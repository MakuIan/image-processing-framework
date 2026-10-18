//! Optionale GPU-Beschleunigung fuer die SOM-Quantisierung.
//!
//! Abgebildet wird ausschliesslich der Hot Path des Encode-Loops: die
//! Distanz-Matrix `Bloecke x Neuronen` der BMU-Suche
//! ([`SomMatrix::quantize_all_flat`]). Training und Rekonstruktion bleiben
//! bewusst auf der CPU (sequenzielles Kohonen-Training bzw. bereits
//! Rayon-parallel).
//!
//! Die CPU-Implementierung ist die Referenz: [`SomGpuQuantizer::new`]
//! liefert ohne GPU-Adapter `None` und der Caller faellt - wie bei den
//! Filtern - auf die CPU zurueck. Der Quantizer selbst enthaelt keine
//! Fallback-Logik.

use std::cell::RefCell;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::{GpuContext, get_gpu_context};
use crate::block::BlockBuffer;

/// Workgroups pro Zeile des 2D-Dispatch-Gitters. Eine Workgroup
/// quantisiert [`BLOCKS_PER_WG`] Bloecke; die Aufteilung in X/Y haelt die
/// Gesamtzahl der Workgroups unterhalb des WebGPU-Standardlimits
/// `max_compute_workgroups_per_dimension` (65.535), das ein 1D-Dispatch
/// bei 4K-Bloeckzahlen (~295.000) auf vielen Adaptern ueberschreiten
/// wuerde.
const GRID_DIM_X: u32 = 256;

/// Bloecke, die eine Workgroup sequenziell quantisiert: Die Dimensionen
/// laufen aussen, sodass die Gewichtszeile eines Lanes pro Workgroup nur
/// einmal geladen wird (statt `BLOCKS_PER_WG`-fachem Leseverkehr in den
/// Caches), und der Dispatch-Aufwand wird ueber mehrere Bloecke
/// amortisiert. Messung (4K, RX 7900 XT): 1 Block 20.9 ms, 8 Bloecke
/// 6.6 ms, 16 Bloecke 6.0 ms, 32 Bloecke 6.7 ms.
const BLOCKS_PER_WG: u32 = 16;

/// Hard-Limit des Compilers: Die Akkumulatoren-Schleife im Kernel
/// (Feldgroesse von `acc`) ist auf diese Blockzahl ausgelegt.
const MAX_BLOCKS_PER_WG: u32 = 32;

const _: () = assert!(BLOCKS_PER_WG <= MAX_BLOCKS_PER_WG);

/// Hoechstmenge an Bloecken, die das 2D-Dispatch-Gitter abdeckt.
const MAX_DISPATCHABLE_BLOCKS: usize = (GRID_DIM_X as usize * 65_535) * BLOCKS_PER_WG as usize;

/// Speicher-Rundung fuer die wiederverwendeten Storage-/Staging-Buffer:
/// verhindert bei konstanter Frame-Groesse eine Neuinstantiierung pro
/// Frame (VRAM-Fragmentierung, Treiber-Overhead).
const ALLOC_GRANULARITY: u64 = 16 * 1024;

const SOM_BMU_WGSL: &str = r#"
struct Params {
    neuron_count: u32,   // grid_width * grid_height (<= 256)
    vector_dim:   u32,   // z. B. 48
    block_count:  u32,   // Anzahl Bloecke in diesem Frame
    grid_dim_x:   u32,   // Anzahl Workgroups in X-Richtung (fuer 2D-Indexberechnung)
    blocks_per_wg: u32,  // Bloecke je Workgroup (<= 32)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<f32>;  // neuron_count * vector_dim
@group(0) @binding(2) var<storage, read> blocks:  array<f32>;  // block_count * vector_dim
@group(0) @binding(3) var<storage, read_write> out_indices: array<u32>; // block_count

// 256 Eintraege: ein Distanz-/Index-Slot pro (max.) Neuron.
var<workgroup> dist_scratch: array<f32, 256>;
var<workgroup> idx_scratch:  array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // Der Blockindex haengt nur von der Workgroup ab - alle Lanes sind in
    // jeder Schleife und vor jedem Barrier identisch gesteuert.
    let wg_index = wid.y * params.grid_dim_x + wid.x;
    let first_block = wg_index * params.blocks_per_wg;
    let neuron_idx = lid.x;
    let valid_lane = neuron_idx < params.neuron_count;

    // Phase 1: Distanzen zu allen Bloecken dieser Workgroup. Die
    // Dimensionen laufen aussen: die Gewichtszeile des Lanes wird pro
    // Dimension genau einmal geladen und ueber alle Bloecke wiederverwendet
    // (sonst `blocks_per_wg`-facher Leseverkehr in den Caches).
    // Einheitlicher Kontrollfluss ohne divergentes if: ungueltige Lanes
    // lesen gefahrlos Neuron 0 und werden in Phase 2 via select neutralisiert.
    var acc: array<f32, 32>;
    for (var p = 0u; p < params.blocks_per_wg; p++) {
        acc[p] = 0.0;
    }
    let safe_neuron_offset = select(0u, neuron_idx * params.vector_dim, valid_lane);
    for (var d = 0u; d < params.vector_dim; d++) {
        let w = weights[safe_neuron_offset + d];
        for (var p = 0u; p < params.blocks_per_wg; p++) {
            let block_idx = first_block + p;
            if (block_idx < params.block_count) {
                let diff = blocks[block_idx * params.vector_dim + d] - w;
                acc[p] += diff * diff;
            }
        }
    }

    // Phase 2: Reduktion Block fuer Block ueber den 256-Eintrag-Scratch.
    for (var p = 0u; p < params.blocks_per_wg; p++) {
        let block_idx = first_block + p;
        let dist = select(1e38, acc[p], valid_lane && block_idx < params.block_count);
        dist_scratch[neuron_idx] = dist;
        idx_scratch[neuron_idx] = neuron_idx;

        // Parallele Reduktion ueber alle 256 Lanes der Workgroup.
        for (var s = 128u; s > 0u; s >>= 1u) {
            workgroupBarrier();
            if (neuron_idx < s) {
                let d1 = dist_scratch[neuron_idx];
                let d2 = dist_scratch[neuron_idx + s];
                let i1 = idx_scratch[neuron_idx];
                let i2 = idx_scratch[neuron_idx + s];

                // Tie-Break: kleinerer Index gewinnt bei gleicher Distanz
                // (entspricht dem ersten Minimum der CPU-Referenz).
                if (d2 < d1 || (d2 == d1 && i2 < i1)) {
                    dist_scratch[neuron_idx] = d2;
                    idx_scratch[neuron_idx] = i2;
                }
            }
        }

        if (neuron_idx == 0u && block_idx < params.block_count) {
            out_indices[block_idx] = idx_scratch[0];
        }

        // Sicherstellen, dass alle Threads diesen Block abgeschlossen haben,
        // bevor ein Thread in der naechsten Iteration den Scratch-Speicher ueberschreibt.
        workgroupBarrier();
    }
}
"#;

/// Uniform-Daten des Quantize-Kernels (32 Byte, auf 16 Byte ausgerichtet).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SomQuantizeUniform {
    neuron_count: u32,
    vector_dim: u32,
    block_count: u32,
    grid_dim_x: u32,
    blocks_per_wg: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Ein wiederverwendeter GPU-Puffer samt seiner Capacitaet in Bytes.
struct Slot {
    buffer: wgpu::Buffer,
    capacity: u64,
}

/// Puffer-Satz eines [`SomGpuQuantizer`]s. Wird nur bei wachsendem
/// Speicherbedarf neu instantiiert, sonst pro Frame ueber
/// `Queue::write_buffer` befuellt.
#[derive(Default)]
struct SomGpuBuffers {
    uniform: Option<Slot>,
    weights: Option<Slot>,
    blocks: Option<Slot>,
    output: Option<Slot>,
    staging: Option<Slot>,
}

fn create_or_grow(
    slot: &mut Option<Slot>,
    device: &wgpu::Device,
    label: &str,
    usage: wgpu::BufferUsages,
    size: u64,
) {
    if slot.as_ref().is_some_and(|s| s.capacity >= size) {
        return;
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    });
    *slot = Some(Slot {
        buffer,
        capacity: size,
    });
}

impl SomGpuBuffers {
    fn ensure(
        &mut self,
        device: &wgpu::Device,
        weights_size: u64,
        blocks_size: u64,
        out_size: u64,
    ) {
        create_or_grow(
            &mut self.uniform,
            device,
            "som_quantize_uniform",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            std::mem::size_of::<SomQuantizeUniform>() as u64,
        );
        create_or_grow(
            &mut self.weights,
            device,
            "som_quantize_weights",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            weights_size.next_multiple_of(ALLOC_GRANULARITY),
        );
        create_or_grow(
            &mut self.blocks,
            device,
            "som_quantize_blocks",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            blocks_size.next_multiple_of(ALLOC_GRANULARITY),
        );
        create_or_grow(
            &mut self.output,
            device,
            "som_quantize_output",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            out_size.next_multiple_of(ALLOC_GRANULARITY),
        );
        create_or_grow(
            &mut self.staging,
            device,
            "som_quantize_staging",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            out_size.next_multiple_of(ALLOC_GRANULARITY),
        );
    }
}

/// GPU-Pfad der SOM-Quantisierung (BMU-Suche ueber alle Blöcke).
///
/// Der Konstruktor baut die Compute-Pipeline einmal auf; pro Aufruf von
/// [`SomGpuQuantizer::quantize_flat`] werden nur noch die Daten-Buffer
/// beschrieben und dispatched.
pub struct SomGpuQuantizer {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    buffers: RefCell<SomGpuBuffers>,
}

impl SomGpuQuantizer {
    /// Erzeugt den Quantizer, sofern ein GPU-Adapter gefunden wird.
    ///
    /// Liefert `None` ohne Adapter (headless/unterstuetzt nicht) - der
    /// Caller entscheidet dann ueber den CPU-Fallback.
    pub fn new() -> Option<Self> {
        let ctx = get_gpu_context()?;
        Some(Self::with_context(ctx))
    }

    /// Baut Layout und Compute-Pipeline (Shader-Kompilierung) auf.
    fn with_context(ctx: Arc<GpuContext>) -> Self {
        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SomBmuShader"),
            source: wgpu::ShaderSource::Wgsl(SOM_BMU_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("som_quantize_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("som_quantize_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("som_quantize_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            ctx,
            pipeline,
            bind_group_layout,
            buffers: RefCell::new(SomGpuBuffers::default()),
        }
    }

    /// Quantisiert alle Bloecke von `blocks` gegen die flachen `weights_flat`
    /// (zeilenmajor, `neuron_count * vector_dim` Werte) und liefert die
    /// BMU-Indizes in Eingabereihenfolge.
    ///
    /// Die Reihenfolge und der Tie-Break (kleinerer Index bei gleicher
    /// Distanz) entsprechen [`SomMatrix::quantize_all_flat`]; wegen
    /// abweichender Gleitkomma-Rundung (GPU-FMA vs. AVX2) koennen
    /// quasi-identische Distanzen in seltenen Faellen eine andere BMU
    /// waehlen.
    ///
    /// # Panics
    /// Panics, wenn `weights_flat` nicht ein Vielfaches der Blockdimension
    /// ist, mehr als 256 Neuronen beschreibt oder der Frame mehr Bloecke
    /// als das Dispatch-Gitter aufnehmen kann.
    pub fn quantize_flat(&self, weights_flat: &[f32], blocks: &BlockBuffer) -> Vec<u8> {
        let dim = blocks.dim();
        assert!(!weights_flat.is_empty(), "weight buffer must not be empty");
        assert!(
            weights_flat.len().is_multiple_of(dim),
            "weight buffer has {} value(s), not a multiple of block dim {dim}",
            weights_flat.len()
        );
        let neuron_count = weights_flat.len() / dim;
        assert!(
            neuron_count <= 256,
            "som grid has {neuron_count} neurons, at most 256 are supported (indices are u8)"
        );

        let block_count = blocks.len();
        if block_count == 0 {
            return Vec::new();
        }
        assert!(
            block_count <= MAX_DISPATCHABLE_BLOCKS,
            "frame has {block_count} blocks, the 2D dispatch grid holds at most {MAX_DISPATCHABLE_BLOCKS}"
        );

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let uniform = SomQuantizeUniform {
            neuron_count: neuron_count as u32,
            vector_dim: dim as u32,
            block_count: block_count as u32,
            grid_dim_x: GRID_DIM_X,
            blocks_per_wg: BLOCKS_PER_WG,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let weights_bytes: &[u8] = bytemuck::cast_slice(weights_flat);
        let blocks_bytes: &[u8] = bytemuck::cast_slice(blocks.data());
        let out_size = (block_count * std::mem::size_of::<u32>()) as u64;

        {
            let mut buffers = self.buffers.borrow_mut();
            buffers.ensure(
                device,
                weights_bytes.len() as u64,
                blocks_bytes.len() as u64,
                out_size,
            );
        }

        let buffers = self.buffers.borrow();
        let uniform_buffer = &buffers
            .uniform
            .as_ref()
            .expect("uniform buffer must be ensured")
            .buffer;
        let weights_buffer = &buffers
            .weights
            .as_ref()
            .expect("weights buffer must be ensured")
            .buffer;
        let blocks_buffer = &buffers
            .blocks
            .as_ref()
            .expect("blocks buffer must be ensured")
            .buffer;
        let output_buffer = &buffers
            .output
            .as_ref()
            .expect("output buffer must be ensured")
            .buffer;
        let staging_buffer = &buffers
            .staging
            .as_ref()
            .expect("staging buffer must be ensured")
            .buffer;

        queue.write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        queue.write_buffer(weights_buffer, 0, weights_bytes);
        queue.write_buffer(blocks_buffer, 0, blocks_bytes);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("som_quantize_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: blocks_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("som_quantize_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("som_quantize_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // Ein Workgroup quantisiert `BLOCKS_PER_WG` Bloecke; das
            // Gesamt-Workgroup-Gitter wird 2D aufgeteilt, damit es nicht
            // das Dimensionen-Limit ueberschreitet.
            let workgroups = (block_count as u32).div_ceil(BLOCKS_PER_WG);
            compute_pass.dispatch_workgroups(GRID_DIM_X, workgroups.div_ceil(GRID_DIM_X), 1);
        }

        encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, out_size);
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..out_size);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("map_async callback dropped")
            .expect("reading the SOM indices from the GPU must succeed");

        let data = buffer_slice.get_mapped_range();
        let out_indices: &[u32] = bytemuck::cast_slice(&data);
        let indices: Vec<u8> = out_indices.iter().map(|&index| index as u8).collect();
        drop(data);
        staging_buffer.unmap();

        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::SomMatrix;

    /// Liefert den Quantizer oder `None` (kein Adapter): Der Test wird dann
    /// uebersprungen, damit die Suite auch headless gruen bleibt.
    fn quantizer_or_skip() -> Option<SomGpuQuantizer> {
        match SomGpuQuantizer::new() {
            Some(quantizer) => Some(quantizer),
            None => {
                println!("kein GPU-Adapter - Test uebersprungen");
                None
            }
        }
    }

    /// Deterministisches SOM plus Bloecke mit disjunkten Minima: Jeder
    /// Block ist exakt eine Gewichtszeile (Distanz 0) oder eine davon
    /// minimal gestoerte Variante, sodass CPU und GPU bit-identisch
    /// entscheiden muessen.
    fn disjoint_fixture(grid: usize) -> (SomMatrix, BlockBuffer) {
        let dim = 48usize;
        let neuron_count = grid * grid;
        let som = SomMatrix {
            grid_width: grid,
            grid_height: grid,
            vector_dim: dim,
            weights: (0..neuron_count)
                .map(|i| {
                    (0..dim)
                        .map(|d| ((i * 31 + d * 7) % 251) as f32 / 255.0)
                        .collect()
                })
                .collect(),
        };

        let mut data: Vec<f32> = Vec::with_capacity((neuron_count + 16) * dim);
        for weight in &som.weights {
            data.extend_from_slice(weight);
        }
        for k in 0..16 {
            let mut block = som.weights[(k * 13) % neuron_count].clone();
            block[k % dim] += 0.001;
            data.extend_from_slice(&block);
        }

        let blocks = BlockBuffer::new(data, dim);
        (som, blocks)
    }

    #[test]
    fn quantizer_creation_is_optional() {
        match SomGpuQuantizer::new() {
            Some(_) => println!("GPU-Adapter gefunden"),
            None => println!("kein GPU-Adapter - Fallback auf CPU erwartet"),
        }
    }

    #[test]
    fn gpu_matches_cpu_on_disjoint_minima() {
        let Some(quantizer) = quantizer_or_skip() else {
            return;
        };

        // 8x8 = 64 Neuronen: uebt den Pfad mit inaktiven Lanes (neuron_count < 256),
        // 16x16 = 256 Neuronen den vollen Reduktionsbaum.
        for grid in [8usize, 16] {
            let (som, blocks) = disjoint_fixture(grid);
            let weights_flat: Vec<f32> = som.weights.iter().flatten().copied().collect();

            let cpu = som.quantize_all_flat(&blocks);
            let gpu = quantizer.quantize_flat(&weights_flat, &blocks);

            assert_eq!(gpu.len(), cpu.len(), "grid {grid}x{grid}");
            assert_eq!(gpu, cpu, "grid {grid}x{grid}");
            assert!(gpu.iter().all(|&i| (i as usize) < som.neuron_count()));
        }
    }

    #[test]
    fn gpu_indices_stay_within_codebook_for_random_blocks() {
        let Some(quantizer) = quantizer_or_skip() else {
            return;
        };

        let dim = 48usize;
        let samples: Vec<Vec<f32>> = (0..64)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * 7 + j * 13) % 256) as f32 / 255.0)
                    .collect()
            })
            .collect();
        let som = SomMatrix::new(16, 16, dim, &samples);

        let block_count = 2_000usize;
        let blocks = BlockBuffer::new(
            (0..block_count * dim)
                .map(|i| ((i * 3 + 11) % 256) as f32 / 256.0)
                .collect(),
            dim,
        );

        let weights_flat: Vec<f32> = som.weights.iter().flatten().copied().collect();
        let gpu = quantizer.quantize_flat(&weights_flat, &blocks);

        assert_eq!(gpu.len(), block_count);
        assert!(gpu.iter().all(|&i| (i as usize) < som.neuron_count()));
    }

    #[test]
    fn gpu_handles_empty_block_buffer() {
        let Some(quantizer) = quantizer_or_skip() else {
            return;
        };
        let weights = vec![0.5f32; 256 * 48];
        let blocks = BlockBuffer::new(Vec::new(), 48);
        assert!(quantizer.quantize_flat(&weights, &blocks).is_empty());
    }
}
