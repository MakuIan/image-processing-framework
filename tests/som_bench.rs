use image::RgbImage;
use image_processing_framework::{
    BlockBuffer, BlockConfig, Frame, SomGpuQuantizer, SomMatrix, SomVideoConfig, calculate_psnr,
    compress_indices, decode_video_stream, decompress_indices, deserialize_som,
    encode_video_stream, encode_video_stream_with_progress, extract_blocks_flat,
    reconstruct_image_from_codebook,
};
use std::time::Instant;

/// Liefert den GPU-Quantizer oder `None` ohne Adapter: Der Test wird dann
/// uebersprungen, damit die Suite auch headless gruen bleibt.
fn gpu_quantizer_or_skip() -> Option<SomGpuQuantizer> {
    match SomGpuQuantizer::new() {
        Some(quantizer) => Some(quantizer),
        None => {
            println!("kein GPU-Adapter - GPU-Test uebersprungen");
            None
        }
    }
}

/// Deterministisches SOM plus Bloecke mit disjunkten Minima: Jeder Block
/// ist exakt eine Gewichtszeile (Distanz 0.0) oder eine davon minimal
/// gestoerte Variante. CPU und GPU muessen bit-identisch entscheiden.
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

    let mut data: Vec<f32> = Vec::with_capacity((neuron_count + 32) * dim);
    for weight in &som.weights {
        data.extend_from_slice(weight);
    }
    for k in 0..32 {
        let mut block = som.weights[(k * 13) % neuron_count].clone();
        block[k % dim] += 0.001;
        data.extend_from_slice(&block);
    }

    let blocks = BlockBuffer::new(data, dim);
    (som, blocks)
}

/// GPU-vs-CPU-Aequivalenz auf Disjunkten-Minima-Fixtures: bit-identische
/// Indizes (inkl. inaktiver Lanes bei 8x8-Gittern und Tie-Break).
#[test]
fn gpu_quantize_matches_cpu_on_disjoint_minima() {
    let Some(gpu) = gpu_quantizer_or_skip() else {
        return;
    };

    for grid in [8usize, 16] {
        let (som, blocks) = disjoint_fixture(grid);
        let weights_flat: Vec<f32> = som.weights.iter().flatten().copied().collect();

        let cpu = som.quantize_all_flat(&blocks);
        let gpu_indices = gpu.quantize_flat(&weights_flat, &blocks);

        assert_eq!(gpu_indices.len(), cpu.len(), "grid {grid}x{grid}");
        assert_eq!(gpu_indices, cpu, "grid {grid}x{grid}");
    }
}

/// Zufaellige Bloecke: alle Indizes liegen im Codebook, und die
/// Rekonstruktionsqualitaet (PSNR) von GPU- und CPU-Indizes weicht um
/// weniger als 1e-3 dB voneinander ab.
#[test]
fn gpu_quantize_reconstruction_matches_cpu_psnr() {
    let Some(gpu) = gpu_quantizer_or_skip() else {
        return;
    };

    let (width, height) = (64u32, 64u32);
    let data: Vec<u8> = (0..(width * height * 3) as usize)
        .map(|i| ((i * 37 + (i >> 5) * 11 + 3) % 256) as u8)
        .collect();
    let original = Frame::new(
        RgbImage::from_raw(width, height, data.clone()).expect("64x64 frame buffer"),
        0,
        0.0,
    );

    let block_config = BlockConfig::new(4, 3);
    let blocks = extract_blocks_flat(&data, width, height, &block_config);
    let samples: Vec<Vec<f32>> = (0..blocks.len())
        .step_by(3)
        .map(|i| blocks.block(i).to_vec())
        .collect();
    let mut som = SomMatrix::new(16, 16, blocks.dim(), &samples);
    som.train(&samples, 5, 0.3);

    let cpu = som.quantize_all_flat(&blocks);
    let weights_flat: Vec<f32> = som.weights.iter().flatten().copied().collect();
    let gpu_indices = gpu.quantize_flat(&weights_flat, &blocks);

    assert_eq!(gpu_indices.len(), cpu.len());
    assert!(
        gpu_indices
            .iter()
            .all(|&index| (index as usize) < som.neuron_count())
    );
    let differing = cpu
        .iter()
        .zip(gpu_indices.iter())
        .filter(|(a, b)| a != b)
        .count();

    let psnr_of = |indices: &[u8]| -> f64 {
        let raw = reconstruct_image_from_codebook(
            indices,
            &som.weights,
            width,
            height,
            &block_config,
        );
        let recon = Frame::new(
            RgbImage::from_raw(width, height, raw).expect("recon buffer must fit"),
            0,
            0.0,
        );
        calculate_psnr(&original, &recon)
    };

    let psnr_cpu = psnr_of(&cpu);
    let psnr_gpu = psnr_of(&gpu_indices);
    println!(
        "PSNR CPU {psnr_cpu:.4} dB | GPU {psnr_gpu:.4} dB | abweichende Indizes: {differing}/{}",
        cpu.len()
    );

    assert!(
        (psnr_cpu - psnr_gpu).abs() < 1e-3,
        "PSNR CPU {psnr_cpu:.6} dB vs. GPU {psnr_gpu:.6} dB"
    );
}

/// Stream-Rundtrip mit `with_gpu(true)`: Der Encoder rechnet den
/// Block-Index-Strom auf der GPU; die Indizes muessen fuer die im Paket
/// uebertragenen Codebooks exakt der CPU-Referenz entsprechen.
///
/// Zwei unabhaengig trainierte Streams sind **nicht** vergleichbar
/// (`SomMatrix::new` initialisiert die Gewichte zufaellig) - deshalb wird
/// der Vergleich pro Frame gegen den Codebook-Zustand gefuehrt, den auch
/// der Decoder aus dem Stream rekonstruiert.
#[test]
fn gpu_stream_matches_cpu_reference_indices() {
    let Some(_gpu) = gpu_quantizer_or_skip() else {
        return;
    };

    let (width, height) = (64u32, 64u32);
    let frames: Vec<Frame> = (0..6)
        .map(|k| {
            let data: Vec<u8> = (0..(width * height * 3) as usize)
                .map(|i| {
                    let base = (i * 13 + (i / 64) * 29) % 256;
                    ((base + k * 17) % 256) as u8
                })
                .collect();
            Frame::new(
                RgbImage::from_raw(width, height, data).expect("64x64 frame buffer"),
                k,
                k as f64 / 30.0,
            )
        })
        .collect();

    let config = SomVideoConfig::default().with_gpu(true);
    let result = encode_video_stream(&frames, &config);
    assert_eq!(result.frames.len(), frames.len());
    assert!(result.frames[0].is_keyframe);

    // Codebook-Zustand wie der Decoder fortschreiben und die GPU-Indizes
    // jedes Frames gegen `quantize_all_flat` pruefen.
    let block_config = BlockConfig::new(config.block_size, 3);
    let vector_dim = block_config.block_dim();
    let mut current: Option<SomMatrix> = None;

    for (k, enc) in result.frames.iter().enumerate() {
        if enc.is_keyframe {
            let payload = enc
                .som_matrix
                .as_ref()
                .expect("keyframes carry a codebook");
            current = Some(deserialize_som(
                payload,
                config.som_grid_size,
                config.som_grid_size,
                vector_dim,
            ));
        }
        let som = current.as_ref().expect("codebook exists after frame 0");

        let blocks = extract_blocks_flat(
            frames[k].data().as_raw(),
            frames[k].width(),
            frames[k].height(),
            &block_config,
        );
        let cpu = som.quantize_all_flat(&blocks);
        let gpu_indices =
            decompress_indices(&enc.compressed_indices).expect("indices must decompress");

        let differing = cpu
            .iter()
            .zip(gpu_indices.iter())
            .filter(|(a, b)| a != b)
            .count();
        println!(
            "Frame {k} ({}): {differing}/{} Indizes weichen ab | Keyframe {}",
            if enc.is_keyframe { "Keyframe" } else { "Interframe" },
            cpu.len(),
            enc.is_keyframe
        );
        assert_eq!(gpu_indices, cpu, "frame {k}: GPU weicht von der CPU-Referenz ab");
    }

    // Der GPU-Stream muss zusaetzlich gueltig dekodieren.
    let decoded = decode_video_stream(&result.frames, width, height, &config, 30.0)
        .expect("decoding must succeed");
    for (k, (original, decoded_frame)) in frames.iter().zip(decoded.iter()).enumerate() {
        let psnr = calculate_psnr(original, decoded_frame);
        println!("Frame {k}: Deko-PSNR {psnr:.4} dB");
        assert!(psnr > 20.0 || psnr == f64::INFINITY, "frame {k} PSNR {psnr}");
    }
}

#[test]
#[ignore]
fn bench_som_performance() {
    let dim = 48; // 4x4 RGB
    let num_blocks = 129_600; // 1080p frame
    let num_neurons = 256; // 16x16 grid

    // Generate deterministic pseudo-random blocks
    let mut blocks = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let mut b = Vec::with_capacity(dim);
        for d in 0..dim {
            b.push(((i * 31 + d * 17) % 256) as f32 / 255.0);
        }
        blocks.push(b);
    }

    let mut som = SomMatrix::new(16, 16, dim, &blocks[0..num_neurons]);

    // Benchmark SIMD quantize_all
    let start_quant = Instant::now();
    let indices = som.quantize_all(&blocks);
    let dur_quant = start_quant.elapsed();
    println!("SIMD quantize_all ({} blocks): {:?}", num_blocks, dur_quant);
    assert_eq!(indices.len(), num_blocks);

    // Benchmark 1 epoch of train (5,000 blocks)
    let train_subset = &blocks[0..5000];
    let start_train = Instant::now();
    som.train(train_subset, 1, 0.3);
    let dur_train = start_train.elapsed();
    println!("SIMD train (5000 blocks, 1 epoch): {:?}", dur_train);
}

/// Enkodierungs-Benchmark mit dem 4K-Format des Optimierungsplans
/// (4096x1152): misst die reine `encode_video_stream`-Laufzeit inklusive
/// adaptiver Keyframe-Erkennung und extrapolert auf die Plan-Referenz von
/// 914 Frames.
///
/// ```
/// cargo test --release -- --ignored --nocapture bench_som_encode_4k
/// ```
#[test]
#[ignore]
fn bench_som_encode_4k() {
    let (width, height) = (4096u32, 1152u32);
    let frame_count = 60usize;

    // Wrap-freier Farbverlauf: Basiswerte <= 126, zeitlicher Shift <= 118,
    // damit keine Bytes ueberlaufen (ein Wrap waere ein >250er Blockfehler
    // und wuerde die Keyframe-Erkennung kuenstlich in Permanenz-Training
    // treiben - das waere ein Inhalt-Artefakt, kein Codec-Verhalten).
    let mut base = vec![0u8; (width as usize) * (height as usize) * 3];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let i = (y * width as usize + x) * 3;
            base[i] = ((x % 64) + (y % 64)) as u8;
            base[i + 1] = (((x / 2) % 64) + ((y / 2) % 64)) as u8;
            base[i + 2] = (((x + y) / 4) % 128) as u8;
        }
    }

    let frames: Vec<Frame> = (0..frame_count)
        .map(|k| {
            let shift = (k as u8).wrapping_mul(2);
            let data: Vec<u8> = base.iter().map(|&v| v.wrapping_add(shift)).collect();
            Frame::new(
                RgbImage::from_raw(width, height, data).expect("4K frame buffer must fit"),
                k,
                k as f64 / 60.0,
            )
        })
        .collect();

    let config = SomVideoConfig::default();
    let mut previous = Instant::now();
    let mut interframe_times = Vec::new();
    let mut keyframe_times = Vec::new();
    let start = Instant::now();
    let result = encode_video_stream_with_progress(&frames, &config, |prog| {
        let now = Instant::now();
        if prog.frame_index > 1 {
            let took = now.duration_since(previous);
            if prog.is_keyframe {
                keyframe_times.push(took);
            } else {
                interframe_times.push(took);
            }
        }
        previous = now;
    });
    let elapsed = start.elapsed();

    let mean = |durations: &[std::time::Duration]| -> std::time::Duration {
        if durations.is_empty() {
            std::time::Duration::ZERO
        } else {
            durations.iter().sum::<std::time::Duration>() / durations.len() as u32
        }
    };

    let per_frame = elapsed / frame_count as u32;
    let extrapolated = elapsed.mul_f64(914.0 / frame_count as f64);
    println!(
        "encode_video_stream {}x{} ({} frames): {:?} ({:?}/Frame)",
        width, height, frame_count, elapsed, per_frame
    );
    println!(
        "  Interframe ({}x): {:?}/Frame | Keyframe ({}x): {:?}/Frame",
        interframe_times.len(),
        mean(&interframe_times),
        keyframe_times.len(),
        mean(&keyframe_times)
    );
    println!(
        "extrapoliert auf 914 Frames: {:?} | Keyframes: {} ({:.1}%) | avg PSNR: {:.2} dB",
        extrapolated,
        result.stats.keyframe_count,
        result.stats.keyframe_count as f64 / frame_count as f64 * 100.0,
        result.stats.avg_psnr_db
    );

    // Komponenten-Einzelmessung an Frame 0 ( ohne Progress-Overhead ).
    let block_config = BlockConfig::new(4, 3);
    let raw = frames[0].data().as_raw();

    let t = Instant::now();
    let blocks = extract_blocks_flat(raw, width, height, &block_config);
    let d_extract = t.elapsed();

    let train_samples: Vec<Vec<f32>> = (0..blocks.len().min(10_000))
        .map(|i| blocks.block(i).to_vec())
        .collect();
    let mut som = SomMatrix::new(
        config.som_grid_size,
        config.som_grid_size,
        block_config.block_dim(),
        &train_samples,
    );
    let t = Instant::now();
    som.train(&train_samples, config.train_epochs, 0.3);
    let d_train = t.elapsed();

    let t = Instant::now();
    let indices = som.quantize_all_flat(&blocks);
    let d_quantize = t.elapsed();

    let t = Instant::now();
    let recon_raw =
        reconstruct_image_from_codebook(&indices, &som.weights, width, height, &block_config);
    let d_reconstruct = t.elapsed();

    let recon_frame = Frame::new(
        RgbImage::from_raw(width, height, recon_raw).expect("recon buffer must fit"),
        0,
        0.0,
    );
    let t = Instant::now();
    let frame_psnr = calculate_psnr(&frames[0], &recon_frame);
    let d_psnr = t.elapsed();

    let t = Instant::now();
    let compressed = compress_indices(&indices);
    let d_compress = t.elapsed();

    println!(
        "Komponenten (1 Frame, 4K): extract {:?} | train(10k, 5ep) {:?} | quantize {:?} | \
         reconstruct {:?} | psnr {:?} | compress {:?} ({} -> {} Bytes, {:.1} dB)",
        d_extract,
        d_train,
        d_quantize,
        d_reconstruct,
        d_psnr,
        d_compress,
        indices.len(),
        compressed.len(),
        frame_psnr
    );

    assert_eq!(result.frames.len(), frame_count);
    assert!(
        result.stats.keyframe_count * 4 <= frame_count,
        "adaptive keyframe detection should stay well below 100%, got {}/{}",
        result.stats.keyframe_count,
        frame_count
    );
}

/// CPU-vs-GPU-Benchmark der reinen Quantisierung auf 4K-Bloeckmenge
/// (4096x1152 -> 294.912 Bloecke x 256 Neuronen x 48 Dims).
///
/// ```
/// cargo test --release -- --ignored --nocapture bench_som_quantize_4k_gpu
/// ```
#[test]
#[ignore]
fn bench_som_quantize_4k_gpu() {
    let Some(gpu) = gpu_quantizer_or_skip() else {
        return;
    };

    let (width, height) = (4096u32, 1152u32);
    let data: Vec<u8> = (0..(width * height * 3) as usize)
        .map(|i| ((i * 31 + (i >> 7) * 17) % 251) as u8)
        .collect();

    let block_config = BlockConfig::new(4, 3);
    let blocks = extract_blocks_flat(&data, width, height, &block_config);

    let samples: Vec<Vec<f32>> = (0..blocks.len())
        .step_by(29)
        .map(|i| blocks.block(i).to_vec())
        .collect();
    let mut som = SomMatrix::new(16, 16, blocks.dim(), &samples);
    som.train(&samples, 1, 0.3);
    let weights_flat: Vec<f32> = som.weights.iter().flatten().copied().collect();

    // Warmup: Pipeline- und Pufferaufbau nicht in der Messung.
    let warmup = gpu.quantize_flat(&weights_flat, &blocks);
    assert_eq!(warmup.len(), blocks.len());

    let start = Instant::now();
    let cpu_indices = som.quantize_all_flat(&blocks);
    let d_cpu = start.elapsed();

    let start = Instant::now();
    let gpu_indices = gpu.quantize_flat(&weights_flat, &blocks);
    let d_gpu = start.elapsed();

    let differing = cpu_indices
        .iter()
        .zip(gpu_indices.iter())
        .filter(|(a, b)| a != b)
        .count();

    println!(
        "quantize {} Bloecke: CPU {:?} | GPU {:?} ({:.1}x)",
        blocks.len(),
        d_cpu,
        d_gpu,
        d_cpu.as_secs_f64() / d_gpu.as_secs_f64().max(1e-12)
    );
    println!("  abweichende Indizes: {differing}/{}", cpu_indices.len());

    assert_eq!(cpu_indices.len(), gpu_indices.len());
    assert!(gpu_indices.iter().all(|&i| (i as usize) < som.neuron_count()));
}
