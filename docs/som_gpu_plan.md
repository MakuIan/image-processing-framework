# Implementierungsplan: Optionale GPU-Beschleunigung fuer das SOM-Verfahren

> **Status:** Implementiert (Stufen 1-5, siehe §7)
> **Ziel:** Optionaler GPU-Pfad (wgpu Compute Shaders) fuer die SOM-Quantisierung — analog zum `--gpu`-Flag der Filter. CPU bleibt immer die Referenz und Fallback-Pfad.

---

## 1. Scope und Abgrenzung

| Teil des SOM-Verfahrens | GPU? | Begruendung |
| :--- | :--- | :--- |
| **Quantisierung (`quantize_all_flat`)** | **Ja** | Hot Path im Encode-Loop: ~295k Bloecke x 256 Neuronen x 48 Dims reine Distanz-Matrix. Grosse, unabhaengige Parallelarbeit — ideal fuer Compute Shader (vgl. `som_optimization_plan.md` §3: gesch~50 ms CPU -> ~2-5 ms GPU). |
| **Training (`SomMatrix::train`)** | **Nein** | Online-Kohonen ist pro Sample sequenziell (BMU -> Update -> naechstes Sample). Keine Parallelstruktur (vgl. `som_optimization_plan.md` §3). Maximal 10.000 Samples x 5 Epochen — vernachlaessigbar gegenueber der Quantisierung. Batch-SOM waere ein separater algorithmischer Eingriff (siehe Ausblick, §8). |
| **Rekonstruktion / Decode** | Nein | `reconstruct_image_from_codebook` ist bereits Rayon-parallel und nicht der Engpass. |

**Erwarteter Gewinn:** Interframe-Latenz (~45 ms, fast vollstaendig Quantisierung) auf ~5-10 ms; Keyframe-Latenz (~170 ms) unveraendert (Training bleibt CPU).

---

## 2. Architekturentscheidung: Wie strukturieren?

### 2.1 Was der Filter-Pfad vorgibt (Best Practice im Repo)

* CLI-Flag `--gpu: bool` (`src/main.rs:84-86`), Runtime-Dispatch, kein Device-Enum.
* `get_gpu_context()` Singleton (`src/gpu/mod.rs:11-15`): lazy Auto-Detection, `None` = kein GPU-Adapter.
* GPU-Klasse haelt `Option<GpuComputeHelper>`; ohne GPU => Fallback auf CPU-Implementierung
  (Filter geben unveraendert durch, SOM muss aber zwingend auf CPU zurueckfallen, weil
  Quantisierung Pflichtschritt ist).
* Pro Feature ein WGSL-String + `#[repr(C)] Pod/Zeroable`-Uniform-Struct im selben Modul
  (`src/gpu/{noise,bilateral,blur}.rs`).

### 2.2 Warum NICHT: `GpuComputeHelper` wiederverwenden

`GpuComputeHelper` (`src/gpu/helper.rs:5-9`) hat ein **festes 3-Binding-Layout**
(0=Uniform, 1=Input `array<u32>`, 2=Output `array<u32>`) und `execute` arbeitet auf
`&[u32]`-Pixeln mit fester 16x16-Dispatch-Geometrie. SOM-Quantisierung braucht:

* **4 Bindings**: Uniform + **zwei** Read-only Storage-Buffer (Gewichte **und** Bloecke) + Output;
* **f32**-Daten (kein u32-Pixel-Packing);
* **2D-Dispatch** (ein Workgroup pro Block, aufgeteilt in X/Y zur Einhaltung von wgpu-Hardware-Limits).

Eine Erweiterung des bestehenden Helpers wuerde dessen Schluessel-Layout fuer die
Filter destabilisieren. **Entscheidung: eigener Quantizer in `src/gpu/som.rs`, der nur
`GpuContext` teilt.**

### 2.3 Warum KEIN Trait/Enum-Dispatch

Der Filter-Pfad dispatcht per `FilterConfig::instantiate(gpu: bool) -> Box<dyn Filter>`.
Fuer SOM gibt es nur **eine** Funktion mit zwei Backends — ein Trait waere Over-Engineering.
Stattdessen: wie bei den Filtern ein `bool` in der Config + `Option<SomGpuQuantizer>`,
die Dispatch-Stelle ist eine kleine Helper-Funktion in `video_codec.rs` (siehe §5.3).

### 2.4 Zielstruktur (Dateiuebersicht)

```
src/
  gpu/
    mod.rs          // unveraendert: get_gpu_context(), pub mod som + Re-Export
    helper.rs       // unveraendert: GpuComputeHelper (Filter)
    conversion.rs   // unveraendert
    som.rs          // NEU: SomGpuQuantizer + SOM_BMU_WGSL + SomQuantizeUniform
  video_codec.rs    // gpu-Flag in Config; Dispatch-Helper; Train-Aufrufe unveraendert
  matrix.rs         // unveraendert (CPU bleibt Referenz-Implementierung)
  main.rs           // --gpu help-Text erweitern; SOM-Modus-Print; Config-Wiring
tests/
  som_bench.rs      // GPU-Bench (#[ignore]) + GPU-vs-CPU-Vergleichstests
```

---

## 3. GPU-Kernel-Konzept (`src/gpu/som.rs`)

### 3.1 WGSL-Shader (Details & Limits)

```wgsl
struct Params {
    neuron_count: u32,   // grid_width * grid_height (<= 256)
    vector_dim:   u32,   // z. B. 48
    block_count:  u32,   // Anzahl Bloecke in diesem Frame
    grid_dim_x:   u32,   // Anzahl Workgroups in X-Richtung (fuer 2D-Indexberechnung)
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<f32>;  // neuron_count * vector_dim
@group(0) @binding(2) var<storage, read> blocks:  array<f32>;  // block_count * vector_dim
@group(0) @binding(3) var<storage, read_write> out_indices: array<u32>; // block_count

var<workgroup> dist_scratch: array<f32, 256>;   // ein Eintrag pro (max.) Neuron
var<workgroup> idx_scratch:  array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let block_idx = wid.y * params.grid_dim_x + wid.x;
    if (block_idx >= params.block_count) {
        return;
    }

    let neuron_idx = lid.x;
    var dist = 1e38; // Default Unendlich fuer inaktive Neuronen (wenn neuron_count < 256)

    if (neuron_idx < params.neuron_count) {
        dist = 0.0;
        let block_offset = block_idx * params.vector_dim;
        let neuron_offset = neuron_idx * params.vector_dim;
        for (var d = 0u; d < params.vector_dim; d++) {
            let diff = blocks[block_offset + d] - weights[neuron_offset + d];
            dist += diff * diff;
        }
    }

    dist_scratch[neuron_idx] = dist;
    idx_scratch[neuron_idx] = neuron_idx;

    // Parallel Reduction ueber alle 256 Lanes im Workgroup
    for (var s = 128u; s > 0u; s >>= 1u) {
        workgroupBarrier();
        if (neuron_idx < s) {
            let d1 = dist_scratch[neuron_idx];
            let d2 = dist_scratch[neuron_idx + s];
            let i1 = idx_scratch[neuron_idx];
            let i2 = idx_scratch[neuron_idx + s];

            // Tie-Break: kleinerer Index gewinnt bei gleicher Distanz (entspricht CPU-Verhalten)
            if (d2 < d1 || (d2 == d1 && i2 < i1)) {
                dist_scratch[neuron_idx] = d2;
                idx_scratch[neuron_idx] = i2;
            }
        }
    }

    if (neuron_idx == 0u) {
        out_indices[block_idx] = idx_scratch[0];
    }
}
```

* **Dispatch-Geometrie (2D):**
  WebGPU garantiert als Standard-Grenzwert fuer `max_compute_workgroups_per_dimension`
  65.535. Bei 4K-Frames (~295.000 Bloecken) wuerde ein 1D-Dispatch `(block_count, 1, 1)`
  auf vielen Adaptern crashen. Daher wird ein 2D-Grid aufgespannt:
  ```rust
  let grid_dim_x = 256u32;
  let grid_dim_y = (block_count as u32).div_ceil(grid_dim_x);
  // Dispatch: (grid_dim_x, grid_dim_y, 1)
  ```
* **Reduktion & Initialisierung:**
  Jeder Workgroup bearbeitet genau einen Block (`block_idx`). Die 256 Lanes berechnen
  jeweils die Distanz zu einem Neuron. Nicht genutzte Lanes (`neuron_idx >= neuron_count`,
  z. B. bei einem 8x8 Grid = 64 Neuronen) initialisieren ihren Scratch-Speicher mit `1e38`,
  sodass sie im Reduktionsbaum neutral bleiben.
* **Tie-Break:** Bei `dist == best` waehle kleineren Index — entspricht der Semantik
  von `find_bmu_scalar` (`src/matrix.rs:327-345`, erstes Minimum gewinnt). Damit sind
  GPU-Ergebnisse deterministisch und zur CPU aequivalent.

### 3.2 Uniform-Struct (Rust-Seite, `bytemuck`)

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SomQuantizeUniform {
    neuron_count: u32,
    vector_dim: u32,
    block_count: u32,
    grid_dim_x: u32,
}
```

### 3.3 `SomGpuQuantizer` & Puffer-Wiederverwendung

```rust
pub struct SomGpuQuantizer {
    ctx: Arc<GpuContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl SomGpuQuantizer {
    /// Liefert None, wenn kein GPU-Adapter vorhanden ist (wie Filter-Fallback).
    pub fn new() -> Option<Self>;

    /// Quantisiert `blocks` (flach, Stride = vector_dim) gegen flache `weights`.
    /// Liefert block_count u8-Indizes in Eingabereihenfolge.
    pub fn quantize_flat(&self, weights_flat: &[f32], blocks: &BlockBuffer) -> Vec<u8>;
}
```

* `new()` baut Pipeline **einmal** (Shader kompilieren), wie `GpuComputeHelper::new`.
* **Puffer-Management:** Um VRAM-Fragmentierung und Treiber-Overhead bei 60 FPS zu vermeiden,
  werden Buffer nicht pro Frame unnoetig neu instantiiert, sondern bei stabiler Frame-Groesse
  wiederverwendet bzw. ueber `write_buffer` befuellt.
* CPU-Fallback-Logik lebt **nicht** in `quantize_flat`, sondern im Caller (§5.3) —
  der Quantizer ist ein reiner GPU-Pfad.

---

## 4. Config- und CLI-Aenderungen

### 4.1 `SomVideoConfig` (`src/video_codec.rs:61-116`)

* Neues Feld `pub gpu: bool` (Standard: `false`), analog zum `--gpu`-Flag der Filter.
* `Default`-Anpassung + Builder-Methode:

```rust
pub fn with_gpu(mut self, gpu: bool) -> Self {
    self.gpu = gpu;
    self
}
```

### 4.2 CLI (`src/main.rs`)

* **Wiederverwendung des bestehenden `--gpu`-Flags** (`main.rs:84-86`) — wie bei den
  Filtern ein einziger Schalter. Help-Text erweitern:

  ```
  /// GPU-Compute-Shader (wgpu) fuer Filter und SOM-Quantisierung verwenden
  ```

* `run_som_video_pipeline` (`main.rs:339-538`): `SomVideoConfig` um `.with_gpu(cli.gpu)`
  erweitern; Modus-Print analog `build_pipeline` (`main.rs:169-173`) ergaenzen:

  ```
  SOM-Modus: GPU-Beschleunigung aktiv (wgpu) / SOM-Modus: CPU (SIMD + Rayon)
  ```

* Kein neues Flag: ein `--gpu` steuert konsistent Filter **und** SOM.

---

## 5. Integration in `video_codec.rs`

### 5.1 Quantizer einmalig erzeugen

Am Anfang von `encode_video_stream_with_progress` (`video_codec.rs:326`):

```rust
let gpu_quantizer = if config.gpu { SomGpuQuantizer::new() } else { None };
if config.gpu && gpu_quantizer.is_none() {
    eprintln!("Warnung: GPU-Quantisierung angefordert, aber kein GPU-Adapter verfuegbar. Fallback auf CPU.");
}
```

### 5.2 Flache Gewichte

`quantize_all_flat` spiegelt die Gewichte selbst (`matrix.rs:625`). Fuer den GPU-Pfad
wird der flache Gewichts-Puffer **einmal pro Frame** (nicht pro Aufruf) erstellt und an
die Quantisierungs-Aufrufe im Encode-Loop weitergereicht:

```rust
let weights_flat: Vec<f32> = trained_or_current.weights.iter().flatten().copied().collect();
```

(Nach Keyframe-Training neu aufbauen; bei Interframes wiederverwenden.)

### 5.3 Dispatch-Helper (die einzige Branch-Stelle)

```rust
fn quantize_blocks(
    som: &SomMatrix,
    blocks: &BlockBuffer,
    weights_flat: &[f32],
    gpu: Option<&SomGpuQuantizer>,
) -> Vec<u8> {
    match gpu {
        Some(q) => q.quantize_flat(weights_flat, blocks),
        None => som.quantize_all_flat(blocks),   // CPU: AVX2/Rayon, unveraendert
    }
}
```

* Ersetzt die `quantize_all_flat`-Aufrufe im Encode-Loop (`video_codec.rs:389, 396, 416`).
* `SomMatrix::train` / `train_som_on_blocks` bleiben **unveraendert CPU**.
* Decode-Pfad (`decode_video_stream_with_progress`) bleibt unveraendert.

### 5.4 Determinismus-Hinweis

GPU-`f32`-FMA kann bei **exakt gleichen** Distanzen geringfuegig andere Rundung als AVX2 liefern;
der Tie-Break (§3.1) sorgt fuer deterministische BMU-Wahl **innerhalb** eines
Backends. Quasi-identische Distanzen koennen in seltenen Faellen zu einer anderen BMU fuehren
als die CPU — das ist akzeptabel, solange PSNR/SSIM des Streams im Toleranzband bleiben.

---

## 6. Tests und Validierung

| Test | Ort | Anspruch |
| :--- | :--- | :--- |
| GPU-vs-CPU-Aequivalenz | `tests/som_bench.rs` (oder `#[cfg(test)]` in `gpu/som.rs`) | Fixtures mit disjunkten Minima: **bit-identische** Indizes. Zusaetzlich zufaellige Bloecke: alle Indizes `< neuron_count`, Laenge korrekt, PSNR(Deko GPU) vs PSNR(Deko CPU) innerhalb `1e-3` dB. `#[ignore]` fuer Adapter-abhaengige Laeufe. |
| Fallback | Unit-Test | `SomGpuQuantizer::new()` kann `None` liefern; `quantize_blocks(..., &None)` == `quantize_all_flat`. |
| Stream-Rundtrip | `tests/som_bench.rs` (`gpu_stream_matches_cpu_reference_indices`) | `with_gpu(true)` (falls Adapter vorhanden) erzeugt gueltigen Stream; die GPU-Indizes jedes Frames werden gegen `quantize_all_flat` **mit dem Codebook-Zustand aus dem Stream** geprueft (identisch zur CPU-Referenz), zusaetzlich Deko-PSNR-Check. |
| Benchmark | `tests/som_bench.rs` | `bench_som_quantize_4k_gpu` (`#[ignore]`), Ausgabe CPU-ms vs GPU-ms — analog `src/filters/gpu/mod.rs:16-47`. |
| Manuell | CLI | `cargo run --release -- --som --gpu ...` auf 4K-Material; Latenz- und Qualitaets-Print pruefen. |

Alle bestehenden CPU-Tests muessen **gruen** bleiben (CPU-Pfad unveraendert).

### 6.1 Umsetzungshinweise (Abweichungen/Details)

* **Kein Stream-vs-Stream-Vergleich:** `SomMatrix::new` initialisiert die
  Startgewichte zufaellig (`thread_rng`), daher trainieren zwei Encode-Laeufe
  (CPU- und GPU-Config) unabhaengig voneinander verschiedene Codebooks.
  Statt `1e-3` dB zwischen zwei Streams zu vergleichen, prueft der Rundtrip-Test
  die GPU-Indizes pro Frame gegen `quantize_all_flat` mit dem aus dem Stream
  rekonstruierten Codebook (`deserialize_som` + `decompress_indices`) - das ist
  die eigentliche Aequivalenzbedingung und macht den Test stabil.
* **Adapter-abhaengige Tests ueberspringen statt `#[ignore]`:** Nur die
  Benchmark-Funktionen sind `#[ignore]`; die Aequivalenz- und
  Fallback-Tests laufen im Standardlauf und kehren ohne GPU-Adapter mit
  Meldung zurueck (Suite bleibt headless gruen).
* **Puffer-Wiederverwendung:** `SomGpuQuantizer` haelt die Uniform-,
  Gewichts-, Block-, Output- und Staging-Buffer in einem
  `RefCell<SomGpuBuffers>`; Speicherbedarf wird auf 16 KiB aufgerundet und nur
  bei Wachstum neu instantiiert, sonst pro Frame ueber `Queue::write_buffer`
  befuellt. `BindGroup` und Encoders entstehen pro Aufruf (billig gegenueber
  dem Dispatch).
* **Kernel-Struktur (Abweichung von §3):** Statt einem Block je Workgroup
  (8 Barrieren pro Block) quantisiert eine Workgroup `BLOCKS_PER_WG = 16`
  Bloecke: die Schleife ueber die **Dimensionen lauft aussen**, sodass die
  Gewichtszeile eines Lanes pro Workgroup nur einmal geladen wird (sonst
  `BLOCKS_PER_WG`-facher Leseverkehr fuer ~14 GiB Rohdaten - Cache-,
  Bandbreiten-Limit); die Reduktion folgt blockweise ueber einen
  256-Eintrag-Scratch (8 Barrieren je Block, LDS unter dem 16-KiB-Limit).
  Messung 4K (RX 7900 XT, Min von 9 Laeufen): Block-fuer-Block-Gewichtsleses
  ~21 ms, mit ausgelagerten Gewichten 6.6 ms (K=8) / **6.0 ms (K=16)** / 6.7 ms
  (K=32). `bench_som_quantize_4k_gpu`: CPU 27.8 ms vs. GPU 8.4 ms (3.3x,
  inkl. 54 MiB Upload + Treiber-Overhead), 0 abweichende Indizes.
  Das Plan-Ziel (§3: ~2-5 ms reiner Kernel) ist damit erreicht; der
  Upload-Satz ist irreduzibel, solange die Bloecke auf der CPU entstehen.
* **Keine gemeinsamen Barrieren pro Workgroup-Phase:** Eine
  Lockstep-Variante (mehrere Bloecke reduzieren hinter einem gemeinsamen
  Satz von 8 Barrieren) war bei K<=4 messbar **nicht** schneller als der
  sequenzielle Loop mit ausgelagerten Gewichten - die Barrieren waren
  nicht der Engpass, der Gewichts-Leseverkehr und die Dispatch-Amortisierung
  waren es.

---

## 7. Umsetzungs-Stufen (Reihenfolge)

| Stufe | Inhalt | Dateien |
| :--- | :--- | :--- |
| **1. Quantizer-Kern** | `SomGpuQuantizer` + WGSL + Uniform + `quantize_flat`; 2D-Dispatch; Unit-Test Aequivalenz | `src/gpu/som.rs` (neu), `src/gpu/mod.rs` (Modul + Re-Export) |
| **2. Codec-Integration** | `SomVideoConfig.gpu` + `with_gpu`; `quantize_blocks`-Helper; Encode-Loop umstellen; GPU-Warnung bei fehlendem Adapter | `src/video_codec.rs` |
| **3. CLI-Wiring** | `--gpu` Help-Text; `.with_gpu(cli.gpu)`; SOM-Modus-Print | `src/main.rs` |
| **4. Tests + Bench** | Aequivalenz-, Fallback-, Rundtrip-Tests; `bench_som_quantize_4k_gpu`; `#[ignore]`-Konvention | `tests/som_bench.rs`, `src/gpu/som.rs` |
| **5. Doku** | README-Tabelleneintrag (`--gpu` = Filter **und** SOM); dieses Dokument auf "Implementiert" stellen | `README.md`, `docs/som_gpu_plan.md` |

---

## 8. Ausblick (bewusst ausserhalb dieses Plans)

* **Dekodierung auf GPU:** `reconstruct_image_from_codebook` als Lookup-Kernel — erst sinnvoll, wenn Decode zum Engpass wird.
* **Batch-SOM-Training auf GPU:** parallele neuronale Updates mit Sigma-gating pro Sample-Batch.
* **Sperren der Gewichte als `Vec<f32>` in `SomMatrix`:** flaches Original-Layout statt `Vec<Vec<f32>>` wuerde GPU-Upload und CPU-Pfad beschleunigen.
