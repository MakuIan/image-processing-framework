# Image Processing Framework

Ein hochperformantes Framework zur Bild- und Videoverarbeitung in **Rust** mit modularer Streaming-Pipeline, CPU-Multithreading via [Rayon](https://crates.io/crates/rayon), GPU-Compute-Shader-Beschleunigung via [wgpu](https://crates.io/crates/wgpu) (Vulkan / DirectX 12 / Metal) und **SOM-basierter Videokompression** (Kohonen Self-Organizing Maps, Vektorquantisierung und Entropiekodierung).

---

## Inhaltsverzeichnis

- [Systemanforderungen & Installation](#systemanforderungen--installation)
- [Projekt- & Modulstruktur](#projekt---modulstruktur)
- [Framework starten](#framework-starten)
- [CLI-Referenz](#cli-referenz)
- [SOM-Videokompression (Neuronale Vektorquantisierung)](#som-videokompression-neuronale-vektorquantisierung)
- [Praxisbeispiele](#praxisbeispiele)
  - [Bildverarbeitung](#bildverarbeitung)
  - [Videoverarbeitung & Filter](#videoverarbeitung--filter)
  - [SOM-Videokompression & Qualitätsuntersuchung](#som-videokompression--qualitätsuntersuchung)
- [Filter, Architektur & Qualitätsmetriken](#filter-architektur--qualitätsmetriken)

---

## Systemanforderungen & Installation

Bevor das Framework ausgeführt werden kann, muss folgendes auf dem System installiert sein:

### 1. Rust Toolchain (Pflicht)
Das Framework basiert auf **Rust Edition 2024** (Mindestversion: **1.99.0**).
Die Toolchain ist über die enthaltene [`rust-toolchain.toml`](rust-toolchain.toml) fixiert (`1.99.0` inkl. `rustfmt` und `clippy`).

- **Installation:** Über das offizielle Setup-Tool [rustup.rs](https://rustup.rs/):
  - **Windows:** Den Installer `rustup-init.exe` herunterladen und ausführen (erfordert die Visual Studio C++ Build Tools).
  - **Linux / macOS:**
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```
- **Toolchain 1.99.0 installieren / aktualisieren:**
  ```bash
  rustup toolchain install 1.99.0
  ```
- **Prüfen:**
  ```bash
  cargo --version
  rustc --version
  ```

### 2. FFmpeg & ffprobe (Erforderlich für Videos)
Für das Dekodieren und Enkodieren von Videos (`mp4`, `avi`, `mov`, etc.) wird FFmpeg benötigt. Die Befehle `ffmpeg` und `ffprobe` müssen im System-`PATH` verfügbar sein.
- **Windows:**
  - Via Windows Package Manager (Winget):
    ```powershell
    winget install Gyan.FFmpeg
    ```
  - Oder via Chocolatey: `choco install ffmpeg`
  - Oder manuell von [ffmpeg.org](https://ffmpeg.org/download.html) herunterladen und den `bin`-Ordner in die Systemumgebungsvariable `PATH` eintragen.
- **Linux (Ubuntu / Debian):**
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
- **macOS:**
  ```bash
  brew install ffmpeg
  ```
- **Prüfen:**
  ```bash
  ffmpeg -version
  ffprobe -version
  ```

### 3. GPU-Treiber (Optional, für `--gpu`)
Für die GPU-Beschleunigung nutzt das Framework standardkonforme WGSL Compute-Shader über `wgpu`.
- **Voraussetzung:** Ein Grafiktreiber mit Unterstützung für **DirectX 12** oder **Vulkan** (Windows/Linux) bzw. **Metal** (macOS).
- Kompatibel mit modernen GPUs von **AMD**, **NVIDIA** und **Intel**.

---

### Wo sollen die Eingabedateien hin?
- Lege deine Testbilder oder -videos in den Ordner:
  ```text
  assets/input/
  ```
  *(Beispiel: `assets/input/test.png` oder `assets/input/test.mp4`)*

### Unterstützte Dateiformate
- **Bilder:** `.png`, `.jpg`, `.jpeg`
- **Videos:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

---

## Framework starten

Das Framework wird über `cargo` kompiliert und gestartet:

```powershell
# Im Entwicklungsmodus
cargo run -- <INPUT> [OPTIONEN]

# Im Release-Modus (EMPFEHLUNG)
cargo run --release -- <INPUT> [OPTIONEN]
```

> [!TIP]
> **Performance-Tipp:** Für die Bild- und insbesondere Videoverarbeitung empfiehlt sich stets das Flag `--release`, da die Rust-Compiler-Optimierungen die CPU-Verarbeitung um ein Vielfaches beschleunigen.

---

## CLI-Referenz

### Syntax
```text
cargo run --release -- <INPUT> [OPTIONEN]
```

### Argumente
| Argument | Typ | Beschreibung |
| :--- | :--- | :--- |
| `<INPUT>` | Pfad (Pflicht) | Pfad zur Eingabedatei (Bild oder Video). Beispiel: `assets/input/test_input.png` |

### Allgemeine Optionen & Filter
| Option / Flag | Standardwert | Beschreibung |
| :--- | :--- | :--- |
| `-o`, `--output <PFAD>` | `assets/output/<stem>_processed.<ext>` | Individueller Zielpfad für die Ausgabedatei. |
| `--gpu` | `false` (CPU-Modus) | Aktiviert **wgpu Compute-Shader** auf der GPU (Vulkan / DX12 / Metal). Standardmäßig rechnet die CPU parallel via Rayon. |
| `--noise` | `false` | Wendet Gaussian-Noise (normalverteiltes Rauschen) an. |
| `--noise-sigma <FLOAT>` | `15.0` | Standardabweichung / Stärke des Rauschens. |
| `--noise-mean <FLOAT>` | `0.0` | Mittelwert des Rauschens. |
| `--blur` | `false` | Wendet Gaussian-Blur (Weichzeichner) an. |
| `--blur-sigma <FLOAT>` | `1.0` | Stärke der Weichzeichnung (Sigma). |
| `--bilateral` | `false` | Wendet einen Bilateral-Filter (kantenerhaltender Glätter) an. |
| `--bilateral-radius <INT>` | `3` | Suchradius in Pixeln. |
| `--bilateral-color <FLOAT>`| `50.0` | Farbtoleranz (σ_color) – Schwellenwert für Kanten. |
| `--bilateral-space <FLOAT>`| `50.0` | Raumdistanz (σ_space) – räumliche Gewichtung. |
| `--encoder <NAME>` | `auto` | Video-Encoder für FFmpeg: `auto`, `amf` (AMD), `nvenc` (NVIDIA), `qsv` (Intel), `cpu` (libx264). |
| `--max-frames <ANZAHL>` | `None` (alle) | Begrenzt die Frame-Anzahl bei Videos (ideal für Tests). |
| `-h`, `--help` | - | Zeigt die Hilfe und alle CLI-Parameter an. |
| `-V`, `--version` | - | Gibt die Versionsnummer des Frameworks aus. |

> [!NOTE]
> **Filter-Aktivierungslogik:**
> - **Keine Filter-Flags gesetzt:** Das Framework wendet automatisch die Standard-Pipeline (`Noise` ➔ `Bilateral` ➔ `Blur`) an.
> - **Filter explizit auswählen:** Sobald mindestens ein Filter oder ein Filterparameter gesetzt wird (z. B. `--blur` oder `--noise-sigma 20.0`), werden **nur** die ausgewählten Filter ausgeführt.
> - **Implizite Aktivierung:** Ein Parameter wie `--blur-sigma 2.5` aktiviert den Blur-Filter automatisch, ohne dass man zwingend zusätzlich `--blur` angeben muss.

### SOM-Kompression (Videoverfahren)
| Option / Flag | Standardwert | Beschreibung |
| :--- | :--- | :--- |
| `--som` | `false` | Aktiviert das neuronale SOM-Kompressionsexperiment für Videos. |
| `--som-block-size <INT>` | `4` | Kantenlänge quadratischer Kacheln (z. B. `4` für 4x4 oder `8` für 8x8 Pixel). |
| `--som-grid <INT>` | `16` | Gittergröße des SOM (z. B. `16` für 16x16 = 256 Neuronen, oder `8` für 64 Neuronen). |
| `--som-threshold <FLOAT>`| `28.0` | PSNR-Schwellenwert in dB. Unterschreitet ein Folgeframe diese Qualität, wird ein neuer Keyframe trainiert. |
| `--som-epochs <INT>` | `5` | Anzahl der Kohonen-Trainings-Epochen pro Keyframe. |

---

## SOM-Videokompression (Neuronale Vektorquantisierung)

Das Modul nutzt ein klassisches Verfahren der neuronalen Bild- und Videokompression zur Reduktion von Redundanzen:

1. **Räumliche Redundanz (Intra-Frame):**  
   Bilder werden in Kacheln zerlegt (z. B. 4x4 Pixel mit 3 Kanälen = 48 Dimensionen). Ein zweidimensionales Kohonen-Gitter (SOM) lernt ein repräsentatives Codebuch. Jeder Block wird durch die Best Matching Unit (BMU) approximiert und als 1-Byte-Index (`u8`) dargestellt.
2. **Topologische Ordnung & Entropiekodierung:**  
   Da Nachbarneuronen ähnliche Bildmuster darstellen, weisen die quantisierten Blockindizes hohe lokale Redundanz auf. Diese wird durch **Run-Length-Encoding (RLE)** und anschließende **Deflate-Kompression (`flate2`)** verlustfrei komprimiert.
3. **Zeitliche Redundanz & Keyframe-Entscheidung (Inter-Frame):**  
   Frame 0 startet als Keyframe mit neu trainiertem Codebuch. Für jeden Folgeframe wird das bestehende Codebuch getestet:
   - Erreicht die Rekonstruktion mindestens den Schwellenwert (z. B. `28.0 dB` PSNR), wird das SOM beibehalten und nur der kompakte Index-Datenstrom übertragen.
   - Sinkt die Qualität (z. B. durch Szenenwechsel oder starke Bewegung), wird ein neuer Keyframe getriggert, ein neues SOM trainiert und die Codebuch-Gewichte übertragen.

---

## Praxisbeispiele

### Bildverarbeitung
```powershell
# Bild mit Standard-Pipeline filtern
cargo run --release -- assets/input/test.png

# Bild mit spezifischem Rausch- und Glättungsfilter
cargo run --release -- assets/input/test.png --noise-sigma 20.0 --bilateral-radius 5
```

### Videoverarbeitung & Filter
```powershell
# Erste 60 Frames eines Videos filtern
cargo run --release -- assets/input/video.mp4 --max-frames 60 --blur-sigma 2.0
```

### SOM-Videokompression & Qualitätsuntersuchung

```powershell
# Standard-Kompression (4x4 Blöcke, 16x16 SOM = 256 Neuronen, 28 dB Schwelle)
cargo run --release -- assets/input/video.mp4 --som --max-frames 30

# Schnelleres Training mit kleinerem Codebuch (8x8 SOM = 64 Neuronen)
cargo run --release -- assets/input/video.mp4 --som --som-grid 8 --max-frames 30

# Größere Kacheln (8x8 Blöcke) für höhere Kompression bei geringerer Detailtreue
cargo run --release -- assets/input/video.mp4 --som --som-block-size 8 --max-frames 30

# Strengere Qualitätsschwelle (32 dB -> führt zu mehr Keyframes bei Szenenwechseln)
cargo run --release -- assets/input/video.mp4 --som --som-threshold 32.0 --max-frames 30
```

---

## Filter, Architektur & Qualitätsmetriken

### Enthaltene Filter
1. **Gaussian Noise Filter** (`GaussianNoiseFilter` / `GpuGaussianNoiseFilter`):  
   Fügt normalverteiltes Zufallsrauschen hinzu. Auf der CPU via Box-Muller-Transformation, auf der GPU via WGSL-Hash & Box-Muller Compute-Shader.
2. **Bilateral Filter** (`BilateralFilter` / `GpuBilateralFilter`):  
   Glättet Oberflächenrauschen unter Erhaltung harter Kanten.
   - **CPU:** Berechnet vorab eine Look-Up-Table (LUT) für Farbgewichte und verteilt die Zeilen via Rayon.
   - **GPU:** 2D-Workgroup-Compute-Shader (16 × 16), bei dem Pixel parallel in Hardware berechnet werden.
3. **Gaussian Blur Filter** (`BlurFilter` / `GpuBlurFilter`):  
   Klassischer Gaußscher Weichzeichner.

### Zentrale Standardwerte (`FilterConfig`)
Alle Standardwerte sind zentral in `src/filters/config.rs` als Konstanten definiert:
- `DEFAULT_NOISE_MEAN = 0.0`
- `DEFAULT_NOISE_SIGMA = 15.0`
- `DEFAULT_BILATERAL_RADIUS = 3`
- `DEFAULT_BILATERAL_COLOR = 50.0`
- `DEFAULT_BILATERAL_SPACE = 50.0`
- `DEFAULT_BLUR_SIGMA = 1.0`

### Evaluierung & Qualitätsmetriken
Das Framework berechnet automatisch etablierte Maße zur Bewertung der Übertragungs- und Rekonstruktionsqualität:
- **PSNR (Peak Signal-to-Noise Ratio):** Maß für den Signal-zu-Rausch-Abstand in Dezibel (dB).
- **SSIM (Structural Similarity Index):** Strukturelle Ähnlichkeit zwischen Original und Ergebnis bzw. rekonstruiertem Frame (0.0 bis 1.0).
- **Kompressionsfaktor (CR):** Verhältnis zwischen Rohdaten-Bytegröße und komprimierter Index-Bytegröße (`CR = Raw / Compressed`).
- **Diff-Heatmap (`*_diff_heatmap.png`):** Farbkodierte Abweichungskarte bei Einzelbildern (Schwarz/Blau = identisch, Rot/Gelb = hohe Abweichung).
