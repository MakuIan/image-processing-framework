# Image Processing Framework

Ein hochperformantes Framework zur Bild- und Videoverarbeitung in **Rust** mit modularer Streaming-Pipeline, CPU-Multithreading via [Rayon](https://crates.io/crates/rayon) und GPU-Compute-Shader-Beschleunigung via [wgpu](https://crates.io/crates/wgpu) (Vulkan / DirectX 12 / Metal).

---

## Inhaltsverzeichnis

- [Systemanforderungen & Installation](#systemanforderungen--installation)
- [Projekt- & Modulstruktur](#projekt---modulstruktur)
- [Framework starten](#framework-starten)
- [CLI-Referenz](#cli-referenz)
- [Praxisbeispiele](#praxisbeispiele)
  - [Bildverarbeitung](#bildverarbeitung)
  - [Videoverarbeitung](#videoverarbeitung)
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

### Optionen & Flags
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
Bei Bildern berechnet das Framework automatisch:
- **PSNR (Peak Signal-to-Noise Ratio):** Maß für den Signal-zu-Rausch-Abstand in Dezibel (dB).
- **SSIM (Structural Similarity Index):** Strukturelle Ähnlichkeit zwischen Original und Ergebnis (0.0 bis 1.0).
- **Differenz-Heatmap (`*_diff_heatmap.png`):** Farbkodierte Abweichungskarte (Schwarz/Blau = identisch, Rot/Gelb = hohe Abweichung).
