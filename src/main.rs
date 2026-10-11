use std::path::{Path, PathBuf};

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use image_processing_framework::{
    calculate_psnr, calculate_ssim, generate_diff_map, FilterConfig, ImageSink, ImageSource,
    Pipeline, Sink, Source, VideoEncoder, VideoSink, VideoSource,
};

// --- Standard-Konstanten fuer Pfade und CLI ---
const ASSETS_DIR: &str = "assets";
const OUTPUT_DIR: &str = "output";
const DEFAULT_FILE_STEM: &str = "output";
const DEFAULT_IMAGE_EXT: &str = "png";
const PROCESSED_SUFFIX: &str = "_processed";
const DIFF_HEATMAP_SUFFIX: &str = "_diff_heatmap";
const DEFAULT_ENCODER: &str = "auto";

/// Image Processing Framework – Bild- und Video-Verarbeitung über die CLI
#[derive(Parser)]
#[command(name = "image-processing-framework", version, about)]
struct Cli {
    /// Eingabedatei (Bild: png/jpg/jpeg, Video: mp4/avi/mov/mkv/webm)
    input: PathBuf,

    /// Ausgabepfad (standardmäßig automatisch in 'assets/output/' generiert)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Gaussian-Noise-Filter anwenden
    #[arg(long)]
    noise: bool,

    /// Stärke des Rauschens (Standard: 15.0)
    #[arg(long)]
    noise_sigma: Option<f64>,

    /// Mittelwert des Rauschens (Standard: 0.0)
    #[arg(long)]
    noise_mean: Option<f64>,

    /// Blur-Filter (Weichzeichner) anwenden
    #[arg(long)]
    blur: bool,

    /// Stärke der Weichzeichnung / Sigma (Standard: 1.0)
    #[arg(long)]
    blur_sigma: Option<f32>,

    /// Bilateral-Filter (Kantenerhaltender Glätter) anwenden
    #[arg(long)]
    bilateral: bool,

    /// Bilateral-Filter Radius in Pixeln (Standard: 3)
    #[arg(long)]
    bilateral_radius: Option<i32>,

    /// Bilateral-Filter Farbtoleranz / Sigma Color (Standard: 50.0)
    #[arg(long)]
    bilateral_color: Option<f64>,

    /// Bilateral-Filter Raumdistanz / Sigma Space (Standard: 50.0)
    #[arg(long)]
    bilateral_space: Option<f64>,

    /// Video-Encoder für FFmpeg (auto, amf, nvenc, qsv, cpu)
    #[arg(long, default_value = DEFAULT_ENCODER)]
    encoder: String,

    /// Maximale Anzahl Frames verarbeiten (optional zum Testen)
    #[arg(long)]
    max_frames: Option<usize>,

    /// GPU-Compute-Shader (wgpu) für Filter verwenden (AMD/Nvidia/Intel)
    #[arg(long)]
    gpu: bool,
}

/// Klassifiziert die Eingabedatei anhand ihrer Dateiendung
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputKind {
    Image,
    Video,
    Unknown,
}

fn detect_input_kind(path: &Path) -> InputKind {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());

    match ext.as_deref() {
        Some("png") | Some("jpg") | Some("jpeg") => InputKind::Image,
        Some("mp4") | Some("avi") | Some("mov") | Some("mkv") | Some("webm") => InputKind::Video,
        _ => InputKind::Unknown,
    }
}

/// Generiert den Ausgabepfad standardmäßig nach `assets/output/<stem>_processed.<ext>`
/// Falls der Input nicht in assets/input liegt, bleibt der Zielordner assets/output.
fn auto_output_path(input_path: &Path) -> PathBuf {
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(DEFAULT_FILE_STEM);
    let ext = input_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or(DEFAULT_IMAGE_EXT);

    PathBuf::from(ASSETS_DIR)
        .join(OUTPUT_DIR)
        .join(format!("{}{}.{}", stem, PROCESSED_SUFFIX, ext))
}

/// Ermittelt den Pfad für die Diff-Heatmap passend zum Ausgabepfad
fn diff_heatmap_path(output_path: &Path) -> PathBuf {
    let stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("diff")
        .trim_end_matches(PROCESSED_SUFFIX);
    let ext = output_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or(DEFAULT_IMAGE_EXT);

    let parent = output_path.parent().unwrap_or(Path::new("."));
    parent.join(format!("{}{}.{}", stem, DIFF_HEATMAP_SUFFIX, ext))
}

/// Baut die Pipeline anhand der gesetzten CLI-Filter-Flags auf
fn build_pipeline(cli: &Cli) -> Pipeline {
    if cli.gpu {
        println!("  Modus: GPU-Beschleunigung aktiv (wgpu Compute-Shader auf Vulkan/DX12)");
    } else {
        println!("  Modus: CPU-Verarbeitung (Rayon Multithreading)");
    }

    let use_noise = cli.noise || cli.noise_sigma.is_some() || cli.noise_mean.is_some();
    let use_bilateral = cli.bilateral
        || cli.bilateral_radius.is_some()
        || cli.bilateral_color.is_some()
        || cli.bilateral_space.is_some();
    let use_blur = cli.blur || cli.blur_sigma.is_some();

    let any_filter_set = use_noise || use_bilateral || use_blur;

    let filter_configs: Vec<FilterConfig> = if any_filter_set {
        let mut configs = Vec::new();

        if use_noise {
            configs.push(FilterConfig::GaussianNoise {
                mean: cli.noise_mean.unwrap_or(FilterConfig::DEFAULT_NOISE_MEAN),
                sigma: cli.noise_sigma.unwrap_or(FilterConfig::DEFAULT_NOISE_SIGMA),
            });
        }
        if use_bilateral {
            configs.push(FilterConfig::Bilateral {
                radius: cli.bilateral_radius.unwrap_or(FilterConfig::DEFAULT_BILATERAL_RADIUS),
                sigma_color: cli.bilateral_color.unwrap_or(FilterConfig::DEFAULT_BILATERAL_COLOR),
                sigma_space: cli.bilateral_space.unwrap_or(FilterConfig::DEFAULT_BILATERAL_SPACE),
            });
        }
        if use_blur {
            configs.push(FilterConfig::Blur {
                sigma: cli.blur_sigma.unwrap_or(FilterConfig::DEFAULT_BLUR_SIGMA),
            });
        }

        configs
    } else {
        println!(
            "  Keine Filter-Flags gesetzt – verwende Standard-Pipeline (noise + bilateral + blur)."
        );
        FilterConfig::default_pipeline()
    };

    let mut pipeline = Pipeline::new();
    for config in filter_configs {
        pipeline = pipeline.add_boxed_filter(config.instantiate(cli.gpu));
    }
    pipeline
}

/// Führt die Bild-Pipeline für eine einzelne Eingabedatei durch
fn run_image_pipeline(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let input_path = &cli.input;

    if !input_path.is_file() {
        eprintln!("Fehler: Eingabedatei {:?} nicht gefunden!", input_path);
        return Err("Input file not found".into());
    }

    let output_path = match &cli.output {
        Some(p) => p.clone(),
        None => auto_output_path(input_path),
    };
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    println!("=== Bild-Pipeline ===\n");
    println!("Input:  {:?}", input_path);
    println!("Output: {:?}", output_path);

    // Step 1: Original-Bild über ImageSource laden
    println!("Step 1: Bild laden via ImageSource...");
    let mut source = ImageSource::new(input_path);
    let original_frame = source
        .next_frame()
        .ok_or_else(|| format!("Konnte Bild nicht laden: {:?}", input_path))?;
    source.release();

    println!(
        "  Geladen: {}x{} (Index: {}, Timestamp: {:.2}s)",
        original_frame.width(),
        original_frame.height(),
        original_frame.index(),
        original_frame.timestamp()
    );

    // Step 2: Pipeline anhand der CLI-Flags aufbauen
    println!("\nStep 2: Pipeline aufbauen...");
    let pipeline = build_pipeline(cli);

    for (idx, filter) in pipeline.filters().iter().enumerate() {
        println!("  Filter {}: {}", idx + 1, filter.name());
    }

    // Step 3: Pipeline ausführen
    println!("\nStep 3: Pipeline ausführen...");
    let processed_frame = pipeline.process_frame(original_frame.clone());

    // Metadaten der verarbeiteten Frames anzeigen
    println!("  Angewandte Filter aus Frame-Metadaten:");
    for (key, value) in processed_frame.metadata() {
        println!("    {}: {}", key, value);
    }

    // Step 4: Ergebnis über ImageSink speichern
    println!("\nStep 4: Speichere Ergebnis via ImageSink...");
    let mut sink = ImageSink::new(&output_path);
    sink.write(&processed_frame)?;
    sink.release();
    println!("  Verarbeitetes Bild: {:?}", output_path);

    // Diff-Heatmap berechnen und passend zum Ausgabepfad speichern
    let diff_path = diff_heatmap_path(&output_path);
    let diff_frame = generate_diff_map(&original_frame, &processed_frame);
    let mut diff_sink = ImageSink::new(&diff_path);
    diff_sink.write(&diff_frame)?;
    diff_sink.release();
    println!("  Diff-Heatmap: {:?}", diff_path);

    // Step 5: Qualitätsmetriken berechnen
    println!("\nStep 5: Qualitätsmetriken berechnen...");
    let psnr = calculate_psnr(&original_frame, &processed_frame);
    println!(
        "  PSNR: {:.2} dB",
        if psnr.is_infinite() {
            f64::INFINITY
        } else {
            psnr
        }
    );

    let ssim = calculate_ssim(&original_frame, &processed_frame);
    println!("  SSIM: {:.4}", ssim);

    println!("\n=== Bild-Pipeline erfolgreich abgeschlossen ===");
    Ok(())
}

/// Führt die Video-Pipeline für eine einzelne Eingabedatei durch
fn run_video_pipeline(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let input_path = &cli.input;

    if !input_path.is_file() {
        eprintln!("Fehler: Eingabedatei {:?} nicht gefunden!", input_path);
        return Err("Input file not found".into());
    }

    let output_path = match &cli.output {
        Some(p) => p.clone(),
        None => auto_output_path(input_path),
    };
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    println!("=== Video-Pipeline ===\n");
    println!("Input:  {:?}", input_path);
    println!("Output: {:?}", output_path);

    println!("Step 1: Initialisiere VideoSource...");
    let mut source = VideoSource::new(input_path)?;
    println!(
        "  Video-Eigenschaften: {}x{} @ {:.2} FPS{}",
        source.width(),
        source.height(),
        source.fps(),
        if let Some(total) = source.total_frames() {
            format!(" (ca. {} Frames)", total)
        } else {
            String::new()
        }
    );

    let encoder = VideoEncoder::from_str_loose(&cli.encoder);
    let mut sink = VideoSink::new(&output_path, source.fps() as f32).with_encoder(encoder);

    println!("\nStep 2: Pipeline aufbauen...");
    let mut pipeline = build_pipeline(cli);

    for (idx, filter) in pipeline.filters().iter().enumerate() {
        println!("  Filter {}: {}", idx + 1, filter.name());
    }

    println!("\nStep 3: Streaming-Verarbeitung ausführen...");

    let total_frames = source.total_frames();
    let display_total = match (total_frames, cli.max_frames) {
        (Some(total), Some(limit)) => Some(total.min(limit)),
        (Some(total), None) => Some(total),
        (None, Some(limit)) => Some(limit),
        (None, None) => None,
    };

    let pb = if let Some(total) = display_total {
        let bar = ProgressBar::new(total as u64);
        bar.set_draw_target(indicatif::ProgressDrawTarget::stdout_with_hz(5));
        bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:30.cyan/blue}] {pos}/{len} ({per_sec}, ETA {eta})")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("#>-"),
        );
        bar
    } else {
        let bar = ProgressBar::new_spinner();
        bar.set_draw_target(indicatif::ProgressDrawTarget::stdout_with_hz(5));
        bar.set_style(
            ProgressStyle::default_spinner()
                .template("[{elapsed_precise}] {pos} Frames ({per_sec})")
                .unwrap_or_else(|_| ProgressStyle::default_spinner()),
        );
        bar
    };

    let pb_clone = pb.clone();
    let processed_count = pipeline.run_with_progress(
        &mut source,
        &mut sink,
        cli.max_frames,
        move |_count| {
            pb_clone.inc(1);
        },
    )?;

    pb.finish_and_clear();
    println!("  {} Frame(s) erfolgreich verarbeitet!", processed_count);

    sink.release();

    if output_path.is_file() {
        let meta = std::fs::metadata(&output_path)?;
        println!("  Ausgabedatei: {:?} ({} Bytes)", output_path, meta.len());
    }

    println!("\n=== Video-Pipeline erfolgreich abgeschlossen ===");
    Ok(())
}

fn main() {
    let cli = Cli::parse();

    // Dateityp anhand der Dateiendung bestimmen
    match detect_input_kind(&cli.input) {
        InputKind::Image => {
            if let Err(e) = run_image_pipeline(&cli) {
                eprintln!("Bild-Pipeline fehlgeschlagen: {}", e);
                std::process::exit(1);
            }
        }
        InputKind::Video => {
            if let Err(e) = run_video_pipeline(&cli) {
                eprintln!("Video-Pipeline fehlgeschlagen: {}", e);
                eprintln!("Hinweis: Stelle sicher, dass 'ffmpeg' und 'ffprobe' installiert sind.");
                std::process::exit(1);
            }
        }
        InputKind::Unknown => {
            eprintln!(
                "Fehler: Unbekannte Dateiendung für {:?}. \
                 Unterstützt werden: png, jpg, jpeg (Bild) oder mp4, avi, mov, mkv, webm (Video).",
                cli.input
            );
            std::process::exit(1);
        }
    }
}
