use std::path::{Path, PathBuf};

use clap::Parser;
use image_processing_framework::{
    FilterConfig, ImageSink, ImageSource, Pipeline, Sink, SomVideoConfig, Source, VideoCodec,
    VideoEncoder, VideoSink, VideoSource, calculate_psnr, calculate_ssim, create_progress_bar,
    create_spinner, create_task_progress, decode_video_stream_with_progress,
    encode_video_stream_with_progress, generate_diff_map,
};

// --- Standard-Konstanten fuer Pfade und CLI ---
const ASSETS_DIR: &str = "assets";
const OUTPUT_DIR: &str = "output";
const DEFAULT_FILE_STEM: &str = "output";
const DEFAULT_IMAGE_EXT: &str = "png";
const PROCESSED_SUFFIX: &str = "_processed";
const DIFF_HEATMAP_SUFFIX: &str = "_diff_heatmap";
const DEFAULT_ENCODER: &str = "auto";
const DEFAULT_CODEC: &str = "h265";

/// Image Processing Framework - Bild- und Video-Verarbeitung ueber die CLI
#[derive(Parser)]
#[command(name = "image-processing-framework", version, about)]
struct Cli {
    /// Eingabedatei (Bild: png/jpg/jpeg, Video: mp4/avi/mov/mkv/webm)
    input: PathBuf,

    /// Ausgabepfad (standardmaessig automatisch in 'assets/output/' generiert)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Gaussian-Noise-Filter anwenden
    #[arg(long)]
    noise: bool,

    /// Staerke des Rauschens (Standard: 15.0)
    #[arg(long)]
    noise_sigma: Option<f64>,

    /// Mittelwert des Rauschens (Standard: 0.0)
    #[arg(long)]
    noise_mean: Option<f64>,

    /// Blur-Filter (Weichzeichner) anwenden
    #[arg(long)]
    blur: bool,

    /// Staerke der Weichzeichnung / Sigma (Standard: 1.0)
    #[arg(long)]
    blur_sigma: Option<f32>,

    /// Bilateral-Filter (Kantenerhaltender Glaetter) anwenden
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

    /// Video-Encoder fuer FFmpeg (auto, amf, nvenc, qsv, cpu)
    #[arg(long, default_value = DEFAULT_ENCODER)]
    encoder: String,

    /// Video-Codec fuer FFmpeg (h265/hevc oder h264/avc, Standard: h265)
    #[arg(long, default_value = DEFAULT_CODEC)]
    codec: String,

    /// Optionaler CRF/QP-Wert fuer den Video-Encoder (Standard: 32 fuer HEVC, 26 fuer H264)
    #[arg(long)]
    crf: Option<u32>,

    /// Maximale Anzahl Frames verarbeiten (optional zum Testen)
    #[arg(long)]
    max_frames: Option<usize>,

    /// GPU-Compute-Shader (wgpu) fuer Filter und SOM-Quantisierung verwenden (AMD/Nvidia/Intel)
    #[arg(long)]
    gpu: bool,

    // --- SOM Video Kompression Flags ---
    /// Aktiviert das SOM-Kompressionsexperiment fuer Videos
    #[arg(long)]
    som: bool,

    /// Kantenlaenge der quadratischen Kacheln / Bloecke (Standard: 4 fuer 4x4)
    #[arg(long, default_value_t = 4)]
    som_block_size: usize,

    /// Gittergroesse des SOM (Standard: 16 -> 16x16 = 256 Neuronen, alternativ z. B. 8 -> 64)
    #[arg(long, default_value_t = 16)]
    som_grid: usize,

    /// PSNR-Schwellenwert in dB fuer Keyframe-Generierung (Standard: 28.0 dB).
    /// Ist die Schwelle fuer das Material unerreichbar, senkt sie sich
    /// automatisch relativ zum letzten Keyframe (minus 2.5 dB Toleranz) ab;
    /// unendlich erzwingt jeden Frame als Keyframe
    #[arg(long, default_value_t = 28.0)]
    som_threshold: f64,

    /// Anzahl der Trainings-Epochen pro Keyframe (Standard: 5)
    #[arg(long, default_value_t = 5)]
    som_epochs: usize,
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

/// Generiert den Ausgabepfad standardmaessig nach `assets/output/<stem>_processed.<ext>`
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

/// Ermittelt den Pfad fuer die Diff-Heatmap passend zum Ausgabepfad
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
                radius: cli
                    .bilateral_radius
                    .unwrap_or(FilterConfig::DEFAULT_BILATERAL_RADIUS),
                sigma_color: cli
                    .bilateral_color
                    .unwrap_or(FilterConfig::DEFAULT_BILATERAL_COLOR),
                sigma_space: cli
                    .bilateral_space
                    .unwrap_or(FilterConfig::DEFAULT_BILATERAL_SPACE),
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
            "  Keine Filter-Flags gesetzt - verwende Standard-Pipeline (noise + bilateral + blur)."
        );
        FilterConfig::default_pipeline()
    };

    let mut pipeline = Pipeline::new();
    for config in filter_configs {
        pipeline = pipeline.add_boxed_filter(config.instantiate(cli.gpu));
    }
    pipeline
}

/// Fuehrt die Bild-Pipeline fuer eine einzelne Eingabedatei durch
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

    // Step 1: Original-Bild ueber ImageSource laden
    println!("Step 1: Bild laden via ImageSource...");
    let pb_load = create_spinner("Lade Bilddatei...");
    let mut source = ImageSource::new(input_path);
    let original_frame = source
        .next_frame()
        .ok_or_else(|| format!("Konnte Bild nicht laden: {:?}", input_path))?;
    source.release();
    pb_load.finish_and_clear();

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

    // Step 3: Pipeline ausfuehren
    println!("\nStep 3: Pipeline ausfuehren...");
    let filter_count = pipeline.filters().len();
    let pb_filters = create_progress_bar(filter_count as u64, "Wende Filter an");
    let processed_frame =
        pipeline.process_frame_with_progress(original_frame.clone(), |idx, total, name| {
            pb_filters.set_message(format!("Filter {idx}/{total}: {name}"));
            pb_filters.inc(1);
        });
    pb_filters.finish_and_clear();

    // Metadaten der verarbeiteten Frames anzeigen
    println!("  Angewandte Filter aus Frame-Metadaten:");
    for (key, value) in processed_frame.metadata() {
        println!("    {}: {}", key, value);
    }

    // Step 4: Ergebnis ueber ImageSink speichern
    println!("\nStep 4: Speichere Ergebnis via ImageSink...");
    let pb_sink = create_progress_bar(2, "Speichere Bilddateien");
    pb_sink.set_message("Schreibe verarbeitetes Bild...");
    let mut sink = ImageSink::new(&output_path);
    sink.write(&processed_frame)?;
    sink.release();
    pb_sink.inc(1);

    pb_sink.set_message("Schreibe Diff-Heatmap...");
    let diff_path = diff_heatmap_path(&output_path);
    let diff_frame = generate_diff_map(&original_frame, &processed_frame);
    let mut diff_sink = ImageSink::new(&diff_path);
    diff_sink.write(&diff_frame)?;
    diff_sink.release();
    pb_sink.inc(1);
    pb_sink.finish_and_clear();

    println!("  Verarbeitetes Bild: {:?}", output_path);
    println!("  Diff-Heatmap:       {:?}", diff_path);

    // Step 5: Qualitaetsmetriken berechnen
    println!("\nStep 5: Qualitaetsmetriken berechnen...");
    let pb_metrics = create_progress_bar(2, "Berechne Metriken");
    pb_metrics.set_message("Berechne PSNR...");
    let psnr = calculate_psnr(&original_frame, &processed_frame);
    pb_metrics.inc(1);

    pb_metrics.set_message("Berechne SSIM...");
    let ssim = calculate_ssim(&original_frame, &processed_frame);
    pb_metrics.inc(1);
    pb_metrics.finish_and_clear();

    println!(
        "  PSNR: {:.2} dB",
        if psnr.is_infinite() {
            f64::INFINITY
        } else {
            psnr
        }
    );
    println!("  SSIM: {:.4}", ssim);

    println!("\n=== Bild-Pipeline erfolgreich abgeschlossen ===");
    Ok(())
}

/// Fuehrt das SOM-Video-Kompressionsexperiment durch
fn run_som_video_pipeline(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
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

    println!("=== SOM-Video-Kompressionsexperiment ===\n");
    println!("Input:  {:?}", input_path);
    println!("Output: {:?}", output_path);

    let config = SomVideoConfig::new(
        cli.som_block_size,
        cli.som_grid,
        cli.som_threshold,
        cli.som_epochs,
    )
    .with_gpu(cli.gpu);

    println!("Codec-Konfiguration:");
    if config.gpu {
        println!("  SOM-Modus:            GPU-Beschleunigung aktiv (wgpu)");
    } else {
        println!("  SOM-Modus:            CPU (SIMD + Rayon)");
    }
    println!(
        "  Blockgroesse:         {}x{} Pixel",
        config.block_size, config.block_size
    );
    println!(
        "  SOM-Gitter:          {}x{} ({} Neuronen)",
        config.som_grid_size,
        config.som_grid_size,
        config.som_grid_size * config.som_grid_size
    );
    println!("  PSNR-Schwellenwert:  {:.1} dB", config.psnr_threshold_db);
    println!(
        "  Relativer Margin:    {:.1} dB (Keyframe bei PSNR-Abfall zum letzten Keyframe)",
        config.psnr_relative_margin_db
    );
    println!("  Kohonen-Epochen:     {}", config.train_epochs);

    println!("\nStep 1: Frames aus Video laden...");
    let mut source = VideoSource::new(input_path)?;
    let fps = source.fps();
    let width = source.width();
    let height = source.height();

    let total_hint = source.total_frames().map(|t| match cli.max_frames {
        Some(limit) => t.min(limit),
        None => t,
    });
    let pb_load = create_task_progress(total_hint, "Lade Frames");
    let mut raw_frames = Vec::new();
    while let Some(frame) = source.next_frame() {
        raw_frames.push(frame);
        pb_load.inc(1);
        if let Some(limit) = cli.max_frames {
            if raw_frames.len() >= limit {
                break;
            }
        }
    }
    source.release();
    pb_load.finish_and_clear();

    if raw_frames.is_empty() {
        return Err("Keine Frames im Video gefunden".into());
    }
    println!(
        "  {} Frame(s) geladen ({}x{} @ {:.2} FPS)",
        raw_frames.len(),
        width,
        height,
        fps
    );

    println!("\nStep 2: SOM-Kompression durchfuehren (Intra- & Inter-Frame Redundanz)...");
    let pb_encode = create_progress_bar(raw_frames.len() as u64, "SOM-Kompression");
    let result = encode_video_stream_with_progress(&raw_frames, &config, |prog| {
        let tag = if prog.is_keyframe {
            "Keyframe"
        } else {
            "Interframe"
        };
        pb_encode.set_message(format!("{tag} | PSNR: {:.1} dB", prog.psnr_db));
        pb_encode.inc(1);
    });
    pb_encode.finish_and_clear();

    println!("\n=== Kompressions- & Qualitaets-Statistik ===");
    println!("  Verarbeitete Frames:         {}", result.stats.total_frames);
    println!(
        "  Keyframes generiert:         {} ({:.1}% aller Frames)",
        result.stats.keyframe_count,
        (result.stats.keyframe_count as f64 / result.stats.total_frames as f64) * 100.0
    );
    println!(
        "  Durchschnittliche PSNR:      {:.2} dB",
        result.stats.avg_psnr_db
    );
    println!(
        "  Unkomprimiert (RGB-Rohdaten):{:.2} MB ({} Bytes)",
        result.stats.raw_size as f64 / 1_048_576.0,
        result.stats.raw_size
    );
    println!(
        "  SOM-Datenstrom (Index+SOM):  {:.2} MB ({} Bytes)",
        result.stats.compressed_size as f64 / 1_048_576.0,
        result.stats.compressed_size
    );
    println!(
        "  CR gegenueber Rohdaten:      {:.2} : 1",
        result.stats.compression_ratio
    );
    if let Ok(meta) = std::fs::metadata(input_path) {
        let input_bytes = meta.len();
        println!(
            "  Original-Videodatei (Input): {:.2} MB ({} Bytes)",
            input_bytes as f64 / 1_048_576.0,
            input_bytes
        );
        let ratio_to_orig = input_bytes as f64 / result.stats.compressed_size as f64;
        let pct_of_orig = (result.stats.compressed_size as f64 / input_bytes as f64) * 100.0;
        println!(
            "  SOM vs. Original-Container:  {:.2} : 1 ({:.1}% der Original-Dateigroesse)",
            ratio_to_orig,
            pct_of_orig
        );
    }

    println!("\nStep 3: Dekodiere Videostrom & schreibe Ausgabevideo zur Qualitaetspruefung...");
    let pb_decode = create_progress_bar(result.frames.len() as u64, "Dekodiere Frames");
    let decoded_frames = decode_video_stream_with_progress(
        &result.frames,
        width,
        height,
        &config,
        fps,
        |_k, _total| {
            pb_decode.inc(1);
        },
    )
    .map_err(|e| format!("Fehler beim Dekodieren: {}", e))?;
    pb_decode.finish_and_clear();

    // Berechne durchschnittlichen SSIM ueber alle dekodierten Frames
    let pb_ssim = create_progress_bar(raw_frames.len() as u64, "Berechne SSIM");
    let mut total_ssim = 0.0;
    for (orig, dec) in raw_frames.iter().zip(decoded_frames.iter()) {
        total_ssim += calculate_ssim(orig, dec);
        pb_ssim.inc(1);
    }
    pb_ssim.finish_and_clear();
    let avg_ssim = total_ssim / raw_frames.len() as f64;
    println!("  Durchschnittlicher SSIM:  {:.4}", avg_ssim);

    // Ausgabe-Video via FFmpeg schreiben
    let encoder = VideoEncoder::from_str_loose(&cli.encoder);
    let codec = VideoCodec::from_str_loose(&cli.codec);
    let mut sink = VideoSink::new(&output_path, fps as f32)
        .with_encoder(encoder)
        .with_codec(codec);
    if let Some(crf) = cli.crf {
        sink.set_crf(Some(crf));
    }

    let pb_write = create_progress_bar(decoded_frames.len() as u64, "Schreibe Video (FFmpeg)");
    for frame in &decoded_frames {
        sink.write(frame)?;
        pb_write.inc(1);
    }
    sink.release();
    pb_write.finish_and_clear();

    if output_path.is_file() {
        let meta = std::fs::metadata(&output_path)?;
        let out_bytes = meta.len();
        println!(
            "  Rekonstruiertes Video:       {:?} ({:.2} MB, {} Bytes)",
            output_path,
            out_bytes as f64 / 1_048_576.0,
            out_bytes
        );
        if let Ok(in_meta) = std::fs::metadata(input_path) {
            let in_bytes = in_meta.len();
            let ratio = in_bytes as f64 / out_bytes as f64;
            println!(
                "  Verhaeltnis Export / Input:  {:.1}% ({:.2} : 1)",
                (out_bytes as f64 / in_bytes as f64) * 100.0,
                ratio
            );
        }
    }

    println!("\n=== SOM-Kompression erfolgreich abgeschlossen ===");
    Ok(())
}

/// Fuehrt die Video-Pipeline fuer eine einzelne Eingabedatei durch
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
    let codec = VideoCodec::from_str_loose(&cli.codec);
    let mut sink = VideoSink::new(&output_path, source.fps() as f32)
        .with_encoder(encoder)
        .with_codec(codec);
    if let Some(crf) = cli.crf {
        sink.set_crf(Some(crf));
    }

    println!("\nStep 2: Pipeline aufbauen...");
    let mut pipeline = build_pipeline(cli);

    for (idx, filter) in pipeline.filters().iter().enumerate() {
        println!("  Filter {}: {}", idx + 1, filter.name());
    }

    println!("\nStep 3: Streaming-Verarbeitung ausfuehren...");

    let total_frames = source.total_frames();
    let display_total = match (total_frames, cli.max_frames) {
        (Some(total), Some(limit)) => Some(total.min(limit)),
        (Some(total), None) => Some(total),
        (None, Some(limit)) => Some(limit),
        (None, None) => None,
    };

    let pb = create_task_progress(display_total, "Streaming-Verarbeitung");
    let pb_clone = pb.clone();
    let processed_count =
        pipeline.run_with_progress(&mut source, &mut sink, cli.max_frames, move |_count| {
            pb_clone.inc(1);
        })?;

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
            if cli.som {
                if let Err(e) = run_som_video_pipeline(&cli) {
                    eprintln!("SOM-Video-Pipeline fehlgeschlagen: {}", e);
                    eprintln!(
                        "Hinweis: Stelle sicher, dass 'ffmpeg' und 'ffprobe' installiert sind."
                    );
                    std::process::exit(1);
                }
            } else if let Err(e) = run_video_pipeline(&cli) {
                eprintln!("Video-Pipeline fehlgeschlagen: {}", e);
                eprintln!("Hinweis: Stelle sicher, dass 'ffmpeg' und 'ffprobe' installiert sind.");
                std::process::exit(1);
            }
        }
        InputKind::Unknown => {
            eprintln!(
                "Fehler: Unbekannte Dateiendung fuer {:?}. \
                 Unterstuetzt werden: png, jpg, jpeg (Bild) oder mp4, avi, mov, mkv, webm (Video).",
                cli.input
            );
            std::process::exit(1);
        }
    }
}
