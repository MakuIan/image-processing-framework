mod base;
mod evaluator;
mod filters;
mod frame;
mod io;
mod pipeline;
mod video_sink;
mod video_source;

use image::{Rgb, RgbImage};

use base::{Sink, Source};
use evaluator::{calculate_psnr, calculate_ssim, generate_diff_map};
use filters::{BilateralFilter, BlurFilter, GaussianNoiseFilter};
use io::{ImageSink, ImageSource};
use pipeline::Pipeline;
use video_sink::VideoSink;
use video_source::VideoSource;

/// Erstellt ein Schachbrett-Muster-Bild
fn create_checkerboard(width: u32, height: u32, tile_size: u32) -> RgbImage {
    let mut img = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let row = y / tile_size;
            let col = x / tile_size;
            if (row + col) % 2 == 0 {
                img.put_pixel(x, y, Rgb([255, 255, 255]));
            } else {
                img.put_pixel(x, y, Rgb([128, 128, 128]));
            }
        }
    }
    img
}

/// Führt das Bild-Beispiel durch (originaler Code)
fn run_image_pipeline() {
    println!("=== Bild-Pipeline ===\n");

    // Step 1: Create synthetic checkerboard image (512x512)
    let tile_size = 64u32;
    let original_image = create_checkerboard(512, 512, tile_size);
    let original_path = "test_input.png";

    println!("Step 1: Creating checkerboard image ({}x{})", 512, 512);
    if let Err(e) = original_image.save(original_path) {
        eprintln!("Error saving original image: {}", e);
        return;
    }
    println!("  Saved to: {}", original_path);

    // Step 2: Create ImageSource
    println!("\nStep 2: Creating ImageSource...");
    let mut source = ImageSource::new(original_path);

    let original_frame = match source.next_frame() {
        Some(frame) => frame,
        None => {
            eprintln!("Error loading image!");
            return;
        }
    };
    source.release();
    println!(
        "  Loaded: {}x{}",
        original_frame.width(),
        original_frame.height()
    );

    // Step 3: Build Pipeline with filters
    println!("\nStep 3: Building Pipeline...");
    let mut pipeline = Pipeline::new();

    println!("  Adding GaussianNoiseFilter(mean=0.0, sigma=30.0)...");
    pipeline = pipeline.add_filter(GaussianNoiseFilter::new(0.0, 30.0));

    println!("  Adding BilateralFilter(radius=3, sigma_color=75.0, sigma_space=75.0)...");
    let bilateral = BilateralFilter::new(3, 75.0, 75.0);
    pipeline = pipeline.add_filter(bilateral);

    println!("  Adding BlurFilter(sigma=1.5)...");
    pipeline = pipeline.add_filter(BlurFilter::new(1.5));

    // Step 4: Run Pipeline and save result - pass sink directly as &mut K
    println!("\nStep 4: Running Pipeline...");
    let mut pipeline_source = ImageSource::new(original_path);
    let mut sink = ImageSink::new("test_output.png");

    let processed_frames = pipeline.run(&mut pipeline_source, &mut sink);
    pipeline_source.release();

    if !processed_frames.is_empty() {
        let processed_frame = &processed_frames[0];
        println!(
            "  Processed frame: {}x{}",
            processed_frame.width(),
            processed_frame.height()
        );
        println!("  Saved to: test_output.png");
    } else {
        eprintln!("ERROR: No frames were processed!");
        return;
    }

    // Step 5: Calculate quality metrics (PSNR, SSIM)
    println!("\nStep 5: Calculating quality metrics...");

    let psnr = calculate_psnr(&original_frame, &processed_frames[0]);
    println!(
        "  PSNR: {:.2} dB",
        if psnr.is_infinite() {
            f64::INFINITY
        } else {
            psnr
        }
    );

    let ssim = calculate_ssim(&original_frame, &processed_frames[0]);
    println!("  SSIM: {:.4}", ssim);

    // Step 6: Generate difference heatmap
    println!("\nStep 6: Generating difference heatmap...");
    let diff_image = generate_diff_map(&original_frame, &processed_frames[0]);
    if let Err(e) = diff_image.data().save("diff_heatmap.png") {
        eprintln!("Error saving diff image: {}", e);
    } else {
        println!("  Saved to: diff_heatmap.png");
    }

    // Step 7: Summary
    println!("\n=== Bild-Pipeline Summary ===");
    println!(
        "Original image: {}x{}",
        original_frame.width(),
        original_frame.height()
    );
    println!("Pipeline filters: Gaussian Noise -> Bilateral Filter -> Blur");
    println!(
        "PSNR: {:.2} dB",
        if psnr.is_infinite() {
            f64::INFINITY
        } else {
            psnr
        }
    );
    println!("SSIM: {:.4}", ssim);
    println!("\nOutput files:");
    println!("  - test_input.png (original checkerboard)");
    println!("  - test_output.png (processed image)");
    println!("  - diff_heatmap.png (difference visualization)");
}

/// Führt das Video-Beispiel durch mit VideoSource und VideoSink
fn run_video_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let video_input_path = "test_video.mp4";
    let video_output_path = "output_video.mp4";

    println!("=== Video-Pipeline ===\n");

    // Check if video file exists, otherwise use fallback to image
    if !std::path::Path::new(video_input_path).exists() {
        eprintln!("Hinweis: {} nicht gefunden.", video_input_path);
        eprintln!("Verwende Bild-Benchmark als Fallback.");
        return run_image_pipeline_with_video_sink();
    }

    println!("Step 1: Creating VideoSource for '{}'", video_input_path);

    // Erstelle VideoSource mit ggfs. FPS von Video ermitteln
    let mut source = match VideoSource::new(video_input_path) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("Fehler beim Starten von VideoSource: {}", e);
            return Err("VideoSource konnte nicht erstellt werden".into());
        }
    };

    // Hole Metadaten aus dem ersten Frame
    let first_frame = match source.next_frame() {
        Some(f) => f,
        None => {
            eprintln!("Fehler: Kein Frame lesbar!");
            return Err("Kein Frame von VideoSource verfügbar".into());
        }
    };

    println!(
        "  Videogröße: {}x{}",
        first_frame.width(),
        first_frame.height()
    );
    println!("  FPS: aus ffprobe ermittelt");

    // Erstelle VideoSink und setze Auflösung
    let mut sink = VideoSink::new(video_output_path, 25.0);
    sink.set_width(first_frame.width());
    sink.set_height(first_frame.height());

    // Step 2: Erstelle Pipeline mit Filtern
    println!("\nStep 2: Building Pipeline...");
    let mut pipeline = Pipeline::new();

    println!("  Adding GaussianNoiseFilter(mean=0.0, sigma=15.0)...");
    pipeline = pipeline.add_filter(GaussianNoiseFilter::new(0.0, 15.0));

    println!("  Adding BlurFilter(sigma=3.0)...");
    pipeline = pipeline.add_filter(BlurFilter::new(3.0));

    // Step 3: Führe Pipeline aus (verarbeite alle Frames)
    println!("\nStep 3: Running Pipeline...");

    let processed_frames = pipeline.run(&mut source, &mut sink);

    println!("  {} Frame(s) verarbeitet", processed_frames.len());
    source.release();
    sink.release();

    // Step 4: Ausgabe-Informationen
    println!("\nStep 4: Video-Pipeline abgeschlossen!");
    if std::path::Path::new(video_output_path).exists() {
        let metadata = std::fs::metadata(video_output_path)?;
        println!("  Output-Datei: {}", video_output_path);
        println!("  Grösse: {} Bytes", metadata.len());
    }

    // Schritt 5: Erstelle Original (ohne Filter) Ausgabe
    println!("\nStep 5: Creating original output (no filters)...");
    let mut source = match VideoSource::new(video_input_path) {
        Ok(source) => source,
        Err(e) => return Err(e.into()),
    };

    // Erhalte Metadaten erneut für Original-Videostreaming
    if let Some(first_frame) = source.next_frame() {
        let mut sink = VideoSink::new("original_video.mp4", 25.0);
        sink.set_width(first_frame.width());
        sink.set_height(first_frame.height());

        println!("  Running source-to-sink without pipeline (original video)...");
        pipeline.run(&mut source, &mut sink);
        source.release();
        sink.release();
    }

    println!("\n=== Video-Pipeline Summary ===");
    println!("Input: {}", video_input_path);
    println!("Processed Output: {}", video_output_path);
    println!("Original Output: original_video.mp4");
    println!("Filters: Gaussian Noise (sigma=15) -> Blur (sigma=3)");
    println!("Codec: H.264 libx264, Format: YUV420P");

    Ok(())
}

/// Fallback-Modus wenn keine Videodatei vorhanden
fn run_image_pipeline_with_video_sink() -> Result<(), Box<dyn std::error::Error>> {
    let original_image = create_checkerboard(1920, 1080, 64);
    let temp_path = "test_temp.png";

    println!("=== VideoSink Fallback Mode ===\n");

    println!("Step 1: Creating test image for video sink testing...");
    if let Err(e) = original_image.save(temp_path) {
        eprintln!("Error saving test image: {}", e);
        return Err("Cannot create test image".into());
    }

    // Verwende ImageSource anstelle VideoSource
    let mut source = ImageSource::new(temp_path);
    let frame = source.next_frame().ok_or("No image available")?;

    println!("Test Image: {}x{}", frame.width(), frame.height());

    // Erstelle VideoSink mit Auflösung
    let mut sink = VideoSink::new("output_video.mp4", 1.0);
    sink.set_width(frame.width());
    sink.set_height(frame.height());

    println!("\nStep 2: Testing VideoSink with test image...");

    // Schreibe Bild als Frame aus
    sink.write(&frame)?;

    // Release um FFmpeg zu beenden
    sink.release();

    if std::path::Path::new("output_video.mp4").exists() {
        let metadata = std::fs::metadata("output_video.mp4")?;
        println!("VideoSink test successful!");
        println!("Output: output_video.mp4 ({} bytes)", metadata.len());
    } else {
        eprintln!("VideoSink test warning: file not created (ffmpeg might not be installed)");
    }

    Ok(())
}

fn main() {
    println!("=== Image Processing Framework Demo ===\n");

    // ---- Bild-Beispiel (originaler Code) ----
    run_image_pipeline();

    // ---- Video-Beispiel (neuer Code) ----
    if let Err(e) = run_video_pipeline() {
        eprintln!("Video-Pipeline failed: {}", e);
        eprintln!("Hint: Make sure 'ffmpeg' and 'ffprobe' are installed.");
        return;
    }
}
