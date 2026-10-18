use image::RgbImage;
use image_processing_framework::{
    BlockConfig, Frame, SomMatrix, SomVideoConfig, calculate_psnr, compress_indices,
    encode_video_stream_with_progress, extract_blocks_flat, reconstruct_image_from_codebook,
};
use std::time::Instant;

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
