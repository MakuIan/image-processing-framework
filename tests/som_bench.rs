use image_processing_framework::SomMatrix;
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
