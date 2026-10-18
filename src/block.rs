//! Tiling of image frames into square blocks for vector quantization.
//!
//! A frame is a flat, row-major byte buffer where each pixel occupies
//! `channels` consecutive bytes (1 for grayscale, 3 for RGB). This module
//! splits the frame into `block_size x block_size` tiles, normalizing every
//! sample to the `[0.0, 1.0]` range, and provides the lossless inverse that
//! reassembles the tiles back into a byte buffer.
//!
//! Only the region that is exactly covered by whole blocks is processed.
//! When `width` or `height` is not a multiple of `block_size`, the leftover
//! right and bottom strip is simply ignored during extraction and left as
//! zero during reconstruction, so the two operations stay consistent.

use rayon::prelude::*;

/// Parameters that describe how a frame is tiled into square blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockConfig {
    /// Edge length of every square block in pixels (e.g. `4` or `8`).
    pub block_size: usize,
    /// Number of color channels per pixel (`1` for gray, `3` for RGB).
    pub channels: usize,
}

impl BlockConfig {
    /// Creates a new configuration.
    ///
    /// Panics if `block_size` or `channels` is zero, as those values would
    /// cause a division by zero or an empty, meaningless block dimension.
    pub fn new(block_size: usize, channels: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0, got {block_size}");
        assert!(channels > 0, "channels must be > 0, got {channels}");
        Self {
            block_size,
            channels,
        }
    }

    /// Dimension (number of `f32` samples) of a single block:
    /// `block_size^2 * channels`.
    pub fn block_dim(&self) -> usize {
        self.block_size * self.block_size * self.channels
    }

    /// How many whole blocks fit along the width.
    fn blocks_per_row(&self, width: u32) -> usize {
        (width as usize) / self.block_size
    }

    /// How many whole blocks fit along the height.
    fn blocks_per_column(&self, height: u32) -> usize {
        (height as usize) / self.block_size
    }

    /// Total number of whole blocks that fit into an image of the given size.
    pub fn total_blocks(&self, width: u32, height: u32) -> usize {
        self.blocks_per_row(width) * self.blocks_per_column(height)
    }
}

/// Flat storage for a sequence of block vectors (`Vec<Vec<f32>>` replaced by
/// one contiguous `Vec<f32>` with stride access).
///
/// Keeping every block in a single allocation removes the per-block heap
/// allocations and the pointer chasing of nested vectors, which matters for
/// 4K frames with hundreds of thousands of blocks. Block `i` occupies the
/// samples `i * block_dim .. (i + 1) * block_dim`.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockBuffer {
    data: Vec<f32>,
    block_dim: usize,
}

impl BlockBuffer {
    /// Wraps a flat sample array as a block buffer.
    ///
    /// # Panics
    /// Panics if `block_dim` is zero or `data.len()` is not a multiple of
    /// `block_dim`.
    pub fn new(data: Vec<f32>, block_dim: usize) -> Self {
        assert!(block_dim > 0, "block_dim must be > 0, got {block_dim}");
        assert_eq!(
            data.len() % block_dim,
            0,
            "buffer has {} sample(s), not a multiple of block_dim {block_dim}",
            data.len()
        );
        Self { data, block_dim }
    }

    /// Number of blocks stored in the buffer.
    pub fn len(&self) -> usize {
        self.data.len() / self.block_dim
    }

    /// `true` if the buffer holds no blocks.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Dimension (number of `f32` samples) of a single block.
    pub fn dim(&self) -> usize {
        self.block_dim
    }

    /// The underlying flat sample array.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns block `index` as a slice of `block_dim` samples.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds.
    pub fn block(&self, index: usize) -> &[f32] {
        let start = index * self.block_dim;
        &self.data[start..start + self.block_dim]
    }

    /// Iterates over all blocks in storage order.
    pub fn blocks(&self) -> impl Iterator<Item = &[f32]> {
        self.data.chunks(self.block_dim)
    }
}

/// Splits `frame_data` into a flat [`BlockBuffer`] of normalized block
/// vectors.
///
/// This is the zero-allocation hot path of the codec: the samples of all
/// blocks live in one contiguous `Vec<f32>` (stride `config.block_dim()`),
/// so extraction performs a single heap allocation per frame instead of one
/// per block.
///
/// Traversal order and normalization are identical to [`extract_blocks`]:
/// block by block, top to bottom and left to right, row-major inside a
/// block, every byte scaled to `[0.0, 1.0]`. The buffer holds
/// `config.total_blocks(width, height)` blocks.
///
/// # Panics
/// Panics if `frame_data` does not hold at least `width * height * channels`
/// bytes.
pub fn extract_blocks_flat(
    frame_data: &[u8],
    width: u32,
    height: u32,
    config: &BlockConfig,
) -> BlockBuffer {
    let bs = config.block_size;
    let channels = config.channels;
    let dim = config.block_dim();

    let w = width as usize;
    let h = height as usize;

    let expected = w * h * channels;
    assert!(
        frame_data.len() >= expected,
        "frame_data too small: expected {expected} bytes for {width}x{height} \
         with {channels} channel(s), got {}",
        frame_data.len()
    );

    let blocks_per_row = config.blocks_per_row(width);
    let blocks_per_column = config.blocks_per_column(height);
    let total_blocks = blocks_per_row * blocks_per_column;
    let scale = 1.0_f32 / 255.0_f32;

    // Block-Reihenparallel (Rayon): Die Schreibposition jedes Blocks ist
    // dessen Index, daher bleibt die Speicherreihenfolge identisch zum
    // sequenziellen Pfad, waehrend Lese-/Schreibbandbreite über alle Kerne
    // ausgelastet wird.
    let mut data = vec![0.0_f32; total_blocks * dim];
    data.par_chunks_mut(dim)
        .enumerate()
        .for_each(|(index, block)| {
            let by = index / blocks_per_row;
            let bx = index % blocks_per_row;
            let x0 = bx * bs;
            let y0 = by * bs;

            let mut i = 0usize;
            for r in 0..bs {
                let row_base = (y0 + r) * w * channels;
                for c in 0..bs {
                    let pixel_base = row_base + (x0 + c) * channels;
                    for ch in 0..channels {
                        block[i] = frame_data[pixel_base + ch] as f32 * scale;
                        i += 1;
                    }
                }
            }
        });

    BlockBuffer::new(data, dim)
}

/// Splits `frame_data` into normalized block vectors.
///
/// The image is traversed block by block, top to bottom and left to right.
/// Inside a block samples are ordered row-major: for each pixel row, then
/// each pixel column, then each channel. Every byte is scaled to `[0.0, 1.0]`.
///
/// The returned vector contains `config.total_blocks(width, height)` entries,
/// each of length `config.block_dim()`. For a `4x4` RGB tile the dimension is
/// `4 * 4 * 3 == 48`.
///
/// Thin wrapper around [`extract_blocks_flat`] that splits the flat buffer
/// into one heap-allocated `Vec` per block; use the flat variant directly on
/// hot paths.
///
/// # Panics
/// Panics if `frame_data` does not hold at least `width * height * channels`
/// bytes.
pub fn extract_blocks(
    frame_data: &[u8],
    width: u32,
    height: u32,
    config: &BlockConfig,
) -> Vec<Vec<f32>> {
    extract_blocks_flat(frame_data, width, height, config)
        .blocks()
        .map(<[f32]>::to_vec)
        .collect()
}

/// Writes the given blocks back into a `width * height * channels` byte
/// buffer, consuming at most `config.total_blocks(width, height)` entries.
///
/// Sequential worker of [`reconstruct_image`]. The hot codec path uses
/// [`reconstruct_image_from_codebook`], which assembles from indices directly
/// and in parallel.
fn assemble_blocks<'a, I>(blocks: I, width: u32, height: u32, config: &BlockConfig) -> Vec<u8>
where
    I: IntoIterator<Item = &'a [f32]>,
{
    let bs = config.block_size;
    let channels = config.channels;
    let dim = config.block_dim();

    let w = width as usize;
    let h = height as usize;

    let mut out = vec![0u8; w * h * channels];

    let blocks_per_row = config.blocks_per_row(width);
    let blocks_per_column = config.blocks_per_column(height);
    let total_blocks = blocks_per_row * blocks_per_column;

    for (block_index, block) in blocks.into_iter().enumerate().take(total_blocks) {
        let block_len = block.len();
        assert_eq!(
            block_len, dim,
            "block {block_index} has {block_len} sample(s), expected {dim}"
        );

        let by = block_index / blocks_per_row;
        let bx = block_index % blocks_per_row;

        let x0 = bx * bs;
        let y0 = by * bs;

        let mut i = 0usize;
        for r in 0..bs {
            let row_base = (y0 + r) * w * channels;
            for c in 0..bs {
                let pixel_base = row_base + (x0 + c) * channels;
                for ch in 0..channels {
                    let v = block[i];
                    i += 1;
                    out[pixel_base + ch] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
    }

    out
}

/// Inverse of [`extract_blocks`]: assembles the block vectors back into a byte
/// buffer of size `width * height * channels`.
///
/// Every sample is clamped to `[0.0, 1.0]`, multiplied by `255` and rounded to
/// the nearest byte. Only as many blocks are consumed as fit into the image
/// (any surplus entries are ignored). Samples that do not belong to a covered
/// block are left as zero, which keeps the operation consistent with the
/// extraction of a non-divisible image.
///
/// # Panics
/// Panics if a consumed block does not have exactly `config.block_dim()`
/// samples.
pub fn reconstruct_image(
    blocks: &[Vec<f32>],
    width: u32,
    height: u32,
    config: &BlockConfig,
) -> Vec<u8> {
    assemble_blocks(blocks.iter().map(Vec::as_slice), width, height, config)
}

/// Reassembles a frame directly from quantized block `indices` and a
/// `codebook` (one `vector_dim`-sized codevector per index).
///
/// Avoids materializing an intermediate `Vec<Vec<f32>>` of codevectors: each
/// codevector slice is written into the output buffer on the fly. Semantics
/// match calling [`reconstruct_image`] on the looked-up codevectors.
///
/// This is the hot reconstruction path of the codec and runs block-row
/// parallel via Rayon; each block-row writes into a disjoint, contiguous
/// region of the output buffer, so the byte layout is identical to the
/// sequential [`assemble_blocks`] worker.
///
/// # Panics
/// Panics if an index is out of `codebook` bounds or the referenced codevector
/// does not have exactly `config.block_dim()` samples.
pub fn reconstruct_image_from_codebook(
    indices: &[u8],
    codebook: &[Vec<f32>],
    width: u32,
    height: u32,
    config: &BlockConfig,
) -> Vec<u8> {
    let bs = config.block_size;
    let channels = config.channels;
    let dim = config.block_dim();

    let w = width as usize;
    let h = height as usize;

    let mut out = vec![0u8; w * h * channels];

    let blocks_per_row = config.blocks_per_row(width);
    let blocks_per_column = config.blocks_per_column(height);
    let total_blocks = blocks_per_row * blocks_per_column;
    let used_blocks = indices.len().min(total_blocks);

    if used_blocks == 0 || blocks_per_row == 0 {
        return out;
    }

    // Jede Blockreihe belegt genau `bs` volle Pixelzeilen, also einen
    // zusammenhaengenden Abschnitt im zeilenweisen Ausgabepuffer. Die
    // par_chunks_mut-Bloecke liegen damit disjunkt; ein nicht gedeckter
    // Rest am rechten/unteren Rand bleibt null wie im sequenziellen Pfad.
    let region_len = bs * w * channels;
    out.par_chunks_mut(region_len)
        .enumerate()
        .for_each(|(by, region)| {
            let row_start = by * blocks_per_row;
            for bx in 0..blocks_per_row {
                let block_index = row_start + bx;
                if block_index >= used_blocks {
                    break;
                }
                let block = &codebook[indices[block_index] as usize];
                let block_len = block.len();
                assert_eq!(
                    block_len, dim,
                    "block {block_index} has {block_len} sample(s), expected {dim}"
                );

                let x0 = bx * bs;
                let mut i = 0usize;
                for r in 0..bs {
                    let row_base = (r * w + x0) * channels;
                    for c in 0..bs {
                        let pixel_base = row_base + c * channels;
                        for ch in 0..channels {
                            let v = block[i];
                            i += 1;
                            region[pixel_base + ch] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                        }
                    }
                }
            }
        });

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_dim_matches_formula() {
        assert_eq!(BlockConfig::new(4, 3).block_dim(), 48);
        assert_eq!(BlockConfig::new(8, 3).block_dim(), 192);
        assert_eq!(BlockConfig::new(4, 1).block_dim(), 16);
        assert_eq!(BlockConfig::new(8, 1).block_dim(), 64);
    }

    #[test]
    #[should_panic(expected = "block_size must be > 0")]
    fn rejects_zero_block_size() {
        let _ = BlockConfig::new(0, 3);
    }

    #[test]
    #[should_panic(expected = "channels must be > 0")]
    fn rejects_zero_channels() {
        let _ = BlockConfig::new(4, 0);
    }

    #[test]
    fn total_blocks_truncates_non_divisible() {
        let config = BlockConfig::new(4, 3);
        assert_eq!(config.total_blocks(8, 8), 4);
        assert_eq!(config.total_blocks(10, 10), 4);
        assert_eq!(config.total_blocks(3, 100), 0);
    }

    #[test]
    fn extract_single_block_exact() {
        let width = 4u32;
        let height = 4u32;
        let channels = 3;
        let config = BlockConfig::new(4, channels);

        let mut data = vec![0u8; ((width * height) as usize) * channels];
        for y in 0..height as usize {
            for x in 0..width as usize {
                for ch in 0..channels {
                    let idx = y * (width as usize) * channels + x * channels + ch;
                    data[idx] = (x * 10 + y * 3 + ch * 7) as u8;
                }
            }
        }

        let blocks = extract_blocks(&data, width, height, &config);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].len(), 48);

        for (i, &v) in blocks[0].iter().enumerate() {
            let r = i / (4 * 3);
            let c = (i / 3) % 4;
            let ch = i % 3;
            let expected = ((c * 10 + r * 3 + ch * 7) % 256) as f32 / 255.0;
            assert!(
                (v - expected).abs() < 1e-6,
                "sample {i}: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn extract_block_count_and_bounds() {
        let width = 8u32;
        let height = 8u32;
        let channels = 3;
        let config = BlockConfig::new(4, channels);

        let data = vec![128u8; ((width * height) as usize) * channels];
        let blocks = extract_blocks(&data, width, height, &config);

        assert_eq!(blocks.len(), 4);
        for block in &blocks {
            assert_eq!(block.len(), 48);
            assert!(block.iter().all(|v| (0.0..=1.0).contains(v)));
        }
    }

    #[test]
    fn extract_truncates_non_divisible_image() {
        // 10x10 with 4x4 blocks: only the top-left 8x8 is covered -> 2x2 blocks.
        let width = 10u32;
        let height = 10u32;
        let channels = 1;
        let config = BlockConfig::new(4, channels);

        let data = vec![200u8; ((width * height) as usize) * channels];
        let blocks = extract_blocks(&data, width, height, &config);

        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].len(), 16);
    }

    #[test]
    fn roundtrip_exact_rgb() {
        let width = 10u32;
        let height = 10u32;
        let channels = 3;
        let config = BlockConfig::new(4, channels);

        let expected_len = ((width * height) as usize) * channels;
        let data: Vec<u8> = (0..expected_len as u32).map(|i| (i % 256) as u8).collect();

        let blocks = extract_blocks(&data, width, height, &config);
        assert_eq!(blocks.len(), 4);
        let out = reconstruct_image(&blocks, width, height, &config);

        assert_eq!(out.len(), expected_len);

        // The covered 8x8 region must round-trip exactly.
        for y in 0..8 {
            for x in 0..8 {
                for ch in 0..channels {
                    let idx = (y * (width as usize) + x) * channels + ch;
                    assert_eq!(data[idx], out[idx], "pixel ({x},{y}) ch{ch}");
                }
            }
        }
    }

    #[test]
    fn roundtrip_exact_gray() {
        let width = 8u32;
        let height = 8u32;
        let channels = 1;
        let config = BlockConfig::new(8, channels);

        let data: Vec<u8> = (0..(width * height))
            .map(|i| (i % 256) as u8)
            .collect();
        let blocks = extract_blocks(&data, width, height, &config);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].len(), 64);

        let out = reconstruct_image(&blocks, width, height, &config);
        assert_eq!(data, out);
    }

    #[test]
    fn reconstruct_clamps_out_of_range() {
        let width = 4u32;
        let height = 4u32;
        let config = BlockConfig::new(4, 1);

        let block = vec![
            -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 0.001, -1.0, 0.25, 0.75, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5,
        ];

        let out = reconstruct_image(&[block], width, height, &config);
        assert_eq!(out.len(), 16);
        assert_eq!(out[0], 0); // clamped low
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 128); // 0.5 * 255 == 127.5 -> 128
        assert_eq!(out[3], 255); // 1.0
        assert_eq!(out[4], 255); // 1.5 clamped high
        assert_eq!(out[5], 255); // 2.0 clamped high
        assert_eq!(out[6], 0); // 0.001 * 255 == 0.255 -> 0
        assert_eq!(out[7], 0); // clamped low
        assert_eq!(out[9], 191); // 0.75 * 255 == 191.25 -> 191
    }

    #[test]
    fn reconstruct_ignores_surplus_blocks() {
        let config = BlockConfig::new(4, 1);
        // Provide more blocks than fit; only the first is used.
        let block = vec![0.5f32; 16];
        let extra = vec![0.9f32; 16];
        let out = reconstruct_image(&[block.clone(), extra], 4, 4, &config);
        assert_eq!(out, vec![128u8; 16]);
    }

    #[test]
    #[should_panic(expected = "has 16 sample(s), expected 48")]
    fn reconstruct_rejects_misshapen_block() {
        let config = BlockConfig::new(4, 3);
        let too_short = vec![0.0f32; 16];
        let _ = reconstruct_image(&[too_short], 4, 4, &config);
    }

    #[test]
    fn extract_empty_when_smaller_than_block() {
        let config = BlockConfig::new(4, 3);
        let data = vec![0u8; 3 * 3 * 3];
        let blocks = extract_blocks(&data, 3, 3, &config);
        assert_eq!(blocks.len(), 0);

        let out = reconstruct_image(&blocks, 3, 3, &config);
        assert_eq!(out, vec![0u8; 3 * 3 * 3]);
    }

    #[test]
    #[should_panic(expected = "frame_data too small")]
    fn extract_rejects_undersized_buffer() {
        let config = BlockConfig::new(4, 3);
        let data = vec![0u8; 10]; // far less than 8*8*3
        let _ = extract_blocks(&data, 8, 8, &config);
    }

    #[test]
    fn extract_blocks_flat_matches_nested_extraction() {
        let (width, height) = (10u32, 10u32);
        let config = BlockConfig::new(4, 3);
        let data: Vec<u8> = (0..(width * height * 3)).map(|i| (i % 256) as u8).collect();

        let nested = extract_blocks(&data, width, height, &config);
        let flat = extract_blocks_flat(&data, width, height, &config);

        assert_eq!(flat.len(), nested.len());
        assert_eq!(flat.dim(), config.block_dim());
        assert_eq!(flat.data().len(), nested.len() * config.block_dim());
        for (index, expected) in nested.iter().enumerate() {
            assert_eq!(flat.block(index), expected.as_slice());
        }
        assert_eq!(flat.blocks().count(), nested.len());
    }

    #[test]
    #[should_panic(expected = "not a multiple of block_dim")]
    fn block_buffer_rejects_partial_block() {
        // 5 samples with dim 4: the trailing sample is not a full block.
        let _ = BlockBuffer::new(vec![0.0; 5], 4);
    }

    #[test]
    fn reconstruct_from_codebook_matches_block_reconstruction() {
        let config = BlockConfig::new(4, 3);

        let codebook: Vec<Vec<f32>> = (0..6)
            .map(|c| {
                (0..config.block_dim())
                    .map(|i| ((i + c * 13) % 256) as f32 / 255.0)
                    .collect()
            })
            .collect();

        // Teilbare und nicht-teilbare Dimensionen decken sowohl den vollen
        // Pfad als auch den nicht-gedeckten Randstreifen ab.
        for (width, height) in [(8u32, 8u32), (10u32, 10u32), (12u32, 8u32)] {
            let indices: Vec<u8> = (0..(config.total_blocks(width, height)))
                .map(|i| (i % codebook.len()) as u8)
                .collect();
            let blocks: Vec<Vec<f32>> = indices
                .iter()
                .map(|&i| codebook[i as usize].clone())
                .collect();

            let expected = reconstruct_image(&blocks, width, height, &config);
            let actual =
                reconstruct_image_from_codebook(&indices, &codebook, width, height, &config);
            assert_eq!(actual, expected, "mismatch for {width}x{height}");
        }
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn reconstruct_from_codebook_rejects_unknown_index() {
        let config = BlockConfig::new(4, 1);
        let codebook: Vec<Vec<f32>> = vec![vec![0.5; config.block_dim()]];
        let _ = reconstruct_image_from_codebook(&[7], &codebook, 4, 4, &config);
    }
}
