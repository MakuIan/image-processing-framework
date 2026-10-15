//! SOM-gestÃ¼tzte Video-Kompression.
//!
//! Das Frame-Stream-Codec quantisiert 4x4-RGB-BildblÃ¶cke mit einer
//! Self-Organizing Map (SOM) auf Indizes `0..=255` und Ã¼bertrÃ¤gt pro Frame
//! nur diese Indizes (RLE + Deflate). Der Codebook-Zustand (SOM-Gewichte)
//! wird nicht bei jedem Frame gesendet, sondern nur in **Keyframes**:
//!
//! 1. **Frame 0 ist immer ein Keyframe**: Es gibt noch keine gemeinsame
//!    SOM, daher wird auf den BlÃ¶cken von Frame 0 ein neues SOM trainiert.
//!    Das Paket enthÃ¤lt die SOM-Gewichte plus die komprimierten Indizes.
//!
//! 2. **Frame k > 0**: Die BlÃ¶cke werden mit der *bisherigen* SOM
//!    enkodiert und der Frame testweise rekonstruiert. Ist die PSNR
//!    gegenÃ¼ber dem Original mindestens `psnr_threshold_db`, ist der
//!    Codebook-Zustand weiterhin gÃ¼ltig: Es wird kein Keyframe erzeugt und
//!    nur das Index-Paket gesendet. Liegt die PSNR darunter, wird auf den
//!    BlÃ¶cken von Frame k ein neues SOM trainiert, Frame k wird damit als
//!    Keyframe erneut enkodiert und das Paket enthÃ¤lt die neuen
//!    SOM-Gewichte plus die komprimierten Indizes.
//!
//! Der Decoder arbeitet damit sequenziell: Er hÃ¤lt die zuletzt empfangene
//! SOM, liest bei jedem Frame die Indizes, und Ã¼bernimmt bei Keyframes
//! zusÃ¤tzlich die neuen SOM-Gewichte, bevor die BlÃ¶cke rekonstruiert werden.

use crate::block::{BlockConfig, extract_blocks, reconstruct_image};
use crate::compression::{compress_indices, decompress_indices};
use crate::evaluator::calculate_psnr;
use crate::frame::Frame;
use crate::matrix::SomMatrix;
use image::RgbImage;

/// Anfangslernrate fÃ¼r das Kohonen-Training (lineare Abklingung erfolgt in
/// [`SomMatrix::train`]).
const INITIAL_LEARNING_RATE: f32 = 0.3;

/// Parameter des SOM-Video-Codecs.
#[derive(Clone, Debug)]
pub struct SomVideoConfig {
    /// KantenlÃ¤nge der Quadrate-BlÃ¶cke in Pixeln (Standard: `4`).
    pub block_size: usize,
    /// Gitterseite des SOM: `som_grid_size x som_grid_size` Neuronen
    /// (Standard: `16` -> 256 Codevektoren).
    pub som_grid_size: usize,
    /// Mindest-PSNR in dB, unterhalb derer ein Frame als Keyframe neu
    /// trainiert wird (Standard: `28.0`).
    pub psnr_threshold_db: f64,
    /// Anzahl der Kohonen-Epochen beim Training (Standard: `5`).
    pub train_epochs: usize,
}

impl Default for SomVideoConfig {
    fn default() -> Self {
        Self {
            block_size: 4,
            som_grid_size: 16,
            psnr_threshold_db: 28.0,
            train_epochs: 5,
        }
    }
}

impl SomVideoConfig {
    /// Erzeugt eine Konfiguration mit expliziten Parametern.
    pub fn new(
        block_size: usize,
        som_grid_size: usize,
        psnr_threshold_db: f64,
        train_epochs: usize,
    ) -> Self {
        Self {
            block_size,
            som_grid_size,
            psnr_threshold_db,
            train_epochs,
        }
    }
}

/// Datenpaket pro Frame des komprimierten Streams.
#[derive(Clone, Debug)]
pub struct EncodedFrame {
    /// `true`, wenn `som_matrix` eine neue SOM enthÃ¤lt, die der Decoder
    /// Ã¼bernehmen muss, bevor die Indizes dekodiert werden.
    pub is_keyframe: bool,
    /// Serialisierte SOM-Gewichte (`neuron_count * vector_dim` f32-Werte,
    /// little-endian). Exakt dann vorhanden, wenn `is_keyframe == true`.\
    pub som_matrix: Option<Vec<u8>>,
    /// RLE+Deflate-Paket der Blockindizes.
    pub compressed_indices: Vec<u8>,
}

/// Aggregierte Codec-Statistiken eines Streams.
#[derive(Clone, Debug)]
pub struct VideoCodecStats {
    /// Anzahl der enkodierten Frames.
    pub total_frames: usize,
    /// Anzahl der Keyframes (Frames mit neuem SOM).
    pub keyframe_count: usize,
    /// PSNR-Mittel Ã¼ber alle Frames mit endlicher PSNR (dB). `f64::INFINITY`,
    /// wenn alle Frames verlustfrei rekonstruiert wurden; `0.0`, wenn es
    /// keine Frames (oder keine endlichen PSNR-Werte) gab.
    pub avg_psnr_db: f64,
    /// Summe der Rohdaten-GrÃ¶ÃŸen aller Frames in Bytes.
    pub raw_size: u64,
    /// Summe der komprimierten Index-Pakete aller Frames in Bytes.
    /// (Keyframe-SOM-Gewichte sind nicht enthalten; sie werden als
    /// Codebook-Zustand separat Ã¼bertragen.)
    pub compressed_size: u64,
    /// Kompressionsfaktor `raw_size / compressed_size` (oder
    /// `f64::INFINITY`, wenn die komprimierten Pakete leer sind).
    pub compression_ratio: f64,
}

/// Ergebnis von [`encode_video_stream`]: die Frame-Pakete plus Statistiken.
#[derive(Clone, Debug)]
pub struct EncodeResult {
    /// Eines pro Eingabe-Frame, in Stream-Reihenfolge.
    pub frames: Vec<EncodedFrame>,
    /// Aggregierte Statistiken.
    pub stats: VideoCodecStats,
}

/// Serialisiert die SOM-Gewichte zu Bytes: pro Neuron `vector_dim`
/// f32-Werte im Little-Endian-Format, in Gitter-Reihenfolge.
pub fn serialize_som(som: &SomMatrix) -> Vec<u8> {
    let mut out = Vec::with_capacity(som.neuron_count() * som.vector_dim * 4);
    for weights in &som.weights {
        for value in weights {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

/// Inverse von [`serialize_som`]: stellt das SOM aus serialisierten
/// Gewichten wieder her.
///
/// # Panics
/// Panics, wenn `data` nicht die erwartete GrÃ¶ÃŸe
/// (`grid_width * grid_height * vector_dim * 4` Bytes) hat.
pub fn deserialize_som(
    data: &[u8],
    grid_width: usize,
    grid_height: usize,
    vector_dim: usize,
) -> SomMatrix {
    let expected = grid_width * grid_height * vector_dim * 4;
    assert_eq!(
        data.len(),
        expected,
        "som_matrix payload has {} byte(s), expected {expected}",
        data.len()
    );

    let values: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    SomMatrix {
        grid_width,
        grid_height,
        vector_dim,
        weights: values
            .chunks_exact(vector_dim)
            .map(|c| c.to_vec())
            .collect(),
    }
}

/// Wandelt ein RGB-Frame in den rohen Byte-Puffer
/// (`width * height * 3`, Zeilenweise) um.
fn frame_to_raw(frame: &Frame) -> Vec<u8> {
    let raw = frame.data().as_raw();
    let expected = (frame.width() as usize) * (frame.height() as usize) * 3;
    assert_eq!(
        raw.len(),
        expected,
        "frame data has {} byte(s), expected {expected} for {}x{}",
        raw.len(),
        frame.width(),
        frame.height()
    );
    raw.to_vec()
}

/// Trainiert auf den BlÃ¶cken von `frame` eine neue SOM gemÃ¤ÃŸ `config`.
fn train_som_on_frame(
    frame: &Frame,
    config: &BlockConfig,
    som_grid: usize,
    epochs: usize,
) -> SomMatrix {
    let blocks = extract_blocks(&frame_to_raw(frame), frame.width(), frame.height(), config);
    assert!(
        !blocks.is_empty(),
        "frame is smaller than one {}x{} block, nothing to train on",
        config.block_size,
        config.block_size
    );

    let mut som = SomMatrix::new(som_grid, som_grid, config.block_dim(), &blocks);
    som.train(&blocks, epochs, INITIAL_LEARNING_RATE);
    som
}

/// Einkodiert die BlÃ¶cke von `frame` mit `som` und liefert das
/// RLE+Deflate-Paket der Indizes.
fn encode_frame_indices(frame: &Frame, som: &SomMatrix, config: &BlockConfig) -> Vec<u8> {
    let blocks = extract_blocks(&frame_to_raw(frame), frame.width(), frame.height(), config);
    let indices = som.quantize_all(&blocks);
    compress_indices(&indices)
}

/// Rekonstruiert `frame` testweise Ã¼ber die Indizes der SOM `som`:
/// BlÃ¶cke extrahieren, quantisieren, Codevektoren in ein Frame zerlegen.
fn reconstruct_frame_with_som(frame: &Frame, som: &SomMatrix, config: &BlockConfig) -> Frame {
    let blocks = extract_blocks(&frame_to_raw(frame), frame.width(), frame.height(), config);
    let indices = som.quantize_all(&blocks);

    let recon_blocks: Vec<Vec<f32>> = indices
        .iter()
        .map(|&i| som.lookup_vector(i).to_vec())
        .collect();
    let recon_raw = reconstruct_image(&recon_blocks, frame.width(), frame.height(), config);

    let image = RgbImage::from_raw(frame.width(), frame.height(), recon_raw);
    frame.clone_with_data(image.expect("reconstructed image buffer must match dimensions"))
}

/// Einkodiert einen Frame-Stream gemÃ¤ÃŸ `config`.
///
/// Siehe Modul-Dokumentation fÃ¼r den Keyframe-Algorithmus. FÃ¼r jeden
/// Eingabe-Frame liefert die Ausgabe genau ein [`EncodedFrame`]-Paket;
/// Frame 0 ist immer ein Keyframe.
///
/// # Panics
/// Panics auf leeren Streams oder auf Frames, die kleiner als ein einzelner
/// Block sind (es gÃ¤be nichts zu quantisieren).
/// Fortschritts-Info fuer einen kodierten Frame.
#[derive(Clone, Copy, Debug)]
pub struct EncodeFrameProgress {
    pub frame_index: usize,
    pub total_frames: usize,
    pub is_keyframe: bool,
    pub psnr_db: f64,
}

/// Einkodiert einen Frame-Stream gemaess `config` mit Fortschritts-Callback.
///
/// Siehe Modul-Dokumentation fuer den Keyframe-Algorithmus. Fuer jeden
/// Eingabe-Frame liefert die Ausgabe genau ein [`EncodedFrame`]-Paket;
/// Frame 0 ist immer ein Keyframe.
///
/// # Panics
/// Panics auf leeren Streams oder auf Frames, die kleiner als ein einzelner
/// Block sind (es gaebe nichts zu quantisieren).
pub fn encode_video_stream_with_progress<F: FnMut(EncodeFrameProgress)>(
    frames: &[Frame],
    config: &SomVideoConfig,
    mut on_progress: F,
) -> EncodeResult {
    assert!(!frames.is_empty(), "frames must not be empty");
    assert!(
        config.block_size > 0,
        "block_size must be > 0, got {}",
        config.block_size
    );
    assert!(
        config.som_grid_size > 0,
        "som_grid_size must be > 0, got {}",
        config.som_grid_size
    );
    assert!(
        config.som_grid_size * config.som_grid_size <= 256,
        "som grid has {} neurons, at most 256 are supported (indices are u8)",
        config.som_grid_size * config.som_grid_size
    );
    assert!(
        config.train_epochs > 0,
        "train_epochs must be > 0, got {}",
        config.train_epochs
    );

    let block_config = BlockConfig::new(config.block_size, 3);

    // Frame 0: immer Keyframe, neues SOM trainieren.
    let mut som = train_som_on_frame(
        &frames[0],
        &block_config,
        config.som_grid_size,
        config.train_epochs,
    );

    let mut frames_out: Vec<EncodedFrame> = Vec::with_capacity(frames.len());
    let mut keyframe_count = 0usize;
    let mut finite_psnrs: Vec<f64> = Vec::with_capacity(frames.len());
    let mut all_finite = true;
    let mut raw_size: u64 = 0;
    let mut compressed_size: u64 = 0;

    for (k, frame) in frames.iter().enumerate() {
        // Testweise Rekonstruktion mit der bisherigen SOM (Frame 0 nutzt
        // das soeben auf ihm selbst trainierte SOM).
        let mut recon = reconstruct_frame_with_som(frame, &som, &block_config);
        let mut frame_psnr = calculate_psnr(frame, &recon);

        let is_keyframe = if k == 0 {
            // Neues SOM auf Frame 0: Paket enthaelt Gewichte + Indizes.
            true
        } else if frame_psnr < config.psnr_threshold_db {
            // Codebook passt nicht mehr: neues SOM trainieren, neu kodieren.
            som = train_som_on_frame(
                frame,
                &block_config,
                config.som_grid_size,
                config.train_epochs,
            );
            recon = reconstruct_frame_with_som(frame, &som, &block_config);
            frame_psnr = calculate_psnr(frame, &recon);
            true
        } else {
            false
        };
        if is_keyframe {
            keyframe_count += 1;
        }

        let compressed_indices = encode_frame_indices(frame, &som, &block_config);
        let compressed_len = compressed_indices.len() as u64;

        frames_out.push(EncodedFrame {
            is_keyframe,
            som_matrix: if is_keyframe {
                Some(serialize_som(&som))
            } else {
                None
            },
            compressed_indices,
        });

        if frame_psnr.is_finite() {
            finite_psnrs.push(frame_psnr);
        } else {
            all_finite = false;
        }

        raw_size += (frame.width() as u64) * (frame.height() as u64) * 3;
        compressed_size += compressed_len;

        on_progress(EncodeFrameProgress {
            frame_index: k + 1,
            total_frames: frames.len(),
            is_keyframe,
            psnr_db: frame_psnr,
        });
    }

    let avg_psnr_db = if finite_psnrs.is_empty() {
        0.0
    } else if all_finite {
        finite_psnrs.iter().sum::<f64>() / finite_psnrs.len() as f64
    } else {
        f64::INFINITY
    };

    let compression_ratio = if compressed_size == 0 {
        f64::INFINITY
    } else {
        raw_size as f64 / compressed_size as f64
    };

    EncodeResult {
        frames: frames_out,
        stats: VideoCodecStats {
            total_frames: frames.len(),
            keyframe_count,
            avg_psnr_db,
            raw_size,
            compressed_size,
            compression_ratio,
        },
    }
}

/// Einkodiert einen Frame-Stream gemaess `config`.
pub fn encode_video_stream(frames: &[Frame], config: &SomVideoConfig) -> EncodeResult {
    encode_video_stream_with_progress(frames, config, |_| {})
}

/// Dekodiert einen Strom von [`EncodedFrame`]-Paketen zurueck in RGB-[`Frame`]s mit Fortschritts-Callback.
pub fn decode_video_stream_with_progress<F: FnMut(usize, usize)>(
    encoded_frames: &[EncodedFrame],
    width: u32,
    height: u32,
    config: &SomVideoConfig,
    fps: f64,
    mut on_progress: F,
) -> Result<Vec<Frame>, String> {
    if encoded_frames.is_empty() {
        return Ok(Vec::new());
    }

    let block_config = BlockConfig::new(config.block_size, 3);
    let vector_dim = block_config.block_dim();
    let mut current_som: Option<SomMatrix> = None;
    let mut frames_out = Vec::with_capacity(encoded_frames.len());

    for (k, enc) in encoded_frames.iter().enumerate() {
        if enc.is_keyframe {
            let som_data = enc
                .som_matrix
                .as_ref()
                .ok_or_else(|| format!("Keyframe {k} fehlt SOM-Payload"))?;
            current_som = Some(deserialize_som(
                som_data,
                config.som_grid_size,
                config.som_grid_size,
                vector_dim,
            ));
        }

        let som = current_som
            .as_ref()
            .ok_or_else(|| format!("Frame {k} empfangen, bevor ein initiales SOM gesetzt wurde"))?;

        let indices = decompress_indices(&enc.compressed_indices)?;
        let recon_blocks: Vec<Vec<f32>> = indices
            .iter()
            .map(|&idx| som.lookup_vector(idx).to_vec())
            .collect();

        let recon_raw = reconstruct_image(&recon_blocks, width, height, &block_config);
        let image = RgbImage::from_raw(width, height, recon_raw)
            .ok_or_else(|| "Ungueltige Dimensionen fuer rekonstruiertes Bild".to_string())?;

        let timestamp = if fps > 0.0 { k as f64 / fps } else { 0.0 };
        frames_out.push(Frame::new(image, k, timestamp));
        on_progress(k + 1, encoded_frames.len());
    }

    Ok(frames_out)
}

/// Dekodiert einen Strom von [`EncodedFrame`]-Paketen zurueck in RGB-[`Frame`]s.
pub fn decode_video_stream(
    encoded_frames: &[EncodedFrame],
    width: u32,
    height: u32,
    config: &SomVideoConfig,
    fps: f64,
) -> Result<Vec<Frame>, String> {
    decode_video_stream_with_progress(encoded_frames, width, height, config, fps, |_, _| {})
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Erzeugt ein Frame mit einheitlicher Farbe.
    fn make_solid_frame(index: usize, val: u8) -> Frame {
        let (width, height) = (16u32, 16u32);
        let data = vec![val; (width * height) as usize * 3];
        Frame::new(
            RgbImage::from_raw(width, height, data)
                .expect("test frame buffer must match dimensions"),
            index,
            index as f64 / 30.0,
        )
    }

    /// 16x16-RGB-Frame mit deterministischem Muster.
    fn make_frame(index: usize, pattern_offset: u8) -> Frame {
        let (width, height) = (16u32, 16u32);
        let mut data = Vec::with_capacity((width * height) as usize * 3);
        for y in 0..height {
            for x in 0..width {
                let base = (x.wrapping_mul(7) + y.wrapping_mul(3) + pattern_offset as u32) as u8;
                data.push(base);
                data.push(base.wrapping_mul(2));
                data.push(base.wrapping_mul(3));
            }
        }
        Frame::new(
            RgbImage::from_raw(width, height, data)
                .expect("test frame buffer must match dimensions"),
            index,
            index as f64 / 30.0,
        )
    }

    fn default_config() -> SomVideoConfig {
        SomVideoConfig::default()
    }

    #[test]
    fn frame_zero_is_always_keyframe() {
        let frames = (0..4).map(|i| make_frame(i, 0)).collect::<Vec<_>>();
        let result = encode_video_stream(&frames, &default_config());

        assert_eq!(result.stats.total_frames, 4);
        assert!(result.frames[0].is_keyframe);
        assert!(result.frames[0].som_matrix.is_some());
        assert!(result.stats.keyframe_count >= 1);
    }

    #[test]
    fn som_serialization_roundtrip() {
        let frame = make_frame(0, 0);
        let block_config = BlockConfig::new(4, 3);
        let blocks = extract_blocks(
            &frame_to_raw(&frame),
            frame.width(),
            frame.height(),
            &block_config,
        );
        let mut som = SomMatrix::new(4, 4, block_config.block_dim(), &blocks);
        som.train(&blocks, 2, 0.3);

        let bytes = serialize_som(&som);
        assert_eq!(bytes.len(), som.neuron_count() * som.vector_dim * 4);

        let restored = deserialize_som(&bytes, som.grid_width, som.grid_height, som.vector_dim);
        assert_eq!(restored.weights, som.weights);
    }

    #[test]
    fn indices_roundtrip_through_decompression() {
        let frames = (0..2).map(|i| make_frame(i, 0)).collect::<Vec<_>>();
        let result = encode_video_stream(&frames, &default_config());

        for (k, enc) in result.frames.iter().enumerate() {
            let indices =
                decompress_indices(&enc.compressed_indices).expect("indices must decompress");
            let block_count = (frames[k].width() as usize / 4) * (frames[k].height() as usize / 4);
            assert_eq!(indices.len(), block_count);
            assert!(indices.iter().all(|&i| (i as usize) < 256));
        }
    }

    #[test]
    fn decode_video_stream_roundtrip() {
        let frames = vec![make_frame(0, 0), make_frame(1, 0), make_frame(2, 5)];
        let config = default_config();
        let result = encode_video_stream(&frames, &config);

        let decoded = decode_video_stream(&result.frames, 16, 16, &config, 30.0)
            .expect("decoding must succeed");
        assert_eq!(decoded.len(), frames.len());
        for (orig, dec) in frames.iter().zip(decoded.iter()) {
            assert_eq!(orig.width(), dec.width());
            assert_eq!(orig.height(), dec.height());
            let psnr = calculate_psnr(orig, dec);
            assert!(psnr > 20.0 || psnr == f64::INFINITY);
        }
    }

    #[test]
    fn identical_frames_yield_few_keyframes() {
        // Bei identischen, gleichfÃ¶rmigen Frames bleibt das trainierte SOM exakt
        // und die PSNR ist unendlich oder weit Ã¼ber dem Schwellenwert.
        let frames = vec![
            make_solid_frame(0, 100),
            make_solid_frame(1, 100),
            make_solid_frame(2, 100),
        ];
        let result = encode_video_stream(&frames, &default_config());

        assert_eq!(result.stats.keyframe_count, 1);
        assert!(!result.frames[1].is_keyframe);
        assert!(!result.frames[2].is_keyframe);
        assert!(result.frames[1].som_matrix.is_none());
        assert!(result.stats.avg_psnr_db.is_finite() || result.stats.avg_psnr_db == f64::INFINITY);
    }

    #[test]
    fn low_threshold_forces_keyframes() {
        // Unrealistisch hohe Schwelle (INFINITY) -> jeder Frame wird neu trainiert.
        let frames = vec![make_frame(0, 0), make_frame(1, 0)];
        let config = SomVideoConfig {
            psnr_threshold_db: f64::INFINITY,
            ..default_config()
        };

        let result = encode_video_stream(&frames, &config);
        assert_eq!(result.stats.keyframe_count, 2);
        assert!(
            result
                .frames
                .iter()
                .all(|f| f.is_keyframe && f.som_matrix.is_some())
        );
    }

    #[test]
    fn stats_sizes_are_consistent() {
        let frames = vec![make_frame(0, 0), make_frame(1, 5), make_frame(2, 10)];
        let result = encode_video_stream(&frames, &default_config());

        let expected_raw: u64 = frames
            .iter()
            .map(|f| (f.width() as u64) * (f.height() as u64) * 3)
            .sum();
        let expected_compressed: u64 = result
            .frames
            .iter()
            .map(|f| f.compressed_indices.len() as u64)
            .sum();

        assert_eq!(result.stats.raw_size, expected_raw);
        assert_eq!(result.stats.compressed_size, expected_compressed);
        assert!(result.stats.compression_ratio > 0.0);
    }

    #[test]
    #[should_panic(expected = "frames must not be empty")]
    fn rejects_empty_stream() {
        encode_video_stream(&[], &default_config());
    }

    #[test]
    #[should_panic(expected = "train_epochs must be > 0")]
    fn rejects_zero_epochs() {
        let frames = vec![make_frame(0, 0)];
        let config = SomVideoConfig {
            train_epochs: 0,
            ..default_config()
        };
        encode_video_stream(&frames, &config);
    }

    #[test]
    fn encode_and_decode_with_progress_callbacks() {
        let frames = vec![make_frame(0, 0), make_frame(1, 10)];
        let mut progress_counts = Vec::new();
        let result = encode_video_stream_with_progress(&frames, &default_config(), |prog| {
            progress_counts.push(prog.frame_index);
        });
        assert_eq!(progress_counts, vec![1, 2]);

        let mut decode_counts = Vec::new();
        let decoded = decode_video_stream_with_progress(
            &result.frames,
            16,
            16,
            &default_config(),
            30.0,
            |idx, total| {
                decode_counts.push((idx, total));
            },
        ).expect("decoding should succeed");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decode_counts, vec![(1, 2), (2, 2)]);
    }
}