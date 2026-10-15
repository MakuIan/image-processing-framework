//! Komprimierung von SOM-Blockindizes.
//!
//! Topologische SOMs mappen ähnliche Bildmuster auf benachbarte Indizes, sodass
//! lange Wiederholungen gleicher Indizes typisch sind. Deshalb wird erst ein
//! Run-Length-Encoding (RLE) angewendet und anschließend Deflate (via `flate2`),
//! um auch die verbleibenden Muster weiter auszunutzen.

use flate2::{
    Compression,
    write::{ZlibDecoder, ZlibEncoder},
};
use std::io::Write;

/// Kodiert `data` als RLE: Jede maximale Sequenz gleicher Bytes wird zu
/// `[Länge (u8), Wert (u8)]` komprimiert. Läuft länger als 255 Byte, wird der
/// Run in mehrere Pairs aufgeteilt.
pub fn encode_rle(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2 + data.len() / 255);
    let mut i = 0usize;
    while i < data.len() {
        let value = data[i];
        let mut run_len: usize = 1;
        while i + run_len < data.len() && data[i + run_len] == value && run_len < 255 {
            run_len += 1;
        }
        out.push(run_len as u8);
        out.push(value);
        i += run_len;
    }
    out
}

/// Dekodiert RLE-Daten, die von [`encode_rle`] erzeugt wurden.
///
/// Schlägt fehl, wenn die Länge ungerade ist, ein Pair fehlt (unvollständige Daten)
/// oder ein Run-Länge-Byte null ist.
pub fn decode_rle(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() % 2 != 0 {
        return Err(format!(
            "Ungültige RLE-Länge: {} (muss gerade sein)",
            data.len()
        ));
    }
    let mut out = Vec::new();
    for chunk in data.chunks_exact(2) {
        let run_len = chunk[0] as usize;
        let value = chunk[1];
        if run_len == 0 {
            return Err("Ungültige RLE-Länge: 0 gefunden".to_string());
        }
        out.extend(std::iter::repeat_n(value, run_len));
    }
    Ok(out)
}

/// Komprimiert `data` mit Deflate (Zlib-Wrapper) via [`flate2::write::ZlibEncoder`].
pub fn compress_deflate(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data)?;
    encoder.finish()
}

/// Dekomprimiert zuvor mit [`compress_deflate`] erzeugte Daten.
pub fn decompress_deflate(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    let mut decoder = ZlibDecoder::new(Vec::new());
    decoder.write_all(data)?;
    decoder.finish()
}

/// Kombinierte Pipeline: Erst RLE, dann Deflate.
pub fn compress_indices(indices: &[u8]) -> Vec<u8> {
    let rle = encode_rle(indices);
    match compress_deflate(&rle) {
        Ok(compressed) => compressed,
        Err(err) => {
            eprintln!(
                "Warnung: Deflate-Kompression fehlgeschlagen ({}), RLE wird unverändert geliefert",
                err
            );
            rle
        }
    }
}

/// Inverse der [`compress_indices`]-Pipeline: Deflate, dann RLE.
pub fn decompress_indices(compressed: &[u8]) -> Result<Vec<u8>, String> {
    let rle = decompress_deflate(compressed)
        .map_err(|e| format!("Deflate-Dekompression fehlgeschlagen: {e}"))?;
    decode_rle(&rle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rle_roundtrip_basic() {
        let data = b"aaaabbbcc";
        let encoded = encode_rle(data);
        assert_eq!(encoded, [4, b'a', 3, b'b', 2, b'c']);
        assert_eq!(decode_rle(&encoded).unwrap(), data.to_vec());
    }

    #[test]
    fn rle_empty() {
        let encoded = encode_rle(b"");
        assert!(encoded.is_empty());
        assert_eq!(decode_rle(&encoded).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn rle_single_byte() {
        let encoded = encode_rle(&[42]);
        assert_eq!(encoded, [1, 42]);
        assert_eq!(decode_rle(&encoded).unwrap(), vec![42]);
    }

    #[test]
    fn rle_splits_runs_longer_than_255() {
        let data = vec![7u8; 600];
        let encoded = encode_rle(&data);
        // 255 + 255 + 90 -> drei Pairs
        assert_eq!(encoded, [255, 7, 255, 7, 90, 7]);
        assert_eq!(decode_rle(&encoded).unwrap(), data);
    }

    #[test]
    fn rle_alternating_bytes_no_compression_gain() {
        // [0, 1, 2, 3, ...] ohne Wiederholungen
        let data: Vec<u8> = (0..100).collect();
        let encoded = encode_rle(&data);
        // Da jedes Byte einzeln vorkommt: 100 * 2 = 200 Bytes
        assert_eq!(encoded.len(), data.len() * 2);
        assert_eq!(decode_rle(&encoded).unwrap(), data);
    }

    #[test]
    fn rle_decode_rejects_odd_length() {
        assert!(decode_rle(&[1, b'a', b'b']).is_err());
    }

    #[test]
    fn rle_decode_rejects_zero_run_length() {
        assert!(decode_rle(&[0, b'a']).is_err());
    }

    #[test]
    fn rle_roundtrip_random() {
        // Deterministische Pseudo-Zufallsdaten mit gemischten Runs
        let mut state = 0x2F6E2B1u32;
        let mut rand = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as u8
        };
        let mut data = Vec::with_capacity(10_000);
        while data.len() < 10_000 {
            let value = rand();
            let run = (rand() % 30) as usize + 1;
            data.extend(std::iter::repeat_n(value, run));
        }
        data.truncate(10_000);
        let encoded = encode_rle(&data);
        assert!(encoded.len() < data.len());
        assert_eq!(decode_rle(&encoded).unwrap(), data);
    }

    #[test]
    fn deflate_roundtrip() {
        // Wiederholendes Muster, das gut komprimiert
        let data = vec![42u8; 1000];
        let compressed = compress_deflate(&data).unwrap();
        assert!(compressed.len() < data.len());
        assert_eq!(decompress_deflate(&compressed).unwrap(), data);
    }

    #[test]
    fn deflate_roundtrip_empty() {
        let compressed = compress_deflate(b"").unwrap();
        assert_eq!(decompress_deflate(&compressed).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn deflate_rejects_garbage() {
        assert!(decompress_deflate(b"definitely not zlib").is_err());
    }

    #[test]
    fn pipeline_roundtrip_typical_indices() {
        // Typisches SOM-Indexmuster: lange Wiederholungen mit Sprüngen
        let mut indices = Vec::new();
        for run in [40, 120, 7, 255, 256, 3, 500] {
            let value = (indices.len() % 256) as u8;
            indices.extend(std::iter::repeat_n(value, run));
        }
        let compressed = compress_indices(&indices);
        assert!(compressed.len() < indices.len());
        let restored = decompress_indices(&compressed).unwrap();
        assert_eq!(restored, indices);
    }

    #[test]
    fn pipeline_roundtrip_incompressible() {
        // Pseudo-zufällige 64-bit Daten
        let mut state = 0xDEAD_BEEF_CAFE_BABE_u64;
        let data: Vec<u8> = (0..5000)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as u8
            })
            .collect();
        let restored = decompress_indices(&compress_indices(&data)).unwrap();
        assert_eq!(restored, data);
    }

    #[test]
    fn pipeline_rejects_garbage() {
        assert!(decompress_indices(b"not a valid payload").is_err());
    }
}
