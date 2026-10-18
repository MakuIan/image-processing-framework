//! Self-Organizing Map (SOM, Kohonen-Netz) für die Vektorquantisierung von
//! Bildblöcken.
//!
//! Die Neuronen liegen auf einem 2D-Gitter von `grid_width x grid_height`
//! (z. B. 16x16 = 256 Codevektoren). Jedes Neuron speichert einen
//! Gewichtsvektor der Dimension `vector_dim` (z. B. 48 für 4x4-RGB-Blöcke),
//! zeilenweise (row-major) in `weights` angeordnet: `weights[i]` ist der
//! Codevektor des Neurons mit 1D-Index `i`.
//!
//! Das Training folgt dem klassischen Kohonen-Verfahren: Für jedes Sample
//! wird die Best Matching Unit (BMU) bestimmt — das Neuron mit dem kleinsten
//! quadratischen euklidischen Abstand im Merkmalsraum — und alle Neuronen in
//! ihrer Gitter-Nachbarschaft (Radius `sigma`) in Richtung des Samples
//! bewegt:
//!
//! ```text
//! w_i <- w_i + alpha(t) * exp(-d^2 / (2 * sigma(t)^2)) * (sample - w_i)
//! ```
//!
//! Lernrate `alpha(t)` und Nachbarschaftsradius `sigma(t)` klingen während
//! des Trainings ab.
//!
//! Performance:
//! - **SIMD-Vektorisierung (AVX2 + FMA / Scalar-Fallback)**: Beschleunigt die
//!   Abstandsberechnung ([`SomMatrix::squared_distance`]), die BMU-Suche und
//!   die Vektor-Aktualisierungen auf Instruktionsebene um ein Vielfaches.
//! - **Rayon (Multithreading)**: Beschleunigt die unabhängige BMU-Suche aller
//!   Blöcke in [`SomMatrix::quantize_all`].
//! - Beim Training mit typischen Gittergrößen (bis 256 Neuronen) vermeidet
//!   die sequentielle SIMD-Aktualisierung den Thread-Synchronisations-Overhead pro Sample,
//!   sodass das Training extrem schnell im CPU-L1/L2-Cache abläuft.

use crate::block::BlockBuffer;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
mod simd_x86 {
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn hsum256_ps(v: __m256) -> f32 {
        let vlow = _mm256_castps256_ps128(v);
        let vhigh = _mm256_extractf128_ps(v, 1);
        let v128 = _mm_add_ps(vlow, vhigh);
        let shuf = _mm_movehl_ps(v128, v128);
        let sums = _mm_add_ps(v128, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 1);
        let res = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(res)
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn squared_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();

        let chunks_a = a.chunks_exact(16);
        let chunks_b = b.chunks_exact(16);
        let rem_a = chunks_a.remainder();
        let rem_b = chunks_b.remainder();

        for (ca, cb) in chunks_a.zip(chunks_b) {
            let va0 = unsafe { _mm256_loadu_ps(ca.as_ptr()) };
            let vb0 = unsafe { _mm256_loadu_ps(cb.as_ptr()) };
            let diff0 = _mm256_sub_ps(va0, vb0);
            sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);

            let va1 = unsafe { _mm256_loadu_ps(ca.as_ptr().add(8)) };
            let vb1 = unsafe { _mm256_loadu_ps(cb.as_ptr().add(8)) };
            let diff1 = _mm256_sub_ps(va1, vb1);
            sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
        }

        let mut sum = _mm256_add_ps(sum0, sum1);

        let chunks8_a = rem_a.chunks_exact(8);
        let chunks8_b = rem_b.chunks_exact(8);
        let final_rem_a = chunks8_a.remainder();
        let final_rem_b = chunks8_b.remainder();

        for (ca, cb) in chunks8_a.zip(chunks8_b) {
            let va = unsafe { _mm256_loadu_ps(ca.as_ptr()) };
            let vb = unsafe { _mm256_loadu_ps(cb.as_ptr()) };
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let mut total = unsafe { hsum256_ps(sum) };
        for (x, y) in final_rem_a.iter().zip(final_rem_b.iter()) {
            let d = x - y;
            total += d * d;
        }
        total
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn update_vector_avx2(code: &mut [f32], sample: &[f32], eta: f32) {
        let v_eta = _mm256_set1_ps(eta);
        let rem_len = code.len() % 8;
        let split_idx = code.len() - rem_len;
        let (main_code, rem_code) = code.split_at_mut(split_idx);
        let (main_sample, rem_sample) = sample.split_at(split_idx);

        for (cc, cs) in main_code
            .chunks_exact_mut(8)
            .zip(main_sample.chunks_exact(8))
        {
            let vc = unsafe { _mm256_loadu_ps(cc.as_ptr()) };
            let vs = unsafe { _mm256_loadu_ps(cs.as_ptr()) };
            let diff = _mm256_sub_ps(vs, vc);
            let updated = _mm256_fmadd_ps(v_eta, diff, vc);
            unsafe { _mm256_storeu_ps(cc.as_mut_ptr(), updated) };
        }

        for (c, s) in rem_code.iter_mut().zip(rem_sample.iter()) {
            *c += eta * (*s - *c);
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn find_bmu_avx2(weights: &[Vec<f32>], sample: &[f32]) -> usize {
        let mut best_index = 0usize;
        let mut best_dist = f32::INFINITY;
        for (i, code) in weights.iter().enumerate() {
            let dist = unsafe { squared_distance_avx2(sample, code) };
            if dist < best_dist {
                best_dist = dist;
                best_index = i;
            }
        }
        best_index
    }

    /// Wie [`find_bmu_avx2`], aber über einen flachen Gewichtsspeicher mit
    /// `vector_dim` Stichproben je Neuron (Zeilenmajor).
    ///
    /// Die sequenzielle Speicheranordnung erlaubt dem Hardware-Prefetcher,
    /// die Gewichte als durchgehenden Stream zu laden. Fuer Vielfache von 16
    /// (der heiße Pfad: 48, 64, 192 ...) werden zudem je zwei Neuronen
    /// interleaved verarbeitet, damit die FMA-Latenz der beiden unabhaengigen
    /// Akkumulatoren-Paare versteckt wird. Die Summationsreihenfolge bleibt
    /// bit-identisch zu [`squared_distance_avx2`] (gleiche even/odd-Gruppierung).
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn find_bmu_flat_avx2(
        flat_weights: &[f32],
        vector_dim: usize,
        sample: &[f32],
    ) -> usize {
        debug_assert!(vector_dim > 0 && flat_weights.len().is_multiple_of(vector_dim));
        debug_assert_eq!(
            sample.len(),
            vector_dim,
            "sample must have exactly vector_dim entries"
        );
        let neuron_count = flat_weights.len() / vector_dim;
        let mut best_index = 0usize;
        let mut best_dist = f32::INFINITY;

        let mut i = 0usize;
        if vector_dim.is_multiple_of(16) {
            let chunk_count = vector_dim / 16;
            while i + 2 <= neuron_count {
                let code0 = &flat_weights[i * vector_dim..(i + 1) * vector_dim];
                let code1 = &flat_weights[(i + 1) * vector_dim..(i + 2) * vector_dim];

                // Genau wie squared_distance_avx2 geht bei jedem 16er-Chunk
                // die untere Haelfte (8 lanes) in even und die obere in odd;
                // die Akkumulationsreihenfolge bleibt bit-identisch, waehrend
                // die zwei Neuronen die FMA-Pipeline fuellen.
                let mut even0 = _mm256_setzero_ps();
                let mut odd0 = _mm256_setzero_ps();
                let mut even1 = _mm256_setzero_ps();
                let mut odd1 = _mm256_setzero_ps();

                let mut c = 0usize;
                while c + 1 < chunk_count {
                    // Chunk c.
                    let s = unsafe { sample.as_ptr().add(c * 16) };
                    let vs_low = unsafe { _mm256_loadu_ps(s) };
                    let vs_high = unsafe { _mm256_loadu_ps(s.add(8)) };
                    let off = c * 16;

                    let d =
                        _mm256_sub_ps(vs_low, unsafe { _mm256_loadu_ps(code0.as_ptr().add(off)) });
                    even0 = _mm256_fmadd_ps(d, d, even0);
                    let d = _mm256_sub_ps(vs_high, unsafe {
                        _mm256_loadu_ps(code0.as_ptr().add(off + 8))
                    });
                    odd0 = _mm256_fmadd_ps(d, d, odd0);
                    let d =
                        _mm256_sub_ps(vs_low, unsafe { _mm256_loadu_ps(code1.as_ptr().add(off)) });
                    even1 = _mm256_fmadd_ps(d, d, even1);
                    let d = _mm256_sub_ps(vs_high, unsafe {
                        _mm256_loadu_ps(code1.as_ptr().add(off + 8))
                    });
                    odd1 = _mm256_fmadd_ps(d, d, odd1);

                    // Chunk c+1.
                    let s = unsafe { sample.as_ptr().add((c + 1) * 16) };
                    let vs_low = unsafe { _mm256_loadu_ps(s) };
                    let vs_high = unsafe { _mm256_loadu_ps(s.add(8)) };
                    let off = (c + 1) * 16;

                    let d =
                        _mm256_sub_ps(vs_low, unsafe { _mm256_loadu_ps(code0.as_ptr().add(off)) });
                    even0 = _mm256_fmadd_ps(d, d, even0);
                    let d = _mm256_sub_ps(vs_high, unsafe {
                        _mm256_loadu_ps(code0.as_ptr().add(off + 8))
                    });
                    odd0 = _mm256_fmadd_ps(d, d, odd0);
                    let d =
                        _mm256_sub_ps(vs_low, unsafe { _mm256_loadu_ps(code1.as_ptr().add(off)) });
                    even1 = _mm256_fmadd_ps(d, d, even1);
                    let d = _mm256_sub_ps(vs_high, unsafe {
                        _mm256_loadu_ps(code1.as_ptr().add(off + 8))
                    });
                    odd1 = _mm256_fmadd_ps(d, d, odd1);

                    c += 2;
                }
                if c < chunk_count {
                    // Ungerade Chunk-Anzahl: letzter Chunk, gleiches Schema.
                    let s = unsafe { sample.as_ptr().add(c * 16) };
                    let vs_low = unsafe { _mm256_loadu_ps(s) };
                    let vs_high = unsafe { _mm256_loadu_ps(s.add(8)) };
                    let off = c * 16;

                    let d =
                        _mm256_sub_ps(vs_low, unsafe { _mm256_loadu_ps(code0.as_ptr().add(off)) });
                    even0 = _mm256_fmadd_ps(d, d, even0);
                    let d = _mm256_sub_ps(vs_high, unsafe {
                        _mm256_loadu_ps(code0.as_ptr().add(off + 8))
                    });
                    odd0 = _mm256_fmadd_ps(d, d, odd0);
                    let d =
                        _mm256_sub_ps(vs_low, unsafe { _mm256_loadu_ps(code1.as_ptr().add(off)) });
                    even1 = _mm256_fmadd_ps(d, d, even1);
                    let d = _mm256_sub_ps(vs_high, unsafe {
                        _mm256_loadu_ps(code1.as_ptr().add(off + 8))
                    });
                    odd1 = _mm256_fmadd_ps(d, d, odd1);
                }

                let dist0 = unsafe { hsum256_ps(_mm256_add_ps(even0, odd0)) };
                let dist1 = unsafe { hsum256_ps(_mm256_add_ps(even1, odd1)) };
                if dist0 < best_dist {
                    best_dist = dist0;
                    best_index = i;
                }
                if dist1 < best_dist {
                    best_dist = dist1;
                    best_index = i + 1;
                }
                i += 2;
            }
        }

        // Restliche Neuronen (ungerade Anzahl bzw. nicht teilbare Dimension)
        // ueber den klassischen Einzel-Distanzpfad.
        while i < neuron_count {
            let code = &flat_weights[i * vector_dim..(i + 1) * vector_dim];
            let dist = unsafe { squared_distance_avx2(sample, code) };
            if dist < best_dist {
                best_dist = dist;
                best_index = i;
            }
            i += 1;
        }
        best_index
    }
}

mod simd_fallback {
    #[inline]
    pub fn squared_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        let chunks_a = a.chunks_exact(4);
        let chunks_b = b.chunks_exact(4);
        let rem_a = chunks_a.remainder();
        let rem_b = chunks_b.remainder();

        for (ca, cb) in chunks_a.zip(chunks_b) {
            let d0 = ca[0] - cb[0];
            let d1 = ca[1] - cb[1];
            let d2 = ca[2] - cb[2];
            let d3 = ca[3] - cb[3];
            sum0 += d0 * d0;
            sum1 += d1 * d1;
            sum2 += d2 * d2;
            sum3 += d3 * d3;
        }

        let mut total = (sum0 + sum1) + (sum2 + sum3);
        for (x, y) in rem_a.iter().zip(rem_b.iter()) {
            let d = x - y;
            total += d * d;
        }
        total
    }

    #[inline]
    pub fn update_vector_scalar(code: &mut [f32], sample: &[f32], eta: f32) {
        let rem_len = code.len() % 4;
        let split_idx = code.len() - rem_len;
        let (main_code, rem_code) = code.split_at_mut(split_idx);
        let (main_sample, rem_sample) = sample.split_at(split_idx);

        for (c, s) in main_code
            .chunks_exact_mut(4)
            .zip(main_sample.chunks_exact(4))
        {
            c[0] += eta * (s[0] - c[0]);
            c[1] += eta * (s[1] - c[1]);
            c[2] += eta * (s[2] - c[2]);
            c[3] += eta * (s[3] - c[3]);
        }

        for (c, s) in rem_code.iter_mut().zip(rem_sample.iter()) {
            *c += eta * (*s - *c);
        }
    }

    #[inline]
    pub fn find_bmu_scalar(weights: &[Vec<f32>], sample: &[f32]) -> usize {
        let mut best_index = 0usize;
        let mut best_dist = f32::INFINITY;
        for (i, code) in weights.iter().enumerate() {
            let dist = squared_distance_scalar(sample, code);
            if dist < best_dist {
                best_dist = dist;
                best_index = i;
            }
        }
        best_index
    }
}

/// Self-Organizing Map (Kohonen-Netz) über einem 2D-Neurongitter.
#[derive(Debug, Clone)]
pub struct SomMatrix {
    /// Gitterweite in x-Richtung (z. B. `16`).
    pub grid_width: usize,
    /// Gitterhöhe in y-Richtung (z. B. `16`).
    pub grid_height: usize,
    /// Dimension der Codevektoren (z. B. `48`).
    pub vector_dim: usize,
    /// Codevektor pro Neuron: `weights[i]` ist der Gewichtvektor des Neurons
    /// mit 1D-Index `i` (`0..grid_width * grid_height`), zeilenweise
    /// (row-major) auf dem Gitter angeordnet.
    pub weights: Vec<Vec<f32>>,
}

impl SomMatrix {
    /// Erzeugt eine neue SOM mit `grid_width x grid_height` Neuronen.
    ///
    /// Die Startgewichte werden per Neuron zufällig (mit Wiederholung) aus
    /// `samples` gezogen, sodass die initialen Codevektoren typische
    /// Trainingsblöcke abbilden.
    ///
    /// # Panics
    /// Panics, wenn eine Dimension null ist, das Gitter mehr als 256 Neuronen
    /// hätte (die quantisierten Indizes sind `u8`) oder ein Sample nicht
    /// genau `vector_dim` Samples hat.
    pub fn new(
        grid_width: usize,
        grid_height: usize,
        vector_dim: usize,
        samples: &[Vec<f32>],
    ) -> Self {
        assert!(grid_width > 0, "grid_width must be > 0, got {grid_width}");
        assert!(
            grid_height > 0,
            "grid_height must be > 0, got {grid_height}"
        );
        assert!(vector_dim > 0, "vector_dim must be > 0, got {vector_dim}");
        assert!(!samples.is_empty(), "samples must not be empty");
        assert!(
            grid_width * grid_height <= u8::MAX as usize + 1,
            "grid must not exceed 256 neurons (quantized indices are u8), got {}x{}",
            grid_width,
            grid_height
        );
        for (i, s) in samples.iter().enumerate() {
            assert_eq!(
                s.len(),
                vector_dim,
                "sample {i} has {} sample(s), expected {vector_dim}",
                s.len()
            );
        }

        let neuron_count = grid_width * grid_height;
        let mut rng = thread_rng();
        let weights = (0..neuron_count)
            .map(|_| samples[rng.gen_range(0..samples.len())].clone())
            .collect();

        Self {
            grid_width,
            grid_height,
            vector_dim,
            weights,
        }
    }

    /// Gesamtzahl der Neuronen im Gitter.
    pub fn neuron_count(&self) -> usize {
        self.grid_width * self.grid_height
    }

    /// 2D-Position `(y, x)` eines Neurons mit 1D-Index `index` im Gitter.
    fn grid_position(index: usize, width: usize) -> (usize, usize) {
        (index / width, index % width)
    }

    /// Quadratische euklidische Distanz zweier Vektoren (SIMD-beschleunigt).
    #[inline]
    pub fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { simd_x86::squared_distance_avx2(a, b) };
            }
        }
        simd_fallback::squared_distance_scalar(a, b)
    }

    /// Findet die Best Matching Unit: das Neuron, dessen Codevektor den
    /// kleinsten quadratischen euklidischen Abstand zum Sample hat.
    #[inline]
    fn find_bmu(&self, sample: &[f32]) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { simd_x86::find_bmu_avx2(&self.weights, sample) };
            }
        }
        simd_fallback::find_bmu_scalar(&self.weights, sample)
    }

    /// Trainiert die SOM mit dem Kohonen-Verfahren.
    ///
    /// Pro Epoche `e` (von `0` bis `epochs - 1`) klingen ab:
    /// - die Lernrate `alpha(e)` **linear** von `initial_lr` bis
    ///   `initial_lr / epochs`,
    /// - der Nachbarschaftsradius `sigma(e)` **exponentiell** von der halben
    ///   Gitterkante bis `1.0`.
    ///
    /// Für jedes Sample wird die BMU bestimmt und jedes Neuron `i` innerhalb
    /// eines Gitter-Radius von `sigma` aktualisiert:
    /// `w_i += alpha * exp(-d^2 / (2 * sigma^2)) * (sample - w_i)`, wobei `d`
    /// der Gitterabstand von `i` zur BMU ist.
    ///
    /// # Panics
    /// Panics, wenn `samples` leer ist, `epochs` null ist, `initial_lr` nicht
    /// endlich und positiv ist oder ein Sample nicht genau `vector_dim`
    /// Samples hat.
    pub fn train(&mut self, samples: &[Vec<f32>], epochs: usize, initial_lr: f32) {
        assert!(!samples.is_empty(), "samples must not be empty");
        assert!(epochs > 0, "epochs must be > 0, got {epochs}");
        assert!(
            initial_lr.is_finite() && initial_lr > 0.0,
            "initial_lr must be finite and > 0, got {initial_lr}"
        );

        let initial_sigma = (self.grid_width.max(self.grid_height) / 2).max(1) as f32;
        let final_sigma = 1.0_f32;

        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;

        for epoch in 0..epochs {
            // Lineare Abklingung: initial_lr -> initial_lr / epochs (bleibt > 0).
            let alpha = initial_lr * (epochs - epoch) as f32 / epochs as f32;
            // Exponentielle Abklingung: initial_sigma -> final_sigma.
            let sigma = if epochs == 1 {
                initial_sigma
            } else {
                initial_sigma
                    * (final_sigma / initial_sigma).powf(epoch as f32 / (epochs - 1) as f32)
            };
            let sigma_sq = sigma * sigma;

            for (sample_idx, sample) in samples.iter().enumerate() {
                assert_eq!(
                    sample.len(),
                    self.vector_dim,
                    "sample {sample_idx} has {} sample(s), expected {}",
                    sample.len(),
                    self.vector_dim
                );

                let bmu = if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        simd_x86::find_bmu_avx2(&self.weights, sample)
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        self.find_bmu(sample)
                    }
                } else {
                    simd_fallback::find_bmu_scalar(&self.weights, sample)
                };

                self.update_neighborhood_simd(sample, bmu, alpha, sigma_sq, use_avx2);
            }
        }
    }

    /// Aktualisiert alle Neuronen innerhalb des Radius `sigma` (Gitterdistanz
    /// `d` mit `d <= sigma`) in Richtung des Samples unter Nutzung von SIMD.
    fn update_neighborhood_simd(
        &mut self,
        sample: &[f32],
        bmu: usize,
        alpha: f32,
        sigma_sq: f32,
        use_avx2: bool,
    ) {
        let (bmy, bmx) = Self::grid_position(bmu, self.grid_width);
        let width = self.grid_width;
        let two_sigma_sq = 2.0 * sigma_sq;

        for (i, code) in self.weights.iter_mut().enumerate() {
            let (ny, nx) = Self::grid_position(i, width);
            let dy = ny as i32 - bmy as i32;
            let dx = nx as i32 - bmx as i32;
            let dist_sq = (dy * dy + dx * dx) as f32;
            if dist_sq > sigma_sq {
                continue;
            }

            let eta = alpha * (-dist_sq / two_sigma_sq).exp();
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    simd_x86::update_vector_avx2(code, sample, eta);
                }
            } else {
                simd_fallback::update_vector_scalar(code, sample, eta);
            }
        }
    }

    /// Quantisiert einen Block: liefert den 1D-Index (`0..255`) der BMU,
    /// also des Codevektors mit dem kleinsten quadratischen euklidischen
    /// Abstand.
    ///
    /// # Panics
    /// Panics, wenn `block` nicht genau `vector_dim` Samples hat.
    pub fn quantize_block(&self, block: &[f32]) -> u8 {
        assert_eq!(
            block.len(),
            self.vector_dim,
            "block has {} sample(s), expected {}",
            block.len(),
            self.vector_dim
        );
        self.find_bmu(block) as u8
    }

    /// Quantisiert alle Blöcke. Die BMU-Suche pro Block ist unabhängig und
    /// wird daher parallel mit Rayon und pro Block mit SIMD ausgeführt;
    /// die Reihenfolge bleibt erhalten.
    pub fn quantize_all(&self, blocks: &[Vec<f32>]) -> Vec<u8> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return blocks
                    .par_iter()
                    .map(|b| {
                        assert_eq!(
                            b.len(),
                            self.vector_dim,
                            "block has {} sample(s), expected {}",
                            b.len(),
                            self.vector_dim
                        );
                        unsafe { simd_x86::find_bmu_avx2(&self.weights, b) as u8 }
                    })
                    .collect();
            }
        }

        blocks.par_iter().map(|b| self.quantize_block(b)).collect()
    }

    /// Quantisiert alle Blöcke eines flachen [`BlockBuffer`]s (Zero-Allocation
    /// Pfad): Die BMU-Suche pro Block ist unabhängig und wird parallel mit
    /// Rayon über die zusammenhängenden Speicherbereiche und pro Block mit
    /// SIMD ausgeführt; die Reihenfolge bleibt erhalten.
    ///
    /// Gegenüber [`SomMatrix::quantize_all`] entfallen die Deref-Sprünge über
    /// verschachtelte Vektoren und die Längen-Prüfung pro Block (sie erfolgt
    /// einzig am Puffer). Die Gewichte werden einmalig in einen flachen
    /// Puffer gespiegelt, damit die BMU-Suche sie als sequenziellen Stream
    /// laden kann (Prefetcher-freundlich); die Ergebnisse bleiben bit-identisch.
    ///
    /// # Panics
    /// Panics, wenn die Blockdimension des Puffers nicht `vector_dim` ist.
    pub fn quantize_all_flat(&self, blocks: &BlockBuffer) -> Vec<u8> {
        let dim = blocks.dim();
        assert_eq!(
            dim, self.vector_dim,
            "block has {dim} sample(s), expected {}",
            self.vector_dim
        );

        let data = blocks.data();

        // 256 x 48 f32 = 49 KiB; eine Kopie pro Aufruf (nicht pro Block).
        let flat_weights: Vec<f32> = self.weights.iter().flatten().copied().collect();

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return data
                    .par_chunks(dim)
                    .map(|b| unsafe { simd_x86::find_bmu_flat_avx2(&flat_weights, dim, b) as u8 })
                    .collect();
            }
        }

        data.par_chunks(dim)
            .map(|b| self.quantize_block(b))
            .collect()
    }

    /// Liefert den Codevektor (Lookup-Eintrag) am 1D-Index `index`.
    ///
    /// # Panics
    /// Panics, wenn `index` außerhalb des Gitters liegt.
    pub fn lookup_vector(&self, index: u8) -> &[f32] {
        &self.weights[index as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2x2-Gitter, 2D-Codevektoren, bekannte Gewichte.
    fn test_som() -> SomMatrix {
        SomMatrix {
            grid_width: 2,
            grid_height: 2,
            vector_dim: 2,
            weights: vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
        }
    }

    #[test]
    fn new_initializes_weights_from_samples() {
        let samples = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let som = SomMatrix::new(2, 2, 3, &samples);

        assert_eq!(som.neuron_count(), 4);
        assert_eq!(som.weights.len(), 4);
        for w in &som.weights {
            assert_eq!(w.len(), 3);
            assert!(
                w == &samples[0] || w == &samples[1],
                "initial weight {w:?} is not one of the samples"
            );
        }
    }

    #[test]
    fn quantize_block_picks_closest_codevector() {
        let som = test_som();
        assert_eq!(som.quantize_block(&[0.1, 0.1]), 0);
        assert_eq!(som.quantize_block(&[0.1, 0.9]), 1);
        assert_eq!(som.quantize_block(&[0.9, 0.1]), 2);
        assert_eq!(som.quantize_block(&[0.9, 0.9]), 3);
    }

    #[test]
    fn quantize_all_matches_quantize_block() {
        let som = test_som();
        let blocks = vec![vec![0.2, 0.2], vec![0.8, 0.2], vec![0.8, 0.8]];

        let codes = som.quantize_all(&blocks);
        assert_eq!(codes, vec![0u8, 2, 3]);

        for (block, code) in blocks.iter().zip(codes.iter()) {
            assert_eq!(som.quantize_block(block), *code);
        }
    }

    #[test]
    fn quantize_all_flat_matches_quantize_all() {
        let som = test_som();
        let blocks = vec![
            vec![0.2, 0.2],
            vec![0.8, 0.2],
            vec![0.8, 0.8],
            vec![0.0, 1.0],
        ];

        let nested = som.quantize_all(&blocks);
        let flat = BlockBuffer::new(blocks.iter().flatten().copied().collect(), 2);

        assert_eq!(flat.len(), blocks.len());
        assert_eq!(som.quantize_all_flat(&flat), nested);
    }

    #[test]
    fn quantize_all_flat_matches_quantize_block_at_48_dims() {
        // Uebt den flachen Interleaved-BMU-Pfad (vector_dim % 16 == 0) gegen
        // den Einzel-Distanzpfad von quantize_block.
        let dim = 48usize;
        let samples: Vec<Vec<f32>> = (0..64)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * 7 + j * 13) % 256) as f32 / 255.0)
                    .collect()
            })
            .collect();
        let som = SomMatrix::new(16, 16, dim, &samples);

        let blocks = BlockBuffer::new(
            (0..100 * dim)
                .map(|i| ((i * 3) % 256) as f32 / 256.0)
                .collect(),
            dim,
        );

        let flat = som.quantize_all_flat(&blocks);
        assert_eq!(flat.len(), 100);
        for (k, block) in blocks.blocks().enumerate() {
            assert_eq!(flat[k], som.quantize_block(block), "block {k}");
        }
    }

    #[test]
    #[should_panic(expected = "block has 4 sample(s), expected 2")]
    fn quantize_all_flat_rejects_wrong_dim() {
        let som = test_som();
        let buffer = BlockBuffer::new(vec![0.0; 8], 4);
        let _ = som.quantize_all_flat(&buffer);
    }

    #[test]
    fn lookup_vector_returns_requested_codevector() {
        let som = test_som();
        assert_eq!(som.lookup_vector(0), &som.weights[0]);
        assert_eq!(som.lookup_vector(3), &som.weights[3]);
        assert_eq!(som.lookup_vector(1), &[0.0, 1.0]);
    }

    #[test]
    fn train_single_step_matches_manual_formula() {
        // alpha = 0.2 (Epoche 0 von 1), sigma = 1 (halbe Gitterkante 2/2),
        // BMU = Neuron 0 (alle Codevektoren gleich weit entfernt -> erster).
        let sample = vec![0.5, 0.5];
        let mut som = test_som();
        som.train(&[sample], 1, 0.2);

        let eta0 = 0.2; // Gitterdistanz 0
        let eta1 = 0.2 * (-0.5f32).exp(); // Gitterdistanz 1

        fn close(actual: f32, expected: f32) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "got {actual}, expected {expected}"
            );
        }

        // Neuron 0: BMU, volle Lernrate.
        close(som.weights[0][0], eta0 * 0.5);
        close(som.weights[0][1], eta0 * 0.5);
        // Neuron 1: Gitterdistanz 1, gedämpfte Lernrate.
        close(som.weights[1][0], eta1 * 0.5);
        close(som.weights[1][1], 1.0 - eta1 * 0.5);
        // Neuron 2: Gitterdistanz 1, gedämpfte Lernrate.
        close(som.weights[2][0], 1.0 - eta1 * 0.5);
        close(som.weights[2][1], eta1 * 0.5);
        // Neuron 3: Gitterdistanz sqrt(2) > sigma -> unverändert.
        close(som.weights[3][0], 1.0);
        close(som.weights[3][1], 1.0);
    }

    #[test]
    fn train_moves_bmu_towards_sample() {
        let sample = vec![0.3, 0.3];
        let before = SomMatrix::squared_distance(&sample, &test_som().weights[0]);

        let mut som = test_som();
        som.train(std::slice::from_ref(&sample), 20, 0.3);

        let after = SomMatrix::squared_distance(&sample, &som.weights[0]);
        assert!(
            after < before,
            "BMU should move closer: before {before}, after {after}"
        );
    }

    #[test]
    fn train_keeps_grid_layout() {
        let mut som = test_som();
        som.train(&[vec![0.5, 0.5], vec![0.1, 0.9]], 5, 0.2);
        assert_eq!(som.grid_width, 2);
        assert_eq!(som.grid_height, 2);
        assert_eq!(som.weights.len(), 4);
        assert!(som.weights.iter().all(|w| w.len() == 2));
    }

    #[test]
    fn simd_and_scalar_squared_distance_match() {
        let v1: Vec<f32> = (0..48).map(|i| (i as f32 * 0.3).sin()).collect();
        let v2: Vec<f32> = (0..48).map(|i| (i as f32 * 0.7).cos()).collect();

        let scalar_dist = simd_fallback::squared_distance_scalar(&v1, &v2);
        let simd_dist = SomMatrix::squared_distance(&v1, &v2);

        assert!(
            (scalar_dist - simd_dist).abs() < 1e-4,
            "Scalar {scalar_dist} and SIMD {simd_dist} differ by {}",
            (scalar_dist - simd_dist).abs()
        );
    }

    #[test]
    fn simd_and_scalar_update_vector_match() {
        let mut code_scalar: Vec<f32> = (0..48).map(|i| i as f32 * 0.1).collect();
        let mut code_simd = code_scalar.clone();
        let sample: Vec<f32> = (0..48).map(|i| (i as f32 * 0.2).sin()).collect();
        let eta = 0.25f32;

        simd_fallback::update_vector_scalar(&mut code_scalar, &sample, eta);
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                simd_x86::update_vector_avx2(&mut code_simd, &sample, eta);
            }
            for (i, (s, a)) in code_scalar.iter().zip(code_simd.iter()).enumerate() {
                assert!(
                    (s - a).abs() < 1e-5,
                    "Mismatch at index {i}: scalar={s}, avx2={a}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "block has 4 sample(s), expected 2")]
    fn quantize_block_rejects_wrong_dim() {
        test_som().quantize_block(&[0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "samples must not be empty")]
    fn new_rejects_empty_samples() {
        let _ = SomMatrix::new(2, 2, 2, &[]);
    }

    #[test]
    #[should_panic(expected = "sample 0 has 3 sample(s), expected 2")]
    fn new_rejects_wrong_dim_samples() {
        let _ = SomMatrix::new(2, 2, 2, &[vec![0.0, 0.0, 0.0]]);
    }

    #[test]
    #[should_panic(expected = "epochs must be > 0")]
    fn train_rejects_zero_epochs() {
        let mut som = test_som();
        som.train(&[vec![0.5, 0.5]], 0, 0.1);
    }

    #[test]
    #[should_panic(expected = "initial_lr must be finite and > 0")]
    fn train_rejects_invalid_lr() {
        let mut som = test_som();
        som.train(&[vec![0.5, 0.5]], 1, 0.0);
    }
}
