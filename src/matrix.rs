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
//! Rayon beschleunigt die unabhängigen Anteile: die parallele BMU-Suche in
//! [`SomMatrix::quantize_all`] sowie die parallele
//! Nachbarschafts-Aktualisierung beim Training (disjunkte Neuronen-Vektoren).
//! Die Reihenfolge der Samples im Training bleibt sequentiell, da jede
//! Aktualisierung auf den jeweils aktuellen Gewichten aufbaut.

use rand::{Rng, thread_rng};
use rayon::prelude::*;

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

    /// Quadratische euklidische Distanz zweier Vektoren.
    #[inline]
    fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Findet die Best Matching Unit: das Neuron, dessen Codevektor den
    /// kleinsten quadratischen euklidischen Abstand zum Sample hat.
    fn find_bmu(&self, sample: &[f32]) -> usize {
        let mut best_index = 0usize;
        let mut best_dist = f32::INFINITY;
        for (i, code) in self.weights.iter().enumerate() {
            let dist = Self::squared_distance(sample, code);
            if dist < best_dist {
                best_dist = dist;
                best_index = i;
            }
        }
        best_index
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

                let bmu = self.find_bmu(sample);
                self.update_neighborhood(sample, bmu, alpha, sigma_sq);
            }
        }
    }

    /// Aktualisiert alle Neuronen innerhalb des Radius `sigma` (Gitterdistanz
    /// `d` mit `d <= sigma`) in Richtung des Samples.
    ///
    /// Die Neuronen halten disjunkte Vektoren, ihre Aktualisierung ist
    /// unabhängig und wird daher parallel mit Rayon ausgeführt.
    fn update_neighborhood(&mut self, sample: &[f32], bmu: usize, alpha: f32, sigma_sq: f32) {
        let (bmy, bmx) = Self::grid_position(bmu, self.grid_width);
        let width = self.grid_width;
        let dim = self.vector_dim;
        let two_sigma_sq = 2.0 * sigma_sq;

        self.weights
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, code)| {
                let (ny, nx) = Self::grid_position(i, width);
                let dy = ny as i32 - bmy as i32;
                let dx = nx as i32 - bmx as i32;
                let dist_sq = (dy * dy + dx * dx) as f32;
                if dist_sq > sigma_sq {
                    return;
                }

                let eta = alpha * (-dist_sq / two_sigma_sq).exp();
                for d in 0..dim {
                    code[d] += eta * (sample[d] - code[d]);
                }
            });
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
    /// wird daher parallel mit Rayon ausgeführt; die Reihenfolge bleibt
    /// erhalten.
    pub fn quantize_all(&self, blocks: &[Vec<f32>]) -> Vec<u8> {
        blocks.par_iter().map(|b| self.quantize_block(b)).collect()
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
