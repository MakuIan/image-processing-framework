use crate::base::Filter;
use crate::filters::config::{DEFAULT_NOISE_MEAN, DEFAULT_NOISE_SIGMA};
use crate::frame::Frame;
use image::RgbImage;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Fuehrt die Gauss-Rauschen-Berechnung parallel auf der CPU durch.
pub fn apply_gaussian_noise_cpu(img: &RgbImage, mean: f64, sigma: f64) -> RgbImage {
    let width = img.width();
    let height = img.height();

    if sigma <= 0.0 || width == 0 || height == 0 {
        return img.clone();
    }

    let normal = match Normal::new(mean, sigma) {
        Ok(dist) => dist,
        Err(_) => return img.clone(),
    };

    let orig_raw = img.as_raw();
    let stride = (width * 3) as usize;
    let mut noise_raw = vec![0u8; (width * height * 3) as usize];

    noise_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y_idx, row_out)| {
            let mut rng = thread_rng();
            let orig_row = &orig_raw[y_idx * stride..(y_idx + 1) * stride];

            for x in 0..width as usize {
                let base = x * 3;
                let noise_r = normal.sample(&mut rng);
                let noise_g = normal.sample(&mut rng);
                let noise_b = normal.sample(&mut rng);

                row_out[base] = (orig_row[base] as f64 + noise_r).clamp(0.0, 255.0) as u8;
                row_out[base + 1] =
                    (orig_row[base + 1] as f64 + noise_g).clamp(0.0, 255.0) as u8;
                row_out[base + 2] =
                    (orig_row[base + 2] as f64 + noise_b).clamp(0.0, 255.0) as u8;
            }
        });

    RgbImage::from_raw(width, height, noise_raw).unwrap_or_else(|| img.clone())
}

#[derive(Clone, Debug)]
pub struct GaussianNoiseFilter {
    pub mean: f64,
    pub sigma: f64,
}

impl GaussianNoiseFilter {
    pub fn new(mean: f64, sigma: f64) -> Self {
        Self { mean, sigma }
    }
}

impl Default for GaussianNoiseFilter {
    fn default() -> Self {
        Self::new(DEFAULT_NOISE_MEAN, DEFAULT_NOISE_SIGMA)
    }
}

impl Filter for GaussianNoiseFilter {
    fn name(&self) -> &str {
        "GaussianNoiseFilter"
    }

    fn apply(&self, frame: &Frame) -> Frame {
        let result = apply_gaussian_noise_cpu(frame.data(), self.mean, self.sigma);
        frame.clone_with_data(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_noise_parallel() {
        let img = RgbImage::new(8, 8);
        let frame = Frame::new(img, 0, 0.0);
        let filter = GaussianNoiseFilter::new(0.0, 10.0);
        let filtered = filter.apply(&frame);
        assert_eq!(filtered.width(), 8);
        assert_eq!(filtered.height(), 8);
    }
}
