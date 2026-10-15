use crate::base::Filter;
use crate::filters::config::{
    DEFAULT_BILATERAL_COLOR, DEFAULT_BILATERAL_RADIUS, DEFAULT_BILATERAL_SPACE,
};
use crate::frame::Frame;
use image::RgbImage;
use rayon::prelude::*;

/// Berechnet den Gauss-Gewichtungswert.
#[inline]
pub fn gaussian_weight(dist_sq: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return 0.0;
    }
    let exponent = -dist_sq / (2.0 * sigma * sigma);
    if exponent < -700.0 {
        0.0
    } else {
        exponent.exp()
    }
}

/// Fuehrt die bilaterale Filterung parallel auf der CPU aus.
pub fn apply_bilateral_cpu(
    img: &RgbImage,
    radius: i32,
    spatial_kernel: &[f64],
    color_lut: &[f64],
) -> RgbImage {
    let width = img.width();
    let height = img.height();

    if width == 0 || height == 0 || radius < 0 {
        return img.clone();
    }

    let dim = (2 * radius + 1) as usize;
    let input_raw = img.as_raw();
    let stride = (width * 3) as usize;
    let mut output_raw = vec![0u8; (width * height * 3) as usize];

    output_raw
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y_idx, row_out)| {
            let y = y_idx as i32;
            let start_y = (y - radius).max(0) as usize;
            let end_y = (y + radius).min(height as i32 - 1) as usize;

            for x in 0..width as usize {
                let center_idx = (y_idx * stride) + (x * 3);
                let center_r = input_raw[center_idx] as i32;
                let center_g = input_raw[center_idx + 1] as i32;
                let center_b = input_raw[center_idx + 2] as i32;

                let mut sum_r: f64 = 0.0;
                let mut sum_g: f64 = 0.0;
                let mut sum_b: f64 = 0.0;
                let mut sum_weights: f64 = 0.0;

                let start_x = (x as i32 - radius).max(0) as usize;
                let end_x = (x as i32 + radius).min(width as i32 - 1) as usize;

                for ny in start_y..=end_y {
                    let row_offset = ny * stride;
                    let k_y = (ny as i32 - y + radius) as usize;

                    for nx in start_x..=end_x {
                        let k_x = (nx as i32 - x as i32 + radius) as usize;
                        let spatial_weight = spatial_kernel[k_y * dim + k_x];

                        let n_idx = row_offset + (nx * 3);
                        let nr = input_raw[n_idx] as i32;
                        let ng = input_raw[n_idx + 1] as i32;
                        let nb = input_raw[n_idx + 2] as i32;

                        let diff_r = nr - center_r;
                        let diff_g = ng - center_g;
                        let diff_b = nb - center_b;
                        let color_dist_sq =
                            (diff_r * diff_r + diff_g * diff_g + diff_b * diff_b) as usize;

                        let color_weight = color_lut[color_dist_sq];
                        let weight = spatial_weight * color_weight;

                        sum_r += nr as f64 * weight;
                        sum_g += ng as f64 * weight;
                        sum_b += nb as f64 * weight;
                        sum_weights += weight;
                    }
                }

                let out_idx = x * 3;
                if sum_weights > 0.0 {
                    row_out[out_idx] = (sum_r / sum_weights).round().clamp(0.0, 255.0) as u8;
                    row_out[out_idx + 1] = (sum_g / sum_weights).round().clamp(0.0, 255.0) as u8;
                    row_out[out_idx + 2] = (sum_b / sum_weights).round().clamp(0.0, 255.0) as u8;
                } else {
                    row_out[out_idx] = center_r as u8;
                    row_out[out_idx + 1] = center_g as u8;
                    row_out[out_idx + 2] = center_b as u8;
                }
            }
        });

    RgbImage::from_raw(width, height, output_raw).unwrap_or_else(|| img.clone())
}

#[derive(Clone, Debug)]
pub struct BilateralFilter {
    pub radius: i32,
    pub sigma_color: f64,
    pub sigma_space: f64,
    pub name: String,
    spatial_kernel: Vec<f64>,
    color_lut: Vec<f64>,
}

impl BilateralFilter {
    pub fn new(radius: i32, sigma_color: f64, sigma_space: f64) -> Self {
        let r = radius.max(0);
        let dim = (2 * r + 1) as usize;
        let mut spatial_kernel = vec![0.0f64; dim * dim];

        for dy in -r..=r {
            for dx in -r..=r {
                let dist_sq = (dx * dx + dy * dy) as f64;
                let idx = (dy + r) as usize * dim + (dx + r) as usize;
                spatial_kernel[idx] = gaussian_weight(dist_sq, sigma_space);
            }
        }

        // Maximaler Farbabstand im RGB-Raum: 3 * 255^2 = 195.075
        let max_color_dist = 3 * 255 * 255;
        let color_lut = (0..=max_color_dist)
            .map(|d| gaussian_weight(d as f64, sigma_color))
            .collect();

        Self {
            radius,
            sigma_color,
            sigma_space,
            name: format!(
                "BilateralFilter(radius={}, sigma_color={}, sigma_space={})",
                radius, sigma_color, sigma_space
            ),
            spatial_kernel,
            color_lut,
        }
    }
}

impl Default for BilateralFilter {
    fn default() -> Self {
        Self::new(
            DEFAULT_BILATERAL_RADIUS,
            DEFAULT_BILATERAL_COLOR,
            DEFAULT_BILATERAL_SPACE,
        )
    }
}

impl Filter for BilateralFilter {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, frame: &Frame) -> Frame {
        let result = apply_bilateral_cpu(
            frame.data(),
            self.radius,
            &self.spatial_kernel,
            &self.color_lut,
        );
        frame.clone_with_data(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn test_bilateral_filter_parallel() {
        let mut img = RgbImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                img.put_pixel(x, y, Rgb([x as u8 * 20, y as u8 * 20, 128]));
            }
        }
        let frame = Frame::new(img, 0, 0.0);
        let filter = BilateralFilter::new(2, 25.0, 10.0);
        let filtered = filter.apply(&frame);
        assert_eq!(filtered.width(), 10);
        assert_eq!(filtered.height(), 10);
    }
}
