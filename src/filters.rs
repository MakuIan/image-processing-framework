use crate::base::Filter;
use crate::frame::Frame;
use image::{Rgb, RgbImage};

pub struct GaussianNoiseFilter {
    pub mean: f64,
    pub sigma: f64,
}

impl GaussianNoiseFilter {
    pub fn new(mean: f64, sigma: f64) -> Self {
        Self { mean, sigma }
    }
}

impl Filter for GaussianNoiseFilter {
    fn name(&self) -> &str {
        "GaussianNoiseFilter"
    }

    fn apply(&self, frame: &Frame) -> Frame {
        let mut noise_image = RgbImage::new(frame.width(), frame.height());

        let sigma = self.sigma as f64;
        let mean = self.mean as f64;

        for y in 0..frame.height() {
            for x in 0..frame.width() {
                let orig_pixel = frame.get_pixel(x, y);

                let new_r = (orig_pixel[0] as f64 + sigma * mean).clamp(0.0, 255.0);
                let new_g = (orig_pixel[1] as f64 + sigma * mean).clamp(0.0, 255.0);
                let new_b = (orig_pixel[2] as f64 + sigma * mean).clamp(0.0, 255.0);

                noise_image.put_pixel(x, y, Rgb([new_r as u8, new_g as u8, new_b as u8]));
            }
        }

        Frame::new(noise_image, 0, 0.0)
    }
}

pub struct BlurFilter {
    pub sigma: f32,
}

impl BlurFilter {
    pub fn new(sigma: f32) -> Self {
        Self { sigma }
    }
}

impl Filter for BlurFilter {
    fn name(&self) -> &str {
        "BlurFilter"
    }

    fn apply(&self, frame: &Frame) -> Frame {
        let blurred = imageproc::filter::gaussian_blur_f32(frame.data(), self.sigma);
        Frame::new(blurred, 0, 0.0)
    }
}

pub struct BilateralFilter {
    pub radius: i32,
    pub sigma_color: f64,
    pub sigma_space: f64,
    pub name: String,
}

impl BilateralFilter {
    pub fn new(radius: i32, sigma_color: f64, sigma_space: f64) -> Self {
        Self {
            radius,
            sigma_color,
            sigma_space,
            name: format!(
                "BilateralFilter(radius={}, sigma_color={}, sigma_space={})",
                radius, sigma_color, sigma_space
            ),
        }
    }

    fn exp(&self, x: f64) -> f64 {
        if x < -700.0 {
            0.0
        } else {
            (-x).exp()
        }
    }
}

impl Filter for BilateralFilter {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, frame: &Frame) -> Frame {
        let width = frame.width();
        let height = frame.height();
        let radius = self.radius as u32;
        let sigma_color = self.sigma_color;
        let sigma_space = self.sigma_space;

        let mut result = RgbImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let mut sum_r: f64 = 0.0;
                let mut sum_g: f64 = 0.0;
                let mut sum_b: f64 = 0.0;
                let mut sum_weights: f64 = 0.0;

                let start_x = if x > radius as u32 { x - radius } else { 0 };
                let end_x = (x + radius).min((width - 1) as u32);
                let start_y = if y > radius as u32 { y - radius } else { 0 };
                let end_y = (y + radius).min((height - 1) as u32);

                for ny in start_y..=end_y {
                    for nx in start_x..=end_x {
                        let dx = (x as i32 - nx as i32) as f64;
                        let dy = (y as i32 - ny as i32) as f64;
                        let spatial_dist_sq = dx * dx + dy * dy;

                        let neighbor_pixel = frame.get_pixel(nx, ny);
                        let color_dist_sq = (neighbor_pixel[0] as f64
                            - frame.get_pixel(x, y)[0] as f64)
                            .powi(2)
                            + (neighbor_pixel[1] as f64 - frame.get_pixel(x, y)[1] as f64).powi(2)
                            + (neighbor_pixel[2] as f64 - frame.get_pixel(x, y)[2] as f64).powi(2);

                        let spatial_weight =
                            self.exp(-spatial_dist_sq / (2.0 * sigma_space * sigma_space));
                        let color_weight =
                            self.exp(-color_dist_sq / (2.0 * sigma_color * sigma_color));
                        let weight = spatial_weight * color_weight;

                        sum_r += neighbor_pixel[0] as f64 * weight;
                        sum_g += neighbor_pixel[1] as f64 * weight;
                        sum_b += neighbor_pixel[2] as f64 * weight;
                        sum_weights += weight;
                    }
                }

                let final_r = if sum_weights > 0.0 {
                    sum_r / sum_weights
                } else {
                    frame.get_pixel(x, y)[0] as f64
                };
                let final_g = if sum_weights > 0.0 {
                    sum_g / sum_weights
                } else {
                    frame.get_pixel(x, y)[1] as f64
                };
                let final_b = if sum_weights > 0.0 {
                    sum_b / sum_weights
                } else {
                    frame.get_pixel(x, y)[2] as f64
                };

                result.put_pixel(x, y, Rgb([final_r as u8, final_g as u8, final_b as u8]));
            }
        }

        Frame::new(result, 0, 0.0)
    }
}
