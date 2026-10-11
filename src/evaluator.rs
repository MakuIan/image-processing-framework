use crate::frame::Frame;
use image::GrayImage;
use image::{Rgb, RgbImage};
use imageproc::filter::gaussian_blur_f32;
use rayon::prelude::*;

pub fn calculate_psnr(original: &Frame, processed: &Frame) -> f64 {
    if original.width() != processed.width() || original.height() != processed.height() {
        return 0.0;
    }

    let width = original.width();
    let height = original.height();
    if width == 0 || height == 0 {
        return 0.0;
    }

    let num_pixels = (width * height) as f64;
    let total_channels = num_pixels * 3.0;

    let orig_raw = original.data().as_raw();
    let proc_raw = processed.data().as_raw();

    let sum_squared_diff: f64 = orig_raw
        .par_iter()
        .zip(proc_raw.par_iter())
        .map(|(&o, &p)| {
            let diff = o as f64 - p as f64;
            diff * diff
        })
        .sum();

    let mse = sum_squared_diff / total_channels;

    if mse == 0.0 {
        return f64::INFINITY;
    }

    10.0 * (255.0f64.powi(2) / mse).log10()
}

pub fn calculate_ssim(original: &Frame, processed: &Frame) -> f64 {
    if original.width() != processed.width() || original.height() != processed.height() {
        return 0.0;
    }

    let width = original.width();
    let height = original.height();
    if width == 0 || height == 0 {
        return 0.0;
    }

    // Convert to grayscale
    let gray1 = rgb_to_gray(original);
    let gray2 = rgb_to_gray(processed);

    // Apply Gaussian blur
    let sigma = 1.5f32;
    let blurred1 = gaussian_blur_f32(&gray1, sigma);
    let blurred2 = gaussian_blur_f32(&gray2, sigma);

    let total_pixels = (width * height) as f64;

    let raw1 = blurred1.as_raw();
    let raw2 = blurred2.as_raw();

    // Calculate means using parallel sum
    let sum1: f64 = raw1.par_iter().map(|&p| p as f64).sum();
    let sum2: f64 = raw2.par_iter().map(|&p| p as f64).sum();
    let mu1 = sum1 / total_pixels;
    let mu2 = sum2 / total_pixels;

    // Calculate variances and covariance in parallel
    let (var1_sum, var2_sum, cov_sum): (f64, f64, f64) = raw1
        .par_iter()
        .zip(raw2.par_iter())
        .map(|(&p1, &p2)| {
            let d1 = p1 as f64 - mu1;
            let d2 = p2 as f64 - mu2;
            (d1 * d1, d2 * d2, d1 * d2)
        })
        .reduce(
            || (0.0, 0.0, 0.0),
            |acc, item| (acc.0 + item.0, acc.1 + item.1, acc.2 + item.2),
        );

    let sigma1_sq = var1_sum / total_pixels;
    let sigma2_sq = var2_sum / total_pixels;
    let sigma12 = cov_sum / total_pixels;

    // Standard SSIM stability constants
    const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
    const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0);

    let numerator = (2.0 * mu1 * mu2 + C1) * (2.0 * sigma12 + C2);
    let denominator = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2);

    if denominator == 0.0 {
        return 0.0;
    }

    numerator / denominator
}

fn rgb_to_gray(img: &Frame) -> GrayImage {
    let width = img.width();
    let height = img.height();
    if width == 0 || height == 0 {
        return GrayImage::new(width, height);
    }

    let orig_raw = img.data().as_raw();
    let stride = (width * 3) as usize;
    let mut gray_raw = vec![0u8; (width * height) as usize];

    gray_raw
        .par_chunks_mut(width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let in_row = &orig_raw[y * stride..(y + 1) * stride];
            for x in 0..width as usize {
                let r = in_row[x * 3] as f32;
                let g = in_row[x * 3 + 1] as f32;
                let b = in_row[x * 3 + 2] as f32;
                let luma: f32 = 0.299 * r + 0.587 * g + 0.114 * b;
                row[x] = luma.round().clamp(0.0, 255.0) as u8;
            }
        });

    GrayImage::from_raw(width, height, gray_raw).unwrap()
}

pub fn generate_diff_map(original: &Frame, processed: &Frame) -> Frame {
    let width = original.width();
    let height = original.height();
    if width == 0 || height == 0 {
        return Frame::default();
    }

    let orig_raw = original.data().as_raw();
    let proc_raw = processed.data().as_raw();
    let total_pixels = (width * height) as usize;

    // Parallel pass 1: compute channel differences
    let diff_values: Vec<f64> = (0..total_pixels)
        .into_par_iter()
        .map(|idx| {
            let base = idx * 3;
            let diff_r = (orig_raw[base] as f64 - proc_raw[base] as f64).abs();
            let diff_g = (orig_raw[base + 1] as f64 - proc_raw[base + 1] as f64).abs();
            let diff_b = (orig_raw[base + 2] as f64 - proc_raw[base + 2] as f64).abs();
            (diff_r + diff_g + diff_b) / 3.0
        })
        .collect();

    let max_diff = diff_values.par_iter().copied().reduce(|| 0.0f64, f64::max);

    // Parallel pass 2: normalize differences into jet pseudocolor map
    let mut diff_raw = vec![0u8; (width * height * 3) as usize];

    diff_raw
        .par_chunks_mut(3)
        .zip(diff_values.par_iter())
        .for_each(|(pixel, &avg_diff)| {
            let normalized = if max_diff > 0.0 {
                (avg_diff / max_diff * 255.0).clamp(0.0, 255.0) as u8
            } else {
                0u8
            };

            let jet_color = jet_pseudocolor(normalized);
            pixel[0] = jet_color[0];
            pixel[1] = jet_color[1];
            pixel[2] = jet_color[2];
        });

    let diff_map = RgbImage::from_raw(width, height, diff_raw).unwrap();
    processed.clone_with_data(diff_map)
}

fn jet_pseudocolor(value: u8) -> Rgb<u8> {
    let v = value as f64;
    if v < 51.0 {
        Rgb([
            (v / 51.0 * 255.0) as u8,
            (v / 255.0 * 51.0) as u8,
            ((255.0 - v) / 255.0 * 255.0) as u8,
        ])
    } else if v < 102.0 {
        Rgb([
            255u8,
            ((v - 51.0) / 51.0 * 255.0) as u8,
            (((255.0 - (v - 51.0)) / 51.0 * 51.0) as u8),
        ])
    } else if v < 153.0 {
        Rgb([
            (((255.0 - (v - 102.0)) / 51.0 * 255.0) as u8),
            255u8,
            ((v - 102.0) / 51.0 * 255.0) as u8,
        ])
    } else if v < 204.0 {
        Rgb([
            ((v - 153.0) / 51.0 * 255.0) as u8,
            (((255.0 - (v - 153.0)) / 51.0 * 51.0) as u8),
            255u8,
        ])
    } else {
        Rgb([
            0u8,
            ((v - 204.0) / 51.0 * 255.0) as u8,
            ((v - 204.0) / 102.0 * 255.0) as u8,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psnr_identical() {
        let img = RgbImage::new(16, 16);
        let f1 = Frame::new(img.clone(), 0, 0.0);
        let f2 = Frame::new(img, 0, 0.0);
        assert_eq!(calculate_psnr(&f1, &f2), f64::INFINITY);
    }

    #[test]
    fn test_ssim_identical() {
        let mut img = RgbImage::new(16, 16);
        for p in img.pixels_mut() {
            *p = Rgb([120, 130, 140]);
        }
        let f1 = Frame::new(img.clone(), 0, 0.0);
        let f2 = Frame::new(img, 0, 0.0);
        let ssim = calculate_ssim(&f1, &f2);
        assert!((ssim - 1.0).abs() < 1e-4);
    }
}
