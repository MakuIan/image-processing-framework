use crate::frame::Frame;
use image::{GrayImage, Luma};
use image::{Rgb, RgbImage};
use imageproc::filter::gaussian_blur_f32;

pub fn calculate_psnr(original: &Frame, processed: &Frame) -> f64 {
    if original.width() != processed.width() || original.height() != processed.height() {
        return 0.0;
    }

    let width = original.width();
    let height = original.height();
    let num_pixels = (width as u32 * height as u32) as f64;
    let total_channels = num_pixels * 3.0;

    let mut sum_squared_diff: f64 = 0.0;

    for y in 0..height {
        for x in 0..width {
            let orig_pixel = original.get_pixel(x, y);
            let proc_pixel = processed.get_pixel(x, y);

            sum_squared_diff += (orig_pixel[0] as f64 - proc_pixel[0] as f64).powi(2);
            sum_squared_diff += (orig_pixel[1] as f64 - proc_pixel[1] as f64).powi(2);
            sum_squared_diff += (orig_pixel[2] as f64 - proc_pixel[2] as f64).powi(2);
        }
    }

    let mse = sum_squared_diff / total_channels;

    if mse == 0.0 {
        return f64::INFINITY;
    }

    10.0 * (f64::from(255.0).powi(2) / mse).log10()
}

pub fn calculate_ssim(original: &Frame, processed: &Frame) -> f64 {
    if original.width() != processed.width() || original.height() != processed.height() {
        return 0.0;
    }

    // RgbImage in Graustufen konvertieren (über get_pixel aus dem Frame)
    let gray1 = rgb_to_gray(original);
    let gray2 = rgb_to_gray(processed);

    // Gauß-Glättung anwenden
    let sigma = 1.5f32;
    let blurred1 = gaussian_blur_f32(&gray1, sigma);
    let blurred2 = gaussian_blur_f32(&gray2, sigma);

    let width = original.width() as usize;
    let height = original.height() as usize;

    // Mittelwerte berechnen
    let mut sum1: f64 = 0.0;
    let mut sum2: f64 = 0.0;
    for y in 0..height {
        for x in 0..width {
            let p1 = blurred1.get_pixel(x as u32, y as u32);
            let p2 = blurred2.get_pixel(x as u32, y as u32);
            sum1 += (*p1)[0] as f64;
            sum2 += (*p2)[0] as f64;
        }
    }
    let mu1 = sum1 / (width as f64 * height as f64);
    let mu2 = sum2 / (width as f64 * height as f64);

    // Varianz und Kovarianz berechnen
    let mut var_sum: f64 = 0.0;
    let mut cov_sum: f64 = 0.0;
    for y in 0..height {
        for x in 0..width {
            let p1 = blurred1.get_pixel(x as u32, y as u32);
            let p2 = blurred2.get_pixel(x as u32, y as u32);

            let d1 = (*p1)[0] as f64 - mu1;
            let d2 = (*p2)[0] as f64 - mu2;

            var_sum += d1 * d1;
            cov_sum += d1 * d2;
        }
    }

    let sigma12_sq = if width * height > 0 {
        var_sum / (width as f64 * height as f64)
    } else {
        0.0
    };
    let sigma12_cov = if width * height > 0 {
        cov_sum / (width as f64 * height as f64)
    } else {
        0.0
    };

    // SSIM berechnen
    const C1: f64 = (0.01 * 255.0f64) * (0.01 * 255.0f64);
    const C2: f64 = (0.03 * 255.0f64) * (0.03 * 255.0f64);

    let numerator = (2.0 * mu1 * mu2 + C1) * (2.0 * sigma12_cov + C2);
    let denominator = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma12_sq + C2);

    if denominator == 0.0 {
        return 0.0;
    }

    numerator / denominator
}

fn rgb_to_gray(img: &Frame) -> GrayImage {
    let width = img.width() as usize;
    let height = img.height() as usize;
    let mut gray = GrayImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let rgb = img.get_pixel(x as u32, y as u32);
            // Luminance formula
            let luma: f32 = 0.299 * rgb[0] as f32 + 0.587 * rgb[1] as f32 + 0.114 * rgb[2] as f32;
            gray.put_pixel(x as u32, y as u32, Luma([((luma * 255.0f32) as u8)]));
        }
    }

    gray
}

pub fn generate_diff_map(original: &Frame, processed: &Frame) -> Frame {
    let width = original.width();
    let height = original.height();
    let mut result = RgbImage::new(width, height);

    let mut max_diff = 0f64;

    for y in 0..height {
        for x in 0..width {
            let orig_pixel = original.get_pixel(x, y);
            let proc_pixel = processed.get_pixel(x, y);

            let diff_r = (orig_pixel[0] as f64 - proc_pixel[0] as f64).abs();
            let diff_g = (orig_pixel[1] as f64 - proc_pixel[1] as f64).abs();
            let diff_b = (orig_pixel[2] as f64 - proc_pixel[2] as f64).abs();
            let avg_diff = (diff_r + diff_g + diff_b) / 3.0;

            result.put_pixel(x, y, Rgb([avg_diff as u8, avg_diff as u8, avg_diff as u8]));

            if avg_diff > max_diff {
                max_diff = avg_diff;
            }
        }
    }

    let mut diff_map = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let orig_pixel = original.get_pixel(x, y);
            let proc_pixel = processed.get_pixel(x, y);

            let diff_r = (orig_pixel[0] as f64 - proc_pixel[0] as f64).abs();
            let diff_g = (orig_pixel[1] as f64 - proc_pixel[1] as f64).abs();
            let diff_b = (orig_pixel[2] as f64 - proc_pixel[2] as f64).abs();
            let avg_diff = (diff_r + diff_g + diff_b) / 3.0;

            let normalized = if max_diff > 0.0 {
                (avg_diff / max_diff * 255.0f64) as u8
            } else {
                0u8
            };

            let jet_color = jet_pseudocolor(normalized);
            diff_map.put_pixel(x, y, jet_color);
        }
    }

    Frame::new(diff_map, 0, 0.0)
}

fn jet_pseudocolor(value: u8) -> Rgb<u8> {
    let v = value as f64;
    if v < 51.0f64 {
        Rgb([
            (v / 51.0f64 * 255.0f64) as u8,
            (v / 255.0f64 * 51.0f64) as u8,
            ((255.0f64 - v) / 255.0f64 * 255.0f64) as u8,
        ])
    } else if v < 102.0f64 {
        Rgb([
            255u8,
            ((v - 51.0f64) / 51.0f64 * 255.0f64) as u8,
            (((255.0f64 - (v - 51.0f64)) / 51.0f64 * 51.0f64) as u8),
        ])
    } else if v < 153.0f64 {
        Rgb([
            (((255.0f64 - (v - 102.0f64)) / 51.0f64 * 255.0f64) as u8),
            255u8,
            ((v - 102.0f64) / 51.0f64 * 255.0f64) as u8,
        ])
    } else if v < 204.0f64 {
        Rgb([
            ((v - 153.0f64) / 51.0f64 * 255.0f64) as u8,
            (((255.0f64 - (v - 153.0f64)) / 51.0f64 * 51.0f64) as u8),
            255u8,
        ])
    } else {
        Rgb([
            0u8,
            ((v - 204.0f64) / 51.0f64 * 255.0f64) as u8,
            ((v - 204.0f64) / 102.0f64 * 255.0f64) as u8,
        ])
    }
}
