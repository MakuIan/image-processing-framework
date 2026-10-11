use crate::frame::Frame;
use image::RgbImage;
use rayon::prelude::*;

/// Konvertiert einen Frame (RGB8) in ein gepacktes RGBA-Format (u32),
/// welches direkt von Compute-Shadern verarbeitet werden kann.
pub fn frame_to_rgba_u32(frame: &Frame) -> (u32, u32, Vec<u32>) {
    let width = frame.width();
    let height = frame.height();
    let raw = frame.data().as_raw();
    let total_pixels = (width * height) as usize;

    let mut packed = vec![0u32; total_pixels];
    packed.par_iter_mut().enumerate().for_each(|(i, out)| {
        let base = i * 3;
        let r = raw[base] as u32;
        let g = raw[base + 1] as u32;
        let b = raw[base + 2] as u32;
        *out = r | (g << 8) | (b << 16) | 0xFF000000;
    });

    (width, height, packed)
}

/// Konvertiert ein von der GPU zurueckgegebenes u32-RGBA-Array
/// wieder in einen Frame (RGB8).
pub fn rgba_u32_to_frame(frame: &Frame, width: u32, height: u32, packed: &[u32]) -> Frame {
    let total_pixels = (width * height) as usize;
    let mut rgb_raw = vec![0u8; total_pixels * 3];

    rgb_raw.par_chunks_mut(3).enumerate().for_each(|(i, chunk)| {
        let pixel = packed[i];
        chunk[0] = (pixel & 0xFF) as u8;
        chunk[1] = ((pixel >> 8) & 0xFF) as u8;
        chunk[2] = ((pixel >> 16) & 0xFF) as u8;
    });

    let img = RgbImage::from_raw(width, height, rgb_raw).unwrap();
    frame.clone_with_data(img)
}
