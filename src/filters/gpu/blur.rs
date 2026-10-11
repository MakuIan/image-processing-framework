use crate::base::Filter;
use crate::filters::config::DEFAULT_BLUR_SIGMA;
use crate::frame::Frame;
use crate::gpu::{frame_to_rgba_u32, get_gpu_context, rgba_u32_to_frame, GpuComputeHelper};
use bytemuck::{Pod, Zeroable};

const BLUR_WGSL: &str = r#"
struct BlurParams {
    width: u32,
    height: u32,
    radius: i32,
    sigma: f32,
};

@group(0) @binding(0) var<uniform> params: BlurParams;
@group(0) @binding(1) var<storage, read> input_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<u32>;

fn unpack_rgb(pixel: u32) -> vec3<f32> {
    return vec3<f32>(
        f32(pixel & 255u),
        f32((pixel >> 8u) & 255u),
        f32((pixel >> 16u) & 255u)
    );
}

fn pack_rgb(rgb: vec3<f32>) -> u32 {
    let r = u32(clamp(round(rgb.r), 0.0, 255.0));
    let g = u32(clamp(round(rgb.g), 0.0, 255.0));
    let b = u32(clamp(round(rgb.b), 0.0, 255.0));
    return r | (g << 8u) | (b << 16u) | 0xFF000000u;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let w = i32(params.width);
    let h = i32(params.height);

    if (x >= w || y >= h) {
        return;
    }

    let center_idx = u32(y * w + x);
    let r = params.radius;
    let two_sigma_sq = 2.0 * params.sigma * params.sigma;

    var sum_rgb = vec3<f32>(0.0, 0.0, 0.0);
    var sum_weights = 0.0;

    let start_y = max(y - r, 0);
    let end_y = min(y + r, h - 1);
    let start_x = max(x - r, 0);
    let end_x = min(x + r, w - 1);

    for (var ny = start_y; ny <= end_y; ny++) {
        let dy = f32(y - ny);
        let dy_sq = dy * dy;
        let row_offset = ny * w;

        for (var nx = start_x; nx <= end_x; nx++) {
            let dx = f32(x - nx);
            let dist_sq = dx * dx + dy_sq;
            let weight = exp(-dist_sq / two_sigma_sq);

            let n_idx = u32(row_offset + nx);
            let n_rgb = unpack_rgb(input_data[n_idx]);
            sum_rgb += n_rgb * weight;
            sum_weights += weight;
        }
    }

    if (sum_weights > 0.0) {
        output_data[center_idx] = pack_rgb(sum_rgb / sum_weights);
    } else {
        output_data[center_idx] = input_data[center_idx];
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurUniform {
    width: u32,
    height: u32,
    radius: i32,
    sigma: f32,
}

pub struct GpuBlurFilter {
    pub sigma: f32,
    pub radius: i32,
    pub name: String,
    helper: Option<GpuComputeHelper>,
}

impl GpuBlurFilter {
    pub fn new(sigma: f32) -> Self {
        let radius = (sigma * 3.0).ceil() as i32;
        let helper = get_gpu_context().map(|ctx| {
            GpuComputeHelper::new(ctx, BLUR_WGSL, "GpuBlur")
        });

        Self {
            sigma,
            radius,
            name: format!("GpuBlurFilter(sigma={}, radius={})", sigma, radius),
            helper,
        }
    }
}

impl Default for GpuBlurFilter {
    fn default() -> Self {
        Self::new(DEFAULT_BLUR_SIGMA)
    }
}

impl Filter for GpuBlurFilter {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, frame: &Frame) -> Frame {
        let Some(helper) = &self.helper else {
            return frame.clone();
        };

        let (width, height, packed) = frame_to_rgba_u32(frame);
        if width == 0 || height == 0 {
            return frame.clone();
        }

        let uniform = BlurUniform {
            width,
            height,
            radius: self.radius,
            sigma: self.sigma,
        };

        let result = helper.execute(bytemuck::bytes_of(&uniform), &packed, width, height);
        rgba_u32_to_frame(frame, width, height, &result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn test_gpu_blur_filter() {
        let mut img = RgbImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                img.put_pixel(x, y, Rgb([x as u8 * 10, y as u8 * 10, 128]));
            }
        }
        let frame = Frame::new(img, 0, 0.0);
        let filter = GpuBlurFilter::new(1.0);
        let filtered = filter.apply(&frame);
        assert_eq!(filtered.width(), 16);
        assert_eq!(filtered.height(), 16);
    }
}
