use crate::base::Filter;
use crate::filters::config::{
    DEFAULT_BILATERAL_COLOR, DEFAULT_BILATERAL_RADIUS, DEFAULT_BILATERAL_SPACE,
};
use crate::frame::Frame;
use crate::gpu::{frame_to_rgba_u32, get_gpu_context, rgba_u32_to_frame, GpuComputeHelper};
use bytemuck::{Pod, Zeroable};

const BILATERAL_WGSL: &str = r#"
struct Params {
    width: u32,
    height: u32,
    radius: i32,
    _pad: u32,
    sigma_color: f32,
    sigma_space: f32,
    _pad2: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
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
    let center_rgb = unpack_rgb(input_data[center_idx]);

    let r = params.radius;
    let two_sigma_space_sq = 2.0 * params.sigma_space * params.sigma_space;
    let two_sigma_color_sq = 2.0 * params.sigma_color * params.sigma_color;

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
            let spatial_dist_sq = dx * dx + dy_sq;
            let spatial_weight = exp(-spatial_dist_sq / two_sigma_space_sq);

            let n_idx = u32(row_offset + nx);
            let n_rgb = unpack_rgb(input_data[n_idx]);
            let diff = n_rgb - center_rgb;
            let color_dist_sq = dot(diff, diff);
            let color_weight = exp(-color_dist_sq / two_sigma_color_sq);

            let weight = spatial_weight * color_weight;
            sum_rgb += n_rgb * weight;
            sum_weights += weight;
        }
    }

    if (sum_weights > 0.0) {
        output_data[center_idx] = pack_rgb(sum_rgb / sum_weights);
    } else {
        output_data[center_idx] = pack_rgb(center_rgb);
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BilateralUniform {
    width: u32,
    height: u32,
    radius: i32,
    _pad: u32,
    sigma_color: f32,
    sigma_space: f32,
    _pad2: [f32; 2],
}

pub struct GpuBilateralFilter {
    pub radius: i32,
    pub sigma_color: f64,
    pub sigma_space: f64,
    pub name: String,
    helper: Option<GpuComputeHelper>,
}

impl GpuBilateralFilter {
    pub fn new(radius: i32, sigma_color: f64, sigma_space: f64) -> Self {
        let helper = get_gpu_context().map(|ctx| {
            GpuComputeHelper::new(ctx, BILATERAL_WGSL, "GpuBilateral")
        });

        Self {
            radius,
            sigma_color,
            sigma_space,
            name: format!(
                "GpuBilateralFilter(radius={}, sigma_color={}, sigma_space={})",
                radius, sigma_color, sigma_space
            ),
            helper,
        }
    }
}

impl Default for GpuBilateralFilter {
    fn default() -> Self {
        Self::new(
            DEFAULT_BILATERAL_RADIUS,
            DEFAULT_BILATERAL_COLOR,
            DEFAULT_BILATERAL_SPACE,
        )
    }
}

impl Filter for GpuBilateralFilter {
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

        let uniform = BilateralUniform {
            width,
            height,
            radius: self.radius,
            _pad: 0,
            sigma_color: self.sigma_color as f32,
            sigma_space: self.sigma_space as f32,
            _pad2: [0.0, 0.0],
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
    fn test_gpu_bilateral_filter() {
        let mut img = RgbImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                img.put_pixel(x, y, Rgb([x as u8 * 10, y as u8 * 10, 128]));
            }
        }
        let frame = Frame::new(img, 0, 0.0);
        let filter = GpuBilateralFilter::new(2, 25.0, 10.0);
        let filtered = filter.apply(&frame);
        assert_eq!(filtered.width(), 16);
        assert_eq!(filtered.height(), 16);
    }
}
