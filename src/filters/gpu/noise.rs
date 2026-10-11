use crate::base::Filter;
use crate::filters::config::{DEFAULT_NOISE_MEAN, DEFAULT_NOISE_SIGMA};
use crate::frame::Frame;
use crate::gpu::{frame_to_rgba_u32, get_gpu_context, rgba_u32_to_frame, GpuComputeHelper};
use bytemuck::{Pod, Zeroable};

const NOISE_WGSL: &str = r#"
struct NoiseParams {
    width: u32,
    height: u32,
    mean: f32,
    sigma: f32,
    seed: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: NoiseParams;
@group(0) @binding(1) var<storage, read> input_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<u32>;

fn hash(x: u32) -> u32 {
    var state = x * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(state: ptr<function, u32>) -> f32 {
    *state = hash(*state);
    return f32(*state) / 4294967296.0;
}

fn box_muller(state: ptr<function, u32>, mean: f32, sigma: f32) -> f32 {
    let u1 = max(rand_f32(state), 1e-7);
    let u2 = rand_f32(state);
    let z0 = sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
    return mean + z0 * sigma;
}

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
    let x = global_id.x;
    let y = global_id.y;
    let w = params.width;
    let h = params.height;

    if (x >= w || y >= h) {
        return;
    }

    let idx = y * w + x;
    var state = idx + params.seed * 1999999973u;

    let orig = unpack_rgb(input_data[idx]);
    let noise_r = box_muller(&state, params.mean, params.sigma);
    let noise_g = box_muller(&state, params.mean, params.sigma);
    let noise_b = box_muller(&state, params.mean, params.sigma);

    let noisy = orig + vec3<f32>(noise_r, noise_g, noise_b);
    output_data[idx] = pack_rgb(noisy);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NoiseUniform {
    width: u32,
    height: u32,
    mean: f32,
    sigma: f32,
    seed: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct GpuGaussianNoiseFilter {
    pub mean: f64,
    pub sigma: f64,
    pub name: String,
    helper: Option<GpuComputeHelper>,
}

impl GpuGaussianNoiseFilter {
    pub fn new(mean: f64, sigma: f64) -> Self {
        let helper = get_gpu_context().map(|ctx| {
            GpuComputeHelper::new(ctx, NOISE_WGSL, "GpuNoise")
        });

        Self {
            mean,
            sigma,
            name: format!("GpuGaussianNoiseFilter(mean={}, sigma={})", mean, sigma),
            helper,
        }
    }
}

impl Default for GpuGaussianNoiseFilter {
    fn default() -> Self {
        Self::new(DEFAULT_NOISE_MEAN, DEFAULT_NOISE_SIGMA)
    }
}

impl Filter for GpuGaussianNoiseFilter {
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

        let seed = (frame.index() as u32).wrapping_add(42);
        let uniform = NoiseUniform {
            width,
            height,
            mean: self.mean as f32,
            sigma: self.sigma as f32,
            seed,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let result = helper.execute(bytemuck::bytes_of(&uniform), &packed, width, height);
        rgba_u32_to_frame(frame, width, height, &result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_gpu_noise_filter() {
        let img = RgbImage::new(16, 16);
        let frame = Frame::new(img, 0, 0.0);
        let filter = GpuGaussianNoiseFilter::new(0.0, 15.0);
        let filtered = filter.apply(&frame);
        assert_eq!(filtered.width(), 16);
        assert_eq!(filtered.height(), 16);
    }
}
