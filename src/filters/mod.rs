pub mod config;
pub mod cpu;
pub mod gpu;

pub use config::{
    FilterConfig, DEFAULT_BILATERAL_COLOR, DEFAULT_BILATERAL_RADIUS, DEFAULT_BILATERAL_SPACE,
    DEFAULT_BLUR_SIGMA, DEFAULT_NOISE_MEAN, DEFAULT_NOISE_SIGMA,
};

pub use cpu::{
    apply_bilateral_cpu, apply_blur_cpu, apply_gaussian_noise_cpu,
    BilateralFilter, BlurFilter, GaussianNoiseFilter,
};

pub use gpu::{
    GpuBilateralFilter, GpuBlurFilter, GpuGaussianNoiseFilter,
};
