use crate::base::Filter;
use crate::filters::config::DEFAULT_BLUR_SIGMA;
use crate::frame::Frame;
use image::RgbImage;

/// Fuehrt den Gauss-Weichzeichner auf der CPU aus.
pub fn apply_blur_cpu(img: &RgbImage, sigma: f32) -> RgbImage {
    imageproc::filter::gaussian_blur_f32(img, sigma)
}

#[derive(Clone, Debug)]
pub struct BlurFilter {
    pub sigma: f32,
}

impl BlurFilter {
    pub fn new(sigma: f32) -> Self {
        Self { sigma }
    }
}

impl Default for BlurFilter {
    fn default() -> Self {
        Self::new(DEFAULT_BLUR_SIGMA)
    }
}

impl Filter for BlurFilter {
    fn name(&self) -> &str {
        "BlurFilter"
    }

    fn apply(&self, frame: &Frame) -> Frame {
        let blurred = apply_blur_cpu(frame.data(), self.sigma);
        frame.clone_with_data(blurred)
    }
}
