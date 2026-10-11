pub mod bilateral;
pub mod blur;
pub mod noise;

pub use bilateral::GpuBilateralFilter;
pub use blur::GpuBlurFilter;
pub use noise::GpuGaussianNoiseFilter;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::Filter;
    use crate::frame::Frame;
    use image::RgbImage;

    #[test]
    #[ignore]
    fn benchmark_gpu_pipeline_4k() {
        let width = 4096;
        let height = 1152;
        let raw = vec![128u8; (width * height * 3) as usize];
        let img = RgbImage::from_raw(width, height, raw).unwrap();
        let frame = Frame::new(img, 0, 0.0);

        let noise = GpuGaussianNoiseFilter::new(0.0, 15.0);
        let bilateral = GpuBilateralFilter::new(3, 50.0, 50.0);
        let blur = GpuBlurFilter::new(1.0);

        let _ = bilateral.apply(&frame);

        let start = std::time::Instant::now();
        let f1 = noise.apply(&frame);
        let dur_noise = start.elapsed();

        let start = std::time::Instant::now();
        let f2 = bilateral.apply(&f1);
        let dur_bilateral = start.elapsed();

        let start = std::time::Instant::now();
        let _f3 = blur.apply(&f2);
        let dur_blur = start.elapsed();

        println!("GPU Noise:     {:?}", dur_noise);
        println!("GPU Bilateral: {:?}", dur_bilateral);
        println!("GPU Blur:      {:?}", dur_blur);
        println!("Total GPU:     {:?}", dur_noise + dur_bilateral + dur_blur);
    }
}
