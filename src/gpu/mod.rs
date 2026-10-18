pub mod conversion;
pub mod helper;
pub mod som;

use std::sync::{Arc, OnceLock};

pub use conversion::{frame_to_rgba_u32, rgba_u32_to_frame};
pub use helper::GpuComputeHelper;
pub use som::SomGpuQuantizer;

pub static GPU_CONTEXT: OnceLock<Option<Arc<GpuContext>>> = OnceLock::new();

pub fn get_gpu_context() -> Option<Arc<GpuContext>> {
    GPU_CONTEXT
        .get_or_init(|| GpuContext::new().map(Arc::new))
        .clone()
}

pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let adapter_info = adapter.get_info();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ImageProcessingDevice"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .ok()?;

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        match GpuContext::new() {
            Some(ctx) => {
                println!("GPU Adapter Name: {}", ctx.adapter_info.name);
                println!("GPU Backend: {:?}", ctx.adapter_info.backend);
                println!("GPU Driver: {}", ctx.adapter_info.driver);
            }
            _ => {
                println!("No GPU adapter found (headless or not supported)");
            }
        }
    }
}
