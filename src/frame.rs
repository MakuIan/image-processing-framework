use image::RgbImage;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Frame {
    data: RgbImage,
    index: usize,
    timestamp: f64,
    metadata: HashMap<String, String>,
}

impl Frame {
    pub fn new(data: RgbImage, index: usize, timestamp: f64) -> Self {
        Self {
            data,
            index,
            timestamp,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn width(&self) -> u32 {
        self.data.width()
    }

    pub fn height(&self) -> u32 {
        self.data.height()
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> &image::Rgb<u8> {
        self.data.get_pixel(x, y)
    }

    pub fn data(&self) -> &RgbImage {
        &self.data
    }
}
