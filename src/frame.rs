use image::RgbImage;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Frame {
    data: RgbImage,
    index: usize,
    timestamp: f64,
    metadata: HashMap<String, String>,
}

impl Default for Frame {
    fn default() -> Self {
        Self {
            data: RgbImage::new(0, 0),
            index: 0,
            timestamp: 0.0,
            metadata: HashMap::new(),
        }
    }
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

    /// Clones the frame with the same metadata and index but new data
    pub fn clone_with_data(&self, data: RgbImage) -> Self {
        Self {
            data,
            index: self.index,
            timestamp: self.timestamp,
            metadata: self.metadata.clone(),
        }
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
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

    pub fn into_data(self) -> RgbImage {
        self.data
    }

    /// Returns the frame index
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the timestamp of the frame in seconds
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }

    /// Returns a reference to the metadata HashMap
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
}
