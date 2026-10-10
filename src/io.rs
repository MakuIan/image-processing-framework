use crate::base::{Sink, Source};
use crate::frame::Frame;
use std::path::PathBuf;

pub struct ImageSource {
    path: PathBuf,
    consumed: bool,
}

impl ImageSource {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            path: path.into(),
            consumed: false,
        }
    }
}

impl Source for ImageSource {
    fn next_frame(&mut self) -> Option<crate::frame::Frame> {
        if self.consumed {
            return None;
        }

        self.consumed = true;

        let img = image::open(&self.path).ok()?;

        let rgb = img.to_rgb8();

        Some(Frame::new(rgb, 0, 0.0))
    }

    fn release(&mut self) {
        // Nothing to release for ImageSource
    }
}

pub struct ImageSink {
    path: PathBuf,
}

impl ImageSink {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self { path: path.into() }
    }
}

impl Sink for ImageSink {
    fn write(&mut self, frame: &Frame) -> Result<(), Box<dyn std::error::Error>> {
        let _ = frame.data().save(&self.path)?;

        Ok(())
    }

    fn release(&mut self) {
        // Nothing to release for ImageSink
    }
}
