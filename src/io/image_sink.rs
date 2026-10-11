use crate::base::Sink;
use crate::frame::Frame;
use std::path::PathBuf;

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
        frame.data().save(&self.path)?;
        Ok(())
    }

    fn release(&mut self) {
        // Nothing to release for ImageSink
    }
}
