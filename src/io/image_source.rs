use crate::base::Source;
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
    fn next_frame(&mut self) -> Option<Frame> {
        if self.consumed {
            return None;
        }

        self.consumed = true;

        let img = image::open(&self.path).ok()?;

        let frame = Frame::new(img.to_rgb8(), 0, 0.0);

        Some(frame)
    }

    fn release(&mut self) {
        // Nothing to release for ImageSource
    }
}
