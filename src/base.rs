use crate::frame::Frame;

pub trait Filter: Send + Sync {
    fn name(&self) -> &str;
    fn apply(&self, frame: &Frame) -> Frame;
}

pub trait Source {
    fn next_frame(&mut self) -> Option<Frame>;
    fn release(&mut self);
}

pub trait Sink {
    fn write(&mut self, frame: &Frame) -> Result<(), Box<dyn std::error::Error>>;
    fn release(&mut self);
}
