use crate::base::{Filter, Sink, Source};
use crate::frame::Frame;

pub struct Pipeline {
    filters: Vec<Box<dyn Filter>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    pub fn add_filter<F: Filter + 'static>(mut self, filter: F) -> Self {
        self.filters.push(Box::new(filter));
        self
    }

    pub fn process_frame(&self, mut frame: Frame) -> Frame {
        for filter in &self.filters {
            frame = filter.apply(&frame);
        }
        frame
    }

    pub fn run<S: Source, K: Sink>(&mut self, source: &mut S, sink: &mut K) -> Vec<Frame> {
        let mut processed_frames = Vec::new();

        while let Some(frame) = source.next_frame() {
            let processed_frame = self.process_frame(frame);

            sink.write(&processed_frame).ok();

            processed_frames.push(processed_frame);
        }

        sink.release();

        processed_frames
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}
