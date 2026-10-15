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

    pub fn add_boxed_filter(mut self, filter: Box<dyn Filter>) -> Self {
        self.filters.push(filter);
        self
    }

    pub fn filters(&self) -> &[Box<dyn Filter>] {
        &self.filters
    }

    pub fn process_frame(&self, mut frame: Frame) -> Frame {
        for (i, filter) in self.filters.iter().enumerate() {
            let filter_name = filter.name();
            let mut next_frame = filter.apply(&frame);
            next_frame.set_metadata(&format!("filter_{}", i), filter_name);
            frame = next_frame;
        }
        frame
    }

    /// Processes frames in a stream from source to sink with an optional progress callback and optional max_frames limit.
    pub fn run_with_progress<S: Source, K: Sink, F: FnMut(usize)>(
        &mut self,
        source: &mut S,
        sink: &mut K,
        max_frames: Option<usize>,
        mut on_frame: F,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut count = 0;

        while let Some(frame) = source.next_frame() {
            if let Some(max) = max_frames
                && count >= max {
                    break;
                }
            let processed_frame = self.process_frame(frame);
            sink.write(&processed_frame)?;
            count += 1;
            on_frame(count);
        }

        sink.release();
        Ok(count)
    }

    /// Processes frames in a stream from source to sink without keeping all frames in memory.
    pub fn run<S: Source, K: Sink>(
        &mut self,
        source: &mut S,
        sink: &mut K,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        self.run_with_progress(source, sink, None, |_| {})
    }

    /// Convenience method for test scenarios where all processed frames are needed in memory.
    pub fn run_and_collect<S: Source, K: Sink>(
        &mut self,
        source: &mut S,
        sink: &mut K,
    ) -> Result<Vec<Frame>, Box<dyn std::error::Error>> {
        let mut processed_frames = Vec::new();

        while let Some(frame) = source.next_frame() {
            let processed_frame = self.process_frame(frame);
            sink.write(&processed_frame)?;
            processed_frames.push(processed_frame);
        }

        sink.release();
        Ok(processed_frames)
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}
