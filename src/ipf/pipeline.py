from typing import List
from typing import Optional

from ipf.base import BaseFilter
from ipf.base import BaseSink
from ipf.base import BaseSource
from ipf.frame import Frame


class Pipeline:
    """Verarbeitet Frames durch eine Kette von Filtern."""

    def __init__(self):
        self.filters: List[BaseFilter] = []

    def add_filter(self, filter_module: BaseFilter) -> "Pipeline":
        """Fügt einen Filter hinzu und gibt selbst zurück (Method Chaining)."""
        self.filters.append(filter_module)
        return self

    def process_frame(self, frame: Frame) -> Frame:
        """Reicht den Frame nacheinander durch alle filter.apply(frame) und gibt den fertigen Frame zurück."""
        for filter_module in self.filters:
            frame = filter_module.apply(frame)
        return frame

    def run(
        self, source: BaseSource, sink: Optional[BaseSink] = None
    ) -> List[Frame]:
        """
        Verarbeitet alle Frames von der Quelle.

        Args:
            source: Die Datenquelle (Video oder Bild).
            sink: Optionale Ausgabesink zur Speicherung.

        Returns:
            Liste aller verarbeiteten Frames.
        """
        processed_frames: List[Frame] = []
        for frame in source:
            processed_frame = self.process_frame(frame)
            if sink is not None:
                sink.write(processed_frame)
            processed_frames.append(processed_frame)
        source.release()
        if sink is not None:
            sink.release()
        return processed_frames
