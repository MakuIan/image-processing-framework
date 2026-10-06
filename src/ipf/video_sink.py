from typing import Any
import cv2
import importlib

from ipf.base import BaseSink
from ipf.frame import Frame


# Workaround for OpenCV versions that don't expose VideoWriter_fourcc in __all__
try:
    from cv2 import VideoWriter_fourcc as vfw_fourcc
except ImportError:
    vfw_fourcc = None


class VideoSink(BaseSink):
    """Speichert Frames als Videodatei."""

    def __init__(self, output_path: str, fps: float = 30.0, fourcc_str: str = "mp4v"):
        self.output_path = output_path
        self.fps = fps
        self.fourcc_str = fourcc_str
        self.writer = None

    def write(self, frame: Frame) -> None:
        if self.writer is None:
            height, width = frame.data.shape[:2]
            try:
                fourcc = vfw_fourcc(*self.fourcc_str)
            except AttributeError:
                fourcc = cv2.VideoWriter_fourcc(*self.fourcc_str)
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (width, height)
            )
        if self.writer is None:
            raise RuntimeError(f"Fehler beim {self.output_path}")
        self.writer.write(frame.data)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
