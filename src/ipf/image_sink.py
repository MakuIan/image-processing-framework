import cv2
from ipf.base import BaseSink
from ipf.frame import Frame


class ImageSink(BaseSink):
    """Speichert einzelne Frames als Bilddatei."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def write(self, frame: Frame) -> None:
        cv2.imwrite(self.output_path, frame.data)

    def release(self) -> None:
        pass
