from typing import Iterator
import cv2
import os

from ipf.base import BaseSource
from ipf.frame import Frame


class ImageSource(BaseSource):
    """Quelle für einzelne Bilddateien."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Bild nicht gefunden: {file_path}")
        self.img = cv2.imread(file_path)

    def __iter__(self) -> Iterator[Frame]:
        yield Frame(data=self.img, index=0)

    def release(self) -> None:
        self.img = None
