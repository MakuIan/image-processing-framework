from typing import Iterator, Union
import cv2

from ipf.base import BaseSource
from ipf.frame import Frame


class VideoSource(BaseSource):
    """Quelle für Videodateien oder Kameras."""

    def __init__(self, file_path: Union[str, int] | None = None, camera_index: int = 0):
        if file_path is not None and isinstance(file_path, str):
            self.cap = cv2.VideoCapture(file_path)
        else:
            self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Video/Kamera konnte nicht geffnet werden: {file_path or camera_index}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps > 0 else 30.0

    def __iter__(self) -> Iterator[Frame]:
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield Frame(data=frame, index=idx, timestamp=idx / self._fps)
            idx += 1

    def release(self) -> None:
        self.cap.release()
