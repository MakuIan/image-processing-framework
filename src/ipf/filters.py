import cv2
import numpy as np
from ipf.base import BaseFilter
from ipf.frame import Frame


class GaussianNoiseFilter(BaseFilter):
    """Fügt Gaußsches Rauschen zum Bild hinzu."""

    def __init__(self, mean: float = 0.0, sigma: float = 25.0, name: str = "GaussianNoise"):
        super().__init__(name)
        self.mean = mean
        self.sigma = sigma

    def apply(self, frame: Frame) -> Frame:
        noise = np.random.normal(self.mean, self.sigma, frame.data.shape).astype(np.float32)
        noisy_data = frame.data.astype(np.float32) + noise
        clipped = np.clip(noisy_data, 0, 255).astype(np.uint8)
        return Frame(data=clipped, index=frame.index, timestamp=frame.timestamp, metadata=frame.metadata.copy())


class BilateralDenoiseFilter(BaseFilter):
    """Bilaterale Entzung zum Entfernen von Rauschen bei Kanten."""

    def __init__(self, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0,
                 name: str = "BilateralDenoise"):
        super().__init__(name)
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, frame: Frame) -> Frame:
        denoised_data = cv2.bilateralFilter(frame.data.astype(np.float32), self.d,
                                            self.sigma_color, self.sigma_space)
        return Frame(data=denoised_data, index=frame.index, timestamp=frame.timestamp,
                     metadata=frame.metadata.copy())


class BlurFilter(BaseFilter):
    """Gaußsches Blur."""

    def __init__(self, kernel_size: int = 5, name: str = "Blur"):
        super().__init__(name)
        self.kernel_size = kernel_size

    def apply(self, frame: Frame) -> Frame:
        blurred_data = cv2.GaussianBlur(frame.data.astype(np.float32), (self.kernel_size, self.kernel_size), 0)
        return Frame(data=blurred_data, index=frame.index, timestamp=frame.timestamp,
                     metadata=frame.metadata.copy())
