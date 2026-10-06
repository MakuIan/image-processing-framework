from abc import ABC
from abc import abstractmethod
from typing import Iterator

from ipf.frame import Frame


class BaseSource(ABC):
    """Abstrakte Basisklasse für Bild- und Video-Quellen."""

    @abstractmethod
    def __iter__(self) -> Iterator:
        pass

    @abstractmethod
    def release(self) -> None:
        pass


class BaseFilter(ABC):
    """Abstrakte Basisklasse für Bild- und Video-Filter."""

    def __init__(self, name: str = "Filter"):
        self.name = name

    @abstractmethod
    def apply(self, frame: Frame) -> Frame:
        pass


class BaseSink(ABC):
    """Abstrakte Basisklasse für Bild- und Video-Sinks."""

    @abstractmethod
    def write(self, frame: Frame):
        pass

    @abstractmethod
    def release(self) -> None:
        pass
