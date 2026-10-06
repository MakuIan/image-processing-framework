from dataclasses import dataclass, field
from typing import Any


@dataclass
class Frame:
    """Repräsentiert einen Bild- oder Video-Frame."""

    data: Any  # np.ndarray (uint8 oder float32)
    index: int = 0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Frame":
        """Gibt eine tiefe Kopie des Frames zurück."""
        return Frame(
            data=self.data.copy(),
            index=self.index,
            timestamp=self.timestamp,
            metadata=self.metadata.copy(),
        )
