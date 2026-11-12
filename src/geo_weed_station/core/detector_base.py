"""Abstract base class interface for object detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from geo_weed_station.schemas.entities import DetectedWeed


class BaseDetector(ABC):
    """Abstract base detector interface for weed identification."""

    @abstractmethod
    def load(self) -> None:
        """Load detector model weights into execution session."""

    @abstractmethod
    def detect(self, image: np.ndarray, *, classification_image: np.ndarray | None = None) -> list[DetectedWeed]:
        """Perform object detection on an image array.
        
        Args:
            image: RGB uint8 (H,W,3), or normalized MS5 float32 (H,W,5).
            classification_image: Optional aligned RGB uint8 for crop classification.
            
        Returns:
            List of DetectedWeed instances with normalized BBox [0..1] in original frame coordinates.
        """

    @abstractmethod
    def close(self) -> None:
        """Release underlying inference session and free memory."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready for inference."""
