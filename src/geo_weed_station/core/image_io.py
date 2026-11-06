"""Image input/output utilities with safety limits and format conversion."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from geo_weed_station.core.exceptions import ImageReadError

MAX_IMAGE_WIDTH: int = 8000
MAX_IMAGE_HEIGHT: int = 6000


def read_image(path: Path) -> np.ndarray:
    """Safely read an image file using OpenCV and convert it to RGB format.
    
    If the image exceeds MAX_IMAGE_WIDTH x MAX_IMAGE_HEIGHT, it is safely downscaled
    with aspect ratio preserved to prevent out-of-memory errors on large aerial imagery.
    
    Args:
        path: Path to the image file.
        
    Returns:
        np.ndarray: Image array in RGB format (H, W, 3).
        
    Raises:
        ImageReadError: If the file does not exist, cannot be opened, or is corrupted.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageReadError(f"Image file does not exist: {path}")

    try:
        # cv2.imread does not raise exceptions on corrupted files; it returns None
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ImageReadError(f"Corrupted or unsupported image file: {path.name}")

        h, w = img_bgr.shape[:2]
        if h <= 0 or w <= 0:
            raise ImageReadError(f"Invalid image dimensions ({w}x{h}) in {path.name}")

        # Limit maximum image resolution to 8000x6000 to prevent OOM
        if w > MAX_IMAGE_WIDTH or h > MAX_IMAGE_HEIGHT:
            scale = min(MAX_IMAGE_WIDTH / w, MAX_IMAGE_HEIGHT / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert OpenCV BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    except ImageReadError:
        raise
    except Exception as e:
        raise ImageReadError(f"Unexpected error decoding image {path.name}: {e}") from e
