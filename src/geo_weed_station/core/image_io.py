"""Image input/output utilities with safety limits and format conversion."""

from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np

from geo_weed_station.core.exceptions import ImageReadError

MAX_IMAGE_WIDTH: int = 8000
MAX_IMAGE_HEIGHT: int = 6000


def get_image_size(path: Path) -> tuple[int, int]:
    """Fast extraction of image dimensions (width, height) without decoding pixel buffers.
    
    Inspects JPEG SOF markers, PNG IHDR chunk, or TIFF tags in microseconds.
    Falls back to (1024, 1024) if dimensions cannot be determined.
    """
    path = Path(path)
    if not path.is_file():
        return (1024, 1024)

    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic.startswith(b"\xff\xd8"):  # JPEG
                f.seek(2)
                while True:
                    m = f.read(2)
                    if len(m) < 2 or m[0] != 0xFF:
                        break
                    marker = m[1]
                    if marker in (0xD8, 0xD9):
                        continue
                    if marker == 0xDA:  # SOS: compressed scan data starts
                        break
                    len_bytes = f.read(2)
                    if len(len_bytes) < 2:
                        break
                    length = struct.unpack(">H", len_bytes)[0]
                    # SOF markers: 0xC0..0xCF except 0xC4, 0xC8, 0xCC
                    if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        f.read(1)  # precision
                        h, w = struct.unpack(">HH", f.read(4))
                        if w > 0 and h > 0:
                            return (int(w), int(h))
                    f.seek(length - 2, 1)

            elif magic.startswith(b"\x89PNG"):  # PNG
                f.seek(16)
                w, h = struct.unpack(">II", f.read(8))
                if w > 0 and h > 0:
                    return (int(w), int(h))

            elif magic.startswith((b"II\x2a\x00", b"MM\x00\x2a")):  # TIFF
                endian = "<" if magic.startswith(b"II") else ">"
                f.seek(4)
                ifd_bytes = f.read(4)
                if len(ifd_bytes) == 4:
                    ifd_offset = struct.unpack(f"{endian}I", ifd_bytes)[0]
                    f.seek(ifd_offset)
                    n_entries = struct.unpack(f"{endian}H", f.read(2))[0]
                    w, h = None, None
                    for _ in range(n_entries):
                        entry = f.read(12)
                        if len(entry) < 12:
                            break
                        tag, tag_type, count, val = struct.unpack(f"{endian}HHI I", entry)
                        if tag == 256:  # ImageWidth
                            w = val
                        elif tag == 257:  # ImageLength
                            h = val
                    if w and h:
                        return (int(w), int(h))
    except Exception:
        pass

    return (1024, 1024)


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


def validate_ms5(image: np.ndarray) -> None:
    """Validate normalized, registered MS5 data in R,G,B,RE,NIR order."""
    if image.ndim != 3 or image.shape[2] != 5 or min(image.shape[:2]) < 1:
        raise ImageReadError("MS5: нужны пять совмещённых каналов [R,G,B,RE,NIR]")
    if image.dtype != np.float32 or not np.isfinite(image).all():
        raise ImageReadError("MS5: ожидается конечный float32-массив")
    if image.min() < 0 or image.max() > 1:
        raise ImageReadError("MS5: значения float должны находиться в диапазоне [0,1]")


def read_ms5_image(path: Path) -> np.ndarray:
    """Read a registered five-band TIFF; integer scaling matches D04 inference.

    Band order is a required input contract, not guessed from pixel values.
    Accept interleaved YXS/YXC and planar SYX/CYX/QYX TIFF series.
    """
    import tifffile

    path = Path(path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ImageReadError("MS5: выберите пятиканальный TIFF [R,G,B,RE,NIR], не RGB-снимок")
    try:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            axes = series.axes
            if len(series.shape) != 3:
                raise ImageReadError("MS5: TIFF должен содержать ровно пять полос одного кадра")
            if axes in {"YXS", "YXC"} and series.shape[-1] == 5:
                image = series.asarray()
            elif axes in {"SYX", "CYX", "QYX", "IYX"} and series.shape[0] == 5:
                image = series.asarray().transpose(1, 2, 0)
            else:
                raise ImageReadError(f"MS5: неподдерживаемые оси/форма TIFF: {axes} {series.shape}")
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif np.issubdtype(image.dtype, np.floating):
            image = image.astype(np.float32)
        else:
            raise ImageReadError("MS5: поддерживаются TIFF uint8, uint16 и float [0,1]")
        validate_ms5(image)
        h, w = image.shape[:2]
        if w > MAX_IMAGE_WIDTH or h > MAX_IMAGE_HEIGHT:
            scale = min(MAX_IMAGE_WIDTH / w, MAX_IMAGE_HEIGHT / h)
            image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image)
    except ImageReadError:
        raise
    except Exception as exc:
        raise ImageReadError(f"Не удалось прочитать MS5 TIFF {path.name}: {exc}") from exc


def ms5_rgb_preview(image: np.ndarray) -> np.ndarray:
    """RGB uint8 on the same pixel grid; conversion matches reference D04 preview."""
    return np.ascontiguousarray((image[..., :3] * 255.0).clip(0, 255).astype(np.uint8))
