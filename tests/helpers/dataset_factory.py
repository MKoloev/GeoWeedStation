"""Dataset factory creating synthetic geotagged images, video clips, and mixed test datasets."""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Optional, Tuple
import cv2
import numpy as np

from geo_weed_station.schemas.entities import CoordinateSystem

_XMP_JPEG_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


def make_geotagged_jpeg(
    path: Path,
    *,
    cs: CoordinateSystem = CoordinateSystem.LOCAL_ARUCO,
    altitude_m: Optional[float] = 10.0,
    yaw_deg: float = 0.0,
    xy: Tuple[float, float] = (5.0, 5.0),
    size: Tuple[int, int] = (64, 48),
) -> Path:
    """Create a decodable JPEG file with embedded XMP spatial metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    alt_str = f"{altitude_m:+.2f}" if altitude_m is not None else "+0.00"
    if cs == CoordinateSystem.LOCAL_ARUCO:
        coord_xml = f'agro:CoordinateSystem="local_aruco" agro:CameraX="{xy[0]:.2f}" agro:CameraY="{xy[1]:.2f}"'
    else:
        coord_xml = f'agro:CoordinateSystem="wgs84" drone-dji:GpsLatitude="{xy[1]:.6f}" drone-dji:GpsLongitude="{xy[0]:.6f}"'

    xmp_xml = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
        xmlns:Camera="http://pix4d.com/camera/1.0/"
        xmlns:agro="http://geosaero.ru/agro/1.0/"
        drone-dji:RelativeAltitude="{alt_str}"
        drone-dji:GimbalYawDegree="{yaw_deg:+.1f}"
        drone-dji:GimbalPitchDegree="-90.0"
        drone-dji:GimbalRollDegree="0.0"
        Camera:PerspectiveFocalLength="4.5"
        Camera:SensorWidth="6.4"
        Camera:SensorHeight="4.8"
        {coord_xml}/>
  </rdf:RDF>
</x:xmpmeta>"""

    # Generate image buffer
    img = np.full((size[1], size[0], 3), 120, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    data = buf.tobytes()

    # Insert APP1 XMP segment after SOI marker
    payload = _XMP_JPEG_HEADER + xmp_xml.encode("utf-8")
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    data = data[:2] + segment + data[2:]

    path.write_bytes(data)
    return path


def make_broken_jpeg(path: Path) -> Path:
    """Create a corrupt, truncated JPEG file that fails image decoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00CORRUPTED_DATA")
    return path


def make_video(path: Path, frames: int = 15, fps: int = 15, size: Tuple[int, int] = (64, 48)) -> Path:
    """Generate a test mp4 video stream."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), (i * 15) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def make_dataset(root: Path, n_valid: int = 3, n_no_xmp: int = 2, n_broken: int = 1) -> Path:
    """Create a synthetic dataset folder containing valid XMP, missing XMP, and corrupted files."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    for i in range(n_valid):
        make_geotagged_jpeg(
            root / f"valid_{i:03d}.jpg",
            cs=CoordinateSystem.LOCAL_ARUCO,
            altitude_m=12.0,
            yaw_deg=float(i * 10),
            xy=(float(i * 5), float(i * 3)),
        )

    for i in range(n_no_xmp):
        # Plain JPEG with zero metadata
        img = np.full((48, 64, 3), 100, dtype=np.uint8)
        cv2.imwrite(str(root / f"no_xmp_{i:03d}.jpg"), img)

    for i in range(n_broken):
        make_broken_jpeg(root / f"broken_{i:03d}.jpg")

    return root
