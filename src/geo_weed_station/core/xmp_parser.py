"""High-performance direct XMP metadata parser for JPEG and TIFF imagery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Dict, Optional, Tuple
import xml.etree.ElementTree as ET

from geo_weed_station.core.exceptions import (
    XmpError,
    XmpMalformedError,
    XmpNotFoundError,
)
from geo_weed_station.schemas.entities import (
    CameraSpatialPose,
    CoordinateSystem,
    SensorOptics,
)
from geo_weed_station.schemas.payload import FramePayload

# Standard XMP header in JPEG APP1 marker
_XMP_JPEG_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


@dataclass(frozen=True)
class XmpSpatialData:
    """Parsed spatial and optical metadata container from XMP packet."""

    pose: Optional[CameraSpatialPose]
    optics: Optional[SensorOptics]
    raw_tags: Dict[str, str]

    @property
    def has_spatial_data(self) -> bool:
        return self.pose is not None


def _read_jpeg_xmp(path: Path) -> str:
    """Extract XMP XML string from JPEG APP1 markers without decoding pixel data."""
    with open(path, "rb") as f:
        soi = f.read(2)
        if soi != b"\xff\xd8":
            raise XmpNotFoundError(f"File {path.name} is not a valid JPEG (missing SOI marker)")

        while True:
            marker_header = f.read(2)
            if len(marker_header) < 2:
                break

            # JPEG markers start with 0xFF followed by non-0x00 and non-0xFF byte
            if marker_header[0] != 0xFF:
                # Synchronize or break if corrupted
                break

            marker = marker_header[1]

            # Standalone markers with no length
            if marker in (0xD8, 0xD9):  # SOI, EOI
                continue
            if marker == 0xDA:  # SOS (Start of Scan) - compressed pixel stream begins
                break

            length_bytes = f.read(2)
            if len(length_bytes) < 2:
                break
            length = struct.unpack(">H", length_bytes)[0]
            if length < 2:
                break
            payload_len = length - 2

            # APP1 marker: 0xFFE1
            if marker == 0xE1:
                prefix = f.read(min(len(_XMP_JPEG_HEADER), payload_len))
                if prefix == _XMP_JPEG_HEADER:
                    remaining_bytes = payload_len - len(_XMP_JPEG_HEADER)
                    xmp_bytes = f.read(remaining_bytes)
                    try:
                        return xmp_bytes.decode("utf-8", errors="replace")
                    except Exception as e:
                        raise XmpMalformedError(f"Failed to decode XMP payload in {path.name}: {e}") from e
                else:
                    # Skip rest of this APP1 (e.g. EXIF)
                    f.seek(payload_len - len(prefix), 1)
            else:
                # Skip other markers (APP0, APP2, DQT, DHT, SOF, etc.)
                f.seek(payload_len, 1)

    raise XmpNotFoundError(f"No XMP APP1 packet found in JPEG: {path.name}")


def _read_tiff_xmp(path: Path) -> str:
    """Extract XMP XML string from TIFF Tag 700 (0x02BC) without reading image raster."""
    with open(path, "rb") as f:
        header = f.read(4)
        if len(header) < 4:
            raise XmpNotFoundError(f"File {path.name} is too short to be a valid TIFF")

        if header[:2] == b"II" and header[2:4] in (b"*\x00", b"\x2a\x00"):
            endian = "<"
        elif header[:2] == b"MM" and header[2:4] in (b"\x00*", b"\x00\x2a"):
            endian = ">"
        else:
            raise XmpNotFoundError(f"File {path.name} is not a valid TIFF header")

        ifd_offset_bytes = f.read(4)
        if len(ifd_offset_bytes) < 4:
            raise XmpNotFoundError(f"Missing IFD0 offset in TIFF: {path.name}")
        ifd_offset = struct.unpack(f"{endian}I", ifd_offset_bytes)[0]

        while ifd_offset > 0:
            f.seek(ifd_offset)
            num_entries_bytes = f.read(2)
            if len(num_entries_bytes) < 2:
                break
            num_entries = struct.unpack(f"{endian}H", num_entries_bytes)[0]

            for _ in range(num_entries):
                entry = f.read(12)
                if len(entry) < 12:
                    break
                tag, tag_type, count, val_offset = struct.unpack(f"{endian}HHI I", entry)
                # Tag 700 = 0x02BC (XMP packet)
                if tag == 700:
                    if count <= 4:
                        # Pack value offset bytes
                        raw_data = struct.pack(f"{endian}I", val_offset)[:count]
                    else:
                        f.seek(val_offset)
                        raw_data = f.read(count)
                    # Strip any trailing null bytes
                    cleaned = raw_data.rstrip(b"\x00")
                    return cleaned.decode("utf-8", errors="replace")

            # Next IFD offset
            next_ifd_bytes = f.read(4)
            if len(next_ifd_bytes) < 4:
                break
            ifd_offset = struct.unpack(f"{endian}I", next_ifd_bytes)[0]

    raise XmpNotFoundError(f"No XMP Tag 700 found in TIFF: {path.name}")


def read_xmp_packet(path: Path) -> str:
    """Read raw XMP XML packet from JPEG or TIFF image file.
    
    Raises:
        FileNotFoundError: if path does not exist.
        XmpNotFoundError: if file contains no XMP packet.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    # Inspect file signature
    with open(path, "rb") as f:
        magic = f.read(4)

    if magic.startswith(b"\xff\xd8"):
        return _read_jpeg_xmp(path)
    elif magic.startswith(b"II") or magic.startswith(b"MM"):
        return _read_tiff_xmp(path)

    # Fallback: scan beginning of file for embedded XMP packet
    with open(path, "rb") as f:
        chunk = f.read(131072)
    start_tag = b"<x:xmpmeta"
    end_tag = b"</x:xmpmeta>"
    start_idx = chunk.find(start_tag)
    end_idx = chunk.find(end_tag)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        xmp_slice = chunk[start_idx : end_idx + len(end_tag)]
        return xmp_slice.decode("utf-8", errors="replace")

    raise XmpNotFoundError(f"Unsupported image format or missing XMP in {path.name}")


def parse_xmp(xmp_xml: str | Path) -> XmpSpatialData:
    """Parse raw XMP XML string or extract and parse from image path.
    
    Supports both attribute-based and child-element-based RDF representations.
    
    Raises:
        XmpMalformedError: if XML syntax is invalid or coordinates are out of bounds.
        XmpNotFoundError: if image path has no XMP metadata.
    """
    if isinstance(xmp_xml, Path) or (isinstance(xmp_xml, str) and not xmp_xml.strip().startswith("<") and Path(xmp_xml).exists()):
        xmp_xml = read_xmp_packet(Path(xmp_xml))

    try:
        root = ET.fromstring(xmp_xml)
    except ET.ParseError as e:
        raise XmpMalformedError(f"Malformed XMP XML: {e}") from e

    raw_tags: Dict[str, str] = {}

    # Traverse all elements to collect tags (ignoring XML namespace prefixes)
    for elem in root.iter():
        # Tag name without namespace
        local_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if elem.text and elem.text.strip():
            raw_tags[local_name] = elem.text.strip()

        # Collect attributes on elements (e.g. rdf:Description attributes)
        for attr_key, attr_val in elem.attrib.items():
            attr_local = attr_key.split("}")[-1] if "}" in attr_key else attr_key
            raw_tags[attr_local] = attr_val.strip()

    # 1. Parse Camera Orientation Angles
    yaw_val = raw_tags.get("GimbalYawDegree")
    pitch_val = raw_tags.get("GimbalPitchDegree")
    roll_val = raw_tags.get("GimbalRollDegree")

    angles_valid = False
    yaw_deg = pitch_deg = roll_deg = 0.0
    if yaw_val is not None and pitch_val is not None and roll_val is not None:
        try:
            yaw_deg = float(yaw_val)
            pitch_deg = float(pitch_val)
            roll_deg = float(roll_val)
            angles_valid = True
        except ValueError:
            angles_valid = False

    # 2. Parse Relative Altitude H
    alt_val = raw_tags.get("RelativeAltitude")
    altitude_m = 0.0
    alt_valid = False
    if alt_val is not None:
        try:
            altitude_m = float(alt_val)
            if altitude_m > 0.0:
                alt_valid = True
        except ValueError:
            alt_valid = False

    # 3. Determine Coordinate System and Coordinates
    pose: Optional[CameraSpatialPose] = None
    cs_tag = raw_tags.get("CoordinateSystem", "").lower()

    if angles_valid and alt_valid:
        # Check local_aruco mode
        if cs_tag == "local_aruco" or ("CameraX" in raw_tags and "CameraY" in raw_tags and cs_tag != "wgs84"):
            cam_x_val = raw_tags.get("CameraX")
            cam_y_val = raw_tags.get("CameraY")
            if cam_x_val is not None and cam_y_val is not None:
                try:
                    cam_x = float(cam_x_val)
                    cam_y = float(cam_y_val)
                    pose = CameraSpatialPose(
                        coordinate_system=CoordinateSystem.LOCAL_ARUCO,
                        x=cam_x,
                        y=cam_y,
                        altitude_m=altitude_m,
                        roll_deg=roll_deg,
                        pitch_deg=pitch_deg,
                        yaw_deg=yaw_deg,
                    )
                except ValueError:
                    pose = None

        # Check wgs84 mode
        elif cs_tag == "wgs84" or ("GpsLatitude" in raw_tags and "GpsLongitude" in raw_tags):
            lat_val = raw_tags.get("GpsLatitude")
            lon_val = raw_tags.get("GpsLongitude")
            if lat_val is not None and lon_val is not None:
                try:
                    lat = float(lat_val)
                    lon = float(lon_val)
                except ValueError as e:
                    raise XmpMalformedError(f"Invalid WGS84 float values: {e}") from e

                if not (-90.0 <= lat <= 90.0):
                    raise XmpMalformedError(f"WGS84 latitude out of range [-90, 90]: {lat}")
                if not (-180.0 <= lon <= 180.0):
                    raise XmpMalformedError(f"WGS84 longitude out of range [-180, 180]: {lon}")

                pose = CameraSpatialPose(
                    coordinate_system=CoordinateSystem.WGS84,
                    x=lon,  # X = Longitude
                    y=lat,  # Y = Latitude
                    altitude_m=altitude_m,
                    roll_deg=roll_deg,
                    pitch_deg=pitch_deg,
                    yaw_deg=yaw_deg,
                )

    # 4. Parse Optics parameters
    optics: Optional[SensorOptics] = None
    focal_val = raw_tags.get("PerspectiveFocalLength")
    if focal_val is not None:
        try:
            focal_length_mm = float(focal_val)
            principal_point: Optional[Tuple[float, float]] = None
            pp_val = raw_tags.get("PrincipalPoint")
            if pp_val:
                parts = pp_val.split(",")
                if len(parts) == 2:
                    principal_point = (float(parts[0].strip()), float(parts[1].strip()))

            band_names: list[str] = []
            band_val = raw_tags.get("BandName")
            if band_val:
                band_names = [b.strip() for b in band_val.split(",") if b.strip()]

            optics = SensorOptics(
                focal_length_mm=focal_length_mm,
                principal_point_px=principal_point,
                band_names=band_names,
            )
        except ValueError:
            optics = None

    return XmpSpatialData(
        pose=pose,
        optics=optics,
        raw_tags=raw_tags,
    )


def extract_spatial_data(path: Path) -> XmpSpatialData:
    """Extract and parse spatial data directly from image file path."""
    xml_packet = read_xmp_packet(path)
    return parse_xmp(xml_packet)


def has_xmp(path: Path) -> bool:
    """Fast check whether an image file contains readable XMP metadata."""
    try:
        read_xmp_packet(path)
        return True
    except (XmpError, FileNotFoundError, OSError):
        return False


def build_frame_payload(path: Path, image_size_px: Tuple[int, int]) -> FramePayload:
    """Construct a FramePayload from image file.
    
    If XMP is missing or corrupted, returns FramePayload with pose=None.
    Does not crash or raise exceptions on missing metadata.
    """
    path = Path(path)
    pose: Optional[CameraSpatialPose] = None
    optics: Optional[SensorOptics] = None

    try:
        spatial_data = extract_spatial_data(path)
        pose = spatial_data.pose
        optics = spatial_data.optics
    except Exception:
        # File is treated as image without georeferencing
        pose = None
        optics = None

    return FramePayload(
        frame_id=path.stem,
        source_path=str(path),
        image_size_px=image_size_px,
        image=None,
        pose=pose,
        optics=optics,
        detections=[],
    )
