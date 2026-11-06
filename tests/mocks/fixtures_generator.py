"""Generator of mock test fixtures: JPEGs with XMP, summary JSON, and mission ZIP archives."""

from __future__ import annotations

import io
import struct
import zipfile

from geo_weed_station.schemas.mission import Waypoint
from geo_weed_station.schemas.onboard_api import (
    SummaryDetection,
    SummaryDetections,
    SummaryItem,
    SummaryTargetCoords,
)

_XMP_JPEG_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


def make_xmp_xml(
    cs: str,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 2.0,
    yaw: float = 0.0,
    pitch: float = -90.0,
    roll: float = 0.0,
    lat: float = 55.751244,
    lon: float = 37.618423,
    focal_length_mm: float = 3.6,
) -> str:
    """Generate XMP metadata XML string with drone-dji, Camera, and agro namespaces."""
    norm_cs = "local_aruco" if "aruco" in cs.lower() else "wgs84"
    if norm_cs == "local_aruco":
        return f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
        xmlns:Camera="http://pix4d.com/camera/1.0/"
        xmlns:agro="http://geosaero.ru/agro/1.0/"
        drone-dji:RelativeAltitude="{z:+.2f}"
        drone-dji:GimbalYawDegree="{yaw:+.1f}"
        drone-dji:GimbalPitchDegree="{pitch:+.1f}"
        drone-dji:GimbalRollDegree="{roll:+.1f}"
        Camera:PerspectiveFocalLength="{focal_length_mm:.1f}"
        Camera:PrincipalPoint="1280.0,960.0"
        Camera:BandName="RGB"
        agro:CoordinateSystem="local_aruco"
        agro:CameraX="{x:.2f}"
        agro:CameraY="{y:.2f}"/>
  </rdf:RDF>
</x:xmpmeta>"""
    return f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
        xmlns:Camera="http://pix4d.com/camera/1.0/"
        xmlns:agro="http://geosaero.ru/agro/1.0/"
        drone-dji:RelativeAltitude="{z:+.2f}"
        drone-dji:GimbalYawDegree="{yaw:+.1f}"
        drone-dji:GimbalPitchDegree="{pitch:+.1f}"
        drone-dji:GimbalRollDegree="{roll:+.1f}"
        drone-dji:GpsLatitude="{lat:.6f}"
        drone-dji:GpsLongitude="{lon:.6f}"
        Camera:PerspectiveFocalLength="{focal_length_mm:.1f}"
        Camera:PrincipalPoint="2000.0,1500.0"
        Camera:BandName="RGB"
        agro:CoordinateSystem="wgs84"/>
  </rdf:RDF>
</x:xmpmeta>"""


def make_jpeg_bytes(xmp_xml: str | None = None) -> bytes:
    """Generate minimal valid JPEG byte stream with embedded APP1 XMP marker."""
    buf = io.BytesIO()
    # SOI
    buf.write(b"\xff\xd8")

    # APP0 marker
    app0_payload = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    buf.write(b"\xff\xe0" + struct.pack(">H", len(app0_payload) + 2) + app0_payload)

    # APP1 XMP marker
    if xmp_xml is not None:
        xmp_payload = _XMP_JPEG_HEADER + xmp_xml.encode("utf-8")
        buf.write(b"\xff\xe1" + struct.pack(">H", len(xmp_payload) + 2) + xmp_payload)

    # DQT
    dqt = b"\x00" + (b"\x10" * 64)
    buf.write(b"\xff\xdb" + struct.pack(">H", len(dqt) + 2) + dqt)

    # SOF0
    sof = b"\x08\x02\x80\x02\x80\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01"  # 640x640
    buf.write(b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof)

    # SOS
    sos = b"\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00"
    buf.write(b"\xff\xda" + struct.pack(">H", len(sos) + 2) + sos)

    # Pixel stream
    buf.write(b"\x00" * 32)

    # EOI
    buf.write(b"\xff\xd9")
    return buf.getvalue()


def make_summary_detections(
    run_id: str,
    waypoints: list[Waypoint],
    coordinate_system: str = "aruco_local",
) -> SummaryDetections:
    """Generate SummaryDetections with realistic weed detections for given waypoints."""
    items: list[SummaryItem] = []
    total_detections = 0

    weed_classes = ["taraxacum", "cirsium", "sonchus"]

    for idx, wp in enumerate(waypoints):
        dets: list[SummaryDetection] = []
        if idx % 2 == 0:
            cls_name = weed_classes[idx % len(weed_classes)]
            dets.append(
                SummaryDetection(
                    class_name=cls_name,
                    confidence=0.88,
                    bbox_px=[100, 120, 240, 280],
                    target_local_coords=SummaryTargetCoords(x=wp.x + 0.1, y=wp.y + 0.1),
                )
            )
            total_detections += 1

        items.append(
            SummaryItem(
                wp_index=idx,
                timestamp=1700000000.0 + idx * 5.0,
                camera_x=wp.x,
                camera_y=wp.y,
                altitude_m=wp.z,
                detections=dets,
            )
        )

    return SummaryDetections(
        run_id=run_id,
        coordinate_system=coordinate_system,
        detections_count=total_detections,
        items=items,
    )


def make_run_zip_bytes(
    run_id: str,
    waypoints: list[Waypoint],
    coordinate_system: str = "aruco_local",
) -> bytes:
    """Generate a mission ZIP archive containing JPEGs with XMP and summary.json."""
    zip_buf = io.BytesIO()
    summary = make_summary_detections(run_id, waypoints, coordinate_system)

    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, wp in enumerate(waypoints):
            xmp = make_xmp_xml(
                cs=coordinate_system,
                x=wp.x,
                y=wp.y,
                z=wp.z,
                yaw=getattr(wp, "yaw", 0.0),
                pitch=-90.0,
            )
            img_bytes = make_jpeg_bytes(xmp)
            zf.writestr(f"wp_{idx:04d}.jpg", img_bytes)

        summary_json_str = summary.model_dump_json(indent=2)
        zf.writestr("summary.json", summary_json_str)

    return zip_buf.getvalue()
