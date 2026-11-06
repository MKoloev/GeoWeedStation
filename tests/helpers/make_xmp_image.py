"""Test helper module to create JPEG and TIFF test fixture files with embedded XMP packets."""

from __future__ import annotations

import struct
from pathlib import Path

_XMP_JPEG_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"

SAMPLE_XMP_LOCAL_ARUCO = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
        xmlns:Camera="http://pix4d.com/camera/1.0/"
        xmlns:agro="http://geosaero.ru/agro/1.0/"
        drone-dji:RelativeAltitude="+2.00"
        drone-dji:GimbalYawDegree="+15.0"
        drone-dji:GimbalPitchDegree="-90.0"
        drone-dji:GimbalRollDegree="0.0"
        Camera:PerspectiveFocalLength="3.6"
        Camera:PrincipalPoint="1280.0,960.0"
        Camera:BandName="RGB"
        agro:CoordinateSystem="local_aruco"
        agro:CameraX="1.50"
        agro:CameraY="0.80"/>
  </rdf:RDF>
</x:xmpmeta>"""

SAMPLE_XMP_WGS84 = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
        xmlns:Camera="http://pix4d.com/camera/1.0/"
        xmlns:agro="http://geosaero.ru/agro/1.0/"
        drone-dji:RelativeAltitude="+15.50"
        drone-dji:GimbalYawDegree="-45.0"
        drone-dji:GimbalPitchDegree="-85.0"
        drone-dji:GimbalRollDegree="1.2"
        Camera:PerspectiveFocalLength="8.8"
        Camera:PrincipalPoint="2000.0,1500.0"
        Camera:BandName="RGB,RedEdge"
        agro:CoordinateSystem="wgs84"
        drone-dji:GpsLatitude="55.751244"
        drone-dji:GpsLongitude="37.618423"/>
  </rdf:RDF>
</x:xmpmeta>"""

SAMPLE_XMP_ELEMENT_FORM = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
        xmlns:agro="http://geosaero.ru/agro/1.0/">
      <drone-dji:RelativeAltitude>+3.20</drone-dji:RelativeAltitude>
      <drone-dji:GimbalYawDegree>0.0</drone-dji:GimbalYawDegree>
      <drone-dji:GimbalPitchDegree>-90.0</drone-dji:GimbalPitchDegree>
      <drone-dji:GimbalRollDegree>0.0</drone-dji:GimbalRollDegree>
      <agro:CoordinateSystem>local_aruco</agro:CoordinateSystem>
      <agro:CameraX>2.10</agro:CameraX>
      <agro:CameraY>1.80</agro:CameraY>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"""


def create_jpeg_with_xmp(path: Path, xmp_xml: str | None = None, padding_bytes: int = 0) -> Path:
    """Create a syntactically valid minimal JPEG file with an optional APP1 XMP marker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        # SOI (Start of Image)
        f.write(b"\xff\xd8")

        # APP0 marker (JFIF header)
        app0_payload = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        f.write(b"\xff\xe0" + struct.pack(">H", len(app0_payload) + 2) + app0_payload)

        # APP1 marker with XMP
        if xmp_xml is not None:
            xmp_payload = _XMP_JPEG_HEADER + xmp_xml.encode("utf-8")
            f.write(b"\xff\xe1" + struct.pack(">H", len(xmp_payload) + 2) + xmp_payload)

        # DQT marker (minimal quant table)
        dqt_payload = b"\x00" + (b"\x10" * 64)
        f.write(b"\xff\xdb" + struct.pack(">H", len(dqt_payload) + 2) + dqt_payload)

        # SOF0 (Start of Frame) - 8000x6000 or default
        sof_payload = b"\x08\x17\x70\x1f\x40\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01"
        f.write(b"\xff\xc0" + struct.pack(">H", len(sof_payload) + 2) + sof_payload)

        # SOS (Start of Scan) - pixel stream starts
        sos_payload = b"\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00"
        f.write(b"\xff\xda" + struct.pack(">H", len(sos_payload) + 2) + sos_payload)

        # Simulated compressed stream / padding bytes
        if padding_bytes > 0:
            # Write in chunks
            chunk = b"\x00" * 65536
            f.writelines(chunk for _ in range(padding_bytes // 65536))
            remainder = padding_bytes % 65536
            if remainder > 0:
                f.write(b"\x00" * remainder)
        else:
            f.write(b"\x00" * 32)

        # EOI (End of Image)
        f.write(b"\xff\xd9")

    return path


def create_tiff_with_xmp(path: Path, xmp_xml: str) -> Path:
    """Create a minimal valid TIFF file containing Tag 700 (XMP)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xmp_bytes = xmp_xml.encode("utf-8") + b"\x00"
    xmp_len = len(xmp_bytes)

    with open(path, "wb") as f:
        # Little-endian header: 'II', version 42, IFD0 offset 8
        f.write(b"II\x2a\x00\x08\x00\x00\x00")

        # IFD0 at offset 8:
        # 1 entry: Tag 700 (0x02BC, type BYTE=1, count=xmp_len, value_offset)
        num_entries = 1
        f.write(struct.pack("<H", num_entries))

        tag = 700
        tag_type = 1  # BYTE
        count = xmp_len
        # Offset where XMP data is stored (after IFD0 header: 8 + 2 + 12 + 4 = 26)
        xmp_offset = 26
        f.write(struct.pack("<HHI I", tag, tag_type, count, xmp_offset))

        # Next IFD offset = 0
        f.write(struct.pack("<I", 0))

        # Write XMP payload at xmp_offset
        f.write(xmp_bytes)

    return path
