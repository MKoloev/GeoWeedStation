"""Unit tests for DownloadManager: chunked downloads, cancellation, zip extraction, and Zip-Slip protection."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from geo_weed_station.network.download_manager import DownloadManager
from geo_weed_station.network.drone_client import DroneClient
from geo_weed_station.network.exceptions import DownloadCancelled


class MockDroneClientForDownload(DroneClient):
    """Mock client returning controllable binary chunk stream for stream_archive."""

    def __init__(self, archive_bytes: bytes) -> None:
        super().__init__(host="127.0.0.1", port=8000)
        self.archive_bytes = archive_bytes

    def stream_archive(self, run_id: str, chunk_bytes: int = 65536):
        buf = io.BytesIO(self.archive_bytes)
        while True:
            chunk = buf.read(chunk_bytes)
            if not chunk:
                break
            yield chunk


def create_sample_zip(files: dict[str, bytes]) -> bytes:
    """Create in-memory zip bytes with specified files."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        for fname, content in files.items():
            zf.writestr(fname, content)
    return bio.getvalue()


def test_download_run_chunked_success(tmp_path: Path):
    """Verify chunked download creates valid .zip and invokes progress callback."""
    zip_bytes = create_sample_zip({"manifest.json": b'{"run_id": "test_run"}'})
    mock_client = MockDroneClientForDownload(zip_bytes)

    dl_dir = tmp_path / "downloads"
    manager = DownloadManager(client=mock_client, target_dir=dl_dir, chunk_bytes=10)

    progress_reports = []

    def on_progress(downloaded, total):
        progress_reports.append(downloaded)

    zip_file = manager.download_run("run_123", progress_cb=on_progress)

    assert zip_file.exists()
    assert zip_file.name == "run_123.zip"
    assert not (dl_dir / "run_123.zip.part").exists()  # .part should be cleaned up
    assert zip_file.read_bytes() == zip_bytes
    assert len(progress_reports) > 1  # Verify multiple chunks were processed


def test_download_cancellation_cleans_up_part(tmp_path: Path):
    """Verify cancelling download raises DownloadCancelled and removes temporary .part file."""
    # 100 KB payload
    zip_bytes = b"X" * 102400
    mock_client = MockDroneClientForDownload(zip_bytes)

    dl_dir = tmp_path / "downloads"
    manager = DownloadManager(client=mock_client, target_dir=dl_dir, chunk_bytes=1024)

    cancel_counter = [0]

    def cancel_after_two_chunks():
        cancel_counter[0] += 1
        return cancel_counter[0] >= 3

    with pytest.raises(DownloadCancelled):
        manager.download_run("run_cancel", cancel_flag=cancel_after_two_chunks)

    # Ensure neither final zip nor partial file remain on disk
    assert not (dl_dir / "run_cancel.zip").exists()
    assert not (dl_dir / "run_cancel.zip.part").exists()


def test_extract_run_success(tmp_path: Path):
    """Verify regular zip extraction creates target directory structure."""
    zip_bytes = create_sample_zip(
        {
            "images/photo_01.jpg": b"\xff\xd8\xff\xe0",
            "images/photo_02.png": b"\x89PNG\r\n",
            "metadata.json": b"{}",
        }
    )
    zip_path = tmp_path / "my_run.zip"
    zip_path.write_bytes(zip_bytes)

    manager = DownloadManager(
        client=MockDroneClientForDownload(b""),
        target_dir=tmp_path / "extracted",
    )
    run_dir = manager.extract_run(zip_path)

    assert run_dir.is_dir()
    assert (run_dir / "metadata.json").is_file()
    assert (run_dir / "images" / "photo_01.jpg").is_file()

    # Test list_images
    imgs = manager.list_images(run_dir)
    assert len(imgs) == 2
    img_names = [img.name for img in imgs]
    assert "photo_01.jpg" in img_names
    assert "photo_02.png" in img_names


def test_extract_run_zip_slip_protection(tmp_path: Path):
    """Verify Zip-Slip path traversal vulnerability attempt raises PermissionError."""
    # Craft malicious zip with entry attempting to escape directory
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr("../../evil_script.sh", b"malicious content")
    malicious_zip = tmp_path / "malicious.zip"
    malicious_zip.write_bytes(bio.getvalue())

    manager = DownloadManager(
        client=MockDroneClientForDownload(b""),
        target_dir=tmp_path / "target",
    )

    with pytest.raises(PermissionError, match="Zip-Slip attempt"):
        manager.extract_run(malicious_zip)

    # Ensure file was not extracted outside target directory
    assert not (tmp_path / "evil_script.sh").exists()
