"""Mission archive download manager with chunked streaming and Zip-Slip protection."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional
import zipfile

from geo_weed_station.network.drone_client import DroneClient
from geo_weed_station.network.exceptions import DownloadCancelled

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class DownloadManager:
    """Manages streaming downloads of run archives, unpacking, and local file discovery."""

    def __init__(
        self,
        client: DroneClient,
        target_dir: Path,
        chunk_bytes: int = 65536,
    ) -> None:
        self.client = client
        self.target_dir = Path(target_dir)
        self.chunk_bytes = max(1, int(chunk_bytes))
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def download_run(
        self,
        run_id: str,
        progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
        cancel_flag: Optional[Callable[[], bool]] = None,
    ) -> Path:
        """Download mission ZIP archive with chunked streaming and atomic renaming.
        
        Args:
            run_id: Target mission run identifier.
            progress_cb: Optional callback(downloaded_bytes, total_bytes).
            cancel_flag: Optional callable returning True to abort download.
            
        Returns:
            Path to downloaded .zip archive.
            
        Raises:
            DownloadCancelled: If cancel_flag returned True.
            DroneNetworkError: On network or API error.
        """
        final_path = self.target_dir / f"{run_id}.zip"
        part_path = self.target_dir / f"{run_id}.zip.part"

        if part_path.exists():
            part_path.unlink()

        downloaded_bytes = 0
        try:
            chunks_iter = self.client.stream_archive(run_id, chunk_bytes=self.chunk_bytes)
            with open(part_path, "wb") as f:
                for chunk in chunks_iter:
                    if cancel_flag is not None and cancel_flag():
                        raise DownloadCancelled(f"Download of run '{run_id}' was cancelled by user.")

                    if chunk:
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        if progress_cb is not None:
                            progress_cb(downloaded_bytes, None)

            # Atomic rename from .part to final .zip
            part_path.replace(final_path)
            return final_path

        except Exception:
            # Clean up partial temporary file on failure or cancellation
            if part_path.exists():
                part_path.unlink()
            raise

    def extract_run(self, archive: Path) -> Path:
        """Extract a downloaded ZIP archive with strict Zip-Slip path traversal protection.
        
        Args:
            archive: Path to the .zip archive.
            
        Returns:
            Path to the directory containing extracted files.
            
        Raises:
            PermissionError: If malicious path traversal (Zip-Slip) is detected.
            FileNotFoundError: If archive does not exist.
        """
        archive_path = Path(archive)
        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive file not found: {archive_path}")

        run_dir = self.target_dir / archive_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        resolved_base = run_dir.resolve()

        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                # Check for Zip-Slip vulnerability
                target_dest = (run_dir / member.filename).resolve()
                if not target_dest.is_relative_to(resolved_base):
                    raise PermissionError(
                        f"Malicious archive member detected (Zip-Slip attempt): {member.filename}"
                    )
                zf.extract(member, run_dir)

        return run_dir

    def list_images(self, run_dir: Path) -> List[Path]:
        """Scan directory recursively for supported aerial imagery formats."""
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            return []

        images: List[Path] = []
        for p in run_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                images.append(p)

        images.sort(key=lambda x: x.name)
        return images
