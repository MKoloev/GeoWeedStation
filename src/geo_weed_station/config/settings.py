"""Application settings module using Pydantic v2 BaseSettings."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings


class NetworkSettings(BaseModel):
    """Network connection and protocol settings for drone onboard communication."""

    host: str = "192.168.4.1"
    port: int = 8000
    request_timeout_s: float = 2.0  # Must be <= 2.0s per NFR
    poll_rate_hz: float = 1.0  # Allowed range: 0.5..2.0 Hz
    reconnect_backoff_initial_s: float = 1.0
    reconnect_backoff_max_s: float = 10.0
    reconnect_backoff_factor: float = 2.0
    download_chunk_bytes: int = 65536

    @field_validator("request_timeout_s")
    @classmethod
    def validate_request_timeout(cls, v: float) -> float:
        if v <= 0.0 or v > 2.0:
            raise ValueError(f"request_timeout_s must be in range (0.0, 2.0], got {v}")
        return v

    @field_validator("poll_rate_hz")
    @classmethod
    def validate_poll_rate(cls, v: float) -> float:
        if not (0.5 <= v <= 2.0):
            raise ValueError(f"poll_rate_hz must be in range [0.5, 2.0], got {v}")
        return v

    @field_validator("reconnect_backoff_initial_s")
    @classmethod
    def validate_initial_backoff(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("reconnect_backoff_initial_s must be > 0")
        return v

    @field_validator("reconnect_backoff_max_s")
    @classmethod
    def validate_max_backoff(cls, v: float, info: ValidationInfo) -> float:
        initial = info.data.get("reconnect_backoff_initial_s", 1.0)
        if v < initial:
            raise ValueError(
                f"reconnect_backoff_max_s ({v}) must be >= reconnect_backoff_initial_s ({initial})"
            )
        return v

    @field_validator("download_chunk_bytes")
    @classmethod
    def validate_chunk_bytes(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("download_chunk_bytes must be > 0")
        return v


class SshSettings(BaseModel):
    """SSH connection and remote launch settings for Skyris onboard computer."""

    host: str = "10.42.0.1"
    port: int = 22
    username: str = "orangepi"
    password: str = "orangepi"
    connect_timeout_s: float = 10.0
    remote_dir: str = "/home/orangepi/tsu/CollectOnboardImage"
    launch_command: str = (
        'bash -l -c "cd /home/orangepi/tsu/CollectOnboardImage && python3 main.py '
        '--input-dir data_image/06_geosaero_corner_25_wgs84_10pct_aruc_ex '
        '--model models/yolox_m_rk3588_int8.rknn '
        '--waypoints waypoints/zigzag_skyris.json '
        '--geotag-mode drone_telemetry '
        '--altitude 2.0 '
        '--speed 0.5 '
        '--hover-delay 1.0 '
        '--nav-timeout 20.0 '
        '--conf-thresh 0.35"'
    )


class StreamSettings(BaseModel):
    """HTTP web_video_server stream and snapshot polling settings."""

    host: str = "10.42.0.1"
    port: int = 8080
    topic: str = "/yolo/debug"
    poll_rate_hz: float = 3.0
    request_timeout_s: float = 1.5

    @field_validator("poll_rate_hz")
    @classmethod
    def validate_poll_rate(cls, v: float) -> float:
        if not (0.5 <= v <= 10.0):
            raise ValueError(f"poll_rate_hz must be in range [0.5, 10.0], got {v}")
        return v


class DetectorSettings(BaseModel):
    """ONNX neural network detector parameters."""

    model_path: Path = Path("models/weed_detection_yolo.onnx")
    classes_path: Path = Path("models/classes.json")
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.45
    input_size: tuple[int, int] = (640, 640)
    video_frame_skip: int = 3  # Detect on every N-th frame

    @field_validator("confidence_threshold")
    @classmethod
    def validate_conf_thresh(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence_threshold must be in range [0.0, 1.0], got {v}")
        return v

    @field_validator("nms_iou_threshold")
    @classmethod
    def validate_iou_thresh(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"nms_iou_threshold must be in range [0.0, 1.0], got {v}")
        return v

    @field_validator("video_frame_skip")
    @classmethod
    def validate_frame_skip(cls, v: int) -> int:
        if v < 1:
            raise ValueError("video_frame_skip must be >= 1")
        return v


class MapSettings(BaseModel):
    """Digital mapping, heatmap and coordinate system settings."""

    heatmap_cell_m: float = 0.5
    critical_density_threshold: float = 5.0
    max_local_extent_m: float = 3.0
    wgs84_origin: tuple[float, float] | None = None  # Reference point (lat, lon)

    @field_validator("heatmap_cell_m")
    @classmethod
    def validate_heatmap_cell(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"heatmap_cell_m must be > 0, got {v}")
        return v

    @field_validator("critical_density_threshold")
    @classmethod
    def validate_density_thresh(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("critical_density_threshold must be > 0")
        return v

    @field_validator("max_local_extent_m")
    @classmethod
    def validate_max_extent(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("max_local_extent_m must be > 0")
        return v

    @field_validator("wgs84_origin")
    @classmethod
    def validate_origin(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        if v is not None:
            lat, lon = v
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(f"wgs84_origin latitude must be in [-90, 90], got {lat}")
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(f"wgs84_origin longitude must be in [-180, 180], got {lon}")
        return v


class AppSettings(BaseSettings):
    """Root application configuration."""

    network: NetworkSettings = Field(default_factory=NetworkSettings)
    ssh: SshSettings = Field(default_factory=SshSettings)
    stream: StreamSettings = Field(default_factory=StreamSettings)
    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    map: MapSettings = Field(default_factory=MapSettings)
    workdir: Path = Path("workdir")

    @classmethod
    def default_config_path(cls) -> Path:
        """Return the default configuration path bundled with the package."""
        return Path(__file__).parent / "default_config.json"

    @classmethod
    def load(cls, path: Path | None = None) -> AppSettings:
        """Load configuration from a JSON file.
        
        If path is None, loads from default_config.json bundled in the package.
        """
        target_path = path if path is not None else cls.default_config_path()
        if not target_path.exists():
            if path is None:
                return cls()
            raise FileNotFoundError(f"Configuration file not found: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def save(self, path: Path) -> None:
        """Save configuration to the specified JSON file path."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))


_global_settings: AppSettings | None = None


def get_settings(path: Path | None = None, reload: bool = False) -> AppSettings:
    """Get the active application settings singleton, optionally loading from a path or reloading."""
    global _global_settings
    if _global_settings is None or reload or path is not None:
        _global_settings = AppSettings.load(path)
    return _global_settings
