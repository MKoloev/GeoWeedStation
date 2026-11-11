"""Data schemas for comprehensive folder detection and agronomic reports."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from geo_weed_station.schemas.map_models import PrescriptionZone


class FrameReportItem(BaseModel):
    """Per-frame summary row inside a folder detection report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_id: str
    file_name: str
    file_path: str
    size_bytes: int = 0
    status: str = "pending"
    has_xmp: bool = False
    coordinate_system: str | None = None
    camera_x: float | None = None
    camera_y: float | None = None
    altitude_m: float | None = None
    yaw_deg: float | None = None
    detections_count: int = 0
    mean_confidence: float = 0.0
    max_confidence: float = 0.0
    by_class: dict[str, int] = Field(default_factory=dict)
    by_stage: dict[str, int] = Field(default_factory=dict)


class DetectionReportRow(BaseModel):
    """Single detected object record in the report (supports georeferenced and non-georeferenced frames)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    weed_id: str
    source_frame: str
    class_id: int = 0
    class_name: str
    stage_name: str | None = None
    confidence: float
    class_prob: float | None = None
    bbox_x: float = 0.0
    bbox_y: float = 0.0
    bbox_w: float = 0.0
    bbox_h: float = 0.0
    coord_system: str | None = None
    world_x: float | None = None
    world_y: float | None = None
    lat: float | None = None
    lon: float | None = None
    is_unique: bool = True
    duplicate_count: int = 1
    best_frame_id: str | None = None


class FolderDetectionReport(BaseModel):
    """Complete structured report for a folder of processed images."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # 1. Metadata & detector settings
    report_id: str
    created_at: str
    folder_name: str
    folder_path: str
    model_name: str = "unknown"
    classifier_type: str = "none"
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.45

    # 2. File & batch statistics
    total_files: int = 0
    processed_files: int = 0
    files_with_xmp: int = 0
    files_without_xmp: int = 0
    error_files: int = 0
    frames_with_detections: int = 0
    empty_frames: int = 0
    total_size_bytes: int = 0

    # 3. Detection & classification statistics
    total_detections: int = 0
    unique_weeds: int = 0
    deduplication_ratio: float = 1.0
    mean_confidence: float = 0.0
    by_class: dict[str, int] = Field(default_factory=dict)
    by_stage: dict[str, int] = Field(default_factory=dict)
    class_stage_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)

    # 4. Spatial & agronomic summary
    has_georef: bool = False
    coordinate_system: str | None = None
    surveyed_area_m2: float = 0.0
    mean_density_per_m2: float = 0.0
    critical_area_percent: float = 0.0
    prescription_zones_count: int = 0
    zones: list[PrescriptionZone] = Field(default_factory=list)

    # 5. Detailed tables
    frames: list[FrameReportItem] = Field(default_factory=list)
    detections: list[DetectionReportRow] = Field(default_factory=list)
