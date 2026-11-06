"""Offline processing session schemas: file entries, statuses and persisted session state."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ProcessingStatus(str, Enum):
    """Processing status of a single file entry in a batch."""

    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"
    NO_METADATA = "no_metadata"


class FileEntry(BaseModel):
    """Single file (image / video / run frame) tracked by an offline session."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    path: str
    size_bytes: int = 0
    status: ProcessingStatus = ProcessingStatus.PENDING
    has_xmp: bool = False
    detections_count: int | None = None


class SessionState(BaseModel):
    """Serializable snapshot of an offline processing session."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    source_kind: str | None = None
    source_path: str | None = None
    run_inference: bool = False
    entries: list[FileEntry] = Field(default_factory=list)
