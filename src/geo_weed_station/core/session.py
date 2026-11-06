"""Offline processing session with JSON persistence and periodic autosave (no Qt dependencies)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import List, Optional

from geo_weed_station.schemas.session import FileEntry, ProcessingStatus, SessionState


class ProcessingSession:
    """Mutable processing session holding file entries and their statuses."""

    def __init__(
        self,
        entries: Optional[List[FileEntry]] = None,
        source_kind: Optional[str] = None,
        source_path: Optional[str] = None,
        run_inference: bool = False,
    ) -> None:
        self.entries: List[FileEntry] = list(entries) if entries else []
        self.source_kind = source_kind
        self.source_path = source_path
        self.run_inference = run_inference
        self._lock = threading.Lock()
        self._autosave_path: Optional[Path] = None
        self._autosave_every: int = 0
        self._updates_since_save: int = 0

    # ------------------------------------------------------------------ state
    def to_state(self) -> SessionState:
        """Return a deep-copied serializable snapshot."""
        with self._lock:
            return SessionState(
                source_kind=self.source_kind,
                source_path=self.source_path,
                run_inference=self.run_inference,
                entries=[e.model_copy() for e in self.entries],
            )

    @classmethod
    def from_state(cls, state: SessionState) -> "ProcessingSession":
        """Restore session from a snapshot."""
        return cls(
            entries=list(state.entries),
            source_kind=state.source_kind,
            source_path=state.source_path,
            run_inference=state.run_inference,
        )

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        """Atomically write the session to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(self.to_state().model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "ProcessingSession":
        """Load session from JSON; any in-flight PROCESSING entries return to PENDING."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        state = SessionState.model_validate(data)
        session = cls.from_state(state)
        for entry in session.entries:
            if entry.status == ProcessingStatus.PROCESSING:
                entry.status = ProcessingStatus.PENDING
        return session

    # --------------------------------------------------------------- autosave
    def enable_autosave(self, path: Path, every: int = 1) -> None:
        """Save automatically after each `every` entry updates."""
        self._autosave_path = Path(path)
        self._autosave_every = max(1, int(every))
        self._updates_since_save = 0

    def update_entry(
        self,
        index: int,
        status: ProcessingStatus,
        detections_count: Optional[int] = None,
    ) -> None:
        """Update status of an entry and trigger autosave if configured."""
        with self._lock:
            entry = self.entries[index]
            entry.status = status
            if detections_count is not None:
                entry.detections_count = detections_count
            self._updates_since_save += 1
            need_save = (
                self._autosave_path is not None
                and self._updates_since_save >= self._autosave_every
            )
            if need_save:
                self._updates_since_save = 0
        if need_save and self._autosave_path is not None:
            self.save(self._autosave_path)

    def flush(self) -> None:
        """Force-save to the autosave path (if configured)."""
        if self._autosave_path is not None:
            self._updates_since_save = 0
            self.save(self._autosave_path)

    def pending_indices(self) -> List[int]:
        """Indices of entries still awaiting processing."""
        return [i for i, e in enumerate(self.entries) if e.status == ProcessingStatus.PENDING]
