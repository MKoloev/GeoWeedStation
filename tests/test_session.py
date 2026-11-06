"""Tests for ProcessingSession persistence and autosave."""

from __future__ import annotations

from pathlib import Path

from geo_weed_station.core.session import ProcessingSession
from geo_weed_station.schemas.session import FileEntry, ProcessingStatus


def _entries(n: int = 3):
    return [FileEntry(path=f"/tmp/img_{i}.jpg", size_bytes=i) for i in range(n)]


def test_save_load_roundtrip(tmp_path: Path):
    s = ProcessingSession(_entries(), source_kind="folder", source_path="/tmp", run_inference=True)
    s.update_entry(0, ProcessingStatus.DONE, 4)
    f = tmp_path / "sub" / "session.json"
    s.save(f)
    assert f.exists() and not (tmp_path / "sub" / "session.json.tmp").exists()

    loaded = ProcessingSession.load(f)
    assert loaded.source_kind == "folder"
    assert loaded.run_inference is True
    assert loaded.entries[0].status == ProcessingStatus.DONE
    assert loaded.entries[0].detections_count == 4
    assert loaded.entries[1].status == ProcessingStatus.PENDING


def test_load_resets_processing_to_pending(tmp_path: Path):
    s = ProcessingSession(_entries())
    s.update_entry(1, ProcessingStatus.PROCESSING)
    s.save(tmp_path / "s.json")
    loaded = ProcessingSession.load(tmp_path / "s.json")
    assert loaded.entries[1].status == ProcessingStatus.PENDING
    assert loaded.pending_indices() == [0, 1, 2]


def test_autosave_every_n(tmp_path: Path):
    f = tmp_path / "auto.json"
    s = ProcessingSession(_entries(4))
    s.enable_autosave(f, every=2)
    s.update_entry(0, ProcessingStatus.DONE)
    assert not f.exists()
    s.update_entry(1, ProcessingStatus.ERROR)
    assert f.exists()
    assert ProcessingSession.load(f).entries[1].status == ProcessingStatus.ERROR


def test_flush_writes_file(tmp_path: Path):
    f = tmp_path / "flush.json"
    s = ProcessingSession(_entries())
    s.enable_autosave(f, every=100)
    s.update_entry(0, ProcessingStatus.DONE)
    s.flush()
    assert ProcessingSession.load(f).entries[0].status == ProcessingStatus.DONE
