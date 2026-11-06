"""Tests for BatchImageWorker, XmpIndexTask and FrameQueue (pytest-qt)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QThreadPool

from geo_weed_station.core.detector_base import BaseDetector
from geo_weed_station.core.session import ProcessingSession
from geo_weed_station.core.source_factory import SourceFactory
from geo_weed_station.schemas.entities import BBox, DetectedWeed
from geo_weed_station.schemas.session import ProcessingStatus
from geo_weed_station.workers.base_worker import start_in_thread
from geo_weed_station.workers.batch_image_worker import BatchImageWorker
from geo_weed_station.workers.frame_queue import FrameQueue
from geo_weed_station.workers.xmp_index_task import start_xmp_indexing
from tests.helpers.make_media import make_real_jpeg
from tests.helpers.make_xmp_image import SAMPLE_XMP_LOCAL_ARUCO


class FakeDetector(BaseDetector):
    instances: list[FakeDetector] = []

    def __init__(self, delay: float = 0.0) -> None:
        self._loaded = False
        self.closed = False
        self.delay = delay
        self.threads: list[int] = []
        FakeDetector.instances.append(self)

    def load(self) -> None:
        self._loaded = True

    def detect(self, image: np.ndarray):
        self.threads.append(threading.get_ident())
        if self.delay:
            time.sleep(self.delay)
        return [
            DetectedWeed(
                weed_id="w", class_id=0, class_name="weed", confidence=0.9,
                bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2),
            )
        ]

    def close(self) -> None:
        self._loaded = False
        self.closed = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeDetector.instances.clear()


def _make_batch(tmp_path: Path):
    """10 JPEGs: 2 without XMP (idx 3, 7), 1 corrupted (idx 5)."""
    for i in range(10):
        p = tmp_path / f"img_{i:02d}.jpg"
        if i == 5:
            p.write_bytes(b"corrupted")
        elif i in (3, 7):
            make_real_jpeg(p, None, seed=i)
        else:
            make_real_jpeg(p, SAMPLE_XMP_LOCAL_ARUCO, seed=i)
    return SourceFactory.create("folder", tmp_path)


def _run(worker, entries, run_inference, qtbot, thread_name="batch_t"):
    done = []
    worker.batch_finished.connect(lambda ok, bad: done.append((ok, bad)))
    thread = start_in_thread(worker, thread_name)
    return thread, done


def test_batch_statuses_and_counters(tmp_path: Path, qtbot):
    src = _make_batch(tmp_path)
    entries = src.entries()
    session = ProcessingSession(entries)
    session.enable_autosave(tmp_path / "out" / "s.json", every=1)
    worker = BatchImageWorker(src, FakeDetector, session)

    statuses = {}
    progress = []
    frames = []
    worker.entry_status_changed.connect(lambda i, s: statuses.__setitem__(i, s))
    worker.progress.connect(lambda i, n, t: progress.append((i, n, t)))
    worker.frame_processed.connect(frames.append)
    done = []
    worker.batch_finished.connect(lambda ok, bad: done.append((ok, bad)))

    thread = start_in_thread(worker, "batch_t1")
    try:
        from PySide6.QtCore import QObject, Signal

        class Trigger(QObject):
            go = Signal(list, bool)

        trigger = Trigger()
        trigger.go.connect(worker.process)  # cross-thread -> queued
        trigger.go.emit(entries, True)
        qtbot.waitUntil(lambda: len(done) == 1, timeout=15000)
    finally:
        thread.quit()
        thread.wait(3000)

    assert done[0] == (9, 1)
    assert statuses[5] == ProcessingStatus.ERROR
    assert statuses[3] == ProcessingStatus.NO_METADATA
    assert statuses[7] == ProcessingStatus.NO_METADATA
    assert statuses[0] == ProcessingStatus.DONE
    assert len(frames) == 9
    assert all(len(f.detections) == 1 for f in frames)
    # progress: text and monotonic 0..100%
    assert progress[0][2] == "инференс кадра 1 из 10"
    assert progress[-1][:2] == (10, 10)
    pcts = [100 * i // n for i, n, _ in progress]
    assert pcts == sorted(pcts) and pcts[-1] == 100
    # detector lived in worker thread and was closed
    det = FakeDetector.instances[0]
    assert det.closed and threading.get_ident() not in det.threads
    # session persisted
    saved = ProcessingSession.load(tmp_path / "out" / "s.json")
    assert saved.entries[5].status == ProcessingStatus.ERROR
    assert saved.entries[0].detections_count == 1


def test_frames_are_copies(tmp_path: Path, qtbot):
    make_real_jpeg(tmp_path / "a.jpg", SAMPLE_XMP_LOCAL_ARUCO)
    src = SourceFactory.create("folder", tmp_path)
    entries = src.entries()
    loaded = []
    orig_load = src.load
    src.load = lambda e: (loaded.append(orig_load(e)), loaded[-1])[1]  # type: ignore[method-assign]
    worker = BatchImageWorker(src)
    frames = []
    worker.frame_processed.connect(frames.append)
    worker.process(entries, False)
    assert frames[0].image is not loaded[0].image
    assert np.array_equal(frames[0].image, loaded[0].image)


def test_lazy_onnx_not_created_before_inference(tmp_path: Path):
    src = _make_batch(tmp_path)
    worker = BatchImageWorker(src, FakeDetector)
    assert not worker.detector_created
    assert FakeDetector.instances == []
    worker.process(src.entries()[:2], False)  # no inference requested
    assert FakeDetector.instances == []
    worker.process(src.entries()[:1], True)
    assert len(FakeDetector.instances) == 1


def test_lazy_real_onnx_detector_not_loaded(tmp_path: Path):
    from geo_weed_station.config.settings import AppSettings
    from geo_weed_station.core.onnx_detector import OnnxDetector
    from tests.helpers.make_dummy_onnx import create_dummy_yolo_onnx

    model = create_dummy_yolo_onnx(tmp_path / "m.onnx")
    s = AppSettings.load().detector
    s = s.model_copy(update={"model_path": model})
    created = []

    def factory():
        d = OnnxDetector(s)
        created.append(d)
        return d

    make_real_jpeg(tmp_path / "a.jpg", SAMPLE_XMP_LOCAL_ARUCO, size=(64, 64))
    src = SourceFactory.create("folder", tmp_path)
    worker = BatchImageWorker(src, factory)
    assert created == []  # nothing before first inference
    worker.process(src.entries(), True)
    assert len(created) == 1 and not created[0].is_loaded  # closed in finally


def test_cancel_midway_keeps_partial_results(tmp_path: Path, qtbot):
    src = _make_batch(tmp_path)
    entries = src.entries()
    session = ProcessingSession(entries)
    worker = BatchImageWorker(src, lambda: FakeDetector(delay=0.05), session)
    done = []
    worker.batch_finished.connect(lambda ok, bad: done.append((ok, bad)))
    processed = []

    def on_frame(_):
        processed.append(1)
        if len(processed) == 3:
            worker.cancel()

    worker.frame_processed.connect(on_frame)
    worker.process(entries, True)
    assert done == [(3, 0)]
    pend = session.pending_indices()
    assert len(pend) == 7
    assert FakeDetector.instances[0].closed
    assert all(e.status != ProcessingStatus.PROCESSING for e in session.entries)


def test_inference_failure_marks_error_and_continues(tmp_path: Path):
    for i in range(3):
        make_real_jpeg(tmp_path / f"i{i}.jpg", SAMPLE_XMP_LOCAL_ARUCO, seed=i)
    src = SourceFactory.create("folder", tmp_path)

    class Boom(FakeDetector):
        def detect(self, image):
            raise RuntimeError("boom")

    worker = BatchImageWorker(src, Boom)
    done = []
    errors = []
    worker.batch_finished.connect(lambda ok, bad: done.append((ok, bad)))
    worker.failed.connect(errors.append)
    worker.process(src.entries(), True)
    assert done == [(0, 3)] and len(errors) == 3


def test_xmp_index_pool(tmp_path: Path, qtbot):
    items = []
    for i in range(20):
        p = tmp_path / f"i{i}.jpg"
        make_real_jpeg(p, SAMPLE_XMP_LOCAL_ARUCO if i % 2 == 0 else None, seed=i)
        items.append((i, str(p)))
    results = {}
    finished = []
    pool = QThreadPool()
    job = start_xmp_indexing(items, pool=pool, chunk_size=4)
    job.signals.indexed.connect(lambda i, ok: results.__setitem__(i, ok))
    job.signals.finished.connect(lambda: finished.append(1))
    job.start()
    qtbot.waitUntil(lambda: len(finished) == 1, timeout=5000)
    assert len(results) == 20
    assert all(results[i] == (i % 2 == 0) for i in range(20))
    pool.waitForDone(2000)


def test_xmp_index_cancel(tmp_path: Path, qtbot):
    items = []
    for i in range(8):
        p = tmp_path / f"i{i}.jpg"
        make_real_jpeg(p, SAMPLE_XMP_LOCAL_ARUCO, seed=i)
        items.append((i, str(p)))
    pool = QThreadPool()
    pool.setMaxThreadCount(1)
    job = start_xmp_indexing(items, pool=pool, chunk_size=2)
    results = []
    finished = []
    job.signals.indexed.connect(lambda i, ok: results.append(i))
    job.signals.finished.connect(lambda: finished.append(1))
    job.cancel()
    job.start()
    qtbot.waitUntil(lambda: len(finished) == 1, timeout=3000)
    assert results == []
    pool.waitForDone(2000)


def test_frame_queue_bounded_drop_oldest():
    q = FrameQueue(maxlen=3)
    drops = [q.put(i) for i in range(5)]
    assert drops == [False, False, False, True, True]
    assert len(q) == 3 and q.dropped == 2
    assert [q.get(0), q.get(0), q.get(0)] == [2, 3, 4]
    assert q.get(0) is None


def test_frame_queue_threaded_performance():
    q = FrameQueue(maxlen=8)
    n = 20000
    got = []

    def consumer():
        while True:
            item = q.get(timeout=0.5)
            if item is None or item == -1:
                break
            got.append(item)

    t = threading.Thread(target=consumer)
    t.start()
    start = time.perf_counter()
    for i in range(n):
        q.put(i)
    q.put(-1)
    t.join(5)
    elapsed = time.perf_counter() - start
    assert not t.is_alive()
    assert elapsed < 2.0
    assert len(q) <= 8
    assert got == sorted(got)  # order preserved
    assert len(got) + q.dropped >= n - 8
