"""Tests for VideoWorker using a generated mp4 (pytest-qt)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PySide6.QtCore import Q_ARG, QMetaObject, Qt

from geo_weed_station.workers.base_worker import start_in_thread
from geo_weed_station.workers.video_worker import VideoWorker
from tests.helpers.make_media import make_mp4
from tests.test_batch_image_worker import FakeDetector


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeDetector.instances.clear()


def _connect(worker):
    rec = {"frames": [], "pos": [], "ended": [], "failed": []}
    worker.frame_ready.connect(lambda img, dets: (rec["frames"].append((img, dets)), worker.frame_consumed()))
    worker.position_changed.connect(lambda i, n: rec["pos"].append((i, n)))
    worker.playback_ended.connect(lambda: rec["ended"].append(1))
    worker.failed.connect(rec["failed"].append)
    return rec


def test_open_shows_first_frame(tmp_path: Path, qtbot):
    vid = make_mp4(tmp_path / "v.mp4", n_frames=10)
    w = VideoWorker()
    rec = _connect(w)
    w.open(str(vid))
    assert rec["pos"][0] == (0, 10)
    img, dets = rec["frames"][0]
    assert img.shape == (48, 64, 3) and dets == []
    w.close_resources()


def test_open_invalid_file_fails(tmp_path: Path):
    w = VideoWorker()
    rec = _connect(w)
    w.open(str(tmp_path / "missing.mp4"))
    assert len(rec["failed"]) == 1


def test_play_to_end(tmp_path: Path, qtbot):
    vid = make_mp4(tmp_path / "v.mp4", n_frames=10, fps=100)
    w = VideoWorker()
    rec = _connect(w)
    w.open(str(vid))
    w.play()
    qtbot.waitUntil(lambda: len(rec["ended"]) == 1, timeout=5000)
    assert [p[0] for p in rec["pos"]] == list(range(10))
    assert not w.is_playing
    w.close_resources()


def test_replay_after_reaching_end(tmp_path: Path, qtbot):
    vid = make_mp4(tmp_path / "v.mp4", n_frames=10, fps=100)
    w = VideoWorker()
    rec = _connect(w)
    w.open(str(vid))
    w.play()
    qtbot.waitUntil(lambda: len(rec["ended"]) == 1, timeout=5000)
    assert not w.is_playing
    assert rec["pos"][-1] == (9, 10)

    # Replay after EOF: should rewind to 0 and play through all frames again
    w.play()
    qtbot.waitUntil(lambda: len(rec["ended"]) == 2, timeout=5000)
    second_pass = [p[0] for p in rec["pos"][10:]]
    assert second_pass == list(range(10))
    w.close_resources()


def test_pause_and_seek(tmp_path: Path, qtbot):
    vid = make_mp4(tmp_path / "v.mp4", n_frames=10, fps=10)
    w = VideoWorker()
    rec = _connect(w)
    w.open(str(vid))
    w.play()
    qtbot.waitUntil(lambda: len(rec["pos"]) >= 3, timeout=3000)
    w.pause()
    assert not w.is_playing
    n = len(rec["pos"])
    qtbot.wait(300)
    assert len(rec["pos"]) == n  # paused: no new frames
    w.seek(7)
    assert rec["pos"][-1] == (7, 10)
    w.seek(999)  # clamped
    assert rec["pos"][-1] == (9, 10)
    w.seek(-5)
    assert rec["pos"][-1] == (0, 10)
    w.close_resources()


def test_frame_skip_runs_inference_every_nth_and_holds_boxes(tmp_path: Path, qtbot):
    vid = make_mp4(tmp_path / "v.mp4", n_frames=10, fps=100)
    w = VideoWorker(FakeDetector)
    rec = _connect(w)
    w.set_frame_skip(2)  # inference on frames 0, 3, 6, 9
    w.open(str(vid))
    w.play()
    qtbot.waitUntil(lambda: len(rec["ended"]) == 1, timeout=5000)
    assert len(FakeDetector.instances[0].threads) == 4
    # boxes held on intermediate frames
    assert all(len(d) == 1 for _, d in rec["frames"])
    w.close_resources()


def test_lazy_detector(tmp_path: Path):
    vid = make_mp4(tmp_path / "v.mp4")
    w = VideoWorker(FakeDetector)
    assert not w.detector_created and FakeDetector.instances == []
    w.open(str(vid))  # first frame triggers first inference
    assert w.detector_created
    w.close_resources()
    assert FakeDetector.instances[0].closed and not w.detector_created


def test_no_detector_factory_means_no_inference(tmp_path: Path):
    vid = make_mp4(tmp_path / "v.mp4")
    w = VideoWorker()
    w.open(str(vid))
    assert not w.detector_created
    w.close_resources()


def test_resources_released(tmp_path: Path):
    vid = make_mp4(tmp_path / "v.mp4")
    w = VideoWorker(FakeDetector)
    w.open(str(vid))
    real = w._cap

    class Proxy:
        releases = 0

        def __getattr__(self, name):
            return getattr(real, name)

        def release(self):
            Proxy.releases += 1
            real.release()

    w._cap = Proxy()
    w.close_resources()
    assert Proxy.releases == 1
    assert not real.isOpened()
    assert w._cap is None
    w.close_resources()  # idempotent


def test_reopen_releases_previous_capture(tmp_path: Path):
    a = make_mp4(tmp_path / "a.mp4")
    b = make_mp4(tmp_path / "b.mp4")
    w = VideoWorker()
    w.open(str(a))
    first = w._cap
    w.open(str(b))
    assert w._cap is not first and not first.isOpened()
    w.close_resources()


def test_frames_are_copies(tmp_path: Path):
    vid = make_mp4(tmp_path / "v.mp4")
    w = VideoWorker()
    rec = _connect(w)
    w.open(str(vid))
    w.seek(0)
    a, b = rec["frames"][0][0], rec["frames"][1][0]
    assert a is not b and a.flags["OWNDATA"]
    w.close_resources()


def test_slow_consumer_drops_frames_not_memory(tmp_path: Path, qtbot):
    vid = make_mp4(tmp_path / "v.mp4", n_frames=10, fps=100)
    w = VideoWorker(max_inflight=2)
    frames = []
    ended = []
    w.frame_ready.connect(lambda i, d: frames.append(i))  # never acknowledges
    w.playback_ended.connect(lambda: ended.append(1))
    w.open(str(vid))
    w.play()
    qtbot.waitUntil(lambda: len(ended) == 1, timeout=5000)
    assert len(frames) == 2
    assert w.dropped_frames == 9 - 1  # 10 frames, 2 delivered
    w.close_resources()


def test_runs_in_worker_thread(tmp_path: Path, qtbot):
    vid = make_mp4(tmp_path / "v.mp4", n_frames=6, fps=100)
    w = VideoWorker(FakeDetector)
    rec = _connect(w)
    thread = start_in_thread(w, "video_t")
    try:
        QMetaObject.invokeMethod(w, "open", Qt.QueuedConnection, Q_ARG(str, str(vid)))
        QMetaObject.invokeMethod(w, "play", Qt.QueuedConnection)
        qtbot.waitUntil(lambda: len(rec["ended"]) == 1, timeout=5000)
        assert threading.get_ident() not in FakeDetector.instances[0].threads
        QMetaObject.invokeMethod(w, "close_resources", Qt.BlockingQueuedConnection)
    finally:
        thread.quit()
        thread.wait(3000)
    assert FakeDetector.instances[0].closed
