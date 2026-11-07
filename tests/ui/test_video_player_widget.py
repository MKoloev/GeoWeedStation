"""Tests for VideoPlayerWidget controls, timeline, skip selector and rendering performance."""

from __future__ import annotations

import time

import numpy as np

from geo_weed_station.schemas.entities import BBox, DetectedWeed
from geo_weed_station.ui.widgets.video_player_widget import VideoPlayerWidget


def _sample_frame(w: int = 1280, h: int = 720) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _sample_detections() -> list[DetectedWeed]:
    return [
        DetectedWeed(
            weed_id="v-1",
            class_id=0,
            class_name="Ambrosia",
            confidence=0.88,
            bbox=BBox(x=0.2, y=0.2, w=0.3, h=0.3),
        )
    ]


def test_video_player_playback_signals(qtbot):
    player = VideoPlayerWidget()
    qtbot.addWidget(player)

    plays = []
    pauses = []
    player.play_requested.connect(lambda: plays.append(True))
    player.pause_requested.connect(lambda: pauses.append(True))

    # Click play
    player.btn_play_pause.click()
    assert plays == [True]
    assert "Пауза" in player.btn_play_pause.text()

    # Click pause
    player.btn_play_pause.click()
    assert pauses == [True]
    assert "Воспроизведение" in player.btn_play_pause.text()


def test_video_player_play_at_end_rewinds(qtbot):
    player = VideoPlayerWidget()
    qtbot.addWidget(player)

    seeks = []
    plays = []
    player.seek_requested.connect(seeks.append)
    player.play_requested.connect(lambda: plays.append(True))

    player.set_position(9, 10)
    player.set_playing(False)

    player.btn_play_pause.click()
    assert seeks == [0]
    assert plays == [True]
    assert "Пауза" in player.btn_play_pause.text()


def test_video_player_timeline_and_seek(qtbot):
    player = VideoPlayerWidget()
    qtbot.addWidget(player)

    seeks = []
    player.seek_requested.connect(seeks.append)

    # Set position programmatically -> does NOT emit seek_requested
    player.set_position(10, 100)
    assert player.label_position.text() == "10 / 100"
    assert player.slider.value() == 10
    assert seeks == []

    # User dragging simulation
    player.slider.sliderPressed.emit()
    player.slider.setValue(45)
    player.slider.sliderReleased.emit()
    assert seeks == [45]


def test_video_player_frame_skip(qtbot):
    player = VideoPlayerWidget()
    qtbot.addWidget(player)

    skips = []
    player.frame_skip_changed.connect(skips.append)

    player.combo_skip.setCurrentIndex(1)  # 2nd option: skip 2
    assert skips == [2]

    player.combo_skip.setCurrentIndex(2)  # skip 5
    assert skips == [2, 5]


def test_video_player_rendering_fps_720p(qtbot):
    """Verify player can render 720p frames at >= 30 FPS without blocking UI."""
    player = VideoPlayerWidget()
    qtbot.addWidget(player)
    player.resize(640, 360)
    player.show()

    frame = _sample_frame(1280, 720)
    detections = _sample_detections()

    num_frames = 30
    t0 = time.perf_counter()
    for _ in range(num_frames):
        player.show_frame(frame, detections)
        player.canvas.repaint()
    duration = time.perf_counter() - t0

    fps = num_frames / max(duration, 1e-6)
    assert fps >= 30.0, f"Expected >= 30 FPS for 720p playback, got {fps:.1f} FPS"


def test_video_player_grab(qtbot):
    player = VideoPlayerWidget()
    qtbot.addWidget(player)
    player.resize(400, 300)
    player.show()

    player.show_frame(_sample_frame(640, 480), _sample_detections())
    pix = player.grab()
    assert not pix.isNull()
    assert pix.width() >= 400
    assert pix.height() > 0
