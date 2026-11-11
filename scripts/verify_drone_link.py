#!/usr/bin/env python3
"""Live end-to-end verification of GeoWeedStation integration with Skyris UAV.

Exercises the production OnlineController -> SshMissionWorker + LiveStreamWorker
-> SFTP summary.json download -> SummaryDetections validation -> MapEngine pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from geo_weed_station.config.settings import AppSettings, MapSettings, NetworkSettings
from geo_weed_station.workers.live_stream_worker import DEFAULT_STREAM_PORT, DEFAULT_STREAM_TOPIC
from geo_weed_station.workers.ssh_mission_worker import (
    DEFAULT_SSH_COMMAND,
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_PASSWORD,
    DEFAULT_SSH_PORT,
    DEFAULT_SSH_USER,
)


def build_bench_command(nav_timeout: float = 3.0, hover_delay: float = 0.5) -> str:
    """Build bench-test launch command (propellers detached) per drone_onboard_guide.md."""
    return (
        'bash -l -c "cd /home/orangepi/tsu/CollectOnboardImage && python3 main.py '
        '--input-dir data_image/06_geosaero_corner_25_wgs84_10pct_aruc_ex '
        '--model models/yolox_m_rk3588_int8.rknn '
        '--waypoints waypoints/zigzag_skyris.json '
        '--geotag-mode drone_telemetry '
        '--altitude 2.0 '
        '--speed 0.5 '
        f'--hover-delay {hover_delay} '
        f'--nav-timeout {nav_timeout} '
        '--gsd 0.00022 '
        '--iou-thresh 0.25 '
        '--conf-thresh 0.6 '
        '--containment-thresh 0.7 '
        '--core-mask 0_1_2"'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify live GeoWeedStation <-> Skyris drone SSH, video stream, and summary.json link."
    )
    parser.add_argument("--host", default=DEFAULT_SSH_HOST, help="Drone IP address")
    parser.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT, help="SSH port")
    parser.add_argument("--stream-port", type=int, default=DEFAULT_STREAM_PORT, help="web_video_server port")
    parser.add_argument("--username", default=DEFAULT_SSH_USER, help="SSH username")
    parser.add_argument("--password", default=DEFAULT_SSH_PASSWORD, help="SSH password")
    parser.add_argument("--topic", default=DEFAULT_STREAM_TOPIC, help="ROS debug video topic")
    parser.add_argument(
        "--bench",
        action="store_true",
        help=(
            "Use short bench-test command "
            "(--nav-timeout 3.0, --gsd 0.00022, --iou-thresh 0.25, "
            "--conf-thresh 0.6, --containment-thresh 0.7, --core-mask 0_1_2)"
        ),
    )
    parser.add_argument("--nav-timeout", type=float, default=3.0, help="Navigation timeout for --bench")
    parser.add_argument("--hover-delay", type=float, default=0.5, help="Hover delay for --bench")
    parser.add_argument("--command", default=None, help="Explicit remote command override")
    parser.add_argument("--timeout", type=float, default=180.0, help="Overall wait timeout in seconds")
    parser.add_argument("--workdir", type=Path, default=None, help="Optional workdir (defaults to temp dir)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from geo_weed_station.controllers.map_controller import MapController
    from geo_weed_station.controllers.online_controller import OnlineController
    from geo_weed_station.core.map_engine.projector import NadirProjector
    from geo_weed_station.core.map_engine.weed_map import MapEngine
    from geo_weed_station.ui.main_window import MainWindow
    from geo_weed_station.ui.widgets.map_canvas import MapCanvas

    app = QApplication.instance() or QApplication(sys.argv)

    tmp_ctx = tempfile.TemporaryDirectory(prefix="gws_live_") if args.workdir is None else None
    try:
        workdir = Path(tmp_ctx.name) if tmp_ctx is not None else args.workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        settings = AppSettings(
            network=NetworkSettings(host=args.host, port=8000, request_timeout_s=2.0),
            map=MapSettings(heatmap_cell_m=1.0),
            workdir=workdir,
        )
        settings.ssh.host = args.host
        settings.ssh.port = args.ssh_port
        settings.ssh.username = args.username
        settings.ssh.password = args.password
        settings.stream.host = args.host
        settings.stream.port = args.stream_port
        settings.stream.topic = args.topic

        engine = MapEngine(projector=NadirProjector(), settings=settings.map)
        map_ctrl = MapController(engine=engine)
        window = MainWindow(qsettings=QSettings(str(workdir / "ui.ini"), QSettings.IniFormat))
        canvas = MapCanvas()
        ctrl = OnlineController(
            window=window,
            map_ctrl=map_ctrl,
            settings=settings,
            map_canvas=canvas,
        )

        # Select matching topic in UI widget before stream starts
        idx = ctrl.mission_control.live_stream_widget.combo_topic.findData(args.topic)
        if idx >= 0:
            ctrl.mission_control.live_stream_widget.combo_topic.setCurrentIndex(idx)

        command = args.command or (
            build_bench_command(nav_timeout=args.nav_timeout, hover_delay=args.hover_delay)
            if args.bench
            else DEFAULT_SSH_COMMAND
        )

        metrics: dict[str, object] = {
            "frames": 0,
            "frame_sizes": [],
            "stream_statuses": [],
            "wp_progress": [],
            "mission_finished": None,
            "summary_downloaded": None,
            "map_locked": None,
            "map_updated": 0,
            "logs": [],
        }

        ctrl.mission_control.live_stream_widget.frame_ready_hook = None
        ctrl.start_stream(host=args.host, port=args.stream_port)
        assert ctrl._stream_worker is not None

        def on_frame(pixmap) -> None:
            metrics["frames"] = int(metrics["frames"]) + 1
            sz = (pixmap.width(), pixmap.height())
            sizes = metrics["frame_sizes"]
            assert isinstance(sizes, list)
            if sz not in sizes:
                sizes.append(sz)
            print(f"[STREAM] frame #{metrics['frames']}: {sz[0]}x{sz[1]}", flush=True)

        def on_stream_status(key: str, text: str) -> None:
            statuses = metrics["stream_statuses"]
            assert isinstance(statuses, list)
            entry = (key, text)
            if not statuses or statuses[-1] != entry:
                statuses.append(entry)
                print(f"[STREAM STATUS] {key}: {text}", flush=True)

        def on_map_updated() -> None:
            metrics["map_updated"] = int(metrics["map_updated"]) + 1

        def on_map_locked(locked: bool, reason: str) -> None:
            metrics["map_locked"] = (locked, reason)
            print(f"[MAP LOCK] locked={locked} reason={reason}", flush=True)

        ctrl._stream_worker.frame_ready.connect(on_frame)
        ctrl._stream_worker.status_changed.connect(on_stream_status)
        map_ctrl.map_updated.connect(on_map_updated)
        map_ctrl.map_locked_changed.connect(on_map_locked)

        ctrl.start_ssh_mission(
            {
                "host": args.host,
                "port": args.ssh_port,
                "username": args.username,
                "password": args.password,
                "command": command,
            }
        )
        worker = ctrl._ssh_worker
        if worker is None:
            print("[ERROR] OnlineController failed to create SshMissionWorker", file=sys.stderr)
            return 2

        ssh_done = {"flag": False}

        def on_log(line: str) -> None:
            logs = metrics["logs"]
            assert isinstance(logs, list)
            logs.append(line)
            print(f"[SSH] {line}", flush=True)

        def on_progress(curr: int, total: int) -> None:
            wps = metrics["wp_progress"]
            assert isinstance(wps, list)
            wps.append((curr, total))
            print(f"[PROGRESS] WP {curr}/{total}", flush=True)

        def on_finished(ok: bool, msg: str) -> None:
            metrics["mission_finished"] = (ok, msg)
            print(f"[MISSION FINISHED] ok={ok} msg={msg}", flush=True)

        def on_summary(path_str: str) -> None:
            metrics["summary_downloaded"] = path_str
            print(f"[SFTP SUMMARY] {path_str}", flush=True)

        def on_thread_finished() -> None:
            ssh_done["flag"] = True

        worker.log_received.connect(on_log)
        worker.progress_updated.connect(on_progress)
        worker.mission_finished.connect(on_finished)
        worker.summary_downloaded.connect(on_summary)
        worker.finished.connect(on_thread_finished)

        deadline = time.time() + args.timeout
        while time.time() < deadline:
            app.processEvents()
            if ssh_done["flag"]:
                # Drain any queued signals (e.g. summary_downloaded -> _on_ssh_summary_downloaded)
                for _ in range(10):
                    app.processEvents()
                    time.sleep(0.05)
                break
            time.sleep(0.05)

        ctrl.disconnect()
        window.shutdown_threads(timeout_ms=2000)

        from geo_weed_station.schemas.onboard_api import SummaryDetections

        print("\n=== LIVE DRONE VERIFICATION REPORT ===")
        print(f"SSH mission finished: {metrics['mission_finished']}")
        print(f"Waypoints reported:   {len(metrics['wp_progress'])} -> {metrics['wp_progress']}")
        print(f"Stream statuses:      {metrics['stream_statuses']}")
        print(f"Stream frames:        {metrics['frames']} (sizes={metrics['frame_sizes']})")
        print(f"Summary downloaded:   {metrics['summary_downloaded']}")

        summary_obj = None
        summary_valid = False
        total_detections = 0
        max_alt = 0.0
        if metrics["summary_downloaded"] is not None:
            summary_path = Path(str(metrics["summary_downloaded"]))
            if summary_path.is_file():
                def _reject_non_finite(token: str) -> None:
                    raise ValueError(f"Non-standard JSON token in summary.json: {token}")

                raw = json.loads(summary_path.read_text(encoding="utf-8"), parse_constant=_reject_non_finite)
                summary_obj = SummaryDetections.model_validate(raw)
                summary_valid = len(summary_obj.items) > 0
                total_detections = sum(len(item.detections) for item in summary_obj.items)
                if summary_obj.items:
                    max_alt = max(float(item.altitude_m) for item in summary_obj.items)
                manifest_ok = (summary_path.parent / "manifest.json").is_file()
                print(
                    f"Summary validated:    {summary_valid} "
                    f"(run_id={summary_obj.run_id}, items={len(summary_obj.items)}, "
                    f"detections={total_detections}, max_alt={max_alt:.2f}m, manifest={manifest_ok})"
                )

        map_points = len(map_ctrl.engine.weed_map.points)
        synthetic_map_points = 0
        if summary_obj is not None and map_points == 0 and max_alt <= 0.0 and args.bench:
            patched_items = [item.model_copy(update={"altitude_m": 2.0}) for item in summary_obj.items]
            patched_summary = summary_obj.model_copy(update={"items": patched_items})
            sim_engine = MapEngine(projector=NadirProjector(), settings=settings.map)
            sim_engine.ingest_summary(patched_summary)
            synthetic_map_points = len(sim_engine.weed_map.points)
            print(f"Bench fallback map:   {synthetic_map_points} weed points (with nominal alt=2.0m)")

        print(f"Map updated events:   {metrics['map_updated']}")
        print(f"Map lock state:       {metrics['map_locked']}")
        print(f"Map weed points:      {map_points}")
        print(f"MissionControl errs:  {ctrl.mission_control.lbl_errors.text()!r}")

        ok_mission = isinstance(metrics["mission_finished"], tuple) and metrics["mission_finished"][0] is True
        ok_waypoints = len(metrics["wp_progress"]) > 0
        ok_stream = int(metrics["frames"]) > 0 and any(k == "LIVE" for k, _ in metrics["stream_statuses"])
        ok_summary = summary_valid and total_detections >= 0
        if max_alt > 0.0:
            ok_map = map_points > 0
        elif args.bench:
            locked_state = metrics["map_locked"]
            ok_locked = isinstance(locked_state, tuple) and locked_state[0] is True
            ok_map = ok_locked and (synthetic_map_points > 0 if total_detections > 0 else True)
        else:
            ok_map = map_points > 0

        if ok_mission and ok_waypoints and ok_stream and ok_summary and ok_map:
            print("RESULT: PASS (SSH + WP progress + Stream + SFTP summary + Map pipeline verified)")
            return 0

        print(
            f"RESULT: FAIL (mission={ok_mission}, wp={ok_waypoints}, stream={ok_stream}, "
            f"summary={ok_summary}, map={ok_map})",
            file=sys.stderr,
        )
        return 1
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
