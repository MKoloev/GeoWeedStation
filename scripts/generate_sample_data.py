#!/usr/bin/env python3
"""CLI utility to generate synthetic test datasets for GeoWeedStation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.helpers.dataset_factory import make_dataset, make_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic sample dataset (geotagged images, corrupted files, and video)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "workdir" / "sample_data",
        help="Target folder for generated data (default: workdir/sample_data)",
    )
    parser.add_argument("--num-valid", type=int, default=5, help="Number of valid geotagged images")
    parser.add_argument("--num-no-xmp", type=int, default=2, help="Number of images without spatial XMP")
    parser.add_argument("--num-broken", type=int, default=1, help="Number of corrupt files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    images_dir = out_dir / "images"
    video_dir = out_dir / "video"

    print(f"Generating synthetic dataset in: {out_dir}")
    images_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    make_dataset(
        images_dir,
        n_valid=args.num_valid,
        n_no_xmp=args.num_no_xmp,
        n_broken=args.num_broken,
    )
    print(f"Created {args.num_valid} geotagged, {args.num_no_xmp} unreferenced, and {args.num_broken} corrupt images in {images_dir}")

    video_path = video_dir / "flight_sample.mp4"
    make_video(video_path, frames=45, fps=15)
    print(f"Created sample video: {video_path}")

    print("Generation complete! You can open these files in GeoWeedStation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
