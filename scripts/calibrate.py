"""Calibrate a camera from views of a chessboard and save the model as JSON (Tema 2).

The source can be a video, a folder of photos, a glob pattern or a live camera/stream. Use the
same resolution as the live pipeline: the intrinsics are only valid for that resolution.

Examples::

    # Video recorded through the glasses stream (or with the webcam):
    python scripts/calibrate.py --source data/calibration/glasses_high.mp4 --pattern 9x6 \
        --square-size 25 --name glasses_high

    # Folder of photos, saving annotated corners and undistorted samples for the report:
    python scripts/calibrate.py --source data/calibration/webcam/ --pattern 9x6 \
        --square-size 25 --name webcam --debug-dir outputs/calibration/webcam

    # Live webcam: move the board around until enough views are collected (q or Esc to stop):
    python scripts/calibrate.py --source 0 --pattern 9x6 --square-size 25 --name webcam
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from blackjack_vision.calibration.chessboard import (
    CalibrationResult,
    ChessboardSpec,
    FrameCallback,
    ViewCollection,
    calibrate,
    collect_views,
    draw_corners,
)
from blackjack_vision.calibration.undistort import Undistorter
from blackjack_vision.capture.sources import Frame, ImageSequenceSource, open_source, write_image

WINDOW = "calibration (q / Esc to finish)"
PREVIEW_MAX_HEIGHT = 900
DEFAULT_VIDEO_STRIDE = 5
DEFAULT_MIN_SHIFT = 15.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source", required=True, help="video, image folder, glob pattern, camera index or URL"
    )
    parser.add_argument(
        "--pattern", default="9x6", help="inner corners as COLSxROWS (default: 9x6)"
    )
    parser.add_argument(
        "--square-size", type=float, default=25.0, help="square side in mm (default: 25)"
    )
    parser.add_argument("--name", default="camera", help="camera name, e.g. glasses_high, webcam")
    parser.add_argument(
        "--out", type=Path, help="output JSON (default: data/calibration/<name>.json)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        help=f"process 1 frame out of N (default: {DEFAULT_VIDEO_STRIDE} for videos, 1 otherwise)",
    )
    parser.add_argument(
        "--min-shift",
        type=float,
        help=f"min mean corner motion in px for a new view (default: {DEFAULT_MIN_SHIFT} for "
        "videos and cameras, 0 for photos)",
    )
    parser.add_argument("--max-views", type=int, default=40, help="stop after N views (40)")
    parser.add_argument("--rational", action="store_true", help="8-coefficient rational model")
    parser.add_argument("--fix-k3", action="store_true", help="keep k3 = 0 (narrow lenses)")
    parser.add_argument(
        "--keep-outliers", action="store_true", help="do not discard views with abnormal error"
    )
    parser.add_argument("--preview", action="store_true", help="show detections while collecting")
    parser.add_argument(
        "--debug-dir", type=Path, help="save annotated views, undistorted samples and errors"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    spec = ChessboardSpec.parse(args.pattern, args.square_size)
    out_path = args.out or Path("data/calibration") / f"{args.name}.json"

    with open_source(args.source) as source:
        photos = isinstance(source, ImageSequenceSource)
        recorded_video = not photos and not source.is_live
        stride = args.stride or (DEFAULT_VIDEO_STRIDE if recorded_video else 1)
        min_shift = (
            args.min_shift if args.min_shift is not None else (0.0 if photos else DEFAULT_MIN_SHIFT)
        )
        preview = args.preview or source.is_live
        views = collect_views(
            source,
            spec,
            stride=stride,
            min_shift=min_shift,
            max_views=args.max_views,
            keep_images=args.debug_dir is not None,
            on_frame=_preview_callback(spec, args.max_views) if preview else None,
        )
    if preview:
        cv2.destroyAllWindows()

    print(
        f"Board found in {views.frames_with_board}/{views.frames_processed} processed frames; "
        f"{len(views)} distinct views kept"
    )
    if len(views) < 3 or views.image_size is None:
        print(
            "Not enough views. Check --pattern (inner corners, not squares), lighting and "
            "that the whole board is visible.",
            file=sys.stderr,
        )
        return 1

    result = calibrate(
        views.image_points,
        spec,
        views.image_size,
        view_names=views.names,
        name=args.name,
        rational_model=args.rational,
        fix_k3=args.fix_k3,
        reject_outliers=not args.keep_outliers,
    )
    print(result.report())
    result.camera.save(out_path)
    print(f"Camera model saved to {out_path}")

    if args.debug_dir is not None:
        _save_debug(args.debug_dir, views, result, spec)
        print(f"Debug images saved to {args.debug_dir}")
    return 0


def _preview_callback(spec: ChessboardSpec, max_views: int) -> FrameCallback:
    kept = 0

    def on_frame(frame: Frame, corners: np.ndarray | None, accepted: bool) -> bool:
        nonlocal kept
        kept += accepted
        canvas = frame.image if corners is None else draw_corners(frame.image, corners, spec)
        if accepted:
            colour = (0, 255, 0)  # new view kept
        elif corners is not None:
            colour = (0, 200, 255)  # board found but too similar to a kept view
        else:
            colour = (0, 0, 255)  # no board
        cv2.putText(
            canvas, f"views {kept}/{max_views}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2
        )
        height = canvas.shape[0]
        if height > PREVIEW_MAX_HEIGHT:
            scale = PREVIEW_MAX_HEIGHT / height
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW, canvas)
        return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)

    return on_frame


def _save_debug(
    directory: Path, views: ViewCollection, result: CalibrationResult, spec: ChessboardSpec
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    errors = dict(zip(result.view_names, result.per_view_errors, strict=True))
    with (directory / "per_view_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["view", "rms_px", "used"])
        for name in views.names:
            used = name in errors
            writer.writerow([name, f"{errors[name]:.4f}" if used else "", int(used)])

    undistorter = Undistorter(result.camera, alpha=0.0)
    for index, (name, image, corners) in enumerate(
        zip(views.names, views.images, views.image_points, strict=True)
    ):
        stem = Path(name).stem
        write_image(directory / f"corners_{stem}.jpg", draw_corners(image, corners, spec))
        if index < 3:
            side_by_side = np.hstack([image, undistorter(image)])
            write_image(directory / f"undistorted_{stem}.jpg", side_by_side)


if __name__ == "__main__":
    sys.exit(main())
