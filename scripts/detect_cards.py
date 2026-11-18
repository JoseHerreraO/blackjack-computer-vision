"""Detect and track playing cards live or on recordings, with the FPS drawn on the video.

Each card is shown with its tracker ID and label. Grey boxes are tracks that are not confirmed
yet, green ones are confirmed, and the terminal prints every newly confirmed card once.

Examples::

    # Live from the glasses (the iOS app connects to this PC on port 5050)
    python scripts/detect_cards.py --source bridge

    # Frames saved with mobile/ios/tools/pc_receiver.py, a video or a webcam
    python scripts/detect_cards.py --source data/samples/20261118-201500 --fps 3
    python scripts/detect_cards.py --source data/samples/round_01.mp4 --save outputs/round_01.mp4
    python scripts/detect_cards.py --source 0

    # Undistort first with the glasses calibration
    python scripts/detect_cards.py --source bridge --camera data/calibration/glasses_high.json

Keys: space = pause / resume, s = save snapshot, r = reset tracker, q / Esc = quit.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from blackjack_vision.calibration.camera_model import CameraModel
from blackjack_vision.calibration.undistort import Undistorter
from blackjack_vision.capture.sources import open_source, write_image
from blackjack_vision.detection_tracking.detector import DEFAULT_WEIGHTS, CardDetector, Detection
from blackjack_vision.detection_tracking.tracker import CardTracker, TrackerUpdate

WINDOW = "cards"
PREVIEW_MAX_HEIGHT = 900

logger = logging.getLogger("detect_cards")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", default="bridge", help="bridge[:port], camera, video or images")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLO model file")
    parser.add_argument("--conf", type=float, default=0.4, help="minimum detection score (0.4)")
    parser.add_argument("--tiles", default="1x2", help="tile grid COLSxROWS, 1x1 = whole frame")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size (640)")
    parser.add_argument("--device", default=None, help="cpu, cuda:0... (auto)")
    parser.add_argument("--no-merge", action="store_true", help="show raw corner boxes")
    parser.add_argument("--camera", type=Path, help="camera model JSON to undistort frames")
    parser.add_argument("--fps", type=float, help="nominal rate of an image folder")
    parser.add_argument("--save", type=Path, help="write the annotated video to this file")
    parser.add_argument("--snapshots", type=Path, default=Path("outputs/detect_cards"))
    parser.add_argument("--no-window", action="store_true", help="run without a preview window")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def parse_tiles(text: str) -> tuple[int, int]:
    columns, _, rows = text.lower().partition("x")
    if not (columns.isdigit() and rows.isdigit()):
        raise argparse.ArgumentTypeError(f"--tiles must look like 1x2, got {text!r}")
    return int(columns), int(rows)


def draw(
    image: np.ndarray,
    update: TrackerUpdate,
    raw: list[Detection],
    fps: float,
    show_raw: bool,
) -> np.ndarray:
    view = image.copy()
    if show_raw:
        for detection in raw:
            x1, y1, x2, y2 = (int(v) for v in detection.box)
            cv2.rectangle(view, (x1, y1), (x2, y2), (255, 160, 0), 1)
    for track in update.tracks:
        if track.missed > 0:
            continue
        x1, y1, x2, y2 = (int(v) for v in track.box)
        color = (0, 200, 0) if track.confirmed else (160, 160, 160)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        text = f"#{track.track_id} {track.label} {track.label_confidence:.0%}"
        _label(view, text, (x1, max(18, y1 - 6)), color)
    counted = len(update.visible)
    _label(view, f"{fps:.1f} FPS  {counted} cards", (10, 28), (0, 255, 255), scale=0.8)
    return view


def _label(image, text, origin, color, scale=0.55) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )

    detector = CardDetector(
        args.weights,
        confidence=args.conf,
        image_size=args.imgsz,
        tiles=parse_tiles(args.tiles),
        merge=not args.no_merge,
        device=args.device,
    )
    tracker = CardTracker()
    undistorter: Undistorter | None = None
    camera = CameraModel.load(args.camera) if args.camera else None

    source = open_source(args.source, fps=args.fps)
    if source.is_live and str(args.source).startswith("bridge"):
        logger.info("Waiting for the iPhone. In the app type this PC's IP and port, then Connect")

    writer: cv2.VideoWriter | None = None
    frame_times: deque[float] = deque(maxlen=30)
    paused = False
    image = None
    with source:
        while True:
            if not paused or image is None:
                frame = source.read()
                if frame is None:
                    break
                image = frame.image
                if camera is not None:
                    if undistorter is None:
                        size = (image.shape[1], image.shape[0])
                        model = camera if camera.image_size == size else camera.scaled(size)
                        undistorter = Undistorter(model)
                    image = undistorter(image)

                raw = detector.detect(image)
                update = tracker.update(raw)
                for track in update.confirmed:
                    logger.info("New card #%d: %s", track.track_id, track.label)

                frame_times.append(time.perf_counter())
                fps = (
                    (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                    if len(frame_times) > 1
                    else 0.0
                )
                view = draw(image, update, raw, fps, show_raw=args.no_merge)

                if args.save is not None:
                    if writer is None:
                        args.save.parent.mkdir(parents=True, exist_ok=True)
                        rate = source.fps or 15.0
                        size = (view.shape[1], view.shape[0])
                        writer = cv2.VideoWriter(
                            str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), rate, size
                        )
                    writer.write(view)

            if args.no_window:
                continue
            scale = min(1.0, PREVIEW_MAX_HEIGHT / view.shape[0])
            preview = cv2.resize(view, None, fx=scale, fy=scale) if scale < 1 else view
            cv2.imshow(WINDOW, preview)
            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("r"):
                tracker.reset()
                logger.info("Tracker reset")
            elif key == ord("s"):
                path = args.snapshots / f"snapshot_{int(time.time() * 1000)}.png"
                write_image(path, view)
                logger.info("Saved %s", path)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
