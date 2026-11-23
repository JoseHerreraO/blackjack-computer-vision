"""Show the card detector working on any source: detections, segmentation mask and rectified cards.

Use it to tune the detector on real footage of our mat and lighting before wiring it into the live
assistant. Green outlines are accepted cards (the dot marks their top-left corner), orange ones
are clusters (blobs that are not a single card, usually overlapping cards).

Examples::

    python scripts/preview_detection.py --source 0
    python scripts/preview_detection.py --source data/samples/round_01.mp4 \
        --camera data/calibration/webcam.json
    python scripts/preview_detection.py --source data/samples/photos/ --segmentation hsv

Keys: m = toggle mask view, s = save snapshot (and the rectified cards), space = pause / resume,
n = next frame while paused (photos start paused), q / Esc = quit.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from blackjack_vision.calibration.camera_model import CameraModel
from blackjack_vision.calibration.undistort import Undistorter
from blackjack_vision.capture.sources import ImageSequenceSource, open_source, write_image
from blackjack_vision.detection_tracking.card_detector import (
    VIEWS,
    CardDetector,
    CardDetectorConfig,
    Segmentation,
    config_for_view,
    default_view,
    draw_detections,
)
from blackjack_vision.vision_classic.rectify import CARD_SIZE, rectify_card

WINDOW = "card detection"
PREVIEW_MAX_WIDTH = 1600
STRIP_CARDS = 6

logger = logging.getLogger("preview_detection")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", required=True, help="video, image folder, camera index or URL")
    parser.add_argument(
        "--view",
        choices=VIEWS,
        help="glasses (table seen at an angle) or top (camera above the table); "
        "glasses by default for --source bridge, top otherwise",
    )
    parser.add_argument("--camera", type=Path, help="camera model JSON to undistort frames first")
    parser.add_argument(
        "--segmentation",
        choices=[method.value for method in Segmentation],
        default=Segmentation.WHITENESS.value,
    )
    parser.add_argument("--min-threshold", type=int, default=CardDetectorConfig.min_threshold)
    parser.add_argument(
        "--min-area",
        type=float,
        default=CardDetectorConfig.min_area_fraction,
        help="smallest card as a fraction of the image area",
    )
    parser.add_argument(
        "--snapshots", type=Path, default=Path("outputs/detection"), help="snapshot folder"
    )
    return parser.parse_args(argv)


def card_strip(image: np.ndarray, corners_list: list[np.ndarray], height: int) -> np.ndarray:
    """Rectified cards side by side, scaled to ``height`` (empty slots stay black)."""
    width = round(height * CARD_SIZE[0] / CARD_SIZE[1])
    strip = np.zeros((height, width * STRIP_CARDS, 3), np.uint8)
    for slot, corners in enumerate(corners_list[:STRIP_CARDS]):
        card = cv2.resize(rectify_card(image, corners), (width, height))
        strip[:, slot * width : (slot + 1) * width] = card
        cv2.rectangle(strip, (slot * width, 0), ((slot + 1) * width - 1, height - 1), (90,) * 3)
    return strip


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    detector = CardDetector(
        config_for_view(
            args.view or default_view(args.source),
            segmentation=args.segmentation,
            min_threshold=args.min_threshold,
            min_area_fraction=args.min_area,
        )
    )
    camera = CameraModel.load(args.camera) if args.camera else None
    undistorters: dict[tuple[int, int], Undistorter] = {}

    show_mask, snapshot = False, 0
    with open_source(args.source) as source:
        paused = isinstance(source, ImageSequenceSource)
        frame = source.read()
        while frame is not None:
            image = frame.image
            if camera is not None:
                if frame.size not in undistorters:
                    model = camera if frame.size == camera.image_size else camera.scaled(frame.size)
                    undistorters[frame.size] = Undistorter(model)
                image = undistorters[frame.size](image)

            start = time.perf_counter()
            result = detector.detect(image)
            elapsed_ms = 1000.0 * (time.perf_counter() - start)

            view = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR) if show_mask else image
            canvas = draw_detections(view, result)
            label = (
                f"{len(result.cards)} cards  {len(result.clusters)} clusters  {elapsed_ms:.1f} ms"
            )
            cv2.putText(canvas, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            corners = [card.corners for card in result.cards]
            strip = card_strip(image, corners, height=canvas.shape[0] // 3)
            strip = cv2.resize(strip, (canvas.shape[1], strip.shape[0]))
            canvas = np.vstack([canvas, strip])
            if canvas.shape[1] > PREVIEW_MAX_WIDTH:
                scale = PREVIEW_MAX_WIDTH / canvas.shape[1]
                canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW, canvas)

            pressed = cv2.waitKey(0 if paused else 1) & 0xFF
            if pressed in (ord("q"), 27):
                break
            if pressed == ord("m"):
                show_mask = not show_mask
            elif pressed == ord(" "):
                paused = not paused
            elif pressed == ord("s"):
                folder = args.snapshots / f"snapshot_{snapshot:03d}"
                write_image(folder / "preview.png", canvas)
                for index, card_corners in enumerate(corners):
                    write_image(folder / f"card_{index}.png", rectify_card(image, card_corners))
                snapshot += 1
                print(f"Saved {folder}")
            if not paused or pressed == ord("n"):
                frame = source.read()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
