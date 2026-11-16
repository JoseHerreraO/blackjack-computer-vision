"""Show a source side by side before and after lens-distortion correction.

Use it to check a calibration visually: straight edges of the table, the mat or a door frame
must look straight on the right-hand side, also near the image borders.

Examples::

    python scripts/preview_undistort.py --camera data/calibration/webcam.json --source 0
    python scripts/preview_undistort.py --camera data/calibration/glasses_high.json \
        --source data/samples/round_01.mp4

Keys: a = toggle alpha 0/1 (crop vs keep every pixel), g = toggle grid, s = save snapshot,
space = pause / resume, n = next frame while paused (photos start paused), q / Esc = quit.
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

WINDOW = "raw | undistorted"
PREVIEW_MAX_WIDTH = 1600
GRID_STEP = 80

logger = logging.getLogger("preview_undistort")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--camera", type=Path, required=True, help="camera model JSON")
    parser.add_argument("--source", required=True, help="video, image folder, camera index or URL")
    parser.add_argument("--alpha", type=float, default=0.0, help="initial alpha in [0, 1]")
    parser.add_argument(
        "--snapshots", type=Path, default=Path("outputs/undistort"), help="snapshot folder"
    )
    return parser.parse_args(argv)


def draw_grid(image: np.ndarray, step: int = GRID_STEP) -> np.ndarray:
    """Overlay a regular grid of straight lines to judge residual distortion."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for x in range(0, width, step):
        cv2.line(canvas, (x, 0), (x, height - 1), (255, 255, 0), 1)
    for y in range(0, height, step):
        cv2.line(canvas, (0, y), (width - 1, y), (255, 255, 0), 1)
    return canvas


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    camera = CameraModel.load(args.camera)
    print(camera.summary())

    alpha, show_grid = args.alpha, False
    undistorters: dict[tuple[tuple[int, int], float], Undistorter] = {}
    snapshot = 0
    with open_source(args.source) as source:
        paused = isinstance(source, ImageSequenceSource)  # photos: step with n
        frame = source.read()
        while frame is not None:
            key = (frame.size, alpha)
            if key not in undistorters:
                model = camera
                if frame.size != camera.image_size:
                    logger.warning(
                        "Frame size %s differs from the calibration %s: rescaling the model",
                        frame.size,
                        camera.image_size,
                    )
                    model = camera.scaled(frame.size)
                undistorters[key] = Undistorter(model, alpha=alpha)
            start = time.perf_counter()
            corrected = undistorters[key](frame.image)
            elapsed_ms = 1000.0 * (time.perf_counter() - start)

            raw = frame.image
            if show_grid:
                raw, corrected = draw_grid(raw), draw_grid(corrected)
            canvas = np.hstack([raw, corrected])
            label = f"alpha={alpha:.0f}  remap {elapsed_ms:.1f} ms"
            cv2.putText(canvas, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if canvas.shape[1] > PREVIEW_MAX_WIDTH:
                scale = PREVIEW_MAX_WIDTH / canvas.shape[1]
                canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW, canvas)

            pressed = cv2.waitKey(0 if paused else 1) & 0xFF
            if pressed in (ord("q"), 27):
                break
            if pressed == ord("a"):
                alpha = 1.0 - alpha
            elif pressed == ord("g"):
                show_grid = not show_grid
            elif pressed == ord(" "):
                paused = not paused
            elif pressed == ord("s"):
                path = write_image(args.snapshots / f"snapshot_{snapshot:03d}.png", canvas)
                snapshot += 1
                print(f"Saved {path}")
            if not paused or pressed == ord("n"):
                frame = source.read()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
