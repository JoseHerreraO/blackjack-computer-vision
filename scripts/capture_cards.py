"""Build the card dataset: detect, rectify and save cards from a camera or a video, one label at a time.

Put ONE card on the mat, check that it is outlined in green and press space (or turn on
auto-save with ``a``). Vary its position, rotation, distance and the lighting between shots, and
also move the camera: the recogniser has to cope with the same variety at the table. Images go to
``<out>/<CODE>/NNNN.png``, the layout expected by ``scripts/train_recognizer.py``.

Examples::

    python scripts/capture_cards.py --source 0 --deck                 # walk through all 52 cards
    python scripts/capture_cards.py --source 0 --label QH --auto-every 10
    python scripts/capture_cards.py --source data/samples/deck.mp4 --label 7C \
        --camera data/calibration/webcam.json

Keys: space = save the card now, a = toggle auto-save, n / b = next / previous card (--deck),
q / Esc = quit.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from blackjack_vision.blackjack.cards import FULL_DECK, Card
from blackjack_vision.calibration.camera_model import CameraModel
from blackjack_vision.calibration.undistort import Undistorter
from blackjack_vision.capture.sources import open_source, write_image
from blackjack_vision.detection_tracking.card_detector import (
    VIEWS,
    CardDetector,
    config_for_view,
    default_view,
    draw_detections,
)
from blackjack_vision.vision_classic.rectify import rectify_card

WINDOW = "capture cards"
PREVIEW_MAX_WIDTH = 1400

logger = logging.getLogger("capture_cards")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", required=True, help="camera index, video file, bridge...")
    parser.add_argument(
        "--view",
        choices=VIEWS,
        help="glasses (table seen at an angle) or top (camera above the table); "
        "glasses by default for --source bridge, top otherwise",
    )
    labels = parser.add_mutually_exclusive_group(required=True)
    labels.add_argument("--label", help="card being captured, e.g. AS, TH, 10D, QC")
    labels.add_argument("--deck", action="store_true", help="walk through the 52 cards in order")
    parser.add_argument("--start", default="AS", help="first card in --deck mode")
    parser.add_argument("--out", type=Path, default=Path("data/cards"), help="dataset folder")
    parser.add_argument("--camera", type=Path, help="camera model JSON to undistort frames first")
    parser.add_argument(
        "--auto-every", type=int, default=15, help="frames between auto-saved shots"
    )
    return parser.parse_args(argv)


def next_index(folder: Path) -> int:
    """First free ``NNNN`` number in a label folder."""
    numbers = [int(path.stem) for path in folder.glob("*.png") if path.stem.isdigit()]
    return max(numbers, default=-1) + 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.deck:
        deck = list(FULL_DECK)
        position = deck.index(Card.parse(args.start))
    else:
        deck, position = [Card.parse(args.label)], 0

    detector = CardDetector(config_for_view(args.view or default_view(args.source)))
    camera = CameraModel.load(args.camera) if args.camera else None
    undistorters: dict[tuple[int, int], Undistorter] = {}
    auto, since_last = False, 0

    with open_source(args.source) as source:
        for frame in source:
            image = frame.image
            if camera is not None:
                if frame.size not in undistorters:
                    model = camera if frame.size == camera.image_size else camera.scaled(frame.size)
                    undistorters[frame.size] = Undistorter(model)
                image = undistorters[frame.size](image)

            card = deck[position]
            folder = args.out / card.code
            result = detector.detect(image)
            single = len(result.cards) == 1 and not result.clusters
            canvas = draw_detections(image, result)
            saved = len(list(folder.glob("*.png"))) if folder.is_dir() else 0
            status = f"{card.code}  saved {saved}  {'AUTO' if auto else 'manual'}"
            if not single:
                status += "  (show exactly one card)"
            colour = (0, 255, 0) if single else (0, 0, 255)
            cv2.putText(canvas, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
            if single:
                thumb = rectify_card(image, result.cards[0].corners)
                height = min(canvas.shape[0], thumb.shape[0])
                canvas[:height, -thumb.shape[1] :] = thumb[:height]
            if canvas.shape[1] > PREVIEW_MAX_WIDTH:
                scale = PREVIEW_MAX_WIDTH / canvas.shape[1]
                canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW, canvas)

            pressed = cv2.waitKey(1) & 0xFF
            since_last += 1
            save = pressed == ord(" ") or (auto and since_last >= args.auto_every)
            if save and single:
                path = folder / f"{next_index(folder):04d}.png"
                write_image(path, rectify_card(image, result.cards[0].corners))
                since_last = 0
                logger.info("Saved %s", path)
            elif pressed == ord(" "):
                logger.warning(
                    "Not saved: %d cards and %d clusters in view",
                    len(result.cards),
                    len(result.clusters),
                )
            if pressed in (ord("q"), 27):
                break
            if pressed == ord("a"):
                auto = not auto
            elif pressed == ord("n") and position < len(deck) - 1:
                position, auto = position + 1, False
            elif pressed == ord("b") and position > 0:
                position, auto = position - 1, False
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
