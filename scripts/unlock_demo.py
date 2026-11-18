"""Live demo of the security system: show four aces in the secret suit order to unlock.

The secret is never drawn on screen, only how many aces of it have been shown. It is read from
``--secret`` or, better, from the ``BJ_SECRET`` environment variable (suit codes, e.g. ``SHDC``
for spades, hearts, diamonds, clubs), so it does not end up in the shell history.

The suit templates come from a trained recogniser (``scripts/train_recognizer.py``). With
``--use-rank`` the recogniser also decides which cards are aces; otherwise the single large pip
in the middle of the card does.

Examples::

    set BJ_SECRET=SHDC
    python scripts/unlock_demo.py --source 0 --model data/models/recognizer.joblib
    python scripts/unlock_demo.py --source bridge --model data/models/recognizer.joblib \
        --camera data/calibration/glasses_high.json --use-rank

Keys: r = start over (lock again), q / Esc = quit.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2

from blackjack_vision.calibration.camera_model import CameraModel
from blackjack_vision.calibration.undistort import Undistorter
from blackjack_vision.capture.sources import open_source
from blackjack_vision.detection_tracking.card_detector import CardDetector, draw_detections
from blackjack_vision.security.aces import (
    AceReader,
    DecoderEvent,
    SecuritySystem,
    SequenceDecoder,
    parse_secret,
)
from blackjack_vision.vision_classic.recognition import CardRecognizer
from blackjack_vision.vision_classic.rectify import rectify_card

WINDOW = "unlock"
PREVIEW_MAX_HEIGHT = 900

logger = logging.getLogger("unlock_demo")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", required=True, help="camera index, video, bridge[:port]...")
    parser.add_argument("--model", type=Path, required=True, help="trained recogniser (joblib)")
    parser.add_argument("--secret", default=os.environ.get("BJ_SECRET"), help="e.g. SHDC")
    parser.add_argument("--camera", type=Path, help="camera model JSON to undistort frames")
    parser.add_argument("--use-rank", action="store_true", help="aces decided by the recogniser")
    parser.add_argument("--min-frames", type=int, default=5, help="frames to accept an ace")
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds between aces")
    return parser.parse_args(argv)


def status_line(decoder: SequenceDecoder) -> str:
    """Progress without revealing the secret, e.g. ``LOCKED  ##..``."""
    if decoder.unlocked:
        return "UNLOCKED"
    done = "#" * decoder.progress + "." * (len(decoder.secret) - decoder.progress)
    return f"LOCKED  {done}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.secret:
        print("Give the secret with --secret or the BJ_SECRET variable", file=sys.stderr)
        return 2
    recognizer = CardRecognizer.load(args.model)
    system = SecuritySystem(
        AceReader(recognizer.suit_templates, recognizer.config.index_region),
        SequenceDecoder(
            parse_secret(args.secret), min_frames=args.min_frames, timeout=args.timeout
        ),
    )
    detector = CardDetector()
    camera = CameraModel.load(args.camera) if args.camera else None
    undistorter: Undistorter | None = None

    with open_source(args.source) as source:
        for frame in source:
            image = frame.image
            if camera is not None:
                if undistorter is None:
                    model = camera if frame.size == camera.image_size else camera.scaled(frame.size)
                    undistorter = Undistorter(model)
                image = undistorter(image)

            result = detector.detect(image)
            cards = [rectify_card(image, card.corners) for card in result.cards]
            ranks = None
            if args.use_rank and cards:
                ranks = [prediction.rank for prediction in recognizer.predict_many(cards)]
            event, readings = system.update(cards, frame.timestamp, ranks)
            if event is not DecoderEvent.NONE:
                logger.info("%s (%s)", event.value, status_line(system.decoder))

            canvas = draw_detections(image, result)
            for card, reading in zip(result.cards, readings, strict=True):
                if reading.is_ace:
                    x, y = (int(v) for v in card.center)
                    cv2.putText(
                        canvas,
                        "A",
                        (x - 12, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 215, 255),
                        3,
                        cv2.LINE_AA,
                    )
            colour = (0, 220, 0) if system.unlocked else (0, 0, 255)
            cv2.putText(
                canvas,
                status_line(system.decoder),
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                colour,
                3,
                cv2.LINE_AA,
            )
            if canvas.shape[0] > PREVIEW_MAX_HEIGHT:
                scale = PREVIEW_MAX_HEIGHT / canvas.shape[0]
                canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                system.decoder.reset()
                logger.info("Reset: locked again")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
