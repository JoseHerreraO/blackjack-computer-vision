"""Detect, track and recognise cards in real time, with the FPS drawn on the video.

Classical detector -> tracker -> one recognition per card on the fused sharpest crops (see
``src/blackjack_vision/detection_tracking/pipeline.py``). A card shows as ``?`` until it is
recognised, then with its code and similarity. Every newly recognised card is printed once.

Examples::

    python scripts/track_cards.py --source bridge                       # live from the glasses
    python scripts/track_cards.py --source data/samples/<folder> --fps 15
    python scripts/track_cards.py --source data/samples/<folder> --no-window --save out.mp4

The recogniser is the hybrid one by default (``vision_dl/hybrid_recognizer.py``: the rank from the
classical recogniser, the suit from the embeddings, 97.3 % of cards right on the glasses
footage). ``--recognizer classic`` is the classical one alone (bag of words + SVM for the rank, NCC templates
for the suit; train it with ``scripts/train_recognizer.py``), which is the one the course syllabus
covers. ``--recognizer embedding`` uses the learned embedding recogniser instead, for comparison
(needs ``scripts/download_card_recognizer.py``). ``--no-recognition`` only detects and tracks.
Keys: space = pause / resume, r = reset, s = snapshot, q / Esc = quit.
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

from blackjack_vision.capture.sources import open_source, write_image
from blackjack_vision.detection_tracking.card_detector import CardDetector, CardDetectorConfig
from blackjack_vision.detection_tracking.pipeline import CardPipeline, PipelineResult
from blackjack_vision.vision_dl.hybrid_recognizer import RECOGNIZERS, load_recognizer

WINDOW = "cards"
PREVIEW_MAX_HEIGHT = 760

logger = logging.getLogger("track_cards")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", default="bridge", help="bridge[:port], camera, video or images")
    parser.add_argument("--fps", type=float, help="nominal rate of an image folder")
    parser.add_argument(
        "--recognizer",
        choices=RECOGNIZERS,
        default="hybrid",
        help="hybrid (default): classical rank + embedding suit; classic: bag of words + SVM and "
        "NCC templates (syllabus); embedding: learned model",
    )
    parser.add_argument(
        "--model", type=Path, default=Path("data/models/recognizer.joblib"), help="classic model"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="classic: weakest one-vs-one SVM margin to name a card (default 0.4)",
    )
    parser.add_argument(
        "--embedding-model", type=Path, default=Path("models/irfan-playing-card-detector")
    )
    parser.add_argument(
        "--threshold", type=float, default=0.70, help="embedding: min similarity (0.70)"
    )
    parser.add_argument("--no-recognition", action="store_true", help="detection and tracking only")
    parser.add_argument("--min-crops", type=int, default=5, help="frames before recognising (5)")
    parser.add_argument("--fuse", type=int, default=3, help="sharpest crops averaged (3)")
    parser.add_argument(
        "--view",
        choices=("glasses", "top"),
        default="glasses",
        help="glasses: table seen at an angle (starts the foreshortening at 0.5), top: from above",
    )
    parser.add_argument("--save", type=Path, help="write the annotated video to this file")
    parser.add_argument("--snapshots", type=Path, default=Path("outputs/track_cards"))
    parser.add_argument("--no-window", action="store_true", help="run without a preview window")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def draw(image: np.ndarray, result: PipelineResult, fps: float) -> np.ndarray:
    view = image.copy()
    thickness = max(1, image.shape[1] // 360)
    for card in result.cards:
        colour = (0, 200, 0) if card.recognised else (0, 200, 255)
        cv2.polylines(view, [card.corners.astype(np.int32)], True, colour, 2 * thickness)
        text = f"#{card.track_id} {card.label}"
        if card.recognised:
            text += f" {card.similarity:.2f}"
        x, y = card.corners.min(axis=0).astype(int)
        _label(view, text, (x, max(14, y - 6)), colour, 0.45 * thickness, thickness)
    known = sum(card.recognised for card in result.cards)
    _label(
        view,
        f"{fps:.1f} FPS  {len(result.cards)} cards  {known} recognised",
        (10, 30 * thickness),
        (0, 255, 255),
        0.7 * thickness,
        thickness,
    )
    return view


def _label(image, text, origin, colour, scale, thickness) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, origin, font, scale, (0, 0, 0), 3 * thickness, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, scale, colour, thickness, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    recognizer = None
    if not args.no_recognition:
        recognizer = load_recognizer(
            args.recognizer,
            model=args.model,
            embedding_model=args.embedding_model,
            threshold=args.threshold,
            min_confidence=args.min_confidence,
        )
    config = CardDetectorConfig.for_glasses() if args.view == "glasses" else CardDetectorConfig()
    pipeline = CardPipeline(
        recognizer,
        detector=CardDetector(config),
        min_crops=args.min_crops,
        fuse_top=args.fuse,
    )

    writer: cv2.VideoWriter | None = None
    frame_times: deque[float] = deque(maxlen=30)
    paused = False
    view = None
    with open_source(args.source, fps=args.fps) as source:
        while True:
            if not paused or view is None:
                frame = source.read()
                if frame is None:
                    break
                result = pipeline.process(frame.image)
                for card in result.new_cards:
                    logger.info("Card #%d: %s (%.2f)", card.track_id, card.label, card.similarity)
                frame_times.append(time.perf_counter())
                fps = (
                    (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                    if len(frame_times) > 1
                    else 0.0
                )
                view = draw(frame.image, result, fps)
                if args.save is not None:
                    if writer is None:
                        args.save.parent.mkdir(parents=True, exist_ok=True)
                        size = (view.shape[1], view.shape[0])
                        writer = cv2.VideoWriter(
                            str(args.save),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            source.fps or 15.0,
                            size,
                        )
                    writer.write(view)
            if args.no_window:
                continue
            scale = min(1.0, PREVIEW_MAX_HEIGHT / view.shape[0])
            cv2.imshow(WINDOW, cv2.resize(view, None, fx=scale, fy=scale) if scale < 1 else view)
            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("r"):
                pipeline.reset()
            elif key == ord("s"):
                path = write_image(args.snapshots / f"snapshot_{int(time.time() * 1000)}.png", view)
                logger.info("Saved %s", path)
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
