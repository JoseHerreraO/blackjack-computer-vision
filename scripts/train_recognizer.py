"""Train the card recogniser on a folder of rectified cards and report its accuracy.

The dataset is ``<data>/<CODE>/*.png`` (see ``scripts/capture_cards.py``). A stratified part of
every card is held out to measure the accuracy (rank, suit and full card) and the rank confusion
matrix; then, unless ``--no-refit``, the model is retrained on all the images and saved.

Examples::

    python scripts/train_recognizer.py --data data/cards --out data/models/recognizer.joblib
    python scripts/train_recognizer.py --data data/cards --vocabulary 500 --levels 1 2 4 \
        --errors-dir outputs/recognition_errors
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from blackjack_vision.blackjack.cards import Rank
from blackjack_vision.capture.sources import write_image
from blackjack_vision.vision_classic.features import FeatureConfig, SamplingMode
from blackjack_vision.vision_classic.recognition import (
    CardRecognizer,
    RecognizerConfig,
    load_dataset,
)

logger = logging.getLogger("train_recognizer")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    defaults = RecognizerConfig()
    parser.add_argument("--data", type=Path, default=Path("data/cards"), help="dataset folder")
    parser.add_argument(
        "--out", type=Path, default=Path("data/models/recognizer.joblib"), help="model file"
    )
    parser.add_argument("--vocabulary", type=int, default=defaults.vocabulary_size)
    parser.add_argument("--levels", type=int, nargs="+", default=list(defaults.pyramid_levels))
    parser.add_argument("--sampling", choices=[m.value for m in SamplingMode], default="dense")
    parser.add_argument("--step", type=int, default=defaults.features.step)
    parser.add_argument("--c", type=float, default=defaults.svm_c, help="rank SVM C")
    parser.add_argument("--test-size", type=float, default=0.25, help="held-out fraction")
    parser.add_argument(
        "--test-data",
        type=Path,
        help="separate test folder (e.g. another recording): train on all of --data, test on it",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-refit", action="store_true", help="save the train-split model")
    parser.add_argument("--errors-dir", type=Path, help="save misrecognised test images here")
    return parser.parse_args(argv)


def confusion_table(true: list[str], predicted: list[str], labels: list[str]) -> str:
    """Rank confusion matrix as text (rows = true rank, columns = predicted rank)."""
    counts = Counter(zip(true, predicted, strict=True))
    lines = ["true\\pred " + " ".join(f"{label:>3}" for label in labels)]
    for row in labels:
        cells = " ".join(f"{counts[(row, column)] or '.':>3}" for column in labels)
        lines.append(f"{row:>9} {cells}")
    return "\n".join(lines)


def evaluate(recognizer, images, truth, paths, errors_dir: Path | None) -> None:
    """Print accuracy, the rank confusion matrix and what ``recognise`` would leave unknown."""
    start = time.perf_counter()
    predictions = recognizer.predict_many(images)
    per_card_ms = 1000 * (time.perf_counter() - start) / max(1, len(images))
    rank_ok = np.array([p.rank == c.rank for p, c in zip(predictions, truth, strict=True)])
    suit_ok = np.array([p.suit == c.suit for p, c in zip(predictions, truth, strict=True)])
    card_ok = rank_ok & suit_ok
    print(
        f"Test ({len(images)} images, {per_card_ms:.1f} ms/card): "
        f"rank {rank_ok.mean():.1%}  suit {suit_ok.mean():.1%}  card {card_ok.mean():.1%}"
    )
    known = np.array(
        [
            p.card is not None
            and p.rank_confidence >= recognizer.min_rank_confidence
            and p.suit_margin >= recognizer.min_suit_margin
            for p in predictions
        ]
    )
    if known.any():
        print(
            f"recognise() with min confidence {recognizer.min_rank_confidence}: "
            f"{known.mean():.1%} named, {card_ok[known].mean():.1%} of them right, "
            f"{(~known).sum()} left unknown"
        )
    print(
        confusion_table(
            [c.rank.value for c in truth],
            [p.rank.value for p in predictions],
            [rank.value for rank in Rank],
        )
    )
    if errors_dir is not None:
        for image, card, path, prediction, ok in zip(
            images, truth, paths, predictions, card_ok, strict=True
        ):
            if not ok:
                # "x" for an unread suit: "?" is not allowed in Windows file names.
                guess = prediction.card.code if prediction.card else f"{prediction.rank.value}x"
                write_image(errors_dir / f"{card.code}_as_{guess}_{path.stem}.png", image)
        print(f"Errors saved to {errors_dir}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    images, cards, paths = load_dataset(args.data)
    per_card = Counter(card.code for card in cards)
    print(f"{len(images)} images of {len(per_card)} cards (min {min(per_card.values())} per card)")

    config = RecognizerConfig(
        features=FeatureConfig(mode=args.sampling, step=args.step),
        vocabulary_size=args.vocabulary,
        pyramid_levels=tuple(args.levels),
        svm_c=args.c,
        seed=args.seed,
    )
    if args.test_data is not None:
        test_images, test_cards, test_paths = load_dataset(args.test_data)
        start = time.perf_counter()
        recognizer = CardRecognizer(config).fit(images, cards)
        print(f"Training: {time.perf_counter() - start:.1f} s on all {len(images)} images")
        print(f"Test set: {len(test_images)} images from {args.test_data}")
        evaluate(recognizer, test_images, test_cards, test_paths, args.errors_dir)
    else:
        stratify = [card.code for card in cards] if min(per_card.values()) >= 2 else None
        train_idx, test_idx = train_test_split(
            np.arange(len(images)),
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stratify,
        )
        start = time.perf_counter()
        recognizer = CardRecognizer(config).fit(
            [images[i] for i in train_idx], [cards[i] for i in train_idx]
        )
        print(f"Training: {time.perf_counter() - start:.1f} s on {len(train_idx)} images")
        print("Test set: a random part of the same recording (optimistic, see --test-data)")
        evaluate(
            recognizer,
            [images[i] for i in test_idx],
            [cards[i] for i in test_idx],
            [paths[i] for i in test_idx],
            args.errors_dir,
        )
        if not args.no_refit:
            start = time.perf_counter()
            recognizer = CardRecognizer(config).fit(images, cards)
            print(f"Refit on all {len(images)} images: {time.perf_counter() - start:.1f} s")
    print(f"Saved {recognizer.save(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
