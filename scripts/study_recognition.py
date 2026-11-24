"""Recogniser study for the report: vocabulary size, spatial pyramid, SVM kernel and a baseline.

Trains on one folder of rectified cards and tests on another (``data/<CODE>/*.png``, as written
by ``scripts/capture_cards.py``). Use two separate recording sessions: a random split of one
recording gives optimistic numbers. Without ``--test-data`` a stratified split is used and the
output says so.

Results go to ``--out``: ``results.csv``, ``results.md`` (a table for the report) and, when
matplotlib is installed, ``vocabulary.png`` (accuracy and time against ``K``).

Examples::

    python scripts/study_recognition.py --data data/cards --test-data data/cards_test
    python scripts/study_recognition.py --data data/cards --test-data data/cards_test \
        --vocabulary 50 100 200 300 500 --kernels chi2 intersection linear rbf
    python scripts/study_recognition.py --data data/cards --quick       # small, fast settings
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from blackjack_vision.vision_classic.recognition import RecognizerConfig, load_dataset
from blackjack_vision.vision_classic.study import (
    KERNELS,
    StudyResult,
    kernel_study,
    pyramid_study,
    results_table,
    template_baseline,
    vocabulary_study,
)

logger = logging.getLogger("study_recognition")


def parse_levels(text: str) -> tuple[int, ...]:
    """``"1,2,4"`` -> ``(1, 2, 4)``."""
    try:
        levels = tuple(int(part) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid pyramid {text!r}, e.g. 1,2,4") from error
    if not levels or any(level < 1 for level in levels):
        raise argparse.ArgumentTypeError(f"Invalid pyramid {text!r}")
    return levels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="training folder")
    parser.add_argument("--test-data", type=Path, help="test folder (another recording)")
    parser.add_argument("--test-size", type=float, default=0.25, help="split without --test-data")
    parser.add_argument("--vocabulary", type=int, nargs="+", default=[50, 100, 200, 300])
    parser.add_argument("--levels", type=parse_levels, nargs="+", default=[(1,), (1, 2), (1, 2, 4)])
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=list(KERNELS))
    parser.add_argument("--base-vocabulary", type=int, default=300, help="K for the other studies")
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=("vocabulary", "pyramid", "kernel", "baseline"),
        default=["vocabulary", "pyramid", "kernel", "baseline"],
    )
    parser.add_argument("--quick", action="store_true", help="small vocabularies, for a smoke run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/study"))
    return parser.parse_args(argv)


def write_outputs(results: list[StudyResult], out: Path, note: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].as_dict()))
        writer.writeheader()
        writer.writerows(result.as_dict() for result in results)
    (out / "results.md").write_text(f"{note}\n\n{results_table(results)}\n", encoding="utf-8")
    vocabulary = [r for r in results if r.study == "vocabulary"]
    if len(vocabulary) < 2:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not installed: no plot")
        return
    sizes = [int(r.setting.split("=")[1]) for r in vocabulary]
    fig, left = plt.subplots(figsize=(7, 4))
    left.plot(sizes, [100 * r.rank_accuracy for r in vocabulary], "o-", label="rank")
    left.plot(sizes, [100 * r.card_accuracy for r in vocabulary], "s-", label="card")
    left.set_xlabel("Vocabulary size K (visual words)")
    left.set_ylabel("Test accuracy [%]")
    left.legend(loc="lower left")
    right = left.twinx()
    right.plot(sizes, [r.ms_per_card for r in vocabulary], "k--", label="ms/card")
    right.set_ylabel("Recognition time [ms/card]")
    left.set_title("Bag of visual words: vocabulary size")
    fig.tight_layout()
    fig.savefig(out / "vocabulary.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    images, cards, _ = load_dataset(args.data)
    if args.test_data is not None:
        test_images, test_cards, _ = load_dataset(args.test_data)
        note = (
            f"Train: {args.data} ({len(images)} images). Test: {args.test_data} "
            f"({len(test_images)} images, separate recording)."
        )
        train = (images, cards)
        test = (test_images, test_cards)
    else:
        codes = [card.code for card in cards]
        counts = {code: codes.count(code) for code in set(codes)}
        stratify = codes if min(counts.values()) >= 2 else None
        train_idx, test_idx = train_test_split(
            np.arange(len(images)),
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stratify,
        )
        train = ([images[i] for i in train_idx], [cards[i] for i in train_idx])
        test = ([images[i] for i in test_idx], [cards[i] for i in test_idx])
        note = (
            f"Train/test: random split of {args.data} ({len(train_idx)}/{len(test_idx)} "
            "images). Optimistic: use --test-data with another recording for the report."
        )

    sizes, base_k = args.vocabulary, args.base_vocabulary
    if args.quick:
        sizes, base_k = [20, 40], 40
    base = RecognizerConfig(vocabulary_size=base_k, seed=args.seed)

    results: list[StudyResult] = []
    if "baseline" in args.studies:
        logger.info("Template matching baseline")
        results.append(template_baseline(train, test))
    if "vocabulary" in args.studies:
        logger.info("Vocabulary sizes %s", sizes)
        results += vocabulary_study(train, test, sizes, base)
    if "pyramid" in args.studies:
        logger.info("Pyramids %s", args.levels)
        results += pyramid_study(train, test, args.levels, base)
    if "kernel" in args.studies:
        logger.info("Kernels %s", args.kernels)
        results += kernel_study(train, test, base, args.kernels)

    write_outputs(results, args.out, note)
    print(note)
    print(results_table(results))
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
