"""Experiments on the card recogniser for the Results section of the report.

Three studies of the classical recogniser (:mod:`blackjack_vision.vision_classic.recognition`),
plus a simple baseline, all measured on the same train and test sets:

* **Vocabulary size** ``K`` of the bag of visual words (Tema 4).
* **Spatial pyramid**: plain bag of words (``1``) against the 2x2 and 4x4 grids.
* **SVM kernel** on the same histograms: chi-squared (the default), histogram intersection,
  linear and RBF. The chi-squared and intersection kernels are designed for histograms; the
  other two treat them as ordinary vectors.
* **Template matching baseline** (Tema 3): every card class gets the mean of its training images
  as a template, and a test card takes the class of the template with the highest normalised
  cross-correlation (``cv2.matchTemplate`` with ``TM_CCOEFF_NORMED``), trying both 180-degree
  orientations. It answers how much the bag of words adds over comparing whole images.

Each run reports rank, suit and card accuracy, training time and recognition time per card.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np
from sklearn.metrics.pairwise import additive_chi2_kernel, chi2_kernel
from sklearn.svm import SVC

from blackjack_vision.blackjack.cards import Card, Rank
from blackjack_vision.vision_classic.recognition import CardRecognizer, RecognizerConfig

TEMPLATE_SIZE = (100, 150)
KERNELS = ("chi2", "intersection", "linear", "rbf")


@dataclass(frozen=True)
class StudyResult:
    """One row of a study.

    Attributes:
        study: Which study (``vocabulary``, ``pyramid``, ``kernel``, ``baseline``).
        setting: The value tried, e.g. ``"K=300"`` or ``"chi2"``.
        rank_accuracy: Fraction of test cards with the right rank.
        suit_accuracy: Fraction with the right suit (``nan`` when the setting does not change it).
        card_accuracy: Fraction with both right (``nan`` when not measured).
        train_seconds: Training time.
        ms_per_card: Recognition time per test card.
    """

    study: str
    setting: str
    rank_accuracy: float
    suit_accuracy: float
    card_accuracy: float
    train_seconds: float
    ms_per_card: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


Dataset = tuple[Sequence[np.ndarray], Sequence[Card]]


def _accuracies(
    ranks: Sequence[object], suits: Sequence[object] | None, truth: Sequence[Card]
) -> tuple[float, float, float]:
    rank_ok = np.array([r == c.rank for r, c in zip(ranks, truth, strict=True)])
    if suits is None:
        return float(rank_ok.mean()), float("nan"), float("nan")
    suit_ok = np.array([s == c.suit for s, c in zip(suits, truth, strict=True)])
    return float(rank_ok.mean()), float(suit_ok.mean()), float((rank_ok & suit_ok).mean())


def evaluate_recognizer(
    study: str, setting: str, config: RecognizerConfig, train: Dataset, test: Dataset
) -> StudyResult:
    """Train a :class:`CardRecognizer` with ``config`` and measure it on ``test``."""
    start = time.perf_counter()
    recognizer = CardRecognizer(config).fit(*train)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predictions = recognizer.predict_many(test[0])
    ms = 1000 * (time.perf_counter() - start) / max(1, len(test[0]))
    ranks = [p.rank for p in predictions]
    suits = [p.suit for p in predictions]
    return StudyResult(study, setting, *_accuracies(ranks, suits, test[1]), train_seconds, ms)


def vocabulary_study(
    train: Dataset, test: Dataset, sizes: Sequence[int], base: RecognizerConfig
) -> list[StudyResult]:
    """Rank/suit/card accuracy for every vocabulary size ``K``."""
    return [
        evaluate_recognizer("vocabulary", f"K={k}", replace(base, vocabulary_size=k), train, test)
        for k in sizes
    ]


def pyramid_study(
    train: Dataset,
    test: Dataset,
    levels: Sequence[tuple[int, ...]],
    base: RecognizerConfig,
) -> list[StudyResult]:
    """Accuracy for every spatial pyramid, e.g. ``(1,)``, ``(1, 2)``, ``(1, 2, 4)``."""
    return [
        evaluate_recognizer(
            "pyramid",
            "+".join(f"{n}x{n}" for n in level),
            replace(base, pyramid_levels=tuple(level)),
            train,
            test,
        )
        for level in levels
    ]


def intersection_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Histogram intersection kernel ``k(x, y) = sum(min(x, y))``."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    out = np.empty((len(x), len(y)))
    for i, row in enumerate(x):
        out[i] = np.minimum(row[None, :], y).sum(axis=1)
    return out


def _chi2_with_gamma(train: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    distances = -additive_chi2_kernel(train)
    mean = distances[~np.eye(len(train), dtype=bool)].mean()
    gamma = 1.0 / mean if mean > 0 else 1.0
    return lambda a, b: chi2_kernel(a, b, gamma=gamma)


def kernel_study(
    train: Dataset, test: Dataset, base: RecognizerConfig, kernels: Sequence[str] = KERNELS
) -> list[StudyResult]:
    """Rank accuracy of SVMs with different kernels on the same bag-of-words histograms.

    The vocabulary and histograms of ``base`` are computed once; only the SVM changes. Training
    uses every card in both 180-degree orientations, as :class:`CardRecognizer` does.
    """
    recognizer = CardRecognizer(base).fit(*train)
    images, cards = train
    samples = list(images)
    labels = [c.rank.value for c in cards]
    if base.augment_rotation:
        samples += [cv2.rotate(image, cv2.ROTATE_180) for image in images]
        labels += labels
    x_train = np.stack([recognizer.histogram(image) for image in samples]).astype(np.float64)
    x_test = np.stack([recognizer.histogram(image) for image in test[0]]).astype(np.float64)

    results = []
    for name in kernels:
        start = time.perf_counter()
        if name in ("chi2", "intersection"):
            kernel = _chi2_with_gamma(x_train) if name == "chi2" else intersection_kernel
            svm = SVC(kernel="precomputed", C=base.svm_c).fit(kernel(x_train, x_train), labels)
            train_seconds = time.perf_counter() - start
            start = time.perf_counter()
            predicted = svm.predict(kernel(x_test, x_train))
        elif name in ("linear", "rbf"):
            svm = SVC(kernel=name, C=base.svm_c, gamma="scale").fit(x_train, labels)
            train_seconds = time.perf_counter() - start
            start = time.perf_counter()
            predicted = svm.predict(x_test)
        else:
            raise ValueError(f"Unknown kernel {name!r}, expected one of {KERNELS}")
        ms = 1000 * (time.perf_counter() - start) / max(1, len(x_test))
        ranks = [Rank(value) for value in predicted]
        results.append(
            StudyResult("kernel", name, *_accuracies(ranks, None, test[1]), train_seconds, ms)
        )
    return results


class TemplateMatcher:
    """Baseline: one mean template per card class, matched by normalised cross-correlation.

    Args:
        size: ``(width, height)`` the cards are reduced to before matching.
    """

    def __init__(self, size: tuple[int, int] = TEMPLATE_SIZE) -> None:
        self.size = size
        self.templates: dict[Card, np.ndarray] = {}

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)

    def fit(self, images: Sequence[np.ndarray], cards: Sequence[Card]) -> TemplateMatcher:
        groups: dict[Card, list[np.ndarray]] = {}
        for image, card in zip(images, cards, strict=True):
            groups.setdefault(card, []).append(self._prepare(image))
        self.templates = {card: np.mean(views, axis=0) for card, views in groups.items()}
        return self

    def predict(self, image: np.ndarray) -> Card:
        if not self.templates:
            raise RuntimeError("The template matcher is not fitted")
        patch = self._prepare(image)
        candidates = (patch, cv2.rotate(patch, cv2.ROTATE_180))
        best, best_score = None, -np.inf
        for card, template in self.templates.items():
            score = max(
                float(cv2.matchTemplate(view, template, cv2.TM_CCOEFF_NORMED)[0, 0])
                for view in candidates
            )
            if score > best_score:
                best, best_score = card, score
        return best


def template_baseline(train: Dataset, test: Dataset) -> StudyResult:
    """Accuracy of :class:`TemplateMatcher` (whole-card NCC, Tema 3)."""
    start = time.perf_counter()
    matcher = TemplateMatcher().fit(*train)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predicted = [matcher.predict(image) for image in test[0]]
    ms = 1000 * (time.perf_counter() - start) / max(1, len(test[0]))
    ranks = [card.rank for card in predicted]
    suits = [card.suit for card in predicted]
    return StudyResult(
        "baseline", "template NCC", *_accuracies(ranks, suits, test[1]), train_seconds, ms
    )


def results_table(results: Sequence[StudyResult]) -> str:
    """Markdown table of the results, ready for the report."""
    lines = [
        "| Study | Setting | Rank | Suit | Card | Train (s) | ms/card |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:

        def pct(value: float) -> str:
            return "–" if np.isnan(value) else f"{100 * value:.1f} %"

        lines.append(
            f"| {r.study} | {r.setting} | {pct(r.rank_accuracy)} | {pct(r.suit_accuracy)} | "
            f"{pct(r.card_accuracy)} | {r.train_seconds:.1f} | {r.ms_per_card:.1f} |"
        )
    return "\n".join(lines)
