"""Rank and suit recognition of rectified cards (Tema 4).

* **Rank**: bag of visual words + chi-squared SVM. Dense SIFT descriptors on the whole card, a
  K-Means vocabulary and spatial-pyramid histograms
  (:mod:`blackjack_vision.vision_classic.features`) feed an SVM with the kernel
  ``k(x, y) = exp(-gamma * sum((x - y)^2 / (x + y)))``, where ``gamma`` is the inverse of the
  mean chi-squared distance between training histograms. The number and layout of the pips and
  the face-card artwork tell the 13 ranks apart (what Blackjack and the Hi-Lo count need). The
  detector cannot tell which short side of a card is the top, so every training card is also
  used rotated by 180 degrees.
* **Suit**: colour, then template matching (Tema 3). The fraction of red ink on the card (HSV,
  Tema 2; see :func:`card_colour`, face cards excepted) decides whether the suit is red or black, and the normalised cross-correlation of
  the suit symbol under the rank with one template per suit
  (:mod:`blackjack_vision.vision_classic.suit_templates`) picks one of the two suits of that
  colour. In a bag of words the small symbol is diluted among the pips, which confuses spades
  with clubs and hearts with diamonds.

A single frame can still be wrong (blur, glare), so per-card predictions are meant to be smoothed
over time with :class:`TemporalVoter` once the tracker assigns stable IDs.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Generic, TypeVar

import cv2
import joblib
import numpy as np
from sklearn.metrics.pairwise import additive_chi2_kernel, chi2_kernel
from sklearn.svm import SVC

from blackjack_vision.blackjack.cards import Card, Rank, Suit
from blackjack_vision.capture.sources import IMAGE_EXTENSIONS, read_image
from blackjack_vision.vision_classic.features import (
    BowEncoder,
    FeatureConfig,
    SiftExtractor,
    Vocabulary,
)
from blackjack_vision.vision_classic.suit_templates import SuitTemplates, corner_symbols, ink_mask

logger = logging.getLogger(__name__)

MODEL_FORMAT_VERSION = 2
RED_INK_THRESHOLD = 0.45
"""Share of red ink above which a card is red. Real red ink from the glasses is orange-ish and
pale; on the 18/11 recording red cards measured 0.59-0.80 and black ones 0.00-0.30."""
RED_HUE_MAX = 20
RED_HUE_MIN = 165
RED_MIN_SATURATION = 45
CARD_MARGIN = 0.04
"""Border of the rectified card left out of colour measurements (table pixels may leak in)."""
PICTURE_SIZE = 0.5
"""A blob wider and taller than this fraction of the card body is a face-card picture."""
INDEX_REGION = (0.22, 0.32)
"""Width and height of each index corner, as fractions of the card."""
MIN_RANK_CONFIDENCE = 0.4
"""Weakest one-vs-one SVM margin of the winning rank below which :meth:`CardRecognizer.recognise`
reports the card as unknown. On the synthetic cards every wrong rank stayed below 0.32."""
MIN_SUIT_MARGIN = 0.02
"""NCC score gap between the two best suits below which the suit is considered unsure."""

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class RecognizerConfig:
    """Parameters of :class:`CardRecognizer`.

    Attributes:
        features: SIFT extraction on the whole card.
        vocabulary_size: Number of visual words.
        pyramid_levels: Grid sides of the spatial pyramid.
        svm_c: Regularisation constant of the rank SVM.
        index_region: Width and height of an index corner, as fractions of the card.
        augment_rotation: Also train the rank SVM on every card rotated by 180 degrees.
        colour_filter: Restrict the suit to the colour of the ink in the index corners.
        max_vocabulary_descriptors: Descriptors sampled to build the vocabulary.
        seed: Random seed (vocabulary sampling and K-Means).
    """

    features: FeatureConfig = field(default_factory=FeatureConfig)
    vocabulary_size: int = 300
    pyramid_levels: tuple[int, ...] = (1, 2, 4)
    svm_c: float = 10.0
    index_region: tuple[float, float] = INDEX_REGION
    augment_rotation: bool = True
    colour_filter: bool = True
    max_vocabulary_descriptors: int = 25_000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least 2")
        if self.svm_c <= 0:
            raise ValueError("svm_c must be positive")
        width, height = self.index_region
        if not (0 < width <= 0.5 and 0 < height <= 0.5):
            raise ValueError(f"Invalid index_region {self.index_region}")
        object.__setattr__(self, "pyramid_levels", tuple(self.pyramid_levels))
        object.__setattr__(self, "index_region", tuple(self.index_region))

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form, stored inside saved models."""
        data = {name: getattr(self, name) for name in self.__dataclass_fields__}
        features = self.features
        data["features"] = {
            "mode": features.mode.value,
            "step": features.step,
            "patch_size": features.patch_size,
            "max_keypoints": features.max_keypoints,
            "root_sift": features.root_sift,
            "image_size": list(features.image_size),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecognizerConfig:
        values = dict(data)
        values["features"] = FeatureConfig(**values["features"])
        return cls(**values)


@dataclass(frozen=True)
class Prediction:
    """Recognition result for one card image.

    Attributes:
        rank: Most likely rank.
        suit: Most likely suit, ``None`` if no suit symbol could be read.
        rank_margin: Score gap between the best and the second-best rank (higher = surer).
        suit_margin: Score gap between the best and the second-best allowed suit.
        red_ink: Fraction of the ink of the card that is red (face-card picture left out).
        rank_confidence: Smallest one-vs-one SVM margin between the winning rank and every other
            rank (see :meth:`Chi2SVM.weakest_duel`). It measures how clearly the rank wins its
            closest duel; ``rank_margin`` only counts votes and is about 1 for most cards.
    """

    rank: Rank
    suit: Suit | None
    rank_margin: float
    suit_margin: float
    red_ink: float
    rank_confidence: float = 0.0

    @property
    def card(self) -> Card | None:
        """The full card, or ``None`` if the suit is unknown."""
        return Card(self.rank, self.suit) if self.suit is not None else None


@dataclass(frozen=True)
class ClassicRecognition:
    """Result of :meth:`CardRecognizer.recognise` (same fields the card pipeline reads).

    Attributes:
        card: The card, or ``None`` when the recogniser is not sure.
        similarity: Confidence of the rank (its weakest one-vs-one SVM margin); not a cosine.
        prediction: The full prediction, for logging and evaluation.
    """

    card: Card | None
    similarity: float
    prediction: Prediction

    @property
    def known(self) -> bool:
        return self.card is not None


class Chi2SVM:
    """SVM with a chi-squared kernel on non-negative histograms (precomputed kernel).

    Args:
        c: Regularisation constant.
    """

    def __init__(self, c: float = 10.0) -> None:
        self.c = c
        self.gamma = 1.0
        self.n_train = 0
        self.train: np.ndarray | None = None  # histograms of the support vectors
        self.svm: SVC | None = None

    @property
    def classes(self) -> np.ndarray:
        if self.svm is None:
            raise RuntimeError("The SVM is not trained")
        return self.svm.classes_

    def fit(self, histograms: np.ndarray, labels: Sequence[str]) -> Chi2SVM:
        histograms = np.asarray(histograms, np.float64)
        distances = -additive_chi2_kernel(histograms)
        off_diagonal = distances[~np.eye(len(histograms), dtype=bool)]
        mean = float(off_diagonal.mean()) if off_diagonal.size else 0.0
        self.gamma = 1.0 / mean if mean > 0 else 1.0
        self.svm = SVC(kernel="precomputed", C=self.c).fit(
            chi2_kernel(histograms, gamma=self.gamma), list(labels)
        )
        # Only the support vectors take part in the decision function, so only their
        # histograms are kept (the saved model shrinks accordingly).
        self.n_train = len(histograms)
        self.train = histograms[self.svm.support_].astype(np.float32)
        return self

    def _kernel(self, histograms: np.ndarray) -> np.ndarray:
        if self.svm is None:
            raise RuntimeError("The SVM is not trained")
        histograms = np.asarray(histograms, np.float64)
        kernel = np.zeros((len(histograms), self.n_train))
        kernel[:, self.svm.support_] = chi2_kernel(
            histograms, self.train.astype(np.float64), gamma=self.gamma
        )
        return kernel

    def scores(self, histograms: np.ndarray) -> np.ndarray:
        """One-vs-rest scores ``(n_samples, n_classes)``, columns ordered as :attr:`classes`."""
        kernel = self._kernel(histograms)
        scores = self.svm.decision_function(kernel)
        if scores.ndim == 1:  # binary problem: a single signed distance
            scores = np.column_stack([-scores, scores])
        return scores

    def weakest_duel(self, histograms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Winning class of every sample and the margin of its closest one-vs-one duel.

        A multi-class SVM is a set of one-vs-one SVMs, one per pair of classes. The predicted
        class wins most duels; the smallest signed decision value between it and every other
        class says how clearly it wins: negative if it lost a duel, close to 0 when another class
        is almost as good, large when it beats all of them comfortably.

        Returns:
            Labels ``(n_samples,)`` and margins ``(n_samples,)``.
        """
        if self.svm is None:
            raise RuntimeError("The SVM is not trained")
        kernel = self._kernel(histograms)
        labels = self.svm.predict(kernel)
        self.svm.decision_function_shape = "ovo"
        try:
            values = self.svm.decision_function(kernel)
        finally:
            self.svm.decision_function_shape = "ovr"
        values = np.asarray(values).reshape(len(kernel), -1)
        classes = list(self.svm.classes_)
        n = len(classes)
        margins = np.empty(len(kernel))
        for row, (label, duels) in enumerate(zip(labels, values, strict=True)):
            pairwise = np.zeros((n, n))
            for k, (i, j) in enumerate(combinations(range(n), 2)):
                pairwise[i, j], pairwise[j, i] = duels[k], -duels[k]
            winner = classes.index(label)
            margins[row] = min(pairwise[winner, j] for j in range(n) if j != winner)
        return labels, margins


class CardRecognizer:
    """Recognises the rank and suit of rectified cards.

    Args:
        config: Training parameters, defaults if ``None``.
    """

    def __init__(self, config: RecognizerConfig | None = None) -> None:
        self.config = config or RecognizerConfig()
        self.extractor = SiftExtractor(self.config.features)
        self.encoder: BowEncoder | None = None
        self.rank_svm: Chi2SVM | None = None
        self.suit_templates: SuitTemplates | None = None
        self.min_rank_confidence = MIN_RANK_CONFIDENCE
        self.min_suit_margin = MIN_SUIT_MARGIN

    @property
    def is_trained(self) -> bool:
        return self.rank_svm is not None and self.suit_templates is not None

    def fit(self, images: Sequence[np.ndarray], cards: Sequence[Card]) -> CardRecognizer:
        """Train the vocabulary, the rank SVM and the suit templates.

        Args:
            images: Rectified cards (BGR), e.g. the output of ``rectify_card``.
            cards: Label of every image.

        Raises:
            ValueError: on mismatched lengths or with fewer than two ranks or two suits.
        """
        if len(images) != len(cards):
            raise ValueError(f"{len(images)} images but {len(cards)} labels")
        if len({card.rank for card in cards}) < 2 or len({card.suit for card in cards}) < 2:
            raise ValueError("Training needs at least two ranks and two suits")
        config = self.config

        features, ranks = [], []
        for image, card in zip(images, cards, strict=True):
            features.append(self.extractor.extract(image))
            ranks.append(card.rank.value)
            if config.augment_rotation:
                features.append(self.extractor.extract(cv2.rotate(image, cv2.ROTATE_180)))
                ranks.append(card.rank.value)
        vocabulary = Vocabulary.fit(
            [descriptors for _, descriptors in features],
            config.vocabulary_size,
            max_descriptors=config.max_vocabulary_descriptors,
            seed=config.seed,
        )
        self.encoder = BowEncoder(vocabulary, config.pyramid_levels, config.features.image_size)
        histograms = np.stack([self.encoder.encode(*f) for f in features])
        self.rank_svm = Chi2SVM(config.svm_c).fit(histograms, ranks)
        self.suit_templates = SuitTemplates().fit(
            [index_corners(image, config.index_region) for image in images],
            [card.suit for card in cards],
        )
        logger.info(
            "Trained on %d images (%d rank samples), %d words",
            len(images),
            len(ranks),
            len(vocabulary),
        )
        return self

    def histogram(self, image: np.ndarray) -> np.ndarray:
        """Bag-of-words histogram of one rectified card."""
        if self.encoder is None:
            raise RuntimeError("The recognizer is not trained")
        return self.encoder.encode(*self.extractor.extract(image))

    def predict(self, image: np.ndarray) -> Prediction:
        """Recognise one rectified card."""
        return self.predict_many([image])[0]

    def predict_many(self, images: Sequence[np.ndarray]) -> list[Prediction]:
        """Recognise several rectified cards at once."""
        if not self.is_trained:
            raise RuntimeError("The recognizer is not trained")
        if len(images) == 0:
            return []
        histograms = np.stack([self.histogram(image) for image in images])
        rank_scores = self.rank_svm.scores(histograms)
        rank_labels, rank_confidences = self.rank_svm.weakest_duel(histograms)
        known = self.suit_templates.suits
        both_colours = any(s.is_red for s in known) and any(not s.is_red for s in known)

        predictions = []
        for image, scores, label, confidence in zip(
            images, rank_scores, rank_labels, rank_confidences, strict=True
        ):
            _, rank_margin = _best(scores)
            corners = index_corners(image, self.config.index_region)
            red_ink = red_ink_fraction(image)
            colour = card_colour(image)
            allowed = None
            if self.config.colour_filter and both_colours and colour is not None:
                allowed = [s for s in known if s.is_red == colour]
            suit, suit_margin = self.suit_templates.classify(
                corner_symbols(corners), allowed=allowed
            )
            predictions.append(
                Prediction(Rank(label), suit, rank_margin, suit_margin, red_ink, float(confidence))
            )
        return predictions

    def recognise(self, image: np.ndarray) -> ClassicRecognition:
        """Name one rectified card, or report it as unknown when the evidence is weak.

        Same interface as the embedding recogniser, so the real-time ``CardPipeline`` can use
        either: ``card`` is ``None`` when the rank's weakest one-vs-one margin is below
        :attr:`min_rank_confidence`, or the suit is missing or its NCC margin is below
        :attr:`min_suit_margin`. The pipeline then tries again with fresher crops.
        """
        prediction = self.predict(image)
        sure = (
            prediction.card is not None
            and prediction.rank_confidence >= self.min_rank_confidence
            and prediction.suit_margin >= self.min_suit_margin
        )
        return ClassicRecognition(
            prediction.card if sure else None, prediction.rank_confidence, prediction
        )

    def save(self, path: str | Path) -> Path:
        """Store the trained model (``joblib`` file), creating parent folders if needed."""
        if not self.is_trained:
            raise RuntimeError("The recognizer is not trained")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format": MODEL_FORMAT_VERSION,
            "config": self.config.to_dict(),
            "words": self.encoder.vocabulary.words,
            "rank_svm": self.rank_svm,
            "suit_templates": self.suit_templates,
        }
        joblib.dump(state, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path) -> CardRecognizer:
        """Load a model written by :meth:`save`.

        Only load files you created yourself: ``joblib`` files can execute code when loaded.
        """
        state = joblib.load(Path(path))
        if state.get("format") != MODEL_FORMAT_VERSION:
            raise ValueError(f"Unsupported model format {state.get('format')!r} in {path}")
        recognizer = cls(RecognizerConfig.from_dict(state["config"]))
        config = recognizer.config
        recognizer.encoder = BowEncoder(
            Vocabulary(state["words"]), config.pyramid_levels, config.features.image_size
        )
        recognizer.rank_svm = state["rank_svm"]
        recognizer.suit_templates = state["suit_templates"]
        return recognizer


def index_corners(card: np.ndarray, region: tuple[float, float] = INDEX_REGION) -> list[np.ndarray]:
    """The two index corners of a rectified card, both upright.

    Returns the top-left corner and the bottom-right corner rotated by 180 degrees. When the card
    is upside down the two simply swap places.
    """
    height, width = card.shape[:2]
    corner_w, corner_h = max(1, int(width * region[0])), max(1, int(height * region[1]))
    top_left = card[:corner_h, :corner_w]
    bottom_right = cv2.rotate(card[height - corner_h :, width - corner_w :], cv2.ROTATE_180)
    return [top_left, bottom_right]


def _body(card: np.ndarray, margin: float) -> np.ndarray:
    height, width = card.shape[:2]
    dy, dx = int(margin * height), int(margin * width)
    return card[dy : height - dy, dx : width - dx]


def picture_box(card: np.ndarray, margin: float = CARD_MARGIN) -> tuple[int, int, int, int] | None:
    """Bounding box ``(x, y, w, h)`` of a face-card picture in the card body, or ``None``.

    The ink of the body is thresholded with Otsu and split into connected components; a component
    wider and taller than ``PICTURE_SIZE`` of the body is the picture (or its frame). Coordinates
    are relative to the body (the card without its ``margin``).
    """
    if card.ndim != 3:
        return None
    ink = ink_mask(_body(card, margin), upscale=1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    height, width = ink.shape
    boxes = [
        stats[label, :4]
        for label in range(1, count)
        if stats[label, 2] > PICTURE_SIZE * width and stats[label, 3] > PICTURE_SIZE * height
    ]
    if not boxes:
        return None
    x0 = min(int(b[0]) for b in boxes)
    y0 = min(int(b[1]) for b in boxes)
    x1 = max(int(b[0] + b[2]) for b in boxes)
    y1 = max(int(b[1] + b[3]) for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def red_ink_fraction(card: np.ndarray, margin: float = CARD_MARGIN) -> float:
    """Fraction of the ink of a rectified card that is red (HSV, Tema 2).

    Ink pixels are found with Otsu's threshold on the minimum of the B, G and R channels (dark for
    ink of any colour). A pixel is red when its hue is within ``RED_HUE_MAX`` of red and its
    saturation is at least ``RED_MIN_SATURATION``. The whole card body is used, not only the index
    corners: in the glasses video the indices are small and pale, and real red ink is more orange
    and less saturated than a pure red. The picture of a face card (:func:`picture_box`) is left
    out, because its artwork has red and yellow in black cards too.
    """
    if card.ndim != 3:
        return 0.0
    body = _body(card, margin)
    ink = ink_mask(body, upscale=1) > 0
    box = picture_box(card, margin)
    if box is not None:
        x, y, w, h = box
        ink[y : y + h, x : x + w] = False
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    hue, saturation = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
    red = ((hue <= RED_HUE_MAX) | (hue >= RED_HUE_MIN)) & (saturation >= RED_MIN_SATURATION)
    count = int(ink.sum())
    return float((red & ink).sum() / count) if count else 0.0


def card_colour(card: np.ndarray, margin: float = CARD_MARGIN) -> bool | None:
    """``True`` for a red card, ``False`` for a black one, ``None`` when it cannot be told.

    Number cards and aces are decided by :func:`red_ink_fraction`. For a face card the colour is
    reported as unknown: once the picture is left out only the index corners remain, and in the
    glasses video their pale red ink reads as red for some red cards and not for others (0.00-0.39
    on the 18/11 recording). The suit of a face card is then chosen among the four suits by the
    shape of its index symbol alone.
    """
    if picture_box(card, margin) is not None:
        return None
    return red_ink_fraction(card, margin) >= RED_INK_THRESHOLD


def load_dataset(folder: str | Path) -> tuple[list[np.ndarray], list[Card], list[Path]]:
    """Rectified cards stored as ``folder/<CODE>/*.png`` (e.g. ``data/cards/AS/0001.png``).

    Sub-folders whose name is not a card code are skipped with a warning.

    Returns:
        Images, their cards and their paths.
    """
    folder = Path(folder)
    images: list[np.ndarray] = []
    cards: list[Card] = []
    paths: list[Path] = []
    for sub in sorted(path for path in folder.iterdir() if path.is_dir()):
        try:
            card = Card.parse(sub.name)
        except ValueError:
            logger.warning("Skipping %s: not a card code", sub)
            continue
        for path in sorted(sub.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = read_image(path)
            if image is None:
                logger.warning("Skipping unreadable image %s", path)
                continue
            images.append(image)
            cards.append(card)
            paths.append(path)
    if not images:
        raise FileNotFoundError(f"No card images in {folder}/<CODE>/")
    return images, cards, paths


class TemporalVoter(Generic[T]):
    """Majority vote of the last ``window`` predictions of every tracked object.

    Example::

        voter = TemporalVoter(window=15)
        label, agreement = voter.add(track_id, prediction.card)
    """

    def __init__(self, window: int = 15) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = window
        self._history: dict[Hashable, deque[T]] = {}

    def add(self, key: Hashable, label: T) -> tuple[T, float]:
        """Record a prediction and return the current vote."""
        history = self._history.setdefault(key, deque(maxlen=self.window))
        history.append(label)
        label, count = Counter(history).most_common(1)[0]
        return label, count / len(history)

    def vote(self, key: Hashable) -> tuple[T, float] | None:
        """Most frequent label and the fraction of votes it got, or ``None`` if unseen."""
        history = self._history.get(key)
        if not history:
            return None
        label, count = Counter(history).most_common(1)[0]
        return label, count / len(history)

    def forget(self, key: Hashable) -> None:
        """Drop the history of an object that left the scene."""
        self._history.pop(key, None)


def _best(scores: np.ndarray) -> tuple[int, float]:
    """Index of the highest finite score and its margin over the runner-up."""
    finite = np.sort(scores[np.isfinite(scores)])
    best = int(np.argmax(scores))
    margin = float(finite[-1] - finite[-2]) if len(finite) > 1 else float("inf")
    return best, margin
