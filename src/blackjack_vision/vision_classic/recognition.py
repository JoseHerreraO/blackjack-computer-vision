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
  Tema 2) decides whether the suit is red or black, and the normalised cross-correlation of
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
INDEX_REGION = (0.22, 0.32)
"""Width and height of each index corner, as fractions of the card."""

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
        red_ink: Fraction of the ink of the card that is red.
    """

    rank: Rank
    suit: Suit | None
    rank_margin: float
    suit_margin: float
    red_ink: float

    @property
    def card(self) -> Card | None:
        """The full card, or ``None`` if the suit is unknown."""
        return Card(self.rank, self.suit) if self.suit is not None else None


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

    def scores(self, histograms: np.ndarray) -> np.ndarray:
        """One-vs-rest scores ``(n_samples, n_classes)``, columns ordered as :attr:`classes`."""
        if self.svm is None:
            raise RuntimeError("The SVM is not trained")
        histograms = np.asarray(histograms, np.float64)
        kernel = np.zeros((len(histograms), self.n_train))
        kernel[:, self.svm.support_] = chi2_kernel(
            histograms, self.train.astype(np.float64), gamma=self.gamma
        )
        scores = self.svm.decision_function(kernel)
        if scores.ndim == 1:  # binary problem: a single signed distance
            scores = np.column_stack([-scores, scores])
        return scores


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
        rank_scores = self.rank_svm.scores(np.stack([self.histogram(image) for image in images]))
        rank_classes = self.rank_svm.classes
        known = self.suit_templates.suits
        both_colours = any(s.is_red for s in known) and any(not s.is_red for s in known)

        predictions = []
        for image, scores in zip(images, rank_scores, strict=True):
            rank_index, rank_margin = _best(scores)
            corners = index_corners(image, self.config.index_region)
            red_ink = red_ink_fraction(image)
            allowed = None
            if self.config.colour_filter and both_colours:
                red = red_ink >= RED_INK_THRESHOLD
                allowed = [s for s in known if s.is_red == red]
            suit, suit_margin = self.suit_templates.classify(
                corner_symbols(corners), allowed=allowed
            )
            predictions.append(
                Prediction(Rank(rank_classes[rank_index]), suit, rank_margin, suit_margin, red_ink)
            )
        return predictions

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


def red_ink_fraction(card: np.ndarray, margin: float = CARD_MARGIN) -> float:
    """Fraction of the ink of a rectified card that is red (HSV, Tema 2).

    Ink pixels are found with Otsu's threshold on the minimum of the B, G and R channels (dark for
    ink of any colour). A pixel is red when its hue is within ``RED_HUE_MAX`` of red and its
    saturation is at least ``RED_MIN_SATURATION``. The whole card body is used, not only the index
    corners: in the glasses video the indices are small and pale, and real red ink is more orange
    and less saturated than a pure red.
    """
    if card.ndim != 3:
        return 0.0
    height, width = card.shape[:2]
    dy, dx = int(margin * height), int(margin * width)
    body = card[dy : height - dy, dx : width - dx]
    ink = ink_mask(body, upscale=1) > 0
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    hue, saturation = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
    red = ((hue <= RED_HUE_MAX) | (hue >= RED_HUE_MIN)) & (saturation >= RED_MIN_SATURATION)
    count = int(ink.sum())
    return float((red & ink).sum() / count) if count else 0.0


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
