"""Bag of visual words with SIFT descriptors and a spatial pyramid (Tema 4).

1. :class:`SiftExtractor` computes SIFT descriptors on the rectified card, either on a dense grid
   (default: every card gets the same number of descriptors, also plain ones such as aces) or at
   the keypoints found by the SIFT detector. Descriptors are converted to RootSIFT (L1 normalise
   and square root), so that comparing them with the Euclidean distance amounts to the Hellinger
   kernel, which works better for histograms of gradients.
2. :class:`Vocabulary` clusters training descriptors with K-Means (``cv2.kmeans``); each centre is
   a visual word. New descriptors are assigned to their nearest word with ``cv2.BFMatcher``.
3. :class:`BowEncoder` counts the words in every cell of a spatial pyramid (the whole card,
   then a 2x2 grid...). The plain bag of words ignores *where* the words are and, once
   normalised, *how many* there are, which is what tells a 3 from a 4 of the same suit; the
   pyramid keeps the layout of the pips (Lazebnik et al., CVPR 2006).

The histograms are divided by the number of descriptors of the image, so with the dense grid they
keep the absolute counts. They are meant for a chi-squared (or histogram intersection) kernel.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from blackjack_vision.vision_classic.rectify import CARD_SIZE

DESCRIPTOR_SIZE = 128


class SamplingMode(str, Enum):
    """Where SIFT descriptors are computed."""

    DENSE = "dense"
    KEYPOINTS = "keypoints"


@dataclass(frozen=True)
class FeatureConfig:
    """Parameters of :class:`SiftExtractor`.

    Attributes:
        mode: Dense grid or SIFT detector keypoints.
        step: Grid spacing in pixels (dense mode).
        patch_size: Diameter of each dense patch in pixels.
        max_keypoints: Strongest keypoints kept by the detector (keypoints mode).
        root_sift: Apply the RootSIFT (Hellinger) normalisation.
        image_size: ``(width, height)`` every card is resized to before extraction.
    """

    mode: SamplingMode = SamplingMode.DENSE
    step: int = 8
    patch_size: int = 16
    max_keypoints: int = 300
    root_sift: bool = True
    image_size: tuple[int, int] = CARD_SIZE

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", SamplingMode(self.mode))
        object.__setattr__(self, "image_size", tuple(int(v) for v in self.image_size))
        if self.step <= 0 or self.patch_size <= 0 or self.max_keypoints <= 0:
            raise ValueError("step, patch_size and max_keypoints must be positive")


class SiftExtractor:
    """SIFT descriptors of a card image.

    Args:
        config: Extraction parameters, defaults if ``None``.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self._sift = cv2.SIFT_create(nfeatures=self.config.max_keypoints)
        self._grid = _dense_grid(self.config)

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.extract(image)

    def extract(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Descriptors of one image.

        Args:
            image: BGR or grayscale card, resized to ``config.image_size`` if needed.

        Returns:
            Keypoint positions ``(N, 2)`` in the resized image and descriptors ``(N, 128)``, both
            ``float32``. ``N`` may be 0 in keypoints mode.
        """
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if (gray.shape[1], gray.shape[0]) != self.config.image_size:
            gray = cv2.resize(gray, self.config.image_size, interpolation=cv2.INTER_AREA)
        if self.config.mode is SamplingMode.DENSE:
            keypoints, descriptors = self._sift.compute(gray, self._grid)
        else:
            keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, DESCRIPTOR_SIZE), np.float32)
        points = np.array([kp.pt for kp in keypoints], np.float32)
        descriptors = descriptors.astype(np.float32)
        if self.config.root_sift:
            descriptors /= descriptors.sum(axis=1, keepdims=True) + 1e-7
            descriptors = np.sqrt(descriptors)
        return points, descriptors


class Vocabulary:
    """Visual words: K-Means centres of training descriptors.

    Args:
        words: ``(K, 128)`` cluster centres.
    """

    def __init__(self, words: np.ndarray) -> None:
        self.words = np.ascontiguousarray(words, np.float32)
        if self.words.ndim != 2 or len(self.words) < 2:
            raise ValueError("A vocabulary needs at least 2 words")
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    def __len__(self) -> int:
        return len(self.words)

    @classmethod
    def fit(
        cls,
        descriptors: np.ndarray | Iterable[np.ndarray],
        size: int,
        *,
        max_descriptors: int = 25_000,
        attempts: int = 2,
        max_iterations: int = 50,
        seed: int = 0,
    ) -> Vocabulary:
        """Cluster descriptors with ``cv2.kmeans`` (k-means++ initialisation).

        Args:
            descriptors: ``(N, 128)`` array or iterable of per-image arrays.
            size: Number of visual words ``K``.
            max_descriptors: Random subsample used for clustering, to bound the training time.
            attempts: K-Means restarts; the most compact solution is kept.
            max_iterations: Iteration limit of every K-Means run.
            seed: Seed of the subsampling and of OpenCV's random generator.
        """
        if not isinstance(descriptors, np.ndarray):
            descriptors = np.concatenate([d for d in descriptors if len(d)], axis=0)
        data = np.ascontiguousarray(descriptors, np.float32)
        if len(data) < size:
            raise ValueError(f"Need at least {size} descriptors for {size} words, got {len(data)}")
        rng = np.random.default_rng(seed)
        if len(data) > max_descriptors:
            data = data[rng.choice(len(data), max_descriptors, replace=False)]
        cv2.setRNGSeed(seed)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, 1e-3)
        _, _, centres = cv2.kmeans(data, size, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        return cls(centres)

    def assign(self, descriptors: np.ndarray) -> np.ndarray:
        """Index of the nearest word of every descriptor, ``(N,)`` ``int64``."""
        if len(descriptors) == 0:
            return np.zeros(0, np.int64)
        matches = self._matcher.match(np.ascontiguousarray(descriptors, np.float32), self.words)
        words = np.empty(len(descriptors), np.int64)
        for match in matches:
            words[match.queryIdx] = match.trainIdx
        return words


class BowEncoder:
    """Spatial-pyramid bag-of-words histograms.

    Args:
        vocabulary: Visual words.
        levels: Grid sides of the pyramid, ``(1, 2)`` = whole card plus a 2x2 grid.
        image_size: ``(width, height)`` of the images the keypoint positions refer to.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        levels: Sequence[int] = (1, 2),
        image_size: tuple[int, int] = CARD_SIZE,
    ) -> None:
        if not levels or any(level < 1 for level in levels):
            raise ValueError(f"Invalid pyramid levels {levels}")
        self.vocabulary = vocabulary
        self.levels = tuple(int(level) for level in levels)
        self.image_size = image_size

    @property
    def dimension(self) -> int:
        """Length of an encoded histogram."""
        return len(self.vocabulary) * sum(level * level for level in self.levels)

    def encode(self, points: np.ndarray, descriptors: np.ndarray) -> np.ndarray:
        """Histogram of one image, ``(dimension,)`` ``float32``, divided by the descriptor count."""
        size = len(self.vocabulary)
        histogram = np.zeros(self.dimension, np.float32)
        if len(descriptors) == 0:
            return histogram
        words = self.vocabulary.assign(descriptors)
        width, height = self.image_size
        offset = 0
        for level in self.levels:
            columns = np.clip((points[:, 0] * level / width).astype(int), 0, level - 1)
            rows = np.clip((points[:, 1] * level / height).astype(int), 0, level - 1)
            bins = (rows * level + columns) * size + words
            histogram[offset : offset + level * level * size] = np.bincount(
                bins, minlength=level * level * size
            )
            offset += level * level * size
        return histogram / len(descriptors)


def _dense_grid(config: FeatureConfig) -> list[cv2.KeyPoint]:
    width, height = config.image_size
    half = config.patch_size / 2.0
    xs = np.arange(half, width - half + 1e-6, config.step)
    ys = np.arange(half, height - half + 1e-6, config.step)
    return [cv2.KeyPoint(float(x), float(y), float(config.patch_size), 0.0) for y in ys for x in xs]
