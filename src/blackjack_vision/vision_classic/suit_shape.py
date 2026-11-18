"""Suit recognition from the shape of the suit symbol (Tema 4: shape descriptors).

Every card prints a small suit symbol under the rank in its two index corners. The symbol is
segmented and described with shape features that do not depend on its position or size:

* the first two Hu moment invariants (log-scaled), which summarise the mass distribution (the
  odd-order ones are close to zero for symmetric symbols, so their logarithm is unstable);
* solidity (area / convex hull area): low for the three lobes of a club, high for a diamond;
* aspect ratio and extent (area / bounding box area) of the symbol;
* the width of the symbol at five heights, which tells a pointed top (spade, diamond) from a
  rounded one (heart, club).

An RBF SVM on the standardised features classifies each corner, and the scores of both corners
are added. The same features can describe the large central pip of an ace (security system).
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from blackjack_vision.blackjack.cards import Suit

SYMBOL_UPSCALE = 3
MIN_SYMBOL_AREA = 0.01
PROFILE_HEIGHTS = (0.1, 0.25, 0.5, 0.75, 0.9)
N_FEATURES = 2 + 3 + len(PROFILE_HEIGHTS)


def ink_mask(image: np.ndarray, upscale: int = SYMBOL_UPSCALE) -> np.ndarray:
    """Binary mask (0/255) of the printed ink, magnified ``upscale`` times.

    The ink of both colours is dark in the minimum of the B, G and R channels, so a single Otsu
    threshold on it separates red and black ink from the paper.
    """
    gray = image.min(axis=2) if image.ndim == 3 else image
    if upscale > 1:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def suit_symbol_mask(corner: np.ndarray, upscale: int = SYMBOL_UPSCALE) -> np.ndarray | None:
    """Mask of the suit symbol of an upright index corner, or ``None`` if none is found.

    The symbol is the lowest sizeable ink blob that does not touch the corner's border (the rank
    sits above it; blobs cut by the crop are fragments of the artwork).
    """
    mask = ink_mask(corner, upscale)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    height, width = mask.shape
    best = None
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < MIN_SYMBOL_AREA * height * width:
            continue
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue
        if best is None or centroids[label][1] > centroids[best][1]:
            best = label
    if best is None:
        return None
    return np.where(labels == best, 255, 0).astype(np.uint8)


def shape_features(mask: np.ndarray) -> np.ndarray:
    """Translation- and scale-invariant shape descriptor of a single-blob mask.

    Returns:
        ``(N_FEATURES,)`` ``float64`` vector: 2 log Hu moments, solidity, aspect ratio, extent
        and the relative width at :data:`PROFILE_HEIGHTS`.

    Raises:
        ValueError: if the mask is empty.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Empty mask")
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    x, y, w, h = cv2.boundingRect(contour)
    hu = cv2.HuMoments(cv2.moments(mask, binaryImage=True)).ravel()[:2]
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    filled = mask > 0
    profile = [filled[y + int(f * (h - 1)), x : x + w].sum() / w for f in PROFILE_HEIGHTS]
    return np.concatenate([hu, [area / max(hull_area, 1.0), w / h, area / (w * h)], profile])


class SuitShapeClassifier:
    """RBF SVM on the shape features of suit symbols.

    Args:
        c: SVM regularisation constant.
    """

    def __init__(self, c: float = 10.0) -> None:
        self.c = c
        self.model: Pipeline | None = None

    @property
    def classes(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("The suit classifier is not trained")
        return self.model.classes_

    def fit(self, corners: Sequence[Sequence[np.ndarray]], suits: Sequence[Suit]):
        """Train on the index corners of labelled cards.

        Args:
            corners: For every card, its upright index corners (see ``index_corners``).
            suits: Suit of every card.

        Raises:
            ValueError: if fewer than two suits have a readable symbol.
        """
        features, labels = [], []
        for card_corners, suit in zip(corners, suits, strict=True):
            for vector in self.corner_features(card_corners):
                features.append(vector)
                labels.append(suit.value)
        if len(set(labels)) < 2:
            raise ValueError("Suit symbols of at least two suits are needed")
        self.model = make_pipeline(StandardScaler(), SVC(C=self.c, gamma="scale"))
        self.model.fit(np.array(features), labels)
        return self

    def scores(self, corners: Sequence[np.ndarray]) -> np.ndarray | None:
        """Per-suit scores of one card (summed over its corners), ordered as :attr:`classes`.

        Returns ``None`` when no corner shows a readable symbol.
        """
        if self.model is None:
            raise RuntimeError("The suit classifier is not trained")
        features = self.corner_features(corners)
        if not features:
            return None
        scores = self.model.decision_function(np.array(features))
        if scores.ndim == 1:  # two suits: a single signed distance per corner
            scores = np.column_stack([-scores, scores])
        return scores.sum(axis=0)

    @staticmethod
    def corner_features(corners: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Shape features of the symbol of every corner where one is found."""
        masks = [suit_symbol_mask(corner) for corner in corners]
        return [shape_features(mask) for mask in masks if mask is not None]
