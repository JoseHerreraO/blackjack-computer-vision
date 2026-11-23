"""Suit recognition by template matching with normalised cross-correlation (Tema 3).

Every card prints a small suit symbol under the rank in its two index corners, and an ace has a
large one in the middle. The symbol is isolated and compared with one template per suit:

1. **Thresholding**: the ink of both colours is dark in the minimum of the B, G and R channels,
   so Otsu's threshold on it separates the printed ink from the paper.
2. **Connected components**: the suit symbol is the lowest sizeable blob of an index corner (the
   rank sits above it), or the largest blob in the middle of an ace.
3. **Normalisation**: the blob is cropped to its bounding box, centred in a square and resized
   to ``TEMPLATE_SIZE``, so position and size no longer matter.
4. **Templates**: the template of a suit is the mean of the normalised symbols of that suit in
   the training cards.
5. **Matching**: normalised cross-correlation (``cv2.matchTemplate`` with ``TM_CCOEFF_NORMED``)
   between the symbol and each template; the scores of the symbols of one card are added.

As in the template-matching examples of the course, NCC works here because every symbol of a suit
has the same shape and, after the normalisation, the same scale. Only the 180-degree rotation of
an upside-down card is left, which :meth:`SuitTemplates.scores` can try explicitly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from blackjack_vision.blackjack.cards import Suit

TEMPLATE_SIZE = 32
SYMBOL_UPSCALE = 3
MIN_SYMBOL_AREA = 0.01


def ink_mask(image: np.ndarray, upscale: int = SYMBOL_UPSCALE) -> np.ndarray:
    """Binary mask (0/255) of the printed ink, magnified ``upscale`` times (Otsu threshold)."""
    gray = image.min(axis=2) if image.ndim == 3 else image
    if upscale > 1:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def suit_symbol_mask(corner: np.ndarray, upscale: int = SYMBOL_UPSCALE) -> np.ndarray | None:
    """Mask of the suit symbol of an upright index corner, or ``None`` if none is found.

    The symbol is the lowest sizeable connected component that does not touch the corner's
    border (the rank sits above it; blobs cut by the crop are fragments of the artwork).
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


def normalise_symbol(mask: np.ndarray, size: int = TEMPLATE_SIZE) -> np.ndarray:
    """Crop a symbol mask to its bounding box, centre it in a square and resize it.

    Returns:
        ``(size, size)`` ``float32`` patch in [0, 255].

    Raises:
        ValueError: if the mask is empty.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Empty symbol mask")
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    height, width = crop.shape
    side = max(height, width)
    square = np.zeros((side, side), np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    square[top : top + height, left : left + width] = crop
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def ncc(patch: np.ndarray, template: np.ndarray) -> float:
    """Normalised cross-correlation of two patches of the same size, in [-1, 1]."""
    return float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0, 0])


def corner_symbols(corners: Iterable[np.ndarray]) -> list[np.ndarray]:
    """Normalised suit symbols of the index corners where one is found."""
    masks = [suit_symbol_mask(corner) for corner in corners]
    return [normalise_symbol(mask) for mask in masks if mask is not None]


class SuitTemplates:
    """One template per suit, learned as the mean of normalised symbols.

    Args:
        size: Side of the normalised symbols and templates.
    """

    def __init__(self, size: int = TEMPLATE_SIZE) -> None:
        self.size = size
        self.templates: dict[Suit, np.ndarray] = {}

    @property
    def suits(self) -> list[Suit]:
        return list(self.templates)

    def fit(self, corners: Sequence[Sequence[np.ndarray]], suits: Sequence[Suit]) -> SuitTemplates:
        """Learn the templates from the index corners of labelled cards.

        Args:
            corners: For every card, its upright index corners (see ``index_corners``).
            suits: Suit of every card.

        Raises:
            ValueError: if fewer than two suits have a readable symbol.
        """
        samples: dict[Suit, list[np.ndarray]] = {}
        for card_corners, suit in zip(corners, suits, strict=True):
            samples.setdefault(suit, []).extend(corner_symbols(card_corners))
        samples = {suit: patches for suit, patches in samples.items() if patches}
        if len(samples) < 2:
            raise ValueError("Suit symbols of at least two suits are needed")
        self.templates = {
            suit: np.mean(patches, axis=0).astype(np.float32)
            for suit, patches in sorted(samples.items(), key=lambda item: item[0].value)
        }
        return self

    def scores(
        self,
        symbols: Sequence[np.ndarray],
        *,
        allowed: Iterable[Suit] | None = None,
        both_orientations: bool = False,
    ) -> dict[Suit, float] | None:
        """Summed NCC of ``symbols`` with every template, or ``None`` without symbols.

        Args:
            symbols: Normalised symbols of one card (see :func:`normalise_symbol`).
            allowed: Suits to consider (e.g. only the red ones), all of them by default.
            both_orientations: Also try every template turned by 180 degrees and keep the best
                (for a symbol whose orientation is unknown, like the pip of an ace).
        """
        if not self.templates:
            raise RuntimeError("The suit templates are not trained")
        if not symbols:
            return None
        suits = [suit for suit in self.templates if allowed is None or suit in set(allowed)]
        result = {}
        for suit in suits:
            template = self.templates[suit]
            candidates = [template]
            if both_orientations:
                candidates.append(cv2.rotate(template, cv2.ROTATE_180))
            result[suit] = sum(max(ncc(symbol, t) for t in candidates) for symbol in symbols)
        return result

    def classify(self, symbols: Sequence[np.ndarray], **kwargs) -> tuple[Suit | None, float]:
        """Best suit and its margin over the runner-up (``(None, 0)`` without symbols)."""
        scores = self.scores(symbols, **kwargs)
        if not scores:
            return None, 0.0
        ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        margin = ranking[0][1] - ranking[1][1] if len(ranking) > 1 else float("inf")
        return ranking[0][0], float(margin)
