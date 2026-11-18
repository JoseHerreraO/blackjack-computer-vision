"""Picking and combining the best views of a card seen over several frames.

The glasses move with the head, so many frames of a card are smeared by motion blur and all of
them carry noise and compression artefacts. While a card is tracked its rectified crops (all the
same size, so they line up pixel by pixel) are kept, and recognition uses the sharpest ones:

* :func:`sharpness` scores a crop by the variance of its Laplacian, which is high when edges are
  crisp and drops quickly with blur.
* :func:`fuse` averages the ``k`` sharpest crops after turning each one so it matches the
  sharpest (a card can come out rotated by 180° from one frame to the next). Averaging aligned
  views keeps the detail they share and cancels the noise and JPEG blocks they do not.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian of ``image`` (BGR or grayscale). Higher means sharper."""
    return float(cv2.Laplacian(_gray(image), cv2.CV_64F).var())


def align_to(reference: np.ndarray, image: np.ndarray) -> np.ndarray:
    """``image`` as is or rotated 180°, whichever correlates better with ``reference``."""
    if image.shape != reference.shape:
        raise ValueError(f"Crops must have the same shape, got {image.shape} and {reference.shape}")
    rotated = cv2.rotate(image, cv2.ROTATE_180)
    ref = _gray(reference).astype(np.float32)

    def score(candidate: np.ndarray) -> float:
        return float(
            cv2.matchTemplate(_gray(candidate).astype(np.float32), ref, cv2.TM_CCOEFF_NORMED)[0, 0]
        )

    return rotated if score(rotated) > score(image) else image


def fuse(crops: Sequence[np.ndarray], k: int = 3) -> np.ndarray:
    """Average of the ``k`` sharpest crops, each aligned to the sharpest one.

    With ``k = 1`` this is just the sharpest crop.
    """
    if not crops:
        raise ValueError("fuse needs at least one crop")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    ranked = sorted(crops, key=sharpness, reverse=True)[:k]
    reference = ranked[0]
    if len(ranked) == 1:
        return reference.copy()
    stack = np.stack([reference] + [align_to(reference, crop) for crop in ranked[1:]])
    return np.clip(stack.astype(np.float32).mean(axis=0) + 0.5, 0, 255).astype(np.uint8)
