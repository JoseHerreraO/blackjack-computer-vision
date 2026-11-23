"""Is the camera looking at the table? The coloured mat seen by HSV thresholding (Tema 2).

With the camera on the player's head, a frame without cards can mean two things: the cards were
collected, or the player turned the head away. The mat tells them apart. When enough of the frame
has the mat's hue and saturation, the camera is on the table, and an empty view really means an
empty table.

Our mat is teal: on the 2026-11-18 recordings its pixels have a hue of about 87 (OpenCV's 0-180
scale) and a saturation of about 196. Frames looking at the table had at least 10 % of mat
pixels, and parts of the image without the table at most 3.4 %, hence ``MIN_MAT_FRACTION = 0.08``.
For a different mat colour, change ``hue_range``.
"""

from __future__ import annotations

import cv2
import numpy as np

MAT_HUE_RANGE = (70, 105)
MIN_SATURATION = 60
MIN_VALUE = 40
MIN_MAT_FRACTION = 0.08


def mat_mask(
    image: np.ndarray,
    hue_range: tuple[int, int] = MAT_HUE_RANGE,
    min_saturation: int = MIN_SATURATION,
    min_value: int = MIN_VALUE,
) -> np.ndarray:
    """Binary mask (0/255) of the mat-coloured pixels of a BGR image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low = (hue_range[0], min_saturation, min_value)
    high = (hue_range[1], 255, 255)
    return cv2.inRange(hsv, low, high)


def mat_fraction(image: np.ndarray, **kwargs) -> float:
    """Fraction of the image covered by the mat colour, in ``[0, 1]``."""
    return float(np.count_nonzero(mat_mask(image, **kwargs))) / (image.shape[0] * image.shape[1])


def mat_in_view(image: np.ndarray, min_fraction: float = MIN_MAT_FRACTION, **kwargs) -> bool:
    """``True`` when enough of the mat is visible for the camera to be on the table."""
    return mat_fraction(image, **kwargs) >= min_fraction
