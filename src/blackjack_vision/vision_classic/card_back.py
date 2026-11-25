"""Is a rectified card face down? The back of our deck is a blue pattern (Tema 2, HSV).

The dealer's hole card lies face down. A recogniser trained on faces still names a back as some
card (it read ours as the queen of clubs), which deals a wrong card to the dealer and moves the
count. The back is easy to tell apart by colour: on the 2026-11-25 glasses recording 94-100 % of
the inside of a back has a blue hue, against at most 1.6 % on the 2,091 face crops of our deck
(the court cards' blue details are small). Change ``hue_range`` for a deck with another back.
"""

from __future__ import annotations

import cv2
import numpy as np

BACK_HUE_RANGE = (90, 130)
"""Blue on OpenCV's 0-180 hue scale."""
MIN_SATURATION = 35
MIN_VALUE = 70
MIN_BACK_FRACTION = 0.35
MARGIN = 0.12
"""Border left out on every side (the white frame of the back, and the mat at the edges)."""


def back_fraction(card_image: np.ndarray, hue_range: tuple[int, int] = BACK_HUE_RANGE) -> float:
    """Fraction of the inside of a rectified card that has the colour of the back."""
    height, width = card_image.shape[:2]
    inner = card_image[
        int(MARGIN * height) : int((1 - MARGIN) * height),
        int(MARGIN * width) : int((1 - MARGIN) * width),
    ]
    if inner.size == 0 or inner.ndim != 3:
        return 0.0
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    back = (
        (hue >= hue_range[0]) & (hue <= hue_range[1]) & (sat > MIN_SATURATION) & (val > MIN_VALUE)
    )
    return float(np.count_nonzero(back)) / back.size


def is_card_back(
    card_image: np.ndarray,
    hue_range: tuple[int, int] = BACK_HUE_RANGE,
    min_fraction: float = MIN_BACK_FRACTION,
) -> bool:
    """``True`` when the rectified card shows its back (face down)."""
    return back_fraction(card_image, hue_range) >= min_fraction
