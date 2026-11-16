"""Perspective rectification of a detected card with a homography (Tema 3).

The four ordered corners of a card (see :mod:`blackjack_vision.vision_classic.corners`) are mapped
to the corners of a fixed-size upright image, so every card reaches the recognition stage with the
same size and orientation no matter how it lies on the table or how the camera looks at it.

The homography assumes a pinhole camera: rectify cards on undistorted frames (or undistort the
corners first with :meth:`Undistorter.undistort_points`), otherwise lens distortion bends the
sides of cards near the image borders.
"""

from __future__ import annotations

import cv2
import numpy as np

CARD_SIZE: tuple[int, int] = (200, 300)
"""``(width, height)`` of a rectified card in pixels."""


def target_corners(size: tuple[int, int] = CARD_SIZE) -> np.ndarray:
    """Corners of the rectified image (top-left, top-right, bottom-right, bottom-left)."""
    width, height = size
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)


def card_homography(corners: np.ndarray, size: tuple[int, int] = CARD_SIZE) -> np.ndarray:
    """Homography from image pixels to the rectified card.

    Args:
        corners: Ordered card corners ``(4, 2)`` in image pixels.
        size: ``(width, height)`` of the rectified card.

    Returns:
        ``3x3`` ``float64`` matrix ``H`` such that ``H @ [x, y, 1]`` lands on the rectified card.
    """
    source = np.asarray(corners, np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(source, target_corners(size))


def rectify_card(
    image: np.ndarray,
    corners: np.ndarray,
    size: tuple[int, int] = CARD_SIZE,
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Warp the card bounded by ``corners`` to an upright ``size`` image.

    Args:
        image: Frame containing the card (BGR or grayscale).
        corners: Ordered card corners ``(4, 2)``.
        size: ``(width, height)`` of the output.
        interpolation: Interpolation used by ``cv2.warpPerspective``.
    """
    homography = card_homography(corners, size)
    return cv2.warpPerspective(
        image, homography, size, flags=interpolation, borderMode=cv2.BORDER_REPLICATE
    )
