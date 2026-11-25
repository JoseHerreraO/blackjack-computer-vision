"""How the image moved between two frames: camera motion from sparse optical flow (Lab 5, Tema 4).

The camera is on the player's head, so a head turn moves every card in the image at once, often
by more than a card's own width between two frames. A tracker that predicts each card from its
own past motion then loses all of them together and starts new tracks.

The motion of the whole image is estimated between consecutive frames and applied to the
tracker's predictions before matching:

1. The frame is reduced to a small grayscale image (``width`` pixels wide), which is enough for
   the global motion and keeps the cost at a few milliseconds.
2. Shi-Tomasi corners (``goodFeaturesToTrack``) of the previous frame are followed into the new
   one with pyramidal Lucas-Kanade (``calcOpticalFlowPyrLK``), and checked backwards (a point
   that does not come back to where it started is dropped).
3. A similarity transform (rotation, uniform scale and translation) is fitted with RANSAC
   (``estimateAffinePartial2D``). The table, the mat and the cards on it are most of the points
   and move together, so a moving hand is left out as outliers.

When there are too few points or inliers, the motion is unknown and ``None`` is returned, which
the tracker reads as "no camera motion".
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraMotionConfig:
    """Parameters of :class:`CameraMotion`.

    Attributes:
        width: Width of the reduced frame the flow is computed on.
        max_corners: Corners followed between frames.
        quality_level: ``goodFeaturesToTrack`` quality level.
        min_distance: Minimum distance between corners, in reduced pixels.
        max_back_error: Largest forward-backward error of a followed point, in reduced pixels.
        ransac_threshold: RANSAC reprojection threshold, in reduced pixels.
        min_inliers: Fewest inliers for the motion to be trusted.
    """

    width: int = 320
    max_corners: int = 200
    quality_level: float = 0.01
    min_distance: int = 8
    max_back_error: float = 1.0
    ransac_threshold: float = 2.0
    min_inliers: int = 20


class CameraMotion:
    """Estimates the motion of the image from one frame to the next.

    Call it once per frame, in order. It returns the ``2 x 3`` transform that maps a point of the
    previous frame to the same point in this one (full-resolution pixels), or ``None`` on the
    first frame and whenever the motion cannot be trusted.
    """

    def __init__(self, config: CameraMotionConfig | None = None) -> None:
        self.config = config or CameraMotionConfig()
        self._previous: np.ndarray | None = None
        self._scale = 1.0
        self.last_inliers = 0

    def reset(self) -> None:
        self._previous = None

    def __call__(self, image: np.ndarray) -> np.ndarray | None:
        return self.update(image)

    def update(self, image: np.ndarray) -> np.ndarray | None:
        config = self.config
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        scale = config.width / gray.shape[1] if gray.shape[1] > config.width else 1.0
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        previous, previous_scale = self._previous, self._scale
        self._previous, self._scale = small, scale
        self.last_inliers = 0
        if previous is None or previous.shape != small.shape or previous_scale != scale:
            return None

        points = cv2.goodFeaturesToTrack(
            previous, config.max_corners, config.quality_level, config.min_distance
        )
        if points is None or len(points) < config.min_inliers:
            return None
        moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, small, points, None)
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(small, previous, moved, None)
        error = np.linalg.norm((back - points).reshape(-1, 2), axis=1)
        good = (status.ravel() == 1) & (back_status.ravel() == 1) & (error <= config.max_back_error)
        if good.sum() < config.min_inliers:
            return None
        transform, inliers = cv2.estimateAffinePartial2D(
            points[good],
            moved[good],
            method=cv2.RANSAC,
            ransacReprojThreshold=config.ransac_threshold,
        )
        if transform is None or inliers is None or int(inliers.sum()) < config.min_inliers:
            return None
        self.last_inliers = int(inliers.sum())
        # Back to full-resolution pixels: x_full = x_small / s, so only the translation changes.
        transform = transform.astype(np.float64)
        transform[:, 2] /= scale
        return transform


def apply_motion(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Move ``(N, 2)`` points with a ``2 x 3`` transform."""
    points = np.asarray(points, np.float64).reshape(-1, 2)
    return points @ transform[:, :2].T + transform[:, 2]


def motion_scale(transform: np.ndarray) -> float:
    """Uniform scale of a similarity transform (``1`` = no zoom)."""
    return float(np.sqrt(abs(np.linalg.det(transform[:, :2]))))
