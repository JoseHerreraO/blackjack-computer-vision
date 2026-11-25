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


def mat_far_edge(
    image: np.ndarray,
    hue_range: tuple[int, int] = MAT_HUE_RANGE,
    *,
    step: int = 4,
    min_run: int = 3,
    max_slope: float = 0.6,
) -> np.ndarray:
    """Row of the far edge of the mat in every column of the image (``-1`` if not found).

    The glasses look at the table from the player's seat, so the far edge of the mat is its top
    edge in the image. It moves with the head, but the cards keep their place relative to it: the
    dealer's cards lie just below it and the player's further down (see
    :meth:`blackjack_vision.blackjack.table.Table.zone_of`).

    The mat mask is computed on an image reduced ``step`` times and only its largest connected
    piece is kept (a blue jumper behind the table has nearly the mat's hue, but it is a separate,
    ragged piece). The first row of every column that starts ``min_run`` mat rows in a row is a
    point of the mat's outline, and a nearly horizontal line (``|slope| <= max_slope``) is fitted
    to those points with RANSAC: the far edge, extended over the whole width. Returns the line's
    row in every column, or ``-1`` everywhere when no such line is found.
    """
    small = cv2.resize(image, None, fx=1.0 / step, fy=1.0 / step, interpolation=cv2.INTER_AREA)
    raw = mat_mask(small, hue_range=hue_range)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=4)
    if count <= 1:
        return np.full(image.shape[1], -1.0)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = labels == largest
    height = mask.shape[0]
    # A row starts a run when it and the next ``min_run - 1`` rows are all mat.
    run = mask.copy()
    for k in range(1, min_run):
        run[:-k] &= mask[k:]
        run[height - k :] = False
    found = run.any(axis=0)
    first = run.argmax(axis=0).astype(np.float64)
    xs = np.nonzero(found)[0].astype(np.float64)
    ys = first[found]
    line = _far_edge_line(xs, ys, max_slope, tolerance=2.0)
    if line is None:
        return np.full(image.shape[1], -1.0)
    slope, intercept = line
    columns = (np.arange(image.shape[1]) + 0.5) / step
    return (slope * columns + intercept) * step


def _far_edge_line(
    xs: np.ndarray, ys: np.ndarray, max_slope: float, tolerance: float, iterations: int = 200
) -> tuple[float, float] | None:
    """RANSAC line through the top boundary points that is nearly horizontal.

    Near the sides of the image the first mat row of a column is the mat's side edge, which
    runs steeply down the image; only the far edge is nearly horizontal from the player's seat.
    """
    if len(xs) < 10:
        return None
    rng = np.random.default_rng(0)
    best: tuple[float, float] | None = None
    best_inliers = 0
    for _ in range(iterations):
        i, j = rng.choice(len(xs), 2, replace=False)
        if xs[i] == xs[j]:
            continue
        slope = (ys[j] - ys[i]) / (xs[j] - xs[i])
        if abs(slope) > max_slope:
            continue
        intercept = ys[i] - slope * xs[i]
        inliers = np.abs(ys - (slope * xs + intercept)) <= tolerance
        count = int(inliers.sum())
        if count > best_inliers:
            best_inliers, best = count, (slope, intercept)
    if best is None or best_inliers < 0.2 * len(xs):
        return None
    inliers = np.abs(ys - (best[0] * xs + best[1])) <= tolerance
    slope, intercept = np.polyfit(xs[inliers], ys[inliers], 1)
    return float(slope), float(intercept)
