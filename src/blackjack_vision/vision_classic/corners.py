"""Card vertices: canonical ordering and sub-pixel refinement (Tema 3).

``approxPolyDP`` returns the four vertices of a card in an arbitrary order and, because real cards
have rounded corners, slightly inside the true (virtual, sharp) corners. Both problems are fixed
here before computing the homography:

* :func:`order_corners` sorts the vertices clockwise starting at the top-left one, with the first
  edge being a *short* side of the card, so the rectified card is always upright (portrait).
* :func:`refine_corners` fits a straight line to the contour points of every side, ignoring the
  rounded ends, and intersects consecutive lines. The intersections are the sharp corners of the
  card, which is the rectangle the homography must map.
"""

from __future__ import annotations

import cv2
import numpy as np


def order_corners(quad: np.ndarray) -> np.ndarray:
    """Order the four vertices of a card as top-left, top-right, bottom-right, bottom-left.

    "Top-left" refers to the card in portrait orientation: the edge ``0 -> 1`` is always one of
    the two short sides, and of those the upper one in the image (the left one if both are at the
    same height). The order is clockwise as seen on screen. A card has 180-degree symmetry, so the
    remaining ambiguity (which short side is "top") does not matter for recognition.

    Args:
        quad: Four points, shape ``(4, 2)`` or ``(4, 1, 2)`` (the output of ``approxPolyDP``).

    Returns:
        ``(4, 2)`` ``float32`` array.

    Raises:
        ValueError: if ``quad`` does not hold exactly four points.
    """
    points = np.asarray(quad, np.float64).reshape(-1, 2)
    if len(points) != 4:
        raise ValueError(f"Expected 4 corners, got {len(points)}")
    centre = points.mean(axis=0)
    # With the y axis pointing down, increasing atan2 angles run clockwise on screen.
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    points = points[np.argsort(angles)]
    edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if edges[0] + edges[2] > edges[1] + edges[3]:
        points = np.roll(points, -1, axis=0)  # make 0 -> 1 a short side
    first = (points[0] + points[1]) / 2.0
    second = (points[2] + points[3]) / 2.0
    tolerance = 1e-6 * max(1.0, float(np.abs(points).max()))
    if second[1] < first[1] - tolerance or (
        abs(second[1] - first[1]) <= tolerance and second[0] < first[0]
    ):
        points = np.roll(points, -2, axis=0)
    return points.astype(np.float32)


def side_lengths(corners: np.ndarray) -> np.ndarray:
    """Lengths of the edges ``0 -> 1``, ``1 -> 2``, ``2 -> 3`` and ``3 -> 0``."""
    points = np.asarray(corners, np.float64).reshape(-1, 2)
    return np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)


def aspect_ratio(corners: np.ndarray) -> float:
    """Mean long side over mean short side of an ordered quadrilateral (``>= 1``)."""
    edges = side_lengths(corners)
    short, long = sorted((edges[0] + edges[2], edges[1] + edges[3]))
    return float(long / short) if short > 0 else float("inf")


def refine_corners(
    contour: np.ndarray,
    corners: np.ndarray,
    *,
    trim: float = 0.15,
    max_distance: float = 0.04,
    max_shift: float = 0.2,
) -> np.ndarray:
    """Sharp corners of a card from its contour, by line fitting and intersection.

    For each side ``i`` (from corner ``i`` to corner ``i + 1``) the contour points that project
    onto its central part and lie close to it are fitted with a robust line (``cv2.fitLine`` with
    the Huber loss). Corner ``i`` is then the intersection of the lines of sides ``i - 1`` and
    ``i``. A side with too few supporting points keeps the line through its approximate vertices.

    Args:
        contour: Dense contour of the card (``CHAIN_APPROX_NONE``), shape ``(N, 1, 2)`` or
            ``(N, 2)``.
        corners: Approximate ordered corners, ``(4, 2)``.
        trim: Fraction of each side ignored at both ends (where the rounded corners are).
        max_distance: Maximum distance from a point to the approximate side, as a fraction of the
            side length (at least 2 px), for the point to be used in the fit.
        max_shift: A refined corner that moves more than this fraction of the shortest side is
            considered unreliable and the approximate corner is kept.

    Returns:
        ``(4, 2)`` ``float32`` refined corners, in the same order as ``corners``.
    """
    points = np.asarray(contour, np.float64).reshape(-1, 2)
    quad = np.asarray(corners, np.float64).reshape(4, 2)
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(4):
        start, end = quad[i], quad[(i + 1) % 4]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length == 0:
            return quad.astype(np.float32)
        unit = direction / length
        normal = np.array([-unit[1], unit[0]])
        relative = points - start
        along = relative @ unit / length
        across = np.abs(relative @ normal)
        selected = points[
            (along > trim) & (along < 1.0 - trim) & (across < max(2.0, max_distance * length))
        ]
        if len(selected) >= 5:
            vx, vy, x0, y0 = cv2.fitLine(
                selected.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
            ).ravel()
            lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
        else:
            lines.append((start, unit))

    refined = quad.copy()
    limit = max_shift * float(side_lengths(quad).min())
    for i in range(4):
        corner = _intersect(lines[i - 1], lines[i])
        if corner is not None and np.linalg.norm(corner - quad[i]) <= limit:
            refined[i] = corner
    return refined.astype(np.float32)


def _intersect(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> np.ndarray | None:
    """Intersection of two lines given as ``(point, direction)``, or ``None`` if parallel."""
    (p1, d1), (p2, d2) = first, second
    matrix = np.column_stack([d1, -d2])
    if abs(np.linalg.det(matrix)) < 1e-9:
        return None
    s, _ = np.linalg.solve(matrix, p2 - p1)
    return p1 + s * d1
