"""Split a blob of touching or overlapping cards into single cards (Temas 3 and 5).

When two cards touch or overlap, the card mask holds one blob for both and the quadrilateral fit
of :class:`~blackjack_vision.detection_tracking.card_detector.CardDetector` rejects it. The blob
still carries the cards' sides: on its outline, where a card borders the mat, and inside it, where
the edge of the upper card shows as a thin line over the card below (Canny).

Everything is done on the table corrected for the foreshortening of the glasses (``y / c``), where
cards are real rectangles with the card's aspect ratio:

1. The outline is simplified with ``approxPolyDP`` and every vertex is classified as a convex
   right-angle corner or not.
2. **Whole sides.** Each side of the outline between two such corners proposes two cards: that
   side as the short side of the card or as its long side, the other side ``88/63`` times longer
   or shorter, towards the inside of the blob. The size comes from the side itself, so cards nearer
   or further from the camera need no size prior. The top card of a fanned hand, and the bottom
   one, always show a whole side.
3. **Corners.** Cards in the middle of a fan may show no whole side, only a corner. Once some
   cards are known their size is used to propose a card at every right-angle corner, along its
   two sides, in both orientations.
4. A proposal is kept only if nearly all of it lies inside the blob. Its *support* is the fraction
   of its outline that runs along an edge, either the blob outline or an inner Canny edge: a real
   card follows edges on its visible sides, a wrong proposal cuts across the card faces.
5. Proposals are accepted greedily by support, skipping those that are almost the same card as an
   accepted one or that explain almost nothing new.
6. The split is returned only if the accepted cards explain most of the blob, so hands, arms and
   other non-card blobs, which seldom have straight sides and right angles, come back empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CARD_ASPECT_RATIO = 88.0 / 63.0


@dataclass(frozen=True)
class SplitConfig:
    """Parameters of :func:`split_cluster`.

    Attributes:
        approx_epsilon: ``approxPolyDP`` tolerance, as a fraction of the outline perimeter.
        angle_tolerance: Largest deviation from 90 degrees of a card corner, in degrees.
        min_side: Shortest whole side that can propose a card, as a fraction of the square root
            of the blob area (skips the short steps where two cards meet).
        min_inside: Lowest fraction of a proposal that must lie inside the blob.
        min_support: Lowest fraction of a proposal's outline that must run along an edge.
        support_distance: Distance to an edge that still counts as "along" it, as a fraction of
            the proposal's short side (at least 2 px).
        canny_thresholds: Canny thresholds for the inner edges (only with an ``image``).
        max_overlap: Proposals whose IoU with an accepted card is above this are the same card.
        min_new_support: Lowest fraction of a proposal's outline that must run along an edge
            no accepted card explains yet. The card below another one shares little area with
            it but has its own visible sides, a wrong proposal reuses the sides of accepted cards.
        min_gain: Lowest fraction of a proposal that must be blob not explained yet.
        max_size_ratio: The cards of one blob lie at about the same distance from the camera,
            so a proposal whose short side differs from the best card's by more than this
            factor is rejected (a card read in the wrong orientation is 1.4 times off).
        min_coverage: Lowest fraction of the blob that the accepted cards must explain.
        residual: Also take the card-shaped pieces left once the accepted cards are taken away
            (the second of two cards side by side whose shared outline is blurred).
        min_card_fraction: Smallest card, as a fraction of the blob area (``0`` = any size).
        max_cards: Most cards returned from one blob.
        min_cards: Fewest cards for a split to count. A blob that is one card would have been
            found by the detector itself, except a blob cut by the image border, where a whole
            card can sit next to a cut one.
    """

    approx_epsilon: float = 0.012
    angle_tolerance: float = 15.0
    min_side: float = 0.2
    min_inside: float = 0.92
    min_support: float = 0.5
    support_distance: float = 0.05
    canny_thresholds: tuple[float, float] = (30.0, 90.0)
    max_overlap: float = 0.75
    min_new_support: float = 0.25
    min_gain: float = 0.02
    max_size_ratio: float = 1.2
    min_coverage: float = 0.88
    residual: bool = True
    min_card_fraction: float = 0.0
    max_cards: int = 6
    min_cards: int = 2


@dataclass
class _Proposal:
    quad: np.ndarray
    mask: np.ndarray
    support: float
    samples: np.ndarray  # points along the outline of the proposal, (M, 2) int
    near: np.ndarray  # which samples lie near an edge, (M,) bool
    tolerance: float


class _Canvas:
    """The blob rasterised in corrected coordinates, with its edges."""

    def __init__(
        self,
        points: np.ndarray,
        scale: float,
        image: np.ndarray | None,
        config: SplitConfig,
    ) -> None:
        self.scale = scale
        corrected = points * np.array([1.0, 1.0 / scale])
        self.origin = corrected.min(axis=0) - 4.0
        size = np.ceil(corrected.max(axis=0) - self.origin + 4.0).astype(int)
        self.points = corrected - self.origin
        polygon = np.round(self.points).astype(np.int32).reshape(-1, 1, 2)
        self.blob = np.zeros((size[1], size[0]), np.uint8)
        cv2.drawContours(self.blob, [polygon], -1, 255, -1)
        self.area = float(np.count_nonzero(self.blob))

        edges = np.zeros_like(self.blob)
        cv2.drawContours(edges, [polygon], -1, 255, 1)
        if image is not None:
            edges |= self._inner_edges(image, size, config) & self.blob
        self.distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)

    def _inner_edges(self, image: np.ndarray, size: np.ndarray, config: SplitConfig) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        height, width = gray.shape[:2]
        # Bounding box of the blob in the image, with a small margin.
        x0 = max(int(np.floor(self.origin[0])), 0)
        y0 = max(int(np.floor(self.origin[1] * self.scale)), 0)
        x1 = min(int(np.ceil(self.origin[0] + size[0])), width)
        y1 = min(int(np.ceil((self.origin[1] + size[1]) * self.scale)), height)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((size[1], size[0]), np.uint8)
        crop = cv2.GaussianBlur(np.ascontiguousarray(gray[y0:y1, x0:x1]), (3, 3), 0)
        canny = cv2.Canny(crop, *config.canny_thresholds)
        # Image (x, y) -> canvas (x - ox, y / c - oy), for the crop that starts at (x0, y0).
        warp = np.array(
            [
                [1.0, 0.0, x0 - self.origin[0]],
                [0.0, 1.0 / self.scale, y0 / self.scale - self.origin[1]],
            ]
        )
        warped = cv2.warpAffine(canny, warp, (int(size[0]), int(size[1])), flags=cv2.INTER_NEAREST)
        return cv2.dilate(warped, np.ones((3, 3), np.uint8))

    def to_image(self, quad: np.ndarray) -> np.ndarray:
        return ((quad + self.origin) * np.array([1.0, self.scale])).astype(np.float32)

    def score(self, quad: np.ndarray, config: SplitConfig) -> _Proposal | None:
        mask = np.zeros_like(self.blob)
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
        # The true area, not the pixels drawn: the canvas clips what falls outside the blob.
        area = float(cv2.contourArea(quad.astype(np.float32).reshape(-1, 1, 2)))
        if area <= 0 or np.count_nonzero(mask & self.blob) / area < config.min_inside:
            return None
        short_side = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
        tolerance = max(config.support_distance * short_side, 2.0)
        samples, near = _support(quad, self.distance, tolerance)
        support = float(near.mean()) if len(near) else 0.0
        if support < config.min_support:
            return None
        return _Proposal(quad, mask > 0, support, samples, near, tolerance)


def split_cluster(
    contour: np.ndarray,
    foreshortening: float = 1.0,
    image: np.ndarray | None = None,
    config: SplitConfig | None = None,
) -> list[np.ndarray]:
    """Corners of the single cards that make up a blob, or ``[]`` if it cannot be split.

    Args:
        contour: Dense external contour of the blob, ``(N, 1, 2)``, in image coordinates.
        foreshortening: Vertical scale of the table in the image (see the card detector).
        image: The frame (BGR or gray). Its inner edges help to find cards in the middle of a
            fan; without it only the blob outline is used.
        config: Parameters, defaults if ``None``.

    Returns:
        One ``(4, 2)`` ``float32`` array per card, in image coordinates. The corners are not
        ordered; the caller orders them on the corrected shape like any other card.
    """
    config = config or SplitConfig()
    points = np.asarray(contour, np.float64).reshape(-1, 2)
    if len(points) < 8:
        return []
    # The cheap test first: without a whole side there is nothing to propose (most hands).
    approx, right = _corners(points * np.array([1.0, 1.0 / float(foreshortening)]), config)
    if len(approx) < 4 or not np.any(right & np.roll(right, -1)):
        return []
    canvas = _Canvas(points, float(foreshortening), image, config)
    if canvas.area <= 0:
        return []
    approx = approx - canvas.origin

    accepted: list[_Proposal] = []
    explained = np.zeros(canvas.blob.shape, bool)
    accepted_sides = np.zeros(canvas.blob.shape, np.uint8)
    blob = canvas.blob > 0

    def accept_from(groups: list[list[np.ndarray]]) -> None:
        nonlocal explained
        # The proposals of one side or corner exclude each other: keep the best supported one.
        proposals = []
        for group in groups:
            scored = [p for p in (canvas.score(q, config) for q in group) if p is not None]
            if scored:
                proposals.append(max(scored, key=lambda p: p.support))
        proposals.sort(key=lambda p: -p.support)
        for proposal in proposals:
            if len(accepted) >= config.max_cards:
                return
            if any(_iou(proposal.mask, other.mask) > config.max_overlap for other in accepted):
                continue
            if _area(proposal.quad) < config.min_card_fraction * canvas.area:
                continue
            if accepted:
                ratio = _short_side(proposal.quad) / _short_side(accepted[0].quad)
                if not 1.0 / config.max_size_ratio <= ratio <= config.max_size_ratio:
                    continue
            gain = float((proposal.mask & blob & ~explained).sum()) / float(proposal.mask.sum())
            if gain < config.min_gain:
                continue
            x, y = proposal.samples[:, 0], proposal.samples[:, 1]
            new = proposal.near & (accepted_sides[y, x] == 0)
            if new.mean() < config.min_new_support:
                continue
            accepted.append(proposal)
            explained |= proposal.mask
            thickness = 2 * int(np.ceil(proposal.tolerance)) + 1
            polygon = np.round(proposal.quad).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(accepted_sides, [polygon], True, 255, thickness)

    accept_from(_side_proposals(approx, right, config.min_side * np.sqrt(canvas.area)))
    quads = [p.quad for p in accepted]
    if config.residual and accepted and len(accepted) < config.max_cards:
        quads += _residual_cards(blob & ~explained, accepted, config)

    if len(quads) < config.min_cards:
        return []
    explained_all = explained.copy()
    for quad in quads[len(accepted) :]:
        mask = np.zeros(canvas.blob.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
        explained_all |= mask > 0
    if float((explained_all & blob).sum()) / canvas.area < config.min_coverage:
        return []
    return [canvas.to_image(quad) for quad in quads]


def _residual_cards(
    residual: np.ndarray, accepted: list[_Proposal], config: SplitConfig
) -> list[np.ndarray]:
    """Card-shaped pieces of the blob left once the accepted cards are taken away.

    Two cards side by side show whole sides only on the outer ends when the outline between them
    is rounded or blurred: the first card comes from its whole side, the second is what is left.
    """
    short = float(np.median([_short_side(p.quad) for p in accepted]))
    card_area = short * short * CARD_ASPECT_RATIO
    size = max(int(0.15 * short) | 1, 3)
    pieces = cv2.morphologyEx(
        residual.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)),
    )
    contours, _ = cv2.findContours(pieces, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.5 * card_area:
            continue
        (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
        if min(w, h) <= 0:
            continue
        aspect = max(w, h) / min(w, h)
        box_area = w * h
        if not 1.15 <= aspect <= 1.9 or area / box_area < 0.85:
            continue
        if not 1.0 / config.max_size_ratio**2 <= box_area / card_area <= config.max_size_ratio**2:
            continue
        cards.append(cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.float64))
    return cards


def _corners(points: np.ndarray, config: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    """Simplified outline and, for each vertex, whether it is a convex right-angle corner."""
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    epsilon = config.approx_epsilon * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float64)
    n = len(approx)
    right = np.zeros(n, bool)
    if n < 4:
        return approx, right
    # Convex vertices turn the same way as the polygon itself (sign of its shoelace area).
    x, y = approx[:, 0], approx[:, 1]
    orientation = np.sign(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    for i in range(n):
        incoming = approx[i] - approx[i - 1]
        outgoing = approx[(i + 1) % n] - approx[i]
        norm = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
        if norm <= 0:
            continue
        turn = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        # Interior angle 90 degrees <=> the direction turns by 90 degrees.
        angle = np.degrees(np.arccos(np.clip(np.dot(incoming, outgoing) / norm, -1.0, 1.0)))
        right[i] = turn * orientation > 0 and abs(angle - 90.0) <= config.angle_tolerance
    return approx, right


def _side_proposals(
    approx: np.ndarray, right: np.ndarray, min_side: float
) -> list[list[np.ndarray]]:
    """Cards built on every whole side (both ends right-angle corners), one group per side."""
    n = len(approx)
    groups = []
    for i in range(n):
        j = (i + 1) % n
        if not (right[i] and right[j]):
            continue
        start, end = approx[i], approx[j]
        length = float(np.linalg.norm(end - start))
        if length < min_side:
            continue
        direction = (end - start) / length
        normal = np.array([-direction[1], direction[0]])
        quads = []
        for depth in (length * CARD_ASPECT_RATIO, length / CARD_ASPECT_RATIO):
            for sign in (1.0, -1.0):  # the one pointing out of the blob fails the inside test
                offset = sign * normal * depth
                quads.append(np.array([start, end, end + offset, start + offset]))
        groups.append(quads)
    return groups


def _area(quad: np.ndarray) -> float:
    return float(cv2.contourArea(quad.astype(np.float32).reshape(-1, 1, 2)))


def _short_side(quad: np.ndarray) -> float:
    return float(min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1])))


def _support(
    quad: np.ndarray, distance: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Points sampled along the sides of ``quad`` (inside the canvas) and which lie near an edge.

    Points outside the canvas count as not near (they are part of the denominator only).
    """
    samples = []
    for k in range(4):
        start, end = quad[k], quad[(k + 1) % 4]
        steps = max(int(np.linalg.norm(end - start)), 2)
        t = np.linspace(0.0, 1.0, steps, endpoint=False)[:, None]
        samples.append(start + (end - start) * t)
    points = np.round(np.vstack(samples)).astype(int)
    height, width = distance.shape
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    near = distance[points[:, 1], points[:, 0]] <= tolerance
    return points, near


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 0.0
