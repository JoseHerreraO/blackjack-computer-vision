"""Card detection on a single frame with classical segmentation and contour analysis (Tema 5).

Pipeline:

1. **Segmentation** (:meth:`CardDetector.segment`). Cards are white and the mat is coloured, so a
   "whiteness" channel (the minimum of B, G and R, high only for bright *and* unsaturated pixels)
   is thresholded with Otsu. Grayscale Otsu and fixed HSV thresholds are also available. Otsu is
   never allowed below ``min_threshold``: on an empty table it would otherwise split the mat
   texture in two classes.
2. **Morphology**: an opening removes specks and thin bridges, a closing fills small gaps in the
   card outline.
3. **Contours**: external contours only (the printed symbols are holes and do not matter).
   Blobs that are too small, too large or touch the image border (cards only partly visible) are
   discarded.
4. **Quadrilateral fit**: ``approxPolyDP`` on the convex hull must give exactly 4 vertices; the
   vertices are ordered and refined to the sharp corners
   (:mod:`blackjack_vision.vision_classic.corners`).
5. **Shape checks**: aspect ratio around that of a playing card (63 x 88 mm, about 1.4, wider
   range to tolerate perspective) and a blob area close to the area of the fitted quadrilateral.

Blobs large enough to hold a card that fail the shape checks are returned as *clusters*: usually
several overlapping cards of one hand, which a later stage can split.

Example::

    detector = CardDetector()
    result = detector.detect(frame.image)
    for card in result.cards:
        upright = rectify_card(frame.image, card.corners)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from blackjack_vision.vision_classic.corners import aspect_ratio, order_corners, refine_corners

CARD_ASPECT_RATIO = 88.0 / 63.0
"""Long side over short side of a standard poker-size card."""


class Segmentation(str, Enum):
    """How card pixels are separated from the mat."""

    WHITENESS = "whiteness"
    GRAY = "gray"
    HSV = "hsv"


@dataclass(frozen=True)
class CardDetectorConfig:
    """Parameters of :class:`CardDetector`.

    Attributes:
        segmentation: Segmentation method (see :class:`Segmentation`).
        blur_ksize: Side of the Gaussian blur applied before thresholding (odd, ``0`` = none).
        min_threshold: Lowest threshold Otsu may choose (``WHITENESS`` and ``GRAY``).
        hsv_max_saturation: Highest saturation of a card pixel (``HSV``).
        hsv_min_value: Lowest value (brightness) of a card pixel (``HSV``).
        open_ksize: Side of the elliptical opening kernel (``0`` = no opening).
        close_ksize: Side of the elliptical closing kernel (``0`` = no closing).
        min_area_fraction: Smallest blob, as a fraction of the image area.
        max_area_fraction: Largest blob, as a fraction of the image area.
        aspect_range: Accepted long/short side ratios measured in the image.
        min_fill: Lowest ratio between the blob area and the area of its quadrilateral (and
            vice versa); rejects non-rectangular blobs such as overlapping cards.
        approx_epsilons: ``approxPolyDP`` tolerances to try, as fractions of the hull perimeter.
        refine: Refine the vertices to the sharp corners by line fitting.
        border_margin: Blobs whose bounding box comes closer than this to the image border are
            ignored (``-1`` keeps them).
    """

    segmentation: Segmentation = Segmentation.WHITENESS
    blur_ksize: int = 5
    min_threshold: int = 100
    hsv_max_saturation: int = 60
    hsv_min_value: int = 150
    open_ksize: int = 3
    close_ksize: int = 5
    min_area_fraction: float = 0.002
    max_area_fraction: float = 0.5
    aspect_range: tuple[float, float] = (1.15, 1.9)
    min_fill: float = 0.92
    approx_epsilons: tuple[float, ...] = (0.01, 0.015, 0.02, 0.025)
    refine: bool = True
    border_margin: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "segmentation", Segmentation(self.segmentation))
        if self.blur_ksize < 0 or (self.blur_ksize and self.blur_ksize % 2 == 0):
            raise ValueError(f"blur_ksize must be 0 or odd, got {self.blur_ksize}")
        if not 0 <= self.min_area_fraction < self.max_area_fraction <= 1:
            raise ValueError("Need 0 <= min_area_fraction < max_area_fraction <= 1")
        low, high = self.aspect_range
        if not 1.0 <= low < high:
            raise ValueError(f"Invalid aspect_range {self.aspect_range}")
        if not 0.0 < self.min_fill <= 1.0:
            raise ValueError(f"min_fill must be in (0, 1], got {self.min_fill}")


@dataclass(frozen=True, eq=False)
class CardDetection:
    """One card found in a frame.

    Attributes:
        corners: ``(4, 2)`` ``float32`` corners ordered top-left, top-right, bottom-right,
            bottom-left in portrait orientation (see :func:`order_corners`).
        contour: Dense external contour of the blob, ``(N, 1, 2)`` ``int32``.
        area: Blob area in pixels.
    """

    corners: np.ndarray
    contour: np.ndarray
    area: float

    @property
    def center(self) -> np.ndarray:
        """Centroid of the four corners, ``(2,)`` ``float64``."""
        return self.corners.astype(np.float64).mean(axis=0)

    @property
    def aspect_ratio(self) -> float:
        """Long side over short side measured in the image."""
        return aspect_ratio(self.corners)


@dataclass(eq=False)
class DetectionResult:
    """Output of :meth:`CardDetector.detect`.

    Attributes:
        cards: Single, fully visible cards, sorted left to right.
        clusters: Contours of blobs big enough to hold a card that are not one card (usually
            overlapping cards).
        mask: Binary card mask after morphology (``uint8``, 0 or 255).
    """

    cards: list[CardDetection] = field(default_factory=list)
    clusters: list[np.ndarray] = field(default_factory=list)
    mask: np.ndarray | None = None


class CardDetector:
    """Finds isolated playing cards on a plain coloured mat.

    Args:
        config: Detection parameters, defaults if ``None``.
    """

    def __init__(self, config: CardDetectorConfig | None = None) -> None:
        self.config = config or CardDetectorConfig()
        self._open_kernel = _ellipse(self.config.open_ksize)
        self._close_kernel = _ellipse(self.config.close_ksize)

    def __call__(self, image: np.ndarray) -> DetectionResult:
        return self.detect(image)

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Binary mask (0/255) of card-coloured pixels, after morphology."""
        config = self.config
        if config.segmentation is Segmentation.HSV:
            image = self._blur(_as_bgr(image))
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(
                hsv,
                (0, 0, config.hsv_min_value),
                (180, config.hsv_max_saturation, 255),
            )
        else:
            if config.segmentation is Segmentation.WHITENESS and image.ndim == 3:
                channel = image.min(axis=2)
            elif image.ndim == 3:
                channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                channel = image
            channel = self._blur(np.ascontiguousarray(channel, np.uint8))
            otsu, _ = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold = max(float(otsu), float(config.min_threshold))
            _, mask = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY)
        if self._open_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        if self._close_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        return mask

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect the cards of a BGR (or grayscale) frame."""
        config = self.config
        mask = self.segment(image)
        height, width = mask.shape[:2]
        image_area = float(height * width)
        min_area = config.min_area_fraction * image_area
        max_area = config.max_area_fraction * image_area

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        result = DetectionResult(mask=mask)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not min_area <= area <= max_area:
                continue
            if config.border_margin >= 0 and _touches_border(contour, width, height, config):
                continue
            corners = self._fit_card(contour, area)
            if corners is None:
                result.clusters.append(contour)
            else:
                result.cards.append(CardDetection(corners, contour, area))
        result.cards.sort(key=lambda card: (card.center[0], card.center[1]))
        return result

    def _fit_card(self, contour: np.ndarray, area: float) -> np.ndarray | None:
        """Ordered corners if the blob is a single card, otherwise ``None``."""
        config = self.config
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        quad = None
        for epsilon in config.approx_epsilons:
            approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
            if len(approx) == 4:
                quad = approx
                break
        if quad is None:
            return None
        corners = order_corners(quad)
        if config.refine:
            corners = refine_corners(contour, corners)
        quad_area = float(cv2.contourArea(corners.reshape(-1, 1, 2)))
        if quad_area <= 0 or min(area, quad_area) / max(area, quad_area) < config.min_fill:
            return None
        low, high = config.aspect_range
        if not low <= aspect_ratio(corners) <= high:
            return None
        return corners

    def _blur(self, image: np.ndarray) -> np.ndarray:
        size = self.config.blur_ksize
        return cv2.GaussianBlur(image, (size, size), 0) if size else image


def draw_detections(
    image: np.ndarray,
    result: DetectionResult,
    *,
    card_color: tuple[int, int, int] = (0, 255, 0),
    cluster_color: tuple[int, int, int] = (0, 165, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Copy of ``image`` with the detected cards (and clusters) drawn on top.

    The first corner of every card (its top-left) is marked with a filled circle.
    """
    canvas = _as_bgr(image).copy()
    if result.clusters:
        cv2.drawContours(canvas, result.clusters, -1, cluster_color, thickness)
    for index, card in enumerate(result.cards):
        polygon = np.round(card.corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [polygon], True, card_color, thickness, cv2.LINE_AA)
        top_left = tuple(int(v) for v in np.round(card.corners[0]))
        cv2.circle(canvas, top_left, thickness + 3, card_color, -1, cv2.LINE_AA)
        centre = tuple(int(v) for v in np.round(card.center))
        cv2.putText(
            canvas, str(index), centre, cv2.FONT_HERSHEY_SIMPLEX, 0.8, card_color, 2, cv2.LINE_AA
        )
    return canvas


def _ellipse(size: int) -> np.ndarray | None:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) if size > 1 else None


def _as_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def _touches_border(
    contour: np.ndarray, width: int, height: int, config: CardDetectorConfig
) -> bool:
    x, y, w, h = cv2.boundingRect(contour)
    margin = config.border_margin
    return x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin
