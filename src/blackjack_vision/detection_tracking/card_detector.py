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
5. **Shape checks**: a blob area close to the area of the fitted quadrilateral, and an aspect
   ratio around that of a playing card (63 x 88 mm, about 1.4).
6. **Foreshortening** (:func:`estimate_foreshortening`). A camera that looks at the table at an
   angle, like the glasses on the player's head, squashes the table vertically: a card lying
   upright looks almost square and one lying sideways looks twice as long as it is wide, so the
   raw aspect ratio of real cards falls anywhere between about 1.0 and 2.5. With at least
   ``foreshortening_min_cards`` candidates in the frame, the vertical scale ``c`` that makes
   their aspect ratios closest to a card's is estimated, and the aspect check (and the corner
   order) is done on the quadrilaterals stretched by ``1 / c``. A square stays square after the
   correction, so it is still rejected. With fewer candidates the detector reuses the last
   value it estimated (the head pitch changes slowly), or ``foreshortening_default`` before the
   first estimate.

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
        foreshortening: Vertical scale of the table in the image (``1`` = seen from above).
            ``None`` estimates it in every frame from the candidate cards.
        foreshortening_range: Values tried when ``foreshortening`` is estimated.
        foreshortening_min_cards: Candidates needed to estimate it. With fewer, the last
            estimate is reused, or ``foreshortening_default`` before the first one.
        foreshortening_default: Value used before anything was estimated (``1`` = none; about
            ``0.5`` for the glasses looking down at the table).
        candidate_aspect_range: Raw aspect ratios a quadrilateral may have to take part in the
            estimation.
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
    foreshortening: float | None = None
    foreshortening_range: tuple[float, float] = (0.4, 1.0)
    foreshortening_min_cards: int = 3
    foreshortening_default: float = 1.0
    candidate_aspect_range: tuple[float, float] = (1.0, 3.2)

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
        if not 0.0 < self.foreshortening_default <= 1.0:
            raise ValueError(
                f"foreshortening_default must be in (0, 1], got {self.foreshortening_default}"
            )
        if self.foreshortening is not None and not 0.0 < self.foreshortening <= 1.0:
            raise ValueError(f"foreshortening must be in (0, 1], got {self.foreshortening}")
        f_low, f_high = self.foreshortening_range
        if not 0.0 < f_low < f_high <= 1.0:
            raise ValueError(f"Invalid foreshortening_range {self.foreshortening_range}")


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
        foreshortening: Vertical scale of the table used for the shape checks (``1`` = none).
    """

    cards: list[CardDetection] = field(default_factory=list)
    clusters: list[np.ndarray] = field(default_factory=list)
    mask: np.ndarray | None = None
    foreshortening: float = 1.0


class CardDetector:
    """Finds isolated playing cards on a plain coloured mat.

    Args:
        config: Detection parameters, defaults if ``None``.
    """

    def __init__(self, config: CardDetectorConfig | None = None) -> None:
        self.config = config or CardDetectorConfig()
        self._last_foreshortening: float | None = None
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
        candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not min_area <= area <= max_area:
                continue
            if config.border_margin >= 0 and _touches_border(contour, width, height, config):
                continue
            corners = self._fit_quad(contour, area)
            if corners is None:
                result.clusters.append(contour)
            else:
                candidates.append((corners, contour, area))

        scale = config.foreshortening
        if scale is None:
            estimate = estimate_foreshortening(
                [corners for corners, _, _ in candidates],
                config.foreshortening_range,
                config.foreshortening_min_cards,
                config.candidate_aspect_range,
            )
            if estimate is not None:
                self._last_foreshortening = estimate
            scale = self._last_foreshortening or config.foreshortening_default
        result.foreshortening = scale

        low, high = config.aspect_range
        for corners, contour, area in candidates:
            corrected = _stretch(corners, 1.0 / scale)
            if not low <= aspect_ratio(corrected) <= high:
                result.clusters.append(contour)
                continue
            # Order the corners on the corrected shape so that 0 -> 1 is a real short side.
            ordered = _stretch(order_corners(corrected), scale).astype(np.float32)
            result.cards.append(CardDetection(ordered, contour, area))
        result.cards.sort(key=lambda card: (card.center[0], card.center[1]))
        return result

    def reset(self) -> None:
        """Forget the last foreshortening estimate (for example when switching sources)."""
        self._last_foreshortening = None

    def _fit_quad(self, contour: np.ndarray, area: float) -> np.ndarray | None:
        """Refined corners if the blob is a well-filled quadrilateral, otherwise ``None``."""
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
        return corners

    def _blur(self, image: np.ndarray) -> np.ndarray:
        size = self.config.blur_ksize
        return cv2.GaussianBlur(image, (size, size), 0) if size else image


def estimate_foreshortening(
    quads: list[np.ndarray],
    search_range: tuple[float, float] = (0.4, 1.0),
    min_cards: int = 3,
    candidate_aspect_range: tuple[float, float] = (1.0, 3.2),
    steps: int = 61,
) -> float | None:
    """Vertical scale of the table that makes the quadrilaterals look most like cards.

    Every candidate is stretched vertically by ``1 / c`` for each ``c`` in ``search_range`` and
    the ``c`` with the smallest median ``|log(aspect / CARD_ASPECT_RATIO)|`` wins. The median keeps
    a few non-card rectangles from pulling the estimate. Returns ``None`` when fewer than
    ``min_cards`` quadrilaterals have a plausible raw aspect ratio.
    """
    low, high = candidate_aspect_range
    usable = [quad for quad in quads if low <= aspect_ratio(quad) <= high]
    if len(usable) < min_cards:
        return None
    best_scale, best_error = 1.0, float("inf")
    for scale in np.linspace(search_range[1], search_range[0], steps):
        errors = [
            abs(np.log(aspect_ratio(_stretch(quad, 1.0 / scale)) / CARD_ASPECT_RATIO))
            for quad in usable
        ]
        error = float(np.median(errors))
        if error < best_error - 1e-9:
            best_scale, best_error = float(scale), error
    return best_scale


def _stretch(corners: np.ndarray, factor: float) -> np.ndarray:
    """Scale the y coordinate of a quadrilateral by ``factor``."""
    stretched = np.asarray(corners, dtype=np.float64).reshape(-1, 2).copy()
    stretched[:, 1] *= factor
    return stretched


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
