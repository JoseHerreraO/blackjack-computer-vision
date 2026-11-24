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

   When every card lies the same way the estimate is ambiguous: an upright card squashed by
   ``c`` looks exactly like a sideways card squashed by ``(88/63)^2 c``, about ``1.95 c``, so
   with the glasses (``c`` about 0.5) both 0.5 and about 1.0 explain the frame, and picking 1.0
   would rectify every card rotated by 90 degrees. The estimate is therefore pulled towards the
   previous one (or the default) by ``foreshortening_prior_weight * |log(c / previous)|``. The
   pull is small against real evidence (cards in both orientations), so a real change of head
   pitch still moves it, but it decides between near-equal explanations. With the glasses use
   :meth:`CardDetectorConfig.for_glasses`, which starts from 0.5.

   A single frame can still mislead the estimate: a double blob of two touching cards looks like
   one card for the wrong ``c``. A jump of the estimate (more than ``foreshortening_jump`` in
   log scale) is therefore adopted only after ``foreshortening_confirm_frames`` frames in a row
   agree on it. The head pitch never changes that fast, a passing blob does.

7. **Clusters** (:func:`~blackjack_vision.vision_classic.split_cards.split_cluster`). Blobs large
   enough to hold a card that fail the shape checks are usually two or more touching or
   overlapping cards. They are split into single cards from the whole sides of their outline,
   on the shape corrected for the foreshortening. What cannot be split stays in
   ``DetectionResult.clusters``.

Example::

    detector = CardDetector()
    result = detector.detect(frame.image)
    for card in result.cards:
        upright = rectify_card(frame.image, card.corners)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

import cv2
import numpy as np

from blackjack_vision.vision_classic.corners import (
    aspect_ratio,
    order_corners,
    refine_corners,
    side_lengths,
)
from blackjack_vision.vision_classic.mat import MAT_HUE_RANGE, mat_mask
from blackjack_vision.vision_classic.split_cards import SplitConfig, split_cluster

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
        foreshortening_default: Value used before anything was estimated, and the first
            prior (``1`` = seen from above; about ``0.5`` for the glasses looking down at the
            table, see :meth:`for_glasses`).
        foreshortening_prior_weight: How strongly the estimate is pulled towards the previous
            one (``0`` = every frame on its own).
        candidate_aspect_range: Raw aspect ratios a quadrilateral may have to take part in the
            estimation.
        foreshortening_jump: Change of the estimate, as ``|log(new / current)|``, that counts as
            a jump and needs confirming.
        foreshortening_confirm_frames: Frames in a row that must agree on a jump before it is
            adopted (``1`` adopts every estimate at once).
        split_clusters: Split blobs of touching or overlapping cards into single cards.
        split: Parameters of the cluster split.
        split_mat_hue: Hue range of the mat (OpenCV 0-180). When set, only clusters lying on
            the mat (at least ``split_min_mat`` of a thin ring around them has the mat colour)
            are split: hands, faces and objects off the table are skipped, which saves time and
            false cards. ``None`` splits every cluster.
        split_min_mat: See ``split_mat_hue``.
        split_size_ratio: A card split from a cluster is kept only if its short side is within
            this factor of the single card found nearest to it in height in the same frame
            (cards at the same distance from the camera have the same size). It rejects strips
            of a single card cut the wrong way.
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
    foreshortening_prior_weight: float = 0.1
    candidate_aspect_range: tuple[float, float] = (1.0, 3.2)
    foreshortening_jump: float = 0.25
    foreshortening_confirm_frames: int = 5
    split_clusters: bool = True
    split: SplitConfig = field(default_factory=SplitConfig)
    split_mat_hue: tuple[int, int] | None = None
    split_min_mat: float = 0.5
    split_size_ratio: float = 1.25

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
        if self.foreshortening_prior_weight < 0:
            raise ValueError("foreshortening_prior_weight must be >= 0")
        if self.foreshortening is not None and not 0.0 < self.foreshortening <= 1.0:
            raise ValueError(f"foreshortening must be in (0, 1], got {self.foreshortening}")
        if self.foreshortening_confirm_frames < 1:
            raise ValueError("foreshortening_confirm_frames must be >= 1")
        f_low, f_high = self.foreshortening_range
        if not 0.0 < f_low < f_high <= 1.0:
            raise ValueError(f"Invalid foreshortening_range {self.foreshortening_range}")

    @classmethod
    def for_glasses(cls, **overrides: object) -> CardDetectorConfig:
        """Settings for the Meta glasses looking down at the table (table squashed to ~0.5).

        Clusters are split only on our teal mat (see :mod:`blackjack_vision.vision_classic.mat`).
        """
        defaults = {"foreshortening_default": 0.5, "split_mat_hue": MAT_HUE_RANGE}
        return cls(**{**defaults, **overrides})


VIEWS = ("glasses", "top")
"""Camera views: the glasses looking down at the table, or a camera above it."""


def default_view(source: object) -> str:
    """``"glasses"`` for the glasses stream (``bridge[:port]``), ``"top"`` for anything else."""
    return "glasses" if str(source).strip().lower().startswith("bridge") else "top"


def config_for_view(view: str, **overrides: object) -> CardDetectorConfig:
    """Detector settings for a camera view (see :data:`VIEWS`).

    With the glasses a single card on the table gives no foreshortening estimate, so the
    detector must start from the glasses' value (:meth:`CardDetectorConfig.for_glasses`);
    with the default 1.0 an upright card comes out rotated by 90 degrees and a sideways one is
    not found.
    """
    if view == "glasses":
        return CardDetectorConfig.for_glasses(**overrides)
    if view == "top":
        return CardDetectorConfig(**overrides)
    raise ValueError(f"Unknown view {view!r}, expected one of {VIEWS}")


@dataclass(frozen=True, eq=False)
class CardDetection:
    """One card found in a frame.

    Attributes:
        corners: ``(4, 2)`` ``float32`` corners ordered top-left, top-right, bottom-right,
            bottom-left in portrait orientation (see :func:`order_corners`).
        contour: Dense external contour of the blob, ``(N, 1, 2)`` ``int32`` (for a card split
            from a cluster, the contour of the whole cluster).
        area: Blob area in pixels (for a split card, the area of its quadrilateral).
        split: Whether the card was split from a cluster of touching or overlapping cards.
    """

    corners: np.ndarray
    contour: np.ndarray
    area: float
    split: bool = False

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
        cut: Contours of blobs cut by the image border (not cards themselves, but they may hide a
            known card next to a cut one, see the pipeline).
    """

    cards: list[CardDetection] = field(default_factory=list)
    clusters: list[np.ndarray] = field(default_factory=list)
    mask: np.ndarray | None = None
    foreshortening: float = 1.0
    cut: list[np.ndarray] = field(default_factory=list)


class CardDetector:
    """Finds isolated playing cards on a plain coloured mat.

    Args:
        config: Detection parameters, defaults if ``None``.
    """

    def __init__(self, config: CardDetectorConfig | None = None) -> None:
        self.config = config or CardDetectorConfig()
        self._last_foreshortening: float | None = None
        self._pending_foreshortening: tuple[float, int] | None = None
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
        # Blobs cut by the border: a card only partly visible is no card, but a blob of touching
        # cards cut by the border may still hold whole cards (split below, the cut ones dropped).
        border_blobs: list[np.ndarray] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not min_area <= area <= max_area:
                continue
            if config.border_margin >= 0 and _touches_border(contour, width, height, config):
                border_blobs.append(contour)
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
                prior=self._last_foreshortening or config.foreshortening_default,
                prior_weight=config.foreshortening_prior_weight,
            )
            if estimate is not None:
                self._accept_foreshortening(estimate)
            scale = self._last_foreshortening or config.foreshortening_default
        result.foreshortening = scale
        result.cut = border_blobs

        low, high = config.aspect_range
        quad_blobs: list[np.ndarray] = []  # clean quadrilaterals with the wrong aspect ratio
        for corners, contour, area in candidates:
            corrected = _stretch(corners, 1.0 / scale)
            if not low <= aspect_ratio(corrected) <= high:
                result.clusters.append(contour)
                quad_blobs.append(contour)
                continue
            result.cards.append(CardDetection(_portrait(corners, scale), contour, area))

        if config.split_clusters:
            kept = []
            mat = None
            singles = [card.corners for card in result.cards]
            for contour in result.clusters + border_blobs:
                cut = any(contour is blob for blob in border_blobs)
                if config.split_mat_hue is not None and image.ndim == 3:
                    if mat is None:
                        mat = mat_mask(image, hue_range=config.split_mat_hue)
                    if _mat_around(contour, mat) < config.split_min_mat:
                        if not cut:
                            kept.append(contour)
                        continue
                split_config = config.split
                if cut:
                    # Part of the blob is outside the image, so it cannot be fully explained;
                    # one whole card is enough, but it must follow the edges closely.
                    split_config = replace(
                        split_config,
                        min_cards=1,
                        min_coverage=0.0,
                        min_support=max(split_config.min_support, 0.75),
                    )
                elif any(contour is blob for blob in quad_blobs):
                    # A clean quadrilateral with the wrong aspect ratio is not split: cards side
                    # by side and one card seen at an odd angle look the same (a 2:1.4 block is
                    # two upright cards or one sideways card), so guessing would invent cards.
                    kept.append(contour)
                    continue
                parts = split_cluster(contour, scale, image, split_config)
                if cut:
                    parts = [q for q in parts if not _quad_near_border(q, width, height, config)]
                parts = [q for q in parts if _size_fits(q, singles, scale, config.split_size_ratio)]
                if not parts:
                    if not cut:
                        kept.append(contour)
                    continue
                for corners in parts:
                    quad_area = float(cv2.contourArea(corners.reshape(-1, 1, 2)))
                    card = CardDetection(_portrait(corners, scale), contour, quad_area, True)
                    result.cards.append(card)
            result.clusters = kept
        result.cards.sort(key=lambda card: (card.center[0], card.center[1]))
        return result

    def reset(self) -> None:
        """Forget the last foreshortening estimate (for example when switching sources)."""
        self._last_foreshortening = None
        self._pending_foreshortening = None

    def _accept_foreshortening(self, estimate: float) -> None:
        """Adopt an estimate at once if it is close to the current one, a jump once confirmed."""
        config = self.config
        current = self._last_foreshortening
        if current is None or abs(np.log(estimate / current)) <= config.foreshortening_jump:
            self._last_foreshortening = estimate
            self._pending_foreshortening = None
            return
        pending = self._pending_foreshortening
        if pending is not None and abs(np.log(estimate / pending[0])) <= config.foreshortening_jump:
            count = pending[1] + 1
        else:
            count = 1
        if count >= config.foreshortening_confirm_frames:
            self._last_foreshortening = estimate
            self._pending_foreshortening = None
        else:
            self._pending_foreshortening = (estimate, count)

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
    prior: float | None = None,
    prior_weight: float = 0.1,
) -> float | None:
    """Vertical scale of the table that makes the quadrilaterals look most like cards.

    Every candidate is stretched vertically by ``1 / c`` for each ``c`` in ``search_range`` and
    the ``c`` with the smallest median ``|log(aspect / CARD_ASPECT_RATIO)|`` wins. The median keeps
    a few non-card rectangles from pulling the estimate. With a ``prior`` (the previous estimate)
    ``prior_weight * |log(c / prior)|`` is added, which settles the ambiguity of frames where all
    the cards lie the same way (see the module docstring). Returns ``None`` when fewer than
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
        if prior is not None and prior_weight > 0:
            error += prior_weight * abs(float(np.log(scale / prior)))
        if error < best_error - 1e-9:
            best_scale, best_error = float(scale), error
    return best_scale


def _portrait(corners: np.ndarray, scale: float) -> np.ndarray:
    """Corners ordered on the shape corrected by ``1 / scale``, so that 0 -> 1 is a short side."""
    return _stretch(order_corners(_stretch(corners, 1.0 / scale)), scale).astype(np.float32)


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


def _size_fits(
    corners: np.ndarray, singles: list[np.ndarray], scale: float, max_ratio: float
) -> bool:
    """Whether a split card is about as big as the single card nearest to it in height."""
    if not singles:
        return True
    y = float(corners[:, 1].mean())
    nearest = min(singles, key=lambda other: abs(float(other[:, 1].mean()) - y))
    ratio = _short(corners, scale) / _short(nearest, scale)
    return 1.0 / max_ratio <= ratio <= max_ratio


def _short(corners: np.ndarray, scale: float) -> float:
    sides = side_lengths(_stretch(corners, 1.0 / scale))
    return float(min(sides))


def _mat_around(contour: np.ndarray, mat: np.ndarray, width: int = 6) -> float:
    """Fraction of a ring ``width`` pixels wide around the blob that is mat."""
    x, y, w, h = cv2.boundingRect(contour)
    x0, y0 = max(x - width, 0), max(y - width, 0)
    x1, y1 = min(x + w + width, mat.shape[1]), min(y + h + width, mat.shape[0])
    blob = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.drawContours(blob, [contour - np.array([x0, y0], np.int32)], -1, 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * width + 1, 2 * width + 1))
    ring = cv2.dilate(blob, kernel) & ~blob
    total = np.count_nonzero(ring)
    return float(np.count_nonzero(ring & mat[y0:y1, x0:x1])) / total if total else 0.0


def _quad_near_border(
    corners: np.ndarray, width: int, height: int, config: CardDetectorConfig
) -> bool:
    margin = max(config.border_margin, 0) + 2
    x, y = corners[:, 0], corners[:, 1]
    return bool(
        x.min() <= margin
        or y.min() <= margin
        or x.max() >= width - 1 - margin
        or y.max() >= height - 1 - margin
    )


def _touches_border(
    contour: np.ndarray, width: int, height: int, config: CardDetectorConfig
) -> bool:
    x, y, w, h = cv2.boundingRect(contour)
    margin = config.border_margin
    return x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin
