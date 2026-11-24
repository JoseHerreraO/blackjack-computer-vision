"""Real-time card pipeline: classical detection, tracking and one-off recognition per card.

Every frame (cheap, keeps the stream at full rate):

1. :class:`~blackjack_vision.detection_tracking.card_detector.CardDetector` finds the cards and
   their four corners.
2. :class:`~blackjack_vision.vision_classic.camera_motion.CameraMotion` measures how the whole
   image moved since the previous frame (head turns), and
   :class:`~blackjack_vision.detection_tracking.tracker.CardTracker` follows the cards with
   stable IDs, its predictions moved with the camera (it gets the bounding box of each card,
   the label is ``"?"`` until recognised).
3. Each tracked card that is not recognised yet gets its rectified crop added to a small buffer,
   together with its :func:`~blackjack_vision.vision_classic.sharpness.sharpness`.

Once per card (after ``min_crops`` frames, about a third of a second at 15 fps):

4. The ``fuse_top`` sharpest crops are aligned and averaged
   (:func:`~blackjack_vision.vision_classic.sharpness.fuse`), which removes most motion blur,
   noise and JPEG artefacts of a single frame.
5. The recognizer names the fused crop. If it is not sure (below its threshold) the buffer keeps
   filling and it tries again ``retry_frames`` later with fresher crops.

**Cards that merge.** When two cards come to touch (a card dealt next to another, the dealer's
hole card next to the up card), the detector sees one blob and the cards it knew vanish. The
tracker still knows where they were: their last corners, moved with the camera motion, are taken
as detections again if the blob covers them (``track_cover``). What is left of the blob, if it is
card-shaped and card-sized, is the new card (:meth:`CardPipeline._cards_from_tracks`). The
geometric split of the detector covers blobs with no history.

A recognised card keeps its name for as long as it is tracked, so recognition does not run on
every frame and the Hi-Lo count sees each card once (:attr:`PipelineResult.new_cards`).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from blackjack_vision.blackjack.cards import Card
from blackjack_vision.detection_tracking.card_detector import (
    CardDetection,
    CardDetector,
    DetectionResult,
    _mat_around,
    _portrait,
    _stretch,
)
from blackjack_vision.detection_tracking.tracker import CardTracker
from blackjack_vision.detection_tracking.yolo_detector import Detection
from blackjack_vision.vision_classic.camera_motion import CameraMotion, apply_motion
from blackjack_vision.vision_classic.mat import mat_mask
from blackjack_vision.vision_classic.rectify import rectify_card
from blackjack_vision.vision_classic.sharpness import fuse, sharpness

UNKNOWN_LABEL = "?"


class Recognizer(Protocol):
    """Anything that names a rectified card (see ``vision_dl.embedding_recognizer``)."""

    def recognise(self, card_image: np.ndarray): ...


@dataclass
class TrackedCard:
    """A card followed across frames and what is known about it."""

    track_id: int
    corners: np.ndarray | None = None
    card: Card | None = None
    similarity: float = 0.0
    attempts: int = 0
    """Recognitions tried so far."""
    frames_since_attempt: int = 0
    crops: deque[tuple[float, np.ndarray]] = field(default_factory=deque, repr=False)

    @property
    def label(self) -> str:
        return self.card.code if self.card is not None else UNKNOWN_LABEL

    @property
    def recognised(self) -> bool:
        return self.card is not None


@dataclass
class PipelineResult:
    """Output of :meth:`CardPipeline.process` for one frame."""

    cards: list[TrackedCard]
    """Cards seen in this frame, recognised or not, with their corners in this frame."""
    new_cards: list[TrackedCard]
    """Cards recognised in this frame (each card appears here once)."""
    timings_ms: dict[str, float]
    """Time spent in detection, tracking and recognition in this frame."""
    motion: np.ndarray | None = None
    """Camera motion from the previous frame to this one (``2 x 3``), ``None`` if unknown."""


class CardPipeline:
    """See the module docstring.

    Args:
        recognizer: Names rectified cards. ``None`` keeps every card as ``"?"`` (detection and
            tracking only).
        detector: Classical card detector (a new one by default).
        tracker: Card tracker. By default IoU matching only (see the comment in ``__init__``).
        motion: Camera motion estimator (a new one by default). ``False`` disables it, so the
            tracker assumes a still camera.
        min_crops: Crops collected before the first recognition of a card.
        fuse_top: Sharpest crops averaged for recognition (``1`` = only the sharpest).
        buffer_size: Crops kept per card (the oldest are dropped).
        retry_frames: Frames to wait before trying again after an unsure recognition.
        max_recognitions_per_frame: Recognitions allowed in one frame (about 25-60 ms each on a
            CPU). When many cards appear at once the rest wait for the next frames, so the frame
            time does not spike.
        track_cover: Fraction of a known card's predicted quadrilateral that a cluster must cover
            for the card to be taken from its track (see "Cards that merge" above).
    """

    def __init__(
        self,
        recognizer: Recognizer | None,
        *,
        detector: CardDetector | None = None,
        tracker: CardTracker | None = None,
        motion: CameraMotion | bool | None = None,
        min_crops: int = 5,
        fuse_top: int = 3,
        buffer_size: int = 10,
        retry_frames: int = 5,
        max_recognitions_per_frame: int = 2,
        track_cover: float = 0.85,
    ) -> None:
        if not 1 <= fuse_top <= buffer_size or not 1 <= min_crops <= buffer_size:
            raise ValueError("Need 1 <= fuse_top, min_crops <= buffer_size")
        self.recognizer = recognizer
        self.detector = detector or CardDetector()
        # Every detection reaches the tracker as "?", so its second pass (same label and close
        # centres) would hand a lost card's ID, and its name, to any nearby card. IoU only; a card
        # that is lost and found again is recognised again, and the table does not recount it.
        # With the camera motion applied, a track may wait longer (20 frames, ~1.3 s at 15 fps)
        # for a card hidden by a hand without drifting away from it.
        self.tracker = tracker or CardTracker(min_hits=1, max_center_distance=0.0, max_missed=20)
        if motion is None or motion is True:
            motion = CameraMotion()
        self.motion: CameraMotion | None = motion or None
        self.min_crops = min_crops
        self.fuse_top = fuse_top
        self.buffer_size = buffer_size
        self.retry_frames = retry_frames
        self.max_recognitions_per_frame = max_recognitions_per_frame
        self.track_cover = track_cover
        self._cards: dict[int, TrackedCard] = {}

    def reset(self) -> None:
        self.tracker.reset()
        self.detector.reset()
        if self.motion is not None:
            self.motion.reset()
        self._cards.clear()

    def process(self, image: np.ndarray) -> PipelineResult:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        detection = self.detector(image)
        detected = list(detection.cards)
        timings["detect"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        motion = self.motion(image) if self.motion is not None else None
        if motion is not None:
            # Where every known card should be now (visible ones are overwritten below).
            for state in self._cards.values():
                if state.corners is not None:
                    state.corners = apply_motion(motion, state.corners).astype(np.float32)
        timings["motion"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        detected = self._cards_from_tracks(image, detection, detected)
        timings["merge"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        detections = [
            Detection(
                np.concatenate([card.corners.min(axis=0), card.corners.max(axis=0)]).astype(float),
                UNKNOWN_LABEL,
                1.0,
            )
            for card in detected
        ]
        corners_of = {id(d): card.corners for d, card in zip(detections, detected)}
        # Corners of split or recovered cards are a guess (part of the card may be hidden), so
        # their crops are not used for recognition: the card is named from clean views.
        guessed = {id(d) for d, card in zip(detections, detected) if card.split}
        update = self.tracker.update(detections, motion)
        for track in update.removed:
            self._cards.pop(track.track_id, None)
        timings["track"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        visible: list[TrackedCard] = []
        new_cards: list[TrackedCard] = []
        budget = self.max_recognitions_per_frame
        for track in update.tracks:
            if track.missed > 0 or track.last_detection is None:
                continue
            state = self._cards.setdefault(track.track_id, TrackedCard(track.track_id))
            state.corners = corners_of[id(track.last_detection)]
            visible.append(state)
            if state.recognised or self.recognizer is None:
                continue
            if id(track.last_detection) in guessed:
                continue
            crop = rectify_card(image, state.corners)
            state.crops.append((sharpness(crop), crop))
            while len(state.crops) > self.buffer_size:
                state.crops.popleft()
            state.frames_since_attempt += 1
            if budget > 0 and self._should_try(state):
                budget -= 1
                if self._recognise(state):
                    new_cards.append(state)
        timings["recognise"] = (time.perf_counter() - start) * 1000
        return PipelineResult(cards=visible, new_cards=new_cards, timings_ms=timings, motion=motion)

    def _cards_from_tracks(
        self, image: np.ndarray, detection: DetectionResult, detected: list[CardDetection]
    ) -> list[CardDetection]:
        """The detections, with known cards hidden in clusters taken from their tracks.

        Returns ``detected`` plus the known cards found inside clusters and the card-shaped rest
        of those clusters. A detection that swallows a known card and is about twice its size is
        two cards side by side (their union can look exactly like one bigger card), so it is
        treated as a cluster too.
        """
        if self._cards and detection.mask is not None:
            known = [s.corners for s in self._cards.values() if s.corners is not None]
            merged = [card for card in detected if _swallows(card, known, self.track_cover)]
        else:
            merged = []
        blobs = list(detection.clusters) + list(detection.cut)
        config = self.detector.config
        if blobs and config.split_mat_hue is not None and image.ndim == 3:
            # Only blobs on the mat: a known card predicted onto a hand, a sleeve or the laptop
            # would otherwise be "found" there and keep its track alive as a ghost.
            mat = mat_mask(image, hue_range=config.split_mat_hue)
            blobs = [b for b in blobs if _mat_around(b, mat) >= config.split_min_mat]
        if not (blobs or merged) or not self._cards or detection.mask is None:
            return detected
        detected = [card for card in detected if not any(card is m for m in merged)]
        shape = detection.mask.shape[:2]
        clusters = np.zeros(shape, np.uint8)
        cv2.drawContours(clusters, blobs, -1, 255, -1)
        for card in merged:
            cv2.fillConvexPoly(clusters, np.round(card.corners).astype(np.int32), 255)
        boxes = [_box(card.corners) for card in detected]
        found: list[CardDetection] = []
        claimed = np.zeros(shape, np.uint8)
        for state in self._cards.values():
            if state.corners is None:
                continue
            box = _box(state.corners)
            if any(_box_overlap(box, other) > 0.4 for other in boxes):
                continue  # the detector sees it already (or another known card is there)
            if box[0] <= 2 or box[1] <= 2 or box[2] >= shape[1] - 3 or box[3] >= shape[0] - 3:
                continue  # leaving the view: only part of it is in the image
            quad = np.zeros(shape, np.uint8)
            cv2.fillConvexPoly(quad, np.round(state.corners).astype(np.int32), 255)
            area = np.count_nonzero(quad)
            if area == 0 or np.count_nonzero(quad & clusters) / area < self.track_cover:
                continue
            found.append(
                CardDetection(state.corners.copy(), np.empty((0, 1, 2), np.int32), area, True)
            )
            boxes.append(box)
            claimed |= quad
        if not found:
            return detected + merged  # nothing known inside: keep the detections as they were
        found += _rest_of_clusters(clusters, claimed, found, detection.foreshortening)
        return detected + found

    def _should_try(self, state: TrackedCard) -> bool:
        if len(state.crops) < self.min_crops:
            return False
        return state.attempts == 0 or state.frames_since_attempt >= self.retry_frames

    def _recognise(self, state: TrackedCard) -> bool:
        fused = fuse([crop for _, crop in state.crops], k=self.fuse_top)
        result = self.recognizer.recognise(fused)
        state.attempts += 1
        state.frames_since_attempt = 0
        state.similarity = float(result.similarity)
        if result.card is None:
            return False
        state.card = result.card
        state.crops.clear()  # not needed any more
        return True


def _swallows(card: CardDetection, known: list[np.ndarray], cover: float) -> bool:
    """Whether a detection holds a known card and is about twice as big (or more)."""
    polygon = card.corners.reshape(-1, 1, 2).astype(np.float32)
    area = float(cv2.contourArea(polygon))
    for corners in known:
        known_area = float(cv2.contourArea(corners.reshape(-1, 1, 2).astype(np.float32)))
        if known_area <= 0 or area < 1.6 * known_area:
            continue
        inside = sum(
            cv2.pointPolygonTest(polygon, (float(x), float(y)), True) >= -2 for x, y in corners
        )
        if inside == 4 and _box_iou(_box(card.corners), _box(corners)) > 0.2 * cover:
            return True
    return False


def _box(corners: np.ndarray) -> np.ndarray:
    return np.concatenate([corners.min(axis=0), corners.max(axis=0)]).astype(float)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _box_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the smaller of the two boxes."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / smaller if smaller > 0 else 0.0


def _rest_of_clusters(
    clusters: np.ndarray, claimed: np.ndarray, known: list[CardDetection], scale: float
) -> list[CardDetection]:
    """Card-shaped pieces of the clusters left once the known cards are taken away.

    A piece is a card if, on the table corrected for the foreshortening, its minimum-area
    rectangle has a card's aspect ratio, the piece fills it, and it is about as big as the known
    cards (the new card next to the known ones).
    """
    card_area = float(np.median([cv2.contourArea(k.corners.reshape(-1, 1, 2)) for k in known]))
    card_area /= scale  # in corrected pixels
    side = max(int(0.1 * np.sqrt(card_area)) | 1, 3)
    rest = cv2.morphologyEx(
        clusters & ~claimed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (side, side))
    )
    contours, _ = cv2.findContours(rest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = clusters.shape[:2]
    cards = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
            continue  # cut by the border: not a whole card
        corrected = _stretch(contour.reshape(-1, 2), 1.0 / scale).astype(np.float32)
        area = float(cv2.contourArea(corrected))
        if area < 0.5 * card_area:
            continue
        (cx, cy), (w, h), angle = cv2.minAreaRect(corrected)
        if min(w, h) <= 0:
            continue
        if not 1.15 <= max(w, h) / min(w, h) <= 1.9 or area / (w * h) < 0.8:
            continue
        if not 0.6 <= w * h / card_area <= 1.6:
            continue
        box = _stretch(cv2.boxPoints(((cx, cy), (w, h), angle)), scale)
        cards.append(CardDetection(_portrait(box, scale), contour, area * scale, True))
    return cards
