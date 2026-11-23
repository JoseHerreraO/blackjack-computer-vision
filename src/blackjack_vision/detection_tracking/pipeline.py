"""Real-time card pipeline: classical detection, tracking and one-off recognition per card.

Every frame (cheap, keeps the stream at full rate):

1. :class:`~blackjack_vision.detection_tracking.card_detector.CardDetector` finds the cards and
   their four corners.
2. :class:`~blackjack_vision.detection_tracking.tracker.CardTracker` follows them with stable
   IDs (it gets the bounding box of each card, the label is ``"?"`` until recognised).
3. Each tracked card that is not recognised yet gets its rectified crop added to a small buffer,
   together with its :func:`~blackjack_vision.vision_classic.sharpness.sharpness`.

Once per card (after ``min_crops`` frames, about a third of a second at 15 fps):

4. The ``fuse_top`` sharpest crops are aligned and averaged
   (:func:`~blackjack_vision.vision_classic.sharpness.fuse`), which removes most motion blur,
   noise and JPEG artefacts of a single frame.
5. The recognizer names the fused crop. If it is not sure (below its threshold) the buffer keeps
   filling and it tries again ``retry_frames`` later with fresher crops.

A recognised card keeps its name for as long as it is tracked, so recognition does not run on
every frame and the Hi-Lo count sees each card once (:attr:`PipelineResult.new_cards`).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from blackjack_vision.blackjack.cards import Card
from blackjack_vision.detection_tracking.card_detector import CardDetector
from blackjack_vision.detection_tracking.tracker import CardTracker
from blackjack_vision.detection_tracking.yolo_detector import Detection
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


class CardPipeline:
    """See the module docstring.

    Args:
        recognizer: Names rectified cards. ``None`` keeps every card as ``"?"`` (detection and
            tracking only).
        detector: Classical card detector (a new one by default).
        tracker: Card tracker (a new one by default).
        min_crops: Crops collected before the first recognition of a card.
        fuse_top: Sharpest crops averaged for recognition (``1`` = only the sharpest).
        buffer_size: Crops kept per card (the oldest are dropped).
        retry_frames: Frames to wait before trying again after an unsure recognition.
        max_recognitions_per_frame: Recognitions allowed in one frame (about 25-60 ms each on a
            CPU). When many cards appear at once the rest wait for the next frames, so the frame
            time does not spike.
    """

    def __init__(
        self,
        recognizer: Recognizer | None,
        *,
        detector: CardDetector | None = None,
        tracker: CardTracker | None = None,
        min_crops: int = 5,
        fuse_top: int = 3,
        buffer_size: int = 10,
        retry_frames: int = 5,
        max_recognitions_per_frame: int = 2,
    ) -> None:
        if not 1 <= fuse_top <= buffer_size or not 1 <= min_crops <= buffer_size:
            raise ValueError("Need 1 <= fuse_top, min_crops <= buffer_size")
        self.recognizer = recognizer
        self.detector = detector or CardDetector()
        self.tracker = tracker or CardTracker(min_hits=1)
        self.min_crops = min_crops
        self.fuse_top = fuse_top
        self.buffer_size = buffer_size
        self.retry_frames = retry_frames
        self.max_recognitions_per_frame = max_recognitions_per_frame
        self._cards: dict[int, TrackedCard] = {}

    def reset(self) -> None:
        self.tracker.reset()
        self.detector.reset()
        self._cards.clear()

    def process(self, image: np.ndarray) -> PipelineResult:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        detected = self.detector(image).cards
        timings["detect"] = (time.perf_counter() - start) * 1000

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
        update = self.tracker.update(detections)
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
        return PipelineResult(cards=visible, new_cards=new_cards, timings_ms=timings)

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
