"""Multi-card tracker: one Kalman filter per card and Hungarian matching on IoU.

YOLO detects cards frame by frame and has no memory, so the same card would be counted again on
every frame and a single misread would change its label. The tracker gives every card a stable
ID across frames:

1. Each track predicts where its card is now with a constant-velocity Kalman filter
   (``cv2.KalmanFilter``) on the box centre and size.
2. Predicted boxes and new detections are matched with the Hungarian algorithm
   (``scipy.optimize.linear_sum_assignment``) on ``1 - IoU``.
3. Matched tracks are corrected with their detection. Unmatched detections start new tracks and
   tracks that stay unmatched for ``max_missed`` frames are dropped, so a card hidden for a
   moment by a hand keeps its ID when it shows up again.
4. The label of a track is a confidence-weighted vote over its last ``label_window``
   detections, so one wrong frame does not change the card.

A track is *confirmed* after ``min_hits`` detections. :meth:`CardTracker.update` reports the
tracks confirmed in that frame, which is the moment to count a card once (Hi-Lo).

If this tracker falls short, ByteTrack (built into Ultralytics) is the fallback.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from blackjack_vision.detection_tracking.yolo_detector import Detection, box_iou


def _to_measurement(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in box)
    return np.array([[(x1 + x2) / 2], [(y1 + y2) / 2], [x2 - x1], [y2 - y1]], dtype=np.float32)


class BoxKalmanFilter:
    """Constant-velocity Kalman filter on a box: state ``[cx, cy, w, h, vx, vy]``.

    Velocities are in pixels per frame. The size has no velocity: cards on a table barely change
    size between frames, and a size velocity makes boxes balloon while a card is occluded.
    """

    def __init__(
        self, box: np.ndarray, *, process_noise: float = 1.0, measurement_noise: float = 4.0
    ) -> None:
        kf = cv2.KalmanFilter(6, 4)
        kf.transitionMatrix = np.eye(6, dtype=np.float32)
        kf.transitionMatrix[0, 4] = 1.0
        kf.transitionMatrix[1, 5] = 1.0
        kf.measurementMatrix = np.eye(4, 6, dtype=np.float32)
        kf.processNoiseCov = np.diag([1, 1, 1, 1, 0.5, 0.5]).astype(np.float32) * process_noise
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * measurement_noise
        kf.errorCovPost = np.diag([10, 10, 10, 10, 100, 100]).astype(np.float32)
        kf.statePost = np.vstack([_to_measurement(box), np.zeros((2, 1), np.float32)])
        self._kf = kf

    def predict(self) -> np.ndarray:
        return self._box(self._kf.predict())

    def correct(self, box: np.ndarray) -> np.ndarray:
        return self._box(self._kf.correct(_to_measurement(box)))

    @property
    def box(self) -> np.ndarray:
        return self._box(self._kf.statePost)

    @property
    def velocity(self) -> tuple[float, float]:
        state = self._kf.statePost
        return float(state[4, 0]), float(state[5, 0])

    @staticmethod
    def _box(state: np.ndarray) -> np.ndarray:
        cx, cy, w, h = (float(v) for v in state[:4, 0])
        w, h = max(w, 1.0), max(h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class Track:
    """One card followed across frames."""

    track_id: int
    filter: BoxKalmanFilter = field(repr=False)
    label_window: int
    hits: int = 1
    missed: int = 0
    age: int = 1
    confirmed: bool = False
    _votes: deque[tuple[str, float]] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        self._votes = deque(self._votes, maxlen=self.label_window)

    @property
    def box(self) -> np.ndarray:
        return self.filter.box

    @property
    def label(self) -> str:
        """Most voted label, weighted by detector confidence."""
        scores: Counter[str] = Counter()
        for label, confidence in self._votes:
            scores[label] += confidence
        return scores.most_common(1)[0][0]

    @property
    def label_confidence(self) -> float:
        """Share of the vote that the winning label got, in ``[0, 1]``."""
        total = sum(confidence for _, confidence in self._votes)
        if total <= 0:
            return 0.0
        winner = self.label
        return sum(c for label, c in self._votes if label == winner) / total

    def vote(self, detection: Detection) -> None:
        self._votes.append((detection.label, detection.confidence))


@dataclass
class TrackerUpdate:
    """Result of one :meth:`CardTracker.update`."""

    tracks: list[Track]
    """Every live track, including those not seen in this frame."""
    confirmed: list[Track]
    """Tracks that became confirmed in this frame (count these cards once)."""
    removed: list[Track]
    """Tracks dropped in this frame."""

    @property
    def visible(self) -> list[Track]:
        """Confirmed tracks matched in this frame."""
        return [track for track in self.tracks if track.confirmed and track.missed == 0]


class CardTracker:
    """Tracks detections across frames (see the module docstring).

    Args:
        min_iou: Minimum IoU between a predicted box and a detection to match them.
        min_hits: Detections needed before a track is confirmed.
        max_missed: Consecutive frames a track may go unmatched before it is dropped.
        label_window: Detections that vote for the label of a track.
        process_noise: Kalman process noise (higher follows fast motion, lower smooths more).
        measurement_noise: Kalman measurement noise in pixels squared.
    """

    def __init__(
        self,
        *,
        min_iou: float = 0.2,
        min_hits: int = 3,
        max_missed: int = 10,
        label_window: int = 15,
        process_noise: float = 1.0,
        measurement_noise: float = 4.0,
    ) -> None:
        self.min_iou = min_iou
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.label_window = label_window
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(self, detections: Sequence[Detection]) -> TrackerUpdate:
        predicted = np.array([track.filter.predict() for track in self.tracks]).reshape(-1, 4)
        for track in self.tracks:
            track.age += 1

        matches, unmatched_tracks, unmatched_detections = self._match(predicted, detections)

        confirmed: list[Track] = []
        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            detection = detections[detection_index]
            track.filter.correct(detection.box)
            track.vote(detection)
            track.hits += 1
            track.missed = 0
            if not track.confirmed and track.hits >= self.min_hits:
                track.confirmed = True
                confirmed.append(track)

        for track_index in unmatched_tracks:
            self.tracks[track_index].missed += 1

        for detection_index in unmatched_detections:
            track = self._new_track(detections[detection_index])
            if self.min_hits <= 1:
                track.confirmed = True
                confirmed.append(track)

        removed = [track for track in self.tracks if track.missed > self.max_missed]
        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]
        return TrackerUpdate(tracks=list(self.tracks), confirmed=confirmed, removed=removed)

    def _match(
        self, predicted: np.ndarray, detections: Sequence[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        n_tracks, n_detections = len(predicted), len(detections)
        if n_tracks == 0 or n_detections == 0:
            return [], list(range(n_tracks)), list(range(n_detections))
        iou = box_iou(predicted, np.array([d.box for d in detections]))
        rows, cols = linear_sum_assignment(1.0 - iou)
        matches = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= self.min_iou]
        matched_tracks = {r for r, _ in matches}
        matched_detections = {c for _, c in matches}
        return (
            [(int(r), int(c)) for r, c in matches],
            [i for i in range(n_tracks) if i not in matched_tracks],
            [j for j in range(n_detections) if j not in matched_detections],
        )

    def _new_track(self, detection: Detection) -> Track:
        track = Track(
            track_id=self._next_id,
            filter=BoxKalmanFilter(
                detection.box,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
            ),
            label_window=self.label_window,
        )
        track.vote(detection)
        self._next_id += 1
        self.tracks.append(track)
        return track
