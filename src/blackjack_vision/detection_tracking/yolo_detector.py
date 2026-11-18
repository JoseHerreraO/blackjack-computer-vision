"""Playing-card detection with a YOLO model (Ultralytics).

The default weights are ``lgd-cards-gen3`` (YOLO11s, 52 classes ``"10H"``, ``"AS"``, ``"KD"``...),
see ``scripts/download_card_model.py``. That model was trained on the *corner index* of each
card (the small rank and suit printed in two opposite corners), not on the whole card, so one
card usually yields two boxes with the same label. :func:`merge_corners` joins them into one
detection per card.

The glasses stream is a tall 720x1280 frame and the model runs at 640 px, so the frame is split
into overlapping tiles (one column and two rows by default). Each tile is close to the model's
native size, which keeps the small corner indices readable.

``ultralytics`` is an optional dependency (``pip install -e ".[yolo]"``) and is only imported when
a :class:`YoloCardDetector` is created, so the rest of the package works without it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = Path("models/lgd-cards-gen3/best.pt")


@dataclass(frozen=True)
class Detection:
    """One detected box in image coordinates.

    Attributes:
        box: ``(x1, y1, x2, y2)`` in pixels, float.
        label: Card code as the model names it, e.g. ``"10H"`` or ``"QS"``.
        confidence: Detector score in ``[0, 1]``.
        parts: Number of model boxes merged into this one (2 when both corners were seen).
    """

    box: np.ndarray = field(repr=False)
    label: str
    confidence: float
    parts: int = 1

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def size(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return x2 - x1, y2 - y1


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` boxes in ``x1, y1, x2, y2`` format."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1e-9), 0.0)


def non_max_suppression(detections: Sequence[Detection], iou_threshold: float) -> list[Detection]:
    """Per-label NMS, used to remove duplicates where tiles overlap."""
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
        same_label = [k.box for k in kept if k.label == detection.label]
        if same_label and box_iou(detection.box, np.array(same_label)).max() > iou_threshold:
            continue
        kept.append(detection)
    return kept


def merge_corners(detections: Sequence[Detection], link_factor: float = 6.0) -> list[Detection]:
    """Join the two corner boxes of the same card into one detection.

    Two boxes are joined when they have the same label and their centres are closer than
    ``link_factor`` times the larger box diagonal (a card is about 5-6 corner indices tall).
    The merged box spans both corners, so it covers most of the card, and keeps the best score.
    Groups are built transitively, so three boxes of one card also end up together.
    """
    remaining = list(detections)
    merged: list[Detection] = []
    while remaining:
        group = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for candidate in list(remaining):
                if candidate.label != group[0].label:
                    continue
                if any(_close(candidate, member, link_factor) for member in group):
                    group.append(candidate)
                    remaining.remove(candidate)
                    changed = True
        boxes = np.array([d.box for d in group])
        merged.append(
            Detection(
                box=np.concatenate([boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)]),
                label=group[0].label,
                confidence=max(d.confidence for d in group),
                parts=sum(d.parts for d in group),
            )
        )
    return merged


def _close(a: Detection, b: Detection, link_factor: float) -> bool:
    diagonal = max(float(np.hypot(*a.size)), float(np.hypot(*b.size)))
    distance = float(np.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]))
    return distance <= link_factor * diagonal


def tile_grid(
    width: int, height: int, columns: int, rows: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles ``(x, y, w, h)`` that cover a ``width x height`` image."""
    if columns < 1 or rows < 1:
        raise ValueError("columns and rows must be at least 1")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    def spans(length: int, count: int) -> list[tuple[int, int]]:
        if count == 1:
            return [(0, length)]
        size = int(np.ceil(length / (count - (count - 1) * overlap)))
        step = (length - size) / (count - 1)
        return [(round(i * step), size) for i in range(count)]

    return [(x, y, w, h) for y, h in spans(height, rows) for x, w in spans(width, columns)]


class YoloCardDetector:
    """YOLO card detector with tiling and corner merging.

    Args:
        weights: Ultralytics ``.pt`` (or exported) model file.
        confidence: Minimum detection score.
        iou: IoU threshold of the model's own NMS and of the cross-tile NMS.
        image_size: Inference size passed to YOLO.
        tiles: ``(columns, rows)`` grid. ``(1, 1)`` runs on the whole frame.
        overlap: Fraction of each tile shared with its neighbour, so no corner index is cut.
        merge: Join the two corners of each card (see :func:`merge_corners`).
        device: ``"cpu"``, ``"cuda:0"``... ``None`` lets Ultralytics choose.
    """

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        *,
        confidence: float = 0.4,
        iou: float = 0.5,
        image_size: int = 640,
        tiles: tuple[int, int] = (1, 2),
        overlap: float = 0.15,
        merge: bool = True,
        device: str | None = None,
    ) -> None:
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"Card model not found: {weights}. Run scripts/download_card_model.py first."
            )
        try:
            from ultralytics import YOLO
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError(
                'YoloCardDetector needs ultralytics: pip install -e ".[yolo]"'
            ) from error

        self.model = YOLO(str(weights))
        self.names: dict[int, str] = dict(self.model.names)
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.tiles = tiles
        self.overlap = overlap
        self.merge = merge
        self.device = device
        logger.info("Loaded %s with %d classes", weights, len(self.names))

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Cards in a BGR image, one detection per card when ``merge`` is on."""
        height, width = image.shape[:2]
        grid = tile_grid(width, height, *self.tiles, self.overlap)
        crops = [image[y : y + h, x : x + w] for x, y, w, h in grid]
        results = self.model.predict(
            crops,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        for (x, y, _, _), result in zip(grid, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy() + np.array([x, y, x, y], dtype=float)
            for box, cls, score in zip(xyxy, boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                detections.append(Detection(box, self.names[int(cls)], float(score)))

        if len(grid) > 1:
            detections = non_max_suppression(detections, self.iou)
        return merge_corners(detections) if self.merge else detections
