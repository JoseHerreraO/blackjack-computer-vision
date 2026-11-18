import numpy as np
import pytest

from blackjack_vision.detection_tracking.tracker import BoxKalmanFilter, CardTracker
from blackjack_vision.detection_tracking.yolo_detector import (
    Detection,
    box_iou,
    merge_corners,
    non_max_suppression,
    tile_grid,
)


def det(x, y, w=20, h=40, label="AH", confidence=0.9):
    return Detection(np.array([x, y, x + w, y + h], dtype=float), label, confidence)


# Detector helpers -------------------------------------------------------------------------------


def test_box_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert box_iou(a, b) == pytest.approx(np.array([[1.0, 1 / 3, 0.0]]))


def test_nms_keeps_best_per_label():
    kept = non_max_suppression(
        [det(0, 0, confidence=0.6), det(2, 0, confidence=0.9), det(2, 0, label="KS")], 0.5
    )
    assert sorted((d.label, d.confidence) for d in kept) == [("AH", 0.9), ("KS", 0.9)]


def test_merge_corners_joins_both_corners_of_a_card():
    # Top-left and bottom-right index of the same card, plus a different card next to it.
    merged = merge_corners([det(100, 100), det(200, 260), det(400, 100, label="7C")])
    by_label = {d.label: d for d in merged}
    assert set(by_label) == {"AH", "7C"}
    assert by_label["AH"].parts == 2
    assert by_label["AH"].box.tolist() == [100, 100, 220, 300]
    assert by_label["7C"].parts == 1


def test_merge_corners_keeps_far_cards_with_the_same_label_apart():
    # Two decks in the shoe: two aces of hearts on opposite sides of the table.
    assert len(merge_corners([det(0, 0), det(1000, 1000)])) == 2


@pytest.mark.parametrize(("columns", "rows"), [(1, 1), (1, 2), (2, 3)])
def test_tile_grid_covers_the_image(columns, rows):
    tiles = tile_grid(720, 1280, columns, rows, overlap=0.15)
    assert len(tiles) == columns * rows
    covered = np.zeros((1280, 720), bool)
    for x, y, w, h in tiles:
        assert x >= 0 and y >= 0 and x + w <= 720 and y + h <= 1280
        covered[y : y + h, x : x + w] = True
    assert covered.all()


def test_tile_grid_overlaps_neighbours():
    (_, y0, _, h0), (_, y1, _, _) = tile_grid(720, 1280, 1, 2, overlap=0.15)
    assert y1 < y0 + h0


# Tracker ----------------------------------------------------------------------------------------


def test_kalman_follows_constant_motion():
    kf = BoxKalmanFilter(np.array([0, 0, 20, 40]))
    for step in range(1, 15):
        kf.predict()
        kf.correct(np.array([5 * step, 0, 5 * step + 20, 40]))
    predicted = kf.predict()
    assert predicted[0] == pytest.approx(75, abs=3)
    assert kf.velocity[0] == pytest.approx(5, abs=0.5)


def test_track_keeps_id_and_confirms_once():
    tracker = CardTracker(min_hits=3)
    confirmed = []
    for step in range(6):
        update = tracker.update([det(100 + 3 * step, 100)])
        confirmed += update.confirmed
    assert len(tracker.tracks) == 1
    assert [t.track_id for t in confirmed] == [1]
    assert update.visible[0].label == "AH"


def test_two_cards_get_two_ids():
    tracker = CardTracker(min_hits=1)
    for step in range(4):
        update = tracker.update(
            [det(100 + step, 100, label="AH"), det(300 - step, 100, label="KS")]
        )
    labels = {t.track_id: t.label for t in update.tracks}
    assert labels == {1: "AH", 2: "KS"}


def test_track_survives_short_occlusion():
    tracker = CardTracker(min_hits=2, max_missed=5)
    for _ in range(3):
        tracker.update([det(100, 100)])
    for _ in range(4):  # hidden by a hand
        tracker.update([])
    update = tracker.update([det(100, 100)])
    assert [t.track_id for t in update.tracks] == [1]
    assert update.confirmed == []  # not counted twice


def test_track_is_dropped_after_max_missed():
    tracker = CardTracker(max_missed=2)
    tracker.update([det(100, 100)])
    removed = []
    for _ in range(3):
        removed += tracker.update([]).removed
    assert tracker.tracks == [] and [t.track_id for t in removed] == [1]


def test_label_vote_ignores_a_single_misread():
    tracker = CardTracker(min_hits=1, label_window=10)
    labels = ["8S"] * 6 + ["3S"] + ["8S"] * 2
    for label in labels:
        update = tracker.update([det(100, 100, label=label, confidence=0.8)])
    track = update.tracks[0]
    assert track.label == "8S"
    assert track.label_confidence == pytest.approx(8 / 9)
