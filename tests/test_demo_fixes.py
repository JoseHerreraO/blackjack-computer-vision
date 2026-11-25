"""Fixes from the first full test with the glasses (2026-11-25 recording)."""

from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

import cv2
import numpy as np
import pytest

from blackjack_vision.blackjack.cards import Card
from blackjack_vision.blackjack.table import Table, Zone
from blackjack_vision.vision_classic.card_back import back_fraction, is_card_back
from blackjack_vision.vision_classic.mat import mat_far_edge

DATA = Path(__file__).parent / "data"
TEAL = (150, 140, 20)  # BGR of a teal mat (hue ~87)
NAVY = (110, 40, 20)  # BGR of a navy jumper (hue ~106, also inside the mat's hue range)
_ids = count(1000)


@dataclass
class Seen:
    corners: np.ndarray
    card: Card | None = None
    track_id: int = field(default_factory=lambda: next(_ids))


def card_at(x: float, y: float, code: str | None = None, height: float = 80) -> Seen:
    w, h = height * 0.7 / 2, height / 2
    corners = np.array([[x - w, y - h], [x + w, y - h], [x + w, y + h], [x - w, y + h]], float)
    return Seen(corners, Card.parse(code) if code else None)


# Card backs ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted((DATA / "card_backs").glob("*.png")), ids=lambda p: p.stem)
def test_real_card_backs_are_seen_face_down(path):
    assert is_card_back(cv2.imread(str(path)))


@pytest.mark.parametrize("path", sorted((DATA / "real_crops").glob("*.png"))[:20], ids=str)
def test_real_card_faces_are_not_backs(path):
    image = cv2.imread(str(path))
    assert back_fraction(image) < 0.05


# Far edge of the mat ------------------------------------------------------------------------------


def table_view(width=720, height=1280, top=500, jumper=True):
    """Wooden table, a trapezoid mat whose far edge is at ``top`` and a navy jumper above it."""
    image = np.full((height, width, 3), (90, 140, 190), np.uint8)
    mat = np.array([[60, top], [width - 60, top - 20], [width + 200, height], [-200, height]])
    cv2.fillPoly(image, [mat.astype(np.int32)], TEAL)
    if jumper:
        cv2.ellipse(image, (width // 2, top - 200), (250, 150), 0, 0, 360, NAVY, -1)
    return image


def test_far_edge_is_found_under_a_jumper_of_the_same_hue():
    edge = mat_far_edge(table_view())
    assert edge[360] == pytest.approx(490, abs=8)  # middle of the far edge (500 to 480)
    # Near the sides the first mat row is the side edge, much lower; the line ignores it.
    assert edge[10] == pytest.approx(500, abs=12)


def test_no_mat_gives_no_edge():
    image = np.full((640, 360, 3), (90, 140, 190), np.uint8)
    assert (mat_far_edge(image) == -1).all()


# Zones from the mat -------------------------------------------------------------------------------


def test_zone_follows_the_mat_not_the_middle_of_the_image():
    table = Table(dealer_line=0.5)
    edge = np.full(720, 700.0)  # head tilted up: the far edge is low in the image
    dealer = card_at(360, 760)  # below the middle of the image, but just below the edge
    player = card_at(360, 1000)
    assert table.zone_of(dealer, 1280) is Zone.PLAYER  # by the middle of the image: wrong
    assert table.zone_of(dealer, 1280, edge) is Zone.DEALER
    assert table.zone_of(player, 1280, edge) is Zone.PLAYER


def test_zone_falls_back_to_the_image_line_without_the_mat():
    table = Table(dealer_line=0.5)
    no_mat = np.full(720, -1.0)
    assert table.zone_of(card_at(360, 300), 1280, no_mat) is Zone.DEALER
    assert table.zone_of(card_at(360, 900), 1280, no_mat) is Zone.PLAYER


# Looking away -------------------------------------------------------------------------------------


def test_card_found_again_after_looking_away_is_not_recounted():
    table = Table()
    up = card_at(360, 300, "QD")
    table.update([up], [up], 1280, image_width=720)
    # Head turned: the camera motion moves the lost card out of the image.
    away = np.array([[1.0, 0.0, 900.0], [0.0, 1.0, 0.0]])
    table.update([], [], 1280, mat_in_view=False, motion=away, image_width=720)
    # Looking back, the same card shows up far from the (wrongly) predicted place.
    back = card_at(250, 250, "QD")
    update = table.update([back], [back], 1280, image_width=720)
    assert len(table.round.dealer.cards) == 1
    assert any("not recounted" in event for event in update.events)


def test_corner_of_the_mat_while_looking_away_is_not_an_empty_table():
    table = Table(clear_frames=3)
    up, a, b = card_at(360, 300, "TC"), card_at(300, 900, "TS"), card_at(420, 900, "6D")
    table.update([up, a, b], [up, a, b], 1280, image_width=720)
    away = np.array([[1.0, 0.0, -900.0], [0.0, 1.0, 0.0]])  # the cards are now off to the left
    table.update([], [], 1280, mat_in_view=True, motion=away, image_width=720)
    for _ in range(10):  # some mat in view, no cards: but we are not looking where they were
        table.update([], [], 1280, mat_in_view=True, image_width=720)
    assert table.rounds_played == 0 and table.round.hands[0].total == 16


def test_empty_mat_where_the_cards_were_starts_a_new_round():
    table = Table(clear_frames=3)
    up, a, b = card_at(360, 300, "TC"), card_at(300, 900, "TS"), card_at(420, 900, "6D")
    table.update([up, a, b], [up, a, b], 1280, image_width=720)
    for _ in range(3):
        table.update([], [], 1280, mat_in_view=True, image_width=720)
    assert table.rounds_played == 1


# Pipeline: face-down and moving cards are not read ------------------------------------------------


class CountingRecognizer:
    def __init__(self):
        self.calls = 0

    def recognise(self, card_image):
        self.calls += 1
        return type("R", (), {"card": Card.parse("KH"), "similarity": 0.9})()


def _scene(centre, back=False):
    from synthetic_cards import placed_corners, scene_with_cards

    corners = placed_corners(centre, 200)
    image = scene_with_cards([corners])
    if back:
        inner = corners.mean(axis=0) + (corners - corners.mean(axis=0)) * 0.85
        cv2.fillPoly(image, [np.round(inner).astype(np.int32)], (200, 90, 40))  # blue back
    return image


def _pipeline(recognizer):
    from blackjack_vision.detection_tracking.card_detector import CardDetector, CardDetectorConfig
    from blackjack_vision.detection_tracking.pipeline import CardPipeline

    detector = CardDetector(CardDetectorConfig(foreshortening=1.0))
    return CardPipeline(recognizer, detector=detector, motion=False, min_crops=3)


def test_face_down_card_is_not_named():
    recognizer = CountingRecognizer()
    pipeline = _pipeline(recognizer)
    for _ in range(10):
        result = pipeline.process(_scene((320, 240), back=True))
    assert recognizer.calls == 0
    assert result.cards and result.cards[0].face_down and result.cards[0].card is None


def test_card_is_read_once_it_lies_still():
    recognizer = CountingRecognizer()
    pipeline = _pipeline(recognizer)
    for k in range(6):  # being dealt: 25 px per frame across the mat
        pipeline.process(_scene((200 + 25 * k, 240)))
    assert recognizer.calls == 0
    for _ in range(12):
        result = pipeline.process(_scene((350, 240)))
    assert recognizer.calls == 1 and result.cards[0].card is not None
