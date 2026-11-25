"""Splitting blobs of touching or overlapping cards (issue #26)."""

import cv2
import numpy as np
from synthetic_cards import mat, paste_card, placed_corners

from blackjack_vision.detection_tracking.card_detector import CardDetector, CardDetectorConfig
from blackjack_vision.vision_classic.split_cards import split_cluster

EDGE_BGR = (205, 205, 205)


def table(placements, size=(640, 480)):
    """Cards pasted in order, each with the thin grey edge real cards show over the one below."""
    scene = mat(size)
    for corners in placements:
        scene = paste_card(scene, corners)
        cv2.polylines(scene, [np.round(corners).astype(np.int32)], True, EDGE_BGR, 1, cv2.LINE_AA)
    return scene


def detect(scene):
    return CardDetector(CardDetectorConfig(foreshortening=1.0)).detect(scene)


def closest(cards, expected):
    centres = np.array([card.center for card in cards])
    return cards[int(np.argmin(np.linalg.norm(centres - expected.mean(axis=0), axis=1)))]


def test_fanned_pair_is_split_into_both_cards():
    bottom = placed_corners((260, 230), 180)
    top = placed_corners((290, 260), 180)
    result = detect(table([bottom, top]))
    assert not result.clusters
    assert len(result.cards) == 2
    for expected in (bottom, top):
        card = closest(result.cards, expected)
        assert card.split
        assert np.abs(card.center - expected.mean(axis=0)).max() < 5


def test_cards_touching_in_an_l_are_split():
    upright = placed_corners((230, 240), 180)
    sideways = placed_corners((370, 300), 180, angle_deg=90)
    result = detect(table([upright, sideways]))
    assert len(result.cards) == 2 and not result.clusters


def test_fan_of_three_gives_the_outer_cards_and_no_wrong_one():
    # The middle card shows no whole side, only steps of its outline: it is left out rather
    # than guessed (the tracker keeps it if it was seen before the cards were fanned).
    fan = [placed_corners((240 + 30 * k, 220 + 25 * k), 160, angle_deg=10) for k in range(3)]
    result = detect(table(fan))
    assert len(result.cards) == 2
    for expected in (fan[0], fan[2]):
        assert np.abs(closest(result.cards, expected).center - expected.mean(axis=0)).max() < 6


def test_blob_without_straight_sides_is_not_split():
    blob = np.zeros((480, 640), np.uint8)
    cv2.ellipse(blob, (320, 240), (120, 70), 20, 0, 360, 255, -1)
    cv2.circle(blob, (420, 200), 50, 255, -1)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert split_cluster(contours[0]) == []


def test_card_next_to_one_cut_by_the_border_is_kept():
    whole = placed_corners((480, 240), 180)
    cut = placed_corners((600, 250), 180)  # runs past the right border, touching the first
    result = detect(table([whole, cut]))
    assert len(result.cards) == 1
    assert np.abs(result.cards[0].center - whole.mean(axis=0)).max() < 5


def test_splitting_can_be_turned_off():
    pair = [placed_corners((260, 230), 180), placed_corners((290, 260), 180)]
    config = CardDetectorConfig(foreshortening=1.0, split_clusters=False)
    result = CardDetector(config).detect(table(pair))
    assert result.cards == [] and len(result.clusters) == 1
