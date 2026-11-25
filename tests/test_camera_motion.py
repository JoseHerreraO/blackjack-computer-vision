"""Camera motion between frames, and tracking cards through head turns."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from synthetic_cards import mat, paste_card, placed_corners

from blackjack_vision.blackjack.table import Table
from blackjack_vision.detection_tracking.card_detector import (
    CardDetector,
    CardDetectorConfig,
    config_for_view,
)
from blackjack_vision.detection_tracking.pipeline import CardPipeline
from blackjack_vision.vision_classic.camera_motion import CameraMotion, apply_motion

REAL_FRAMES = Path(__file__).parent / "data" / "merge_18nov"


def scene(placements, shift=(0.0, 0.0)):
    """Cards on a mat with some texture, the whole view moved by ``shift`` pixels."""
    dx, dy = shift
    image = mat((640, 480), seed=1)
    rng = np.random.default_rng(3)
    for x, y in rng.integers(0, [640, 480], (150, 2)):
        cv2.circle(image, (int(x + dx), int(y + dy)), 3, (20, 60, 25), -1)
    for corners in placements:
        moved = corners + np.array([dx, dy], np.float32)
        image = paste_card(image, moved)
        cv2.polylines(image, [np.round(moved).astype(np.int32)], True, (205, 205, 205), 1)
    return image


def test_translation_of_the_view_is_measured():
    motion = CameraMotion()
    cards = [placed_corners((250, 240), 160)]
    assert motion(scene(cards)) is None  # first frame
    transform = motion(scene(cards, (12, -5)))
    assert transform is not None
    moved = apply_motion(transform, np.array([[100.0, 100.0], [500.0, 400.0]]))
    np.testing.assert_allclose(moved, [[112, 95], [512, 395]], atol=1.0)


def test_blank_frames_give_no_motion():
    motion = CameraMotion()
    blank = np.full((480, 640, 3), 128, np.uint8)
    motion(blank)
    assert motion(blank) is None


def test_card_keeps_its_id_through_a_head_turn():
    # The view jumps by more than a card's width: without the camera motion the IoU is zero.
    card = [placed_corners((250, 240), 160)]
    frames = [scene(card)] * 3 + [scene(card, (90 * k, 0)) for k in range(1, 4)]
    detector = CardDetectorConfig(foreshortening=1.0)
    with_motion = CardPipeline(None, detector=CardDetector(detector))
    without = CardPipeline(None, detector=CardDetector(detector), motion=False)
    ids_with, ids_without = set(), set()
    for frame in frames:
        ids_with |= {card.track_id for card in with_motion.process(frame).cards}
        ids_without |= {card.track_id for card in without.process(frame).cards}
    assert ids_with == {1}
    assert len(ids_without) > 1


def test_card_dealt_against_a_known_one_is_found_from_the_track():
    # Side by side, the two cards make one block that looks like a single bigger card.
    first = placed_corners((250, 240), 160)
    second = first + np.array([115, 0], np.float32)
    pipeline = CardPipeline(None, detector=CardDetector(CardDetectorConfig(foreshortening=1.0)))
    for _ in range(3):
        pipeline.process(scene([first]))
    for _ in range(3):
        result = pipeline.process(scene([first, second]))
    by_id = {card.track_id: card.corners.mean(axis=0) for card in result.cards}
    assert set(by_id) == {1, 2}
    np.testing.assert_allclose(by_id[1], first.mean(axis=0), atol=3)
    np.testing.assert_allclose(by_id[2], second.mean(axis=0), atol=6)


def test_table_moves_lost_cards_with_the_camera():
    table = Table()
    shift = np.array([[1.0, 0.0, 30.0], [0.0, 1.0, 0.0]])
    table._dealt.append(  # a dealt card out of view
        type("D", (), {"code": "KH", "zone": None, "track_id": 9, "centre": np.zeros(2)})()
    )
    table.update([], [], 480, motion=shift)
    np.testing.assert_allclose(table._dealt[0].centre, [30.0, 0.0])


@pytest.mark.skipif(not REAL_FRAMES.exists(), reason="real frames not available")
def test_real_dealer_cards_merged_on_the_glasses_are_both_kept():
    """2026-11-18 glasses recording: the dealer's 3 of clubs and the face-down card touch."""
    frames = sorted(REAL_FRAMES.glob("*.jpg"))
    pipeline = CardPipeline(None, detector=CardDetector(config_for_view("glasses")))
    player_ids = []
    for path in frames:
        result = pipeline.process(cv2.imread(str(path)))
        player_ids.append({c.track_id for c in result.cards if c.corners[:, 1].mean() > 400})
    dealer = [c for c in result.cards if c.corners[:, 1].mean() < 400]
    assert len(dealer) == 2  # the up card and the hole card, not one blob
    assert len(result.cards) == 4
    # The player's two cards keep their IDs from the moment both are seen.
    seen = [ids for ids in player_ids if len(ids) == 2]
    assert seen and all(ids == seen[0] for ids in seen)
