"""Cards seen at an angle, like from the glasses on the player's head.

The synthetic scenes are drawn as seen from above and then squashed vertically, which is how the
table looks when the camera is tilted towards it: upright cards look almost square and cards
lying sideways look very long, so the raw aspect check alone would reject them.
"""

import cv2
import numpy as np
import pytest
from synthetic_cards import placed_corners, scene_with_cards

from blackjack_vision.detection_tracking.card_detector import (
    CARD_ASPECT_RATIO,
    CardDetector,
    CardDetectorConfig,
    estimate_foreshortening,
)
from blackjack_vision.vision_classic.corners import side_lengths

SQUASH = 0.5
SIZE = (640, 960)  # drawn from above, then squashed to 640 x 480


def squashed(scene: np.ndarray, factor: float = SQUASH) -> np.ndarray:
    height, width = scene.shape[:2]
    return cv2.resize(scene, (width, round(height * factor)), interpolation=cv2.INTER_AREA)


def table(angles: list[float], square: bool = False) -> tuple[np.ndarray, int]:
    centres = [(130, 230), (330, 230), (520, 260), (180, 650), (430, 680), (330, 450)]
    placements = [
        placed_corners(centre, 170, angle_deg=angle) for centre, angle in zip(centres, angles)
    ]
    scene = scene_with_cards(placements, size=SIZE)
    if square:
        cv2.rectangle(scene, (500, 560), (600, 660), (250, 250, 250), -1)
    return squashed(scene), len(placements)


def test_squashed_cards_at_any_angle_are_found():
    image, n_cards = table([0, 30, 90, 135, 60])
    result = CardDetector().detect(image)
    assert len(result.cards) == n_cards
    assert result.foreshortening == pytest.approx(SQUASH, abs=0.06)


def test_raw_aspect_check_alone_misses_them():
    image, n_cards = table([0, 30, 90, 135, 60])
    result = CardDetector(CardDetectorConfig(foreshortening=1.0)).detect(image)
    assert len(result.cards) < n_cards


def test_corners_are_in_portrait_order_after_the_correction():
    image, _ = table([0, 30, 90, 135, 60])
    result = CardDetector().detect(image)
    for card in result.cards:
        corrected = card.corners.astype(float) * [1.0, 1.0 / result.foreshortening]
        edges = side_lengths(corrected.astype(np.float32))
        assert edges[0] < edges[1]


def test_a_square_is_still_rejected_after_the_correction():
    image, n_cards = table([0, 30, 90, 135], square=True)
    result = CardDetector().detect(image)
    assert len(result.cards) == n_cards


def test_last_estimate_is_reused_when_few_cards_are_visible():
    detector = CardDetector()
    image, _ = table([0, 30, 90, 135, 60])
    detector.detect(image)

    # One card lying sideways: squashed, it looks almost three times as long as it is wide.
    lone = squashed(scene_with_cards([placed_corners((320, 480), 170, angle_deg=90)], size=SIZE))
    assert len(detector.detect(lone).cards) == 1
    assert len(CardDetector().detect(lone).cards) == 0  # no estimate yet, no correction

    detector.reset()
    assert len(detector.detect(lone).cards) == 0


def test_default_is_used_before_the_first_estimate():
    lone = squashed(scene_with_cards([placed_corners((320, 480), 170, angle_deg=90)], size=SIZE))
    config = CardDetectorConfig(foreshortening_default=SQUASH)
    assert len(CardDetector(config).detect(lone).cards) == 1


def test_estimate_needs_enough_cards():
    # 63 x 88 card lying upright, squashed to half height (63 x 44) and lying sideways (88 x 31.5)
    upright = np.array([[0, 0], [63, 0], [63, 44], [0, 44]], np.float32)
    sideways = np.array([[0, 0], [88, 0], [88, 31.5], [0, 31.5]], np.float32)
    assert estimate_foreshortening([upright, sideways], min_cards=3) is None
    estimate = estimate_foreshortening([upright, sideways, sideways])
    assert estimate == pytest.approx(SQUASH, abs=0.02)
    assert 88 / (31.5 / estimate) == pytest.approx(CARD_ASPECT_RATIO, rel=0.05)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"foreshortening": 0.0},
        {"foreshortening": 1.5},
        {"foreshortening_range": (0.8, 0.4)},
        {"foreshortening_default": 0.0},
    ],
)
def test_invalid_foreshortening_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CardDetectorConfig(**kwargs)
