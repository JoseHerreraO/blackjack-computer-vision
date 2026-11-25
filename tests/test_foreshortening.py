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


# All cards lying the same way (review of PR #19) ----------------------------------------------


def noisy_card(rng, cx, c, angle=0.0, scale=1.6):
    """Card corners on a 63 x 88 table, rotated, squashed by ``c`` and with 1.5 px of noise."""
    quad = np.array([[-31.5, -44], [31.5, -44], [31.5, 44], [-31.5, 44]]) * scale
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    corners = quad @ rotation.T + [cx, 300]
    corners[:, 1] *= c
    return (corners + rng.normal(0, 1.5, (4, 2))).astype(np.float32)


def share_right(true_c, angles, prior, frames=150, weight=0.1, seed=0):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(frames):
        quads = [noisy_card(rng, 100 + 150 * i, true_c, a) for i, a in enumerate(angles)]
        estimate = estimate_foreshortening(quads, prior=prior, prior_weight=weight)
        hits += abs(np.log(estimate / true_c)) <= 0.12
    return hits / frames


@pytest.mark.parametrize("true_c", [0.45, 0.5])
def test_upright_cards_alone_are_ambiguous_without_a_prior(true_c):
    # An upright card squashed by c looks like a sideways one squashed by ~1.95 c.
    assert share_right(true_c, [0, 0, 0, 0], prior=None) < 0.8


@pytest.mark.parametrize("true_c", [0.45, 0.5, 0.55])
def test_prior_settles_upright_cards_with_the_glasses(true_c):
    assert share_right(true_c, [0, 0, 0, 0], prior=0.5) >= 0.99


def test_prior_settles_upright_cards_seen_from_above():
    assert share_right(1.0, [0, 0, 0, 0], prior=1.0) >= 0.99


@pytest.mark.parametrize(("true_c", "prior"), [(0.8, 0.5), (0.5, 0.9)])
def test_real_evidence_overrides_the_prior(true_c, prior):
    # The head pitch changed: with cards in several orientations the data wins.
    assert share_right(true_c, [0, 90, 30, 60], prior=prior) >= 0.99


def test_glasses_detector_keeps_upright_cards_portrait():
    centres = [(110, 300), (260, 300), (410, 300), (560, 300)]
    scene = scene_with_cards([placed_corners(c, 170) for c in centres], size=SIZE)
    image = squashed(scene)
    detector = CardDetector(CardDetectorConfig.for_glasses())
    for _ in range(3):
        result = detector.detect(image)
        assert len(result.cards) == 4
        assert result.foreshortening == pytest.approx(SQUASH, abs=0.06)
        for card in result.cards:
            edges = side_lengths(card.corners)  # in the image an upright card is wider than tall
            corrected = side_lengths(
                (card.corners * [1, 1 / result.foreshortening]).astype(np.float32)
            )
            assert corrected[0] < corrected[1], (
                "rectified card would come out rotated by 90 degrees"
            )
            assert edges[0] > 0


def test_for_glasses_preset():
    config = CardDetectorConfig.for_glasses(min_fill=0.9)
    assert config.foreshortening_default == 0.5 and config.min_fill == 0.9


def test_negative_prior_weight_is_rejected():
    with pytest.raises(ValueError):
        CardDetectorConfig(foreshortening_prior_weight=-0.1)


def test_one_card_seen_from_above_gives_the_scale():
    # Glasses looking down at a single card (the lock shows the aces one at a time): the card
    # looks taller than wide, which only an upright card with c > 0.72 explains.
    from blackjack_vision.detection_tracking.card_detector import estimate_from_few

    quad = squashed_quad(angle=0, scale=0.85)
    assert estimate_from_few([quad]) == pytest.approx(0.85, abs=0.03)


def test_one_card_that_looks_wide_gives_nothing():
    from blackjack_vision.detection_tracking.card_detector import estimate_from_few

    # Upright at c=0.5 looks wider than tall, and so does a card lying sideways: ambiguous.
    assert estimate_from_few([squashed_quad(angle=0, scale=0.5)]) is None


def test_glasses_detector_finds_a_lone_card_seen_from_above():
    # Before the fix the detector kept the glasses' default 0.5 and rejected the card.
    detector = CardDetector(CardDetectorConfig.for_glasses())
    lone = squashed(scene_with_cards([placed_corners((320, 480), 220)], size=SIZE), 0.9)
    results = [detector.detect(lone) for _ in range(3)]
    assert all(len(result.cards) == 1 for result in results)
    # The card gets its own scale; the scale of the frame is not moved by one card (with the
    # glasses the near cards look less squashed than the far ones).
    assert results[-1].foreshortening == pytest.approx(0.5)
    edges = side_lengths(results[-1].cards[0].corners)
    assert edges[0] < edges[1]  # upright: the first edge is the short top side


def squashed_quad(angle: float, scale: float) -> np.ndarray:
    """Corners of an upright 63 x 88 card (turned by ``angle``) squashed vertically by ``scale``."""
    quad = placed_corners((300, 300), 176, angle_deg=angle).astype(np.float64)
    quad[:, 1] *= scale
    return quad
