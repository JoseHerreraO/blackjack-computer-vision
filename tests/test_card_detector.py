import cv2
import numpy as np
import pytest
from synthetic_cards import (
    card_texture,
    mat,
    match_corners,
    paste_card,
    placed_corners,
    scene_with_cards,
)

from blackjack_vision.detection_tracking.card_detector import (
    CardDetector,
    CardDetectorConfig,
    Segmentation,
    draw_detections,
)
from blackjack_vision.vision_classic.corners import side_lengths
from blackjack_vision.vision_classic.rectify import CARD_SIZE, rectify_card

CORNER_TOLERANCE_PX = 1.5


def test_single_upright_card_is_found_with_accurate_corners():
    expected = placed_corners((320, 240), height_px=220)
    result = CardDetector().detect(scene_with_cards([expected]))

    assert len(result.cards) == 1
    assert not result.clusters
    card = result.cards[0]
    assert match_corners(card.corners, expected) < CORNER_TOLERANCE_PX
    assert np.allclose(card.corners[0], expected[0], atol=CORNER_TOLERANCE_PX)  # top-left first
    assert card.aspect_ratio == pytest.approx(88 / 63, rel=0.02)


def test_rotated_and_tilted_cards_are_all_found():
    placements = [
        placed_corners((110, 130), 150, angle_deg=0),
        placed_corners((320, 120), 150, angle_deg=30, tilt=0.1),
        placed_corners((530, 140), 150, angle_deg=-60, tilt=0.15),
        placed_corners((200, 350), 140, angle_deg=90),
        placed_corners((450, 350), 160, angle_deg=170, tilt=0.2),
    ]
    result = CardDetector().detect(scene_with_cards(placements))

    assert len(result.cards) == len(placements)
    centres = [card.center for card in result.cards]
    assert [c[0] for c in centres] == sorted(c[0] for c in centres)  # left to right
    for expected in placements:
        card = min(result.cards, key=lambda c: np.linalg.norm(c.center - expected.mean(axis=0)))
        assert match_corners(card.corners, expected) < CORNER_TOLERANCE_PX
        edges = side_lengths(card.corners)
        assert edges[0] < edges[1]  # portrait order: the first edge is short


@pytest.mark.parametrize("angle", [0, 25, 180, 205])
def test_rectified_card_matches_the_texture(angle):
    corners = placed_corners((320, 240), 260, angle_deg=angle, tilt=0.1)
    scene = scene_with_cards([corners])
    card = CardDetector().detect(scene).cards[0]

    rectified = rectify_card(scene, card.corners)
    texture, _ = card_texture()
    reference = cv2.resize(texture, CARD_SIZE, interpolation=cv2.INTER_AREA)

    assert rectified.shape == (CARD_SIZE[1], CARD_SIZE[0], 3)
    errors = [
        np.abs(rectified.astype(int) - candidate.astype(int)).mean()
        for candidate in (reference, cv2.rotate(reference, cv2.ROTATE_180))
    ]
    # The card comes out upright (possibly turned by 180 degrees, a symmetry of real cards) and
    # the matching orientation is clearly better than the other one.
    assert min(errors) < 3
    assert max(errors) > 3 * min(errors)


@pytest.mark.parametrize("method", list(Segmentation))
def test_every_segmentation_method_finds_the_cards(method):
    placements = [placed_corners((200, 240), 200, 10), placed_corners((450, 240), 200, -15)]
    detector = CardDetector(CardDetectorConfig(segmentation=method))
    assert len(detector.detect(scene_with_cards(placements)).cards) == 2


def test_uneven_lighting_is_tolerated():
    placements = [placed_corners((180, 240), 200, 5), placed_corners((460, 240), 200, -5)]
    scene = scene_with_cards(placements).astype(np.float32)
    scene *= np.linspace(0.55, 1.0, scene.shape[1], dtype=np.float32)[None, :, None]
    result = CardDetector().detect(scene.astype(np.uint8))
    assert len(result.cards) == 2


def test_empty_mat_gives_no_detections():
    result = CardDetector().detect(mat(seed=3, noise=8.0))
    assert result.cards == []
    assert result.clusters == []
    assert not result.mask.any()


def test_white_shapes_that_are_not_cards_are_rejected():
    scene = mat()
    cv2.rectangle(scene, (60, 60), (200, 200), (250, 250, 250), -1)  # square: aspect 1
    cv2.circle(scene, (330, 130), 70, (250, 250, 250), -1)  # circle
    cv2.rectangle(scene, (440, 60), (480, 360), (250, 250, 250), -1)  # strip: aspect 7.5
    cv2.circle(scene, (100, 400), 4, (250, 250, 250), -1)  # speck below the minimum area
    result = CardDetector().detect(scene)
    assert result.cards == []
    assert len(result.clusters) == 3  # the speck is not even a candidate


def test_overlapping_cards_are_reported_as_a_cluster():
    first = placed_corners((300, 240), 220)
    second = first + np.array([60, 45], np.float32)
    result = CardDetector().detect(scene_with_cards([first, second]))
    assert result.cards == []
    assert len(result.clusters) == 1


def test_card_cut_by_the_image_border_is_ignored():
    scene = scene_with_cards([placed_corners((40, 240), 220), placed_corners((400, 240), 220)])
    result = CardDetector().detect(scene)
    assert len(result.cards) == 1
    assert result.cards[0].center[0] == pytest.approx(400, abs=2)

    keep_all = CardDetector(CardDetectorConfig(border_margin=-1)).detect(scene)
    assert len(keep_all.cards) + len(keep_all.clusters) == 2


def test_grayscale_input_is_accepted():
    scene = scene_with_cards([placed_corners((320, 240), 220)])
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    detector = CardDetector(CardDetectorConfig(segmentation="gray"))
    assert len(detector(gray).cards) == 1


def test_drawing_does_not_modify_the_input():
    scene = paste_card(mat(), placed_corners((320, 240), 220))
    original = scene.copy()
    canvas = draw_detections(scene, CardDetector().detect(scene))
    assert np.array_equal(scene, original)
    assert not np.array_equal(canvas, scene)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"blur_ksize": 4},
        {"min_area_fraction": 0.6, "max_area_fraction": 0.5},
        {"aspect_range": (0.5, 2.0)},
        {"min_fill": 0.0},
        {"segmentation": "canny"},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CardDetectorConfig(**kwargs)
