"""The classical recogniser inside the real-time pipeline, the camera views and --test-data."""

import runpy
from pathlib import Path

import cv2
import numpy as np
import pytest
from synthetic_cards import mat, paste_card, placed_corners
from synthetic_faces import card_face, photographed

from blackjack_vision.blackjack.cards import Card, Rank, Suit
from blackjack_vision.capture.sources import write_image
from blackjack_vision.detection_tracking.card_detector import (
    CardDetector,
    config_for_view,
    default_view,
)
from blackjack_vision.detection_tracking.pipeline import CardPipeline
from blackjack_vision.vision_classic.recognition import CardRecognizer, RecognizerConfig

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CARDS = [Card(rank, suit) for rank in (Rank.ACE, Rank.FOUR, Rank.NINE, Rank.KING) for suit in Suit]
FAST = RecognizerConfig(vocabulary_size=60, seed=2)


@pytest.fixture(scope="module")
def recognizer() -> CardRecognizer:
    rng = np.random.default_rng(0)
    images = [photographed(card_face(card), rng) for card in CARDS for _ in range(3)]
    return CardRecognizer(FAST).fit(images, [card for card in CARDS for _ in range(3)])


def _scene_with_face(card: Card, corners: np.ndarray) -> np.ndarray:
    """A real card face (not the plain texture) pasted on the mat at ``corners``."""
    face = card_face(card)
    height, width = face.shape[:2]
    source = np.array(
        [[-0.5, -0.5], [width - 0.5, -0.5], [width - 0.5, height - 0.5], [-0.5, height - 0.5]],
        np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, np.asarray(corners, np.float32))
    scene = mat((640, 480))
    warped = cv2.warpPerspective(face, homography, (640, 480))
    weight = cv2.warpPerspective(np.ones((height, width), np.float32), homography, (640, 480))
    blended = scene * (1 - weight[..., None]) + warped * weight[..., None]
    return blended.astype(np.uint8)


# --------------------------------------------------------------------------- recognise()


def test_recognise_names_clear_cards_and_reports_unsure_ones(recognizer):
    result = recognizer.recognise(card_face(Card(Rank.NINE, Suit.HEARTS)))
    assert result.card == Card(Rank.NINE, Suit.HEARTS)
    assert result.known and result.similarity == result.prediction.rank_confidence
    assert result.similarity >= recognizer.min_rank_confidence

    recognizer.min_rank_confidence = 1e9  # nothing is that sure
    try:
        unsure = recognizer.recognise(card_face(Card(Rank.NINE, Suit.HEARTS)))
    finally:
        recognizer.min_rank_confidence = 0.4
    assert unsure.card is None and not unsure.known
    assert unsure.prediction.rank is Rank.NINE  # the guess is still there for logging


def test_weakest_duel_is_lower_for_an_ambiguous_card(recognizer):
    clear = recognizer.predict(card_face(Card(Rank.FOUR, Suit.CLUBS)))
    blank = recognizer.predict(np.full((300, 200, 3), 245, np.uint8))
    assert clear.rank_confidence > blank.rank_confidence


def test_classic_recogniser_in_the_pipeline(recognizer):
    corners = placed_corners((320, 240), 240, angle_deg=15)
    scene = _scene_with_face(Card(Rank.KING, Suit.SPADES), corners)
    pipeline = CardPipeline(recognizer, detector=CardDetector(), min_crops=3, fuse_top=2)
    named = []
    for _ in range(4):
        result = pipeline.process(scene)
        named += [card.card for card in result.new_cards]
    assert named == [Card(Rank.KING, Suit.SPADES)]  # recognised once


# --------------------------------------------------------------------------- views


def test_default_view_follows_the_source():
    assert default_view("bridge") == "glasses" and default_view("BRIDGE:5051") == "glasses"
    assert default_view(0) == "top" and default_view("clip.mp4") == "top"
    with pytest.raises(ValueError):
        config_for_view("side")


@pytest.mark.parametrize("angle", [0, 90])
def test_glasses_view_keeps_a_single_card_upright(angle):
    corners = placed_corners((320, 240), 220, angle_deg=angle)
    corners[:, 1] = 240 + (corners[:, 1] - 240) * 0.5  # table squashed as seen by the glasses
    scene = paste_card(mat(), corners)

    glasses = CardDetector(config_for_view("glasses")).detect(scene).cards
    assert len(glasses) == 1
    first_edge = glasses[0].corners[1] - glasses[0].corners[0]
    # The first edge is the card's short side: horizontal for an upright card, vertical otherwise.
    assert (abs(first_edge[0]) > abs(first_edge[1])) == (angle == 0)

    top = CardDetector(config_for_view("top")).detect(scene).cards
    if top:  # with the default scale the card is lost or comes out rotated by 90 degrees
        first_edge = top[0].corners[1] - top[0].corners[0]
        assert (abs(first_edge[0]) > abs(first_edge[1])) != (angle == 0)


# --------------------------------------------------------------------------- train_recognizer


def test_train_recognizer_with_a_separate_test_folder(tmp_path, capsys):
    rng = np.random.default_rng(4)
    for folder, copies in (("train", 3), ("test", 1)):
        for card in CARDS:
            for i in range(copies):
                image = photographed(card_face(card), rng)
                write_image(tmp_path / folder / card.code / f"{i:03d}.png", image)
    main = runpy.run_path(str(SCRIPTS / "train_recognizer.py"), run_name="train_cli")["main"]
    code = main(
        [
            "--data", str(tmp_path / "train"),
            "--test-data", str(tmp_path / "test"),
            "--vocabulary", "40",
            "--out", str(tmp_path / "model.joblib"),
        ]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0
    assert "Test set: 16 images" in out and "recognise() with min confidence" in out
    assert (tmp_path / "model.joblib").is_file()
    assert CardRecognizer.load(tmp_path / "model.joblib").is_trained
