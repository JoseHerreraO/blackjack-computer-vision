from dataclasses import dataclass

import numpy as np
import pytest
from synthetic_cards import placed_corners, scene_with_cards

from blackjack_vision.blackjack.cards import Card
from blackjack_vision.detection_tracking.pipeline import CardPipeline


@dataclass
class FakeRecognition:
    card: Card | None
    similarity: float


class FakeRecognizer:
    """Answers with the queued results in order, then with ``default``."""

    def __init__(self, *answers, default="KH"):
        self.answers = list(answers)
        self.default = default
        self.calls = []

    def recognise(self, card_image):
        self.calls.append(card_image.shape)
        code = self.answers.pop(0) if self.answers else self.default
        return FakeRecognition(Card.parse(code) if code else None, 0.9 if code else 0.5)


@pytest.fixture
def table():
    placements = [placed_corners((160, 240), 170), placed_corners((460, 240), 170, angle_deg=20)]
    return scene_with_cards(placements), len(placements)


def test_cards_are_recognised_once_after_min_crops(table):
    image, n_cards = table
    recognizer = FakeRecognizer()
    pipeline = CardPipeline(recognizer, min_crops=4, fuse_top=2)

    for _ in range(3):
        result = pipeline.process(image)
        assert len(result.cards) == n_cards
        assert all(card.label == "?" for card in result.cards) and result.new_cards == []

    result = pipeline.process(image)
    assert len(result.new_cards) == n_cards
    assert all(card.label == "KH" and card.similarity == 0.9 for card in result.cards)

    for _ in range(5):
        result = pipeline.process(image)
        assert result.new_cards == []
    assert len(recognizer.calls) == n_cards  # never recognised again
    assert all(shape == (300, 200, 3) for shape in recognizer.calls)


def test_unsure_recognition_is_retried(table):
    image, _ = table
    recognizer = FakeRecognizer(None, None, default="5C")  # both cards unsure the first time
    pipeline = CardPipeline(recognizer, min_crops=2, retry_frames=3)

    pipeline.process(image)
    assert pipeline.process(image).new_cards == []  # tried, unsure
    for _ in range(2):
        assert pipeline.process(image).new_cards == []  # waiting retry_frames
    result = pipeline.process(image)
    assert {card.label for card in result.new_cards} == {"5C"}
    assert len(result.new_cards) == 2


def test_without_recognizer_cards_stay_unknown(table):
    image, n_cards = table
    pipeline = CardPipeline(None)
    for _ in range(8):
        result = pipeline.process(image)
    assert len(result.cards) == n_cards and all(card.label == "?" for card in result.cards)
    assert set(result.timings_ms) == {"detect", "motion", "merge", "track", "recognise"}


def test_card_that_leaves_is_forgotten(table):
    image, _ = table
    pipeline = CardPipeline(FakeRecognizer(), min_crops=1)
    pipeline.process(image)
    empty = np.full_like(image, image[5, 5])
    for _ in range(pipeline.tracker.max_missed + 2):
        assert pipeline.process(empty).cards == []
    assert pipeline._cards == {}


def test_invalid_buffer_settings():
    with pytest.raises(ValueError):
        CardPipeline(None, fuse_top=0)
    with pytest.raises(ValueError):
        CardPipeline(None, min_crops=20, buffer_size=10)


def test_recognitions_are_spread_over_frames():
    placements = [placed_corners((110 + 140 * i, 240), 150) for i in range(4)]
    image = scene_with_cards(placements)
    recognizer = FakeRecognizer()
    pipeline = CardPipeline(recognizer, min_crops=1, max_recognitions_per_frame=2)
    counts = [len(pipeline.process(image).new_cards) for _ in range(3)]
    assert counts == [2, 2, 0]
    assert len(recognizer.calls) == 4
