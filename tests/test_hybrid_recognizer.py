import numpy as np
import pytest

from blackjack_vision.blackjack.cards import Card, Rank, Suit
from blackjack_vision.vision_classic.recognition import Prediction
from blackjack_vision.vision_dl.embedding_recognizer import Recognition
from blackjack_vision.vision_dl.hybrid_recognizer import HybridRecognizer, load_recognizer

IMAGE = np.zeros((300, 200, 3), np.uint8)


class FakeClassic:
    min_rank_confidence = 0.4

    def __init__(self, rank, suit, confidence):
        self.prediction = Prediction(rank, suit, 1.0, 0.5, 0.0, confidence)

    def predict(self, image):
        return self.prediction


class FakeEmbedding:
    def __init__(self, name, similarity):
        card = None
        self.result = Recognition(card, similarity, name)

    def recognise(self, image):
        return self.result


def hybrid(classic, embedding):
    return HybridRecognizer(classic, embedding)


def test_rank_from_classic_and_suit_from_embedding():
    # The classical recogniser reads the rank well but the suit wrong (black read as red).
    result = hybrid(
        FakeClassic(Rank.NINE, Suit.HEARTS, 0.9), FakeEmbedding("ten_of_spades", 0.82)
    ).recognise(IMAGE)
    assert result.card == Card(Rank.NINE, Suit.SPADES)
    assert (result.rank_source, result.suit_source) == ("classic", "embedding")
    assert result.similarity == pytest.approx(0.82) and result.rank_confidence == 0.9


def test_unsure_classic_rank_falls_back_to_the_embedding():
    result = hybrid(
        FakeClassic(Rank.SEVEN, Suit.CLUBS, 0.1), FakeEmbedding("queen_of_hearts", 0.91)
    ).recognise(IMAGE)
    assert result.card == Card(Rank.QUEEN, Suit.HEARTS)
    assert result.rank_source == "embedding"


def test_both_unsure_is_unknown():
    result = hybrid(
        FakeClassic(Rank.SEVEN, Suit.CLUBS, 0.1), FakeEmbedding("queen_of_hearts", 0.5)
    ).recognise(IMAGE)
    assert result.card is None and not result.known


def test_no_embedding_card_uses_the_classic_suit():
    result = hybrid(FakeClassic(Rank.TWO, Suit.DIAMONDS, 0.8), FakeEmbedding("joker", 0.9))
    out = result.recognise(IMAGE)
    assert out.card == Card(Rank.TWO, Suit.DIAMONDS) and out.suit_source == "classic"


def test_weak_embedding_match_uses_the_classic_suit():
    out = hybrid(
        FakeClassic(Rank.TWO, Suit.DIAMONDS, 0.8), FakeEmbedding("two_of_hearts", 0.2)
    ).recognise(IMAGE)
    assert out.card == Card(Rank.TWO, Suit.DIAMONDS)


def test_load_recognizer_checks_kind_and_model(tmp_path):
    with pytest.raises(ValueError):
        load_recognizer("magic")
    with pytest.raises(SystemExit):
        load_recognizer("classic", model=tmp_path / "missing.joblib")
