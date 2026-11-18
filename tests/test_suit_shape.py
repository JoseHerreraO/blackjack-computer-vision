import numpy as np
import pytest
from synthetic_faces import PAPER, card_face, draw_suit

from blackjack_vision.blackjack.cards import Card, Rank, Suit
from blackjack_vision.vision_classic.recognition import index_corners
from blackjack_vision.vision_classic.suit_shape import (
    N_FEATURES,
    SuitShapeClassifier,
    shape_features,
    suit_symbol_mask,
)


def _symbol(suit: Suit, size: float, centre=(60.0, 60.0), canvas=120) -> np.ndarray:
    image = np.full((canvas, canvas, 3), PAPER, np.uint8)
    draw_suit(image, suit, centre, size)
    return image


def _mask(image: np.ndarray) -> np.ndarray:
    mask = suit_symbol_mask(image, upscale=1)
    assert mask is not None
    return mask


@pytest.mark.parametrize("suit", list(Suit))
def test_features_do_not_depend_on_position_or_size(suit):
    small = shape_features(_mask(_symbol(suit, 40, centre=(45, 50))))
    large = shape_features(_mask(_symbol(suit, 80, centre=(62, 58))))
    assert small.shape == (N_FEATURES,)
    same = np.linalg.norm(small - large)
    for other in Suit:
        if other is not suit:
            assert same < np.linalg.norm(small - shape_features(_mask(_symbol(other, 80))))


def test_solidity_separates_lobed_and_convex_symbols():
    solidity = {suit: shape_features(_mask(_symbol(suit, 80)))[2] for suit in Suit}
    assert solidity[Suit.DIAMONDS] > 0.95
    assert solidity[Suit.CLUBS] < solidity[Suit.SPADES] < solidity[Suit.DIAMONDS]


def test_symbol_is_the_lowest_blob_of_the_index():
    corner = index_corners(card_face(Card(Rank.TEN, Suit.HEARTS)))[0]
    mask = suit_symbol_mask(corner)
    rows = np.nonzero(mask.any(axis=1))[0]
    assert rows.min() > mask.shape[0] * 0.4  # below the "10"
    assert suit_symbol_mask(np.full((96, 44, 3), 245, np.uint8)) is None


def test_classifier_reads_every_suit_from_the_corners():
    cards = [Card(rank, suit) for rank in (Rank.ACE, Rank.FIVE, Rank.QUEEN) for suit in Suit]
    corners = [index_corners(card_face(card)) for card in cards]
    classifier = SuitShapeClassifier().fit(corners, [card.suit for card in cards])

    for suit in Suit:
        scores = classifier.scores(index_corners(card_face(Card(Rank.NINE, suit))))
        assert classifier.classes[int(np.argmax(scores))] == suit.value
    assert classifier.scores([np.full((96, 44, 3), 245, np.uint8)]) is None


def test_classifier_needs_two_suits():
    corners = [index_corners(card_face(Card(Rank.ACE, Suit.SPADES)))]
    with pytest.raises(ValueError):
        SuitShapeClassifier().fit(corners, [Suit.SPADES])
    with pytest.raises(RuntimeError):
        SuitShapeClassifier().scores(corners[0])
