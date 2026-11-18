import cv2
import numpy as np
import pytest
from synthetic_faces import PAPER, card_face, draw_suit, photographed

from blackjack_vision.blackjack.cards import FULL_DECK, Card, Rank, Suit
from blackjack_vision.vision_classic.recognition import index_corners
from blackjack_vision.vision_classic.suit_templates import (
    TEMPLATE_SIZE,
    SuitTemplates,
    corner_symbols,
    largest_blob_mask,
    ncc,
    normalise_symbol,
    suit_symbol_mask,
)


def _symbol(suit: Suit, size: float, centre=(60.0, 60.0), canvas=120) -> np.ndarray:
    image = np.full((canvas, canvas, 3), PAPER, np.uint8)
    draw_suit(image, suit, centre, size)
    return image


def _patch(image: np.ndarray) -> np.ndarray:
    mask = suit_symbol_mask(image, upscale=1)
    assert mask is not None
    return normalise_symbol(mask)


@pytest.fixture(scope="module")
def templates() -> SuitTemplates:
    rng = np.random.default_rng(0)
    cards = [card for card in FULL_DECK if card.rank in (Rank.ACE, Rank.FIVE, Rank.QUEEN)]
    images = [photographed(card_face(card), rng) for card in cards for _ in range(2)]
    labels = [card.suit for card in cards for _ in range(2)]
    return SuitTemplates().fit([index_corners(image) for image in images], labels)


def test_normalisation_removes_position_and_size():
    for suit in Suit:
        small = _patch(_symbol(suit, 40, centre=(45, 50)))
        large = _patch(_symbol(suit, 80, centre=(62, 58)))
        assert small.shape == (TEMPLATE_SIZE, TEMPLATE_SIZE)
        same = ncc(small, large)
        others = [ncc(small, _patch(_symbol(other, 80))) for other in Suit if other is not suit]
        assert same > 0.9
        assert same > max(others)


def test_normalise_rejects_an_empty_mask():
    with pytest.raises(ValueError):
        normalise_symbol(np.zeros((10, 10), np.uint8))


def test_symbol_is_the_lowest_blob_of_the_index():
    corner = index_corners(card_face(Card(Rank.TEN, Suit.HEARTS)))[0]
    mask = suit_symbol_mask(corner)
    rows = np.nonzero(mask.any(axis=1))[0]
    assert rows.min() > mask.shape[0] * 0.4  # below the "10"
    assert suit_symbol_mask(np.full((96, 44, 3), 245, np.uint8)) is None


def test_largest_blob_share():
    image = np.full((100, 100, 3), 245, np.uint8)
    assert largest_blob_mask(image) == (None, 0.0)
    cv2.circle(image, (50, 50), 20, (0, 0, 0), -1)
    cv2.circle(image, (90, 90), 3, (0, 0, 0), -1)
    mask, share = largest_blob_mask(image)
    assert mask[50, 50] == 255 and mask[90, 90] == 0
    assert 0.9 < share < 1.0


def test_templates_classify_every_suit(templates):
    assert set(templates.suits) == set(Suit)
    rng = np.random.default_rng(1)
    for suit in Suit:
        for rank in (Rank.TWO, Rank.NINE, Rank.KING):
            card = photographed(card_face(Card(rank, suit)), rng)
            best, margin = templates.classify(corner_symbols(index_corners(card)))
            assert best is suit and margin > 0


def test_colour_restriction_and_missing_symbols(templates):
    symbols = corner_symbols(index_corners(card_face(Card(Rank.FOUR, Suit.SPADES))))
    best, _ = templates.classify(symbols, allowed=[Suit.HEARTS, Suit.DIAMONDS])
    assert best in (Suit.HEARTS, Suit.DIAMONDS)
    assert templates.scores([]) is None
    assert templates.classify([]) == (None, 0.0)


def test_upside_down_symbol_needs_both_orientations(templates):
    pip = _patch(cv2.rotate(_symbol(Suit.SPADES, 80), cv2.ROTATE_180))
    best, _ = templates.classify([pip], allowed=[Suit.SPADES, Suit.CLUBS], both_orientations=True)
    assert best is Suit.SPADES


def test_training_errors():
    with pytest.raises(ValueError):
        SuitTemplates().fit([index_corners(card_face(Card(Rank.ACE, Suit.SPADES)))], [Suit.SPADES])
    with pytest.raises(RuntimeError):
        SuitTemplates().scores([np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), np.float32)])
