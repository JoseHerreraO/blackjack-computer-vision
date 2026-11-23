"""Tests on real rectified cards from the glasses stream (see tests/data/real_crops/README.md).

They pin down the behaviour that the synthetic cards could not show: our deck's layout (a 3 or a
5 has a single pip in the centre), the pale orange-red ink of the video and the red and yellow
artwork of black face cards.
"""

from pathlib import Path

import pytest

from blackjack_vision.blackjack.cards import Card, Rank
from blackjack_vision.capture.sources import read_image
from blackjack_vision.security.aces import AceReader
from blackjack_vision.vision_classic.recognition import (
    RED_INK_THRESHOLD,
    card_colour,
    index_corners,
    picture_box,
    red_ink_fraction,
)
from blackjack_vision.vision_classic.suit_templates import SuitTemplates, corner_symbols

CROPS = sorted((Path(__file__).parent / "data" / "real_crops").glob("*.png"))
FACES = (Rank.JACK, Rank.QUEEN, Rank.KING)


def _label(path: Path) -> Card:
    return Card.parse(path.stem.split("_")[0])


@pytest.fixture(scope="module")
def crops() -> list[tuple[str, Card, object]]:
    assert len(CROPS) >= 13, "real crops missing"
    return [(path.stem, _label(path), read_image(path)) for path in CROPS]


def test_colour_of_number_cards_and_aces(crops):
    for name, card, image in crops:
        if card.rank in FACES:
            continue
        assert card_colour(image) is card.suit.is_red, name


def test_black_face_cards_do_not_read_red(crops):
    for name, card, image in crops:
        if card.rank not in FACES or card.suit.is_red:
            continue
        assert picture_box(image) is not None, name
        assert red_ink_fraction(image) < RED_INK_THRESHOLD, name  # artwork left out
        assert card_colour(image) is None, name  # colour left to the suit shape


def test_red_face_cards_read_red_but_colour_stays_unknown(crops):
    """The red index ink of a red face card is measured, but the colour is still left to the
    suit shape, like for every face card."""
    red_faces = [(n, c, i) for n, c, i in crops if c.rank in FACES and c.suit.is_red]
    assert red_faces, "no red face card among the crops"
    for name, _, image in red_faces:
        assert picture_box(image) is not None, name
        assert red_ink_fraction(image) >= RED_INK_THRESHOLD, name
        assert card_colour(image) is None, name


def test_pips_are_counted_and_only_aces_pass(crops):
    templates = SuitTemplates().fit(
        [index_corners(image) for _, _, image in crops], [card.suit for _, card, _ in crops]
    )
    reader = AceReader(templates)
    for name, card, image in crops:
        if card.rank not in FACES:
            assert len(reader.body_pips(image)) == card.rank.blackjack_value, name
        reading = reader.read(image)
        assert reading.is_ace == (card.rank is Rank.ACE), name
        if reading.is_ace:
            assert reading.suit is card.suit, name


def test_suit_by_shape_alone_leave_one_out(crops):
    """Each crop is classified by templates learned from the other crops, without colour."""
    for name, card, image in crops:
        others = [(c, i) for n, c, i in crops if n != name]
        templates = SuitTemplates().fit(
            [index_corners(i) for _, i in others], [c.suit for c, _ in others]
        )
        suit, margin = templates.classify(corner_symbols(index_corners(image)))
        assert suit is card.suit and margin > 0, name
