import pytest

from blackjack_vision.blackjack.cards import FULL_DECK, Card, Rank, Suit


@pytest.mark.parametrize(
    ("text", "rank", "suit"),
    [
        ("AS", Rank.ACE, Suit.SPADES),
        ("th", Rank.TEN, Suit.HEARTS),
        ("10D", Rank.TEN, Suit.DIAMONDS),
        (" qc ", Rank.QUEEN, Suit.CLUBS),
        ("7H", Rank.SEVEN, Suit.HEARTS),
    ],
)
def test_parse(text, rank, suit):
    card = Card.parse(text)
    assert (card.rank, card.suit) == (rank, suit)
    assert Card.parse(card.code) == card


@pytest.mark.parametrize("text", ["", "A", "1S", "AX", "11H", "ZZ"])
def test_parse_rejects_invalid_codes(text):
    with pytest.raises(ValueError):
        Card.parse(text)


def test_blackjack_values_and_hilo_tags():
    values = {rank: rank.blackjack_value for rank in Rank}
    assert values[Rank.ACE] == 1
    assert values[Rank.SEVEN] == 7
    assert {values[r] for r in (Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING)} == {10}
    assert [Rank.parse(code).hilo for code in "23456789TJQKA"] == [1] * 5 + [0] * 3 + [-1] * 5
    assert sum(card.rank.hilo for card in FULL_DECK) == 0  # a balanced count


def test_full_deck():
    assert len(FULL_DECK) == len(set(FULL_DECK)) == 52
    assert sum(card.suit.is_red for card in FULL_DECK) == 26
    assert str(Card(Rank.KING, Suit.DIAMONDS)) == "KD"
