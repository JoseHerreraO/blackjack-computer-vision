"""Playing cards: ranks, suits and their values in Blackjack and in the Hi-Lo count.

Cards are written with a two-character code, rank then suit: ``"AS"`` (ace of spades), ``"TH"``
(ten of hearts), ``"QD"``, ``"7C"``. :meth:`Card.parse` also accepts ``"10H"`` and lower case.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Rank(str, Enum):
    """Card rank. The value is the one-character code (``"T"`` for ten)."""

    ACE = "A"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"

    @property
    def blackjack_value(self) -> int:
        """Points in Blackjack: ace 1 (a hand may count one ace as 11), face cards 10."""
        if self is Rank.ACE:
            return 1
        if self in (Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING):
            return 10
        return int(self.value)

    @property
    def hilo(self) -> int:
        """Hi-Lo tag: ``+1`` for 2-6, ``0`` for 7-9, ``-1`` for tens and aces."""
        value = self.blackjack_value
        if 2 <= value <= 6:
            return 1
        if 7 <= value <= 9:
            return 0
        return -1

    @classmethod
    def parse(cls, text: str) -> Rank:
        """Rank from ``"A"``, ``"2"``... ``"T"``/``"10"``, ``"J"``, ``"Q"``, ``"K"`` (any case)."""
        code = text.strip().upper()
        return cls("T" if code == "10" else code)


class Suit(str, Enum):
    """Card suit. The value is the one-character code."""

    SPADES = "S"
    HEARTS = "H"
    DIAMONDS = "D"
    CLUBS = "C"

    @property
    def is_red(self) -> bool:
        return self in (Suit.HEARTS, Suit.DIAMONDS)

    @classmethod
    def parse(cls, text: str) -> Suit:
        return cls(text.strip().upper())


@dataclass(frozen=True)
class Card:
    """A playing card."""

    rank: Rank
    suit: Suit

    @property
    def code(self) -> str:
        """Two-character code such as ``"AS"`` or ``"TH"``."""
        return self.rank.value + self.suit.value

    def __str__(self) -> str:
        return self.code

    @classmethod
    def parse(cls, text: str) -> Card:
        """Card from a code such as ``"AS"``, ``"th"`` or ``"10H"``.

        Raises:
            ValueError: if the code is not a valid card.
        """
        code = text.strip()
        if len(code) < 2:
            raise ValueError(f"Invalid card code {text!r}")
        try:
            return cls(Rank.parse(code[:-1]), Suit.parse(code[-1]))
        except ValueError as exc:
            raise ValueError(f"Invalid card code {text!r}") from exc


FULL_DECK: tuple[Card, ...] = tuple(Card(rank, suit) for suit in Suit for rank in Rank)
"""The 52 cards, grouped by suit."""
