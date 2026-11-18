"""A Blackjack hand: totals, soft hands, pairs, blackjack and bust."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from blackjack_vision.blackjack.cards import Card, Rank


@dataclass
class Hand:
    """Cards of one player hand or of the dealer.

    Attributes:
        cards: Cards in the order they were dealt.
        from_split: The hand comes from splitting a pair (a two-card 21 is then not a
            blackjack, and some rules restrict further actions).
        doubled: The bet was doubled (one more card only, payout x2).
    """

    cards: list[Card] = field(default_factory=list)
    from_split: bool = False
    doubled: bool = False

    @classmethod
    def of(cls, *codes: str) -> Hand:
        """Hand from card codes, e.g. ``Hand.of("AS", "7D")`` (handy in tests and notebooks)."""
        return cls([Card.parse(code) for code in codes])

    def add(self, card: Card) -> None:
        self.cards.append(card)

    def extend(self, cards: Iterable[Card]) -> None:
        self.cards.extend(cards)

    def __len__(self) -> int:
        return len(self.cards)

    @property
    def hard_total(self) -> int:
        """Total counting every ace as 1."""
        return sum(card.rank.blackjack_value for card in self.cards)

    @property
    def total(self) -> int:
        """Best total: one ace counts as 11 when that does not bust the hand."""
        hard = self.hard_total
        if self.has_ace and hard + 10 <= 21:
            return hard + 10
        return hard

    @property
    def has_ace(self) -> bool:
        return any(card.rank is Rank.ACE for card in self.cards)

    @property
    def is_soft(self) -> bool:
        """An ace is being counted as 11."""
        return self.has_ace and self.hard_total + 10 <= 21

    @property
    def is_blackjack(self) -> bool:
        """Two-card 21 that does not come from a split."""
        return len(self.cards) == 2 and self.total == 21 and not self.from_split

    @property
    def is_bust(self) -> bool:
        return self.hard_total > 21

    @property
    def is_pair(self) -> bool:
        """Two cards of the same value (any two ten-value cards count as a pair)."""
        return (
            len(self.cards) == 2
            and self.cards[0].rank.blackjack_value == self.cards[1].rank.blackjack_value
        )

    @property
    def pair_value(self) -> int | None:
        """Blackjack value of the paired cards (1 for aces), or ``None`` if not a pair."""
        return self.cards[0].rank.blackjack_value if self.is_pair else None

    def describe(self) -> str:
        """Short text such as ``"A7 (soft 18)"`` or ``"T6 (16)"``."""
        cards = "".join(card.rank.value for card in self.cards)
        if self.is_bust:
            return f"{cards} (bust {self.total})"
        if self.is_blackjack:
            return f"{cards} (blackjack)"
        kind = "soft " if self.is_soft else ""
        return f"{cards} ({kind}{self.total})"
