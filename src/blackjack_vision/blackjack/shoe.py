"""Shoe composition and the Hi-Lo card count.

:class:`ShoeTracker` is fed every card the vision pipeline confirms (each physical card once,
e.g. when its track is confirmed). It keeps:

* the cards left of every rank, so the HUD can draw the shoe histogram;
* the Hi-Lo **running count** (+1 for 2-6, 0 for 7-9, -1 for tens and aces);
* the **true count**, the running count divided by the decks still in the shoe, which is what
  measures the player's advantage;
* an estimate of that advantage and a bet suggestion in betting units.

The advantage estimate is the usual rule of thumb: about -0.5 % off the top of a 6-deck S17
shoe with basic strategy, plus about 0.5 % per true-count point.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

from blackjack_vision.blackjack.cards import Card, Rank

logger = logging.getLogger(__name__)

CARDS_PER_DECK = 52
BASE_ADVANTAGE = -0.005
ADVANTAGE_PER_TRUE_COUNT = 0.005


class ShoeTracker:
    """Cards seen from a shoe of ``decks`` decks, with the Hi-Lo count.

    Args:
        decks: Decks in the shoe.
        min_decks_remaining: Floor used when dividing by the decks left, so the true count does
            not explode at the very end of the shoe.
    """

    def __init__(self, decks: int = 6, *, min_decks_remaining: float = 0.25) -> None:
        if decks < 1:
            raise ValueError(f"decks must be at least 1, got {decks}")
        self.decks = decks
        self.min_decks_remaining = min_decks_remaining
        self.running_count = 0
        self._seen: Counter[Rank] = Counter()
        self.history: list[Card] = []

    # Feeding ------------------------------------------------------------------------------------

    def see(self, card: Card) -> None:
        """Count one card that left the shoe.

        Seeing more copies of a rank than the shoe holds means a card was counted twice
        (tracking error) or the deck count is wrong; it is logged and the card still counts.
        """
        if self.remaining(card.rank) <= 0:
            logger.warning("More %s than a %d-deck shoe holds: counted twice?", card, self.decks)
        self._seen[card.rank] += 1
        self.running_count += card.rank.hilo
        self.history.append(card)

    def undo(self, card: Card) -> None:
        """Remove a card that was counted by mistake."""
        if self._seen[card.rank] <= 0:
            raise ValueError(f"{card} was not counted")
        self._seen[card.rank] -= 1
        self.running_count -= card.rank.hilo
        for index in range(len(self.history) - 1, -1, -1):
            if self.history[index] == card:
                del self.history[index]
                break

    def reshuffle(self) -> None:
        """New shoe: everything back to the start."""
        self.running_count = 0
        self._seen.clear()
        self.history.clear()

    # Composition --------------------------------------------------------------------------------

    @property
    def total_cards(self) -> int:
        return self.decks * CARDS_PER_DECK

    @property
    def cards_seen(self) -> int:
        return sum(self._seen.values())

    @property
    def cards_remaining(self) -> int:
        return max(0, self.total_cards - self.cards_seen)

    @property
    def decks_remaining(self) -> float:
        return self.cards_remaining / CARDS_PER_DECK

    @property
    def penetration(self) -> float:
        """Fraction of the shoe already dealt."""
        return min(1.0, self.cards_seen / self.total_cards)

    def remaining(self, rank: Rank) -> int:
        """Cards of ``rank`` left in the shoe (4 per deck and rank)."""
        return max(0, 4 * self.decks - self._seen[rank])

    def remaining_by_value(self) -> dict[int, int]:
        """Cards left grouped by Blackjack value (1 = ace, 10 = any ten-value card)."""
        counts: Counter[int] = Counter()
        for rank in Rank:
            counts[rank.blackjack_value] += self.remaining(rank)
        return {value: counts[value] for value in range(1, 11)}

    def probability(self, value: int) -> float:
        """Probability that the next card has this Blackjack value (1 = ace)."""
        left = self.cards_remaining
        return self.remaining_by_value().get(value, 0) / left if left else 0.0

    # Count --------------------------------------------------------------------------------------

    @property
    def true_count(self) -> float:
        """Running count per remaining deck."""
        return self.running_count / max(self.decks_remaining, self.min_decks_remaining)

    @property
    def advantage(self) -> float:
        """Estimated player advantage (fraction of the bet) with basic strategy."""
        return BASE_ADVANTAGE + ADVANTAGE_PER_TRUE_COUNT * self.true_count

    def bet_units(self, *, max_units: int = 8) -> int:
        """Suggested bet in units: 1 unit until the count is favourable, then ``TC - 1``.

        The true count is floored, as players do at the table. The ramp only grows when the
        estimated advantage is positive (true count about +1 or more).
        """
        units = math.floor(self.true_count) - 1
        return int(min(max_units, max(1, units)))

    def summary(self) -> str:
        """One line for the HUD or the log."""
        return (
            f"RC {self.running_count:+d}  TC {self.true_count:+.1f}  "
            f"edge {100 * self.advantage:+.1f}%  decks left {self.decks_remaining:.1f}  "
            f"bet {self.bet_units()}u"
        )
