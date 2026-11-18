"""Blackjack basic strategy for multi-deck shoes (4-8 decks).

The tables are the standard ones for a shoe where the dealer **stands on soft 17** (S17),
doubling is allowed on any two cards and, optionally, after splitting (DAS). Dealer hitting soft
17 (H17) and late surrender are handled as the usual small set of changes on top. Columns are
the dealer's up card 2-9, ten and ace.

Table codes:

=====  ==========================================================
``H``  hit
``S``  stand
``D``  double if allowed, otherwise hit
``Ds`` double if allowed, otherwise stand
``P``  split
``Ph`` split if doubling after split is allowed, otherwise hit
``Rh`` surrender if allowed, otherwise hit
``Rs`` surrender if allowed, otherwise stand
``Rp`` surrender if allowed, otherwise split
=====  ==========================================================

References: E. Thorp, *Beat the Dealer* (1966); the 4-8 deck tables published by the Wizard
of Odds (wizardofodds.com/games/blackjack/strategy/4-decks/).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from blackjack_vision.blackjack.cards import Card, Rank
from blackjack_vision.blackjack.hand import Hand

DEALER_COLUMNS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
"""Dealer up card values, in table order (11 = ace)."""


class Action(str, Enum):
    """What the player should do."""

    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"
    SURRENDER = "surrender"

    @property
    def spanish(self) -> str:
        """Short Spanish label for the HUD and the voice feedback."""
        return {
            Action.HIT: "Pide carta",
            Action.STAND: "Plántate",
            Action.DOUBLE: "Dobla",
            Action.SPLIT: "Separa",
            Action.SURRENDER: "Ríndete",
        }[self]


@dataclass(frozen=True)
class Rules:
    """Table rules that change the strategy.

    Attributes:
        decks: Decks in the shoe (the tables are valid for 4-8; also used by the count).
        dealer_hits_soft_17: The dealer hits soft 17 (H17) instead of standing (S17).
        double_after_split: Doubling is allowed after splitting (DAS).
        surrender: Late surrender is offered.
        blackjack_payout: Payout of a natural (1.5 for 3:2, 1.2 for 6:5).
    """

    decks: int = 6
    dealer_hits_soft_17: bool = False
    double_after_split: bool = True
    surrender: bool = False
    blackjack_payout: float = 1.5

    def __post_init__(self) -> None:
        if self.decks < 1:
            raise ValueError(f"decks must be at least 1, got {self.decks}")
        if self.blackjack_payout <= 0:
            raise ValueError("blackjack_payout must be positive")


def _rows(text: str) -> dict[int, tuple[str, ...]]:
    table = {}
    for line in text.strip().splitlines():
        key, *codes = line.split()
        if len(codes) != len(DEALER_COLUMNS):
            raise ValueError(f"Bad strategy row {line!r}")
        table[int(key)] = tuple(codes)
    return table


#                   2  3  4  5  6  7  8  9  T  A
HARD = _rows("""
    8              H  H  H  H  H  H  H  H  H  H
    9              H  D  D  D  D  H  H  H  H  H
    10             D  D  D  D  D  D  D  D  H  H
    11             D  D  D  D  D  D  D  D  D  H
    12             H  H  S  S  S  H  H  H  H  H
    13             S  S  S  S  S  H  H  H  H  H
    14             S  S  S  S  S  H  H  H  H  H
    15             S  S  S  S  S  H  H  H  Rh H
    16             S  S  S  S  S  H  H  Rh Rh Rh
    17             S  S  S  S  S  S  S  S  S  S
""")
"""Hard totals (S17). Totals below 8 always hit, above 17 always stand."""

#                   2  3  4  5  6  7  8  9  T  A
SOFT = _rows("""
    12             H  H  H  H  H  H  H  H  H  H
    13             H  H  H  D  D  H  H  H  H  H
    14             H  H  H  D  D  H  H  H  H  H
    15             H  H  D  D  D  H  H  H  H  H
    16             H  H  D  D  D  H  H  H  H  H
    17             H  D  D  D  D  H  H  H  H  H
    18             S  Ds Ds Ds Ds S  S  H  H  H
    19             S  S  S  S  S  S  S  S  S  S
    20             S  S  S  S  S  S  S  S  S  S
""")
"""Soft totals (an ace counted as 11), S17. Soft 12 is a pair of aces that cannot be split."""

#                   2  3  4  5  6  7  8  9  T  A
PAIRS = _rows("""
    1              P  P  P  P  P  P  P  P  P  P
    2              Ph Ph P  P  P  P  H  H  H  H
    3              Ph Ph P  P  P  P  H  H  H  H
    4              H  H  H  Ph Ph H  H  H  H  H
    5              -  -  -  -  -  -  -  -  -  -
    6              Ph P  P  P  P  H  H  H  H  H
    7              P  P  P  P  P  P  H  H  H  H
    8              P  P  P  P  P  P  P  P  P  P
    9              P  P  P  P  P  S  P  P  S  S
    10             -  -  -  -  -  -  -  -  -  -
""")
"""Pairs, by the value of each card (1 = aces). ``-`` means "never split, play the total"."""

# Changes when the dealer hits soft 17 (4-8 decks): (table, player key, dealer value) -> code.
H17_CHANGES = {
    ("hard", 11, 11): "D",
    ("hard", 15, 11): "Rh",
    ("hard", 17, 11): "Rs",
    ("soft", 18, 2): "Ds",
    ("soft", 19, 6): "Ds",
    ("pair", 8, 11): "Rp",
}


def dealer_value(card: Card | Rank | int) -> int:
    """Value used to index the tables: 2-10, and 11 for an ace."""
    if isinstance(card, Card):
        card = card.rank
    if isinstance(card, Rank):
        return 11 if card is Rank.ACE else card.blackjack_value
    value = int(card)
    if value == 1:
        return 11
    if value not in DEALER_COLUMNS:
        raise ValueError(f"Invalid dealer up card value {card}")
    return value


def table_code(table: str, key: int, dealer: int, rules: Rules) -> str:
    """Raw code of a table cell, with the H17 changes applied."""
    if rules.dealer_hits_soft_17 and (table, key, dealer) in H17_CHANGES:
        return H17_CHANGES[(table, key, dealer)]
    source = {"hard": HARD, "soft": SOFT, "pair": PAIRS}[table]
    return source[key][DEALER_COLUMNS.index(dealer)]


def basic_strategy(
    hand: Hand,
    dealer_up: Card | Rank | int,
    rules: Rules | None = None,
    *,
    can_double: bool | None = None,
    can_split: bool = True,
    can_surrender: bool | None = None,
) -> Action:
    """Basic-strategy action for ``hand`` against the dealer's up card.

    Args:
        hand: The player's hand (at least two cards).
        dealer_up: Dealer up card (a :class:`Card`, a :class:`Rank` or a value, ace = 1 or 11).
        rules: Table rules, the defaults if ``None``.
        can_double: Doubling is possible now. By default only on the first two cards (and,
            after a split, only with DAS).
        can_split: Splitting is possible now (e.g. not after the maximum number of splits).
        can_surrender: Surrendering is possible now. By default only on the first two cards of
            an unsplit hand and when the rules offer it.

    Raises:
        ValueError: if the hand has fewer than two cards or is already bust.
    """
    rules = rules or Rules()
    if len(hand) < 2:
        raise ValueError("The strategy needs at least two player cards")
    if hand.is_bust:
        raise ValueError("The hand is bust")
    dealer = dealer_value(dealer_up)
    first_decision = len(hand) == 2
    if can_double is None:
        can_double = first_decision and (not hand.from_split or rules.double_after_split)
    if can_surrender is None:
        can_surrender = rules.surrender and first_decision and not hand.from_split

    if hand.is_pair and can_split:
        code = table_code("pair", hand.pair_value, dealer, rules)
        if code == "P" or (code == "Ph" and rules.double_after_split):
            return Action.SPLIT
        if code == "Rp":
            return Action.SURRENDER if can_surrender else Action.SPLIT
        if code not in ("-", "Ph"):
            return _resolve(code, can_double, can_surrender)

    total = hand.total
    if hand.is_soft:
        if total >= 20:
            return Action.STAND
        code = table_code("soft", total, dealer, rules)
    elif total < 8:
        return Action.HIT
    elif total >= 18:
        return Action.STAND
    else:
        code = table_code("hard", total, dealer, rules)
    return _resolve(code, can_double, can_surrender)


def _resolve(code: str, can_double: bool, can_surrender: bool) -> Action:
    if code == "H":
        return Action.HIT
    if code == "S":
        return Action.STAND
    if code == "D":
        return Action.DOUBLE if can_double else Action.HIT
    if code == "Ds":
        return Action.DOUBLE if can_double else Action.STAND
    if code == "Rh":
        return Action.SURRENDER if can_surrender else Action.HIT
    if code == "Rs":
        return Action.SURRENDER if can_surrender else Action.STAND
    raise ValueError(f"Unexpected strategy code {code!r}")


INSURANCE_TRUE_COUNT = 3.0
"""Hi-Lo true count from which insurance (and even money) becomes a good bet."""


def take_insurance(true_count: float) -> bool:
    """Insurance is only worth it with a high count (basic strategy alone never takes it)."""
    return true_count >= INSURANCE_TRUE_COUNT
