"""State machine of one Blackjack round, driven by the cards the vision pipeline sees.

The assistant cannot hear the player, so the round advances from what appears on the table:

* ``WAITING``: cards are being dealt. Once the player has two cards and the dealer an up card,
  the round moves to ``PLAYER_TURN`` (or straight to the dealer when the player has blackjack).
* ``PLAYER_TURN``: :attr:`Round.recommendation` gives the basic-strategy action for the active
  hand. A hand ends by itself on 21, bust, after the single card of a double or of a split ace,
  or when :meth:`Round.stand` / :meth:`Round.surrender` is called. Splits are announced with
  :meth:`Round.split` (two separate hands appear on the table).
* ``DEALER_TURN``: any dealer card after the player has started acting (the hole card being
  turned over, or the dealer drawing) means the player is done.
* ``FINISHED``: the dealer stands (17 or more, hitting soft 17 under H17) or busts, or every
  player hand is already lost. :meth:`Round.results` gives the outcome and payout of each hand.

Every card dealt is also passed to the optional :class:`~blackjack_vision.blackjack.shoe.ShoeTracker`
so the Hi-Lo count stays up to date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from blackjack_vision.blackjack.cards import Card, Rank
from blackjack_vision.blackjack.hand import Hand
from blackjack_vision.blackjack.shoe import ShoeTracker
from blackjack_vision.blackjack.strategy import Action, Rules, basic_strategy

MAX_HANDS = 4
"""Maximum number of hands after splitting (three splits)."""


class Phase(str, Enum):
    WAITING = "waiting"
    PLAYER_TURN = "player_turn"
    DEALER_TURN = "dealer_turn"
    FINISHED = "finished"


class Outcome(str, Enum):
    BLACKJACK = "blackjack"
    WIN = "win"
    PUSH = "push"
    LOSE = "lose"
    SURRENDER = "surrender"


@dataclass(frozen=True)
class HandResult:
    """Final result of one player hand.

    Attributes:
        hand: The hand.
        outcome: How it ended.
        payout: Net result in units of the initial bet (``+1.5`` blackjack 3:2, ``+2`` a won
            double, ``-0.5`` surrender...).
    """

    hand: Hand
    outcome: Outcome
    payout: float


@dataclass
class _PlayerHand:
    hand: Hand
    done: bool = False
    surrendered: bool = False
    split_aces: bool = False


@dataclass
class Round:
    """One round between the player and the dealer.

    Args:
        rules: Table rules (strategy and dealer behaviour).
        shoe: Tracker that counts every card dealt in the round (optional).
    """

    rules: Rules = field(default_factory=Rules)
    shoe: ShoeTracker | None = None
    dealer: Hand = field(default_factory=Hand)
    phase: Phase = Phase.WAITING
    _hands: list[_PlayerHand] = field(default_factory=lambda: [_PlayerHand(Hand())])
    _active: int = 0

    # Reading the state --------------------------------------------------------------------------

    @property
    def hands(self) -> list[Hand]:
        return [entry.hand for entry in self._hands]

    @property
    def active_index(self) -> int | None:
        """Index of the hand being played, ``None`` outside the player's turn."""
        return self._active if self.phase is Phase.PLAYER_TURN else None

    @property
    def active_hand(self) -> Hand | None:
        index = self.active_index
        return None if index is None else self._hands[index].hand

    @property
    def dealer_up(self) -> Card | None:
        return self.dealer.cards[0] if self.dealer.cards else None

    @property
    def recommendation(self) -> Action | None:
        """Basic-strategy action for the active hand, ``None`` when there is nothing to decide."""
        hand = self.active_hand
        if hand is None or self.dealer_up is None or len(hand) < 2:
            return None
        entry = self._hands[self._active]
        return basic_strategy(
            hand,
            self.dealer_up,
            self.rules,
            can_split=len(self._hands) < MAX_HANDS and not entry.split_aces,
        )

    # Events from the table ---------------------------------------------------------------------

    def deal_to_player(self, card: Card, hand_index: int | None = None) -> None:
        """A card was placed on a player hand (the active one by default)."""
        if self.phase in (Phase.DEALER_TURN, Phase.FINISHED):
            raise RuntimeError(f"No player cards are dealt during {self.phase.value}")
        index = self._active if hand_index is None else hand_index
        entry = self._hands[index]
        if entry.done:
            raise RuntimeError(f"Hand {index} is already finished")
        self._count(card)
        entry.hand.add(card)
        self._update_after_player_card(index)
        self._maybe_start_player_turn()

    def deal_to_dealer(self, card: Card) -> None:
        """A dealer card appeared (up card, hole card turned over or a drawn card)."""
        if self.phase is Phase.FINISHED:
            raise RuntimeError("The round is finished")
        self._count(card)
        self.dealer.add(card)
        if self.phase is Phase.PLAYER_TURN and len(self.dealer) > 1:
            # The dealer only acts once the player has finished.
            for entry in self._hands:
                entry.done = True
            self.phase = Phase.DEALER_TURN
        self._maybe_start_player_turn()
        self._maybe_finish()

    def split(self) -> None:
        """The active pair was split into two hands."""
        index = self._require_turn()
        entry = self._hands[index]
        if not entry.hand.is_pair:
            raise RuntimeError("Only a pair can be split")
        if len(self._hands) >= MAX_HANDS:
            raise RuntimeError(f"At most {MAX_HANDS} hands")
        aces = entry.hand.cards[0].rank is Rank.ACE
        first, second = entry.hand.cards
        self._hands[index] = _PlayerHand(Hand([first], from_split=True), split_aces=aces)
        self._hands.insert(index + 1, _PlayerHand(Hand([second], from_split=True), split_aces=aces))

    def double(self) -> None:
        """The player doubled the active hand: one more card and the hand ends."""
        index = self._require_turn()
        self._hands[index].hand.doubled = True

    def stand(self) -> None:
        """The player stands on the active hand."""
        index = self._require_turn()
        self._hands[index].done = True
        self._advance()

    def surrender(self) -> None:
        """The player surrendered the active hand (half the bet is lost)."""
        index = self._require_turn()
        entry = self._hands[index]
        entry.surrendered = entry.done = True
        self._advance()

    # Results ------------------------------------------------------------------------------------

    def results(self) -> list[HandResult]:
        """Outcome of every player hand once the round is finished."""
        if self.phase is not Phase.FINISHED:
            raise RuntimeError("The round is not finished")
        dealer_total = self.dealer.total
        dealer_bust = self.dealer.is_bust
        results = []
        for entry in self._hands:
            hand = entry.hand
            stake = 2.0 if hand.doubled else 1.0
            if entry.surrendered:
                outcome, payout = Outcome.SURRENDER, -0.5
            elif hand.is_bust:
                outcome, payout = Outcome.LOSE, -stake
            elif hand.is_blackjack and not self.dealer.is_blackjack:
                outcome, payout = Outcome.BLACKJACK, self.rules.blackjack_payout
            elif self.dealer.is_blackjack and not hand.is_blackjack:
                outcome, payout = Outcome.LOSE, -stake
            elif hand.is_blackjack or dealer_total == hand.total:
                outcome, payout = Outcome.PUSH, 0.0
            elif dealer_bust or hand.total > dealer_total:
                outcome, payout = Outcome.WIN, stake
            else:
                outcome, payout = Outcome.LOSE, -stake
            results.append(HandResult(hand, outcome, payout))
        return results

    @property
    def net(self) -> float:
        """Net result of the round in units of the initial bet."""
        return sum(result.payout for result in self.results())

    # Internals ----------------------------------------------------------------------------------

    def _count(self, card: Card) -> None:
        if self.shoe is not None:
            self.shoe.see(card)

    def _require_turn(self) -> int:
        if self.phase is not Phase.PLAYER_TURN:
            raise RuntimeError("It is not the player's turn")
        return self._active

    def _update_after_player_card(self, index: int) -> None:
        entry = self._hands[index]
        hand = entry.hand
        finished = (
            hand.is_bust
            or hand.total == 21
            or (hand.doubled and len(hand) >= 3)
            or (entry.split_aces and len(hand) >= 2)
        )
        if finished:
            entry.done = True
            if self.phase is Phase.PLAYER_TURN and index == self._active:
                self._advance()

    def _maybe_start_player_turn(self) -> None:
        if self.phase is not Phase.WAITING:
            return
        if len(self._hands[0].hand) >= 2 and self.dealer.cards:
            self.phase = Phase.PLAYER_TURN
            if self._hands[0].done:  # blackjack or 21 on the deal
                self._advance()

    def _advance(self) -> None:
        while self._active < len(self._hands) and self._hands[self._active].done:
            self._active += 1
        if self._active >= len(self._hands):
            self._active = len(self._hands) - 1
            self.phase = Phase.DEALER_TURN
            self._maybe_finish()

    def _maybe_finish(self) -> None:
        if self.phase is not Phase.DEALER_TURN:
            return
        if all(entry.hand.is_bust or entry.surrendered for entry in self._hands):
            self.phase = Phase.FINISHED
            return
        if len(self.dealer) < 2:
            return
        if self.dealer.is_bust or self.dealer.is_blackjack or self._dealer_stands():
            self.phase = Phase.FINISHED

    def _dealer_stands(self) -> bool:
        total = self.dealer.total
        if total > 17:
            return True
        if total == 17:
            return not (self.rules.dealer_hits_soft_17 and self.dealer.is_soft)
        return False
