"""Live table: turns the cards the pipeline recognises into a Blackjack round and advice.

The glasses look at the table from the player's seat, so the dealer's cards are on the far side
(top of the image) and the player's on the near side (bottom). A horizontal ``dealer_line``, as a
fraction of the image height, separates the two zones. Every card the pipeline names for the first
time (:attr:`~blackjack_vision.detection_tracking.pipeline.PipelineResult.new_cards`) is dealt to
the hand of its zone in :class:`~blackjack_vision.blackjack.game.Round`, which also feeds the
Hi-Lo count of the shared :class:`~blackjack_vision.blackjack.shoe.ShoeTracker`.

What the table infers on its own:

* **Hole card.** A card in the dealer zone that stays unrecognised for ``hidden_frames`` frames
  while the dealer shows a single card is the face-down hole card. It is shown as such and not
  counted until it is turned over (then it is recognised and dealt like any other card).
* **Re-found cards.** A card hidden for a while (by a hand, for instance) loses its track and
  comes back with a new one, so the pipeline names it again. If a card with the same code was
  already dealt to the same zone in this round and its track is no longer in view, it is the same
  card and is not dealt (or counted) twice. Two equal cards in view at the same time are two cards
  (several decks).
* **New round.** When the table stays empty for ``clear_frames`` frames, or a card appears after
  the round finished, a new round starts. The shoe, and so the count, carries over.
* **Advice.** Whenever the basic-strategy recommendation changes it is queued as speech (Spanish,
  e.g. "Pide carta"). When the round ends its result is spoken too.

Limits: splits and doubles are not seen by the camera, so the player's hand is a single hand and a
third card after "Dobla" counts as a normal hit. The result is still right for a single hand.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

from blackjack_vision.blackjack.cards import Card
from blackjack_vision.blackjack.game import Outcome, Phase, Round
from blackjack_vision.blackjack.shoe import ShoeTracker
from blackjack_vision.blackjack.strategy import Action, Rules

logger = logging.getLogger(__name__)

OUTCOME_SPEECH = {
    Outcome.BLACKJACK: "Blackjack",
    Outcome.WIN: "Ganas",
    Outcome.PUSH: "Empate",
    Outcome.LOSE: "Pierdes",
    Outcome.SURRENDER: "Te has rendido",
}


class Zone(str, Enum):
    DEALER = "dealer"
    PLAYER = "player"


class SeenCard(Protocol):
    """What the table needs from a tracked card (``pipeline.TrackedCard`` fits)."""

    track_id: int
    corners: np.ndarray | None
    card: Card | None


@dataclass
class TableUpdate:
    """What happened in one :meth:`Table.update`."""

    speech: list[str] = field(default_factory=list)
    """Phrases to say through the glasses, in order."""
    events: list[str] = field(default_factory=list)
    """Short log lines (new round, card dealt, card ignored...)."""


class Table:
    """See the module docstring.

    Args:
        rules: Table rules for the strategy and the dealer.
        decks: Decks in the shoe (for the true count).
        dealer_line: Height of the line between the dealer's and the player's zone, as a
            fraction of the image height (cards above it are the dealer's).
        hidden_frames: Frames a dealer card must stay unrecognised to count as the hole card.
        clear_frames: Frames without cards that start a new round.
    """

    def __init__(
        self,
        rules: Rules | None = None,
        *,
        decks: int = 6,
        dealer_line: float = 0.5,
        hidden_frames: int = 15,
        clear_frames: int = 15,
    ) -> None:
        if not 0.0 < dealer_line < 1.0:
            raise ValueError(f"dealer_line must be in (0, 1), got {dealer_line}")
        self.rules = rules or Rules(decks=decks)
        self.shoe = ShoeTracker(decks=decks)
        self.dealer_line = dealer_line
        self.hidden_frames = hidden_frames
        self.clear_frames = clear_frames
        self.round = Round(rules=self.rules, shoe=self.shoe)
        self.rounds_played = 0
        self.last_results: list[str] = []
        self.hole_card_visible = False
        self._zones: dict[int, Zone] = {}
        self._dealt: list[tuple[str, Zone, int]] = []
        self._unknown_frames: dict[int, int] = {}
        self._empty_frames = 0
        self._last_advice: Action | None = None

    # Reading the state --------------------------------------------------------------------------

    @property
    def advice(self) -> Action | None:
        return self.round.recommendation

    def zone_of(self, card: SeenCard, image_height: int) -> Zone:
        """Zone of a card by the height of its centre in the image."""
        if card.corners is None:
            return Zone.PLAYER
        centre_y = float(np.asarray(card.corners)[:, 1].mean())
        return Zone.DEALER if centre_y < self.dealer_line * image_height else Zone.PLAYER

    # Feeding ------------------------------------------------------------------------------------

    def update(
        self, visible: Sequence[SeenCard], new_cards: Sequence[SeenCard], image_height: int
    ) -> TableUpdate:
        """Feed the cards of one frame (``visible``) and those recognised in it (``new_cards``)."""
        update = TableUpdate()

        if not visible:
            self._empty_frames += 1
            if self._empty_frames == self.clear_frames and self._round_has_cards():
                self._new_round(update, "table cleared")
        else:
            self._empty_frames = 0

        in_view = {card.track_id for card in visible}
        for card in new_cards:
            if card.card is None:
                continue
            zone = self.zone_of(card, image_height)
            self._zones[card.track_id] = zone
            if self.round.phase is Phase.FINISHED:
                self._new_round(update, "new cards after the round ended")
            if self._refound(card.card, zone, in_view):
                update.events.append(
                    f"{card.card} seen again in the {zone.value} zone, not recounted"
                )
                continue
            if self._deal(card.card, zone, update):
                self._dealt.append((card.card.code, zone, card.track_id))

        self._update_hole_card(visible, image_height)
        self._update_advice(update)
        return update

    def reset(self, *, reshuffle: bool = False) -> None:
        """Start over: a new round, and a fresh count if the shoe was reshuffled."""
        if reshuffle:
            self.shoe.reshuffle()
        self.round = Round(rules=self.rules, shoe=self.shoe)
        self._zones.clear()
        self._dealt.clear()
        self._unknown_frames.clear()
        self._last_advice = None
        self.hole_card_visible = False

    # Internals ----------------------------------------------------------------------------------

    def _round_has_cards(self) -> bool:
        return bool(self.round.dealer.cards) or any(len(hand) for hand in self.round.hands)

    def _new_round(self, update: TableUpdate, reason: str) -> None:
        self.rounds_played += 1
        self.reset()
        update.events.append(f"New round ({reason})")

    def _refound(self, card: Card, zone: Zone, in_view: set[int]) -> bool:
        """An equal card already dealt to this zone whose track has been lost."""
        return any(
            code == card.code and dealt_zone is zone and track_id not in in_view
            for code, dealt_zone, track_id in self._dealt
        )

    def _deal(self, card: Card, zone: Zone, update: TableUpdate) -> bool:
        before = self.round.phase
        try:
            if zone is Zone.DEALER:
                self.round.deal_to_dealer(card)
            else:
                self.round.deal_to_player(card)
        except RuntimeError as error:
            update.events.append(f"Ignored {card} for the {zone.value}: {error}")
            return False
        update.events.append(f"{card} to the {zone.value}")
        if self.round.phase is Phase.FINISHED and before is not Phase.FINISHED:
            self._announce_results(update)
        return True

    def _announce_results(self, update: TableUpdate) -> None:
        results = self.round.results()
        self.last_results = [result.outcome.value for result in results]
        phrases = [OUTCOME_SPEECH[result.outcome] for result in results]
        update.speech.append(", ".join(phrases))
        update.events.append(
            f"Round over: {', '.join(self.last_results)} (net {self.round.net:+.1f})"
        )

    def _update_hole_card(self, visible: Sequence[SeenCard], image_height: int) -> None:
        seen = set()
        for card in visible:
            seen.add(card.track_id)
            if card.card is None and self.zone_of(card, image_height) is Zone.DEALER:
                self._unknown_frames[card.track_id] = self._unknown_frames.get(card.track_id, 0) + 1
            else:
                self._unknown_frames.pop(card.track_id, None)
        for track_id in list(self._unknown_frames):
            if track_id not in seen:
                del self._unknown_frames[track_id]
        self.hole_card_visible = len(self.round.dealer) == 1 and any(
            frames >= self.hidden_frames for frames in self._unknown_frames.values()
        )

    def _update_advice(self, update: TableUpdate) -> None:
        advice = self.advice
        if advice is not None and advice != self._last_advice:
            update.speech.append(advice.spanish)
        self._last_advice = advice
