"""Live table: turns the cards the pipeline recognises into a Blackjack round and advice.

The glasses look at the table from the player's seat, so the dealer's cards are on the far side
(top of the image) and the player's on the near side (bottom). When the far edge of the mat is
known (:func:`~blackjack_vision.vision_classic.mat.mat_far_edge`), a card whose centre is less than
``dealer_depth`` card heights below that edge is the dealer's: the edge moves with the head, the
cards keep their place on the table. Otherwise a horizontal ``dealer_line``, as a fraction of the
image height, separates the two zones. Every card the pipeline names for the first
time (:attr:`~blackjack_vision.detection_tracking.pipeline.PipelineResult.new_cards`) is dealt to
the hand of its zone in :class:`~blackjack_vision.blackjack.game.Round`, which also feeds the
Hi-Lo count of the shared :class:`~blackjack_vision.blackjack.shoe.ShoeTracker`.

What the table infers on its own:

* **Hole card.** A card in the dealer zone that stays unrecognised for ``hidden_frames`` frames
  while the dealer shows a single card is the face-down hole card. It is shown as such and not
  counted until it is turned over (then it is recognised and dealt like any other card).
* **Re-found cards.** A card hidden for a while (by a hand, for instance) loses its track and
  comes back with a new one, so the pipeline names it again. If a card with the same code was
  already dealt to the same zone in this round, its track is no longer in view, and the new one
  appears near where it was last seen (``refound_distance`` card sizes), it is the same card: it
  is not dealt (or counted) twice and the dealt card takes over the new track, so a later equal
  card is a new card. Two equal cards in view at the same time are two cards (several decks).
* **New round.** A card after the round finished starts a new round. An empty view starts one
  after ``clear_frames`` frames only when that cannot lose a hand: the round is over (or has no
  cards), or the mat is in view and empty (the cards were collected). With the head turned away
  (``mat_in_view=False``) the round waits for the cards to come back. The shoe, and so the count,
  carries over.
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
        clear_frames: Frames without cards that start a new round (see the module docstring).
        refound_distance: How far, in card sizes, a lost card may reappear and still be taken
            for the same card.
        dealer_depth: With the mat's far edge known, how far below it (in heights of the card
            itself) the dealer's zone reaches. On the 2026-11-25 recording the dealer's cards
            were about 1 card height below the edge and the player's more than 2.5.
    """

    def __init__(
        self,
        rules: Rules | None = None,
        *,
        decks: int = 6,
        dealer_line: float = 0.5,
        hidden_frames: int = 15,
        clear_frames: int = 15,
        refound_distance: float = 3.0,
        dealer_depth: float = 1.8,
    ) -> None:
        if not 0.0 < dealer_line < 1.0:
            raise ValueError(f"dealer_line must be in (0, 1), got {dealer_line}")
        self.rules = rules or Rules(decks=decks)
        self.shoe = ShoeTracker(decks=decks)
        self.dealer_line = dealer_line
        self.hidden_frames = hidden_frames
        self.clear_frames = clear_frames
        self.refound_distance = refound_distance
        self.dealer_depth = dealer_depth
        self.mat_edge: np.ndarray | None = None
        """Last far edge of the mat given to :meth:`update` (for the HUD)."""
        self.round = Round(rules=self.rules, shoe=self.shoe)
        self.rounds_played = 0
        self.last_results: list[str] = []
        self.hole_card_visible = False
        self._zones: dict[int, Zone] = {}
        self._dealt: list[_Dealt] = []
        self._unknown_frames: dict[int, int] = {}
        self._empty_frames = 0
        self._last_advice: Action | None = None

    # Reading the state --------------------------------------------------------------------------

    @property
    def advice(self) -> Action | None:
        return self.round.recommendation

    def dealt_zone(self, track_id: int) -> Zone | None:
        """Zone a recognised card was dealt to (``None`` if it was not dealt by this table)."""
        return self._zones.get(track_id)

    def boundary(self, image_width: int, image_height: int, card_height: float) -> np.ndarray:
        """Row of the line between the zones in every column, as :meth:`zone_of` draws it.

        ``mat_edge + dealer_depth * card_height`` when the mat's far edge is known, otherwise the
        fixed ``dealer_line``.
        """
        edge = self.mat_edge
        if edge is not None and len(edge) == image_width and (edge >= 0).all():
            return np.asarray(edge, np.float64) + self.dealer_depth * card_height
        return np.full(image_width, self.dealer_line * image_height)

    def zone_of(
        self, card: SeenCard, image_height: int, mat_edge: np.ndarray | None = None
    ) -> Zone:
        """Zone of a card: by its distance to the mat's far edge, or by its height in the image.

        ``mat_edge`` is the row of the mat's far edge in every column (``-1`` where there is no
        mat), see :func:`~blackjack_vision.vision_classic.mat.mat_far_edge`.
        """
        if card.corners is None:
            return Zone.PLAYER
        corners = np.asarray(card.corners, np.float64)
        centre_x, centre_y = corners.mean(axis=0)
        if mat_edge is not None and len(mat_edge):
            column = int(np.clip(round(centre_x), 0, len(mat_edge) - 1))
            edge = float(mat_edge[column])
            if edge >= 0:
                card_height = float(corners[:, 1].max() - corners[:, 1].min())
                depth = (centre_y - edge) / max(card_height, 1.0)
                return Zone.DEALER if depth < self.dealer_depth else Zone.PLAYER
        return Zone.DEALER if centre_y < self.dealer_line * image_height else Zone.PLAYER

    # Feeding ------------------------------------------------------------------------------------

    def update(
        self,
        visible: Sequence[SeenCard],
        new_cards: Sequence[SeenCard],
        image_height: int,
        *,
        mat_in_view: bool | None = None,
        motion: np.ndarray | None = None,
        mat_edge: np.ndarray | None = None,
        image_width: int | None = None,
    ) -> TableUpdate:
        """Feed the cards of one frame (``visible``) and those recognised in it (``new_cards``).

        ``mat_in_view`` tells an empty table from a camera looking away (see
        :func:`~blackjack_vision.vision_classic.mat.mat_in_view`). ``None`` means unknown, and then
        an empty view only starts a new round once the current one is over.

        ``motion`` is the camera motion since the previous frame (``2 x 3``, see
        :class:`~blackjack_vision.vision_classic.camera_motion.CameraMotion`). It moves the last
        position of the cards out of view, so a card found again after a head turn is still
        recognised as the same card. ``mat_edge`` places the cards in their zone (see
        :meth:`zone_of`). With ``image_width`` (and the camera motion) the cards only count as
        collected when the place where they were is in view: looking at the laptop with a corner
        of the mat in the image is not an empty table.
        """
        update = TableUpdate()
        if mat_edge is not None:
            self.mat_edge = mat_edge if (np.asarray(mat_edge) >= 0).any() else None

        if not visible:
            self._empty_frames += 1
            # Checked on every empty frame from clear_frames on (not only the first): the view may
            # come back to an empty mat only after looking away. A new round has no cards, so it
            # does not repeat.
            if self._empty_frames >= self.clear_frames and self._round_has_cards():
                if self.round.phase is Phase.FINISHED:
                    self._new_round(update, "table cleared after the round")
                elif mat_in_view and self._dealt_in_view(image_width, image_height):
                    self._new_round(update, "cards collected")
        else:
            self._empty_frames = 0

        in_view = {card.track_id for card in visible}
        positions = {card.track_id: _centre(card) for card in visible if card.corners is not None}
        for dealt in self._dealt:
            if dealt.track_id in positions:
                dealt.centre = positions[dealt.track_id]
            elif motion is not None:
                dealt.centre = motion[:, :2] @ dealt.centre + motion[:, 2]

        for card in new_cards:
            if card.card is None:
                continue
            zone = self.zone_of(card, image_height, mat_edge)
            self._zones[card.track_id] = zone
            if self.round.phase is Phase.FINISHED:
                self._new_round(update, "new cards after the round ended")
            lost = self._refound(card, zone, in_view, image_width, image_height)
            if lost is not None:
                lost.track_id = card.track_id  # the dealt card now follows the new track
                lost.centre = _centre(card)
                update.events.append(
                    f"{card.card} seen again in the {zone.value} zone, not recounted"
                )
                continue
            if self._deal(card.card, zone, update):
                self._dealt.append(_Dealt(card.card.code, zone, card.track_id, _centre(card)))

        self._update_hole_card(visible, image_height, mat_edge)
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

    def _dealt_in_view(self, image_width: int | None, image_height: int) -> bool:
        """Whether the last positions of the dealt cards are inside the image."""
        if image_width is None:
            return True
        return all(self._inside(dealt.centre, image_width, image_height) for dealt in self._dealt)

    @staticmethod
    def _inside(point: np.ndarray, width: int | None, height: int | None) -> bool:
        """Whether a point is inside the image, 5 % away from its border (``True`` if unknown)."""
        if width is None or height is None:
            return True
        x, y = point
        margin = 0.05
        return bool(
            margin * width <= x <= (1 - margin) * width
            and margin * height <= y <= (1 - margin) * height
        )

    def _round_has_cards(self) -> bool:
        return bool(self.round.dealer.cards) or any(len(hand) for hand in self.round.hands)

    def _new_round(self, update: TableUpdate, reason: str) -> None:
        self.rounds_played += 1
        self.reset()
        update.events.append(f"New round ({reason})")

    def _refound(
        self,
        card: SeenCard,
        zone: Zone,
        in_view: set[int],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> _Dealt | None:
        """The dealt card this one is, if an equal card of the zone lost its track nearby.

        "Nearby" only applies when the lost card's last position is inside the image (we were
        looking there, so a card elsewhere is another one). When it fell outside, the head was
        turned away and its position is only a guess, so the code and the zone decide.
        """
        centre = _centre(card)
        size = _size(card)
        best: _Dealt | None = None
        best_distance = float("inf")
        for dealt in self._dealt:
            if dealt.code != card.card.code or dealt.zone is not zone or dealt.track_id in in_view:
                continue
            distance = float(np.hypot(*(centre - dealt.centre)))
            near = distance <= self.refound_distance * size
            if not near and not self._inside(dealt.centre, image_width, image_height):
                near = True  # lost while looking away
            if near and distance < best_distance:
                best, best_distance = dealt, distance
        return best

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

    def _update_hole_card(
        self, visible: Sequence[SeenCard], image_height: int, mat_edge: np.ndarray | None
    ) -> None:
        seen = set()
        for card in visible:
            seen.add(card.track_id)
            if card.card is None and self.zone_of(card, image_height, mat_edge) is Zone.DEALER:
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


@dataclass
class _Dealt:
    """A card dealt in this round and where its track was last seen."""

    code: str
    zone: Zone
    track_id: int
    centre: np.ndarray


def _centre(card: SeenCard) -> np.ndarray:
    return np.asarray(card.corners, dtype=float).mean(axis=0)


def _size(card: SeenCard) -> float:
    """Diagonal of the card's bounding box, in pixels."""
    corners = np.asarray(card.corners, dtype=float)
    return float(np.hypot(*(corners.max(axis=0) - corners.min(axis=0))))
