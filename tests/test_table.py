from dataclasses import dataclass, field
from itertools import count

import numpy as np
import pytest

from blackjack_vision.blackjack.cards import Card
from blackjack_vision.blackjack.game import Phase
from blackjack_vision.blackjack.strategy import Action
from blackjack_vision.blackjack.table import Table, Zone
from blackjack_vision.ui.hud import draw_hud

HEIGHT = 1000
_ids = count(1)


@dataclass
class Seen:
    corners: np.ndarray
    card: Card | None = None
    track_id: int = field(default_factory=lambda: next(_ids))


def at(y: float, code: str | None = None, x: float = 300) -> Seen:
    corners = np.array(
        [[x - 30, y - 40], [x + 30, y - 40], [x + 30, y + 40], [x - 30, y + 40]], float
    )
    return Seen(corners, Card.parse(code) if code else None)


def dealer(code: str | None = None) -> Seen:
    return at(200, code)


def player(code: str | None = None) -> Seen:
    return at(800, code)


def deal(table: Table, *cards: Seen, visible: list[Seen] | None = None):
    """One frame in which ``cards`` are recognised (and all of ``visible`` are on the table)."""
    return table.update(visible if visible is not None else list(cards), list(cards), HEIGHT)


def test_zone_by_height():
    table = Table(dealer_line=0.5)
    assert table.zone_of(dealer(), HEIGHT) is Zone.DEALER
    assert table.zone_of(player(), HEIGHT) is Zone.PLAYER


def test_round_gives_advice_and_result():
    table = Table()
    on_table = [player("TH"), dealer("6C"), player("6D")]
    update = deal(table, *on_table)
    assert table.round.phase is Phase.PLAYER_TURN
    assert table.advice is Action.STAND  # hard 16 against a 6
    assert update.speech == [Action.STAND.spanish]

    # The dealer turns the hole card (9) and draws a ten: 25, bust.
    hole = dealer("9S")
    on_table.append(hole)
    deal(table, hole, visible=on_table)
    assert table.round.phase is Phase.DEALER_TURN
    draw = dealer("TC")
    on_table.append(draw)
    update = deal(table, draw, visible=on_table)
    assert table.round.phase is Phase.FINISHED
    assert update.speech == ["Ganas"]
    assert table.last_results == ["win"]


def test_advice_is_spoken_only_when_it_changes():
    table = Table()
    on_table = [player("5H"), dealer("TC"), player("6D")]  # 11 against a ten: double
    assert deal(table, *on_table).speech == [Action.DOUBLE.spanish]
    assert table.update(on_table, [], HEIGHT).speech == []  # same advice, silent


def test_count_carries_over_to_the_next_round():
    table = Table(clear_frames=3)
    deal(table, player("TH"), dealer("KC"), player("QD"))  # three high cards: RC -3
    assert table.shoe.running_count == -3
    for _ in range(3):
        update = table.update([], [], HEIGHT)
    assert any("New round" in event for event in update.events)
    assert table.round.dealer.cards == [] and table.rounds_played == 1
    assert table.shoe.running_count == -3


def test_new_card_after_the_round_ended_starts_a_new_round():
    table = Table()
    deal(table, player("TH"), dealer("5C"), player("QD"))  # 20 against 5
    deal(table, dealer("KS"))  # 15
    deal(table, dealer("TS"))  # 25: bust, finished
    assert table.round.phase is Phase.FINISHED
    update = deal(table, player("2H"))
    assert any("New round" in event for event in update.events)
    assert [card.code for card in table.round.hands[0].cards] == ["2H"]


def test_hole_card_is_recognised_after_a_while():
    table = Table(hidden_frames=5)
    up, hole = dealer("7C"), dealer(None)
    hole.corners = hole.corners + [100, 0]
    deal(table, up, visible=[up, hole])
    for _ in range(3):
        table.update([up, hole], [], HEIGHT)
    assert not table.hole_card_visible
    for _ in range(3):
        table.update([up, hole], [], HEIGHT)
    assert table.hole_card_visible


def test_card_that_does_not_fit_the_round_is_ignored():
    table = Table()
    deal(table, player("TH"), dealer("6C"), player("7D"))
    deal(table, dealer("TC"))  # dealer turn: the player is done
    update = deal(table, player("2S"))
    assert any(event.startswith("Ignored 2S") for event in update.events)


def test_reset_with_reshuffle_clears_the_count():
    table = Table()
    deal(table, player("TH"), dealer("KC"), player("QD"))
    table.reset(reshuffle=True)
    assert table.shoe.running_count == 0 and table.round.dealer.cards == []


def test_invalid_dealer_line():
    with pytest.raises(ValueError):
        Table(dealer_line=1.5)


@pytest.mark.parametrize("width", [360, 720])
def test_hud_draws_without_changing_the_frame(width):
    image = np.full((width * 16 // 9, width, 3), 60, np.uint8)
    table = Table()
    cards = [player("TH"), dealer("6C"), player("6D")]
    deal(table, *cards)
    scale = image.shape[0] / HEIGHT
    for card in cards:
        card.corners = card.corners * scale
    view = draw_hud(image, cards, 14.8, table=table, lock_text="UNLOCKED")
    assert view.shape == image.shape and (view != image).any()
    assert (image == 60).all()  # the input is untouched
    locked = draw_hud(image, cards, 14.8, lock_text="LOCKED  #...", locked=True)
    assert locked.shape == image.shape


def test_card_found_again_after_losing_its_track_is_not_recounted():
    table = Table()
    ten = player("TH")
    deal(table, ten, dealer("6C"))
    assert table.shoe.running_count == -1 + 1  # T is -1, 6 is +1
    again = player("TH")  # same card, new track (the old one was lost behind a hand)
    update = deal(table, again, visible=[again])
    assert any("not recounted" in event for event in update.events)
    assert [card.code for card in table.round.hands[0].cards] == ["TH"]
    assert table.shoe.running_count == 0


def test_two_equal_cards_in_view_are_two_cards():
    table = Table()
    first = player("TH")
    deal(table, first)
    second = player("TH")
    second.corners = second.corners + [150, 0]
    deal(table, second, visible=[first, second])  # both in view: several decks
    assert [card.code for card in table.round.hands[0].cards] == ["TH", "TH"]
