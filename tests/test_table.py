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
        update = table.update([], [], HEIGHT, mat_in_view=True)  # cards collected
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


# Review of #25 -------------------------------------------------------------------------------


def seen_at(track_id: int, code: str | None, y: float, x: float) -> Seen:
    corners = np.array(
        [[x - 30, y - 40], [x + 30, y - 40], [x + 30, y + 40], [x - 30, y + 40]], float
    )
    return Seen(corners, Card.parse(code) if code else None, track_id)


def test_second_equal_card_after_a_refound_one_is_dealt():
    table = Table()
    up = seen_at(1, "9H", 150, 300)
    p1, p2 = seen_at(2, "8S", 550, 250), seen_at(3, "3D", 550, 350)
    table.update([up, p1, p2], [up, p1, p2], 720)  # player 8S 3D = 11
    table.update([up, p2], [], 720)  # the 8S is covered: track 2 lost
    p1b = seen_at(4, "8S", 550, 250)
    update = table.update([up, p1b, p2], [p1b], 720)  # found again: not recounted
    assert any("not recounted" in event for event in update.events)
    hit = seen_at(5, "8S", 550, 450)
    table.update([up, p1b, p2, hit], [hit], 720)  # a different 8S
    hand = table.round.hands[0]
    assert [card.code for card in hand.cards] == ["8S", "3D", "8S"] and hand.total == 19


def test_lost_card_reappearing_far_away_is_a_new_card():
    table = Table()
    up = seen_at(1, "9H", 150, 300)
    p1, p2 = seen_at(2, "8S", 550, 200), seen_at(3, "3D", 550, 300)
    table.update([up, p1, p2], [up, p1, p2], 720)
    table.update([up, p2], [], 720)  # 8S lost
    far = seen_at(4, "8S", 650, 650)  # an 8S, but far from where the lost one was
    table.update([up, p2, far], [far], 720)
    assert len(table.round.hands[0]) == 3


def test_looking_away_mid_round_keeps_the_hand():
    table = Table(clear_frames=15)
    up, a, b = seen_at(1, "TC", 150, 300), seen_at(2, "TS", 550, 250), seen_at(3, "6D", 550, 350)
    table.update([up, a, b], [up, a, b], 720)  # player 16 against a ten
    for _ in range(60):
        table.update([], [], 720, mat_in_view=False)  # head turned away
    assert table.round.phase is Phase.PLAYER_TURN and table.round.hands[0].total == 16
    assert table.rounds_played == 0


def test_empty_mat_mid_round_starts_a_new_round():
    table = Table(clear_frames=15)
    up, a, b = seen_at(1, "TC", 150, 300), seen_at(2, "TS", 550, 250), seen_at(3, "6D", 550, 350)
    table.update([up, a, b], [up, a, b], 720)
    for _ in range(15):
        update = table.update([], [], 720, mat_in_view=True)  # cards collected, mat empty
    assert any("New round" in event for event in update.events)
    assert table.rounds_played == 1


def test_empty_table_after_the_round_ended_starts_a_new_round_even_without_the_mat():
    table = Table(clear_frames=15)
    deal(table, player("TH"), dealer("5C"), player("QD"))
    deal(table, dealer("KS"))
    deal(table, dealer("TS"))  # dealer busts, round finished
    for _ in range(15):
        table.update([], [], HEIGHT, mat_in_view=None)
    assert table.rounds_played == 1


def test_mat_seen_empty_after_looking_away_starts_a_new_round():
    # Review of #25: the cards are collected while the player looks away, then the player looks
    # back at an empty mat. The old round must close before the next round's cards arrive.
    table = Table(clear_frames=15)
    up, a, b = seen_at(1, "TC", 150, 300), seen_at(2, "TS", 550, 250), seen_at(3, "6D", 550, 350)
    table.update([up, a, b], [up, a, b], 720)  # dealer TC, player TS 6D: 16
    for _ in range(20):
        table.update([], [], 720, mat_in_view=False)  # looking away
    for _ in range(20):
        table.update([], [], 720, mat_in_view=True)  # back: empty mat
    assert table.rounds_played == 1 and not table.round.hands[0].cards
    n1, n2, n3 = (
        seen_at(10, "5C", 550, 250),
        seen_at(11, "KD", 150, 300),
        seen_at(12, "7H", 550, 350),
    )
    table.update([n1, n2, n3], [n1, n2, n3], 720)
    assert [card.code for card in table.round.hands[0].cards] == ["5C", "7H"]
    assert [card.code for card in table.round.dealer.cards] == ["KD"]


def test_new_round_from_an_empty_view_happens_once():
    table = Table(clear_frames=3)
    deal(table, player("TH"), dealer("5C"), player("QD"))
    for _ in range(10):
        table.update([], [], HEIGHT, mat_in_view=True)
    assert table.rounds_played == 1
