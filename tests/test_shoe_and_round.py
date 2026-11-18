import pytest

from blackjack_vision.blackjack.cards import FULL_DECK, Card
from blackjack_vision.blackjack.game import Outcome, Phase, Round
from blackjack_vision.blackjack.shoe import ShoeTracker
from blackjack_vision.blackjack.strategy import Action, Rules


def cards(*codes: str) -> list[Card]:
    return [Card.parse(code) for code in codes]


# --------------------------------------------------------------------------- shoe and count


def test_running_and_true_count():
    shoe = ShoeTracker(decks=6)
    for card in cards("2S", "3D", "4C", "5H", "6S", "7D", "8C", "9H", "TS", "AD"):
        shoe.see(card)
    assert shoe.running_count == 5 - 2
    assert shoe.cards_seen == 10 and shoe.cards_remaining == 302
    assert shoe.true_count == pytest.approx(3 / (302 / 52))
    assert shoe.advantage == pytest.approx(-0.005 + 0.005 * shoe.true_count)


def test_a_full_deck_balances_the_count():
    shoe = ShoeTracker(decks=1)
    for card in FULL_DECK:
        shoe.see(card)
    assert shoe.running_count == 0
    assert shoe.cards_remaining == 0 and shoe.penetration == 1.0
    assert shoe.true_count == 0.0  # the decks-left floor avoids a division by zero


def test_composition_and_probabilities():
    shoe = ShoeTracker(decks=2)
    assert shoe.remaining_by_value()[10] == 32 and shoe.remaining_by_value()[1] == 8
    assert shoe.probability(10) == pytest.approx(32 / 104)
    for card in cards("KS", "QD", "TC", "JH"):
        shoe.see(card)
    assert shoe.remaining_by_value()[10] == 28
    assert shoe.probability(10) == pytest.approx(28 / 100)


def test_undo_reshuffle_and_double_counting(caplog):
    shoe = ShoeTracker(decks=1)
    shoe.see(Card.parse("5S"))
    shoe.undo(Card.parse("5S"))
    assert shoe.running_count == 0 and shoe.cards_seen == 0 and shoe.history == []
    with pytest.raises(ValueError):
        shoe.undo(Card.parse("5S"))
    for _ in range(5):  # a one-deck shoe has only four fives
        shoe.see(Card.parse("5S"))
    assert "counted twice" in caplog.text
    shoe.reshuffle()
    assert shoe.cards_seen == 0 and shoe.running_count == 0


@pytest.mark.parametrize(("running", "units"), [(-10, 1), (0, 1), (12, 1), (18, 2), (60, 8)])
def test_bet_ramp(running, units):
    shoe = ShoeTracker(decks=6)
    shoe.running_count = running  # 6 decks left: TC = running / 6
    assert shoe.bet_units() == units


# --------------------------------------------------------------------------- rounds


def deal_start(round_: Round, player: tuple[str, str], dealer_up: str) -> None:
    first, second = cards(*player)
    round_.deal_to_player(first)
    round_.deal_to_dealer(Card.parse(dealer_up))
    round_.deal_to_player(second)


def test_simple_round_with_recommendations_and_count():
    shoe = ShoeTracker(decks=6)
    round_ = Round(shoe=shoe)
    deal_start(round_, ("TS", "2D"), "4C")
    assert round_.phase is Phase.PLAYER_TURN
    assert round_.recommendation is Action.STAND

    round_.stand()
    assert round_.phase is Phase.DEALER_TURN and round_.recommendation is None
    for code in ("TD", "9H"):  # dealer: 4 + 10 + 9 = 23
        round_.deal_to_dealer(Card.parse(code))
    assert round_.phase is Phase.FINISHED
    [result] = round_.results()
    assert result.outcome is Outcome.WIN and result.payout == 1.0
    assert shoe.running_count == 1 - 1 + 1 - 1 + 0  # T, 2, 4, T, 9


def test_hitting_until_bust_ends_the_round():
    round_ = Round()
    deal_start(round_, ("TS", "6D"), "TC")
    assert round_.recommendation is Action.HIT
    round_.deal_to_player(Card.parse("9H"))
    assert round_.phase is Phase.FINISHED  # the only hand is bust, the dealer needs not play
    assert round_.results()[0].outcome is Outcome.LOSE


def test_blackjack_pays_three_to_two_and_skips_the_player_turn():
    round_ = Round()
    deal_start(round_, ("AS", "KD"), "9C")
    assert round_.phase is Phase.DEALER_TURN
    round_.deal_to_dealer(Card.parse("8H"))  # dealer 17
    assert round_.results()[0].outcome is Outcome.BLACKJACK
    assert round_.net == 1.5


def test_blackjack_against_dealer_blackjack_is_a_push():
    round_ = Round()
    deal_start(round_, ("AS", "KD"), "AC")
    round_.deal_to_dealer(Card.parse("QH"))
    assert round_.results()[0].outcome is Outcome.PUSH


def test_dealer_blackjack_beats_twenty():
    round_ = Round()
    deal_start(round_, ("TS", "KD"), "AC")
    round_.stand()
    round_.deal_to_dealer(Card.parse("QH"))
    assert round_.results()[0].outcome is Outcome.LOSE


def test_double_down_takes_one_card_and_doubles_the_payout():
    round_ = Round()
    deal_start(round_, ("6S", "5D"), "6C")
    assert round_.recommendation is Action.DOUBLE
    round_.double()
    round_.deal_to_player(Card.parse("4H"))  # 15 after doubling: the hand ends anyway
    assert round_.phase is Phase.DEALER_TURN
    for code in ("TD", "7S"):  # dealer 6 + 10 + 7 = 23
        round_.deal_to_dealer(Card.parse(code))
    assert round_.results()[0].payout == 2.0


def test_split_plays_each_hand_in_turn():
    round_ = Round()
    deal_start(round_, ("8S", "8D"), "9C")
    assert round_.recommendation is Action.SPLIT
    round_.split()
    assert [len(hand) for hand in round_.hands] == [1, 1] and round_.active_index == 0
    round_.deal_to_player(Card.parse("3H"))  # first hand 11 vs 9: double
    assert round_.recommendation is Action.DOUBLE
    round_.double()
    round_.deal_to_player(Card.parse("TH"))  # 21: first hand done
    assert round_.active_index == 1
    round_.deal_to_player(Card.parse("TC"))  # second hand 18
    assert round_.recommendation is Action.STAND
    round_.stand()
    for code in ("KD",):  # dealer 9 + K = 19
        round_.deal_to_dealer(Card.parse(code))
    first, second = round_.results()
    assert (first.outcome, first.payout) == (Outcome.WIN, 2.0)
    assert (second.outcome, second.payout) == (Outcome.LOSE, -1.0)
    assert not first.hand.is_blackjack


def test_split_aces_get_one_card_each():
    round_ = Round()
    deal_start(round_, ("AS", "AD"), "6C")
    round_.split()
    round_.deal_to_player(Card.parse("5H"))
    assert round_.active_index == 1  # split ace hand ends after one card
    round_.deal_to_player(Card.parse("KC"))  # 21 but not blackjack
    assert round_.phase is Phase.DEALER_TURN
    for code in ("TD", "5S"):  # dealer 6 + 10 + 5 = 21
        round_.deal_to_dealer(Card.parse(code))
    outcomes = [result.outcome for result in round_.results()]
    assert outcomes == [Outcome.LOSE, Outcome.PUSH]


def test_dealer_card_during_player_turn_ends_it():
    round_ = Round()
    deal_start(round_, ("TS", "7D"), "9C")
    round_.deal_to_dealer(Card.parse("TH"))  # hole card turned over: the player stood
    assert round_.phase is Phase.FINISHED  # dealer 19 stands
    assert round_.results()[0].outcome is Outcome.LOSE


@pytest.mark.parametrize(("h17", "finished"), [(False, True), (True, False)])
def test_dealer_soft_17(h17, finished):
    round_ = Round(rules=Rules(dealer_hits_soft_17=h17))
    deal_start(round_, ("TS", "8D"), "AC")
    round_.stand()
    round_.deal_to_dealer(Card.parse("6H"))  # soft 17
    assert (round_.phase is Phase.FINISHED) == finished


def test_surrender_loses_half():
    round_ = Round(rules=Rules(surrender=True))
    deal_start(round_, ("TS", "6D"), "TC")
    assert round_.recommendation is Action.SURRENDER
    round_.surrender()
    assert round_.phase is Phase.FINISHED
    assert round_.results()[0].payout == -0.5


def test_invalid_events_are_rejected():
    round_ = Round()
    with pytest.raises(RuntimeError):
        round_.stand()
    deal_start(round_, ("TS", "6D"), "TC")
    with pytest.raises(RuntimeError):
        round_.split()
    with pytest.raises(RuntimeError):
        round_.results()
    round_.stand()
    with pytest.raises(RuntimeError):
        round_.deal_to_player(Card.parse("5H"))
