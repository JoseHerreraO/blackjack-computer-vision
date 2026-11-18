import pytest

from blackjack_vision.blackjack.cards import Card, Rank, Suit
from blackjack_vision.blackjack.hand import Hand
from blackjack_vision.blackjack.strategy import (
    DEALER_COLUMNS,
    HARD,
    PAIRS,
    SOFT,
    Action,
    Rules,
    basic_strategy,
    dealer_value,
    take_insurance,
)

H, S, D, P, R = Action.HIT, Action.STAND, Action.DOUBLE, Action.SPLIT, Action.SURRENDER
S17_DAS = Rules()
H17 = Rules(dealer_hits_soft_17=True)
NO_DAS = Rules(double_after_split=False)
SURRENDER = Rules(surrender=True)


def play(codes: str, dealer: str, rules: Rules = S17_DAS, **kwargs) -> Action:
    return basic_strategy(Hand.of(*codes.split()), Rank.parse(dealer), rules, **kwargs)


# --------------------------------------------------------------------------- hands


@pytest.mark.parametrize(
    ("codes", "total", "soft"),
    [
        ("AS 6D", 17, True),
        ("AS 6D TC", 17, False),
        ("AS AD", 12, True),
        ("AS AD 9C", 21, True),
        ("TS 6D", 16, False),
        ("KS QD 5C", 25, False),
        ("AS AD AC AH 7S", 21, True),
    ],
)
def test_hand_totals(codes, total, soft):
    hand = Hand.of(*codes.split())
    assert (hand.total, hand.is_soft) == (total, soft)


def test_blackjack_bust_and_pairs():
    assert Hand.of("AS", "KD").is_blackjack
    assert not Hand([Card.parse("AS"), Card.parse("KD")], from_split=True).is_blackjack
    assert not Hand.of("7S", "7D", "7C").is_blackjack
    assert Hand.of("KS", "QD", "2C").is_bust
    assert Hand.of("KS", "QD").is_pair and Hand.of("KS", "QD").pair_value == 10
    assert Hand.of("AS", "AD").pair_value == 1
    assert not Hand.of("9S", "TD").is_pair
    assert Hand.of("AS", "6D").describe() == "A6 (soft 17)"


# --------------------------------------------------------------------------- tables


def test_tables_are_complete_and_use_known_codes():
    allowed = {"H", "S", "D", "Ds", "P", "Ph", "Rh", "Rs", "Rp", "-"}
    assert set(HARD) == set(range(8, 18))
    assert set(SOFT) == set(range(12, 21))
    assert set(PAIRS) == set(range(1, 11))
    for table in (HARD, SOFT, PAIRS):
        for row in table.values():
            assert len(row) == len(DEALER_COLUMNS)
            assert set(row) <= allowed


@pytest.mark.parametrize(
    ("codes", "dealer", "expected"),
    [
        # Hard totals
        ("5S 3D", "6", H),
        ("5S 4D", "2", H),
        ("5S 4D", "3", D),
        ("6S 4D", "9", D),
        ("6S 4D", "T", H),
        ("6S 5D", "T", D),
        ("6S 5D", "A", H),  # S17
        ("TS 2D", "3", H),
        ("TS 2D", "4", S),
        ("TS 3D", "2", S),
        ("TS 6D", "6", S),
        ("TS 6D", "7", H),
        ("TS 6D", "T", H),  # no surrender
        ("TS 7D", "A", S),
        # Soft totals
        ("AS 2D", "5", D),
        ("AS 2D", "4", H),
        ("AS 4D", "4", D),
        ("AS 6D", "2", H),
        ("AS 6D", "3", D),
        ("AS 7D", "2", S),  # S17
        ("AS 7D", "6", D),
        ("AS 7D", "8", S),
        ("AS 7D", "9", H),
        ("AS 8D", "6", S),  # S17
        # Pairs
        ("AS AD", "A", P),
        ("8S 8D", "T", P),
        ("TS KD", "6", S),
        ("9S 9D", "7", S),
        ("9S 9D", "8", P),
        ("9S 9D", "A", S),
        ("7S 7D", "7", P),
        ("7S 7D", "8", H),
        ("6S 6D", "2", P),  # DAS
        ("5S 5D", "9", D),
        ("4S 4D", "5", P),  # DAS
        ("4S 4D", "4", H),
        ("2S 2D", "3", P),  # DAS
        ("3S 3D", "8", H),
    ],
)
def test_basic_strategy_s17_das(codes, dealer, expected):
    assert play(codes, dealer) == expected


@pytest.mark.parametrize(
    ("codes", "dealer", "s17", "h17"),
    [
        ("6S 5D", "A", H, D),
        ("AS 7D", "2", S, D),
        ("AS 8D", "6", S, D),
    ],
)
def test_dealer_hits_soft_17_changes(codes, dealer, s17, h17):
    assert play(codes, dealer, S17_DAS) == s17
    assert play(codes, dealer, H17) == h17


@pytest.mark.parametrize(
    ("codes", "dealer", "expected"),
    [("6S 6D", "2", H), ("4S 4D", "5", H), ("2S 2D", "3", H), ("2S 2D", "4", P)],
)
def test_without_double_after_split(codes, dealer, expected):
    assert play(codes, dealer, NO_DAS) == expected


@pytest.mark.parametrize(
    ("codes", "dealer", "rules", "expected"),
    [
        ("TS 6D", "T", SURRENDER, R),
        ("TS 6D", "9", SURRENDER, R),
        ("TS 5D", "T", SURRENDER, R),
        ("TS 5D", "9", SURRENDER, H),
        ("8S 8D", "T", SURRENDER, P),  # S17: split, never surrender 8-8
        ("TS 7D", "A", Rules(surrender=True, dealer_hits_soft_17=True), R),
        ("8S 8D", "A", Rules(surrender=True, dealer_hits_soft_17=True), R),
        ("8S 8D", "A", H17, P),
    ],
)
def test_late_surrender(codes, dealer, rules, expected):
    assert play(codes, dealer, rules) == expected


def test_double_and_surrender_only_on_the_first_two_cards():
    assert play("2S 3D 6C", "5") == H  # hard 11 with three cards: hit, not double
    assert play("AS 2D 4C", "4") == H  # soft 17 vs 4: D -> hit when doubling is not allowed
    assert play("AS 3D 4C", "4") == S  # soft 18 vs 4: Ds -> stand when doubling is not allowed
    assert play("TS 3D 3C", "T", SURRENDER) == H  # 16 with three cards: no surrender


def test_split_hands_and_split_limits():
    split_hand = Hand([Card(Rank.SIX, Suit.SPADES), Card(Rank.FIVE, Suit.CLUBS)], from_split=True)
    assert basic_strategy(split_hand, Rank.SIX, S17_DAS) == D
    assert basic_strategy(split_hand, Rank.SIX, NO_DAS) == H
    assert play("AS AD", "6", can_split=False) == H
    assert play("8S 8D", "T", can_split=False) == H


def test_dealer_values_and_errors():
    assert dealer_value(Card.parse("AH")) == 11
    assert dealer_value(1) == 11
    assert dealer_value(Rank.KING) == 10
    with pytest.raises(ValueError):
        dealer_value(12)
    with pytest.raises(ValueError):
        basic_strategy(Hand.of("TS"), Rank.TWO)
    with pytest.raises(ValueError):
        basic_strategy(Hand.of("TS", "QD", "5C"), Rank.TWO)
    with pytest.raises(ValueError):
        Rules(decks=0)


def test_insurance_needs_a_high_count():
    assert not take_insurance(2.9)
    assert take_insurance(3.0)


def test_spanish_labels():
    assert {action.spanish for action in Action} == {
        "Pide carta",
        "Plántate",
        "Dobla",
        "Separa",
        "Ríndete",
    }
