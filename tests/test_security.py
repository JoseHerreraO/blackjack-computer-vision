import numpy as np
import pytest
from synthetic_faces import card_face, photographed

from blackjack_vision.blackjack.cards import FULL_DECK, Card, Rank, Suit
from blackjack_vision.security.aces import (
    AceReader,
    DecoderEvent,
    SecuritySystem,
    SequenceDecoder,
    parse_secret,
)
from blackjack_vision.vision_classic.recognition import index_corners
from blackjack_vision.vision_classic.suit_templates import SuitTemplates

SECRET = (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)


@pytest.fixture(scope="module")
def reader() -> AceReader:
    rng = np.random.default_rng(0)
    cards = [card for card in FULL_DECK if card.rank in (Rank.THREE, Rank.EIGHT, Rank.JACK)]
    images = [photographed(card_face(card), rng) for card in cards for _ in range(2)]
    labels = [card.suit for card in cards for _ in range(2)]
    return AceReader(SuitTemplates().fit([index_corners(i) for i in images], labels))


# --------------------------------------------------------------------------- ace reader


def test_aces_and_their_suits_are_read(reader):
    rng = np.random.default_rng(1)
    for suit in Suit:
        for _ in range(5):  # half of them come out upside down
            reading = reader.read(photographed(card_face(Card(Rank.ACE, suit)), rng))
            assert reading.is_ace and reading.suit is suit and reading.margin > 0


@pytest.mark.parametrize("rank", [r for r in Rank if r is not Rank.ACE])
def test_other_ranks_are_not_aces(reader, rank):
    rng = np.random.default_rng(2)
    for suit in (Suit.HEARTS, Suit.CLUBS):
        reading = reader.read(photographed(card_face(Card(rank, suit)), rng))
        assert not reading.is_ace and reading.suit is None


def test_rank_from_the_recogniser_overrides_the_pip_test(reader):
    seven = card_face(Card(Rank.SEVEN, Suit.DIAMONDS))
    reading = reader.read(seven, rank=Rank.ACE)
    assert reading.is_ace and reading.suit is Suit.DIAMONDS  # suit from the index corners
    ace = card_face(Card(Rank.ACE, Suit.SPADES))
    assert not reader.read(ace, rank=Rank.KING).is_ace


def test_reader_needs_trained_templates():
    with pytest.raises(ValueError):
        AceReader(SuitTemplates())


# --------------------------------------------------------------------------- decoder


def feed(decoder: SequenceDecoder, suits, start=0.0, step=0.1):
    """Feed one reading per frame; returns the non-trivial events."""
    events = []
    for i, suit in enumerate(suits):
        event = decoder.update(suit, start + i * step)
        if event is not DecoderEvent.NONE:
            events.append(event)
    return events


def show(suit: Suit, frames: int = 6, gap: int = 4):
    return [suit] * frames + [None] * gap


def test_correct_sequence_unlocks():
    decoder = SequenceDecoder(SECRET)
    frames = [frame for suit in SECRET for frame in show(suit)]
    events = feed(decoder, frames)
    assert events == [DecoderEvent.ACCEPTED] * 3 + [DecoderEvent.UNLOCKED]
    assert decoder.unlocked and decoder.expected is None
    assert decoder.update(Suit.SPADES, 99.0) is DecoderEvent.NONE  # stays unlocked


def test_wrong_ace_resets_and_can_restart_the_sequence():
    decoder = SequenceDecoder(SECRET)
    feed(decoder, show(Suit.SPADES) + show(Suit.HEARTS))
    assert decoder.progress == 2
    assert feed(decoder, show(Suit.CLUBS), start=5.0) == [DecoderEvent.WRONG]
    assert decoder.progress == 0
    feed(decoder, show(Suit.HEARTS), start=6.0)
    assert decoder.progress == 0
    feed(decoder, show(Suit.SPADES), start=7.0)  # a wrong ace equal to the first one restarts
    assert decoder.progress == 1


def test_short_glimpses_do_not_count():
    decoder = SequenceDecoder(SECRET, min_frames=5)
    feed(decoder, [Suit.SPADES] * 4 + [None] + [Suit.HEARTS, Suit.SPADES] * 3)
    assert decoder.progress == 0


def test_holding_one_ace_counts_once():
    decoder = SequenceDecoder((Suit.SPADES, Suit.SPADES), gap_frames=3)
    feed(decoder, [Suit.SPADES] * 30)
    assert decoder.progress == 1  # still the same ace
    feed(decoder, [None] * 3 + [Suit.SPADES] * 5, start=3.0)
    assert decoder.unlocked  # shown again after leaving the view


def test_timeout_resets():
    decoder = SequenceDecoder(SECRET, timeout=2.0)
    feed(decoder, show(Suit.SPADES))
    assert decoder.progress == 1
    assert decoder.update(None, 10.0) is DecoderEvent.TIMEOUT
    assert decoder.progress == 0


def test_manual_reset_locks_again():
    decoder = SequenceDecoder((Suit.HEARTS,))
    feed(decoder, show(Suit.HEARTS))
    assert decoder.unlocked
    assert decoder.reset() is DecoderEvent.RESET
    assert not decoder.unlocked and decoder.progress == 0


def test_secret_parsing_and_validation():
    assert parse_secret("SHDC") == SECRET
    assert parse_secret(" s h ") == (Suit.SPADES, Suit.HEARTS)
    with pytest.raises(ValueError):
        parse_secret("")
    with pytest.raises(ValueError):
        parse_secret("SX")
    with pytest.raises(ValueError):
        SequenceDecoder(())
    with pytest.raises(ValueError):
        SequenceDecoder(SECRET, min_frames=0)


# --------------------------------------------------------------------------- whole system


def test_security_system_unlocks_from_card_images(reader):
    rng = np.random.default_rng(3)
    system = SecuritySystem(reader, SequenceDecoder(SECRET, min_frames=3, gap_frames=2))
    other = photographed(card_face(Card(Rank.NINE, Suit.CLUBS)), rng)
    t = 0.0
    for suit in SECRET:
        ace = photographed(card_face(Card(Rank.ACE, suit)), rng)
        for _ in range(3):
            system.update([ace, other], t)  # a non-ace card in view does not matter
            t += 0.1
        for _ in range(2):
            system.update([other], t)
            t += 0.1
    assert system.unlocked


def test_two_aces_in_view_are_ignored(reader):
    system = SecuritySystem(reader, SequenceDecoder(SECRET, min_frames=2))
    spade = card_face(Card(Rank.ACE, Suit.SPADES))
    heart = card_face(Card(Rank.ACE, Suit.HEARTS))
    for i in range(5):
        event, readings = system.update([spade, heart], i * 0.1)
        assert event is DecoderEvent.NONE and len(readings) == 2
    assert system.decoder.progress == 0


# --------------------------------------------------------------------------- real-deck regressions


def _real_layout(rank: Rank, suit: Suit) -> np.ndarray:
    """A card laid out like our real deck: the 3 and the 5 have one pip alone in the centre and
    their other pips near the top and bottom edges, outside the old centre region."""
    from synthetic_faces import PAPER, draw_suit

    card = card_face(Card(rank, suit))
    card[100:200, 50:150] = PAPER
    card[30:100, 50:150] = PAPER
    card[200:270, 50:150] = PAPER
    height, width = card.shape[:2]
    layouts = {
        Rank.THREE: [(0.5, 0.17), (0.5, 0.5), (0.5, 0.83)],
        Rank.FIVE: [(0.32, 0.17), (0.68, 0.17), (0.5, 0.5), (0.32, 0.83), (0.68, 0.83)],
    }
    for fx, fy in layouts[rank]:
        draw_suit(card, suit, (fx * width, fy * height), 46)  # pips as big as in our deck
    return card


@pytest.mark.parametrize("rank", [Rank.THREE, Rank.FIVE])
def test_a_single_central_pip_is_not_enough_to_be_an_ace(reader, rank):
    for suit in (Suit.CLUBS, Suit.DIAMONDS):
        card = _real_layout(rank, suit)
        assert not reader.read(card).is_ace
        assert len(reader.body_pips(card)) == int(rank.value)


def test_pale_orange_red_ink_reads_as_red(reader):
    import cv2

    from blackjack_vision.vision_classic.recognition import RED_INK_THRESHOLD, red_ink_fraction

    # Red ink measured on the glasses video: orange-ish and pale (median hue 9, saturation 79
    # in OpenCV HSV), with a light background behind it.
    card = card_face(Card(Rank.ACE, Suit.HEARTS))
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    inked = hsv[..., 1] > 40
    hsv[inked, 0] = 9
    hsv[inked, 1] = np.minimum(hsv[inked, 1], 79)
    hsv[inked, 2] = np.maximum(hsv[inked, 2], 190)
    card = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert red_ink_fraction(card) >= RED_INK_THRESHOLD
    reading = reader.read(card)
    assert reading.is_ace and reading.suit is Suit.HEARTS
    assert red_ink_fraction(card_face(Card(Rank.ACE, Suit.SPADES))) < RED_INK_THRESHOLD
