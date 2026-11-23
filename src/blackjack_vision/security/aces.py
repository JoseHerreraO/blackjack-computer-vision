"""Security system: unlock the assistant by showing four aces in a secret suit order.

Two parts, built only with techniques of the course:

* :class:`AceReader` looks at a rectified card. The ink of the card body (everything but the
  margin and the four index corners) is thresholded with Otsu (Tema 3) and split into connected
  components (Tema 5); the sizeable ones are the pips. An ace has exactly one, a 3 has three, a
  face card has a large picture. Counting over the whole body matters: in a real deck the centre
  of a 3 or a 5 holds a single pip too. Its suit comes from template matching with normalised
  cross-correlation (Tema 3) of that pip and of the symbols of the index corners against the suit
  templates, restricted to the colour of the ink (HSV, Tema 2). The pip may be upside down, so
  both orientations of each template are tried for it. When the recogniser already knows the
  rank, it can be passed in instead of the pip test.
* :class:`SequenceDecoder` turns the per-frame readings into ace *events*: a suit seen for
  ``min_frames`` consecutive frames, after the previous ace left the view (or changed). Each event
  is checked against the secret sequence: a correct ace advances, a wrong one resets (and counts
  as a first ace when it matches the start of the secret), too long between aces resets, and
  :meth:`SequenceDecoder.reset` restarts on demand (e.g. a key press).

:class:`SecuritySystem` puts both together for the cards detected in a frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from blackjack_vision.blackjack.cards import Rank, Suit
from blackjack_vision.vision_classic.recognition import (
    INDEX_REGION,
    RED_INK_THRESHOLD,
    index_corners,
    red_ink_fraction,
)
from blackjack_vision.vision_classic.suit_templates import (
    SuitTemplates,
    corner_symbols,
    ink_mask,
    normalise_symbol,
)

BODY_MARGIN = 0.05
"""Border of the rectified card left out of the pip count (table pixels, card edge)."""
MIN_PIP_AREA = 0.004
"""Smallest blob that can be a pip, as a fraction of the card area."""
PIP_RELATIVE_AREA = 0.2
"""A blob counts as a pip when it is at least this fraction of the largest blob."""
MIN_ACE_PIP_AREA = 0.01
"""Smallest ace pip, as a fraction of the card area (an ace pip is larger than a 2's)."""
MAX_ACE_PIP_SIZE = 0.7
"""Largest ace pip, as a fraction of the body width and height (a face-card frame is larger)."""


@dataclass(frozen=True)
class AceReading:
    """What :class:`AceReader` saw on one card.

    Attributes:
        is_ace: The card is an ace.
        suit: Its suit, ``None`` when it is not an ace or the suit could not be read.
        margin: NCC score gap between the best suit and the runner-up (higher = surer).
        red_ink: Fraction of the ink of the card that is red.
    """

    is_ace: bool
    suit: Suit | None = None
    margin: float = 0.0
    red_ink: float = 0.0


class AceReader:
    """Tells whether a rectified card is an ace and reads its suit.

    Args:
        templates: Trained suit templates (e.g. ``CardRecognizer.suit_templates``).
        index_region: Size of the index corners, as fractions of the card.
    """

    def __init__(
        self, templates: SuitTemplates, index_region: tuple[float, float] = INDEX_REGION
    ) -> None:
        if not templates.templates:
            raise ValueError("The suit templates are not trained")
        self.templates = templates
        self.index_region = index_region

    def body_pips(self, card: np.ndarray) -> list[np.ndarray]:
        """Masks (full card size) of the pips of the card body, largest first.

        The body is the card without a ``BODY_MARGIN`` border and without the four index corners
        (some decks print the index in all four). Its ink is thresholded with Otsu and split into
        connected components; a component is a pip when its area is at least ``MIN_PIP_AREA`` of
        the card and ``PIP_RELATIVE_AREA`` of the largest one (so specks and decorations do not
        count).
        """
        height, width = card.shape[:2]
        mask = ink_mask(card, upscale=1)
        dy, dx = int(BODY_MARGIN * height), int(BODY_MARGIN * width)
        body = np.zeros_like(mask)
        body[dy : height - dy, dx : width - dx] = 255
        corner_w, corner_h = int(self.index_region[0] * width), int(self.index_region[1] * height)
        for y in (0, height - corner_h):
            for x in (0, width - corner_w):
                body[y : y + corner_h, x : x + corner_w] = 0
        mask = cv2.bitwise_and(mask, body)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return []
        areas = stats[1:, cv2.CC_STAT_AREA]
        threshold = max(MIN_PIP_AREA * height * width, PIP_RELATIVE_AREA * areas.max())
        order = np.argsort(areas)[::-1]
        return [
            np.where(labels == 1 + int(i), 255, 0).astype(np.uint8)
            for i in order
            if areas[i] >= threshold
        ]

    def ace_pip(self, card: np.ndarray) -> np.ndarray | None:
        """Mask of the pip if the card is an ace (exactly one pip in the body), else ``None``."""
        pips = self.body_pips(card)
        if len(pips) != 1:
            return None
        pip = pips[0]
        height, width = card.shape[:2]
        if np.count_nonzero(pip) < MIN_ACE_PIP_AREA * height * width:
            return None
        rows, cols = np.nonzero(pip)
        body_h, body_w = (1 - 2 * BODY_MARGIN) * height, (1 - 2 * BODY_MARGIN) * width
        if (
            np.ptp(rows) + 1 > MAX_ACE_PIP_SIZE * body_h
            or np.ptp(cols) + 1 > MAX_ACE_PIP_SIZE * body_w
        ):
            return None  # a face-card picture or frame, not a pip
        return pip

    def read(self, card: np.ndarray, rank: Rank | None = None) -> AceReading:
        """Read one rectified card (BGR).

        Args:
            card: Rectified card, e.g. the output of ``rectify_card``.
            rank: Rank given by the recogniser, if known (preferred). Without it the card counts
                as an ace when its body holds exactly one pip.
        """
        red_ink = red_ink_fraction(card)
        pip = self.ace_pip(card)
        is_ace = rank is Rank.ACE if rank is not None else pip is not None
        if not is_ace:
            return AceReading(False, red_ink=red_ink)

        known = self.templates.suits
        allowed = None
        if any(s.is_red for s in known) and any(not s.is_red for s in known):
            red = red_ink >= RED_INK_THRESHOLD
            allowed = [s for s in known if s.is_red == red]
        scores: dict[Suit, float] = dict.fromkeys(allowed or known, 0.0)
        if pip is not None:
            pip_scores = self.templates.scores(
                [normalise_symbol(pip)], allowed=allowed, both_orientations=True
            )
            for suit, value in (pip_scores or {}).items():
                scores[suit] += value
        index_scores = self.templates.scores(
            corner_symbols(index_corners(card, self.index_region)), allowed=allowed
        )
        for suit, value in (index_scores or {}).items():
            scores[suit] += value
        if pip is None and index_scores is None:
            return AceReading(True, None, 0.0, red_ink)
        ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        margin = ranking[0][1] - ranking[1][1] if len(ranking) > 1 else float("inf")
        return AceReading(True, ranking[0][0], float(margin), red_ink)


class DecoderEvent(str, Enum):
    """What happened in the last :meth:`SequenceDecoder.update`."""

    NONE = "none"
    ACCEPTED = "accepted"
    WRONG = "wrong"
    TIMEOUT = "timeout"
    UNLOCKED = "unlocked"
    RESET = "reset"


def parse_secret(text: str) -> tuple[Suit, ...]:
    """Secret from suit codes, e.g. ``"SHDC"`` (spades, hearts, diamonds, clubs)."""
    secret = tuple(Suit.parse(char) for char in text.strip() if not char.isspace())
    if not secret:
        raise ValueError("The secret needs at least one suit")
    return secret


@dataclass
class SequenceDecoder:
    """Checks the aces shown against a secret sequence of suits.

    Args:
        secret: Suits in the order they must be shown.
        min_frames: Consecutive frames with the same suit needed to accept an ace (filters
            single-frame misreadings).
        gap_frames: Frames without an ace needed before the same suit counts again.
        timeout: Seconds allowed between two aces before the sequence starts over.
    """

    secret: tuple[Suit, ...]
    min_frames: int = 5
    gap_frames: int = 3
    timeout: float = 10.0
    progress: int = 0
    unlocked: bool = False

    def __post_init__(self) -> None:
        self.secret = tuple(self.secret)
        if not self.secret:
            raise ValueError("The secret needs at least one suit")
        if self.min_frames < 1 or self.gap_frames < 1 or self.timeout <= 0:
            raise ValueError("min_frames and gap_frames must be >= 1 and timeout > 0")
        self._candidate: Suit | None = None
        self._streak = 0
        self._absent = 0
        self._last_event_suit: Suit | None = None
        self._last_event_time: float | None = None

    @property
    def expected(self) -> Suit | None:
        """Suit the next ace must have (``None`` once unlocked)."""
        return None if self.unlocked else self.secret[self.progress]

    def reset(self) -> DecoderEvent:
        """Start over (and lock again)."""
        self.progress = 0
        self.unlocked = False
        self._candidate = None
        self._streak = self._absent = 0
        self._last_event_suit = None
        self._last_event_time = None
        return DecoderEvent.RESET

    def update(self, suit: Suit | None, timestamp: float) -> DecoderEvent:
        """Feed the ace suit seen in one frame (``None`` = no ace) at ``timestamp`` seconds."""
        if self.unlocked:
            return DecoderEvent.NONE
        event = DecoderEvent.NONE
        if (
            self.progress > 0
            and self._last_event_time is not None
            and timestamp - self._last_event_time > self.timeout
        ):
            self.progress = 0
            self._last_event_time = None
            event = DecoderEvent.TIMEOUT

        if suit is None:
            self._candidate, self._streak = None, 0
            self._absent += 1
            if self._absent >= self.gap_frames:
                self._last_event_suit = None
            return event

        self._absent = 0
        if suit == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = suit, 1
        if self._streak != self.min_frames or suit == self._last_event_suit:
            return event

        self._last_event_suit = suit
        self._last_event_time = timestamp
        if suit == self.secret[self.progress]:
            self.progress += 1
            if self.progress == len(self.secret):
                self.unlocked = True
                return DecoderEvent.UNLOCKED
            return DecoderEvent.ACCEPTED
        self.progress = 1 if suit == self.secret[0] else 0
        return DecoderEvent.WRONG


class SecuritySystem:
    """Ace reader plus sequence decoder, fed with the cards of every frame.

    Args:
        reader: Reads aces from rectified cards.
        decoder: Checks the sequence.
    """

    def __init__(self, reader: AceReader, decoder: SequenceDecoder) -> None:
        self.reader = reader
        self.decoder = decoder

    @property
    def unlocked(self) -> bool:
        return self.decoder.unlocked

    def update(
        self,
        cards: Sequence[np.ndarray],
        timestamp: float,
        ranks: Sequence[Rank | None] | None = None,
    ) -> tuple[DecoderEvent, list[AceReading]]:
        """Read the rectified cards of one frame and update the decoder.

        Exactly one readable ace must be in view for the frame to count; with several aces the
        frame is treated as "no ace", since the order between them would be ambiguous.
        """
        ranks = list(ranks) if ranks is not None else [None] * len(cards)
        readings = [self.reader.read(card, rank) for card, rank in zip(cards, ranks, strict=True)]
        aces = [reading for reading in readings if reading.is_ace and reading.suit is not None]
        suit = aces[0].suit if len(aces) == 1 else None
        return self.decoder.update(suit, timestamp), readings
