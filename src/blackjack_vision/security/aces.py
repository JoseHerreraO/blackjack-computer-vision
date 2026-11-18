"""Security system: unlock the assistant by showing four aces in a secret suit order.

Two parts, built only with techniques of the course:

* :class:`AceReader` looks at a rectified card. An ace has a single large pip in the middle, so
  after thresholding the centre of the card (Otsu, Tema 3), the largest connected component
  (Tema 5) holds almost all the ink. Its suit comes from template matching with normalised
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
    largest_blob_mask,
    normalise_symbol,
)

CENTRE_REGION = (0.15, 0.25, 0.85, 0.75)
"""``(x0, y0, x1, y1)`` of the centre of a rectified card where an ace has its pip."""
MIN_PIP_SHARE = 0.6
"""Share of the ink of the centre that the largest blob must hold for the card to be an ace."""
MIN_PIP_AREA = 0.03
"""Smallest pip, as a fraction of the centre region."""


@dataclass(frozen=True)
class AceReading:
    """What :class:`AceReader` saw on one card.

    Attributes:
        is_ace: The card is an ace.
        suit: Its suit, ``None`` when it is not an ace or the suit could not be read.
        margin: NCC score gap between the best suit and the runner-up (higher = surer).
        red_ink: Fraction of red ink in the index corners.
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

    def central_pip(self, card: np.ndarray) -> np.ndarray | None:
        """Mask of the single large pip in the centre of the card, or ``None`` if there is none."""
        height, width = card.shape[:2]
        x0, y0, x1, y1 = CENTRE_REGION
        centre = card[int(y0 * height) : int(y1 * height), int(x0 * width) : int(x1 * width)]
        mask, share = largest_blob_mask(centre)
        if mask is None or share < MIN_PIP_SHARE:
            return None
        if np.count_nonzero(mask) < MIN_PIP_AREA * mask.size:
            return None
        rows, cols = np.nonzero(mask)
        if rows.min() == 0 or cols.min() == 0:  # cut by the region: artwork, not a pip
            return None
        if rows.max() == mask.shape[0] - 1 or cols.max() == mask.shape[1] - 1:
            return None
        return mask

    def read(self, card: np.ndarray, rank: Rank | None = None) -> AceReading:
        """Read one rectified card (BGR).

        Args:
            card: Rectified card, e.g. the output of ``rectify_card``.
            rank: Rank given by the recogniser, if known. Without it the card counts as an ace
                when its centre holds a single large pip.
        """
        red_ink = red_ink_fraction(card, self.index_region)
        pip = self.central_pip(card)
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
