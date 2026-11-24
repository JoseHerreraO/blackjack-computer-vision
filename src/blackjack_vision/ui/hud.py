"""HUD drawn over the live video: FPS, lock state, cards by zone, hands, advice and the count.

Layout (scaled to the frame width, so it reads the same at 360x640 and 720x1280):

* top-left: FPS (always on, the assignment requires it in the video) and LOCKED / UNLOCKED;
* a dashed line splitting the dealer's zone (top) from the player's (bottom);
* every card outlined in its zone colour with its code, ``?`` while unrecognised, or HOLE for
  the dealer's face-down card;
* bottom panel: dealer and player hands with totals, the advice in large letters, the Hi-Lo
  count and the suggested bet, and the result of the last round.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from blackjack_vision.blackjack.hand import Hand
from blackjack_vision.blackjack.table import SeenCard, Table, Zone

DEALER_BGR = (255, 0, 200)
PLAYER_BGR = (0, 210, 0)
PENDING_BGR = (0, 200, 255)
LOCKED_BGR = (0, 0, 255)
TEXT_BGR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _scale(image: np.ndarray) -> float:
    return image.shape[1] / 720.0


def _text(image, text, origin, scale, colour=TEXT_BGR, thickness=2) -> None:
    cv2.putText(image, text, origin, FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, scale, colour, thickness, cv2.LINE_AA)


def _hand_text(hand: Hand) -> str:
    if not hand.cards:
        return "-"
    cards = " ".join(card.code for card in hand.cards)
    soft = " soft" if hand.is_soft and not hand.is_bust else ""
    return f"{cards}  ({hand.total}{soft})"


def draw_hud(
    image: np.ndarray,
    cards: Sequence[SeenCard],
    fps: float,
    *,
    table: Table | None = None,
    lock_text: str | None = None,
    locked: bool = False,
    panel: bool = True,
) -> np.ndarray:
    """Return a copy of ``image`` with the HUD.

    Args:
        image: Frame (BGR).
        cards: Cards seen in this frame (with ``corners`` and ``card``).
        fps: Frame rate of the whole loop.
        table: The table state. ``None`` while locked (no hands, no advice).
        lock_text: Lock status line, e.g. ``"LOCKED  ##.."`` or ``"UNLOCKED"``.
        locked: Draw the lock status in red and hide everything else.
        panel: Draw the bottom panel (hands, advice, count). The web dashboard shows them
            itself, so its video goes without it.
    """
    view = image.copy()
    height, width = view.shape[:2]
    s = _scale(view)
    thick = max(1, round(2 * s))

    if table is not None and not locked:
        line_y = int(table.dealer_line * height)
        for x in range(0, width, int(30 * s) or 1):
            cv2.line(
                view, (x, line_y), (x + int(15 * s), line_y), (200, 200, 200), max(1, thick // 2)
            )
        _text(view, "DEALER", (int(10 * s), line_y - int(8 * s)), 0.5 * s, DEALER_BGR, thick)
        _text(view, "PLAYER", (int(10 * s), line_y + int(22 * s)), 0.5 * s, PLAYER_BGR, thick)

    for card in cards:
        if card.corners is None:
            continue
        corners = np.asarray(card.corners)
        if locked or table is None:
            colour, label = PENDING_BGR, card.card.code if card.card else ""
        else:
            zone = table.zone_of(card, height)
            if card.card is None:
                hole = table.hole_card_visible and zone is Zone.DEALER
                colour, label = PENDING_BGR, "HOLE" if hole else "?"
            else:
                colour = DEALER_BGR if zone is Zone.DEALER else PLAYER_BGR
                label = card.card.code
        cv2.polylines(view, [corners.astype(np.int32)], True, colour, thick)
        if label:
            x, y = corners.min(axis=0).astype(int)
            _text(view, label, (x, max(int(16 * s), y - int(6 * s))), 0.6 * s, colour, thick)

    _text(view, f"{fps:.1f} FPS", (int(10 * s), int(34 * s)), 0.9 * s, (0, 255, 255), thick)
    if lock_text:
        colour = LOCKED_BGR if locked else (0, 220, 0)
        _text(view, lock_text, (int(10 * s), int(70 * s)), 0.8 * s, colour, thick)

    if table is not None and not locked and panel:
        _draw_panel(view, table, s, thick)
    return view


def _draw_panel(view: np.ndarray, table: Table, s: float, thick: int) -> None:
    height, width = view.shape[:2]
    line = int(30 * s)
    top = height - 5 * line - int(10 * s)
    overlay = view.copy()
    cv2.rectangle(overlay, (0, top), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, view, 0.45, 0, dst=view)

    x = int(10 * s)
    round_ = table.round
    dealer = _hand_text(round_.dealer) + ("  + hole" if table.hole_card_visible else "")
    _text(view, f"Dealer: {dealer}", (x, top + line), 0.6 * s, DEALER_BGR, thick)
    hands = round_.hands
    player = "   |   ".join(_hand_text(hand) for hand in hands) if hands else "-"
    _text(view, f"Player: {player}", (x, top + 2 * line), 0.6 * s, PLAYER_BGR, thick)

    advice = table.advice
    if advice is not None:
        _text(
            view,
            advice.spanish.upper(),
            (x, top + int(3.4 * line)),
            1.1 * s,
            (0, 255, 255),
            thick + 1,
        )
    elif table.last_results:
        _text(view, "Result: " + ", ".join(table.last_results), (x, top + int(3.2 * line)), 0.7 * s)
    else:
        _text(view, f"Phase: {round_.phase.value}", (x, top + int(3.2 * line)), 0.6 * s)

    shoe = table.shoe
    count = (
        f"RC {shoe.running_count:+d}  TC {shoe.true_count:+.1f}  "
        f"bet {shoe.bet_units()}u  round {table.rounds_played + 1}"
    )
    _text(view, count, (x, top + int(4.6 * line)), 0.55 * s, TEXT_BGR, max(1, thick - 1))
