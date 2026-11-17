"""Synthetic rectified card faces (rank index, suit pips, colours) for the recognition tests."""

from __future__ import annotations

import cv2
import numpy as np

from blackjack_vision.blackjack.cards import Card, Rank, Suit
from blackjack_vision.vision_classic.rectify import CARD_SIZE

RED = (30, 30, 200)
BLACK = (25, 25, 25)
PAPER = (245, 245, 245)

# Pip layouts (x, y in card fractions) of the number cards.
_COLUMNS = (0.32, 0.68)
_PIPS: dict[int, list[tuple[float, float]]] = {
    1: [(0.5, 0.5)],
    2: [(0.5, 0.25), (0.5, 0.75)],
    3: [(0.5, 0.25), (0.5, 0.5), (0.5, 0.75)],
    4: [(x, y) for x in _COLUMNS for y in (0.25, 0.75)],
    5: [(x, y) for x in _COLUMNS for y in (0.25, 0.75)] + [(0.5, 0.5)],
    6: [(x, y) for x in _COLUMNS for y in (0.25, 0.5, 0.75)],
    7: [(x, y) for x in _COLUMNS for y in (0.25, 0.5, 0.75)] + [(0.5, 0.37)],
    8: [(x, y) for x in _COLUMNS for y in (0.25, 0.5, 0.75)] + [(0.5, 0.37), (0.5, 0.63)],
    9: [(x, y) for x in _COLUMNS for y in (0.22, 0.41, 0.59, 0.78)] + [(0.5, 0.5)],
    10: [(x, y) for x in _COLUMNS for y in (0.22, 0.41, 0.59, 0.78)] + [(0.5, 0.32), (0.5, 0.68)],
}


def draw_suit(image: np.ndarray, suit: Suit, centre: tuple[float, float], size: float) -> None:
    """Draw a suit symbol of the given ``size`` (pixels) centred at ``centre``."""
    colour = RED if suit.is_red else BLACK
    cx, cy = centre
    r = size / 2.0

    def pt(dx: float, dy: float) -> tuple[int, int]:
        return round(cx + dx * r), round(cy + dy * r)

    def disc(dx: float, dy: float, radius: float) -> None:
        cv2.circle(image, pt(dx, dy), max(1, round(radius * r)), colour, -1, cv2.LINE_AA)

    def poly(points: list[tuple[float, float]]) -> None:
        polygon = np.array([pt(dx, dy) for dx, dy in points], np.int32)
        cv2.fillPoly(image, [polygon], colour, cv2.LINE_AA)

    if suit is Suit.DIAMONDS:
        poly([(0, -1), (0.7, 0), (0, 1), (-0.7, 0)])
    elif suit is Suit.HEARTS:
        disc(-0.45, -0.35, 0.5)
        disc(0.45, -0.35, 0.5)
        poly([(-0.93, -0.2), (0.93, -0.2), (0, 1)])
    elif suit is Suit.SPADES:
        disc(-0.45, 0.2, 0.5)
        disc(0.45, 0.2, 0.5)
        poly([(-0.93, 0.05), (0.93, 0.05), (0, -1)])
        poly([(0, 0.3), (0.3, 1), (-0.3, 1)])
    else:  # clubs
        disc(0, -0.5, 0.42)
        disc(-0.5, 0.15, 0.42)
        disc(0.5, 0.15, 0.42)
        poly([(0, 0), (0.3, 1), (-0.3, 1)])


def card_face(card: Card, size: tuple[int, int] = CARD_SIZE) -> np.ndarray:
    """Upright rectified card: indices in two corners, pips or a face picture in the middle."""
    width, height = size
    image = np.full((height, width, 3), PAPER, np.uint8)
    colour = RED if card.suit.is_red else BLACK
    text = "10" if card.rank is Rank.TEN else card.rank.value

    corner = np.full((90, 40, 3), PAPER, np.uint8)
    scale = 0.8 if len(text) == 2 else 1.1
    cv2.putText(corner, text, (3, 32), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)
    draw_suit(corner, card.suit, (19, 55), 24)
    image[4:94, 4:44] = corner
    image[height - 94 : height - 4, width - 44 : width - 4] = cv2.rotate(corner, cv2.ROTATE_180)

    value = card.rank.blackjack_value
    if card.rank in (Rank.JACK, Rank.QUEEN, Rank.KING):
        cv2.rectangle(image, (50, 50), (width - 51, height - 51), colour, 2)
        cv2.rectangle(image, (60, 60), (width - 61, height - 61), (40, 160, 200), 3)
        cv2.putText(
            image, card.rank.value, (72, 180), cv2.FONT_HERSHEY_TRIPLEX, 3.0, colour, 5, cv2.LINE_AA
        )
        draw_suit(image, card.suit, (width / 2, 220), 30)
    else:
        pip = 80 if value == 1 else 34
        for fx, fy in _PIPS[value]:
            draw_suit(image, card.suit, (fx * width, fy * height), pip)
    return image


def photographed(face: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Imitate a card detected and rectified from a real frame.

    Small corner errors (homography jitter), brightness and colour cast changes, blur and sensor
    noise; half of the cards come out rotated by 180 degrees, like the detector may return them.
    """
    height, width = face.shape[:2]
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
    )
    jittered = corners + rng.uniform(-5, 5, corners.shape).astype(np.float32)
    homography = cv2.getPerspectiveTransform(corners, jittered)
    image = cv2.warpPerspective(face, homography, (width, height), borderMode=cv2.BORDER_REPLICATE)
    image = image.astype(np.float32) * rng.uniform(0.7, 1.05) * rng.uniform(0.92, 1.0, 3)
    image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.3, 1.2))
    image += rng.normal(0, 4, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        image = cv2.rotate(image, cv2.ROTATE_180)
    return image
