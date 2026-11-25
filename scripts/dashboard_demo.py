"""Demo of the web dashboard without glasses or models: simulated rounds on a drawn table.

The cards are drawn from a shuffled shoe and placed on a synthetic table image, and the real
:class:`~blackjack_vision.blackjack.table.Table` follows them as it does in ``scripts/run.py``:
zones, hole card, basic-strategy advice, Hi-Lo count and results. The player follows the advice
(hit and double take a card; stand, split and surrender go to the dealer). It starts locked and
"shows" the four aces, so the lock screen can be seen too.

Useful to try the page, to record it for the video and for screenshots of the report::

    python scripts/dashboard_demo.py            # then open http://localhost:8000
    python scripts/dashboard_demo.py --port 8080 --speed 2 --no-lock

Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from blackjack_vision.blackjack.cards import Card, Rank, Suit
from blackjack_vision.blackjack.game import Phase
from blackjack_vision.blackjack.strategy import Action
from blackjack_vision.blackjack.table import Table
from blackjack_vision.ui.dashboard import Dashboard
from blackjack_vision.ui.hud import draw_hud

WIDTH, HEIGHT = 360, 640
CARD_W, CARD_H = 64, 90
FPS = 15.0
SECRET_LENGTH = 4

logger = logging.getLogger("dashboard_demo")


@dataclass
class DemoCard:
    """A card on the drawn table (fits ``SeenCard``: track id, corners and name)."""

    track_id: int
    corners: np.ndarray
    card: Card | None
    shown: Card | None = None  # what is drawn (the hole card is drawn face down until revealed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast")
    parser.add_argument("--decks", type=int, default=6)
    parser.add_argument("--no-lock", action="store_true", help="start unlocked")
    parser.add_argument("--seed", type=int, help="shoe order (random by default)")
    parser.add_argument("--rounds", type=int, help="stop after this many rounds")
    return parser.parse_args(argv)


def new_shoe(decks: int, rng: random.Random) -> list[Card]:
    cards = [Card(rank, suit) for _ in range(decks) for rank in Rank for suit in Suit]
    rng.shuffle(cards)
    return cards


def felt() -> np.ndarray:
    """Green table seen from above, a bit darker at the edges."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    r = np.hypot((x - WIDTH / 2) / WIDTH, (y - HEIGHT / 2) / HEIGHT)
    shade = np.clip(1.0 - 0.9 * r, 0.45, 1.0)[..., None]
    image = (np.array([70, 110, 20], np.float32) * shade).astype(np.uint8)
    cv2.ellipse(image, (WIDTH // 2, -40), (WIDTH, 300), 0, 0, 180, (60, 160, 200), 2)
    return image


def corners_at(x: float, y: float) -> np.ndarray:
    return np.array(
        [[x, y], [x + CARD_W, y], [x + CARD_W, y + CARD_H], [x, y + CARD_H]], np.float32
    )


def draw_card(image: np.ndarray, card: DemoCard) -> None:
    x0, y0 = card.corners[0].astype(int)
    x1, y1 = card.corners[2].astype(int)
    cv2.rectangle(image, (x0 + 3, y0 + 4), (x1 + 3, y1 + 4), (20, 40, 10), -1)  # shadow
    if card.shown is None:
        cv2.rectangle(image, (x0, y0), (x1, y1), (245, 245, 245), -1)
        back = np.full((y1 - y0 - 9, x1 - x0 - 9, 3), (50, 40, 150), np.uint8)
        for k in range(-back.shape[0], back.shape[1], 10):
            cv2.line(back, (k, 0), (k + back.shape[0], back.shape[0]), (70, 60, 180), 2)
        image[y0 + 5 : y0 + 5 + back.shape[0], x0 + 5 : x0 + 5 + back.shape[1]] = back
        return
    cv2.rectangle(image, (x0, y0), (x1, y1), (250, 250, 247), -1)
    colour = (40, 40, 200) if card.shown.suit.is_red else (30, 25, 20)
    rank = "10" if card.shown.rank is Rank.TEN else card.shown.rank.value
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, rank, (x0 + 6, y0 + 24), font, 0.75, colour, 2, cv2.LINE_AA)
    cv2.putText(image, card.shown.suit.value, (x0 + 22, y0 + 64), font, 1.1, colour, 2, cv2.LINE_AA)


class Demo:
    """One simulated table: a generator of frames that deals rounds following the advice."""

    def __init__(self, args: argparse.Namespace, dashboard: Dashboard) -> None:
        self.args = args
        self.dashboard = dashboard
        self.rng = random.Random(args.seed)
        self.table = Table(decks=args.decks)
        self.shoe = new_shoe(args.decks, self.rng)
        self.background = felt()
        self.cards: list[DemoCard] = []
        self.new: list[DemoCard] = []
        self.next_id = 0
        self.locked = not args.no_lock
        self.lock_progress = 0
        self.restart = False
        self.paused = False

    # Dealing ------------------------------------------------------------------------------------

    def draw(self) -> Card:
        if len(self.shoe) < 15:
            self.shoe = new_shoe(self.args.decks, self.rng)
            self.table.reset(reshuffle=True)
            self.dashboard.log("Sabot nuevo: cuenta a 0")
        return self.shoe.pop()

    def place(self, card: Card | None, dealer: bool, shown: Card | None = None) -> DemoCard:
        row = [c for c in self.cards if (c.corners[0, 1] < HEIGHT / 2) == dealer]
        x = 40 + 46 * len(row)
        y = 120 if dealer else 420
        demo = DemoCard(self.next_id, corners_at(x, y), card, shown or card)
        self.next_id += 1
        self.cards.append(demo)
        if card is not None:
            self.new.append(demo)
        return demo

    def hold(self, seconds: float) -> Iterator[None]:
        for _ in range(max(1, round(seconds * FPS / self.args.speed))):
            yield
            if self.restart:
                return

    def lock_sequence(self) -> Iterator[None]:
        self.locked, self.lock_progress = True, 0
        yield from self.hold(2.0)
        for suit in (Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)[:SECRET_LENGTH]:
            ace = self.place(None, dealer=False, shown=Card(Rank.ACE, suit))
            ace.corners = corners_at(WIDTH / 2 - CARD_W / 2, HEIGHT / 2 - CARD_H / 2)
            yield from self.hold(1.0)
            self.lock_progress += 1
            self.cards.clear()
            yield from self.hold(0.5)
        self.locked = False
        self.dashboard.log("Desbloqueado", "say")

    def round(self) -> Iterator[None]:
        table = self.table
        self.place(self.draw(), dealer=False)
        yield from self.hold(1.0)
        self.place(self.draw(), dealer=True)
        yield from self.hold(1.0)
        self.place(self.draw(), dealer=False)
        yield from self.hold(1.0)
        hole_card = self.draw()
        hole = self.place(None, dealer=True, shown=None)
        yield from self.hold(2.5)

        while table.round.phase is Phase.PLAYER_TURN and table.advice in (
            Action.HIT,
            Action.DOUBLE,
        ):
            doubled = table.advice is Action.DOUBLE
            self.place(self.draw(), dealer=False)
            yield from self.hold(2.0)
            if doubled:
                break
        yield from self.hold(1.0)

        if table.round.phase is not Phase.FINISHED:  # a bust ends the round without it
            hole.card = hole.shown = hole_card  # turned over: recognised on the same track
            self.new.append(hole)
        yield from self.hold(1.5)
        while table.round.phase is Phase.DEALER_TURN:
            self.place(self.draw(), dealer=True)
            yield from self.hold(1.5)
        yield from self.hold(3.5)
        self.cards.clear()
        # Empty table long enough for the table to start a new round (whatever the speed),
        # otherwise a card in the same place is taken for the last round's card seen again.
        for _ in range(self.table.clear_frames + 3):
            yield

    def frames(self) -> Iterator[None]:
        rounds = 0
        while self.args.rounds is None or rounds < self.args.rounds:
            if self.locked:
                yield from self.lock_sequence()
            if self.restart:
                self.restart = False
                self.cards.clear()
                continue
            yield from self.round()
            rounds += 1
            if self.restart:
                self.restart = False
                self.cards.clear()

    # Commands -----------------------------------------------------------------------------------

    def command(self, name: str) -> None:
        if name == "pause":
            self.paused = not self.paused
            return
        if name == "new_shoe":
            self.shoe = new_shoe(self.args.decks, self.rng)
            self.table.reset(reshuffle=True)
            self.dashboard.log("Nuevo sabot")
        elif name == "new_round":
            self.table.reset()
            self.dashboard.log("Nueva ronda")
        elif name == "lock":
            self.locked = True
            self.dashboard.log("Bloqueado")
        self.restart = True

    # Frame --------------------------------------------------------------------------------------

    def step(self, fps: float) -> None:
        """Feed the table with the current cards and publish the frame and the state."""
        image = self.background.copy()
        for card in self.cards:
            draw_card(image, card)
        if self.locked:
            text = (
                "LOCKED  " + "#" * self.lock_progress + "." * (SECRET_LENGTH - self.lock_progress)
            )
            view = draw_hud(image, self.cards, fps, lock_text=text, locked=True)
        else:
            update = self.table.update(self.cards, self.new, HEIGHT, mat_in_view=True)
            self.new = []
            for event in update.events:
                logger.info(event)
                self.dashboard.log(event)
            for phrase in update.speech:
                logger.info("SAY: %s", phrase)
                self.dashboard.log(phrase, "say")
            lock = None if self.args.no_lock else "UNLOCKED"
            view = draw_hud(image, self.cards, fps, table=self.table, lock_text=lock, panel=False)
        self.dashboard.publish_frame(view)
        self.dashboard.publish_state(
            self.table,
            fps=fps,
            locked=self.locked,
            lock_progress=self.lock_progress,
            lock_length=0 if self.args.no_lock else SECRET_LENGTH,
            paused=self.paused,
            source="demo",
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dashboard = Dashboard(args.host, args.port).start()
    print(f"Open {dashboard.url}  (Ctrl+C to stop)")
    demo = Demo(args, dashboard)
    frames = demo.frames()
    period = 1.0 / FPS
    times: list[float] = []
    try:
        while True:
            start = time.perf_counter()
            for name in dashboard.commands():
                demo.command(name)
            if not demo.paused:
                try:
                    next(frames)
                except StopIteration:
                    break
            times = [*times[-29:], start]
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else FPS
            demo.step(fps)
            time.sleep(max(0.0, period - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
