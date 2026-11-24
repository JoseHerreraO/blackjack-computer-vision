"""Web dashboard on localhost: the live video and the state of the table in the browser.

``scripts/run.py --web`` starts it next to the OpenCV window (or instead of it, with
``--no-window``). It only uses the Python standard library:

* ``GET /`` the page (``dashboard.html``): the video, the dealer's and the player's cards, the
  advice in large letters, the Hi-Lo count with the suggested bet, the cards left in the shoe by
  value, the results of the round and a log of what was said through the glasses.
* ``GET /stream.mjpg`` the annotated video as MJPEG (``multipart/x-mixed-replace``), which any
  browser plays in an ``<img>``. Frames are only JPEG-encoded while someone is watching.
* ``GET /state`` the state as JSON (:func:`table_state`), polled by the page.
* ``POST /command/<name>`` buttons of the page: ``new_round``, ``new_shoe``, ``lock`` and
  ``pause``. They are queued and the main loop runs them (:meth:`Dashboard.commands`), as it does
  with the keys of the OpenCV window.

The server listens on ``127.0.0.1`` by default, so only this PC sees it. The lock secret never
leaves the main loop: the state only has how many aces of it have been shown.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files

import cv2
import numpy as np

from blackjack_vision.blackjack.cards import Card, Rank
from blackjack_vision.blackjack.hand import Hand
from blackjack_vision.blackjack.table import Table

logger = logging.getLogger(__name__)

COMMANDS = ("new_round", "new_shoe", "lock", "pause")
BOUNDARY = "frame"
JPEG_QUALITY = 80
LOG_LINES = 12


def card_state(card: Card) -> dict[str, object]:
    """``{"code": "TH", "rank": "10", "suit": "H", "red": true}`` for the page."""
    rank = "10" if card.rank is Rank.TEN else card.rank.value
    return {"code": card.code, "rank": rank, "suit": card.suit.value, "red": card.suit.is_red}


def hand_state(hand: Hand, *, active: bool = False) -> dict[str, object]:
    return {
        "cards": [card_state(card) for card in hand.cards],
        "total": hand.total if hand.cards else None,
        "soft": bool(hand.cards) and hand.is_soft and not hand.is_bust,
        "bust": bool(hand.cards) and hand.is_bust,
        "blackjack": hand.is_blackjack,
        "active": active,
    }


def table_state(table: Table | None) -> dict[str, object] | None:
    """Everything the page shows about the table and the shoe (``None`` while locked)."""
    if table is None:
        return None
    round_ = table.round
    active = round_.active_index
    advice = table.advice
    shoe = table.shoe
    remaining = shoe.remaining_by_value()
    full = {value: 4 * shoe.decks * (4 if value == 10 else 1) for value in range(1, 11)}
    return {
        "phase": round_.phase.value,
        "round": table.rounds_played + 1,
        "dealer": {**hand_state(round_.dealer), "hole": table.hole_card_visible},
        "hands": [hand_state(hand, active=i == active) for i, hand in enumerate(round_.hands)],
        "advice": None if advice is None else {"action": advice.value, "label": advice.spanish},
        "results": list(table.last_results),
        "count": {
            "running": shoe.running_count,
            "true": round(shoe.true_count, 2),
            "bet": shoe.bet_units(),
            "advantage": round(100 * shoe.advantage, 2),
            "decks_remaining": round(shoe.decks_remaining, 2),
            "penetration": round(100 * shoe.penetration, 1),
            "seen": shoe.cards_seen,
            "total": shoe.total_cards,
        },
        "shoe": [
            {"value": value, "left": remaining[value], "full": full[value]}
            for value in range(1, 11)
        ],
    }


class Dashboard:
    """The web server and what it shows. Thread-safe: the main loop publishes, the server reads.

    Args:
        host: Address to listen on (``127.0.0.1``: only this PC).
        port: TCP port (``0`` picks a free one, see :attr:`port`).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._jpeg: bytes | None = None
        self._frame_id = 0
        self._viewers = 0
        self._state: dict[str, object] = {"locked": False, "fps": 0.0, "table": None}
        self._log: deque[dict[str, object]] = deque(maxlen=LOG_LINES)
        self._commands: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._closed = False
        self._page = (files("blackjack_vision.ui") / "dashboard.html").read_bytes()
        self._server = ThreadingHTTPServer((host, port), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # Life cycle ---------------------------------------------------------------------------------

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        host = self._server.server_address[0]
        return f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{self.port}/"

    def start(self) -> Dashboard:
        self._thread.start()
        logger.info("Dashboard on %s", self.url)
        return self

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._frame_ready.notify_all()
        self._server.shutdown()
        self._server.server_close()

    # Main loop side -----------------------------------------------------------------------------

    @property
    def watching(self) -> bool:
        """Whether a browser is playing the video (frames are only encoded then)."""
        return self._viewers > 0

    def publish_frame(self, image: np.ndarray) -> None:
        """New annotated frame (BGR). Skipped when nobody is watching the video."""
        if not self.watching:
            return
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        with self._lock:
            self._jpeg = jpeg.tobytes()
            self._frame_id += 1
            self._frame_ready.notify_all()

    def publish_state(
        self,
        table: Table | None,
        *,
        fps: float,
        locked: bool,
        lock_progress: int = 0,
        lock_length: int = 0,
        paused: bool = False,
        source: str = "",
    ) -> None:
        """New state of the table and of the lock (``lock_length`` 0: no lock)."""
        state = {
            "time": time.time(),
            "fps": round(float(fps), 1),
            "paused": paused,
            "source": source,
            "lock": {"enabled": lock_length > 0, "progress": lock_progress, "length": lock_length},
            "locked": locked,
            "table": None if locked else table_state(table),
        }
        with self._lock:
            self._state = state

    def log(self, text: str, kind: str = "event") -> None:
        """A line for the page's log (``kind``: ``say``, ``event`` or ``result``)."""
        with self._lock:
            self._log.append({"time": time.time(), "kind": kind, "text": text})

    def commands(self) -> list[str]:
        """Commands from the page's buttons since the last call, in order."""
        pending = []
        while True:
            try:
                pending.append(self._commands.get_nowait())
            except queue.Empty:
                return pending

    # Server side --------------------------------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {**self._state, "log": list(self._log)}

    def _frames(self) -> Iterator[bytes]:
        """JPEG frames as they are published (each one once), until the dashboard closes."""
        last = -1
        while True:
            with self._lock:
                self._frame_ready.wait_for(
                    lambda seen=last: self._closed or self._frame_id != seen, timeout=1.0
                )
                if self._closed:
                    return
                if self._frame_id == last or self._jpeg is None:
                    continue
                last, jpeg = self._frame_id, self._jpeg
            yield jpeg

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "BlackjackDashboard"

            def log_message(self, format: str, *args: object) -> None:
                logger.debug("web: " + format, *args)

            def _send(self, body: bytes, content_type: str, status=HTTPStatus.OK) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    self._send(dashboard._page, "text/html; charset=utf-8")
                elif path == "/state":
                    body = json.dumps(dashboard.snapshot()).encode()
                    self._send(body, "application/json")
                elif path == "/stream.mjpg":
                    self._stream()
                else:
                    self._send(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                name = self.path.removeprefix("/command/")
                if not self.path.startswith("/command/") or name not in COMMANDS:
                    self._send(b"Unknown command", "text/plain", HTTPStatus.NOT_FOUND)
                    return
                dashboard._commands.put(name)
                self._send(b'{"ok": true}', "application/json")

            def _stream(self) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with dashboard._lock:
                    dashboard._viewers += 1
                try:
                    for jpeg in dashboard._frames():
                        self.wfile.write(
                            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                            + jpeg
                            + b"\r\n"
                        )
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # the browser closed the page
                finally:
                    with dashboard._lock:
                        dashboard._viewers -= 1

        return Handler
