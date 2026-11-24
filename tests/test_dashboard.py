"""Web dashboard: the state for the page, the HTTP server, the video stream and the demo."""

import json
import runpy
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from blackjack_vision.blackjack.cards import Card
from blackjack_vision.blackjack.table import Table
from blackjack_vision.ui.dashboard import Dashboard, card_state, table_state

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@dataclass
class Seen:
    track_id: int
    corners: np.ndarray
    card: Card | None


def _at(track_id: int, code: str | None, y: float) -> Seen:
    corners = np.array([[10, y], [60, y], [60, y + 70], [10, y + 70]], np.float32)
    return Seen(track_id, corners, Card.parse(code) if code else None)


def _deal(table: Table, *cards: Seen) -> None:
    seen: list[Seen] = []
    for card in cards:
        seen.append(card)
        table.update(seen, [card], 480)


@pytest.fixture
def dashboard():
    server = Dashboard(port=0).start()
    yield server
    server.close()


def _get(server: Dashboard, path: str):
    return urllib.request.urlopen(f"http://127.0.0.1:{server.port}{path}", timeout=5)


def test_card_state_for_the_page():
    assert card_state(Card.parse("TH")) == {"code": "TH", "rank": "10", "suit": "H", "red": True}
    assert card_state(Card.parse("AS"))["red"] is False


def test_table_state_has_hands_advice_count_and_shoe():
    table = Table(decks=6)
    _deal(table, _at(1, "TS", 300), _at(2, "6H", 50), _at(3, "6C", 300))
    state = table_state(table)
    assert [c["code"] for c in state["dealer"]["cards"]] == ["6H"]
    assert [c["code"] for c in state["hands"][0]["cards"]] == ["TS", "6C"]
    assert state["hands"][0]["total"] == 16 and state["hands"][0]["active"]
    assert state["advice"] == {"action": "stand", "label": "Plántate"}  # 16 against a 6
    assert state["count"]["running"] == 1 and state["count"]["seen"] == 3
    shoe = {s["value"]: s for s in state["shoe"]}
    assert shoe[10] == {"value": 10, "left": 95, "full": 96}
    assert shoe[6] == {"value": 6, "left": 22, "full": 24}
    assert shoe[1]["full"] == 24
    json.dumps(state)  # what the server sends
    assert table_state(None) is None


def test_server_serves_page_state_and_commands(dashboard):
    page = _get(dashboard, "/").read().decode()
    assert "<title>Blackjack Vision</title>" in page

    dashboard.publish_state(Table(), fps=14.6, locked=True, lock_progress=2, lock_length=4)
    dashboard.log("Desbloqueado", "say")
    state = json.loads(_get(dashboard, "/state").read())
    assert state["locked"] and state["table"] is None  # nothing about the table while locked
    assert state["lock"] == {"enabled": True, "progress": 2, "length": 4}
    assert state["fps"] == 14.6 and state["log"][-1]["text"] == "Desbloqueado"

    for name in ("new_round", "pause"):
        request = urllib.request.Request(
            f"http://127.0.0.1:{dashboard.port}/command/{name}", method="POST"
        )
        assert json.loads(urllib.request.urlopen(request).read()) == {"ok": True}
    assert dashboard.commands() == ["new_round", "pause"]
    assert dashboard.commands() == []

    bad = urllib.request.Request(f"http://127.0.0.1:{dashboard.port}/command/quit", method="POST")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(bad)
    assert error.value.code == 404  # the page cannot stop the assistant
    with pytest.raises(urllib.error.HTTPError):
        _get(dashboard, "/secret")


def test_video_stream_sends_jpeg_frames_only_while_watched(dashboard):
    frame = np.zeros((64, 48, 3), np.uint8)
    dashboard.publish_frame(frame)
    assert not dashboard.watching and dashboard._jpeg is None  # nobody watching: no encoding

    response = _get(dashboard, "/stream.mjpg")
    assert response.headers["Content-Type"].startswith("multipart/x-mixed-replace")

    def publish():
        for _ in range(50):
            dashboard.publish_frame(frame)
            time.sleep(0.02)

    thread = threading.Thread(target=publish)
    thread.start()
    head = response.read(400)
    thread.join()
    assert head.startswith(b"--frame\r\nContent-Type: image/jpeg")
    assert b"\xff\xd8" in head  # JPEG start
    assert dashboard.watching
    response.close()


def test_run_parses_the_web_option():
    parse = runpy.run_path(str(SCRIPTS / "run.py"), run_name="run_cli")["parse_args"]
    assert parse([]).web is None
    assert parse(["--web"]).web == 8000
    assert parse(["--web", "8080"]).web == 8080
    assert parse(["--web"]).web_host == "127.0.0.1"


def test_demo_plays_a_round():
    demo = runpy.run_path(str(SCRIPTS / "dashboard_demo.py"), run_name="demo_cli")
    args = demo["parse_args"](["--port", "0", "--no-lock", "--seed", "3", "--speed", "50"])
    server = Dashboard(port=0).start()
    try:
        game = demo["Demo"](args, server)
        frames = game.frames()
        for _ in range(200):
            next(frames)
            game.step(15.0)
            if game.table.rounds_played:
                break
        assert game.table.rounds_played == 1
        log = [line["text"] for line in server.snapshot()["log"]]
        assert any(text.startswith("Round over") for text in log)
    finally:
        server.close()


def test_run_feeds_the_dashboard(tmp_path):
    """``run.py --web --no-window`` on a folder of frames, with a fake recogniser."""
    from synthetic_cards import mat, paste_card, placed_corners

    from blackjack_vision.capture.sources import write_image

    corners = placed_corners((320, 380), 150, angle_deg=5)
    scene = paste_card(mat((640, 480)), corners)
    for i in range(25):
        write_image(tmp_path / "frames" / f"{i:03d}.png", scene)

    @dataclass
    class Named:
        card: Card | None
        similarity: float = 0.9

    class FakeRecognizer:
        def recognise(self, card_image):
            return Named(Card.parse("9H"))

    created: list[Dashboard] = []

    class Captured(Dashboard):
        def __init__(self, host, port):
            super().__init__(host, 0)
            created.append(self)
            self._commands.put("new_shoe")

    run = runpy.run_path(str(SCRIPTS / "run.py"), run_name="run_cli")
    run["main"].__globals__["build_recognizer"] = lambda args: FakeRecognizer()
    run["main"].__globals__["Dashboard"] = Captured
    code = run["main"](
        [
            "--source", str(tmp_path / "frames"),
            "--fps", "50",
            "--no-lock", "--no-window", "--no-voice",
            "--web",
        ]
    )  # fmt: skip
    assert code == 0
    snapshot = created[0].snapshot()
    hand = snapshot["table"]["hands"][0]
    assert [card["code"] for card in hand["cards"]] == ["9H"]
    log = [line["text"] for line in snapshot["log"]]
    assert "Nuevo sabot" in log and "9H to the player" in log
    assert snapshot["lock"]["enabled"] is False and not snapshot["locked"]
