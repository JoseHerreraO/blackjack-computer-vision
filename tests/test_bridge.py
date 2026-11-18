import json
import socket
import struct
import time

import cv2
import numpy as np
import pytest

from blackjack_vision.capture.bridge import COMMAND, FRAME, HELLO, LOG, BridgeSource, encode_message
from blackjack_vision.capture.sources import SourceKind, parse_source_spec


def _jpeg(value: int, size=(72, 128)) -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.full((size[1], size[0], 3), value, np.uint8))
    assert ok
    return buffer.tobytes()


def _frame_message(capture_ms: int, value: int) -> bytes:
    return encode_message(FRAME, struct.pack(">q", capture_ms) + _jpeg(value))


def _read_message(sock: socket.socket) -> tuple[int, bytes]:
    header = b""
    while len(header) < 5:
        header += sock.recv(5 - len(header))
    kind, length = struct.unpack(">BI", header)
    payload = b""
    while len(payload) < length:
        payload += sock.recv(length - len(payload))
    return kind, payload


@pytest.fixture
def bridge():
    source = BridgeSource(port=0, host="127.0.0.1", timeout=2.0)
    yield source
    source.release()


@pytest.fixture
def iphone(bridge):
    client = socket.create_connection(("127.0.0.1", bridge.port), timeout=2.0)
    yield client
    client.close()


def _wait_until(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.mark.parametrize(
    ("spec", "value"), [("bridge", 5050), ("bridge:6000", 6000), ("BRIDGE:5051", 5051)]
)
def test_parse_bridge_spec(spec, value):
    assert parse_source_spec(spec) == (SourceKind.BRIDGE, value)


def test_parse_bridge_spec_rejects_bad_port():
    with pytest.raises(ValueError):
        parse_source_spec("bridge:abc")


def test_hello_frames_and_logs(bridge, iphone):
    hello = {"device": "test", "width": 72, "height": 128, "fps": 15, "jpeg_quality": 0.8}
    iphone.sendall(encode_message(HELLO, json.dumps(hello).encode()))
    iphone.sendall(encode_message(LOG, b"hello from the phone"))
    iphone.sendall(_frame_message(1_000, 50))

    first = bridge.read()
    assert first is not None
    assert first.index == 0 and first.timestamp == 0.0 and first.size == (72, 128)
    assert bridge.hello == hello and bridge.fps == 15.0

    iphone.sendall(_frame_message(1_500, 150))
    second = bridge.read()
    assert second.index == 1 and second.timestamp == pytest.approx(0.5)
    assert abs(int(second.image.mean()) - 150) < 5


def test_read_returns_only_the_latest_frame(bridge, iphone):
    for i in range(5):
        iphone.sendall(_frame_message(1_000 + i, 20 * i))
    _wait_until(lambda: bridge.frames_received == 5)

    frame = bridge.read_bridge_frame()
    assert frame.capture_ms == 1_004
    assert bridge.frames_dropped == 4


def test_read_times_out_without_frames(bridge):
    start = time.monotonic()
    bridge._timeout = 0.2
    assert bridge.read() is None
    assert time.monotonic() - start >= 0.2


def test_say_sends_a_command(bridge, iphone):
    _wait_until(lambda: bridge.connected)
    assert bridge.say("Pide carta")
    kind, payload = _read_message(iphone)
    assert kind == COMMAND
    assert json.loads(payload) == {"type": "say", "text": "Pide carta", "lang": "es-ES"}


def test_say_without_iphone_returns_false(bridge):
    assert not bridge.say("nadie escucha")


def test_reconnect_after_disconnect(bridge, iphone):
    iphone.sendall(_frame_message(1_000, 10))
    assert bridge.read() is not None
    iphone.close()
    _wait_until(lambda: not bridge.connected)

    again = socket.create_connection(("127.0.0.1", bridge.port), timeout=2.0)
    try:
        again.sendall(_frame_message(2_000, 90))
        frame = bridge.read()
        assert frame is not None and frame.name == "2000"
    finally:
        again.close()


def test_release_unblocks_read(bridge):
    bridge._timeout = None
    bridge.release()
    assert bridge.read() is None


def test_new_connection_replaces_a_dead_one(bridge, iphone):
    # The first iPhone goes silent without closing, like a phone that lost the Wi-Fi.
    iphone.sendall(_frame_message(1_000, 10))
    assert bridge.read() is not None

    again = socket.create_connection(("127.0.0.1", bridge.port), timeout=2.0)
    try:
        again.sendall(_frame_message(2_000, 90))
        frame = bridge.read()
        assert frame is not None and frame.name == "2000"
        _wait_until(lambda: bridge.connected)
        assert bridge.say("sigo aqui")
        kind, _ = _read_message(again)
        assert kind == COMMAND
    finally:
        again.close()
