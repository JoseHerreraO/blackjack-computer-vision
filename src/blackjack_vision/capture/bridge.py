"""Live frames from the BJ Bridge iOS app (Meta glasses -> iPhone -> PC over TCP).

The PC is the TCP server and the iPhone connects to it. Wire format, shared with
``mobile/ios/BlackjackBridge/Networking/BridgeProtocol.swift`` (do not change one side only)::

    [1 byte type][uint32 big-endian payload length][payload]

    iPhone -> PC  0x01 HELLO   JSON {"device","width","height","fps","jpeg_quality","app_version"}
                  0x02 FRAME   int64 big-endian capture time (ms since epoch) + JPEG
                  0x03 LOG     UTF-8 text
    PC -> iPhone  0x10 COMMAND JSON {"type": "say", "text", "lang"} or {"type": "status", "text"}

Frames are latest-only: :meth:`BridgeSource.read` always returns the newest frame and drops any
frame that arrived while the pipeline was busy, so a slow stage never builds up latency.

Example::

    with BridgeSource(port=5050) as source:
        for frame in source:
            ...
            source.say("Pide carta")
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from blackjack_vision.capture.sources import Frame, FrameSource

logger = logging.getLogger(__name__)

DEFAULT_PORT = 5050
HELLO, FRAME, LOG, COMMAND = 0x01, 0x02, 0x03, 0x10
HEADER = struct.Struct(">BI")
CAPTURE_TIME = struct.Struct(">q")
MAX_PAYLOAD = 16 * 1024 * 1024


def encode_message(kind: int, payload: bytes) -> bytes:
    """Frame one message of the bridge protocol."""
    return HEADER.pack(kind, len(payload)) + payload


def encode_command(command: dict[str, Any]) -> bytes:
    """COMMAND message for the iPhone."""
    return encode_message(COMMAND, json.dumps(command, ensure_ascii=False).encode("utf-8"))


@dataclass(frozen=True)
class BridgeFrame:
    """A decoded frame plus the raw bytes and timing it arrived with."""

    image: np.ndarray
    jpeg: bytes
    capture_ms: int
    received_at: float


class BridgeSource(FrameSource):
    """TCP server that receives the glasses stream from the iPhone app.

    The server starts listening immediately in a background thread and accepts one iPhone at a
    time; when it disconnects, the next connection is accepted (the app retries every 2 s).

    Args:
        port: TCP port (5050 is the one the app expects by default).
        host: Interface to bind, all of them by default.
        timeout: Seconds :meth:`read` waits for a new frame before giving up and returning
            ``None``. ``None`` waits until :meth:`release`.
    """

    def __init__(
        self, port: int = DEFAULT_PORT, *, host: str = "0.0.0.0", timeout: float | None = None
    ) -> None:
        self._timeout = timeout
        self._condition = threading.Condition()
        self._latest: BridgeFrame | None = None
        self._latest_seq = 0
        self._read_seq = 0
        self._index = 0
        self._first_capture_ms: int | None = None
        self._client: socket.socket | None = None
        self._closed = False
        self.hello: dict[str, Any] | None = None
        self.frames_received = 0
        self.frames_dropped = 0

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(1)
        self.port = self._server.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, name="bridge-server", daemon=True)
        self._thread.start()
        logger.info("Waiting for the iPhone on port %d", self.port)

    # FrameSource -------------------------------------------------------------------------------

    @property
    def is_live(self) -> bool:
        return True

    @property
    def fps(self) -> float | None:
        value = (self.hello or {}).get("fps")
        return float(value) if value else None

    @property
    def connected(self) -> bool:
        with self._condition:
            return self._client is not None

    def read(self) -> Frame | None:
        bridge_frame = self.read_bridge_frame()
        if bridge_frame is None:
            return None
        if self._first_capture_ms is None:
            self._first_capture_ms = bridge_frame.capture_ms
        frame = Frame(
            image=bridge_frame.image,
            index=self._index,
            timestamp=(bridge_frame.capture_ms - self._first_capture_ms) / 1000.0,
            name=str(bridge_frame.capture_ms),
        )
        self._index += 1
        return frame

    def read_bridge_frame(self) -> BridgeFrame | None:
        """Newest frame not returned yet, with its JPEG bytes and capture time."""
        deadline = None if self._timeout is None else time.monotonic() + self._timeout
        with self._condition:
            while not self._closed and self._latest_seq == self._read_seq:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._closed:
                return None
            skipped = self._latest_seq - self._read_seq - 1
            self.frames_dropped += max(0, skipped)
            self._read_seq = self._latest_seq
            return self._latest

    def release(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            client = self._client
            self._condition.notify_all()
        for sock in (client, self._server):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    # Commands to the iPhone --------------------------------------------------------------------

    def send_command(self, command: dict[str, Any]) -> bool:
        """Send a COMMAND. Returns ``False`` when no iPhone is connected or the send fails."""
        with self._condition:
            client = self._client
        if client is None:
            return False
        try:
            client.sendall(encode_command(command))
            return True
        except OSError as error:
            logger.warning("Could not send a command to the iPhone: %s", error)
            return False

    def say(self, text: str, lang: str = "es-ES") -> bool:
        """Have the iPhone speak ``text`` through the glasses."""
        return self.send_command({"type": "say", "text": text, "lang": lang})

    def status(self, text: str) -> bool:
        """Show ``text`` as a status line in the app."""
        return self.send_command({"type": "status", "text": text})

    # Server thread -----------------------------------------------------------------------------

    def _serve(self) -> None:
        while not self._closed:
            try:
                client, address = self._server.accept()
            except OSError:
                return  # released
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            logger.info("iPhone connected from %s:%d", *address)
            with self._condition:
                self._client = client
            try:
                self._read_messages(client)
            except (ConnectionError, OSError) as error:
                if not self._closed:
                    logger.info("iPhone disconnected: %s", error)
            finally:
                with self._condition:
                    self._client = None
                try:
                    client.close()
                except OSError:
                    pass

    def _read_messages(self, client: socket.socket) -> None:
        while not self._closed:
            kind, length = HEADER.unpack(_read_exact(client, HEADER.size))
            if length > MAX_PAYLOAD:
                raise ConnectionError(
                    f"message too large ({length} bytes), is this the BJ Bridge app?"
                )
            payload = _read_exact(client, length)
            if kind == HELLO:
                self.hello = json.loads(payload)
                logger.info("HELLO %s", self.hello)
            elif kind == FRAME:
                self._handle_frame(payload)
            elif kind == LOG:
                logger.debug("[iPhone] %s", payload.decode("utf-8", errors="replace"))
            else:
                logger.warning("Unknown message type 0x%02X (%d bytes)", kind, length)

    def _handle_frame(self, payload: bytes) -> None:
        if len(payload) <= CAPTURE_TIME.size:
            logger.warning("FRAME without image data")
            return
        (capture_ms,) = CAPTURE_TIME.unpack_from(payload)
        jpeg = payload[CAPTURE_TIME.size :]
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Could not decode a JPEG frame")
            return
        frame = BridgeFrame(image=image, jpeg=jpeg, capture_ms=capture_ms, received_at=time.time())
        with self._condition:
            self._latest = frame
            self._latest_seq += 1
            self.frames_received += 1
            self._condition.notify_all()


def _read_exact(client: socket.socket, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = client.recv(length - len(data))
        if not chunk:
            raise ConnectionError("closed by the iPhone")
        data.extend(chunk)
    return bytes(data)
