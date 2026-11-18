"""Test receiver for the BJ Bridge iOS app.

A minimal PC-side TCP server for checking the iPhone app without the full pipeline.
It shows the glasses' frames in a window with the receive rate and latency, prints
HELLO and LOG messages, and sends commands typed in the terminal back to the iPhone.

    python mobile/ios/tools/pc_receiver.py [--port 5050]

Terminal commands (sent as COMMAND messages):
    say <text>      {"type": "say", "text": ..., "lang": "es-ES"}
    status <text>   {"type": "status", "text": ...}

Press q in the video window or Ctrl+C in the terminal to quit.

Wire format (see mobile/ios/BlackjackBridge/Networking/BridgeProtocol.swift):
[1 byte type][uint32 BE length][payload]. 0x01 HELLO (JSON), 0x02 FRAME (int64 BE
capture ms + JPEG), 0x03 LOG (UTF-8), 0x10 COMMAND (JSON, PC -> iPhone).
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

HELLO, FRAME, LOG, COMMAND = 0x01, 0x02, 0x03, 0x10
HEADER = struct.Struct(">BI")
MAX_PAYLOAD = 16 * 1024 * 1024


class Receiver:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.client: socket.socket | None = None
        self.latest: tuple[np.ndarray, int, float] | None = None  # image, capture ms, arrival
        self.arrivals: deque[float] = deque(maxlen=60)
        self.frames = 0
        self.hello: dict | None = None

    def send_command(self, command: dict) -> bool:
        payload = json.dumps(command, ensure_ascii=False).encode("utf-8")
        with self.lock:
            client = self.client
        if client is None:
            print("! No iPhone connected")
            return False
        try:
            client.sendall(HEADER.pack(COMMAND, len(payload)) + payload)
            return True
        except OSError as error:
            print(f"! Send failed: {error}")
            return False

    def serve(self, port: int) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", port))
        server.listen(1)
        print(f"Listening on port {port}. Type this in the app: {local_ip()}:{port}")
        while True:
            client, address = server.accept()
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"+ iPhone connected from {address[0]}:{address[1]}")
            with self.lock:
                self.client = client
            try:
                self.read_messages(client)
            except (ConnectionError, OSError) as error:
                print(f"- Connection lost: {error}")
            finally:
                with self.lock:
                    self.client = None
                client.close()
                print("- iPhone disconnected, waiting for it to reconnect")

    def read_messages(self, client: socket.socket) -> None:
        while True:
            kind, length = HEADER.unpack(read_exact(client, HEADER.size))
            if length > MAX_PAYLOAD:
                raise ConnectionError(f"message too large ({length} bytes)")
            payload = read_exact(client, length)
            if kind == HELLO:
                self.hello = json.loads(payload)
                print(f"HELLO {self.hello}")
            elif kind == FRAME:
                self.handle_frame(payload)
            elif kind == LOG:
                print(f"[iPhone] {payload.decode('utf-8', errors='replace')}")
            else:
                print(f"? Unknown message type 0x{kind:02X} ({length} bytes)")

    def handle_frame(self, payload: bytes) -> None:
        (captured_ms,) = struct.unpack(">q", payload[:8])
        image = cv2.imdecode(np.frombuffer(payload[8:], np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print("! Could not decode a JPEG frame")
            return
        now = time.time()
        with self.lock:
            self.latest = (image, captured_ms, now)
            self.arrivals.append(now)
            self.frames += 1

    def stats(self) -> tuple[float, int]:
        with self.lock:
            arrivals = list(self.arrivals)
            frames = self.frames
        recent = [t for t in arrivals if arrivals[-1] - t <= 2] if arrivals else []
        fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 and recent[-1] > recent[0] else 0.0
        return fps, frames


def read_exact(client: socket.socket, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = client.recv(length - len(data))
        if not chunk:
            raise ConnectionError("closed by the iPhone")
        data.extend(chunk)
    return bytes(data)


def local_ip() -> str:
    """LAN address of this PC (no packet is sent, it only picks the outgoing interface)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def read_terminal(receiver: Receiver) -> None:
    for line in sys.stdin:
        verb, _, text = line.strip().partition(" ")
        if verb == "say" and text:
            receiver.send_command({"type": "say", "text": text, "lang": "es-ES"})
        elif verb == "status" and text:
            receiver.send_command({"type": "status", "text": text})
        elif verb:
            print("Commands: say <text> | status <text>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    receiver = Receiver()
    threading.Thread(target=receiver.serve, args=(args.port,), daemon=True).start()
    threading.Thread(target=read_terminal, args=(receiver,), daemon=True).start()

    window = "BJ Bridge receiver"
    shown_at = 0.0
    try:
        while True:
            with receiver.lock:
                latest = receiver.latest
            if latest is not None and latest[2] != shown_at:
                image, captured_ms, shown_at = latest
                fps, frames = receiver.stats()
                latency_ms = shown_at * 1000 - captured_ms
                h, w = image.shape[:2]
                view = image.copy()
                label = f"{w}x{h}  {fps:.1f} fps  {frames} frames  latency {latency_ms:.0f} ms"
                cv2.putText(view, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(view, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(window, view)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
