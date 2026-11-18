"""Test receiver for the BJ Bridge iOS app.

A minimal PC-side TCP server for checking the iPhone app without the full pipeline.
It shows the glasses' frames in a window with the receive rate and latency, prints
HELLO and LOG messages, sends commands typed in the terminal back to the iPhone and
saves frames to disk for calibration and for building the card dataset.

    python mobile/ios/tools/pc_receiver.py [--port 5050] [--out DIR] [--every N]

Keys (with the video window focused):
    s   save the current frame
    r   start or stop recording every N-th frame (all of them by default)
    q   quit

Frames are saved as the JPEG bytes received from the iPhone, without re-encoding, named
<sequence>_<capture ms>.jpg. They go to --out, by default data/samples/<date-time>/, which
git ignores. For the glasses calibration record into data/calibration/glasses_high/ and
pass that folder to scripts/calibrate.py.

Terminal commands (sent as COMMAND messages):
    say <text>      {"type": "say", "text": ..., "lang": "es-ES"}
    status <text>   {"type": "status", "text": ...}

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
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HELLO, FRAME, LOG, COMMAND = 0x01, 0x02, 0x03, 0x10
HEADER = struct.Struct(">BI")
MAX_PAYLOAD = 16 * 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parents[3]


class Recorder:
    """Writes received JPEG frames to a folder, untouched."""

    def __init__(self, out_dir: Path, every: int) -> None:
        self.out_dir = out_dir
        self.every = max(1, every)
        self.recording = False
        self.saved = 0
        self._seen = 0
        self._sequence = 0
        self._snapshot_requested = False
        self._lock = threading.Lock()

    def toggle(self) -> None:
        with self._lock:
            self.recording = not self.recording
            self._seen = 0
            state = "started" if self.recording else "stopped"
        print(f"Recording {state} ({self.saved} frames saved in {self.out_dir})")

    def request_snapshot(self) -> None:
        with self._lock:
            self._snapshot_requested = True

    def offer(self, jpeg: bytes, captured_ms: int) -> None:
        """Called for every received frame (network thread)."""
        with self._lock:
            take = self._snapshot_requested
            self._snapshot_requested = False
            if self.recording:
                take = take or self._seen % self.every == 0
                self._seen += 1
            if not take:
                return
            self._sequence += 1
            name = f"{self._sequence:05d}_{captured_ms}.jpg"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / name).write_bytes(jpeg)
        with self._lock:
            self.saved += 1
            if not self.recording:
                print(f"Saved {self.out_dir / name}")


class Receiver:
    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder
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
        server.listen(4)
        addresses = ", ".join(f"{ip}:{port}" for ip in local_ips())
        print(f"Listening on port {port}. Type one of these in the app: {addresses}")
        print(
            "(192.168.137.1 is the Windows mobile hotspot, the one to use on the university Wi-Fi)"
        )
        while True:
            client, address = server.accept()
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            enable_keepalive(client)
            # If the iPhone vanished without closing (Wi-Fi lost, app suspended), the old
            # connection never reports an error. A new connection means the app reconnected,
            # so it replaces the old one instead of waiting behind it.
            with self.lock:
                previous, self.client = self.client, client
            if previous is not None:
                print("- A new connection replaces the previous one")
                close_quietly(previous)
            threading.Thread(target=self.handle_client, args=(client, address), daemon=True).start()

    def handle_client(self, client: socket.socket, address: tuple[str, int]) -> None:
        print(f"+ iPhone connected from {address[0]}:{address[1]}")
        try:
            self.read_messages(client)
        except (ConnectionError, OSError) as error:
            print(f"- Connection lost: {error}")
        finally:
            with self.lock:
                if self.client is client:
                    self.client = None
            close_quietly(client)
            print("- iPhone disconnected, waiting for it to reconnect")

    def read_messages(self, client: socket.socket) -> None:
        while True:
            kind, length = HEADER.unpack(read_exact(client, HEADER.size))
            if length > MAX_PAYLOAD:
                raise ConnectionError(
                    f"message too large ({length} bytes), is this the BJ Bridge app?"
                )
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
        jpeg = payload[8:]
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print("! Could not decode a JPEG frame")
            return
        self.recorder.offer(jpeg, captured_ms)
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
        fps = (
            (len(recent) - 1) / (recent[-1] - recent[0])
            if len(recent) > 1 and recent[-1] > recent[0]
            else 0.0
        )
        return fps, frames


def read_exact(client: socket.socket, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = client.recv(length - len(data))
        if not chunk:
            raise ConnectionError("closed by the iPhone")
        data.extend(chunk)
    return bytes(data)


def enable_keepalive(sock: socket.socket) -> None:
    """Detect a vanished iPhone in about 15 s instead of waiting forever."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    if hasattr(socket, "SIO_KEEPALIVE_VALS"):  # Windows: on, 5 s idle, 1 s between probes
        sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 5000, 1000))
    elif hasattr(socket, "TCP_KEEPIDLE"):  # Linux
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 5)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 10)


def close_quietly(sock: socket.socket) -> None:
    """Close a socket and wake up any thread blocked reading it."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


def local_ips() -> list[str]:
    """IPv4 addresses of this PC, the outgoing interface first."""
    found: list[str] = []
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))  # picks the outgoing interface, sends nothing
        found.append(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in found and not ip.startswith(("127.", "169.254.")):
                found.append(ip)
    except OSError:
        pass
    return found or ["127.0.0.1"]


def read_terminal(receiver: Receiver) -> None:
    for line in sys.stdin:
        verb, _, text = line.strip().partition(" ")
        if verb == "say" and text:
            receiver.send_command({"type": "say", "text": text, "lang": "es-ES"})
        elif verb == "status" and text:
            receiver.send_command({"type": "status", "text": text})
        elif verb:
            print("Commands: say <text> | status <text>")


def draw_label(
    image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument(
        "--out", type=Path, help="folder for saved frames (default data/samples/<date-time>)"
    )
    parser.add_argument(
        "--every", type=int, default=1, help="while recording, keep 1 frame out of N (1)"
    )
    parser.add_argument(
        "--max-height", type=int, default=760, help="preview height limit in pixels (760)"
    )
    args = parser.parse_args()

    out_dir = args.out or REPO_ROOT / "data" / "samples" / datetime.now().strftime("%Y%m%d-%H%M%S")
    recorder = Recorder(out_dir, args.every)
    receiver = Receiver(recorder)
    threading.Thread(target=receiver.serve, args=(args.port,), daemon=True).start()
    threading.Thread(target=read_terminal, args=(receiver,), daemon=True).start()
    print(f"Frames will be saved in {out_dir}. Keys: s = save frame, r = record, q = quit")

    window = "BJ Bridge receiver"
    shown_at = 0.0
    try:
        while True:
            with receiver.lock:
                latest = receiver.latest
            if latest is not None and latest[2] != shown_at:
                image, captured_ms, shown_at = latest
                fps, _ = receiver.stats()
                latency_ms = shown_at * 1000 - captured_ms
                h, w = image.shape[:2]
                # Shrink the preview to fit the screen (720x1280 is taller than most laptop
                # screens). Saved frames keep the full resolution.
                scale = min(1.0, args.max_height / h)
                view = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
                draw_label(
                    view,
                    f"{w}x{h}  {fps:.1f} fps  latency {latency_ms:.0f} ms",
                    (10, 30),
                    (0, 255, 0),
                )
                if recorder.recording:
                    draw_label(view, f"REC {recorder.saved}", (10, 60), (0, 0, 255))
                cv2.imshow(window, view)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                recorder.request_snapshot()
            elif key == ord("r"):
                recorder.toggle()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
