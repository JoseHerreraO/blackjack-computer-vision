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
                raise ConnectionError(f"message too large ({length} bytes), is this the BJ Bridge app?")
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


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--out", type=Path, help="folder for saved frames (default data/samples/<date-time>)")
    parser.add_argument("--every", type=int, default=1, help="while recording, keep 1 frame out of N (1)")
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
                fps, frames = receiver.stats()
                latency_ms = shown_at * 1000 - captured_ms
                h, w = image.shape[:2]
                view = image.copy()
                draw_label(view, f"{w}x{h}  {fps:.1f} fps  latency {latency_ms:.0f} ms", (10, 30), (0, 255, 0))
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
