"""Live Blackjack assistant: glasses -> PC -> advice spoken through the glasses.

1. **Locked** until four aces are shown one at a time in the secret suit order
   (``security/aces.py``, secret from ``BJ_SECRET`` or ``--secret``, never drawn).
2. **Unlocked**: every frame goes through the card pipeline (classical detector, tracker and one
   fused recognition per card). Each newly recognised card is dealt to the dealer or the player by
   its zone (``--dealer-line``), the round follows the cards, and the basic-strategy advice is
   spoken through the glasses ("Pide carta", "Plántate"...) and drawn on the HUD with the Hi-Lo
   count and the suggested bet. The FPS are always drawn on the video.

Examples::

    set BJ_SECRET=SHDC
    python scripts/run.py --source bridge --model data/models/recognizer.joblib   # hybrid recogniser
    python scripts/run.py --source bridge --recognizer embedding --no-lock
    python scripts/run.py --source bridge --model data/models/recognizer.joblib --web --no-window
    python scripts/run.py --source data/samples/<folder> --fps 15 --no-lock --recognizer embedding \
        --no-window --save outputs/demo.mp4

With ``--source bridge`` the iOS app connects to this PC and the advice is spoken by the iPhone
through the glasses. With other sources it is only logged.

``--web`` also serves a dashboard on http://localhost:8000 (``ui/dashboard.py``): the video, the
cards as cards, the advice, the count, the shoe and the voice log, with buttons for the commands
below. With ``--no-window`` the browser is the only screen.

Keys: q / Esc = quit, space = pause, r = new round, n = new shoe (count to 0), l = lock again,
s = snapshot.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from blackjack_vision.blackjack.cards import Card
from blackjack_vision.blackjack.table import Table
from blackjack_vision.capture.bridge import BridgeSource
from blackjack_vision.capture.sources import FrameSource, open_source, write_image
from blackjack_vision.detection_tracking.card_detector import (
    VIEWS,
    CardDetector,
    config_for_view,
    default_view,
)
from blackjack_vision.detection_tracking.pipeline import CardPipeline
from blackjack_vision.security.aces import (
    AceReader,
    DecoderEvent,
    SecuritySystem,
    SequenceDecoder,
    parse_secret,
)
from blackjack_vision.ui.dashboard import Dashboard
from blackjack_vision.ui.hud import draw_hud
from blackjack_vision.vision_classic.mat import mat_far_edge, mat_in_view
from blackjack_vision.vision_classic.rectify import rectify_card
from blackjack_vision.vision_dl.hybrid_recognizer import RECOGNIZERS, load_recognizer

WINDOW = "Blackjack assistant"
PREVIEW_MAX_HEIGHT = 760

logger = logging.getLogger("run")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", default="bridge", help="bridge[:port], camera, video or images")
    parser.add_argument("--fps", type=float, help="nominal rate of an image folder")
    parser.add_argument("--view", choices=VIEWS, help="camera view (glasses for bridge sources)")
    parser.add_argument(
        "--recognizer",
        choices=RECOGNIZERS,
        default="hybrid",
        help="hybrid: classical rank + embedding suit (default); classic; embedding",
    )
    parser.add_argument("--model", type=Path, default=Path("data/models/recognizer.joblib"))
    parser.add_argument("--min-confidence", type=float, help="classic: rank confidence threshold")
    parser.add_argument(
        "--embedding-model", type=Path, default=Path("models/irfan-playing-card-detector")
    )
    parser.add_argument("--threshold", type=float, default=0.70, help="embedding: min similarity")
    parser.add_argument("--secret", default=os.environ.get("BJ_SECRET"), help="e.g. SHDC")
    parser.add_argument("--no-lock", action="store_true", help="skip the four-aces lock")
    parser.add_argument("--use-rank", action="store_true", help="aces decided by the recogniser")
    parser.add_argument("--decks", type=int, default=6, help="decks in the shoe (6)")
    parser.add_argument(
        "--dealer-line", type=float, default=0.5, help="dealer/player split, fraction of height"
    )
    parser.add_argument("--no-voice", action="store_true", help="do not send advice to the glasses")
    parser.add_argument("--save", type=Path, help="write the annotated video to this file")
    parser.add_argument("--snapshots", type=Path, default=Path("outputs/run"))
    parser.add_argument("--no-window", action="store_true", help="run without a preview window")
    parser.add_argument(
        "--web",
        type=int,
        nargs="?",
        const=8000,
        metavar="PORT",
        help="web dashboard on http://localhost:PORT (8000 if no port is given)",
    )
    parser.add_argument(
        "--web-host",
        default="127.0.0.1",
        help="dashboard address (127.0.0.1: only this PC; 0.0.0.0 lets anyone on the network "
        "watch it and press its buttons)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


@dataclass
class LockCard:
    """A detected card while locked (only drawn, not tracked)."""

    track_id: int
    corners: np.ndarray
    card: Card | None = None


class Feedback:
    """Speech and status for the player: through the glasses with the bridge, logged otherwise."""

    def __init__(
        self, source: FrameSource, enabled: bool, dashboard: Dashboard | None = None
    ) -> None:
        self.bridge = source if isinstance(source, BridgeSource) and enabled else None
        self.dashboard = dashboard
        self._last_status: str | None = None

    def say(self, text: str) -> None:
        logger.info("SAY: %s", text)
        if self.dashboard is not None:
            self.dashboard.log(text, "say")
        if self.bridge is not None and not self.bridge.say(text):
            logger.debug("iPhone not connected, advice not spoken")

    def status(self, text: str) -> None:
        if text == self._last_status:
            return
        self._last_status = text
        if self.bridge is not None:
            self.bridge.status(text)


def build_recognizer(args: argparse.Namespace):
    return load_recognizer(
        args.recognizer,
        model=args.model,
        embedding_model=args.embedding_model,
        threshold=args.threshold,
        min_confidence=args.min_confidence,
    )


def build_security(args: argparse.Namespace) -> tuple[SecuritySystem, object] | None:
    """Lock system, or ``None`` with ``--no-lock``. Needs the classic model's suit templates."""
    if args.no_lock:
        return None
    if not args.secret:
        raise SystemExit("Give the secret with BJ_SECRET or --secret, or run with --no-lock.")
    from blackjack_vision.vision_classic.recognition import CardRecognizer

    if not args.model.is_file():
        raise SystemExit(
            f"The lock reads the ace suits with the classic model's templates: {args.model} "
            "not found. Train it with scripts/train_recognizer.py or run with --no-lock."
        )
    classic = CardRecognizer.load(args.model)
    system = SecuritySystem(
        AceReader(classic.suit_templates, classic.config.index_region),
        SequenceDecoder(parse_secret(args.secret)),
    )
    return system, classic


def lock_text(system: SecuritySystem) -> str:
    """Progress without revealing the secret, e.g. ``LOCKED  ##..``."""
    decoder = system.decoder
    if decoder.unlocked:
        return "UNLOCKED"
    return "LOCKED  " + "#" * decoder.progress + "." * (len(decoder.secret) - decoder.progress)


CLEAR_AFTER_UNLOCK = 10
"""Frames without cards needed after the unlock before the game starts."""

KEYS = {
    ord("q"): "quit",
    27: "quit",
    ord(" "): "pause",
    ord("r"): "new_round",
    ord("n"): "new_shoe",
    ord("l"): "lock",
    ord("s"): "snapshot",
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    view_name = args.view or default_view(args.source)
    recognizer = build_recognizer(args)
    security = build_security(args)
    pipeline = CardPipeline(recognizer, detector=CardDetector(config_for_view(view_name)))
    table = Table(decks=args.decks, dealer_line=args.dealer_line)
    dashboard = Dashboard(args.web_host, args.web).start() if args.web is not None else None

    frame_times: deque[float] = deque(maxlen=30)
    writer: cv2.VideoWriter | None = None
    paused = False
    hud = None
    fps = 0.0
    locked = security is not None
    # After the unlock the last ace is still on the table: the game starts once the table has
    # been seen empty for CLEAR_AFTER_UNLOCK frames, so the aces of the lock are not dealt.
    clearing = False
    empty_frames = 0

    def run_command(command: str) -> bool:
        """Keys and dashboard buttons. Returns ``False`` to quit."""
        nonlocal paused
        if command == "quit":
            return False
        if command == "pause":
            paused = not paused
        elif command == "new_round":
            table.reset()
            pipeline.reset()
            logger.info("New round (key)")
        elif command == "new_shoe":
            table.reset(reshuffle=True)
            pipeline.reset()
            logger.info("New shoe: count back to 0")
        elif command == "lock" and security is not None:
            security[0].decoder.reset()
            logger.info("Locked again")
        elif command == "snapshot" and hud is not None:
            path = write_image(args.snapshots / f"run_{int(time.time() * 1000)}.png", hud)
            logger.info("Saved %s", path)
        logged = (
            ("new_round", "new_shoe", "lock") if security is not None else ("new_round", "new_shoe")
        )
        if dashboard is not None and command in logged:
            label = {"new_round": "Nueva ronda", "new_shoe": "Nuevo sabot", "lock": "Bloqueado"}
            dashboard.log(label[command])
        return True

    try:
        with open_source(args.source, fps=args.fps) as source:
            feedback = Feedback(source, enabled=not args.no_voice, dashboard=dashboard)
            if isinstance(source, BridgeSource):
                logger.info("Waiting for the iPhone on port %d", source.port)
            running = True
            while running:
                if not paused or hud is None:
                    frame = source.read()
                    if frame is None:
                        break
                    image = frame.image
                    frame_times.append(time.perf_counter())
                    fps = (
                        (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                        if len(frame_times) > 1
                        else 0.0
                    )

                    locked = security is not None and not security[0].unlocked
                    if locked:
                        system, classic = security
                        detected = pipeline.detector(image).cards
                        crops = [rectify_card(image, card.corners) for card in detected]
                        ranks = None
                        if args.use_rank and crops:
                            ranks = [p.rank for p in classic.predict_many(crops)]
                        event, _ = system.update(crops, frame.timestamp, ranks)
                        if event is DecoderEvent.UNLOCKED:
                            feedback.say("Desbloqueado")
                            logger.info("Unlocked: waiting for the table to be cleared")
                            clearing, empty_frames = True, 0
                        elif event in (DecoderEvent.WRONG, DecoderEvent.TIMEOUT):
                            logger.info("Lock: %s (%s)", event.value, lock_text(system))
                        cards = [LockCard(i, card.corners) for i, card in enumerate(detected)]
                        text = lock_text(system)
                        hud = draw_hud(image, cards, fps, lock_text=text, locked=True)
                        web_view = hud
                    elif clearing:
                        detected = pipeline.detector(image).cards
                        empty_frames = 0 if detected else empty_frames + 1
                        if empty_frames >= CLEAR_AFTER_UNLOCK:
                            clearing = False
                            table.reset()
                            pipeline.reset()
                            logger.info("Table cleared: the game starts")
                        cards = [LockCard(i, card.corners) for i, card in enumerate(detected)]
                        hud = draw_hud(image, cards, fps, lock_text="UNLOCKED  retira las cartas")
                        web_view = hud
                    else:
                        result = pipeline.process(image)
                        # Only needed when no card is seen: empty table, or head turned away?
                        on_mat = mat_in_view(image) if not result.cards else None
                        # Zones from the far edge of the mat, which moves with the head.
                        edge = mat_far_edge(image) if result.new_cards or result.cards else None
                        update = table.update(
                            result.cards,
                            result.new_cards,
                            image.shape[0],
                            mat_in_view=on_mat,
                            motion=result.motion,
                            mat_edge=edge,
                            image_width=image.shape[1],
                        )
                        for event in update.events:
                            logger.info(event)
                            if dashboard is not None:
                                dashboard.log(event)
                        for phrase in update.speech:
                            feedback.say(phrase)
                        feedback.status(
                            f"RC {table.shoe.running_count:+d} TC {table.shoe.true_count:+.1f}"
                            f" bet {table.shoe.bet_units()}u"
                        )
                        text = lock_text(security[0]) if security is not None else None
                        hud = draw_hud(image, result.cards, fps, table=table, lock_text=text)
                        web_view = hud
                        if dashboard is not None and dashboard.watching:
                            # The page shows the hands and the count itself.
                            web_view = draw_hud(
                                image, result.cards, fps, table=table, lock_text=text, panel=False
                            )

                    if dashboard is not None:
                        dashboard.publish_frame(web_view)
                    if args.save is not None:
                        if writer is None:
                            args.save.parent.mkdir(parents=True, exist_ok=True)
                            rate = source.fps or 15.0
                            size = (hud.shape[1], hud.shape[0])
                            writer = cv2.VideoWriter(
                                str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), rate, size
                            )
                        writer.write(hud)

                if dashboard is not None:
                    decoder = security[0].decoder if security is not None else None
                    dashboard.publish_state(
                        table,
                        fps=fps,
                        locked=locked,
                        lock_progress=decoder.progress if decoder else 0,
                        lock_length=len(decoder.secret) if decoder else 0,
                        paused=paused,
                        source=str(args.source),
                    )
                    for command in dashboard.commands():
                        running = run_command(command) and running

                if args.no_window:
                    if paused:
                        time.sleep(0.05)  # nothing to read: wait for the dashboard
                    continue
                scale = min(1.0, PREVIEW_MAX_HEIGHT / hud.shape[0])
                cv2.imshow(WINDOW, cv2.resize(hud, None, fx=scale, fy=scale) if scale < 1 else hud)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key in KEYS:
                    running = run_command(KEYS[key]) and running
    except KeyboardInterrupt:
        logger.info("Stopped (Ctrl+C)")
    finally:
        # First of all, also on Ctrl+C: an MP4 that is not closed has no index and cannot be
        # played, so nothing that may block or be interrupted (the dashboard) goes before it.
        if writer is not None:
            writer.release()
            logger.info("Video saved to %s", args.save)
        if dashboard is not None:
            dashboard.close()
        cv2.destroyAllWindows()
    logger.info(
        "Rounds: %d  %s", table.rounds_played + 1, table.shoe.summary() if table.shoe else ""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
