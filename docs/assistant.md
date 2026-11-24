# Live assistant (`scripts/run.py`)

```
glasses -> iPhone -> BridgeSource -> [LOCKED: four aces] -> CardPipeline -> Table -> Round
                                                                              |-> say() advice to the glasses
                                                                              |-> HUD (FPS, hands, advice, count)
```

```bash
set BJ_SECRET=SHDC
python scripts/run.py --source bridge --model data/models/recognizer.joblib
python scripts/run.py --source bridge --recognizer embedding --no-lock          # without the lock
python scripts/run.py --source data/samples/<folder> --fps 15 --no-lock --recognizer embedding \
    --no-window --save outputs/demo.mp4                                          # on a recording
```

Keys: `q` quit, space pause, `r` new round, `n` new shoe (count to 0), `l` lock again,
`s` snapshot.

## What it does

1. **Locked** until four aces are shown one at a time in the secret order
   (`security/aces.py`, #21). The HUD shows only the progress (`LOCKED  ##..`). The lock reads
   the suits with the classic model's templates, so it needs `--model` (or `--no-lock`).
2. **Unlocked**: `CardPipeline` (#23) detects, tracks and recognises each card once.
   `blackjack/table.py` deals every newly recognised card to the dealer or the player by its zone,
   and `Round` (#20) follows the round and gives the basic-strategy advice.
3. **Feedback**: the advice is spoken through the glasses when it changes ("Pide carta",
   "Plántate"...), and so is the result when the round ends ("Ganas", "Pierdes"...). The count goes
   to the app as a `status` line. With sources other than the bridge everything is only logged.
4. **HUD** (`ui/hud.py`): FPS (always on), lock state, the dashed line between the zones, every
   card in its zone colour (`?` until named, `HOLE` for the dealer's face-down card), both hands
   with totals, the advice and the Hi-Lo count with the suggested bet.

## Table rules it infers (`blackjack/table.py`)

- **Zones**: cards above `--dealer-line` (fraction of the image height, 0.5 by default) are the
  dealer's. Sit so that the dealer's cards are on the far half of the image.
- **Hole card**: a dealer card that stays unrecognised for 15 frames while the dealer shows one
  card is the face-down card. It is counted once it is turned over.
- **New round**: a card after the round has ended, or 15 frames without cards when that cannot
  lose a hand, either because the round is over or because the mat is in view and empty (the
  cards were collected). The mat is found by its colour in HSV (`vision_classic/mat.py`: teal,
  hue about 87). With the head turned away the mat is not in view, so the round waits for the
  cards to come back. The count carries over. A new shoe is the `n` key.
- **Cards found again**: a card hidden for a moment comes back with a new track and is named again.
  If the same code was already dealt to the same zone in this round, its old track is gone and it
  reappears within 1.5 card sizes of where it was last seen, it is not dealt or counted twice, and
  the dealt card takes over the new track, so a later equal card counts. Two equal cards in view at
  once are two cards.

## Web dashboard (`--web`)

`--web` serves a page on http://localhost:8000 (`ui/dashboard.py`, Python standard library only)
with what the assistant knows, laid out for the demo and the video:

- the glasses video with the detections and the FPS (MJPEG, only encoded while the page is open);
- the dealer's and the player's cards drawn as cards, with totals and the hole card face down;
- the advice in large letters (or the result of the round);
- the Hi-Lo count: running and true count, suggested bet, player edge and shoe penetration;
- what is left in the shoe by value against the average (red: good for the player);
- the log of what the glasses said and of the table events;
- buttons for new round, new shoe, pause and lock (the same as the keys);
- while locked, a lock screen with how many aces of the secret have been shown, never which.

```powershell
python scripts/run.py --source bridge --model data/models/recognizer.joblib --web --no-window
python scripts/run.py --source bridge --web 8080      # another port; the OpenCV window stays
python scripts/dashboard_demo.py                      # simulated rounds, no glasses or models
```

The server listens on `127.0.0.1` only. `--web-host 0.0.0.0` opens it to the local network (for
example to watch it from a phone); the page cannot stop the assistant or read the secret.
`scripts/dashboard_demo.py` deals rounds from a shuffled shoe on a drawn table through the real
`Table`, so the page can be tried and recorded without the glasses.

## Known limits

- **Touching or overlapping cards** are one blob for the classical detector, which reports them as
  a cluster. They are not recognised, and they can also pull the foreshortening estimate. On the
  first recording the 2♠ and the J♥ touched and were lost together. Splitting clusters into cards
  is the main thing left for real hands, which often overlap.
- The camera cannot see splits or doubles, so the player plays a single hand, and the third card
  after "Dobla" counts as a hit.
- Tested end to end on the 2026-11-18 recordings (`--no-lock`, embedding recogniser): cards are
  dealt by zone, the advice is spoken and the count follows. The real test is the
  52-card session and a live round with the lock.
