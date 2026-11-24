# 07 Starting a game

Everything needed to go from a switched-off table to the assistant giving advice, in order.

## One-time setup on the PC

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e ".[yolo]"                     # makes blackjack_vision importable and adds PyTorch
python scripts/download_card_recognizer.py   # embedding recogniser, pinned and SHA-256 checked
```

The classical model is trained from the card recording (`03_card_recognition.md`).

```bash
python scripts/capture_cards.py --source bridge --deck --auto-every 5     # the 52 cards
python scripts/train_recognizer.py --data data/cards --out data/models/recognizer.joblib
```

## One-time setup on the iPhone

Install the latest `BlackjackBridge-<run>.ipa` from GitHub Actions with Sideloadly (it has to be
installed again every 7 days), enable Developer Mode on the iPhone and in Meta AI, and approve BJ
Bridge in Meta AI (`01_phone_connection.md`).

## Before every session

1. **Windows mobile hotspot** on, **5 GHz band** (Settings, Network and Internet, Mobile hotspot,
   Edit). Connect the iPhone to it.
2. Good, even light over the mat, and the deck ready.
3. Choose the secret for the lock, for example `set BJ_SECRET=SHDC`.

## Starting the assistant

```bash
python scripts/run.py --source bridge
```

The default recogniser is the hybrid one and the default view is the glasses. Useful options are
`--no-lock` to skip the aces, `--recognizer classic` or `embedding` to compare, `--dealer-line 0.5`
to move the line between the zones, and `--save outputs/demo.mp4` to record the video with the HUD.

Then, in the BJ Bridge app with the glasses on:

1. Type `192.168.137.1:5050` and tap **Connect** (the status must say *connected*).
2. Choose **720×1280 @ 15**.
3. Tap **Start session**, then **Start stream**.

## Playing

1. **Unlock.** Show the four aces one at a time in the secret order, taking each away before the
   next one. The HUD goes from `LOCKED ....` to `UNLOCKED` and the glasses say "Desbloqueado".
2. **Sit as when playing.** The dealer's cards go on the far half of the image and the player's on
   the near half (the dashed line on the HUD).
3. **Deal.** Each card shows `?` for a third of a second and then its name. With two player cards
   and the dealer's up card the advice appears in large letters and is spoken through the glasses.
4. **Play the hand.** Each new card updates the total and the advice. The dealer turning over the
   hole card ends the player's turn, and the result is spoken when the round ends.
5. **Next round.** Collect the cards. With the mat empty in view a new round starts, and the Hi-Lo
   count carries over.

## Keys

| Key | Action |
|---|---|
| `q` / Esc | Quit |
| space | Pause |
| `r` | New round |
| `n` | New shoe (count back to 0) |
| `l` | Lock again |
| `s` | Snapshot |

## If something goes wrong

| Symptom | What to do |
|---|---|
| The app says `PC not reachable` | The iPhone is not on the PC's hotspot, or the address is wrong |
| The stream drops to 504×896 or 360×640 | The hotspot is on 2.4 GHz. Switch it to 5 GHz |
| Cards are not outlined | Too much tilt of the head, or cards touching each other (they come back as one cluster) |
| A card stays `?` | Blur or glare. Hold the head still for a moment |
| The voice is heard from the phone | Pick the glasses as the audio output on the iPhone |
