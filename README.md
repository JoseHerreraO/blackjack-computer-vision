# Blackjack Computer Vision 🃏🕶️

Blackjack assistant built on **classical computer vision**. The player wears **Meta glasses** whose camera captures the table from a first-person view. A phone receives the video from the glasses through Meta's SDK (Wearables Device Access Toolkit) and forwards it over the local network to a computer, where all the analysis happens. The computer recognises the player's and the dealer's cards, recommends the optimal move according to basic strategy (hit, stand, double down or split) and shows it in a HUD over the video, together with the Hi-Lo count, the cards left in the shoe and a decision history.

Before it can be used, the assistant has to be unlocked by showing the camera **four aces in a secret suit order**.

Final project for *Computer Vision I*, 3rd year of Mathematical Engineering, Universidad Pontificia Comillas ICAI (2026/27).

**Authors.** José De Oña · José Herrera (Group A)

> 🚧 **Status.** The repository contains the package structure, tests and CI. The modules are under development (see [Schedule](#schedule)).

---

## Architecture

```
┌──────────────┐   Bluetooth (Meta SDK)    ┌──────────────┐   Local Wi-Fi (frames)  ┌───────────────────────┐
│ Meta glasses │ ────────────────────────► │  Phone app   │ ──────────────────────► │   Computer (Python)   │
│   (camera)   │                           │ Android/iOS  │                         │    Frame receiver     │
└──────────────┘                           └──────────────┘                         └───────────┬───────────┘
                                                                                                │
[Offline calibration] ──► K, distortion coefficients, RMS                                       │
                                   │                                                            ▼
                                   └──────────► Distortion correction (remap) ──► Card detector
                                                                                        │
   ┌──────────────── SECURITY SYSTEM ───────────────────────────────────────────────────┴───┐
   │  Ace suit detector ──► Sequence decoder (4)                                            │
   └───────────────────────────────┬────────────────────────────────────────────────────────┘
                                   │ access granted
   ┌──────────────── PROPOSED SYSTEM ───────────────────────────────────────────────────────┐
   │  Tracker (cards + chips) ──► Rank/suit recognition                                     │
   │  ──► Round logic ──► Basic strategy + Hi-Lo                                            │
   │  ──► Video output (HUD + FPS)                                                          │
   └────────────────────────────────────────────────────────────────────────────────────────┘
```

The phone does no vision processing at all. It only acts as a bridge between the glasses and the computer, which runs the whole pipeline in Python with OpenCV.

| Module | Technique |
|---|---|
| Capture | Meta glasses camera through the Wearables Device Access Toolkit (up to 720p at 30 FPS, limited by Bluetooth bandwidth). The phone app forwards the frames to the computer over the local network. Fallback with a fixed overhead webcam plugged into the computer |
| Calibration | Chessboard photographed with the glasses themselves at the stream resolution. `findChessboardCorners` + `cornerSubPix` + `calibrateCamera` (offline). Precomputed `initUndistortRectifyMap` maps |
| Card detection | Otsu / HSV, morphology, `findContours`, `approxPolyDP` (4 vertices, aspect ratio ≈ 1.4), homography to 200×300 |
| Security | Suit of the ace's central pip by colour (HSV) and shape (Hu moments). Decoder for 4 patterns with reset on error, timeout or key press |
| Recognition | Bag of visual words. SIFT on the rectified card, K-Means vocabulary, histogram + SVM (χ² kernel). Suit colour filter and temporal voting over N frames |
| Tracker | IDs by centroid association + Kalman filter (cards), CamShift (chips) |
| Game | Round state machine and basic strategy tables (hard, soft, pairs) |
| Extensions | Hi-Lo (running and true count, edge, bet sizing), shoe histogram, CSV log, vocabulary/kernel study and BoW vs. template matching comparison |

### What a first-person camera changes

The camera is no longer fixed above the table but on the player's head, so the image moves and its perspective changes. Each card's homography corrects its perspective individually, but the tracker has to tolerate camera motion between frames and not only the motion of the cards. Input resolution and FPS are limited by the glasses stream, which also lowers them automatically when the Bluetooth link is tight. Calibration has to be done with images captured by the glasses at the final stream resolution, because changing the resolution invalidates the intrinsic matrix.

### Fallback with a fixed overhead camera

As a fallback, the system also works with a webcam (or a phone) fixed overhead above the table and plugged directly into the computer. In this mode the glasses → phone bridge is skipped and the rest of the pipeline is identical, so only the frame source changes. It is useful for developing and testing without the glasses, and for the demo if the Bluetooth stream fails or drops in quality. With a fixed camera the tracker has an easier job, because only the cards and chips move. Each camera is calibrated separately and its parameters are stored in `data/calibration/`, with different files for the glasses and for the webcam.

## Repository structure

```
.
├── src/blackjack_vision/
│   ├── capture/              # Frame sources: webcam, video files, image folders, glasses stream
│   ├── calibration/          # Chessboard calibration, camera model and undistortion (Tema 2)
│   ├── detection_tracking/   # Card and hand detection, multi-object tracking (Tema 5)
│   ├── vision_classic/       # Vertex refinement, homography rectification, rank/suit recognition
│   ├── security/             # Ace-sequence unlock: pattern detector + sequence decoder
│   ├── blackjack/            # Cards, hands, basic strategy, Hi-Lo count and round state machine
│   └── ui/                   # HUD on the PC and feedback to the glasses
├── scripts/                  # Command-line entry points (calibration, previews, live assistant)
├── tests/                    # pytest tests (synthetic data, no hardware needed)
├── docs/                     # Technical guides and design notes
├── data/
│   ├── calibration/          # Camera models (JSON) and calibration captures
│   └── samples/              # Test photos and videos (not committed)
├── .github/                  # CI (ruff + pytest) and pull request template
├── pyproject.toml
└── requirements.txt
```

The computer-side frame receiver and the phone app that uses Meta's SDK are still to be added.

## Installation and usage

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (on Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .                # makes blackjack_vision importable from scripts/
```

```bash
python scripts/calibrate.py --source <video|folder|camera> --name glasses_high   # see docs/calibration.md
python scripts/run.py           # real-time assistant
```

Before opening a pull request, run the same checks as the CI.

```bash
ruff check .
pytest
```

## Hardware

Meta glasses supported by the Wearables Device Access Toolkit, an Android or iOS phone running the bridge app, a computer on the same Wi-Fi network, a standard 52-card deck, coloured chips and a plain-coloured table mat. The fallback also needs a webcam (or a phone) with a stand to mount it overhead above the table.

## Schedule

| Dates | Tasks | Status |
|---|---|---|
| 4–7 Nov | Repository, camera capture and proposal | ✅ |
| 8–14 Nov | Package structure and CI. Calibration, card detection and recognition | ⏳ |
| 15–20 Nov | Glasses → phone → computer bridge app. Security system, round logic and basic strategy | ⏳ |
| 21–26 Nov | Tracker, Hi-Lo, shoe, log, HUD and optimisation | ⏳ |
| 27–30 Nov | Report, demo video and poster | ⏳ |
| 2 Dec | Submission and presentation | ⏳ |

## License

[MIT](LICENSE)
