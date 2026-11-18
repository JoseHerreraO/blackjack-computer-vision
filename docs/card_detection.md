# Card detection and tracking

Cards are detected with a YOLO model and followed across frames with our own Kalman tracker.
ByteTrack (built into Ultralytics) is the fallback if the tracker falls short.

```
glasses -> iPhone (BJ Bridge app) --TCP 5050--> BridgeSource -> [Undistorter] -> CardDetector -> CardTracker
```

## 1. Setup

```bash
# CPU build of PyTorch (skip on a PC with an NVIDIA GPU and install the CUDA build instead)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[yolo]"
python scripts/download_card_model.py
```

The weights go to `models/lgd-cards-gen3/` and are not versioned (AGPL-3.0, 19 MB).

## 2. The pretrained model

[`sroot/lgd-cards-gen3`](https://huggingface.co/sroot/lgd-cards-gen3) is YOLO11s fine-tuned at
640 px on the public synthetic dataset
[Roboflow playing-cards-ow27d](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d)
plus two days of real table video. It has 52 classes named like `10H`, `AS` or `QD`
(rank + `C`/`D`/`H`/`S`).

The model boxes the **corner index** of a card (the small rank and suit), not the whole card, so
a card usually gives two boxes with the same label. `merge_corners` joins boxes with the same
label that are close together into one detection per card. Its reported recall is 0.85 on its
authors' own casino video, so it has to be measured on our glasses stream before trusting it.

## 3. Running it

```bash
python scripts/detect_cards.py --source bridge                       # live from the glasses
python scripts/detect_cards.py --source data/samples/<folder> --fps 3  # saved frames
python scripts/detect_cards.py --source bridge --camera data/calibration/glasses_high.json
```

`--source bridge` starts the TCP server the iOS app connects to (see
`src/blackjack_vision/capture/bridge.py` for the protocol). The video shows the FPS, every card
with its track ID and label, grey while unconfirmed and green once confirmed. Each card is printed
once in the terminal when it is confirmed, which is when Hi-Lo should count it.

Useful flags are `--tiles` (default `1x2`, see below), `--conf` (minimum score, 0.4), `--no-merge`
(show the raw corner boxes) and `--save out.mp4`.

## 4. Tiling

The glasses stream is 720×1280 and the model runs at 640 px. On the whole frame every corner
index would be shrunk to half its size, so by default the frame is cut into two overlapping
720×~690 tiles (`--tiles 1x2`), which YOLO sees at nearly native resolution. It costs two
inferences per frame.

Measured on the team laptop's CPU (PyTorch CPU, no GPU, 2026-11-18):

| Tiles | Time per frame | FPS |
|-------|----------------|-----|
| `1x1` | 130 ms         | 7.7 |
| `1x2` | 390 ms         | 2.6 |

Both are below the 15 fps of the stream. Options, from cheapest to most work: export the model to
OpenVINO (`yolo export model=models/lgd-cards-gen3/best.pt format=openvino`, usually 2-3× faster
on Intel CPUs), detect only every few frames and let the Kalman tracker predict in between, or
fine-tune a smaller YOLO11n.

## 5. Tracker

`CardTracker` (`src/blackjack_vision/detection_tracking/tracker.py`):

- one `cv2.KalmanFilter` per card, state `[cx, cy, w, h, vx, vy]` (constant velocity);
- Hungarian matching (`scipy.optimize.linear_sum_assignment`) on `1 - IoU` between predicted
  boxes and detections, with `min_iou = 0.2`;
- a track is confirmed after `min_hits = 3` detections and dropped after `max_missed = 10`
  frames without one (about 0.7 s at 15 fps), so a card hidden by a hand keeps its ID;
- the label is a confidence-weighted vote over the last 15 detections.

## 6. Next steps

1. Record real cards with the glasses at playing distance
   (`mobile/ios/tools/pc_receiver.py --every 5`) and check the pretrained model on them.
2. Label the frames where it fails (for example with Roboflow or CVAT, YOLO format) and fine-tune
   from `best.pt` on our deck and our table.
3. Measure precision, recall and FPS on a held-out recording for the report, with and without
   tiling and calibration.
