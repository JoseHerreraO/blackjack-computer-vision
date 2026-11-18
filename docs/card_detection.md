# Card detection and tracking

Cards are detected with a YOLO model and followed across frames with our own Kalman tracker.
ByteTrack (built into Ultralytics) is the fallback if the tracker falls short.

```
glasses -> iPhone (BJ Bridge app) --TCP 5050--> BridgeSource -> [Undistorter] -> YoloCardDetector -> CardTracker
```

## 1. Setup

```bash
# CPU build of PyTorch (skip on a PC with an NVIDIA GPU and install the CUDA build instead)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[yolo]"
python scripts/download_card_model.py
```

The weights go to `models/lgd-cards-gen3/` and are not versioned (AGPL-3.0, 19 MB). The script
downloads a pinned commit of the model repo and checks the SHA-256 of `best.pt`, because a `.pt`
is a pickle that runs code when it is loaded.

## 2. The pretrained model

[`sroot/lgd-cards-gen3`](https://huggingface.co/sroot/lgd-cards-gen3) is YOLO11s fine-tuned at
640 px on the public synthetic dataset
[Roboflow playing-cards-ow27d](https://universe.roboflow.com/augmented-startups/playing-cards-ow27d)
plus two days of real table video. It has 52 classes named like `10H`, `AS` or `QD`
(rank + `C`/`D`/`H`/`S`).

The model boxes the **corner index** of a card (the small rank and suit), not the whole card, so
a card usually gives two boxes with the same label. `merge_corners` pairs them **two by two**
with an optimal matching among the boxes of each label: two boxes can be one card when their
distance is about 2.6 corner-box diagonals (median 2.4 on our footage), each box is used once,
and a box without a partner stays a one-corner card. Two cards with the same value next to each
other (several decks in the shoe) therefore stay two cards.
Its reported recall is 0.85 on its
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
  boxes and detections, with `min_iou = 0.2`, then a second pass for what is left that matches
  the same label within `max_center_distance = 2` box diagonals, so a head turn that moves a card
  further than its size between two detections does not create a new track (and a new count);
- a track is confirmed after `min_hits = 3` detections and dropped after `max_missed = 10`
  frames without one (about 0.7 s at 15 fps), so a card hidden by a hand keeps its ID;
- the label is a confidence-weighted vote over the last 15 detections.

## 6. First test on real footage (2026-11-18)

448 frames recorded through the bridge with the glasses at playing distance, cards dealt on a
green mat. Over the stream (2.4 GHz Windows hotspot) the SDK dropped to 360×640, which is what we
use now (on 5 GHz it is stable). Every 4th frame was evaluated (112 frames) on the laptop CPU,
with `--conf 0.3` and the raw corner boxes:

| Setting                  | Frames with detections | Time per frame |
|--------------------------|------------------------|----------------|
| whole frame, `imgsz 640` | 0 / 112                | 59 ms          |
| whole frame, `imgsz 1280`| 70 / 112               | 225 ms         |
| `--tiles 1x2`, 640       | 70 / 112               | 211 ms         |

At 360×640 the corner indices are too small at the model's native size and the frame has to be
upscaled about 2× (`--imgsz 1280`, or tiles). Most frames without detections are from the start
of the clip, before any card is on the mat. The ace of diamonds and the two of spades are read
correctly with scores of 0.5-0.8. The king of hearts is often misread as `JH` or `2H`, and
sometimes only one corner of a card is found. The face cards are where fine-tuning on our deck
should help most.

## 7. Next steps

1. Record real cards with the glasses at playing distance
   (`mobile/ios/tools/pc_receiver.py --every 5`) and check the pretrained model on them.
2. Label the frames where it fails (for example with Roboflow or CVAT, YOLO format) and fine-tune
   from `best.pt` on our deck and our table.
3. Measure precision, recall and FPS on a held-out recording for the report, with and without
   tiling and calibration.
