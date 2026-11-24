# 04 Real-time pipeline

The assistant has to keep up with the 15 fps of the glasses and answer before the player decides.
The design idea is to do only cheap work on every frame and to recognise each card **once**, from
its best views, instead of recognising every card on every frame.

## Per frame and per card

`detection_tracking/pipeline.py` (`CardPipeline`) runs this loop.

**Every frame**

1. The classical detector finds the cards and their corners (`02_card_detection.md`).
2. The tracker gives each card a stable ID across frames.
3. Each card without a name yet has its straightened crop added to a buffer of ten, together with
   its sharpness.

**Once per card, after five frames (a third of a second)**

4. The three sharpest crops are aligned and averaged.
5. The recogniser names the averaged crop (`03_card_recognition.md`). If it is unsure, the card
   stays `?` and is tried again five frames later with fresher crops.

A named card keeps its name for as long as it is tracked, and it is reported once as a new card,
which is the moment the Hi-Lo count should see it. At most two cards are recognised in the same
frame, so a burst of new cards does not stall the video.

## Sharp crops

The glasses move with the head, so many frames are smeared by motion blur, and every frame carries
noise and JPEG blocks. `vision_classic/sharpness.py` scores a crop by the variance of its Laplacian,
which is high for crisp edges and drops fast with blur. The three sharpest crops are aligned (a
crop can come out rotated by 180° from one frame to the next, so each one is turned to match the
sharpest) and averaged, which keeps the detail they share and cancels the noise. On the first
recording, recognising from the first frame named every card but got 8♣, K♥ and 2♦ wrong. With the
fused sharpest crops they came out right, or stayed `?` instead of getting a wrong name.

## The tracker

`detection_tracking/tracker.py` (`CardTracker`) follows the cards between frames.

- One `cv2.KalmanFilter` per card, with state `[cx, cy, w, h, vx, vy]` (constant velocity on the
  centre, constant size).
- The predicted boxes are matched to the detections with the Hungarian algorithm
  (`scipy.optimize.linear_sum_assignment`) on `1 − IoU`, accepting IoU of at least 0.2.
- A card unseen for ten frames is dropped, so a card hidden for a moment by a hand keeps its ID.

The tracker also has a second matching pass by label and distance, meant for head turns that move a
card further than its own size. In the pipeline every detection arrives unnamed (`?`), and that pass
handed the ID and the name of a lost card to a nearby different card. On the recording a dealer
card showed up as "AD". The pipeline therefore matches by IoU only. A card that is lost and found
again is simply recognised again, and the table makes sure it is not counted twice
(`06_game_logic.md`).

## Frame budget

Measured on the laptop CPU (no GPU), with every frame of the recording through the whole pipeline.

| Stream | Time per frame | Of which detection |
|---|---|---|
| 720×1280 | 54 ms (about 18 fps) | 51 ms |
| 504×896 | 42 ms (about 24 fps) | 35 ms |
| 360×640 | 14 ms | 12 ms |

Recognition (63 + 50 ms with the hybrid) only happens on the frames where a card gets its name, and
at most twice per frame. On the stream the receiver always hands the newest frame to the pipeline,
so a slower frame skips old frames instead of building up delay. The video always shows the FPS of
the whole loop, as the assignment requires.

## Why YOLO was not needed for speed

The pretrained YOLO card detector ran at about 4 fps on this CPU with the upscaling it needed. The
classical detector, at 51 ms per 720×1280 frame, is fast enough and needs no GPU.

See also `docs/card_pipeline.md`.
