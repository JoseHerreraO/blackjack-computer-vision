# 09 Results

Every number measured on real glasses footage, with its source, to build the Results section of the
report. Hardware is a laptop with no GPU (PyTorch on the CPU). Recordings are in `data/samples/`
(not committed).

## Recordings

| Date | Content | Frames |
|---|---|---|
| 18/11 | First hand on the mat, 360×640 (2.4 GHz hotspot) | 448 |
| 18/11 | 12 to 13 cards on the mat at 360×640, 720×1280 and 504×896 | 1,245 |
| 23/11 | The 52 cards one by one, 720×1280 (`capture_cards.py`) | 2,129 crops, 2,091 after cleaning |

## Stream

| Measurement | Value |
|---|---|
| Frame rate at 720×1280 (asked 15 or 24 fps) | 15.0 fps, 0 frames lost in 30 s |
| 720×1280 on the 5 GHz hotspot | steady 15.0 fps for 28 s |
| 2.4 GHz hotspot | the SDK drops to 504×896 and then 360×640, and the Wi-Fi link breaks |
| Field of view across resolutions | same (scale 0.98 to 0.99 by feature matching) |

## Card detection (classical)

| Stream | Cards per frame, 12 on the mat, before / after the foreshortening correction |
|---|---|
| 720×1280 | 5.6 / **11.8** |
| 504×896 | 5.7 / **11.8** |
| 360×640 | 3.7 / 5.7 |

| Single card, fresh detector (capture case) | Found |
|---|---|
| Head as when playing (c ≈ 0.50), glasses view | 92/96, all upright |
| Head tilted further (c ≈ 0.83), glasses view | 1/27 |

Foreshortening with only upright cards (synthetic, 1.5 px noise) was wrong in 58 to 69 % of frames
without the prior and right in 100 % with it.

## YOLO (comparison)

| Model | Result on our footage |
|---|---|
| lgd-cards-gen3 (corner index), 640 px | 0 of 112 frames with detections (360×640) |
| same, upscaled 2× | 70 of 112 frames, 225 ms per frame, K♥ misread |
| same, 12-card scene at 720×1280 | 0 to 0.7 cards per frame |
| Whole-card YOLO11n (irfan) | all cards, 46 ms per frame, but the laptop boxed at 0.91 |

## Card recognition (52-card recording, 523 held-out crops of the same session)

| Recogniser | Rank | Suit | Card | Time per card |
|---|---|---|---|---|
| Classical, original colour test | 99.2 % | 62.1 % | 61.8 % | 63 ms |
| Classical, colour fixed | 99.2 % | 81.5 % | 81.1 % | 63 ms |
| Embedding (all 2,091 crops, never trained on our deck) | 96.4 % | 98.2 % | 95.2 % | 50 ms |
| **Hybrid** | 99.2 % | 98.2 % | **97.3 %** | 113 ms |
| Embedding, classical only when unsure | | | 95.0 % | |

Hybrid by group was face cards 93.8 %, black 97.8 % and red 99.1 %. It names 99.4 % of the cards
and 97.9 % of those are right. These are within-session numbers and the second recording will give
the held-out ones.

| Colour test (2,091 crops) | Black read as red |
|---|---|
| Hue and saturation only | 543 |
| With brightness ≥ 110 | 94 |

## Real-time pipeline

| Stream | Time per frame | Detection |
|---|---|---|
| 720×1280 | 54 ms (about 18 fps) | 51 ms |
| 504×896 | 42 ms | 35 ms |
| 360×640 | 14 ms | 12 ms |

Recognising from the first frame named every card with more mistakes (8♣ as 8♠, K♥ as K♠, 2♦ as
2♠). The fused three sharpest of five frames got them right or left them as `?`.

## Security

| Version | Aces found | Non-aces taken as aces | Red cards read as red |
|---|---|---|---|
| First (centre only, index colour) | 14/14 | 85/157 | 0/14 |
| Pips over the whole body, colour over the body | 14/14 | 0/157 | 14/14 |

## Calibration (photo mode)

RMS 0.329 px (leave-one-out 0.394 px) against 0.553 px for the standard calibration, and the
distortion is almost zero (`k1 ≈ −0.016`).

## Tests

About 380 automated tests (synthetic scenes and real crops) run on every pull request.
