# 08 Calibration

The camera of the glasses was calibrated with Zhang's method and a printed chessboard, following
the procedure of Lab 2 (José Herrera, `notebooks/01_calibracion_camara.ipynb`, `scripts/calibrate.py`,
`docs/calibration.md`).

## What was done

Fifteen photos of a 6×4 board (25 mm squares) were taken with the glasses in photo mode
(3024×4032). Corners were found with `findChessboardCornersSB` and refined. Because a printed sheet
is never perfectly flat, the board shape was estimated together with the camera
(`calibrateCameraRO`, release object), and views far from a flat board were dropped.

| Setting | RMS error | Leave-one-out |
|---|---|---|
| Standard calibration | 0.553 px | 0.598 px |
| Accurate corners + release object | **0.329 px** | 0.394 px |

The model is `data/calibration/gafas_foto.json`. The distortion came out almost zero
(`k1 ≈ −0.016`), which suggests the glasses already correct their photos in software.

## Why it is not applied live

The model describes the photo mode, a 3:4 frame, while the video stream is a 9:16 crop of the
sensor, so the matrix cannot simply be scaled to the stream. Calibrating the stream itself was
possible, but the distortion measured is negligible, the card homographies already correct each
card's perspective individually, and the foreshortening of the table is estimated from the cards in
every frame (`02_card_detection.md`). The calibration therefore stays as a module of the project
(it is graded as such), and the live pipeline works on the raw frames. The scripts still accept
`--camera <model.json>` to undistort first if a stream model is made.
