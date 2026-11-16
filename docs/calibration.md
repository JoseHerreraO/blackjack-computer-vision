# Camera calibration guide

Every camera we use (the glasses stream and the fallback webcam) needs its intrinsic matrix `K`
and its distortion coefficients before we can undistort frames and compute metrically correct
homographies of the cards. We use Zhang's method with a planar chessboard (Tema 2).

## 1. The pattern

- Print a chessboard of **10 × 7 squares (9 × 6 inner corners)**, for example OpenCV's
  `doc/pattern.png`, on matte paper (glossy paper reflects the lamps).
- Glue it to something **rigid and flat** (foam board, a hardcover book). A bent sheet ruins the
  calibration even if the error looks low.
- Measure one square with a ruler and pass it as `--square-size` in millimetres. It only scales
  the extrinsics; the intrinsics do not depend on it.

## 2. One calibration per camera and resolution

`K` is only valid at the resolution it was estimated with. The glasses stream (Meta Wearables
Device Access Toolkit) is portrait and offers three resolutions with the same aspect ratio:

| Stream quality | Resolution  | Model file                          |
|----------------|-------------|-------------------------------------|
| `HIGH`         | 720 × 1280  | `data/calibration/glasses_high.json`   |
| `MEDIUM`       | 504 × 896   | `data/calibration/glasses_medium.json` |
| `LOW`          | 360 × 640   | `data/calibration/glasses_low.json`    |
| Fallback webcam| its own     | `data/calibration/webcam.json`         |

If the lower resolutions are pure downscales of the same field of view, `CameraModel.scaled()`
converts a model between them. Verify it once by calibrating `HIGH` and `MEDIUM` independently and
comparing `scaled()` against the real `MEDIUM` calibration.

The glasses must be calibrated with frames that went through the **same stream** we use live
(glasses → phone → PC), not with photos taken by the glasses: photos use another resolution and
crop. Until the bridge works, calibrate the webcam.

## 3. Recording the views

Record a 30–60 s video (or take 15–25 photos) while moving the board **slowly** (motion blur
shifts the corners):

- cover the whole image, especially the **corners and borders**, where distortion is largest;
- tilt the board about ±30–45° around both axes, not only frontal views;
- vary the distance (board filling ~25 % to ~80 % of the image);
- keep the whole board, including its white margin, inside the image;
- use even lighting without reflections on the board.

## 4. Running the calibration

```bash
# From a video (1 frame out of 5 is analysed; near-duplicate views are skipped)
python scripts/calibrate.py --source data/calibration/glasses_high.mp4 --pattern 9x6 \
    --square-size 25 --name glasses_high --debug-dir outputs/calibration/glasses_high

# From a folder of photos
python scripts/calibrate.py --source data/calibration/webcam/ --pattern 9x6 --square-size 25 \
    --name webcam

# Live from the webcam (press q or Esc when enough views are collected)
python scripts/calibrate.py --source 0 --pattern 9x6 --square-size 25 --name webcam
```

Views whose reprojection error is far above the others (blur, wrong detections) are discarded
automatically and the calibration is repeated once; use `--keep-outliers` to disable it.
`--fix-k3` helps with few views or narrow lenses, `--rational` with very wide-angle lenses.

## 5. Checking the result

1. **RMS reprojection error**: below 0.5 px is acceptable, 0.1–0.3 px is good. A low error alone
   does not prove a good calibration: few or similar poses, or too many distortion coefficients,
   can overfit.
2. **Uncertainty**: the report shows the standard deviation of `fx`, `fy`, `cx`, `cy`. A few
   pixels or more means the poses are not varied enough.
3. **Visual check**: straight edges must look straight after correction, also near the borders:

   ```bash
   python scripts/preview_undistort.py --camera data/calibration/webcam.json --source 0
   ```

   With `--debug-dir`, `undistorted_*.jpg` shows raw vs corrected views and
   `per_view_errors.csv` the error of every view (useful plots for the report).
4. **Sanity**: the principal point should be near the image centre and the field of view
   plausible for the lens.

The JSON models are small and human-readable: commit them. The calibration videos and photos are
ignored by git.
