# Security system

The assistant stays locked until four aces are shown to the camera in a secret suit order.
Everything uses techniques of the course:

| Step | Technique | Tema |
|------|-----------|------|
| Find and rectify the card | `CardDetector` + homography to 200 × 300 | 3, 4 |
| Red or black | Share of the card's ink (Otsu) that is red in HSV: hue ≤ 20 or ≥ 165 and saturation ≥ 45, threshold 0.45 over the whole card body. A face-card picture (a blob wider and taller than half the body) is left out, and for face cards the colour is not used. | 2 |
| Is it an ace? | Otsu thresholding of the card body (without the margin and the four index corners) and connected components: the sizeable blobs are the pips, and an ace has exactly one, not too small and not as large as a face-card picture. The recogniser's rank is preferred when available (`--use-rank`). | 3, 5 |
| Which suit? | The pip and the index symbols are cropped, centred and resized to 32 × 32, then compared with one template per suit by normalised cross-correlation (`cv2.matchTemplate`, `TM_CCOEFF_NORMED`). The pip is also tried upside down. Only suits of the detected colour compete. | 3 |
| Sequence | `SequenceDecoder` (see below) | – |

The suit templates are the mean of the normalised index symbols of every suit in the training
cards; the same templates give the suit in the recogniser (`vision_classic/suit_templates.py`).

## Sequence decoder

* An ace counts once its suit has been read in `min_frames` consecutive frames (5 by default),
  which filters single-frame misreadings. With two aces in view the frame is ignored.
* The same suit only counts again after the ace has left the view for `gap_frames` frames, so
  holding one ace in front of the camera counts once.
* A correct ace advances; a wrong one starts over (and counts as the first ace if it matches
  the start of the secret); more than `timeout` seconds between aces starts over; `r` restarts
  and locks again.
* The screen only shows the progress (`LOCKED ##..`), never the secret.
* **Take each ace away before showing the next one.** With two aces in view the frame is
  ignored, so aces left on the table stop the sequence.

## Running it

```bash
set BJ_SECRET=SHDC
python scripts/unlock_demo.py --source 0 --model data/models/recognizer.joblib
```

The secret is a string of suit codes (`S` spades, `H` hearts, `D` diamonds, `C` clubs). Pass it
through the `BJ_SECRET` environment variable rather than `--secret`, so it does not stay in the
shell history.

On the synthetic cards of the tests, the pip test separates the 40 aces from the other 480 cards
without errors and reads all their suits, also upside down.

## Lessons from real glasses footage

The first version, tested on 171 rectified crops of the 18/11 recording (720×1280), had two
problems that the synthetic cards did not show; both are now regression tests:

* It looked only at the centre of the card for "one big pip". In our deck the 3 and the 5 have a
  single pip there too, so 85 of 157 non-aces were taken as aces. Counting the pips over the whole
  body fixes it.
* It measured red ink in the index corners with hue ≤ 10 and saturation ≥ 90. In the video the
  indices are small and pale and the real red ink is orange-ish (median hue 9, saturation 79), so
  every red card read as black (0.000). Measured over the whole card with the thresholds above,
  red cards give 0.59–0.80 and black ones 0.00–0.30.
* Measuring on the whole card then made black face cards read red (J♣ 0.54, Q♠ 0.62–0.67):
  their artwork has red and yellow. The picture is now left out of the measurement, and a face
  card's colour is reported as unknown, because without the picture only the pale index corners
  remain and they do not separate red from black reliably (0.00–0.39 on red number cards). The
  suit of a face card is chosen among the four suits by the shape of its index symbol alone,
  which got all 13 real crops right in a leave-one-out test.
* `tests/data/real_crops/` holds 13 labelled real crops; `tests/test_real_crops.py` checks the
  colour, the pip count, the ace test and the suit by shape on them.
