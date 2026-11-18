# Security system

The assistant stays locked until four aces are shown to the camera in a secret suit order.
Everything uses techniques of the course:

| Step | Technique | Tema |
|------|-----------|------|
| Find and rectify the card | `CardDetector` + homography to 200 × 300 | 3, 4 |
| Red or black | Fraction of red ink (HSV) in the index corners | 2 |
| Is it an ace? | Otsu thresholding of the centre of the card and connected components: the largest blob must hold ≥ 60 % of the ink and not touch the region border (one big pip). The recogniser's rank can be used instead (`--use-rank`). | 3, 5 |
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
