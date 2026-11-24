# Card recognition guide

The recogniser receives a card already detected and rectified to 200 × 300 px
(`CardDetector` + `rectify_card`) and returns its rank and suit.

| Output | Method | Module |
|--------|--------|--------|
| Rank (13 classes) | Dense RootSIFT on the whole card → K-Means vocabulary (`cv2.kmeans`) → spatial-pyramid bag of words (1×1, 2×2, 4×4) → SVM with χ² kernel | `vision_classic/features.py`, `vision_classic/recognition.py` |
| Suit (4 classes) | Red or black from the share of red ink on the card body (HSV, Tema 2; face cards: picture left out and colour not used), then the suit symbol under the rank, isolated with Otsu thresholding and connected components, normalised to 32 × 32 and matched by normalised cross-correlation against one template per suit (Tema 3) | `vision_classic/suit_templates.py` |

The detector does not know which short side of a card is the top, so cards may arrive rotated
by 180°: the rank SVM is trained on both orientations and the suit uses both index corners.

OpenCV 5 removed the `ml` module and the `BOW*` classes, so the vocabulary and the word
assignment use `cv2.kmeans` and `cv2.BFMatcher` and the SVMs come from scikit-learn. The code
works with OpenCV 4.x and 5.x.

## 1. Capturing the dataset

Use the deck we will play with, on the same mat and under the same kind of light. The glasses are
not needed: the webcam (or a phone used as a webcam) is enough, as long as the image is sharp.

```bash
python scripts/capture_cards.py --source 0 --deck
```

- Show **one card at a time**; it must be outlined in green (orange means "not a single card").
- Press space to save, or `a` to auto-save every 15 frames while you move the card.
- `n` / `b` go to the next / previous card. Images are stored as `data/cards/<CODE>/NNNN.png`
  (`AS`, `TH`, `QD`...), which git ignores.
- Aim for **10–20 images per card**: different positions and rotations, nearer and farther,
  tilted camera, light from different sides, a little blur. Variety matters more than quantity.
- Once the glasses stream works, add images captured through it: its colours, resolution and
  compression differ from the webcam.

## 2. Training and evaluation

Capture the cards with the same camera and view they will be used with. With the glasses
(`--source bridge`) `capture_cards.py` uses `--view glasses` automatically: a single card gives no
foreshortening estimate, and with the default view an upright card would be saved rotated by
90 degrees.

Record **two separate sessions** (different time, light and positions): one to train and one to
test, and evaluate with `--test-data`. A random split of the same recording is optimistic,
because its training and test crops are almost the same pictures.

```bash
python scripts/train_recognizer.py --data data/cards --test-data data/cards_test     --out data/models/recognizer.joblib --errors-dir outputs/recognition_errors
```

The report includes how many cards `recognise()` would name at its confidence threshold (the
weakest one-vs-one SVM margin of the winning rank, 0.4 by default) and how many of those are
right. This is what the real-time pipeline uses (`scripts/track_cards.py --recognizer classic`):
below the threshold a card stays `?` and is tried again with fresher crops.


```bash
python scripts/train_recognizer.py --data data/cards --out data/models/recognizer.joblib \
    --errors-dir outputs/recognition_errors
```

A stratified 25 % of every card is held out to report rank, suit and card accuracy and the rank
confusion matrix; then the model is retrained on all the images and saved. `--errors-dir`
stores the misrecognised test images, which usually show what is missing from the dataset
(glare, blur, cards cut by the rectification).

Useful knobs for the vocabulary/kernel study of the report: `--vocabulary` (number of words),
`--levels` (pyramid, e.g. `--levels 1` for a plain bag of words), `--sampling keypoints`
(SIFT detector instead of the dense grid), `--step` and `--c`.

Model files are `joblib` pickles: only load models you trained yourselves.

## 3. Reference numbers (synthetic cards)

On the synthetic cards of the tests (8 photos per card, 25 % held out): rank 97 %, suit 100 %,
about 20 ms per card and 25–30 s of training. The only rank errors are among J, Q and K, whose
synthetic faces differ only in a letter; real face cards differ much more. These numbers only
show that the pipeline works: the accuracy that counts is the one measured on our own photos.

In the live assistant a card does not need to be recognised in every frame: predictions of the
same tracked card are combined with `TemporalVoter` (majority over the last N frames).

## 4. Recogniser study for the report

`scripts/study_recognition.py` runs the experiments of the Results section on one train folder
and one test folder (two separate recordings; without `--test-data` it uses a random split and
the output warns that the numbers are optimistic):

| Study | What changes | Syllabus |
|---|---|---|
| `vocabulary` | number of visual words `K` (default 50, 100, 200, 300) | Tema 4, bag of words |
| `pyramid` | `1x1`, `1x1+2x2`, `1x1+2x2+4x4` | Tema 4, spatial pyramid |
| `kernel` | SVM on the same histograms: χ², histogram intersection, linear, RBF | Tema 4 |
| `baseline` | whole-card template matching with NCC (`TM_CCOEFF_NORMED`), one mean template per card, both 180° orientations | Tema 3 |

```powershell
python scripts/study_recognition.py --data data/cards --test-data data/cards_test --vocabulary 50 100 200 300 500
```

It writes `outputs/study/results.csv`, `results.md` (the table for the report) and
`vocabulary.png` (accuracy and ms/card against `K`). The kernel rows only measure the rank
(the suit does not depend on the SVM) and their ms/card is the SVM alone: the histograms are
computed once for all kernels. `--quick` runs tiny settings as a smoke test; `--studies` picks
a subset.

Synthetic cards, 4 photos per card for training and 2 from another seed for testing
(52 cards, 2 min 44 s for everything):

| Study | Setting | Rank | Suit | Card | Train (s) | ms/card |
|---|---|---|---|---|---|---|
| baseline | template NCC | 95.2 % | 64.4 % | 60.6 % | 0.0 | 19.5 |
| vocabulary | K=50 | 88.5 % | 100.0 % | 88.5 % | 9.6 | 29.1 |
| vocabulary | K=100 | 89.4 % | 100.0 % | 89.4 % | 10.5 | 29.9 |
| vocabulary | K=200 | 93.3 % | 100.0 % | 93.3 % | 11.8 | 32.1 |
| vocabulary | K=300 | 99.0 % | 100.0 % | 99.0 % | 14.2 | 35.2 |
| vocabulary | K=500 | 100.0 % | 100.0 % | 100.0 % | 17.5 | 40.8 |
| pyramid | 1x1 | 98.1 % | 100.0 % | 98.1 % | 11.1 | 29.0 |
| pyramid | 1x1+2x2 | 98.1 % | 100.0 % | 98.1 % | 11.3 | 30.0 |
| pyramid | 1x1+2x2+4x4 | 99.0 % | 100.0 % | 99.0 % | 13.6 | 34.2 |
| kernel | chi2 | 99.0 % | – | – | 3.2 | 4.0 |
| kernel | intersection | 100.0 % | – | – | 3.2 | 8.1 |
| kernel | linear | 92.3 % | – | – | 0.2 | 0.6 |
| kernel | rbf | 99.0 % | – | – | 0.2 | 0.9 |

What it already shows: whole-card NCC gets the rank of these clean synthetic faces but confuses
suits of the same colour (a spade and a club differ in a few pixels of a 100×150 image), while
the bag of words with the suit templates never does; accuracy grows with `K` up to about 300
words at a moderate cost in time; and the kernels designed for histograms (χ², intersection)
beat the linear one. The numbers for the report must come from the real recordings.
