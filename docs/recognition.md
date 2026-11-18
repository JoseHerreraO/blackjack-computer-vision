# Card recognition guide

The recogniser receives a card already detected and rectified to 200 × 300 px
(`CardDetector` + `rectify_card`) and returns its rank and suit.

| Output | Method | Module |
|--------|--------|--------|
| Rank (13 classes) | Dense RootSIFT on the whole card → K-Means vocabulary (`cv2.kmeans`) → spatial-pyramid bag of words (1×1, 2×2, 4×4) → SVM with χ² kernel | `vision_classic/features.py`, `vision_classic/recognition.py` |
| Suit (4 classes) | Red or black from the ink of the two index corners (HSV), then the shape of the suit symbol under the rank (log Hu moments, solidity, aspect, extent, width profile) → RBF SVM | `vision_classic/suit_shape.py` |

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
