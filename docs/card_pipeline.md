# Real-time card pipeline

```
frame -> CardDetector (classical, corners) -> CardTracker (IDs) -> crop buffer per card
                                                                     |
                     once per card, after ~5 frames: fuse the 3 sharpest -> EmbeddingRecognizer
```

`src/blackjack_vision/detection_tracking/pipeline.py`, run live with
`python scripts/track_cards.py --source bridge`.

## Why this split

- **Finding the cards** is done by the classical detector (#14, with the foreshortening fix of
  #19). It gives the four corners of every card, which the homography needs to straighten it,
  and it does not fire on the laptop or the phone. The whole-card YOLO of the recognizer's repo
  finds the cards too but only gives axis-aligned boxes and also boxes the laptop (0.91).
- **Naming the card** is done by the classical recogniser by default (bag of words + χ² SVM for
  the rank, NCC templates for the suit, Temas 3-4; `--recognizer classic`, trained with
  `scripts/train_recognizer.py`), which reports a card as unknown when its rank's weakest
  one-vs-one SVM margin is below 0.4. For comparison it can be a learned recognizer
  (`--recognizer embedding`), the embedding network of
  [`irfanhossainsust/playing-card-detector`](https://huggingface.co/irfanhossainsust/playing-card-detector)
  (AGPL-3.0), trained on photos of many deck designs with blur, glare, perspective and
  low-resolution augmentations. It compares the straightened card with 7,624 reference
  embeddings and needs no training on our deck. Below a cosine similarity of 0.70 the card is
  reported as unknown instead of guessed. Both are to be measured on the same held-out
  recording of the 52 cards; the classical one stays the main recogniser if it gets close, as
  the course syllabus asks.

## Real time

Every frame only runs the cheap steps (detection, tracking, one rectification per unrecognised
card), so the loop keeps up with the 15 fps stream. Each card is recognised **once**:

1. While a card is unrecognised, its rectified 200x300 crops go into a buffer of 10 with their
   sharpness (variance of the Laplacian, `vision_classic/sharpness.py`).
2. After `min_crops = 5` frames (a third of a second) the 3 sharpest crops are aligned (a crop can
   come out rotated 180°) and averaged. Picking the sharpest drops the frames smeared by head
   movement, and averaging aligned views removes noise and JPEG blocks.
3. The fused crop is recognised. If the recognizer is unsure it tries again 5 frames later with
   fresher crops. Once named, a card keeps its name while it is tracked and appears once in
   `PipelineResult.new_cards`, which is when Hi-Lo should count it.

## Setup

```bash
pip install -e ".[yolo]"                  # PyTorch + torchvision (see pyproject.toml for CPU)
python scripts/download_card_recognizer.py  # pinned commit, every file checked by SHA-256
```

The `.pt` files are pickles that run code when loaded, which is why the download is pinned and
verified. The reference database is a FAISS flat index read directly with NumPy (no FAISS).

## Results on our recordings (2026-11-18, laptop CPU, no GPU)

Every frame of the 2026-11-18 recording through the whole pipeline. "Named" is the share of
tracked cards that got a name (the rest stayed `?` because the recognizer was never sure).

| Stream   | Frame time | Named, first frame | Named, fused 3 of 5 | Mean similarity (fused) |
|----------|------------|--------------------|---------------------|-------------------------|
| 720x1280 | 54 ms (~18 fps), detection 51 ms | 16/16 | 15/16 | 0.88 |
| 504x896  | 42 ms (~24 fps)                  | 14/15 | 14/15 | 0.86 |
| 360x640  | 14 ms                            | 9/14  | 9/14  | 0.82 |

The first-frame strategy names more cards but gets more of them wrong, with the 8 of clubs as
8♠, the king of hearts as K♠ and two of diamonds as 2♠. With the sharpest crops fused, those
come out right (8♣, J♣, 5♣) or stay `?` instead of a wrong name. The rest of the errors are on
crops that are still very blurred, taken while the head was moving when the card appeared.
Recognition costs 25-60 ms per card and is capped at 2 cards per frame, so many cards appearing
together do not stall the stream. 720x1280 is the resolution to use. At 360x640 a third of the
cards are never named.
