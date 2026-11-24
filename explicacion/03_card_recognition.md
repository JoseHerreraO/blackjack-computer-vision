# 03 Card recognition

Once a card is found and straightened to 200×300 px (`rectify_card`), a recogniser names its rank
and suit. Three recognisers exist. After measuring them on our own deck through the glasses, the
**hybrid** one is the default. It takes the rank from the classical recogniser and the suit from a
learned embedding.

## The 52-card dataset

The deck was recorded through the glasses with `scripts/capture_cards.py --source bridge --deck`
(720×1280, on our mat, one card at a time, near the player and on the dealer's side, rotated and
upside down). That gave **2,129 crops**.

Checking them found two problems.

- **519 crops (24 %) came out rotated by 90°**, mostly clubs and diamonds. It is the single-card
  foreshortening issue of `02_card_detection.md`. The head was tilted more than the starting value
  assumed, and upright cards were taken for sideways ones. These crops were not discarded. A crop
  rectified with the axes swapped has its content stretched the other way, so turning it by 90° and
  resizing it back to 200×300 gives a correct card. They were found by asking the embedding model
  whether the crop or its 90° rotation looked more like the labelled card.
- **38 crops were not cards** (hands, the wooden table, a fanned hand). They were moved to
  `data/cards_rejected/` after checking each one by eye.

The result is **2,091 clean crops**, 18 to 60 per card, in `data/cards/` (the original is kept in
`data/cards_raw/`). A sheet with one crop per card confirmed that every label is right and every card
is upright.

## The classical recogniser (José Herrera)

- **Rank.** Dense RootSIFT on the whole card, a K-Means vocabulary of 300 words, a spatial-pyramid
  bag of words (1×1, 2×2, 4×4) and an SVM with a χ² kernel, trained on both 180° orientations.
- **Suit.** First red or black from the share of red ink on the card (HSV), then the suit symbol of
  the index corner, normalised to 32×32 and compared by normalised cross-correlation with one
  template per suit, only among the suits of that colour.

Trained on the dataset (1,568 crops, 9 minutes on the laptop) and tested on the other 523, it read
the **rank 99.2 %** right but the **suit only 62.1 %**. The cause was the colour. Under the warm
light of that session black ink looks dark brown (hue 19, saturation 60, value 81), and the red
test (hue ≤ 20, saturation ≥ 45) took it for red. Spades were read as red in 306 of 369 crops, and
once the colour is wrong the suit cannot be right. Real red ink is bright (value about 210), so the
fix requires red pixels to be bright too (value ≥ 110). The existing threshold of 0.45 then
separates both recordings (red cards 0.70 to 1.00 and black ones 0.00 to 0.35). The suit went up to
81.5 %, still limited by spades against clubs and hearts against diamonds.

## The embedding recogniser

`vision_dl/embedding_recognizer.py` uses the recogniser of `irfanhossainsust/playing-card-detector`
(AGPL-3.0). An EfficientNet-B0 maps a card to a 256-dimensional vector, trained on photos of many
deck designs with blur, glare, perspective and low-resolution augmentation. The card is named after
the closest of 7,624 reference vectors by cosine similarity, trying it also upside down, and below
0.70 it is reported as unknown. It **never saw our deck**, and on the 2,091 crops it named 95.2 % of
the cards right. Its weak spots are 9 against 10 and K♣ against K♠. The weights are downloaded by
`scripts/download_card_recognizer.py`, pinned to one commit and checked by SHA-256, because a `.pt`
file is a pickle that runs code when it is loaded.

## The hybrid recogniser

Each recogniser is good at a different half of the card.

| Recogniser | Rank | Suit | Card |
|---|---|---|---|
| Classical | **99.2 %** | 81.5 % | 81.1 % |
| Embedding | 96.4 % | **98.2 %** | 95.2 % |
| **Hybrid** (classical rank + embedding suit) | 99.2 % | 98.2 % | **97.3 %** |
| Embedding, classical only when unsure | | | 95.0 % |

`vision_dl/hybrid_recognizer.py` takes the rank from the classical recogniser and the suit from the
embedding. It falls back to the embedding rank when the classical rank is unsure, and to the
classical suit when the embedding has no usable card. It names 99.4 % of the cards and gets 97.9 %
of those right (face cards 93.8 %, black 97.8 %, red 99.1 %). The simpler idea of using the
embedding and asking the classical recogniser only when the embedding is unsure did not help
(95.0 %), because when the embedding is unsure the classical one is usually wrong too.

The hybrid also fits the course. The rank, which decides the blackjack value of a card, comes from
the syllabus technique, and the learned model only gives the suit, which matters for the lock and
the dealer's hole card but not for the strategy. It costs about 63 ms (classical) and 50 ms
(embedding) per card on the CPU, but only once per card (see `04_real_time_pipeline.md`).

## What is still needed

These numbers come from one session, split at random into training and test crops, so they are
optimistic for the classical recogniser. A **second recording** (another time of day, other light
and positions) will be the test set, with `train_recognizer.py --test-data`. José Herrera's study script
(`scripts/study_recognition.py`, #27) then measures the vocabulary size, the pyramid and the kernel
on the same sets for the report.

See also `docs/recognition.md` (training) and `docs/card_pipeline.md`.
