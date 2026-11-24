# 05 Security, the four-aces lock

The assistant stays **locked** until four aces are shown one at a time in a secret order of suits,
for example spades, hearts, diamonds, clubs. While locked the HUD shows only the progress
(`LOCKED ##..`), never the secret, and no advice is given. The lock was built by José Herrera
(`security/aces.py`) with techniques of the course only.

## How an ace is read

| Step | Technique | Tema |
|---|---|---|
| Find and straighten the card | Classical detector and homography to 200×300 | 3, 4 |
| Red or black | Share of the card's ink (Otsu) that is red in HSV, with the face-card picture left out | 2 |
| Is it an ace? | Otsu threshold of the card body without its margin and index corners, then connected components. The sizeable blobs are the pips, and an ace has exactly one | 3, 5 |
| Which suit? | The pip and the index symbols are normalised to 32×32 and compared with one template per suit by normalised cross-correlation, also upside down, only among the suits of the detected colour | 3 |

When the recogniser already gives the rank (`--use-rank`), it decides which cards are aces.

## The sequence

`SequenceDecoder` turns the readings of each frame into ace events.

- An ace counts once its suit has been read in five consecutive frames, which filters
  single-frame misreadings.
- The same suit counts again only after the ace has left the view, so holding an ace counts once.
- A correct ace advances. A wrong one starts over (and counts as the first ace if it matches the
  start of the secret). Ten seconds without an ace also start over, and a key locks again.
- With two aces in view the frame is ignored, because their order would be ambiguous. **Each ace
  must be taken away before showing the next one.**

The secret is given in the `BJ_SECRET` environment variable (suit codes such as `SHDC`), so it does
not stay in the shell history.

## What real footage taught

The first version worked on synthetic cards and failed on the glasses footage in three ways. All of
them are now regression tests with real crops (`tests/data/real_crops/`, `tests/test_real_crops.py`).

1. It looked only at the centre of the card for "one big pip". In our deck the 3 and the 5 also
   have a single pip there, so 85 of 157 non-aces were taken as aces. Counting the pips over the
   whole body fixed it (14 of 14 aces, 0 false of 157), and the pip count is now exact for every card.
2. It measured red ink in the small, pale index corners, and every red card read as black. Measuring
   over the whole card fixed it, and black face cards, whose artwork has red and yellow, read red
   until the picture was left out of the measurement.
3. Under the warm light of the 52-card recording black ink looked dark brown and read as red. Red
   pixels now also have to be bright, which separates red from black in both recordings.

The suit templates come from the trained classical recogniser, so the lock needs that model even
when the hybrid recogniser is used for the game.

See also `docs/security.md`.
