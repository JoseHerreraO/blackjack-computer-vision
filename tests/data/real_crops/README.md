# Real card crops from the glasses

Rectified 200x300 cards cut from the Meta glasses stream (BJ Bridge, 720x1280 at 15 fps,
recording of 2026-11-18, cards dealt on our green mat). Each one was found by the classical
`CardDetector` (with the foreshortening correction of #19) and warped with `rectify_card`.

The file name is the card code and a number, e.g. `QS_2.png` is the queen of spades. The labels
were proposed by the embedding recognizer (cosine similarity >= 0.85) and then checked by eye.
For each card the sharpest crops of the recording were kept, so they show the usual video
quality: small cards upscaled, some blur, pale and slightly orange red ink.

| Cards | Why they are here |
|---|---|
| `AC_*`, `AD_*` | black and red aces (one pip in the body) |
| `2D_1`, `3D_*`, `5C_1`, `5S_1`, `9S_1` | number cards. The 3 and the 5 have a pip in the centre |
| `QS_*`, `JC_1` | black face cards, whose artwork has red and yellow |
