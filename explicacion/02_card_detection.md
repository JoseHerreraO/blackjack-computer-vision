# 02 Card detection

Finding the cards is done with classical vision (`detection_tracking/card_detector.py`). It gives
the **four corners of every card**, which the homography needs to straighten the card before it is
recognised. YOLO was tried as well and is kept only as a comparison, for the reasons below.

## The classical detector

1. **Segmentation.** Cards are white and the mat is coloured, so a "whiteness" channel (the minimum
   of B, G and R, high only for bright and unsaturated pixels) is thresholded with Otsu. Otsu is
   never allowed below a minimum, so an empty mat is not split in two.
2. **Morphology.** An opening removes specks and a closing fills small gaps in the outline.
3. **Contours.** Only external contours are used, so the printed symbols do not matter. Blobs that
   are too small, too large or touch the image border are discarded.
4. **Quadrilateral.** `approxPolyDP` on the convex hull must give four vertices, which are ordered
   and refined to the sharp corners by fitting a line to each side. The blob must fill its
   quadrilateral (fill ratio at least 0.92), which rejects shapes that are not cards.
5. **Aspect ratio**, after the foreshortening correction below.

Blobs big enough for a card that fail the checks are reported as **clusters**, usually several
cards touching each other.

## The foreshortening of the glasses

The first tests on real footage found only half of the cards, and none of the face cards. The
segmentation was fine. The cards were rejected by the aspect-ratio check (a playing card is 88/63,
about 1.4). The glasses look at the table at an angle, which squashes it vertically. An upright
card measured an aspect of 1.02 to 1.11 and a sideways card 2.0 to 2.4, both outside the accepted
1.15 to 1.9.

The detector now estimates that vertical squash `c` in every frame. Each candidate quadrilateral
is stretched by `1/c` for `c` between 0.4 and 1.0, and the `c` that brings their aspect ratios
closest to 88/63 wins. The median is used, so a few rectangles that are not cards cannot pull it.
The aspect check and the corner ordering are then done on the corrected shapes, so the rectified
card always comes out upright. A square stays a square after the correction, so it is still
rejected. With the glasses `c` is about 0.5.

José Herrera's review found an important case. When every card lies the same way, which is normal in
blackjack because cards are dealt facing the player, the estimate is ambiguous. An upright card
squashed by `c` looks exactly like a sideways card squashed by about `1.95 c`. With four upright
cards and corner noise, the estimator chose the wrong solution in 58 to 69 % of the frames at
`c = 0.45` to `0.5`, and every card would have been rectified rotated by 90°. The fix pulls the
estimate towards the previous one, adding `0.1 · |log(c / c_previous)|` to the error, because the
head pitch changes slowly. The pull is small next to real evidence (cards in both orientations),
so a real change of head pitch still moves it, but it decides between near-equal explanations.
`CardDetectorConfig.for_glasses()` starts from 0.5, and every script uses it when the source is the
glasses. With this, upright-only frames are right in 100 % of the synthetic tests.

With fewer than three candidates there is no estimate and the last one (or the starting value) is
used. This matters when a single card is captured. If the head tilts much more than when playing
(we measured `c ≈ 0.83` in one recording), an upright card is taken for a sideways one. That is
what rotated 519 of the 2,129 crops of the 52-card recording (see `03_card_recognition.md`). The
advice for capturing is to sit as when playing and to start with three or four cards on the mat
so the estimate is learned.

## Results on real footage

On the 23/11 recording with 12 to 13 cards on the mat (every tenth frame):

| Stream | Before the correction | After |
|---|---|---|
| 720×1280 | 5.6 cards per frame | **11.8** |
| 504×896 | 5.7 | **11.8** |
| 360×640 | 3.7 | 5.7 |

No false positives were seen. At 360×640 the small, far cards are still missed, so the system uses
**720×1280**.

## Known limit, touching cards

Two cards that touch or overlap form one blob, which comes back as a cluster and is not
recognised. They can also pull the foreshortening estimate (0.47 instead of about 0.81 in one frame
of the first recording, where the 2♠ and the J♥ touched). In blackjack the cards of a hand often
overlap, so splitting clusters is the main thing left for real hands. It is tracked in issue #26,
with classical ideas (distance transform and watershed, convexity defects, splitting by the known
card size).

## Why not YOLO

Two pretrained YOLO models were tried on our footage.

The first, `lgd-cards-gen3` (YOLO11s, 52 classes), boxes the small **corner index** of each card,
not the whole card. At the model's native 640 px it found nothing in our 360×640 frames. Upscaling
the image twice it found cards in 70 of 112 frames of the first recording, reading A♦ and 2♠ well
but mistaking K♥ for J♥ or 2♥. On the 12-card recording it found 0 to 0.7 cards per frame. The
reasons are a strong domain shift. The corner index is about 10×7 px in our frames, it is squashed
by the perspective, and the model was trained on large synthetic cards and overhead casino video.
It also ran at only 4 fps on the laptop CPU.

The second, the whole-card detector of `irfanhossainsust/playing-card-detector` (YOLO11n, one class),
found every card at 46 ms per frame, but also boxed the laptop (0.91) and the phone. And it only
gives axis-aligned boxes, without the four corners needed to straighten a card.

So detection stays classical. It needs no training, gives the corners, does not fire on other
objects and is the technique of the syllabus. The YOLO code (`detection_tracking/yolo_detector.py`,
`scripts/detect_cards.py`) remains for the comparison in the report.

See also `docs/card_detection.md` (YOLO details).
