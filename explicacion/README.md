# How the Blackjack assistant works

This folder explains the whole project, one document per part, in the order the data flows. It
gathers what was built, why each decision was taken and what was measured on our own glasses
recordings, so it can be read without going through the pull requests. The technical guides in
`docs/` stay as the short reference for each module, and these documents are the long version.

```
Meta glasses ──Bluetooth──► iPhone (BJ Bridge) ──Wi-Fi, TCP 5050──► PC (Python, OpenCV)
                                                                      │
       ┌──────────────────────────────────────────────────────────────┘
       ▼
   LOCKED until four aces in the secret order
       │
       ▼
   classical card detector ──► Kalman tracker ──► one hybrid recognition per card
       │                                                   │
       ▼                                                   ▼
   table zones (dealer / player) ──► round, basic strategy, Hi-Lo ──► voice through the glasses
                                                                   └─► HUD with FPS on the PC
```

| Document | What it covers |
|---|---|
| [01 Phone connection](01_phone_connection.md) | The iOS bridge app, how it is built and installed without a Mac, the network and the protocol |
| [02 Card detection](02_card_detection.md) | Finding the cards with classical vision, the foreshortening of the glasses and the YOLO experiments |
| [03 Card recognition](03_card_recognition.md) | The 52-card dataset, the classical, embedding and hybrid recognisers and their results |
| [04 Real-time pipeline](04_real_time_pipeline.md) | How everything runs at the stream rate, the tracker and the per-card recognition |
| [05 Security](05_security.md) | The four-aces lock |
| [06 Game logic](06_game_logic.md) | Table zones, the round, basic strategy, Hi-Lo and the spoken advice |
| [07 Starting a game](07_starting_a_game.md) | Step by step from switching everything on to playing a hand |
| [08 Calibration](08_calibration.md) | The camera calibration and why it is not applied live |
| [09 Results](09_results.md) | Every number measured on real footage, in one place, for the report |

## Who did what

José Herrera built the classical core (card detector and homography, bag-of-words recogniser and
suit templates, round logic and strategy, the security system, the calibration notebook and the
recogniser study). José De Oña built the glasses bridge (iOS app, PC receiver and bridge source),
the foreshortening correction of the detector, the tracker, the real-time pipeline with the
embedding and hybrid recognisers, the live assistant (`scripts/run.py`, table logic and HUD) and
the evaluation on real footage. Every change went through a pull request reviewed by the other.

## Status (24 November 2026)

The full chain works on recorded glasses footage and its parts were tested live (video from the
glasses to the PC, card capture with the glasses). Still open are the live test of the voice
through the glasses (#22), splitting touching cards (#26), and the second card recording that will
give the definitive recogniser numbers.
