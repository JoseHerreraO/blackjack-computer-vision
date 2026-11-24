# 06 Game logic, from cards on the table to spoken advice

The assistant cannot hear the player, so the round has to follow from what appears on the table.
Two modules do this. `blackjack/game.py` (José Herrera) is the round itself, with basic strategy and the
Hi-Lo count. `blackjack/table.py` (José De Oña) turns the cards seen by the camera into events for
that round.

## The round (`Round`)

| Phase | What happens |
|---|---|
| Waiting | Cards are being dealt. With two player cards and a dealer up card the player's turn starts (or the dealer's, on a player blackjack) |
| Player's turn | `recommendation` gives the basic-strategy action for the hand. The hand ends by itself on 21, on bust, or after the card of a double |
| Dealer's turn | Any dealer card after the player started acting (the hole card turned over, or a drawn card) means the player is done |
| Finished | The dealer stands on 17 (S17 by default) or busts. `results()` gives the outcome and the payout |

Basic strategy uses the usual tables for hard totals, soft totals and pairs, with 6 decks, dealer
stands on soft 17, double after split and 3:2 for a blackjack. The actions are said in Spanish
("Pide carta", "Plántate", "Dobla", "Separa", "Ríndete").

The **Hi-Lo** count (`blackjack/shoe.py`) adds +1 for 2 to 6, 0 for 7 to 9 and −1 for tens and aces, divides the
running count by the decks left to get the true count, estimates the player's edge and suggests a
bet (one unit until the count is favourable, then the true count minus one).

## The table (`Table`)

- **Zones.** The glasses look at the table from the player's seat, so the dealer's cards are on the
  far side (top of the image) and the player's on the near side. A line at half the image height
  (`--dealer-line`) separates them, and every newly recognised card is dealt to the hand of its zone.
- **Hole card.** A dealer card that stays unrecognised for 15 frames while the dealer shows one card
  is the face-down card. It is shown as HOLE and counted only when it is turned over.
- **Cards found again.** A card hidden by a hand loses its track and comes back with a new one. If
  a card with the same code was already dealt to the same zone in this round, its old track is gone
  and the new one appears within 1.5 card sizes of where it was last seen, it is the same card and
  is not counted twice. The dealt card then follows the new track, so a later equal card (there are
  several decks) is counted.
- **New round.** A card after the round has finished starts a new round. An empty view starts one
  only when that cannot lose a hand, that is when the round is over, or when the mat is in view and
  empty (the cards were collected). The mat is found by its colour (`vision_classic/mat.py`, HSV,
  Tema 2). Our teal mat has a hue of about 87 and covers at least 10 % of the frame when looking at
  the table and at most 3.4 % elsewhere, so 8 % is the threshold. With the head turned away the round
  waits for the cards to come back. The count always carries over, and a key resets it for a new shoe.
- **Advice.** The recommendation is queued as speech only when it changes, and the result when the
  round ends ("Ganas", "Pierdes", "Empate", "Blackjack"). The count goes to the phone as a status
  line, only when it changes.

The three cases about found cards and empty views were found in José Herrera's review of #25 and are tests
now.

## Limits

The camera cannot see a split or a double. The player is followed as a single hand, and the third
card after "Dobla" counts as a normal hit, which still gives the right total and result for that
hand.

See also `docs/assistant.md`.
