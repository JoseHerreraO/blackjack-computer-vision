"""Hybrid card recogniser: the rank from the classical recogniser, the suit from the embeddings.

Each recogniser is good at a different half of the card on the glasses footage (52-card recording
of 2026-11-23, 523 held-out crops of the same session):

=============================== ======= ======= ======
Recogniser                      Rank    Suit    Card
=============================== ======= ======= ======
Classical (BoW + SVM, NCC suit) 99.2 %  81.5 %  81.1 %
Embedding (EfficientNet-B0)     96.4 %  98.2 %  95.2 %
**Hybrid** (this module)        99.2 %  98.2 %  97.3 %
=============================== ======= ======= ======

The classical rank (dense RootSIFT, bag of words with a spatial pyramid and a chi-squared SVM)
is the syllabus technique and decides the blackjack value. The suit, which is weak in the
classical recogniser because of the ink colour under warm light, comes from the nearest
reference embedding.

Fallbacks: when the classical rank is unsure (``rank_confidence`` below its threshold) and the
embedding match is good, the embedding rank is used. When the embedding has no usable card (a
joker, or a match below ``min_suit_similarity``), the classical suit is used. The card is unknown
when neither recogniser can give a rank, and then the pipeline tries again with fresher crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from blackjack_vision.blackjack.cards import Card, Rank, Suit
from blackjack_vision.vision_dl.embedding_recognizer import EmbeddingRecognizer, card_from_name

if TYPE_CHECKING:
    from blackjack_vision.vision_classic.recognition import CardRecognizer


@dataclass(frozen=True)
class HybridRecognition:
    """Result of :meth:`HybridRecognizer.recognise`.

    Attributes:
        card: The card, or ``None`` when no rank is trusted.
        similarity: Cosine similarity of the embedding match (what the HUD shows).
        rank_source: ``"classic"`` or ``"embedding"``.
        suit_source: ``"embedding"`` or ``"classic"``.
        rank_confidence: Classical rank confidence (weakest one-vs-one SVM margin).
    """

    card: Card | None
    similarity: float
    rank_source: str = "classic"
    suit_source: str = "embedding"
    rank_confidence: float = 0.0

    @property
    def known(self) -> bool:
        return self.card is not None


class HybridRecognizer:
    """See the module docstring.

    Args:
        classic: Trained ``CardRecognizer`` (gives the rank).
        embedding: ``EmbeddingRecognizer`` (gives the suit).
        min_rank_confidence: Classical rank confidence needed to trust its rank. ``None`` uses
            the classical recogniser's own threshold.
        min_embedding_rank_similarity: Embedding similarity needed to use its rank as fallback.
        min_suit_similarity: Embedding similarity needed to use its suit.
    """

    def __init__(
        self,
        classic: CardRecognizer,
        embedding: EmbeddingRecognizer,
        *,
        min_rank_confidence: float | None = None,
        min_embedding_rank_similarity: float = 0.70,
        min_suit_similarity: float = 0.30,
    ) -> None:
        self.classic = classic
        self.embedding = embedding
        self.min_rank_confidence = (
            classic.min_rank_confidence if min_rank_confidence is None else min_rank_confidence
        )
        self.min_embedding_rank_similarity = min_embedding_rank_similarity
        self.min_suit_similarity = min_suit_similarity

    def recognise(self, card_image: np.ndarray) -> HybridRecognition:
        """Name one rectified card."""
        prediction = self.classic.predict(card_image)
        match = self.embedding.recognise(card_image)
        matched = card_from_name(match.name)

        rank: Rank | None = None
        rank_source = "classic"
        if prediction.rank_confidence >= self.min_rank_confidence:
            rank = prediction.rank
        elif matched is not None and match.similarity >= self.min_embedding_rank_similarity:
            rank, rank_source = matched.rank, "embedding"

        suit: Suit | None
        suit_source = "embedding"
        if matched is not None and match.similarity >= self.min_suit_similarity:
            suit = matched.suit
        else:
            suit, suit_source = prediction.suit, "classic"

        card = Card(rank, suit) if rank is not None and suit is not None else None
        return HybridRecognition(
            card=card,
            similarity=float(match.similarity),
            rank_source=rank_source,
            suit_source=suit_source,
            rank_confidence=float(prediction.rank_confidence),
        )


RECOGNIZERS = ("hybrid", "classic", "embedding")


def load_recognizer(
    kind: str,
    *,
    model: str | Path = Path("data/models/recognizer.joblib"),
    embedding_model: str | Path = Path("models/irfan-playing-card-detector"),
    threshold: float = 0.70,
    min_confidence: float | None = None,
):
    """Load ``"hybrid"`` (default in the scripts), ``"classic"`` or ``"embedding"``.

    Raises:
        SystemExit: with a hint when a model file is missing.
    """
    if kind not in RECOGNIZERS:
        raise ValueError(f"Unknown recogniser {kind!r}, expected one of {RECOGNIZERS}")
    classic = embedding = None
    if kind in ("hybrid", "classic"):
        from blackjack_vision.vision_classic.recognition import CardRecognizer

        if not Path(model).is_file():
            raise SystemExit(
                f"Classic model not found: {model}. Train it with scripts/train_recognizer.py "
                "or use --recognizer embedding."
            )
        classic = CardRecognizer.load(model)
        if min_confidence is not None:
            classic.min_rank_confidence = min_confidence
    if kind in ("hybrid", "embedding"):
        embedding = EmbeddingRecognizer.load(embedding_model, threshold=threshold)
    if kind == "classic":
        return classic
    if kind == "embedding":
        return embedding
    return HybridRecognizer(classic, embedding)
