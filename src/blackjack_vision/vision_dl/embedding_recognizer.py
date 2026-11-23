"""Card recognition by nearest neighbour in an embedding space.

Uses the recognizer of ``irfanhossainsust/playing-card-detector`` (Hugging Face, AGPL-3.0,
downloaded with ``scripts/download_card_recognizer.py``): an EfficientNet-B0 that maps a 224x224
card face to a 256-d unit vector, trained with a CosFace loss on photos of many deck designs with
blur, glare, perspective and low-resolution augmentations, plus 7,624 reference vectors of the 52
cards (and the joker). A rectified card from the classical detector is embedded, compared by
cosine similarity with every reference, and named after the closest one. The card is also tried
rotated by 180° (the detector cannot tell which short side is the top) and the better view wins.
Below ``threshold`` the result is *unknown* rather than a guess.

The reference vectors are stored as a FAISS ``IndexFlatIP``. That format is a short header plus
the raw float32 matrix, so it is read directly with NumPy and FAISS is not needed.

PyTorch and torchvision are only imported by :meth:`EmbeddingRecognizer.load`. The matching logic
can be built from any embedding function, which is how the tests run without the model.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from blackjack_vision.blackjack.cards import Card, Rank, Suit

DEFAULT_MODEL_DIR = Path("models/irfan-playing-card-detector")
DEFAULT_THRESHOLD = 0.70
INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

_RANKS = {
    "ace": Rank.ACE,
    "two": Rank.TWO,
    "three": Rank.THREE,
    "four": Rank.FOUR,
    "five": Rank.FIVE,
    "six": Rank.SIX,
    "seven": Rank.SEVEN,
    "eight": Rank.EIGHT,
    "nine": Rank.NINE,
    "ten": Rank.TEN,
    "jack": Rank.JACK,
    "queen": Rank.QUEEN,
    "king": Rank.KING,
}
_SUITS = {
    "clubs": Suit.CLUBS,
    "diamonds": Suit.DIAMONDS,
    "hearts": Suit.HEARTS,
    "spades": Suit.SPADES,
}

EmbedFunction = Callable[[Sequence[np.ndarray]], np.ndarray]


def card_from_name(name: str) -> Card | None:
    """``"queen_of_hearts"`` -> ``Card(QUEEN, HEARTS)``. ``None`` for the joker or unknown names."""
    rank, _, suit = name.lower().partition("_of_")
    if rank in _RANKS and suit in _SUITS:
        return Card(_RANKS[rank], _SUITS[suit])
    return None


def read_flat_index(path: str | Path) -> np.ndarray:
    """Vectors of a FAISS ``IndexFlatIP`` / ``IndexFlatL2`` file as an ``(ntotal, d)`` array."""
    data = Path(path).read_bytes()
    if data[:4] not in (b"IxFI", b"IxF2"):
        raise ValueError(f"{path} is not a FAISS flat index (header {data[:4]!r})")
    dim, ntotal = struct.unpack_from("<iq", data, 4)
    offset = 4 + 4 + 8 + 8 + 8 + 1  # fourcc, d, ntotal, two unused int64, is_trained
    (metric,) = struct.unpack_from("<i", data, offset)
    offset += 4
    if metric > 1:  # metric types beyond L2/IP store an extra float argument
        offset += 4
    (count,) = struct.unpack_from("<q", data, offset)
    offset += 8
    if count != dim * ntotal:
        raise ValueError(f"{path}: expected {dim * ntotal} floats, found {count}")
    vectors = np.frombuffer(data, dtype="<f4", count=count, offset=offset)
    return vectors.reshape(ntotal, dim).copy()


@dataclass(frozen=True)
class Recognition:
    """Result of :meth:`EmbeddingRecognizer.recognise`.

    Attributes:
        card: The card, or ``None`` when the best match is below the threshold (or a joker).
        similarity: Cosine similarity of the best match, in ``[-1, 1]``.
        name: Name of the best reference, e.g. ``"king_of_hearts"``, even when below threshold.
        flipped: The best match was the card rotated by 180°.
    """

    card: Card | None
    similarity: float
    name: str
    flipped: bool = False

    @property
    def known(self) -> bool:
        return self.card is not None


class EmbeddingRecognizer:
    """Names a rectified card by its nearest reference embedding.

    Args:
        embed: Maps a list of BGR card images to an ``(N, d)`` array of unit vectors.
        references: ``(M, d)`` unit reference vectors.
        labels: Card name of every reference row (``"ace_of_spades"``...).
        threshold: Minimum cosine similarity to name a card.
    """

    def __init__(
        self,
        embed: EmbedFunction,
        references: np.ndarray,
        labels: Sequence[str],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        references = np.asarray(references, dtype=np.float32)
        if references.ndim != 2 or len(references) != len(labels):
            raise ValueError("references must be (M, d) with one label per row")
        self._embed = embed
        self.references = references
        self.labels = list(labels)
        self.threshold = threshold

    @classmethod
    def load(
        cls,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cpu",
    ) -> EmbeddingRecognizer:
        """Load the embedder and the reference database downloaded into ``model_dir``."""
        model_dir = Path(model_dir)
        weights = model_dir / "models/card_embedder.pt"
        if not weights.is_file():
            raise FileNotFoundError(
                f"Recognizer not found in {model_dir}. Run scripts/download_card_recognizer.py first."
            )
        try:
            import torch
            from torch import nn
            from torchvision.models import efficientnet_b0
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError(
                'EmbeddingRecognizer needs PyTorch and torchvision: pip install -e ".[yolo]"'
            ) from error

        # The checkpoint stores the architecture name, the embedding size and a state dict.
        # The file is checked by its SHA-256 when downloaded (it is a pickle).
        checkpoint = torch.load(weights, map_location=device, weights_only=False)
        if checkpoint.get("arch") != "efficientnet_b0":
            raise ValueError(f"Unsupported embedder architecture {checkpoint.get('arch')!r}")
        dim = int(checkpoint["dim"])

        class EmbeddingNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = efficientnet_b0(weights=None)
                self.backbone.classifier = nn.Identity()
                self.head = nn.Sequential(
                    nn.Dropout(0.2), nn.Linear(1280, dim), nn.BatchNorm1d(dim)
                )

            def forward(self, x):
                return nn.functional.normalize(self.head(self.backbone(x)), dim=1)

        net = EmbeddingNet()
        net.load_state_dict(checkpoint["state_dict"])
        net.eval().to(device)

        def embed(images: Sequence[np.ndarray]) -> np.ndarray:
            batch = np.stack([preprocess(image) for image in images])
            with torch.no_grad():
                return net(torch.from_numpy(batch).to(device)).cpu().numpy()

        references = read_flat_index(model_dir / "card_database/embeddings/card_embeddings.faiss")
        labels = json.loads((model_dir / "card_database/embeddings/labels.json").read_text())
        return cls(embed, references, labels["labels"], threshold)

    def recognise(self, card_image: np.ndarray) -> Recognition:
        """Name a rectified BGR card image (any size, portrait)."""
        views = [card_image, cv2.rotate(card_image, cv2.ROTATE_180)]
        similarities = self._embed(views) @ self.references.T  # (2, M)
        view, index = np.unravel_index(int(np.argmax(similarities)), similarities.shape)
        similarity = float(similarities[view, index])
        name = self.labels[index]
        card = card_from_name(name) if similarity >= self.threshold else None
        return Recognition(card=card, similarity=similarity, name=name, flipped=bool(view))


def preprocess(image: np.ndarray) -> np.ndarray:
    """BGR card image -> normalised ``(3, 224, 224)`` float32 RGB tensor data."""
    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
