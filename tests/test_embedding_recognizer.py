import struct

import cv2
import numpy as np
import pytest

from blackjack_vision.blackjack.cards import Card, Rank, Suit
from blackjack_vision.vision_dl.embedding_recognizer import (
    EmbeddingRecognizer,
    card_from_name,
    preprocess,
    read_flat_index,
)


@pytest.mark.parametrize(
    ("name", "card"),
    [
        ("ace_of_spades", Card(Rank.ACE, Suit.SPADES)),
        ("ten_of_hearts", Card(Rank.TEN, Suit.HEARTS)),
        ("queen_of_diamonds", Card(Rank.QUEEN, Suit.DIAMONDS)),
        ("two_of_clubs", Card(Rank.TWO, Suit.CLUBS)),
        ("joker", None),
        ("eleven_of_cups", None),
    ],
)
def test_card_from_name(name, card):
    assert card_from_name(name) == card


def write_flat_index(path, vectors, metric=0):
    """Same layout as faiss.write_index for an IndexFlatIP (metric 0) or IndexFlatL2 (1)."""
    ntotal, dim = vectors.shape
    header = b"IxFI" + struct.pack("<iqqqBi", dim, ntotal, 1 << 20, 1 << 20, 1, metric)
    path.write_bytes(header + struct.pack("<q", ntotal * dim) + vectors.astype("<f4").tobytes())


def test_read_flat_index(tmp_path):
    vectors = np.random.default_rng(0).normal(size=(7, 5)).astype(np.float32)
    write_flat_index(tmp_path / "index.faiss", vectors)
    np.testing.assert_array_equal(read_flat_index(tmp_path / "index.faiss"), vectors)


def test_read_flat_index_rejects_other_files(tmp_path):
    (tmp_path / "x.faiss").write_bytes(b"not an index at all")
    with pytest.raises(ValueError):
        read_flat_index(tmp_path / "x.faiss")


def marked_card(top_left: int, bottom_right: int) -> np.ndarray:
    card = np.full((300, 200, 3), 128, np.uint8)
    card[0, 0] = top_left
    card[-1, -1] = bottom_right
    return card


KING = np.array([1.0, 0.0, 0.0], np.float32)
TWO = np.array([0.0, 1.0, 0.0], np.float32)


def fake_embed(images):
    """Embeds a card by its top-left pixel: 10 -> king of hearts, 200 -> two of spades."""
    out = []
    for image in images:
        value = int(image[0, 0, 0])
        vector = KING if value == 10 else TWO if value == 200 else np.array([0.6, 0.0, 0.8])
        out.append(vector)
    return np.array(out, np.float32)


@pytest.fixture
def recognizer():
    refs = np.stack([KING, TWO, KING])
    return EmbeddingRecognizer(
        fake_embed, refs, ["king_of_hearts", "two_of_spades", "king_of_hearts"]
    )


def test_recognise_names_the_nearest_reference(recognizer):
    result = recognizer.recognise(marked_card(10, 50))
    assert result.card == Card(Rank.KING, Suit.HEARTS)
    assert result.similarity == pytest.approx(1.0)
    assert result.known and not result.flipped


def test_recognise_tries_the_card_upside_down(recognizer):
    # Upright it looks like nothing, rotated 180° its top-left pixel is 200: two of spades.
    result = recognizer.recognise(marked_card(77, 200))
    assert result.card == Card(Rank.TWO, Suit.SPADES)
    assert result.flipped


def test_below_threshold_is_unknown(recognizer):
    result = recognizer.recognise(marked_card(77, 77))  # embeds to (0.6, 0, 0.8): sim 0.6
    assert result.card is None and not result.known
    assert result.similarity == pytest.approx(0.6)
    assert result.name == "king_of_hearts"  # best guess is still reported


def test_references_and_labels_must_match():
    with pytest.raises(ValueError):
        EmbeddingRecognizer(fake_embed, np.zeros((2, 3)), ["ace_of_spades"])


def test_preprocess_shape_and_normalisation():
    data = preprocess(np.full((300, 200, 3), 255, np.uint8))
    assert data.shape == (3, 224, 224) and data.dtype == np.float32
    # White in RGB is (1 - mean) / std for every channel.
    expected = (1.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np.testing.assert_allclose(data[:, 0, 0], expected, rtol=1e-5)
    bgr = np.zeros((300, 200, 3), np.uint8)
    bgr[..., 2] = 255  # pure red in BGR must land in the R channel
    assert preprocess(bgr)[0, 0, 0] > preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))[0, 0, 0]
