import cv2
import numpy as np
import pytest
from synthetic_faces import card_face, photographed

from blackjack_vision.blackjack.cards import Card, Rank, Suit
from blackjack_vision.capture.sources import write_image
from blackjack_vision.vision_classic.features import (
    BowEncoder,
    FeatureConfig,
    SamplingMode,
    SiftExtractor,
    Vocabulary,
)
from blackjack_vision.vision_classic.recognition import (
    CardRecognizer,
    RecognizerConfig,
    TemporalVoter,
    load_dataset,
    red_ink_fraction,
)

RANKS = [Rank.ACE, Rank.TWO, Rank.THREE, Rank.SEVEN, Rank.TEN, Rank.KING]
CARDS = [Card(rank, suit) for rank in RANKS for suit in Suit]
FAST = RecognizerConfig(vocabulary_size=80, seed=1)


def _photos(per_card: int, seed: int) -> tuple[list[np.ndarray], list[Card]]:
    rng = np.random.default_rng(seed)
    pairs = [(photographed(card_face(card), rng), card) for card in CARDS for _ in range(per_card)]
    return [image for image, _ in pairs], [card for _, card in pairs]


@pytest.fixture(scope="module")
def recognizer() -> CardRecognizer:
    images, cards = _photos(per_card=3, seed=0)
    return CardRecognizer(FAST).fit(images, cards)


def test_recognises_unseen_photos(recognizer):
    images, cards = _photos(per_card=2, seed=1)
    predictions = recognizer.predict_many(images)

    rank_accuracy = np.mean([p.rank == c.rank for p, c in zip(predictions, cards, strict=True)])
    suit_accuracy = np.mean([p.suit == c.suit for p, c in zip(predictions, cards, strict=True)])
    colour_accuracy = np.mean(
        [p.suit.is_red == c.suit.is_red for p, c in zip(predictions, cards, strict=True)]
    )
    assert rank_accuracy >= 0.95
    assert colour_accuracy == 1.0  # the colour filter never mixes red and black suits
    assert suit_accuracy >= 0.95


def test_card_without_a_readable_suit_symbol_has_no_suit(recognizer):
    blank = np.full((300, 200, 3), 245, np.uint8)
    prediction = recognizer.predict(blank)
    assert prediction.suit is None
    assert prediction.card is None


def test_upside_down_cards_get_the_same_rank(recognizer):
    for card in (Card(Rank.TWO, Suit.HEARTS), Card(Rank.KING, Suit.SPADES)):
        face = card_face(card)
        upright, flipped = recognizer.predict_many([face, cv2.rotate(face, cv2.ROTATE_180)])
        assert upright.card == flipped.card == card
        assert upright.rank_margin > 0


def test_saved_model_predicts_the_same(recognizer, tmp_path):
    images, _ = _photos(per_card=1, seed=2)
    path = recognizer.save(tmp_path / "models" / "recognizer.joblib")
    loaded = CardRecognizer.load(path)
    assert loaded.config == recognizer.config
    original = [(p.rank, p.suit) for p in recognizer.predict_many(images[:8])]
    assert [(p.rank, p.suit) for p in loaded.predict_many(images[:8])] == original


def test_untrained_recognizer_refuses_to_predict():
    with pytest.raises(RuntimeError):
        CardRecognizer().predict(card_face(Card(Rank.ACE, Suit.SPADES)))


def test_training_needs_several_ranks_and_suits():
    faces = [card_face(Card(rank, Suit.SPADES)) for rank in (Rank.ACE, Rank.TWO)]
    with pytest.raises(ValueError, match="two suits"):
        CardRecognizer(FAST).fit(faces, [Card(Rank.ACE, Suit.SPADES), Card(Rank.TWO, Suit.SPADES)])
    with pytest.raises(ValueError, match="labels"):
        CardRecognizer(FAST).fit(faces, [Card(Rank.ACE, Suit.SPADES)])


def test_red_ink_is_measured_on_the_index_corners():
    assert red_ink_fraction(card_face(Card(Rank.FIVE, Suit.HEARTS))) > 0.8
    assert red_ink_fraction(card_face(Card(Rank.FIVE, Suit.CLUBS))) < 0.05
    # The red/yellow frame of a black king is in the middle, not in the corners.
    assert red_ink_fraction(card_face(Card(Rank.KING, Suit.SPADES))) < 0.05


def test_dense_extractor_gives_a_fixed_number_of_descriptors():
    extractor = SiftExtractor(FeatureConfig(step=10, patch_size=20))
    blank = np.full((300, 200, 3), 255, np.uint8)
    points, descriptors = extractor(blank)
    rich_points, rich_descriptors = extractor(card_face(Card(Rank.TEN, Suit.CLUBS)))
    assert len(points) == len(rich_points) == len(descriptors) == len(rich_descriptors) > 0
    assert descriptors.shape[1] == 128
    assert np.allclose((rich_descriptors**2).sum(axis=1), 1.0, atol=1e-3)  # RootSIFT


def test_keypoint_extractor_resizes_and_may_find_nothing():
    extractor = SiftExtractor(FeatureConfig(mode=SamplingMode.KEYPOINTS))
    points, descriptors = extractor(np.full((150, 100), 255, np.uint8))
    assert points.shape == (0, 2) and descriptors.shape == (0, 128)
    points, _ = extractor(cv2.resize(card_face(Card(Rank.NINE, Suit.SPADES)), (400, 600)))
    assert len(points) > 0
    assert points[:, 0].max() < 200 and points[:, 1].max() < 300  # in the 200x300 frame


def test_vocabulary_assigns_the_nearest_word():
    words = np.eye(4, 128, dtype=np.float32)
    vocabulary = Vocabulary(words)
    queries = words[[2, 0, 3]] + 0.01
    assert vocabulary.assign(queries).tolist() == [2, 0, 3]
    assert vocabulary.assign(np.zeros((0, 128), np.float32)).shape == (0,)


def test_vocabulary_fit_finds_the_clusters():
    rng = np.random.default_rng(0)
    centres = rng.uniform(0, 1, (3, 128)).astype(np.float32)
    data = np.concatenate([c + rng.normal(0, 0.01, (200, 128)) for c in centres]).astype(np.float32)
    vocabulary = Vocabulary.fit(data, 3, seed=0)
    distances = np.linalg.norm(vocabulary.words[:, None] - centres[None], axis=2)
    assert distances.min(axis=0).max() < 0.05
    with pytest.raises(ValueError):
        Vocabulary.fit(data[:2], 3)


def test_spatial_pyramid_places_words_in_their_cell():
    vocabulary = Vocabulary(np.eye(2, 128, dtype=np.float32))
    encoder = BowEncoder(vocabulary, levels=(1, 2), image_size=(200, 300))
    points = np.array([[10, 10], [190, 290]], np.float32)  # top-left and bottom-right cells
    descriptors = np.eye(2, 128, dtype=np.float32)  # word 0, word 1
    histogram = encoder.encode(points, descriptors)

    assert encoder.dimension == histogram.size == 2 * (1 + 4)
    assert histogram[:2].tolist() == [0.5, 0.5]  # level 1: whole card
    level2 = histogram[2:].reshape(4, 2)  # cells in row-major order
    assert level2[0].tolist() == [0.5, 0.0]
    assert level2[3].tolist() == [0.0, 0.5]
    assert level2[1:3].sum() == 0


def test_temporal_voter_follows_the_majority():
    voter = TemporalVoter(window=3)
    assert voter.vote("a") is None
    assert voter.add("a", "7H") == ("7H", 1.0)
    voter.add("a", "1H")
    assert voter.add("a", "7H") == ("7H", pytest.approx(2 / 3))
    voter.add("a", "TH")
    voter.add("a", "TH")  # the window only keeps the last three: 7H, TH, TH
    assert voter.vote("a") == ("TH", pytest.approx(2 / 3))
    voter.forget("a")
    assert voter.vote("a") is None
    with pytest.raises(ValueError):
        TemporalVoter(window=0)


def test_load_dataset_reads_card_folders(tmp_path):
    for code in ("AS", "TH"):
        for i in range(2):
            write_image(tmp_path / code / f"{i:03d}.png", card_face(Card.parse(code)))
    write_image(tmp_path / "unsorted" / "x.png", card_face(Card.parse("2C")))
    (tmp_path / "TH" / "notes.txt").write_text("not an image")

    images, cards, paths = load_dataset(tmp_path)

    assert [card.code for card in cards] == ["AS", "AS", "TH", "TH"]
    assert len(images) == len(paths) == 4
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "unsorted")


def test_config_round_trips_through_a_dict():
    config = RecognizerConfig(vocabulary_size=50, pyramid_levels=(1, 3), colour_filter=False)
    assert RecognizerConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        RecognizerConfig(index_region=(0.8, 0.2))
