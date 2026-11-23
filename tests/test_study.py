import csv
import runpy
from pathlib import Path

import numpy as np
import pytest
from synthetic_faces import card_face, photographed

from blackjack_vision.blackjack.cards import Card, Rank, Suit
from blackjack_vision.capture.sources import write_image
from blackjack_vision.vision_classic.recognition import RecognizerConfig
from blackjack_vision.vision_classic.study import (
    KERNELS,
    TemplateMatcher,
    intersection_kernel,
    kernel_study,
    pyramid_study,
    results_table,
    template_baseline,
    vocabulary_study,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "study_recognition.py"
CARDS = [Card(rank, suit) for rank in (Rank.TWO, Rank.SEVEN, Rank.KING) for suit in Suit]
BASE = RecognizerConfig(vocabulary_size=30, seed=3)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)

    def photos(copies):
        images = [photographed(card_face(card), rng) for card in CARDS for _ in range(copies)]
        return images, [card for card in CARDS for _ in range(copies)]

    return photos(3), photos(1)


def test_intersection_kernel():
    x = np.array([[1.0, 0.0, 2.0], [0.5, 0.5, 0.0]])
    assert intersection_kernel(x, x).tolist() == [[3.0, 0.5], [0.5, 1.0]]


def test_template_matcher_and_baseline(data):
    train, test = data
    matcher = TemplateMatcher().fit(*train)
    assert set(matcher.templates) == set(CARDS)
    upside_down = matcher.predict(np.rot90(card_face(CARDS[4]), 2).copy())
    assert upside_down == CARDS[4]  # both orientations are tried
    result = template_baseline(train, test)
    assert result.study == "baseline" and 0.0 <= result.card_accuracy <= 1.0
    assert result.rank_accuracy > 1 / 3  # better than chance between the 3 ranks
    with pytest.raises(RuntimeError):
        TemplateMatcher().predict(card_face(CARDS[0]))


def test_vocabulary_and_pyramid_studies(data):
    train, test = data
    vocab = vocabulary_study(train, test, [10, 30], BASE)
    assert [r.setting for r in vocab] == ["K=10", "K=30"]
    pyramid = pyramid_study(train, test, [(1,), (1, 2)], BASE)
    assert [r.setting for r in pyramid] == ["1x1", "1x1+2x2"]
    for result in vocab + pyramid:
        assert 0.0 <= result.rank_accuracy <= 1.0 and result.train_seconds > 0
        assert not np.isnan(result.card_accuracy)


def test_kernel_study_covers_every_kernel(data):
    train, test = data
    results = kernel_study(train, test, BASE)
    assert [r.setting for r in results] == list(KERNELS)
    assert all(np.isnan(r.suit_accuracy) for r in results)  # kernels only change the rank
    chi2 = results[0]
    assert chi2.rank_accuracy >= 0.8
    with pytest.raises(ValueError):
        kernel_study(train, test, BASE, ["poly"])


def test_results_table_marks_missing_values(data):
    train, test = data
    table = results_table(kernel_study(train, test, BASE, ["linear"]))
    assert table.splitlines()[0].startswith("| Study | Setting |")
    assert "| kernel | linear |" in table and "–" in table


def test_script_writes_csv_table_and_plot(tmp_path, data):
    train, test = data
    for folder, (images, cards) in (("train", train), ("test", test)):
        for i, (image, card) in enumerate(zip(images, cards, strict=True)):
            write_image(tmp_path / folder / card.code / f"{i:03d}.png", image)
    main = runpy.run_path(str(SCRIPT), run_name="study_cli")["main"]
    code = main(
        [
            "--data", str(tmp_path / "train"),
            "--test-data", str(tmp_path / "test"),
            "--quick",
            "--kernels", "chi2", "linear",
            "--levels", "1", "1,2",
            "--out", str(tmp_path / "out"),
        ]
    )  # fmt: skip
    assert code == 0
    with open(tmp_path / "out" / "results.csv", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert {row["study"] for row in rows} == {"baseline", "vocabulary", "pyramid", "kernel"}
    assert "separate recording" in (tmp_path / "out" / "results.md").read_text(encoding="utf-8")
    assert (tmp_path / "out" / "vocabulary.png").is_file()
