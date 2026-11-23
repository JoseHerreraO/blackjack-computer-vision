import cv2
import numpy as np
import pytest
from synthetic_cards import card_texture

from blackjack_vision.vision_classic.sharpness import align_to, fuse, sharpness


@pytest.fixture
def card():
    texture, _ = card_texture()
    return cv2.resize(texture, (200, 300), interpolation=cv2.INTER_AREA)


def noisy(image, seed, sigma=12.0):
    rng = np.random.default_rng(seed)
    return np.clip(image + rng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)


def test_blur_lowers_sharpness(card):
    sharp = sharpness(card)
    assert sharpness(cv2.GaussianBlur(card, (7, 7), 2)) < 0.5 * sharp
    assert sharpness(cv2.blur(card, (1, 9))) < sharp  # vertical motion blur


def test_align_undoes_a_half_turn(card):
    rotated = cv2.rotate(card, cv2.ROTATE_180)
    np.testing.assert_array_equal(align_to(card, rotated), card)
    np.testing.assert_array_equal(align_to(card, card), card)


def test_align_rejects_different_sizes(card):
    with pytest.raises(ValueError):
        align_to(card, card[:100])


def test_fuse_with_k_1_is_the_sharpest(card):
    blurred = cv2.GaussianBlur(card, (9, 9), 3)
    np.testing.assert_array_equal(fuse([blurred, card, blurred], k=1), card)


def test_fuse_skips_blurred_crops_and_reduces_noise(card):
    crops = [noisy(card, seed) for seed in range(3)]
    crops += [cv2.GaussianBlur(card, (15, 15), 5) for _ in range(3)]  # motion-smeared frames
    crops[1] = cv2.rotate(crops[1], cv2.ROTATE_180)  # one frame came out upside down
    fused = fuse(crops, k=3)
    single_error = np.abs(crops[0].astype(float) - card).mean()
    fused_error = np.abs(fused.astype(float) - card).mean()
    assert fused_error < 0.75 * single_error


def test_fuse_needs_crops(card):
    with pytest.raises(ValueError):
        fuse([])
    with pytest.raises(ValueError):
        fuse([card], k=0)
