import cv2
import numpy as np
from synthetic_cards import mat

from blackjack_vision.vision_classic.mat import mat_fraction, mat_in_view

TEAL_BGR = (110, 104, 10)  # hue ~88, saturation ~232, like our mat


def test_teal_mat_is_in_view():
    image = np.full((640, 360, 3), 200, np.uint8)
    image[300:] = TEAL_BGR  # the lower half shows the mat
    assert mat_fraction(image) > 0.4
    assert mat_in_view(image)


def test_wall_and_person_are_not_the_mat():
    image = np.full((640, 360, 3), (190, 195, 200), np.uint8)  # grey wall
    cv2.rectangle(
        image, (100, 100), (260, 500), (100, 115, 95), -1
    )  # dull green jumper, low saturation
    assert mat_fraction(image) < 0.05
    assert not mat_in_view(image)


def test_threshold_and_hue_range_can_change():
    image = np.full((100, 100, 3), 200, np.uint8)
    image[:10] = TEAL_BGR  # 10 % of mat
    assert mat_in_view(image)  # default threshold 0.08
    assert not mat_in_view(image, min_fraction=0.2)
    assert not mat_in_view(image, hue_range=(0, 30))  # a red mat would not match


def test_synthetic_mat_of_the_tests_counts_as_a_mat():
    assert mat_in_view(mat((320, 240)), hue_range=(35, 105))
