import itertools

import cv2
import numpy as np
import pytest

from blackjack_vision.vision_classic.corners import (
    aspect_ratio,
    order_corners,
    refine_corners,
    side_lengths,
)
from blackjack_vision.vision_classic.rectify import CARD_SIZE, card_homography, target_corners

UPRIGHT = np.array([[100, 50], [240, 50], [240, 250], [100, 250]], np.float32)


@pytest.mark.parametrize("permutation", list(itertools.permutations(range(4))))
def test_order_is_independent_of_the_input_order(permutation):
    assert np.allclose(order_corners(UPRIGHT[list(permutation)]), UPRIGHT)


def test_landscape_card_starts_with_a_short_side():
    landscape = np.array([[50, 100], [250, 100], [250, 240], [50, 240]], np.float32)
    ordered = order_corners(landscape)
    edges = side_lengths(ordered)
    assert edges[0] < edges[1]
    assert np.allclose(ordered[0], [50, 240])  # left short side, walked clockwise


def test_rotated_card_keeps_a_short_upper_edge_first():
    centre = UPRIGHT.mean(axis=0)
    rotation = cv2.getRotationMatrix2D(tuple(map(float, centre)), 30, 1.0)
    rotated = cv2.transform(UPRIGHT.reshape(-1, 1, 2), rotation).reshape(-1, 2)
    ordered = order_corners(rotated[::-1])
    edges = side_lengths(ordered)
    assert edges[0] < edges[1]
    assert (ordered[0, 1] + ordered[1, 1]) < (ordered[2, 1] + ordered[3, 1])
    assert aspect_ratio(ordered) == pytest.approx(200 / 140, rel=1e-4)


def test_order_rejects_wrong_number_of_points():
    with pytest.raises(ValueError):
        order_corners(UPRIGHT[:3])


def test_refinement_recovers_the_sharp_corners_of_a_rounded_card():
    mask = np.zeros((300, 340), np.uint8)
    x0, y0, x1, y1, r = 100, 50, 239, 249, 14  # the rounded rectangle covers [x0, x1] x [y0, y1]
    cv2.rectangle(mask, (x0 + r, y0), (x1 - r, y1), 255, -1)
    cv2.rectangle(mask, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x1 - r, y1 - r), (x0 + r, y1 - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(contours[0])
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    assert len(approx) == 4
    rough = order_corners(approx)
    sharp = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)

    refined = refine_corners(contours[0], rough)

    assert np.abs(rough - sharp).max() > 2.0  # approxPolyDP cuts the rounded corners
    assert np.abs(refined - sharp).max() < 0.6


def test_homography_maps_corners_to_the_rectified_frame():
    homography = card_homography(UPRIGHT)
    mapped = cv2.perspectiveTransform(UPRIGHT.reshape(-1, 1, 2), homography).reshape(-1, 2)
    assert np.allclose(mapped, target_corners(CARD_SIZE), atol=1e-3)
