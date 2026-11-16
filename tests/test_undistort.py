import cv2
import numpy as np
import pytest

from blackjack_vision.calibration.camera_model import CameraModel
from blackjack_vision.calibration.undistort import Undistorter

SIZE = (640, 480)
K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
BARREL = np.array([-0.3, 0.1, 0.0, 0.0, 0.0])


def camera(dist=BARREL) -> CameraModel:
    return CameraModel(K, dist, SIZE)


def test_without_distortion_the_image_is_unchanged():
    image = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    corrected = Undistorter(camera(np.zeros(5)))(image)
    assert corrected.shape == image.shape
    inner = (slice(5, -5), slice(5, -5))
    assert np.abs(corrected[inner].astype(int) - image[inner]).mean() < 1.0


def test_points_match_the_ideal_pinhole_projection():
    undistorter = Undistorter(camera(), alpha=0.0)
    rng = np.random.default_rng(1)
    world = np.column_stack([rng.uniform(-0.4, 0.4, (50, 2)), np.ones(50)])
    zero = np.zeros(3)
    distorted, _ = cv2.projectPoints(world, zero, zero, K, BARREL)
    ideal, _ = cv2.projectPoints(world, zero, zero, undistorter.new_camera_matrix, None)

    corrected = undistorter.undistort_points(distorted.reshape(-1, 2))

    assert np.allclose(corrected, ideal.reshape(-1, 2), atol=0.05)
    assert np.allclose(undistorter.distort_points(corrected), distorted.reshape(-1, 2), atol=0.05)


def test_image_remap_agrees_with_point_mapping():
    undistorter = Undistorter(camera(), alpha=0.0)
    raw_point = np.array([[100.0, 80.0]])  # near a corner: strong barrel distortion
    image = np.zeros((480, 640), np.uint8)
    cv2.circle(image, (100, 80), 3, 255, -1)

    corrected = undistorter(image)
    moments = cv2.moments(corrected, binaryImage=False)
    blob = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])

    assert np.linalg.norm(blob - undistorter.undistort_points(raw_point)[0]) < 1.0
    assert np.linalg.norm(blob - raw_point[0]) > 10.0  # the correction actually moved it


def test_alpha_one_keeps_more_field_of_view_than_alpha_zero():
    cropped = Undistorter(camera(), alpha=0.0).new_camera_matrix
    full = Undistorter(camera(), alpha=1.0).new_camera_matrix
    assert full[0, 0] < cropped[0, 0]  # shorter focal length = wider view


def test_wrong_image_size_is_rejected():
    with pytest.raises(ValueError, match="scaled"):
        Undistorter(camera())(np.zeros((240, 320, 3), np.uint8))


def test_invalid_alpha_is_rejected():
    with pytest.raises(ValueError):
        Undistorter(camera(), alpha=1.5)
