"""Tests of the chessboard calibration on synthetic data with a known ground truth."""

import cv2
import numpy as np
import pytest
from synthetic_board import render_homography

from blackjack_vision.calibration.chessboard import (
    ChessboardSpec,
    calibrate,
    collect_views,
    find_corners,
    subpixel_half_window,
)
from blackjack_vision.capture.sources import Frame

SPEC = ChessboardSpec((9, 6), square_size=25.0)
IMAGE_SIZE = (1280, 720)
K_TRUE = np.array([[900.0, 0.0, 650.0], [0.0, 900.0, 350.0], [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([-0.25, 0.08, 0.0005, -0.0008, 0.0])


# --------------------------------------------------------------------------- helpers


def render_board(homography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return render_homography(SPEC, homography, IMAGE_SIZE)


def homography(offset_x: float, offset_y: float, scale: float, tilt: float) -> np.ndarray:
    return np.array(
        [[scale, 0.12, offset_x], [0.05, scale, offset_y], [tilt, 0.0, 1.0]], np.float64
    )


def synthetic_views(n_views: int, noise_px: float, seed: int = 0) -> list[np.ndarray]:
    """Project the board with K_TRUE / DIST_TRUE from random poses that stay inside the image."""
    rng = np.random.default_rng(seed)
    object_points = SPEC.object_points().astype(np.float64)
    centre = object_points.mean(axis=0)
    views = []
    while len(views) < n_views:
        rvec = np.deg2rad(rng.uniform([-35, -35, -20], [35, 35, 20]))
        rotation, _ = cv2.Rodrigues(rvec)
        offset = rng.uniform([-120, -60, 380], [120, 60, 650])
        tvec = offset - rotation @ centre  # board centre ends up at `offset`
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K_TRUE, DIST_TRUE)
        points = projected.reshape(-1, 2) + rng.normal(0.0, noise_px, (SPEC.n_corners, 2))
        inside = (points > 5).all() and (points < np.array(IMAGE_SIZE) - 5).all()
        if inside:
            views.append(points.astype(np.float32))
    return views


# --------------------------------------------------------------------------- spec


def test_spec_parse_and_object_points():
    spec = ChessboardSpec.parse("9x6", square_size=25.0)
    points = spec.object_points()

    assert spec.inner_corners == (9, 6)
    assert spec.n_corners == 54
    assert points.shape == (54, 3)
    assert points.dtype == np.float32
    assert np.allclose(points[1] - points[0], (25, 0, 0))  # x varies fastest
    assert np.allclose(points[9] - points[0], (0, 25, 0))  # next row
    assert np.all(points[:, 2] == 0)


@pytest.mark.parametrize("text", ["9", "9x", "axb", "9x6x2"])
def test_spec_parse_rejects_bad_input(text):
    with pytest.raises(ValueError):
        ChessboardSpec.parse(text)


def test_spec_validation():
    with pytest.raises(ValueError):
        ChessboardSpec((1, 6))
    with pytest.raises(ValueError):
        ChessboardSpec((9, 6), square_size=0)


# --------------------------------------------------------------------------- corners


def test_find_corners_is_subpixel_accurate():
    image, expected = render_board(homography(300, 150, 1.4, 2e-4))
    corners = find_corners(image, SPEC)

    assert corners is not None
    assert corners.shape == (54, 2)
    # OpenCV may start from either end of the board: match each corner to its nearest truth.
    distances = np.linalg.norm(expected[:, None, :] - corners[None, :, :], axis=2).min(axis=1)
    assert distances.mean() < 0.15
    assert distances.max() < 0.4


def test_find_corners_returns_none_without_board():
    assert find_corners(np.full((480, 640, 3), 128, np.uint8), SPEC) is None


def test_subpixel_window_shrinks_for_small_squares():
    columns, rows = SPEC.inner_corners
    grid = np.array([(x, y) for y in range(rows) for x in range(columns)], np.float32)
    assert subpixel_half_window(grid * 60, SPEC) == 11  # large squares: OpenCV's usual 11 px
    assert subpixel_half_window(grid * 10, SPEC) == 4  # small squares: 40 % of the side
    assert subpixel_half_window(grid * 2, SPEC) == 2  # lower bound


# --------------------------------------------------------------------------- collection


def test_collect_views_skips_duplicates_and_empty_frames():
    first, _ = render_board(homography(300, 150, 1.4, 2e-4))
    second, _ = render_board(homography(450, 220, 1.2, -1e-4))
    blank = np.full_like(first, 128)
    images = [first, first, blank, second, second]
    frames = [Frame(image, index, index / 30) for index, image in enumerate(images)]
    seen = []

    views = collect_views(
        frames, SPEC, min_shift=10.0, keep_images=True, on_frame=lambda *a: seen.append(a[2])
    )

    assert len(views) == 2
    assert views.names == ["frame_00000", "frame_00003"]
    assert views.image_size == IMAGE_SIZE
    assert views.frames_processed == 5
    assert views.frames_with_board == 4
    assert len(views.images) == 2
    assert seen == [True, False, False, True, False]


def test_collect_views_stops_when_callback_returns_false():
    image, _ = render_board(homography(300, 150, 1.4, 2e-4))
    frames = (Frame(image, index, 0.0) for index in range(10))
    views = collect_views(frames, SPEC, on_frame=lambda *a: False)
    assert views.frames_processed == 1


def test_collect_views_rejects_mixed_resolutions():
    image, _ = render_board(homography(300, 150, 1.4, 2e-4))
    frames = [Frame(image, 0, 0.0), Frame(image[:400], 1, 0.1)]
    with pytest.raises(ValueError, match="calibrate each resolution separately"):
        collect_views(frames, SPEC)


# --------------------------------------------------------------------------- calibration


def test_calibration_recovers_known_intrinsics():
    result = calibrate(synthetic_views(20, noise_px=0.1), SPEC, IMAGE_SIZE, name="synthetic")
    camera = result.camera

    # Gaussian noise of 0.1 px per coordinate gives an RMS point distance of 0.1 * sqrt(2).
    assert result.rms == pytest.approx(0.1 * np.sqrt(2), abs=0.02)
    assert result.quality == "good"
    assert camera.fx == pytest.approx(900.0, rel=0.005)
    assert camera.fy == pytest.approx(900.0, rel=0.005)
    assert camera.cx == pytest.approx(650.0, abs=2.0)
    assert camera.cy == pytest.approx(350.0, abs=2.0)
    assert camera.dist_coeffs[0, 0] == pytest.approx(-0.25, abs=0.02)
    assert camera.image_size == IMAGE_SIZE
    assert camera.metadata["pattern"] == "9x6"
    assert len(result.per_view_errors) == 20
    assert result.std_intrinsics[0] < 2.0  # sigma(fx) in pixels
    assert not result.rejected_views
    assert "RMS reprojection error" in result.report()


def test_calibration_rejects_corrupted_views():
    views = synthetic_views(15, noise_px=0.1, seed=1)
    rng = np.random.default_rng(7)
    views[4] = views[4] + rng.normal(0.0, 4.0, views[4].shape).astype(np.float32)
    names = [f"img_{i:02d}.png" for i in range(15)]

    result = calibrate(views, SPEC, IMAGE_SIZE, view_names=names)

    assert result.rejected_views == ["img_04.png"]
    assert "img_04.png" not in result.view_names
    assert result.rms < 0.2
    assert result.camera.fx == pytest.approx(900.0, rel=0.01)


def test_calibration_needs_enough_valid_views():
    views = synthetic_views(3, noise_px=0.1)
    with pytest.raises(ValueError, match="at least"):
        calibrate(views[:2], SPEC, IMAGE_SIZE)
    with pytest.raises(ValueError, match="corners"):
        calibrate([views[0], views[1], views[2][:10]], SPEC, IMAGE_SIZE)
