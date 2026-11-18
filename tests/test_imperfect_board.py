"""Calibration with imperfect boards: SB detector, plane-fit errors and release-object method."""

import cv2
import numpy as np
import pytest
from synthetic_board import render_homography

from blackjack_vision.calibration.chessboard import (
    ChessboardSpec,
    calibrate,
    find_corners,
    orient_corners,
    plane_fit_errors,
)

SPEC = ChessboardSpec((9, 6), square_size=25.0)
IMAGE_SIZE = (1280, 720)
K_TRUE = np.array([[900.0, 0.0, 650.0], [0.0, 900.0, 350.0], [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([-0.2, 0.05, 0.0, 0.0, 0.0])


def bent_board(amplitude_mm: float) -> np.ndarray:
    """Board corners of a sheet that bulges like a curled page (z grows along both axes)."""
    points = SPEC.object_points().astype(np.float64)
    columns, rows = SPEC.inner_corners
    u = points[:, 0] / ((columns - 1) * SPEC.square_size)
    v = points[:, 1] / ((rows - 1) * SPEC.square_size)
    points[:, 2] = amplitude_mm * (np.sin(np.pi * u) + 0.5 * v**2)
    return points


def views_of(
    board: np.ndarray, n_views: int, noise_px: float, seed: int = 0, dist=DIST_TRUE
) -> list[np.ndarray]:
    """Project one physical board from random poses, all seen from the same side."""
    rng = np.random.default_rng(seed)
    centre = board.mean(axis=0)
    views = []
    while len(views) < n_views:
        rvec = np.deg2rad(rng.uniform([-30, -30, -20], [30, 30, 20]))
        rotation, _ = cv2.Rodrigues(rvec)
        tvec = rng.uniform([-120, -60, 380], [120, 60, 600]) - rotation @ centre
        projected, _ = cv2.projectPoints(board, rvec, tvec, K_TRUE, dist)
        points = projected.reshape(-1, 2) + rng.normal(0.0, noise_px, (len(board), 2))
        if (points > 5).all() and (points < np.array(IMAGE_SIZE) - 5).all():
            views.append(points.astype(np.float32))
    return views


def test_sb_detector_finds_accurate_corners():
    homography = np.array([[1.4, 0.15, 260.0], [0.05, 1.3, 120.0], [2e-4, 0.0, 1.0]])
    image, expected = render_homography(SPEC, homography, IMAGE_SIZE)
    corners = find_corners(image, SPEC, detector="sb")
    assert corners is not None and corners.shape == (SPEC.n_corners, 2)
    errors = np.linalg.norm(orient_corners(corners, SPEC) - orient_corners(expected, SPEC), axis=1)
    assert errors.mean() < 0.15
    assert errors.max() < 0.4


def test_unknown_detector_is_rejected():
    with pytest.raises(ValueError, match="detector"):
        find_corners(np.zeros((100, 100), np.uint8), SPEC, detector="magic")


def test_orient_corners_puts_the_upper_row_first():
    corners = views_of(SPEC.object_points().astype(np.float64), 1, 0.0)[0]
    upright = orient_corners(corners, SPEC)
    flipped = orient_corners(corners[::-1], SPEC)
    assert np.array_equal(upright, flipped)
    grid = upright.reshape(6, 9, 2)
    assert grid[0, :, 1].mean() < grid[-1, :, 1].mean()


def test_plane_fit_error_is_small_for_a_flat_board_and_grows_when_bent():
    lens_free = SPEC.object_points().astype(np.float64)
    flat_views = []
    for rvec, tvec in (([0.3, -0.2, 0.1], [-100, -60, 500]), ([-0.2, 0.3, 0.0], [-80, -70, 450])):
        projected, _ = cv2.projectPoints(
            lens_free, np.array(rvec, np.float64), np.array(tvec, np.float64), K_TRUE, None
        )
        flat_views.append(projected.reshape(-1, 2))
    bent = bent_board(3.0)
    bent_views = []
    for rvec, tvec in (([0.3, -0.2, 0.1], [-100, -60, 500]), ([-0.2, 0.3, 0.0], [-80, -70, 450])):
        projected, _ = cv2.projectPoints(
            bent, np.array(rvec, np.float64), np.array(tvec, np.float64), K_TRUE, None
        )
        bent_views.append(projected.reshape(-1, 2))
    assert plane_fit_errors(flat_views, SPEC).max() < 1e-4
    assert plane_fit_errors(bent_views, SPEC).min() > 0.3


def test_release_object_absorbs_a_bent_sheet():
    views = views_of(bent_board(1.5), 15, noise_px=0.05, seed=4)
    common = {"fix_k3": True, "zero_tangent": True, "reject_outliers": False}

    standard = calibrate(views, SPEC, IMAGE_SIZE, **common)
    released = calibrate(views, SPEC, IMAGE_SIZE, release_object=True, **common)

    assert standard.rms > 0.3
    assert released.rms < 0.12
    assert released.camera.metadata["release_object"] is True
    assert np.abs(released.board_points[:, 2]).max() > 0.5  # the bulge was estimated
    assert released.camera.fx == pytest.approx(K_TRUE[0, 0], rel=0.01)
    assert abs(released.camera.fx - K_TRUE[0, 0]) < abs(standard.camera.fx - K_TRUE[0, 0])
    assert "release object" in released.report()


def test_release_object_handles_views_numbered_from_either_end():
    views = views_of(bent_board(1.5), 12, noise_px=0.05, seed=5)
    mixed = [v if i % 2 else v[::-1].copy() for i, v in enumerate(views)]
    common = {
        "fix_k3": True,
        "zero_tangent": True,
        "reject_outliers": False,
        "release_object": True,
    }
    assert calibrate(mixed, SPEC, IMAGE_SIZE, **common).rms == pytest.approx(
        calibrate(views, SPEC, IMAGE_SIZE, **common).rms, abs=1e-6
    )


def test_max_plane_error_drops_the_bad_views():
    flat = SPEC.object_points().astype(np.float64)
    views = views_of(flat, 10, noise_px=0.05, seed=6, dist=None)  # no lens: plane error ~0
    noisy = [
        v + np.random.default_rng(i).normal(0, 2.0, v.shape).astype(np.float32)
        for i, v in enumerate(views[:3])
    ]
    names = [f"v{i}" for i in range(10)]
    result = calibrate(
        noisy + views[3:],
        SPEC,
        IMAGE_SIZE,
        view_names=names,
        max_plane_error=0.5,
        reject_outliers=False,
    )
    assert result.rejected_views == ["v0", "v1", "v2"]
    assert len(result.view_names) == 7
    with pytest.raises(ValueError, match="plane-fit"):
        calibrate(noisy + views[3:], SPEC, IMAGE_SIZE, max_plane_error=1e-6)
