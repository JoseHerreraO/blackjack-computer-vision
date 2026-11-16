"""Synthetic chessboard images with an exactly known geometry, shared by the calibration tests."""

from __future__ import annotations

import cv2
import numpy as np

from blackjack_vision.calibration.chessboard import ChessboardSpec

SQUARE_PX = 40
MARGIN_PX = 40
BACKGROUND = 128
_EXACT = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)


def board_texture(spec: ChessboardSpec, square_px: int = SQUARE_PX, margin: int = MARGIN_PX):
    """Printed board: ``(columns + 1) x (rows + 1)`` squares surrounded by a white margin."""
    columns, rows = spec.inner_corners
    height, width = (rows + 1) * square_px, (columns + 1) * square_px
    texture = np.full((height + 2 * margin, width + 2 * margin), 255, np.uint8)
    for row in range(rows + 1):
        for column in range(columns + 1):
            if (row + column) % 2 == 0:
                y0, x0 = margin + row * square_px, margin + column * square_px
                texture[y0 : y0 + square_px, x0 : x0 + square_px] = 0
    return texture


def texture_corners(spec: ChessboardSpec, square_px: int = SQUARE_PX, margin: int = MARGIN_PX):
    """Exact inner-corner positions in the texture (pixel centres at integer coordinates)."""
    columns, rows = spec.inner_corners
    xs = margin + square_px * np.arange(1, columns + 1) - 0.5
    ys = margin + square_px * np.arange(1, rows + 1) - 0.5
    return np.array([(x, y) for y in ys for x in xs], np.float64)


def render_homography(spec: ChessboardSpec, homography: np.ndarray, size: tuple[int, int]):
    """Board warped by a texture-to-image homography (no lens distortion).

    Returns:
        The BGR image and the exact image positions of the inner corners.
    """
    canvas = cv2.warpPerspective(
        board_texture(spec), homography, size, flags=cv2.INTER_LINEAR, borderValue=BACKGROUND
    )
    expected = cv2.perspectiveTransform(texture_corners(spec).reshape(-1, 1, 2), homography)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), expected.reshape(-1, 2)


def normalised_rays(camera_matrix, dist_coeffs, size: tuple[int, int]) -> np.ndarray:
    """Ideal normalised coordinates ``(x, y)`` seen by every pixel of a distorted camera image."""
    width, height = size
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
    if hasattr(cv2, "undistortPointsIter"):  # OpenCV 4.x
        rays = cv2.undistortPointsIter(pixels, camera_matrix, dist_coeffs, None, None, _EXACT)
    else:  # OpenCV 5.x merged the iterative variant into undistortPoints
        rays = cv2.undistortPoints(pixels, camera_matrix, dist_coeffs, criteria=_EXACT)
    return rays.reshape(height, width, 2)


def render_camera_view(
    spec: ChessboardSpec,
    rays: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """Board seen by a distorted camera at pose ``(rvec, tvec)`` with a single interpolation.

    Each output pixel looks along its ideal ray (``rays`` from :func:`normalised_rays`); the ray
    is intersected with the board plane and the texture is sampled there.
    """
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    plane_to_rays = np.column_stack([rotation[:, 0], rotation[:, 1], np.ravel(tvec)])
    scale = spec.square_size / SQUARE_PX
    offset = MARGIN_PX + SQUARE_PX - 0.5  # texture position of the first inner corner
    texture_to_plane = np.array(
        [[scale, 0.0, -offset * scale], [0.0, scale, -offset * scale], [0.0, 0.0, 1.0]]
    )
    rays_to_texture = np.linalg.inv(plane_to_rays @ texture_to_plane)
    height, width = rays.shape[:2]
    homogeneous = np.concatenate([rays.reshape(-1, 2), np.ones((height * width, 1))], axis=1)
    texture_points = homogeneous @ rays_to_texture.T
    texture_points = texture_points[:, :2] / texture_points[:, 2:3]
    map_x = texture_points[:, 0].reshape(height, width).astype(np.float32)
    map_y = texture_points[:, 1].reshape(height, width).astype(np.float32)
    image = cv2.remap(board_texture(spec), map_x, map_y, cv2.INTER_LINEAR, borderValue=BACKGROUND)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
