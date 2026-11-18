"""Synthetic table scenes with playing cards at exactly known positions, for the detection tests."""

from __future__ import annotations

import cv2
import numpy as np

CARD_TEXTURE_SIZE = (252, 352)  # 63 x 88 mm at 4 px/mm
CORNER_RADIUS_PX = 12  # 3 mm
MAT_BGR = (40, 110, 45)
MARK_BGR = (0, 0, 220)


def card_texture() -> tuple[np.ndarray, np.ndarray]:
    """A white card with rounded corners and an asymmetric pattern.

    The top-left index is a red block, the bottom-right one a black block, and a dark pip sits
    in the centre, so the rectified card reveals its orientation.

    Returns:
        BGR texture and its alpha mask (``float32`` in [0, 1], 0 outside the rounded corners).
    """
    width, height = CARD_TEXTURE_SIZE
    texture = np.full((height, width, 3), 250, np.uint8)
    cv2.rectangle(texture, (16, 16), (52, 80), MARK_BGR, -1)
    cv2.rectangle(texture, (width - 53, height - 81), (width - 17, height - 17), (20, 20, 20), -1)
    cv2.circle(texture, (width // 2, height // 2), 40, (30, 30, 30), -1)

    alpha = np.zeros((height, width), np.uint8)
    r = CORNER_RADIUS_PX
    cv2.rectangle(alpha, (r, 0), (width - 1 - r, height - 1), 255, -1)
    cv2.rectangle(alpha, (0, r), (width - 1, height - 1 - r), 255, -1)
    for cx, cy in (
        (r, r),
        (width - 1 - r, r),
        (width - 1 - r, height - 1 - r),
        (r, height - 1 - r),
    ):
        cv2.circle(alpha, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return texture, alpha.astype(np.float32) / 255.0


def texture_corners() -> np.ndarray:
    """Outer corners of the texture (pixel centres at integer coordinates), TL, TR, BR, BL."""
    width, height = CARD_TEXTURE_SIZE
    return np.array(
        [[-0.5, -0.5], [width - 0.5, -0.5], [width - 0.5, height - 0.5], [-0.5, height - 0.5]],
        np.float32,
    )


def placed_corners(
    center: tuple[float, float], height_px: float, angle_deg: float = 0.0, tilt: float = 0.0
) -> np.ndarray:
    """Image corners (TL, TR, BR, BL of the card) of a card placed on the table.

    Args:
        center: Card centre in the image.
        height_px: Length of the long side in pixels.
        angle_deg: Clockwise rotation on screen.
        tilt: Perspective foreshortening: the top edge is ``1 - tilt`` times the bottom edge.
    """
    width_px = height_px * CARD_TEXTURE_SIZE[0] / CARD_TEXTURE_SIZE[1]
    half_w, half_h = width_px / 2.0, height_px / 2.0
    top = half_w * (1.0 - tilt)
    local = np.array([[-top, -half_h], [top, -half_h], [half_w, half_h], [-half_w, half_h]])
    theta = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (local @ rotation.T + np.asarray(center, np.float64)).astype(np.float32)


def mat(size: tuple[int, int] = (640, 480), seed: int = 0, noise: float = 4.0) -> np.ndarray:
    """Plain green mat with a little sensor noise."""
    width, height = size
    rng = np.random.default_rng(seed)
    image = np.empty((height, width, 3), np.float32)
    image[:] = MAT_BGR
    image += rng.normal(0.0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def paste_card(scene: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw a card whose outer corners land on ``corners`` (TL, TR, BR, BL), anti-aliased."""
    texture, alpha = card_texture()
    homography = cv2.getPerspectiveTransform(texture_corners(), np.asarray(corners, np.float32))
    height, width = scene.shape[:2]
    warped = cv2.warpPerspective(texture, homography, (width, height), flags=cv2.INTER_LINEAR)
    weight = cv2.warpPerspective(alpha, homography, (width, height), flags=cv2.INTER_LINEAR)
    weight = weight[..., None]
    blended = scene.astype(np.float32) * (1.0 - weight) + warped.astype(np.float32) * weight
    return np.clip(blended, 0, 255).astype(np.uint8)


def scene_with_cards(placements: list[np.ndarray], size=(640, 480), seed: int = 0) -> np.ndarray:
    """Mat with one card per entry of ``placements`` (corner arrays from :func:`placed_corners`)."""
    scene = mat(size, seed)
    for corners in placements:
        scene = paste_card(scene, corners)
    return scene


def match_corners(detected: np.ndarray, expected: np.ndarray) -> float:
    """Largest distance between each expected corner and the nearest detected one."""
    distances = np.linalg.norm(expected[:, None, :] - detected[None, :, :], axis=2)
    return float(distances.min(axis=1).max())
