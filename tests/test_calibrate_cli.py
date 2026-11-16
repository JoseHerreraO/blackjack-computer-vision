"""End-to-end test of ``scripts/calibrate.py`` on synthetic photos of a distorted camera."""

import csv
import json
import runpy
from pathlib import Path

import cv2
import numpy as np
import pytest
from synthetic_board import normalised_rays, render_camera_view

from blackjack_vision.calibration.chessboard import ChessboardSpec
from blackjack_vision.capture.sources import write_image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate.py"
SPEC = ChessboardSpec((9, 6), square_size=25.0)
SIZE = (640, 480)
K_TRUE = np.array([[520.0, 0.0, 322.0], [0.0, 520.0, 238.0], [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([-0.28, 0.09, 0.0, 0.0, 0.0])


def board_poses(n_views: int, seed: int = 3):
    """Random poses whose whole board (with margin) stays inside the image."""
    rng = np.random.default_rng(seed)
    columns, rows = SPEC.inner_corners
    s = SPEC.square_size
    outline = np.array(
        [[-2 * s, -2 * s, 0], [(columns + 1) * s, -2 * s, 0],
         [(columns + 1) * s, (rows + 1) * s, 0], [-2 * s, (rows + 1) * s, 0]],
        np.float64,
    )  # fmt: skip
    centre = SPEC.object_points().astype(np.float64).mean(axis=0)
    poses = []
    while len(poses) < n_views:
        rvec = np.deg2rad(rng.uniform([-35, -35, -25], [35, 35, 25]))
        rotation, _ = cv2.Rodrigues(rvec)
        tvec = rng.uniform([-60, -40, 330], [60, 40, 480]) - rotation @ centre
        projected, _ = cv2.projectPoints(outline, rvec, tvec, K_TRUE, DIST_TRUE)
        corners = projected.reshape(-1, 2)
        if (corners > 2).all() and (corners < np.array(SIZE) - 2).all():
            poses.append((rvec, tvec))
    return poses


@pytest.fixture(scope="module")
def synthetic_photos(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("views")
    rays = normalised_rays(K_TRUE, DIST_TRUE, SIZE)
    for index, (rvec, tvec) in enumerate(board_poses(12)):
        write_image(folder / f"view_{index:02d}.png", render_camera_view(SPEC, rays, rvec, tvec))
    return folder


def run_cli(*args: str) -> int:
    namespace = runpy.run_path(str(SCRIPT), run_name="calibrate_cli")
    return namespace["main"](list(args))


def test_cli_recovers_the_synthetic_camera(synthetic_photos, tmp_path):
    out = tmp_path / "synthetic.json"
    debug = tmp_path / "debug"

    exit_code = run_cli(
        "--source", str(synthetic_photos), "--pattern", "9x6", "--square-size", "25",
        "--name", "synthetic", "--out", str(out), "--debug-dir", str(debug),
    )  # fmt: skip

    assert exit_code == 0
    model = json.loads(out.read_text(encoding="utf-8"))
    matrix = np.array(model["camera_matrix"])
    assert model["image_size"] == {"width": 640, "height": 480}
    assert matrix[0, 0] == pytest.approx(520.0, rel=0.01)
    assert matrix[1, 1] == pytest.approx(520.0, rel=0.01)
    assert matrix[0, 2] == pytest.approx(322.0, abs=3.0)
    assert matrix[1, 2] == pytest.approx(238.0, abs=3.0)
    assert model["dist_coeffs"][0] == pytest.approx(-0.28, abs=0.03)
    assert model["rms"] < 0.3
    assert model["metadata"]["n_views"] >= 10

    with (debug / "per_view_errors.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert len(list(debug.glob("corners_view_*.jpg"))) == 12
    assert len(list(debug.glob("undistorted_view_*.jpg"))) == 3


def test_cli_fails_cleanly_without_views(tmp_path):
    for index in range(3):
        write_image(tmp_path / f"blank_{index}.png", np.full((120, 160, 3), 128, np.uint8))
    exit_code = run_cli("--source", str(tmp_path), "--out", str(tmp_path / "x.json"))
    assert exit_code == 1
    assert not (tmp_path / "x.json").exists()
