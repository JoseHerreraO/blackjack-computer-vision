import numpy as np
import pytest

from blackjack_vision.calibration.camera_model import CameraModel

K = np.array([[800.0, 0.0, 639.5], [0.0, 810.0, 359.5], [0.0, 0.0, 1.0]])
DIST = np.array([-0.21, 0.05, 0.001, -0.002, 0.01])


def make_model(**overrides) -> CameraModel:
    params = {"camera_matrix": K, "dist_coeffs": DIST, "image_size": (1280, 720), "rms": 0.2}
    params.update(overrides)
    return CameraModel(**params)


def test_json_roundtrip(tmp_path):
    model = make_model(name="webcam", metadata={"pattern": "9x6", "n_views": 18})
    loaded = CameraModel.load(model.save(tmp_path / "models" / "webcam.json"))

    assert np.allclose(loaded.camera_matrix, K)
    assert np.allclose(loaded.dist_coeffs, DIST)
    assert loaded.dist_coeffs.shape == (1, 5)
    assert loaded.image_size == (1280, 720)
    assert loaded.rms == pytest.approx(0.2)
    assert loaded.name == "webcam"
    assert loaded.metadata == {"pattern": "9x6", "n_views": 18}


def test_properties_and_field_of_view():
    model = make_model(camera_matrix=[[640.0, 0, 640], [0, 640.0, 360], [0, 0, 1]])
    assert (model.fx, model.fy, model.cx, model.cy) == (640.0, 640.0, 640.0, 360.0)
    horizontal, vertical = model.field_of_view
    assert horizontal == pytest.approx(90.0)
    assert vertical == pytest.approx(2 * np.degrees(np.arctan(360 / 640)))


def test_scaled_model_follows_resize_convention():
    half = make_model().scaled((640, 360))

    assert half.image_size == (640, 360)
    assert half.fx == pytest.approx(400.0)
    assert half.fy == pytest.approx(405.0)
    # The centre of a 1280x720 image (639.5, 359.5) is the centre of the 640x360 image.
    assert half.cx == pytest.approx(319.5)
    assert half.cy == pytest.approx(179.5)
    assert np.allclose(half.dist_coeffs, DIST)
    assert half.metadata["rescaled_from"] == [1280, 720]


def test_scaling_to_a_different_aspect_ratio_is_rejected():
    with pytest.raises(ValueError, match="aspect ratio"):
        make_model().scaled((640, 480))


@pytest.mark.parametrize(
    "overrides",
    [
        {"camera_matrix": np.eye(2)},
        {"dist_coeffs": np.zeros(3)},
        {"image_size": (0, 720)},
    ],
)
def test_invalid_parameters_raise(overrides):
    with pytest.raises(ValueError):
        make_model(**overrides)


def test_summary_mentions_key_values():
    text = make_model(name="glasses_high").summary()
    assert "glasses_high" in text
    assert "fx=800.00" in text
    assert "0.200 px" in text
