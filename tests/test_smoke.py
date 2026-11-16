import importlib

import pytest

SUBPACKAGES = [
    "capture",
    "calibration",
    "detection_tracking",
    "vision_classic",
    "security",
    "blackjack",
    "ui",
]


def test_package_imports():
    import blackjack_vision  # noqa: F401


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(f"blackjack_vision.{name}")
