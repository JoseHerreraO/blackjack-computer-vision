"""Pinhole camera model with lens distortion (Tema 2): intrinsics, distortion and persistence.

Conventions (the same as OpenCV):

* ``K = [[fx, s, cx], [0, fy, cy], [0, 0, 1]]`` in pixels, with pixel centres at integer
  coordinates.
* ``dist_coeffs = (k1, k2, p1, p2[, k3[, k4, k5, k6, ...]])``: radial ``k`` and tangential ``p``
  terms applied to normalised (dimensionless) coordinates, so they do not depend on resolution.
* ``image_size = (width, height)``: the resolution the model is valid for. A pure rescaling of
  the image only rescales ``K`` (see :meth:`CameraModel.scaled`); a crop needs a new calibration.

The glasses stream offers 720x1280, 504x896 and 360x640 (portrait, same aspect ratio), so a model
calibrated at one resolution can be rescaled to another one *if* the lower resolutions are pure
downscales of the same field of view, which should be checked once by calibrating both.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VALID_DISTORTION_LENGTHS: tuple[int, ...] = (4, 5, 8, 12, 14)
_ASPECT_TOLERANCE = 0.01
# A JSON list of numbers spread over several lines (raw newlines never occur inside strings).
_NUMERIC_LIST = re.compile(r"\[\n\s*([-+0-9.eE,\s]+?)\s*\]")


@dataclass(eq=False)
class CameraModel:
    """Intrinsic parameters of a camera at a given resolution.

    Attributes:
        camera_matrix: Intrinsic matrix ``K`` (3x3, pixels).
        dist_coeffs: Distortion coefficients, shape ``(1, N)`` with ``N`` in 4, 5, 8, 12 or 14.
        image_size: ``(width, height)`` in pixels that ``camera_matrix`` refers to.
        rms: RMS reprojection error of the calibration in pixels, if known.
        name: Identifier, e.g. ``"glasses_high"`` or ``"webcam"``.
        metadata: JSON-serialisable details of how the model was obtained.
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    rms: float | None = None
    name: str = "camera"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        if self.camera_matrix.shape != (3, 3):
            raise ValueError(f"camera_matrix must be 3x3, got shape {self.camera_matrix.shape}")
        self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(1, -1)
        if self.dist_coeffs.shape[1] not in VALID_DISTORTION_LENGTHS:
            raise ValueError(
                f"dist_coeffs must have {VALID_DISTORTION_LENGTHS} elements, "
                f"got {self.dist_coeffs.shape[1]}"
            )
        width, height = (int(value) for value in self.image_size)
        if width <= 0 or height <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        self.image_size = (width, height)
        if self.rms is not None:
            self.rms = float(self.rms)

    @property
    def fx(self) -> float:
        """Focal length along x, in pixels."""
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        """Focal length along y, in pixels."""
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        """Principal point x coordinate, in pixels."""
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        """Principal point y coordinate, in pixels."""
        return float(self.camera_matrix[1, 2])

    @property
    def field_of_view(self) -> tuple[float, float]:
        """Horizontal and vertical field of view in degrees (pinhole approximation)."""
        width, height = self.image_size
        horizontal = math.degrees(2.0 * math.atan(width / (2.0 * self.fx)))
        vertical = math.degrees(2.0 * math.atan(height / (2.0 * self.fy)))
        return horizontal, vertical

    def scaled(self, new_size: tuple[int, int]) -> CameraModel:
        """Return the model for the same camera when its images are resized to ``new_size``.

        Focal lengths scale with the image and the principal point follows the pixel-centre
        convention of ``cv2.resize``: ``c' = (c + 0.5) * s - 0.5``. Distortion coefficients act on
        normalised coordinates and are unchanged.

        Raises:
            ValueError: if the aspect ratio changes, which means the new image is a crop rather
                than a rescaled copy and the camera has to be recalibrated.
        """
        width, height = self.image_size
        new_width, new_height = (int(value) for value in new_size)
        scale_x, scale_y = new_width / width, new_height / height
        if abs(scale_x - scale_y) > _ASPECT_TOLERANCE * max(scale_x, scale_y):
            raise ValueError(
                f"Cannot rescale {width}x{height} to {new_width}x{new_height}: the aspect ratio "
                "changes (crop), so the camera must be recalibrated at that resolution"
            )
        matrix = self.camera_matrix.copy()
        matrix[0, :2] *= scale_x
        matrix[1, 1] *= scale_y
        matrix[0, 2] = (matrix[0, 2] + 0.5) * scale_x - 0.5
        matrix[1, 2] = (matrix[1, 2] + 0.5) * scale_y - 0.5
        return CameraModel(
            camera_matrix=matrix,
            dist_coeffs=self.dist_coeffs.copy(),
            image_size=(new_width, new_height),
            rms=None if self.rms is None else self.rms * scale_x,
            name=self.name,
            metadata={**self.metadata, "rescaled_from": [width, height]},
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (inverse of :meth:`from_dict`)."""
        width, height = self.image_size
        return {
            "name": self.name,
            "image_size": {"width": width, "height": height},
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.ravel().tolist(),
            "rms": self.rms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraModel:
        """Build a model from the dictionary produced by :meth:`to_dict`."""
        size = data["image_size"]
        return cls(
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64),
            image_size=(size["width"], size["height"]),
            rms=data.get("rms"),
            name=data.get("name", "camera"),
            metadata=dict(data.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        """Write the model as human-readable JSON (easy to review in a pull request)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2)
        # One matrix row / coefficient vector per line instead of one number per line.
        text = _NUMERIC_LIST.sub(
            lambda match: "[" + ", ".join(item.strip() for item in match[1].split(",")) + "]", text
        )
        path.write_text(text + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> CameraModel:
        """Read a model saved with :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        """Multi-line, human-readable description of the model."""
        width, height = self.image_size
        horizontal, vertical = self.field_of_view
        names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tx", "ty")
        coefficients = ", ".join(
            f"{name}={value:+.4f}"
            for name, value in zip(names, self.dist_coeffs.ravel(), strict=False)
        )
        lines = [
            f"Camera '{self.name}' at {width}x{height} px",
            f"  fx={self.fx:.2f}  fy={self.fy:.2f}  cx={self.cx:.2f}  cy={self.cy:.2f}",
            f"  field of view: {horizontal:.1f} deg x {vertical:.1f} deg",
            f"  distortion: {coefficients}",
        ]
        if self.rms is not None:
            lines.append(f"  RMS reprojection error: {self.rms:.3f} px")
        return "\n".join(lines)
