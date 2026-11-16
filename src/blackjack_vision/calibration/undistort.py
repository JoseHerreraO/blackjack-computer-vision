"""Lens distortion correction with precomputed remap tables (Tema 2).

For every pixel of the corrected image the inverse mapping is evaluated once: pixel -> normalised
ideal coordinates (``K'^-1``) -> forward distortion model -> distorted pixel (``K``). The result is
stored as remap tables (``initUndistortRectifyMap``), so correcting each frame is a single
``cv2.remap`` with bilinear interpolation, cheap enough for real time.

When only a few points are needed (e.g. the four vertices of a card before computing its
homography) it is cheaper and more accurate to correct just those points with
:meth:`Undistorter.undistort_points` instead of the whole frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from blackjack_vision.calibration.camera_model import CameraModel


class Undistorter:
    """Removes lens distortion from frames of a calibrated camera.

    Args:
        camera: Calibrated camera model. Frames must have exactly ``camera.image_size``; use
            ``camera.scaled(size)`` for a rescaled stream.
        alpha: Free scaling parameter of ``getOptimalNewCameraMatrix``. ``0`` crops the output so
            every pixel is valid (no black borders, slightly narrower view); ``1`` keeps every
            source pixel (curved black borders).
        output_size: Size of the corrected image, ``camera.image_size`` by default.
        interpolation: Interpolation used by ``cv2.remap``.

    Attributes:
        new_camera_matrix: Intrinsics of the corrected (distortion-free) image.
        valid_roi: ``(x, y, width, height)`` of the region with valid pixels in the output.
    """

    def __init__(
        self,
        camera: CameraModel,
        *,
        alpha: float = 0.0,
        output_size: tuple[int, int] | None = None,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.camera = camera
        self.alpha = alpha
        self.output_size = output_size or camera.image_size
        self.interpolation = interpolation
        new_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera.camera_matrix,
            camera.dist_coeffs,
            camera.image_size,
            alpha,
            self.output_size,
        )
        self.new_camera_matrix: np.ndarray = new_matrix
        self.valid_roi: tuple[int, int, int, int] = tuple(int(v) for v in roi)
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            camera.camera_matrix,
            camera.dist_coeffs,
            None,
            new_matrix,
            self.output_size,
            cv2.CV_16SC2,
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return the distortion-free version of ``image``.

        Raises:
            ValueError: if the image size differs from the calibrated resolution.
        """
        height, width = image.shape[:2]
        if (width, height) != self.camera.image_size:
            expected_width, expected_height = self.camera.image_size
            raise ValueError(
                f"Image is {width}x{height} but the camera was calibrated at "
                f"{expected_width}x{expected_height}; build the Undistorter with "
                f"camera.scaled(({width}, {height})) if it is the same view rescaled"
            )
        return cv2.remap(
            image,
            self._map_x,
            self._map_y,
            self.interpolation,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Map pixel coordinates of the raw image to the corrected image.

        Args:
            points: Array of shape ``(N, 2)`` (or ``(N, 1, 2)``) in raw image pixels.

        Returns:
            ``(N, 2)`` ``float64`` coordinates in the output of :meth:`__call__`.
        """
        raw = np.asarray(points, np.float64).reshape(-1, 1, 2)
        corrected = cv2.undistortPoints(
            raw,
            self.camera.camera_matrix,
            self.camera.dist_coeffs,
            P=self.new_camera_matrix,
        )
        return corrected.reshape(-1, 2)

    def distort_points(self, points: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`undistort_points`: corrected-image pixels to raw-image pixels.

        Useful to draw results computed on corrected images over the raw video.
        """
        corrected = np.asarray(points, np.float64).reshape(-1, 2)
        homogeneous = np.hstack([corrected, np.ones((len(corrected), 1))])
        rays = homogeneous @ np.linalg.inv(self.new_camera_matrix).T
        raw, _ = cv2.projectPoints(
            rays.reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            self.camera.camera_matrix,
            self.camera.dist_coeffs,
        )
        return raw.reshape(-1, 2)
