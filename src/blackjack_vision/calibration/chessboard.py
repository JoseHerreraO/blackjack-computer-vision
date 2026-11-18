"""Chessboard camera calibration with Zhang's method (Tema 2).

Pipeline:

1. Detect the inner corners of the printed pattern (:func:`find_corners`) with
   ``findChessboardCorners`` and refine them to sub-pixel accuracy with ``cornerSubPix``.
2. Gather enough *different* views (:func:`collect_views`): 10-20 poses that cover the whole
   image, tilted in both axes and at several distances. Near-duplicate video frames are skipped.
3. Estimate intrinsics, distortion and per-view extrinsics (:func:`calibrate`): closed-form
   initialisation assuming no distortion, then joint Levenberg-Marquardt refinement of every
   parameter (``calibrateCamera``). Views with an abnormal error (blur, wrong detections) are
   discarded and the calibration is repeated once.
4. Judge the RMS reprojection error: below 0.5 px is acceptable and 0.1-0.3 px is good. A low
   error alone does not prove a good calibration: poor pose variety or too many distortion
   coefficients can overfit.

Two options help with real, imperfect boards:

* ``detector="sb"`` uses ``findChessboardCornersSB`` with ``CALIB_CB_ACCURACY`` (Duda and
  Frese's sector-based detector), more accurate than ``findChessboardCorners`` + ``cornerSubPix``
  on noisy or compressed photos.
* ``release_object=True`` uses ``calibrateCameraRO`` (Strobl and Hirzinger's "releasing the
  object" method): the 3D board points are estimated together with the camera, so a printed
  sheet that is not perfectly flat or not printed exactly to scale no longer inflates the error.
  :func:`plane_fit_errors` measures how far each view is from an ideal flat board, which helps to
  spot blurred views and curled paper before calibrating.

References: Z. Zhang, "A flexible new technique for camera calibration", IEEE TPAMI 22(11), 2000.
K. H. Strobl and G. Hirzinger, "More accurate pinhole camera calibration with imperfect planar
target", ICCV Workshops, 2011.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

import cv2
import numpy as np

from blackjack_vision.calibration.camera_model import CameraModel
from blackjack_vision.capture.sources import Frame

logger = logging.getLogger(__name__)

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
MIN_VIEWS = 3
GOOD_RMS = 0.3
ACCEPTABLE_RMS = 0.5
_SUBPIX_HALF_WINDOW_LIMITS = (2, 11)
_INTRINSIC_NAMES = ("fx", "fy", "cx", "cy")
DETECTORS = ("classic", "sb")

FrameCallback = Callable[[Frame, np.ndarray | None, bool], bool | None]


@dataclass(frozen=True)
class ChessboardSpec:
    """Geometry of the printed calibration pattern.

    Attributes:
        inner_corners: ``(columns, rows)`` of *inner* corners, e.g. ``(9, 6)`` for a board of
            10x7 squares. One odd and one even number of squares avoids orientation ambiguity.
        square_size: Side of a square in world units (millimetres by convention). It does not
            affect the intrinsics, only the scale of the extrinsics.
    """

    inner_corners: tuple[int, int]
    square_size: float = 1.0

    def __post_init__(self) -> None:
        columns, rows = self.inner_corners
        if columns < 2 or rows < 2:
            raise ValueError(f"Need at least 2x2 inner corners, got {self.inner_corners}")
        if self.square_size <= 0:
            raise ValueError(f"square_size must be positive, got {self.square_size}")

    @classmethod
    def parse(cls, text: str, square_size: float = 1.0) -> ChessboardSpec:
        """Build a spec from ``"COLSxROWS"`` inner corners, e.g. ``"9x6"``."""
        try:
            columns, rows = (int(part) for part in text.lower().replace("×", "x").split("x"))
        except ValueError as exc:
            raise ValueError(f"Invalid pattern {text!r}: expected COLSxROWS, e.g. 9x6") from exc
        return cls((columns, rows), square_size)

    @property
    def n_corners(self) -> int:
        """Number of inner corners."""
        return self.inner_corners[0] * self.inner_corners[1]

    def object_points(self) -> np.ndarray:
        """3D corner coordinates on the board plane (``Z = 0``), shape ``(N, 3)``, ``float32``.

        The order matches OpenCV's corner order: row by row, with ``x`` varying fastest.
        """
        columns, rows = self.inner_corners
        points = np.zeros((rows * columns, 3), np.float32)
        points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * self.square_size
        return points

    def __str__(self) -> str:
        return f"{self.inner_corners[0]}x{self.inner_corners[1]}"


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a BGR/BGRA image to grayscale; grayscale images are returned unchanged."""
    if image.ndim == 2:
        return image
    code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, code)


def subpixel_half_window(corners: np.ndarray, spec: ChessboardSpec) -> int:
    """Half size of the ``cornerSubPix`` search window, adapted to the apparent square size.

    The search window must not reach the neighbouring corners. With small squares (a distant
    board or the low-resolution glasses stream) the usual 11 px half-window would mix several
    corners, so it is limited to 40 % of the smallest square side seen in the image.
    """
    columns, rows = spec.inner_corners
    grid = np.asarray(corners, np.float64).reshape(rows, columns, 2)
    horizontal = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    vertical = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    side = min(float(horizontal.min()), float(vertical.min()))
    low, high = _SUBPIX_HALF_WINDOW_LIMITS
    return int(np.clip(0.4 * side, low, high))


def find_corners(
    image: np.ndarray,
    spec: ChessboardSpec,
    *,
    refine: bool = True,
    fast_check: bool = True,
    detector: str = "classic",
) -> np.ndarray | None:
    """Locate the inner chessboard corners with sub-pixel accuracy.

    Args:
        image: BGR or grayscale image.
        spec: Pattern geometry.
        refine: Refine the corners with ``cornerSubPix`` (classic detector only; the SB detector
            is already sub-pixel accurate).
        fast_check: Quickly reject images without a board (``CALIB_CB_FAST_CHECK``), which
            matters for videos where most frames do not show the complete pattern (classic
            detector only).
        detector: ``"classic"`` (``findChessboardCorners`` + ``cornerSubPix``) or ``"sb"``
            (``findChessboardCornersSB`` with ``CALIB_CB_ACCURACY``).

    Returns:
        Corners as an ``(N, 2)`` ``float32`` array in OpenCV order, or ``None`` if the complete
        pattern was not found.
    """
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector {detector!r}, expected one of {DETECTORS}")
    gray = to_gray(image)
    if detector == "sb":
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCornersSB(gray, spec.inner_corners, flags=flags)
        if not found or corners is None:
            return None
        return corners.reshape(-1, 2).astype(np.float32)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast_check:
        flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, spec.inner_corners, flags=flags)
    if not found or corners is None:
        return None
    if refine:
        half = subpixel_half_window(corners, spec)
        corners = cv2.cornerSubPix(gray, corners, (half, half), (-1, -1), SUBPIX_CRITERIA)
    return corners.reshape(-1, 2).astype(np.float32)


def draw_corners(image: np.ndarray, corners: np.ndarray, spec: ChessboardSpec) -> np.ndarray:
    """Return a copy of ``image`` with the detected corners drawn (for previews and reports)."""
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    points = np.asarray(corners, np.float32).reshape(-1, 1, 2)
    cv2.drawChessboardCorners(canvas, spec.inner_corners, points, True)
    return canvas


def orient_corners(corners: np.ndarray, spec: ChessboardSpec) -> np.ndarray:
    """Corners numbered so that the first row is the upper one in the image.

    A board with an odd number of squares on both sides looks the same after a 180-degree turn,
    so the detector may number its corners from either end. That is harmless for the standard
    calibration, but ``calibrate(release_object=True)`` estimates one board shape shared by all
    views and needs every index to be the same physical corner. Taking the upper row first is
    consistent as long as all photos are taken from the same side of the board.
    """
    points = np.asarray(corners, np.float32).reshape(-1, 2)
    columns, rows = spec.inner_corners
    grid = points.reshape(rows, columns, 2)
    if grid[0, :, 1].mean() > grid[-1, :, 1].mean():
        return points[::-1].copy()
    return points


def plane_fit_errors(image_points: Sequence[np.ndarray], spec: ChessboardSpec) -> np.ndarray:
    """RMS distance (px) between each view's corners and the best homography of a flat board.

    A homography is the most general mapping of a plane into an image, so this error does not
    depend on any camera model: with a flat, accurately printed board, sharp images and a lens
    with little distortion over the board it should be about 0.1-0.2 px. Larger values point to
    blur, heavy compression or a sheet that is not flat. Views far above the rest are good
    candidates to drop (``calibrate(max_plane_error=...)``).
    """
    board = spec.object_points()[:, :2].astype(np.float64)
    errors = []
    for points in image_points:
        points = np.asarray(points, np.float64).reshape(-1, 2)
        homography, _ = cv2.findHomography(board, points, 0)
        projected = cv2.perspectiveTransform(board.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors.append(float(np.sqrt(((projected - points) ** 2).sum(axis=1).mean())))
    return np.array(errors)


@dataclass
class ViewCollection:
    """Board detections gathered from a source, ready for :func:`calibrate`.

    Attributes:
        image_points: Corners of every kept view, each ``(N, 2)``.
        names: Identifier of every kept view (file name or ``frame_<index>``).
        images: The kept frames themselves, only filled when ``keep_images=True``.
        image_size: ``(width, height)`` shared by every view.
        frames_processed: Frames on which detection was attempted.
        frames_with_board: Frames where the complete pattern was found.
    """

    image_points: list[np.ndarray] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    images: list[np.ndarray] = field(default_factory=list)
    image_size: tuple[int, int] | None = None
    frames_processed: int = 0
    frames_with_board: int = 0

    def __len__(self) -> int:
        return len(self.image_points)


def collect_views(
    frames: Iterable[Frame],
    spec: ChessboardSpec,
    *,
    stride: int = 1,
    min_shift: float = 0.0,
    max_views: int | None = None,
    keep_images: bool = False,
    on_frame: FrameCallback | None = None,
    detector: str = "classic",
) -> ViewCollection:
    """Detect the board in a stream of frames and keep a diverse set of views.

    Consecutive video frames are nearly identical and add no information, so a detection is
    kept only if its corners moved more than ``min_shift`` pixels on average with respect to
    every view kept so far.

    Args:
        frames: Any iterable of frames, typically a :class:`~blackjack_vision.capture.sources.FrameSource`.
        spec: Pattern geometry.
        stride: Process one frame out of ``stride`` (by frame index).
        min_shift: Minimum mean corner displacement in pixels for a view to count as new.
        max_views: Stop after keeping this many views.
        keep_images: Also store the kept frames (to save annotated images for the report).
        on_frame: Called as ``on_frame(frame, corners_or_None, accepted)`` after every processed
            frame. Returning ``False`` stops the collection (e.g. the user pressed ``q``).
        detector: Corner detector, see :func:`find_corners`.

    Raises:
        ValueError: if frames of different sizes are mixed (a calibration is per resolution).
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    views = ViewCollection()
    for frame in frames:
        if frame.index % stride:
            continue
        if views.image_size is None:
            views.image_size = frame.size
        elif frame.size != views.image_size:
            raise ValueError(
                f"Frame {frame.index} is {frame.size}, previous frames are {views.image_size}: "
                "calibrate each resolution separately"
            )
        views.frames_processed += 1
        corners = find_corners(frame.image, spec, detector=detector)
        accepted = False
        if corners is not None:
            views.frames_with_board += 1
            if _is_new_view(corners, views.image_points, min_shift):
                views.image_points.append(corners)
                views.names.append(frame.name or f"frame_{frame.index:05d}")
                if keep_images:
                    views.images.append(frame.image)
                accepted = True
        if on_frame is not None and on_frame(frame, corners, accepted) is False:
            break
        if max_views is not None and len(views) >= max_views:
            break
    return views


def _is_new_view(corners: np.ndarray, kept: Sequence[np.ndarray], min_shift: float) -> bool:
    """Whether ``corners`` differ enough from every kept view (robust to reversed corner order)."""
    if min_shift <= 0:
        return True
    for previous in kept:
        direct = float(np.linalg.norm(corners - previous, axis=1).mean())
        reversed_order = float(np.linalg.norm(corners - previous[::-1], axis=1).mean())
        if min(direct, reversed_order) < min_shift:
            return False
    return True


@dataclass(eq=False)
class CalibrationResult:
    """Outcome of :func:`calibrate`.

    Attributes:
        camera: Estimated camera model (with ``rms`` and metadata filled in).
        per_view_errors: RMS reprojection error of each used view, in pixels.
        view_names: Names of the used views, aligned with ``per_view_errors``.
        rvecs: Rotation (Rodrigues vector) of the board in each used view.
        tvecs: Translation of the board in each used view, in ``square_size`` units.
        std_intrinsics: Standard deviation of every intrinsic parameter estimated by OpenCV
            (``fx, fy, cx, cy, k1, k2, p1, p2, k3, ...``).
        rejected_views: Views discarded as outliers before the final estimation.
        board_points: Board corners used in the final estimation, ``(N, 3)``. The ideal grid,
            or the shape estimated by the release-object method.
    """

    camera: CameraModel
    per_view_errors: np.ndarray
    view_names: list[str]
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    std_intrinsics: np.ndarray
    rejected_views: list[str] = field(default_factory=list)
    board_points: np.ndarray | None = None

    @property
    def rms(self) -> float:
        """Overall RMS reprojection error in pixels."""
        return float(self.camera.rms if self.camera.rms is not None else np.nan)

    @property
    def quality(self) -> str:
        """Verdict on the RMS error using the course thresholds."""
        if self.rms <= GOOD_RMS:
            return "good"
        if self.rms <= ACCEPTABLE_RMS:
            return "acceptable"
        return "poor: check blur, board flatness and pose variety"

    def report(self) -> str:
        """Human-readable summary for the console and the project report."""
        camera = self.camera
        width, height = camera.image_size
        sigma = dict(zip(_INTRINSIC_NAMES, self.std_intrinsics[:4], strict=False))
        values = {"fx": camera.fx, "fy": camera.fy, "cx": camera.cx, "cy": camera.cy}
        intrinsics = "  ".join(
            f"{name}={values[name]:.2f} +/- {sigma.get(name, np.nan):.2f}"
            for name in _INTRINSIC_NAMES
        )
        horizontal, vertical = camera.field_of_view
        lines = [
            (
                f"Calibration of '{camera.name}' at {width}x{height} px "
                f"from {len(self.view_names)} views"
            ),
            f"  RMS reprojection error: {self.rms:.3f} px ({self.quality})",
            (
                f"  per-view error: median {np.median(self.per_view_errors):.3f} px, "
                f"max {self.per_view_errors.max():.3f} px"
            ),
            f"  {intrinsics}",
            f"  distortion: {np.array2string(camera.dist_coeffs.ravel(), precision=4)}",
            f"  field of view: {horizontal:.1f} deg x {vertical:.1f} deg",
        ]
        if self.board_points is not None and camera.metadata.get("release_object"):
            ideal = camera.metadata.get("square_size", 1.0)
            deviation = np.linalg.norm(self.board_points - self._ideal_board(), axis=1)
            lines.append(
                f"  release object: board deviates up to {deviation.max():.3f} "
                f"(square = {ideal:g}) from the ideal flat grid"
            )
        if self.rejected_views:
            lines.append(f"  rejected views: {', '.join(self.rejected_views)}")
        return "\n".join(lines)

    def _ideal_board(self) -> np.ndarray:
        columns, rows = (int(v) for v in self.camera.metadata["pattern"].split("x"))
        spec = ChessboardSpec((columns, rows), float(self.camera.metadata["square_size"]))
        return spec.object_points().astype(np.float64)


def calibrate(
    image_points: Sequence[np.ndarray],
    spec: ChessboardSpec,
    image_size: tuple[int, int],
    *,
    view_names: Sequence[str] | None = None,
    name: str = "camera",
    rational_model: bool = False,
    fix_k3: bool = False,
    zero_tangent: bool = False,
    reject_outliers: bool = True,
    min_views: int = 5,
    fix_aspect_ratio: bool = False,
    release_object: bool = False,
    max_plane_error: float | None = None,
) -> CalibrationResult:
    """Estimate the camera intrinsics and distortion from several views of the chessboard.

    Args:
        image_points: Detected corners of each view, ``(N, 2)`` each, in OpenCV order.
        spec: Pattern geometry (gives the 3D object points).
        image_size: ``(width, height)`` of the images.
        view_names: Optional names of the views, used in reports.
        name: Name stored in the resulting :class:`CameraModel`.
        rational_model: Use the 8-coefficient rational model (very wide-angle lenses).
        fix_k3: Keep ``k3 = 0`` (recommended for narrow lenses or few views, avoids overfitting).
        zero_tangent: Assume no tangential distortion (``p1 = p2 = 0``).
        reject_outliers: Discard views whose error is far above the rest and recalibrate once.
        min_views: Never keep fewer views than this when rejecting outliers.
        fix_aspect_ratio: Force ``fx = fy`` (square pixels, true for practically every camera).
        release_object: Estimate the board points too (``calibrateCameraRO``), for printed boards
            that are not perfectly flat. Views are re-numbered with :func:`orient_corners`, so
            all photos must be taken from the same side of the board.
        max_plane_error: Drop, before calibrating, the views whose :func:`plane_fit_errors` is
            above this value in pixels (blurred views, curled paper).

    Raises:
        ValueError: with fewer than :data:`MIN_VIEWS` views or mismatched corner counts.
    """
    n_views = len(image_points)
    if n_views < MIN_VIEWS:
        raise ValueError(f"Need at least {MIN_VIEWS} views (10-20 recommended), got {n_views}")
    names = (
        list(view_names) if view_names is not None else [f"view_{i:03d}" for i in range(n_views)]
    )
    if len(names) != n_views:
        raise ValueError("view_names and image_points have different lengths")
    for points, view in zip(image_points, names, strict=True):
        if np.asarray(points).reshape(-1, 2).shape[0] != spec.n_corners:
            raise ValueError(f"View {view!r} does not have {spec.n_corners} corners")

    if max_plane_error is not None:
        plane = plane_fit_errors(image_points, spec)
        keep = plane <= max_plane_error
        if int(keep.sum()) < max(MIN_VIEWS, min(min_views, n_views)):
            raise ValueError(
                f"Only {int(keep.sum())} views have a plane-fit error <= {max_plane_error} px"
            )
        dropped = [view for view, kept in zip(names, keep, strict=True) if not kept]
        if dropped:
            logger.info("Dropping %d view(s) with a high plane-fit error", len(dropped))
        image_points = [p for p, kept in zip(image_points, keep, strict=True) if kept]
        names = [view for view, kept in zip(names, keep, strict=True) if kept]
    else:
        dropped = []
    if release_object:
        image_points = [orient_corners(points, spec) for points in image_points]

    flags = 0
    if fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if fix_k3:
        flags |= cv2.CALIB_FIX_K3
    if zero_tangent:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST

    n_views = len(image_points)
    result = _calibrate_once(image_points, names, spec, image_size, flags, name, release_object)
    if reject_outliers and n_views > min_views:
        keep = result.per_view_errors <= _outlier_threshold(result.per_view_errors)
        if not keep.all() and int(keep.sum()) >= min_views:
            rejected = [view for view, kept in zip(names, keep, strict=True) if not kept]
            logger.info("Rejecting %d outlier view(s): %s", len(rejected), ", ".join(rejected))
            result = _calibrate_once(
                [points for points, kept in zip(image_points, keep, strict=True) if kept],
                [view for view, kept in zip(names, keep, strict=True) if kept],
                spec,
                image_size,
                flags,
                name,
                release_object,
            )
            result.rejected_views = rejected
    if dropped:
        result.rejected_views = dropped + result.rejected_views
        result.camera.metadata["max_plane_error"] = max_plane_error
    if result.rejected_views:
        result.camera.metadata["rejected_views"] = result.rejected_views
    return result


def _outlier_threshold(errors: np.ndarray) -> float:
    """Robust error threshold: median + 3 * MAD, never below 2 x median nor below 0.5 px.

    The absolute floor keeps views that are fine by the course standard (< 0.5 px) even when all
    views are so consistent that the MAD is tiny.
    """
    median = float(np.median(errors))
    mad = 1.4826 * float(np.median(np.abs(errors - median)))
    return max(median + 3.0 * mad, 2.0 * median, ACCEPTABLE_RMS)


def _calibrate_once(
    image_points: Sequence[np.ndarray],
    names: Sequence[str],
    spec: ChessboardSpec,
    image_size: tuple[int, int],
    flags: int,
    name: str,
    release_object: bool = False,
) -> CalibrationResult:
    object_points = [spec.object_points() for _ in image_points]
    points = [np.asarray(view, np.float32).reshape(-1, 1, 2) for view in image_points]
    size = tuple(int(v) for v in image_size)
    board = spec.object_points()
    if release_object:
        # The far end of the first row stays fixed (with the origin) to keep the scale, as the
        # OpenCV documentation recommends.
        fixed_point = spec.inner_corners[0] - 1
        (
            rms,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            new_board,
            std_intrinsics,
            _std_extrinsics,
            _std_board,
            per_view_errors,
        ) = cv2.calibrateCameraROExtended(
            object_points, points, size, fixed_point, None, None, flags=flags
        )
        board = np.asarray(new_board, np.float64).reshape(-1, 3)
    else:
        (
            rms,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            std_intrinsics,
            _std_extrinsics,
            per_view_errors,
        ) = cv2.calibrateCameraExtended(object_points, points, size, None, None, flags=flags)
    camera = CameraModel(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        rms=float(rms),
        name=name,
        metadata={
            "method": (
                "Zhang chessboard calibration with release object (cv2.calibrateCameraRO)"
                if release_object
                else "Zhang chessboard calibration (cv2.calibrateCamera)"
            ),
            "pattern": str(spec),
            "square_size": spec.square_size,
            "n_views": len(names),
            "flags": int(flags),
            "release_object": bool(release_object),
            "opencv_version": cv2.__version__,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return CalibrationResult(
        camera=camera,
        per_view_errors=np.asarray(per_view_errors, np.float64).ravel(),
        view_names=list(names),
        rvecs=[np.asarray(r).ravel() for r in rvecs],
        tvecs=[np.asarray(t).ravel() for t in tvecs],
        std_intrinsics=np.asarray(std_intrinsics, np.float64).ravel(),
        board_points=np.asarray(board, np.float64),
    )
