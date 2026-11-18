"""Frame sources: one interface for webcams, video files, image sequences and network streams.

Every downstream stage (calibration, detection, tracking, HUD) consumes :class:`Frame` objects
produced by a :class:`FrameSource`. Recorded test videos, still photos and the live glasses
stream are therefore interchangeable, which lets us validate each module offline before
connecting it to the live stream.

Example::

    from blackjack_vision.capture.sources import open_source

    with open_source("data/samples/round_01.mp4") as source:
        for frame in source:
            print(frame.index, frame.timestamp, frame.size)
"""

from __future__ import annotations

import glob
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
STREAM_PREFIXES: tuple[str, ...] = ("http://", "https://", "rtsp://", "rtmp://", "tcp://", "udp://")


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image from disk, also when the path contains non-ASCII characters.

    ``cv2.imread`` cannot open paths such as ``C:/Users/José/...`` on Windows, so the bytes are
    read with NumPy and decoded in memory instead.

    Args:
        path: Image file.
        flags: ``cv2.IMREAD_*`` flags (colour BGR by default).

    Returns:
        The decoded image, or ``None`` if the file is missing or cannot be decoded.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image(path: str | Path, image: np.ndarray, params: Sequence[int] = ()) -> Path:
    """Write an image to disk (non-ASCII safe), creating parent folders if needed.

    The format is taken from the file extension (``.png``, ``.jpg``...).

    Raises:
        OSError: if the image cannot be encoded.
    """
    path = Path(path)
    ok, buffer = cv2.imencode(path.suffix or ".png", image, list(params))
    if not ok:
        raise OSError(f"Could not encode image as {path.suffix!r}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(str(path))
    return path


@dataclass(frozen=True, eq=False)
class Frame:
    """One BGR image and its position in the stream.

    Attributes:
        image: BGR image, shape ``(H, W, 3)``, dtype ``uint8``.
        index: 0-based position of the frame within its source.
        timestamp: Seconds since the first frame. Recorded videos use the container timestamps,
            live sources a monotonic clock and image sequences ``index / fps``.
        name: Optional identifier (the file name for image sequences).
    """

    image: np.ndarray
    index: int
    timestamp: float
    name: str = ""

    @property
    def size(self) -> tuple[int, int]:
        """Image size as ``(width, height)``, the convention of OpenCV's calibration functions."""
        height, width = self.image.shape[:2]
        return width, height


class FrameSource(ABC):
    """Base class of every frame producer.

    Sources are iterable and work as context managers, so resources are always released::

        with open_source(0) as source:  # default webcam
            for frame in source:
                ...
    """

    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or ``None`` when the source is exhausted or has failed."""

    @abstractmethod
    def release(self) -> None:
        """Free the underlying resources. Calling it more than once is harmless."""

    @property
    def is_live(self) -> bool:
        """``True`` for real-time sources (cameras, network streams), ``False`` for recordings."""
        return False

    @property
    def fps(self) -> float | None:
        """Nominal frame rate reported by the source, if known."""
        return None

    def __iter__(self) -> Iterator[Frame]:
        while (frame := self.read()) is not None:
            yield frame

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


class VideoCaptureSource(FrameSource):
    """Frames from ``cv2.VideoCapture``: webcams, video files and network streams (MJPEG, RTSP).

    Args:
        source: Camera index (``0`` is the default webcam), video file path or stream URL.
        width: Requested capture width (cameras only; the driver may ignore it).
        height: Requested capture height (cameras only; the driver may ignore it).
        fps: Requested frame rate (cameras only; the driver may ignore it).
        live: Whether the source is real-time. Inferred from ``source`` when ``None``.
        backend: OpenCV capture API, e.g. ``cv2.CAP_DSHOW`` to open Windows webcams faster.

    Raises:
        OSError: if OpenCV cannot open the source.
    """

    def __init__(
        self,
        source: int | str | Path,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        live: bool | None = None,
        backend: int = cv2.CAP_ANY,
    ) -> None:
        target: int | str = source if isinstance(source, int) else str(source)
        self._capture = cv2.VideoCapture(target, backend)
        if not self._capture.isOpened():
            raise OSError(f"Cannot open video source {source!r}")
        for prop, value in (
            (cv2.CAP_PROP_FRAME_WIDTH, width),
            (cv2.CAP_PROP_FRAME_HEIGHT, height),
            (cv2.CAP_PROP_FPS, fps),
        ):
            if value is not None:
                self._capture.set(prop, value)
        if live is None:
            live = isinstance(source, int) or str(source).lower().startswith(STREAM_PREFIXES)
        self._live = live
        self._index = 0
        self._clock_start: float | None = None
        self._last_timestamp = -1.0

    @property
    def is_live(self) -> bool:
        return self._live

    @property
    def fps(self) -> float | None:
        value = self._capture.get(cv2.CAP_PROP_FPS)
        return float(value) if value and value > 0 else None

    @property
    def frame_count(self) -> int | None:
        """Number of frames of a video file, or ``None`` for live sources or when unknown."""
        if self._live:
            return None
        count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else None

    def read(self) -> Frame | None:
        ok, image = self._capture.read()
        if not ok or image is None:
            return None
        frame = Frame(image=image, index=self._index, timestamp=self._next_timestamp())
        self._index += 1
        return frame

    def release(self) -> None:
        self._capture.release()

    def _next_timestamp(self) -> float:
        if self._live:
            now = time.monotonic()
            if self._clock_start is None:
                self._clock_start = now
            return now - self._clock_start
        # Recorded video: the container timestamp of the frame just decoded. Some backends do not
        # report it (always 0), so fall back to index / fps and keep the sequence increasing.
        position = self._capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if position <= self._last_timestamp:
            position = max(self._index / (self.fps or 30.0), self._last_timestamp + 1e-6)
        self._last_timestamp = position
        return position


class ImageSequenceSource(FrameSource):
    """Frames from a list of image files, e.g. calibration photos or test stills.

    Files that cannot be decoded are skipped with a warning. Timestamps are synthetic
    (``index / fps``) because still images carry no timing information.

    Args:
        paths: Image files, read in the given order.
        fps: Nominal rate used to build the synthetic timestamps.

    Raises:
        FileNotFoundError: if ``paths`` is empty.
    """

    def __init__(self, paths: Sequence[str | Path], *, fps: float = 1.0) -> None:
        self._paths = [Path(path) for path in paths]
        if not self._paths:
            raise FileNotFoundError("The image sequence is empty")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self._fps = float(fps)
        self._cursor = 0
        self._index = 0

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        *,
        extensions: Sequence[str] = IMAGE_EXTENSIONS,
        fps: float = 1.0,
    ) -> ImageSequenceSource:
        """All images in ``folder`` (not recursive) in natural order (``img2`` before ``img10``)."""
        folder = Path(folder)
        suffixes = {extension.lower() for extension in extensions}
        paths = [path for path in folder.iterdir() if path.suffix.lower() in suffixes]
        if not paths:
            raise FileNotFoundError(f"No images ({', '.join(sorted(suffixes))}) in {folder}")
        return cls(sorted(paths, key=_natural_key), fps=fps)

    @classmethod
    def from_glob(cls, pattern: str, *, fps: float = 1.0) -> ImageSequenceSource:
        """Images matching a glob pattern such as ``data/calibration/webcam/*.jpg``."""
        paths = [Path(path) for path in glob.glob(pattern)]
        paths = [path for path in paths if path.suffix.lower() in IMAGE_EXTENSIONS]
        if not paths:
            raise FileNotFoundError(f"No images match {pattern!r}")
        return cls(sorted(paths, key=_natural_key), fps=fps)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def paths(self) -> list[Path]:
        return list(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def read(self) -> Frame | None:
        while self._cursor < len(self._paths):
            path = self._paths[self._cursor]
            self._cursor += 1
            image = read_image(path)
            if image is None:
                logger.warning("Skipping unreadable image %s", path)
                continue
            frame = Frame(image, self._index, self._index / self._fps, name=path.name)
            self._index += 1
            return frame
        return None

    def release(self) -> None:
        self._cursor = len(self._paths)


class SourceKind(str, Enum):
    """Kinds of source that :func:`open_source` can open."""

    CAMERA = "camera"
    VIDEO = "video"
    STREAM = "stream"
    IMAGES = "images"
    BRIDGE = "bridge"


def parse_source_spec(spec: int | str | Path) -> tuple[SourceKind, int | str]:
    """Classify a command-line source specification without opening it.

    * ``0`` or ``"1"``: camera index.
    * ``"bridge"`` or ``"bridge:5050"``: live glasses stream from the BJ Bridge iOS app.
    * ``"http://..."``, ``"rtsp://..."``...: network stream.
    * An existing folder, a glob pattern (``*``, ``?``, ``[``) or an image file: image sequence.
    * Anything else (``clip.mp4``): video file.

    Returns:
        The kind of source and the normalised value (``int`` for cameras, ``str`` otherwise).
    """
    if isinstance(spec, int):
        return SourceKind.CAMERA, spec
    text = str(spec).strip()
    if text.isdigit():
        return SourceKind.CAMERA, int(text)
    if text.lower() == "bridge" or text.lower().startswith("bridge:"):
        port = text.partition(":")[2] or "5050"
        if not port.isdigit():
            raise ValueError(f"Invalid bridge port in {text!r}, use e.g. bridge:5050")
        return SourceKind.BRIDGE, int(port)
    if text.lower().startswith(STREAM_PREFIXES):
        return SourceKind.STREAM, text
    path = Path(text)
    is_pattern = any(char in text for char in "*?[")
    if path.is_dir() or is_pattern or path.suffix.lower() in IMAGE_EXTENSIONS:
        return SourceKind.IMAGES, text
    return SourceKind.VIDEO, text


def open_source(
    spec: int | str | Path,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
) -> FrameSource:
    """Open any supported source from a single specification (see :func:`parse_source_spec`).

    Args:
        spec: Camera index, video file, image folder, glob pattern, image file, stream URL or
            ``bridge[:port]`` for the glasses stream sent by the iOS app.
        width: Requested camera width (cameras only).
        height: Requested camera height (cameras only).
        fps: Requested camera frame rate, or the nominal rate of an image sequence.

    Raises:
        FileNotFoundError: if a video file or image set does not exist.
        OSError: if OpenCV cannot open a camera, video or stream.
    """
    kind, value = parse_source_spec(spec)
    if kind is SourceKind.CAMERA:
        return VideoCaptureSource(int(value), width=width, height=height, fps=fps, live=True)
    if kind is SourceKind.STREAM:
        return VideoCaptureSource(str(value), live=True)
    if kind is SourceKind.BRIDGE:
        from blackjack_vision.capture.bridge import BridgeSource  # imports this module

        return BridgeSource(port=int(value))
    path = Path(str(value))
    if kind is SourceKind.IMAGES:
        rate = fps or 1.0
        if path.is_dir():
            return ImageSequenceSource.from_folder(path, fps=rate)
        if path.is_file():
            return ImageSequenceSource([path], fps=rate)
        return ImageSequenceSource.from_glob(str(value), fps=rate)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    return VideoCaptureSource(path, live=False)


def _natural_key(path: Path) -> list[int | str]:
    """Sort key that orders embedded numbers numerically: ``img2`` < ``img10``."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]
