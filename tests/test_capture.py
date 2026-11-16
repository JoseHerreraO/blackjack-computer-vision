from pathlib import Path

import cv2
import numpy as np
import pytest

from blackjack_vision.capture.sources import (
    ImageSequenceSource,
    SourceKind,
    VideoCaptureSource,
    open_source,
    parse_source_spec,
    read_image,
    write_image,
)


def _write_video(path: Path, n_frames: int, fps: float = 10.0, size=(64, 48)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened(), "OpenCV build without an MJPG encoder"
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), 10 + 20 * i, np.uint8))
    writer.release()


@pytest.mark.parametrize(
    ("spec", "kind", "value"),
    [
        (0, SourceKind.CAMERA, 0),
        ("1", SourceKind.CAMERA, 1),
        ("rtsp://192.168.1.20:8554/glasses", SourceKind.STREAM, "rtsp://192.168.1.20:8554/glasses"),
        ("HTTP://10.0.0.5:8080/video", SourceKind.STREAM, "HTTP://10.0.0.5:8080/video"),
        ("data/calibration/*.jpg", SourceKind.IMAGES, "data/calibration/*.jpg"),
        ("photo.PNG", SourceKind.IMAGES, "photo.PNG"),
        ("round_01.mp4", SourceKind.VIDEO, "round_01.mp4"),
    ],
)
def test_parse_source_spec(spec, kind, value):
    assert parse_source_spec(spec) == (kind, value)


def test_parse_source_spec_detects_folders(tmp_path):
    assert parse_source_spec(tmp_path) == (SourceKind.IMAGES, str(tmp_path))


def test_image_io_roundtrip_with_non_ascii_path(tmp_path):
    image = np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    path = write_image(tmp_path / "visión_ñ" / "carta.png", image)
    assert np.array_equal(read_image(path), image)
    assert read_image(tmp_path / "missing.png") is None


def test_image_sequence_natural_order_and_timestamps(tmp_path):
    for i in (10, 2, 1):
        write_image(tmp_path / f"img{i}.png", np.full((8, 12, 3), i, np.uint8))
    (tmp_path / "notes.txt").write_text("not an image")

    with open_source(tmp_path, fps=2.0) as source:
        frames = list(source)

    assert [frame.name for frame in frames] == ["img1.png", "img2.png", "img10.png"]
    assert [frame.index for frame in frames] == [0, 1, 2]
    assert [frame.timestamp for frame in frames] == [0.0, 0.5, 1.0]
    assert frames[0].size == (12, 8)


def test_image_sequence_skips_unreadable_files(tmp_path):
    write_image(tmp_path / "a.png", np.zeros((4, 4, 3), np.uint8))
    (tmp_path / "b.png").write_bytes(b"corrupted")
    frames = list(ImageSequenceSource.from_folder(tmp_path))
    assert [frame.name for frame in frames] == ["a.png"]


def test_image_sequence_from_glob(tmp_path):
    for name in ("x1.jpg", "x2.jpg", "y1.png"):
        write_image(tmp_path / name, np.zeros((4, 4, 3), np.uint8))
    source = open_source(str(tmp_path / "x*.jpg"))
    assert isinstance(source, ImageSequenceSource)
    assert len(source) == 2


def test_video_file_frames_and_timestamps(tmp_path):
    path = tmp_path / "clip.avi"
    _write_video(path, n_frames=12, fps=10.0)

    with open_source(path) as source:
        assert isinstance(source, VideoCaptureSource)
        assert not source.is_live
        assert source.fps == pytest.approx(10.0)
        assert source.frame_count == 12
        frames = list(source)

    assert len(frames) == 12
    assert [frame.index for frame in frames] == list(range(12))
    timestamps = np.array([frame.timestamp for frame in frames])
    assert np.all(np.diff(timestamps) > 0)
    assert timestamps == pytest.approx(np.arange(12) / 10.0, abs=0.02)
    # MJPEG is lossy: compare the mean grey level of each frame with a tolerance.
    means = np.array([frame.image.mean() for frame in frames])
    assert means == pytest.approx(10 + 20 * np.arange(12), abs=6)


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_source(tmp_path / "nope.mp4")
