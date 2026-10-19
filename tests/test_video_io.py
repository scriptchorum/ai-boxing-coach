from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import av
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boxing_coach.video_io import (  # noqa: E402
    DisplayTransform,
    FrameTiming,
    MediaInputError,
    _frame_timing,
    _timing_summary,
    apply_display_transform,
    decode_video,
    probe_video,
)


def write_synthetic_video(
    path: Path,
    timestamps_ms: list[int],
    *,
    width: int = 64,
    height: int = 48,
    rotation: int = 0,
    horizontal_flip: bool = False,
    codec: str = "mpeg4",
) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=30)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        if rotation or horizontal_flip:
            stream.set_display_rotation(rotation, hflip=horizontal_flip)
        for index, timestamp_ms in enumerate(timestamps_ms):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[:, : width // 2, 0] = 240
            image[:, width // 2 :, 1] = 240
            image[index % height, :, 2] = 240
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = timestamp_ms
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class VideoIoTests(unittest.TestCase):
    def test_fixed_rate_30_and_60_preserve_source_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fps, timestamps in (
                (30, [0, 33, 67, 100, 133, 167]),
                (60, [0, 17, 33, 50, 67, 83]),
            ):
                with self.subTest(fps=fps):
                    path = root / f"synthetic-{fps}.mkv"
                    write_synthetic_video(path, timestamps)
                    inspection = probe_video(path)
                    self.assertEqual(inspection.frame_count, len(timestamps))
                    self.assertFalse(inspection.variable_frame_rate)
                    self.assertIsNotNone(inspection.nominal_fps)
                    self.assertAlmostEqual(inspection.nominal_fps or 0, fps, delta=0.5)
                    self.assertEqual(
                        [frame.source_frame_index for frame in inspection.timeline],
                        list(range(len(timestamps))),
                    )
                    self.assertEqual(
                        [frame.timestamp_ms for frame in inspection.timeline],
                        [float(value) for value in timestamps],
                    )

    def test_variable_frame_rate_uses_pts_not_index_divided_by_fps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-vfr.mkv"
            timestamps = [0, 33, 67, 120, 153]
            write_synthetic_video(path, timestamps)
            inspection = probe_video(path)
            self.assertTrue(inspection.variable_frame_rate)
            self.assertIsNone(inspection.nominal_fps)
            self.assertEqual(
                [frame.timestamp_ms for frame in inspection.timeline],
                [float(value) for value in timestamps],
            )
            self.assertNotEqual(inspection.timeline[3].timestamp_ms, 3 * 1000 / 30)

    def test_all_right_angle_rotations_and_reflections_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rotation in (0, 90, 180, 270):
                for horizontal_flip in (False, True):
                    with self.subTest(rotation=rotation, horizontal_flip=horizontal_flip):
                        path = root / f"synthetic-{rotation}-{horizontal_flip}.mp4"
                        write_synthetic_video(
                            path,
                            [0, 33, 67],
                            width=64,
                            height=48,
                            rotation=rotation,
                            horizontal_flip=horizontal_flip,
                            codec="libx264",
                        )
                        inspection = probe_video(path)
                        self.assertEqual(inspection.codec, "h264")
                        self.assertEqual(
                            inspection.display_transform,
                            DisplayTransform(rotation, horizontal_flip),
                        )
                        self.assertEqual(
                            (inspection.coded_width, inspection.coded_height),
                            (64, 48),
                        )
                        expected_size = (48, 64) if rotation in (0, 180) else (64, 48)
                        self.assertEqual((inspection.height, inspection.width), expected_size)
                        decoded = next(decode_video(path, inspection=inspection))
                        self.assertEqual(decoded.image_rgb.shape[:2], expected_size)
                        with av.open(str(path), mode="r") as container:
                            raw = next(container.decode(container.streams.video[0])).to_ndarray(
                                format="rgb24"
                            )
                        expected_pixels = np.rot90(raw, k=rotation // 90)
                        if horizontal_flip:
                            expected_pixels = np.fliplr(expected_pixels)
                        self.assertTrue(np.array_equal(decoded.image_rgb, expected_pixels))

    def test_container_and_user_mirror_corrections_are_explicit(self) -> None:
        image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((2, 3, 3))
        metadata_flip = apply_display_transform(image, DisplayTransform(0, True))
        self.assertTrue(np.array_equal(metadata_flip, image[:, ::-1, :]))
        corrected = apply_display_transform(
            image,
            DisplayTransform(0, True),
            correct_mirror=True,
        )
        self.assertTrue(np.array_equal(corrected, image))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-mirror.mp4"
            write_synthetic_video(path, [0, 33, 67], horizontal_flip=True)
            inspection = probe_video(path)
            self.assertEqual(inspection.display_transform, DisplayTransform(0, True))

    def test_contract_media_contains_only_contract_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-contract.mkv"
            write_synthetic_video(path, [0, 40, 80])
            inspection = probe_video(path)
            media = inspection.to_contract_media()
            self.assertEqual(
                set(media),
                {
                    "codec",
                    "width",
                    "height",
                    "frame_count",
                    "rotation_applied_degrees",
                    "original_pts_start_ms",
                    "timestamp_source",
                    "nominal_fps",
                },
            )
            self.assertEqual(media["timestamp_source"], "decoded_pts")

    def test_hash_matches_input_without_exposing_path_in_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private-name.mkv"
            write_synthetic_video(path, [0, 40, 80])
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            result = probe_video(path).to_dict(include_timeline=True)
            self.assertEqual(result["input_sha256"], expected)
            self.assertNotIn(str(path), json.dumps(result))

    def test_decode_rejects_an_inspection_from_another_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.mkv"
            second = root / "second.mkv"
            write_synthetic_video(first, [0, 40, 80])
            write_synthetic_video(second, [0, 33, 67])
            inspection = probe_video(first)
            with self.assertRaises(MediaInputError) as caught:
                next(decode_video(second, inspection=inspection))
            self.assertEqual(caught.exception.code, "input_changed")

    def test_negative_start_pts_is_preserved_and_normalized(self) -> None:
        first_frame = av.VideoFrame(16, 16, "rgb24")
        first_frame.pts = -3
        first_frame.time_base = Fraction(1, 30)
        first, anchor = _frame_timing(0, first_frame, None)
        second_frame = av.VideoFrame(16, 16, "rgb24")
        second_frame.pts = -2
        second_frame.time_base = Fraction(1, 30)
        second, _ = _frame_timing(1, second_frame, anchor)
        self.assertEqual(first.original_pts_ms, -100.0)
        self.assertEqual(first.timestamp_ms, 0.0)
        self.assertAlmostEqual(second.timestamp_ms, 1000 / 30, places=5)

    def test_non_monotonic_timeline_is_rejected(self) -> None:
        timeline = [
            FrameTiming(0, 0, 1, 1000, 0.0, 0.0, 33.0),
            FrameTiming(1, 0, 1, 1000, 0.0, 0.0, 33.0),
        ]
        with self.assertRaises(MediaInputError) as caught:
            _timing_summary(timeline)
        self.assertEqual(caught.exception.code, "non_monotonic_pts")

    def test_duration_limit_and_multiple_video_streams_are_rejected(self) -> None:
        timeline = [
            FrameTiming(0, 0, 1, 1000, 0.0, 0.0, 33.0),
            FrameTiming(1, 120001, 1, 1000, 120001.0, 120001.0, 33.0),
        ]
        with self.assertRaises(MediaInputError) as caught:
            _timing_summary(timeline)
        self.assertEqual(caught.exception.code, "video_too_long")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multiple-streams.mkv"
            with av.open(str(path), mode="w") as container:
                streams = []
                for _ in range(2):
                    stream = container.add_stream("mpeg4", rate=30)
                    stream.width = 16
                    stream.height = 16
                    stream.pix_fmt = "yuv420p"
                    streams.append(stream)
                for stream in streams:
                    frame = av.VideoFrame.from_ndarray(
                        np.zeros((16, 16, 3), dtype=np.uint8),
                        format="rgb24",
                    )
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for stream in streams:
                    for packet in stream.encode():
                        container.mux(packet)
            with self.assertRaises(MediaInputError) as caught:
                probe_video(path)
            self.assertEqual(caught.exception.code, "multiple_video_streams")

    def test_non_right_angle_display_transform_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-45.mp4"
            write_synthetic_video(path, [0, 33, 67], rotation=45)
            with self.assertRaises(MediaInputError) as caught:
                probe_video(path)
            self.assertEqual(caught.exception.code, "unsupported_display_transform")

    def test_corrupt_and_unsupported_inputs_have_stable_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrupt = root / "private-corrupt.mp4"
            corrupt.write_bytes(b"not a video")
            with self.assertRaises(MediaInputError) as caught:
                probe_video(corrupt)
            self.assertEqual(caught.exception.code, "decode_failed")
            self.assertNotIn(str(corrupt), str(caught.exception))

            unsupported = root / "private-video.txt"
            unsupported.write_bytes(b"not a video")
            with self.assertRaises(MediaInputError) as caught:
                probe_video(unsupported)
            self.assertEqual(caught.exception.code, "unsupported_extension")

    def test_cli_reports_timeline_and_does_not_echo_private_path_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "private-good.mkv"
            write_synthetic_video(good, [0, 40, 80])
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "boxing_coach.video_io",
                    "probe",
                    str(good),
                    "--timeline",
                    "--decode-first-frame",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(len(payload["timeline"]), 3)
            self.assertEqual(payload["first_frame"]["rgb_shape"], [48, 64, 3])
            self.assertNotIn(str(good), completed.stdout + completed.stderr)

            bad = root / "private-bad.mp4"
            bad.write_bytes(b"not a video")
            completed = subprocess.run(
                [sys.executable, "-m", "boxing_coach.video_io", "probe", str(bad)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn("VIDEO_INPUT_ERROR[decode_failed]", completed.stderr)
            self.assertNotIn(str(bad), completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
