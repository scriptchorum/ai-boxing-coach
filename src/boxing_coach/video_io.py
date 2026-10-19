"""Local video probing, PTS preservation, and display-normalized decoding."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
from typing import Any, Iterator

import av
import numpy as np
from av.sidedata.sidedata import Type as SideDataType


SUPPORTED_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
MAX_FILE_SIZE_BYTES = 512 * 1024 * 1024
MAX_DURATION_MS = 120_000.0
MAX_FRAME_COUNT = 20_000
MAX_FRAME_PIXELS = 3840 * 2160


class MediaInputError(RuntimeError):
    """A safe, user-facing video input failure without a private file path."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DisplayTransform:
    """Container display transform: counter-clockwise rotation, then horizontal flip."""

    rotation_degrees: int = 0
    horizontal_flip: bool = False

    def __post_init__(self) -> None:
        if self.rotation_degrees not in (0, 90, 180, 270):
            raise ValueError("rotation_degrees must be one of 0, 90, 180, or 270")


@dataclass(frozen=True)
class FrameTiming:
    source_frame_index: int
    pts: int
    time_base_numerator: int
    time_base_denominator: int
    original_pts_ms: float
    timestamp_ms: float
    duration_ms: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "source_frame_index": self.source_frame_index,
            "pts": self.pts,
            "time_base_numerator": self.time_base_numerator,
            "time_base_denominator": self.time_base_denominator,
            "original_pts_ms": self.original_pts_ms,
            "timestamp_ms": self.timestamp_ms,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class VideoInspection:
    input_sha256: str
    file_size_bytes: int
    container_format: str
    codec: str
    coded_width: int
    coded_height: int
    width: int
    height: int
    duration_ms: float
    frame_count: int
    original_pts_start_ms: float
    nominal_fps: float | None
    variable_frame_rate: bool
    display_transform: DisplayTransform
    timeline: tuple[FrameTiming, ...]

    def to_contract_media(self) -> dict[str, int | float | str | None]:
        """Return exactly the media fields defined by contracts schema v0.1.0."""

        return {
            "codec": self.codec,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "rotation_applied_degrees": self.display_transform.rotation_degrees,
            "original_pts_start_ms": self.original_pts_start_ms,
            "timestamp_source": "decoded_pts",
            "nominal_fps": self.nominal_fps,
        }

    def to_dict(self, *, include_timeline: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "probe_schema_version": "1.0.0",
            "input_sha256": self.input_sha256,
            "file_size_bytes": self.file_size_bytes,
            "container_format": self.container_format,
            "coded_width": self.coded_width,
            "coded_height": self.coded_height,
            "display_horizontal_flip": self.display_transform.horizontal_flip,
            "variable_frame_rate": self.variable_frame_rate,
            "duration_ms": self.duration_ms,
            "media": self.to_contract_media(),
        }
        if include_timeline:
            result["timeline"] = [frame.to_dict() for frame in self.timeline]
        return result


@dataclass(frozen=True)
class DecodedFrame:
    source_frame_index: int
    timestamp_ms: float
    image_rgb: np.ndarray


_ORIENTATION_MATRICES: dict[tuple[int, bool], tuple[tuple[float, float], tuple[float, float]]] = {
    (0, False): ((1.0, 0.0), (0.0, 1.0)),
    (90, False): ((0.0, -1.0), (1.0, 0.0)),
    (180, False): ((-1.0, 0.0), (0.0, -1.0)),
    (270, False): ((0.0, 1.0), (-1.0, 0.0)),
    (0, True): ((-1.0, 0.0), (0.0, 1.0)),
    (90, True): ((0.0, -1.0), (-1.0, 0.0)),
    (180, True): ((1.0, 0.0), (0.0, -1.0)),
    (270, True): ((0.0, 1.0), (1.0, 0.0)),
}


def _round_ms(value: Fraction | float) -> float:
    return round(float(value), 6)


def _fraction(value: Any) -> Fraction:
    return Fraction(int(value.numerator), int(value.denominator))


def _safe_file(path: Path) -> tuple[Path, int]:
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise MediaInputError(
            "unsupported_extension",
            "Supported video extensions are AVI, M4V, MKV, MOV, MP4, and WEBM.",
        )
    try:
        stat = resolved.stat()
    except OSError as exc:
        raise MediaInputError("file_unavailable", "The selected video cannot be read.") from exc
    if not resolved.is_file() or stat.st_size <= 0:
        raise MediaInputError("file_unavailable", "The selected video is not a readable file.")
    if stat.st_size > MAX_FILE_SIZE_BYTES:
        raise MediaInputError("file_too_large", "The selected video exceeds the 512 MiB limit.")
    return resolved, stat.st_size


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MediaInputError("file_unavailable", "The selected video cannot be read.") from exc
    return digest.hexdigest()


def _rotation_from_metadata(metadata: dict[str, str]) -> DisplayTransform:
    raw = next((value for key, value in metadata.items() if key.lower() == "rotate"), None)
    if raw is None:
        return DisplayTransform()
    try:
        degrees = float(raw)
    except (TypeError, ValueError) as exc:
        raise MediaInputError("unsupported_display_transform", "Invalid rotation metadata.") from exc
    quarter_turns = round(degrees / 90.0)
    if not math.isclose(degrees, quarter_turns * 90.0, abs_tol=0.5):
        raise MediaInputError(
            "unsupported_display_transform",
            "Only right-angle video rotation metadata is supported.",
        )
    return DisplayTransform(rotation_degrees=(quarter_turns * 90) % 360)


def _transform_from_display_matrix(payload: bytes) -> DisplayTransform:
    if len(payload) != 36:
        raise MediaInputError("unsupported_display_transform", "Invalid display matrix metadata.")
    matrix = struct.unpack("<9i", payload)
    values = ((float(matrix[0]), float(matrix[1])), (float(matrix[3]), float(matrix[4])))
    magnitude = math.sqrt(sum(value * value for row in values for value in row) / 2.0)
    if magnitude <= 0:
        raise MediaInputError("unsupported_display_transform", "Empty display matrix metadata.")
    normalized = tuple(tuple(value / magnitude for value in row) for row in values)

    best: tuple[int, bool] | None = None
    best_error = math.inf
    for orientation, expected in _ORIENTATION_MATRICES.items():
        error = math.sqrt(
            sum(
                (normalized[row][column] - expected[row][column]) ** 2
                for row in range(2)
                for column in range(2)
            )
        )
        if error < best_error:
            best = orientation
            best_error = error
    if best is None or best_error > 0.1:
        raise MediaInputError(
            "unsupported_display_transform",
            "Only right-angle rotation and mirror display transforms are supported.",
        )
    return DisplayTransform(rotation_degrees=best[0], horizontal_flip=best[1])


def _display_transform(frame: av.VideoFrame, metadata: dict[str, str]) -> DisplayTransform:
    if SideDataType.DISPLAYMATRIX in frame.side_data:
        side_data = frame.side_data[SideDataType.DISPLAYMATRIX]
        return _transform_from_display_matrix(bytes(side_data))
    return _rotation_from_metadata(metadata)


def _frame_timing(
    index: int,
    frame: av.VideoFrame,
    start_ms: Fraction | None,
) -> tuple[FrameTiming, Fraction]:
    if frame.pts is None or frame.time_base is None:
        raise MediaInputError("missing_pts", "A decoded frame has no presentation timestamp.")
    time_base = _fraction(frame.time_base)
    original_ms = Fraction(int(frame.pts)) * time_base * 1000
    anchor = original_ms if start_ms is None else start_ms
    duration_ms: float | None = None
    if frame.duration is not None and int(frame.duration) > 0:
        duration_ms = _round_ms(Fraction(int(frame.duration)) * time_base * 1000)
    return (
        FrameTiming(
            source_frame_index=index,
            pts=int(frame.pts),
            time_base_numerator=time_base.numerator,
            time_base_denominator=time_base.denominator,
            original_pts_ms=_round_ms(original_ms),
            timestamp_ms=_round_ms(original_ms - anchor),
            duration_ms=duration_ms,
        ),
        anchor,
    )


def _timing_summary(timeline: list[FrameTiming]) -> tuple[float, float | None, bool]:
    if not timeline:
        raise MediaInputError("no_video_frames", "The video stream decoded no frames.")
    if len(timeline) > MAX_FRAME_COUNT:
        raise MediaInputError("too_many_frames", "The video exceeds the 20,000-frame limit.")
    for previous, current in zip(timeline, timeline[1:]):
        if current.timestamp_ms <= previous.timestamp_ms:
            raise MediaInputError(
                "non_monotonic_pts",
                "Decoded presentation timestamps repeat or move backwards.",
            )

    deltas = [
        current.timestamp_ms - previous.timestamp_ms
        for previous, current in zip(timeline, timeline[1:])
    ]
    variable = False
    nominal_fps: float | None = None
    fallback_duration: float | None = None
    if deltas:
        median_delta = statistics.median(deltas)
        tolerance = max(1.0, median_delta * 0.05)
        variable = any(abs(delta - median_delta) > tolerance for delta in deltas)
        fallback_duration = median_delta
        if not variable:
            span = timeline[-1].timestamp_ms - timeline[0].timestamp_ms
            if span > 0:
                nominal_fps = round((len(timeline) - 1) * 1000.0 / span, 6)

    last_duration = timeline[-1].duration_ms or fallback_duration
    if last_duration is None or last_duration <= 0:
        raise MediaInputError("invalid_duration", "The decoded video duration is unavailable.")
    duration_ms = round(timeline[-1].timestamp_ms + last_duration, 6)
    if duration_ms <= 0:
        raise MediaInputError("invalid_duration", "The decoded video duration is invalid.")
    if duration_ms > MAX_DURATION_MS:
        raise MediaInputError("video_too_long", "The decoded video exceeds the 120-second limit.")
    return duration_ms, nominal_fps, variable


def _select_video_stream(container: av.container.InputContainer) -> av.video.stream.VideoStream:
    streams = list(container.streams.video)
    if not streams:
        raise MediaInputError("no_video_stream", "The selected file contains no video stream.")
    if len(streams) != 1:
        raise MediaInputError(
            "multiple_video_streams",
            "Files with more than one video stream are not supported yet.",
        )
    return streams[0]


def _display_dimensions(width: int, height: int, transform: DisplayTransform) -> tuple[int, int]:
    if transform.rotation_degrees in (90, 270):
        return height, width
    return width, height


def probe_video(path: Path | str) -> VideoInspection:
    """Decode a video to validate every frame and preserve its source PTS timeline."""

    resolved, file_size = _safe_file(Path(path))
    digest = _sha256_file(resolved)
    try:
        with av.open(str(resolved), mode="r") as container:
            stream = _select_video_stream(container)
            coded_width = int(stream.codec_context.width)
            coded_height = int(stream.codec_context.height)
            if coded_width <= 0 or coded_height <= 0:
                raise MediaInputError("invalid_dimensions", "The video dimensions are invalid.")
            if coded_width * coded_height > MAX_FRAME_PIXELS:
                raise MediaInputError("resolution_too_large", "Video resolution exceeds the 4K pixel limit.")

            timeline: list[FrameTiming] = []
            start_ms: Fraction | None = None
            transform: DisplayTransform | None = None
            for index, frame in enumerate(container.decode(stream)):
                if frame.is_corrupt:
                    raise MediaInputError("corrupt_frame", "A corrupt video frame was detected.")
                if transform is None:
                    transform = _display_transform(frame, dict(stream.metadata))
                timing, start_ms = _frame_timing(index, frame, start_ms)
                timeline.append(timing)
                if timing.timestamp_ms > MAX_DURATION_MS:
                    raise MediaInputError("video_too_long", "The decoded video exceeds the 120-second limit.")
                if len(timeline) > MAX_FRAME_COUNT:
                    raise MediaInputError("too_many_frames", "The video exceeds the 20,000-frame limit.")

            if transform is None or start_ms is None:
                raise MediaInputError("no_video_frames", "The video stream decoded no frames.")
            duration_ms, nominal_fps, variable = _timing_summary(timeline)
            width, height = _display_dimensions(coded_width, coded_height, transform)
            codec = stream.codec_context.name or "unknown"
            container_format = container.format.name or "unknown"
    except MediaInputError:
        raise
    except av.error.DecoderNotFoundError as exc:
        raise MediaInputError("unsupported_codec", "No decoder is available for this video codec.") from exc
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise MediaInputError("decode_failed", "The selected file could not be decoded as video.") from exc

    return VideoInspection(
        input_sha256=digest,
        file_size_bytes=file_size,
        container_format=container_format,
        codec=codec,
        coded_width=coded_width,
        coded_height=coded_height,
        width=width,
        height=height,
        duration_ms=duration_ms,
        frame_count=len(timeline),
        original_pts_start_ms=_round_ms(start_ms),
        nominal_fps=nominal_fps,
        variable_frame_rate=variable,
        display_transform=transform,
        timeline=tuple(timeline),
    )


def apply_display_transform(
    image_rgb: np.ndarray,
    transform: DisplayTransform,
    *,
    correct_mirror: bool = False,
) -> np.ndarray:
    """Apply container display metadata, then optional user-confirmed mirror correction."""

    result = np.rot90(image_rgb, k=transform.rotation_degrees // 90)
    if transform.horizontal_flip:
        result = np.fliplr(result)
    if correct_mirror:
        result = np.fliplr(result)
    return np.ascontiguousarray(result)


def decode_video(
    path: Path | str,
    *,
    inspection: VideoInspection | None = None,
    correct_mirror: bool = False,
) -> Iterator[DecodedFrame]:
    """Yield RGB frames mapped to source frame indices and normalized decoded PTS."""

    resolved, _ = _safe_file(Path(path))
    expected = inspection or probe_video(resolved)
    if _sha256_file(resolved) != expected.input_sha256:
        raise MediaInputError("input_changed", "The selected video changed after it was probed.")
    try:
        with av.open(str(resolved), mode="r") as container:
            stream = _select_video_stream(container)
            start_ms: Fraction | None = None
            count = 0
            for index, frame in enumerate(container.decode(stream)):
                timing, start_ms = _frame_timing(index, frame, start_ms)
                if index >= len(expected.timeline):
                    raise MediaInputError("timeline_changed", "The decoded frame count changed after probing.")
                expected_timing = expected.timeline[index]
                if (
                    timing.pts != expected_timing.pts
                    or timing.time_base_numerator != expected_timing.time_base_numerator
                    or timing.time_base_denominator != expected_timing.time_base_denominator
                ):
                    raise MediaInputError("timeline_changed", "The decoded PTS timeline changed after probing.")
                image = frame.to_ndarray(format="rgb24")
                image = apply_display_transform(
                    image,
                    expected.display_transform,
                    correct_mirror=correct_mirror,
                )
                count += 1
                yield DecodedFrame(index, timing.timestamp_ms, image)
            if count != expected.frame_count:
                raise MediaInputError("timeline_changed", "The decoded frame count changed after probing.")
    except MediaInputError:
        raise
    except av.error.DecoderNotFoundError as exc:
        raise MediaInputError("unsupported_codec", "No decoder is available for this video codec.") from exc
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise MediaInputError("decode_failed", "The selected file could not be decoded as video.") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe a local video without uploading it or printing its private path."
    )
    parser.add_argument("action", choices=("probe",))
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Include source frame PTS and normalized timestamps in JSON output.",
    )
    parser.add_argument(
        "--decode-first-frame",
        action="store_true",
        help="Also decode one display-normalized RGB frame and report its shape.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inspection = probe_video(args.video)
        result = inspection.to_dict(include_timeline=args.timeline)
        if args.decode_first_frame:
            first = next(decode_video(args.video, inspection=inspection))
            result["first_frame"] = {
                "source_frame_index": first.source_frame_index,
                "timestamp_ms": first.timestamp_ms,
                "rgb_shape": list(first.image_rgb.shape),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except (MediaInputError, StopIteration) as exc:
        if isinstance(exc, MediaInputError):
            code = exc.code
            message = str(exc)
        else:
            code = "no_video_frames"
            message = "The video stream decoded no frames."
        print(f"VIDEO_INPUT_ERROR[{code}]: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
