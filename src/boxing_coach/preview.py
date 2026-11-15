"""Restricted local preview helpers; no inference, uploads or persistent media cache."""

from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
from pathlib import Path

import av
import numpy as np

from boxing_coach.video_io import (
    SUPPORTED_EXTENSIONS,
    DecodedFrame,
    MediaInputError,
    VideoInspection,
    decode_video,
    probe_video,
)


@dataclass(frozen=True)
class PreviewVideo:
    path: Path
    inspection: VideoInspection
    content: bytes


def _build_browser_playable_mp4_bytes(source: Path, inspection: VideoInspection) -> bytes:
    """Transcode rotated/HEVC/MOV streams into upright browser-playable H.264 MP4 bytes in memory."""
    buf = io.BytesIO()
    even_w = max(2, inspection.width - (inspection.width % 2))
    even_h = max(2, inspection.height - (inspection.height % 2))
    with av.open(buf, mode="w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=1000)
        stream.width = even_w
        stream.height = even_h
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.options = {"preset": "ultrafast", "crf": "23"}
        with closing(decode_video(source, inspection=inspection, correct_mirror=False)) as decoded:
            for item in decoded:
                rgb = item.image_rgb[:even_h, :even_w]
                vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
                vf.pts = int(round(item.timestamp_ms))
                vf.time_base = Fraction(1, 1000)
                for packet in stream.encode(vf):
                    out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return buf.getvalue()


def load_preview(path: str | Path, *, allow_extended_formats: bool = False) -> PreviewVideo:
    source = Path(str(path).strip().strip('"')).expanduser().resolve()
    suffix = source.suffix.lower()
    if not allow_extended_formats and suffix != ".mp4":
        raise MediaInputError("preview_format", "首版仅支持 H.264 编码的 MP4 文件。")
    if allow_extended_formats and suffix not in SUPPORTED_EXTENSIONS:
        raise MediaInputError("preview_format", "仅支持 MP4、MOV、M4V、MKV、WEBM 或 AVI 视频文件。")
    inspection = probe_video(source)
    is_mp4_container = "mp4" in inspection.container_format.split(",")
    if not allow_extended_formats and (inspection.codec != "h264" or not is_mp4_container):
        raise MediaInputError("preview_format", "首版仅支持 H.264 编码的 MP4 文件；改扩展名不能转换编码。")
    transform = inspection.display_transform
    has_transform = bool(transform.rotation_degrees or transform.horizontal_flip)
    if not allow_extended_formats and has_transform:
        raise MediaInputError("preview_transform", "首版暂不接受带旋转或反射元数据的视频，请保留原片，等待后续格式支持。")
    try:
        with av.open(str(source)) as container:
            pix_fmt = container.streams.video[0].codec_context.format.name
            if not allow_extended_formats and pix_fmt != "yuv420p":
                raise MediaInputError("preview_pixel_format", "首版仅支持 8-bit yuv420p H.264 视频。")
        raw_bytes = source.read_bytes()
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise MediaInputError("file_unavailable", "视频无法读取，请重新选择。") from exc
    if hashlib.sha256(raw_bytes).hexdigest() != inspection.input_sha256:
        raise MediaInputError("input_changed", "视频在检查后发生变化，请重新打开。")

    needs_browser_norm = (
        allow_extended_formats
        and (inspection.codec != "h264" or not is_mp4_container or has_transform or pix_fmt != "yuv420p")
    )
    if needs_browser_norm:
        try:
            content = _build_browser_playable_mp4_bytes(source, inspection)
        except Exception:
            content = raw_bytes
    else:
        content = raw_bytes
    return PreviewVideo(source, inspection, content)


def read_preview_frame(video: PreviewVideo, index: int, *, correct_mirror: bool = False) -> DecodedFrame:
    if not 0 <= index < video.inspection.frame_count:
        raise ValueError("帧号超出视频范围。")
    # Closing the generator also closes the AV container when returning early.
    with closing(decode_video(video.path, inspection=video.inspection, correct_mirror=correct_mirror)) as frames:
        for frame in frames:
            if frame.source_frame_index == index:
                return frame
    raise MediaInputError("timeline_changed", "目标帧已不存在，请重新打开视频。")


def selection_times(inspection: VideoInspection, start: int, end: int) -> tuple[float, float]:
    """Inclusive frame selection -> half-open PTS interval in milliseconds."""
    if not 0 <= start <= end < inspection.frame_count:
        raise ValueError("起始帧不能晚于结束帧，且必须在视频范围内。")
    stop = inspection.timeline[end + 1].timestamp_ms if end + 1 < inspection.frame_count else inspection.duration_ms
    return inspection.timeline[start].timestamp_ms, stop
