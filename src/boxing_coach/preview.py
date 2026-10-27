"""Restricted local preview helpers; no inference, uploads or persistent media cache."""

from contextlib import closing
from dataclasses import dataclass
import hashlib
from pathlib import Path

import av

from boxing_coach.video_io import DecodedFrame, MediaInputError, VideoInspection, decode_video, probe_video


@dataclass(frozen=True)
class PreviewVideo:
    path: Path
    inspection: VideoInspection
    content: bytes


def load_preview(path: str | Path) -> PreviewVideo:
    source = Path(str(path).strip().strip('"')).expanduser().resolve()
    if source.suffix.lower() != ".mp4":
        raise MediaInputError("preview_format", "首版仅支持 H.264 编码的 MP4 文件。")
    inspection = probe_video(source)
    if inspection.codec != "h264" or "mp4" not in inspection.container_format.split(","):
        raise MediaInputError("preview_format", "首版仅支持 H.264 编码的 MP4 文件；改扩展名不能转换编码。")
    transform = inspection.display_transform
    if transform.rotation_degrees or transform.horizontal_flip:
        raise MediaInputError("preview_transform", "首版暂不接受带旋转或反射元数据的视频，请保留原片，等待后续格式支持。")
    try:
        with av.open(str(source)) as container:
            if container.streams.video[0].codec_context.format.name != "yuv420p":
                raise MediaInputError("preview_pixel_format", "首版仅支持 8-bit yuv420p H.264 视频。")
        content = source.read_bytes()
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise MediaInputError("file_unavailable", "视频无法读取，请重新选择。") from exc
    if hashlib.sha256(content).hexdigest() != inspection.input_sha256:
        raise MediaInputError("input_changed", "视频在检查后发生变化，请重新打开。")
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
