"""Browser-side paired preview, independent of pose inference and exact PTS review."""
from pathlib import Path
import json
import struct

import streamlit.components.v2 as components

from boxing_coach.preview import PreviewVideo


MAX_SYNC_BYTES = 64 * 1024 * 1024
ASSETS = Path(__file__).with_name('player')
_player = components.component(
    'boxing_synced_player',
    html=(ASSETS / 'player.html').read_text(encoding='utf-8'),
    css=(ASSETS / 'player.css').read_text(encoding='utf-8'),
    js=(ASSETS / 'player.js').read_text(encoding='utf-8'),
)


def player_payload(video: PreviewVideo, correct_mirror: bool, pose_run=None) -> bytes:
    if len(video.content) > MAX_SYNC_BYTES:
        raise ValueError('联动播放暂限 64 MiB；较大文件请切换“手动逐帧”。')
    # Raw binary transport: one mirror flag byte, then the immutable MP4 snapshot.
    # No filenames, paths, external endpoints, or executable user content.
    if pose_run is None:
        return bytes([int(correct_mirror)]) + video.content
    from .pose import EDGES, POINT_STYLES, CENTER_COLOR, drawable, run_key
    if pose_run.key != run_key(video, pose_run.start, pose_run.end, correct_mirror):
        raise ValueError('关键点与当前视频或镜像不一致，请重新识别。')
    # Non-zero container origins need separate browser timeline validation.
    if abs(video.inspection.original_pts_start_ms) > 0.001:
        raise ValueError('此视频的起始时间偏移尚未验证骨架联动，请使用逐帧核对。')
    frames = []
    from .quality import assess_frames, reason_text, display_frame, VERSION
    decisions = assess_frames(pose_run)
    for frame, diagnostic, decision in zip(pose_run.frames, pose_run.diagnostics, decisions):
        points = {str(p['index']): [p['image']['x'], p['image']['y']]
                  for p in display_frame(frame,decision)['landmarks'] if drawable(p)}
        frames.append({'time': frame['timestamp_ms']/1000,
                       'index': frame['source_frame_index'],
                       'status': diagnostic['status'] if decision['show'] else 'quality_rejected',
                       'reason':reason_text(decision), 'hidden_parts':decision['hidden_parts'],
                       'points': points if decision['show'] else {}})
    metadata = json.dumps({'frames':frames,'edges':EDGES,'quality_version':VERSION,
                           'point_styles':POINT_STYLES,'center_color':CENTER_COLOR}, separators=(',',':'),allow_nan=False).encode('utf-8')
    # Flag bit 1 introduces a little-endian metadata length + JSON, before MP4.
    return bytes([2 + int(correct_mirror)]) + struct.pack('<I',len(metadata)) + metadata + video.content


def render_synced_player(video: PreviewVideo, correct_mirror: bool, pose_run=None) -> None:
    _player(data=player_payload(video, correct_mirror, pose_run),
            key='pose_player' if pose_run is not None else 'synced_player', height='content')
