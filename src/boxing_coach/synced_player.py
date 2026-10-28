"""Browser-side paired preview, independent of pose inference and exact PTS review."""
from pathlib import Path

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


def player_payload(video: PreviewVideo, correct_mirror: bool) -> bytes:
    if len(video.content) > MAX_SYNC_BYTES:
        raise ValueError('联动播放暂限 64 MiB；较大文件请切换“手动逐帧”。')
    # Raw binary transport: one mirror flag byte, then the immutable MP4 snapshot.
    # No filenames, paths, external endpoints, or executable user content.
    return bytes([int(correct_mirror)]) + video.content


def render_synced_player(video: PreviewVideo, correct_mirror: bool) -> None:
    _player(data=player_payload(video, correct_mirror), key='synced_player', height='content')
