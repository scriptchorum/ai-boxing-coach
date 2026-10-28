"""Local video review and experimental pose inspection; no action assessment."""

from __future__ import annotations

import streamlit as st
from boxing_coach.preview import load_preview, read_preview_frame, selection_times
from boxing_coach.video_io import MediaInputError
from boxing_coach.synced_player import render_synced_player
from boxing_coach.pose_ui import render_pose_panel


def clear_video() -> None:
    for key in ("video", "frame_index", "start_frame", "end_frame", "mirror", "view_mode", "pose_run", "pose_index"):
        st.session_state.pop(key, None)


def move_frame(delta: int) -> None:
    video = st.session_state.video
    st.session_state.frame_index = max(0, min(video.inspection.frame_count - 1, st.session_state.frame_index + delta))

st.set_page_config(page_title="女性搏击视频复盘工具", page_icon="🥊", layout="wide")
st.title("女性搏击视频复盘工具")
st.caption("以女性身体多样性为核心 · 本地自用预览版")
st.info("现在可以看视频、选取训练区间，并手动启动本地人体关键点试跑。尚无动作分析，不提供评分、拳力或伤病判断。")
st.write("视频仅在这台电脑读取和显示，不上传云端，也不修改原片。请在文件资源管理器中复制视频路径，粘贴到下方。")
with st.form("open_video"):
    path = st.text_input("本机 MP4 视频路径", placeholder='粘贴完整文件路径，可带双引号', key="input_path")
    submitted = st.form_submit_button("打开视频")
if submitted:
    clear_video()
    try:
        with st.spinner("检查完整视频与时间轴…"):
            st.session_state.video = load_preview(path)
    except MediaInputError as exc:
        st.error(f"无法打开视频 [{exc.code}]：{exc}")
    except (ValueError, OSError):
        st.error("路径无法读取，请重新复制文件路径。")

video = st.session_state.get("video")
if video is None:
    st.caption("首版范围：无旋转/反射元数据的 H.264 / MP4、8-bit yuv420p；单片最多 120 秒、512 MiB。推荐固定机位、全身留边、背景无人。")
    st.stop()

inspection = video.inspection
last = inspection.frame_count - 1
st.button("关闭当前视频", on_click=clear_video)
st.success(f"已打开：{video.path.name}")
st.caption(f"{inspection.width} × {inspection.height} · {inspection.duration_ms / 1000:.2f} 秒 · {inspection.frame_count} 帧 · 使用原始视频时间戳")
mirror = st.checkbox("拍摄时开启了镜像保存：纠正右侧画面", value=False, key="mirror")
st.caption("未开启镜像保存时保持不勾选。不确定时先保留原画面；这里不会自动猜测左右手。")
mode = st.radio("查看方式", ["联动播放", "手动逐帧"], horizontal=True, key="view_mode")
if mode == "联动播放":
    try:
        # Revalidate the original file before rendering the immutable local snapshot.
        read_preview_frame(video, 0)
        render_synced_player(video, mirror)
        st.caption("同步画面来自浏览器当前播放帧；精确源帧号与 PTS 核对请切换手动逐帧。更改镜像或选段后，联动播放器会重新加载。")
    except MediaInputError as exc:
        clear_video()
        st.error(f"视频已失效 [{exc.code}]，请重新打开。")
        st.stop()
    except ValueError as exc:
        st.warning(str(exc))
else:
    st.number_input("查看帧号（从 0 开始）", min_value=0, max_value=last, step=1, key="frame_index")
    prev, nxt = st.columns(2)
    prev.button("上一帧", on_click=move_frame, args=(-1,), disabled=st.session_state.frame_index == 0)
    nxt.button("下一帧", on_click=move_frame, args=(1,), disabled=st.session_state.frame_index == last)

    try:
        with st.spinner("定位源帧…"):
            frame = read_preview_frame(video, int(st.session_state.frame_index), correct_mirror=mirror)
    except MediaInputError as exc:
        clear_video()
        st.error(f"视频已失效 [{exc.code}]，请重新打开。")
        st.stop()

    left, right = st.columns(2)
    with left:
        st.subheader("原片播放")
        st.video(video.content, format="video/mp4")
        st.caption("保留拍摄画面和原始音轨。播放器用于观看，不与右侧帧号自动联动；精确时间以右侧为准。")
    with right:
        st.subheader("逐帧核对")
        st.image(frame.image_rgb, caption=f"源帧 {frame.source_frame_index} · {frame.timestamp_ms / 1000:.6f} 秒" + (" · 已纠正镜像" if mirror else " · 保留原画面"), width="stretch")

st.subheader("选择稳定的训练区间")
st.write("避开拿取手机、出框和遮挡；这只是手动选择，不代表系统判断动作合格。")
start_col, end_col = st.columns(2)
start = start_col.number_input("起始帧（包含）", min_value=0, max_value=last, value=0, step=1, key="start_frame")
end = end_col.number_input("结束帧（包含）", min_value=0, max_value=last, value=last, step=1, key="end_frame")
try:
    begin_ms, stop_ms = selection_times(inspection, int(start), int(end))
    selection_valid = True
    st.write(f"选中 {int(end) - int(start) + 1} 帧；区间 {begin_ms / 1000:.6f}–{stop_ms / 1000:.6f} 秒（结束时间不包含），共 {(stop_ms - begin_ms) / 1000:.3f} 秒。")
except ValueError as exc:
    selection_valid = False
    st.error(str(exc))
st.caption("本次选择仅保留在当前会话，关闭或刷新后需重新选择；不会裁剪、覆盖原片或保存训练历史。")
if selection_valid:
    render_pose_panel(video,int(start),int(end),mirror)
