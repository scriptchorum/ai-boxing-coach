"""Local video review and experimental pose inspection; no action assessment."""

from __future__ import annotations

import streamlit as st
from boxing_coach.preview import load_preview, read_preview_frame, selection_times
from boxing_coach.video_io import MediaInputError
from boxing_coach.synced_player import render_synced_player
from boxing_coach.pose_ui import render_pose_panel


def clear_video() -> None:
    for key in ("video", "frame_index", "start_frame", "end_frame", "mirror", "view_mode",
                "pose_run", "pose_index", "pose_reviews", "pose_review_filter",
                "review_unrecorded_only", "pose_evidence_bundle", "pose_strike_audit",
                "pose_rear_straight_overrides", "pose_review_annotation", "pose_local_history",
                "pose_demo_frames", "pose_demo_report", "auto_run_pose"):
        st.session_state.pop(key, None)


def _auto_clamped_end_frame(inspection, start_frame: int = 0) -> int:
    """Find the largest end frame within the 30s / 1800-frame single-run limit."""
    last_idx = inspection.frame_count - 1
    max_end = min(last_idx, start_frame + 1800 - 1)
    start_ms = inspection.timeline[start_frame].timestamp_ms
    chosen = start_frame
    for idx in range(start_frame, max_end + 1):
        _, stop_ms = selection_times(inspection, start_frame, idx)
        if stop_ms - start_ms <= 30000.0:
            chosen = idx
        else:
            break
    return chosen


def move_frame(delta: int) -> None:
    video = st.session_state.video
    st.session_state.frame_index = max(0, min(video.inspection.frame_count - 1, st.session_state.frame_index + delta))

st.set_page_config(page_title="女性搏击视频复盘工具", page_icon="🥊", layout="wide")
st.title("女性搏击视频复盘工具")
st.caption("以女性身体多样性为核心 · 一站式本地自动复盘版")
st.markdown('**极简流程：粘贴路径 → 点击【🚀 一键打开并直接生成动作复盘】→ 直接查看回防评分、最高优先建议、纠错处方与骨架回放。**')
st.info("视频仅在本机读取和分析，不上传云端，也不修改原片。若视频超过 30 秒，一键复盘会自动截取前 30 秒合规区间直接出分。")
with st.form("open_video"):
    path = st.text_input("本机 MP4 视频路径", placeholder='粘贴完整文件路径（可带双引号），回车即可一键打开并分析', key="input_path")
    btn_c1, btn_c2 = st.columns([2, 1])
    submitted_auto = btn_c1.form_submit_button("🚀 一键打开并直接生成动作复盘", type="primary")
    submitted_open = btn_c2.form_submit_button("打开视频")
if submitted_auto or submitted_open:
    clear_video()
    try:
        with st.spinner("检查完整视频与时间轴…"):
            loaded = load_preview(path)
            st.session_state.video = loaded
            if submitted_auto:
                st.session_state.start_frame = 0
                st.session_state.end_frame = _auto_clamped_end_frame(loaded.inspection, 0)
                st.session_state.auto_run_pose = True
    except MediaInputError as exc:
        st.error(f"无法打开视频 [{exc.code}]：{exc}")
    except (ValueError, OSError):
        st.error("路径无法读取，请重新复制文件路径。")

video = st.session_state.get("video")
if video is None:
    st.caption("支持范围：无旋转/反射元数据的 H.264 / MP4、8-bit yuv420p；单片最多 120 秒、512 MiB。支持全身及半身（肩肘腕可见）训练视频。")
    st.stop()

inspection = video.inspection
last = inspection.frame_count - 1
top_bar1, top_bar2, top_bar3 = st.columns([3, 2, 1])
top_bar1.success(f"当前视频：{video.path.name}（{inspection.width}×{inspection.height} · {inspection.duration_ms / 1000:.2f} 秒 · {inspection.frame_count} 帧）")
mirror = top_bar2.checkbox("拍摄时开启了镜像保存：纠正右侧画面", value=False, key="mirror")
top_bar3.button("关闭当前视频", on_click=clear_video)

# Quick switcher for sibling MP4 files in the same directory
try:
    sibling_mp4s = sorted(
        p for p in video.path.parent.glob("*.mp4") if p.is_file()
    )
except OSError:
    sibling_mp4s = []
if len(sibling_mp4s) > 1:
    with st.expander(f"📂 同文件夹下发现 {len(sibling_mp4s)} 个 MP4 视频（点击可一键切换并直接复盘）", expanded=False):
        sc1, sc2 = st.columns([3, 1])
        chosen_sibling = sc1.selectbox(
            "选择同目录训练视频",
            sibling_mp4s,
            index=sibling_mp4s.index(video.path) if video.path in sibling_mp4s else 0,
            format_func=lambda p: p.name,
            key="sibling_video_select",
        )
        if sc2.button("🚀 切换并直接复盘该视频", key="switch_sibling_auto_btn"):
            clear_video()
            try:
                loaded_sib = load_preview(str(chosen_sibling))
                st.session_state.video = loaded_sib
                st.session_state.start_frame = 0
                st.session_state.end_frame = _auto_clamped_end_frame(loaded_sib.inspection, 0)
                st.session_state.auto_run_pose = True
                st.rerun()
            except (MediaInputError, ValueError, OSError) as exc:
                st.error(f"切换视频失败：{exc}")

with st.expander("🎬 原片预览与训练区间调整（需截取特定片段或逐帧看原片时展开）", expanded=(st.session_state.get("pose_run") is None and not st.session_state.get("auto_run_pose"))):
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
    st.write("请选择单人可见、画面稳定的区间（单次分析最多 30 秒、1800 帧）。")
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
