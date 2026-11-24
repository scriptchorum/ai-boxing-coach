"""One-command diagnostic script to inspect punch, kick, and footwork counts for a local video."""

from __future__ import annotations

import argparse
import pathlib
import sys

from boxing_coach.muaythai_metrics import analyze_muaythai_arsenal
from boxing_coach.pose_backend import MediaPipePoseConfig, extract_pose_run
from boxing_coach.preview import load_preview
from boxing_coach.rear_straight_review import (
    analyze_jab_and_one_two,
    compute_rear_straight_metrics,
    segment_rear_straights,
)
from boxing_coach.stance_calibration import calibrate_stance_from_frames
from boxing_coach.view_gate import classify_view_from_frames


def main() -> int:
    parser = argparse.ArgumentParser(description="导出单条视频的拳腿计数与时间点对账摘要")
    parser.add_argument("video_path", help="本地视频路径")
    parser.add_argument("--stance", default="auto", choices=["auto", "orthodox", "southpaw"], help="站架")
    parser.add_argument("--max-seconds", type=float, default=30.0, help="最多分析前多少秒")
    args = parser.parse_args()

    vp = pathlib.Path(args.video_path).expanduser().resolve()
    if not vp.exists():
        print(f"找不到视频文件: {vp}", file=sys.stderr)
        return 1

    preview = load_preview(str(vp), allow_extended_formats=True)
    fps = float(preview.fps or 30.0)
    end_frame = min(preview.frame_count - 1, max(1, int(args.max_seconds * fps)))
    pose_run = extract_pose_run(
        preview,
        start_frame=0,
        end_frame=end_frame,
        config=MediaPipePoseConfig(),
    )
    frames = pose_run["frames"]
    w, h = int(preview.width), int(preview.height)

    stance = args.stance
    if stance == "auto":
        cal = calibrate_stance_from_frames(frames)
        stance = cal.get("stance", "orthodox")
        if stance not in ("orthodox", "southpaw"):
            stance = "orthodox"
    view_info = classify_view_from_frames(frames)

    rear_seg = segment_rear_straights(frames, width=w, height=h, stance=stance)
    rear_metrics = compute_rear_straight_metrics(rear_seg, single_person_confirmed=True)
    combo_res = analyze_jab_and_one_two(
        frames,
        rear_segmentation=rear_seg,
        width=w,
        height=h,
        stance=stance,
        single_person_confirmed=True,
    )
    mt_res = analyze_muaythai_arsenal(
        frames,
        stance=stance,
        single_person_confirmed=True,
        width=w,
        height=h,
        punch_events=combo_res["straight_events"],
    )

    print(f"=== 视频对账摘要: {vp.name} ===")
    print(
        f"分辨率: {w}x{h}, FPS: {fps:.2f}, 分析帧数: {len(frames)}, "
        f"自动站架: {stance}, 机位: {view_info.get('view_label')}"
    )
    print(
        f"后手直拳(已回落): {rear_metrics['completed_count']} | "
        f"前手刺拳: {combo_res['jab_attempt_count']} | "
        f"平勾拳: {combo_res['hook_count']} | 上勾拳: {combo_res['uppercut_count']} | "
        f"1-2: {combo_res['combo_count']} | 1-2-3: {combo_res['one_two_three_count']}"
    )
    print(
        f"泰拳 -> 扫踢: {mt_res['kick_count']} | 正蹬(Stab): {mt_res['teep_count']} | "
        f"冲膝: {mt_res['knee_count']} | 提膝格挡: {mt_res['shin_check_count']} | "
        f"步伐: {mt_res['step_count']} | 换架: {mt_res['stance_switch_count']}"
    )

    punches = sorted(
        combo_res["straight_events"] + combo_res["bent_arm_punches"] + combo_res.get("unorthodox_punches", []),
        key=lambda e: e["timestamp_ms"],
    )
    print("\n--- 出拳明细 ---")
    for e in punches:
        print(f"  t={e['timestamp_ms']/1000:6.2f}s  frame={e['frame_index']:4d}  {e['punch_type']}")

    print("\n--- 腿法与步伐明细 ---")
    for e in sorted(mt_res["leg_events"] + mt_res["shin_checks"] + mt_res["footwork_events"], key=lambda x: x["timestamp_ms"]):
        print(f"  t={e['timestamp_ms']/1000:6.2f}s  frame={e['frame_index']:4d}  {e['technique_name']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
