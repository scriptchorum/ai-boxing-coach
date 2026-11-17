"""Standalone offline HTML training scorecard and multi-video batch comparison helper."""
from __future__ import annotations

import base64
import html
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

from .muaythai_metrics import analyze_muaythai_arsenal
from .pose import draw_pose, extract_pose
from .preview import load_preview, read_preview_frame, selection_times
from .rear_straight_review import (
    analyze_jab_and_one_two,
    build_history_entry,
    compute_rear_straight_metrics,
    segment_rear_straights,
)


def summarize_frames_for_batch(
    frames: Sequence[dict[str, Any]],
    *,
    entry_id: str,
    label: str,
    width: int = 1280,
    height: int = 720,
    stance: str = "orthodox",
    view: str = "side",
    mirror: bool = False,
    model_id: str = "mediapipe_pose_landmarker_lite",
) -> dict[str, Any]:
    """Compute a complete single-clip summary row and history entry from extracted pose frames."""
    seg = segment_rear_straights(
        frames,
        width=width,
        height=height,
        stance=stance,
        sample_id=entry_id,
    )
    rear_metrics = compute_rear_straight_metrics(seg, single_person_confirmed=True)
    combo_res = analyze_jab_and_one_two(
        frames,
        rear_segmentation=seg,
        width=width,
        height=height,
        stance=stance,
        sample_id=entry_id,
        single_person_confirmed=True,
    )
    mt_res = analyze_muaythai_arsenal(
        frames, width=width, height=height,
        punch_events=combo_res["straight_events"], stance=stance,
        single_person_confirmed=True,
    )
    hist_entry = build_history_entry(
        rear_metrics,
        entry_id=entry_id,
        label=label,
        mirror=mirror,
        view=view,
        model_id=model_id,
    )
    hg_rate = mt_res.get("high_guard_retention_rate")
    row = {
        "视频片段": label,
        "回防评分": rear_metrics.get("defense_grade", "暂无"),
        "后手直拳 (已回落/保持满 100 ms)": f"{rear_metrics.get('completed_count', rear_metrics['valid_count'])} / {rear_metrics['valid_count']}",
        "中位回手时间 (ms)": (
            f"{hist_entry['median_return_time_ms']:.1f}"
            if hist_entry.get("median_return_time_ms") is not None
            else "-"
        ),
        "已识别出拳": len(combo_res["straight_events"]) + len(combo_res["bent_arm_punches"]),
        "步伐 / 换架": f"{mt_res['step_count']} / {mt_res['stance_switch_count']}",
        "刺拳 / 平勾 / 上勾": (
            f"{combo_res['jab_attempt_count']} / {combo_res['hook_count']} / {combo_res['uppercut_count']}"
        ),
        "1-2 / 1-2-3 连击": f"{combo_res['combo_count']} 组 / {combo_res['one_two_three_count']} 组",
        "泰拳高抱架率": f"{hg_rate * 100:.0f}%" if hg_rate is not None else "-",
        "扫踢 / 正蹬 / 冲膝 / 提膝格挡": (
            f"{mt_res['kick_count']} / {mt_res['teep_count']} / {mt_res['knee_count']} / {mt_res['shin_check_count']}"
        ),
        "头号优先建议": rear_metrics.get("top_priority_advice", ""),
        "训练处方 (B01)": (rear_metrics.get("drill_prescription") or {}).get("title", "-"),
    }
    return {
        "summary_row": row,
        "history_entry": hist_entry,
        "rear_metrics": rear_metrics,
        "combo_res": combo_res,
        "mt_res": mt_res,
    }


def _auto_clamped_end_frame(inspection: Any, start_frame: int = 0) -> int:
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


def batch_review_video_files(
    video_paths: Sequence[Path | str],
    *,
    stance: str = "orthodox",
    view: str = "side",
    mirror: bool = False,
    max_clips: int = 8,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Batch-open and analyze multiple .mp4 files in a folder for 1-click comparison."""
    selected = [Path(p) for p in video_paths[:max_clips]]
    rows: list[dict[str, Any]] = []
    history_entries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    total = len(selected)
    for idx, p in enumerate(selected, start=1):
        if progress_callback is not None:
            progress_callback(idx - 1, total, p.name)
        try:
            vid = load_preview(str(p), allow_extended_formats=True)
            end_frame = _auto_clamped_end_frame(vid.inspection, 0)
            run = extract_pose(vid, 0, end_frame, correct_mirror=mirror)
            item = summarize_frames_for_batch(
                run.frames,
                entry_id=f"batch-{idx:02d}",
                label=f"{p.name} (0–{end_frame})",
                width=vid.inspection.width,
                height=vid.inspection.height,
                stance=stance,
                view=view,
                mirror=mirror,
                model_id=run.model.get("id", "mediapipe_pose_landmarker_lite"),
            )
            rows.append(item["summary_row"])
            history_entries.append(item["history_entry"])
        except Exception as exc:
            skipped.append({"file": p.name, "reason": str(exc)})

    if progress_callback is not None:
        progress_callback(total, total, "完成")

    return {
        "processed_count": len(rows),
        "skipped_count": len(skipped),
        "rows": rows,
        "history_entries": history_entries,
        "skipped": skipped,
    }


def extract_keyframe_previews(
    video: Any,
    pose_run: Any,
    *,
    mirror: bool,
    rear_metrics: dict[str, Any],
    combo_res: dict[str, Any],
    mt_res: dict[str, Any],
    max_frames: int = 4,
) -> list[dict[str, Any]]:
    """Render up to max_frames keyframe skeleton overlay PNGs (base64) for the standalone HTML scorecard."""
    if video is None or pose_run is None or not getattr(pose_run, "frames", None):
        return []
    frame_map = {int(f["source_frame_index"]): f for f in pose_run.frames}
    candidates: list[tuple[int, str]] = []
    top_f = rear_metrics.get("top_priority_frame_index")
    if top_f is not None and int(top_f) in frame_map:
        candidates.append((int(top_f), "头号优先建议对应帧"))
    for act in rear_metrics.get("action_metrics", []):
        ext_f = act.get("extension_frame_index")
        if act.get('status') == 'valid' and ext_f is not None and int(ext_f) in frame_map:
            candidates.append((int(ext_f), f"后手直拳峰值 ({act['action_id']})"))
    for bp in combo_res.get("bent_arm_punches", []):
        bf = bp.get("frame_index")
        if bf is not None and int(bf) in frame_map:
            candidates.append((int(bf), str(bp["punch_type"])))
    for le in (*mt_res.get("leg_events", []), *mt_res.get("shin_checks", [])):
        lf = le.get("frame_index")
        if lf is not None and int(lf) in frame_map:
            candidates.append((int(lf), str(le["technique_name"])))

    seen: set[int] = set()
    previews: list[dict[str, Any]] = []
    for f_idx, title in candidates:
        if f_idx in seen:
            continue
        seen.add(f_idx)
        try:
            pf = frame_map[f_idx]
            raw = read_preview_frame(video, f_idx, correct_mirror=mirror)
            over = draw_pose(raw.image_rgb, pf)
            buf = BytesIO()
            Image.fromarray(over).save(buf, format="PNG", optimize=False)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            previews.append(
                {
                    "frame_index": f_idx,
                    "timestamp_ms": float(pf.get("timestamp_ms", 0.0)),
                    "title": title,
                    "png_b64": b64,
                }
            )
            if len(previews) >= max_frames:
                break
        except Exception:
            continue
    return previews


def build_standalone_scorecard_html(
    *,
    clip_label: str,
    stance: str,
    view: str,
    rear_metrics: dict[str, Any],
    combo_res: dict[str, Any],
    mt_res: dict[str, Any],
    keyframe_previews: Sequence[dict[str, Any]] = (),
) -> bytes:
    """Build a self-contained offline HTML training scorecard report (B01/B02/B03/T08-T11)."""
    esc = html.escape
    grade = esc(str(rear_metrics.get("defense_grade", "暂无")))
    top_advice = esc(str(rear_metrics.get("top_priority_advice", "")))
    drill = rear_metrics.get("drill_prescription") or {}
    hg_rate = mt_res.get("high_guard_retention_rate")
    hg_label = f"{hg_rate * 100:.0f}%" if hg_rate is not None else "暂无"

    rep = rear_metrics.get("repeatability")
    med_label = f"{rep['median_ms']:.1f} ms" if rep else "需更多完整回手样本"
    iqr_label = f"{rep['iqr_ms']:.1f} ms" if rep else "-"

    action_rows_html = []
    for a in rear_metrics.get("action_metrics", []):
        ret_str = f"{a['return_time_ms']:.1f} ms" if a.get("return_time_ms") is not None else "-"
        hip_str = (
            f"{a['torso_projection_shift_ratio']:.3f}"
            if a.get("torso_projection_shift_ratio") is not None
            else "-"
        )
        reasons_str = "、".join(a.get("quality_reasons", [])) or "完整合规"
        action_rows_html.append(
            f"<tr><td>{esc(str(a['action_id']))}</td>"
            f"<td>{esc(str(a['status']))}</td>"
            f"<td>{esc(str(a['start_frame_index']))}</td>"
            f"<td>{esc(str(a['extension_frame_index']))}</td>"
            f"<td>{esc(str(a.get('return_frame_index') if a.get('return_frame_index') is not None else '-'))}</td>"
            f"<td>{esc(ret_str)}</td>"
            f"<td>{esc(hip_str)}</td>"
            f"<td>{esc(reasons_str)}</td></tr>"
        )
    actions_table_body = "".join(action_rows_html) or "<tr><td colspan='8'>当前区间未检出后手直拳动作</td></tr>"

    combo_rows_html = []
    for c in combo_res.get("one_two_combos", []):
        combo_rows_html.append(
            f"<tr><td>{esc(str(c['combo_id']))}</td>"
            f"<td>1-2 (Jab→Cross)</td>"
            f"<td>{esc(str(c['jab_extension_frame_index']))} → {esc(str(c['rear_extension_frame_index']))}</td>"
            f"<td>{c['combo_interval_ms']:.1f} ms</td>"
            f"<td>{'是' if c['both_returned'] else '否'}</td></tr>"
        )
    for c3 in combo_res.get("one_two_three_combos", []):
        combo_rows_html.append(
            f"<tr><td>{esc(str(c3['combo_id']))}</td>"
            f"<td>{esc(str(c3['combo_name']))}</td>"
            f"<td>{esc(str(c3['jab_frame_index']))} → {esc(str(c3['rear_frame_index']))} → {esc(str(c3['third_punch_frame_index']))}</td>"
            f"<td>2→3: {c3['two_to_three_ms']:.1f} ms</td>"
            f"<td>{'是' if c3['all_returned'] else '否'}</td></tr>"
        )
    combos_table_body = "".join(combo_rows_html) or "<tr><td colspan='5'>当前区间未检出 1-2 或 1-2-3 组合连击</td></tr>"

    mt_rows_html = []
    for mc in mt_res.get("punch_leg_combos", []):
        mt_rows_html.append(
            f"<tr><td>{esc(str(mc['combo_id']))}</td>"
            f"<td>{esc(str(mc['combo_name']))}</td>"
            f"<td>拳帧 {esc(str(mc['punch_frame_index']))} → 腿膝帧 {esc(str(mc['leg_frame_index']))}</td>"
            f"<td>{mc['transition_ms']:.1f} ms</td>"
            f"<td>{'保持' if mc['guard_kept_on_leg'] else '下掉'}</td></tr>"
        )
    mt_table_body = "".join(mt_rows_html) or "<tr><td colspan='5'>当前区间未检出拳腿/拳膝立体连击</td></tr>"

    kf_cards = []
    for kp in keyframe_previews:
        kf_cards.append(
            f"<div class='metric' style='text-align:center'>"
            f"<div class='label'>{esc(str(kp.get('title', '关键帧')))} · 源帧 {esc(str(kp.get('frame_index')))} ({float(kp.get('timestamp_ms', 0.0))/1000.0:.2f}s)</div>"
            f"<img src='data:image/png;base64,{kp.get('png_b64', '')}' style='max-width:100%;border-radius:6px;margin-top:6px' alt='关键帧骨架图'>"
            f"</div>"
        )
    kf_section_html = (
        f"<div class='card'><h2>五、关键动作峰值与纠错证据帧骨架图</h2><div class='grid'>{''.join(kf_cards)}</div></div>"
        if kf_cards
        else ""
    )

    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI 拳击/泰拳完整训练复盘评分报告 - {esc(clip_label)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px auto; max-width: 980px; padding: 0 16px; color: #1e293b; background: #f8fafc; }}
  h1, h2, h3 {{ color: #0f172a; }}
  .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 20px; margin-bottom: 18px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 12px 0; }}
  .metric {{ background: #f1f5f9; border-radius: 8px; padding: 12px; }}
  .metric .label {{ font-size: 12px; color: #64748b; }}
  .metric .value {{ font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 4px; }}
  .advice {{ background: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 6px; margin: 10px 0; }}
  .drill {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 6px; margin: 10px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
  .muted {{ color: #64748b; font-size: 13px; }}
</style>
</head>
<body>
  <h1>🥊 AI 拳击与泰拳动作复盘评分报告</h1>
  <p class="muted">片段标识：{esc(clip_label)} ｜ 站架：{esc(stance)} ｜ 机位：{esc(view)} ｜ 本地离线生成（无外部脚本与网络请求）</p>

  <div class="card">
    <h2>一、核心回防评分与最高优先建议（T08–T11）</h2>
    <div class="grid">
      <div class="metric"><div class="label">后手直拳回防评分</div><div class="value">{grade}</div></div>
      <div class="metric"><div class="label">后手直拳（已回落 / 保持满 100 ms）</div><div class="value">{rear_metrics.get('completed_count', rear_metrics['valid_count'])} / {rear_metrics['valid_count']} 次</div></div>
      <div class="metric"><div class="label">同场次中位回手时间</div><div class="value">{esc(med_label)}</div></div>
      <div class="metric"><div class="label">四分位距 (IQR)</div><div class="value">{esc(iqr_label)}</div></div>
      <div class="metric"><div class="label">泰拳高抱架保持率</div><div class="value">{esc(hg_label)}</div></div>
    </div>
    <div class="advice">
      <strong>💡 本段最高优先建议（Top 1）：</strong><br>{top_advice}
    </div>
    <div class="drill">
      <strong>🏋️ 针对性纠错训练处方（B01）—— {esc(str(drill.get('title', '基础回防定型练习')))}</strong><br>
      <span><strong>针对问题：</strong>{esc(str(drill.get('target_issue', '-')))}</span><br>
      <span><strong>推荐训练量：</strong>{esc(str(drill.get('dosage', '-')))}</span><br>
      <span><strong>执行要点：</strong>{esc(str(drill.get('steps', '-')))}</span><br>
      <span><strong>自查口令：</strong>{esc(str(drill.get('cue', '-')))}</span>
    </div>
  </div>

  <div class="card">
    <h2>二、全拳法统计：刺拳、平勾拳、上勾拳与 1-2 / 1-2-3 连击（B02）</h2>
    <div class="grid">
      <div class="metric"><div class="label">所选区间已识别的出拳</div><div class="value">{len(combo_res['straight_events']) + len(combo_res['bent_arm_punches']) if 'straight_events' in combo_res and 'bent_arm_punches' in combo_res else '—'} 次</div></div>
      <div class="metric"><div class="label">前手刺拳 (Jab)</div><div class="value">{combo_res['jab_attempt_count']} 次（完整 {combo_res['jab_valid_count']}）</div></div>
      <div class="metric"><div class="label">前手刺拳回防评分</div><div class="value">{esc(str(combo_res['jab_defense_grade']))}</div></div>
      <div class="metric"><div class="label">平勾拳 / 上勾拳</div><div class="value">{combo_res.get('hook_count', 0)} 次 / {combo_res.get('uppercut_count', 0)} 次</div></div>
      <div class="metric"><div class="label">1-2 / 1-2-3 连击组数</div><div class="value">{combo_res['combo_count']} 组 / {combo_res.get('one_two_three_count', 0)} 组</div></div>
    </div>
    <p class="muted">{esc(str(combo_res.get('combo_feedback', '')))}</p>
    <table>
      <thead><tr><th>连击编号</th><th>组合类型</th><th>关键峰值源帧</th><th>衔接间隔</th><th>完整回防</th></tr></thead>
      <tbody>{combos_table_body}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>三、泰拳专项：高抱架、扫踢、正蹬、冲膝、提膝格挡与拳腿立体连击（B03）</h2>
    <div class="grid">
      <div class="metric"><div class="label">步伐 / 换架</div><div class="value">{mt_res.get('step_count', '—')} / {mt_res.get('stance_switch_count', '—')}</div></div>
      <div class="metric"><div class="label">正蹬 / Stab (Teep)</div><div class="value">{str(mt_res['teep_count']) + ' 次' if 'teep_count' in mt_res else '旧报告未统计'}</div></div>
      <div class="metric"><div class="label">泰式扫踢 (Roundhouse)</div><div class="value">{mt_res['kick_count']} 次</div></div>
      <div class="metric"><div class="label">泰拳冲膝 (Knee Strike)</div><div class="value">{mt_res['knee_count']} 次</div></div>
      <div class="metric"><div class="label">提膝格挡 (Shin Check)</div><div class="value">{mt_res['shin_check_count']} 次</div></div>
      <div class="metric"><div class="label">拳腿/拳膝立体连击</div><div class="value">{mt_res['combo_count']} 组</div></div>
    </div>
    <p class="muted">{esc(str(mt_res.get('muaythai_feedback', '')))}</p>
    <table>
      <thead><tr><th>连击编号</th><th>立体组合</th><th>拳帧 → 腿膝帧</th><th>拳腿转换耗时</th><th>起腿护头状态</th></tr></thead>
      <tbody>{mt_table_body}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>四、后手直拳逐拳分段明细表（T08–T10）</h2>
    <table>
      <thead><tr><th>动作编号</th><th>状态</th><th>起手帧</th><th>最大伸展帧</th><th>回手进入帧</th><th>回手时间</th><th>二维髋偏移比</th><th>原因说明</th></tr></thead>
      <tbody>{actions_table_body}</tbody>
    </table>
    <p class="muted">{esc(str(rear_metrics.get('limitations_notice', '')))}</p>
  </div>
  {kf_section_html}
</body>
</html>"""
    return doc.encode("utf-8")