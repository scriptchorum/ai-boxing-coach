"""T13: Female-first cohort evaluation, G2 protocol checks, and replayable synthetic regression benchmark.

Implements DATA_EVALUATION.md rules:
1. Frozen version manifest for reproducible replay after model/rule updates.
2. One-to-one temporal IoU >= 0.5 matching between predicted and annotated ActionSegments.
   Even when a clip is rejected by the system, human-annotatable ground-truth actions count as FN.
3. Conditional return_time_ms MAE and P90 on matched valid actions alongside coverage and rejection rates.
4. Per-participant breakdown and female internal subgroup reporting (attire_group, torso_scale_group, stance).
   Subgroups with < 5 independent participants are flagged as exploratory_only; inter-group gaps > 10 percentage
   points trigger investigate_gap_over_10pp.
5. Never fabricates G2 pass scores when evaluated on synthetic data or when real holdout size is below G2 minimums.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from boxing_coach.quality_gate import VERSION as QUALITY_GATE_VERSION
from boxing_coach.rear_straight_review import (
    RULE_VERSION as REAR_STRAIGHT_RULE_VERSION,
    _quantile,
    check_participant_split_leakage,
)

FROZEN_EVALUATION_VERSION = {
    "evaluation_protocol_version": "0.1.0",
    "contract_schema_version": "0.1.0",
    "quality_gate_version": QUALITY_GATE_VERSION,
    "rear_straight_rule_version": REAR_STRAIGHT_RULE_VERSION,
}


def compute_temporal_iou(
    seg_a: tuple[float, float],
    seg_b: tuple[float, float],
) -> float:
    """Compute 1D temporal Intersection-over-Union for [start_ms, end_ms] intervals."""
    a0, a1 = float(seg_a[0]), float(seg_a[1])
    b0, b1 = float(seg_b[0]), float(seg_b[1])
    if a1 <= a0 or b1 <= b0:
        return 0.0
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    if inter <= 0.0:
        return 0.0
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0.0 else 0.0


def evaluate_clip_actions(
    predicted_actions: Sequence[dict[str, Any]],
    ground_truth_actions: Sequence[dict[str, Any]],
    *,
    clip_rejected: bool = False,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Match predicted vs ground-truth actions with 1-to-1 temporal IoU >= 0.5.

    If clip_rejected is True, effective predictions are empty, and all annotatable
    ground_truth_actions are counted as FN (per DATA_EVALUATION.md).
    """
    active_preds = (
        []
        if clip_rejected
        else [
            p
            for p in predicted_actions
            if p.get("start_ms") is not None and p.get("return_ms") is not None and p["return_ms"] > p["start_ms"]
        ]
    )
    valid_gts = [
        g
        for g in ground_truth_actions
        if g.get("start_ms") is not None and g.get("return_ms") is not None and g["return_ms"] > g["start_ms"]
    ]

    candidates: list[tuple[float, int, int]] = []
    for p_idx, pred in enumerate(active_preds):
        p_iv = (float(pred["start_ms"]), float(pred["return_ms"]))
        for g_idx, gt in enumerate(valid_gts):
            g_iv = (float(gt["start_ms"]), float(gt["return_ms"]))
            iou = compute_temporal_iou(p_iv, g_iv)
            if iou >= iou_threshold - 1e-9:
                candidates.append((iou, p_idx, g_idx))

    # Greedy 1-to-1 matching by descending IoU
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    return_time_errors_ms: list[float] = []

    for iou, p_idx, g_idx in candidates:
        if p_idx in matched_preds or g_idx in matched_gts:
            continue
        matched_preds.add(p_idx)
        matched_gts.add(g_idx)
        pred = active_preds[p_idx]
        gt = valid_gts[g_idx]
        if (
            pred.get("status") == "valid"
            and pred.get("extension_ms") is not None
            and pred.get("return_ms") is not None
            and gt.get("extension_ms") is not None
            and gt.get("return_ms") is not None
        ):
            pred_ret = float(pred["return_ms"]) - float(pred["extension_ms"])
            gt_ret = float(gt["return_ms"]) - float(gt["extension_ms"])
            return_time_errors_ms.append(abs(pred_ret - gt_ret))

    tp = len(matched_preds)
    fp = len(active_preds) - tp
    fn = len(valid_gts) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(valid_gts) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    count_abs_error = abs(len(active_preds) - len(valid_gts))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_count": len(active_preds),
        "ground_truth_count": len(valid_gts),
        "count_abs_error": count_abs_error,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "return_time_errors_ms": return_time_errors_ms,
    }


def _aggregate_group(clip_evals: Sequence[dict[str, Any]], participant_ids: set[str]) -> dict[str, Any]:
    tp = sum(c["tp"] for c in clip_evals)
    fp = sum(c["fp"] for c in clip_evals)
    fn = sum(c["fn"] for c in clip_evals)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    all_errors = [err for c in clip_evals for err in c["return_time_errors_ms"]]
    normative = [c for c in clip_evals if c["clip_kind"] == "normative"]
    usable = [c for c in normative if c["fully_analyzable"]]
    coverage = (len(usable) / len(normative)) if normative else None

    return {
        "participant_count": len(participant_ids),
        "clip_count": len(clip_evals),
        "ground_truth_action_count": sum(c["ground_truth_count"] for c in clip_evals),
        "exploratory_only": len(participant_ids) < 5,
        "f1": round(f1, 4),
        "coverage_rate": round(coverage, 4) if coverage is not None else None,
        "return_time_mae_ms": round(sum(all_errors) / len(all_errors), 3) if all_errors else None,
        "return_time_p90_ms": round(_quantile(all_errors, 0.90), 3) if all_errors else None,
    }


def evaluate_cohort_runs(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """T13: Evaluate a cohort of clip records following DATA_EVALUATION.md and G2 gates."""
    manifests = [r["manifest"] for r in records if "manifest" in r]
    leakage_errors = check_participant_split_leakage(manifests)

    any_synthetic = any(bool(r.get("synthetic", True)) for r in records)
    participants: dict[str, dict[str, Any]] = {}
    clip_summaries: list[dict[str, Any]] = []

    for rec in records:
        m = rec["manifest"]
        p_id = str(m["participant_id"])
        gender = str(rec.get("participant_gender", "female")).lower()
        attire = str(rec.get("attire_group", "unspecified"))
        scale_group = str(rec.get("torso_scale_group", "unspecified"))
        stance = str(m.get("capture", {}).get("stance", "orthodox")).lower()
        clip_kind = str(rec.get("clip_kind", "normative"))  # "normative" or "bad"
        clip_rejected = bool(rec.get("clip_rejected", False))
        preds = rec.get("predicted_actions", [])
        gts = rec.get("ground_truth_actions", [])

        ev = evaluate_clip_actions(preds, gts, clip_rejected=clip_rejected)
        valid_pred_count = 0 if clip_rejected else sum(1 for a in preds if a.get("status") == "valid")
        fully_analyzable = (not clip_rejected) and (valid_pred_count >= 5)

        entry = {
            "sample_id": m["sample_id"],
            "participant_id": p_id,
            "gender": gender,
            "attire_group": attire,
            "torso_scale_group": scale_group,
            "stance": stance,
            "clip_kind": clip_kind,
            "clip_rejected": clip_rejected,
            "fully_analyzable": fully_analyzable,
            **ev,
        }
        clip_summaries.append(entry)
        participants.setdefault(p_id, {"gender": gender, "clips": []})["clips"].append(entry)

    total_participants = len(participants)
    female_participants = sum(1 for p in participants.values() if p["gender"] == "female")
    female_ratio = (female_participants / total_participants) if total_participants > 0 else None

    normative_clips = [c for c in clip_summaries if c["clip_kind"] == "normative"]
    bad_clips = [c for c in clip_summaries if c["clip_kind"] == "bad"]

    tp = sum(c["tp"] for c in normative_clips)
    fp = sum(c["fp"] for c in normative_clips)
    fn = sum(c["fn"] for c in normative_clips)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    overall_f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    count_errors = [c["count_abs_error"] for c in normative_clips]
    mean_count_error = (sum(count_errors) / len(count_errors)) if count_errors else None
    max_count_error = max(count_errors) if count_errors else None

    all_ret_errors = [err for c in normative_clips for err in c["return_time_errors_ms"]]
    return_mae_ms = round(sum(all_ret_errors) / len(all_ret_errors), 3) if all_ret_errors else None
    return_p90_ms = round(_quantile(all_ret_errors, 0.90), 3) if all_ret_errors else None

    usable_normative = sum(1 for c in normative_clips if c["fully_analyzable"])
    coverage_rate = (usable_normative / len(normative_clips)) if normative_clips else None
    false_rejection_rate = (
        sum(1 for c in normative_clips if c["clip_rejected"]) / len(normative_clips)
        if normative_clips
        else None
    )
    bad_rejection_recall = (
        sum(1 for c in bad_clips if c["clip_rejected"]) / len(bad_clips)
        if bad_clips
        else None
    )

    per_participant = {
        p_id: _aggregate_group(info["clips"], {p_id})
        for p_id, info in sorted(participants.items())
    }

    # Female subgroup breakdown (attire_group & torso_scale_group)
    female_clips = [c for c in normative_clips if c["gender"] == "female"]
    subgroups: dict[str, dict[str, Any]] = {}
    for axis in ("attire_group", "torso_scale_group", "stance"):
        buckets: dict[str, list[dict[str, Any]]] = {}
        for c in female_clips:
            buckets.setdefault(str(c[axis]), []).append(c)
        axis_report: dict[str, Any] = {}
        f1_vals: list[float] = []
        for label, items in sorted(buckets.items()):
            pids = {c["participant_id"] for c in items}
            grp = _aggregate_group(items, pids)
            axis_report[label] = grp
            f1_vals.append(grp["f1"])
        gap = (max(f1_vals) - min(f1_vals)) if len(f1_vals) >= 2 else 0.0
        subgroups[axis] = {
            "groups": axis_report,
            "max_f1_gap": round(gap, 4),
            "investigate_gap_over_10pp": gap > 0.10 + 1e-9,
        }

    total_gt_actions = sum(c["ground_truth_count"] for c in normative_clips)
    g2_sample_met = (
        (not any_synthetic)
        and (not leakage_errors)
        and female_participants >= 5
        and len(normative_clips) >= 20
        and total_gt_actions >= 100
        and len(bad_clips) >= 20
    )

    return {
        "frozen_version": dict(FROZEN_EVALUATION_VERSION),
        "synthetic_only": any_synthetic,
        "split_leakage_errors": leakage_errors,
        "denominators": {
            "total_participants": total_participants,
            "female_participants": female_participants,
            "female_ratio": round(female_ratio, 4) if female_ratio is not None else None,
            "normative_clips": len(normative_clips),
            "bad_clips": len(bad_clips),
            "total_ground_truth_actions": total_gt_actions,
        },
        "overall_metrics": {
            "segmentation_precision": round(precision, 4),
            "segmentation_recall": round(recall, 4),
            "segmentation_f1": round(overall_f1, 4),
            "count_abs_error_mean": round(mean_count_error, 4) if mean_count_error is not None else None,
            "count_abs_error_max": max_count_error,
            "return_time_mae_ms": return_mae_ms,
            "return_time_p90_ms": return_p90_ms,
            "valid_return_time_comparisons": len(all_ret_errors),
            "coverage_rate": round(coverage_rate, 4) if coverage_rate is not None else None,
            "normative_false_rejection_rate": (
                round(false_rejection_rate, 4) if false_rejection_rate is not None else None
            ),
            "bad_clip_rejection_recall": (
                round(bad_rejection_recall, 4) if bad_rejection_recall is not None else None
            ),
        },
        "per_participant": per_participant,
        "female_subgroups": subgroups,
        "g2_gate_check": {
            "g2_sample_minimums_met": g2_sample_met,
            "status": "ready_for_g2_claim" if g2_sample_met else "pending_real_holdout_sample",
            "note": (
                "当前为合成回归或样本量未达 G2 独立留出集下限（需 ≥5 名女性、≥20 条规范片、≥100 次动作、≥20 条坏片）；"
                "所有指标仅验证评估管线与回归重放，不宣称通过真实人群公平性或 G2 准确率门禁。"
                if not g2_sample_met
                else "已满足 G2 样本规模与分区隔离要求。"
            ),
        },
    }
