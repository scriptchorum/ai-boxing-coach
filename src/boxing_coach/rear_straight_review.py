"""T08-T11 rear straight segmentation, descriptive metrics, optional event override, and local history comparison.

Implements:
- T08: ReviewAnnotation schema validation, participant/session/lineage split leakage check, and optional manual event override on top of automatic pre-labeling.
- T09: Pixel-restored rear straight state machine (start -> extension -> return -> hold_end) with personal starting guard region and feint/incomplete-return handling.
- T10: Descriptive metrics using decoded PTS timestamps and pixel-restored coordinates (return_time_ms, return_time_repeatability median/IQR for >=5 valid actions, and 2D torso projection shift ratio).
- T11: Observation report assembly and same-condition local history comparison (refusing to plot cross-condition records as an improvement trend).
"""
from __future__ import annotations

import json
import math
from typing import Any, Sequence

from jsonschema import Draft202012Validator

from boxing_coach.model_assets import find_project_root

RULE_VERSION = "rear-straight-0.1.0"
APP_VERSION = "local-review-0.1.0"
EVENTS = ("start", "extension", "return", "hold_end")

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

_ANNOTATION_SCHEMA = json.loads(
    (find_project_root() / "schemas" / "review-annotation.schema.json").read_text(encoding="utf-8")
)
Draft202012Validator.check_schema(_ANNOTATION_SCHEMA)
REVIEW_ANNOTATION_VALIDATOR = Draft202012Validator(_ANNOTATION_SCHEMA)


def _quantile(values: Sequence[float], p: float) -> float:
    """Linear interpolation at (n - 1) * p matching tools/validate_contracts.py."""
    ordered = sorted(float(v) for v in values)
    index = (len(ordered) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def _median(values: Sequence[float]) -> float:
    return _quantile(values, 0.5)


def _point_map(frame: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(p["index"]): p for p in frame.get("landmarks", []) if "index" in p}


def _observed_pixel_xy(
    points: dict[int, dict[str, Any]],
    index: int,
    width: int,
    height: int,
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
) -> tuple[float, float] | None:
    """Return pixel-restored (x_px, y_px) only if directly observed, in-frame, and above thresholds."""
    point = points.get(index)
    if not point or point.get("state") != "observed":
        return None
    img = point.get("image")
    if not img:
        return None
    x, y = img.get("x"), img.get("y")
    vis, pres = point.get("visibility"), point.get("presence")
    if x is None or y is None or vis is None or pres is None:
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(vis) and math.isfinite(pres)):
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    if vis < min_visibility or pres < min_presence:
        return None
    return (float(x) * float(width), float(y) * float(height))


def _torso_scale_and_hip_px(
    points: dict[int, dict[str, Any]],
    width: int,
    height: int,
    *,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
) -> tuple[float | None, tuple[float, float] | None]:
    """Compute projected torso pixel scale and hip midpoint in pixels."""
    l_sh = _observed_pixel_xy(points, LEFT_SHOULDER, width, height, min_visibility=min_visibility, min_presence=min_presence)
    r_sh = _observed_pixel_xy(points, RIGHT_SHOULDER, width, height, min_visibility=min_visibility, min_presence=min_presence)
    l_hip = _observed_pixel_xy(points, LEFT_HIP, width, height, min_visibility=min_visibility, min_presence=min_presence)
    r_hip = _observed_pixel_xy(points, RIGHT_HIP, width, height, min_visibility=min_visibility, min_presence=min_presence)

    hip_mid = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0) if (l_hip and r_hip) else None
    measures: list[float] = []
    if l_sh and r_sh and l_hip and r_hip:
        sh_mid = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
        trunk_len = math.hypot(sh_mid[0] - hip_mid[0], sh_mid[1] - hip_mid[1])
        if trunk_len > 1e-3:
            measures.append(trunk_len)
    if l_sh and r_sh:
        sh_w = math.hypot(l_sh[0] - r_sh[0], l_sh[1] - r_sh[1])
        if sh_w > 1e-3:
            measures.append(sh_w)
    if l_hip and r_hip:
        hip_w = math.hypot(l_hip[0] - r_hip[0], l_hip[1] - r_hip[1])
        if hip_w > 1e-3:
            measures.append(hip_w)

    if not measures:
        return None, hip_mid
    return max(measures), hip_mid


def _critical_indices_for_hand(hand: str) -> tuple[int, int, int]:
    if hand == "right":
        return (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    return (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)


def _hand_for_stance(stance: str) -> str:
    s = stance.lower()
    if s == "orthodox":
        return "right"
    if s == "southpaw":
        return "left"
    return "unknown"


def check_participant_split_leakage(manifests: Sequence[dict[str, Any]]) -> list[str]:
    """T08: Detect split leakage across participant_id, lineage_id, session_id, and input_sha256."""
    splits: dict[tuple[str, str], set[str]] = {}
    for m in manifests:
        split = m.get("split", "unassigned")
        if split not in ("development", "holdout"):
            continue
        for field in ("participant_id", "lineage_id", "session_id", "input_sha256"):
            val = m.get(field)
            if isinstance(val, str) and val:
                splits.setdefault((field, val), set()).add(split)

    errors: list[str] = []
    for (field, _), assigned in splits.items():
        if "development" in assigned and "holdout" in assigned:
            errors.append(f"split_leakage:{field}")
    return sorted(set(errors))


def segment_rear_straights(
    frames: Sequence[dict[str, Any]],
    *,
    width: int = 1280,
    height: int = 720,
    stance: str = "orthodox",
    sample_id: str = "local-sample",
    reference_start_ms: float = 0.0,
    reference_end_ms: float = 400.0,
    return_radius_ratio: float = 0.30,
    return_hold_ms: float = 100.0,
    min_extension_ratio: float = 0.68,
    feint_extension_ratio: float = 0.50,
    min_reach_scale_ratio: float = 0.85,
    feint_reach_scale_ratio: float = 0.45,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
) -> dict[str, Any]:
    """T09: Segment rear straight punches using pixel-restored coordinates and decoded PTS."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers.")
    hand = _hand_for_stance(stance)
    sh_idx, el_idx, wr_idx = _critical_indices_for_hand(hand if hand != "unknown" else "right")

    # 1. Estimate personal reference scale and starting rear-hand offset in pixels.
    ref_scales: list[float] = []
    ref_offsets: list[tuple[float, float]] = []
    ref_hips: list[tuple[float, float]] = []

    frame_features: list[dict[str, Any]] = []
    for pos, frame in enumerate(frames):
        src_idx = int(frame["source_frame_index"])
        ts_ms = float(frame["timestamp_ms"])
        pts = _point_map(frame)
        scale_px, hip_mid = _torso_scale_and_hip_px(
            pts, width, height, min_visibility=min_visibility, min_presence=min_presence
        )
        sh_xy = _observed_pixel_xy(pts, sh_idx, width, height, min_visibility=min_visibility, min_presence=min_presence)
        el_xy = _observed_pixel_xy(pts, el_idx, width, height, min_visibility=min_visibility, min_presence=min_presence)
        wr_xy = _observed_pixel_xy(pts, wr_idx, width, height, min_visibility=min_visibility, min_presence=min_presence)

        critical_ok = sh_xy is not None and el_xy is not None and wr_xy is not None
        offset_xy: tuple[float, float] | None = None
        reach_px: float | None = None
        ext_ratio: float | None = None
        elbow_angle_deg: float | None = None

        if critical_ok and sh_xy and el_xy and wr_xy:
            offset_xy = (wr_xy[0] - sh_xy[0], wr_xy[1] - sh_xy[1])
            reach_px = math.hypot(offset_xy[0], offset_xy[1])
            upper_px = math.hypot(el_xy[0] - sh_xy[0], el_xy[1] - sh_xy[1])
            fore_px = math.hypot(wr_xy[0] - el_xy[0], wr_xy[1] - el_xy[1])
            arm_len = upper_px + fore_px
            if arm_len > 1e-3:
                ext_ratio = min(1.0, reach_px / arm_len)
            if upper_px > 1e-3 and fore_px > 1e-3:
                v1 = (sh_xy[0] - el_xy[0], sh_xy[1] - el_xy[1])
                v2 = (wr_xy[0] - el_xy[0], wr_xy[1] - el_xy[1])
                cos_val = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (upper_px * fore_px)))
                elbow_angle_deg = math.degrees(math.acos(cos_val))

        if reference_start_ms <= ts_ms <= reference_end_ms:
            if scale_px is not None:
                ref_scales.append(scale_px)
            if offset_xy is not None:
                ref_offsets.append(offset_xy)
            if hip_mid is not None:
                ref_hips.append(hip_mid)

        frame_features.append(
            {
                "pos": pos,
                "source_frame_index": src_idx,
                "timestamp_ms": ts_ms,
                "scale_px": scale_px,
                "hip_mid_px": hip_mid,
                "offset_xy": offset_xy,
                "reach_px": reach_px,
                "ext_ratio": ext_ratio,
                "elbow_angle_deg": elbow_angle_deg,
                "critical_ok": critical_ok,
            }
        )

    # Fallback if no frames fell inside [reference_start_ms, reference_end_ms]
    if not ref_scales:
        ref_scales = [f["scale_px"] for f in frame_features[:5] if f["scale_px"] is not None]
    if not ref_offsets:
        ref_offsets = [f["offset_xy"] for f in frame_features[:5] if f["offset_xy"] is not None]
    if not ref_hips:
        ref_hips = [f["hip_mid_px"] for f in frame_features[:5] if f["hip_mid_px"] is not None]

    baseline_scale_px = round(_median(ref_scales), 3) if ref_scales else None
    baseline_offset_px = (
        (_median([o[0] for o in ref_offsets]), _median([o[1] for o in ref_offsets]))
        if ref_offsets
        else None
    )
    baseline_hip_px = (
        (_median([h[0] for h in ref_hips]), _median([h[1] for h in ref_hips]))
        if ref_hips
        else None
    )

    guard_radius_px = (
        baseline_scale_px * return_radius_ratio
        if baseline_scale_px is not None
        else None
    )

    for feat in frame_features:
        if baseline_offset_px is not None and feat["offset_xy"] is not None and guard_radius_px is not None:
            dist_from_guard = math.hypot(
                feat["offset_xy"][0] - baseline_offset_px[0],
                feat["offset_xy"][1] - baseline_offset_px[1],
            )
            feat["guard_dist_px"] = dist_from_guard
            feat["in_guard"] = dist_from_guard <= guard_radius_px
        else:
            feat["guard_dist_px"] = None
            feat["in_guard"] = False

    # 2. State machine over frames after reference_end_ms
    actions: list[dict[str, Any]] = []
    n = len(frame_features)
    idx = 0
    action_seq = 0
    last_valid_hold_end_ms = -1.0

    while idx < n:
        feat = frame_features[idx]
        if feat["timestamp_ms"] < reference_end_ms or feat["timestamp_ms"] < last_valid_hold_end_ms:
            idx += 1
            continue

        # Look for excursion leaving the guard region
        if not feat["critical_ok"] or feat["in_guard"] or feat["guard_dist_px"] is None:
            idx += 1
            continue

        # Start of excursion is the immediately preceding in-guard frame (if >= reference_end_ms and >= last_valid_hold_end_ms),
        # otherwise the current frame where excursion is first observed.
        start_pos = idx
        if idx > 0:
            prev_feat = frame_features[idx - 1]
            if (
                prev_feat["critical_ok"]
                and prev_feat["timestamp_ms"] >= reference_end_ms
                and prev_feat["timestamp_ms"] >= last_valid_hold_end_ms
            ):
                start_pos = idx - 1

        # Scan forward until the wrist returns into the guard region (or clip ends)
        excursion_end = idx
        while excursion_end < n and not frame_features[excursion_end]["in_guard"]:
            excursion_end += 1

        # Find maximum visible extension frame within [idx, excursion_end - 1]
        excursion_slice = [
            frame_features[k]
            for k in range(idx, excursion_end)
            if frame_features[k]["critical_ok"] and frame_features[k]["reach_px"] is not None
        ]
        if not excursion_slice:
            idx = excursion_end + 1
            continue

        peak_feat = max(
            excursion_slice,
            key=lambda x: (x["reach_px"] or 0.0, x["ext_ratio"] or 0.0),
        )
        peak_pos = peak_feat["pos"]
        peak_ext_ratio = peak_feat["ext_ratio"] or 0.0
        peak_reach_scale_ratio = (
            (peak_feat["reach_px"] / baseline_scale_px)
            if (peak_feat["reach_px"] is not None and baseline_scale_px is not None and baseline_scale_px > 1e-3)
            else 0.0
        )

        # Ensure start_pos is strictly before peak_pos if possible
        if start_pos >= peak_pos and peak_pos > 0:
            cand_start = peak_pos - 1
            if (
                frame_features[cand_start]["timestamp_ms"] >= reference_end_ms
                and frame_features[cand_start]["timestamp_ms"] >= last_valid_hold_end_ms
            ):
                start_pos = cand_start

        # Ignore tiny jitter that is not even a feint
        if peak_ext_ratio < feint_extension_ratio or peak_reach_scale_ratio < feint_reach_scale_ratio:
            idx = max(excursion_end, idx + 1)
            continue

        action_seq += 1
        action_id = f"{sample_id}-action-{action_seq}"
        start_feat = frame_features[start_pos]

        reasons: list[str] = []
        if hand == "unknown":
            reasons.append("unknown_stance")
        if baseline_scale_px is None:
            reasons.append("reference_scale_missing")
        if peak_pos <= start_pos:
            reasons.append("missing_start_before_extension")
        if peak_ext_ratio < min_extension_ratio or peak_reach_scale_ratio < min_reach_scale_ratio:
            reasons.append("feint_or_incomplete_extension")

        # Locate return_pos and hold_end_pos after peak_pos
        return_feat: dict[str, Any] | None = None
        hold_end_feat: dict[str, Any] | None = None

        if excursion_end < n and frame_features[excursion_end]["in_guard"]:
            return_feat = frame_features[excursion_end]
            # Verify continuous hold in guard region for >= return_hold_ms
            hold_cursor = excursion_end
            while hold_cursor < n and frame_features[hold_cursor]["in_guard"]:
                if frame_features[hold_cursor]["timestamp_ms"] - return_feat["timestamp_ms"] >= return_hold_ms - 1e-9:
                    hold_end_feat = frame_features[hold_cursor]
                    break
                hold_cursor += 1
            if hold_end_feat is None:
                reasons.append("insufficient_return_hold")
                next_idx = hold_cursor + 1
            else:
                next_idx = hold_end_feat["pos"] + 1
        else:
            reasons.append("missing_return")
            next_idx = excursion_end + 1

        for ev_feat in (start_feat, peak_feat, return_feat, hold_end_feat):
            if ev_feat is not None and not ev_feat["critical_ok"]:
                if "critical_point_unusable" not in reasons:
                    reasons.append("critical_point_unusable")

        if not reasons and return_feat is not None and hold_end_feat is not None:
            status = "valid"
            last_valid_hold_end_ms = hold_end_feat["timestamp_ms"]
        elif "unknown_stance" in reasons or "critical_point_unusable" in reasons or "feint_or_incomplete_extension" in reasons:
            status = "rejected"
        else:
            status = "partial"

        actions.append(
            {
                "schema_version": "0.1.0",
                "sample_id": sample_id,
                "action_id": action_id,
                "action_kind": "rear_straight",
                "hand": hand,
                "source": "automatic",
                "status": status,
                "quality_reasons": reasons,
                "start_frame_index": start_feat["source_frame_index"],
                "start_ms": start_feat["timestamp_ms"],
                "extension_frame_index": peak_feat["source_frame_index"],
                "extension_ms": peak_feat["timestamp_ms"],
                "return_frame_index": return_feat["source_frame_index"] if return_feat else None,
                "return_ms": return_feat["timestamp_ms"] if return_feat else None,
                "hold_end_frame_index": hold_end_feat["source_frame_index"] if hold_end_feat else None,
                "hold_end_ms": hold_end_feat["timestamp_ms"] if hold_end_feat else None,
            }
        )
        idx = max(next_idx, idx + 1)

    return {
        "sample_id": sample_id,
        "stance": stance.lower() if stance.lower() in ("orthodox", "southpaw") else "unknown",
        "hand": hand,
        "width": width,
        "height": height,
        "metric_config": {
            "version": RULE_VERSION,
            "reference_start_ms": reference_start_ms,
            "reference_end_ms": reference_end_ms,
            "scale_kind": "projected_torso",
            "scale_px": baseline_scale_px,
            "return_radius_ratio": return_radius_ratio,
            "return_hold_ms": return_hold_ms,
        },
        "baseline_hip_px": baseline_hip_px,
        "frame_features": frame_features,
        "actions": actions,
    }


def apply_action_override(
    segmentation: dict[str, Any],
    frames: Sequence[dict[str, Any]],
    action_id: str,
    *,
    start_frame_index: int | None = None,
    extension_frame_index: int | None = None,
    return_frame_index: int | None = None,
    hold_end_frame_index: int | None = None,
    reject_reason: str | None = None,
    min_visibility: float = 0.5,
    min_presence: float = 0.5,
) -> dict[str, Any]:
    """T08: Apply optional manual frame override to an auto-segmented action while preserving PTS mapping."""
    frame_lookup = {int(f["source_frame_index"]): f for f in frames}
    feat_lookup = {int(f["source_frame_index"]): f for f in segmentation.get("frame_features", [])}
    return_hold_ms = float(segmentation["metric_config"]["return_hold_ms"])
    hand = segmentation["hand"]
    sh_idx, el_idx, wr_idx = _critical_indices_for_hand(hand if hand != "unknown" else "right")
    width, height = int(segmentation["width"]), int(segmentation["height"])

    updated_actions: list[dict[str, Any]] = []
    found = False
    for action in segmentation["actions"]:
        if action["action_id"] != action_id:
            updated_actions.append(dict(action))
            continue
        found = True
        patched = dict(action)
        patched["source"] = "manual"

        overrides = {
            "start": start_frame_index if start_frame_index is not None else patched["start_frame_index"],
            "extension": extension_frame_index if extension_frame_index is not None else patched["extension_frame_index"],
            "return": return_frame_index if return_frame_index is not None else patched["return_frame_index"],
            "hold_end": hold_end_frame_index if hold_end_frame_index is not None else patched["hold_end_frame_index"],
        }
        reasons: list[str] = []
        if reject_reason:
            reasons.append(reject_reason)

        times: list[float] = []
        for ev in EVENTS:
            f_idx = overrides[ev]
            if f_idx is None:
                patched[f"{ev}_frame_index"] = None
                patched[f"{ev}_ms"] = None
                continue
            if f_idx not in frame_lookup:
                raise ValueError(f"Frame index {f_idx} does not exist in PoseRun.")
            f_obj = frame_lookup[f_idx]
            ts_ms = float(f_obj["timestamp_ms"])
            patched[f"{ev}_frame_index"] = int(f_idx)
            patched[f"{ev}_ms"] = ts_ms
            times.append(ts_ms)

            # Check critical landmarks on overridden event frame
            feat = feat_lookup.get(int(f_idx))
            if feat is not None:
                crit_ok = bool(feat["critical_ok"])
            else:
                pts = _point_map(f_obj)
                crit_ok = all(
                    _observed_pixel_xy(
                        pts, c_idx, width, height, min_visibility=min_visibility, min_presence=min_presence
                    )
                    is not None
                    for c_idx in (sh_idx, el_idx, wr_idx)
                )
            if not crit_ok and "critical_point_unusable" not in reasons:
                reasons.append("critical_point_unusable")

        if any(a >= b for a, b in zip(times, times[1:])):
            reasons.append("event_order")
        if patched["return_ms"] is None:
            reasons.append("missing_return")
        elif patched["hold_end_ms"] is None or (patched["hold_end_ms"] - patched["return_ms"] < return_hold_ms - 1e-9):
            reasons.append("insufficient_return_hold")

        if not reasons and all(patched[f"{ev}_ms"] is not None for ev in EVENTS):
            patched["status"] = "valid"
        elif reject_reason or "critical_point_unusable" in reasons or "event_order" in reasons:
            patched["status"] = "rejected"
        else:
            patched["status"] = "partial"
        patched["quality_reasons"] = reasons
        updated_actions.append(patched)

    if not found:
        raise KeyError(f"Action {action_id} not found.")

    result = dict(segmentation)
    result["actions"] = updated_actions
    return result


def build_review_annotation(
    segmentation: dict[str, Any],
    *,
    annotation_id: str = "local-annotation-1",
    annotator_code: str = "local-user",
    review_status: str = "auto_prelabeled",
    ambiguity_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """T08: Build and schema-validate a ReviewAnnotation payload."""
    actions = segmentation.get("actions", [])
    valid_count = sum(1 for a in actions if a["status"] == "valid")
    if any(a["source"] == "manual" for a in actions) and review_status == "auto_prelabeled":
        review_status = "adjusted"
    if valid_count > 0:
        analyzable_status = "analyzable" if valid_count == len(actions) else "partial"
    else:
        analyzable_status = "rejected" if actions else "partial"

    payload = {
        "schema_version": "0.1.0",
        "annotation_id": annotation_id,
        "sample_id": segmentation["sample_id"],
        "annotator_code": annotator_code,
        "stance": segmentation["stance"],
        "analyzable_status": analyzable_status,
        "review_status": review_status,
        "ambiguity_reasons": list(dict.fromkeys(ambiguity_reasons)),
        "actions": [
            {
                "action_id": a["action_id"],
                "action_kind": "rear_straight",
                "hand": a["hand"],
                "source": a["source"],
                "status": a["status"],
                "quality_reasons": list(a["quality_reasons"]),
                "start_frame_index": a["start_frame_index"],
                "start_ms": a["start_ms"],
                "extension_frame_index": a["extension_frame_index"],
                "extension_ms": a["extension_ms"],
                "return_frame_index": a["return_frame_index"],
                "return_ms": a["return_ms"],
                "hold_end_frame_index": a["hold_end_frame_index"],
                "hold_end_ms": a["hold_end_ms"],
            }
            for a in actions
        ],
    }
    REVIEW_ANNOTATION_VALIDATOR.validate(payload)
    return payload


def export_review_annotation_json(
    segmentation: dict[str, Any],
    *,
    annotation_id: str = "local-annotation-1",
    annotator_code: str = "local-user",
    review_status: str = "auto_prelabeled",
    ambiguity_reasons: Sequence[str] = (),
) -> bytes:
    """Serialize validated ReviewAnnotation to deterministic UTF-8 JSON bytes."""
    payload = build_review_annotation(
        segmentation,
        annotation_id=annotation_id,
        annotator_code=annotator_code,
        review_status=review_status,
        ambiguity_reasons=ambiguity_reasons,
    )
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def compute_rear_straight_metrics(
    segmentation: dict[str, Any],
    *,
    single_person_confirmed: bool = True,
) -> dict[str, Any]:
    """T10: Compute return_time_ms, repeatability (median/IQR for >=5 valid actions), and 2D torso projection shift."""
    actions = segmentation.get("actions", [])
    feat_by_src = {int(f["source_frame_index"]): f for f in segmentation.get("frame_features", [])}
    scale_px = segmentation["metric_config"].get("scale_px")
    baseline_hip_px = segmentation.get("baseline_hip_px")

    action_metrics: list[dict[str, Any]] = []
    valid_return_times: list[float] = []
    valid_action_ids: list[str] = []
    repeatability_evidence: list[dict[str, Any]] = []

    for a in actions:
        ext_idx = a["extension_frame_index"]
        ext_feat = feat_by_src.get(ext_idx) if ext_idx is not None else None

        # Torso projection shift at extension relative to reference hip midpoint, normalized by personal torso scale
        torso_shift_ratio: float | None = None
        if (
            single_person_confirmed
            and scale_px is not None
            and scale_px > 1e-3
            and baseline_hip_px is not None
            and ext_feat is not None
            and ext_feat.get("hip_mid_px") is not None
        ):
            hip_now = ext_feat["hip_mid_px"]
            shift_px = math.hypot(hip_now[0] - baseline_hip_px[0], hip_now[1] - baseline_hip_px[1])
            torso_shift_ratio = round(shift_px / scale_px, 4)

        if (
            single_person_confirmed
            and a["status"] == "valid"
            and a["extension_ms"] is not None
            and a["return_ms"] is not None
            and a["hold_end_ms"] is not None
        ):
            ret_ms = round(float(a["return_ms"]) - float(a["extension_ms"]), 3)
            valid_return_times.append(ret_ms)
            valid_action_ids.append(a["action_id"])
            ev_items = [
                {
                    "action_id": a["action_id"],
                    "event": ev,
                    "source_frame_index": a[f"{ev}_frame_index"],
                    "timestamp_ms": a[f"{ev}_ms"],
                }
                for ev in ("extension", "return", "hold_end")
            ]
            repeatability_evidence.extend(ev_items)
        else:
            ret_ms = None
            ev_items = []

        reasons = list(a["quality_reasons"])
        if not single_person_confirmed and "single_person_unconfirmed" not in reasons:
            reasons.append("single_person_unconfirmed")

        action_metrics.append(
            {
                "action_id": a["action_id"],
                "source": a["source"],
                "status": a["status"] if single_person_confirmed else "rejected",
                "start_frame_index": a["start_frame_index"],
                "start_ms": a["start_ms"],
                "extension_frame_index": a["extension_frame_index"],
                "extension_ms": a["extension_ms"],
                "return_frame_index": a["return_frame_index"],
                "return_ms": a["return_ms"],
                "hold_end_frame_index": a["hold_end_frame_index"],
                "hold_end_ms": a["hold_end_ms"],
                "return_time_ms": ret_ms,
                "torso_projection_shift_ratio": torso_shift_ratio,
                "quality_reasons": reasons,
                "evidence": ev_items,
            }
        )

    candidate_count = len(actions)
    valid_count = len(valid_return_times) if single_person_confirmed else 0
    usable_ratio = (valid_count / candidate_count) if candidate_count > 0 else None

    if single_person_confirmed and valid_count >= 5:
        median_ms = round(_quantile(valid_return_times, 0.5), 3)
        iqr_ms = round(_quantile(valid_return_times, 0.75) - _quantile(valid_return_times, 0.25), 3)
        repeatability = {
            "metric_id": "return_time_repeatability",
            "action_ids": valid_action_ids,
            "unit": "ms",
            "median_ms": median_ms,
            "iqr_ms": iqr_ms,
            "evidence": repeatability_evidence,
            "limitations": [
                "view_dependent",
                "not_coaching_assessment",
            ],
        }
    else:
        repeatability = None

    report_reasons: list[str] = []
    if not single_person_confirmed:
        report_reasons.append("single_person_unconfirmed")
    if candidate_count == 0:
        report_reasons.append("no_candidate_actions")
    elif valid_count == 0:
        report_reasons.append("no_valid_actions")
    elif valid_count < 5:
        report_reasons.append("fewer_than_5_valid_actions_for_repeatability")

    if not single_person_confirmed or (candidate_count > 0 and valid_count == 0):
        report_status = "rejected"
    elif valid_count == 0:
        report_status = "no_observations"
    elif valid_count >= 5:
        report_status = "complete"
    else:
        report_status = "partial"

    coaching = _build_defense_score_and_top_advice(
        action_metrics,
        valid_return_times=valid_return_times,
        repeatability=repeatability,
        single_person_confirmed=bool(single_person_confirmed),
    )

    return {
        "sample_id": segmentation["sample_id"],
        "stance": segmentation["stance"],
        "hand": segmentation["hand"],
        "rule_version": RULE_VERSION,
        "app_version": APP_VERSION,
        "status": report_status,
        "single_person_confirmed": bool(single_person_confirmed),
        "candidate_count": candidate_count,
        "valid_count": valid_count,
        "usable_ratio": round(usable_ratio, 4) if usable_ratio is not None else None,
        "metric_config": dict(segmentation["metric_config"]),
        "action_metrics": action_metrics,
        "repeatability": repeatability,
        "defense_score": coaching["defense_score"],
        "defense_grade": coaching["defense_grade"],
        "top_priority_advice": coaching["top_priority_advice"],
        "top_priority_frame_index": coaching["top_priority_frame_index"],
        "quality_reasons": report_reasons,
        "limitations_notice": (
            "本报告仅描述画面内可见关键点的回手时间（毫秒）、同场次中位数/四分位距及二维髋部投影偏移比；"
            "不代表真实三维拳速、拳力、身体重心平衡或动作优劣评价，不包含任何体型或外貌评价。"
        ),
    }


def _build_defense_score_and_top_advice(
    action_metrics: Sequence[dict[str, Any]],
    *,
    valid_return_times: Sequence[float],
    repeatability: dict[str, Any] | None,
    single_person_confirmed: bool,
) -> dict[str, Any]:
    """Compute an intuitive 0-100 return-defense score and single highest-priority advice."""
    if not single_person_confirmed:
        return {
            "defense_score": None,
            "defense_grade": "待开启单人确认",
            "top_priority_advice": "请勾选“确认当前片段为单人训练”，即可查看本段回防评分与最高优先建议。",
            "top_priority_frame_index": None,
        }

    real_attempts = [
        m for m in action_metrics if "feint_or_incomplete_extension" not in m.get("quality_reasons", [])
    ]
    if not real_attempts:
        return {
            "defense_score": None,
            "defense_grade": "暂无有效出拳",
            "top_priority_advice": (
                "当前选段内未检出完整伸展的后手直拳（或关键点被遮挡）。"
                "建议选取侧面/前斜 45°、肩肘腕清晰可见的出拳片段，或确认站架（正架/反架）是否匹配。"
            ),
            "top_priority_frame_index": action_metrics[0]["extension_frame_index"] if action_metrics else None,
        }

    valid_count = len(valid_return_times)
    completion_ratio = valid_count / len(real_attempts)

    # 1. Completion component (up to 60 points)
    completion_pts = 60.0 * completion_ratio

    # 2. Return speed component (up to 30 points): <=220ms gets full 30, 550ms+ gets 8
    if valid_return_times:
        ref_ms = float(repeatability["median_ms"]) if repeatability else _median(valid_return_times)
        speed_factor = max(0.0, min(1.0, (550.0 - ref_ms) / (550.0 - 220.0)))
        speed_pts = 8.0 + 22.0 * speed_factor
    else:
        ref_ms = None
        speed_pts = 0.0

    # 3. Consistency component (up to 10 points)
    if repeatability is not None:
        iqr_ms = float(repeatability["iqr_ms"])
        consistency_factor = max(0.0, min(1.0, (180.0 - iqr_ms) / 150.0))
        consistency_pts = 10.0 * consistency_factor
    elif valid_count > 0:
        consistency_pts = 7.0
    else:
        consistency_pts = 0.0

    score = int(round(max(0.0, min(100.0, completion_pts + speed_pts + consistency_pts))))
    if score >= 85:
        grade = f"{score} 分（A · 回防稳固）"
    elif score >= 70:
        grade = f"{score} 分（B · 回防良好）"
    elif score >= 50:
        grade = f"{score} 分（C · 回防待加强）"
    else:
        grade = f"{score} 分（D · 急需强化收手护头）"

    # Determine single highest-priority actionable advice
    missing_ret = [
        m for m in real_attempts if "missing_return" in m.get("quality_reasons", [])
    ]
    short_hold = [
        m for m in real_attempts if "insufficient_return_hold" in m.get("quality_reasons", [])
    ]
    slow_punches = [
        m for m in real_attempts if m.get("return_time_ms") is not None and float(m["return_time_ms"]) > 380.0
    ]

    if missing_ret:
        first = missing_ret[0]
        f_idx = first["extension_frame_index"]
        advice = (
            f"【最高优先：出拳后及时收手】动作 {first['action_id']}（最大伸展源帧 {f_idx}）出拳后后手未回到个人起始护头区。"
            "建议击打到点后立即沿原路弹回下颌/腮旁，避免出拳后掉手或停留在外。"
        )
        target_frame = f_idx
    elif short_hold:
        first = short_hold[0]
        f_idx = first.get("return_frame_index") or first["extension_frame_index"]
        advice = (
            f"【最高优先：回防后稳住护头】动作 {first['action_id']}（回手源帧 {f_idx}）虽然触达护头区，但保持不足 100ms 即再次离开或掉手。"
            "建议收拳归位后先稳固护头再做下一动作。"
        )
        target_frame = f_idx
    elif slow_punches:
        slowest = max(slow_punches, key=lambda m: float(m["return_time_ms"]))
        f_idx = slowest["extension_frame_index"]
        advice = (
            f"【最高优先：加快回手速度】动作 {slowest['action_id']}（最大伸展源帧 {f_idx}）回手耗时 {slowest['return_time_ms']:.0f} ms 偏慢。"
            "建议减少击打最远端的推压停留，做到“快打快收”。"
        )
        target_frame = f_idx
    elif valid_return_times and ref_ms is not None:
        best_or_first = next((m for m in real_attempts if m.get("return_time_ms") is not None), real_attempts[0])
        f_idx = best_or_first["extension_frame_index"]
        shifts = [
            float(m["torso_projection_shift_ratio"])
            for m in real_attempts
            if m.get("torso_projection_shift_ratio") is not None
        ]
        if shifts and _median(shifts) < 0.04:
            advice = (
                f"【最高优先：巩固回防并带动送髋】当前回防扎实（中位回手 {ref_ms:.0f} ms），"
                f"但出拳时二维髋部偏移较小（约 {_median(shifts):.3f}）。可在保持快收护头的前提下，适度加强后脚蹬地转髋。"
            )
        else:
            advice = (
                f"【最高优先：保持当前收手节奏】本段 {valid_count} 次有效后手直拳均完整回防并稳固保持 ≥100ms"
                f"（参考回手时间 {ref_ms:.0f} ms），收手护头习惯良好！"
            )
        target_frame = f_idx
    else:
        first = real_attempts[0]
        target_frame = first["extension_frame_index"]
        advice = (
            f"【最高优先：改善机位与关键点可见度】动作 {first['action_id']}（源帧 {target_frame}）期间肩/肘/腕关键点存在遮挡或置信度不足，"
            "建议保持正侧面或前斜 45° 全身无遮挡拍摄。"
        )

    return {
        "defense_score": score,
        "defense_grade": grade,
        "top_priority_advice": advice,
        "top_priority_frame_index": target_frame,
    }


def build_history_entry(
    metrics_result: dict[str, Any],
    *,
    entry_id: str,
    label: str,
    mirror: bool,
    view: str = "side",
    model_id: str = "mediapipe_pose_landmarker_lite",
) -> dict[str, Any]:
    """T11: Create a privacy-safe local history record bound to capture/model/rule conditions."""
    valid_times = [
        m["return_time_ms"]
        for m in metrics_result.get("action_metrics", [])
        if m.get("return_time_ms") is not None
    ]
    rep = metrics_result.get("repeatability")
    return {
        "entry_id": entry_id,
        "label": label,
        "condition": {
            "stance": metrics_result["stance"],
            "mirror": bool(mirror),
            "view": view,
            "model_id": model_id,
            "rule_version": metrics_result["rule_version"],
        },
        "status": metrics_result["status"],
        "candidate_count": metrics_result["candidate_count"],
        "valid_count": metrics_result["valid_count"],
        "mean_return_time_ms": round(sum(valid_times) / len(valid_times), 2) if valid_times else None,
        "median_return_time_ms": rep["median_ms"] if rep else (round(_median(valid_times), 2) if valid_times else None),
        "iqr_return_time_ms": rep["iqr_ms"] if rep else None,
    }


def compare_same_condition_history(
    entries: Sequence[dict[str, Any]],
    *,
    reference_condition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """T11: Filter local history to strictly matching capture/model/rule conditions.

    Refuses to combine records with different camera view, stance, mirror, model_id,
    or rule_version into a single comparison series.
    """
    if not entries:
        return {
            "reference_condition": reference_condition,
            "compatible_entries": [],
            "excluded_entries": [],
            "comparison_note": "当前会话尚无已保存的本地复盘记录。",
        }

    ref = reference_condition or dict(entries[-1]["condition"])
    compatible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for item in entries:
        cond = item.get("condition", {})
        mismatches = [
            key
            for key in ("stance", "mirror", "view", "model_id", "rule_version")
            if cond.get(key) != ref.get(key)
        ]
        if not mismatches:
            compatible.append(item)
        else:
            excluded.append(
                {
                    "entry_id": item.get("entry_id"),
                    "label": item.get("label"),
                    "mismatch_fields": mismatches,
                }
            )

    return {
        "reference_condition": ref,
        "compatible_entries": compatible,
        "excluded_entries": excluded,
        "comparison_note": (
            f"仅并列展示站架（{ref.get('stance')}）、机位（{ref.get('view')}）、镜像（{ref.get('mirror')}）、"
            f"模型（{ref.get('model_id')}）与规则版本（{ref.get('rule_version')}）完全相同的记录（共 {len(compatible)} 条）；"
            f"已隔离 {len(excluded)} 条不同条件的记录，不跨条件绘制趋势，也不将回手时间长短直接等同于技术好坏。"
        ),
    }
