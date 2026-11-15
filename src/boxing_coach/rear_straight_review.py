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

    # Half-body / upper-body fallback when hips (23, 24) are outside frame or occluded:
    if not (l_hip and r_hip):
        if l_sh and r_sh:
            sh_w = math.hypot(l_sh[0] - r_sh[0], l_sh[1] - r_sh[1])
            if sh_w > 1e-3:
                measures.append(sh_w * 1.35)
        for sh_i, el_i, wr_i in ((LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)):
            s_pt = _observed_pixel_xy(points, sh_i, width, height, min_visibility=min_visibility, min_presence=min_presence)
            e_pt = _observed_pixel_xy(points, el_i, width, height, min_visibility=min_visibility, min_presence=min_presence)
            w_pt = _observed_pixel_xy(points, wr_i, width, height, min_visibility=min_visibility, min_presence=min_presence)
            if s_pt and e_pt and w_pt:
                arm_len = math.hypot(e_pt[0] - s_pt[0], e_pt[1] - s_pt[1]) + math.hypot(w_pt[0] - e_pt[0], w_pt[1] - e_pt[1])
                if arm_len > 1e-3:
                    measures.append(arm_len * 0.85)
            elif s_pt and e_pt:
                upper_len = math.hypot(e_pt[0] - s_pt[0], e_pt[1] - s_pt[1])
                if upper_len > 1e-3:
                    measures.append(upper_len * 1.65)
        if hip_mid is None:
            if l_sh and r_sh:
                hip_mid = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
            elif r_sh:
                hip_mid = r_sh
            elif l_sh:
                hip_mid = l_sh

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
            if offset_xy is not None and (ext_ratio is None or ext_ratio < min_extension_ratio):
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
                "sh_xy": sh_xy,
                "el_xy": el_xy,
                "wr_xy": wr_xy,
                "offset_xy": offset_xy,
                "reach_px": reach_px,
                "ext_ratio": ext_ratio,
                "elbow_angle_deg": elbow_angle_deg,
                "critical_ok": critical_ok,
            }
        )

    # Fallback if no frames fell inside [reference_start_ms, reference_end_ms] (e.g. sub-range start_frame > 0)
    if not ref_scales:
        ref_scales = [f["scale_px"] for f in frame_features if f["scale_px"] is not None]
    if not ref_offsets:
        tucked = sorted(
            [f for f in frame_features if f["offset_xy"] is not None and f["reach_px"] is not None],
            key=lambda x: float(x["reach_px"]),
        )
        take_n = max(1, min(8, len(tucked) // 3 or 1))
        ref_offsets = [f["offset_xy"] for f in tucked[:take_n]]
    if not ref_hips:
        ref_hips = [f["hip_mid_px"] for f in frame_features[:10] if f["hip_mid_px"] is not None]

    baseline_scale_px = round(_median(ref_scales), 3) if ref_scales else None
    baseline_offset_px = (
        (_median([o[0] for o in ref_offsets]), _median([o[1] for o in ref_offsets]))
        if ref_offsets
        else None
    )
    baseline_reach_px = (
        math.hypot(baseline_offset_px[0], baseline_offset_px[1])
        if baseline_offset_px is not None
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
            retracted_guard_ok = (
                baseline_reach_px is not None
                and feat["reach_px"] is not None
                and feat["ext_ratio"] is not None
                and feat["reach_px"] <= baseline_reach_px + 0.55 * guard_radius_px
                and feat["ext_ratio"] <= 0.62
                and dist_from_guard <= guard_radius_px * 1.45
            )
            feat["in_guard"] = (dist_from_guard <= guard_radius_px) or retracted_guard_ok
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
        if feat["timestamp_ms"] < last_valid_hold_end_ms:
            idx += 1
            continue

        # Look for excursion leaving the guard region
        if not feat["critical_ok"] or feat["in_guard"] or feat["guard_dist_px"] is None:
            idx += 1
            continue

        # Start of excursion is the immediately preceding in-guard frame (if >= last_valid_hold_end_ms),
        # otherwise the current frame where excursion is first observed.
        start_pos = idx
        if idx > 0:
            prev_feat = frame_features[idx - 1]
            if (
                prev_feat["critical_ok"]
                and prev_feat["timestamp_ms"] >= last_valid_hold_end_ms
            ):
                start_pos = idx - 1

        # Scan forward until the wrist returns into the guard region (or split at retracted valley between consecutive punches)
        excursion_end = idx
        running_max_reach = 0.0
        running_max_ext = 0.0
        valley_pos: int | None = None
        valley_reach: float | None = None
        while excursion_end < n and not frame_features[excursion_end]["in_guard"]:
            cur_f = frame_features[excursion_end]
            cur_r = cur_f["reach_px"]
            cur_e = cur_f["ext_ratio"] or 0.0
            if cur_r is not None:
                if cur_r >= running_max_reach:
                    running_max_reach = cur_r
                    running_max_ext = max(running_max_ext, cur_e)
                    valley_pos = None
                    valley_reach = None
                elif (
                    running_max_ext >= feint_extension_ratio
                    and guard_radius_px is not None
                    and cur_r <= running_max_reach * 0.75
                ):
                    if valley_reach is None or cur_r <= valley_reach:
                        valley_reach = cur_r
                        valley_pos = excursion_end
                    elif valley_pos is not None and cur_r >= valley_reach + 0.35 * guard_radius_px:
                        # Arm retracted to a local valley and is now extending for a new punch
                        excursion_end = valley_pos
                        break
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
        peak_elbow_angle = peak_feat["elbow_angle_deg"] or 0.0
        peak_reach_scale_ratio = (
            (peak_feat["reach_px"] / baseline_scale_px)
            if (peak_feat["reach_px"] is not None and baseline_scale_px is not None and baseline_scale_px > 1e-3)
            else 0.0
        )

        # Ensure start_pos is strictly before peak_pos if possible
        if start_pos >= peak_pos and peak_pos > 0:
            cand_start = peak_pos - 1
            if frame_features[cand_start]["timestamp_ms"] >= last_valid_hold_end_ms:
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
        foreshortened_full_ext = peak_elbow_angle >= 140.0 and peak_ext_ratio >= 0.78
        if peak_ext_ratio < min_extension_ratio or (
            peak_reach_scale_ratio < min_reach_scale_ratio and not foreshortened_full_ext
        ):
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
        elif (
            single_person_confirmed
            and a["status"] == "partial"
            and a["extension_ms"] is not None
            and a["return_ms"] is not None
        ):
            ret_ms = round(float(a["return_ms"]) - float(a["extension_ms"]), 3)
            ev_items = [
                {
                    "action_id": a["action_id"],
                    "event": ev,
                    "source_frame_index": a[f"{ev}_frame_index"],
                    "timestamp_ms": a[f"{ev}_ms"],
                }
                for ev in ("extension", "return")
            ]
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
        "drill_prescription": coaching["drill_prescription"],
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
            "drill_prescription": None,
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
            "drill_prescription": {
                "drill_id": "drill_camera_and_full_extension",
                "title": "对镜慢速定格直拳 + 机位自检",
                "target_issue": "未检出完整直拳伸展或关键点被遮挡",
                "dosage": "2 组 × 10 次慢速定格",
                "steps": "站在镜头正侧面或前斜 45°，确保双肩、双肘、双腕始终在画面内；慢速打出后手直拳并在最远端定格 0.5 秒后原路收回下颌。",
                "cue": "“肩肘腕入画，打直再快收”",
            },
        }

    valid_count = len(valid_return_times)
    partial_returned_times = [
        float(m["return_time_ms"])
        for m in real_attempts
        if m.get("return_time_ms") is not None
    ]
    partial_return_count = max(0, len(partial_returned_times) - valid_count)
    completion_ratio = (valid_count + 0.65 * partial_return_count) / len(real_attempts)

    # 1. Completion component (up to 60 points)
    completion_pts = 60.0 * min(1.0, completion_ratio)

    # 2. Return speed component (up to 30 points): <=220ms gets full 30, 550ms+ gets 8
    all_observed_times = list(valid_return_times) if valid_return_times else partial_returned_times
    if all_observed_times:
        ref_ms = float(repeatability["median_ms"]) if repeatability else _median(all_observed_times)
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
    elif len(all_observed_times) > 0:
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
        drill = {
            "drill_id": "drill_chin_touch_retraction",
            "title": "后手触腮弹回定型练习（B01 纠错处方）",
            "target_issue": "出拳后掉手或未沿原路收回护头区",
            "dosage": "3 组 × 15 次/组（组间休 30 秒）",
            "steps": "1. 起手前拳套轻触同侧颧骨/下颌；2. 直线打出后手直拳，想象终点是一块烫手钢板，触达瞬间立即原路拉回；3. 每次收回后必须再次物理触碰同侧腮部才算完成 1 次。",
            "cue": "“出去一条线，收回贴腮边”",
        }
    elif short_hold:
        first = short_hold[0]
        f_idx = first.get("return_frame_index") or first["extension_frame_index"]
        advice = (
            f"【最高优先：回防后稳住护头】动作 {first['action_id']}（回手源帧 {f_idx}）虽然触达护头区，但保持不足 100ms 即再次离开或掉手。"
            "建议收拳归位后先稳固护头再做下一动作。"
        )
        target_frame = f_idx
        drill = {
            "drill_id": "drill_one_second_guard_freeze",
            "title": "回防“一秒定格”护头强化练习（B01 纠错处方）",
            "target_issue": "回手触达护头区后停留不足、习惯性二次掉手",
            "dosage": "3 组 × 12 次/组（组间休 30 秒）",
            "steps": "1. 全速打出后手直拳并迅速收回护头位；2. 拳套回到腮旁后强制静止定格满 1 秒（默数“滴-答”），检查双肘收肋、下巴微收；3. 定格稳固后再出下一拳。",
            "cue": "“快打快收，回防定格一秒”",
        }
    elif slow_punches:
        slowest = max(slow_punches, key=lambda m: float(m["return_time_ms"]))
        f_idx = slowest["extension_frame_index"]
        advice = (
            f"【最高优先：加快回手速度】动作 {slowest['action_id']}（最大伸展源帧 {f_idx}）回手耗时 {slowest['return_time_ms']:.0f} ms 偏慢。"
            "建议减少击打最远端的推压停留，做到“快打快收”。"
        )
        target_frame = f_idx
        drill = {
            "drill_id": "drill_snap_retraction_speed",
            "title": "弹力带/空击“收比打快”鞭打练习（B01 纠错处方）",
            "target_issue": "击打末端推拳停留、回手耗时偏长（>380ms）",
            "dosage": "3 组 × 15 次/组（组间休 40 秒）",
            "steps": "1. 将注意力从“往前推重拳”改为“从最远端把拳扯回来”；2. 击打到最远端瞬间手腕微紧即刻放松回抽，使回手速度体感快于出拳速度；3. 配合短促呼气“嘶—”同步收拳。",
            "cue": "“点到即收，拉回比打出更快”",
        }
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
            drill = {
                "drill_id": "drill_rear_pivot_hip_drive",
                "title": "后脚前脚掌旋踩送髋直拳练习（B01 纠错处方）",
                "target_issue": "上肢回防良好但下肢与髋部协同带动偏少",
                "dosage": "3 组 × 12 次/组（组间休 30 秒）",
                "steps": "1. 站好实战架，出后手直拳同时后脚跟主动外旋提起（如踩灭烟头），带动后侧髋部向前送出；2. 出拳同时前手紧贴前腮护头；3. 后脚旋回原位同时后手同步弹回下颌。",
                "cue": "“后脚蹬转送髋，收胯同步收拳”",
            }
        else:
            advice = (
                f"【最高优先：保持当前收手节奏】本段 {valid_count} 次有效后手直拳均完整回防并稳固保持 ≥100ms"
                f"（参考回手时间 {ref_ms:.0f} ms），收手护头习惯良好！"
            )
            drill = {
                "drill_id": "drill_step_and_cross_consistency",
                "title": "前后滑步移动中后手直拳回防巩固练习（B01 进阶处方）",
                "target_issue": "原地后手直拳回防已达标，进阶巩固移动中护头稳定性",
                "dosage": "3 组 × 10 次（每次含前滑步→后手直拳→回防→后撤步）",
                "steps": "1. 前滑步半步站稳瞬间打出后手直拳；2. 确保拳套先回到下颌护头位，再启动后撤步移动；3. 全程保持回手时间稳定在 300ms 以内。",
                "cue": "“先收拳护头，再动步撤离”",
            }
        target_frame = f_idx
    else:
        first = real_attempts[0]
        target_frame = first["extension_frame_index"]
        advice = (
            f"【最高优先：改善机位与关键点可见度】动作 {first['action_id']}（源帧 {target_frame}）期间肩/肘/腕关键点存在遮挡或置信度不足，"
            "建议保持正侧面或前斜 45° 全身无遮挡拍摄。"
        )
        drill = {
            "drill_id": "drill_camera_and_full_extension",
            "title": "对镜慢速定格直拳 + 机位自检（B01 纠错处方）",
            "target_issue": "关键点遮挡或回防轨迹不完整",
            "dosage": "2 组 × 10 次",
            "steps": "调整镜头至正侧面或前斜 45°，确保出拳侧肩、肘、腕全程无遮挡，完成慢速出拳与原路收手。",
            "cue": "“侧面或斜 45°，快打快收贴腮”",
        }

    return {
        "defense_score": score,
        "defense_grade": grade,
        "top_priority_advice": advice,
        "top_priority_frame_index": target_frame,
        "drill_prescription": drill,
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


def _detect_bent_arm_punches(
    seg_dict: dict[str, Any],
    *,
    hand_label: str,
    straight_frames: set[int],
) -> list[dict[str, Any]]:
    """Detect Hook (平勾拳) and Uppercut (上勾拳) from flexed-elbow arm kinematics."""
    feats = seg_dict.get("frame_features", [])
    scale_px = seg_dict.get("metric_config", {}).get("scale_px")
    if not feats or scale_px is None or scale_px <= 1e-3:
        return []

    detected: list[dict[str, Any]] = []
    last_pos = -5
    n = len(feats)
    for i in range(1, n - 1):
        f = feats[i]
        src_idx = int(f["source_frame_index"])
        if any(abs(src_idx - sf) <= 3 for sf in straight_frames):
            continue
        if i - last_pos < 4 or not f.get("critical_ok"):
            continue
        sh_xy = f.get("sh_xy")
        el_xy = f.get("el_xy")
        wr_xy = f.get("wr_xy")
        reach_px = f.get("reach_px")
        ext_ratio = f.get("ext_ratio")
        elbow_ang = f.get("elbow_angle_deg")
        if (
            sh_xy is None
            or el_xy is None
            or wr_xy is None
            or reach_px is None
            or ext_ratio is None
            or elbow_ang is None
        ):
            continue

        arm_reach_px = math.hypot(wr_xy[0] - sh_xy[0], wr_xy[1] - sh_xy[1])
        guard_disp_px = float(f.get("guard_dist_px") or arm_reach_px)
        disp_ratio = guard_disp_px / float(scale_px)
        # Bent-arm punch must have flexed elbow (not a straight lockout) and active excursion out of guard
        if not (45.0 <= float(elbow_ang) <= 135.0 and disp_ratio >= 0.36 and float(ext_ratio) < 0.85):
            continue
        if f.get("in_guard"):
            continue

        prev_f = feats[i - 1]
        next_f = feats[i + 1]
        prev_d = float(prev_f.get("guard_dist_px") or prev_f.get("reach_px") or 0.0)
        next_d = float(next_f.get("guard_dist_px") or next_f.get("reach_px") or 0.0)
        if not (guard_disp_px >= prev_d - 1e-3 and guard_disp_px >= next_d - 1e-3):
            continue

        # Distinguish Hook (elbow raised near shoulder level) vs Uppercut (elbow dipped below shoulder & wrist rises)
        el_drop_ratio = (el_xy[1] - sh_xy[1]) / float(scale_px)
        wr_above_el_ratio = (el_xy[1] - wr_xy[1]) / float(scale_px)

        punch_kind: str | None = None
        punch_label: str | None = None
        if -0.25 <= el_drop_ratio <= 0.30 and abs(wr_xy[1] - sh_xy[1]) <= 0.42 * float(scale_px):
            punch_kind = f"{'lead' if hand_label == '前手' else 'rear'}_hook"
            punch_label = f"{hand_label}平勾拳 (Hook)"
        elif el_drop_ratio > 0.30 and wr_above_el_ratio >= 0.16:
            punch_kind = f"{'lead' if hand_label == '前手' else 'rear'}_uppercut"
            punch_label = f"{hand_label}上勾拳 (Uppercut)"

        if punch_kind is not None and punch_label is not None:
            last_pos = i
            returned = any(bool(feats[j].get("in_guard")) for j in range(i + 1, min(n, i + 15)))
            detected.append(
                {
                    "punch_id": f"{punch_kind}_{len(detected) + 1:02d}",
                    "punch_kind": punch_kind,
                    "punch_type": punch_label,
                    "frame_index": src_idx,
                    "timestamp_ms": round(float(f["timestamp_ms"]), 2),
                    "elbow_angle_deg": round(float(elbow_ang), 1),
                    "returned_to_guard": returned,
                }
            )
    return detected


def analyze_jab_and_one_two(
    frames: Sequence[dict[str, Any]],
    *,
    rear_segmentation: dict[str, Any] | None = None,
    width: int = 1280,
    height: int = 720,
    stance: str = "orthodox",
    sample_id: str = "local-sample",
    single_person_confirmed: bool = True,
    max_combo_gap_ms: float = 750.0,
) -> dict[str, Any]:
    """B02: Segment lead straight (Jab) actions and pair Jab -> Rear Straight (1-2) combinations."""
    s_norm = stance.lower()
    lead_stance_proxy = "southpaw" if s_norm == "orthodox" else ("orthodox" if s_norm == "southpaw" else "unknown")
    jab_seg = segment_rear_straights(
        frames,
        width=width,
        height=height,
        stance=lead_stance_proxy,
        sample_id=f"{sample_id}-jab",
        min_reach_scale_ratio=0.70,
        feint_reach_scale_ratio=0.40,
    )
    for act in jab_seg["actions"]:
        act["action_kind"] = "lead_straight"
    jab_metrics = compute_rear_straight_metrics(jab_seg, single_person_confirmed=single_person_confirmed)

    rear_seg = rear_segmentation or segment_rear_straights(
        frames,
        width=width,
        height=height,
        stance=stance,
        sample_id=sample_id,
    )
    rear_metrics = compute_rear_straight_metrics(rear_seg, single_person_confirmed=single_person_confirmed)

    jab_feat_map = {int(f["source_frame_index"]): f for f in jab_seg.get("frame_features", [])}
    rear_feat_map = {int(f["source_frame_index"]): f for f in rear_seg.get("frame_features", [])}

    def _is_bent_punch_peak(feat_map: dict[int, dict[str, Any]], frame_idx: int | None) -> bool:
        if frame_idx is None or int(frame_idx) not in feat_map:
            return False
        ft = feat_map[int(frame_idx)]
        ang = ft.get("elbow_angle_deg")
        ext = ft.get("ext_ratio")
        return ang is not None and ext is not None and float(ang) < 128.0 and float(ext) < 0.82

    jab_Real = [
        m for m in jab_metrics["action_metrics"]
        if "feint_or_incomplete_extension" not in m.get("quality_reasons", [])
        and not _is_bent_punch_peak(jab_feat_map, m.get("extension_frame_index"))
    ]
    rear_Real = [
        m for m in rear_metrics["action_metrics"]
        if "feint_or_incomplete_extension" not in m.get("quality_reasons", [])
        and not _is_bent_punch_peak(rear_feat_map, m.get("extension_frame_index"))
    ]

    combos: list[dict[str, Any]] = []
    used_rear_ids: set[str] = set()
    for j_act in jab_Real:
        j_ext_ms = j_act.get("extension_ms")
        if j_ext_ms is None:
            continue
        best_rear = None
        for r_act in rear_Real:
            if r_act["action_id"] in used_rear_ids:
                continue
            r_ext_ms = r_act.get("extension_ms")
            if r_ext_ms is None:
                continue
            gap_ms = float(r_ext_ms) - float(j_ext_ms)
            if 40.0 <= gap_ms <= max_combo_gap_ms:
                best_rear = (r_act, round(gap_ms, 1))
                break
        if best_rear is not None:
            r_act, gap_ms = best_rear
            used_rear_ids.add(r_act["action_id"])
            combos.append(
                {
                    "combo_id": f"{sample_id}-12-{len(combos) + 1}",
                    "jab_action_id": j_act["action_id"],
                    "rear_action_id": r_act["action_id"],
                    "jab_extension_frame_index": j_act["extension_frame_index"],
                    "rear_extension_frame_index": r_act["extension_frame_index"],
                    "combo_interval_ms": gap_ms,
                    "jab_return_time_ms": j_act.get("return_time_ms"),
                    "rear_return_time_ms": r_act.get("return_time_ms"),
                    "both_returned": (
                        j_act.get("return_time_ms") is not None
                        and r_act.get("return_time_ms") is not None
                    ),
                }
            )

    if combos:
        intervals = [c["combo_interval_ms"] for c in combos]
        med_interval = round(_median(intervals), 1)
        both_ok = sum(1 for c in combos if c["both_returned"])
        if med_interval <= 320.0 and both_ok == len(combos):
            combo_feedback = (
                f"检出 {len(combos)} 组 1-2 连击，前手刺拳到后手直拳中位衔接间隔 {med_interval:.0f} ms，"
                "连击节奏紧凑且前后手均完成回防！"
            )
        elif both_ok < len(combos):
            combo_feedback = (
                f"检出 {len(combos)} 组 1-2 连击（中位衔接 {med_interval:.0f} ms），"
                f"其中 {len(combos) - both_ok} 组在连击过程中出现前手或后手未完整回防，注意“打二收一、打完双收”。"
            )
        else:
            combo_feedback = (
                f"检出 {len(combos)} 组 1-2 连击，中位衔接间隔 {med_interval:.0f} ms；"
                "可在刺拳触点回抽瞬间同步启动后手转髋，进一步缩短 1-2 衔接停顿。"
            )
    elif jab_Real:
        combo_feedback = (
            f"检出前手刺拳 (Jab) {len(jab_Real)} 次（有效完整回手 {jab_metrics['valid_count']} 次），"
            "未检出紧随刺拳（≤750ms）的 1-2 连击组合。"
        )
    else:
        combo_feedback = "当前片段以单拳或后手练习为主，未检出明显的前手刺拳或 1-2 连击。"

    straight_frames = {
        int(m["extension_frame_index"])
        for m in (jab_Real + rear_Real)
        if m.get("extension_frame_index") is not None
    }
    lead_bent = _detect_bent_arm_punches(jab_seg, hand_label="前手", straight_frames=straight_frames)
    rear_bent = _detect_bent_arm_punches(rear_seg, hand_label="后手", straight_frames=straight_frames)
    bent_punches = sorted(lead_bent + rear_bent, key=lambda p: int(p["frame_index"]))
    hook_count = sum(1 for p in bent_punches if p["punch_kind"].endswith("_hook"))
    uppercut_count = sum(1 for p in bent_punches if p["punch_kind"].endswith("_uppercut"))

    one_two_three_combos: list[dict[str, Any]] = []
    used_bent: set[str] = set()
    for c in combos:
        r_act = next((r for r in rear_Real if r["action_id"] == c["rear_action_id"]), None)
        if r_act is None or r_act.get("extension_ms") is None:
            continue
        r_ms = float(r_act["extension_ms"])
        for bp in lead_bent:
            if bp["punch_id"] in used_bent:
                continue
            dt3 = float(bp["timestamp_ms"]) - r_ms
            if 40.0 <= dt3 <= max_combo_gap_ms:
                used_bent.add(bp["punch_id"])
                one_two_three_combos.append(
                    {
                        "combo_id": f"{sample_id}-123-{len(one_two_three_combos) + 1}",
                        "combo_name": f"1-2-3 连击（刺拳 → 后手直拳 → {bp['punch_type']}）",
                        "jab_frame_index": c["jab_extension_frame_index"],
                        "rear_frame_index": c["rear_extension_frame_index"],
                        "third_punch_frame_index": bp["frame_index"],
                        "third_punch_type": bp["punch_type"],
                        "two_to_three_ms": round(dt3, 1),
                        "all_returned": bool(c["both_returned"] and bp["returned_to_guard"]),
                    }
                )
                break

    if bent_punches:
        combo_feedback += (
            f" 另检出屈臂拳法：平勾拳 {hook_count} 次、上勾拳 {uppercut_count} 次"
            + (f"、1-2-3 经典三连击 {len(one_two_three_combos)} 组。" if one_two_three_combos else "。")
        )

    return {
        "jab_segmentation": jab_seg,
        "jab_metrics": jab_metrics,
        "jab_attempt_count": len(jab_Real),
        "jab_valid_count": jab_metrics["valid_count"],
        "jab_defense_grade": jab_metrics["defense_grade"],
        "jab_defense_score": jab_metrics["defense_score"],
        "one_two_combos": combos,
        "combo_count": len(combos),
        "hook_count": hook_count,
        "uppercut_count": uppercut_count,
        "bent_arm_punches": bent_punches,
        "one_two_three_combos": one_two_three_combos,
        "one_two_three_count": len(one_two_three_combos),
        "combo_feedback": combo_feedback,
    }
