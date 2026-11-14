"""Subject tracking, multi-pose disambiguation, and persistent recovery protection.

Designed to resolve T05 tracking loss and impostor switching:
1. Multi-candidate scoring: Picks the athlete rather than background persons when count > 1.
2. Long-horizon recovery gate: Validates torso position after dropout across up to 1500ms.
3. Kinematic continuity: Smooths short-term occlusions (<= 5 frames).
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Sequence

TORSO_JOINTS = (11, 12, 23, 24)  # Left shoulder, Right shoulder, Left hip, Right hip
KEY_BOXING_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24)  # Upper body & core


def compute_torso_center(landmarks: Sequence[dict]) -> tuple[float, float] | None:
    """Calculate the normalized (x, y) centroid of the torso quad."""
    points_by_idx = {p['index']: p for p in landmarks if 'index' in p}
    valid_coords = []
    for idx in TORSO_JOINTS:
        p = points_by_idx.get(idx)
        if p and p.get('image') and p.get('state') == 'observed':
            img = p['image']
            x, y = img.get('x'), img.get('y')
            if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                valid_coords.append((x, y))

    if len(valid_coords) < 3:  # Need at least 3 torso points for a reliable center
        return None

    avg_x = sum(c[0] for c in valid_coords) / len(valid_coords)
    avg_y = sum(c[1] for c in valid_coords) / len(valid_coords)
    return (avg_x, avg_y)


def compute_torso_bbox(landmarks: Sequence[dict]) -> tuple[float, float, float, float] | None:
    """Compute (min_x, min_y, max_x, max_y) bounding box for torso."""
    points_by_idx = {p['index']: p for p in landmarks if 'index' in p}
    xs, ys = [], []
    for idx in TORSO_JOINTS:
        p = points_by_idx.get(idx)
        if p and p.get('image') and p.get('state') == 'observed':
            img = p['image']
            x, y = img.get('x'), img.get('y')
            if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def score_boxing_visibility(landmarks: Sequence[dict]) -> float:
    """Return average visibility score across key boxing landmarks."""
    points_by_idx = {p['index']: p for p in landmarks if 'index' in p}
    scores = []
    for idx in KEY_BOXING_JOINTS:
        p = points_by_idx.get(idx)
        if p and p.get('visibility') is not None:
            v = p['visibility']
            if math.isfinite(v):
                scores.append(v)
    return sum(scores) / len(scores) if scores else 0.0


class SubjectTracker:
    """Stateful tracking filter to keep focus on the primary athlete."""

    def __init__(
        self,
        max_recovery_gap_ms: float = 1500.0,
        max_torso_jump_normalized: float = 0.35,
        center_bias_weight: float = 0.2,
    ) -> None:
        self.max_recovery_gap_ms = max_recovery_gap_ms
        self.max_torso_jump_normalized = max_torso_jump_normalized
        self.center_bias_weight = center_bias_weight

        # State
        self.last_seen_center: tuple[float, float] | None = None
        self.last_seen_timestamp_ms: float | None = None
        self.last_seen_landmarks: list[dict] | None = None
        self.total_frames_tracked: int = 0
        self.consecutive_missing: int = 0

    def reset(self) -> None:
        self.last_seen_center = None
        self.last_seen_timestamp_ms = None
        self.last_seen_landmarks = None
        self.total_frames_tracked = 0
        self.consecutive_missing = 0

    def select_best_candidate(
        self,
        candidates: Sequence[Sequence[dict]],
        timestamp_ms: float,
    ) -> tuple[Sequence[dict] | None, str]:
        """Disambiguate among multiple pose candidates.

        Returns: (chosen_landmarks or None, reason_code)
        """
        if not candidates:
            self.consecutive_missing += 1
            return None, 'no_candidate'

        # Score each candidate
        scored = []
        for cand in candidates:
            center = compute_torso_center(cand)
            if not center:
                continue

            vis = score_boxing_visibility(cand)
            dist_to_prev = None
            if self.last_seen_center is not None and self.last_seen_timestamp_ms is not None:
                dt = timestamp_ms - self.last_seen_timestamp_ms
                if 0 <= dt <= self.max_recovery_gap_ms:
                    dx = center[0] - self.last_seen_center[0]
                    dy = center[1] - self.last_seen_center[1]
                    dist_to_prev = math.hypot(dx, dy)

            # Centrality bias (distance from (0.5, 0.5))
            dist_from_center = math.hypot(center[0] - 0.5, center[1] - 0.5)

            scored.append({
                'landmarks': cand,
                'center': center,
                'visibility': vis,
                'dist_to_prev': dist_to_prev,
                'dist_from_center': dist_from_center,
            })

        if not scored:
            self.consecutive_missing += 1
            return None, 'no_reliable_torso'

        # If we have previous history within recovery window, prioritize proximity to last seen
        if self.last_seen_center is not None and self.last_seen_timestamp_ms is not None:
            dt = timestamp_ms - self.last_seen_timestamp_ms
            if 0 <= dt <= self.max_recovery_gap_ms:
                # Filter out candidates that jump unrealistically far (e.g. background person)
                valid_proximities = [
                    c for c in scored
                    if c['dist_to_prev'] is not None and c['dist_to_prev'] <= self.max_torso_jump_normalized
                ]
                if not valid_proximities:
                    # All candidates jumped too far -> likely impostors in background
                    self.consecutive_missing += 1
                    return None, 'impostor_jump_rejected'

                # Best is minimum distance to previous known position
                best = min(valid_proximities, key=lambda c: c['dist_to_prev'])
                self._update_state(best['center'], timestamp_ms, best['landmarks'])
                return best['landmarks'], 'tracked_match'

        # Cold start or post-gap reset: pick the most central candidate with best visibility
        best = max(
            scored,
            key=lambda c: c['visibility'] - (self.center_bias_weight * c['dist_from_center']),
        )
        self._update_state(best['center'], timestamp_ms, best['landmarks'])
        return best['landmarks'], 'initial_or_reacquired'

    def _update_state(
        self,
        center: tuple[float, float],
        timestamp_ms: float,
        landmarks: Sequence[dict],
    ) -> None:
        self.last_seen_center = center
        self.last_seen_timestamp_ms = timestamp_ms
        self.last_seen_landmarks = list(landmarks)
        self.total_frames_tracked += 1
        self.consecutive_missing = 0


def resolve_tracker(tracker: SubjectTracker | bool | None) -> SubjectTracker | None:
    """Resolve an optional tracker flag or instance into a fresh SubjectTracker."""
    if tracker is True:
        return SubjectTracker()
    if isinstance(tracker, SubjectTracker):
        tracker.reset()
        return tracker
    if tracker in (None, False):
        return None
    raise TypeError('tracker 必须是 SubjectTracker、bool 或 None。')


def compare_recovery_points(
    run: Any,
    tracker: SubjectTracker | None = None,
) -> dict[str, Any]:
    """Compare baseline display decisions against SubjectTracker on recovery points."""
    from .quality import assess_frames
    from .review import recovery_references

    baseline_diagnostics = [
        {k: v for k, v in diag.items() if k != 'tracker_reason'}
        for diag in run.diagnostics
    ]
    baseline_run = SimpleNamespace(
        frames=run.frames,
        diagnostics=baseline_diagnostics,
        config=getattr(run, 'config', None),
    )
    before_decisions = assess_frames(baseline_run)

    active_tracker = resolve_tracker(True if tracker is None else tracker)
    assert active_tracker is not None
    shadow_tracker = SubjectTracker(
        max_recovery_gap_ms=active_tracker.max_recovery_gap_ms,
        max_torso_jump_normalized=active_tracker.max_torso_jump_normalized,
        center_bias_weight=active_tracker.center_bias_weight,
    )
    after_decisions = assess_frames(run, tracker=active_tracker)

    tracker_reasons: dict[int, str] = {}
    for frame, diag in zip(run.frames, run.diagnostics, strict=True):
        idx = frame['source_frame_index']
        reason = diag.get('tracker_reason')
        if reason is None:
            if diag['status'] == 'single_model_pose':
                _, reason = shadow_tracker.select_best_candidate(
                    [frame['landmarks']],
                    frame['timestamp_ms'],
                )
            else:
                _, reason = shadow_tracker.select_best_candidate([], frame['timestamp_ms'])
        tracker_reasons[idx] = reason

    multi_refs = recovery_references(run.frames, before_decisions)
    frames_by_idx = {f['source_frame_index']: f for f in run.frames}
    after_by_idx = {
        f['source_frame_index']: dec
        for f, dec in zip(run.frames, after_decisions, strict=True)
    }
    before_by_idx = {
        f['source_frame_index']: dec
        for f, dec in zip(run.frames, before_decisions, strict=True)
    }

    recovery_refs: dict[int, int | None] = {}
    last_visible: int | None = None
    for offset, (frame, diag, before_dec) in enumerate(
        zip(run.frames, baseline_diagnostics, before_decisions, strict=True)
    ):
        idx = frame['source_frame_index']
        if idx in multi_refs:
            recovery_refs[idx] = multi_refs[idx]
        elif offset > 0:
            prev_diag = baseline_diagnostics[offset - 1]
            prev_before = before_decisions[offset - 1]
            if (not prev_before['show'] or prev_diag['status'] != 'single_model_pose') and diag['status'] == 'single_model_pose':
                recovery_refs.setdefault(idx, last_visible)
        if before_dec['show']:
            last_visible = idx

    points = []
    kept_visible_count = 0
    newly_rejected_jump_count = 0
    unchanged_hidden_count = 0

    for frame in run.frames:
        idx = frame['source_frame_index']
        if idx not in recovery_refs:
            continue
        ref_idx = recovery_refs[idx]
        before_dec = before_by_idx[idx]
        after_dec = after_by_idx[idx]

        gap_ms: float | None = None
        jump_dist: float | None = None
        if ref_idx is not None and ref_idx in frames_by_idx:
            ref_frame = frames_by_idx[ref_idx]
            gap_ms = round(float(frame['timestamp_ms']) - float(ref_frame['timestamp_ms']), 3)
            ref_center = compute_torso_center(ref_frame['landmarks'])
            curr_center = compute_torso_center(frame['landmarks'])
            if ref_center is not None and curr_center is not None:
                jump_dist = round(
                    math.hypot(curr_center[0] - ref_center[0], curr_center[1] - ref_center[1]),
                    4,
                )

        before_show = bool(before_dec['show'])
        after_show = bool(after_dec['show'])
        if before_show and after_show:
            kept_visible_count += 1
        elif before_show and not after_show:
            newly_rejected_jump_count += 1
        elif not before_show and not after_show:
            unchanged_hidden_count += 1

        points.append({
            'source_frame_index': idx,
            'reference_frame_index': ref_idx,
            'gap_ms': gap_ms,
            'torso_jump_distance': jump_dist,
            'tracker_reason': tracker_reasons[idx],
            'before_show': before_show,
            'after_show': after_show,
            'before_reasons': list(before_dec['reasons']),
            'after_reasons': list(after_dec['reasons']),
        })

    return {
        'total_recovery_points': len(points),
        'kept_visible_count': kept_visible_count,
        'newly_rejected_jump_count': newly_rejected_jump_count,
        'unchanged_hidden_count': unchanged_hidden_count,
        'points': points,
    }
