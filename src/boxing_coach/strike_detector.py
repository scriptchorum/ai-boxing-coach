"""Strike event detection and biomechanical technique auditor.

Identifies strike events (Lead Jab / Rear Cross) from landmark timelines
and evaluates the boxer's defensive posture and power generation:
- Rear guard position at extension peak
- Hip rotation engagement
- Strike cadence and lockout
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from .muaythai_metrics import (
    check_rear_guard,
    compute_arm_extension_ratio,
    compute_hip_rotation_angle,
)


class StrikeAuditor:
    """Detects strikes and audits form across a video sequence."""

    def __init__(
        self,
        stance: str = 'orthodox',
        extension_threshold: float = 0.82,
        min_strike_interval_frames: int = 5,
    ) -> None:
        self.stance = stance.lower()
        self.extension_threshold = extension_threshold
        self.min_strike_interval_frames = min_strike_interval_frames

    def analyze_timeline(self, frames: Sequence[dict]) -> dict[str, Any]:
        """Analyze a sequence of frames for strike events and form quality.

        Each frame must have:
            - 'source_frame_index': int
            - 'timestamp_ms': float
            - 'landmarks': list[dict]
        """
        lead_ratios = []
        rear_ratios = []
        hip_angles = []

        for f in frames:
            lms = f.get('landmarks', [])
            lead = compute_arm_extension_ratio(lms, arm='lead', stance=self.stance)
            rear = compute_arm_extension_ratio(lms, arm='rear', stance=self.stance)
            hip = compute_hip_rotation_angle(lms)

            lead_ratios.append(lead.get('extension_ratio'))
            rear_ratios.append(rear.get('extension_ratio'))
            hip_angles.append(hip.get('angle_degrees'))

        # Detect strike peaks for lead and rear arms
        lead_strikes = self._find_extension_peaks(lead_ratios, 'lead_straight', frames, hip_angles)
        rear_strikes = self._find_extension_peaks(rear_ratios, 'rear_straight', frames, hip_angles)

        all_strikes = sorted(lead_strikes + rear_strikes, key=lambda s: s['frame_index'])

        # Aggregate statistics
        total_strikes = len(all_strikes)
        guard_drops = sum(1 for s in all_strikes if s['rear_guard_dropped'])
        guard_drop_rate = (guard_drops / total_strikes) if total_strikes > 0 else 0.0

        # Hip rotation stats
        observed_hip_angles = [s['hip_rotation_deg'] for s in all_strikes if s['hip_rotation_deg'] is not None]
        avg_hip_rotation = (sum(observed_hip_angles) / len(observed_hip_angles)) if observed_hip_angles else None

        # Practical feedback generation
        feedback = []
        if total_strikes == 0:
            feedback.append("未检测到明显的直拳完全伸展，请检查镜头机位或出拳幅度。")
        else:
            if guard_drop_rate > 0.35:
                feedback.append(
                    f"防守警告：在 {guard_drops}/{total_strikes} 次击打中（{guard_drop_rate*100:.1f}%），"
                    "后手明显掉出下巴防守区，实战中极易遭到对手迎击勾拳！出拳时请务必将后手紧贴下颌骨。"
                )
            else:
                feedback.append(f"抱架防守优秀：后手护头失守率仅为 {guard_drop_rate*100:.1f}%，防守意识良好。")

        return {
            'total_strikes': total_strikes,
            'lead_strikes': len(lead_strikes),
            'rear_strikes': len(rear_strikes),
            'guard_drops': guard_drops,
            'guard_drop_rate': round(guard_drop_rate, 3),
            'avg_hip_rotation_deg': round(avg_hip_rotation, 2) if avg_hip_rotation is not None else None,
            'strikes': all_strikes,
            'coaching_feedback': feedback,
        }

    def _find_extension_peaks(
        self,
        ratios: list[float | None],
        strike_type: str,
        frames: Sequence[dict],
        hip_angles: list[float | None],
    ) -> list[dict[str, Any]]:
        peaks = []
        n = len(ratios)
        last_peak_idx = -self.min_strike_interval_frames

        for i in range(1, n - 1):
            curr = ratios[i]
            prev = ratios[i - 1]
            nxt = ratios[i + 1]

            if curr is None or curr < self.extension_threshold:
                continue

            # Local maximum condition
            is_peak = (prev is None or curr >= prev) and (nxt is None or curr >= nxt)
            if is_peak and (i - last_peak_idx >= self.min_strike_interval_frames):
                last_peak_idx = i
                f = frames[i]
                lms = f.get('landmarks', [])

                # Check guard status at strike impact
                guard_res = check_rear_guard(lms, stance=self.stance)
                rear_guard_safe = guard_res.get('rear_guard_safe')
                dropped = (rear_guard_safe is False)

                peaks.append({
                    'strike_type': strike_type,
                    'frame_index': f.get('source_frame_index', i),
                    'timestamp_ms': f.get('timestamp_ms', 0.0),
                    'extension_ratio': curr,
                    'rear_guard_safe': rear_guard_safe,
                    'rear_guard_dropped': dropped,
                    'rear_guard_drop_distance': guard_res.get('drop_distance'),
                    'hip_rotation_deg': hip_angles[i],
                })

        return peaks
