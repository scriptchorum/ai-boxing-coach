"""Strike event detection and biomechanical technique auditor.

Identifies strike events (Lead Jab / Rear Cross) from landmark timelines
and evaluates the boxer's defensive posture and power generation:
- Rear guard position at extension peak
- Hip rotation engagement
- Strike cadence and lockout
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Sequence

from .muaythai_metrics import (
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    check_rear_guard,
    compute_arm_extension_ratio,
    compute_hip_rotation_angle,
)
from .quality_gate import clip_quality_report, metric_quality_gate

if TYPE_CHECKING:
    from .pose import PoseRun
    from .subject_tracker import SubjectTracker


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

    @staticmethod
    def required_indices_for_stance(stance: str) -> dict[str, tuple[int, ...]]:
        """Return required MediaPipe landmark indices for gated strike metrics."""
        normalized = stance.lower()
        if normalized == 'orthodox':
            return {
                'lead_straight': (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
                'rear_straight': (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
                'rear_guard': (LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_WRIST),
                'hip_rotation': (LEFT_HIP, RIGHT_HIP),
            }
        if normalized == 'southpaw':
            return {
                'lead_straight': (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
                'rear_straight': (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
                'rear_guard': (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST),
                'hip_rotation': (LEFT_HIP, RIGHT_HIP),
            }
        raise ValueError(f'Unsupported stance: {stance}')

    def audit_pose_run(
        self,
        run: PoseRun,
        *,
        single_person_confirmed: bool = False,
        declared_reasons: Iterable[str] = (),
        tracker: SubjectTracker | bool | None = None,
    ) -> dict[str, Any]:
        """Gate strike events and 2D metrics with analysis-quality rules."""
        declared = list(declared_reasons)
        clip_report = clip_quality_report(run, declared_reasons=declared, tracker=tracker)
        raw_analysis = self.analyze_timeline(run.frames)
        indices_map = self.required_indices_for_stance(self.stance)

        gated_strikes: list[dict[str, Any]] = []
        for strike in raw_analysis['strikes']:
            frame_index = strike['frame_index']
            strike_type = strike['strike_type']
            ext_metric_id = (
                'lead_extension_ratio' if strike_type == 'lead_straight' else 'rear_extension_ratio'
            )
            event_frames = {'strike_peak': frame_index}

            ext_gate = metric_quality_gate(
                run,
                ext_metric_id,
                indices_map[strike_type],
                event_frames,
                single_person_confirmed=single_person_confirmed,
                declared_reasons=declared,
                tracker=tracker,
            )
            guard_gate = metric_quality_gate(
                run,
                'rear_guard_drop_distance',
                indices_map['rear_guard'],
                event_frames,
                single_person_confirmed=single_person_confirmed,
                declared_reasons=declared,
                tracker=tracker,
            )
            hip_gate = metric_quality_gate(
                run,
                'hip_rotation_deg',
                indices_map['hip_rotation'],
                event_frames,
                single_person_confirmed=single_person_confirmed,
                declared_reasons=declared,
                tracker=tracker,
            )

            ext_eligible = ext_gate['status'] == 'eligible' and ext_gate['value_policy'] == 'compute_later'
            guard_eligible = guard_gate['status'] == 'eligible' and guard_gate['value_policy'] == 'compute_later'
            hip_eligible = hip_gate['status'] == 'eligible' and hip_gate['value_policy'] == 'compute_later'

            gated_strikes.append({
                'strike_type': strike_type,
                'frame_index': frame_index,
                'timestamp_ms': strike['timestamp_ms'],
                'status': ext_gate['status'],
                'extension_ratio': strike['extension_ratio'] if ext_eligible else None,
                'rear_guard_safe': strike['rear_guard_safe'] if guard_eligible else None,
                'rear_guard_dropped': strike['rear_guard_dropped'] if guard_eligible else None,
                'rear_guard_drop_distance': strike['rear_guard_drop_distance'] if guard_eligible else None,
                'hip_rotation_deg': strike['hip_rotation_deg'] if hip_eligible else None,
                'gates': {
                    'extension': ext_gate,
                    'rear_guard': guard_gate,
                    'hip_rotation': hip_gate,
                },
            })

        eligible_events = [item for item in gated_strikes if item['status'] == 'eligible']
        lead_strikes = sum(1 for item in eligible_events if item['strike_type'] == 'lead_straight')
        rear_strikes = sum(1 for item in eligible_events if item['strike_type'] == 'rear_straight')

        eligible_guard_events = [
            item for item in eligible_events if item['rear_guard_dropped'] is not None
        ]
        if eligible_guard_events:
            guard_drops: int | None = sum(
                1 for item in eligible_guard_events if item['rear_guard_dropped']
            )
            guard_drop_rate: float | None = round(guard_drops / len(eligible_guard_events), 3)
        else:
            guard_drops = None
            guard_drop_rate = None

        observed_hip_angles = [
            item['hip_rotation_deg'] for item in eligible_events if item['hip_rotation_deg'] is not None
        ]
        avg_hip_rotation_deg: float | None = (
            round(sum(observed_hip_angles) / len(observed_hip_angles), 2)
            if observed_hip_angles
            else None
        )

        observations: list[str] = []
        if eligible_events:
            observations.append(
                f'已通过门禁的局部伸展峰值共 {len(eligible_events)} 次'
                f'（前手直拳 {lead_strikes} 次，后手直拳 {rear_strikes} 次；二维归一化画面测量）。'
            )
            if guard_drops is not None and guard_drop_rate is not None:
                observations.append(
                    f'在通过护头关键点门禁的 {len(eligible_guard_events)} 次峰值帧中，'
                    f'后手腕低于下巴基准线容差出现 {guard_drops} 次（{guard_drop_rate * 100:.1f}%）。'
                )
            if avg_hip_rotation_deg is not None:
                observations.append(
                    f'在通过双髋关键点门禁的 {len(observed_hip_angles)} 次峰值帧中，'
                    f'二维双髋连线角度均值为 {avg_hip_rotation_deg:.2f}°（受机位视角影响，仅作当前画面描述）。'
                )

        return {
            'clip_report': clip_report,
            'single_person_confirmed': single_person_confirmed,
            'candidate_strike_count': raw_analysis['total_strikes'],
            'eligible_strike_count': len(eligible_events),
            'lead_strikes': lead_strikes,
            'rear_strikes': rear_strikes,
            'guard_drops': guard_drops,
            'guard_drop_rate': guard_drop_rate,
            'avg_hip_rotation_deg': avg_hip_rotation_deg,
            'strikes': gated_strikes,
            'observations': observations,
        }
