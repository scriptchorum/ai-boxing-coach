"""Strike event detection and biomechanical technique auditor.

Identifies strike events (Lead Jab / Rear Cross) from landmark timelines
and evaluates the boxer's defensive posture and power generation:
- Rear guard position at extension peak
- Hip rotation engagement
- Strike cadence and lockout
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from jsonschema import Draft202012Validator

from .model_assets import find_project_root
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
    compute_hip_rotation_angle,
)
from .rear_straight_review import straight_events
from .quality_gate import clip_quality_report, metric_quality_gate

if TYPE_CHECKING:
    from .pose import PoseRun
    from .subject_tracker import SubjectTracker


def _quality_validator() -> Draft202012Validator:
    schema = json.loads((find_project_root() / 'schemas/analysis-quality.schema.json').read_text(encoding='utf-8'))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


QUALITY_VALIDATOR = _quality_validator()


class StrikeAuditor:
    """Detects strikes and audits form across a video sequence."""

    def __init__(
        self,
        stance: str = 'orthodox',
        extension_threshold: float = 0.82,
        *,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.stance = stance.lower()
        self.extension_threshold = extension_threshold
        self.width = width
        self.height = height

    def analyze_timeline(self, frames: Sequence[dict]) -> dict[str, Any]:
        """Analyze a sequence of frames for strike events and form quality.

        Each frame must have:
            - 'source_frame_index': int
            - 'timestamp_ms': float
            - 'landmarks': list[dict]
        """
        all_strikes = []
        for event in straight_events(frames, self.stance, width=self.width, height=self.height):
            if event["extension_ratio"] < self.extension_threshold:
                continue
            landmarks = frames[event["pos"]]["landmarks"]
            guard = check_rear_guard(landmarks, stance=self.stance)
            all_strikes.append({
                "strike_type": event["side"] + "_straight",
                "frame_index": event["frame_index"],
                "timestamp_ms": event["timestamp_ms"],
                "extension_ratio": event["extension_ratio"],
                "rear_guard_safe": guard["rear_guard_safe"],
                "rear_guard_dropped": guard["rear_guard_safe"] is False,
                "rear_guard_drop_distance": guard["drop_distance"],
                "hip_rotation_deg": compute_hip_rotation_angle(
                    landmarks, width=self.width, height=self.height
                )["angle_degrees"],
            })
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
                    f"防守警告：在 {guard_drops}/{total_strikes} 个峰值帧中（{guard_drop_rate*100:.1f}%），"
                    "后手明显掉出下巴防守区，实战中极易遭到对手迎击勾拳！出拳时请务必将后手紧贴下颌骨。"
                )
            else:
                feedback.append(f"抱架防守优秀：后手护头失守率仅为 {guard_drop_rate*100:.1f}%，防守意识良好。")

        return {
            'total_strikes': total_strikes,
            'lead_strikes': sum(s['strike_type'] == 'lead_straight' for s in all_strikes),
            'rear_strikes': sum(s['strike_type'] == 'rear_straight' for s in all_strikes),
            'guard_drops': guard_drops,
            'guard_drop_rate': round(guard_drop_rate, 3),
            'avg_hip_rotation_deg': round(avg_hip_rotation, 2) if avg_hip_rotation is not None else None,
            'strikes': all_strikes,
            'coaching_feedback': feedback,
        }

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
            'stance': self.stance,
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


def export_strike_audit_json(audit_result: dict[str, Any]) -> bytes:
    """Serialize a privacy-safe gated strike audit report verified against contracts."""
    reports_to_check = [audit_result.get('clip_report')]
    for strike in audit_result.get('strikes', ()):
        gates = strike.get('gates', {})
        for gate_name in ('extension', 'rear_guard', 'hip_rotation'):
            reports_to_check.append(gates.get(gate_name))
    for report in reports_to_check:
        if not isinstance(report, dict) or next(QUALITY_VALIDATOR.iter_errors(report), None) is not None:
            raise ValueError('击打审计报告未通过质量门禁契约校验。')

    payload = {
        'format': 'boxing-strike-audit-0.1',
        'stance': audit_result.get('stance', 'orthodox'),
        'single_person_confirmed': bool(audit_result['single_person_confirmed']),
        'candidate_strike_count': int(audit_result['candidate_strike_count']),
        'eligible_strike_count': int(audit_result['eligible_strike_count']),
        'lead_strikes': int(audit_result['lead_strikes']),
        'rear_strikes': int(audit_result['rear_strikes']),
        'guard_drops': audit_result['guard_drops'],
        'guard_drop_rate': audit_result['guard_drop_rate'],
        'avg_hip_rotation_deg': audit_result['avg_hip_rotation_deg'],
        'observations': list(audit_result['observations']),
        'clip_report': audit_result['clip_report'],
        'strikes': list(audit_result['strikes']),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
