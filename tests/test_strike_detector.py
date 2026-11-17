"""Tests for strike event detection and form quality auditing."""
import json
import unittest

from boxing_coach.pose import CONFIG, PoseRun
from boxing_coach.strike_detector import StrikeAuditor, export_strike_audit_json
from tools.validate_quality_reports import validate_report


MODEL={'id':'pose_landmarker_lite','version':'1','asset_sha256':'a'*64}
RUNTIME={'mediapipe':'1.1.1','numpy':'2.5.4','av':'19.0.1'}


def point(index: int, x: float, y: float, *, visibility: float = 0.9, presence: float = 0.9) -> dict:
    return {
        'index': index,
        'image': {'x': x, 'y': y, 'z': 0.0},
        'world': None,
        'visibility': visibility,
        'presence': presence,
        'state': 'observed',
    }


def make_strike_frame(frame_idx: int, elbow_bend_y: float, wrist_x: float, rear_wrist_pos: tuple[float, float] = (0.58, 0.25)) -> dict:
    """Create a synthetic frame with lead arm bent at elbow_bend_y and extending to wrist_x."""
    el_x = 0.4 + (wrist_x - 0.4) * 0.5
    lms = [
        point(0, 0.5, 0.2),              # Nose
        point(11, 0.4, 0.35),            # L shoulder
        point(12, 0.6, 0.35),            # R shoulder
        point(13, el_x, elbow_bend_y),   # L elbow (offset y = bent)
        point(15, wrist_x, 0.35),        # L wrist
        point(14, 0.6, 0.48),            # R elbow (bent in guard)
        point(16, rear_wrist_pos[0], rear_wrist_pos[1]),  # R wrist
        point(23, 0.4, 0.6),             # L hip
        point(24, 0.6, 0.6),             # R hip
    ]
    by_idx = {p['index']: p for p in lms}
    for i in range(33):
        if i not in by_idx:
            if i in (25, 26, 27, 28):
                by_idx[i] = point(i, 0.5, 0.8)
            else:
                by_idx[i] = {
                    'index': i,
                    'image': None,
                    'world': None,
                    'visibility': None,
                    'presence': None,
                    'state': 'missing',
                }

    return {
        'source_frame_index': frame_idx,
        'timestamp_ms': frame_idx * 33.3,
        'landmarks': [by_idx[i] for i in range(33)],
    }


def make_strike_pose_run(frames: list[dict], *, key: str = 'synthetic-strike-run') -> PoseRun:
    diagnostics = [
        {
            'source_frame_index': f['source_frame_index'],
            'engine_timestamp_ms': round(f['timestamp_ms']),
            'detected_count_capped': 1,
            'status': 'single_model_pose',
        }
        for f in frames
    ]
    return PoseRun(
        key,
        dict(MODEL),
        dict(RUNTIME),
        dict(CONFIG),
        frames[0]['source_frame_index'],
        frames[-1]['source_frame_index'],
        False,
        frames,
        diagnostics,
        0.5,
    )


class StrikeDetectorTests(unittest.TestCase):
    def setUp(self):
        self.auditor = StrikeAuditor(stance='orthodox', extension_threshold=0.85)

    def test_empty_frames(self):
        res = self.auditor.analyze_timeline([])
        self.assertEqual(res['total_strikes'], 0)
        self.assertEqual(res['guard_drops'], 0)

    def test_detect_lead_jab_with_safe_guard(self):
        # Frame 0-1: arm bent (elbow y=0.48, ratio ~0.4)
        # Frame 2: lockout (elbow y=0.35, ratio = 1.0)
        # Frame 3-4: arm retracting (elbow y=0.48, ratio ~0.4)
        frames = [
            make_strike_frame(0, elbow_bend_y=0.48, wrist_x=0.45),
            make_strike_frame(1, elbow_bend_y=0.45, wrist_x=0.55),
            make_strike_frame(2, elbow_bend_y=0.35, wrist_x=0.75, rear_wrist_pos=(0.58, 0.25)),  # Lockout peak
            make_strike_frame(3, elbow_bend_y=0.45, wrist_x=0.55),
            make_strike_frame(4, elbow_bend_y=0.48, wrist_x=0.45),
        ]
        res = self.auditor.analyze_timeline(frames)
        self.assertEqual(res['total_strikes'], 1)
        self.assertEqual(res['lead_strikes'], 1)
        self.assertEqual(res['rear_strikes'], 0)
        self.assertEqual(res['guard_drops'], 0)
        self.assertEqual(res['guard_drop_rate'], 0.0)

        strike = res['strikes'][0]
        self.assertEqual(strike['strike_type'], 'lead_straight')
        self.assertEqual(strike['frame_index'], 2)
        self.assertFalse(strike['rear_guard_dropped'])

    def test_detect_lead_jab_with_dropped_guard(self):
        # Lockout at frame 2, but rear wrist dropped to (0.50, 0.45) (below chin, arm still bent)
        frames = [
            make_strike_frame(0, elbow_bend_y=0.48, wrist_x=0.45),
            make_strike_frame(1, elbow_bend_y=0.45, wrist_x=0.55),
            make_strike_frame(2, elbow_bend_y=0.35, wrist_x=0.75, rear_wrist_pos=(0.50, 0.45)),  # Dropped guard!
            make_strike_frame(3, elbow_bend_y=0.45, wrist_x=0.55),
            make_strike_frame(4, elbow_bend_y=0.48, wrist_x=0.45),
        ]
        res = self.auditor.analyze_timeline(frames)
        self.assertEqual(res['total_strikes'], 1)
        self.assertEqual(res['lead_strikes'], 1)
        self.assertEqual(res['guard_drops'], 1)
        self.assertEqual(res['guard_drop_rate'], 1.0)
        self.assertTrue(res['strikes'][0]['rear_guard_dropped'])
        self.assertTrue(any("防守警告" in msg for msg in res['coaching_feedback']))

    def test_required_indices_for_stance(self):
        ortho = StrikeAuditor.required_indices_for_stance('orthodox')
        self.assertEqual(ortho['lead_straight'], (11, 13, 15))
        self.assertEqual(ortho['rear_straight'], (12, 14, 16))
        self.assertEqual(ortho['rear_guard'], (11, 12, 16))
        self.assertEqual(ortho['hip_rotation'], (23, 24))

        south = StrikeAuditor.required_indices_for_stance('southpaw')
        self.assertEqual(south['lead_straight'], (12, 14, 16))
        self.assertEqual(south['rear_straight'], (11, 13, 15))
        self.assertEqual(south['rear_guard'], (11, 12, 15))
        self.assertEqual(south['hip_rotation'], (23, 24))

        with self.assertRaises(ValueError):
            StrikeAuditor.required_indices_for_stance('switch')

    def test_audit_pose_run_enforces_single_person_review_and_null_policy(self):
        frames = [
            make_strike_frame(0, elbow_bend_y=0.48, wrist_x=0.45),
            make_strike_frame(1, elbow_bend_y=0.45, wrist_x=0.55),
            make_strike_frame(2, elbow_bend_y=0.35, wrist_x=0.75, rear_wrist_pos=(0.50, 0.45)),
            make_strike_frame(3, elbow_bend_y=0.45, wrist_x=0.55),
            make_strike_frame(4, elbow_bend_y=0.48, wrist_x=0.45),
        ]
        run = make_strike_pose_run(frames)

        pending = self.auditor.audit_pose_run(run, single_person_confirmed=False)
        self.assertEqual(validate_report(pending['clip_report']), [])
        self.assertEqual(pending['candidate_strike_count'], 1)
        self.assertEqual(pending['eligible_strike_count'], 0)
        self.assertIsNone(pending['guard_drops'])
        self.assertIsNone(pending['guard_drop_rate'])
        self.assertIsNone(pending['avg_hip_rotation_deg'])
        self.assertEqual(pending['observations'], [])
        event = pending['strikes'][0]
        self.assertEqual(event['status'], 'review_required')
        self.assertIsNone(event['extension_ratio'])
        self.assertIsNone(event['rear_guard_safe'])
        self.assertIsNone(event['rear_guard_dropped'])
        self.assertIsNone(event['rear_guard_drop_distance'])
        self.assertIsNone(event['hip_rotation_deg'])
        for gate in event['gates'].values():
            self.assertEqual(validate_report(gate), [])

        eligible = self.auditor.audit_pose_run(run, single_person_confirmed=True)
        self.assertEqual(validate_report(eligible['clip_report']), [])
        self.assertEqual(eligible['eligible_strike_count'], 1)
        self.assertEqual(eligible['lead_strikes'], 1)
        self.assertEqual(eligible['rear_strikes'], 0)
        self.assertEqual(eligible['guard_drops'], 1)
        self.assertEqual(eligible['guard_drop_rate'], 1.0)
        self.assertAlmostEqual(eligible['avg_hip_rotation_deg'], 0.0, places=2)
        self.assertTrue(eligible['observations'])
        self.assertTrue(all('迎击' not in text and '优秀' not in text for text in eligible['observations']))
        eligible_event = eligible['strikes'][0]
        self.assertEqual(eligible_event['status'], 'eligible')
        self.assertGreaterEqual(eligible_event['extension_ratio'], 0.99)
        self.assertFalse(eligible_event['rear_guard_safe'])
        self.assertTrue(eligible_event['rear_guard_dropped'])
        for gate in eligible_event['gates'].values():
            self.assertEqual(validate_report(gate), [])

    def test_audit_pose_run_rejects_low_signal_declared_reason_and_tracker_jump(self):
        frames = [
            make_strike_frame(0, elbow_bend_y=0.48, wrist_x=0.45),
            make_strike_frame(1, elbow_bend_y=0.45, wrist_x=0.55),
            make_strike_frame(2, elbow_bend_y=0.35, wrist_x=0.75, rear_wrist_pos=(0.58, 0.25)),
            make_strike_frame(3, elbow_bend_y=0.45, wrist_x=0.55),
            make_strike_frame(4, elbow_bend_y=0.48, wrist_x=0.45),
        ]
        frames[2]['landmarks'][15]['visibility'] = 0.1  # Lead wrist low signal
        low_sig_run = make_strike_pose_run(frames)
        low_sig = self.auditor.audit_pose_run(low_sig_run, single_person_confirmed=True)
        self.assertEqual(low_sig['eligible_strike_count'], 0)
        # A gap at the peak interrupts the excursion before an audit event exists.
        self.assertEqual(low_sig['candidate_strike_count'], 0)
        self.assertEqual(low_sig['strikes'], [])
        self.assertIsNone(low_sig['guard_drops'])
        self.assertEqual(low_sig['observations'], [])

        clean_frames = [
            make_strike_frame(0, elbow_bend_y=0.48, wrist_x=0.45),
            make_strike_frame(1, elbow_bend_y=0.45, wrist_x=0.55),
            make_strike_frame(2, elbow_bend_y=0.35, wrist_x=0.75, rear_wrist_pos=(0.58, 0.25)),
            make_strike_frame(3, elbow_bend_y=0.45, wrist_x=0.55),
            make_strike_frame(4, elbow_bend_y=0.48, wrist_x=0.45),
        ]
        declared = self.auditor.audit_pose_run(
            make_strike_pose_run(clean_frames),
            single_person_confirmed=True,
            declared_reasons=['declared_occlusion'],
        )
        self.assertEqual(declared['clip_report']['clip_status'], 'rejected')
        self.assertEqual(declared['eligible_strike_count'], 0)
        self.assertEqual(declared['strikes'][0]['status'], 'rejected')
        self.assertIsNone(declared['strikes'][0]['extension_ratio'])
        self.assertIsNone(declared['strikes'][0]['rear_guard_safe'])
        self.assertIsNone(declared['strikes'][0]['hip_rotation_deg'])
        self.assertEqual(declared['observations'], [])

        jump_frames = [
            make_strike_frame(0, elbow_bend_y=0.48, wrist_x=0.45),
            make_strike_frame(1, elbow_bend_y=0.45, wrist_x=0.55),
            make_strike_frame(2, elbow_bend_y=0.35, wrist_x=0.75, rear_wrist_pos=(0.58, 0.25)),
            make_strike_frame(3, elbow_bend_y=0.45, wrist_x=0.55),
            make_strike_frame(4, elbow_bend_y=0.48, wrist_x=0.45),
        ]
        jump_run = make_strike_pose_run(jump_frames)
        jump_run.diagnostics[2]['tracker_reason'] = 'impostor_jump_rejected'
        jumped = self.auditor.audit_pose_run(jump_run, single_person_confirmed=True, tracker=True)
        self.assertEqual(jumped['eligible_strike_count'], 0)
        self.assertEqual(jumped['strikes'][0]['status'], 'rejected')
        self.assertIn('torso_position_jump', jumped['strikes'][0]['gates']['extension']['reason_codes'])
        self.assertIsNone(jumped['strikes'][0]['extension_ratio'])
        self.assertEqual(jumped['observations'], [])

    def test_export_strike_audit_json_is_deterministic_and_privacy_safe(self):
        frames = [
            make_strike_frame(0, elbow_bend_y=0.48, wrist_x=0.45),
            make_strike_frame(1, elbow_bend_y=0.45, wrist_x=0.55),
            make_strike_frame(2, elbow_bend_y=0.35, wrist_x=0.75, rear_wrist_pos=(0.58, 0.25)),
            make_strike_frame(3, elbow_bend_y=0.45, wrist_x=0.55),
            make_strike_frame(4, elbow_bend_y=0.48, wrist_x=0.45),
        ]
        res = self.auditor.audit_pose_run(make_strike_pose_run(frames), single_person_confirmed=True)
        blob1 = export_strike_audit_json(res)
        blob2 = export_strike_audit_json(res)
        self.assertEqual(blob1, blob2)

        text = blob1.decode('utf-8')
        self.assertNotIn('run_key', text)
        self.assertNotIn('"landmarks"', text)
        parsed = json.loads(text)
        self.assertEqual(parsed['format'], 'boxing-strike-audit-0.1')
        self.assertEqual(parsed['stance'], 'orthodox')
        self.assertTrue(parsed['single_person_confirmed'])
        self.assertEqual(parsed['eligible_strike_count'], 1)
        self.assertAlmostEqual(parsed['avg_hip_rotation_deg'], 0.0, places=2)


if __name__ == '__main__':
    unittest.main()
