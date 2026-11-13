"""Tests for strike event detection and form quality auditing."""
import unittest

from boxing_coach.strike_detector import StrikeAuditor


def point(index: int, x: float, y: float) -> dict:
    return {
        'index': index,
        'image': {'x': x, 'y': y, 'z': 0.0},
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
    for i in range(33):
        if i not in (0, 11, 12, 13, 14, 15, 16, 23, 24):
            lms.append({'index': i, 'image': None, 'state': 'missing'})

    return {
        'source_frame_index': frame_idx,
        'timestamp_ms': frame_idx * 33.3,
        'landmarks': lms,
    }


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


if __name__ == '__main__':
    unittest.main()
