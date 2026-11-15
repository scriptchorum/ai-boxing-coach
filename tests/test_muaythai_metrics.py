"""Tests for Muay Thai & boxing biomechanical form assessment metrics."""
import unittest

from boxing_coach.muaythai_metrics import (
    analyze_muaythai_arsenal,
    check_muaythai_high_guard,
    check_rear_guard,
    compute_hip_rotation_angle,
    compute_arm_extension_ratio,
)


def point(index: int, x: float, y: float, state: str = 'observed') -> dict:
    return {
        'index': index,
        'image': {'x': x, 'y': y, 'z': 0.0} if state == 'observed' else None,
        'state': state,
    }


class MuayThaiMetricsTests(unittest.TestCase):
    def test_rear_guard_drop_detection(self):
        # Normal high guard: nose at 0.2, shoulders at 0.35, chin ~0.26
        # Rear wrist (16) is at y=0.25 (above chin -> safe)
        safe_pose = [
            point(0, 0.5, 0.2),    # Nose
            point(11, 0.4, 0.35),  # L shoulder
            point(12, 0.6, 0.35),  # R shoulder
            point(16, 0.58, 0.25), # R wrist (high guard)
        ]
        res = check_rear_guard(safe_pose, stance='orthodox')
        self.assertEqual(res['status'], 'observed')
        self.assertTrue(res['rear_guard_safe'])
        self.assertLessEqual(res['drop_distance'], 0.04)

        # Dropped guard: Rear wrist (16) drops to y=0.45 (at chest level, below chin)
        dropped_pose = [
            point(0, 0.5, 0.2),
            point(11, 0.4, 0.35),
            point(12, 0.6, 0.35),
            point(16, 0.58, 0.45), # R wrist dropped!
        ]
        res_drop = check_rear_guard(dropped_pose, stance='orthodox')
        self.assertEqual(res_drop['status'], 'observed')
        self.assertFalse(res_drop['rear_guard_safe'])
        self.assertGreater(res_drop['drop_distance'], 0.1)

    def test_hip_rotation_angle(self):
        # Square hips (parallel to horizon): left hip at (0.4, 0.6), right hip at (0.6, 0.6)
        square_pose = [
            point(23, 0.4, 0.6),
            point(24, 0.6, 0.6),
        ]
        res = compute_hip_rotation_angle(square_pose)
        self.assertEqual(res['status'], 'observed')
        self.assertAlmostEqual(res['angle_degrees'], 0.0, places=1)

        # Rotated hips (right hip forward/upward in 2D perspective): (0.4, 0.6) to (0.55, 0.5)
        turned_pose = [
            point(23, 0.4, 0.6),
            point(24, 0.55, 0.5),
        ]
        res_turned = compute_hip_rotation_angle(turned_pose)
        self.assertEqual(res_turned['status'], 'observed')
        self.assertNotEqual(res_turned['angle_degrees'], 0.0)

    def test_lead_arm_extension_ratio(self):
        # Full lockout jab: shoulder(0.4, 0.3) -> elbow(0.4, 0.45) -> wrist(0.4, 0.6)
        lockout = [
            point(11, 0.4, 0.3),
            point(13, 0.4, 0.45),
            point(15, 0.4, 0.6),
        ]
        res = compute_arm_extension_ratio(lockout, arm='lead', stance='orthodox')
        self.assertEqual(res['status'], 'observed')
        self.assertTrue(res['is_extended'])
        self.assertAlmostEqual(res['extension_ratio'], 1.0, places=2)

        # Bent arm (e.g. hook or guard): shoulder(0.4, 0.3) -> elbow(0.5, 0.3) -> wrist(0.5, 0.4)
        bent = [
            point(11, 0.4, 0.3),
            point(13, 0.5, 0.3),
            point(15, 0.5, 0.4),
        ]
        res_bent = compute_arm_extension_ratio(bent, arm='lead', stance='orthodox')
        self.assertEqual(res_bent['status'], 'observed')
        self.assertFalse(res_bent['is_extended'])
        self.assertLess(res_bent['extension_ratio'], 0.8)

    def test_muaythai_arsenal_detects_punch_to_kick_combo_knee_and_shin_check(self):
        def base_frame(idx: int) -> dict:
            by_idx = {
                0: point(0, 0.50, 0.20),    # Nose
                11: point(11, 0.42, 0.34),  # L shoulder
                12: point(12, 0.58, 0.34),  # R shoulder
                13: point(13, 0.40, 0.44),  # L elbow (bent guard)
                14: point(14, 0.60, 0.44),  # R elbow (bent guard)
                15: point(15, 0.44, 0.25),  # L wrist (high guard)
                16: point(16, 0.56, 0.25),  # R wrist (high guard)
                23: point(23, 0.44, 0.56),  # L hip
                24: point(24, 0.56, 0.56),  # R hip
                25: point(25, 0.44, 0.74),  # L knee
                26: point(26, 0.56, 0.74),  # R knee
                27: point(27, 0.44, 0.90),  # L ankle
                28: point(28, 0.56, 0.90),  # R ankle
            }
            lms = [by_idx.get(i, point(i, 0.0, 0.0, state='missing')) for i in range(33)]
            return {
                'source_frame_index': idx,
                'timestamp_ms': idx * 40.0,
                'landmarks': lms,
            }

        frames = [base_frame(i) for i in range(30)]
        # Frame 4: Rear straight punch lockout (R shoulder 0.58,0.34 -> R elbow 0.72,0.34 -> R wrist 0.86,0.34)
        frames[4]['landmarks'][14] = point(14, 0.72, 0.34)
        frames[4]['landmarks'][16] = point(16, 0.86, 0.34)

        # Frame 10 (240ms after frame 4): Rear roundhouse kick (R knee 26 & R ankle 28 swing high & extended)
        frames[10]['landmarks'][26] = point(26, 0.42, 0.58)
        frames[10]['landmarks'][28] = point(28, 0.28, 0.58)

        # Frame 18: Lead shin check defense (L knee 25 rises while L ankle 27 stays vertically under L knee, both hands high)
        frames[18]['landmarks'][25] = point(25, 0.44, 0.64)
        frames[18]['landmarks'][27] = point(27, 0.45, 0.78)

        # Frame 25: Rear knee strike (R knee 26 drives up & forward toward hip level)
        frames[25]['landmarks'][26] = point(26, 0.46, 0.55)
        frames[25]['landmarks'][28] = point(28, 0.54, 0.72)

        res = analyze_muaythai_arsenal(frames, stance='orthodox', single_person_confirmed=True)
        self.assertEqual(res['body_framing'], 'full_body')
        self.assertGreaterEqual(res['kick_count'], 1)
        self.assertGreaterEqual(res['knee_count'], 1)
        self.assertGreaterEqual(res['shin_check_count'], 1)
        self.assertEqual(res['combo_count'], 1)
        self.assertIn('后手直拳 → 后腿泰式扫踢', res['punch_leg_combos'][0]['combo_name'])
        self.assertIsNotNone(res['muaythai_drill'])

    def test_muaythai_arsenal_gracefully_handles_upper_body_only_clip(self):
        frames = []
        for idx in range(10):
            frames.append({
                'source_frame_index': idx,
                'timestamp_ms': idx * 33.3,
                'landmarks': [
                    point(0, 0.50, 0.20),
                    point(11, 0.42, 0.34),
                    point(12, 0.58, 0.34),
                    point(13, 0.40, 0.44),
                    point(14, 0.60, 0.44),
                    point(15, 0.44, 0.25),
                    point(16, 0.56, 0.25),
                ],
            })
        hg = check_muaythai_high_guard(frames[0]['landmarks'], stance='orthodox')
        self.assertTrue(hg['high_guard_active'])
        res = analyze_muaythai_arsenal(frames, stance='orthodox', single_person_confirmed=True)
        self.assertEqual(res['body_framing'], 'upper_body_only')
        self.assertEqual(res['high_guard_retention_rate'], 1.0)
        self.assertEqual(res['kick_count'], 0)
        self.assertIn('半身/近景画面', res['muaythai_feedback'])


if __name__ == '__main__':
    unittest.main()
