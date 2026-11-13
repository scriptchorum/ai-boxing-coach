"""Tests for Muay Thai & boxing biomechanical form assessment metrics."""
import unittest

from boxing_coach.muaythai_metrics import (
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


if __name__ == '__main__':
    unittest.main()
