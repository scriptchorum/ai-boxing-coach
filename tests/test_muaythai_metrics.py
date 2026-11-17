"""Tests for Muay Thai & boxing biomechanical form assessment metrics."""
import unittest

from boxing_coach.scorecard_report import build_standalone_scorecard_html, summarize_frames_for_batch

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
        'presence': 0.95, 'visibility': 0.95,
    }


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
        frames = [base_frame(i) for i in range(70)]
        def place(indices, coords):
            for i in indices:
                for joint, (x, y) in coords.items():
                    frames[i]['landmarks'][joint] = point(joint, x, y)
        place([10, 12], {14: (.66, .39), 16: (.74, .31)})
        place([11], {14: (.72, .34), 16: (.86, .34)})
        # Same high foot with hip turn and sustained depth sweep.
        place(range(16, 19), {26: (.51, .59), 28: (.52, .73)})
        place(range(19, 23), {24: (.50, .51), 26: (.43, .56), 28: (.29, .56)})
        for i in range(19, 23):
            frames[i]['landmarks'][28]['image']['z'] = -.15
        place(range(28, 33), {25: (.44, .64), 27: (.45, .78)})
        place(range(37, 42), {26: (.46, .55), 28: (.48, .72)})
        # Stable pelvis, knee rises first, then a straight foot path.
        place(range(51, 54), {26: (.51, .59), 28: (.52, .73)})
        place(range(54, 56), {26: (.48, .57), 28: (.42, .66)})
        place(range(56, 59), {26: (.45, .56), 28: (.31, .58)})
        res = analyze_muaythai_arsenal(frames)
        self.assertEqual(res['body_framing'], 'full_body')
        self.assertEqual([e['category'] for e in res['leg_events']],
                         ['roundhouse_kick', 'knee_strike', 'teep_kick'])
        self.assertEqual([res[k] for k in ('kick_count', 'teep_count', 'knee_count', 'shin_check_count')],
                         [1, 1, 1, 1])
        self.assertEqual(res['combo_count'], 1)
        self.assertIn('后手直拳 → 后腿泰式扫踢', res['punch_leg_combos'][0]['combo_name'])
        self.assertIsNotNone(res['muaythai_drill'])
        self.assertIn('正蹬 1 次', res['muaythai_feedback'])
        batch = summarize_frames_for_batch(frames, entry_id="synthetic", label="simulated")
        self.assertEqual(batch["summary_row"]["扫踢 / 正蹬 / 冲膝 / 提膝格挡"], "1 / 1 / 1 / 1")
        html = build_standalone_scorecard_html(clip_label="simulated", stance="orthodox", view="side",
                                               rear_metrics=batch["rear_metrics"], combo_res=batch["combo_res"],
                                               mt_res=batch["mt_res"]).decode()
        self.assertIn('正蹬 / Stab (Teep)</div><div class="value">1 次', html)
        for event in res['leg_events']:
            self.assertLess(event['start_ms'], event['timestamp_ms'])
            self.assertLess(event['timestamp_ms'], event['return_ms'])

    def test_rotating_hip_with_projected_arc_or_depth_only_sweep(self):
        for depth_only in (False, True):
            with self.subTest(depth_only=depth_only):
                frames = [base_frame(i) for i in range(38)]
                for i in range(13, 23):
                    p = frames[i]["landmarks"]
                    p[24] = point(24, .51, .55)
                    p[26] = point(26, .48, .59)
                    if depth_only:
                        p[28] = point(28, .49, .72)
                        p[28]["image"]["z"] = -.18
                    elif i < 16:
                        p[28] = point(28, .49, .74)
                    elif i < 19:
                        p[28] = point(28, .42, .56)
                    else:
                        p[28] = point(28, .29, .57)
                res = analyze_muaythai_arsenal(frames)
                self.assertEqual([e["category"] for e in res["leg_events"]], ["roundhouse_kick"])

    def test_leg_cooldown_is_per_side_and_has_an_inclusive_time_boundary(self):
        for interval, same_side, expected in [(449, True, 1), (450, True, 2), (230, False, 2)]:
            frames = []
            for ts in sorted({0, 50, 100, 150, 200, 250, 300, 350, 400, 450,
                              *(t + delta for t in (550, 550 + interval)
                                for delta in (-80, -40, 0, 40, 80, 120, 160)), 1250, 1300}):
                f = base_frame(len(frames))
                f['timestamp_ms'] = ts
                for peak, side in [(550, 'rear'), (550 + interval, 'rear' if same_side else 'lead')]:
                    if peak <= ts <= peak + 80:
                        k, a, x = (26, 28, .56) if side == 'rear' else (25, 27, .44)
                        f['landmarks'][k] = point(k, x, .64)
                        f['landmarks'][a] = point(a, x+.01, .78)
                frames.append(f)
            self.assertEqual(analyze_muaythai_arsenal(frames)['shin_check_count'], expected,
                             (interval, same_side))

    def test_low_camera_occlusions_and_unresolved_extensions_do_not_score(self):
        for case in ('single_jump', 'occluded', 'static_high_foot', 'no_return', 'no_turn_or_chamber'):
            frames = [base_frame(i) for i in range(42)]
            for i, f in enumerate(frames):
                moving = ((case == 'single_jump' and i == 22) or case == 'static_high_foot'
                          or (case == 'no_return' and i >= 25)
                          or (case in ('occluded', 'no_turn_or_chamber') and 20 <= i <= 26))
                if moving:
                    f['landmarks'][25] = point(25, .40, .55)
                    f['landmarks'][27] = point(27, .26, .55)
                    if case == 'occluded':
                        f['landmarks'][25]['visibility'] = .52
            res = analyze_muaythai_arsenal(frames)
            self.assertEqual((res['leg_events'], res['shin_checks']), ([], []), case)

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

    def test_footwork_stance_switch_and_depth_noise_do_not_trigger_roundhouse_kicks(self):
        """Stepping, stance switching, and z-depth noise while feet stay near the floor must yield 0 kicks."""
        frames = [base_frame(i) for i in range(40)]
        for i in range(12, 26):
            p = frames[i]['landmarks']
            # Simulate a stance switch / pivot step: hips rotate, knees bend slightly, ankles slide on floor + z noise
            step = 0.08 if i < 20 else 0.0
            tilt_y = -0.025
            p[23] = point(23, 0.48, 0.56 + tilt_y)
            p[24] = point(24, 0.52, 0.55 + tilt_y)
            p[25] = point(25, 0.42 - step, 0.69 + tilt_y)
            p[26] = point(26, 0.58 + step, 0.68 + tilt_y)
            p[27] = point(27, 0.36 - step * 1.4, 0.87 + tilt_y)
            p[28] = point(28, 0.64 + step * 1.4, 0.86 + tilt_y)
            p[28]['image']['z'] = -0.22
        res = analyze_muaythai_arsenal(frames, stance='orthodox', single_person_confirmed=True)
        self.assertEqual(res['kick_count'], 0)
        self.assertEqual(res['teep_count'], 0)
        self.assertEqual(res['knee_count'], 0)


if __name__ == '__main__':
    unittest.main()
