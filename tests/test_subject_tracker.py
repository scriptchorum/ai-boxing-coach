"""Tests for subject tracking, multi-candidate selection, and recovery jump rejection."""
import unittest

from boxing_coach.subject_tracker import (
    SubjectTracker,
    compute_torso_center,
    compute_torso_bbox,
    score_boxing_visibility,
)


def make_landmark(index: int, x: float, y: float, visibility: float = 0.9) -> dict:
    return {
        'index': index,
        'image': {'x': x, 'y': y, 'z': 0.0},
        'visibility': visibility,
        'presence': 0.9,
        'state': 'observed',
    }


def make_person(center_x: float, center_y: float, vis: float = 0.9) -> list[dict]:
    """Create a minimal 33-point landmark list centered around (center_x, center_y)."""
    points = []
    # Shoulders at y = center_y - 0.1, hips at y = center_y + 0.1
    points.append(make_landmark(11, center_x - 0.08, center_y - 0.1, vis))  # L shoulder
    points.append(make_landmark(12, center_x + 0.08, center_y - 0.1, vis))  # R shoulder
    points.append(make_landmark(13, center_x - 0.10, center_y, vis))        # L elbow
    points.append(make_landmark(14, center_x + 0.10, center_y, vis))        # R elbow
    points.append(make_landmark(15, center_x - 0.12, center_y - 0.05, vis)) # L wrist
    points.append(make_landmark(16, center_x + 0.12, center_y - 0.05, vis)) # R wrist
    points.append(make_landmark(23, center_x - 0.06, center_y + 0.1, vis))  # L hip
    points.append(make_landmark(24, center_x + 0.06, center_y + 0.1, vis))  # R hip
    # Pad others
    for i in range(33):
        if i not in (11, 12, 13, 14, 15, 16, 23, 24):
            points.append({'index': i, 'image': None, 'state': 'missing'})
    return points


class SubjectTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = SubjectTracker(max_recovery_gap_ms=1500.0, max_torso_jump_normalized=0.3)

    def test_torso_center_and_bbox(self):
        person = make_person(0.5, 0.5)
        center = compute_torso_center(person)
        self.assertIsNotNone(center)
        self.assertAlmostEqual(center[0], 0.5, places=2)
        self.assertAlmostEqual(center[1], 0.5, places=2)

        bbox = compute_torso_bbox(person)
        self.assertIsNotNone(bbox)
        min_x, min_y, max_x, max_y = bbox
        self.assertLess(min_x, max_x)
        self.assertLess(min_y, max_y)

    def test_multi_candidate_picks_active_subject_over_background(self):
        # Frame 0: athlete in center
        athlete_f0 = make_person(0.5, 0.5)
        chosen, reason = self.tracker.select_best_candidate([athlete_f0], timestamp_ms=0.0)
        self.assertEqual(reason, 'initial_or_reacquired')
        self.assertIsNotNone(chosen)

        # Frame 1: athlete moves slightly (0.51, 0.5), background person appears at (0.1, 0.3)
        athlete_f1 = make_person(0.51, 0.5)
        bystander_f1 = make_person(0.1, 0.3)

        # Pass both candidates in arbitrary order (bystander first)
        chosen, reason = self.tracker.select_best_candidate([bystander_f1, athlete_f1], timestamp_ms=33.0)
        self.assertEqual(reason, 'tracked_match')
        # Ensure chosen center is near 0.51, not 0.1
        chosen_center = compute_torso_center(chosen)
        self.assertAlmostEqual(chosen_center[0], 0.51, places=2)

    def test_impostor_jump_rejected_after_dropout(self):
        # Track athlete at center
        athlete = make_person(0.5, 0.5)
        self.tracker.select_best_candidate([athlete], timestamp_ms=0.0)

        # Gap of 400ms where athlete is occluded
        # Then, only a background person on far left (0.05, 0.2) is detected
        bystander = make_person(0.05, 0.2)
        chosen, reason = self.tracker.select_best_candidate([bystander], timestamp_ms=400.0)

        # Should be rejected because jump distance is too large (> 0.30)
        self.assertEqual(reason, 'impostor_jump_rejected')
        self.assertIsNone(chosen)


if __name__ == '__main__':
    unittest.main()
