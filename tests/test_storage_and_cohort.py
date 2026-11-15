"""Unit tests for T12 local storage safety and T13/T15 female cohort evaluation & benchmark replay."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from boxing_coach.cohort_evaluation import (
    FROZEN_EVALUATION_VERSION,
    compute_temporal_iou,
    evaluate_clip_actions,
    evaluate_cohort_runs,
)
from boxing_coach.local_storage import (
    inspect_local_storage,
    is_protected_file,
    purge_local_caches_and_reports,
)


class StorageAndCohortTests(unittest.TestCase):
    def test_purge_local_caches_deletes_artifacts_but_never_source_videos_or_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs_dir = root / "outputs"
            runs_dir = root / "runs"
            outside_dir = root / "user_videos"
            outputs_dir.mkdir(parents=True)
            runs_dir.mkdir(parents=True)
            outside_dir.mkdir(parents=True)

            # Removable artifacts inside managed directories
            report_zip = outputs_dir / "boxing-demo-report.private.zip"
            audit_json = runs_dir / "boxing-strike-audit.json"
            report_zip.write_bytes(b"zip-bytes")
            audit_json.write_bytes(b"{}")

            # Protected files inside and outside managed directories
            in_outputs_mp4 = outputs_dir / "accidental_copy.mp4"
            in_outputs_model = outputs_dir / "pose_landmarker_lite.task"
            bound_note = outputs_dir / "bound_source.json"
            external_mp4 = outside_dir / "my_training.mp4"

            in_outputs_mp4.write_bytes(b"video")
            in_outputs_model.write_bytes(b"model")
            bound_note.write_bytes(b"bound")
            external_mp4.write_bytes(b"source-video")

            info_before = inspect_local_storage(
                root, protected_source_paths=[external_mp4, bound_note]
            )
            self.assertEqual(info_before["removable_files"], 2)
            self.assertEqual(info_before["protected_files_skipped"], 3)

            res = purge_local_caches_and_reports(
                root, protected_source_paths=[external_mp4, bound_note]
            )
            self.assertEqual(res["deleted_files"], 2)
            self.assertEqual(res["preserved_protected_files"], 3)

            # Removable files deleted; protected files untouched
            self.assertFalse(report_zip.exists())
            self.assertFalse(audit_json.exists())
            self.assertTrue(in_outputs_mp4.exists())
            self.assertTrue(in_outputs_model.exists())
            self.assertTrue(bound_note.exists())
            self.assertTrue(external_mp4.exists())

            protected, reason = is_protected_file(external_mp4, project_root=root)
            self.assertTrue(protected)
            self.assertEqual(reason, "user_video_media")

    def test_evaluate_clip_actions_counts_rejected_clip_gt_as_fn(self):
        preds = [
            {
                "action_id": "p1",
                "status": "valid",
                "start_ms": 1000.0,
                "extension_ms": 1150.0,
                "return_ms": 1400.0,
            }
        ]
        gts = [
            {
                "action_id": "g1",
                "status": "valid",
                "start_ms": 1020.0,
                "extension_ms": 1160.0,
                "return_ms": 1420.0,
            },
            {
                "action_id": "g2",
                "status": "valid",
                "start_ms": 2000.0,
                "extension_ms": 2150.0,
                "return_ms": 2400.0,
            },
        ]
        self.assertGreater(compute_temporal_iou((1000.0, 1400.0), (1020.0, 1420.0)), 0.8)

        normal = evaluate_clip_actions(preds, gts, clip_rejected=False)
        self.assertEqual(normal["tp"], 1)
        self.assertEqual(normal["fp"], 0)
        self.assertEqual(normal["fn"], 1)
        self.assertEqual(normal["return_time_errors_ms"], [10.0])

        # Rejected clip must still count all annotatable ground-truth actions as FN
        rejected = evaluate_clip_actions(preds, gts, clip_rejected=True)
        self.assertEqual(rejected["tp"], 0)
        self.assertEqual(rejected["fn"], 2)

    def test_evaluate_cohort_runs_reports_female_subgroups_and_g2_honesty_gate(self):
        five_actions = [
            {
                "action_id": f"a{i}",
                "status": "valid",
                "start_ms": 1000.0 + i * 600.0,
                "extension_ms": 1150.0 + i * 600.0,
                "return_ms": 1400.0 + i * 600.0,
            }
            for i in range(5)
        ]
        records = [
            {
                "synthetic": True,
                "manifest": {
                    "sample_id": "s1",
                    "participant_id": "fem-1",
                    "session_id": "sess-1",
                    "lineage_id": "lin-1",
                    "input_sha256": "a1" * 32,
                    "split": "holdout",
                    "capture": {"stance": "orthodox", "is_mirrored": False, "view": "side"},
                },
                "participant_gender": "female",
                "attire_group": "fitted_top",
                "torso_scale_group": "small_scale",
                "clip_kind": "normative",
                "clip_rejected": False,
                "predicted_actions": five_actions,
                "ground_truth_actions": five_actions,
            },
            {
                "synthetic": True,
                "manifest": {
                    "sample_id": "s2",
                    "participant_id": "fem-2",
                    "session_id": "sess-2",
                    "lineage_id": "lin-2",
                    "input_sha256": "b2" * 32,
                    "split": "holdout",
                    "capture": {"stance": "southpaw", "is_mirrored": False, "view": "side"},
                },
                "participant_gender": "female",
                "attire_group": "loose_top",
                "torso_scale_group": "large_scale",
                "clip_kind": "normative",
                "clip_rejected": False,
                "predicted_actions": five_actions[:2],
                "ground_truth_actions": five_actions,
            },
        ]

        report = evaluate_cohort_runs(records)
        self.assertEqual(report["frozen_version"], FROZEN_EVALUATION_VERSION)
        self.assertEqual(report["denominators"]["female_participants"], 2)
        self.assertEqual(report["denominators"]["female_ratio"], 1.0)
        self.assertTrue(
            report["female_subgroups"]["attire_group"]["groups"]["fitted_top"]["exploratory_only"]
        )
        self.assertTrue(
            report["female_subgroups"]["attire_group"]["investigate_gap_over_10pp"]
        )
        self.assertEqual(report["g2_gate_check"]["status"], "pending_real_holdout_sample")


if __name__ == "__main__":
    unittest.main()
