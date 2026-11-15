"""Unit tests for T08-T11 rear straight segmentation, descriptive metrics, override, and history comparison."""
from __future__ import annotations

import json
import unittest

from boxing_coach.rear_straight_review import (
    REVIEW_ANNOTATION_VALIDATOR,
    analyze_jab_and_one_two,
    apply_action_override,
    build_history_entry,
    check_participant_split_leakage,
    compare_same_condition_history,
    compute_rear_straight_metrics,
    export_review_annotation_json,
    segment_rear_straights,
)
from boxing_coach.scorecard_report import (
    build_standalone_scorecard_html,
    summarize_frames_for_batch,
)


def _make_landmark(index: int, x: float, y: float, *, state: str = "observed", vis: float = 0.95) -> dict:
    if state == "missing":
        return {"index": index, "image": None, "world": None, "visibility": None, "presence": None, "state": "missing"}
    return {
        "index": index,
        "image": {"x": x, "y": y, "z": 0.0},
        "world": None,
        "visibility": vis,
        "presence": vis,
        "state": state,
    }


def _make_frame(
    source_frame_index: int,
    timestamp_ms: float,
    *,
    right_wrist_x: float = 0.44,
    right_elbow_x: float = 0.42,
    right_elbow_y: float = 0.44,
    left_wrist_x: float = 0.56,
    left_elbow_x: float = 0.54,
    left_elbow_y: float = 0.44,
    hip_shift_x: float = 0.0,
    right_wrist_state: str = "observed",
) -> dict:
    return {
        "source_frame_index": source_frame_index,
        "timestamp_ms": float(timestamp_ms),
        "landmarks": [
            _make_landmark(11, 0.52, 0.35),
            _make_landmark(12, 0.40, 0.35),
            _make_landmark(13, left_elbow_x, left_elbow_y),
            _make_landmark(14, right_elbow_x, right_elbow_y),
            _make_landmark(15, left_wrist_x, 0.35),
            _make_landmark(16, right_wrist_x, 0.35, state=right_wrist_state),
            _make_landmark(23, 0.51 + hip_shift_x, 0.65),
            _make_landmark(24, 0.41 + hip_shift_x, 0.65),
        ],
    }


def _build_five_punch_sequence(*, stance: str = "orthodox") -> list[dict]:
    """Construct a synthetic 5-punch sequence with 50ms frame step."""
    frames: list[dict] = []
    # Reference window: frames 0..10 (0..500 ms) in guard posture
    for i in range(11):
        frames.append(_make_frame(i, i * 50.0))

    f_idx = 11
    for p in range(5):
        # 2 frames guard before punch
        for _ in range(2):
            frames.append(_make_frame(f_idx, f_idx * 50.0))
            f_idx += 1
        # Extending (2 frames) + Peak extension (1 frame) + Returning (2 frames)
        if stance == "orthodox":
            frames.append(_make_frame(f_idx, f_idx * 50.0, right_elbow_x=0.50, right_wrist_x=0.60, hip_shift_x=0.01))
            f_idx += 1
            frames.append(
                _make_frame(
                    f_idx,
                    f_idx * 50.0,
                    right_elbow_x=0.56,
                    right_wrist_x=0.74,
                    hip_shift_x=0.02 + 0.002 * p,
                )
            )
            f_idx += 1
            frames.append(_make_frame(f_idx, f_idx * 50.0, right_elbow_x=0.50, right_wrist_x=0.58, hip_shift_x=0.01))
            f_idx += 1
        else:
            frames.append(_make_frame(f_idx, f_idx * 50.0, left_elbow_x=0.44, left_wrist_x=0.34, hip_shift_x=-0.01))
            f_idx += 1
            frames.append(_make_frame(f_idx, f_idx * 50.0, left_elbow_x=0.36, left_wrist_x=0.20, hip_shift_x=-0.02))
            f_idx += 1
            frames.append(_make_frame(f_idx, f_idx * 50.0, left_elbow_x=0.44, left_wrist_x=0.34, hip_shift_x=-0.01))
            f_idx += 1
        # Return into guard and hold for 150ms (4 frames at 50ms step)
        for _ in range(4):
            frames.append(_make_frame(f_idx, f_idx * 50.0))
            f_idx += 1
    return frames


class RearStraightReviewTests(unittest.TestCase):
    def test_check_participant_split_leakage_catches_participant_lineage_session_and_hash(self):
        clean = [
            {
                "participant_id": "p1",
                "lineage_id": "l1",
                "session_id": "s1",
                "input_sha256": "h1",
                "split": "development",
            },
            {
                "participant_id": "p2",
                "lineage_id": "l2",
                "session_id": "s2",
                "input_sha256": "h2",
                "split": "holdout",
            },
        ]
        self.assertEqual(check_participant_split_leakage(clean), [])

        leaked = clean + [
            {
                "participant_id": "p1",
                "lineage_id": "l3",
                "session_id": "s1",
                "input_sha256": "h3",
                "split": "holdout",
            }
        ]
        self.assertEqual(
            check_participant_split_leakage(leaked),
            ["split_leakage:participant_id", "split_leakage:session_id"],
        )

    def test_segment_orthodox_and_southpaw_rear_straights(self):
        ortho_frames = _build_five_punch_sequence(stance="orthodox")
        ortho_seg = segment_rear_straights(ortho_frames, stance="orthodox", sample_id="ortho-sample")
        self.assertEqual(ortho_seg["hand"], "right")
        self.assertEqual(len(ortho_seg["actions"]), 5)
        self.assertTrue(all(a["status"] == "valid" for a in ortho_seg["actions"]))

        south_frames = _build_five_punch_sequence(stance="southpaw")
        south_seg = segment_rear_straights(south_frames, stance="southpaw", sample_id="south-sample")
        self.assertEqual(south_seg["hand"], "left")
        self.assertEqual(len(south_seg["actions"]), 5)
        self.assertTrue(all(a["status"] == "valid" for a in south_seg["actions"]))

    def test_segment_feint_incomplete_return_and_missing_critical_points(self):
        frames = [
            _make_frame(i, i * 50.0) for i in range(10)
        ]
        # 1) Feint: leaves guard slightly (ext_ratio between 0.50 and 0.68), then returns
        frames.append(_make_frame(10, 500.0, right_elbow_x=0.45, right_wrist_x=0.51))
        frames.append(_make_frame(11, 550.0))
        frames.append(_make_frame(12, 600.0))
        frames.append(_make_frame(13, 650.0))
        # 2) Full extension at frame 15, but never returns to guard before clip ends
        frames.append(_make_frame(14, 700.0))
        frames.append(_make_frame(15, 750.0, right_elbow_x=0.56, right_wrist_x=0.74))
        frames.append(_make_frame(16, 800.0, right_elbow_x=0.54, right_wrist_x=0.70))

        seg = segment_rear_straights(frames, stance="orthodox", sample_id="edge-sample")
        self.assertEqual(len(seg["actions"]), 2)
        feint_act, no_ret_act = seg["actions"]
        self.assertEqual(feint_act["status"], "rejected")
        self.assertIn("feint_or_incomplete_extension", feint_act["quality_reasons"])
        self.assertEqual(no_ret_act["status"], "partial")
        self.assertIn("missing_return", no_ret_act["quality_reasons"])
        self.assertIsNone(no_ret_act["return_ms"])
        self.assertIsNone(no_ret_act["hold_end_ms"])

    def test_apply_action_override_and_export_review_annotation_json(self):
        frames = _build_five_punch_sequence(stance="orthodox")
        seg = segment_rear_straights(frames, stance="orthodox", sample_id="override-sample")
        first_id = seg["actions"][0]["action_id"]
        orig_ext = seg["actions"][0]["extension_frame_index"]

        patched = apply_action_override(
            seg,
            frames,
            first_id,
            extension_frame_index=orig_ext,
        )
        self.assertEqual(patched["actions"][0]["source"], "manual")
        self.assertEqual(patched["actions"][0]["status"], "valid")

        raw = export_review_annotation_json(patched, annotation_id="ann-01")
        parsed = json.loads(raw.decode("utf-8"))
        REVIEW_ANNOTATION_VALIDATOR.validate(parsed)
        self.assertEqual(parsed["review_status"], "adjusted")
        self.assertEqual(parsed["actions"][0]["source"], "manual")

    def test_compute_metrics_return_time_repeatability_and_null_policy(self):
        frames = _build_five_punch_sequence(stance="orthodox")
        seg = segment_rear_straights(frames, stance="orthodox", sample_id="metrics-sample")
        metrics = compute_rear_straight_metrics(seg, single_person_confirmed=True)

        self.assertEqual(metrics["status"], "complete")
        self.assertEqual(metrics["valid_count"], 5)
        self.assertIsNotNone(metrics["repeatability"])
        self.assertEqual(metrics["repeatability"]["median_ms"], 100.0)
        self.assertEqual(metrics["repeatability"]["iqr_ms"], 0.0)
        self.assertEqual(metrics["defense_score"], 100)
        self.assertIn("A · 回防稳固", metrics["defense_grade"])
        self.assertIn("最高优先", metrics["top_priority_advice"])
        self.assertIsNotNone(metrics["top_priority_frame_index"])
        self.assertTrue(all(m["return_time_ms"] == 100.0 for m in metrics["action_metrics"]))
        self.assertTrue(all(m["torso_projection_shift_ratio"] is not None for m in metrics["action_metrics"]))

        # Unconfirmed single person enforces null policy
        unconfirmed = compute_rear_straight_metrics(seg, single_person_confirmed=False)
        self.assertEqual(unconfirmed["status"], "rejected")
        self.assertEqual(unconfirmed["valid_count"], 0)
        self.assertIsNone(unconfirmed["repeatability"])
        self.assertIsNone(unconfirmed["defense_score"])
        self.assertTrue(all(m["return_time_ms"] is None for m in unconfirmed["action_metrics"]))
        self.assertTrue(all(m["torso_projection_shift_ratio"] is None for m in unconfirmed["action_metrics"]))

    def test_compare_same_condition_history_isolates_mismatched_conditions(self):
        frames = _build_five_punch_sequence(stance="orthodox")
        seg = segment_rear_straights(frames, stance="orthodox", sample_id="hist-sample")
        metrics = compute_rear_straight_metrics(seg, single_person_confirmed=True)

        e1 = build_history_entry(metrics, entry_id="h1", label="第 1 组", mirror=False, view="side")
        e2 = build_history_entry(metrics, entry_id="h2", label="第 2 组", mirror=False, view="side")
        e_diff_view = build_history_entry(metrics, entry_id="h3", label="斜角组", mirror=False, view="front_45")
        e_diff_mirror = build_history_entry(metrics, entry_id="h4", label="镜像组", mirror=True, view="side")

        comp = compare_same_condition_history(
            [e1, e2, e_diff_view, e_diff_mirror],
            reference_condition=e1["condition"],
        )
        self.assertEqual([x["entry_id"] for x in comp["compatible_entries"]], ["h1", "h2"])
        self.assertEqual([x["entry_id"] for x in comp["excluded_entries"]], ["h3", "h4"])
        self.assertEqual(comp["excluded_entries"][0]["mismatch_fields"], ["view"])
        self.assertEqual(comp["excluded_entries"][1]["mismatch_fields"], ["mirror"])

    def test_half_body_sequence_without_hips_produces_full_scoring(self):
        frames = _build_five_punch_sequence(stance="orthodox")
        for f in frames:
            for lm in f["landmarks"]:
                if lm["index"] in (23, 24):
                    lm["state"] = "missing"
                    lm["image"] = None
                    lm["visibility"] = None
                    lm["presence"] = None

        seg = segment_rear_straights(frames, stance="orthodox", sample_id="half-body-sample")
        self.assertEqual(len(seg["actions"]), 5)
        metrics = compute_rear_straight_metrics(seg, single_person_confirmed=True)
        self.assertEqual(metrics["valid_count"], 5)
        self.assertEqual(metrics["defense_score"], 100)
        self.assertIn("A · 回防稳固", metrics["defense_grade"])
        self.assertIsNotNone(metrics["drill_prescription"])
        self.assertIn("title", metrics["drill_prescription"])
        self.assertIsNotNone(metrics["repeatability"])
        self.assertEqual(metrics["repeatability"]["median_ms"], 100.0)

    def test_jab_and_one_two_combo_detection(self):
        frames: list[dict] = [_make_frame(i, i * 50.0) for i in range(11)]
        # Lead jab at frame 13 (650 ms), returns at 15..18, followed by rear straight at frame 17 (850 ms)
        frames.append(_make_frame(11, 550.0))
        frames.append(_make_frame(12, 600.0, left_elbow_x=0.44, left_wrist_x=0.34))
        frames.append(_make_frame(13, 650.0, left_elbow_x=0.36, left_wrist_x=0.20))
        frames.append(_make_frame(14, 700.0, left_elbow_x=0.44, left_wrist_x=0.34))
        frames.append(_make_frame(15, 750.0))
        frames.append(_make_frame(16, 800.0, right_elbow_x=0.50, right_wrist_x=0.60))
        frames.append(_make_frame(17, 850.0, right_elbow_x=0.56, right_wrist_x=0.74))
        frames.append(_make_frame(18, 900.0, right_elbow_x=0.50, right_wrist_x=0.58))
        for k in range(19, 25):
            frames.append(_make_frame(k, k * 50.0))

        combo = analyze_jab_and_one_two(frames, stance="orthodox", sample_id="combo-test")
        self.assertEqual(combo["jab_attempt_count"], 1)
        self.assertEqual(combo["combo_count"], 1)
        self.assertEqual(combo["one_two_combos"][0]["combo_interval_ms"], 200.0)
        self.assertTrue(combo["one_two_combos"][0]["both_returned"])

    def test_hook_uppercut_one_two_three_and_scorecard_html(self):
        frames: list[dict] = [_make_frame(i, i * 50.0) for i in range(11)]
        # 1. Lead jab at frame 13 (650 ms)
        frames.append(_make_frame(11, 550.0))
        frames.append(_make_frame(12, 600.0, left_elbow_x=0.44, left_wrist_x=0.34))
        frames.append(_make_frame(13, 650.0, left_elbow_x=0.36, left_wrist_x=0.20))
        frames.append(_make_frame(14, 700.0, left_elbow_x=0.44, left_wrist_x=0.34))
        frames.append(_make_frame(15, 750.0))
        # 2. Rear straight at frame 17 (850 ms)
        frames.append(_make_frame(16, 800.0, right_elbow_x=0.50, right_wrist_x=0.60))
        frames.append(_make_frame(17, 850.0, right_elbow_x=0.56, right_wrist_x=0.74))
        frames.append(_make_frame(18, 900.0, right_elbow_x=0.50, right_wrist_x=0.58))
        frames.append(_make_frame(19, 950.0))
        frames.append(_make_frame(20, 1000.0))
        # 3. Lead hook at frame 21 (1050 ms): elbow raised at shoulder height (y=0.35), flexed ~90 deg, wrist reaches out
        # L shoulder is at (0.52, 0.35). Put L elbow at (0.42, 0.35) and L wrist at (0.42, 0.25) -> 90 deg flexed hook!
        f21 = _make_frame(21, 1050.0, left_elbow_x=0.42, left_elbow_y=0.35, left_wrist_x=0.39)
        f21["landmarks"][4] = _make_landmark(15, 0.42, 0.23)
        frames.append(f21)
        for k in range(22, 28):
            frames.append(_make_frame(k, k * 50.0))

        combo = analyze_jab_and_one_two(frames, stance="orthodox", sample_id="combo-123")
        self.assertGreaterEqual(combo["hook_count"], 1)
        self.assertEqual(combo["one_two_three_count"], 1)
        self.assertIn("1-2-3 连击", combo["one_two_three_combos"][0]["combo_name"])

        batch_item = summarize_frames_for_batch(
            frames,
            entry_id="batch-01",
            label="clip_123.mp4",
            stance="orthodox",
            view="side",
        )
        self.assertEqual(batch_item["summary_row"]["视频片段"], "clip_123.mp4")
        self.assertIn("1 组 / 1 组", batch_item["summary_row"]["1-2 / 1-2-3 连击"])

        html_bytes = build_standalone_scorecard_html(
            clip_label="clip_123.mp4",
            stance="orthodox",
            view="side",
            rear_metrics=batch_item["rear_metrics"],
            combo_res=batch_item["combo_res"],
            mt_res=batch_item["mt_res"],
        )
        html_text = html_bytes.decode("utf-8")
        self.assertIn("AI 拳击与泰拳动作复盘评分报告", html_text)
        self.assertIn("1-2-3 连击", html_text)
        self.assertNotIn("<script", html_text.lower())


if __name__ == "__main__":
    unittest.main()
