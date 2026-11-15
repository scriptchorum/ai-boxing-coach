"""Run the T13/T15 synthetic female-first cohort regression benchmark offline."""
from __future__ import annotations

import json
import sys

from boxing_coach.cohort_evaluation import evaluate_cohort_runs
from boxing_coach.rear_straight_review import segment_rear_straights


def _make_landmark(index: int, x: float, y: float) -> dict:
    return {
        "index": index,
        "image": {"x": x, "y": y, "z": 0.0},
        "world": None,
        "visibility": 0.96,
        "presence": 0.96,
        "state": "observed",
    }


def _make_frame(source_frame_index: int, timestamp_ms: float, *, right_wrist_x: float = 0.44, right_elbow_x: float = 0.42, right_elbow_y: float = 0.44) -> dict:
    return {
        "source_frame_index": source_frame_index,
        "timestamp_ms": float(timestamp_ms),
        "landmarks": [
            _make_landmark(11, 0.52, 0.35),
            _make_landmark(12, 0.40, 0.35),
            _make_landmark(13, 0.54, 0.44),
            _make_landmark(14, right_elbow_x, right_elbow_y),
            _make_landmark(15, 0.56, 0.35),
            _make_landmark(16, right_wrist_x, 0.35),
            _make_landmark(23, 0.51, 0.65),
            _make_landmark(24, 0.41, 0.65),
        ],
    }


def build_synthetic_benchmark_records() -> list[dict]:
    frames: list[dict] = [_make_frame(i, i * 50.0) for i in range(11)]
    f_idx = 11
    for _ in range(5):
        for _ in range(2):
            frames.append(_make_frame(f_idx, f_idx * 50.0))
            f_idx += 1
        frames.append(_make_frame(f_idx, f_idx * 50.0, right_elbow_x=0.50, right_elbow_y=0.35, right_wrist_x=0.60))
        f_idx += 1
        frames.append(_make_frame(f_idx, f_idx * 50.0, right_elbow_x=0.56, right_elbow_y=0.35, right_wrist_x=0.74))
        f_idx += 1
        frames.append(_make_frame(f_idx, f_idx * 50.0, right_elbow_x=0.50, right_elbow_y=0.35, right_wrist_x=0.58))
        f_idx += 1
        for _ in range(4):
            frames.append(_make_frame(f_idx, f_idx * 50.0))
            f_idx += 1

    seg = segment_rear_straights(frames, stance="orthodox", sample_id="syn-bench-1")
    preds = seg["actions"]

    return [
        {
            "synthetic": True,
            "manifest": {
                "sample_id": "syn-bench-1",
                "participant_id": "syn-female-01",
                "session_id": "syn-sess-01",
                "lineage_id": "syn-lin-01",
                "input_sha256": "11" * 32,
                "split": "holdout",
                "capture": {"stance": "orthodox", "is_mirrored": False, "view": "side"},
            },
            "participant_gender": "female",
            "attire_group": "fitted_top",
            "torso_scale_group": "medium_scale",
            "clip_kind": "normative",
            "clip_rejected": False,
            "predicted_actions": preds,
            "ground_truth_actions": preds,
        },
        {
            "synthetic": True,
            "manifest": {
                "sample_id": "syn-bench-bad-1",
                "participant_id": "syn-female-02",
                "session_id": "syn-sess-02",
                "lineage_id": "syn-lin-02",
                "input_sha256": "22" * 32,
                "split": "holdout",
                "capture": {"stance": "orthodox", "is_mirrored": False, "view": "side"},
            },
            "participant_gender": "female",
            "attire_group": "loose_top",
            "torso_scale_group": "medium_scale",
            "clip_kind": "bad",
            "clip_rejected": True,
            "predicted_actions": [],
            "ground_truth_actions": [],
        },
    ]


def main() -> int:
    records = build_synthetic_benchmark_records()
    report = evaluate_cohort_runs(records)
    if report["split_leakage_errors"]:
        print("FAIL: split leakage detected", file=sys.stderr)
        return 1
    if report["overall_metrics"]["segmentation_f1"] < 0.99:
        print("FAIL: synthetic regression F1 dropped", file=sys.stderr)
        return 1
    if report["g2_gate_check"]["status"] != "pending_real_holdout_sample":
        print("FAIL: synthetic run must not claim real G2 pass", file=sys.stderr)
        return 1
    print(
        f"PASS: synthetic cohort benchmark replay OK "
        f"(version={report['frozen_version']['rear_straight_rule_version']}, "
        f"F1={report['overall_metrics']['segmentation_f1']}, "
        f"status={report['g2_gate_check']['status']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
