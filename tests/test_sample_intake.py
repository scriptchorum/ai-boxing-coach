import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.validate_sample_intake import (
    ROOT,
    blank_record,
    g1_readiness,
    load_json,
    validate_intake,
)


def participant(index, *, gender="woman", purpose="demo_g1"):
    return {
        "participant_id": f"participant-{index:02d}",
        "adult_confirmed": True,
        "self_reported_gender_group": gender,
        "authorization": {
            "status": "granted",
            "reference_id": f"authorization-{index:02d}",
            "allowed_uses": [purpose],
            "retention_review_date": "2027-01-31",
            "delete_on_withdrawal": True,
        },
    }


def sample(index, participant_index, *, expected_class="normative"):
    is_bad = expected_class == "bad"
    return {
        "sample_id": f"sample-{index:03d}",
        "participant_id": f"participant-{participant_index:02d}",
        "session_id": f"session-{participant_index:02d}",
        "lineage_id": f"lineage-{index:03d}",
        "state": "probed",
        "use_status": "eligible",
        "local_file_label": f"clip-{index:03d}.mp4",
        "video_sha256": f"{index % 16:x}" * 64,
        "duration_ms": 10000,
        "expected_class": expected_class,
        "bad_reasons": ["occlusion"] if is_bad else [],
        "capture": {
            "view": "side" if index % 2 else "front_45",
            "stance": "orthodox",
            "camera": "rear",
            "mirror_saved": False,
            "target_fps": 30,
            "repetitions": 5,
            "clothing_visibility_group": "fitted_top",
            "single_person": "fail" if is_bad else "pass",
            "full_body_visible": "pass",
            "phone_fixed": "pass",
            "foreground_clear": "pass",
            "lighting_usable": "pass",
        },
    }


class SampleIntakeTests(unittest.TestCase):
    def setUp(self):
        self.good = blank_record(purpose="demo_g1")
        self.good["participants"] = [participant(1)]
        self.good["samples"] = [sample(1, 1)]

    def assert_bad(self, data, code):
        self.assertIn(code, validate_intake(data))

    def test_valid_private_record(self):
        self.assertEqual(validate_intake(self.good), [])
        report = g1_readiness(self.good)
        self.assertFalse(report["g1_ready"])
        self.assertEqual(report["normative_samples"], 1)

    def test_unknown_and_duplicate_ids_are_rejected(self):
        duplicate = copy.deepcopy(self.good)
        duplicate["participants"].append(copy.deepcopy(duplicate["participants"][0]))
        duplicate["samples"].append(copy.deepcopy(duplicate["samples"][0]))
        self.assert_bad(duplicate, "duplicate_participant_id")
        self.assert_bad(duplicate, "duplicate_sample_id")
        missing = copy.deepcopy(self.good)
        missing["samples"][0]["participant_id"] = "participant-99"
        self.assert_bad(missing, "unknown_participant")

    def test_lineage_variants_count_once_and_keep_one_owner(self):
        derivative = copy.deepcopy(self.good["samples"][0])
        derivative.update(sample_id="sample-002", local_file_label="clip-002.mp4")
        self.good["samples"].append(derivative)
        self.assertEqual(validate_intake(self.good), [])
        self.assertEqual(g1_readiness(self.good)["normative_samples"], 1)
        self.good["participants"].append(participant(2))
        derivative["participant_id"] = "participant-02"
        self.assert_bad(self.good, "lineage_metadata_mismatch")

    def test_authorization_gates_eligible_samples(self):
        cases = [
            ("eligible_without_authorization", lambda d: d["participants"][0]["authorization"].update(status="pending", reference_id=None, allowed_uses=[])),
            ("eligible_outside_authorized_use", lambda d: d["participants"][0]["authorization"].update(allowed_uses=["local_media_probe"])),
            ("eligible_participant_not_confirmed_adult", lambda d: d["participants"][0].update(adult_confirmed=False)),
            ("inactive_sample_marked_eligible", lambda d: d["samples"][0].update(state="withdrawn")),
        ]
        for code, mutate in cases:
            with self.subTest(code=code):
                data = copy.deepcopy(self.good)
                mutate(data)
                self.assert_bad(data, code)

    def test_authorization_record_consistency(self):
        data = copy.deepcopy(self.good)
        data["participants"][0]["authorization"]["reference_id"] = None
        self.assert_bad(data, "authorization_reference_required")
        data = copy.deepcopy(self.good)
        data["participants"][0]["authorization"].update(status="withdrawn", allowed_uses=[])
        data["samples"][0]["use_status"] = "do_not_use"
        self.assertEqual(validate_intake(data), [])
        data["participants"][0]["authorization"]["allowed_uses"] = ["demo_g1"]
        self.assert_bad(data, "authorization_scope_without_grant")

    def test_probe_and_bad_sample_fields_are_consistent(self):
        data = copy.deepcopy(self.good)
        data["samples"][0]["video_sha256"] = None
        data["samples"][0]["duration_ms"] = None
        self.assert_bad(data, "probed_hash_required")
        self.assert_bad(data, "probed_duration_required")
        data = copy.deepcopy(self.good)
        data["samples"][0]["bad_reasons"] = ["blur"]
        self.assert_bad(data, "normative_has_bad_reason")
        data = copy.deepcopy(self.good)
        data["samples"][0].update(expected_class="bad", bad_reasons=[])
        self.assert_bad(data, "bad_sample_reason_required")

    def test_schema_excludes_direct_identifiers_and_absolute_paths(self):
        for mutate in (
            lambda d: d["participants"][0].update(real_name="not-allowed"),
            lambda d: d["participants"][0].update(email="person@example.test"),
            lambda d: d["samples"][0].update(local_file_label="C:\\Users\\person\\clip.mp4"),
            lambda d: d["samples"][0].update(notes="free text is intentionally excluded"),
        ):
            data = copy.deepcopy(self.good)
            mutate(data)
            self.assertTrue(any(error.startswith("schema:") for error in validate_intake(data)))

    def test_g1_readiness_uses_frozen_counts(self):
        data = blank_record(purpose="demo_g1")
        data["participants"] = [participant(1), participant(2), participant(3)]
        data["samples"] = [sample(index, (index - 1) % 3 + 1) for index in range(1, 11)]
        data["samples"] += [sample(index, 1, expected_class="bad") for index in range(11, 16)]
        self.assertEqual(validate_intake(data), [])
        report = g1_readiness(data)
        self.assertTrue(report["g1_ready"])
        self.assertEqual(report["normative_samples"], 10)
        self.assertEqual(report["bad_samples"], 5)
        data["participants"][1]["self_reported_gender_group"] = "man"
        report = g1_readiness(data)
        self.assertFalse(report["g1_ready"])
        self.assertIn("g1_women_share_below_70_percent", report["blockers"])
        data = copy.deepcopy(data)
        data["purpose"] = "local_pose_engineering"
        for item in data["participants"]:
            item["authorization"]["allowed_uses"] = ["local_pose_engineering"]
        self.assertEqual(validate_intake(data), [])
        self.assertIn("g1_purpose_not_demo_g1", g1_readiness(data)["blockers"])

    def test_nonfinite_and_duplicate_json_keys_are_rejected(self):
        self.good["samples"][0]["duration_ms"] = float("nan")
        self.assert_bad(self.good, "non_finite_number")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.private.json"
            for content in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_json(path)

    def test_cli_initializes_without_overwrite_and_does_not_echo_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intake.private.json"
            command = [sys.executable, str(ROOT / "tools/validate_sample_intake.py")]
            created = subprocess.run(command + ["--init", str(path)], capture_output=True, text=True)
            self.assertEqual(created.returncode, 0, created.stdout + created.stderr)
            self.assertNotIn(str(path), created.stdout + created.stderr)
            self.assertEqual(validate_intake(load_json(path)), [])
            refused = subprocess.run(command + ["--init", str(path)], capture_output=True, text=True)
            self.assertEqual(refused.returncode, 1)
            validated = subprocess.run(command + [str(path)], capture_output=True, text=True)
            self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)
            self.assertIn("G1_READY: no", validated.stdout)
            self.assertNotIn(str(path), validated.stdout + validated.stderr)


if __name__ == "__main__":
    unittest.main()
