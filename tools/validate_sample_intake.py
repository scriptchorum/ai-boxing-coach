"""Create and validate a private sample-intake record without reading any video."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schemas/sample-intake.schema.json").read_text(encoding="utf-8"))
Draft202012Validator.check_schema(SCHEMA)
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())
CAPTURE_CHECKS = (
    "single_person",
    "full_body_visible",
    "phone_fixed",
    "foreground_clear",
    "lighting_usable",
)
PURPOSES = (
    "local_media_probe",
    "local_pose_engineering",
    "demo_g1",
    "alpha_development",
    "alpha_holdout",
)


def load_json(path: Path):
    def reject_constant(_value):
        raise ValueError("Non-finite JSON number")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_keys,
    )


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def blank_record(*, collection_id="collection-local-001", purpose="local_pose_engineering"):
    return {
        "schema_version": "0.1.0",
        "record_kind": "private_sample_intake",
        "synthetic": False,
        "collection_id": collection_id,
        "purpose": purpose,
        "participants": [],
        "samples": [],
    }


def validate_intake(data) -> list[str]:
    """Return stable errors without echoing private values or local file labels."""
    if not _finite(data):
        return ["non_finite_number"]
    errors = [
        "schema:" + str(error.validator) + ":" + "/".join(map(str, error.absolute_path))
        for error in VALIDATOR.iter_errors(data)
    ]
    if errors:
        return sorted(set(errors))

    def check(condition, code):
        if not condition and code not in errors:
            errors.append(code)

    participant_ids = [item["participant_id"] for item in data["participants"]]
    sample_ids = [item["sample_id"] for item in data["samples"]]
    check(len(participant_ids) == len(set(participant_ids)), "duplicate_participant_id")
    check(len(sample_ids) == len(set(sample_ids)), "duplicate_sample_id")
    participants = {item["participant_id"]: item for item in data["participants"]}
    lineage_metadata = {}

    for participant in data["participants"]:
        authorization = participant["authorization"]
        if authorization["status"] == "granted":
            check(authorization["reference_id"] is not None, "authorization_reference_required")
            check(bool(authorization["allowed_uses"]), "authorization_scope_required")
        else:
            check(not authorization["allowed_uses"], "authorization_scope_without_grant")

    for sample in data["samples"]:
        participant = participants.get(sample["participant_id"])
        check(participant is not None, "unknown_participant")
        lineage_signature = (
            sample["participant_id"],
            sample["session_id"],
            sample["expected_class"],
        )
        existing_signature = lineage_metadata.setdefault(sample["lineage_id"], lineage_signature)
        check(existing_signature == lineage_signature, "lineage_metadata_mismatch")
        if sample["state"] in ("captured", "probed"):
            check(sample["local_file_label"] is not None, "captured_file_label_required")
        if sample["state"] == "probed":
            check(sample["video_sha256"] is not None, "probed_hash_required")
            check(sample["duration_ms"] is not None, "probed_duration_required")
        if sample["state"] in ("planned", "withdrawn"):
            check(sample["use_status"] == "do_not_use", "inactive_sample_marked_eligible")
        if sample["expected_class"] == "normative":
            check(not sample["bad_reasons"], "normative_has_bad_reason")
        else:
            check(bool(sample["bad_reasons"]), "bad_sample_reason_required")
        if sample["use_status"] != "eligible" or participant is None:
            continue
        authorization = participant["authorization"]
        check(participant["adult_confirmed"], "eligible_participant_not_confirmed_adult")
        check(authorization["status"] == "granted", "eligible_without_authorization")
        check(data["purpose"] in authorization["allowed_uses"], "eligible_outside_authorized_use")
        check(sample["state"] in ("captured", "probed"), "eligible_sample_not_captured")

    return sorted(set(errors))


def g1_readiness(data) -> dict:
    """Summarize frozen G1 collection counts; it does not judge pose or technique."""
    participants = {item["participant_id"]: item for item in data["participants"]}
    normative = {}
    bad = {}
    for sample in data["samples"]:
        if sample["use_status"] != "eligible" or sample["state"] != "probed":
            continue
        if sample["expected_class"] == "bad":
            if sample["bad_reasons"]:
                bad.setdefault(sample["lineage_id"], sample)
            continue
        capture = sample["capture"]
        checks_pass = all(capture[name] == "pass" for name in CAPTURE_CHECKS)
        repetitions = capture["repetitions"]
        if (
            checks_pass
            and capture["view"] in ("side", "front_45")
            and repetitions is not None
            and 5 <= repetitions <= 10
        ):
            normative.setdefault(sample["lineage_id"], sample)

    participant_ids = {sample["participant_id"] for sample in normative.values()}
    women = sum(
        participants[participant_id]["self_reported_gender_group"] == "woman"
        for participant_id in participant_ids
        if participant_id in participants
    )
    total_participants = len(participant_ids)
    blockers = []
    if data["purpose"] != "demo_g1":
        blockers.append("g1_purpose_not_demo_g1")
    if len(normative) < 10:
        blockers.append("g1_normative_samples_below_10")
    if total_participants < 3:
        blockers.append("g1_participants_below_3")
    if total_participants == 0 or women * 100 < total_participants * 70:
        blockers.append("g1_women_share_below_70_percent")
    if len(bad) < 5:
        blockers.append("g1_bad_samples_below_5")
    return {
        "g1_ready": not blockers,
        "normative_samples": len(normative),
        "bad_samples": len(bad),
        "participants": total_participants,
        "women_participants": women,
        "blockers": blockers,
    }


def _write_template(path: Path, *, collection_id: str, purpose: str) -> int:
    if not path.name.endswith(".private.json"):
        print("FAIL: private_suffix_required")
        return 1
    if not path.parent.is_dir():
        print("FAIL: parent_directory_unavailable")
        return 1
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(blank_record(collection_id=collection_id, purpose=purpose), handle, indent=2)
            handle.write("\n")
    except FileExistsError:
        print("FAIL: output_already_exists")
        return 1
    except OSError:
        print("FAIL: output_unavailable")
        return 1
    print("PASS: created ignored private intake template")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="Private intake JSON files")
    parser.add_argument("--init", type=Path, help="Create a new .private.json template without overwriting")
    parser.add_argument("--collection-id", default="collection-local-001")
    parser.add_argument("--purpose", choices=PURPOSES, default="local_pose_engineering")
    args = parser.parse_args(argv)
    if args.init is not None:
        if args.files:
            print("FAIL: init_cannot_validate_files")
            return 1
        candidate = blank_record(collection_id=args.collection_id, purpose=args.purpose)
        if validate_intake(candidate):
            print("FAIL: invalid_template_arguments")
            return 1
        return _write_template(args.init, collection_id=args.collection_id, purpose=args.purpose)
    if not args.files:
        print("FAIL: no_input_files")
        return 1

    failed = False
    for index, path in enumerate(args.files, start=1):
        try:
            data = load_json(path)
        except (OSError, ValueError):
            print(f"FAIL_RECORD[{index}]: unreadable_or_invalid_json")
            failed = True
            continue
        errors = validate_intake(data)
        if errors:
            for error in errors:
                print(f"FAIL_RECORD[{index}]: {error}")
            failed = True
            continue
        report = g1_readiness(data)
        print(f"PASS_RECORD[{index}]: structure_and_authorization_gates")
        print(
            "G1_COUNTS: "
            f"normative={report['normative_samples']} "
            f"bad={report['bad_samples']} "
            f"participants={report['participants']} "
            f"women={report['women_participants']}"
        )
        if report["g1_ready"]:
            print("G1_READY: yes")
        else:
            print("G1_READY: no; " + ",".join(report["blockers"]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
