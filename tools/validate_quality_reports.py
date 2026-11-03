"""Validate analysis-quality-0.1 reports without echoing private values."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / 'schemas/analysis-quality.schema.json').read_text(encoding='utf-8'))
Draft202012Validator.check_schema(SCHEMA)
VALIDATOR = Draft202012Validator(SCHEMA)


def load_json(path: Path):
    def reject_constant(_):
        raise ValueError('Non-finite JSON number')

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant,
                      object_pairs_hook=unique_keys)


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def validate_report(data) -> list[str]:
    """Return stable structural and semantic error codes without report values."""
    if not _finite(data):
        return ['non_finite_number']
    errors = sorted({
        'schema:' + str(error.validator) + ':' + '/'.join(map(str,error.absolute_path))
        for error in VALIDATOR.iter_errors(data)
    })
    if errors:
        return errors

    def check(condition, code):
        if not condition and code not in errors:
            errors.append(code)

    interpolation=data['interpolation']
    disabled=interpolation['policy'] == 'disabled'
    check((interpolation['max_gap_ms'] == 0) == disabled, 'interpolation_policy_mismatch')
    check((interpolation['applied_frame_count'] == 0) ==
          (interpolation['applied_landmark_count'] == 0), 'interpolation_count_mismatch')

    if 'frame_count' in data:
        frame_count=data['frame_count']
        check(sum(data['model_status_frames'].values()) == frame_count, 'model_status_total')
        check(sum(data['display_state_frames'].values()) == frame_count, 'display_state_total')
        check(all(count <= frame_count for count in data['frame_reason_counts'].values()),
              'frame_reason_overflow')
        check(interpolation['applied_frame_count'] <= frame_count, 'interpolation_frame_overflow')
        check(data['config']['min_visibility'] == data['thresholds']['min_visibility'] and
              data['config']['min_presence'] == data['thresholds']['min_presence'],
              'threshold_config_mismatch')
        check(data['config']['interpolation_max_gap_ms'] == interpolation['max_gap_ms'],
              'interpolation_config_mismatch')
        rejected=bool(
            data['declared_clip_reasons'] or
            data['display_state_frames']['hidden'] == frame_count or
            not disabled or interpolation['applied_landmark_count']
        )
        check(data['clip_status'] == ('rejected' if rejected else 'review_required'),
              'clip_status_mismatch')
    else:
        status=data['status']
        reasons=data['reason_codes']
        review=data['review_reasons']
        policy=data['value_policy']
        check(('interpolation_not_supported' in reasons) ==
              (not disabled or interpolation['applied_landmark_count'] > 0),
              'interpolation_reason_mismatch')
        if status == 'eligible':
            check(not reasons and not review and policy == 'compute_later',
                  'eligible_policy_mismatch')
        elif status == 'review_required':
            check(not reasons and review == ['single_person_not_confirmed'] and
                  policy == 'must_be_null', 'review_policy_mismatch')
        else:
            check(bool(reasons) and policy == 'must_be_null', 'rejected_policy_mismatch')
    return sorted(errors)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files',nargs='+',type=Path,help='Local quality report JSON files')
    args=parser.parse_args(argv)
    failed=False
    for path in args.files:
        try:
            data=load_json(path)
        except (OSError,ValueError):
            print('FAIL: unreadable_or_invalid_json')
            failed=True
            continue
        for error in validate_report(data):
            print('FAIL: '+error)
            failed=True
    if failed:
        return 1
    print(f'PASS: {len(args.files)} analysis quality reports; structural and semantic checks only')
    return 0


if __name__ == '__main__':
    sys.exit(main())
