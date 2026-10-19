"""Validate v0.1.0 contracts offline; this does not analyze video or infer poses."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / 'schemas/contracts.schema.json').read_text(encoding='utf-8'))
Draft202012Validator.check_schema(SCHEMA)
VALIDATOR = Draft202012Validator(SCHEMA)
EVENTS = ('start', 'extension', 'return', 'hold_end')


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
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return True


def _quantile(values, p):
    """Linear interpolation at (n - 1) * p; contract convention, not a benchmark."""
    values = sorted(values)
    index = (len(values) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def validate_bundle(data) -> list[str]:
    """Return stable error codes, without echoing participant data or JSON values."""
    if not _finite(data):
        return ['non_finite_number']
    errors = ['schema:' + str(e.validator) + ':' + '/'.join(map(str, e.absolute_path))
              for e in VALIDATOR.iter_errors(data)]
    if errors:
        return errors

    def check(condition, code):
        if not condition and code not in errors:
            errors.append(code)

    m, p, r, actions = (data[k] for k in ('manifest', 'pose', 'report', 'actions'))
    check(all(x['sample_id'] == m['sample_id'] for x in [p, r, *actions]), 'sample_mismatch')
    check(r['input_sha256'] == m['input_sha256'], 'input_hash_mismatch')
    check(r['model'] == p['model'], 'model_mismatch')
    check(r['capture'] == m['capture'], 'capture_mismatch')
    try:
        datetime.fromisoformat(r['generated_at'].replace('Z', '+00:00'))
    except ValueError:
        check(False, 'invalid_generated_at')
    consent = m['consent']['status']
    if consent == 'granted':
        check(m['consent']['reference_id'] is not None, 'consent_reference_required')
    else:
        code = 'consent_missing' if consent == 'not_recorded' else 'consent_withdrawn'
        check(r['status'] == 'rejected' and code in r['quality_reasons'], 'consent_gate')
        check(not p['frames'] and not actions, 'consent_processing_forbidden')

    frames = p['frames']
    frame_map = {}
    previous_index, previous_ms = -1, -1
    for f in frames:
        index, ms = f['source_frame_index'], f['timestamp_ms']
        check(index > previous_index and ms > previous_ms, 'non_monotonic_frames')
        check(index < m['media']['frame_count'] and ms <= m['duration_ms'], 'frame_out_of_bounds')
        if index == 0:
            check(ms == 0, 'first_pts_not_zero')
        previous_index, previous_ms = index, ms
        points = {x['index']: x for x in f['landmarks']}
        check(len(points) == len(f['landmarks']), 'duplicate_landmark')
        for point in points.values():
            if point['state'] == 'missing':
                check(point['image'] is None and point['world'] is None, 'missing_has_coordinates')
            else:
                check(point['image'] is not None, 'present_without_coordinates')
            if point['state'] == 'interpolated':
                check(point['visibility'] is None and point['presence'] is None, 'interpolated_confidence')
        frame_map[index] = (ms, points)
    # An interpolated point needs directly observed endpoints within the configured gap.
    for i, f in enumerate(frames):
        for point in f['landmarks']:
            if point['state'] != 'interpolated':
                continue
            neighbors = []
            for candidates in (reversed(frames[:i]), iter(frames[i+1:])):
                endpoint = None
                for other in candidates:
                    match = next((q for q in other['landmarks'] if q['index'] == point['index']), None)
                    if match and match['state'] == 'observed' and match['image'] is not None:
                        endpoint = other, match
                        break
                neighbors.append(endpoint)
            bounded = all(neighbors)
            check(bounded, 'interpolation_unbounded')
            if bounded:
                (left, lp), (right, rp) = neighbors
                gap = right['timestamp_ms'] - left['timestamp_ms']
                check(0 < gap <= p['config']['interpolation_max_gap_ms'] + 1e-9, 'interpolation_gap')
                if gap > 0 and point['image'] is not None:
                    fraction = (f['timestamp_ms'] - left['timestamp_ms']) / gap
                    check(all(math.isclose(point['image'][axis], lp['image'][axis] + fraction *
                        (rp['image'][axis] - lp['image'][axis]), abs_tol=1e-9) for axis in ('x','y','z')),
                        'interpolation_coordinates')

    metric = r['metric_config']
    check(metric['reference_start_ms'] < metric['reference_end_ms'] <= m['duration_ms'], 'reference_window')
    expected_hand = {'orthodox':'right', 'southpaw':'left', 'unknown':'unknown'}[m['capture']['stance']]
    ids = [a['action_id'] for a in actions]
    check(len(ids) == len(set(ids)), 'duplicate_action')
    action_map = {a['action_id']:a for a in actions}
    valid = [a for a in actions if a['status'] == 'valid']
    previous_end = -1
    for a in actions:
        check(a['hand'] == expected_hand, 'rear_hand_mismatch')
        check((a['status'] == 'valid') == (not a['quality_reasons']), 'action_quality_reasons')
        times = []
        for event in EVENTS:
            ms, index = a[event+'_ms'], a[event+'_frame_index']
            check((ms is None) == (index is None), 'event_pair_missing')
            if ms is not None:
                times.append(ms)
                check(ms <= m['duration_ms'], 'event_out_of_bounds')
                check(index in frame_map and frame_map[index][0] == ms, 'event_frame_mismatch')
        check(all(a < b for a, b in zip(times, times[1:])), 'event_order')
        if a['status'] != 'valid':
            continue
        check(len(times) == 4, 'valid_event_missing')
        check(expected_hand != 'unknown', 'unknown_stance_gate')
        check(m['capture']['view'] in ('side','front_45'), 'view_gate')
        check(metric['scale_px'] is not None and metric['return_radius_ratio'] is not None, 'reference_scale_missing')
        if len(times) == 4:
            check(a['hold_end_ms'] - a['return_ms'] >= metric['return_hold_ms'], 'return_hold_too_short')
            check(a['start_ms'] >= metric['reference_end_ms'], 'action_before_reference')
            check(a['start_ms'] >= previous_end, 'valid_actions_overlap')
            previous_end = a['hold_end_ms']
        critical = (12,14,16) if a['hand'] == 'right' else (11,13,15)
        for event in EVENTS:
            entry = frame_map.get(a[event+'_frame_index'])
            if entry is None:
                continue
            for index in critical:
                point = entry[1].get(index)
                check(point is not None and point['state'] == 'observed' and point['image'] is not None
                    and 0 <= point['image']['x'] <= 1 and 0 <= point['image']['y'] <= 1
                    and point['visibility'] is not None and point['presence'] is not None
                    and point['visibility'] >= p['config']['min_visibility']
                    and point['presence'] >= p['config']['min_presence'], 'event_point_unusable')

    check(r['candidate_count'] == len(actions), 'candidate_count_mismatch')
    check(r['valid_count'] == len(valid), 'valid_count_mismatch')
    ratio = len(valid)/len(actions) if actions else None
    check(r['usable_ratio'] == ratio if ratio is None or r['usable_ratio'] is None else
          math.isclose(r['usable_ratio'], ratio, abs_tol=1e-9), 'usable_ratio_mismatch')
    seen = set()
    reported_actions = set()
    reported_metrics = set()
    for o in r['observations']:
        metric_id = o['metric_id']
        key = metric_id, tuple(sorted(o['action_ids']))
        check(key not in seen, 'duplicate_observation')
        seen.add(key)
        reported_metrics.add(metric_id)
        check(metric_id in r['enabled_metrics'], 'disabled_metric')
        check('not_coaching_assessment' in o['limitations'], 'missing_interpretation_limit')
        evidence = set()
        for e in o['evidence']:
            a = action_map.get(e['action_id'])
            check(e['action_id'] in o['action_ids'], 'unrelated_evidence')
            check(a is not None and a[e['event']+'_ms'] == e['timestamp_ms']
                and a[e['event']+'_frame_index'] == e['source_frame_index'], 'evidence_mismatch')
            evidence.add((e['action_id'], e['event']))
        selected = [action_map.get(i) for i in o['action_ids']]
        accepted = all(a is not None and a['status'] == 'valid' and all(a[e+'_ms'] is not None for e in EVENTS)
                       for a in selected)
        check(accepted, 'observation_action_unusable')
        for action_id in o['action_ids']:
            check(all((action_id, e) in evidence for e in ('extension','return','hold_end')), 'evidence_incomplete')
        if not accepted:
            continue
        values = [a['return_ms'] - a['extension_ms'] for a in selected]
        if metric_id == 'return_time_ms':
            check(len(values) == 1, 'return_metric_cardinality')
            check(math.isclose(o['value'], values[0], abs_tol=1e-9), 'return_value_mismatch')
            reported_actions.update(o['action_ids'])
        else:
            check(len(values) >= 5, 'repeatability_too_few_actions')
            check(set(o['action_ids']) == {a['action_id'] for a in valid}, 'repeatability_selection')
            check(math.isclose(o['median_ms'], _quantile(values,0.5), abs_tol=1e-9)
                and math.isclose(o['iqr_ms'], _quantile(values,0.75)-_quantile(values,0.25), abs_tol=1e-9),
                'repeatability_value_mismatch')
    if r['status'] == 'complete':
        check(len(valid) >= 5 and set(r['enabled_metrics']) == reported_metrics, 'incomplete_report')
        if 'return_time_ms' in r['enabled_metrics']:
            check(reported_actions == {a['action_id'] for a in valid}, 'incomplete_action_metrics')
    elif r['status'] == 'partial':
        check(bool(r['observations']) and bool(valid), 'empty_partial_report')
    else:
        check(not r['observations'], 'empty_status_has_observations')
        if r['status'] == 'rejected':
            check(not valid, 'rejected_has_valid_actions')
    if r['status'] != 'complete':
        check(bool(r['quality_reasons']), 'report_reason_required')
    return errors


def validate_dataset(bundles) -> list[str]:
    """Detect participant/lineage/content leakage across assigned evaluation splits."""
    errors = []
    splits = {}
    for bundle in bundles:
        found = validate_bundle(bundle)
        errors.extend(found)
        if found:
            continue
        m = bundle['manifest']
        for field in ('participant_id','lineage_id','input_sha256'):
            key = field, m[field]
            splits.setdefault(key,set()).add(m['split'])
    for (field,_), values in splits.items():
        if 'development' in values and 'holdout' in values:
            errors.append('split_leakage:'+field)
    return sorted(set(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files',nargs='*',type=Path,help='Local bundle JSON files; defaults to synthetic fixtures')
    args = parser.parse_args(argv)
    paths = args.files or sorted((ROOT/'tests/fixtures').glob('*.synthetic.json'))
    if not paths:
        print('FAIL: no input files')
        return 1
    bundles = []
    failed = False
    for path in paths:
        try:
            data = load_json(path)
        except (OSError, ValueError):
            print('FAIL: unreadable_or_invalid_json')
            failed = True
            continue
        if not args.files and (not isinstance(data, dict) or data.get('synthetic') is not True):
            print('FAIL: fixture_not_synthetic')
            failed = True
            continue
        bundles.append(data)
    errors = validate_dataset(bundles)
    for error in errors:
        print('FAIL: ' + error)
    if errors or failed:
        return 1
    print(f'PASS: {len(bundles)} contract bundles; structural and semantic checks only')
    return 0


if __name__ == '__main__':
    sys.exit(main())
