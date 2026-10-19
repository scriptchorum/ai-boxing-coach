import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.validate_contracts import ROOT, load_json, validate_bundle, validate_dataset


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.good = load_json(ROOT/'tests/fixtures/complete.synthetic.json')

    def assert_bad(self, data, code):
        self.assertIn(code, validate_bundle(data))

    def test_committed_synthetic_fixtures(self):
        paths = sorted((ROOT/'tests/fixtures').glob('*.synthetic.json'))
        self.assertEqual(len(paths),3)
        for path in paths:
            with self.subTest(path=path.name):
                data = load_json(path)
                self.assertIs(data['synthetic'],True)
                self.assertEqual(validate_bundle(data),[])

    def test_mirror_does_not_change_anatomical_hand(self):
        for section in ('manifest','report'):
            self.good[section]['capture']['is_mirrored'] = True
        self.assertEqual(validate_bundle(self.good),[])
        self.good['actions'][0]['hand'] = 'left'
        self.assert_bad(self.good,'rear_hand_mismatch')

    def test_southpaw_rear_is_left(self):
        for section in ('manifest','report'):
            self.good[section]['capture']['stance'] = 'southpaw'
        for action in self.good['actions']:
            action['hand'] = 'left'
        self.assertEqual(validate_bundle(self.good),[])

    def test_vfr_and_sparse_source_frame_mapping(self):
        self.good['manifest']['media']['nominal_fps'] = None
        self.good['pose']['frames'][1]['timestamp_ms'] = 997.125
        self.assertEqual(validate_bundle(self.good),[])

    def test_out_of_frame_coordinates_are_representable(self):
        self.good['pose']['frames'][0]['landmarks'][0]['image']['x'] = -0.1
        self.assertEqual(validate_bundle(self.good),[])

    def test_schema_rejects_unsupported_claims_and_versions(self):
        mutations = [
            lambda b:b['manifest'].update(real_name='synthetic-name'),
            lambda b:b['report'].update(technique_score=90),
            lambda b:b['report'].update(interpretation='injury_prediction'),
            lambda b:b['pose'].update(schema_version='9.0.0'),
            lambda b:b['report']['observations'][0].update(unit='seconds'),
            lambda b:b['report']['observations'][0].update(value=True),
            lambda b:b['pose']['frames'][0]['landmarks'][0].update(visibility=2),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                data=copy.deepcopy(self.good)
                mutate(data)
                self.assertTrue(any(e.startswith('schema:') for e in validate_bundle(data)))

    def test_cross_record_consistency(self):
        mutations = [
            ('sample_mismatch',lambda b:b['pose'].update(sample_id='other-sample')),
            ('input_hash_mismatch',lambda b:b['report'].update(input_sha256='1'*64)),
            ('model_mismatch',lambda b:b['report']['model'].update(version='changed')),
            ('capture_mismatch',lambda b:b['report']['capture'].update(view='front_45')),
            ('candidate_count_mismatch',lambda b:b['report'].update(candidate_count=7)),
            ('valid_count_mismatch',lambda b:b['report'].update(valid_count=2)),
            ('usable_ratio_mismatch',lambda b:b['report'].update(usable_ratio=0.5)),
            ('invalid_generated_at',lambda b:b['report'].update(generated_at='2026-02-30T00:00:00Z')),
            ('duplicate_action',lambda b:b['actions'].append(copy.deepcopy(b['actions'][0]))),
        ]
        for code,mutate in mutations:
            with self.subTest(code=code):
                data=copy.deepcopy(self.good)
                mutate(data)
                self.assert_bad(data,code)

    def test_bad_timeline_is_rejected(self):
        mutations = [
            ('non_monotonic_frames',lambda b:b['pose']['frames'][1].update(timestamp_ms=0)),
            ('non_monotonic_frames',lambda b:b['pose']['frames'][1].update(source_frame_index=0)),
            ('frame_out_of_bounds',lambda b:b['pose']['frames'][-1].update(timestamp_ms=20000)),
            ('event_frame_mismatch',lambda b:b['actions'][0].update(extension_ms=2201)),
            ('event_order',lambda b:b['actions'][0].update(return_ms=2100)),
            ('return_hold_too_short',lambda b:b['actions'][0].update(hold_end_ms=2550)),
            ('event_pair_missing',lambda b:b['actions'][0].update(return_ms=None)),
            ('valid_event_missing',lambda b:b['actions'][0].update(return_ms=None,return_frame_index=None)),
            ('reference_window',lambda b:b['report']['metric_config'].update(reference_end_ms=0)),
        ]
        for code,mutate in mutations:
            with self.subTest(code=code):
                data=copy.deepcopy(self.good)
                mutate(data)
                self.assert_bad(data,code)

    def test_missing_event_cannot_be_zero_or_receive_metric(self):
        data=load_json(ROOT/'tests/fixtures/missing.synthetic.json')
        data['report']['observations']=[copy.deepcopy(self.good['report']['observations'][0])]
        self.assert_bad(data,'observation_action_unusable')
        data['actions'][0].update(return_ms=0,return_frame_index=0)
        self.assert_bad(data,'event_order')

    def test_event_requires_observed_critical_points(self):
        frame=self.good['pose']['frames'][3]  # first extension
        for state in ('missing','interpolated'):
            with self.subTest(state=state):
                data=copy.deepcopy(self.good)
                point=next(p for p in data['pose']['frames'][3]['landmarks'] if p['index']==16)
                point.update(state=state,visibility=None,presence=None)
                if state=='missing':
                    point.update(image=None,world=None)
                self.assert_bad(data,'event_point_unusable')
        point=next(p for p in frame['landmarks'] if p['index']==16)
        point['visibility']=0.1
        self.assert_bad(self.good,'event_point_unusable')
        point['visibility']=0.99
        point['image']['x']=-0.1
        self.assert_bad(self.good,'event_point_unusable')

    def test_absent_or_duplicate_landmark_indices(self):
        self.good['pose']['frames'][3]['landmarks']=[p for p in self.good['pose']['frames'][3]['landmarks'] if p['index']!=16]
        self.assert_bad(self.good,'event_point_unusable')
        self.good['pose']['frames'][0]['landmarks'][1]['index']=11
        self.assert_bad(self.good,'duplicate_landmark')

    def test_bounded_non_event_interpolation(self):
        base=copy.deepcopy(self.good['pose']['frames'][0])
        middle=copy.deepcopy(base)
        right=copy.deepcopy(base)
        middle.update(source_frame_index=1,timestamp_ms=100/3)
        right.update(source_frame_index=2,timestamp_ms=200/3)
        middle['landmarks'][0].update(state='interpolated',visibility=None,presence=None)
        self.good['pose']['frames'][1:1]=[middle,right]
        self.assertEqual(validate_bundle(self.good),[])
        right['timestamp_ms']=100
        self.assert_bad(self.good,'interpolation_gap')
        right['timestamp_ms']=200/3
        middle['landmarks'][0]['image']['x']=999
        self.assert_bad(self.good,'interpolation_coordinates')

    def test_unbounded_interpolation_is_rejected(self):
        point=self.good['pose']['frames'][0]['landmarks'][0]
        point.update(state='interpolated',visibility=None,presence=None)
        self.assert_bad(self.good,'interpolation_unbounded')

    def test_evidence_and_metric_values(self):
        mutations=[
            ('return_value_mismatch',lambda b:b['report']['observations'][0].update(value=999)),
            ('evidence_mismatch',lambda b:b['report']['observations'][0]['evidence'][0].update(timestamp_ms=999)),
            ('evidence_incomplete',lambda b:b['report']['observations'][0]['evidence'].pop()),
            ('repeatability_value_mismatch',lambda b:b['report']['observations'][-1].update(iqr_ms=12)),
            ('repeatability_too_few_actions',lambda b:b['report']['observations'][-1]['action_ids'].pop()),
            ('missing_interpretation_limit',lambda b:b['report']['observations'][0].update(limitations=['projection_only'])),
            ('incomplete_report',lambda b:b['report']['observations'].pop()),
            ('duplicate_observation',lambda b:b['report']['observations'].append(copy.deepcopy(b['report']['observations'][0]))),
        ]
        for code,mutate in mutations:
            with self.subTest(code=code):
                data=copy.deepcopy(self.good)
                mutate(data)
                self.assert_bad(data,code)

    def test_unsupported_view_and_unknown_stance(self):
        for view in ('unsupported','unknown'):
            for section in ('manifest','report'):
                self.good[section]['capture']['view']=view
            self.assert_bad(self.good,'view_gate')
        for section in ('manifest','report'):
            self.good[section]['capture']['stance']='unknown'
        for a in self.good['actions']:
            a['hand']='unknown'
        self.assert_bad(self.good,'unknown_stance_gate')

    def test_consent_gate(self):
        for status in ('not_recorded','withdrawn'):
            self.good['manifest']['consent']['status']=status
            self.assert_bad(self.good,'consent_gate')
            self.assert_bad(self.good,'consent_processing_forbidden')

    def test_participant_and_transformation_split_leakage(self):
        other=copy.deepcopy(self.good)
        other['manifest']['split']='holdout'
        for field in ('participant_id','lineage_id','input_sha256'):
            self.assertIn('split_leakage:'+field,validate_dataset([self.good,other]))
        other['manifest'].update(participant_id='synthetic-person-02',lineage_id='synthetic-lineage-02',input_sha256='1'*64)
        other['report']['input_sha256']='1'*64
        self.assertEqual(validate_dataset([self.good,other]),[])

    def test_nonfinite_numbers_and_duplicate_json_keys(self):
        self.good['report']['observations'][0]['value']=float('nan')
        self.assert_bad(self.good,'non_finite_number')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.json'
            for content in ('{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'):
                path.write_text(content,encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_json(path)
            path.write_text('{"a":1e999}',encoding='utf-8')
            self.assert_bad(load_json(path),'non_finite_number')

    def test_cli_success_and_failure_without_echoing_payload(self):
        good=subprocess.run([sys.executable,str(ROOT/'tools/validate_contracts.py')],capture_output=True,text=True)
        self.assertEqual(good.returncode,0,good.stdout+good.stderr)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.json'
            self.good['manifest']['real_name']='do-not-echo-this-value'
            path.write_text(json.dumps(self.good),encoding='utf-8')
            bad=subprocess.run([sys.executable,str(ROOT/'tools/validate_contracts.py'),str(path)],capture_output=True,text=True)
            self.assertEqual(bad.returncode,1)
            self.assertNotIn('do-not-echo-this-value',bad.stdout+bad.stderr)
            path.write_text('[]',encoding='utf-8')
            bad=subprocess.run([sys.executable,str(ROOT/'tools/validate_contracts.py'),str(path)],capture_output=True,text=True)
            self.assertEqual(bad.returncode,1)


if __name__=='__main__':
    unittest.main()
