import json
import unittest

from boxing_coach.pose import CONFIG, PoseRun, missing_landmarks
from boxing_coach.quality_gate import QualityGateError, clip_quality_report, metric_quality_gate


MODEL={'id':'pose_landmarker_lite','version':'1','asset_sha256':'a'*64}
RUNTIME={'mediapipe':'1.1.1','numpy':'2.5.4','av':'19.0.1'}


def landmarks(x=.5):
    return [{'index':index,'image':{'x':x,'y':.5,'z':0.0},'world':None,
             'visibility':.9,'presence':.9,'state':'observed'} for index in range(33)]


def pose_run(frames, statuses=None, *, config=None, key='synthetic-run'):
    statuses=statuses or ['single_model_pose']*len(frames)
    diagnostics=[{'source_frame_index':index,'engine_timestamp_ms':round(frame['timestamp_ms']),
                  'detected_count_capped':1 if status == 'single_model_pose' else 0,'status':status}
                 for index,(frame,status) in enumerate(zip(frames,statuses,strict=True))]
    return PoseRun(key,dict(MODEL),dict(RUNTIME),dict(CONFIG if config is None else config),
                   0,len(frames)-1,False,frames,diagnostics,1.0)


def frame(index, timestamp, x=.5, *, points=None):
    return {'source_frame_index':index,'timestamp_ms':timestamp,
            'landmarks':landmarks(x) if points is None else points}


class QualityGateTests(unittest.TestCase):
    def test_clip_report_counts_states_reasons_and_declared_blur(self):
        frames=[frame(0,0,.2),frame(1,33,.2,points=missing_landmarks()),
                frame(2,66,.2),frame(3,99,.9),frame(4,132,.9)]
        frames[4]['landmarks'][15]['visibility']=.1
        run=pose_run(frames,['single_model_pose','multiple_poses','single_model_pose',
                             'single_model_pose','single_model_pose'])
        report=clip_quality_report(run,declared_reasons=['declared_blur'])
        self.assertEqual(report['clip_status'],'rejected')
        self.assertEqual(report['display_state_frames'],{
            'full_display':2,'partial_display':1,'hidden':2,
        })
        self.assertEqual(report['frame_reason_counts']['multiple_model_pose_candidates'],1)
        self.assertEqual(report['frame_reason_counts']['torso_position_jump'],1)
        self.assertEqual(report['frame_reason_counts']['critical_landmark_unusable'],1)
        self.assertEqual(report['declared_clip_reasons'],['declared_blur'])
        self.assertEqual(report['interpolation']['applied_frame_count'],0)
        self.assertEqual(report['thresholds']['max_timeline_gap_ms'],100.0)

    def test_single_model_pose_requires_separate_single_person_confirmation(self):
        run=pose_run([frame(0,0)])
        pending=metric_quality_gate(run,'return_time_ms',[11,15],{'extension':0})
        self.assertEqual(pending['status'],'review_required')
        self.assertEqual(pending['value_policy'],'must_be_null')
        self.assertEqual(pending['review_reasons'],['single_person_not_confirmed'])
        eligible=metric_quality_gate(run,'return_time_ms',[11,15],{'extension':0},
                                     single_person_confirmed=True)
        self.assertEqual(eligible['status'],'eligible')
        self.assertEqual(eligible['value_policy'],'compute_later')

    def test_no_pose_all_hidden_clip_and_metric_reject(self):
        run=pose_run([frame(0,0,points=missing_landmarks())],['no_pose'])
        report=clip_quality_report(run)
        self.assertEqual(report['clip_status'],'rejected')
        self.assertEqual(report['frame_reason_counts']['no_model_pose'],1)
        gate=metric_quality_gate(run,'return_time_ms',[11,15],{'extension':0},
                                 single_person_confirmed=True)
        self.assertEqual(gate['reason_codes'],['no_model_pose','critical_landmark_missing'])
        self.assertEqual(gate['value_policy'],'must_be_null')

    def test_missing_event_multiple_candidate_and_declared_occlusion_reject(self):
        run=pose_run([frame(0,0,points=missing_landmarks())],['multiple_poses'])
        gate=metric_quality_gate(run,'return_time_ms',[11,15],
                                 {'extension':0,'return':None},single_person_confirmed=True,
                                 declared_reasons=['declared_occlusion'])
        self.assertEqual(gate['status'],'rejected')
        self.assertEqual(gate['value_policy'],'must_be_null')
        self.assertEqual(gate['reason_codes'],[
            'missing_event','multiple_model_pose_candidates','critical_landmark_missing',
            'declared_occlusion',
        ])

    def test_time_gap_and_explicit_multiple_people_reject(self):
        run=pose_run([frame(0,0),frame(1,200)])
        gate=metric_quality_gate(run,'return_time_ms',[11,15],{'return':1},
                                 single_person_confirmed=True,
                                 declared_reasons=['declared_multiple_people'])
        self.assertEqual(gate['status'],'rejected')
        self.assertEqual(gate['reason_codes'],['timestamp_gap','declared_multiple_people'])
        self.assertEqual(gate['value_policy'],'must_be_null')

    def test_metric_gate_separates_missing_low_signal_outside_and_interpolation(self):
        points=landmarks()
        points[11]['image']=None; points[11]['state']='missing'
        points[12]['visibility']=.1
        points[13]['image']['x']=1.2
        points[14]['state']='interpolated'
        run=pose_run([frame(0,0,points=points)])
        gate=metric_quality_gate(run,'synthetic_metric',[11,12,13,14],{'event':0},
                                 single_person_confirmed=True)
        self.assertEqual(gate['status'],'rejected')
        self.assertEqual(gate['reason_codes'],[
            'critical_landmark_missing','critical_landmark_low_signal',
            'critical_landmark_out_of_frame','interpolation_not_supported',
        ])
        report=clip_quality_report(run)
        self.assertEqual(report['interpolation']['applied_frame_count'],1)
        self.assertEqual(report['interpolation']['applied_landmark_count'],1)
        self.assertEqual(report['clip_status'],'rejected')
        configured=dict(CONFIG,interpolation_max_gap_ms=33)
        configured_gate=metric_quality_gate(
            pose_run([frame(0,0)],config=configured),'synthetic_metric',[11],{'event':0},
            single_person_confirmed=True,
        )
        self.assertIn('interpolation_not_supported',configured_gate['reason_codes'])
        self.assertEqual(configured_gate['interpolation']['policy'],'configured_but_not_implemented')

    def test_run_config_controls_display_diagnostics(self):
        config=dict(CONFIG,min_visibility=.95)
        report=clip_quality_report(pose_run([frame(0,0)],config=config))
        self.assertEqual(report['display_state_frames']['hidden'],1)
        self.assertEqual(report['frame_reason_counts']['critical_landmark_unusable'],1)

    def test_report_omits_run_key_paths_samples_frames_and_landmarks(self):
        report=clip_quality_report(pose_run([frame(0,0)],key=r'C:\private\person.mp4'))
        encoded=json.dumps(report,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()
        decoded=json.loads(encoded)
        self.assertNotIn(b'person.mp4',encoded)
        for forbidden in ('key','path','sample_id','frames','landmarks'):
            self.assertNotIn(forbidden,decoded)

    def test_rejects_invalid_arguments_and_malformed_run(self):
        run=pose_run([frame(0,0)])
        cases=(
            lambda:clip_quality_report(run,declared_reasons=['guessed_blur']),
            lambda:metric_quality_gate(run,'Bad Metric',[11],{'event':0}),
            lambda:metric_quality_gate(run,'metric',[],{'event':0}),
            lambda:metric_quality_gate(run,'metric',[11],{}),
            lambda:metric_quality_gate(run,'metric',[11],{'event':True}),
            lambda:metric_quality_gate(run,'metric',[11],{'event':0},single_person_confirmed='yes'),
            lambda:clip_quality_report(run,declared_reasons=None),
            lambda:metric_quality_gate(run,'metric',None,{'event':0}),
        )
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(QualityGateError):
                    case()
        run.diagnostics=[]
        with self.assertRaises(QualityGateError):
            clip_quality_report(run)
        malformed=pose_run([frame(0,0)])
        malformed.frames[0]['landmarks'][1]['index']=0
        with self.assertRaisesRegex(QualityGateError,'0–32'):
            clip_quality_report(malformed)


if __name__ == '__main__':
    unittest.main()
