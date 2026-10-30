import json
import math
import unittest

from boxing_coach.pose import CONFIG, PoseRun, missing_landmarks
from boxing_coach.pose_batch import PoseBatchError, aggregate_pose_runs, batch_json


MODEL={'id':'pose_landmarker_lite','version':'1','asset_sha256':'a'*64}
RUNTIME={'mediapipe':'1.1.1','numpy':'2.5.4','av':'19.0.1'}


def visible_landmarks(visibility=.9):
    return [{'index':index,'image':{'x':.5,'y':.5,'z':0.0},'world':None,
             'visibility':visibility,'presence':.9,'state':'observed'} for index in range(33)]


def pose_run(key, statuses, elapsed=1.0, *, config=None):
    frames=[]
    diagnostics=[]
    for index,status in enumerate(statuses):
        landmarks=visible_landmarks() if status == 'single_model_pose' else missing_landmarks()
        frames.append({'source_frame_index':index,'timestamp_ms':index*40.0,'landmarks':landmarks})
        diagnostics.append({'source_frame_index':index,'engine_timestamp_ms':index*40,
                            'detected_count_capped':1 if status == 'single_model_pose' else 0,
                            'status':status})
    return PoseRun(key,dict(MODEL),dict(RUNTIME),dict(CONFIG if config is None else config),
                   0,len(frames)-1,False,frames,diagnostics,elapsed)


class PoseBatchTests(unittest.TestCase):
    def test_aggregate_counts_timing_and_fixed_identity(self):
        first=pose_run('run-a',['single_model_pose','no_pose'],1.0)
        second=pose_run('run-b',['multiple_poses','single_model_pose'],3.0)
        report=aggregate_pose_runs([first,second])
        self.assertEqual(report['run_count'],2)
        self.assertEqual(report['frame_count'],4)
        self.assertEqual(report['status_frames'],{
            'no_pose':1,'multiple_poses':1,'single_model_pose':2,
        })
        self.assertTrue(all(count == 2 for count in report['unavailable_joint_frames'].values()))
        self.assertEqual(report['timing_seconds'],{
            'total':4.0,'median_per_run':2.0,'p95_per_run_linear':2.9,
            'aggregate_frames_per_second':1.0,
        })
        self.assertEqual(report['model'],MODEL)
        self.assertEqual(report['config']['version'],CONFIG['version'])

    def test_export_omits_run_keys_paths_samples_and_landmarks(self):
        report=aggregate_pose_runs([pose_run(r'C:\private\alice-video.mp4',['single_model_pose'])])
        encoded=batch_json(report)
        decoded=json.loads(encoded)
        self.assertNotIn(b'alice-video',encoded)
        self.assertNotIn('key',decoded)
        self.assertNotIn('frames',decoded)
        self.assertNotIn('sample_id',decoded)
        self.assertIn('frame_count',decoded)

    def test_run_config_controls_unavailable_threshold(self):
        config=dict(CONFIG,min_visibility=.95)
        run=pose_run('strict',['single_model_pose'],config=config)
        self.assertTrue(all(value == 1 for value in aggregate_pose_runs([run])['unavailable_joint_frames'].values()))

    def test_rejects_empty_duplicates_and_mixed_identity(self):
        with self.assertRaises(PoseBatchError):
            aggregate_pose_runs([])
        with self.assertRaisesRegex(PoseBatchError,'重复'):
            aggregate_pose_runs([pose_run('same',['no_pose']),pose_run('same',['no_pose'])])
        for field,value in [('model',{'version':'2'}),('runtime',{'av':'20'}),
                            ('config',{'min_presence':.8})]:
            with self.subTest(field=field):
                first=pose_run('one',['no_pose'])
                second=pose_run('two',['no_pose'])
                getattr(second,field).update(value)
                with self.assertRaisesRegex(PoseBatchError,'混用'):
                    aggregate_pose_runs([first,second])

    def test_rejects_incomplete_unknown_and_invalid_timing(self):
        incomplete=pose_run('incomplete',['no_pose'])
        incomplete.diagnostics=[]
        unknown=pose_run('unknown',['no_pose'])
        unknown.diagnostics[0]['status']='guessed_pose'
        reordered=pose_run('reordered',['no_pose'])
        reordered.frames[0]['source_frame_index']=1
        for run in (incomplete,unknown,pose_run('negative',['no_pose'],-1),
                    pose_run('infinite',['no_pose'],math.inf),pose_run('boolean',['no_pose'],True),reordered):
            with self.subTest(key=run.key):
                with self.assertRaises(PoseBatchError):
                    aggregate_pose_runs([run])


if __name__ == '__main__':
    unittest.main()
