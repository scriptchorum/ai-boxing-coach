from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from jsonschema import Draft202012Validator
from streamlit.testing.v1 import AppTest

from test_video_io import write_synthetic_video
from boxing_coach.pose import (CONFIG,PoseRun,PoseError,engine_timestamps,normalize_result,drawable,
                               draw_pose,run_key,model_identity,runtime_versions)
from boxing_coach.preview import load_preview

ROOT=Path(__file__).resolve().parents[1]


def raw_point(**changes):
    values={'x':0.5,'y':0.5,'z':0.1,'visibility':0.9,'presence':0.9}
    values.update(changes)
    return SimpleNamespace(**values)


def fixture_run(video,start=0,end=2,mirror=False):
    points,count,status=normalize_result(SimpleNamespace(pose_landmarks=[[raw_point() for _ in range(33)]]))
    frames=[{'source_frame_index':i,'timestamp_ms':video.inspection.timeline[i].timestamp_ms,'landmarks':points} for i in range(start,end+1)]
    diagnostics=[{'source_frame_index':i,'engine_timestamp_ms':round(video.inspection.timeline[i].timestamp_ms),'detected_count_capped':count,'status':status} for i in range(start,end+1)]
    return PoseRun(run_key(video,start,end,mirror),model_identity(),runtime_versions(),dict(CONFIG),start,end,mirror,frames,diagnostics,0.5)


class PoseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'synthetic.mp4'
        write_synthetic_video(self.path,[0,33,120],codec='libx264')
        self.video=load_preview(self.path)

    def test_no_pose_and_multiple_never_select_a_person(self):
        for poses,status in [([], 'no_pose'),([[raw_point()]*33]*2,'multiple_poses')]:
            points,count,actual=normalize_result(SimpleNamespace(pose_landmarks=poses))
            self.assertEqual(actual,status)
            self.assertEqual(count,len(poses))
            self.assertEqual(len(points),33)
            self.assertTrue(all(p['image'] is None and p['state']=='missing' for p in points))

    def test_low_confidence_out_of_frame_and_nonfinite_preserve_missing_evidence(self):
        points=[raw_point() for _ in range(33)]
        points[15]=raw_point(x=1.5)
        points[16]=raw_point(visibility=0.1)
        points[23]=raw_point(x=float('nan'))
        points[24]=raw_point(presence=None)
        result,_,_=normalize_result(SimpleNamespace(pose_landmarks=[points]))
        self.assertEqual(result[15]['image']['x'],1.5)
        self.assertFalse(drawable(result[15]))
        self.assertFalse(drawable(result[16]))
        self.assertEqual(result[23]['state'],'missing')
        self.assertIsNone(result[23]['image'])
        self.assertFalse(drawable(result[24]))
        self.assertTrue(drawable(result[11]))

    def test_pts_clock_adapter_refuses_collisions_and_keeps_fractional_source(self):
        timing=[SimpleNamespace(timestamp_ms=t) for t in [0,33.333333,120.123456]]
        self.assertEqual(engine_timestamps(timing),[0,33,120])
        self.assertEqual(timing[2].timestamp_ms,120.123456)
        with self.assertRaises(PoseError):
            engine_timestamps([SimpleNamespace(timestamp_ms=t) for t in [0,0.4]])

    def test_contract_projection_and_cache_invalidation(self):
        result=fixture_run(self.video)
        schema=json.loads((ROOT/'schemas/contracts.schema.json').read_text())
        validator=Draft202012Validator({'$ref':'#/$defs/PoseSequence','$defs':schema['$defs']})
        validator.validate(result.as_sequence('synthetic-pose'))
        self.assertNotEqual(result.key,run_key(self.video,1,2,False))
        self.assertNotEqual(result.key,run_key(self.video,0,2,True))
        with patch.dict(CONFIG,{'min_presence':0.7}):
            self.assertNotEqual(result.key,run_key(self.video,0,2,False))
        self.assertEqual(result.summary()['unavailable_joint_frames']['左腕'],0)

    def test_overlay_does_not_change_input_or_draw_missing_points(self):
        image=np.zeros((80,160,3),dtype=np.uint8)
        points,_,_=normalize_result(SimpleNamespace(pose_landmarks=[]))
        np.testing.assert_array_equal(draw_pose(image,{'landmarks':points}),image)
        points,_,_=normalize_result(SimpleNamespace(pose_landmarks=[[raw_point()]*33]))
        out=draw_pose(image,{'landmarks':points})
        self.assertTrue(out.any())
        self.assertFalse(image.any())

    def test_ui_explicit_run_and_selection_change_discards_old_result(self):
        app=AppTest.from_file(str(ROOT/'app/main.py'),default_timeout=30).run()
        app.text_input(key='input_path').set_value(str(self.path))
        app.button(key='FormSubmitter:open_video-打开视频').click().run()
        self.assertFalse(any(n.key=='pose_index' for n in app.number_input))
        with patch('boxing_coach.pose_ui.extract_pose',return_value=fixture_run(self.video)) as run:
            app.button(key='run_pose').click().run()
            run.assert_called_once()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.number_input(key='pose_index').value,0)
        app.number_input(key='start_frame').set_value(1).run()
        self.assertFalse(any(n.key=='pose_index' for n in app.number_input))
        self.assertEqual(len(app.exception),0)

    def test_ui_model_failure_does_not_leave_stale_results(self):
        app=AppTest.from_file(str(ROOT/'app/main.py'),default_timeout=30).run()
        app.text_input(key='input_path').set_value(str(self.path))
        app.button(key='FormSubmitter:open_video-打开视频').click().run()
        with patch('boxing_coach.pose_ui.extract_pose',side_effect=PoseError('synthetic failure')):
            app.button(key='run_pose').click().run()
        self.assertEqual(len(app.error),1)
        self.assertFalse(any(n.key=='pose_index' for n in app.number_input))
        self.assertEqual(len(app.exception),0)

    def test_player_payload_uses_source_pts_and_only_drawable_points(self):
        import struct
        from dataclasses import replace
        from boxing_coach.synced_player import player_payload
        run=fixture_run(self.video)
        run.frames[0]['landmarks'][15]['visibility']=0.1
        payload=player_payload(self.video,False,run)
        length=struct.unpack('<I',payload[1:5])[0]
        data=json.loads(payload[5:5+length])
        self.assertEqual(payload[0],2)
        self.assertEqual(payload[5+length:],self.video.content)
        self.assertEqual([f['time'] for f in data['frames']],[0,.033,.120])
        self.assertNotIn('15',data['frames'][0]['points'])
        self.assertNotIn('0',data['frames'][0]['points'])
        with self.assertRaises(ValueError): player_payload(self.video,True,run)
        shifted=replace(self.video,inspection=replace(self.video.inspection,original_pts_start_ms=100))
        with self.assertRaises(ValueError): player_payload(shifted,False,run)


if __name__=='__main__':
    unittest.main()
