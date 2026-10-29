import copy
import unittest
from types import SimpleNamespace
from boxing_coach.quality import assess_frames

def frame(t,x=.5):
    return {'timestamp_ms':t,'landmarks':[{'index':i,'state':'observed','image':{'x':x,'y':.5,'z':0},'visibility':.9,'presence':.9} for i in range(33)]}

def run(frames,statuses=None):
    return SimpleNamespace(frames=frames,diagnostics=[{'status':s} for s in (statuses or ['single_model_pose']*len(frames))])

class QualityTests(unittest.TestCase):
    def test_missing_and_outside_reject_without_mutation(self):
        f=frame(0); f['landmarks'][15]['image']['x']=1.1
        r=run([f]); before=copy.deepcopy(r)
        self.assertEqual(assess_frames(r)[0]['reasons'],['out_of_frame','missing_points'])
        self.assertEqual(r.frames,before.frames)
        f['landmarks'][15]['image']=None
        self.assertEqual(assess_frames(r)[0]['reasons'],['missing_points'])

    def test_time_based_jump_and_gap(self):
        result=assess_frames(run([frame(0,.2),frame(33,.4),frame(66,.4),frame(200,.4)]))
        self.assertEqual([r['show'] for r in result],[True,False,True,False])
        self.assertEqual(result[1]['reasons'],['position_jump'])
        self.assertEqual(result[3]['reasons'],['time_gap'])
        self.assertTrue(assess_frames(run([frame(0,.2),frame(100,.4)]))[1]['show'])

    def test_no_and_multiple_reset_comparison(self):
        r=assess_frames(run([frame(0,.2),frame(33),frame(66,.8)],['single_model_pose','multiple_poses','single_model_pose']))
        self.assertEqual(r[1]['reasons'],['multiple_poses'])
        self.assertTrue(r[2]['show'])
        self.assertEqual(assess_frames(run([frame(0)],['no_pose']))[0]['reasons'],['no_pose'])

    def test_mirror_invariance_and_no_body_proportion_rule(self):
        frames=[frame(0,.3),frame(33,.31)]
        original=assess_frames(run(frames))
        for f in frames:
            for p in f['landmarks']: p['image']['x']=1-p['image']['x']
        self.assertEqual(assess_frames(run(frames)),original)

    def test_filtered_payload_contains_no_rejected_points(self):
        import struct,json
        from test_pose import PoseTests,fixture_run
        from boxing_coach.synced_player import player_payload
        case=PoseTests(); case.setUp()
        try:
            result=fixture_run(case.video)
            result.frames[0]['landmarks'][11]['presence']=.1
            payload=player_payload(case.video,False,result)
            length=struct.unpack('<I',payload[1:5])[0]
            f=json.loads(payload[5:5+length])['frames'][0]
            self.assertEqual(f['status'],'quality_rejected')
            self.assertEqual(f['points'],{})
            self.assertTrue(f['reason'])
        finally: case.doCleanups()

    def test_limb_missing_keeps_other_limbs_and_static_payload_projection_agrees(self):
        from boxing_coach.quality import display_frame
        f=frame(0); f['landmarks'][27]['visibility']=.1
        before=copy.deepcopy(f)
        d=assess_frames(run([f]))[0]
        self.assertTrue(d['show']); self.assertEqual(d['hidden_parts'],['左腿'])
        self.assertIn(15,d['allowed_indices']); self.assertIn(28,d['allowed_indices'])
        self.assertNotIn(25,d['allowed_indices']); self.assertNotIn(27,d['allowed_indices'])
        self.assertEqual({p['index'] for p in display_frame(f,d)['landmarks']},set(d['allowed_indices']))
        self.assertEqual(f,before)
        f['landmarks'][23]['visibility']=.1
        self.assertFalse(assess_frames(run([f]))[0]['show'])
