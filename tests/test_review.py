import unittest
from boxing_coach.review import FILTERS, review_indices, neighbors

class ReviewTests(unittest.TestCase):
    def sample_run(self):
        from types import SimpleNamespace
        from test_quality import frame
        f=frame(0); f['source_frame_index']=50
        return SimpleNamespace(key='synthetic-run',frames=[f],diagnostics=[{'status':'single_model_pose'}])

    def test_review_file_roundtrip_and_matching_identity(self):
        from boxing_coach.review import export_reviews,import_reviews,VERDICTS
        r=self.sample_run(); notes={50:VERDICTS[1]}
        encoded=export_reviews(r,notes)
        self.assertEqual(import_reviews(r,encoded),notes)
        self.assertNotIn(b'landmarks',encoded)
        r.key='different-video-or-settings'
        with self.assertRaises(ValueError): import_reviews(r,encoded)

    def test_review_file_rejects_bad_size_schema_frames_and_rules(self):
        import json
        from boxing_coach.review import export_reviews,import_reviews,MAX_REVIEW_BYTES
        r=self.sample_run(); base=json.loads(export_reviews(r,{50:'无法判断'}))
        invalid=[b'not json',b'0'*(MAX_REVIEW_BYTES+1),b'[]',b'\xff']
        for change in [{'quality_version':'old'}, {'records':[{'frame':True,'verdict':'无法判断'}]},
                       {'records':[{'frame':51,'verdict':'无法判断'}]},
                       {'records':[{'frame':50,'verdict':'动作优秀'}]},
                       {'records':[base['records'][0]]*2}, {'records':None}]:
            invalid.append(json.dumps({**base,**change}).encode())
        for data in invalid:
            with self.subTest(data=data[:80]),self.assertRaises(ValueError): import_reviews(r,data)

    def test_hidden_frame_cannot_import_positive_display_verdict(self):
        from boxing_coach.review import export_reviews,import_reviews
        r=self.sample_run(); data=export_reviews(r,{50:'跟对人，已显示点大致对齐'})
        r.diagnostics[0]['status']='no_pose'
        with self.assertRaises(ValueError): import_reviews(r,data)
        self.assertEqual(import_reviews(r,export_reviews(r,{50:'无法判断'})),{50:'无法判断'})

    def test_filters_preserve_source_indices_and_include_full_frames(self):
        frames=[{'source_frame_index':i} for i in (50,51,52)]
        decisions=[{'show':True,'hidden_parts':[],'reasons':[]},
                   {'show':True,'hidden_parts':['左腿'],'reasons':['missing_points']},
                   {'show':False,'hidden_parts':['左腿'],'reasons':['position_jump']}]
        expected=[[51,52],[51],[52],[52],[50],[50,51,52]]
        for choice,indices in zip(FILTERS,expected):
            self.assertEqual(review_indices(frames,decisions,choice),indices)

    def test_neighbors_empty_edges_and_current_between_matches(self):
        self.assertEqual(neighbors([],7),(None,None))
        self.assertEqual(neighbors([10,20],10),(None,20))
        self.assertEqual(neighbors([10,20],15),(10,20))
        self.assertEqual(neighbors([10,20],20),(10,None))
        self.assertEqual(neighbors([10],10),(None,None))
