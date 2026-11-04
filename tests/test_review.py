import unittest
from boxing_coach.review import FILTERS, review_indices, neighbors

class ReviewTests(unittest.TestCase):
    def sample_run(self):
        from types import SimpleNamespace
        from test_quality import frame
        f=frame(0); f['source_frame_index']=50
        return SimpleNamespace(key='synthetic-run',start=50,end=50,mirror=False,frames=[f],diagnostics=[{'status':'single_model_pose'}])

    def test_review_file_roundtrip_and_matching_identity(self):
        from boxing_coach.review import export_reviews,import_reviews,VERDICTS
        r=self.sample_run(); notes={50:VERDICTS[1]}
        encoded=export_reviews(r,notes)
        self.assertEqual(import_reviews(r,encoded),notes)
        self.assertNotIn(b'landmarks',encoded)
        r.key='different-video-or-settings'
        with self.assertRaises(ValueError): import_reviews(r,encoded)

    def test_settings_preview_and_legacy_import(self):
        import json
        from boxing_coach.review import export_reviews,import_reviews,read_review_file
        run=self.sample_run()
        payload=read_review_file(export_reviews(run,{50:'无法判断'}))
        self.assertEqual(payload['settings'],{'start_frame':50,'end_frame':50,'mirror':False})
        self.assertEqual(set(payload),{'format','settings','run_key','quality_version','records'})
        payload.pop('settings'); payload['format']='boxing-display-review-1'
        old=json.dumps(payload).encode()
        self.assertNotIn('settings',read_review_file(old))
        self.assertEqual(import_reviews(run,old),{50:'无法判断'})

    def test_settings_reject_malformed_and_mismatched_hints(self):
        import json
        from boxing_coach.review import export_reviews,import_reviews,read_review_file
        run=self.sample_run(); payload=json.loads(export_reviews(run,{50:'无法判断'}))
        for changes in ({'start_frame':True},{'end_frame':49},{'end_frame':20000},
                        {'end_frame':1850},{'mirror':'false'},{'path':'private-video.mp4'},
                        {'start_frame':51,'end_frame':52}):
            altered={**payload,'settings':{**payload['settings'],**changes}}
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                read_review_file(json.dumps(altered).encode())
        for changes in ({'mirror':True},{'end_frame':51}):
            altered={**payload,'settings':{**payload['settings'],**changes}}
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                import_reviews(run,json.dumps(altered).encode())

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
        expected=[[51,52],[51],[52],[52],[50],[50,51,52],[]]
        for choice,indices in zip(FILTERS,expected):
            self.assertEqual(review_indices(frames,decisions,choice),indices)

    def test_first_display_after_multiple_candidates(self):
        from boxing_coach.review import REACQUIRED
        states=['shown','no_pose','shown','multiple_poses','multiple_poses','torso_missing',
                'partial','shown','multiple_poses','shown','multiple_poses']
        frames=[{'source_frame_index':100+i} for i in range(len(states))]
        decisions=[{'show':s in ('shown','partial'),'reasons':[s]} for s in states]
        self.assertEqual(review_indices(frames,decisions,REACQUIRED),[106,109])

    def test_neighbors_empty_edges_and_current_between_matches(self):
        self.assertEqual(neighbors([],7),(None,None))
        self.assertEqual(neighbors([10,20],10),(None,20))
        self.assertEqual(neighbors([10,20],15),(10,20))
        self.assertEqual(neighbors([10,20],20),(10,None))
        self.assertEqual(neighbors([10],10),(None,None))

    def test_review_summary_is_aggregate_and_flags_unrejected_issue(self):
        import json
        from boxing_coach.review import summarize_reviews
        run=self.sample_run()
        summary=summarize_reviews(run,{50:'跟错人或落在器械上'})
        self.assertEqual(summary['reviewed_frames'],1)
        self.assertEqual(summary['total_frames'],1)
        self.assertEqual(summary['problem_frames'],1)
        self.assertEqual(summary['display_counts'],{'完整显示':1,'部分显示':0,'全部隐藏':0})
        self.assertEqual(summary['issue_reason_counts'],{'no_display_rule_reason':1})
        encoded=json.dumps(summary,ensure_ascii=False)
        self.assertNotIn(run.key,encoded)
        self.assertNotIn('source_frame_index',encoded)
        self.assertNotIn('landmarks',encoded)

    def test_review_summary_rejects_invalid_session_state(self):
        from boxing_coach.review import summarize_reviews
        run=self.sample_run()
        for reviews in ({51:'无法判断'},{50:'动作优秀'},[]):
            with self.subTest(reviews=reviews),self.assertRaises(ValueError):
                summarize_reviews(run,reviews)
