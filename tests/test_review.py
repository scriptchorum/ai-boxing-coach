import unittest
from boxing_coach.review import FILTERS, review_indices, neighbors

class ReviewTests(unittest.TestCase):
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
