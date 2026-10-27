from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from streamlit.testing.v1 import AppTest

from test_video_io import write_synthetic_video
from boxing_coach.preview import load_preview, read_preview_frame, selection_times
from boxing_coach.video_io import MediaInputError


ROOT = Path(__file__).resolve().parents[1]


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'synthetic.mp4'
        write_synthetic_video(self.path, [0, 33, 67, 120, 153], codec='libx264')

    def test_vfr_frame_and_inclusive_selection_preserve_pts(self):
        video = load_preview(self.path)
        self.assertTrue(video.inspection.variable_frame_rate)
        frame = read_preview_frame(video, 3)
        self.assertAlmostEqual(frame.timestamp_ms, 120, delta=0.01)
        begin, end = selection_times(video.inspection, 1, 2)
        self.assertAlmostEqual(begin, 33, delta=0.01)
        self.assertAlmostEqual(end, 120, delta=0.01)
        self.assertEqual(selection_times(video.inspection, 4, 4)[1], video.inspection.duration_ms)
        flipped = read_preview_frame(video, 3, correct_mirror=True)
        np.testing.assert_array_equal(flipped.image_rgb, np.fliplr(frame.image_rgb))
        for pair in [(-1, 2), (3, 2), (0, 5)]:
            with self.assertRaises(ValueError):
                selection_times(video.inspection, *pair)

    def test_restricted_formats_reject_with_stable_codes(self):
        for filename, kwargs, code in [
            ('synthetic.mov', {'codec': 'libx264'}, 'preview_format'),
            ('other-codec.mp4', {'codec': 'mpeg4'}, 'preview_format'),
            ('rotated.mp4', {'codec': 'libx264', 'rotation': 90}, 'preview_transform'),
            ('reflected.mp4', {'codec': 'libx264', 'horizontal_flip': True}, 'preview_transform'),
        ]:
            with self.subTest(filename=filename):
                path = Path(self.tmp.name) / filename
                write_synthetic_video(path, [0, 33, 67], **kwargs)
                with self.assertRaises(MediaInputError) as caught:
                    load_preview(path)
                self.assertEqual(caught.exception.code, code)

    def test_changed_source_rejected_before_frame_display(self):
        video = load_preview(self.path)
        self.path.write_bytes(b'invalid synthetic replacement')
        with self.assertRaises(MediaInputError) as caught:
            read_preview_frame(video, 0)
        self.assertEqual(caught.exception.code, 'input_changed')

    def open_app(self):
        app = AppTest.from_file(str(ROOT / 'app/main.py'), default_timeout=30).run()
        app.text_input(key='input_path').set_value(str(self.path))
        app.button(key='FormSubmitter:open_video-打开视频').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        return app

    def test_ui_navigation_selection_and_close(self):
        app = self.open_app()
        app.number_input(key='frame_index').set_value(3).run()
        self.assertEqual(app.number_input(key='frame_index').value, 3)
        next(b for b in app.button if b.label == '下一帧').click().run()
        self.assertEqual(app.number_input(key='frame_index').value, 4)
        self.assertTrue(next(b for b in app.button if b.label == '下一帧').disabled)
        app.number_input(key='start_frame').set_value(4).run()
        app.number_input(key='end_frame').set_value(1).run()
        self.assertEqual(len(app.error), 1)
        next(b for b in app.button if b.label == '关闭当前视频').click().run()
        self.assertEqual(len(app.number_input), 0)
        self.assertEqual(len(app.exception), 0)

    def test_ui_failed_open_clears_previous_video_and_indices(self):
        app = self.open_app()
        app.number_input(key='frame_index').set_value(4).run()
        app.text_input(key='input_path').set_value(str(Path(self.tmp.name) / 'missing.mp4'))
        app.button(key='FormSubmitter:open_video-打开视频').click().run()
        self.assertEqual(len(app.error), 1)
        self.assertEqual(len(app.number_input), 0)
        self.assertEqual(len(app.get('video')), 0)
        app.text_input(key='input_path').set_value(str(self.path))
        app.button(key='FormSubmitter:open_video-打开视频').click().run()
        self.assertEqual(app.number_input(key='frame_index').value, 0)
        self.assertEqual(len(app.exception), 0)

    def test_ui_single_frame_video_and_replacement(self):
        write_synthetic_video(self.path, [0], codec='libx264')
        app = self.open_app()
        self.assertEqual(app.number_input(key='end_frame').value, 0)
        self.path.write_bytes(b'synthetic replacement')
        app.checkbox(key='mirror').set_value(True).run()
        self.assertEqual(len(app.error), 1)
        self.assertEqual(len(app.get('video')), 0)
        self.assertEqual(len(app.exception), 0)


if __name__ == '__main__':
    unittest.main()
