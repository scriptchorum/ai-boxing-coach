import copy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

import numpy as np
from jsonschema import Draft202012Validator
from PIL import Image

from boxing_coach.evidence import (EvidenceError, MAX_DEMO_FRAMES, build_demo_report,
                                   build_evidence_bundle, read_evidence_bundle)


ROOT=Path(__file__).resolve().parents[1]


def fixture():
    frame={'source_frame_index':7,'timestamp_ms':233.333,'landmarks':[]}
    diagnostic={'source_frame_index':7,'engine_timestamp_ms':233,
                'detected_count_capped':1,'status':'single_model_pose'}
    run=SimpleNamespace(
        key='a'*64,start=7,end=7,mirror=False,frames=[frame],diagnostics=[diagnostic],
        model={'id':'pose_landmarker_lite','version':'1','asset_sha256':'b'*64},
        runtime={'mediapipe':'1.1.1','numpy':'2.5.4','av':'19.0.1'},
        config={'version':'pose-prototype-0.1','min_visibility':0.5,'min_presence':0.5},
    )
    decision={'show':True,'reasons':[],'allowed_indices':[11,12,23,24],
              'hidden_parts':['左臂','右臂','左腿','右腿']}
    source=np.arange(6*8*3,dtype=np.uint8).reshape(6,8,3)
    overlay=source.copy(); overlay[0,0]=[255,0,0]
    return run,frame,diagnostic,decision,source,overlay


def unpack(data):
    with ZipFile(BytesIO(data)) as archive:
        names=archive.namelist()
        files={name:archive.read(name) for name in names}
    return names,files,json.loads(files['manifest.json'])


def multi_bundles(*,run_key='a'*64):
    run,_,_,decision,source,overlay=fixture()
    frames=[{'source_frame_index':7,'timestamp_ms':233.333,'landmarks':[]},
            {'source_frame_index':8,'timestamp_ms':266.666,'landmarks':[]}]
    diagnostics=[{'source_frame_index':7,'engine_timestamp_ms':233,
                  'detected_count_capped':1,'status':'single_model_pose'},
                 {'source_frame_index':8,'engine_timestamp_ms':267,
                  'detected_count_capped':1,'status':'single_model_pose'}]
    run.key=run_key; run.start=7; run.end=8; run.frames=frames; run.diagnostics=diagnostics
    bundles=[]
    for position in range(2):
        current_source=source.copy(); current_source[0,1,0]=position
        current_overlay=overlay.copy(); current_overlay[0,1,0]=position
        bundles.append(build_evidence_bundle(
            run,frames[position],diagnostics[position],decision,current_source,current_overlay,
            source_timestamp_ms=frames[position]['timestamp_ms']))
    return bundles


class EvidenceBundleTests(unittest.TestCase):
    def test_bundle_is_deterministic_traceable_and_schema_valid(self):
        run,frame,diagnostic,decision,source,overlay=fixture()
        source_before,overlay_before=source.copy(),overlay.copy()
        first=build_evidence_bundle(run,frame,diagnostic,decision,source,overlay,
                                    source_timestamp_ms=233.333)
        second=build_evidence_bundle(run,frame,diagnostic,decision,source,overlay,
                                     source_timestamp_ms=233.333)
        self.assertEqual(first,second)
        names,files,manifest=unpack(first)
        self.assertEqual(names,['manifest.json','source.png','overlay.png'])
        schema=json.loads((ROOT/'schemas/evidence-frame.schema.json').read_text(encoding='utf-8'))
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(manifest)
        self.assertEqual(manifest['source_frame_index'],7)
        self.assertEqual(manifest['timestamp_ms'],233.333)
        self.assertEqual(manifest['display']['state'],'partial_display')
        self.assertEqual(manifest['files']['source']['sha256'],sha256(files['source.png']).hexdigest())
        self.assertEqual(manifest['files']['overlay']['sha256'],sha256(files['overlay.png']).hexdigest())
        np.testing.assert_array_equal(np.asarray(Image.open(BytesIO(files['source.png']))),source)
        np.testing.assert_array_equal(np.asarray(Image.open(BytesIO(files['overlay.png']))),overlay)
        np.testing.assert_array_equal(source,source_before)
        np.testing.assert_array_equal(overlay,overlay_before)
        serialized=json.dumps(manifest,ensure_ascii=False)
        for forbidden in ('path','landmarks','coordinates','action_score','verdict'):
            self.assertNotIn(forbidden,serialized)

    def test_hidden_model_result_is_preserved_without_action_claim(self):
        run,frame,diagnostic,decision,source,overlay=fixture()
        diagnostic['status']='no_pose'
        decision.update(show=False,reasons=['no_pose'],allowed_indices=[],
                        hidden_parts=['左臂','右臂','左腿','右腿'])
        _,_,manifest=unpack(build_evidence_bundle(
            run,frame,diagnostic,decision,source,overlay,source_timestamp_ms=233.333))
        self.assertEqual(manifest['model_status'],'no_pose')
        self.assertEqual(manifest['display']['state'],'hidden')
        self.assertEqual(manifest['display']['reason_codes'],['no_pose'])
        self.assertIn('display_rule_not_action_assessment',manifest['limitations'])

    def test_rejects_mismatched_frame_timestamp_images_and_decisions(self):
        run,frame,diagnostic,decision,source,overlay=fixture()
        calls=[
            (run,frame,diagnostic,decision,source,overlay,230),
            (run,frame,diagnostic,decision,source.astype(float),overlay,233.333),
            (run,frame,diagnostic,decision,source,overlay[:5],233.333),
        ]
        for args in calls:
            with self.subTest(timestamp=args[-1]),self.assertRaises(EvidenceError):
                build_evidence_bundle(*args[:-1],source_timestamp_ms=args[-1])

        bad_run=copy.deepcopy(run); bad_run.key='private-path'
        bad_decision={**decision,'reasons':['action_wrong']}
        bad_diagnostic={**diagnostic,'status':'multiple_poses'}
        for changed_run,changed_diagnostic,changed_decision in (
                (bad_run,diagnostic,decision),(run,diagnostic,bad_decision),
                (run,bad_diagnostic,decision)):
            with self.assertRaises(EvidenceError):
                build_evidence_bundle(changed_run,changed_run.frames[0],changed_diagnostic,
                                      changed_decision,source,overlay,source_timestamp_ms=233.333)

    def test_demo_report_sorts_frames_and_has_verifiable_offline_files(self):
        bundles=multi_bundles()
        first=build_demo_report(list(reversed(bundles)))
        second=build_demo_report(bundles)
        self.assertEqual(first,second)
        names,files,manifest=unpack(first)
        self.assertEqual(names[:3],['manifest.json','report.html','README.txt'])
        self.assertEqual(names[3:],[
            'frames/000/source.png','frames/000/overlay.png',
            'frames/001/source.png','frames/001/overlay.png'])
        schema=json.loads((ROOT/'schemas/demo-report.schema.json').read_text(encoding='utf-8'))
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(manifest)
        self.assertEqual(manifest['frame_count'],2)
        self.assertEqual([frame['source_frame_index'] for frame in manifest['frames']],[7,8])
        self.assertEqual(manifest['files']['report']['sha256'],sha256(files['report.html']).hexdigest())
        self.assertEqual(manifest['files']['readme']['sha256'],sha256(files['README.txt']).hexdigest())
        for frame in manifest['frames']:
            for image in frame['files'].values():
                self.assertEqual(image['sha256'],sha256(files[image['name']]).hexdigest())
        html=files['report.html'].decode('utf-8')
        self.assertIn('源帧 7',html)
        self.assertIn('源帧 8',html)
        self.assertNotIn('<script',html.lower())
        serialized=json.dumps(manifest,ensure_ascii=False)
        for forbidden in ('path','landmarks','coordinates','action_score','verdict'):
            self.assertNotIn(forbidden,serialized)

    def test_demo_report_rejects_duplicates_mixed_runs_and_tampering(self):
        first,second=multi_bundles()
        other=multi_bundles(run_key='c'*64)[0]
        for bundles in ([first,first],[first,other],[first]*(MAX_DEMO_FRAMES+1)):
            with self.subTest(count=len(bundles)),self.assertRaises(EvidenceError):
                build_demo_report(bundles)

        names,files,manifest=unpack(first)
        manifest['private_path']=r'C:\\private\\person.mp4'
        output=BytesIO()
        with ZipFile(output,'w') as archive:
            archive.writestr('manifest.json',json.dumps(manifest).encode())
            archive.writestr('source.png',files['source.png'])
            archive.writestr('overlay.png',files['overlay.png'])
        with self.assertRaises(EvidenceError): read_evidence_bundle(output.getvalue())

        damaged=bytearray(first); damaged[-8:]=b'not-a-zip'
        with self.assertRaises(EvidenceError): read_evidence_bundle(bytes(damaged))


if __name__ == '__main__':
    unittest.main()
