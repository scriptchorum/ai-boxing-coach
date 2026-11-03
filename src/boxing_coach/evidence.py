"""Build private local evidence-frame bundles without action assessment."""
from __future__ import annotations

from hashlib import sha256
from html import escape
from io import BytesIO
import json
import math
import re
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from jsonschema import Draft202012Validator
from PIL import Image

from .model_assets import find_project_root
from .pose import JOINTS
from .quality import LABELS, LIMBS, VERSION


FORMAT = 'boxing-evidence-frame-1'
LIMITATIONS = [
    'private_local_export_contains_video_pixels',
    'model_estimate_not_ground_truth',
    'display_rule_not_action_assessment',
    'no_accuracy_or_coaching_claim',
]
MODEL_STATUSES = {'no_pose','multiple_poses','single_model_pose'}
MAX_EVIDENCE_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_EVIDENCE_IMAGE_BYTES = 32 * 1024 * 1024
MAX_DEMO_FRAMES = 8
MAX_DEMO_SOURCE_BYTES = 96 * 1024 * 1024


def _validator(name):
    schema=json.loads((find_project_root()/'schemas'/name).read_text(encoding='utf-8'))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


EVIDENCE_VALIDATOR=_validator('evidence-frame.schema.json')
DEMO_VALIDATOR=_validator('demo-report.schema.json')


class EvidenceError(ValueError):
    pass


def _selected(mapping, fields, label):
    if not isinstance(mapping,dict) or any(field not in mapping for field in fields):
        raise EvidenceError(f'{label} metadata is incomplete')
    return {field:mapping[field] for field in fields}


def _png(image):
    if (not isinstance(image,np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or
            image.shape[2] != 3 or not 1 <= image.shape[0] <= 8192 or
            not 1 <= image.shape[1] <= 8192):
        raise EvidenceError('Evidence images must be bounded uint8 RGB arrays')
    output=BytesIO()
    Image.fromarray(image).save(output,format='PNG',optimize=False)
    return output.getvalue()


def _write(archive,name,data):
    entry=ZipInfo(name,date_time=(1980,1,1,0,0,0))
    entry.compress_type=ZIP_DEFLATED
    entry.external_attr=0o600 << 16
    archive.writestr(entry,data)


def _json(data):
    def reject_constant(_):
        raise ValueError('Non-finite JSON number')
    def unique_keys(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    return json.loads(data.decode('utf-8'),parse_constant=reject_constant,
                      object_pairs_hook=unique_keys)


def _contract(validator,data,label):
    if next(validator.iter_errors(data),None) is not None:
        raise EvidenceError(f'{label} manifest does not match its contract')


def read_evidence_bundle(data) -> dict:
    """Validate a bounded private frame bundle before report aggregation."""
    if type(data) is not bytes or not 0 < len(data) <= MAX_EVIDENCE_BUNDLE_BYTES:
        raise EvidenceError('Evidence bundle size is invalid')
    try:
        with ZipFile(BytesIO(data)) as archive:
            entries=archive.infolist()
            if ([entry.filename for entry in entries] !=
                    ['manifest.json','source.png','overlay.png'] or
                    any(entry.flag_bits & 1 or entry.is_dir() for entry in entries) or
                    entries[0].file_size > 64 * 1024 or
                    any(entry.file_size > MAX_EVIDENCE_IMAGE_BYTES for entry in entries[1:])):
                raise EvidenceError('Evidence bundle entries are invalid')
            files={entry.filename:archive.read(entry) for entry in entries}
    except (BadZipFile,OSError,RuntimeError,EOFError) as exc:
        raise EvidenceError('Evidence bundle is not a readable ZIP') from exc
    try:
        manifest=_json(files['manifest.json'])
    except (ValueError,UnicodeError,RecursionError) as exc:
        raise EvidenceError('Evidence manifest is not valid UTF-8 JSON') from exc
    _contract(EVIDENCE_VALIDATOR,manifest,'Evidence')
    if (not manifest['selection']['start_frame'] <= manifest['source_frame_index'] <=
            manifest['selection']['end_frame'] or
            manifest['files']['source']['sha256'] != sha256(files['source.png']).hexdigest() or
            manifest['files']['overlay']['sha256'] != sha256(files['overlay.png']).hexdigest()):
        raise EvidenceError('Evidence bundle identity or hashes do not match')
    display=manifest['display']
    status=manifest['model_status']
    reasons=set(display['reason_codes'])
    semantic=(
        display['state'] == ('hidden' if display['allowed_joint_count'] == 0 else
                             ('partial_display' if display['hidden_parts'] else 'full_display')) and
        (status == 'single_model_pose' or display['state'] == 'hidden') and
        ((status == 'single_model_pose' and not reasons & {'no_pose','multiple_poses'}) or
         (status in {'no_pose','multiple_poses'} and status in reasons)) and
        (display['state'] != 'hidden' or set(display['hidden_parts']) == set(LIMBS)) and
        (display['state'] != 'full_display' or
         (display['allowed_joint_count'] == len(JOINTS) and not reasons))
    )
    if not semantic:
        raise EvidenceError('Evidence display state is inconsistent')
    for name in ('source.png','overlay.png'):
        try:
            with Image.open(BytesIO(files[name])) as image:
                image.verify()
            with Image.open(BytesIO(files[name])) as image:
                if (image.format != 'PNG' or image.mode != 'RGB' or
                        image.size != (manifest['image_size']['width'],
                                       manifest['image_size']['height'])):
                    raise EvidenceError('Evidence PNG metadata does not match')
        except (OSError,ValueError) as exc:
            raise EvidenceError('Evidence PNG is invalid') from exc
    return {'manifest':manifest,'source_png':files['source.png'],
            'overlay_png':files['overlay.png']}


def build_evidence_bundle(run, frame, diagnostic, decision, source_rgb, overlay_rgb, *,
                          source_timestamp_ms) -> bytes:
    """Return a deterministic local-only ZIP for one already decoded source frame."""
    if (not isinstance(getattr(run,'key',None),str) or
            re.fullmatch(r'[0-9a-f]{64}',run.key) is None):
        raise EvidenceError('Run identity is invalid')
    if (type(getattr(run,'start',None)) is not int or type(getattr(run,'end',None)) is not int or
            type(getattr(run,'mirror',None)) is not bool or not 0 <= run.start <= run.end < 20000 or
            run.end-run.start+1 > 1800 or len(run.frames) != run.end-run.start+1 or
            len(run.frames) != len(run.diagnostics)):
        raise EvidenceError('Run selection is invalid')
    if not isinstance(frame,dict) or type(frame.get('source_frame_index')) is not int:
        raise EvidenceError('Source frame identity is invalid')
    index=frame['source_frame_index']
    positions=[position for position,item in enumerate(run.frames)
               if isinstance(item,dict) and item.get('source_frame_index') == index]
    if (len(positions) != 1 or not run.start <= index <= run.end or
            run.frames[positions[0]] != frame or run.diagnostics[positions[0]] != diagnostic):
        raise EvidenceError('Frame does not match this pose run')
    timestamp=frame.get('timestamp_ms')
    if (isinstance(timestamp,bool) or not isinstance(timestamp,(int,float)) or
            not math.isfinite(timestamp) or timestamp < 0 or isinstance(source_timestamp_ms,bool) or
            not isinstance(source_timestamp_ms,(int,float)) or not math.isfinite(source_timestamp_ms) or
            not math.isclose(timestamp,source_timestamp_ms,abs_tol=0.001)):
        raise EvidenceError('Decoded frame timestamp does not match pose evidence')
    if not isinstance(diagnostic,dict) or diagnostic.get('status') not in MODEL_STATUSES:
        raise EvidenceError('Model status is invalid')
    if (not isinstance(decision,dict) or
            set(decision) != {'show','reasons','allowed_indices','hidden_parts'} or
            type(decision['show']) is not bool or not isinstance(decision['reasons'],list) or
            not isinstance(decision['allowed_indices'],list) or
            not isinstance(decision['hidden_parts'],list) or
            len(set(decision['reasons'])) != len(decision['reasons']) or
            len(set(decision['allowed_indices'])) != len(decision['allowed_indices']) or
            len(set(decision['hidden_parts'])) != len(decision['hidden_parts']) or
            not set(decision['reasons']) <= set(LABELS) or
            not set(decision['allowed_indices']) <= set(JOINTS) or
            not set(decision['hidden_parts']) <= set(LIMBS) or
            decision['show'] != bool(decision['allowed_indices']) or
            (diagnostic['status'] != 'single_model_pose' and decision['show'])):
        raise EvidenceError('Display decision is invalid')
    if not isinstance(source_rgb,np.ndarray) or not isinstance(overlay_rgb,np.ndarray) or source_rgb.shape != overlay_rgb.shape:
        raise EvidenceError('Source and overlay image sizes do not match')

    source_png,overlay_png=_png(source_rgb),_png(overlay_rgb)
    state='hidden' if not decision['show'] else (
        'partial_display' if decision['hidden_parts'] else 'full_display')
    manifest={
        'format':FORMAT,
        'private_local_bundle':True,
        'run_key':run.key,
        'source_frame_index':index,
        'timestamp_ms':timestamp,
        'selection':{'start_frame':run.start,'end_frame':run.end,
                     'mirror_corrected':run.mirror},
        'model':_selected(run.model,('id','version','asset_sha256'),'Model'),
        'runtime':_selected(run.runtime,('mediapipe','numpy','av'),'Runtime'),
        'config':_selected(run.config,('version','min_visibility','min_presence'),'Config'),
        'model_status':diagnostic['status'],
        'display':{'rule_version':VERSION,'state':state,
                   'reason_codes':list(decision['reasons']),
                   'hidden_parts':list(decision['hidden_parts']),
                   'allowed_joint_count':len(decision['allowed_indices'])},
        'image_size':{'width':int(source_rgb.shape[1]),'height':int(source_rgb.shape[0])},
        'files':{
            'source':{'name':'source.png','sha256':sha256(source_png).hexdigest()},
            'overlay':{'name':'overlay.png','sha256':sha256(overlay_png).hexdigest()},
        },
        'limitations':list(LIMITATIONS),
    }
    try:
        manifest_bytes=json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2,
                                  allow_nan=False).encode('utf-8')
    except (TypeError,ValueError) as exc:
        raise EvidenceError('Evidence metadata is not JSON compatible') from exc
    output=BytesIO()
    with ZipFile(output,'w') as archive:
        _write(archive,'manifest.json',manifest_bytes)
        _write(archive,'source.png',source_png)
        _write(archive,'overlay.png',overlay_png)
    data=output.getvalue()
    read_evidence_bundle(data)
    return data


def _demo_html(frames,model,rule_version):
    cards=[]
    state_text={'full_display':'完整显示','partial_display':'部分显示','hidden':'全部隐藏'}
    for position,frame in enumerate(frames):
        reasons='、'.join(frame['display']['reason_codes']) or '无当前规则原因'
        hidden='、'.join(frame['display']['hidden_parts']) or '无'
        cards.append(f'''<section>
<h2>源帧 {frame['source_frame_index']} · {frame['timestamp_ms']/1000:.3f} 秒</h2>
<p>模型状态：{escape(frame['model_status'])}；显示状态：{state_text[frame['display']['state']]}；原因码：{escape(reasons)}；未显示部位：{escape(hidden)}。</p>
<div class="images"><figure><img src="frames/{position:03d}/source.png" alt="源帧"><figcaption>显示帧</figcaption></figure>
<figure><img src="frames/{position:03d}/overlay.png" alt="骨架对照"><figcaption>过滤后的模型骨架</figcaption></figure></div>
</section>''')
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>私人本地 Demo 报告</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:24px;color:#202124}}.warning{{background:#fff4d6;padding:12px;border-radius:8px}}section{{border-top:1px solid #ddd;margin-top:24px}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}img{{width:100%;height:auto}}figcaption{{text-align:center}}@media(max-width:700px){{.images{{grid-template-columns:1fr}}}}</style></head>
<body><h1>私人本地 Demo 报告</h1><p class="warning">本报告包含私人视频画面和未经核实的模型估计。它不提供动作评分、准确率或教练结论，请勿公开分享。</p>
<p>模型：{escape(model['id'])} / {escape(model['version'])}；显示规则：{escape(rule_version)}；手动选择证据帧：{len(frames)}。</p>
{''.join(cards)}</body></html>'''.encode('utf-8')


def build_demo_report(bundles) -> bytes:
    """Combine manually selected evidence bundles from one run into a local report."""
    if not isinstance(bundles,(list,tuple)) or not 1 <= len(bundles) <= MAX_DEMO_FRAMES:
        raise EvidenceError(f'Demo report requires 1–{MAX_DEMO_FRAMES} evidence frames')
    if any(type(item) is not bytes for item in bundles) or sum(map(len,bundles)) > MAX_DEMO_SOURCE_BYTES:
        raise EvidenceError('Demo report evidence size is invalid')
    evidence=[read_evidence_bundle(item) for item in bundles]
    first=evidence[0]['manifest']
    shared=('run_key','selection','model','runtime','config','image_size')
    if any(any(item['manifest'][field] != first[field] for field in shared) or
           item['manifest']['display']['rule_version'] != first['display']['rule_version']
           for item in evidence[1:]):
        raise EvidenceError('Demo report evidence must come from one run and configuration')
    evidence.sort(key=lambda item:item['manifest']['source_frame_index'])
    indices=[item['manifest']['source_frame_index'] for item in evidence]
    if len(set(indices)) != len(indices):
        raise EvidenceError('Demo report contains duplicate source frames')
    if any(right['manifest']['timestamp_ms'] <= left['manifest']['timestamp_ms']
           for left,right in zip(evidence,evidence[1:])):
        raise EvidenceError('Demo report evidence timestamps are not increasing')

    frames=[]
    image_files=[]
    for position,item in enumerate(evidence):
        source_name=f'frames/{position:03d}/source.png'
        overlay_name=f'frames/{position:03d}/overlay.png'
        source_png,overlay_png=item['source_png'],item['overlay_png']
        source=item['manifest']
        frames.append({
            'source_frame_index':source['source_frame_index'],
            'timestamp_ms':source['timestamp_ms'],
            'model_status':source['model_status'],
            'display':{key:value for key,value in source['display'].items()
                       if key != 'rule_version'},
            'image_size':source['image_size'],
            'files':{
                'source':{'name':source_name,'sha256':sha256(source_png).hexdigest()},
                'overlay':{'name':overlay_name,'sha256':sha256(overlay_png).hexdigest()},
            },
        })
        image_files.extend(((source_name,source_png),(overlay_name,overlay_png)))
    report=_demo_html(frames,first['model'],first['display']['rule_version'])
    readme=('这是私人本地 Demo 报告。解压后用浏览器打开 report.html。\n'
            '报告含视频画面和未经核实的模型骨架，请勿加入 Git 或公开分享。\n'
            'manifest.json 记录来源帧、PTS、版本和文件哈希；本报告不提供动作评分、准确率或教练结论。\n').encode('utf-8')
    manifest={
        'format':'boxing-demo-report-1','private_local_bundle':True,
        'run_key':first['run_key'],'selection':first['selection'],'model':first['model'],
        'runtime':first['runtime'],'config':first['config'],
        'display_rule_version':first['display']['rule_version'],
        'frame_count':len(frames),'frames':frames,
        'files':{
            'report':{'name':'report.html','sha256':sha256(report).hexdigest()},
            'readme':{'name':'README.txt','sha256':sha256(readme).hexdigest()},
        },
        'limitations':list(LIMITATIONS),
    }
    _contract(DEMO_VALIDATOR,manifest,'Demo report')
    manifest_bytes=json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2,
                              allow_nan=False).encode('utf-8')
    output=BytesIO()
    with ZipFile(output,'w') as archive:
        _write(archive,'manifest.json',manifest_bytes)
        _write(archive,'report.html',report)
        _write(archive,'README.txt',readme)
        for name,data in image_files: _write(archive,name,data)
    return output.getvalue()
