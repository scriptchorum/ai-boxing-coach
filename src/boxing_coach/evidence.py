"""Build private local evidence-frame bundles without action assessment."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import math
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image

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


def build_evidence_bundle(run, frame, diagnostic, decision, source_rgb, overlay_rgb, *,
                          source_timestamp_ms) -> bytes:
    """Return a deterministic local-only ZIP for one already decoded source frame."""
    if (not isinstance(getattr(run,'key',None),str) or
            re.fullmatch(r'[0-9a-f]{64}',run.key) is None):
        raise EvidenceError('Run identity is invalid')
    if (type(getattr(run,'start',None)) is not int or type(getattr(run,'end',None)) is not int or
            type(getattr(run,'mirror',None)) is not bool or not 0 <= run.start <= run.end < 20000 or
            run.end-run.start+1 > 1800 or len(run.frames) != len(run.diagnostics)):
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
    return output.getvalue()
