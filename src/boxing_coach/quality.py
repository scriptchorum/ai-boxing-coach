"""Experimental display rejection rules; never a subject or action classifier."""
from math import hypot
from .pose import JOINTS, drawable

VERSION = 'display-quality-0.2'
TORSO = (11,12,23,24)
LIMBS = {'左臂':(11,13,15),'右臂':(12,14,16),'左腿':(23,25,27),'右腿':(24,26,28)}
MAX_TORSO_SPEED = 3.0  # normalized image axes per second, candidate threshold
MAX_GAP_MS = 100.0
LABELS = {
    'no_pose': '未检出人体',
    'multiple_poses': '多个候选，无法确定训练者',
    'out_of_frame': '身体关键点落在画面外',
    'missing_points': '全身关键点不完整或信号不足',
    'position_jump': '躯干位置突变，需人工核对',
    'time_gap': '时间间隔过大，无法连续核对',
    'torso_missing': '躯干关键点不足，无法可靠定位',
}

def assess_frames(run):
    """Return reasons per source frame without editing raw model evidence.

    Compare adjacent reliable torso estimates, including rejected jump frames;
    this detects transitions, not identity, and does not lock onto a subject.
    """
    results=[]
    previous=None
    config=getattr(run,'config',None)
    can_draw=lambda point: drawable(point,config)
    for frame, diagnostic in zip(run.frames,run.diagnostics,strict=True):
        reasons=[]
        allowed=[]
        hidden_parts=[]
        if diagnostic['status'] != 'single_model_pose':
            reasons.append(diagnostic['status'])
            previous=None
        else:
            points={p['index']:p for p in frame['landmarks']}
            required=[points[i] for i in JOINTS]
            if any(p['image'] is not None and
                   not (0 <= p['image']['x'] <= 1 and 0 <= p['image']['y'] <= 1)
                   for p in required):
                reasons.append('out_of_frame')
            if any(not can_draw(p) for p in required):
                reasons.append('missing_points')
            torso=[points[i] for i in TORSO]
            current=None
            if all(can_draw(p) for p in torso):
                allowed.extend(TORSO)
                for name,indices in LIMBS.items():
                    if all(can_draw(points[i]) for i in indices):
                        allowed.extend(indices[1:])
                    else:
                        hidden_parts.append(name)
                center=tuple(sum(p['image'][axis] for p in torso)/4 for axis in ('x','y'))
                current=(frame['timestamp_ms'],center)
                if previous is not None:
                    dt=current[0]-previous[0]
                    if dt <= 0 or dt > MAX_GAP_MS:
                        reasons.append('time_gap')
                    elif hypot(center[0]-previous[1][0],center[1]-previous[1][1])/(dt/1000) > MAX_TORSO_SPEED:
                        reasons.append('position_jump')
            else:
                reasons.append('torso_missing')
            previous=current
        block = any(code not in ('missing_points','out_of_frame') for code in reasons)
        if block: allowed=[]
        results.append({'show':bool(allowed),'reasons':reasons,'allowed_indices':allowed,
                        'hidden_parts':hidden_parts if not block else list(LIMBS)})
    return results

def reason_text(decision):
    return '；'.join(LABELS[code] for code in decision['reasons'])

def display_frame(frame,decision):
    """Project only allowed joints for both renderers, without changing evidence."""
    allowed=set(decision['allowed_indices'])
    return {**frame,'landmarks':[p for p in frame['landmarks'] if p['index'] in allowed]}
