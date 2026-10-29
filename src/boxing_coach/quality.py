"""Experimental display rejection rules; never a subject or action classifier."""
from math import hypot
from .pose import JOINTS, drawable

VERSION = 'display-quality-0.1'
MAX_TORSO_SPEED = 3.0  # normalized image axes per second, candidate threshold
MAX_GAP_MS = 100.0
LABELS = {
    'no_pose': '未检出人体',
    'multiple_poses': '多个候选，无法确定训练者',
    'out_of_frame': '身体关键点落在画面外',
    'missing_points': '全身关键点不完整或信号不足',
    'position_jump': '躯干位置突变，需人工核对',
    'time_gap': '时间间隔过大，无法连续核对',
}

def assess_frames(run):
    """Return reasons per source frame without editing raw model evidence.

    Compare adjacent reliable torso estimates, including rejected jump frames;
    this detects transitions, not identity, and does not lock onto a subject.
    """
    results=[]
    previous=None
    for frame, diagnostic in zip(run.frames,run.diagnostics,strict=True):
        reasons=[]
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
            if any(not drawable(p) for p in required):
                reasons.append('missing_points')
            torso=[points[i] for i in (11,12,23,24)]
            current=None
            if all(drawable(p) for p in torso):
                center=tuple(sum(p['image'][axis] for p in torso)/4 for axis in ('x','y'))
                current=(frame['timestamp_ms'],center)
                if previous is not None:
                    dt=current[0]-previous[0]
                    if dt <= 0 or dt > MAX_GAP_MS:
                        reasons.append('time_gap')
                    elif hypot(center[0]-previous[1][0],center[1]-previous[1][1])/(dt/1000) > MAX_TORSO_SPEED:
                        reasons.append('position_jump')
            previous=current
        results.append({'show':not reasons,'reasons':reasons})
    return results

def reason_text(decision):
    return '；'.join(LABELS[code] for code in decision['reasons'])
