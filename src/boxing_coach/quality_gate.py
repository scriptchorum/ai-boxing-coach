"""Analysis quality gates; separate from display filtering and action assessment."""
import math
import re
from typing import Iterable

from .pose import PoseRun
from .pose_batch import PoseBatchError, aggregate_pose_runs
from .quality import MAX_GAP_MS, MAX_TORSO_SPEED, VERSION as DISPLAY_RULE_VERSION, assess_frames


FORMAT = 'boxing-analysis-quality-1'
VERSION = 'analysis-quality-0.1'
DECLARED_CLIP_REASONS = (
    'declared_multiple_people','declared_out_of_frame','declared_blur',
    'declared_occlusion','declared_unsupported_view',
)
FRAME_REASON_MAP = {
    'no_pose':'no_model_pose',
    'multiple_poses':'multiple_model_pose_candidates',
    'out_of_frame':'critical_landmark_out_of_frame',
    'missing_points':'critical_landmark_unusable',
    'position_jump':'torso_position_jump',
    'time_gap':'timestamp_gap',
    'torso_missing':'torso_landmark_unusable',
}
FRAME_REASON_CODES = tuple(dict.fromkeys(FRAME_REASON_MAP.values()))+('interpolation_not_supported',)
METRIC_REASON_CODES = (
    'missing_event','no_model_pose','multiple_model_pose_candidates',
    'critical_landmark_missing','critical_landmark_low_signal',
    'critical_landmark_out_of_frame','timestamp_gap','torso_position_jump',
    'interpolation_not_supported',*DECLARED_CLIP_REASONS,
)
LIMITATIONS = (
    'engineering_gate_not_accuracy',
    'single_model_pose_not_single_person_proof',
    'thresholds_require_real_sample_calibration',
    'no_action_or_coaching_assessment',
)


class QualityGateError(ValueError):
    pass


def _validated_run(run: PoseRun) -> dict:
    try:
        batch=aggregate_pose_runs([run])
    except (PoseBatchError,TypeError,ValueError) as exc:
        raise QualityGateError('姿态运行记录不完整，无法执行质量门禁。') from exc
    for frame in run.frames:
        landmarks=frame.get('landmarks') if isinstance(frame,dict) else None
        if (not isinstance(landmarks,list) or len(landmarks) != 33
                or any(not isinstance(point,dict) for point in landmarks)
                or {point.get('index') for point in landmarks} != set(range(33))):
            raise QualityGateError('姿态运行记录的关键点布局不是唯一的 0–32。')
    return batch


def _declared_reasons(reasons: Iterable[str]) -> list[str]:
    try:
        values=list(reasons)
    except TypeError as exc:
        raise QualityGateError('显式坏片原因必须是可迭代的稳定原因码。') from exc
    if (any(not isinstance(value,str) for value in values)
            or len(set(values)) != len(values)
            or any(value not in DECLARED_CLIP_REASONS for value in values)):
        raise QualityGateError('显式坏片原因未知或重复。')
    return [code for code in DECLARED_CLIP_REASONS if code in values]


def _thresholds(run: PoseRun) -> dict:
    return {
        'min_visibility':run.config['min_visibility'],
        'min_presence':run.config['min_presence'],
        'max_torso_speed_normalized_per_second':MAX_TORSO_SPEED,
        'max_timeline_gap_ms':MAX_GAP_MS,
    }


def _interpolation(run: PoseRun) -> dict:
    frame_count=0
    landmark_count=0
    for frame in run.frames:
        count=sum(point.get('state') == 'interpolated' for point in frame['landmarks'])
        if count:
            frame_count+=1
            landmark_count+=count
    maximum=run.config.get('interpolation_max_gap_ms')
    return {
        'policy':'disabled' if maximum == 0 else 'configured_but_not_implemented',
        'max_gap_ms':maximum,
        'applied_frame_count':frame_count,
        'applied_landmark_count':landmark_count,
    }


def clip_quality_report(run: PoseRun, *, declared_reasons: Iterable[str] = ()) -> dict:
    """Return privacy-safe, clip-level engineering diagnostics for one pose run."""
    batch=_validated_run(run)
    declared=_declared_reasons(declared_reasons)
    try:
        decisions=assess_frames(run)
    except (IndexError,KeyError,TypeError,ValueError) as exc:
        raise QualityGateError('姿态帧无法按固定显示规则诊断。') from exc
    counts={code:0 for code in FRAME_REASON_CODES}
    states={'full_display':0,'partial_display':0,'hidden':0}
    for decision in decisions:
        state='hidden' if not decision['show'] else ('partial_display' if decision['hidden_parts'] else 'full_display')
        states[state]+=1
        for display_reason in decision['reasons']:
            try:
                counts[FRAME_REASON_MAP[display_reason]]+=1
            except KeyError as exc:
                raise QualityGateError('显示规则返回了未知原因。') from exc
    interpolation=_interpolation(run)
    if interpolation['applied_landmark_count'] or interpolation['policy'] != 'disabled':
        counts['interpolation_not_supported']=interpolation['applied_frame_count'] or len(run.frames)
    rejected=bool(declared or states['hidden'] == len(run.frames) or counts['interpolation_not_supported'])
    return {
        'format':FORMAT,
        'rule_version':VERSION,
        'display_rule_version':DISPLAY_RULE_VERSION,
        'model':batch['model'],
        'runtime':batch['runtime'],
        'config':batch['config'],
        'thresholds':_thresholds(run),
        'frame_count':batch['frame_count'],
        'model_status_frames':batch['status_frames'],
        'display_state_frames':states,
        'frame_reason_counts':counts,
        'declared_clip_reasons':declared,
        'clip_status':'rejected' if rejected else 'review_required',
        'interpolation':interpolation,
        'limitations':list(LIMITATIONS),
    }


def _metric_id(value: str) -> str:
    if not isinstance(value,str) or not re.fullmatch(r'[a-z][a-z0-9_]{1,63}',value):
        raise QualityGateError('指标标识格式无效。')
    return value


def _required_indices(values: Iterable[int]) -> list[int]:
    try:
        indices=list(values)
    except TypeError as exc:
        raise QualityGateError('指标关键点索引必须是可迭代整数集合。') from exc
    if (not indices or len(set(indices)) != len(indices)
            or any(isinstance(index,bool) or not isinstance(index,int) or not 0 <= index <= 32 for index in indices)):
        raise QualityGateError('指标关键点索引必须是唯一的 0–32 整数。')
    return sorted(indices)


def _events(values: dict) -> dict:
    if not isinstance(values,dict) or not values:
        raise QualityGateError('指标至少需要一个事件帧。')
    result={}
    for name,index in values.items():
        if not isinstance(name,str) or not re.fullmatch(r'[a-z][a-z0-9_]{1,31}',name):
            raise QualityGateError('事件标识格式无效。')
        if index is not None and (isinstance(index,bool) or not isinstance(index,int) or index < 0):
            raise QualityGateError('事件帧必须是非负源帧号或 null。')
        result[name]=index
    return dict(sorted(result.items()))


def _point_reasons(point: dict | None, config: dict) -> set[str]:
    if not isinstance(point,dict) or point.get('state') not in ('observed','interpolated') or point.get('image') is None:
        return {'critical_landmark_missing'}
    reasons=set()
    if point.get('state') == 'interpolated':
        reasons.add('interpolation_not_supported')
    visibility=point.get('visibility')
    presence=point.get('presence')
    if (not isinstance(visibility,(int,float)) or isinstance(visibility,bool) or not math.isfinite(visibility)
            or not isinstance(presence,(int,float)) or isinstance(presence,bool) or not math.isfinite(presence)
            or visibility < config['min_visibility'] or presence < config['min_presence']):
        reasons.add('critical_landmark_low_signal')
    image=point['image']
    x=image.get('x') if isinstance(image,dict) else None
    y=image.get('y') if isinstance(image,dict) else None
    if (not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x)
            or not isinstance(y,(int,float)) or isinstance(y,bool) or not math.isfinite(y)):
        reasons.add('critical_landmark_missing')
    elif not (0 <= x <= 1 and 0 <= y <= 1):
        reasons.add('critical_landmark_out_of_frame')
    return reasons


def metric_quality_gate(run: PoseRun, metric_id: str, required_indices: Iterable[int],
                        event_frames: dict, *, single_person_confirmed: bool = False,
                        declared_reasons: Iterable[str] = ()) -> dict:
    """Gate a future metric without calculating or inventing the metric value."""
    _validated_run(run)
    metric=_metric_id(metric_id)
    indices=_required_indices(required_indices)
    events=_events(event_frames)
    declared=_declared_reasons(declared_reasons)
    if not isinstance(single_person_confirmed,bool):
        raise QualityGateError('单人确认必须是显式布尔值。')
    try:
        decisions=assess_frames(run)
    except (IndexError,KeyError,TypeError,ValueError) as exc:
        raise QualityGateError('姿态帧无法按固定显示规则诊断。') from exc
    by_index={frame['source_frame_index']:(frame,diagnostic,decision)
              for frame,diagnostic,decision in zip(run.frames,run.diagnostics,decisions,strict=True)}
    reasons=set(declared)
    for source_index in events.values():
        if source_index is None or source_index not in by_index:
            reasons.add('missing_event')
            continue
        frame,diagnostic,decision=by_index[source_index]
        if diagnostic['status'] == 'no_pose':
            reasons.add('no_model_pose')
        elif diagnostic['status'] == 'multiple_poses':
            reasons.add('multiple_model_pose_candidates')
        points={point.get('index'):point for point in frame['landmarks'] if isinstance(point,dict)}
        for index in indices:
            reasons.update(_point_reasons(points.get(index),run.config))
        if 'time_gap' in decision['reasons']:
            reasons.add('timestamp_gap')
        if 'position_jump' in decision['reasons']:
            reasons.add('torso_position_jump')
    interpolation=_interpolation(run)
    if interpolation['policy'] != 'disabled':
        reasons.add('interpolation_not_supported')
    ordered=[code for code in METRIC_REASON_CODES if code in reasons]
    if ordered:
        status='rejected'
    elif not single_person_confirmed:
        status='review_required'
    else:
        status='eligible'
    return {
        'format':FORMAT,
        'rule_version':VERSION,
        'metric_id':metric,
        'required_landmark_indices':indices,
        'event_names':list(events),
        'status':status,
        'reason_codes':ordered,
        'review_reasons':[] if single_person_confirmed else ['single_person_not_confirmed'],
        'value_policy':'compute_later' if status == 'eligible' else 'must_be_null',
        'thresholds':_thresholds(run),
        'interpolation':interpolation,
        'limitations':list(LIMITATIONS),
    }
