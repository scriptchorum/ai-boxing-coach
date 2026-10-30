"""Privacy-safe aggregate diagnostics for completed local pose runs."""
import json
import math
from statistics import median
from typing import Iterable

from .pose import JOINTS, PoseRun


FORMAT = 'boxing-pose-batch-diagnostics-1'
STATUSES = ('no_pose','multiple_poses','single_model_pose')


class PoseBatchError(ValueError):
    pass


def _canonical(value, label: str) -> str:
    try:
        return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    except (TypeError,ValueError) as exc:
        raise PoseBatchError(f'{label} 无法用于批量诊断。') from exc


def _project_identity(run: PoseRun) -> dict:
    try:
        return {
            'model': {name:run.model[name] for name in ('id','version','asset_sha256')},
            'runtime': {name:run.runtime[name] for name in ('mediapipe','numpy','av')},
            'config': {name:run.config[name] for name in
                       ('version','min_visibility','min_presence','interpolation_max_gap_ms')},
        }
    except (KeyError,TypeError) as exc:
        raise PoseBatchError('运行记录缺少模型、依赖或配置身份。') from exc


def _percentile(values: list[float], fraction: float) -> float:
    """Linear percentile over zero-based ranks; also defined for one value."""
    ordered=sorted(values)
    position=(len(ordered)-1)*fraction
    lower=math.floor(position)
    upper=math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)


def _validate_run(run: PoseRun) -> dict:
    if not isinstance(run,PoseRun):
        raise PoseBatchError('批次只接受完整的姿态运行记录。')
    if not isinstance(run.key,str) or not run.key:
        raise PoseBatchError('运行记录缺少稳定运行标识。')
    expected=run.end-run.start+1
    if expected <= 0 or len(run.frames) != expected or len(run.diagnostics) != expected:
        raise PoseBatchError('运行记录的区间、帧或诊断数量不一致。')
    statuses={name:0 for name in STATUSES}
    for offset,(frame,diagnostic) in enumerate(zip(run.frames,run.diagnostics)):
        source_index=run.start+offset
        if (not isinstance(frame,dict) or not isinstance(diagnostic,dict)
                or frame.get('source_frame_index') != source_index
                or diagnostic.get('source_frame_index') != source_index):
            raise PoseBatchError('运行记录的源帧顺序不连续。')
        status=diagnostic.get('status') if isinstance(diagnostic,dict) else None
        if status not in statuses:
            raise PoseBatchError('运行记录含未知的姿态诊断状态。')
        statuses[status]+=1
    if (isinstance(run.elapsed_seconds,bool) or not isinstance(run.elapsed_seconds,(int,float))
            or not math.isfinite(run.elapsed_seconds) or run.elapsed_seconds < 0):
        raise PoseBatchError('运行耗时必须是有限的非负数。')
    try:
        summary=run.summary()
    except (IndexError,KeyError,TypeError,ValueError) as exc:
        raise PoseBatchError('运行记录的关键点结构无法汇总。') from exc
    if any(summary[name] != statuses[name] for name in STATUSES):
        raise PoseBatchError('运行记录的帧诊断与汇总不一致。')
    unavailable=summary.get('unavailable_joint_frames')
    if not isinstance(unavailable,dict) or set(unavailable) != set(JOINTS.values()):
        raise PoseBatchError('运行记录缺少固定关节可用性统计。')
    return {'statuses':statuses,'unavailable':unavailable}


def aggregate_pose_runs(runs: Iterable[PoseRun]) -> dict:
    """Aggregate runs without exporting run keys, paths, sample IDs, or landmarks."""
    items=list(runs)
    if not items:
        raise PoseBatchError('至少需要一条完整运行记录。')
    validated=[]
    keys=set()
    for run in items:
        result=_validate_run(run)
        if run.key in keys:
            raise PoseBatchError('批次含重复运行标识，已拒绝重复计入。')
        keys.add(run.key)
        validated.append(result)
    identity=_project_identity(items[0])
    canonical_identity=_canonical(
        {'model':items[0].model,'runtime':items[0].runtime,'config':items[0].config},
        '首条运行身份',
    )
    statuses={name:0 for name in STATUSES}
    unavailable={name:0 for name in JOINTS.values()}
    elapsed=[]
    frame_count=0
    for run,result in zip(items,validated):
        if _canonical({'model':run.model,'runtime':run.runtime,'config':run.config},'运行身份') != canonical_identity:
            raise PoseBatchError('批次混用了不同模型、依赖或配置，不能合并。')
        frame_count+=len(run.frames)
        elapsed.append(float(run.elapsed_seconds))
        for name in STATUSES:
            statuses[name]+=result['statuses'][name]
        for name in unavailable:
            unavailable[name]+=result['unavailable'][name]
    total_elapsed=sum(elapsed)
    if not math.isfinite(total_elapsed):
        raise PoseBatchError('批次总耗时超出可表示范围。')
    timing={
        'total':round(total_elapsed,3),
        'median_per_run':round(median(elapsed),3),
        'p95_per_run_linear':round(_percentile(elapsed,.95),3),
        'aggregate_frames_per_second':round(frame_count/total_elapsed,3) if total_elapsed else None,
    }
    return {
        'format':FORMAT,
        'run_count':len(items),
        'frame_count':frame_count,
        **identity,
        'status_frames':statuses,
        'unavailable_joint_frames':unavailable,
        'timing_seconds':timing,
        'interpretation':['engineering_diagnostics_only','not_accuracy_or_action_assessment'],
    }


def batch_json(report: dict) -> bytes:
    """Serialize an aggregate report deterministically for local or reviewed export."""
    if report.get('format') != FORMAT:
        raise PoseBatchError('不是受支持的批量诊断格式。')
    try:
        return (json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    except (TypeError,ValueError) as exc:
        raise PoseBatchError('批量诊断无法序列化。') from exc
