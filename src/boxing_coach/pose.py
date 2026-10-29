"""Local experimental pose extraction; no subject identity or action assessment."""
from contextlib import closing
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np

from .model_assets import AssetError, find_project_root, load_manifest, select_asset, verify_asset, sha256_file
from .preview import PreviewVideo, selection_times
from .video_io import decode_video


CONFIG = {
    'version': 'pose-prototype-0.1', 'mode': 'VIDEO', 'num_poses': 2,
    'detection_confidence': 0.5, 'presence_confidence': 0.5, 'tracking_confidence': 0.5,
    'min_visibility': 0.5, 'min_presence': 0.5, 'interpolation_max_gap_ms': 0,
    'max_duration_ms': 30000, 'max_frames': 1800,
}
JOINTS = {11:'左肩',12:'右肩',13:'左肘',14:'右肘',15:'左腕',16:'右腕',23:'左髋',24:'右髋',25:'左膝',26:'右膝',27:'左踝',28:'右踝'}
EDGES = ((11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(28,30),(30,32))
LEFT_COLOR = '#00bfe8'
RIGHT_COLOR = '#ff9f43'
CENTER_COLOR = '#f5f5f5'
# Anatomical model labels stay attached to their indices, including after mirroring.
POINT_STYLES = {
    **{str(i): {'color': LEFT_COLOR, 'shape': 'circle'} for i in (11,13,15,17,19,21,23,25,27,29,31)},
    **{str(i): {'color': RIGHT_COLOR, 'shape': 'square'} for i in (12,14,16,18,20,22,24,26,28,30,32)},
}


class PoseError(RuntimeError):
    pass


def runtime_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ('mediapipe','numpy','av')}


def model_identity() -> dict:
    asset = select_asset(load_manifest(find_project_root()))
    return {'id':asset['asset_id'], 'version':str(asset['source_generation']), 'asset_sha256':asset['sha256']}


def run_key(video: PreviewVideo, start: int, end: int, mirror: bool) -> str:
    data = {'input':video.inspection.input_sha256,'start':start,'end':end,'mirror':mirror,
            'model':model_identity(),'config':CONFIG,'runtime':runtime_versions()}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def engine_timestamps(timings) -> list[int]:
    """Adapt to integer MediaPipe milliseconds without rewriting source PTS."""
    values = [round(t.timestamp_ms) for t in timings]
    if any(b <= a for a,b in zip(values,values[1:])):
        raise PoseError('所选片段的时间间隔无法映射到模型时钟，请更换片段；未补帧或修改原时间轴。')
    return values


def missing_landmarks() -> list[dict]:
    return [{'index':i,'image':None,'world':None,'visibility':None,'presence':None,'state':'missing'} for i in range(33)]


def finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def confidence(value):
    value = finite(value)
    return value if value is not None and 0 <= value <= 1 else None


def normalize_result(result) -> tuple[list[dict], int, str]:
    count = len(result.pose_landmarks)
    if count != 1:
        return missing_landmarks(), count, 'no_pose' if count == 0 else 'multiple_poses'
    raw = result.pose_landmarks[0]
    if len(raw) != 33:
        raise PoseError('模型返回的关键点布局异常，结果未采用。')
    points = []
    for index, point in enumerate(raw):
        xyz = [finite(getattr(point,axis,None)) for axis in ('x','y','z')]
        image = dict(zip(('x','y','z'),xyz)) if all(v is not None for v in xyz) else None
        points.append({'index':index,'image':image,'world':None,
                       'visibility':confidence(getattr(point,'visibility',None)),
                       'presence':confidence(getattr(point,'presence',None)),
                       'state':'observed' if image is not None else 'missing'})
    return points, count, 'single_model_pose'


def drawable(point: dict) -> bool:
    image = point['image']
    return bool(image is not None and point['state']=='observed'
                and point['visibility'] is not None and point['visibility'] >= CONFIG['min_visibility']
                and point['presence'] is not None and point['presence'] >= CONFIG['min_presence']
                and 0 <= image['x'] <= 1 and 0 <= image['y'] <= 1)


@dataclass
class PoseRun:
    key: str
    model: dict
    runtime: dict
    config: dict
    start: int
    end: int
    mirror: bool
    frames: list[dict]
    diagnostics: list[dict]
    elapsed_seconds: float

    def as_sequence(self, sample_id: str) -> dict:
        return {'schema_version':'0.1.0','sample_id':sample_id,'model':self.model,
                'config':{'version':self.config['version'],'landmark_layout':'mediapipe_33',
                          'coordinate_space':'upright_unmirrored_image','world_origin':'hip_midpoint_model_estimate',
                          'interpolation_max_gap_ms':0,'min_visibility':self.config['min_visibility'],
                          'min_presence':self.config['min_presence']}, 'frames':self.frames}

    def summary(self) -> dict:
        total = len(self.frames)
        statuses = {name:sum(d['status']==name for d in self.diagnostics)
                    for name in ('no_pose','multiple_poses','single_model_pose')}
        unavailable = {name:sum(not drawable(f['landmarks'][index]) for f in self.frames)
                       for index,name in JOINTS.items()}
        return {'frames':total, **statuses, 'unavailable_joint_frames':unavailable,
                'elapsed_seconds':round(self.elapsed_seconds,3)}


def extract_pose(video: PreviewVideo, start: int, end: int, *, correct_mirror=False, progress=None) -> PoseRun:
    begin_ms, stop_ms = selection_times(video.inspection,start,end)
    if stop_ms-begin_ms > CONFIG['max_duration_ms'] or end-start+1 > CONFIG['max_frames']:
        raise PoseError('当前试跑每次最多选择 30 秒、1800 帧，请缩短区间。')
    timings = video.inspection.timeline[start:end+1]
    clocks = engine_timestamps(timings)
    try:
        root = find_project_root()
        asset = select_asset(load_manifest(root))
        model_path = root/'models'/asset['filename']
        verify_asset(model_path,asset)
    except (AssetError,OSError) as exc:
        raise PoseError('模型缺失或校验失败。请按工程说明运行 boxing-coach-models sync/verify 后重试。') from exc
    import mediapipe as mp
    identity = model_identity()
    key = run_key(video,start,end,correct_mirror)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path),delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=CONFIG['num_poses'],
        min_pose_detection_confidence=CONFIG['detection_confidence'],
        min_pose_presence_confidence=CONFIG['presence_confidence'],
        min_tracking_confidence=CONFIG['tracking_confidence'],output_segmentation_masks=False)
    frames, diagnostics = [], []
    started = time.perf_counter()
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            with closing(decode_video(video.path,inspection=video.inspection,correct_mirror=correct_mirror)) as decoded:
                for frame in decoded:
                    if frame.source_frame_index < start:
                        continue
                    offset = frame.source_frame_index-start
                    image = mp.Image(image_format=mp.ImageFormat.SRGB,data=frame.image_rgb)
                    result = detector.detect_for_video(image,clocks[offset])
                    points,count,status = normalize_result(result)
                    frames.append({'source_frame_index':frame.source_frame_index,'timestamp_ms':frame.timestamp_ms,'landmarks':points})
                    diagnostics.append({'source_frame_index':frame.source_frame_index,'engine_timestamp_ms':clocks[offset],
                                        'detected_count_capped':count,'status':status})
                    if progress and (offset % 15 == 0 or frame.source_frame_index == end):
                        progress(offset+1,end-start+1)
                    if frame.source_frame_index == end:
                        break
    except RuntimeError as exc:
        if isinstance(exc,PoseError):
            raise
        raise PoseError('模型执行失败，本次结果未保存。请检查环境和输入后重试。') from exc
    if len(frames) != end-start+1:
        raise PoseError('解码帧数与所选区间不一致，本次结果未采用。')
    if sha256_file(video.path) != video.inspection.input_sha256:
        raise PoseError('视频在处理期间发生变化，本次结果未采用，请重新打开。')
    return PoseRun(key,identity,runtime_versions(),dict(CONFIG),start,end,correct_mirror,frames,diagnostics,time.perf_counter()-started)


def draw_pose(image_rgb: np.ndarray, frame: dict) -> np.ndarray:
    import cv2
    output = image_rgb.copy()
    height,width = output.shape[:2]
    points = {p['index']:p for p in frame['landmarks'] if drawable(p)}
    def pixel(index):
        image = points[index]['image']
        return (min(width-1,round(image['x']*width)),min(height-1,round(image['y']*height)))
    def style(index):
        return POINT_STYLES.get(str(index), {'color': CENTER_COLOR, 'shape': 'circle'})
    def rgb(color):
        return tuple(int(color[i:i+2],16) for i in (1,3,5))
    for a,b in EDGES:
        if a in points and b in points:
            color = style(a)['color'] if style(a)['color']==style(b)['color'] else CENTER_COLOR
            cv2.line(output,pixel(a),pixel(b),rgb(color),max(2,width//300),cv2.LINE_AA)
    for index in points:
        if index >= 11:
            point_style=style(index)
            x,y=pixel(index); radius=max(3,width//180)
            if point_style['shape']=='square':
                cv2.rectangle(output,(x-radius,y-radius),(x+radius,y+radius),rgb(point_style['color']),-1,cv2.LINE_AA)
            else:
                cv2.circle(output,(x,y),radius,rgb(point_style['color']),-1,cv2.LINE_AA)
    return output
