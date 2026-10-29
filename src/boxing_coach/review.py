"""Manual display inspection helpers; no automatic ground truth."""
import json
from .quality import VERSION, assess_frames

MAX_REVIEW_BYTES = 1024 * 1024
FILTERS=('部分显示或隐藏','仅部分显示','仅全部隐藏','仅位置突变','完整显示（检查漏检）','全部帧')
VERDICTS=('未记录','跟对人，已显示点大致对齐','跟错人或落在器械上','跟对人，但已显示点明显错位','无法判断')

def review_indices(frames,decisions,selection):
    if selection not in FILTERS: raise ValueError('Unknown review filter')
    def matches(d):
        partial=d['show'] and bool(d['hidden_parts'])
        return {
            FILTERS[0]:not d['show'] or partial,
            FILTERS[1]:partial,
            FILTERS[2]:not d['show'],
            FILTERS[3]:'position_jump' in d['reasons'],
            FILTERS[4]:d['show'] and not d['hidden_parts'],
            FILTERS[5]:True,
        }[selection]
    return [f['source_frame_index'] for f,d in zip(frames,decisions,strict=True) if matches(d)]

def neighbors(indices,current):
    return (next((i for i in reversed(indices) if i<current),None),
            next((i for i in indices if i>current),None))

def export_reviews(run,reviews):
    payload={'format':'boxing-display-review-1','run_key':run.key,
             'quality_version':VERSION,
             'records':[{'frame':i,'verdict':v} for i,v in sorted(reviews.items())]}
    data=json.dumps(payload,ensure_ascii=False,indent=2).encode('utf-8')
    import_reviews(run,data)  # The same validation guards both directions.
    return data

def import_reviews(run,data):
    """Read bounded JSON data only, never paths or executable object formats."""
    if len(data)>MAX_REVIEW_BYTES:
        raise ValueError('核对文件超过 1 MiB，未导入。')
    try:
        payload=json.loads(data.decode('utf-8'))
    except (ValueError,UnicodeError,RecursionError) as exc:
        raise ValueError('核对文件不是有效的 UTF-8 JSON，未导入。') from exc
    if not isinstance(payload,dict) or set(payload)!= {'format','run_key','quality_version','records'}:
        raise ValueError('核对文件结构不受支持，未导入。')
    if payload['format']!='boxing-display-review-1':
        raise ValueError('核对文件版本不受支持，未导入。')
    if payload['run_key']!=run.key or payload['quality_version']!=VERSION:
        raise ValueError('文件与当前视频、选段、镜像、模型或显示规则不一致，未导入。请恢复原设置并重新识别。')
    records=payload['records']
    if not isinstance(records,list) or len(records)>len(run.frames):
        raise ValueError('核对记录数量不合法，未导入。')
    decisions={f['source_frame_index']:d for f,d in zip(run.frames,assess_frames(run),strict=True)}
    result={}
    for row in records:
        if not isinstance(row,dict) or set(row)!={'frame','verdict'}:
            raise ValueError('核对记录结构不合法，未导入。')
        index,verdict=row['frame'],row['verdict']
        if type(index) is not int or index not in decisions or index in result or verdict not in VERDICTS[1:]:
            raise ValueError('帧号重复、超出选段或记录类型不合法，未导入。')
        if not decisions[index]['show'] and verdict!='无法判断':
            raise ValueError('全部隐藏的帧不能包含已显示骨架的判断，未导入。')
        result[index]=verdict
    return result
