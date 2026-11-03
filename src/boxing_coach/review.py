"""Manual display inspection helpers; no automatic ground truth."""
import json
from .quality import VERSION, assess_frames

MAX_REVIEW_BYTES = 1024 * 1024
FILTERS=('部分显示或隐藏','仅部分显示','仅全部隐藏','仅位置突变','完整显示（检查漏检）','全部帧')
VERDICTS=('未记录','跟对人，已显示点大致对齐','跟错人或落在器械上','跟对人，但已显示点明显错位','无法判断')
DISPLAY_STATES=('完整显示','部分显示','全部隐藏')

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


def summarize_reviews(run,reviews):
    """Return aggregate inspection counts without media or frame identifiers."""
    if not isinstance(reviews,dict):
        raise ValueError('人工核对记录不合法。')
    decisions={f['source_frame_index']:d for f,d in zip(run.frames,assess_frames(run),strict=True)}
    verdict_counts={verdict:0 for verdict in VERDICTS[1:]}
    display_counts={state:0 for state in DISPLAY_STATES}
    issue_reason_counts={}
    problem_verdicts=set(VERDICTS[2:4])
    for index,verdict in reviews.items():
        if type(index) is not int or index not in decisions or verdict not in VERDICTS[1:]:
            raise ValueError('人工核对记录不合法。')
        decision=decisions[index]
        if not decision['show'] and verdict!='无法判断':
            raise ValueError('全部隐藏的帧不能包含已显示骨架的判断。')
        state=DISPLAY_STATES[2] if not decision['show'] else (
            DISPLAY_STATES[1] if decision['hidden_parts'] else DISPLAY_STATES[0])
        verdict_counts[verdict]+=1
        display_counts[state]+=1
        if verdict in problem_verdicts:
            reasons=decision['reasons'] or ['no_display_rule_reason']
            for reason in reasons:
                issue_reason_counts[reason]=issue_reason_counts.get(reason,0)+1
    return {
        'reviewed_frames':len(reviews),
        'total_frames':len(run.frames),
        'problem_frames':sum(verdict_counts[v] for v in problem_verdicts),
        'verdict_counts':verdict_counts,
        'display_counts':display_counts,
        'issue_reason_counts':dict(sorted(issue_reason_counts.items())),
    }

def export_reviews(run,reviews):
    payload={'format':'boxing-display-review-2','run_key':run.key,
             'quality_version':VERSION,
             'settings':{'start_frame':run.start,'end_frame':run.end,'mirror':run.mirror},
             'records':[{'frame':i,'verdict':v} for i,v in sorted(reviews.items())]}
    data=json.dumps(payload,ensure_ascii=False,indent=2).encode('utf-8')
    import_reviews(run,data)  # The same validation guards both directions.
    return data

def read_review_file(data):
    """Validate a bounded document before showing its untrusted settings."""
    if len(data)>MAX_REVIEW_BYTES:
        raise ValueError('核对文件超过 1 MiB，未导入。')
    try:
        payload=json.loads(data.decode('utf-8'))
    except (ValueError,UnicodeError,RecursionError) as exc:
        raise ValueError('核对文件不是有效的 UTF-8 JSON，未导入。') from exc
    if not isinstance(payload,dict):
        raise ValueError('核对文件结构不受支持，未导入。')
    if payload.get('format') not in ('boxing-display-review-1','boxing-display-review-2'):
        raise ValueError('核对文件版本不受支持，未导入。')
    fields={'format','run_key','quality_version','records'}
    settings=None
    if payload['format']=='boxing-display-review-2':
        fields.add('settings')
        settings=payload.get('settings')
        if not isinstance(settings,dict) or set(settings)!={'start_frame','end_frame','mirror'}:
            raise ValueError('核对文件的恢复设置不合法，未导入。')
        start,end=settings['start_frame'],settings['end_frame']
        if (type(start) is not int or type(end) is not int or
                not 0<=start<=end<20000 or end-start+1>1800 or type(settings['mirror']) is not bool):
            raise ValueError('核对文件的恢复设置不合法，未导入。')
    if (set(payload)!=fields or not isinstance(payload['run_key'],str) or
            not isinstance(payload['quality_version'],str)):
        raise ValueError('核对文件结构不受支持，未导入。')
    records=payload['records']
    if not isinstance(records,list) or len(records)>1800:
        raise ValueError('核对记录数量不合法，未导入。')
    seen=set()
    for row in records:
        if not isinstance(row,dict) or set(row)!={'frame','verdict'}:
            raise ValueError('核对记录结构不合法，未导入。')
        index=row['frame']
        if (type(index) is not int or not 0<=index<20000 or index in seen or
                row['verdict'] not in VERDICTS[1:] or
                (settings is not None and not settings['start_frame']<=index<=settings['end_frame'])):
            raise ValueError('帧号重复、超出选段或记录类型不合法，未导入。')
        seen.add(index)
    return payload


def import_reviews(run,data):
    """Settings are hints, never a replacement for the full run identity check."""
    payload=read_review_file(data)
    if payload.get('settings') is not None and payload['settings']!={
            'start_frame':run.start,'end_frame':run.end,'mirror':run.mirror}:
        raise ValueError('文件中的选段或镜像设置与当前识别不一致，未导入。')
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
