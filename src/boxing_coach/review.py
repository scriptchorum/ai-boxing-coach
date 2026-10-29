"""Manual display inspection helpers; no automatic ground truth."""
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
