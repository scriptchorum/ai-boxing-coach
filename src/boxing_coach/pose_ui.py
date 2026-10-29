"""Session-only diagnostic UI for the pose extraction prototype."""
import streamlit as st

from .model_assets import AssetError
from .pose import JOINTS, LEFT_COLOR, RIGHT_COLOR, PoseError, draw_pose, drawable, extract_pose, run_key
from .preview import read_preview_frame
from .video_io import MediaInputError
from .synced_player import render_synced_player
from .quality import assess_frames, reason_text, display_frame, VERSION, LABELS
from .review import FILTERS, VERDICTS, review_indices, neighbors, export_reviews, import_reviews, MAX_REVIEW_BYTES


def clear_pose():
    st.session_state.pop('pose_run',None)
    st.session_state.pop('pose_index',None)
    st.session_state.pop('pose_reviews',None)
    st.session_state.pop('pose_review_filter',None)


def go_review(index):
    st.session_state.pose_index = index


def record_review(index,verdict):
    reviews=st.session_state.setdefault('pose_reviews',{})
    if verdict == '未记录': reviews.pop(index,None)
    else: reviews[index]=verdict


def render_side_legend():
    st.markdown(f'<span style="color:{LEFT_COLOR}">● 左侧肢体（圆点）</span>　'
                f'<span style="color:{RIGHT_COLOR}">■ 右侧肢体（方点）</span>　肩线与髋线为白色',
                unsafe_allow_html=True)
    st.caption('左右指训练者身体的模型标签，不是画面左右，也不代表惯用手或前后手。镜像纠正后颜色仍跟随同一标签；重叠不代表识别错误，模型也可能把左右认反，需对照原片。')


def render_review_files(cached):
    with st.expander('保存或恢复人工核对记录'):
        st.caption('文件仅含帧号、你的判断和匹配标识，不含视频或骨架坐标。仍属于私人训练记录，请仅保存在受控目录，不加入 Git。刷新前请先下载；另一台需准备同一原片和相同识别设置。')
        uploaded=st.file_uploader('选择之前保存的核对记录',type=['json'],key='review_import_file')
        if st.button('导入并替换当前核对记录',key='review_import',disabled=uploaded is None):
            try:
                if uploaded.size>MAX_REVIEW_BYTES: raise ValueError('核对文件超过 1 MiB，未导入。')
                restored=import_reviews(cached,uploaded.getvalue())
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.pose_reviews=restored
                st.success(f'已恢复 {len(restored)} 帧记录；原视频和识别结果未修改。')
        reviews=st.session_state.setdefault('pose_reviews',{})
        st.download_button('下载本次核对记录',data=export_reviews(cached,reviews),
                           file_name='boxing-display-review.private.json',mime='application/json',
                           key='review_download',disabled=not reviews)


def render_pose_panel(video,start,end,mirror):
    st.subheader('播放并查看骨架',anchor='pose-playback')
    st.write('先点击下方“识别所选区间的人体关键点”，完成后播放器会出现在这里。刷新或换片后需要重新识别。')
    st.write('对上方所选区间运行本地模型，再逐帧核对肩、肘、腕、髋、膝与踝。结果只保留在当前会话，不上传或写入训练历史。')
    st.caption('目前是工程试跑：模型信号不等于识别准确率，也不表示动作正确。女性身体多样性的正式验证仍待完成。每次最多 30 秒、1800 帧。')
    try:
        key = run_key(video,start,end,mirror)
    except (AssetError,OSError):
        st.error('无法读取模型配置，请按工程说明检查模型资产。')
        return
    cached = st.session_state.get('pose_run')
    if cached is not None and cached.key != key:
        clear_pose()
        cached = None
    if st.button('识别所选区间的人体关键点',key='run_pose'):
        clear_pose()
        bar = st.progress(0,text='准备模型…')
        try:
            result = extract_pose(video,start,end,correct_mirror=mirror,
                                  progress=lambda done,total:bar.progress(done/total,text=f'已处理 {done}/{total} 帧'))
            st.session_state.pose_run = result
            cached = result
        except (PoseError,MediaInputError) as exc:
            st.error(f'本次识别未完成：{exc}')
            cached = None
        except OSError:
            st.error('本次识别未完成：无法读取本地视频或模型文件，请检查后重试。')
            cached = None
        finally:
            bar.empty()
    if cached is None:
        st.info('尚未生成骨架。点击上方识别按钮，等待处理完成后即可播放。')
        return
    st.button('清除本次关键点结果',on_click=clear_pose)
    render_review_files(cached)
    decisions = assess_frames(cached)
    st.info(f"分部位显示：{sum(not d['show'] for d in decisions)} 帧全部隐藏，{sum(d['show'] and bool(d['hidden_parts']) for d in decisions)} 帧仅显示部分身体，共 {len(decisions)} 帧。隐藏不表示动作错误，未隐藏也不证明识别准确。")
    st.caption('躯干定位不足、多人或位置突变时全部隐藏；否则分别检查左右手臂和腿，缺一处不再让其他部位一起消失。这是待验证规则，不能认人，也不能排除背景人物或器械误检。')
    with st.expander('查看隐藏原因统计'):
        st.dataframe([{'原因':label,'帧数':sum(code in d['reasons'] for d in decisions),'总帧数':len(decisions)} for code,label in LABELS.items()],hide_index=True)
        st.caption(f'规则版本 {VERSION}；同一帧可有多个原因，各项不能直接相加。')
    st.write('点击下面左侧视频的 ▶ 播放，右侧骨架随画面变化；暂停或拖动也可核对。仅识别过的区间显示骨架。')
    render_side_legend()
    try:
        read_preview_frame(video,start,correct_mirror=mirror)
        render_synced_player(video,mirror,cached)
    except ValueError as exc:
        st.warning(str(exc))
    except (MediaInputError,OSError):
        clear_pose()
        st.error('源视频已失效，请重新打开并识别。')
        return
    st.caption('更精确地检查某一帧，可继续使用下方“骨架核对源帧号”。')
    summary = cached.summary()
    st.write(f"已处理 {summary['frames']} 帧，耗时 {summary['elapsed_seconds']:.1f} 秒。单个模型候选 {summary['single_model_pose']} 帧；未检出 {summary['no_pose']} 帧；多个候选 {summary['multiple_poses']} 帧。")
    st.caption('只检出一个候选不证明画面中只有一人，模型也可能跟到背景人物；本版不识别身份、不跨帧确认同一人。请目视核对，勿将背景人物骨架当作训练者。')
    st.warning('骨架是未经核实的模型估计：遮挡、出框或拿取手机时，即使置信信号较高，也可能连到背景人物或器械上。请先选择稳定、单人全身可见的区间；本版尚不能自动排除这些错误。')
    st.dataframe([{'关键点':name,'未满足绘制条件的帧数':count,'总帧数':summary['frames']}
                  for name,count in summary['unavailable_joint_frames'].items()],hide_index=True)
    st.subheader('快速人工核对')
    st.write('选择要看的画面类型，再点上一个或下一个。重点看骨架是否跟对人、已显示的关节是否明显错位。完整显示的帧也可能识别错。')
    selection=st.selectbox('待核对画面',FILTERS,key='pose_review_filter')
    indices=review_indices(cached.frames,decisions,selection)
    current=st.session_state.get('pose_index',start)
    previous,following=neighbors(indices,current)
    first,prev,nxt=st.columns(3)
    first.button('从第一帧开始核对',key='review_first',disabled=not indices,on_click=go_review,args=(indices[0] if indices else start,))
    prev.button('上一个待核对帧',key='review_prev',disabled=previous is None,on_click=go_review,args=(previous,))
    nxt.button('下一个待核对帧',key='review_next',disabled=following is None,on_click=go_review,args=(following,))
    if not indices: st.info('所选区间没有符合此筛选条件的帧。可换一种类型，或手动查看任意帧。')
    else: st.caption(f'此类共 {len(indices)} 帧；当前帧'+('符合筛选。' if current in indices else '不符合筛选，点击按钮跳转。'))
    st.caption('这些按钮只改变下方静态核对帧，不控制上方连续播放器。')
    index = int(st.number_input('骨架核对源帧号',min_value=start,max_value=end,step=1,key='pose_index'))
    pose_frame = cached.frames[index-start]
    diagnostic = cached.diagnostics[index-start]
    decision = decisions[index-start]
    try:
        decoded = read_preview_frame(video,index,correct_mirror=mirror)
    except (MediaInputError,OSError):
        clear_pose()
        st.error('源视频已失效，本次关键点结果已清除，请重新打开。')
        return
    if diagnostic['status']=='multiple_poses':
        st.warning('此帧检出多个候选，没有自动选择训练者，骨架已隐藏。')
    elif diagnostic['status']=='no_pose':
        st.warning('此帧未检出人体关键点，未补点或插值。')
    left,right = st.columns(2)
    left.image(decoded.image_rgb,caption=f'源帧 {index} · {decoded.timestamp_ms/1000:.3f} 秒',width='stretch')
    if not decision['show']:
        st.warning('此帧骨架已隐藏：'+reason_text(decision))
    elif decision['hidden_parts']:
        st.warning('此帧部分显示；未显示：'+'、'.join(decision['hidden_parts'])+'。'+reason_text(decision))
    right.image(draw_pose(decoded.image_rgb,display_frame(pose_frame,decision)),caption='左侧青蓝圆点 · 右侧橙色方点；通过过滤的部位才绘制，线条不代表动作评价',width='stretch')
    reviews=st.session_state.setdefault('pose_reviews',{})
    st.write('此帧人工记录：'+reviews.get(index,'未记录'))
    st.caption('只记录显示情况，不判断拳术动作对错；全部隐藏时请选择“无法判断”。会话记录刷新、重识别或换片会清除；可在上方“保存或恢复人工核对记录”手动下载备份，不自动上传。')
    columns=st.columns(2)
    for n,verdict in enumerate(VERDICTS[1:]):
        columns[n%2].button(verdict,key=f'review_mark_{n}',on_click=record_review,args=(index,verdict),disabled=not decision['show'] and verdict!='无法判断')
    st.button('清除此帧记录',key='review_clear',on_click=record_review,args=(index,'未记录'))
    st.caption(f'本次已记录 {len(reviews)}/{len(cached.frames)} 帧；点过下一帧不会自动算已核对。不是准确率或专业标注。')
    with st.expander('此帧关键点与运行信息'):
        st.dataframe([{'关键点':name,'绘制':idx in decision['allowed_indices'],
                       'visibility':pose_frame['landmarks'][idx]['visibility'],
                       'presence':pose_frame['landmarks'][idx]['presence']} for idx,name in JOINTS.items()],hide_index=True)
        st.json({'model':cached.model,'runtime':cached.runtime,'config':cached.config,
                 'mirror_corrected':cached.mirror,'selected_source_frames':[cached.start,cached.end]})
    st.caption('默认绘制阈值为 visibility 与 presence 均不低于 0.5 且在画面内，只是待验证的工程选择。没有连线不等于动作错误，连线完整也不等于动作合格。')
