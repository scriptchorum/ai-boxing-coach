"""Session-only diagnostic UI for the pose extraction prototype."""
import streamlit as st

from .model_assets import AssetError
from .pose import JOINTS, LEFT_COLOR, RIGHT_COLOR, PoseError, draw_pose, drawable, extract_pose, run_key
from .preview import read_preview_frame
from .video_io import MediaInputError
from .synced_player import render_synced_player
from .quality import assess_frames, reason_text, display_frame, VERSION, LABELS
from .review import FILTERS, REACQUIRED, VERDICTS, review_indices, neighbors, summarize_reviews, export_reviews, import_reviews, read_review_file, MAX_REVIEW_BYTES
from .evidence import EvidenceError, MAX_DEMO_FRAMES, build_demo_report, build_evidence_bundle


def clear_pose():
    st.session_state.pop('pose_run',None)
    st.session_state.pop('pose_index',None)
    st.session_state.pop('pose_reviews',None)
    st.session_state.pop('pose_review_filter',None)
    st.session_state.pop('review_unrecorded_only',None)
    st.session_state.pop('pose_evidence_bundle',None)
    st.session_state.pop('pose_demo_frames',None)
    st.session_state.pop('pose_demo_report',None)


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


def render_review_download(cached, *, key):
    reviews=st.session_state.setdefault('pose_reviews',{})
    if reviews:
        st.caption(f'当前会话已记录 {len(reviews)} 帧。想在刷新后或另一台电脑继续时，可将这些判断备份为本地文件。')
    else:
        st.caption('先在“快速人工核对”记录至少一帧，再下载保存。')
    st.download_button('下载本次核对记录',data=export_reviews(cached,reviews),
                       file_name='boxing-display-review.private.json',mime='application/json',
                       key=key,disabled=not reviews,on_click='ignore')


def render_demo_report(cached, prepared, index):
    with st.expander(f'本地 Demo 报告（手动选择，最多 {MAX_DEMO_FRAMES} 帧）'):
        frames=st.session_state.get('pose_demo_frames',{})
        if not isinstance(frames,dict):
            frames={}
            st.session_state.pop('pose_demo_frames',None)
        if prepared is not None:
            if st.button('将当前证据帧加入 Demo 报告',key='demo_evidence_add',
                         disabled=index in frames or len(frames) >= MAX_DEMO_FRAMES):
                frames[index]=prepared['data']
                st.session_state.pose_demo_frames=frames
                st.session_state.pop('pose_demo_report',None)
        if index in frames and st.button('从 Demo 报告移除当前帧',key='demo_evidence_remove'):
            frames.pop(index)
            if not frames: st.session_state.pop('pose_demo_frames',None)
            st.session_state.pop('pose_demo_report',None)
        if st.button('清空 Demo 报告帧',key='demo_report_clear',disabled=not frames):
            frames.clear()
            st.session_state.pop('pose_demo_frames',None)
            st.session_state.pop('pose_demo_report',None)
        selected=sorted(frames)
        st.caption('已手动选择 '+('、'.join(map(str,selected)) if selected else '0')+
                   f' 个源帧（{len(selected)}/{MAX_DEMO_FRAMES}）。不会自动挑帧或判断动作。')
        report_key=(cached.key,tuple(selected))
        report=st.session_state.get('pose_demo_report')
        if report is not None and report['key'] != report_key:
            st.session_state.pop('pose_demo_report',None)
            report=None
        if st.button('准备本地 Demo 报告',key='demo_report_prepare',disabled=not frames):
            try:
                data=build_demo_report([frames[source_index] for source_index in sorted(frames)])
            except EvidenceError:
                st.error('所选证据帧不一致或已损坏，未生成报告；请清空后重新准备。')
            else:
                report={'key':report_key,'data':data}
                st.session_state.pose_demo_report=report
        if report is not None:
            st.download_button('下载本地 Demo 报告（私人 ZIP）',data=report['data'],
                               file_name='boxing-demo-report.private.zip',mime='application/zip',
                               key='demo_report_download',on_click='ignore')
        st.caption('报告含所选显示帧与模型骨架。请仅保存在受控目录，不要加入 Git 或公开分享；它不是动作评分、准确率或教练结论。')


def render_review_restore_hint():
    with st.expander('从核对文件查看原选段和镜像设置（可选）'):
        uploaded=st.file_uploader('选择之前保存的核对记录',type=['json'],key='review_import_file',max_upload_size=1)
        if uploaded is not None:
            try:
                if uploaded.size>MAX_REVIEW_BYTES: raise ValueError('核对文件超过 1 MiB，未导入。')
                payload=read_review_file(uploaded.getvalue())
            except ValueError as exc:
                st.error(str(exc))
            else:
                settings=payload.get('settings')
                if settings is None:
                    st.info('这是旧版记录，没有保存选段和镜像设置。请按原设置识别，仍可导入。')
                else:
                    st.info(f"文件记录的源帧：{settings['start_frame']}–{settings['end_frame']}（含两端）；镜像纠正："+('开启' if settings['mirror'] else '关闭')+'。')
                st.caption('这里只读取设置提示，不自动切换视频、修改设置或启动识别。请打开同一原片，按提示选择区间与镜像；识别后在保存/恢复区点击导入，系统仍会检查是否匹配。')
    return uploaded


def render_review_files(cached,uploaded):
    with st.expander('保存或恢复人工核对记录'):
        st.caption('文件含帧号、你的判断、选段、镜像设置和匹配标识，不含视频或骨架坐标。仍属于私人训练记录，请仅保存在受控目录，不加入 Git。刷新前请先下载；另一台需准备同一原片和相同识别设置。')
        st.caption(f'当前选段：源帧 {cached.start}–{cached.end}（含两端）；镜像纠正：'+('开启' if cached.mirror else '关闭')+'。恢复时需使用相同设置。')
        if uploaded is None: st.caption('要恢复记录，请先在上方“从核对文件查看原选段和镜像设置”选择文件。')
        if st.button('导入并替换当前核对记录',key='review_import',disabled=uploaded is None):
            try:
                if uploaded.size>MAX_REVIEW_BYTES: raise ValueError('核对文件超过 1 MiB，未导入。')
                restored=import_reviews(cached,uploaded.getvalue())
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state.pose_reviews=restored
                st.success(f'已恢复 {len(restored)} 帧记录；原视频和识别结果未修改。')
        render_review_download(cached,key='review_download')


def render_pose_panel(video,start,end,mirror):
    st.subheader('播放并查看骨架',anchor='pose-playback')
    st.write('先点击下方“识别所选区间的人体关键点”，完成后播放器会出现在这里。刷新或换片后需要重新识别。')
    st.write('对上方所选区间运行本地模型，再逐帧核对肩、肘、腕、髋、膝与踝。结果只保留在当前会话，不上传或写入训练历史。')
    st.caption('目前是工程试跑：模型信号不等于识别准确率，也不表示动作正确。女性身体多样性的正式验证仍待完成。每次最多 30 秒、1800 帧。')
    uploaded=render_review_restore_hint()
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
    render_review_files(cached,uploaded)
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
    reviews=st.session_state.setdefault('pose_reviews',{})
    wrong_target,misaligned=VERDICTS[2:4]
    problem_frames=sorted(i for i,v in reviews.items() if v in (wrong_target,misaligned))
    with st.expander('复查已标记的显示问题'):
        st.caption(f'当前选段的人工记录中，跟错人或落在器械上 {sum(v==wrong_target for v in reviews.values())} 帧，关节明显错位 {sum(v==misaligned for v in reviews.values())} 帧。')
        if problem_frames:
            selected=st.selectbox('选择问题帧',problem_frames,format_func=lambda i:f'源帧 {i} · {reviews[i]}')
            st.button('查看所选问题帧',key='review_issue_jump',on_click=go_review,args=(selected,))
        else:
            st.caption('目前没有标记这两类问题的帧。')
    review_summary=summarize_reviews(cached,reviews)
    with st.expander('查看人工核对汇总'):
        st.caption(f"已记录 {review_summary['reviewed_frames']}/{review_summary['total_frames']} 帧；其中显示问题 {review_summary['problem_frames']} 帧。下列数量只是人工操作记录，不是准确率或专业标注。")
        st.dataframe([{'\u4eba工判断':verdict,'\u5e27数':count}
                      for verdict,count in review_summary['verdict_counts'].items()],hide_index=True)
        st.dataframe([{'\u8bb0录时的显示状态':state,'\u5e27数':count}
                      for state,count in review_summary['display_counts'].items()],hide_index=True)
        if review_summary['issue_reason_counts']:
            st.write('显示问题帧与当前工程规则的对照：')
            st.dataframe([{'\u5f53前规则原因':LABELS.get(code,'未触发当前隐藏原因'),'\u5e27数':count}
                          for code,count in review_summary['issue_reason_counts'].items()],hide_index=True)
            st.caption('同一帧可同时触发多个原因，各项不能直接相加；“未触发”表示人工看到问题时，当前规则没有将该帧拒绝。')
    st.write('选择要看的画面类型，再点上一个或下一个。重点看骨架是否跟对人、已显示的关节是否明显错位。完整显示的帧也可能识别错。')
    selection=st.selectbox('待核对画面',FILTERS,key='pose_review_filter')
    if selection==REACQUIRED:
        st.caption('这些是本选段内出现多个候选后，骨架第一次重新显示的帧。对照原片检查骨架是否仍跟着训练者。')
    matching=review_indices(cached.frames,decisions,selection)
    remaining=[i for i in matching if i not in reviews]
    only_unrecorded=st.checkbox('只看未记录帧',key='review_unrecorded_only')
    indices=remaining if only_unrecorded else matching
    st.caption(f'此类已记录 {len(matching)-len(remaining)}/{len(matching)} 帧，剩余 {len(remaining)} 帧。记录“无法判断”也算已记录，不代表识别正确。')
    current=st.session_state.get('pose_index',start)
    previous,following=neighbors(indices,current)
    first,prev,nxt=st.columns(3)
    first.button('从第一帧开始核对',key='review_first',disabled=not indices,on_click=go_review,args=(indices[0] if indices else start,))
    prev.button('上一个待核对帧',key='review_prev',disabled=previous is None,on_click=go_review,args=(previous,))
    nxt.button('下一个待核对帧',key='review_next',disabled=following is None,on_click=go_review,args=(following,))
    if matching and only_unrecorded and not remaining:
        st.info('此类帧均已有记录。取消“只看未记录帧”可复查，或切换画面类型继续。')
    elif not indices: st.info('所选区间没有符合此筛选条件的帧。可换一种类型，或手动查看任意帧。')
    else: st.caption(f'当前可跳转 {len(indices)} 帧；当前帧'+('符合筛选。' if current in indices else '不符合筛选，点击按钮跳转。'))
    st.caption('这些按钮只改变下方静态核对帧，不控制上方连续播放器。记录后画面保留，点击前后按钮再跳转；取消筛选可修改已有记录。')
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
    overlay=draw_pose(decoded.image_rgb,display_frame(pose_frame,decision))
    right.image(overlay,caption='左侧青蓝圆点 · 右侧橙色方点；通过过滤的部位才绘制，线条不代表动作评价',width='stretch')
    evidence_key=(cached.key,index,VERSION)
    prepared=st.session_state.get('pose_evidence_bundle')
    if prepared is not None and prepared['key'] != evidence_key:
        st.session_state.pop('pose_evidence_bundle',None)
        prepared=None
    st.caption('如需留存这一帧，可在本机准备私人证据包；其中包含上面两张图片、源帧号、PTS、模型和显示规则摘要，不会自动上传。')
    if st.button('准备当前证据帧包',key='evidence_prepare'):
        try:
            data=build_evidence_bundle(cached,pose_frame,diagnostic,decision,
                                       decoded.image_rgb,overlay,
                                       source_timestamp_ms=decoded.timestamp_ms)
        except EvidenceError:
            st.error('当前帧与识别结果不一致，未生成证据包；请重新打开视频并识别。')
        else:
            prepared={'key':evidence_key,'data':data}
            st.session_state.pose_evidence_bundle=prepared
    if prepared is not None:
        st.download_button('下载当前证据帧包（私人 ZIP）',data=prepared['data'],
                           file_name='boxing-evidence-frame.private.zip',mime='application/zip',
                           key='evidence_download',on_click='ignore')
        st.caption('该 ZIP 含视频画面和模型骨架，属于私人训练资料；请保存在受控目录，不要加入 Git 或公开分享。')
    render_demo_report(cached,prepared,index)
    reviews=st.session_state.setdefault('pose_reviews',{})
    st.write('此帧人工记录：'+reviews.get(index,'未记录'))
    st.caption('只记录显示情况，不判断拳术动作对错；全部隐藏时请选择“无法判断”。会话记录刷新、重识别或换片会清除；可在上方“保存或恢复人工核对记录”手动下载备份，不自动上传。')
    columns=st.columns(2)
    for n,verdict in enumerate(VERDICTS[1:]):
        columns[n%2].button(verdict,key=f'review_mark_{n}',on_click=record_review,args=(index,verdict),disabled=not decision['show'] and verdict!='无法判断')
    st.button('清除此帧记录',key='review_clear',on_click=record_review,args=(index,'未记录'))
    st.caption(f'本次已记录 {len(reviews)}/{len(cached.frames)} 帧；点过下一帧不会自动算已核对。不是准确率或专业标注。')
    render_review_download(cached,key='review_download_inline')
    with st.expander('此帧关键点与运行信息'):
        st.dataframe([{'关键点':name,'绘制':idx in decision['allowed_indices'],
                       'visibility':pose_frame['landmarks'][idx]['visibility'],
                       'presence':pose_frame['landmarks'][idx]['presence']} for idx,name in JOINTS.items()],hide_index=True)
        st.json({'model':cached.model,'runtime':cached.runtime,'config':cached.config,
                 'mirror_corrected':cached.mirror,'selected_source_frames':[cached.start,cached.end]})
    st.caption('默认绘制阈值为 visibility 与 presence 均不低于 0.5 且在画面内，只是待验证的工程选择。没有连线不等于动作错误，连线完整也不等于动作合格。')
