"""Session-only diagnostic UI for the pose extraction prototype."""
import streamlit as st

from .model_assets import AssetError
from .pose import JOINTS, LEFT_COLOR, RIGHT_COLOR, PoseError, draw_pose, drawable, extract_pose, run_key
from .preview import read_preview_frame
from .video_io import SUPPORTED_EXTENSIONS, MediaInputError
from .synced_player import render_synced_player
from .quality import assess_frames, reason_text, display_frame, VERSION, LABELS
from .review import FILTERS, REACQUIRED, VERDICTS, recovery_references, review_indices, neighbors, summarize_reviews, export_reviews, import_reviews, read_review_file, MAX_REVIEW_BYTES
from .evidence import EvidenceError, MAX_DEMO_FRAMES, build_demo_report, build_evidence_bundle
from .local_storage import (
    check_offline_readiness,
    inspect_local_storage,
    load_persistent_history,
    purge_local_caches_and_reports,
    save_persistent_history,
)
from .muaythai_metrics import analyze_muaythai_arsenal
from .scorecard_report import (
    batch_review_video_files,
    build_standalone_scorecard_html,
    extract_keyframe_previews,
)
from .rear_straight_review import (
    analyze_jab_and_one_two,
    apply_action_override,
    build_history_entry,
    compare_same_condition_history,
    compute_rear_straight_metrics,
    export_review_annotation_json,
    segment_rear_straights,
)
from .strike_detector import StrikeAuditor, export_strike_audit_json
from .subject_tracker import SubjectTracker


def clear_pose():
    st.session_state.pop('pose_run',None)
    st.session_state.pop('pose_index',None)
    st.session_state.pop('pose_reviews',None)
    st.session_state.pop('pose_review_filter',None)
    st.session_state.pop('review_unrecorded_only',None)
    st.session_state.pop('pose_evidence_bundle',None)
    st.session_state.pop('pose_strike_audit',None)
    st.session_state.pop('pose_rear_straight_overrides',None)
    st.session_state.pop('pose_review_annotation',None)
    st.session_state.pop('pose_scorecard_html',None)
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
    st.caption('颜色对应模型标记的人体左右侧。镜像纠正后标签颜色保持一致，交叉和侧身时请对照原片核对左右与点位。')


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
        st.caption('报告汇总所选显示帧和模型骨架，用于本地逐帧核对。请保存在受控目录，分享前确认素材授权。')


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


def render_strike_audit_section(cached, tracker_enabled):
    with st.expander('泰拳击打与防守动作审计（前/后手直拳峰值、护头掉落率与转髋角度）', expanded=True):
        st.caption('基于姿态关键点在时间线上的伸展峰值审计直拳动作，并根据质量门禁（单人确认、关键点可见度）输出客观观察结果。')
        c1, c2 = st.columns(2)
        stance_label = c1.selectbox('站架选择', ['正架 (右手后手 / Orthodox)', '反架 (左手后手 / Southpaw)'], key='audit_stance_select')
        stance = 'orthodox' if 'Orthodox' in stance_label else 'southpaw'
        single_person = c2.checkbox('目视确认：训练区域无背景人物干扰 (解锁合规指标计算)', value=True, key='audit_single_person_confirmed')

        auditor = StrikeAuditor(stance=stance)
        audit_res = auditor.audit_pose_run(cached, single_person_confirmed=single_person, tracker=tracker_enabled)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric('检测到局部伸展峰值', f"{audit_res['candidate_strike_count']} 次")
        m2.metric('门禁合规击打', f"{audit_res['eligible_strike_count']} 次")
        m3.metric('前手 / 后手直拳', f"{audit_res['lead_strikes']} / {audit_res['rear_strikes']}")
        guard_rate = audit_res.get('guard_drop_rate')
        m4.metric('后手护头掉落率', f"{guard_rate * 100:.1f}%" if guard_rate is not None else "待门禁校验")

        st.markdown('**观察结论与技术反馈：**')
        if audit_res.get('observations'):
            for obs in audit_res['observations']:
                st.info(obs)
        else:
            st.caption('当前所选区间内未检测到符合门禁条件的有效击打或指标数据。')

        gated_strikes = audit_res.get('strikes', [])
        if gated_strikes:
            st.markdown('**击打事件列表（点击可跳转至击打定格帧）：**')
            strike_rows = []
            for s in gated_strikes:
                st_type_name = '前手直拳 (Jab)' if s['strike_type'] == 'lead_straight' else '后手直拳 (Cross)'
                strike_rows.append({
                    '源帧号': s['frame_index'],
                    '时间 (秒)': f"{s['timestamp_ms'] / 1000:.3f}",
                    '动作类型': st_type_name,
                    '门禁状态': s['status'],
                    '伸展比率': f"{s['extension_ratio']:.2f}" if s.get('extension_ratio') is not None else "-",
                    '后手护头状态': '掉落/过低' if s.get('rear_guard_dropped') else ('保持' if s.get('rear_guard_safe') else '-'),
                    '双髋旋转角': f"{s.get('hip_rotation_deg', 0):.1f}°" if s.get('hip_rotation_deg') is not None else "-",
                })
            st.dataframe(strike_rows, hide_index=True)

            strike_indices = [s['frame_index'] for s in gated_strikes]
            sel_strike_frame = st.selectbox('选择要复核的击打瞬间帧', strike_indices, format_func=lambda i: f"源帧 {i}", key='sel_strike_frame_jump')
            st.button('跳转至该击打峰值帧', key='jump_strike_frame_btn', on_click=go_review, args=(sel_strike_frame,))

        audit_key = (cached.key, stance, bool(single_person), bool(tracker_enabled))
        if st.button('生成击打与门禁审计摘要', key='strike_audit_prepare'):
            st.session_state.pose_strike_audit = {
                'key': audit_key,
                'json': export_strike_audit_json(audit_res),
            }
        saved_audit = st.session_state.get('pose_strike_audit')
        if saved_audit and saved_audit.get('key') == audit_key:
            st.download_button(
                '下载击打与门禁审计摘要（JSON）',
                data=saved_audit['json'],
                file_name='boxing-strike-audit.json',
                mime='application/json',
                key='strike_audit_download',
                on_click='ignore',
            )


def render_rear_straight_review_section(video, cached, mirror):
    st.subheader('🎯 后手直拳专项复盘：最高优先建议与回防评分')
    with st.expander('后手直拳自动分段、回防评分、最高优先建议与同条件历史复盘 (T08–T11)', expanded=True):
        st.caption(
            '自动完成“起手→最大伸展→回手进入个人起始区→连续保持 ≥100ms”四阶段分段，计算回防评分、最高优先建议、回手时间、'
            '同场次重复性（中位数/四分位距）与二维髋部投影偏移比；无需从零手标视频，可按需对单拳帧号做可选微调。'
        )
        c1, c2 = st.columns(2)
        stance_opt = c1.selectbox(
            '复盘站架',
            ['正架 (右手后手 / Orthodox)', '反架 (左手后手 / Southpaw)'],
            key='rs_stance_select',
        )
        stance = 'orthodox' if 'Orthodox' in stance_opt else 'southpaw'
        view_opt = c2.selectbox('机位记录', ['侧面 (side)', '前斜 45° (front_45)'], key='rs_view_select')
        view = 'side' if 'side' in view_opt else 'front_45'
        single_ok = st.checkbox(
            '确认当前片段为单人训练（启用回防评分、回手时间与重复性统计）',
            value=True,
            key='rs_single_person_confirmed',
        )

        width = getattr(video.inspection, 'width', 1280)
        height = getattr(video.inspection, 'height', 720)
        seg = segment_rear_straights(
            cached.frames,
            width=width,
            height=height,
            stance=stance,
            sample_id=f'run-{cached.start}-{cached.end}',
        )

        overrides = st.session_state.setdefault('pose_rear_straight_overrides', {})
        for act_id, ov in list(overrides.items()):
            if any(a['action_id'] == act_id for a in seg['actions']):
                seg = apply_action_override(
                    seg,
                    cached.frames,
                    act_id,
                    start_frame_index=ov.get('start_frame_index'),
                    extension_frame_index=ov.get('extension_frame_index'),
                    return_frame_index=ov.get('return_frame_index'),
                    hold_end_frame_index=ov.get('hold_end_frame_index'),
                )

        metrics = compute_rear_straight_metrics(seg, single_person_confirmed=single_ok)
        rep = metrics.get('repeatability')
        observed_times = [
            float(m['return_time_ms'])
            for m in metrics.get('action_metrics', [])
            if m.get('return_time_ms') is not None
        ]
        if rep:
            median_label = f"{rep['median_ms']:.1f} ms"
            iqr_label = f"{rep['iqr_ms']:.1f} ms"
        elif observed_times:
            sorted_t = sorted(observed_times)
            mid = len(sorted_t) // 2
            med_val = sorted_t[mid] if len(sorted_t) % 2 == 1 else (sorted_t[mid - 1] + sorted_t[mid]) / 2.0
            median_label = f"{med_val:.1f} ms（{len(observed_times)} 拳）"
            iqr_label = '需 ≥5 拳算 IQR'
        else:
            median_label = '暂无回手数据'
            iqr_label = '-'
        k0, k1, k2, k3, k4 = st.columns(5)
        k0.metric('回防评分', metrics.get('defense_grade', '暂无'))
        k1.metric('候选后手直拳', f"{metrics['candidate_count']} 次")
        k2.metric('有效完整动作', f"{metrics['valid_count']} 次")
        k3.metric('同场次中位回手时间', median_label)
        k4.metric('同场次四分位距 (IQR)', iqr_label)

        st.markdown('**💡 本段最高优先建议（Top 1）：**')
        score_val = metrics.get('defense_score')
        if score_val is not None and score_val >= 80:
            st.success(metrics['top_priority_advice'])
        else:
            st.warning(metrics['top_priority_advice'])
        top_frame = metrics.get('top_priority_frame_index')
        if top_frame is not None:
            st.button(
                f'定位至最高优先建议对应帧（源帧 {top_frame}）',
                key='rs_jump_top_advice_btn',
                on_click=go_review,
                args=(int(top_frame),),
            )

        drill = metrics.get('drill_prescription')
        if drill:
            st.markdown(f"**🏋️ 针对性纠错训练处方（B01）—— {drill['title']}**")
            st.info(
                f"**针对问题**：{drill['target_issue']}  \n"
                f"**推荐训练量**：{drill['dosage']}  \n"
                f"**执行要点**：{drill['steps']}  \n"
                f"**自查口令**：{drill['cue']}"
            )

        combo_res = analyze_jab_and_one_two(
            cached.frames,
            rear_segmentation=seg,
            width=width,
            height=height,
            stance=stance,
            sample_id=f'run-{cached.start}-{cached.end}',
            single_person_confirmed=single_ok,
        )
        st.markdown('**🥊 全拳法协同复盘：刺拳 (Jab)、平勾 (Hook)、上勾 (Uppercut) 与 1-2 / 1-2-3 连击（B02）：**')
        j1, j2, j3, j4 = st.columns(4)
        j1.metric('前手刺拳 (Jab)', f"{combo_res['jab_attempt_count']} 次（完整回手 {combo_res['jab_valid_count']} 次）")
        j2.metric('前手刺拳回防评分', combo_res['jab_defense_grade'])
        j3.metric('平勾拳 / 上勾拳', f"{combo_res.get('hook_count', 0)} 勾 / {combo_res.get('uppercut_count', 0)} 上勾")
        j4.metric('1-2 / 1-2-3 连击', f"{combo_res['combo_count']} 组 / {combo_res.get('one_two_three_count', 0)} 组")
        st.caption(combo_res['combo_feedback'])
        if combo_res['one_two_combos']:
            st.dataframe(
                [
                    {
                        '连击编号': c['combo_id'],
                        '刺拳最大伸展帧': c['jab_extension_frame_index'],
                        '后手最大伸展帧': c['rear_extension_frame_index'],
                        '1-2 衔接间隔 (ms)': f"{c['combo_interval_ms']:.1f}",
                        '刺拳回手 (ms)': f"{c['jab_return_time_ms']:.1f}" if c['jab_return_time_ms'] is not None else '-',
                        '后手回手 (ms)': f"{c['rear_return_time_ms']:.1f}" if c['rear_return_time_ms'] is not None else '-',
                        '双拳完整回防': '是' if c['both_returned'] else '否',
                    }
                    for c in combo_res['one_two_combos']
                ],
                hide_index=True,
            )
        if combo_res.get('one_two_three_combos'):
            st.dataframe(
                [
                    {
                        '三连击编号': c3['combo_id'],
                        '组合类型': c3['combo_name'],
                        '刺拳峰值帧': c3['jab_frame_index'],
                        '后手峰值帧': c3['rear_frame_index'],
                        '第3拳峰值帧': c3['third_punch_frame_index'],
                        '2→3 衔接间隔 (ms)': f"{c3['two_to_three_ms']:.1f}",
                        '全套回防': '是' if c3['all_returned'] else '否',
                    }
                    for c3 in combo_res['one_two_three_combos']
                ],
                hide_index=True,
            )

        mt_res = analyze_muaythai_arsenal(
            cached.frames,
            stance=stance,
            single_person_confirmed=single_ok,
        )
        st.markdown('**🦵 泰拳专项：高抱架、扫踢、冲膝、提膝格挡与拳腿立体连击（B03）：**')
        m1, m2, m3, m4 = st.columns(4)
        hg_rate = mt_res.get('high_guard_retention_rate')
        m1.metric('泰拳高抱架保持率', f'{hg_rate * 100:.0f}%' if hg_rate is not None else '暂无')
        m2.metric('泰式扫踢 / 冲膝', f"{mt_res['kick_count']} 腿 / {mt_res['knee_count']} 膝")
        m3.metric('提膝格挡 (Shin Check)', f"{mt_res['shin_check_count']} 次")
        m4.metric('拳腿/拳膝立体连击', f"{mt_res['combo_count']} 组")
        st.caption(mt_res['muaythai_feedback'])
        if mt_res['punch_leg_combos']:
            st.dataframe(
                [
                    {
                        '连击编号': c['combo_id'],
                        '立体组合': c['combo_name'],
                        '拳法峰值帧': c['punch_frame_index'],
                        '腿膝峰值帧': c['leg_frame_index'],
                        '拳腿转换间隔 (ms)': f"{c['transition_ms']:.1f}",
                        '起腿高抱架保持': '是' if c['guard_kept_on_leg'] else '否',
                    }
                    for c in mt_res['punch_leg_combos']
                ],
                hide_index=True,
            )
        if mt_res['leg_events']:
            st.dataframe(
                [
                    {
                        '动作编号': e['event_id'],
                        '泰拳技术': e['technique_name'],
                        '峰值源帧': e['frame_index'],
                        '收腿回位 (ms)': f"{e['return_time_ms']:.1f}" if e['return_time_ms'] is not None else '-',
                        '高抱架护头': '保持' if e['guard_kept'] else '下掉',
                    }
                    for e in mt_res['leg_events']
                ],
                hide_index=True,
            )

        st.caption(metrics['limitations_notice'])

        act_rows = []
        for item in metrics['action_metrics']:
            act_rows.append({
                '动作编号': item['action_id'],
                '来源': '人工微调' if item['source'] == 'manual' else '自动预标',
                '状态': item['status'],
                '起手帧': item['start_frame_index'],
                '最大伸展帧': item['extension_frame_index'],
                '回手进入帧': item['return_frame_index'] if item['return_frame_index'] is not None else '-',
                '保持完成帧': item['hold_end_frame_index'] if item['hold_end_frame_index'] is not None else '-',
                '回手时间 (ms)': f"{item['return_time_ms']:.1f}" if item['return_time_ms'] is not None else '-',
                '二维髋偏移比': (
                    f"{item['torso_projection_shift_ratio']:.3f}"
                    if item['torso_projection_shift_ratio'] is not None
                    else '-'
                ),
                '原因说明': '、'.join(item['quality_reasons']) or '完整合规',
            })
        if act_rows:
            st.dataframe(act_rows, hide_index=True)
            action_ids = [a['action_id'] for a in seg['actions']]
            sel_act_id = st.selectbox('选择要定位或微调的动作（可选）', action_ids, key='rs_select_action')
            sel_act = next(a for a in seg['actions'] if a['action_id'] == sel_act_id)
            if sel_act['extension_frame_index'] is not None:
                st.button(
                    '跳转至该拳最大伸展帧',
                    key='rs_jump_extension_btn',
                    on_click=go_review,
                    args=(int(sel_act['extension_frame_index']),),
                )
            if st.checkbox('🔧 展开手动微调该拳的四阶段帧号（可选 · T08）', value=False, key='rs_enable_override_ui'):
                ov_c1, ov_c2, ov_c3, ov_c4 = st.columns(4)
                ov_start = int(ov_c1.number_input(
                    '起手帧', min_value=cached.start, max_value=cached.end,
                    value=int(sel_act['start_frame_index'] if sel_act['start_frame_index'] is not None else cached.start),
                    step=1, key='rs_ov_start',
                ))
                ov_ext = int(ov_c2.number_input(
                    '最大伸展帧', min_value=cached.start, max_value=cached.end,
                    value=int(sel_act['extension_frame_index'] if sel_act['extension_frame_index'] is not None else cached.start),
                    step=1, key='rs_ov_ext',
                ))
                ov_ret = int(ov_c3.number_input(
                    '回手进入帧', min_value=cached.start, max_value=cached.end,
                    value=int(sel_act['return_frame_index'] if sel_act['return_frame_index'] is not None else ov_ext),
                    step=1, key='rs_ov_ret',
                ))
                ov_hold = int(ov_c4.number_input(
                    '保持完成帧', min_value=cached.start, max_value=cached.end,
                    value=int(sel_act['hold_end_frame_index'] if sel_act['hold_end_frame_index'] is not None else ov_ret),
                    step=1, key='rs_ov_hold',
                ))
                ob1, ob2 = st.columns(2)
                if ob1.button('保存并应用该拳帧号微调', key='rs_apply_override_btn'):
                    overrides[sel_act_id] = {
                        'start_frame_index': ov_start,
                        'extension_frame_index': ov_ext,
                        'return_frame_index': ov_ret,
                        'hold_end_frame_index': ov_hold,
                    }
                    st.session_state.pose_rear_straight_overrides = overrides
                    st.rerun()
                if ob2.button('恢复该拳自动预标', key='rs_reset_override_btn'):
                    overrides.pop(sel_act_id, None)
                    st.session_state.pose_rear_straight_overrides = overrides
                    st.rerun()
        else:
            st.caption('当前区间未检出后手直拳候选动作。')

        ann_key = (cached.key, stance, tuple(sorted(overrides.keys())))
        e_col1, e_col2 = st.columns(2)
        if e_col1.button('准备完整图文复盘评分报告（离线 HTML）', key='scorecard_html_prepare'):
            vid_name = getattr(getattr(video, 'path', None), 'name', None)
            clip_label = f'{vid_name} (源帧 {cached.start}–{cached.end})' if vid_name else f'源帧 {cached.start}–{cached.end}'
            kf_previews = extract_keyframe_previews(
                video,
                cached,
                mirror=bool(mirror),
                rear_metrics=metrics,
                combo_res=combo_res,
                mt_res=mt_res,
            )
            st.session_state.pose_scorecard_html = {
                'key': ann_key,
                'html': build_standalone_scorecard_html(
                    clip_label=clip_label,
                    stance=stance,
                    view=view,
                    rear_metrics=metrics,
                    combo_res=combo_res,
                    mt_res=mt_res,
                    keyframe_previews=kf_previews,
                ),
            }
        saved_sc = st.session_state.get('pose_scorecard_html')
        if saved_sc and saved_sc.get('key') == ann_key:
            e_col1.download_button(
                '下载完整图文复盘评分报告（离线 HTML）',
                data=saved_sc['html'],
                file_name='boxing-training-scorecard.html',
                mime='text/html',
                key='scorecard_html_download',
                on_click='ignore',
            )

        if e_col2.button('准备事件标注与分段 JSON（T08/T09）', key='review_annotation_prepare'):
            st.session_state.pose_review_annotation = {
                'key': ann_key,
                'json': export_review_annotation_json(seg),
            }
        saved_ann = st.session_state.get('pose_review_annotation')
        if saved_ann and saved_ann.get('key') == ann_key:
            e_col2.download_button(
                '下载事件标注与分段 JSON',
                data=saved_ann['json'],
                file_name='boxing-rear-straight-annotation.json',
                mime='application/json',
                key='review_annotation_download',
                on_click='ignore',
            )

        if 'pose_local_history' not in st.session_state:
            st.session_state.pose_local_history = load_persistent_history()
        history = st.session_state.pose_local_history
        h_col1, h_col2 = st.columns(2)
        if h_col1.button('将本次统计加入本地同条件历史对比（T11）', key='rs_save_history_btn'):
            vid_name = getattr(getattr(video, 'path', None), 'name', None)
            label_str = f'{vid_name} (源帧 {cached.start}–{cached.end})' if vid_name else f'源帧 {cached.start}–{cached.end}'
            entry = build_history_entry(
                metrics,
                entry_id=f'hist-{len(history) + 1}',
                label=label_str,
                mirror=mirror,
                view=view,
                model_id=cached.model.get('id', 'mediapipe_pose_landmarker_lite'),
            )
            history.append(entry)
            st.session_state.pose_local_history = history
            try:
                save_persistent_history(history)
            except OSError:
                pass

        vid_path_obj = getattr(video, 'path', None)
        sibling_mp4s = []
        if vid_path_obj is not None and getattr(vid_path_obj, 'parent', None) is not None and vid_path_obj.parent.is_dir():
            sibling_mp4s = sorted(
                p for p in vid_path_obj.parent.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        if len(sibling_mp4s) > 1:
            if h_col2.button(
                f'📚 一键批量复盘同目录全部视频（共 {len(sibling_mp4s)} 个）',
                key='batch_folder_review_btn',
            ):
                b_bar = st.progress(0, text='正在批量复盘同目录 MP4 视频…')
                batch_res = batch_review_video_files(
                    sibling_mp4s,
                    stance=stance,
                    view=view,
                    mirror=bool(mirror),
                    progress_callback=lambda done, tot, name: b_bar.progress(
                        done / max(tot, 1), text=f'批量复盘进度 {done}/{tot}：{name}'
                    ),
                )
                st.session_state.pose_batch_table = batch_res['rows']
                existing_labels = {h.get('label') for h in history}
                for he in batch_res['history_entries']:
                    if he.get('label') not in existing_labels:
                        he['entry_id'] = f'hist-{len(history) + 1}'
                        history.append(he)
                        existing_labels.add(he.get('label'))
                st.session_state.pose_local_history = history
                try:
                    save_persistent_history(history)
                except OSError:
                    pass

        batch_rows = st.session_state.get('pose_batch_table')
        if batch_rows:
            st.markdown('**📊 同文件夹多视频横向复盘对比总表：**')
            st.dataframe(batch_rows, hide_index=True)

        if history:
            comp = compare_same_condition_history(
                history,
                reference_condition={
                    'stance': stance,
                    'mirror': bool(mirror),
                    'view': view,
                    'model_id': cached.model.get('id', 'mediapipe_pose_landmarker_lite'),
                    'rule_version': metrics['rule_version'],
                },
            )
            st.caption(comp['comparison_note'])
            if comp['compatible_entries']:
                st.dataframe(comp['compatible_entries'], hide_index=True)


def render_local_storage_section(video):
    with st.expander('离线就绪状态与本地缓存/报告清理（保留原片 · T12）', expanded=False):
        readiness = check_offline_readiness()
        if readiness['offline_ready']:
            st.caption(
                f"离线状态：模型资产 `{readiness['asset_id']}` 已就绪并通过 SHA-256 校验，断网可直接运行。"
            )
        else:
            st.caption(f"离线状态提示：{readiness['reason']}")
        protected = [video.path] if getattr(video, 'path', None) is not None else []
        storage_info = inspect_local_storage(protected_source_paths=protected)
        st.caption(
            f"受管缓存目录（outputs/、runs/、work/matplotlib-cache/）当前可清理派生文件："
            f"{storage_info['removable_files']} 个（{storage_info['removable_bytes']} 字节）；"
            f"已自动保护原片与模型资产。"
        )
        if st.button('清理本地派生缓存与报告（不删原片）', key='purge_local_caches_btn'):
            res = purge_local_caches_and_reports(protected_source_paths=protected)
            st.session_state.pop('pose_evidence_bundle', None)
            st.session_state.pop('pose_strike_audit', None)
            st.session_state.pop('pose_review_annotation', None)
            st.session_state.pop('pose_scorecard_html', None)
            st.session_state.pop('pose_batch_table', None)
            st.session_state.pop('pose_demo_frames', None)
            st.session_state.pop('pose_demo_report', None)
            st.session_state.pop('pose_local_history', None)
            st.success(
                f"已清理 {res['deleted_files']} 个派生缓存文件（{res['deleted_bytes']} 字节），"
                f"保留受保护文件 {res['preserved_protected_files']} 个；用户导入原片未受影响。"
            )


def render_pose_panel(video,start,end,mirror):
    st.subheader('📊 动作复盘与骨架联动工作台',anchor='pose-playback')
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
    auto_run = bool(st.session_state.pop('auto_run_pose', False))
    manual_run = st.button('识别所选区间的人体关键点',key='run_pose')
    if manual_run or (auto_run and cached is None):
        clear_pose()
        bar = st.progress(0,text='准备模型并自动提取动作复盘数据…')
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
        st.info('尚未生成复盘数据。点击上方【识别所选区间的人体关键点】（或在顶部直接点【🚀 一键打开并直接生成动作复盘】），处理完成后即可直接查看评分与建议。')
        return

    top_c1, top_c2, top_c3 = st.columns([2, 2, 1])
    enable_tracker = top_c1.checkbox('试用跟踪过滤（可能隐藏正确骨架）', value=False, key='trial_tracking_filter')
    allow_upper = top_c2.checkbox('半身/近景模式：下肢或髋部未入画时仍显示上肢骨架', value=False, key='allow_upper_body_skeleton')
    top_c3.button('清除本次关键点结果',on_click=clear_pose)

    # 1. Core Coaching & Scoring Dashboard at the very top
    render_rear_straight_review_section(video, cached, mirror)
    render_strike_audit_section(cached, enable_tracker)

    # 2. Synced Video & Skeleton Player
    decisions = assess_frames(cached, tracker=enable_tracker, allow_upper_body=allow_upper)
    st.subheader('🎬 视频与骨架同步回放')
    st.write('点击下面左侧视频的 ▶ 播放，右侧骨架随画面变化；暂停或拖动也可核对。半身视频请勾选上方“半身/近景模式”以显示上肢骨架。')
    render_side_legend()
    try:
        read_preview_frame(video,start,correct_mirror=mirror)
        render_synced_player(video,mirror,cached,decisions=decisions)
    except ValueError as exc:
        st.warning(str(exc))
    except (MediaInputError,OSError):
        clear_pose()
        st.error('源视频已失效，请重新打开并识别。')
        return
    st.info(f"分部位显示：{sum(not d['show'] for d in decisions)} 帧全部隐藏，{sum(d['show'] and bool(d['hidden_parts']) for d in decisions)} 帧仅显示部分身体，共 {len(decisions)} 帧。隐藏不表示动作错误，未隐藏也不证明识别准确。")

    # 3. Secondary Engineering & Frame-by-Frame Inspection Tools (Collapsed by default)
    render_local_storage_section(video)
    render_review_files(cached,uploaded)
    with st.expander('🛠 高级逐帧核对、隐藏原因统计与私人证据包导出（按需展开）', expanded=False):
        st.caption('躯干定位不足、多人或位置突变时全部隐藏；否则分别检查左右手臂和腿，缺一处不再让其他部位一起消失。这是待验证规则，不能认人，也不能排除背景人物或器械误检。')
        st.dataframe([{'原因':label,'帧数':sum(code in d['reasons'] for d in decisions),'总帧数':len(decisions)} for code,label in LABELS.items()],hide_index=True)
        st.caption(f'规则版本 {VERSION}；同一帧可有多个原因，各项不能直接相加。')
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
        st.caption(f'当前选段的人工记录中，跟错人或落在器械上 {sum(v==wrong_target for v in reviews.values())} 帧，关节明显错位 {sum(v==misaligned for v in reviews.values())} 帧。')
        if problem_frames:
            selected=st.selectbox('选择问题帧',problem_frames,format_func=lambda i:f'源帧 {i} · {reviews[i]}')
            st.button('查看所选问题帧',key='review_issue_jump',on_click=go_review,args=(selected,))
        else:
            st.caption('目前没有标记这两类问题的帧。')
        review_summary=summarize_reviews(cached,reviews)
        st.caption(f"已记录 {review_summary['reviewed_frames']}/{review_summary['total_frames']} 帧；其中显示问题 {review_summary['problem_frames']} 帧。下列数量汇总本次人工核对进度与显示问题。")
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
        references = recovery_references(cached.frames, decisions)
        matching = list(references) if selection == REACQUIRED else review_indices(cached.frames, decisions, selection)
        remaining=[i for i in matching if i not in reviews]
        only_unrecorded=st.checkbox('只看未记录帧',key='review_unrecorded_only')
        indices=remaining if only_unrecorded else matching
        st.caption(f'此类已记录 {len(matching)-len(remaining)}/{len(matching)} 帧，剩余 {len(remaining)} 帧。进度包含“无法判断”，各类判断分别汇总。')
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
        overlay=draw_pose(decoded.image_rgb,display_frame(pose_frame,decision))
        if index in references:
            st.subheader('骨架恢复前后对照')
            reference = references[index]
            if reference is None:
                st.info('本选段在这次恢复前没有显示过骨架，可用下方原画核对当前帧。')
            else:
                try:
                    earlier = read_preview_frame(video, reference, correct_mirror=mirror)
                except (MediaInputError, OSError):
                    st.warning('上次显示的画面读取失败，可继续核对下方当前帧。')
                else:
                    offset = reference - start
                    earlier_pose = display_frame(cached.frames[offset], decisions[offset])
                    st.caption(f'两帧相隔 {(decoded.timestamp_ms-earlier.timestamp_ms)/1000:.3f} 秒。请看两幅图的骨架是否落在同一位训练者身上。')
                    before, after = st.columns(2)
                    before.image(draw_pose(earlier.image_rgb, earlier_pose), caption=f'上次显示 · 源帧 {reference}', width='stretch')
                    after.image(overlay, caption=f'首次恢复 · 源帧 {index}', width='stretch')
        left,right = st.columns(2)
        left.image(decoded.image_rgb,caption=f'源帧 {index} · {decoded.timestamp_ms/1000:.3f} 秒',width='stretch')
        if not decision['show']:
            st.warning('此帧骨架已隐藏：'+reason_text(decision))
        elif decision['hidden_parts']:
            st.warning('此帧部分显示；未显示：'+'、'.join(decision['hidden_parts'])+'。'+reason_text(decision))
        right.image(overlay,caption='左侧青蓝圆点 · 右侧橙色方点；通过过滤的部位才绘制，绘制结果用于对照原片',width='stretch')
        tracking = diagnostic.get('subject_tracking')
        if tracking and st.checkbox('查看主训练者跟踪对照', key='compare_subject_tracking'):
            labels = {
                'initial_or_reacquired': '按可见性和画面位置初选',
                'tracked_match': '匹配到邻近候选',
                'impostor_jump_rejected': '位置跳变，本帧隐藏',
                'no_reliable_torso': '可见躯干关键点不足，本帧隐藏',
                'no_candidate': '未检出候选，本帧隐藏',
            }
            st.image(draw_pose(decoded.image_rgb, tracking),
                     caption=f"跟踪试验 · 源帧 {index} · {labels[tracking['reason']]}", width='stretch')
            st.caption('请对照原画确认是否跟对人。下方核对记录和证据包仍对应上方原有骨架。')
        evidence_key=(cached.key,index,VERSION,enable_tracker)
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
        st.caption('请选择当前帧的显示情况；全部隐藏时选择“无法判断”。刷新、重识别或换片会清除会话记录；需要保留时在上方手动下载本地备份。')
        columns=st.columns(2)
        for n,verdict in enumerate(VERDICTS[1:]):
            columns[n%2].button(verdict,key=f'review_mark_{n}',on_click=record_review,args=(index,verdict),disabled=not decision['show'] and verdict!='无法判断')
        st.button('清除此帧记录',key='review_clear',on_click=record_review,args=(index,'未记录'))
        st.caption(f'本次已记录 {len(reviews)}/{len(cached.frames)} 帧；每帧点击判断后计入进度。')
        render_review_download(cached,key='review_download_inline')
        st.dataframe([{'关键点':name,'绘制':idx in decision['allowed_indices'],
                       'visibility':pose_frame['landmarks'][idx]['visibility'],
                       'presence':pose_frame['landmarks'][idx]['presence']} for idx,name in JOINTS.items()],hide_index=True)
        st.json({'model':cached.model,'runtime':cached.runtime,'config':cached.config,
                 'mirror_corrected':cached.mirror,'selected_source_frames':[cached.start,cached.end]})
        st.caption('工程绘制阈值为 visibility 与 presence 均不低于 0.5 且在画面内。请结合原片核对点位与显示结果。')
