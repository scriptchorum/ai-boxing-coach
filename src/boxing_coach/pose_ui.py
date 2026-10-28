"""Session-only diagnostic UI for the pose extraction prototype."""
import streamlit as st

from .model_assets import AssetError
from .pose import JOINTS, PoseError, draw_pose, drawable, extract_pose, run_key
from .preview import read_preview_frame
from .video_io import MediaInputError
from .synced_player import render_synced_player


def clear_pose():
    st.session_state.pop('pose_run',None)
    st.session_state.pop('pose_index',None)


def render_pose_panel(video,start,end,mirror):
    st.subheader('人体关键点试跑')
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
        return
    st.button('清除本次关键点结果',on_click=clear_pose)
    st.subheader('播放并查看骨架')
    st.write('点击下面左侧视频的 ▶ 播放，右侧骨架随画面变化；暂停或拖动也可核对。仅识别过的区间显示骨架。')
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
    index = int(st.number_input('骨架核对源帧号',min_value=start,max_value=end,step=1,key='pose_index'))
    pose_frame = cached.frames[index-start]
    diagnostic = cached.diagnostics[index-start]
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
    right.image(draw_pose(decoded.image_rgb,pose_frame),caption='仅绘制满足当前条件的身体关键点；线条不代表动作评价',width='stretch')
    with st.expander('此帧关键点与运行信息'):
        st.dataframe([{'关键点':name,'绘制':drawable(pose_frame['landmarks'][idx]),
                       'visibility':pose_frame['landmarks'][idx]['visibility'],
                       'presence':pose_frame['landmarks'][idx]['presence']} for idx,name in JOINTS.items()],hide_index=True)
        st.json({'model':cached.model,'runtime':cached.runtime,'config':cached.config,
                 'mirror_corrected':cached.mirror,'selected_source_frames':[cached.start,cached.end]})
    st.caption('默认绘制阈值为 visibility 与 presence 均不低于 0.5 且在画面内，只是待验证的工程选择。没有连线不等于动作错误，连线完整也不等于动作合格。')
