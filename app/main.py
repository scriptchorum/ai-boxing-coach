"""Engineering-only Streamlit shell; no video analysis is implemented yet."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "complete.synthetic.json"

st.set_page_config(page_title="女性搏击视频复盘工具", page_icon="🥊", layout="wide")
st.title("女性搏击视频复盘工具")
st.warning(
    "当前页面仍是工程外壳：T04 已有本地视频探测核心，但本页不导入视频，也没有姿态分析、动作判错、拳力或伤病判断。"
)
st.write(
    "本页面只展示合成契约入口和本地运行环境；视频探测暂通过 boxing-coach-video 本地命令运行。"
)

left, right = st.columns(2)
with left:
    st.subheader("已锁定的本地组件")
    for package in ("av", "mediapipe", "opencv-contrib-python", "numpy", "streamlit"):
        st.code(f"{package}=={importlib.metadata.version(package)}")

with right:
    st.subheader("合成样例摘要")
    bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))
    st.json(
        {
            "synthetic": bundle["synthetic"],
            "sample_id": bundle["manifest"]["sample_id"],
            "candidate_count": bundle["report"]["candidate_count"],
            "valid_count": bundle["report"]["valid_count"],
            "report_status": bundle["report"]["status"],
        }
    )

st.info(
    "合成样例通过只证明数据契约与工程入口可运行，不代表真实女性样本、视觉效果或专业反馈已经验证。"
)
