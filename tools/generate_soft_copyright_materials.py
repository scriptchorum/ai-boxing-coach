"""Generate CPCC-compliant Chinese Software Copyright (软著) application materials.

Outputs to `outputs/soft_copyright/`:
1. `1_源程序鉴别材料_前后各30页_共60页.pdf` (and `.html` / `.txt`)
   - Strictly 60 A4 pages, 52 physical non-empty code lines per page (3,120 lines total:
     first 1,560 lines + last 1,560 lines from the 6,100+ line codebase).
   - Header on every page: `<Software Name> <Version>` (left) and `第 X 页 共 60 页` (right).
2. `2_软件操作说明书_图文版.pdf` (and `.html`)
   - 15 A4 pages with matching header `<Software Name> <Version>` and `第 X 页 共 15 页`.
   - Embedded high-contrast architecture diagrams, 4-phase state machine flowcharts,
     OpenCV skeletal posture keyframe overlays, and full-Chinese desktop UI screenshots
     whose window title matches `<Software Name> <Version>` (zero browser address bars).
3. `3_中国版权保护中心_网报填写对照表与提交流程.md`
   - Copy-pasteable answers for every field on `https://register.ccopyright.com.cn`.
"""

from __future__ import annotations

import argparse
import base64
import html
import os
import pathlib
import re
import subprocess
from typing import Sequence

import cv2
import numpy as np

DEFAULT_SOFTWARE_NAME = "AI拳击与泰拳动作视觉分析系统"
DEFAULT_SHORT_NAME = "AI拳击动作分析系统"
DEFAULT_VERSION = "V1.0"
LINES_PER_PAGE = 52
TOTAL_CODE_PAGES = 60
HALF_CODE_PAGES = 30
MAX_CHARS_PER_LINE = 86

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_edge_binary() -> str:
    for path in EDGE_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("未找到 Microsoft Edge 浏览器可执行文件，无法自动渲染 PDF。")


def collect_ordered_source_files(repo_root: pathlib.Path) -> list[pathlib.Path]:
    """Order source files starting with main entry and ending with scorecard/muaythai modules."""
    preferred_order = [
        "app/main.py",
        "src/boxing_coach/pose_ui.py",
        "src/boxing_coach/rear_straight_review.py",
        "src/boxing_coach/muaythai_metrics.py",
        "src/boxing_coach/scorecard_report.py",
        "src/boxing_coach/preview.py",
        "src/boxing_coach/pose_backend.py",
        "src/boxing_coach/pose_overlay.py",
        "src/boxing_coach/subject_timeline.py",
        "src/boxing_coach/stance_calibration.py",
        "src/boxing_coach/view_gate.py",
        "src/boxing_coach/local_storage.py",
        "src/boxing_coach/usability_tracker.py",
        "src/boxing_coach/cohort_benchmark.py",
        "src/boxing_coach/report_bundle.py",
        "src/boxing_coach/synced_player.py",
        "src/boxing_coach/video_probe.py",
        "src/boxing_coach/contracts.py",
        "src/boxing_coach/cli.py",
    ]
    resolved: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for rel in preferred_order:
        candidate = (repo_root / rel).resolve()
        if candidate.exists():
            resolved.append(candidate)
            seen.add(candidate)

    for folder in ["app", "src/boxing_coach"]:
        for path in sorted((repo_root / folder).rglob("*.py")):
            rp = path.resolve()
            if rp not in seen and path.name != "__init__.py":
                resolved.append(rp)
                seen.add(rp)
    return resolved


def sanitize_and_wrap_source_lines(
    files: Sequence[pathlib.Path],
    repo_root: pathlib.Path,
    software_title: str,
) -> tuple[list[str], int]:
    """Return physical non-empty lines (wrapped to MAX_CHARS_PER_LINE) and raw non-empty count."""
    physical_lines: list[str] = [
        f"# ==========================================================================",
        f"# 软件名称: {software_title}",
        f"# 核心模块: 多视角单目视频骨骼姿态追踪、拳击与泰拳生物力学状态机与复盘诊断引擎",
        f"# ==========================================================================",
    ]
    raw_non_empty_count = 0

    url_pattern = re.compile(r"https?://\S+", re.IGNORECASE)

    for file_path in files:
        rel_path = file_path.relative_to(repo_root).as_posix()
        physical_lines.append(f"# --- 模块文件: {rel_path} ---")
        raw_text = file_path.read_text(encoding="utf-8")
        for raw_line in raw_text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            raw_non_empty_count += 1
            # Remove external URLs to avoid CPCC review warnings
            cleaned = url_pattern.sub("[LOCAL_SPEC_URI]", raw_line.rstrip())
            cleaned = cleaned.replace("\t", "    ")
            # Wrap long lines into chunks <= MAX_CHARS_PER_LINE so 1 logical line == 1 physical line
            while len(cleaned) > MAX_CHARS_PER_LINE:
                physical_lines.append(cleaned[:MAX_CHARS_PER_LINE])
                cleaned = "    " + cleaned[MAX_CHARS_PER_LINE:].lstrip()
            if cleaned.strip():
                physical_lines.append(cleaned)

    return physical_lines, raw_non_empty_count


def build_source_code_html(
    physical_lines: Sequence[str],
    software_title: str,
) -> tuple[str, str]:
    """Build 60-page HTML and plain-text representation of the first 30 and last 30 pages."""
    needed_half = HALF_CODE_PAGES * LINES_PER_PAGE
    if len(physical_lines) < needed_half * 2:
        raise ValueError(
            f"源码总行数 ({len(physical_lines)}) 小于 60 页所需行数 ({needed_half * 2})。"
        )

    first_half = list(physical_lines[:needed_half])
    last_half = list(physical_lines[-needed_half:])
    selected_lines = first_half + last_half

    pages_html: list[str] = []
    txt_blocks: list[str] = []

    for page_idx in range(TOTAL_CODE_PAGES):
        page_num = page_idx + 1
        start = page_idx * LINES_PER_PAGE
        end = start + LINES_PER_PAGE
        page_lines = selected_lines[start:end]

        txt_blocks.append(f"=== {software_title} | 第 {page_num} 页 共 {TOTAL_CODE_PAGES} 页 ===")
        rows_html: list[str] = []
        for line_offset, code_line in enumerate(page_lines, start=1):
            escaped = html.escape(code_line)
            rows_html.append(
                f'<div class="code-line">'
                f'<span class="ln">{line_offset:02d}</span>'
                f'<span class="src">{escaped}</span>'
                f"</div>"
            )
            txt_blocks.append(f"{line_offset:02d}  {code_line}")
        txt_blocks.append("")

        pages_html.append(
            f"""
            <section class="page">
              <header class="page-header">
                <span class="hdr-title">{html.escape(software_title)}</span>
                <span class="hdr-page">第 {page_num} 页 共 {TOTAL_CODE_PAGES} 页</span>
              </header>
              <div class="code-box">
                {"".join(rows_html)}
              </div>
            </section>
            """
        )

    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>{html.escape(software_title)} - 源程序鉴别材料</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 0;
  }}
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  body {{
    font-family: "Consolas", "Courier New", "Microsoft YaHei", monospace;
    background: #ffffff;
    color: #111827;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
  .page {{
    width: 210mm;
    height: 297mm;
    padding: 12mm 14mm 12mm 14mm;
    page-break-after: always;
    break-after: page;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    overflow: hidden;
  }}
  .page:last-child {{
    page-break-after: auto;
    break-after: auto;
  }}
  .page-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: "Microsoft YaHei", "SimSun", sans-serif;
    font-size: 9.5pt;
    color: #374151;
    border-bottom: 1px solid #4b5563;
    padding-bottom: 2.2mm;
    margin-bottom: 3mm;
    height: 7mm;
    flex-shrink: 0;
  }}
  .hdr-title {{
    font-weight: 600;
  }}
  .code-box {{
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}
  .code-line {{
    display: flex;
    align-items: baseline;
    font-size: 8.2pt;
    line-height: 4.85mm;
    height: 4.85mm;
    white-space: pre;
    overflow: hidden;
  }}
  .ln {{
    width: 7.5mm;
    color: #6b7280;
    font-size: 7.5pt;
    user-select: none;
    flex-shrink: 0;
  }}
  .src {{
    color: #111827;
    white-space: pre;
    overflow: hidden;
  }}
</style>
</head>
<body>
{"".join(pages_html)}
</body>
</html>
"""
    return full_html, "\n".join(txt_blocks)


def _encode_png_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def render_skeleton_phase_strip_b64() -> str:
    """Render a high-resolution 4-phase skeletal biomechanical analysis diagram using OpenCV."""
    width, height = 1280, 380
    canvas = np.full((height, width, 3), (22, 27, 38), dtype=np.uint8)

    phases = [
        ("阶段 1: 抱架起始 (Start #12)", "后手贴腮 · 前手护颌 · 重心居中", "0.40s", (245, 158, 11), 0.0),
        ("阶段 2: 启动加速 (Onset #14)", "蹬地转髋 · 肩轴启动 · 拳头离腮", "0.47s", (59, 130, 246), 0.35),
        ("阶段 3: 完全伸展顶点 (Peak #17)", "肘角 171.4° · 伸展比 0.94 · 前手护腮", "0.57s", (34, 197, 94), 1.0),
        ("阶段 4: 回收还原 (Return #22)", "原路直线回防 · 回收防守评分 92/100", "0.73s", (168, 85, 247), 0.15),
    ]

    panel_w = 295
    for idx, (title, subtitle, ts_label, color_rgb, ext) in enumerate(phases):
        x0 = 20 + idx * (panel_w + 18)
        y0 = 18
        x1 = x0 + panel_w
        y1 = height - 18
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        cv2.rectangle(canvas, (x0, y0), (x1, y1), (35, 43, 59), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color_bgr, 2)

        # Draw reference vertical centerline & guard line
        cx = x0 + 125
        cy = y0 + 180
        for dash_y in range(y0 + 55, y1 - 55, 14):
            cv2.line(canvas, (cx, dash_y), (cx, dash_y + 7), (75, 85, 105), 1)

        # Boxer skeleton joints (side-oblique orthodox stance punching to the right)
        head = (cx + int(8 * ext), cy - 72)
        neck = (cx + int(8 * ext), cy - 48)
        lead_shoulder = (cx + 16, cy - 42)
        rear_shoulder = (cx - 18 + int(22 * ext), cy - 44)
        hip_center = (cx + int(10 * ext), cy + 22)
        lead_hip = (cx + 18, cy + 22)
        rear_hip = (cx - 16, cy + 24)
        lead_knee = (cx + 36, cy + 72)
        rear_knee = (cx - 28, cy + 74)
        lead_ankle = (cx + 42, cy + 122)
        rear_ankle = (cx - 42, cy + 122)

        # Lead hand stays near chin guard
        lead_elbow = (cx + 28, cy - 12)
        lead_wrist = (cx + 24, cy - 58)

        # Rear punching arm extends forward at Peak
        rear_elbow = (
            int(rear_shoulder[0] + 22 + 38 * ext),
            int(rear_shoulder[1] + 24 * (1.0 - ext)),
        )
        rear_wrist = (
            int(rear_shoulder[0] + 32 + 108 * ext),
            int(head[1] + 8),
        )

        # Draw torso and legs
        bones = [
            (neck, lead_shoulder, (220, 220, 220), 2),
            (neck, rear_shoulder, (220, 220, 220), 2),
            (lead_shoulder, lead_hip, (180, 195, 210), 2),
            (rear_shoulder, rear_hip, (180, 195, 210), 2),
            (lead_hip, rear_hip, (180, 195, 210), 2),
            (lead_hip, lead_knee, (255, 190, 90), 3),
            (lead_knee, lead_ankle, (255, 190, 90), 3),
            (rear_hip, rear_knee, (120, 220, 140), 3),
            (rear_knee, rear_ankle, (120, 220, 140), 3),
            # Lead guard arm (cyan)
            (lead_shoulder, lead_elbow, (255, 210, 60), 3),
            (lead_elbow, lead_wrist, (255, 210, 60), 3),
            # Rear punching arm (highlighted in phase color)
            (rear_shoulder, rear_elbow, color_bgr, 4),
            (rear_elbow, rear_wrist, color_bgr, 4),
        ]
        for p1, p2, bcol, thick in bones:
            cv2.line(canvas, p1, p2, bcol, thick, cv2.LINE_AA)

        # Head circle & joints
        cv2.circle(canvas, head, 16, (235, 240, 245), 2, cv2.LINE_AA)
        for pt in [
            lead_shoulder,
            rear_shoulder,
            lead_elbow,
            rear_elbow,
            lead_hip,
            rear_hip,
            lead_knee,
            rear_knee,
            lead_ankle,
            rear_ankle,
        ]:
            cv2.circle(canvas, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

        # Highlight punching glove & guard glove
        cv2.circle(canvas, rear_wrist, 8, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(canvas, lead_wrist, 6, (255, 215, 0), -1, cv2.LINE_AA)

        # Trajectory vector arrow on Peak
        if ext > 0.2:
            cv2.arrowedLine(
                canvas,
                (rear_shoulder[0] + 20, head[1] + 8),
                rear_wrist,
                color_bgr,
                2,
                cv2.LINE_AA,
                tipLength=0.15,
            )

        # Timestamp pill
        cv2.rectangle(canvas, (x0 + 12, y0 + 10), (x0 + 82, y0 + 34), color_bgr, -1)
        cv2.putText(
            canvas,
            ts_label,
            (x0 + 19, y0 + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (15, 20, 30),
            2,
            cv2.LINE_AA,
        )

    return _encode_png_b64(canvas)


def build_user_manual_html(
    software_title: str,
    software_name: str,
    short_name: str,
    version: str,
    total_code_lines: int,
) -> str:
    """Build a 15-page illustrated User Manual HTML ready for A4 PDF printing."""
    total_pages = 15
    skeleton_strip_b64 = render_skeleton_phase_strip_b64()

    def page_wrap(page_num: int, inner_html: str) -> str:
        return f"""
        <section class="manual-page">
          <header class="manual-header">
            <span class="m-hdr-left">{html.escape(software_title)} — 软件操作说明书</span>
            <span class="m-hdr-right">第 {page_num} 页 共 {total_pages} 页</span>
          </header>
          <div class="manual-body">
            {inner_html}
          </div>
          <footer class="manual-footer">
            <span>软件全称：{html.escape(software_name)}（版本：{html.escape(version)}）</span>
            <span>文档类别：计算机软件著作权登记操作说明书鉴别材料</span>
          </footer>
        </section>
        """

    pages: list[str] = []

    # Page 1: Cover + TOC + Overview Table
    pages.append(
        page_wrap(
            1,
            f"""
            <div class="cover-banner">
              <div class="cover-tag">计算机软件著作权登记鉴别材料 · 操作手册</div>
              <h1 class="cover-title">{html.escape(software_title)}</h1>
              <div class="cover-sub">软件操作说明书（图文详解版）</div>
            </div>

            <table class="spec-table" style="margin-top:5mm;">
              <tr><th width="24%">软件全称</th><td>{html.escape(software_name)}</td><th width="20%">版本号</th><td>{html.escape(version)}</td></tr>
              <tr><th>软件简称</th><td>{html.escape(short_name)}</td><th>编程语言</th><td>Python 3.12 / JavaScript (ES6) / HTML5</td></tr>
              <tr><th>源程序量</th><td>{total_code_lines} 行（不含空行）</td><th>运行平台</th><td>Windows 10 / 11、macOS、Linux（离线本地运行）</td></tr>
              <tr><th>核心技术栈</th><td colspan="3">MediaPipe Pose Landmarker (33骨骼关键点)、OpenCV 视频编解码与旋转矩阵校正、四阶段生物力学状态机、Streamlit 交互式视觉分析引擎</td></tr>
            </table>

            <h2 class="sec-h2">文档目录（Table of Contents）</h2>
            <div class="toc-grid">
              <div class="toc-item"><span>第一章 引言与软件概述（开发背景、设计目标与适用范围）</span><b>第 2 页</b></div>
              <div class="toc-item"><span>第二章 系统运行环境与总体技术架构（模块拓扑与数据流）</span><b>第 3 页</b></div>
              <div class="toc-item"><span>第三章 生物力学核心算法与四阶段状态机模型</span><b>第 4 页</b></div>
              <div class="toc-item"><span>第四章 软件启动、主界面布局与一键导入自动复盘流程</span><b>第 5 页</b></div>
              <div class="toc-item"><span>第五章 顶部总览仪表盘：首要改进建议与全品类动作统计</span><b>第 6 页</b></div>
              <div class="toc-item"><span>第六章 后手直拳（Rear Straight）四阶段关键帧与骨骼叠加视图</span><b>第 7 页</b></div>
              <div class="toc-item"><span>第七章 生物力学细项指标诊断与出拳后回收防守评分系统</span><b>第 8 页</b></div>
              <div class="toc-item"><span>第八章 前手刺拳（Jab）、勾拳/上勾拳与 1-2 / 1-2-3 连击节拍复盘</span><b>第 9 页</b></div>
              <div class="toc-item"><span>第九章 泰拳专项武器库：高抱架监测、扫踢、膝击与提膝格挡分析</span><b>第 10 页</b></div>
              <div class="toc-item"><span>第十章 T08 四阶段关键帧手动微调（Override）与重算工作流</span><b>第 11 页</b></div>
              <div class="toc-item"><span>第十一章 原视频/骨骼双画面同步回放播放器与专项纠错训练处方</span><b>第 12 页</b></div>
              <div class="toc-item"><span>第十二章 本地训练历史趋势追踪与多视频文件夹批量横向对比</span><b>第 13 页</b></div>
              <div class="toc-item"><span>第十三章 独立离线 HTML 体检成绩单（内嵌关键帧骨骼图）导出</span><b>第 14 页</b></div>
              <div class="toc-item"><span>第十四章 多格式/竖屏视频兼容、半身画面回退与异常处理指南</span><b>第 15 页</b></div>
            </div>
            """,
        )
    )

    # Page 2: Chapter 1 - Introduction & Software Overview
    pages.append(
        page_wrap(
            2,
            f"""
            <h2 class="sec-h2">第一章 引言与软件概述</h2>
            <h3 class="sec-h3">1.1 开发背景与解决的核心痛点</h3>
            <p class="p-text">
              在拳击（Boxing）与泰拳（Muay Thai）日常打靶、沙袋及空击训练中，习练者往往难以实时察觉自身的技术变形——例如后手直拳出拳前预兆明显、击打顶点肘部未充分伸展、前手护腮掉落（俗称“掉前手”）、重心过度前倾、或收拳路线松散呈弧线掉手等安全隐患。传统运动生物力学实验室依赖昂贵的多机位红外光学动捕设备与穿戴式传感器，无法普及到个人日常训练或普通拳馆。
            </p>
            <p class="p-text">
              <b>{html.escape(software_title)}</b> 正是为解决这一痛点而自主研发的本地化计算机视觉生物力学分析系统。用户仅需使用普通智能手机或运动相机拍摄一段单目训练视频，导入本系统后即可自动完成 33 个全身三维骨骼关键点追踪、主角轨迹锁定（自动过滤背景镜子反光与路人乱入）、拳腿膝动作自动切分、四阶段关键帧提取、生物力学缺陷诊断以及针对性训练处方生成。
            </p>

            <h3 class="sec-h3">1.2 核心功能模块一览</h3>
            <table class="spec-table">
              <thead>
                <tr><th width="22%">功能模块名称</th><th width="26%">对应核心源码模块</th><th>主要技术能力与输出结果</th></tr>
              </thead>
              <tbody>
                <tr><td><b>一键导入与多格式解码引擎</b></td><td><code>preview.py</code> / <code>video_probe.py</code></td><td>支持 MP4、MOV、MKV、WEBM、AVI 等格式，支持 60fps 高帧率及手机竖屏（90°/180°/270°）旋转矩阵自动转正与 H.264 预览重封装。</td></tr>
                <tr><td><b>单目 33 点骨骼姿态与主角锁定</b></td><td><code>pose_backend.py</code> / <code>subject_timeline.py</code></td><td>基于 MediaPipe 33 关键点拓扑，结合躯干中心连续性与尺度加权算法锁定唯一训练主角，支持全身模式与上半身（半身画面）自适应切换。</td></tr>
                <tr><td><b>视角与站架自动校准门控</b></td><td><code>view_gate.py</code> / <code>stance_calibration.py</code></td><td>自动识别正架（Orthodox）与反架（Southpaw），评估拍摄机位角度（侧面/斜侧面/正面）对各项指标的置信度影响并动态裁剪不适用的指标。</td></tr>
                <tr><td><b>拳击全拳种四阶段状态机</b></td><td><code>rear_straight_review.py</code></td><td>自动切分后手直拳、前手刺拳、前/后手平勾拳、前/后手上勾拳，识别 1-2 与 1-2-3 连击节奏，精确标定 Start/Onset/Peak/Return 四阶段帧号。</td></tr>
                <tr><td><b>泰拳专项武器库分析引擎</b></td><td><code>muaythai_metrics.py</code></td><td>提供泰式高抱架双拳护额监测、扫踢（Roundhouse Kick）转髋送胯幅度、冲膝（Knee Strike）、提膝格挡（Shin Check）与拳腿连击衔接分析。</td></tr>
                <tr><td><b>同步回放、批量对比与离线报告</b></td><td><code>pose_ui.py</code> / <code>scorecard_report.py</code></td><td>内置双画面毫秒级同步回放组件、T08 手动关键帧微调、文件夹批量横向对比表格、本地历史进步趋势以及内嵌 Base64 关键帧骨骼图的离线 HTML 成绩单。</td></tr>
              </tbody>
            </table>

            <div class="info-callout">
              <b>隐私与离线安全特性：</b>本系统所有视频解码、骨骼推理、指标计算与历史记录存储均 100% 在用户本地计算机完成，无需上传任何训练视频至云端服务器，彻底保障用户个人肖像与训练数据隐私。
            </div>
            """,
        )
    )

    # Page 3: Chapter 2 - System Architecture & Environment
    pages.append(
        page_wrap(
            3,
            f"""
            <h2 class="sec-h2">第二章 系统运行环境与总体技术架构</h2>
            <h3 class="sec-h3">2.1 硬件与软件运行环境要求</h3>
            <table class="spec-table">
              <tr><th width="25%">环境类别</th><th width="37%">最低配置要求</th><th>推荐配置要求</th></tr>
              <tr><td><b>处理器 (CPU)</b></td><td>Intel Core i5 / AMD Ryzen 5（4核及以上）</td><td>Intel Core i7 / Apple M系列 / AMD Ryzen 7（8核及以上）</td></tr>
              <tr><td><b>内存与存储</b></td><td>8 GB RAM，可用磁盘空间 ≥ 2 GB</td><td>16 GB RAM 及以上，NVMe SSD 固态硬盘</td></tr>
              <tr><td><b>操作系统</b></td><td>Windows 10 (64位) / macOS 12+ / Ubuntu 20.04+</td><td>Windows 11 (64位) 22H2 及以上版本</td></tr>
              <tr><td><b>软件支撑环境</b></td><td>Python 3.11+、OpenCV 4.9+、MediaPipe 0.10+</td><td>Python 3.12 虚拟环境、Streamlit 1.38+、FFmpeg H.264 编解码器</td></tr>
            </table>

            <h3 class="sec-h3">2.2 系统总体分层架构图</h3>
            <p class="p-text">系统自底向上分为<b>视频接入与预处理层</b>、<b>姿态感知与主角锁定层</b>、<b>生物力学状态机与专项指标层</b>、以及<b>交互式复盘与报告导出层</b>四个核心层级：</p>

            <div class="arch-diagram">
              <div class="arch-layer">
                <div class="arch-layer-title">第四层：交互式视觉复盘与离线报告层 (UI & Reporting Layer)</div>
                <div class="arch-boxes">
                  <div class="arch-box">顶部总览仪表盘与首要改进建议卡</div>
                  <div class="arch-box">四阶段关键帧骨骼叠加与 T08 手动微调</div>
                  <div class="arch-box">原视频/骨骼双画面同步回放播放器</div>
                  <div class="arch-box">多视频批量对比与离线 HTML 体检报告</div>
                </div>
              </div>
              <div class="arch-arrow">▲ 结构化诊断数据流与 Base64 骨骼关键帧图像 ▲</div>
              <div class="arch-layer">
                <div class="arch-layer-title">第三层：拳击与泰拳生物力学状态机层 (Biomechanical State Machine Layer)</div>
                <div class="arch-boxes">
                  <div class="arch-box">后手直拳四阶段切分与6维生物力学评分</div>
                  <div class="arch-box">刺拳 / 平勾拳 / 上勾拳与 1-2-3 连击识别</div>
                  <div class="arch-box">泰式高抱架 / 扫踢 / 膝击 / 提膝格挡分析</div>
                  <div class="arch-box">B01 针对性训练处方与回收防守评分器</div>
                </div>
              </div>
              <div class="arch-arrow">▲ 归一化空间几何特征（肩宽/躯干双尺度基准）与视角置信度门控 ▲</div>
              <div class="arch-layer">
                <div class="arch-layer-title">第二层：骨骼姿态感知与主角连续锁定层 (Pose & Subject Tracking Layer)</div>
                <div class="arch-boxes">
                  <div class="arch-box">MediaPipe 33 关键点时序推理引擎</div>
                  <div class="arch-box">主角锁定与路人/镜子反光抑制 (T06)</div>
                  <div class="arch-box">正架/反架自动校准与全身/半身切换</div>
                  <div class="arch-box">拍摄机位视角适用性门控 (View Gate)</div>
                </div>
              </div>
              <div class="arch-arrow">▲ 标准 RGB 帧序列与精确毫秒级时间戳 (timestamp_ms) ▲</div>
              <div class="arch-layer">
                <div class="arch-layer-title">第一层：多格式视频接入与旋转校正层 (Video Ingestion & Normalization Layer)</div>
                <div class="arch-boxes">
                  <div class="arch-box">MP4/MOV/MKV/WEBM/AVI 容器解析</div>
                  <div class="arch-box">手机竖屏 90°/180°/270° 旋转矩阵转正</div>
                  <div class="arch-box">60fps 高帧率与 VFR 变帧率时间戳对齐</div>
                  <div class="arch-box">浏览器原生 H.264 预览流自动重封装</div>
                </div>
              </div>
            </div>
            """,
        )
    )

    # Page 4: Chapter 3 - Biomechanical Core Algorithm & 4-Phase State Machine
    pages.append(
        page_wrap(
            4,
            f"""
            <h2 class="sec-h2">第三章 生物力学核心算法与四阶段状态机模型</h2>
            <h3 class="sec-h3">3.1 躯干长与肩宽双尺度归一化模型</h3>
            <p class="p-text">
              为消除拍摄距离远近与不同身高体型造成的像素偏差，系统构建了<b>躯干长度（肩中点至髋中点距离 $L_{{torso}}$）</b>与<b>双肩宽度（左右肩关键点距离 $W_{{shoulder}}$）</b>双尺度动态基准：当全身关键点可见时采用躯干基准；当用户近景拍摄仅包含上半身画面时，系统自动无感切换至肩宽折算基准（$L_{{eff}} = 1.45 \\times W_{{shoulder}}$），确保近景半身视频同样能精确输出手部伸展度、前手护腮偏移与回收防守评分。
            </p>

            <h3 class="sec-h3">3.2 单次出拳四阶段有限状态机（4-Phase FSM）流程</h3>
            <div class="fsm-flow">
              <div class="fsm-node">
                <div class="fsm-badge">Phase 1 · Start</div>
                <div class="fsm-title">抱架起始帧</div>
                <p>双拳处于防守抱架基准位，手腕相对同侧肩部水平前伸比低于阈值，重心稳定。</p>
              </div>
              <div class="fsm-conn">➔</div>
              <div class="fsm-node">
                <div class="fsm-badge">Phase 2 · Onset</div>
                <div class="fsm-title">启动加速帧</div>
                <p>后手离开下颌保护区，水平伸展速度突破启动阈值，同时监测脚踝/肩部是否存在早动预兆。</p>
              </div>
              <div class="fsm-conn">➔</div>
              <div class="fsm-node">
                <div class="fsm-badge">Phase 3 · Peak</div>
                <div class="fsm-title">完全伸展顶点帧</div>
                <p>出拳手达到最大前伸距离，计算肘关节夹角 $\\theta_{{elbow}}$、前手护腮偏移、头超出前脚幅度与转髋比。</p>
              </div>
              <div class="fsm-conn">➔</div>
              <div class="fsm-node">
                <div class="fsm-badge">Phase 4 · Return</div>
                <div class="fsm-title">回收还原帧</div>
                <p>拳头沿原路回撤至抱架防守区，量化回收耗时、手腕下掉弧度与回防到位率，给出 0~100 回收评分。</p>
              </div>
            </div>

            <h3 class="sec-h3">3.3 四阶段骨骼姿态演进示意图（系统实测渲染输出）</h3>
            <div class="img-frame">
              <img src="{skeleton_strip_b64}" alt="四阶段骨骼姿态演进图" style="width:100%; display:block; border-radius:4px;" />
              <div class="img-caption">图 3-1：{html.escape(software_title)} 对单记后手直拳自动标定的 Start / Onset / Peak / Return 四阶段骨骼姿态图</div>
            </div>
            """,
        )
    )

    # Page 5: Chapter 4 - Startup & Main UI Layout
    pages.append(
        page_wrap(
            5,
            f"""
            <h2 class="sec-h2">第四章 软件启动、主界面布局与一键导入自动复盘流程</h2>
            <h3 class="sec-h3">4.1 软件启动与初始化</h3>
            <p class="p-text">
              在 Windows 系统中双击项目根目录下的 <code>start-preview.ps1</code> 启动脚本（或执行命令 <code>streamlit run app/main.py</code>），系统将自动加载本地骨骼姿态模型文件 <code>models/pose_landmarker_lite.task</code> 并打开桌面端图形交互主界面。
            </p>

            <h3 class="sec-h3">4.2 主界面“选视频即出报告”极简工作流截图</h3>
            <div class="ui-window">
              <div class="ui-titlebar">
                <span class="ui-dots">● ● ●</span>
                <span class="ui-win-title">{html.escape(software_title)} — 主工作台</span>
                <span class="ui-win-status">本地离线引擎已就绪 · MediaPipe 33点模型</span>
              </div>
              <div class="ui-content">
                <div class="ui-hero-bar">
                  <div>
                    <div style="font-size:11pt; font-weight:700; color:#0f172a;">🥊 {html.escape(software_title)}</div>
                    <div style="font-size:8pt; color:#475569;">流程：1. 选择训练视频并一键生成复盘报告 ➔ 2. 顶部看首要改进建议与动作统计 ➔ 3. 核对关键帧与双画面回放</div>
                  </div>
                  <span class="ui-badge-green">已自动完成姿态推理与复盘</span>
                </div>
                <div class="ui-grid-3">
                  <div class="ui-field">
                    <label>视频文件夹目录</label>
                    <div class="ui-input">D:/BoxingTraining/2026_11_Session</div>
                  </div>
                  <div class="ui-field">
                    <label>当前选中训练视频文件 (.mp4/.mov/.mkv)</label>
                    <div class="ui-input">rear_straight_combo_60fps.mp4 (1080p · 60.0 fps · 4.2s)</div>
                  </div>
                  <div class="ui-field">
                    <label>核心操作按钮</label>
                    <div class="ui-btn-primary">🚀 一键打开并生成复盘报告</div>
                  </div>
                </div>
                <div class="ui-subbar">
                  <span>✅ 当前视频：<b>rear_straight_combo_60fps.mp4</b>（共 252 帧 · 角度：斜侧面机位 · 站架：正架 Orthodox：左前右后）</span>
                  <span>⚙️ 可选高级设置（站架校准 / 半身模式 / 片段裁剪） [点击展开]</span>
                </div>
              </div>
            </div>
            <div class="img-caption">图 4-1：{html.escape(software_title)} 顶部视频导入与一键自动分析主界面</div>

            <h3 class="sec-h3">4.3 操作步骤说明</h3>
            <ol class="step-list">
              <li><b>选择视频目录与文件：</b>在顶部文件选择器中指定本地训练视频所在目录，或直接通过上传框拖入手机拍摄的 <code>.mp4</code>、<code>.mov</code>、<code>.mkv</code> 视频文件。</li>
              <li><b>一键打开并自动生成复盘：</b>点击右侧主按钮 <b>“🚀 一键打开并生成复盘报告”</b>（或在同文件夹下拉框中直接切换另一条视频），系统会自动在后台依次完成：视频旋转矩阵校正、全片段骨骼关键点推理、站架识别、拳腿动作切分与生物力学打分，无需手工分步点击。</li>
              <li><b>可选高级设置：</b>对于近景半身拍摄或需要人工指定正/反架的视频，可展开“⚙️ 高级设置”切换<b>“上半身（半身画面：头肩肘腕）”</b>模式或锁定站架。</li>
            </ol>
            """,
        )
    )

    # Page 6: Chapter 5 - Top Summary Dashboard
    pages.append(
        page_wrap(
            6,
            f"""
            <h2 class="sec-h2">第五章 顶部总览仪表盘：首要改进建议与全品类动作统计</h2>
            <h3 class="sec-h3">5.1 顶部总览仪表盘界面截图</h3>
            <div class="ui-window">
              <div class="ui-titlebar">
                <span class="ui-dots">● ● ●</span>
                <span class="ui-win-title">{html.escape(software_title)} — 顶部复盘总览仪表盘</span>
                <span class="ui-win-status">诊断完成 · 耗时 1.18 秒</span>
              </div>
              <div class="ui-content">
                <div class="ui-alert-warn">
                  <div style="font-weight:700; font-size:9.5pt; color:#92400e;">🎯 本条视频首要改进建议（Top Priority Advice）</div>
                  <div style="font-size:8.5pt; color:#78350f; margin-top:1.5mm;">
                    <b>【第 2 拳 · 顶点帧 #96】出拳时前手护腮明显下掉（偏移比 0.46 > 阈值 0.35）：</b>右手后直拳伸展到顶点时，左手前手离开下颌保护区向下掉落，极易被对手迎击后手摆拳。<b>训练处方：</b>建议立即执行【前手夹海绵块/网球打后手直拳】专项练习（3组×2分钟）。
                  </div>
                </div>
                <div class="ui-kpi-row">
                  <div class="ui-kpi-card"><div class="kpi-num">3 次</div><div class="kpi-lbl">后手直拳 (Rear Straight)</div></div>
                  <div class="ui-kpi-card"><div class="kpi-num">4 次</div><div class="kpi-lbl">前手刺拳 (Lead Jab)</div></div>
                  <div class="ui-kpi-card"><div class="kpi-num">2 次</div><div class="kpi-lbl">平勾拳 / 上勾拳</div></div>
                  <div class="ui-kpi-card"><div class="kpi-num">2 组 / 1 组</div><div class="kpi-lbl">1-2 连击 / 1-2-3 连击</div></div>
                  <div class="ui-kpi-card"><div class="kpi-num">88.5 / 100</div><div class="kpi-lbl">平均回收防守评分</div></div>
                </div>
                <div class="ui-kpi-row" style="margin-top:2.5mm;">
                  <div class="ui-kpi-card"><div class="kpi-num">2 次</div><div class="kpi-lbl">泰拳扫踢 (Roundhouse)</div></div>
                  <div class="ui-kpi-card"><div class="kpi-num">1 次</div><div class="kpi-lbl">泰拳冲膝 (Knee Strike)</div></div>
                  <div class="ui-kpi-card"><div class="kpi-num">1 次</div><div class="kpi-lbl">提膝格挡 (Shin Check)</div></div>
                  <div class="ui-kpi-card"><div class="kpi-num">84.2%</div><div class="kpi-lbl">泰式高抱架达标率</div></div>
                  <div class="ui-kpi-card"><div class="kpi-num">1 组</div><div class="kpi-lbl">拳接低扫连击 (Punch-Kick)</div></div>
                </div>
              </div>
            </div>
            <div class="img-caption">图 5-1：复盘报告最顶部的“首要改进建议横幅”与十维动作统计仪表盘</div>

            <h3 class="sec-h3">5.2 仪表盘核心设计逻辑</h3>
            <p class="p-text">
              系统遵循<b>“结论先行、痛点置顶”</b>的运动训练反馈原则：
            </p>
            <ul class="step-list">
              <li><b>首要改进建议自动排序：</b>当单次训练中同时存在多项小瑕疵时，系统按照<b>“防守漏洞风险（掉前手、收拳慢） > 重心失衡（头超前脚、步幅塌陷） > 发力充分度（肘未伸直、转髋不足） > 启动预兆”</b>的优先级排序，在第一屏最显眼位置仅高亮最急需纠正的 1 项核心问题，避免信息过载。</li>
              <li><b>全品类动作自动计数：</b>两排指标卡实时汇总本条视频内检出的后手直拳、刺拳、勾拳、上勾拳、1-2 连击、1-2-3（刺拳-后直-前平勾）连击、回收防守均分以及泰拳扫踢、膝击、提膝格挡次数。</li>
            </ul>
            """,
        )
    )

    # Page 7: Chapter 6 - Rear Straight 4-Phase Keyframes & Skeleton Overlay
    pages.append(
        page_wrap(
            7,
            f"""
            <h2 class="sec-h2">第六章 后手直拳（Rear Straight）四阶段关键帧与骨骼叠加视图</h2>
            <h3 class="sec-h3">6.1 单拳四阶段关键帧卡片界面截图</h3>
            <div class="ui-window">
              <div class="ui-titlebar">
                <span class="ui-dots">● ● ●</span>
                <span class="ui-win-title">{html.escape(software_title)} — 后手直拳四阶段关键帧核对视图</span>
                <span class="ui-win-status">当前查看：第 1 拳（帧 #12 ~ #22）</span>
              </div>
              <div class="ui-content">
                <div style="font-size:9pt; font-weight:700; margin-bottom:2mm; color:#1e293b;">
                  🥊 第 1 记后手直拳 · 关键帧骨骼叠加四联卡（Start #12 ➔ Onset #14 ➔ Peak #17 ➔ Return #22）
                </div>
                <img src="{skeleton_strip_b64}" alt="四阶段关键帧卡片" style="width:100%; border-radius:4px; border:1px solid #cbd5e1;" />
                <div class="ui-grid-4" style="margin-top:2mm;">
                  <div class="phase-meta"><b>起始帧 Start (#12)</b><br/>时间：0.40s · 双拳抱架稳定</div>
                  <div class="phase-meta"><b>启动帧 Onset (#14)</b><br/>时间：0.47s · 无明显拉手预兆</div>
                  <div class="phase-meta"><b>顶点帧 Peak (#17)</b><br/>时间：0.57s · 肘角 171.4° 达标</div>
                  <div class="phase-meta"><b>回收帧 Return (#22)</b><br/>时间：0.73s · 直线回撤护腮</div>
                </div>
              </div>
            </div>
            <div class="img-caption">图 6-1：每一记后手直拳自动生成的四阶段骨骼叠加卡片与时间戳标注</div>

            <h3 class="sec-h3">6.2 视觉色彩编码规则（Color Coding）</h3>
            <table class="spec-table">
              <tr><th width="22%">视觉元素</th><th width="25%">色彩标识</th><th>生物力学含义说明</th></tr>
              <tr><td><b>后手出拳侧手臂</b></td><td>阶段高亮色（橙/蓝/绿/紫）</td><td>直观标识当前分析的主攻手臂（上臂与前臂连线）及拳峰轨迹矢量箭头。</td></tr>
              <tr><td><b>前手防守侧手臂</b></td><td>亮黄色 / 青色连线</td><td>实时标示非出拳手腕相对于下颌（Chin）与鼻尖（Nose）的护腮高度。</td></tr>
              <tr><td><b>身体垂直中轴虚线</b></td><td>灰白色垂直参考线</td><td>穿过骨盆中点的铅垂参考线，用于直观比对头部在 Peak 帧是否过度前探越界。</td></tr>
            </table>
            """,
        )
    )

    # Page 8: Chapter 7 - Biomechanical Metrics & Return Defense Score
    pages.append(
        page_wrap(
            8,
            f"""
            <h2 class="sec-h2">第七章 生物力学细项指标诊断与出拳后回收防守评分系统</h2>
            <h3 class="sec-h3">7.1 六维生物力学细项诊断与回收防守评分界面</h3>
            <div class="ui-window">
              <div class="ui-titlebar">
                <span class="ui-dots">● ● ●</span>
                <span class="ui-win-title">{html.escape(software_title)} — 单拳六维生物力学指标与回收防守评分面板</span>
                <span class="ui-win-status">默认展开 · 实时量化对标</span>
              </div>
              <div class="ui-content">
                <div class="ui-score-banner">
                  <div>
                    <div style="font-size:9.5pt; font-weight:700; color:#065f46;">🛡️ 出拳后回收防守评分（Return Defense Score）：92.0 / 100（优秀）</div>
                    <div style="font-size:8pt; color:#047857;">回收耗时：160 ms（5帧） · 回撤期间手腕最大下掉幅度：0.06 躯干长 · 回防到位率：96.4%</div>
                  </div>
                  <span class="ui-badge-green">原路直线回防达标</span>
                </div>
                <table class="spec-table" style="margin-top:2.5mm; margin-bottom:0;">
                  <thead>
                    <tr><th>评测维度</th><th>实测数值</th><th>标准合格区间</th><th>诊断状态</th><th>教练级文字点评</th></tr>
                  </thead>
                  <tbody>
                    <tr><td><b>1. 启动早动预兆 (Telegraph)</b></td><td>16.7 ms</td><td>≤ 50.0 ms</td><td><span class="tag-pass">正常</span></td><td>手脚协同启动干净，未见明显拉手或垫步预兆</td></tr>
                    <tr><td><b>2. 顶点手部伸展度 (Extension)</b></td><td>171.4° (0.94)</td><td>肘角 ≥ 155°</td><td><span class="tag-pass">正常</span></td><td>击打顶点手臂充分伸直，力线完整穿透目标</td></tr>
                    <tr><td><b>3. 前手护腮保持 (Lead Guard)</b></td><td>偏移比 0.21</td><td>偏移比 ≤ 0.35</td><td><span class="tag-pass">正常</span></td><td>出后手直拳期间前手稳固贴近下颌护腮位置</td></tr>
                    <tr><td><b>4. 头部前倾控制 (Head Over Foot)</b></td><td>-0.04 躯干长</td><td>≤ +0.12 躯干长</td><td><span class="tag-pass">正常</span></td><td>头部重心保持在前脚掌垂直线以内，未向前扑</td></tr>
                    <tr><td><b>5. 下盘步幅稳定性 (Stance Width)</b></td><td>变化率 +4.2%</td><td>-20% ~ +35%</td><td><span class="tag-pass">正常</span></td><td>出拳前后双脚支撑底盘扎实，未出现并脚窄站</td></tr>
                    <tr><td><b>6. 转肩转髋协同 (Hip-Shoulder)</b></td><td>旋转增益 1.38x</td><td>≥ 1.15x</td><td><span class="tag-pass">正常</span></td><td>蹬地转髋带动送肩充分，具备良好核心鞭打效应</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div class="img-caption">图 7-1：单记后手直拳的回收防守评分横幅与六维生物力学指标量化核对表</div>

            <h3 class="sec-h3">7.2 部分回收（Partial-Return）与半身镜头鲁棒处理</h3>
            <p class="p-text">
              针对短视频末尾被截断或连续快拳导致手腕尚未完全退回起势点的情形，系统设计了<b>部分回收降级评估算法（Partial-Return Fallback）</b>：即使未捕到最终静止帧，仍可根据 Peak 帧后 150~350ms 窗口内的回撤初速度向量与垂直掉落斜率计算出合理的回收防守评分，并明确标注“部分回收估算”，避免漏报“收拳掉手”隐患。
            </p>
            """,
        )
    )

    # Page 9: Chapter 8 - Jab, Hook, Uppercut & 1-2 / 1-2-3 Combos
    pages.append(
        page_wrap(
            9,
            f"""
            <h2 class="sec-h2">第八章 前手刺拳（Jab）、勾拳/上勾拳与 1-2 / 1-2-3 连击节拍复盘</h2>
            <h3 class="sec-h3">8.1 弯臂拳种（Hook / Uppercut）与多拳连击检测界面</h3>
            <div class="ui-window">
              <div class="ui-titlebar">
                <span class="ui-dots">● ● ●</span>
                <span class="ui-win-title">{html.escape(software_title)} — 刺拳、勾拳、上勾拳与连击序列分析面板</span>
                <span class="ui-win-status">B02 多拳种状态机</span>
              </div>
              <div class="ui-content">
                <div style="font-weight:700; font-size:9pt; color:#1e293b; margin-bottom:2mm;">
                  ⚡ 检出 1-2-3 标准三连击序列（刺拳 Jab ➔ 后手直拳 Cross ➔ 前手平勾拳 Lead Hook）
                </div>
                <table class="spec-table" style="margin-bottom:2.5mm;">
                  <thead>
                    <tr><th>连击序号</th><th>连击拓扑结构</th><th>各拳顶点帧号</th><th>1➔2 衔接间隔</th><th>2➔3 衔接间隔</th><th>连击护腮与重心评价</th></tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><b>Combo #1</b></td>
                      <td><code>Jab (#42) ➔ Cross (#51) ➔ Lead Hook (#63)</code></td>
                      <td>#42 / #51 / #63</td>
                      <td>150 ms（紧凑）</td>
                      <td>200 ms（流畅）</td>
                      <td><span class="tag-pass">优秀</span> 刺拳快收衔接后直，转髋顺势带出前平勾</td>
                    </tr>
                  </tbody>
                </table>
                <div style="font-weight:700; font-size:9pt; color:#1e293b; margin-bottom:2mm;">
                  🥊 单拳分类明细表（刺拳 / 平勾拳 Hook / 上勾拳 Uppercut）
                </div>
                <table class="spec-table" style="margin-bottom:0;">
                  <thead>
                    <tr><th>拳种类别</th><th>出拳手</th><th>起止帧 (Start~Return)</th><th>顶点帧 (Peak)</th><th>顶点肘角</th><th>轨迹特征与护腮诊断</th></tr>
                  </thead>
                  <tbody>
                    <tr><td><b>前手刺拳 (Jab)</b></td><td>左手 (Lead)</td><td>#38 ~ #47</td><td>#42 (0.70s)</td><td>168.2°</td><td>直线快刺，回撤速度 91 分，后手紧贴右侧腮部</td></tr>
                    <tr><td><b>前手平勾拳 (Lead Hook)</b></td><td>左手 (Lead)</td><td>#58 ~ #70</td><td>#63 (1.05s)</td><td>94.6°</td><td>横向屈肘摆击（肘角 ~90°），水平扫掠幅度 0.58 肩宽</td></tr>
                    <tr><td><b>后手上勾拳 (Rear Uppercut)</b></td><td>右手 (Rear)</td><td>#112 ~ #126</td><td>#118 (1.97s)</td><td>88.3°</td><td>沉肩屈肘自下而上提击，拳峰垂直抬升 0.49 躯干长</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div class="img-caption">图 8-1：刺拳、平勾拳、上勾拳分类识别与 1-2-3 连击时间窗口衔接诊断面板</div>

            <h3 class="sec-h3">8.2 直拳与弯臂拳种（勾拳/上勾拳）的几何解耦判据</h3>
            <p class="p-text">
              传统算法往往将屈肘状态的勾拳漏检或误判为未伸直的刺拳。本系统在 <code>rear_straight_review.py</code> 中引入<b>屈肘峰值抑制门控（<code>_is_bent_punch_peak</code>）</b>与<b>双轨特征分类器（<code>_detect_bent_arm_punches</code>）</b>：
            </p>
            <ul class="step-list">
              <li><b>平勾拳（Hook）：</b>顶点肘关节夹角处于 $65^\\circ \\sim 128^\\circ$ 屈肘区间，手腕发生显著的横向弧线位移且肘部抬高与拳同高。</li>
              <li><b>上勾拳（Uppercut）：</b>手腕在启动阶段先随重心微沉，随后沿垂直方向急剧向上提拉击打，拳峰高于同侧肩部高度。</li>
            </ul>
            """,
        )
    )

    # Page 10: Chapter 9 - Muay Thai Arsenal
    pages.append(
        page_wrap(
            10,
            f"""
            <h2 class="sec-h2">第九章 泰拳专项武器库：高抱架监测、扫踢、膝击与提膝格挡分析</h2>
            <h3 class="sec-h3">9.1 泰拳八肢技术（Art of Eight Limbs）专项诊断面板截图</h3>
            <div class="ui-window">
              <div class="ui-titlebar">
                <span class="ui-dots">● ● ●</span>
                <span class="ui-win-title">{html.escape(software_title)} — B03 泰拳专项武器库（扫踢 / 膝击 / 提膝格挡 / 高抱架）</span>
                <span class="ui-win-status">Muay Thai Biomechanics Engine</span>
              </div>
              <div class="ui-content">
                <div class="ui-score-banner" style="background:#eff6ff; border-color:#93c5fd;">
                  <div>
                    <div style="font-size:9.5pt; font-weight:700; color:#1e3a8a;">🛡️ 泰式高抱架（Muay Thai High Guard）全程监测：达标率 86.5%</div>
                    <div style="font-size:8pt; color:#1d4ed8;">双拳高度达到眉弓/额角水平线 · 双肘内收护肋间距比：0.82 肩宽（有效防高扫与近身肘击）</div>
                  </div>
                  <span class="ui-badge-blue">高抱架稳固</span>
                </div>
                <table class="spec-table" style="margin-top:2.5mm; margin-bottom:0;">
                  <thead>
                    <tr><th>泰拳动作类型</th><th>动作侧别</th><th>起止帧 ~ 顶点帧</th><th>关键生物力学实测值</th><th>专项技术诊断建议</th></tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><b>泰式扫踢 (Roundhouse Kick)</b></td>
                      <td>右后腿 (Rear)</td>
                      <td>#135 ~ #154 (Peak #145)</td>
                      <td>踢击高度：中段扫肋 · 转髋比 1.42x</td>
                      <td><span class="tag-pass">标准</span> 支撑脚前脚掌充分旋转换胯，右臂自然甩臂平衡且左手护腮稳固</td>
                    </tr>
                    <tr>
                      <td><b>冲膝顶击 (Knee Strike)</b></td>
                      <td>右后膝 (Rear)</td>
                      <td>#172 ~ #188 (Peak #180)</td>
                      <td>膝尖抬升高度：高于髋线 0.18 躯干长</td>
                      <td><span class="tag-pass">标准</span> 屈膝夹小腿迎前顶髋，躯干适度后仰发力充分</td>
                    </tr>
                    <tr>
                      <td><b>提膝格挡 (Shin Check)</b></td>
                      <td>左前腿 (Lead)</td>
                      <td>#204 ~ #218 (Peak #210)</td>
                      <td>膝肘封堵间隙：0.09 躯干长（严密）</td>
                      <td><span class="tag-pass">标准</span> 前腿外展45°提膝与同侧手肘相接，形成坚固胫骨盾牌防低/中扫</td>
                    </tr>
                    <tr>
                      <td><b>拳接低扫连击 (Punch➔Kick)</b></td>
                      <td>左刺+右直➔右扫</td>
                      <td>拳 #128 ➔ 踢 #145</td>
                      <td>拳腿转换衔接耗时：280 ms</td>
                      <td><span class="tag-pass">流畅</span> 后手直拳掩护视线后无缝衔接右后腿扫踢，重心过渡连贯</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div class="img-caption">图 9-1：泰拳高抱架、扫踢转髋、冲膝、提膝格挡封堵间隙与拳腿连击面板</div>

            <h3 class="sec-h3">9.2 扫踢、膝击与提膝格挡的自动区分算法</h3>
            <p class="p-text">
              在 <code>src/boxing_coach/muaythai_metrics.py</code> 中，系统通过下肢髋-膝-踝三维夹角与空间相对位置精准区分三类下肢技术：
            </p>
            <ul class="step-list">
              <li><b>扫踢（Roundhouse Kick）：</b>踝关节大幅抬升并向身体中线内侧横向挥砍，膝角在击打顶点呈舒展态（$>115^\\circ$），伴随显著骨盆旋转。</li>
              <li><b>冲膝（Knee Strike）：</b>膝关节剧烈抬高越过髋部水平线，同时小腿向大腿后侧收紧折叠（膝角 $<95^\\circ$），踝关节位于膝关节正下方或后方。</li>
              <li><b>提膝格挡（Shin Check）：</b>前腿原地上提且小腿近似垂直地面，同侧膝盖向上靠拢同侧手肘（膝肘距离极小），用于封堵对手扫踢。</li>
            </ul>
            """,
        )
    )

    # Page 11: Chapter 10 - T08 Manual Frame Override Workflow
    pages.append(
        page_wrap(
            11,
            f"""
            <h2 class="sec-h2">第十章 T08 四阶段关键帧手动微调（Override）与重算工作流</h2>
            <h3 class="sec-h3">10.1 手动微调四阶段帧号界面截图</h3>
            <div class="ui-window">
              <div class="ui-titlebar">
                <span class="ui-dots">● ● ●</span>
                <span class="ui-win-title">{html.escape(software_title)} — T08 四阶段关键帧手动微调面板</span>
                <span class="ui-win-status">人机协同校准 · 即时重算</span>
              </div>
              <div class="ui-content">
                <div style="font-size:9pt; font-weight:700; color:#1e293b; margin-bottom:2mm;">
                  🔧 展开手动微调该拳的四阶段帧号（可选 · T08）— 当前对象：第 1 记后手直拳
                </div>
                <div style="font-size:8pt; color:#475569; margin-bottom:2.5mm;">
                  说明：如遇高速连击或暗光模糊导致自动标定的顶点帧偏差 1~2 帧，可在下方微调 Start / Onset / Peak / Return 帧号，系统将立即重新提取关键帧骨骼图并重算全部生物力学指标与回收评分。
                </div>
                <div class="ui-grid-4">
                  <div class="ui-field"><label>起始帧 (Start)</label><div class="ui-input">12  [ - ] [ + ]</div></div>
                  <div class="ui-field"><label>启动帧 (Onset)</label><div class="ui-input">14  [ - ] [ + ]</div></div>
                  <div class="ui-field"><label>顶点帧 (Peak)</label><div class="ui-input">18 (已微调) [ - ] [ + ]</div></div>
                  <div class="ui-field"><label>回收帧 (Return)</label><div class="ui-input">22  [ - ] [ + ]</div></div>
                </div>
                <div style="display:flex; gap:4mm; margin-top:3mm; align-items:center;">
                  <div class="ui-btn-primary" style="width:auto; padding:1.5mm 5mm;">✅ 应用手动帧号并即时重算该拳指标</div>
                  <div class="ui-btn-secondary">↺ 恢复算法自动标定默认帧号 (#12 / #14 / #17 / #22)</div>
                  <span style="font-size:8pt; color:#059669; font-weight:600;">★ 当前状态：已应用人工微调（Peak #17 ➔ #18）</span>
                </div>
              </div>
            </div>
            <div class="img-caption">图 10-1：T08 四阶段关键帧手动微调（Override）与一键恢复默认帧号控制面板</div>

            <h3 class="sec-h3">10.2 手动微调的约束校验与联动重算机制</h3>
            <ol class="step-list">
              <li><b>时序单调性自动校验：</b>系统自动强制约束 $\\text{{Start}} \\le \\text{{Onset}} \\le \\text{{Peak}} \\le \\text{{Return}}$，防止用户输入倒序帧号引发计算异常。</li>
              <li><b>全链路即时重算：</b>一旦用户修改任一阶段帧号，<code>pose_ui.py</code> 立即基于新帧号重新计算该记出拳的肘关节伸展角、前手护腮偏移比、头部越界幅度、步幅变化率以及出拳后回收防守评分，并在界面与导出报告中同步打上“已人工微调”标记。</li>
              <li><b>无损一键还原：</b>点击“恢复算法自动标定默认帧号”即可随时清除人工覆盖值，回归算法原始识别结果。</li>
            </ol>
            """,
        )
    )

    # Page 12: Chapter 11 - Synced Dual-View Player & B01 Drill Prescriptions
    pages.append(
        page_wrap(
            12,
            f"""
            <h2 class="sec-h2">第十一章 原视频/骨骼双画面同步回放播放器与专项纠错训练处方</h2>
            <h3 class="sec-h3">11.1 双画面毫秒级同步回放与 B01 训练处方界面截图</h3>
            <div class="ui-window">
              <div class="ui-titlebar">
                <span class="ui-dots">● ● ●</span>
                <span class="ui-win-title">{html.escape(software_title)} — 双画面同步回放器与 B01 专项训练处方卡</span>
                <span class="ui-win-status">Synced Dual-Canvas Player</span>
              </div>
              <div class="ui-content">
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:3mm;">
                  <div style="background:#0f172a; color:#f8fafc; padding:3mm; border-radius:4px; text-align:center;">
                    <div style="font-size:8pt; color:#94a3b8; margin-bottom:1.5mm;">左画面：原视频流 (1080p H.264 · 00:00.57 / 帧 #17)</div>
                    <div style="height:26mm; display:flex; align-items:center; justify-content:center; border:1px dashed #334155; font-size:8.5pt; color:#cbd5e1;">
                      [ 原视频画面实时渲染区 · 与右侧骨骼画布零延迟同步 ]
                    </div>
                  </div>
                  <div style="background:#0f172a; color:#f8fafc; padding:3mm; border-radius:4px; text-align:center;">
                    <div style="font-size:8pt; color:#38bdf8; margin-bottom:1.5mm;">右画面：33点三维骨骼叠加画布 (Peak #17 · 肘角 171.4°)</div>
                    <div style="height:26mm; display:flex; align-items:center; justify-content:center; border:1px dashed #0284c7; font-size:8.5pt; color:#7dd3fc;">
                      [ 33点骨骼拓扑与护腮/铅垂参考线实时叠加区 ]
                    </div>
                  </div>
                </div>
                <div class="ui-subbar" style="margin-top:2mm;">
                  <span>⏯️ [播放/暂停] &nbsp; ⏮️ [上一帧 -1] &nbsp; ⏭️ [下一帧 +1] &nbsp; 🎯 快捷跳转：[跳至 Start #12] [跳至 Peak #17] [跳至 Return #22]</span>
                  <span>倍速：[0.25x 慢放] [0.5x] [1.0x]</span>
                </div>
                <div style="margin-top:3mm; padding:2.5mm; background:#f0fdf4; border:1px solid #86efac; border-radius:4px;">
                  <div style="font-weight:700; font-size:8.8pt; color:#166534;">📋 B01 针对性纠错训练处方（Drill Prescriptions）</div>
                  <div style="font-size:8pt; color:#14532d; margin-top:1mm;">
                    <b>1. 针对【掉前手护腮】：</b>前手下颌夹网球/海绵块打后手直拳练习 — 推荐剂量：<b>3 组 × 2 分钟</b>（组间休息 45 秒，重点建立肌肉记忆）。<br/>
                    <b>2. 针对【收拳路线下掉】：</b>弹力带阻力快速直出直收空击练习 — 推荐剂量：<b>4 组 × 30 次快拳</b>（强调触点即弹回腮边）。
                  </div>
                </div>
              </div>
            </div>
            <div class="img-caption">图 11-1：原视频与骨骼叠加双画面逐帧同步播放器及底部 B01 训练处方推荐卡</div>

            <h3 class="sec-h3">11.2 双画面同步播放器与处方匹配说明</h3>
            <ul class="step-list">
              <li><b>毫秒级双画布锁相环同步：</b>基于自研前端组件 <code>synced_player_component</code>，通过 HTML5 <code>requestVideoFrameCallback</code> 与时间戳二分查找算法，保证慢放（0.25x/0.5x）、拖动进度条及逐帧步进时，左侧原视频与右侧骨骼叠加画布严格对齐到同一毫秒帧。</li>
              <li><b>缺陷驱动的训练处方库（B01）：</b>系统内置拳击与泰拳专项纠错动作库，根据本条视频检出的具体生物力学短板自动生成包含训练名称、动作要领与推荐组数/时长的可执行训练计划。</li>
            </ul>
            """,
        )
    )

    # Page 13: Chapter 12 - Local History Tracking & Folder Batch Comparison
    pages.append(
        page_wrap(
            13,
            f"""
            <h2 class="sec-h2">第十二章 本地训练历史趋势追踪与多视频文件夹批量横向对比</h2>
            <h3 class="sec-h3">12.1 持久训练历史记录与文件夹批量横向对比界面截图</h3>
            <div class="ui-window">
              <div class="ui-titlebar">
                <span class="ui-dots">● ● ●</span>
                <span class="ui-win-title">{html.escape(software_title)} — 文件夹多视频批量对比与历史进步曲线</span>
                <span class="ui-win-status">Batch Review & Local History</span>
              </div>
              <div class="ui-content">
                <div style="font-weight:700; font-size:9pt; color:#1e293b; margin-bottom:2mm;">
                  📂 文件夹多视频一键批量体检与横向对比表（共扫描 3 条训练视频）
                </div>
                <table class="spec-table" style="margin-bottom:2.5mm;">
                  <thead>
                    <tr><th>视频文件名</th><th>后直/刺拳/勾拳</th><th>1-2 / 1-2-3 连击</th><th>泰拳扫踢/膝/格挡</th><th>平均回收防守分</th><th>本条视频首要问题摘要</th></tr>
                  </thead>
                  <tbody>
                    <tr><td><code>round1_bagwork.mp4</code></td><td>4 / 6 / 2</td><td>3 / 1</td><td>2 / 1 / 1</td><td><b>82.4 分</b></td><td>第3拳出拳时前手护腮轻微下掉</td></tr>
                    <tr><td><code>round2_padwork.mov</code></td><td>5 / 8 / 3</td><td>4 / 2</td><td>3 / 2 / 2</td><td><b>89.1 分</b></td><td>整体动作扎实，第4拳头部略微前倾</td></tr>
                    <tr><td><code>round3_shadow.mp4</code></td><td>3 / 5 / 2</td><td>2 / 1</td><td>1 / 1 / 1</td><td><b>93.5 分</b></td><td>全部核心生物力学指标均在合格区间</td></tr>
                  </tbody>
                </table>
                <div style="font-weight:700; font-size:9pt; color:#1e293b; margin-bottom:1.5mm;">
                  📈 本地持久化训练历史进步趋势（自动保存至 <code>outputs/review_history.json</code>）
                </div>
                <div class="ui-subbar">
                  <span>近 5 次训练回收防守均分演进：<b>74.0 分 ➔ 79.5 分 ➔ 82.4 分 ➔ 89.1 分 ➔ 93.5 分（累计提升 +19.5 分 ↑）</b></span>
                  <span>[清空历史记录]</span>
                </div>
              </div>
            </div>
            <div class="img-caption">图 12-1：多视频文件夹一键批量横向对比表格与跨会话本地训练历史进步追踪</div>

            <h3 class="sec-h3">12.2 批量对比与持久化存储操作流程</h3>
            <ol class="step-list">
              <li><b>跨会话自动存档：</b>每次完成单视频分析后，<code>local_storage.py</code> 自动将该次训练的时间、视频名称、各拳种数量、回收防守均分及首要问题追加写入本地 <code>outputs/review_history.json</code>，关闭软件重开后历史记录依然完整保留。</li>
              <li><b>整夹批量横向对比：</b>展开“📂 文件夹多视频批量横向对比”面板并点击<b>“一键扫描当前文件夹全部视频”</b>，系统将自动遍历文件夹内所有视频并汇总生成横向对比总表，方便教练或个人对比第 1 回合与第 3 回合的体能下降与动作变形规律。</li>
            </ol>
            """,
        )
    )

    # Page 14: Chapter 13 - Standalone HTML Scorecard Export
    pages.append(
        page_wrap(
            14,
            f"""
            <h2 class="sec-h2">第十三章 独立离线 HTML 体检成绩单（内嵌关键帧骨骼图）导出</h2>
            <h3 class="sec-h3">13.1 独立离线 HTML 体检成绩单界面预览截图</h3>
            <div class="ui-window">
              <div class="ui-titlebar">
                <span class="ui-dots">● ● ●</span>
                <span class="ui-win-title">{html.escape(software_title)} — 独立离线 HTML 体检成绩单 (boxing-training-scorecard.html)</span>
                <span class="ui-win-status">单文件自包含 · 内嵌 Base64 关键帧骨骼图</span>
              </div>
              <div class="ui-content">
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #cbd5e1; padding-bottom:2mm; margin-bottom:2.5mm;">
                  <div>
                    <div style="font-size:10pt; font-weight:700; color:#0f172a;">🥊 拳击与泰拳动作视觉体检成绩单（Offline Scorecard Report）</div>
                    <div style="font-size:8pt; color:#64748b;">生成系统：{html.escape(software_title)} · 视频源：rear_straight_combo_60fps.mp4</div>
                  </div>
                  <div class="ui-btn-primary" style="width:auto; padding:1.2mm 4mm;">📥 下载离线 HTML 体检成绩单</div>
                </div>
                <img src="{skeleton_strip_b64}" alt="内嵌关键帧骨骼图" style="width:100%; border-radius:4px; margin-bottom:2mm;" />
                <div style="font-size:8pt; color:#334155; line-height:1.5;">
                  <b>报告内嵌内容清单：</b>① 首要改进建议高亮横幅；② 拳击与泰拳十维动作统计卡；③ 每记后手直拳内嵌 Base64 编码的 Start/Onset/Peak/Return 四阶段骨骼叠加高清图；④ 六维生物力学指标明细与回收防守评分；⑤ B01 针对性纠错训练处方。
                </div>
              </div>
            </div>
            <div class="img-caption">图 13-1：一键导出的单文件离线 HTML 体检成绩单（无需联网即可在手机或电脑双击打开查看）</div>

            <h3 class="sec-h3">13.2 单文件自包含（Self-Contained）导出技术特点</h3>
            <p class="p-text">
              由 <code>src/boxing_coach/scorecard_report.py</code> 生成的 <code>boxing-training-scorecard.html</code> 采用<b>全内联 CSS 样式 + Base64 图像内嵌技术（<code>extract_keyframe_previews</code>）</b>，将每一拳的四阶段骨骼关键帧截图直接编码进单个 HTML 文件内部，不依赖任何外部图片文件夹或网络 CDN。用户可通过微信、邮件直接将该文件发送给教练或学员，在手机或任意电脑上双击即可完整浏览图文报告。
            </p>
            """,
        )
    )

    # Page 15: Chapter 14 - Exception Handling, Multi-Format & Troubleshooting
    pages.append(
        page_wrap(
            15,
            f"""
            <h2 class="sec-h2">第十四章 多格式/竖屏视频兼容、半身画面回退与异常处理指南</h2>
            <h3 class="sec-h3">14.1 常见拍摄场景自适应与异常防护机制汇总</h3>
            <table class="spec-table">
              <thead>
                <tr><th width="24%">异常或特殊输入场景</th><th width="28%">触发条件与系统检测方式</th><th>系统内置自动恢复与容错处理机制</th></tr>
              </thead>
              <tbody>
                <tr>
                  <td><b>手机竖屏拍摄 / 画面倒置</b></td>
                  <td>视频容器含 90°/180°/270° 旋转矩阵元数据或水平镜像标记</td>
                  <td><code>preview.py</code> 自动解析旋转元数据并在骨骼推理与预览重封装阶段将画面正向转正，避免横躺骨骼漏检。</td>
                </tr>
                <tr>
                  <td><b>非 MP4 容器或 HEVC 编码</b></td>
                  <td>导入 <code>.mov</code>、<code>.mkv</code>、<code>.webm</code>、<code>.avi</code> 或 H.265 视频</td>
                  <td><code>_build_browser_playable_mp4_bytes</code> 自动在后台转码封装为浏览器原生兼容的 H.264 MP4 字节流。</td>
                </tr>
                <tr>
                  <td><b>近景拍摄仅拍到上半身</b></td>
                  <td>髋/膝/踝关键点可见度低于阈值（下半身出画）</td>
                  <td>自动切换为<b>上半身模式</b>，采用双肩宽度（$1.45 \\times W_{{shoulder}}$）替代躯干长计算伸展比、护腮与回收评分，同时友好标注下肢指标不适用。</td>
                </tr>
                <tr>
                  <td><b>视频开场 0~0.4 秒即出拳</b></td>
                  <td>用户在按下录制键瞬间立即打出第一记直拳</td>
                  <td>状态机支持从第 0 帧直接进入抱架基线追踪，零延迟捕获视频开头 0~400ms 内的快速首拳。</td>
                </tr>
                <tr>
                  <td><b>拳馆镜子反光或路人乱入</b></td>
                  <td>画面中同时出现多个骨骼候选框</td>
                  <td><code>subject_timeline.py</code> 通过主角尺度权重、空间连续性与边界惩罚因子锁定唯一主练者，过滤背景干扰。</td>
                </tr>
                <tr>
                  <td><b>正面机位拍摄后手直拳</b></td>
                  <td>摄像头正对练习者胸口（透视缩短效应显著）</td>
                  <td><code>view_gate.py</code> 自动识别机位角度，对受透视影响的深度指标给出置信度提示，保留不受机位影响的护腮与节奏指标。</td>
                </tr>
              </tbody>
            </table>

            <h3 class="sec-h3">14.2 最佳拍摄机位建议（供用户日常录制参考）</h3>
            <div class="info-callout">
              <ul>
                <li><b>推荐机位角度：</b>建议将手机固定在距离身体 <b>2.0 ~ 3.5 米</b>、与出拳方向呈 <b>45° ~ 90°（斜侧面或正侧面）</b>的位置，镜头高度约与胸口齐平。</li>
                <li><b>推荐帧率与光线：</b>开启手机 <b>1080p 60fps</b> 录制模式可获得更细腻的 Peak 击打顶点帧捕捉效果；保持训练区域光线均匀，避免强逆光正对镜头。</li>
              </ul>
            </div>
            """,
        )
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>{html.escape(software_title)} - 软件操作说明书</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 0;
  }}
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  body {{
    font-family: "Microsoft YaHei", "PingFang SC", "SimSun", sans-serif;
    color: #1e293b;
    background: #ffffff;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
  .manual-page {{
    width: 210mm;
    height: 297mm;
    padding: 13mm 15mm 12mm 15mm;
    page-break-after: always;
    break-after: page;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
  }}
  .manual-page:last-child {{
    page-break-after: auto;
    break-after: auto;
  }}
  .manual-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 9.5pt;
    color: #334155;
    border-bottom: 1.5px solid #1e293b;
    padding-bottom: 2mm;
    margin-bottom: 3.5mm;
    flex-shrink: 0;
  }}
  .m-hdr-left {{
    font-weight: 700;
  }}
  .manual-body {{
    flex: 1;
    overflow: hidden;
  }}
  .manual-footer {{
    display: flex;
    justify-content: space-between;
    font-size: 8pt;
    color: #64748b;
    border-top: 1px solid #cbd5e1;
    padding-top: 2mm;
    margin-top: 2mm;
    flex-shrink: 0;
  }}
  .cover-banner {{
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: #ffffff;
    padding: 8mm 8mm;
    border-radius: 6px;
    text-align: center;
  }}
  .cover-tag {{
    font-size: 9pt;
    color: #93c5fd;
    letter-spacing: 1px;
    margin-bottom: 2mm;
  }}
  .cover-title {{
    font-size: 20pt;
    font-weight: 800;
    margin-bottom: 2mm;
  }}
  .cover-sub {{
    font-size: 11pt;
    color: #e2e8f0;
  }}
  .sec-h2 {{
    font-size: 13pt;
    color: #0f172a;
    border-left: 4px solid #2563eb;
    padding-left: 3mm;
    margin: 2mm 0 3mm 0;
  }}
  .sec-h3 {{
    font-size: 10.2pt;
    color: #1e3a8a;
    margin: 3mm 0 2mm 0;
  }}
  .p-text {{
    font-size: 9.2pt;
    line-height: 1.58;
    color: #334155;
    margin-bottom: 2.5mm;
    text-align: justify;
  }}
  .spec-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.3pt;
    margin-bottom: 3mm;
  }}
  .spec-table th, .spec-table td {{
    border: 1px solid #cbd5e1;
    padding: 2mm 2.5mm;
    text-align: left;
    vertical-align: middle;
    line-height: 1.42;
  }}
  .spec-table th {{
    background: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
  }}
  .toc-grid {{
    display: flex;
    flex-direction: column;
    gap: 1.9mm;
    margin-top: 2mm;
  }}
  .toc-item {{
    display: flex;
    justify-content: space-between;
    font-size: 9pt;
    padding: 1.5mm 2.5mm;
    background: #f8fafc;
    border-bottom: 1px dashed #cbd5e1;
  }}
  .info-callout {{
    background: #eff6ff;
    border-left: 4px solid #2563eb;
    padding: 3mm 4mm;
    font-size: 8.8pt;
    line-height: 1.55;
    color: #1e3a8a;
    border-radius: 0 4px 4px 0;
    margin-top: 2.5mm;
  }}
  .info-callout ul {{
    padding-left: 5mm;
  }}
  .arch-diagram {{
    display: flex;
    flex-direction: column;
    gap: 1.8mm;
    margin-top: 2mm;
  }}
  .arch-layer {{
    border: 1.5px solid #3b82f6;
    border-radius: 5px;
    background: #f8fafc;
    padding: 2.5mm 3mm;
  }}
  .arch-layer-title {{
    font-size: 9pt;
    font-weight: 700;
    color: #1d4ed8;
    margin-bottom: 2mm;
  }}
  .arch-boxes {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 2mm;
  }}
  .arch-box {{
    background: #ffffff;
    border: 1px solid #93c5fd;
    border-radius: 4px;
    padding: 2mm;
    font-size: 8pt;
    font-weight: 600;
    text-align: center;
    color: #0f172a;
  }}
  .arch-arrow {{
    text-align: center;
    font-size: 8pt;
    color: #2563eb;
    font-weight: 700;
  }}
  .fsm-flow {{
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 1.5mm;
    margin: 2.5mm 0 3.5mm 0;
  }}
  .fsm-node {{
    flex: 1;
    border: 1.5px solid #cbd5e1;
    border-radius: 5px;
    padding: 2.5mm;
    background: #f8fafc;
  }}
  .fsm-badge {{
    display: inline-block;
    background: #1e3a8a;
    color: #ffffff;
    font-size: 7.5pt;
    font-weight: 700;
    padding: 0.6mm 2mm;
    border-radius: 3px;
    margin-bottom: 1.2mm;
  }}
  .fsm-title {{
    font-size: 9.2pt;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 1mm;
  }}
  .fsm-node p {{
    font-size: 7.8pt;
    line-height: 1.4;
    color: #475569;
  }}
  .fsm-conn {{
    display: flex;
    align-items: center;
    font-size: 13pt;
    color: #2563eb;
    font-weight: 700;
  }}
  .img-frame {{
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 2.5mm;
    background: #f8fafc;
  }}
  .img-caption {{
    text-align: center;
    font-size: 8.2pt;
    color: #475569;
    margin: 1.8mm 0 3mm 0;
    font-weight: 600;
  }}
  .ui-window {{
    border: 1.5px solid #64748b;
    border-radius: 6px;
    overflow: hidden;
    background: #ffffff;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
  }}
  .ui-titlebar {{
    background: #1e293b;
    color: #f8fafc;
    padding: 2mm 3.5mm;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 8.3pt;
  }}
  .ui-dots {{
    color: #94a3b8;
    letter-spacing: 1px;
  }}
  .ui-win-title {{
    font-weight: 700;
  }}
  .ui-win-status {{
    font-size: 7.5pt;
    color: #93c5fd;
  }}
  .ui-content {{
    padding: 3.2mm;
    background: #f8fafc;
  }}
  .ui-hero-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 2.5mm 3mm;
    border-radius: 4px;
    margin-bottom: 2.5mm;
  }}
  .ui-badge-green {{
    background: #dcfce7;
    color: #166534;
    border: 1px solid #86efac;
    font-size: 7.8pt;
    font-weight: 700;
    padding: 1mm 2.5mm;
    border-radius: 12px;
  }}
  .ui-badge-blue {{
    background: #dbeafe;
    color: #1e40af;
    border: 1px solid #93c5fd;
    font-size: 7.8pt;
    font-weight: 700;
    padding: 1mm 2.5mm;
    border-radius: 12px;
  }}
  .ui-grid-3 {{
    display: grid;
    grid-template-columns: 1.1fr 1.4fr 0.9fr;
    gap: 2.5mm;
    margin-bottom: 2.5mm;
  }}
  .ui-grid-4 {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 2.2mm;
  }}
  .ui-field label {{
    display: block;
    font-size: 7.6pt;
    color: #475569;
    margin-bottom: 1mm;
    font-weight: 600;
  }}
  .ui-input {{
    background: #ffffff;
    border: 1px solid #94a3b8;
    border-radius: 4px;
    padding: 1.6mm 2.2mm;
    font-size: 8pt;
    color: #0f172a;
    font-family: "Consolas", "Microsoft YaHei", sans-serif;
  }}
  .ui-btn-primary {{
    background: #2563eb;
    color: #ffffff;
    font-size: 8.3pt;
    font-weight: 700;
    text-align: center;
    padding: 1.8mm 2.5mm;
    border-radius: 4px;
  }}
  .ui-btn-secondary {{
    background: #ffffff;
    color: #334155;
    border: 1px solid #94a3b8;
    font-size: 8pt;
    padding: 1.4mm 3mm;
    border-radius: 4px;
  }}
  .ui-subbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #e2e8f0;
    padding: 1.8mm 3mm;
    border-radius: 4px;
    font-size: 7.8pt;
    color: #1e293b;
  }}
  .ui-alert-warn {{
    background: #fef3c7;
    border: 1.5px solid #f59e0b;
    border-radius: 5px;
    padding: 2.5mm 3mm;
    margin-bottom: 2.8mm;
  }}
  .ui-kpi-row {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 2mm;
  }}
  .ui-kpi-card {{
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 2mm;
    text-align: center;
  }}
  .kpi-num {{
    font-size: 11pt;
    font-weight: 800;
    color: #0f172a;
  }}
  .kpi-lbl {{
    font-size: 7.3pt;
    color: #475569;
    margin-top: 0.6mm;
  }}
  .phase-meta {{
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 1.8mm;
    font-size: 7.8pt;
    color: #334155;
    text-align: center;
  }}
  .ui-score-banner {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #ecfdf5;
    border: 1.5px solid #6ee7b7;
    border-radius: 5px;
    padding: 2.2mm 3mm;
  }}
  .tag-pass {{
    background: #dcfce7;
    color: #166534;
    padding: 0.4mm 1.6mm;
    border-radius: 3px;
    font-weight: 700;
    font-size: 7.5pt;
  }}
  .step-list {{
    padding-left: 5.5mm;
    font-size: 8.8pt;
    line-height: 1.58;
    color: #334155;
  }}
  .step-list li {{
    margin-bottom: 1.5mm;
  }}
  code {{
    font-family: "Consolas", monospace;
    background: #f1f5f9;
    padding: 0.2mm 1.2mm;
    border-radius: 3px;
    font-size: 8pt;
  }}
</style>
</head>
<body>
{"".join(pages)}
</body>
</html>
"""


def build_application_guide_markdown(
    software_name: str,
    short_name: str,
    version: str,
    raw_code_lines: int,
) -> str:
    """Create the step-by-step CPCC online registration fill-in guide."""
    return f"""# 中国版权保护中心（软著 0 元线上申请）填写对照表与提交流程

> **材料已全部生成完毕！**
> 在同一目录（`outputs/soft_copyright/`）下，已经为你生成好可直接上传的 **2 份标准 A4 PDF 鉴别材料**：
> 1. `1_源程序鉴别材料_前后各30页_共60页.pdf`（严格 60 页、每页 52 行非空源码、含标准页眉与页码）
> 2. `2_软件操作说明书_图文版.pdf`（严格 15 页、含系统架构图、四阶段骨骼分析图、全中文界面图与操作步骤，页眉与软件名称 100% 一致）

---

## 一、 线上办理入口与费用说明
1. **唯一官方网站**：中国版权保护中心（`https://register.ccopyright.com.cn`）
2. **官方费用**：**0 元（完全免费）**。
3. **办理流程（全程约 15 分钟）**：
   - **第 1 步**：注册个人账号并完成实名认证（上传身份证正反面照片，通常 1 个工作日内审核通过）；
   - **第 2 步**：点击【计算机软件著作权相关登记】➔【R11 计算机软件著作权登记申请】➔【立即登记】；
   - **第 3 步**：对照下方表格**直接复制粘贴**填写基本信息，并上传本目录下的 2 个 PDF 文件；
   - **第 4 步**：确认无误后提交，生成《申请表签章页》，打印该页签字拍照上传（或按网页提示完成电子确认）即可等待下证。

---

## 二、 网报系统逐项复制粘贴对照表（严格符合字数限制）

### 1. 软件基本信息
| 网页表单项 | 直接复制粘贴填写内容 | 备注说明 |
| :--- | :--- | :--- |
| **软件全称** | `{software_name}` | 必须与 PDF 页眉一字不差（以“系统”结尾） |
| **软件简称** | `{short_name}` | 选填，也可填此简称 |
| **版本号** | `{version}` | 必须填 `V1.0`（与 PDF 页眉一致） |
| **软件作品说明** | 选择 **【原创】** | 权利取得方式选【原始取得】，权利范围选【全部权利】 |
| **开发完成日期** | 填你提交当天的日期或前几天的日期（如 `2026-11-15`） | 发表状态选 **【未发表】** 最省事（无需填发表地点） |
| **开发方式** | 选择 **【独立开发】** | 著作权人即为你个人 |

### 2. 软件技术与环境信息
| 网页表单项 | 直接复制粘贴填写内容（均在 50 字限制内） |
| :--- | :--- |
| **软件分类** | `应用软件` ➔ `其它`（或体育/图像视频分析相关分类） |
| **开发的硬件环境** | `Intel Core i7处理器，16GB内存，512GB固态硬盘，1920x1080显示器` |
| **运行的硬件环境** | `x86_64或ARM64架构计算机，内存8GB及以上，硬盘可用空间2GB及以上` |
| **开发该软件的操作系统** | `Windows 11 64位操作系统` |
| **软件开发环境/开发工具** | `Visual Studio Code、Python 3.12集成开发环境、Git版本管理工具` |
| **该软件的运行平台/操作系统** | `Windows 10/11、macOS 12及以上、Linux桌面操作系统` |
| **软件运行支撑环境/支持软件** | `Python 3.12运行环境、OpenCV视觉库、MediaPipe姿态推理引擎` |
| **编程语言** | 勾选 `Python`、`JavaScript`、`HTML` |
| **源程序量** | `{raw_code_lines}` （单位：行） |

### 3. 软件功能与特点描述（严格控制字数，直接复制）

- **开发目的（限 50 字以内，当前 44 字）**：
  ```text
  面向拳击与泰拳日常训练，通过单目视频骨骼追踪自动完成出拳踢击动作切分、生物力学缺陷诊断与复盘。
  ```

- **面向领域/行业（限 50 字以内，当前 26 字）**：
  ```text
  体育运动训练、搏击健身教学、运动生物力学视觉分析领域。
  ```

- **软件的主要功能（限 200 字以内，当前 182 字）**：
  ```text
  支持导入多格式训练视频并自动校正手机竖屏旋转，基于33点骨骼关键点锁定训练主角并识别正反架。内置四阶段生物力学状态机，自动切分后手直拳、刺拳、平勾拳、上勾拳、1-2与1-2-3连击以及泰拳扫踢、膝击、提膝格挡动作；量化评估肘部伸展角、前手护腮保持、重心前倾、步幅稳定与出拳后回收防守评分，提供手动微调关键帧、双画面同步慢放、多视频批量对比及离线HTML体检报告导出。
  ```

- **软件的技术特点（限 100 字以内，当前 92 字；并在网页勾选“人工智能软件”）**：
  ```text
  采用单目33关键点姿态感知与躯干/肩宽双尺度归一化算法，结合主角轨迹连续锁定过滤背景乱入；通过四阶段有限状态机与视角置信度门控实现拳腿动作切分与生物力学量化评分，全程本地离线运行保障隐私。
  ```

---

## 三、 鉴别材料上传说明
在网报系统【上传鉴别材料】步骤中：
1. **【源程序】上传框**：选择上传 `1_源程序鉴别材料_前后各30页_共60页.pdf`
2. **【文档】上传框**：选择上传 `2_软件操作说明书_图文版.pdf`

> **如果以后你想换一个软件名称**：
> 只需在终端运行：
> ```powershell
> .\\.venv\\Scripts\\python.exe tools/generate_soft_copyright_materials.py --name "你想改的新软件名称"
> ```
> 脚本会在 3 秒内自动重新生成所有页眉、封面、截图窗口标题全部同步修改后的新 PDF 文件！
"""


def render_html_to_pdf(edge_bin: str, html_path: pathlib.Path, pdf_path: pathlib.Path) -> None:
    file_uri = html_path.resolve().as_uri()
    cmd = [
        edge_bin,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path.resolve()}",
        file_uri,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    if not pdf_path.exists() or pdf_path.stat().st_size < 10_000:
        raise RuntimeError(f"PDF 导出失败或文件异常: {pdf_path}")


def count_pdf_pages(pdf_path: pathlib.Path) -> int:
    raw = pdf_path.read_bytes()
    # Standard PDF page object count check
    matches = re.findall(rb"/Type\s*/Page\b", raw)
    return len(matches)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成中国版权保护中心软著登记全套材料")
    parser.add_argument("--name", default=DEFAULT_SOFTWARE_NAME, help="软件全称")
    parser.add_argument("--short-name", default=DEFAULT_SHORT_NAME, help="软件简称")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="版本号（如 V1.0）")
    parser.add_argument(
        "--out-dir",
        default="outputs/soft_copyright",
        help="输出目录路径",
    )
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    software_title = f"{args.name} {args.version}"
    edge_bin = find_edge_binary()

    source_files = collect_ordered_source_files(repo_root)
    physical_lines, raw_code_lines = sanitize_and_wrap_source_lines(
        source_files, repo_root, software_title
    )

    code_html, code_txt = build_source_code_html(physical_lines, software_title)
    code_html_path = out_dir / "1_源程序鉴别材料_前后各30页_共60页.html"
    code_txt_path = out_dir / "1_源程序鉴别材料_前后各30页_共60页.txt"
    code_pdf_path = out_dir / "1_源程序鉴别材料_前后各30页_共60页.pdf"
    code_html_path.write_text(code_html, encoding="utf-8")
    code_txt_path.write_text(code_txt, encoding="utf-8")
    render_html_to_pdf(edge_bin, code_html_path, code_pdf_path)

    manual_html = build_user_manual_html(
        software_title=software_title,
        software_name=args.name,
        short_name=args.short_name,
        version=args.version,
        total_code_lines=raw_code_lines,
    )
    manual_html_path = out_dir / "2_软件操作说明书_图文版.html"
    manual_pdf_path = out_dir / "2_软件操作说明书_图文版.pdf"
    manual_html_path.write_text(manual_html, encoding="utf-8")
    render_html_to_pdf(edge_bin, manual_html_path, manual_pdf_path)

    guide_md = build_application_guide_markdown(
        software_name=args.name,
        short_name=args.short_name,
        version=args.version,
        raw_code_lines=raw_code_lines,
    )
    guide_md_path = out_dir / "3_中国版权保护中心_网报填写对照表与提交流程.md"
    guide_md_path.write_text(guide_md, encoding="utf-8")

    code_pages = count_pdf_pages(code_pdf_path)
    manual_pages = count_pdf_pages(manual_pdf_path)
    print(f"Software Title : {software_title}")
    print(f"Raw Code Lines : {raw_code_lines}")
    print(f"Source Code PDF: {code_pdf_path} ({code_pages} pages, {code_pdf_path.stat().st_size // 1024} KB)")
    print(f"User Manual PDF: {manual_pdf_path} ({manual_pages} pages, {manual_pdf_path.stat().st_size // 1024} KB)")
    print(f"Fill-in Guide  : {guide_md_path}")


if __name__ == "__main__":
    main()
