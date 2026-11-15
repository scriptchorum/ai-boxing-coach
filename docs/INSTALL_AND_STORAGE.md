# 安装流程、离线就绪与本地数据安全删除（T12）

本文件说明如何在陌生目录复现安装、检查离线模型就绪状态，以及安全清理本地生成的缓存与报告而不误删任何用户原始训练视频。

## 1. 陌生目录安装与离线启动步骤

在已安装 Python 3.12 的 Windows x64 电脑上执行：

```powershell
git clone https://github.com/scriptchorum/ai-boxing-coach.git
cd ai-boxing-coach
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m boxing_coach.model_assets --verify-or-download
```

- **完全离线运行**：模型权重（`models/pose_landmarker_lite.task`）下载并通过 SHA-256 校验后，断网状态下可直接启动预览与姿态复盘服务：
  ```powershell
  .\start-preview.ps1
  ```
- **无隐式上传**：所有视频解码、MediaPipe 关键点推理、后手直拳分段、回手时间计算与报告打包均在本机内存或本机受控目录完成，无任何外发网络请求。

## 2. 本地存储目录与原片保护规则

`src/boxing_coach/local_storage.py` 严格区分**受保护文件**与**可清理派生缓存**：

- **受保护文件（绝不删除）**：
  - 用户导入的原始视频路径（`protected_source_paths`）及所有视频媒体扩展名（`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.m4v`）；
  - 模型权重文件（`.task`, `.tflite`, `.onnx`, `.pt`, `.pth` 等）；
  - 受控缓存目录（`outputs/`, `runs/`, `work/matplotlib-cache/`）之外的任何文件。
- **可清理派生文件**：
  - 仅限 `outputs/`、`runs/` 与 `work/matplotlib-cache/` 内部生成的 `.zip`、`.json`、`.html`、`.png`、`.cache`、`.tmp`、`.log` 派生缓存与报告文件；
  - 当前浏览器会话内的识别缓存、证据包与历史记录可通过界面“清理本地缓存与会话派生数据（保留原片）”按钮一键清除。
