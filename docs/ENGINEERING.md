# T03 可复现工程环境

日期：2026-10-19。范围：Windows x64、本地 CPU、Python 3.12。这里建立工程骨架、合成契约入口和模型资产校验，不包含视频解码、真实姿态评估或动作反馈。

## 已验证基线

设备 B 实测环境：

- Windows x64。
- CPython 3.12.10；项目明确限制 `>=3.12,<3.13`，尚未把本机已有的 Python 3.14 写成支持版本。
- pip 26.2.1。
- MediaPipe 1.1.1、OpenCV contrib 5.0.0.93、NumPy 2.5.4、Streamlit 1.66.0。
- 契约依赖沿用 `requirements-contracts.txt`，完整解析环境固定在 `requirements.lock`。
- MediaPipe Pose Landmarker Lite 固定到 Google Cloud Storage generation `1682624738331272`，大小与 SHA-256 见 `configs/model-assets.json`。

MediaPipe 依赖 `opencv-contrib-python`，因此不要再安装 `opencv-python` 或 `opencv-python-headless`；这些发行包共享 `cv2` 命名空间。FFmpeg/FFprobe 在本机尚未安装，具体可执行文件版本和视频格式支持由 T04 实测后锁定，不能把 T03 的 Python 导入成功写成视频解码完成。

## Windows 安装

在仓库根目录执行。命令不要求激活 PowerShell 虚拟环境：

```powershell
py install -y 3.12
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip==26.2.1
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

`requirements.lock` 是本次 Windows/Python 3.12 的完整解析结果。升级任何直接或间接依赖前，须在隔离环境重新安装、导入和运行全部测试，再更新锁文件与本文，不能只改顶层版本。

## 模型资产

模型文件不进 Git。下载器只接受清单内的固定 URL，并在落盘前检查文件大小和 SHA-256：

```powershell
.\.venv\Scripts\python.exe -m boxing_coach.model_assets list
.\.venv\Scripts\python.exe -m boxing_coach.model_assets sync
.\.venv\Scripts\python.exe -m boxing_coach.model_assets verify
```

当前 Lite 模型只是 CPU 冒烟与后续 T05 基准候选。尚未在真实拳击、女性身体比例或不同衣物上验证。Full/Heavy 不在本轮自动下载，待 T05 以真实素材和设备性能比较后再决定。

## 验证入口

运行不需要密钥、GPU或真实视频：

```powershell
.\.venv\Scripts\python.exe -m boxing_coach.smoke
.\.venv\Scripts\python.exe -m boxing_coach.smoke --with-model
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

第一条检查 Python 与锁定包版本并运行三个合成契约 bundle；第二条额外校验本地模型并在 64×64 纯黑合成帧上初始化推理。黑帧输出零个人是预期结果，只证明模型文件和本地运行时能加载，不证明姿态准确率。

工程界面外壳可用以下命令启动：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

页面只显示环境与合成样例摘要，不接受真实视频，不提供动作判断。

## 双机验收状态

- 设备 B：已从空虚拟环境按 `requirements.lock` 重装；25 项 unittest、3 个合成契约 bundle、依赖一致性、模型重新下载后的大小/SHA-256、64×64 空白帧加载和 Streamlit 健康检查均由实际运行确认。
- 设备 A：需要拉取包含本实现的提交后按上述命令重新建立环境并记录实际输出。未得到设备 A 结果前，T03 保持 Review，不能标 Done。

设备 A 复现时应记录 Python 与 pip 版本、四条验证命令的退出状态，以及 `boxing_coach.smoke` 输出的锁定依赖版本。模型下载或模型冒烟失败时保留原始错误，不要改清单 URL、哈希或版本来绕过校验。
