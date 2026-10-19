# T03 可复现工程环境

日期：2026-10-19。范围：Windows x64、本地 CPU、Python 3.12。这里建立工程骨架、合成契约入口和模型资产校验，不包含视频解码、真实姿态评估或动作反馈。

## 已验证基线

设备 B 初始实现环境：

- Windows x64。
- CPython 3.12.10；项目明确限制 `>=3.12,<3.13`，尚未把本机已有的 Python 3.14 写成支持版本。
- pip 26.2.1。
- MediaPipe 1.1.1、OpenCV contrib 5.0.0.93、NumPy 2.5.4、Streamlit 1.66.0；T04 在同一锁文件追加 PyAV 19.0.1。
- 契约依赖沿用 `requirements-contracts.txt`，完整解析环境固定在 `requirements.lock`。
- MediaPipe Pose Landmarker Lite 固定到 Google Cloud Storage generation `1682624738331272`，大小与 SHA-256 见 `configs/model-assets.json`。

设备 A 已于 2026-10-19 用 CPython 3.12.14 / pip 26.2.1 在新环境独立复现；48 个锁定依赖逐项一致。详见 [设备 A 验收记录](verification/T03_DEVICE_A.md)。T03 已完成；支持范围仍限本页实际验证的 Windows x64 / Python 3.12。

MediaPipe 依赖 `opencv-contrib-python`，因此不要再安装 `opencv-python` 或 `opencv-python-headless`；这些发行包共享 `cv2` 命名空间。本机没有系统级 FFmpeg/FFprobe；T04 使用 PyAV Windows wheel 随附的 FFmpeg 库读取解码 PTS，不要求另装可执行文件。真实手机编码范围仍须按 [T04 视频输入说明](VIDEO_IO.md) 实测，不能把合成视频通过写成手机格式已兼容。

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

若已具备 Python 3.12，可直接用该解释器创建环境，无须重复安装 Python。独立复现时应使用新目录，保留已有环境；设备 A 使用 `work/t03-device-a/.venv`。采用其他环境位置时，将本文命令中的 `.venv` 换成实际目录。

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

受限环境若提示 Matplotlib 默认缓存目录不可写，可将缓存放到被 Git 忽略的项目 work 下；设备 A 已用此方式复查模型冒烟：

```powershell
New-Item -ItemType Directory -Force -Path work/matplotlib-cache | Out-Null
$env:MPLCONFIGDIR=(Resolve-Path work/matplotlib-cache).Path
```

该设置仅影响当前终端及其子进程，不修改模型、依赖或验收阈值。

工程界面外壳可用以下命令启动：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

页面只显示环境与合成样例摘要，不接受真实视频，不提供动作判断。

## 双机验收状态

- 设备 B：已从空虚拟环境按 `requirements.lock` 重装；25 项 unittest、3 个合成契约 bundle、依赖一致性、模型重新下载后的大小/SHA-256、64×64 空白帧加载和 Streamlit 健康检查均由实际运行确认。
- 设备 A：已在新环境按锁文件安装，48 个版本一致；模型实际下载与哈希、两条冒烟、25 项 unittest、pip check、Streamlit 脚本渲染和本地健康检查全部通过。具体结果见 [独立记录](verification/T03_DEVICE_A.md)。

T03 已标 Done，下一任务 T04 开始实现视频解码与真实时间轴。后续新机器或升级复现仍应记录 Python、pip 和各验证退出状态。模型下载或模型冒烟失败时保留原始错误，不要改清单 URL、哈希或版本来绕过校验。
