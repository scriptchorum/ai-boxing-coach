# T04 视频探测、解码与时间轴

日期：2026-10-19。状态：设备 B 已完成不依赖真人素材的前半段，T04 保持 Doing。这里验证工程接口和合成媒体，不代表真实手机格式或动作画面已经验收。

## 当前实现

`src/boxing_coach/video_io.py` 使用 PyAV 19.0.1 直接访问随 Windows wheel 提供的 FFmpeg 库。系统不需要另装 `ffmpeg.exe` 或 `ffprobe.exe`。选择该接口是为了读取每个解码帧的 PTS 与 time base；OpenCV 仍保留给后续图像处理，但不作为变帧率时间轴的时钟来源。

当前能力：

- 探测 AVI、M4V、MKV、MOV、MP4、WEBM 中的单视频流。
- 完整解码后记录源帧号、原始 PTS、time base、相对首帧时间与帧时长。
- 固定帧率和变帧率都使用解码 PTS；不使用 `frame_index / fps` 代替时间。
- 读取显示矩阵或 `rotate` 元数据，支持 0/90/180/270 度旋转和容器级水平反射。
- 先应用容器显示变换，再按用户明确选择执行镜像纠正；不会根据人物或动作自动猜镜像。
- 输出文件 SHA-256 和契约 v0.1.0 所需的 `media` 字段；诊断 JSON 不包含输入路径。
- 坏文件、无视频流、多个视频流、缺失或倒退 PTS、异常显示矩阵等情况返回稳定错误码，不回显私人路径。

本轮输入上限是工程保护值，不是女性身体或动作标准：单文件不超过 512 MiB、时长不超过 120 秒、最多 20,000 帧、像素总量不超过 3840×2160。产品建议仍是 10–30 秒、720p 或 1080p、30/60fps；真实手机测试后可在不削弱安全边界的前提下调整格式说明。

## 本地命令

安装或更新锁定环境后，在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\boxing-coach-video.exe probe "D:\受控目录\试片.mp4"
```

需要核对逐帧时间和首帧解码尺寸时：

```powershell
.\.venv\Scripts\boxing-coach-video.exe probe "D:\受控目录\试片.mp4" --timeline --decode-first-frame
```

命令只在本机读取文件，不上传视频。输出会包含文件哈希、编码、显示后宽高、帧数、时长、旋转、是否变帧率和可选时间轴，但不会包含输入路径。真实视频及输出诊断仍属于本地数据，不应提交 Git。

Python 调用先探测，再复用同一结果解码；解码前会重新校验文件哈希，避免把另一个文件套用到旧时间轴：

```python
from boxing_coach.video_io import decode_video, probe_video

inspection = probe_video(video_path)
for frame in decode_video(video_path, inspection=inspection, correct_mirror=False):
    use(frame.source_frame_index, frame.timestamp_ms, frame.image_rgb)
```

`correct_mirror=True` 只能来自用户对拍摄方式的明确确认。它是坐标纠正，不改变正架/反架的解剖手别定义。

## 失败原因

命令以 `VIDEO_INPUT_ERROR[错误码]` 返回失败，不回显输入路径。当前错误码按阶段分组：

- 文件：`file_unavailable`、`unsupported_extension`、`file_too_large`。
- 容器/编码：`no_video_stream`、`multiple_video_streams`、`unsupported_codec`、`decode_failed`、`corrupt_frame`。
- 尺寸/显示：`invalid_dimensions`、`resolution_too_large`、`unsupported_display_transform`。
- 时间轴：`no_video_frames`、`missing_pts`、`non_monotonic_pts`、`invalid_duration`、`video_too_long`、`too_many_frames`。
- 一致性：`input_changed`、`timeline_changed`，表示探测后文件或再次解码结果发生变化，必须重新探测，不能套用旧映射。

## 时间与显示定义

- `original_pts_start_ms = 首个解码帧 PTS × time_base × 1000`，允许为负。
- 每帧 `timestamp_ms = 当前原始 PTS 毫秒 − original_pts_start_ms`，首帧为 0。
- 帧号与时间必须严格递增；重复或倒退时间戳会拒绝，不通过排序、补帧或重采样掩盖。
- 当前变帧率标记以相邻 PTS 间隔为工程诊断：相对中位间隔偏差超过 `max(1ms, 5%)` 时标为 VFR，并把契约 `nominal_fps` 设为 null。无论该标记如何，所有计算都使用原始 PTS。
- `rotation_applied_degrees` 表示为得到直立画面而逆时针应用的显示旋转；90/270 度时契约宽高交换。容器反射和用户镜像纠正另外处理。
- 诊断时长取最后一帧相对 PTS 加其解码帧时长；帧时长缺失时才用相邻 PTS 中位数作为最后一帧回退值。

## 合成验证与待办

`tests/test_video_io.py` 在临时目录动态生成媒体，不提交视频二进制。设备 B 还在新的 `work/t04-repro-venv` 中按 49 个锁定包从零安装，运行全套 38 项 unittest、模型冒烟和 `pip check`，并用安装后的 CLI 探测合成 VFR 视频。当前 13 项专项覆盖：

- 合成 30fps、60fps 与不规则 PTS 变帧率。
- 合成 H.264 MP4 的四种直角旋转及其水平反射组合。
- 用户镜像纠正、负起始 PTS、重复时间戳、探测后文件替换。
- 坏 MP4、不支持扩展名、多个视频流、时长上限、契约字段投影、CLI 成功/失败和路径不泄漏。

尚未完成：真实手机 H.264/HEVC、MOV/MP4 旋转、实际 30/60fps、变帧率、坏片的兼容性；也未手工把时间点与真实画面逐帧对照。用户提供本地路径后只在本机运行，记录匿名格式摘要和失败原因，不提交原片、截图或绝对路径。上述真实验证完成前 T04 不能标 Done，也不能据此宣称具备动作分析能力。
