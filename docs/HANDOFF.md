# 项目交接

更新日期：2026-10-19，Asia/Shanghai。

## 当前状态

P00 策划书评审、T01 数据契约和 T03 可复现工程骨架均已完成。设备 B 已认领 T04 视频解码与真实时间轴，并完成不依赖真人素材的核心接口与合成测试；T04 保持 Doing，等待用户提供真实手机片完成格式和画面对照。T02 真实素材尚未接收或检查。

设备 A 本轮基线为 `657697b`，认领提交为 `93032e4`，当前已完成验收并释放任务。本轮未改程序、锁文件或模型哈希；日志保留在本机被忽略的 work/t03-device-a 下，汇总见 [设备 A 记录](verification/T03_DEVICE_A.md)。

用户约束保持：
- 以女性身体为主要设计和验证对象，重视女性内部的身体比例与衣物多样性。
- 用户每周约 20 小时；只有用户与 Codex，没有兼职教练。
- 观察版只描述可见事件，不输出专业判错、拳力、伤病或个性化纠错。
- 设备 B 的结果来自此前交接，设备 A 本轮已独立安装并验证；两台基础工程复现已满足 T03。

## 本次 T04 前半段

- `src/boxing_coach/video_io.py`：本地单视频流探测、文件哈希、完整解码、原始 PTS/time base 映射、显示旋转/反射、用户确认镜像纠正、输入限制和稳定错误码。
- `tests/test_video_io.py`：运行时动态生成合成媒体，覆盖 30/60fps、VFR、全部直角旋转和反射、负起始 PTS、重复 PTS、坏文件、多个视频流、输入上限、文件替换、CLI 与路径隐私；不提交视频二进制。
- `pyproject.toml`、`requirements.lock`、`smoke.py`：锁定并检查 PyAV 19.0.1。Windows wheel 自带 FFmpeg 库，本机仍无系统级 `ffmpeg.exe`/`ffprobe.exe`。
- `docs/VIDEO_IO.md`：本地命令、输入上限、时间与显示定义、错误边界和真实素材待办。
- 在新的 `work/t04-repro-venv` 中按 49 个锁定包从零安装成功；全套 38 项 unittest 通过，其中视频专项 13 项；模型冒烟、安装后 CLI 合成 VFR 探测、`pip check` 与 `git diff --check` 通过。该目录与合成媒体均被 Git 忽略。结果只证明合成媒体接口，不证明真实手机编码兼容或动作分析效果。

## 本次 T03 交付

- `pyproject.toml` 与 `src/boxing_coach/`：Python 3.12 项目、环境/契约冒烟入口和模型资产工具。
- `requirements.lock`：设备 B 从空环境实际安装并验证的 Windows x64 完整依赖快照。
- `configs/model-assets.json` 与 `models/README.md`：固定 MediaPipe Pose Landmarker Lite 对象 generation、大小、SHA-256 和本地下载规则；模型权重被忽略，不进 Git。
- `app/main.py`：仅显示工程状态和合成样例摘要的本地 Streamlit 外壳；不接收视频、不判断动作。
- `tests/test_project_structure.py`：工程根目录、锁文件、模型清单和哈希拒绝路径的 6 项测试。
- `docs/ENGINEERING.md`：Windows 安装、模型同步、冒烟、测试和设备 A 复现步骤。

设备 B 使用 CPython 3.12.10、pip 26.2.1。直接依赖与关键解析版本见工程说明和锁文件。FFmpeg/FFprobe 未安装，留待 T04 对真实时间轴与格式支持进行验证，不能把当前骨架称为视频分析应用。

## 本次 T01 交付

- schemas/contracts.schema.json：正式 VideoManifest / PoseSequence / ActionSegment / ObservationReport，版本 0.1.0。
- tools/validate_contracts.py：离线结构与跨记录语义校验，包括时间轴、左右手、缺失、证据、数值与分区泄漏。
- tests/fixtures/：3 个人工构造样例，分别表示完整、回位缺失、授权撤回；没有真人信息、真实骨架或视频。
- tests/test_contracts.py：19 个测试方法，含有效场景和多组反例。
- requirements-contracts.txt：本机实际安装并验证的契约依赖固定版本；不是完整应用锁文件。
- docs/CONTRACTS.md：字段、单位、指标、限制和另一台电脑的运行命令。
- docs/GATES.md：冻结 v0.1 验收条件，区分 T01 完成与 G0 真实素材尚待验证。
- README、AGENTS、任务、决策与执行方案已更新。

实际用户工时未登记，不把 T01 的 6–10 小时估算记为实耗。用户录制时间归 T02。原有全项目基础工时 194–314 小时和 25% 余量保留，待首周按实际投入重估。

## 已运行的验证

环境：Windows x64、Python 3.12.14、jsonschema 4.26.0；其余依赖见 requirements-contracts.txt。

在项目虚拟环境中：
- python tools/validate_contracts.py：3 个合成 bundle 通过结构与语义校验。
- python -m unittest discover -s tests -v：19 个测试通过。
- python -m pip check：依赖关系通过。
- Git 差异、敏感文件与 Markdown 相对链接检查在提交前执行。

以上 T01 结果只验证契约及校验器行为。项目尚无视频导入、真实视频姿态推理、动作测量、正式用户界面或女性人群效果数据。合成样例没有运动学真实性，不能用其坐标与阈值作为生产算法配置。G0 数据部分和 G1–G4 均未宣布通过；双机基础复现结果见下方 T03 记录。

T03 设备 B 新增验证：

- 从新的隔离虚拟环境执行 `pip install -r requirements.lock` 和 `pip install --no-deps -e .`：成功。
- `python -m boxing_coach.smoke`：锁定运行时检查通过，3 个合成契约 bundle 通过。
- `python -m boxing_coach.smoke --with-model`：固定模型哈希通过，64×64 纯黑合成帧完成推理初始化，检测 0 个人符合预期。
- `python -m unittest discover -s tests -v`：25 项测试通过（原 19 项契约测试加 6 项工程结构测试）。
- `python -m pip check`：无破损依赖。
- 模型下载器的不存在文件分支：重新下载 5,777,746 字节并通过固定 SHA-256 校验。
- Streamlit 外壳：本地启动后 `/_stcore/health` 返回 HTTP 200 和 `ok`。

这些设备 B 结果仅证明基础工程可安装与运行；没有使用真实视频，也不证明姿态准确率、女性人群效果或专业动作判断。

T03 设备 A 独立验证：

- Windows AMD64 / CPython 3.12.14 / pip 26.2.1，在新建 work/t03-device-a/.venv 中安装锁定依赖及本项目；48 个依赖版本逐项一致。
- 模型首次下载 5,777,746 字节，固定 SHA-256 校验通过；两条冒烟均退出 0，黑帧 poses=0。
- 25 项 unittest、3 个合成 bundle、pip check 均通过；Streamlit AppTest 渲染无异常，实际本地服务健康端点返回 HTTP 200 / ok，验证后已关闭本轮服务。
- 默认 Matplotlib 缓存目录在受限环境下不可写；改用项目 work 内缓存后重跑模型冒烟通过，未更改依赖、模型或验收阈值。说明已补入 ENGINEERING.md。
- 本轮实际用户工时未登记，不用预算小时替代；没有收到或分析真实视频。详细环境与证据见 [T03_DEVICE_A.md](verification/T03_DEVICE_A.md)。

## 下一步

1. 用户准备好 3–5 条试片后提供本机受控目录路径；设备 B 按 `docs/VIDEO_IO.md` 探测真实文件并记录匿名格式摘要，不上传原片、截图或绝对路径。
2. 对每条真实片核对 H.264/HEVC、MOV/MP4 旋转、30/60fps 或 VFR、首尾画面时间和镜像选择；另用坏片检查失败原因。证据完整后才把 T04 转 Review/Done。
3. 用户继续 T02：试片保持单人、固定手机、全身留边，侧面与前侧约 45 度各拍一些；开头静止 1–2 秒，再自然做 5–10 次后手直拳，不刻意加速、加力或制造危险错误。
4. T04 完成前不认领依赖它的 T05。当前没有姿态提取、动作判断或报告功能；个人试片不能替代 G1 的多人女性验证。

建议下一次指令：

> 同步仓库并读取 AGENTS.md、HANDOFF.md、BACKLOG.md 和 VIDEO_IO.md。T04 合成媒体前半段已由设备 B 实现；真实视频已准备在本机受控目录，请按文档探测并手工核对格式、旋转、帧时间和坏片失败路径。不要上传视频或路径，未完成真实验收不要把 T04 标 Done。

## 同步与待处理

设备 B 已通过普通 Git 将 T04 认领和本次前半段实现同步到 main；收工时核对本机 HEAD、origin/main 与 GitHub main。设备 A 或其他电脑开始后续工作前必须先快进拉取并重新阅读本文件、BACKLOG.md 和 VIDEO_IO.md；具体提交号以 Git 日志为准。

代码、文档、锁定依赖和运行时生成合成样例的测试进入 Git；合成视频本体仍只在被忽略的 work 目录。真实视频、逐人记录、授权、模型本体、环境和凭据只在本机且不得进入 Git。另一台电脑仍需拉取、安装依赖和实际运行，Git 同步不等于真实手机格式已经验证，也不等于产品已有云端历史功能。

## T01 收工复核：2026-10-19

- T01 交付提交为 `4112fc85e208e0597cb1c948ecc475fb6fd4b52d`；本次收工开始时，本机与 GitHub main 均为此提交，工作区干净，仓库仍为 Private。
- 本次只补充交接与任务收工记录，没有修改程序、schema、样例或依赖；沿用 T01 已通过的 19 项测试结果，不把重复同步算成新的开发验收。
- 已检查 20 个跟踪文件与暂存范围：没有发现真实视频、模型权重、虚拟环境、授权原件或常见密钥格式；三个样例均标记 synthetic，且使用人工构造代号与哈希。
- 已确认 .env、.venv、data、recordings、模型 .task 和 Codex 凭据的 6 项忽略规则生效。检查不表示可以提交后续新增的私人数据，后续仍须审查暂存内容。
- 设备 A 收工、无占用中的开发任务。T03 仍未认领；下一台先拉取，再阅读本文件和 BACKLOG.md。T02 视频仍待用户提供本地路径与授权信息。
- 本次最终收工提交以 Git 日志为准；同步后核对实际远端 main、本机 HEAD、origin/main 和文件树，不在文档中写回自身提交号。

## T03 设备 B 收工复核：2026-10-19

以下保留设备 B 收工时的历史状态；设备 A 后续复现现已通过，当前 T03 状态为 Done。

- 开始实现前已把任务认领提交同步到 main；收工时远端仍指向该预期提交，没有覆盖另一台电脑的新提交。
- 本次暂存 17 个明确属于 T03 的代码、配置、锁文件、测试和文档文件；`git diff --cached --check` 通过，没有未暂存差异。
- 从空虚拟环境安装、25 项 unittest、合成契约冒烟、模型哈希与空白帧加载、模型实际下载分支、依赖一致性、Streamlit 健康检查和 Markdown 相对链接均已通过。
- 已检查暂存文件与常见凭据格式；没有真实视频、逐人记录、授权原件、密钥、虚拟环境或模型权重。`.venv/`、`work/` 和 `*.task` 的忽略规则已实际命中，本地模型未被 Git 跟踪。
- 实现与交接已按用户“结束同步”指令提交并推送；推送后核对本机 HEAD、origin/main 和 GitHub main 一致。T03 保持 Review，直到设备 A 按 `docs/ENGINEERING.md` 独立复现后才能标 Done。

## T04 设备 B 阶段收工复核：2026-10-19

- 本次实现基于已同步的 T04 认领提交 `d52b7b4dbd1c1ebb992e1a8b3a3760c4776ff2c6`；提交前本机 HEAD、origin/main 与 GitHub main 均仍为该预期提交，没有覆盖其他电脑的新工作。
- 本次明确暂存 15 个 T04 代码、测试、依赖和文档文件；没有未暂存差异，`git diff --cached --check` 与 Markdown 相对链接检查通过。
- 在新的被忽略环境中按 49 个锁定包从零安装成功；38 项 unittest 全部通过，其中视频专项 13 项；模型冒烟、安装后 CLI 合成 VFR 探测、`pip check` 和 Streamlit AppTest 均通过。
- 已检查暂存内容与常见凭据、私有绝对路径和敏感文件类型；没有真实视频、模型权重、虚拟环境、逐人记录、授权原件、环境文件或密钥。`.venv/`、`work/` 和 `*.task` 的忽略规则继续生效，运行时合成媒体和本地模型未被 Git 跟踪。
- 实现与交接按用户“结束同步”指令提交并推送；推送后核对本机 HEAD、origin/main、GitHub main 和文件树一致，最终提交号以 Git 日志为准。
- T04 保持 Doing。仍需用户提供受控本地路径下的真实 H.264/HEVC、MOV/MP4、30/60fps、VFR 和坏片，逐条核对旋转、镜像、首尾画面时间与错误路径；完成前不进入 T05。本阶段没有姿态提取、动作判断或专业纠错功能。
