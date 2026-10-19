# 项目交接

更新日期：2026-10-19，Asia/Shanghai。

## 当前状态

P00 策划书评审与执行方案、T01 范围与数据契约均已完成。设备 B 已于 2026-10-19 实现 T03 可复现工程骨架并将其转入 Review；设备 A 尚未按锁文件独立复现，因此 T03 不能标 Done。用户正在录制 T02 起步视频，本轮尚未收到或检查素材。

用户约束保持：
- 以女性身体为主要设计和验证对象，重视女性内部的身体比例与衣物多样性。
- 用户每周约 20 小时；只有用户与 Codex，没有兼职教练。
- 观察版只描述可见事件，不输出专业判错、拳力、伤病或个性化纠错。
- 用户报告第二台电脑已配置；本轮没有远程验证该机。

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

以上 T01 结果只验证契约及校验器行为。项目尚无视频导入、真实视频姿态推理、动作测量、正式用户界面或女性人群效果数据。合成样例没有运动学真实性，不能用其坐标与阈值作为生产算法配置。G0 数据部分、G1–G4 和第二台电脑复现均未宣布通过。

T03 设备 B 新增验证：

- 从新的隔离虚拟环境执行 `pip install -r requirements.lock` 和 `pip install --no-deps -e .`：成功。
- `python -m boxing_coach.smoke`：锁定运行时检查通过，3 个合成契约 bundle 通过。
- `python -m boxing_coach.smoke --with-model`：固定模型哈希通过，64×64 纯黑合成帧完成推理初始化，检测 0 个人符合预期。
- `python -m unittest discover -s tests -v`：25 项测试通过（原 19 项契约测试加 6 项工程结构测试）。
- `python -m pip check`：无破损依赖。
- 模型下载器的不存在文件分支：重新下载 5,777,746 字节并通过固定 SHA-256 校验。
- Streamlit 外壳：本地启动后 `/_stcore/health` 返回 HTTP 200 和 `ok`。

这些结果仅证明设备 B 的工程可安装与运行；没有使用真实视频，也不证明姿态准确率、女性人群效果或专业动作判断。设备 A 的输出不得照抄。

## 下一步

1. 设备 A 拉取包含 T03 实现的 main，按 `docs/ENGINEERING.md` 新建 Python 3.12 环境，依次执行锁文件安装、模型同步、两条冒烟、unittest 和 `pip check`，把实际版本与结果写回本文件；全部通过后再把 T03 标 Done。
2. 用户继续 T02：先录 3–5 条 10–30 秒试片，比较侧面与前侧约 45 度；单人、固定手机、全身留边，开头留 1–2 秒准备，再自然做 5–10 次可分辨后手直拳。不为采集刻意加速、加力或做危险错误动作。
3. 视频保存在本机受控位置，提供文件路径后再整理授权和素材清单。不要上传 GitHub；个人试片不能替代多人女性验证。
4. T03 验收前不认领依赖它的 T04；如果设备 A 复现失败，保留日志并修正锁文件或安装说明，再由两台机器重测。

建议下一次指令：

> 同步仓库并读取 AGENTS.md、HANDOFF.md、BACKLOG.md 和 ENGINEERING.md。T03 在设备 B 已进入 Review；请在设备 A 从零建立 Python 3.12 环境，按工程说明安装锁定依赖、同步并校验模型、运行两条冒烟、全部 unittest 和 pip check。记录实际输出；未全部通过不要把 T03 标 Done，也不要改哈希绕过失败。

## 同步与待处理

设备 B 本轮已通过终端 `git pull --ff-only origin main` 和普通 push 验证 GitHub 网络与仓库写入；T03 认领提交及本次实现收工提交均已同步。收工后已比较本机 HEAD、origin/main 和 GitHub main；具体提交号以 Git 日志为准。设备 A 可以从 main 拉取完整实现开始独立复现。

代码、文档、锁定依赖和合成样例进入 Git。真实视频、逐人记录、授权、模型本体、环境和凭据只在本机。另一台电脑仍需拉取、安装依赖和实际运行，Git 同步不等于产品已有云端历史功能。

## T01 收工复核：2026-10-19

- T01 交付提交为 `4112fc85e208e0597cb1c948ecc475fb6fd4b52d`；本次收工开始时，本机与 GitHub main 均为此提交，工作区干净，仓库仍为 Private。
- 本次只补充交接与任务收工记录，没有修改程序、schema、样例或依赖；沿用 T01 已通过的 19 项测试结果，不把重复同步算成新的开发验收。
- 已检查 20 个跟踪文件与暂存范围：没有发现真实视频、模型权重、虚拟环境、授权原件或常见密钥格式；三个样例均标记 synthetic，且使用人工构造代号与哈希。
- 已确认 .env、.venv、data、recordings、模型 .task 和 Codex 凭据的 6 项忽略规则生效。检查不表示可以提交后续新增的私人数据，后续仍须审查暂存内容。
- 设备 A 收工、无占用中的开发任务。T03 仍未认领；下一台先拉取，再阅读本文件和 BACKLOG.md。T02 视频仍待用户提供本地路径与授权信息。
- 本次最终收工提交以 Git 日志为准；同步后核对实际远端 main、本机 HEAD、origin/main 和文件树，不在文档中写回自身提交号。

## T03 设备 B 收工复核：2026-10-19

- 开始实现前已把任务认领提交同步到 main；收工时远端仍指向该预期提交，没有覆盖另一台电脑的新提交。
- 本次暂存 17 个明确属于 T03 的代码、配置、锁文件、测试和文档文件；`git diff --cached --check` 通过，没有未暂存差异。
- 从空虚拟环境安装、25 项 unittest、合成契约冒烟、模型哈希与空白帧加载、模型实际下载分支、依赖一致性、Streamlit 健康检查和 Markdown 相对链接均已通过。
- 已检查暂存文件与常见凭据格式；没有真实视频、逐人记录、授权原件、密钥、虚拟环境或模型权重。`.venv/`、`work/` 和 `*.task` 的忽略规则已实际命中，本地模型未被 Git 跟踪。
- 实现与交接已按用户“结束同步”指令提交并推送；推送后核对本机 HEAD、origin/main 和 GitHub main 一致。T03 保持 Review，直到设备 A 按 `docs/ENGINEERING.md` 独立复现后才能标 Done。
