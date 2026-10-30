# 分析质量门禁

状态：T06 设备 B 合成基础 `analysis-quality-0.1`，2026-10-30。当前没有在设备 B 使用真人视频，也没有完成阈值校准、女性内部覆盖或人工真值核对；T06 保持 Doing。

## 与显示过滤分开

`display-quality-0.2` 只决定浏览器是否绘制某些骨架部位，方便人工查看。`analysis-quality-0.1` 决定未来某个指标是否可以继续计算。画面显示了骨架，不代表指标可用；指标门禁通过，也不代表动作正确或人物身份已自动确认。

本版输出是 T06 内部工程诊断，不是 T01 的 `ActionSegment` 或 `ObservationReport`，也没有修改现有 schema。T09/T10 接入正式报告前仍需明确版本化映射并补契约测试，不能把本文件的原因码直接冒充已经交付的用户报告。

逐片诊断 `clip_quality_report` 只输出模型/依赖/配置、帧总数、三类模型状态、完整/部分/隐藏帧数、固定原因计数、显式坏片原因、阈值和插值记录。它不输出视频路径、运行标识、样本 ID、逐帧记录或关键点坐标。

逐片状态只有：

- `rejected`：全部帧隐藏、存在显式坏片原因，或出现未支持的插值配置/数据。
- `review_required`：仍有可查看帧，但必须人工核对；本版不会只凭模型结果把整片自动标成可用。

## 逐指标门禁

`metric_quality_gate` 接收指标所需的关键点索引和事件源帧，只检查输入证据，不计算指标值。

- `rejected`：事件缺失、事件帧未检出或有多个模型候选、所需点缺失/低信号/出框、时间异常、未支持插值，或清单显式记录多人、出框、模糊、遮挡及不支持机位。`value_policy` 固定为 `must_be_null`。
- `review_required`：模型证据暂未发现上述问题，但没有从独立素材清单获得单人确认；值仍必须为空。
- `eligible`：调用方另行提供了单人确认，且所需事件、点位和时间证据通过。本状态只允许后续计算，不等于计算已完成、识别准确或动作合格。

`single_person_confirmed` 必须来自用户核对或受控清单，不能由“单个模型候选”自动填写。显式坏片原因也只能来自实际拍摄/人工记录，不从外观、性别、衣物或身体比例猜测。

## 稳定原因码

逐片帧原因：

- `no_model_pose`、`multiple_model_pose_candidates`
- `critical_landmark_out_of_frame`、`critical_landmark_unusable`
- `torso_landmark_unusable`、`torso_position_jump`、`timestamp_gap`
- `interpolation_not_supported`

逐指标会进一步区分 `missing_event`、`critical_landmark_missing`、`critical_landmark_low_signal` 和 `critical_landmark_out_of_frame`。显式坏片原因包括 `declared_multiple_people`、`declared_out_of_frame`、`declared_blur`、`declared_occlusion`、`declared_unsupported_view`。

“多个模型候选”不是已确认多人；“单个模型候选”也不是单人证明。原因码描述触发门禁的工程证据，不冒充人工真值。

## 阈值与插值

报告记录运行自身的 visibility/presence 阈值，以及现有候选显示规则的 3.0 归一化图像坐标/秒躯干突变阈值和 100 ms 时间间隔阈值。这些都是待真人开发集校准的工程候选，不是人体动作标准。

当前插值策略为禁用，`interpolation_max_gap_ms=0`。报告固定记录策略、配置上限、实际插值帧数和点数；如果配置非零或数据出现 `interpolated` 点，门禁拒绝，不静默补点。后续若验证短缺失插值，必须升级规则版本并增加事件边界保护，不能用插值伪造峰值或回位。

## 当前验证与剩余工作

9 项新增质量门禁测试覆盖逐片计数、显式模糊/遮挡/多人、未检出/多候选、缺失/低信号/出框、时间异常、插值拒绝、运行配置阈值、单人确认和隐私安全导出。全套 91 项 Python、10 项播放器测试及 `pip check` 通过。另用固定 MediaPipe 模型处理临时生成的 3 帧 H.264/MP4 黑片，3 帧均未检出；逐片状态为 `rejected`，逐指标状态为 `rejected` 且 `value_policy=must_be_null`，临时媒体随后清理。

当前测试全部为合成反例，只证明门禁分支和空值策略按约定执行。T06 仍需可核验授权的真人规范片与坏片、女性内部衣物/尺度覆盖、逐片人工核对、误拒绝/漏拒绝统计和阈值冻结；完成这些之前不报告准确率，也不将 T06 标 Done。
