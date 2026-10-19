# 数据契约与观察指标 v0.1.0

日期：2026-10-18。T01 交付；T04 的视频输入实现见 [VIDEO_IO.md](VIDEO_IO.md)。范围：成年女性为主要设计与验证对象的单人后手直拳离线观察。没有专业教练复核；所有数值只描述视频中的事件。当前已有合成媒体解码与 PTS 接口，没有姿态模型管线或动作识别实现。

## 规范文件与入口

- [JSON Schema](../schemas/contracts.schema.json)：Draft 2020-12，根为一个分析 bundle；`$defs` 中分别定义 VideoManifest、PoseSequence、ActionSegment、ObservationReport。
- [校验入口](../tools/validate_contracts.py)：先校验结构，再检查跨记录时间、证据、数值与数据分区。
- [回归测试](../tests/test_contracts.py)：包含有效输入和故意损坏的反例。
- [合成样例](../tests/fixtures/)：完整、缺失回位、授权撤回三种**同一样本的替代状态**，不是三份独立参与者数据。

四种契约的 `schema_version` 和 bundle 版本必须都是 `0.1.0`。所有字段必填；允许缺失的字段显式填 `null`；额外字段拒绝。新字段、单位、枚举或解释变化必须更新版本、样例与兼容说明，不能静默改变含义。ReviewAnnotation 留给 T08 的标注工具，本轮不伪造专业动作质量标签。

根字段为 `schema_version`、`synthetic`、`manifest`、`pose`、`actions`、`report`。实际逐人数据只能在本机受控目录保存；`synthetic: false` 只表示真实数据，绝不表示可以进 Git。仓库样例必须为 `true`，所有 ID、坐标、阈值和全零哈希均为人工构造，不能用于验证女性身体适配或用作模型参数。

## 字段与单位

| 对象 | 主要字段 | 语义 |
| --- | --- | --- |
| VideoManifest | sample_id、participant_id、session_id、lineage_id | 不含姓名的内部代号；lineage_id 把裁剪、镜像、转码等衍生片关联到原始采集 |
| VideoManifest | input_sha256、duration_ms、media | 输入文件哈希；时长毫秒；编码、显示变换后直立画面的宽高像素、原始解码帧总数、逆时针应用的旋转角度、原始首帧 PTS、可空的名义 fps |
| VideoManifest | capture | 手动选择 stance=orthodox/southpaw/unknown、is_mirrored、view=side/front_45/unsupported/unknown；两个候选机位仍待真实比较 |
| VideoManifest | split、consent | development/holdout/unassigned；授权状态及本机授权记录代号，不嵌入授权表或联系人 |
| PoseSequence | model、config | 模型 ID、版本、资产 SHA-256；点位布局、坐标空间、配置版本、缺失与可见性阈值 |
| PoseSequence | frames | 按解码源帧顺序的稀疏或完整序列；source_frame_index、timestamp_ms、landmarks |
| ActionSegment | action_id、action_kind、hand、source、status | 固定动作 rear_straight；解剖手别；manual/automatic；valid/partial/rejected；候选可不完整 |
| ActionSegment | start/extension/return/hold_end 的 ms 与 frame_index | 四个事件各有时刻和证据源帧号；成对存在或成对为 null |
| ObservationReport | model、rule_version、app_version、metric_config | 记录生成依赖；起始参考窗口、画面躯干尺度 px、回位区域相对尺度、100ms 保持要求 |
| ObservationReport | candidate_count、valid_count、usable_ratio | 候选总数、可用于本版指标的完整动作数、两者之比；无候选时比例为 null，非 0 |
| ObservationReport | observations、quality_reasons、generated_at | 结构化观察、原因码、UTC 时间；每项观察必须关联动作与事件证据，包含非执教评价的限制 |

`participant_id` 仅是代号，不能保证数据已经匿名。性别与衣物覆盖由用户自愿提供并在本机群体管理记录中保存，不从骨架推断，也不强制写入单条分析输入。女性参与者比例与内部覆盖由独立评估协议验证，不能靠一个 schema 字段宣称完成。

## 坐标、左右与时间

1. 点位 `index` 使用 MediaPipe 33 点的编号；数组允许保留 1–33 个不同点，不能假定数组位置等于编号。省略的点视为未提供。肩为 11/12、肘为 13/14、腕为 15/16、髋为 23/24，前者是解剖左侧。原始全量输出可保留全部 33 点。
2. `image` 是旋转后直立且已还原镜像的图像坐标，x/y 分别按图像宽/高归一化，z 是模型相对深度值。出框坐标允许超出 [0,1] 以保留拒绝依据；不应把它们裁剪后伪装成可见点。计算二维距离或角度前恢复像素比例。
3. `world` 可为 null；非空字段单位为模型估计的米，原点为人体髋中心。它不是标定场地坐标，不支持真实重心、移动速度、发力或伤病结论。
4. 正架 orthodox 的后手是解剖右手，反架 southpaw 的后手是解剖左手。镜像只改变显示与坐标还原，不能交换后手定义。站架未知时不能产生有效后手观察。
5. `timestamp_ms = 当前帧原始 PTS - 首帧原始 PTS`，可有小数；`original_pts_start_ms` 可为负。源帧号从 0 起，保留原始映射；序列允许稀疏但帧号与时间均须严格递增。禁止用 `index/fps` 替代变帧率时间。
6. 数字计算保留原始时间精度；未来界面按实际帧间隔展示精度，不能把小数位数解释为测量准确度。证据只接受实际保存的帧和对应时间，不能通过四舍五入匹配到别帧。

## 缺失、插值与动作有效性

`missing` 点的 image/world 都为 null。`observed` 或 `interpolated` 点必须有 image。插值点没有直接测得的 visibility/presence，二者必须为 null。插值只允许被直接观察到的左右端点包围，端点总时间跨度不超过配置上限（最大 200/3ms）；校验器检查线性图像坐标一致性。该上限是待验证的工程约束。

有效动作须同时满足：start < extension < return < hold_end；hold_end−return ≥100ms；起始事件在参考窗口结束之后；四个事件对应帧中的同侧肩、肘、腕均在画面范围内、直接观察到并达到配置阈值。它们不能由插值替代。缺少事件写 null，并使用 partial 或 rejected 和明确原因；不能用 0 代替未知。相邻有效动作不能重叠，下一动作在前一次保持完成后开始。

这里的 `valid` 意为符合数据与事件完整性契约，**不代表动作正确**。校验器无法从合成坐标证明“这是拳击”“已经回位”或“连续保持 100ms”；实际连续性、参考区域与机位适用性由 T06/T09/T10 的算法及人工视频复核验证。visibility/presence 也不是动作判断准确率。

## 本版可承载的观察

| metric_id | 定义与字段 | 最少证据与限制 |
| --- | --- | --- |
| return_time_ms | return_ms−extension_ms；value，unit=ms；每项只对应一次有效动作 | 最大可见伸展、回到本片个人起始手位区域、保持完成三个事件的帧和时间；不是防守合格或快慢评分 |
| return_time_repeatability | 同一片、同一场次所有有效动作的回手时间 median_ms / iqr_ms；unit=ms | 至少 5 次有效动作，不能挑选表现好的子集；分位数按排序后 (n−1)×p 线性插值；一致不等于正确 |

最大可见伸展使用同侧腕肩投影距离峰值结合肘角去抖定义；回位事件取进入个人起始区域并连续保持至少 100ms 的**进入时刻**，hold_end 是确认完成时刻。二者不是接触/真实三维最大伸展。区域尺度在开发集确定并记录版本，不能使用统一“标准女性骨架”。

躯干投影偏移仍保留在 [指标路线](DATA_EVALUATION.md) 中，待 T10 证明参考可靠后扩充 schema；肩髋与足部旋转只作研究曲线。本契约不接受技术分、拳力、伤病概率、身体审美或纠错 drill 等新输出。

报告 complete 要求至少 5 次有效动作、所有启用指标有结果，并对每次有效动作提供启用的单次回手指标。partial 必须有可用观察并说明缺失原因。no_observations 为零条观察；rejected 则还必须没有有效动作。部分或拒绝状态须有原因，不强行凑出发现。未授权或授权撤回时必须 rejected，且不能包含骨架或动作处理结果。

## 校验范围

Schema 检查类型、必填、版本、枚举、单位和额外字段；Python 语义层检查关联 ID/哈希/模型/机位、有效时间轴、空值配对、左右手、关键点、插值、计数比例、证据与数值一致性、5 次重复性要求，以及同参与者/素材来源/相同文件跨 development 与 holdout 的泄漏。

它不验证授权是否真实存在、代号是否重新分配、模型输出是否准确、女性比例或抽样代表性，也不访问磁盘上的真实视频来核对哈希。实际媒体限制、人体数量、回位保持和图像质量属于后续任务；坏文件也需要能表示为拒绝记录，所以媒体 schema 不冒充已验证的解码支持列表。

## 在另一台电脑复现

契约校验已在设备 A/B 的 Windows x64 / Python 3.12 上运行通过。下面只安装契约校验依赖；完整工程环境的双机结果见 [ENGINEERING.md](ENGINEERING.md) 和 [设备 A 记录](verification/T03_DEVICE_A.md)。项目目录内运行，Python 3.12 需在该机可用：

```powershell
git pull --ff-only origin main
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-contracts.txt
.\.venv\Scripts\python.exe tools/validate_contracts.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

不需要激活 PowerShell 脚本、API 密钥、GPU 或模型下载。默认校验三个合成样例；成功退出码 0，失败为 1。显式指定一个或多个本机 bundle 路径时额外检查分区一致性。报错输出错误码与字段位置，不回显字段值。真实素材与派生记录保持在 Git 忽略的本机目录。
