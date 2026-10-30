# 私有素材采集准备与本地清单

状态：T02 设备 B 已实现非敏感准备工具；真实参与者、授权和样本覆盖仍由用户完成，T02 保持 Doing。本工具不读取视频、不上传内容，也不判断动作质量。

## 使用边界

真实视频、姓名、联系方式、身体信息、授权原件和逐人清单只能保存在用户控制的本地位置，不进入 Git。仓库只保存 schema、校验代码和这份说明。

清单使用匿名 `participant_id`、`sample_id` 和不含路径的 `local_file_label`。授权原件另存，只在清单写不含姓名或路径的 `reference_id`。性别分组只能来自参与者自愿提供；不得根据外观、声音、衣物或骨架推断。拒绝提供非必要性别信息时使用 `not_provided`。

本工具不是法律授权表，也不会代替用户向参与者说明用途、访问人、保留时间、退出和删除方式。撤回后把授权状态改为 `withdrawn`、相关样本改为 `do_not_use`，并按实际约定删除本地派生数据。

## 创建本地私有清单

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe tools\validate_sample_intake.py --init work\sample-intake.private.json --collection-id collection-local-001 --purpose local_pose_engineering
```

输出文件名必须以 `.private.json` 结尾；该后缀和 `work/` 均被 Git 忽略。命令不会覆盖已有文件，也不会在输出中回显本地路径。

按实际情况编辑后校验：

```powershell
.\.venv\Scripts\python.exe tools\validate_sample_intake.py work\sample-intake.private.json
```

校验只读取 JSON，不打开 `local_file_label` 对应的视频。输出包含结构、授权门禁和 G1 聚合计数，不输出参与者代号、文件名或路径。`G1_READY: yes` 仅表示清单达到冻结的采集数量与拍摄条件，不表示姿态、动作分析或女性内部效果已经通过。

## 字段填写

参与者字段：

- `participant_id`：本地匿名代号，不使用姓名、手机号、邮箱或社交账号。
- `adult_confirmed`：参与者已确认是成年人时为 `true`；未确认的样本不能标为可用。
- `self_reported_gender_group`：只按本人自愿提供的信息填写；没有提供时选 `not_provided`。
- `authorization.status`：`pending`、`granted` 或 `withdrawn`。
- `authorization.reference_id`：授权原件的本地不透明编号；不写路径或个人信息。
- `authorization.allowed_uses`：只勾选实际获得同意的用途。当前工程试跑使用 `local_pose_engineering`，媒体兼容检查使用 `local_media_probe`。
- `retention_review_date`：下一次检查是否继续保留的日期，不等于默认永久保留。

样本字段：

- `sample_id`、`session_id`、`lineage_id`：匿名代号；同一原片的裁剪、镜像或重编码版本使用同一 `lineage_id`。G1 数量按唯一 `lineage_id` 统计，不用多个派生版本抬高样本数。
- `state`：计划、已拍摄、已探测或已撤回。只有已拍摄或已探测且授权范围匹配的样本可以设为 `eligible`。
- `local_file_label`：只写文件名标签，例如 `clip-001.mp4`，不写绝对路径。
- `video_sha256`、`duration_ms`：完成媒体探测后填写；`probed` 状态必须有这两项。
- `expected_class`：规范片为 `normative`；预先设计的坏片为 `bad`，并在 `bad_reasons` 写明类别。不得要求参与者故意做危险错误动作。
- `capture`：记录机位、站架、摄像头、镜像保存、目标帧率、动作次数、衣物可见性组和五项拍摄检查。衣物字段用于姿态可见性覆盖，不评价体型或穿着。

## 当前冻结的 G1 数量

- 至少 10 条通过拍摄检查的规范片。
- 至少 3 名独立参与者，女性人数至少占 70%。
- 另有至少 5 条预先分类坏片。
- 规范片为侧面或前侧约 45°，单人、全身可见、手机固定、无遮挡、光线可用，每条 5–10 次自然后手直拳。

只有 `purpose` 为 `demo_g1` 且参与者明确授权该用途的清单才可能显示 G1 就绪。清单达到数量后仍需运行视频、时间轴和骨架人工核对。多人女性内部的衣物与相对身体尺度覆盖也必须单独报告，不能用总数量替代。

## 稳定错误码

校验失败只输出错误码和 JSON 字段位置，不回显值。主要门禁包括：重复匿名 ID、样本引用未知参与者、未授权或超出授权用途、未确认成年人、撤回样本仍标可用、已探测样本缺少哈希/时长，以及规范片和坏片分类不一致。
