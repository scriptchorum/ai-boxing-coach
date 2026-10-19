# T03 设备 A 独立复现记录

日期：2026-10-19。结果：通过。实现基线为设备 B 的 `657697b45c6d55b8bc717eee00925fe4aa695abd`，设备 A 认领提交为 `93032e4`。本轮没有修改程序、模型清单或依赖锁定版本。

## 环境与安装

- Windows / AMD64，CPython 3.12.14，pip 26.2.1。
- 在本机新建 `work/t03-device-a/.venv`；原有 T01 `.venv` 保留。本次没有复用既有 site-packages。
- 从 PyPI 按 `requirements.lock` 安装，再以 `pip install --no-deps -e .` 安装本项目，均成功退出。
- 逐一读取实际包元数据，48 个锁定依赖版本全部一致。构建依赖按 pyproject.toml 的固定版本由安装器处理。
- 关键版本：jsonschema 4.26.0、MediaPipe 1.1.1、NumPy 2.5.4、OpenCV contrib 5.0.0.93、Streamlit 1.66.0。
- 本轮锁文件内容未更改，SHA-256 为 `62c09eb93990ec862301822530927b86f489aba2ce2b004c561ec3f00ab2c95f`（本机 checkout 字节）。

以下命令中的 `python` 均指本轮新环境解释器。原始安装、验证日志及 JSON 摘要保存在被 Git 忽略的 `work/t03-device-a/`，不提交包含本机绝对路径的原始日志。

## 实际检查

| 检查 | 结果 |
| --- | --- |
| 模型下载器 sync | 下载前本机没有模型；下载后大小 5,777,746 字节，固定 SHA-256 校验通过 |
| `python -m boxing_coach.model_assets verify` | 退出 0 |
| `python -m boxing_coach.smoke` | 退出 0，3 个合成契约 bundle 通过，T03_SMOKE_OK |
| `python -m boxing_coach.smoke --with-model` | 退出 0，64×64 黑帧完成 CPU 推理，poses=0，T03_SMOKE_OK |
| `python -m unittest discover -s tests -v` | 退出 0，25 项测试全部通过 |
| `python -m pip check` | 退出 0，无破损依赖 |
| Streamlit AppTest | app/main.py 渲染无异常，中文标题和工程阶段提示存在 |
| Streamlit 本地服务 | 仅绑定 127.0.0.1，健康端点 HTTP 200 / ok；验证后已停止本轮启动的服务 |

模型使用清单中的固定 generation `1682624738331272`；SHA-256 为 `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a`。模型仍由 Git 忽略。

首次模型检查成功退出，但受限环境无法创建默认 Matplotlib 缓存目录。将 `MPLCONFIGDIR` 指向本机 `work/t03-device-a/matplotlib-cache` 后，只重跑相关的模型冒烟，仍退出 0，缓存权限提示消失。MediaPipe 的 feedback tensor 日志仍可出现，本轮没有通过隐藏日志或改模型哈希绕过检查。

## 验收结论与范围

结合设备 B 在 [工程说明](../ENGINEERING.md) 和 [交接](../HANDOFF.md) 中记录的独立安装与测试，T03 的双机基础工程复现条件已满足，可标 Done。设备 A 已释放任务，下一开发项为 T04 视频解码与真实时间轴。

这不是视频分析准确率验证。真实女性样本尚未接收，T02/G0 的素材条件未通过；当前界面仍只是工程外壳。完整产品安装、历史与删除的双机测试仍属于 T15。产品继续先满足用户本机复盘的需要，以女性身体多样性为核心，不提供未经专业复核的动作判错或纠错练习。
