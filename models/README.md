# 本地模型目录

模型权重仅保存在本机，不进入 Git。`configs/model-assets.json` 记录允许下载的文件、固定对象版本、大小和 SHA-256。

安装工程后执行：

```powershell
.\.venv\Scripts\python.exe -m boxing_coach.model_assets sync
.\.venv\Scripts\python.exe -m boxing_coach.model_assets verify
```

当前登记 MediaPipe Pose Landmarker Lite，用于工程加载与后续基准测试。拳击视频和女性样本表现按验证计划评估。
