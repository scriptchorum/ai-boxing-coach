# 双电脑 Codex 同步

## 只有一台电脑有视频时

另一台仍可拉取 main，读取代码、方案、交接与聚合验证记录，继续开发并运行合成测试，无需先复制真实视频。

如果要在另一台检查真实画面或调试人体识别效果，需要用 U 盘或用户选定的私人存储另行传输视频，不通过 GitHub。两台的本地路径可以不同，在页面各自选择即可；模型与 Python 环境也需各自配置。

可以由有素材的电脑做真实验证、另一台做代码开发，但一次仅一台修改共享 main。切换前提交同步，开始后拉取并阅读交接。没有素材的电脑不能声称重复验证了真实视频，只能引用交接中的既有结果。

## 同步范围

GitHub 私有仓库是已提交项目文件的共享位置。两台电脑各自克隆同一仓库，轮流工作。
同步源码、README、AGENTS.md 和交接文档；运行环境、密钥、数据和权重分别配置。

Codex 会读取项目中的 AGENTS.md。聊天记录与本地文件的访问是不同机制；本流程只用 Git 同步仓库文件。
参考：[项目与聊天](https://learn.chatgpt.com/docs/projects)、[AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md)。

## 每台电脑首次接入

1. 确认已安装 Git（终端运行 `git --version`）。
2. 用有权限访问本仓库的 GitHub 账号完成 Git 身份验证。GitHub 插件连接不等于终端 Git 已登录。
   Windows 的 Git Credential Manager 可能在首次 clone/push 时打开登录页面；在浏览器完成登录，不把令牌发到聊天或写入文件。
3. 在希望保存项目的位置打开终端，运行：

```powershell
git clone https://github.com/scriptchorum/ai-boxing-coach.git
cd ai-boxing-coach
git config --local pull.ff only
git config --local fetch.prune true
git config --local push.default simple
git config --local user.name "scriptchorum"
git config --local user.email "149955187+scriptchorum@users.noreply.github.com"
git status --short --branch
```

上面的提交身份使用仓库所有者的 GitHub 隐私邮箱；如使用其他账号，请换成该账号的身份。
不要在已经有项目文件的同名文件夹里再次 clone，也不要重新 git init 建立不相关的提交历史。

4. 在该电脑的 Codex 中添加本地项目，选择刚克隆的 `ai-boxing-coach` 文件夹作为主文件夹。
5. 在该项目中开始聊天并说：“读取 AGENTS.md 和 docs/HANDOFF.md，检查当前进度和 Git 同步状态。”

## 每次开始工作

工作区应当干净，且当前分支应为 main：

```powershell
git status --short --branch
git pull --ff-only origin main
```

如果第一条显示未提交文件，先让 Codex 检查和保留这些改动，再处理同步。
如果拉取报错，就停在当前状态处理错误；不要忽略报错后继续切换电脑。

## 每次结束工作

让 Codex 更新交接文档并检查改动。确认文件均应进入仓库后：

```powershell
git status --short
git diff
git add <本次需要提交的文件路径>
git diff --cached
git commit -m "描述本次工作"
git push origin main
git fetch origin
git rev-parse HEAD
git rev-parse origin/main
git status --short --branch
```

`<本次需要提交的文件路径>` 是占位符，需换成实际路径；可让 Codex 执行这一过程。
只有推送成功、两个提交 ID 一致且工作区干净，才完成本机这次交接。
另一台电脑随后仍需执行 pull，才会取得这次改动。

如果没有新修改，但之前已有未推送提交，仍需执行 push。不要为同步创建无意义的空提交。

## 两台电脑的规则

- A 收工并 push 成功，B 再 pull 开工；反向同理。
- 同时开发时，各用自己的分支，完成后通过合并整合。
- 出现 `non-fast-forward` 或 `Not possible to fast-forward`：保留两边提交，让 Codex 检查差异并处理合并。
- 不使用 force push 或强制重置覆盖另一台电脑的工作。
- 不把包含 `.git` 的工作目录交给网盘自动双向同步。
- 首次接入后可用一个小的交接文档更新进行 A 推送、B 拉取、B 推送、A 拉取的完整验证。

## 常见问题

- `Repository not found`：检查 URL、登录账号和私有仓库访问权限。
- `Authentication failed`：重新完成 Git 的账号授权；不要公开分享访问令牌。
- GitHub 网页/插件可用但终端连接失败：检查这台电脑的网络、代理和终端 Git 身份验证。
- `Author identity unknown`：设置上面的仓库级 user.name/user.email。
- 视频、模型文件没有同步：默认已被忽略；这些资料需要另行复制或选择专用存储。
- 另一台电脑的 Codex 不知道进度：确认已 pull，并让其读取 docs/HANDOFF.md。

不要仅凭网页能看到文件就认定两台电脑都已经接入。
