# 🚀 Google AI Ultra 云端 AI 工作站 (MCP Agent 托管版)

一套专为 **Google AI Ultra（5x 套餐：1,000 CCU/月 + 20TB 云盘；20x 套餐：2,000 CCU/月 + 30TB 云盘）** 打造的云端算力智能调度技能（Agent Skill）与 Colab 自动化工作站模板。

通过结合 Google 官方 [`googlecolab/colab-mcp`](https://github.com/googlecolab/colab-mcp)、[`google-colab-cli`](https://pypi.org/project/google-colab-cli/)、[`deluxebear/colab-dfs-ext`](https://github.com/deluxebear/colab-dfs-ext) 与本仓库的 **`colab-ultra-workstation`** 技能，无需申请任何 API Key，让本地 AI Agent（Claude Code / Cursor / Codex / Antigravity / Windsurf）全自动遥控云端 **96GB Blackwell (G4)**、**L4**、**H100** 等算力与 **20TB/30TB Google Drive** 超大云盘。

---

## 📦 仓库结构

```text
colab-ultra-workstation/
├── SKILL.md                                           # Agent 智能硬件调度与防烧钱拦截技能定义
├── templates/
│   └── workstation_init.py                            # 云端硬件探测与参数自动调优代码模板
├── notebooks/
│   └── Google_AI_Ultra_Workstation.ipynb              # 完整实测 Colab 工作站记事本（含硬件顾问、20TB/30TB 云盘挂载、防烧钱下载器、Qwen2.5-1.5B-X-Writer 实测）
└── skills/
    └── colab-ultra-workstation/                       # 标准 Skill 目录镜像
```

---

## ⚡ 快速开始

### 1. 安装 `colab-mcp`（含 SOCKS 代理兼容修复）

在开启了系统 SOCKS 代理的环境下，直接使用 `uvx` 运行官方 `colab-mcp` 会因缺少 `socksio` 崩溃。推荐使用 `uv tool install --with socksio` 固化为本地二进制命令，实现毫秒级冷启动：

```bash
uv tool install --with socksio --force git+https://github.com/googlecolab/colab-mcp
```

在客户端（Claude Code / Cursor / Antigravity 等）的 `mcpServers` 中添加：

```json
"colab-mcp": {
  "command": "colab-mcp",
  "args": []
}
```

### 2. 安装 `google-colab-cli` 与 `colab-dfs-ext`（零浏览器静默挂载 20TB/30TB 云盘）

传统 `drive.mount('/content/drive')` 每次重连都需在浏览器手动点 OAuth 弹窗，且网页断开易导致临时 Token 过期掉盘。配合 [`deluxebear/colab-dfs-ext`](https://github.com/deluxebear/colab-dfs-ext) 可实现**一次授权、永久零浏览器秒级挂载**，并在云端内部启动 `creds shim` 后台自动续期 Token：

```bash
# 1) 安装官方 google-colab-cli（含 socksio 代理兼容）
uv tool install --with socksio google-colab-cli

# 2) 一键注入 colab-dfs-ext 零侵入扩展（.pth 钩子）
curl -fsSL https://raw.githubusercontent.com/deluxebear/colab-dfs-ext/main/install.sh | sh

# 3) 首次一次性 OAuth 授权（后续永久静默挂载）
colab dfs consent

# 4) 将默认 `colab drivemount` 也路由至零浏览器通道
echo 'export COLAB_DFS_DEFAULT=1' >> ~/.zshrc
```

常用零浏览器云盘运维命令：
```bash
colab dfs mount -s <session>                       # 秒级静默挂载 /content/drive（含后台 Token 自动续期）
colab dfs mount -s <session> --readonly            # 只读挂载（保护 20TB/30TB 模型库防脚本误删）
colab dfs mount -s <session> --force --purge-state # DriveFS/FUSE 假死原地抢救重挂（无需重启 Runtime）
colab dfs status -s <session>                      # 查看云盘挂载与 Token 守护进程状态
```

### 3. 安装 `colab-ultra-workstation` 技能

```bash
git clone https://github.com/yuanyang749/colab-ultra-workstation.git ~/.agents/skills/colab-ultra-workstation
```

---

## 🧠 核心特性：双通道遥控、智能选卡矩阵与防烧钱守门员

1. **全尺寸任务选卡矩阵（视频 + 基础 LLM，兼容 5x 1,000 CCU 与 20x 2,000 CCU）**：
   - **`CPU` (0 CCU/h)**：挂载 20TB/30TB Google Drive、从 HuggingFace 首次下载几十 GB 模型权重。
   - **`T4 GPU` (15GB, 1.19 CCU/h)**：1.5B ~ 9B LLM 的 4-bit 量化推理。
   - **`L4 GPU` (22.5GB Ada, 1.71 CCU/h)**：Wan2.1-1.3B / Flux 轻量生成；1.5B~9B 满血 BF16/FP8 推理与微调；**32B 4-bit (AWQ/Q4) 极限卡位推理**。
   - **`G4 GPU` (96GB RTX PRO 6000 Blackwell, 6.7~8.71 CCU/h)**：**主力王炸卡（全面替代 A100/H100）**。具备 1 PFLOPS BF16 / 2 PFLOPS FP8 / 4 PFLOPS FP4 算力，96GB 超大显存支持 **Wan2.1-14B / HunyuanVideo 100% 显存常驻（零 CPU Offload）**，以及 **32B/70B LLM 的 FP8 超长上下文推理与 QLoRA 微调**。
2. **零浏览器静默挂载与不断联守护（`colab-dfs-ext`）**：
   - 彻底告别 `drive.mount` 人工弹窗确认，支持 `--readonly` 只读防误删挂载、云端后台 Token 自动续期（合盖睡觉跑长任务不掉盘）以及 FUSE 假死原地重挂。
3. **防烧钱硬拦截（Burn-Rate Guard）**：
   - 当检测到正挂载高价显卡（G4 / A100 / H100，$\ge 5$ CCU/h）去首次下载大模型时，自动触发拦截提醒，杜绝开着高价显卡干等下载的算力浪费。
4. **大炮打蚊子预警**：
   - 当检测到使用 G4/H100 运行 <9B 小模型或轻量任务时，自动提醒降级至 L4/T4 节省 CCU。

