# 🚀 Google AI Ultra 云端 AI 工作站 (MCP Agent 托管版)

一套专为 **Google AI Ultra（5x 套餐：1,000 CCU/月 + 20TB 云盘；20x 套餐：2,000 CCU/月 + 30TB 云盘）** 打造的云端算力智能调度技能（Agent Skill）与 Colab 自动化工作站模板。

通过结合 Google 官方 [`googlecolab/colab-mcp`](https://github.com/googlecolab/colab-mcp)、[`google-colab-cli`](https://pypi.org/project/google-colab-cli/)、[`deluxebear/colab-dfs-ext`](https://github.com/deluxebear/colab-dfs-ext) 与本仓库的 **`colab-ultra-workstation`** 技能，无需申请任何 API Key，让本地 AI Agent（Claude Code / Cursor / Codex / Antigravity / Windsurf）全自动遥控云端 **96GB Blackwell (G4)**、**L4**、**H100** 等算力与 **20TB/30TB Google Drive** 超大云盘。

---

## 📦 仓库结构

```text
colab-ultra-workstation/
├── SKILL.md                                           # Agent 智能硬件调度与防烧钱拦截技能定义
├── references/
│   ├── minimax_h3_industrial_workflow_whitepaper.md   # MiniMax-H3 工业级写实长视频与多角色一致性全链路架构白皮书 (v2.5)
│   └── minimax_h3_uncensored_comfyui_guide.md         # MiniMax-H3 无审核破限版 + Turbo 加速 + ComfyUI 高阶实战指南 (含实战踩坑与避坑总表)
├── scripts/
│   ├── colab_meter.py                                 # 开机打点 + 停机双轨 CCU 计费核验与幽灵实例强杀工具
│   ├── launch_g4_comfyui_fast.py                      # G4 96GB 双通道并发 (Drive API + HF CDN) NVMe 极速冷启动引擎
│   ├── launch_cloud_tts_studio.py                     # 独立云端双擎 TTS (Qwen3-TTS + IndexTTS-2) 常驻合成服务
│   ├── run_cloud_tts_remote.py                        # 本地遥控云端 TTS 极速合成与音频回传脚本
│   ├── setup_minimax_h3_comfyui.py                    # 0 CCU/h 免费 CPU 全自动部署 ComfyUI 与核心模型权重脚本
│   └── mcp_dfs_mount.py                               # colab-mcp 零浏览器挂载载荷一键生成器（含凭证擦除）
├── templates/
│   └── workstation_init.py                            # 云端常驻硬件探测、配额校验与防烧钱下载器模块
├── notebooks/
│   └── Google_AI_Ultra_Workstation.ipynb              # 完整实测 Colab 工作站记事本
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

# 3) 首次一次性 OAuth 授权（后续永久静默挂载；开启本地代理时需指定 no_proxy="localhost,127.0.0.1"）
https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 no_proxy="localhost,127.0.0.1" colab dfs consent

# 4) 验证 Token 已包含 https://www.googleapis.com/auth/drive 权限
https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 colab dfs token

# 5) 将默认 `colab drivemount` 也路由至零浏览器通道
echo 'export COLAB_DFS_DEFAULT=1' >> ~/.zshrc
```

> ⚠️ **首次授权必看（Google OAuth 复选框陷阱）**：在浏览器弹出的 Google 授权确认页中，**「查看、编辑、创建和删除您的所有 Google 云端硬盘文件 (Google Drive)」** 旁边的复选框默认是**未勾选**状态，请务必手动勾选后再点击「继续」，否则签发的 Token 缺少 `drive` 权限会导致挂载报 `403 ACCESS_TOKEN_SCOPE_INSUFFICIENT`。

常用零浏览器算力调度与云盘运维命令：
```bash
# 1. 按档位创建并连接实例（全自动选卡）
colab new -s work                                  # 启动免费 CPU (0 CCU/h) 下载模型/整理云盘
colab new -s work --gpu L4                         # 启动 L4 GPU (22.5GB, 1.71 CCU/h)
colab new -s work --gpu G4                         # 启动 G4 GPU (96GB Blackwell, 8.71 CCU/h)
colab new -s work --gpu T4|A100|H100               # 启动 T4 / A100 / H100
colab run train.py --gpu G4                        # “跑完即焚”：开一台 G4 跑完脚本立即自动销毁释放 VM

# 2. 零浏览器云盘挂载与自愈
colab dfs mount -s work                            # 秒级静默挂载 /content/drive（含后台 Token 自动续期）
colab dfs mount -s work --readonly                 # 只读挂载（保护 20TB/30TB 模型库防脚本误删）
colab dfs mount -s work --force --purge-state      # DriveFS/FUSE 假死原地抢救重挂（无需重启 Runtime）
colab dfs status -s work                           # 查看云盘挂载与 Token 守护进程状态

# 3. 断连销毁实例（立停 CCU 计费）与查账
colab stop -s work                                 # CLI 模式：立即销毁释放指定实例，停止扣除 CCU
colab usage                                        # 实时查询账户剩余 CCU 余额与当前消耗速率
colab url -s work                                  # 生成浏览器链接，随时点开查看当前 CLI 会话
```

> 💡 **网页模式 (`colab-mcp`) 自动断连释放**：在 Notebook 单元格中由 Agent 执行 `from google.colab import runtime; runtime.unassign()`（或调用 `workstation_init.release_runtime()`），无需人工点网页菜单即可立即销毁 VM 停止计费。

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
   - **`G4 GPU` (96GB RTX PRO 6000 Blackwell, 6.7~8.71 CCU/h)**：**主力王炸卡（全面替代 A100/H100）**。具备 1 PFLOPS BF16 / 2 PFLOPS FP8 / 4 PFLOPS FP4 算力，96GB 超大显存支持 **33B `MiniMax-H3` 音视频同生大模型 + `Qwen3-VL-32B-Heretic NVFP4` / `Wan2.1-14B` / `HunyuanVideo` 100% 显存常驻（零 CPU Offload）**，以及 **32B/70B LLM 的 FP8 超长上下文推理与 QLoRA 微调**。
2. **零浏览器静默挂载与不断联守护（`colab-dfs-ext`）**：
   - 彻底告别 `drive.mount` 人工弹窗确认，支持 `--readonly` 只读防误删挂载、云端后台 Token 自动续期（合盖睡觉跑长任务不掉盘）以及 FUSE 假死原地重挂。
3. **防烧钱硬拦截（Burn-Rate Guard）与双轨 CCU 计费强杀（`colab_meter.py`）**：
   - 当检测到正挂载高价显卡（G4 / A100 / H100，$\ge 5$ CCU/h）去首次下载大模型时，自动触发拦截提醒；
   - 每次 GPU 任务强制执行 `colab_meter.py start/stop`，自动完成开机余额打点、停机强杀全部残余会话（确认 `Active assignments: 0` 且 `0.00 CCU/h`）并输出双轨 CCU 账单。
4. **大炮打蚊子预警**：
   - 当检测到使用 G4/H100 运行 <9B 小模型或轻量任务时，自动提醒降级至 L4/T4 节省 CCU。

---

## 🎬 MiniMax-H3 (33B Native AV-DiT) 工业级影视工作流 (`v2.5`)

本仓库内置了在 **G4 (96GB Blackwell)** 上实机打磨验证的 **[《MiniMax-H3 工业级写实长视频生成与多角色一致性全链路架构白皮书 v2.5》](references/minimax_h3_industrial_workflow_whitepaper.md)** 与 **[《无审核破限版 ComfyUI 实战指南》](references/minimax_h3_uncensored_comfyui_guide.md)**，核心能力包括：

1. **96GB 显存全量锁死（零 CPU Offload）+ 双通道 NVMe 极速冷启动**：
   - 采用 `Singularity v1.3`（8步极速 DiT，`33GB`）+ `Qwen3-VL-32B-Heretic NVFP4`（无审核视觉语言编码器，`9.5GB`）+ `r34l1sm` 皮肤质感 LoRA + `BUNNY V2` 多人动作逻辑桥，总权重 ~43GB 全量常驻 G4 显存，预留 **49GB 纯动态激活值空间**，单次直出支持 **`3.75s ~ 15.08s`（`90 ~ 362 帧`）弹性分镜**。
   - [`scripts/launch_g4_comfyui_fast.py`](scripts/launch_g4_comfyui_fast.py) 绕开 FUSE 软链接随机读瓶颈，通过「Google Drive 内部 API + HuggingFace 内网 CDN」双通道以 **990 MB/s** 将权重预热至 `/content/local_models` 本地 NVMe SSD，ComfyUI 模型加载仅需 **0.5 秒**。
2. **$17k - 12$ / $17k + 5$ 黄金帧数定律与零跳帧、零起雾无缝接力**：
   - 遵循 5-Token $(1,4,4,4,4)=17$ 帧非均匀时间压缩网格，首段采用 $17k - 12$（如 `124` 帧 / `192` 帧），续接段采用 $17k + 5$（如 `141` 帧 / `192` 帧）配合 `MiniMaxH3MotionContext (context_frames = 22)` 物理裁除前 22 帧重叠区实现零相位硬切。
   - 显式配置 `MiniMaxH3JoinMatch (tuning = "manual", strength = 0.35, min_excess = 2.5, match_spread = False, match_detail = False)`，彻底根除多镜头接力后段画面发雾发灰问题。
3. **H3 原生自锚定声纹 (`Self-Anchor Voice`) + `T8mars` 跨段双人情绪爆发协议**：
   - 摒弃机械感重的外部 TTS 种子，直接提取第 1、2 段 H3 原生直出的高质量干声升频为 `32kHz Stereo` 作为 `<Audio 1>` 声纹锚点；
   - 在情绪爆发段通过 `[Shot 1]` 台词声学副语言前缀（`shouting furiously at the top of his lungs...`）+ `overall_soundscape` 瞬态重音锚点（`loud table slam at 00:00.400`）实现 **`+7.3 dB ~ +10.2 dB` 跨段情绪跃迁**且角色音色 100% 锁死。
4. **武指级 15 秒一镜到底动作暗切 (`V3 ProSeam`)、分阶段同源战损切换 (`V6 BrowStrike`) 与 $17k$ 关键帧控制**：
   - **技巧性暗切（拒绝缠斗硬拆）**：上半段结尾（`6.0s~8.0s`）设计正蹬踹退制造 `0.3s` 物理分离呼吸点（`join_match = 0.1135 ~ 0.1243`，肉眼 0 拼接痕迹），下半段借笼反弹 + 顺势推镜 + 单步下潜扣腕寸拳急停，实现 15.08 秒格斗长镜头连续多抽 100% 零穿模。
   - **写实战损与 1:1 受击对齐**：基于无伤定妆照 1:1 同源重绘战损定妆照（[`damaged.png`](references/minimax_h3_industrial_workflow_whitepaper.md)）分段切换，剔除夸张空中血雾，严格匹配高位拳峰直击左眉弓（`without touching his neck`），实现拳峰落点与眉骨裂口流血 100% 物理严丝合缝。
   - **多关键帧序列补间控制**：严禁将多张关键帧拼成单张宫格大图传入 `<Picture N>`；使用官方 `MiniMaxH3AddGuide` 按 **`frame_idx = 17k`（`0, 34, 51, 85, 136, 170`）** 串联钉入独立关键帧单图，实现零抽卡失败率的确定性动作补间。


