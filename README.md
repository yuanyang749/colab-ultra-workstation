# 🚀 Google AI Ultra 云端 AI 工作站 (MCP Agent 托管版)

一套专为 **Google AI Ultra（5x 套餐：1,000 CCU/月 + 20TB 云盘；20x 套餐：2,000 CCU/月 + 30TB 云盘）** 打造的云端算力智能调度技能（Agent Skill）与 Colab 自动化工作站模板。

通过结合 Google 官方 [`googlecolab/colab-mcp`](https://github.com/googlecolab/colab-mcp)、[`google-colab-cli`](https://pypi.org/project/google-colab-cli/)、[`deluxebear/colab-dfs-ext`](https://github.com/deluxebear/colab-dfs-ext) 与本仓库的 **`colab-ultra-workstation`** 技能，无需申请任何 API Key，让本地 AI Agent（Claude Code / Cursor / Codex / Antigravity / Windsurf）全自动遥控云端 **96GB Blackwell (G4)**、**L4**、**H100** 等算力与 **20TB/30TB Google Drive** 超大云盘。

---

## 📦 仓库结构

```text
colab-ultra-workstation/
├── SKILL.md                                           # Agent 智能硬件调度与防烧钱拦截技能定义
├── references/
│   ├── deepseek_v41_748b_g4_guide.md                  # 单卡 96GB G4 驯服 748B DeepSeek-V4.1 零 Swap 实战与两大 2-Bit 旗舰评测指南
│   ├── minimax_h3_zero_waste_sop.md                   # MiniMax-H3 通用制作 SOP、方案选择与执行前检查
│   ├── minimax_h3_industrial_workflow_whitepaper.md   # MiniMax-H3 工业级写实长视频与多角色一致性全链路架构白皮书 (v2.5)
│   └── minimax_h3_uncensored_comfyui_guide.md         # MiniMax-H3 无审核破限版 + Turbo 加速 + ComfyUI 高阶实战指南 (含实战踩坑与避坑总表)
├── scripts/
│   ├── deepseek_g4/                                   # 748B DeepSeek-V4.1-Flash 单卡 G4 96GB 零 Swap 自动化脚本与「鹈鹕骑自行车」实测集
│   │   ├── fast_start_g4.sh                           # 一键启动 / 22秒一机双模热切换 / 状态巡检 / 释放算力
│   │   ├── step1_cpu_download_to_drive.py             # 0 CCU/h 免费 CPU 预下载 2-bit 主干分片 + 128KB Engram 稀疏头至 Google Drive
│   │   ├── step2_g4_build_engines.sh                  # 编译 Blackwell sm_120a 原生 Delta Engine + llama.cpp 并打包 2 秒解压归档
│   │   ├── step3_g4_hybrid_stage_and_benchmark.py     # 64 并发 Drive-v3 API 直灌 Gen5 NVMe + 三级内存锁定 + OpenAI 网关
│   │   ├── run_pelican_benchmark.py                   # Simon Willison「鹈鹕骑自行车」交互式 SVG+JS 基准测试脚本
│   │   └── examples/                                  # Unsloth UD-IQ2_XXS (原始/修复后) 与 Abiray GSQ-RCO IQ2_XXS 实测 HTML
│   ├── colab_meter.py                                 # 开机打点 + 停机双轨 CCU 计费核验与幽灵实例强杀工具
│   ├── launch_g4_comfyui_fast.py                      # G4 96GB 双通道并发 (Drive API + HF CDN) NVMe 极速冷启动引擎
│   ├── resume_from_latent_checkpoint.py               # 分段原生潜变量 (SaveLatent/LoadLatent) 快照注入与零损耗断点续跑生成器
│   ├── auto_subtitle_video.py                         # 声学包络自动对齐电影级字幕 (SRT/内嵌) + 人物出场动效铭牌压制工具
│   ├── launch_cloud_tts_studio.py                     # 独立云端双擎 TTS (Qwen3-TTS + IndexTTS-2) 常驻合成服务
│   ├── run_cloud_tts_remote.py                        # 本地遥控云端 TTS 极速合成与音频回传脚本
│   ├── setup_minimax_h3_comfyui.py                    # 0 CCU/h 免费 CPU 全自动部署 ComfyUI 与核心模型权重脚本
│   └── mcp_dfs_mount.py                               # colab-mcp 零浏览器挂载载荷一键生成器（含凭证擦除）
├── assets/
│   └── deepseek_v41_g4/                               # 748B DeepSeek-V4.1 G4 架构图、跑分总表与 Engram 置零实证配图
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
colab new -s work --gpu G4                         # 启动 G4 GPU (96GB Blackwell, 10.05 CCU/h)
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
   - **`G4 GPU` (96GB RTX PRO 6000 Blackwell, 8.71~10.05 CCU/h)**：**主力王炸卡（全面替代 A100/H100）**。具备 1 PFLOPS BF16 / 2 PFLOPS FP8 / 4 PFLOPS FP4 算力，96GB 超大显存 + 188GB DDR5 主机内存支持 **33B `MiniMax-H3` 音视频同生大模型 + `Qwen3-VL-32B-Heretic NVFP4` / `Wan2.1-14B` / `HunyuanVideo` 100% 显存常驻（零 CPU Offload）**、**32B/70B LLM 的 FP8 超长上下文推理与 QLoRA 微调**，以及 **748B `DeepSeek-V4.1-Flash-1017`（552B MoE 骨干 `138.6~165.3 GB` 100% 锁入显存+内存，0 Swap 达 `47.4~51.5 tok/s`）**。
2. **零浏览器静默挂载与不断联守护（`colab-dfs-ext`）**：
   - 彻底告别 `drive.mount` 人工弹窗确认，支持 `--readonly` 只读防误删挂载、云端后台 Token 自动续期（合盖睡觉跑长任务不掉盘）以及 FUSE 假死原地重挂。
3. **防烧钱硬拦截（Burn-Rate Guard）与双轨 CCU 计费强杀（`colab_meter.py`）**：
   - 当检测到正挂载高价显卡（G4 / A100 / H100，$\ge 5$ CCU/h）去首次下载大模型时，自动触发拦截提醒；
   - 每次 GPU 任务强制执行 `colab_meter.py start/stop`，自动完成开机余额打点、停机强杀全部残余会话（确认 `Active assignments: 0` 且 `0.00 CCU/h`）并输出双轨 CCU 账单。
4. **大炮打蚊子预警**：
   - 当检测到使用 G4/H100 运行 <9B 小模型或轻量任务时，自动提醒降级至 L4/T4 节省 CCU。

---

## 🐋 单卡 96GB G4 驯服 748B DeepSeek-V4.1-Flash (`0 Swap` 47~52 tok/s + 一机双模网关)

本仓库内置了在单张 **Colab G4 (96GB GDDR7 + 188GB DDR5 + Gen5 NVMe)** 上零磁盘 Swap 跑通 **748B `DeepSeek-V4.1-Flash-1017`** 的全套自动化脚本（[`scripts/deepseek_g4/`](scripts/deepseek_g4/)）与深度实测指南 **[《把 Google AI Ultra 1000 CCU 榨出双倍价值（第三弹）：用单卡 96GB G4 跑通 748B DeepSeek-V4.1 折腾实录》](references/deepseek_v41_748b_g4_guide.md)**：

![96GB G4 × 748B DeepSeek-V4.1-Flash 旗舰量化对比 & 115K 极限压测总表](assets/deepseek_v41_g4/ep3_ds41_fig0_overview_table.png)

1. **552B MoE 骨干网 + 196B Engram 拆包与三级混合零 Swap 架构**：
   - 绕开 FUSE 云盘 `112 GiB` 缓存墙与 `0.4 tok/s` 随机读龟速，通过 **64 并发 Google Drive v3 API Range 流** 将 `Unsloth Dynamic V3.1 UD-IQ2_XXS`（5 个骨干分片 `165.30 GB`）或 `Abiray Uncensored GSQ-RCO IQ2_XXS`（4 个骨干分片 `138.59 GB`）直灌本地 Gen5 NVMe 并同步预热 `188GB DDR5` Page Cache；
   - 将每层必经的 Dense MLA 注意力层、共享专家、`128K` `NVFP4` KV Cache 以及最活跃的 **10,947 个专家（71%，`~87.2 GiB`）** 钉死在 `96GB GDDR7` 显存，剩余 **4,413 个专家（29%）** 全部锁入 `188GB DDR5` 锁页内存池（`--arena-gib 76`），196B Engram 采用 `128 KB` 稀疏 GGUF 头置零启动，实现 **全程 0 磁盘 Swap、稳定输出 `47.4 ~ 51.5 tok/s`、115K 超长代码库 34 秒极速预填充**。
2. **零显存「一机双模」网关与 64K 防截断机制**：
   - 单实例同时对外暴露 `deepseek-v4.1-flash-1017-standard` 与 `deepseek-v4.1-flash-1017-uncensored` 两个 OpenAI 兼容模型 ID，默认托底 `65,536` (`64K`) 最大生成长度并支持多轮 `<think>` 自动剥离与 KV 快照回滚。
3. **「鹈鹕骑自行车」三方同屏对决与 Engram 置零「提笔忘字」第一性原理解剖**：
   - 完整收录 [`Unsloth UD-IQ2_XXS` 原始版](scripts/deepseek_g4/examples/pelican_ud_iq2xxs.html)、[`Unsloth UD-IQ2_XXS` 拼写与原点修复版](scripts/deepseek_g4/examples/pelican_ud_iq2xxs_fixed.html) 与 [`Abiray GSQ-RCO IQ2_XXS` 赛博日落版](scripts/deepseek_g4/examples/pelican_abiray_iq2xxs.html) 交互式源码，实证 552B MoE 骨干网负责硬核数学/逆运动学推理、196B Engram 负责单词拼写与语法肌肉记忆（揭秘 `<ellips>` 丢云朵、双重 `transform-origin` 致喉囊 `76px` 狂抖及 `body.dataset["theheme".replace("theheme","theme")]` 现场自救名场面）。
4. **四条命令极速开跑（[`scripts/deepseek_g4/fast_start_g4.sh`](scripts/deepseek_g4/fast_start_g4.sh)）**：
   ```bash
   ./scripts/deepseek_g4/fast_start_g4.sh start --variant unsloth-ud-iq2xxs
   ./scripts/deepseek_g4/fast_start_g4.sh status
   ./scripts/deepseek_g4/fast_start_g4.sh switch --variant abiray-gsq-iq2xxs
   ./scripts/deepseek_g4/fast_start_g4.sh stop
   ```

---

## 🎬 MiniMax-H3 视频制作：共用流程与专项方案

先读 [制作 SOP](references/minimax_h3_zero_waste_sop.md)，明确模型栈、每段输入和衔接方式，再准备素材和分配 GPU。人物短剧、美女/时装展示、宠物与风格化角色共用身份、分镜、实际状态衔接和验收方法；各题材按自己的解剖、材质与表演要求检查。已安装 `minimax-h3-drama-producer` 时，由它负责具体导演与本地后期，本仓库负责算力和云端执行。

| 方案 | 输入与边界 |
| --- | --- |
| 当前默认：独立 Ref2VA 分段 | 每段独立身份参考 + 第 0 帧场景引导；官方 INT8 模型/编码器、24 步，无额外适配器，默认动态显存。后段不加入 MotionContext、JoinMatch 或续接裁帧。 |
| 明确选用的旧 V9 / 潜变量续接 | 按选定图使用原生 AV latent、相应重叠与音频处理；模型栈、插件和启动 flags 整套核对。`launch_g4_comfyui_fast.py --profile scheme_a` 属于旧模型栈，不能直接启动当前默认图。 |

[SOP](references/minimax_h3_zero_waste_sop.md) 把准备、Drive 凭据预检、150–180 秒启动心跳、后台执行、成果核验、目标实例释放和本地后期放在实际执行阶段。已完成准备再开 GPU；确认成果落盘后及时释放，服务端核验目标 endpoint，不把脚本打印成功或估算消耗当作停机/实扣证据。

已有专项材料保留供选用和复现：

- [历史白皮书](references/minimax_h3_industrial_workflow_whitepaper.md)：v2.5 模块化/时间线、原生声音与潜变量续接案例；旧效果描述不是通用保证。
- [社区模型安装指南](references/minimax_h3_uncensored_comfyui_guide.md)：仅适用于选择对应社区模型栈的项目。
- [真人双角色旧 V9 工作流](workflows/01_dialogue_2p_photorealistic/README.md)：保持原图与脚本，生产改动放在独立项目；不能当作新的独立分段默认图。
- [动作案例目录](workflows/02_action_mma_combat/README.md)、[舞蹈案例目录](workflows/03_dance_choreography/README.md) 与 [竖屏美女/舞蹈专项模板](workflows/04_douyin_beauty_dance/README.md)：动作、节奏和服装按当前作品设计，按具体模板核对输入、裁帧和输出画布。

每段保留实际执行图、来源和原生输出。按剧情/动作需要选时长，按真实对白做字幕；分别验收外观、表演、连续性、声音和文件完整性，关键帧和声纹参考不代表确定性复现。
