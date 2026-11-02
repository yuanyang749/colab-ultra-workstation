---
name: colab-ultra-workstation
description: "Google AI Ultra (5x 1,000 CCU + 20TB / 20x 2,000 CCU + 30TB Google Drive) 云端算力智能调度与 colab-mcp / colab-dfs-ext 工作站规范。当用户要求连接/使用 Google Colab、零浏览器静默挂载 Google Drive (`colab dfs mount`)、跑 AI 视频模型（MiniMax-H3/Wan2.1/Hunyuan/LTX/ComfyUI）、跑基础大语言模型 LLM（7B/14B/32B/70B 及 748B DeepSeek-V4.1-Flash 推理或微调）、下载模型权重到 Google Drive 或咨询 Colab 显卡选择（CPU/T4/L4/G4/A100/H100）时必须激活此技能。"
---

# Google AI Ultra 云端工作站与智能硬件调度规范 (兼容 5x / 20x 套餐)

本技能指导 Agent 通过 **`colab-mcp`（网页桥接通道）** 或 **`google-colab-cli` + `colab-dfs-ext`（纯终端无浏览器通道）** 高效操控用户的 Google Colab 云端工作站，并根据任务类型（AI 视频生成、基础/超大 MoE LLM 推理与微调、大文件下载）自动推荐与校验最优 GPU 档位，最大化利用 **Google AI Ultra 5x（1,000 CCU/月 + 20TB 云盘）** 或 **20x（2,000 CCU/月 + 30TB 云盘）** 订阅资源。

## 1. 视频与多模态硬件调度矩阵

| 任务类型 (`task_type`) | 推荐硬件档位 | 硬件规格 | 费率 (CCU/h) | 月度续航 (5x 1000 / 20x 2000 CCU) | 调度与参数调优铁律 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`download` / `preprocess`**<br>下载模型、解压数据集、挂载云盘 | **`CPU`** | Host CPU | **0 CCU/h** | ♾️ 无限 | **严禁开着高价 GPU (G4/A100/H100) 从 HuggingFace 首次下载几十 GB 模型**。若检测到当前 `ccu_per_hour >= 5.0` 且模型尚未存在于云盘，必须先拦截并切回 CPU 下载。 |
| **`light_video`**<br>Wan2.1-1.3B、LTX-Video、Flux、代码调试 | **`L4 GPU`** | 22.5 GB GDDR6<br>Ada (`sm_89`) | **1.71 CCU/h** | • **5x**: ~584 小时<br>• **20x**: ~1,169 小时 | 支持原生 FP8；显存 < 24GB，建议开启 `enable_model_cpu_offload()`。 |
| **`heavy_video`（主力王炸）**<br>**MiniMax-H3 (33B + 32B 破限编码器)**、Wan2.1-14B、HunyuanVideo | **`G4 GPU`** | **96 GB GDDR7**<br>RTX PRO 6000 Blackwell (`sm_120a`) | **8.71 ~ 10.05 CCU/h** | • **5x**: ~100 ~ 115 小时<br>• **20x**: ~199 ~ 230 小时 | **完全替代 A100 与 H100**！拥有 1 PFLOPS BF16 / 2 PFLOPS FP8 / 4 PFLOPS FP4 算力，关闭 `cpu_offload`，全量常驻 96G 显存。 |

## 2. 基础大语言模型 (LLM) 全尺寸选卡矩阵（推理 & 微调）

显存估算第一性原理（含 KV Cache 余量）：
- **BF16 / FP16**：$\approx \text{参数量(B)} \times 2.4\text{ GB}$
- **FP8 / INT8**：$\approx \text{参数量(B)} \times 1.2\text{ GB}$
- **INT4 / AWQ / GGUF(Q4) / NVFP4**：$\approx \text{参数量(B)} \times 0.7\text{ GB}$

| LLM 参数规模 | 4-bit / 2-bit 量化推理 | 原生 BF16 / FP8 满血推理 | LoRA / QLoRA 微调 | Agent 选卡与避坑指令 |
| :--- | :--- | :--- | :--- | :--- |
| **`llm_small` (1.5B ~ 9B)**<br>*Qwen3-8B / Llama-3.1-8B* | **`T4 GPU`** (15GB)<br>**1.19 CCU/h** | **`L4 GPU`** (22.5GB)<br>**1.71 CCU/h** | **`L4 GPU`** (22.5GB)<br>**1.71 CCU/h** | **严禁开 G4/A100 跑 8B 模型**（大炮打蚊子）。L4 支持 Ada 原生 FP8，性价比最高。 |
| **`llm_medium` (14B ~ 32B)**<br>*Qwen-32B / R1-Distill-32B* | **`L4 GPU`** (22.5GB)<br>**1.71 CCU/h** *(32B Q4 ~19GB)* | • **14B (FP8)** 👉 **`L4`**<br>• **32B (BF16/FP8)** 👉 **`G4`** (**10.05 CCU/h**) | • **14B QLoRA** 👉 **`L4`**<br>• **32B QLoRA** 👉 **`G4`** | 跑 32B 4-bit 量化直接用 **L4** 极限卡位最省钱；跑 32B 满血 BF16 (~68GB) 直接上 **G4 (96GB)**。 |
| **`llm_large` (70B ~ 72B)**<br>*Llama-3.3-70B / Qwen2.5-72B* | **`G4 GPU`** (96GB)<br>**10.05 CCU/h** *(原生 NVFP4)* | **`G4 GPU`** (96GB)<br>**10.05 CCU/h** *(权重 ~72GB + 24GB KV)* | **`G4 GPU`** (96GB)<br>**10.05 CCU/h** | 70B FP8 权重占 72GB，80G 显卡只剩 8GB 给 KV Cache；**G4 (96GB)** 多出 16GB 全部变为上下文空间。 |
| **`llm_ultra_moe` (748B 旗舰 MoE)**<br>*DeepSeek-V4.1-Flash-1017 (`IQ2_XXS`)* | **`G4 GPU`** (96GB GDDR7 + 188GB DDR5)<br>**10.05 CCU/h** *(0 Swap, **47.4~51.5 tok/s**)* | 需使用 2-bit 旗舰量化 (`UD-IQ2_XXS` / `GSQ-RCO IQ2_XXS`) + Delta `sm_120a` | N/A | 552B MoE 主干 (`138.6~165.3 GB`) 100% 锁死在「96GB GDDR7 + 188GB DDR5」，196B Engram 采用 128KB 稀疏头置零启动。详见 [`references/deepseek_v41_748b_g4_guide.md`](references/deepseek_v41_748b_g4_guide.md)。 |
| **`llm_bandwidth` (极致单流吐字)** | **`H100 GPU`** (80GB HBM3, **18.05 CCU/h**) | 仅当总显存 < 75GB 且追求极致 HBM3 带宽时选用 | 不推荐（G4 性价比高一倍） | 日常 LLM 推理/微调优先用 **L4** 和 **G4**。 |

## 3. 本地环境自检与一键安装指南 (`colab-mcp` + `google-colab-cli` + `colab-dfs-ext`)

```bash
# 1) 安装 colab-mcp（网页桥接通道，含 socksio 代理兼容）
uv tool install --with socksio --force git+https://github.com/googlecolab/colab-mcp

# 2) 安装官方 google-colab-cli 与 colab-dfs-ext（零浏览器静默挂载扩展）
uv tool install --with socksio google-colab-cli
curl -fsSL https://raw.githubusercontent.com/deluxebear/colab-dfs-ext/main/install.sh | sh
colab dfs doctor

# 3) 首次一次性 OAuth 授权（⚠️ 浏览器弹窗中务必手动勾选 Google Drive 复选框！）
https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 no_proxy="localhost,127.0.0.1" colab dfs consent
grep -q 'COLAB_DFS_DEFAULT=1' ~/.zshrc || echo 'export COLAB_DFS_DEFAULT=1' >> ~/.zshrc
```

> **🚨 首次授权避坑铁律**：若在 Google 授权页漏勾 Google Drive 复选框导致 `403 ACCESS_TOKEN_SCOPE_INSUFFICIENT`，执行以下命令重签 `drive` 权限：
> `python3 ~/.colab-dfs-ext/colab_dfs_ext/dfs_creds.py consent --flavor dfs --scopes "https://www.googleapis.com/auth/drive" --user-id <GaiaId> --email <Email>`

## 4. Agent 标准作业程序 (SOP) 与全生命周期托管

1. **按需选卡开机与连接**：
   - **免费 CPU (`0 CCU/h`)**：`~/.local/bin/colab new -s <session>`（专用于下载模型、解压数据集与整理云盘）
   - **指定 GPU 开机**：`~/.local/bin/colab new -s <session> --gpu <G4|L4|T4|A100|H100>`
   - **跑完即焚单次任务**：`~/.local/bin/colab run <script.py> --gpu <G4|L4|T4>`

2. **20TB/30TB 云盘零浏览器静默挂载与防只读/假死自愈**：
   - **CLI 模式**：`~/.local/bin/colab dfs mount -s <session>`（只读保护加 `--readonly`；FUSE 假死原地重挂加 `--force --purge-state`）。
   - **`colab-mcp` 网页模式**：运行 `python3 scripts/mcp_dfs_mount.py` 生成载荷执行，挂载成功后立即运行 `python3 scripts/mcp_dfs_mount.py --sanitize` 擦除单元格中的明文 Token。
   - **路径兼容**：自动建立软链接 `os.symlink("/content/drive/My Drive", "/content/drive/MyDrive")`。

3. **跨档位通用防浪费三原则（Zero-Burn Core Rules）**：
   - **本地 `0 CCU` 备料**：所有提示词编排、配置修改与首帧素材准备必须在启动高价 GPU 前在本地完成。
   - **严禁在 FUSE 云盘路径直接 `mmap` 大模型**：FUSE 随机读仅 `~25 MB/s` 且受 `112 GiB` 缓存上限限制；必须通过并发 Google Drive v3 API 或 HuggingFace CDN 将核心权重直灌本地 NVMe SSD（`/content/fast_*`）。
   - **长耗时任务强制 `nohup` 脱机启动**：超过 15 秒的远程任务一律用 `nohup ... > /tmp/task.log 2>&1 &` 后台拉起，再用 `alarm` 超时轮询日志，防止 WebSocket 断连卡死。

4. **关机落盘核验与强制双轨 CCU 结算报告**：
   - **关机前防丢失核验**：向 `/content/drive` 写入大文件后，销毁实例前必须通过 Google Drive v3 API (`files.list`) 确认云端 `size > 0`。
   - **双轨计费核验（`scripts/colab_meter.py`）**：
     - 开机打点：`python3 scripts/colab_meter.py start -s <session> -g <G4|L4|T4>`
     - 停机释放结算：`python3 scripts/colab_meter.py stop -s <session>`
   - **标准释放报告格式（每次释放算力后必须输出）**：
     ```text
     ✅ 算力释放与计费核验报告：
     • 实例释放状态: Active assignments: 0
     • 实时消耗速率: 0.00 CCU/h (已完全停止计费)
     • 本次运行机型: G4 (NVIDIA RTX PRO 6000 Blackwell, 10.05 CCU/h)
     • 本次运行时长: X 分 Y 秒
     • 本次实际消耗: Z.ZZ CCU (按时长精确折算)
     • 当前账户余额: XXX.XX CCU
     ```

## 5. 按需加载的专项领域参考文档 (Progressive Disclosure Routing)

为保持主技能轻量高效，请根据用户的具体任务按需读取以下专项文档（未涉及对应领域时无需加载）：

- **单卡 96GB G4 部署与评测 748B `DeepSeek-V4.1-Flash-1017`（2-Bit 双模型网关 & Engram 置零排障）**：
  - 当用户要求在 G4 上运行 `DeepSeek-V4.1-Flash`、切换 `Unsloth UD-IQ2_XXS` / `Abiray GSQ-RCO IQ2_XXS`、配置 `fast_start_g4.sh` 或排查 Engram 置零导致的拼写/SVG 瑕疵时，读取 **[`references/deepseek_v41_748b_g4_guide.md`](references/deepseek_v41_748b_g4_guide.md)** 及脚本说明 **[`scripts/deepseek_g4/README.md`](scripts/deepseek_g4/README.md)**。
- **MiniMax-H3 视频生成、双擎云端 TTS (`Qwen3-TTS` + `IndexTTS-2`) 与潜变量续跑操作铁律**：
  - 当用户使用 G4 生成 `MiniMax-H3` 短剧/动作长镜头、配置 `F = 17k + 5` 帧数、合成配音、使用 `SaveLatent/LoadLatent` 断点续跑或压制自动字幕时，**必须先读取 [`references/minimax_h3_zero_waste_sop.md`](references/minimax_h3_zero_waste_sop.md)**（包含完整的铁律 A ~ G）。
- **MiniMax-H3 工业级双轨架构与多角色一致性深度白皮书**：
  - 当需要深入了解方案 A/B 架构、22 帧潜空间接力、`H3 Join Match` 手动参数、`Self-Anchor Voice` 原生自锚定声纹或 `V3 ProSeam` 武指分镜语法时，读取 **[`references/minimax_h3_industrial_workflow_whitepaper.md`](references/minimax_h3_industrial_workflow_whitepaper.md)**。
- **MiniMax-H3 无审核破限版与 ComfyUI 节点搭建指南**：
  - 当用户首次搭建 `MiniMax-H3` + `Qwen3-VL-32B-Ultra-Heretic` 无审核编码器或配置 2K 视频超分时，读取 **[`references/minimax_h3_uncensored_comfyui_guide.md`](references/minimax_h3_uncensored_comfyui_guide.md)**。
