---
name: colab-ultra-workstation
description: Google AI Ultra (5x: 1,000 CCU + 20TB / 20x: 2,000 CCU + 30TB Google Drive) 云端算力智能调度与 colab-mcp / colab-dfs-ext 工作站规范。当用户要求连接/使用 Google Colab、零浏览器静默挂载 Google Drive (`colab dfs mount`)、跑 AI 视频模型（Wan2.1/Hunyuan/LTX/ComfyUI）、跑基础大语言模型 LLM（7B/14B/32B/70B 推理或微调）、下载模型权重到 Google Drive 或咨询 Colab 显卡选择（CPU/T4/L4/G4/A100/H100）时必须激活此技能。
---

# Google AI Ultra 云端工作站与智能硬件调度规范 (兼容 5x / 20x 套餐)

本技能指导 Agent 通过 **`colab-mcp`（网页桥接通道）** 或 **`google-colab-cli` + `colab-dfs-ext`（纯终端无浏览器通道）** 高效操控用户的 Google Colab 云端工作站，并根据任务类型（AI 视频生成、不同参数规模的基础 LLM 推理/微调、大文件下载）自动分析、推荐与校验最优 GPU 档位，最大化利用 **Google AI Ultra 5x（1,000 CCU/月 + 20TB 云盘）** 或 **20x（2,000 CCU/月 + 30TB 云盘）** 订阅资源。

## 1. 视频与多模态硬件调度矩阵

| 任务类型 (`task_type`) | 推荐硬件档位 | 硬件规格 | 费率 (CCU/h) | 月度续航 (5x 1000 / 20x 2000 CCU) | 调度与参数调优铁律 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`download` / `preprocess`**<br>下载模型、解压数据集、挂载云盘 | **`CPU`** | Host CPU | **0 CCU/h** | ♾️ 无限 | **严禁开着高价 GPU (G4/A100/H100) 从 HuggingFace 首次下载几十 GB 模型**。若检测到当前 `ccu_per_hour >= 5.0` 且模型尚未存在于 20TB/30TB 云盘，Agent 必须先拦截并提示用户切回 CPU 下载。 |
| **`light_video`**<br>Wan2.1-1.3B、LTX-Video、Flux、代码调试 | **`L4 GPU`** | 22.5 GB GDDR6<br>Ada (`sm_89`) | **1.71 CCU/h** | • **5x**: ~584 小时<br>• **20x**: ~1,169 小时 | 支持原生 FP8；显存 < 24GB，建议开启 `enable_model_cpu_offload()`。 |
| **`heavy_video`（主力王炸）**<br>Wan2.1-14B、HunyuanVideo、1080p 长视频渲染 | **`G4 GPU`**<br>*(无需勾高 RAM)* | **96 GB GDDR7**<br>RTX PRO 6000 Blackwell (`sm_120`) | **6.7 ~ 8.71 CCU/h** | • **5x**: ~115 ~ 149 小时<br>• **20x**: ~230 ~ 298 小时 | **完全替代 A100 与 H100**！拥有 1 PFLOPS BF16 / 2 PFLOPS FP8 / 4 PFLOPS FP4 算力，且显存高达 96GB。<br>**必须关闭 `cpu_offload`**（100% 常驻 96G 显存），使用 `bfloat16` 或原生 `fp8_e4m3fn` + `sdpa`。 |

## 2. 基础大语言模型 (LLM) 全尺寸选卡矩阵（推理 & 微调）

显存估算第一性原理（含 KV Cache 余量）：
- **BF16 / FP16**：$\approx \text{参数量(B)} \times 2.4\text{ GB}$
- **FP8 / INT8**：$\approx \text{参数量(B)} \times 1.2\text{ GB}$
- **INT4 / AWQ / GGUF(Q4) / NVFP4**：$\approx \text{参数量(B)} \times 0.7\text{ GB}$

| LLM 参数规模 | 4-bit 量化推理 (AWQ/GGUF/NVFP4) | 原生 BF16 / FP8 满血推理 (vLLM/SGLang) | LoRA / QLoRA 微调 (Unsloth/LLaMA-Factory) | Agent 选卡与避坑指令 |
| :--- | :--- | :--- | :--- | :--- |
| **`llm_small` (1.5B ~ 9B)**<br>*Qwen3-8B / Llama-3.1-8B / Gemma-2-9B* | **`T4 GPU`** (15GB)<br>**1.19 CCU/h** *(占 ~6GB)* | **`L4 GPU`** (22.5GB)<br>**1.71 CCU/h** *(BF16 占 ~17GB，FP8 占 ~9GB)* | **`L4 GPU`** (22.5GB)<br>**1.71 CCU/h** *(Unsloth 极速)* | **严禁开 G4/A100 跑 8B 模型**（大炮打蚊子）。L4 支持 Ada 原生 FP8，1.71 CCU/h 性价比无敌。 |
| **`llm_medium` (14B ~ 32B)**<br>*Qwen-14B/32B / DeepSeek-R1-Distill-32B / Gemma-27B* | **`L4 GPU`** (22.5GB)<br>**1.71 CCU/h** *(32B AWQ/Q4 占 ~19GB，精准卡进 L4 22.5GB)* | • **14B (FP8)** 👉 **`L4 GPU`**<br>• **32B (BF16/FP8)** 👉 **`G4 GPU`** (96GB, **8.71 CCU/h**) | • **14B QLoRA** 👉 **`L4 GPU`**<br>• **32B QLoRA/LoRA** 👉 **`G4 GPU`** (96GB) | **32B 是黄金分水岭**：<br>1. 跑 32B 4-bit 量化直接用 **L4** 极限卡位最省钱；<br>2. 跑 32B 满血 BF16 (~68GB) 40G A100 装不下，直接上 **G4 (96GB)**。 |
| **`llm_large` (70B ~ 72B & MoE)**<br>*Llama-3.3-70B / Qwen2.5-72B* | **`G4 GPU`** (96GB)<br>**8.71 CCU/h** *(支持 Blackwell 原生 **NVFP4**，4 PFLOPS 算力)* | **`G4 GPU`** (96GB)<br>**8.71 CCU/h** *(70B FP8 权重占 ~72GB + 剩余 24GB 跑超长上下文 KV Cache)* | **`G4 GPU`** (96GB)<br>**8.71 CCU/h** *(70B QLoRA 占 ~48GB，长序列微调不爆显存)* | **为什么 70B 选 G4 甚至比 80G A100/H100 更稳？**<br>70B FP8 权重占 72GB，80G 显卡只剩 8GB 给 KV Cache，上下文稍长必 OOM；**G4 拥有 96GB 显存 + 原生 FP8/FP4**，多出 16GB 全部变成上下文空间！ |
| **`llm_bandwidth` (极致单流吐字速度)** | **`H100 GPU`** (80GB HBM3, **18.05 CCU/h**) | 仅当总显存 < 75GB 且追求极致 Tokens/s（利用 3.35 TB/s HBM3 带宽）、不在乎双倍 CCU 消耗时选用。 | 不推荐（G4 96GB 性价比高一倍） | 日常 LLM 推理/微调优先用 **L4** 和 **G4**。 |

## 3. 本地环境自检与一键安装指南 (`colab-mcp` + `google-colab-cli` + `colab-dfs-ext`)

当 Agent 检测到本地缺少 `colab-mcp`、`colab` CLI 或 `colab dfs` 扩展时，按以下步骤自动为用户安装配置（自带系统 SOCKS 代理 `socksio` 兼容）：

### 3.1 安装 `colab-mcp`（网页桥接通道）
```bash
uv tool install --with socksio --force git+https://github.com/googlecolab/colab-mcp
```
在客户端 `mcp_config.json` 的 `mcpServers` 中注册：
```json
"colab-mcp": {
  "command": "colab-mcp",
  "args": []
}
```

### 3.2 安装 `google-colab-cli` 与 `colab-dfs-ext`（零浏览器静默挂载与无头终端通道）
```bash
# 1) 安装官方 google-colab-cli（含 socksio 代理兼容）
uv tool install --with socksio google-colab-cli

# 2) 一键注入 colab-dfs-ext 零侵入扩展（自动写入 .pth 启动钩子与 ~/.colab-dfs-ext）
curl -fsSL https://raw.githubusercontent.com/deluxebear/colab-dfs-ext/main/install.sh | sh

# 3) 环境与注入状态自检
colab dfs doctor

# 4) 首次一次性 OAuth 授权（生成 ~/.config/colab-cli/dfs_credentials.json）
# ⚠️ 注意：若本地需走代理访问 Google，请加上 https_proxy/http_proxy 并设置 no_proxy="localhost,127.0.0.1"
https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 no_proxy="localhost,127.0.0.1" colab dfs consent

# 5) 验证 Token 必须包含 https://www.googleapis.com/auth/drive 权限
https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 colab dfs token

# 6) 将默认 `colab drivemount` 也自动路由至零浏览器通道
grep -q 'COLAB_DFS_DEFAULT=1' ~/.zshrc || echo 'export COLAB_DFS_DEFAULT=1' >> ~/.zshrc
```

> **🚨 首次授权避坑铁律（Google OAuth 复选框陷阱）**：
> 在执行 `colab dfs consent` 弹出的 Google 授权页面中，**「查看、编辑、创建和删除您的所有 Google 云端硬盘文件 (Google Drive)」** 旁边的复选框默认是**未勾选**状态！
> - 若未勾选直接点击「继续」，签发的 Token 仅有 `openid/email` 权限而缺少 `drive` 权限，挂载时 DriveFS 会报错 `403 ACCESS_TOKEN_SCOPE_INSUFFICIENT` (`CANNOT_INIT_CELLOFS`)。
> - 若遇到该问题，可显式指定单权限重新授权（自动保留已获取的 `user_id` 与 `email`）：
>   ```bash
>   python3 ~/.colab-dfs-ext/colab_dfs_ext/dfs_creds.py consent --flavor dfs --scopes "https://www.googleapis.com/auth/drive" --user-id <GaiaId> --email <Email>
>   ```

## 4. Agent 标准作业程序 (SOP) 与全生命周期托管

1. **双通道按需连接与指定显卡开机**：
   - **通道 A（无头终端 `colab` CLI — 100% 全自动选卡开机/关机）**：
     - **启动免费 CPU (0 CCU/h)**：`~/.local/bin/colab new -s <session>`（默认不传 `--gpu` 即为 CPU，专用于下载模型、解压数据集与整理 20TB/30TB 云盘）
     - **按选卡矩阵精准启动 GPU**：`~/.local/bin/colab new -s <session> --gpu <T4|L4|G4|A100|H100>`（例如主力王炸卡：`colab new -s work --gpu G4`；轻量任务：`colab new -s work --gpu L4`）
     - **跑完即焚单次任务**：`~/.local/bin/colab run <script.py> --gpu <G4|L4|T4>`（自动创建指定显卡实例、运行脚本并在结束瞬间自动销毁释放 VM，绝不多扣一秒 CCU）
     - **查看账户 CCU 余额与实时消耗速率**：`~/.local/bin/colab usage`
     - **查看活跃会话与随时转网页查看**：`~/.local/bin/colab sessions` / `~/.local/bin/colab url -s <session>`
   - **通道 B（网页桥接 `colab-mcp`）**：
     - 当用户希望在浏览器中实时查看 Notebook 时，调用 `colab-mcp` 的 `open_colab_browser_connection` 连接，并运行 `inspect_hardware(task_type)`。

2. **防浪费与防 OOM 双重把关**：
   - 若用户用 G4/A100 跑 7B/8B 小模型，触发「大炮打蚊子预警」，建议降级到 **L4 (1.71 CCU/h)**；
   - 若用户用 40G A100 或 L4 跑 32B BF16 或 70B 模型，触发「OOM 显存不足预警」，建议切换到 **G4 (96GB Blackwell)**。

3. **20TB/30TB 云盘零浏览器静默挂载与持久化（`colab-dfs-ext`）**：
   - **优先零弹窗挂载（支持 CLI 与 `colab-mcp` 双模式一键脚本）**：
     - **CLI 模式**：调用 `~/.local/bin/colab dfs mount -s <session>`（或设置 `COLAB_DFS_DEFAULT=1` 后执行 `colab drivemount`）。
     - **`colab-mcp` 网页模式（零手写代码）**：Agent 直接在本地运行 `python3 ~/.gemini/config/skills/colab-ultra-workstation/scripts/mcp_dfs_mount.py` 生成挂载载荷，通过 `colab-mcp` 的 `add_code_cell` / `run_code_cell` 执行；**挂载成功后立即运行 `python3 ~/.gemini/config/skills/colab-ultra-workstation/scripts/mcp_dfs_mount.py --sanitize` 并用 `update_cell` 擦除单元格中的明文 Token**。
   - **云盘常驻模块免重写（`workstation_init.py`）**：
     - 云盘挂载后，直接从 `/content/drive/MyDrive/AI_Workstation/workstation_init.py` 导入标准函数（若不存在则从仓库自动同步至云盘），无需每次在记事本中重复粘贴百行初始化代码：
       ```python
       import sys; sys.path.insert(0, "/content/drive/MyDrive/AI_Workstation")
       from workstation_init import inspect_hardware, ensure_drive_mounted, smart_download_model, release_runtime
       HW_CONFIG = inspect_hardware("general")
       DIRS = ensure_drive_mounted("/content/drive")
       ```
   - **路径空格兼容与真实 20TB/30TB 容量校验**：
     - 直接启动 `/opt/google/drive/drive` 生成的挂载目录为带空格的 `/content/drive/My Drive`，自动建立软链接 `os.symlink("/content/drive/My Drive", "/content/drive/MyDrive")` 以兼容所有标准路径。
     - 通过容器内 `http://127.0.0.1:8012/computeMetadata/v1/instance/service-accounts/default/token` 获取 Token 调用 `https://www.googleapis.com/drive/v3/about?fields=storageQuota,user`，精准校验并展示 **20.00 TB (5x) / 30.00 TB (20x)** 真实云端容量。
   - **只读防误删模式**：当任务仅为读取 20TB/30TB 云盘中的模型权重进行推理、无需写入云盘时，可使用 `colab dfs mount -s <session> --readonly` 防止脚本误删模型库。
   - **FUSE 假死原地自愈**：若云盘出现 I/O 挂起或 `Transport endpoint is not connected` 报错，严禁直接重启 Runtime 导致环境丢失，应优先执行 `colab dfs mount -s <session> --force --purge-state` 原地清理 DriveFS 状态并重挂。

4. **任务完成自动断连与销毁释放（Stop CCU Billing）**：
   - **CLI 通道主动释放**：任务结束且产物已写入 `/content/drive/MyDrive/AI_Workstation/outputs` 后，Agent 必须执行 `~/.local/bin/colab stop -s <session>` 销毁实例停止计费，并可调用 `~/.local/bin/colab usage` 汇报剩余 CCU。
   - **`colab-mcp` 网页通道主动释放**：无需用户手动点击网页菜单，Agent 可直接在单元格中执行以下代码立即销毁并释放云端运行时（立停计费）：
     ```python
     from google.colab import runtime
     runtime.unassign()  # 或调用 workstation_init.release_runtime()
     ```

5. **全自动跨档位「零浪费」流水线闭环（Zero-Burn Pipeline）**：
   - 当用户要求运行尚未缓存至云盘的大模型（如 Wan2.1-14B / 32B / 70B）时，Agent 可全自动串联：
     1. `colab new -s dl`（0 CCU/h 免费 CPU 开机）$\rightarrow$ `colab dfs mount -s dl` $\rightarrow$ 高速下载模型至 20TB/30TB 云盘 $\rightarrow$ `colab stop -s dl`（销毁 CPU 实例）；
     2. `colab new -s gpu --gpu G4`（启动 96GB Blackwell G4）$\rightarrow$ `colab dfs mount -s gpu` $\rightarrow$ 零网络流量从云盘加载权重完成推理/微调，结果写回云盘 $\rightarrow$ `colab stop -s gpu`（跑完立即销毁 G4 实例停止扣费）$\rightarrow$ `colab usage` 汇报 CCU 余额。
