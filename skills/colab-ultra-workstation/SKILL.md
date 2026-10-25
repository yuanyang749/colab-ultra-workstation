---
name: colab-ultra-workstation
description: "Google AI Ultra (5x 1,000 CCU + 20TB / 20x 2,000 CCU + 30TB Google Drive) 云端算力智能调度与 colab-mcp / colab-dfs-ext 工作站规范。当用户要求连接/使用 Google Colab、零浏览器静默挂载 Google Drive (`colab dfs mount`)、跑 AI 视频模型（MiniMax-H3/Wan2.1/Hunyuan/LTX/ComfyUI）、跑基础大语言模型 LLM（7B/14B/32B/70B 推理或微调）、下载模型权重到 Google Drive 或咨询 Colab 显卡选择（CPU/T4/L4/G4/A100/H100）时必须激活此技能。"
---

# Google AI Ultra 云端工作站与智能硬件调度规范 (兼容 5x / 20x 套餐)

本技能指导 Agent 通过 **`colab-mcp`（网页桥接通道）** 或 **`google-colab-cli` + `colab-dfs-ext`（纯终端无浏览器通道）** 高效操控用户的 Google Colab 云端工作站，并根据任务类型（AI 视频生成、不同参数规模的基础 LLM 推理/微调、大文件下载）自动分析、推荐与校验最优 GPU 档位，最大化利用 **Google AI Ultra 5x（1,000 CCU/月 + 20TB 云盘）** 或 **20x（2,000 CCU/月 + 30TB 云盘）** 订阅资源。

## 1. 视频与多模态硬件调度矩阵

| 任务类型 (`task_type`) | 推荐硬件档位 | 硬件规格 | 费率 (CCU/h) | 月度续航 (5x 1000 / 20x 2000 CCU) | 调度与参数调优铁律 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`download` / `preprocess`**<br>下载模型、解压数据集、挂载云盘 | **`CPU`** | Host CPU | **0 CCU/h** | ♾️ 无限 | **严禁开着高价 GPU (G4/A100/H100) 从 HuggingFace 首次下载几十 GB 模型**。若检测到当前 `ccu_per_hour >= 5.0` 且模型尚未存在于 20TB/30TB 云盘，Agent 必须先拦截并提示用户切回 CPU 下载。 |
| **`light_video`**<br>Wan2.1-1.3B、LTX-Video、Flux、代码调试 | **`L4 GPU`** | 22.5 GB GDDR6<br>Ada (`sm_89`) | **1.71 CCU/h** | • **5x**: ~584 小时<br>• **20x**: ~1,169 小时 | 支持原生 FP8；显存 < 24GB，建议开启 `enable_model_cpu_offload()`。 |
| **`heavy_video`（主力王炸）**<br>**MiniMax-H3 (33B + 32B 破限编码器)**、Wan2.1-14B、HunyuanVideo | **`G4 GPU`**<br>*(无需勾高 RAM)* | **96 GB GDDR7**<br>RTX PRO 6000 Blackwell (`sm_120`) | **6.7 ~ 8.71 CCU/h** | • **5x**: ~115 ~ 149 小时<br>• **20x**: ~230 ~ 298 小时 | **完全替代 A100 与 H100**！拥有 1 PFLOPS BF16 / 2 PFLOPS FP8 / 4 PFLOPS FP4 算力，且显存高达 96GB。<br>**必须关闭 `cpu_offload`**（33B DiT + 32B Qwen3-VL 100% 常驻 96G 显存），使用原生 `fp8_e4m3fn` / `NVFP4` + `SageAttention`。 |

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

4. **任务完成自动断连与销毁释放（Stop CCU Billing & 幽灵实例强杀）**：
   - **🚨 关机前必做：DriveFS 异步云端落盘核验（防大文件丢失铁律）**：
     - 往 `/content/drive/MyDrive/...` 移动或写入几十 GB 的大模型/视频文件时，DriveFS 会先写入容器本地缓存（`/root/.config/Google/DriveFS`）再后台异步上传至云端。**若 `shutil.move` 一返回就立即执行 `colab stop`，尚未传完的缓存文件会随容器销毁而永久丢失！**
     - 销毁实例前必须通过 Google Drive v3 API (`files.list`) 核验目标文件已全部出现在云端且 `size > 0`（参见 `scripts/setup_minimax_h3_comfyui.py` Step 5）。
   - **CLI 通道主动释放**：调用 `~/.local/bin/colab stop -s <session>` 销毁指定会话。
   - **`colab-mcp` 网页通道主动释放**：在单元格中执行 `from google.colab import runtime; runtime.unassign()`（或 `workstation_init.release_runtime()`）。
   - **🚨 强制双轨计费核验与标准化释放报告（Mandatory CCU Meter & Release Report）**：
     - **双轨核验原理（开机打点 + 停机双轨核验）**：Google Colab 后台账单为周期性批量扣款，若单次任务仅运行 1~2 分钟，停机瞬间 API 余额可能尚未刷出扣款。因此必须采取**双轨制**：
       1. **开机瞬间打点**：记录 `{session, gpu, rate, start_time, start_balance}`；
       2. **关机瞬间结算**：强杀全部实例并核验 `Active assignments: 0`，同时计算 `运行时长`、`理论消耗 = (时长/3600) × 费率` 与 `API 实际扣除 = 开机余额 - 关机余额`。
     - **一键自动化命令**：
       - 开机打点：`python3 scripts/colab_meter.py start -s <session> -g <G4|L4|T4>`
       - 停机释放结算：`python3 scripts/colab_meter.py stop -s <session>`
     - **约定返回格式（Agent 每次完成任务释放算力后必须严格按此模板汇报，禁止省略）**：
       ```text
       ✅ 算力释放与计费核验报告：
       • 实例释放状态: Active assignments: 0
       • 实时消耗速率: 0.00 CCU/h (已完全停止计费)
       • 本次运行机型: G4 (NVIDIA RTX PRO 6000 Blackwell, 8.71 CCU/h)
       • 本次运行时长: X 分 Y 秒
       • 本次实际消耗: Z.ZZ CCU (按时长精确折算)
       • 当前账户余额: XXX.XX CCU
       ```
     - 全量底层兜底清场与余额核验：
       ```bash
       https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 no_proxy="localhost,127.0.0.1" \
         ~/.local/share/uv/tools/google-colab-cli/bin/python scripts/colab_meter.py stop
       ```

5. **全自动跨档位「零浪费」流水线闭环（Zero-Burn Pipeline & 严禁 G4 空转）**：
   - **🚨 严禁开着 G4 备料或直读 FUSE 慢速盘（防浪费三大铁律）**：
     - **铁律 A（备料在本地 `0 CCU`，点火即连发）**：严禁先开着 G4 (`8.71 CCU/h`) 再去现场想剧本、生成首帧底图或调试工作流。所有耗时的人工选图、`1344×768` 哑光首帧图生成与提示词编排必须在 **G4 点火前（0 CCU）** 全部在本地备齐。
     - **铁律 B（v2.2 三路全并发 + 单文件插件归档包 + 42.3GB 按需预热本地 NVMe SSD，~90 秒破除 FUSE 限速墙与小文件风暴）**：实测 FUSE `mmap` 仅 ~25 MB/s、FUSE 单流顺序读仅 ~62.7 MB/s、FUSE `rsync` 遍历 3,000+ 个 `.py` 小文件需卡顿 60s+，且若误用云盘旧版脚本串行全量拉取 `66.67 GB` 会拖慢至 6 分钟。在 G4 上必须优先上传并执行 `/tmp/launch_g4_comfyui_fast.py` (`v2.2`，启动时自动覆盖更新云盘旧脚本)：① 默认 `--profile scheme_a` 仅精准预热当前管线必需的 **`42.33 GB`** 权重（跳过未使用的 `fl2va 19GB` 等冗余文件）；② 开启 **「三路全并发引擎（Drive v3 API `150 MB/s` + HF CDN `350~450 MB/s` || 环境构建）」**，仅需 **~75.4 秒（均速 463 MB/s）** 即可将核心权重落入本地 NVMe SSD (`/content/fast_models`)；③ 通过云盘单文件归档包 `custom_nodes_bundle.tar.gz` **1.5 秒** 直接解压 13 套插件至本地 NVMe（彻底跳过 FUSE 小文件遍历），再由 ComfyUI (`--disable-mmap --highvram`) 以 **`2.5 GB/s`（约 16 秒）** 读入并永久锁死在 96GB 显存中（第 2 条起 **0 秒** 读盘）。
     - **铁律 C（长耗时任务强制 `nohup` 脱机启动 + `alarm` 超时轮询）**：严禁直接用 `colab exec -f` 阻塞运行超过 15 秒的脚本（`wsclient.py` 偶发 `RuntimeError: Connection was lost.` 会导致本地 CLI 守护线程无限假死）。一律用 `subprocess.Popen('nohup python3 -u ... > /tmp/task.log 2>&1 &', shell=True)` 1 秒脱机拉起，再用 `perl -e 'alarm 20; exec @ARGV' colab exec` 轮询日志。
     - **铁律 D（`3秒~15秒` 全谱系弹性分镜时长矩阵 `F = 17k + 5` 与动作/舞蹈一镜到底长镜头选型准则）**：
       - **严禁将每个片段机械锁死为 5 秒**！凭借 G4 (96GB) + `NVFP4` 编码器释放的 49GB 激活值显存空间，所有分镜可在 `F(k) = 17k + 5` 合法帧数表内按剧情弹性设定（且不同时长分段可任意通过 `MotionContext(22帧)` 无缝混合续接，优先选 $k \equiv 2 \pmod 3$ 零音频误差黄金帧数）：
       - **G4 四档弹性 `length` 选型矩阵**：
         1. **① 快切/反应短镜档（`3.0~3.75秒`）**：`length = 73` (`3.04s`) 或 **`90` ★ (`3.75s`)**，用于快节奏反应特写、拔刀起势、悬疑闪回；
         2. **② 标准对白主力档（`5.0~6.6秒`）**：`length = 124` (`5.17s`)、**`141` ★ (`5.88s`，续接裁 22 帧后净 `4.96s`)** 或 `158` (`6.58s`)，用于单句/双句核心台词交锋，规避长序列 $\mathcal{O}(L^2)$ 算力开销；
         3. **③ 中长叙事/走位档（`8.0~10.1秒`）**：**`length = 192` ★ (`8.00s`)** 或 **`243` ★ (`10.13s`)**，用于边走边聊（Walk-and-Talk）跟拍、长独白、第一回合过招；
         4. **④ 动作/武打/舞蹈一镜到底长镜档（`12.2~15.1秒` 极限长镜或 `192f + 192f` V3 ProSeam 无痕暗切）**：**`length = 294` ★ (`12.25s`)**、**`345` ★ (`14.38s`)**、**`362` (`15.08s`)** 或 **`192f + 192f` (V3 ProSeam 8s+8s 黄金画质无痕接力)**。编写武戏/舞蹈长镜头时，严禁照搬文戏的 `static shot`、严禁在贴身缠抱半途硬拆拼接、严禁写 `four-punch blitz` 高频碎拳，必须严格执行 **「V3 ProSeam 武指技巧性暗切五法则」**：① 将 `facial-realism-closeup` 特写 LoRA 从 `0.50` 降至 **`0.15`**，`BUNNY_H3_ActionLogic_Bridge_V2` 的 `alpha` 从 `0.10` 提至 **`0.22`**，采用 `28mm` 膝上广角动态手持跟拍；② 从 `00:00.000` 零延迟直接开打，按 **每 `2.0 秒` 一拍（8 秒 4 拍）** 编排，取消文戏 `1.1s holding` 定格；③ **上半段结尾（`6.0~8.0s`）必须设计「正蹬踹退/击退呼吸点」**将对手踹退至笼边/墙边拉开 1.5~2 米干净间距，使 `MotionContext(22帧)` 交接处四肢 100% 物理分离（实测 `join_match=0.1243`，肉眼 0 拼接痕迹）；④ 下半段开头采用**手持镜头借势推近（`28mm` $\to$ `35mm`）+ 对手借铁笼反弹扑击 + 单步俯身下潜闪避 + 单手精准扣腕 + 寸拳急停**，配合 `140-BPM` 强打击乐节拍驱动，连续多抽 100% 零穿模。
     - **铁律 E（独立云端双引擎 TTS 极速工作流：`Qwen3-TTS-1.7B` + `IndexTTS-2`，零 ComfyUI 依赖）**：
       - **严禁为了制作/调试 TTS 配音而启动庞大的 ComfyUI 及拉取 42GB 视频大模型！**
       - 制作或微调配音时，一律调用独立云端双引擎语音工作流 [`scripts/launch_cloud_tts_studio.py`](scripts/launch_cloud_tts_studio.py)（本地一键遥控器 [`scripts/run_cloud_tts_remote.py`](scripts/run_cloud_tts_remote.py)）：
         1. **零 ComfyUI 依赖 + 双引擎显存常驻（`127.0.0.1:8199`）**：不克隆 ComfyUI、不装 15 个视频节点、不拉取 42GB 视频权重。将 **`Qwen3-TTS-12Hz-1.7B` (`VoiceDesign` + `Base`)** 与 **B站 `IndexTTS-2` (`CUDA FP16`)** 同时持久化缓存于 20TB 云盘 `/content/drive/MyDrive/AI_Models/TTS_Cache/`，开机通过 `rsync` 极速同步至本地 NVMe 并双双锁死显存常驻。
         2. **三大合成模式（`--mode hybrid | qwen | indextts`）**：
            - `hybrid`（默认黄金双剑合璧）：先用 `Qwen3-TTS-VoiceDesign` + **基频守卫（$F_0$ Pitch Guard）** 从零捏出 100% 非动漫的电影级成熟声线种子（36岁林薇强制锁定 `160–205 Hz` 女中音区，淘汰 `>215 Hz` 少女/动漫音色；42岁陈默锁定 `90–130 Hz` 深沉男中音），再将该种子喂给 **`IndexTTS-2`**，利用 `IndexTTS-2` 的 **8 维情感向量解耦（`emo_vector`）与真人呼吸气口控制** 合成四句对白，彻底解决刚才因使用 `IndexTTS-2` 官方示例二次元干声导致的动漫感问题！
            - `qwen`：全链路使用 `Qwen3-TTS-VoiceDesign` + `Base` 合成。
            - `indextts`：直接使用 `IndexTTS-2` 克隆 `/content/drive/MyDrive/AI_Models/TTS_VoiceBank/` 中的自定义真人干声。
         3. **一键同步至 ComfyUI 输入区与本地试听**：常驻服务每次重跑单句仅需 **~2.5 秒**，自动生成与正片台词 0 重叠的 `voice_*_seed.wav` 及避开 Seg 2 前 14 帧（`0.68s` 安全区）的 `seg01/02_guide_audio.wav`，同步写入 `/content/drive/MyDrive/AI_Workstation/ComfyUI/input/` 并拉回本地 `tts_takes/` 供用户即时试听。
      - **铁律 F（`MiniMax-H3 Ref2VA` 通用短剧零穿帮与 `H3 Join Match` 零起雾黄金准则）**：
        - **防起雾发灰**：`MiniMaxH3JoinMatch` 严禁使用默认的 `tuning="auto", strength=1.0, match_spread=True, match_detail=True`（会在人物动作或构图变化时压缩 31%+ 对比度导致画面如蒙灰雾），必须显式设为 **`tuning="manual", strength=0.35, min_excess=2.5, match_spread=False, match_detail=False`**。
        - **通用短剧防穿帮五大规范（对齐官方 `h3-prompt-writing/references/ref-en.txt` & `base-en.txt`）**：① 仅向 `MiniMaxH3ReferenceToVideo` 传入纯净角色肖像（`<Picture 1>`, `<Picture 2>`），严禁传入未绑定独立 `<Subject N>` 的合成参考图（否则 0.0s–0.5s 必出静态图闪帧跳切）；② `subject_definitions` 末行必须单独声明 `<Audio 1> is the voice-timbre reference for <Subject N> (Sx)...`，`summary` 以 `[reference generation + audio reference]` 开头，台词以 `<d>[Mandarin]...</d>` 包裹；③ **一镜到底 vs 多机位硬切双轨路由**：同机位连续镜头开启 `MotionContext (22帧)` 并在 `00:00.000–00:01.100` 设置闭口微动气闸舱；若剧情需要**正反打（OTS）、全景切单人特写或跨场景转场**，该切镜段必须**断开 `MotionContext`（`clip_index=0`）**从第 0 帧干净建立新机位，靠 `<Picture N>` + `<Audio 1>` + `180度轴线南侧坐标系` 保证 100% 连戏；④ **大幅度动作位移与机位/景别耦合定律（防切头/防出画）**：当角色执行**站起（`stands up`）、坐下（`sits down`）、走位（`walks`）或打斗**时，在中近景下必须同步搭配相机跟移补偿（如 `the camera tilts up and pulls out with small amplitude at slow speed, dynamically reframing to keep the subject's full head inside the frame with comfortable headroom`），或改用中全景/全景（`medium-full / wide shot`）；若刻意锁死固定中近景（`static MCU/Medium Shot`），则将动作约束在当前画幅平面内；⑤ 按段单说话人门控挂载 `32kHz Stereo` 原生自锚定声纹母本 + `T8mars` 情绪解耦协议，实现跨段 `+7.6 ~ +8.2 dB` 真实情绪爆发。
   - 当用户要求运行尚未缓存至云盘的大模型（如 MiniMax-H3 / Wan2.1-14B / 32B / 70B）时，Agent 可全自动串联：
     1. `colab new -s dl`（0 CCU/h 免费 CPU 开机）$\rightarrow$ `colab dfs mount -s dl` $\rightarrow$ 高速下载模型至 20TB/30TB 云盘并通过 Drive v3 API 核验云端 100% 落盘 $\rightarrow$ `colab stop -s dl`（销毁 CPU 实例）；
     2. 本地（`0 CCU`）备齐所有首帧底图与提示词队列 $\rightarrow$ `colab new -s gpu --gpu G4`（秒开 96GB Blackwell G4）$\rightarrow$ `colab dfs mount -s gpu` $\rightarrow$ `nohup` 启动 `scripts/launch_g4_comfyui_fast.py`（75.4 秒双通道 NVMe 极速落盘 + 16 秒进显存常驻）$\rightarrow$ 批量连发完成渲染 $\rightarrow$ 全量强杀销毁 G4 实例停止扣费 $\rightarrow$ `colab usage` 汇报 CCU 余额。

## 5. 专项高阶实战参考文档 (`references/`)

- **[`references/minimax_h3_industrial_workflow_whitepaper.md`](references/minimax_h3_industrial_workflow_whitepaper.md)**（同时位于工作区根目录 [`MiniMax_H3_Industrial_Workflow_Whitepaper.md`](../MiniMax_H3_Industrial_Workflow_Whitepaper.md)）：**MiniMax-H3 工业级写实长视频生成与多角色一致性全链路架构白皮书 (v2.5)**。涵盖方案 A（模块化巅峰画质管线）与方案 B（Director 时间线导演台）双轨交叉互补架构、22 帧（7-Token）零相位潜空间续航、`H3 Join Match` 零发雾手动模式调优、官方 `h3-prompt-writing` (`ref-en.txt`) 零穿帮分镜语法、原生自锚定声纹 (`Self-Anchor Voice`) + `T8mars` 双人跨段情绪爆发协议（实测 `+8.2 dB` 跃迁）、远景小脸 (<40px) 防崩切脸精修、多人不串台、2026 四大开源 TTS 集成，以及 96GB Blackwell (NVFP4) 自动化部署清单。
- **[`references/minimax_h3_uncensored_comfyui_guide.md`](references/minimax_h3_uncensored_comfyui_guide.md)**：当用户咨询或要求部署 **`MiniMax-H3`（海螺 3 开源 33B 音视频原生同出模型）**、**NSFW 无审核破限编码器 (`Qwen3-VL-32B-Ultra-Heretic`)**、**`MiniMax-H3-Turbo` (4/6/8步加速 LoRA)**、ComfyUI 积木式架构、人物一致性锁脸（`ref2va` / `Flux 首帧 + fl2va`）或 2K 视频超分放大时，请一并参考此指南。

