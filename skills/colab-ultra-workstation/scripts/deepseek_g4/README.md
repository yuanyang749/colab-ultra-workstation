# DeepSeek-V4.1-Flash (748B / 2-Bit) on Single Colab G4 (96GB GDDR7)

本目录包含在 Google Colab Pro+ / AI Ultra 的单卡 **NVIDIA RTX PRO 6000 Blackwell Server Edition (G4, 96GB GDDR7 + 188GB DDR5 + Gen5 NVMe)** 上，以 **零 Swap、47.4 ~ 54.6 tok/s、128K 上下文** 运行 **748B `DeepSeek-V4.1-Flash-1017`** 三大 2-bit 旗舰方案（含 **`ISTA-DASLab GSQ-RCO-Coder 72.12GB + 110.60GB 完整 IQ4_NL Engram`**、`Unsloth UD-IQ2_XXS` 与 `Abiray Uncensored GSQ-RCO IQ2_XXS`）的全套自动化脚本与实测产物。

---

## 📂 目录结构

```text
scripts/deepseek_g4/
├── fast_start_g4.sh                         # 一键启动 / 10 分钟防休眠保活 / 多变体热切换 / Caddy 双模鉴权 / 释放算力
├── step1_cpu_download_to_drive.py           # [0 CCU/h 免费 CPU] 预下载两套 2-bit 主干分片 + 128KB Engram 稀疏头至 Google Drive
├── step2_g4_build_engines.sh                # [首次 G4] 编译 sm_120a 原生 Delta Engine + llama.cpp 并打包 2 秒解压归档
├── step3_g4_hybrid_stage_and_benchmark.py   # [G4 核心调度] 64 并发 Drive-v3/HF 直灌 Gen5 NVMe + 三级内存锁定 + OpenAI/Caddy 网关
├── run_pelican_benchmark.py                 # Simon Willison「鹈鹕骑自行车」交互式 SVG+JS 基准测试脚本
└── examples/
    ├── pelican_ista_coder_engram.html       # [推荐·方案一] ISTA-DASLab GSQ-RCO-Coder (72.1GB 全显存) + 110.6GB 完整 Engram (全内存) 未修复直出原始版（686 行，0 拼写错误，54.63 tok/s）
    ├── pelican_ista_coder_engram_fixed.html # [方案一·微调版] 修复 3 处 SVG 几何坐标与弧线反转后的满血动画版
    ├── pelican_ud_iq2xxs.html               # Unsloth UD-IQ2_XXS 单次直出原始版（含 Engram 置零导致的 <ellips 与 76px 喉囊狂抖现场）
    ├── pelican_ud_iq2xxs_fixed.html         # Unsloth UD-IQ2_XXS 修复拼写与重复 transform-origin 后的满血复原版（883 行）
    └── pelican_abiray_iq2xxs.html           # Abiray Uncensored GSQ-RCO IQ2_XXS 单次直出无删减赛博日落风版本（639 行）
```

---

## 🚀 快速上手（四步流水线）

### 1. 免费 CPU 节点零成本落盘（`0.00 CCU/h`）
在 Colab 免费 CPU 实例挂载 Google Drive 后执行，将 `Unsloth UD-IQ2_XXS`（`165.3 GB`）与 `Abiray GSQ-RCO IQ2_XXS`（`138.6 GB`）主干分片及 `DSpark` 投机解码器（`4.22 GB`）下载到网盘，并为 `110.6 GB` 的 Engram 记忆表生成 `128 KB` 稀疏 GGUF 头：

```bash
python3 scripts/deepseek_g4/step1_cpu_download_to_drive.py --variant both --engram-mode sparse-header
```

### 2. G4 节点极速启动与 `127.0.0.1:8150/v1` 网关映射
切换到 G4 96GB 实例后，一键启动代码首选方案一（**`ista-coder-iq2xxs`：72.12GB 主干 100% 锁显存 + 110.60GB 完整 Engram 100% 锁 DDR5 内存，实测 `54.63 tok/s`**）或全专家方案：

```bash
# 方案一（代码首选，挂载完整 110.6GB Engram，0 拼写错误，54.63 tok/s）
./scripts/deepseek_g4/fast_start_g4.sh start --variant ista-coder-iq2xxs

# 方案一 + 开启原生 BF16 视觉多模态（自动挂载 972.2MB mmproj-BF16.gguf，支持发图/UI截图转代码）
./scripts/deepseek_g4/fast_start_g4.sh start --variant ista-coder-iq2xxs --vision

# 全专家方案（Unsloth V3.1 UD-IQ2_XXS 或 Abiray GSQ-RCO IQ2_XXS）
./scripts/deepseek_g4/fast_start_g4.sh start --variant unsloth-ud-iq2xxs
```

### 3. 一机多模热切换
在保持 `:8150/v1` 端口与客户端连接配置不变的前提下，热切换模型变体：

```bash
./scripts/deepseek_g4/fast_start_g4.sh switch ista-coder-iq2xxs
./scripts/deepseek_g4/fast_start_g4.sh switch abiray-gsq-iq2xxs
```

### 4. 10 分钟防休眠保活、状态巡检与下班关机（`0.00 CCU/h`）

> **💡 为什么需要 `keepalive`？**
> 当通过 Cloudflare Tunnel + API Key（Cherry Studio / Cursor）调用模型时，所有推理流量直接走隧道进入后台 `:18150 -> :8150` 端口，完全绕过了 Colab 的 Jupyter 内核 (`ipykernel`)。若连续 30~60 分钟无 `colab exec` 内核交互，Colab 后台看门狗会误判为空闲并强制回收实例。`fast_start_g4.sh` 在 `start` 时会自动在本地后台挂载 **10 分钟防休眠心跳进程 (`colab_keepalive_<session>`)**，并在 `stop` 时自动销毁。

```bash
# 查看本地保活进程状态、公网网关健康度与 GPU 实时监控
./scripts/deepseek_g4/fast_start_g4.sh status

# 为已运行的 Colab 会话单独补挂/重启本地 10 分钟防休眠心跳守护进程
./scripts/deepseek_g4/fast_start_g4.sh keepalive -s ds_g4

# 自动停止本地保活守护进程并销毁 G4 会话，回归 0.00 CCU/h
./scripts/deepseek_g4/fast_start_g4.sh stop
```

---

## 📖 完整技术白皮书与实测报告

- 第三弹（双模型网关与 Engram 置零实证）：[`references/deepseek_v41_748b_g4_guide.md`](../../references/deepseek_v41_748b_g4_guide.md)
- 第四弹（72.1GB 全显存 + 110.6GB 完整 Engram 四大版本终极横评 & Caddy 双模鉴权）：[`references/deepseek_v41_g4_ep4_full_engram_guide.md`](../../references/deepseek_v41_g4_ep4_full_engram_guide.md)
