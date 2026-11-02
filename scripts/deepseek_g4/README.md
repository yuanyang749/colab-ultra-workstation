# DeepSeek-V4.1-Flash (748B / 2-Bit) on Single Colab G4 (96GB GDDR7)

本目录包含在 Google Colab Pro+ / AI Ultra 的单卡 **NVIDIA RTX PRO 6000 Blackwell Server Edition (G4, 96GB GDDR7 + 188GB DDR5 + Gen5 NVMe)** 上，以 **零 Swap、47.4 ~ 51.5 tok/s、128K 上下文** 运行 **748B `DeepSeek-V4.1-Flash-1017`** 两大 2-bit 旗舰量化版（`Unsloth UD-IQ2_XXS` 与 `Abiray Uncensored GSQ-RCO IQ2_XXS`）的全套自动化脚本与实测产物。

---

## 📂 目录结构

```text
scripts/deepseek_g4/
├── fast_start_g4.sh                         # 一键启动 / 22秒双模热切换 / 状态巡检 / 释放算力
├── step1_cpu_download_to_drive.py           # [0 CCU/h 免费 CPU] 预下载两套 2-bit 主干分片 + 128KB Engram 稀疏头至 Google Drive
├── step2_g4_build_engines.sh                # [首次 G4] 编译 sm_120a 原生 Delta Engine + llama.cpp 并打包 2 秒解压归档
├── step3_g4_hybrid_stage_and_benchmark.py   # [G4 核心调度] 64 并发 Drive-v3 API 直灌 Gen5 NVMe + 三级内存锁定 + OpenAI 网关
├── run_pelican_benchmark.py                 # Simon Willison「鹈鹕骑自行车」交互式 SVG+JS 基准测试脚本
└── examples/
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
切换到 G4 96GB 实例后，一键启动默认精调模型（`Unsloth UD-IQ2_XXS`）：

```bash
./scripts/deepseek_g4/fast_start_g4.sh start --variant unsloth-ud-iq2xxs
```

### 3. 一机双模 22 秒热切换
在保持 `:8150/v1` 端口与客户端连接配置不变的前提下，热切换至无审查极速版（`Abiray GSQ-RCO IQ2_XXS`）：

```bash
./scripts/deepseek_g4/fast_start_g4.sh switch abiray-gsq-iq2xxs
```

### 4. 状态巡检与下班关机（`0.00 CCU/h`）

```bash
# 查看本地隧道、模型健康度与 CCU 实时账单
./scripts/deepseek_g4/fast_start_g4.sh status

# 断开隧道并提示销毁 G4 会话，回归 0.00 CCU/h
./scripts/deepseek_g4/fast_start_g4.sh stop
```

---

## 📖 完整技术白皮书与实测报告

详见仓库文档：[`references/deepseek_v41_748b_g4_guide.md`](../../references/deepseek_v41_748b_g4_guide.md)。
