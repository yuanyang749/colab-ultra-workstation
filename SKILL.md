---
name: colab-ultra-workstation
description: Google AI Ultra 5x (1,000 CCU + 20TB Google Drive) 云端算力智能调度与 colab-mcp 工作站规范。当用户要求连接/使用 Google Colab、跑 AI 视频模型（Wan2.1/Hunyuan/LTX/ComfyUI）、跑基础大语言模型 LLM（7B/14B/32B/70B 推理或微调）、下载模型权重到 Google Drive 或咨询 Colab 显卡选择（CPU/T4/L4/G4/A100/H100）时必须激活此技能。
---

# Google AI Ultra 5x 云端工作站与智能硬件调度规范

本技能指导 Agent 通过 `colab-mcp` 高效操控用户的 Google Colab 云端工作站，并根据任务类型（AI 视频生成、不同参数规模的基础 LLM 推理/微调、大文件下载）自动分析、推荐与校验最优 GPU 档位，最大化利用每月 1,000 CCU 与 20TB Google Drive。

## 1. 视频与多模态硬件调度矩阵

| 任务类型 (`task_type`) | 推荐硬件档位 | 硬件规格 | 费率 (CCU/h) | 1,000 CCU 续航 | 调度与参数调优铁律 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`download` / `preprocess`**<br>下载模型、解压数据集、挂载云盘 | **`CPU`** | Host CPU | **0 CCU/h** | ♾️ 无限 | **严禁开着高价 GPU (G4/A100/H100) 从 HuggingFace 首次下载几十 GB 模型**。若检测到当前 `ccu_per_hour >= 5.0` 且模型尚未存在于 20TB 云盘，Agent 必须先拦截并提示用户切回 CPU 下载。 |
| **`light_video`**<br>Wan2.1-1.3B、LTX-Video、Flux、代码调试 | **`L4 GPU`** | 22.5 GB GDDR6<br>Ada (`sm_89`) | **1.71 CCU/h** | ~584 小时 | 支持原生 FP8；显存 < 24GB，建议开启 `enable_model_cpu_offload()`。 |
| **`heavy_video`（主力王炸）**<br>Wan2.1-14B、HunyuanVideo、1080p 长视频渲染 | **`G4 GPU`**<br>*(无需勾高 RAM)* | **96 GB GDDR7**<br>RTX PRO 6000 Blackwell (`sm_120`) | **6.7 ~ 8.71 CCU/h** | ~115 ~ 149 小时 | **完全替代 A100 与 H100**！拥有 1 PFLOPS BF16 / 2 PFLOPS FP8 / 4 PFLOPS FP4 算力，且显存高达 96GB。<br>**必须关闭 `cpu_offload`**（100% 常驻 96G 显存），使用 `bfloat16` 或原生 `fp8_e4m3fn` + `sdpa`。 |

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

## 3. Agent 标准作业程序 (SOP)

1. **连接与探测**：调用 `colab-mcp` 的 `get_cells` 并运行 `Cell 1` 的 `inspect_hardware(task_type)`。
2. **防浪费与防 OOM 双重把关**：
   - 若用户用 G4/A100 跑 7B/8B 小模型，触发「大炮打蚊子预警」，建议降级到 **L4 (1.71 CCU/h)**；
   - 若用户用 40G A100 或 L4 跑 32B BF16 或 70B 模型，触发「OOM 显存不足预警」，建议切换到 **G4 (96GB Blackwell)**。
3. **20TB 云盘持久化**：模型权重统一存放在 `/content/drive/MyDrive/AI_Workstation/models`，跑完提醒用户「断开连接并删除运行时」。
