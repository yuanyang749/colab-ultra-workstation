# 🎬 MiniMax-H3 (海螺 3) 无审核破限版 + Turbo 极速工作流实战指南

> **适用环境**：Google AI Ultra（5x: 1,000 CCU + 20TB 云盘 / 20x: 2,000 CCU + 30TB 云盘）+ `colab-ultra-workstation` 技能
> **最佳硬件组合**：**0 CCU/h 免费 CPU**（首次下载 50~65GB 全套权重与部署 ComfyUI） + **G4 GPU（96GB RTX PRO 6000 Blackwell, 8.71 CCU/h）**（33B DiT + 32B 破限编码器 100% 显存常驻，原生 FP8/NVFP4 加速）

---

## 1. 模块化“搭积木”三层架构（一次打底，永久复用）

跑出图与视频（尤其是 `MiniMax-H3` 这类 33B 全模态音视频原生同出模型），**只需将底层基础环境一次性持久化安装进 20TB/30TB Google Drive 云盘**；后续所有人物优化、画质超分、镜头控制与特定风格/动作，**100% 依靠挂载 LoRA 或轻量化插件节点解决，无需重装大模型**。

| 层级 | 模块组成 | 存放路径 (`AI_Workstation/ComfyUI/`) | 维护频率与计费策略 |
| :--- | :--- | :--- | :--- |
| **第 1 层：不动产底座**<br>*(约 50~65 GB)* | • **ComfyUI 本体**（原生支持 H3 节点）<br>• **`MiniMax-H3` 33B 主模型 + 音视频双 VAE**<br>• **`Qwen3-VL-32B-Ultra-Heretic` 无审核破限编码器** | • `/content/drive/MyDrive/AI_Workstation/ComfyUI`<br>• `models/diffusion_models/`<br>• `models/text_encoders/`<br>• `models/vae/` | **仅需装 1 次**（使用 **0 CCU/h 免费 CPU** 下载至云盘，永久保存，零算力消耗） |
| **第 2 层：涡轮加速与风格 LoRA**<br>*(几百 MB ~ 几 GB)* | • **`MiniMax-H3-Turbo` (4/6/8 步蒸馏加速 LoRA)**<br>• Civitai / HuggingFace 人物、画风、特定动作 NSFW LoRA | • `models/loras/` | **即插即用**，随时下载单个 `.safetensors` 丢入 `loras/` 目录即可在工作流中串联 |
| **第 3 层：画质与控制插件**<br>*(几 MB 代码仓库)* | • 多参考锁脸 (`ComfyUI-MiniMaxH3Mod`)<br>• 长视频无限续接 (`ComfyUI_MiniMax_H3_Extender`)<br>• 二阶段 2K 超分 (`SeedVR2` / `Real-ESRGAN`)<br>• 色调对齐与工具集 (`ComfyUI-KJNodes` / `VideoHelperSuite`) | • `custom_nodes/` | **即插即用**，`git clone` 至云盘 `custom_nodes/` 目录即可永久生效 |

---

## 2. 核心底座选型与 HuggingFace 仓库清单

### 2.1 扩散主干模型（`fl2va` vs `ref2va` vs `Hybrid` 融合版）

MiniMax-H3 是 **33B 参数的全模态音视频联合扩散模型（Joint Audio-Video DiT）**，在生成 4~15 秒高清视频的同时，**原生同步生成立体声音轨（人声台词、喘息、环境音效、拟音、配乐）**。

| 主模型分支 | 核心特点 | 适用场景 | 推荐仓库 |
| :--- | :--- | :--- | :--- |
| **`fl2va` (`fl2va_pruned`)**<br>*首尾帧/图生音视频版* | **原生画质锐度、皮肤质感、动作流畅度与立体声音质最高**；支持文生音视频 (T2V) 及单图/首尾双图生成音视频 (I2V)。 | **主力推荐**：配合 Flux 高清首帧做写真级人物/NSFW 视频。 | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3)<br>(`MiniMaxAI/MiniMax-H3`) |
| **`ref2va` (`ref2va_pruned`)**<br>*万物多参考锁定版* | 支持同时输入多张角色参考图（不同角度）、参考视频片段与参考音频，**免训练 LoRA 直接锁死人物长相与声线**；但纯画质锐度略逊于 `fl2va`。 | 多镜头连续剧、固定角色/固定声线复现。 | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| **`fl2va + ref2va Hybrid`**<br>*社区神级融合版* | 将 `fl2va` 与 `ref2va` 进行权重融合，**兼顾 `fl2va` 的高画质锐度与 `ref2va` 的多图人物锁定能力**。 | 既想要多图锁脸又追求极致画面锐度的高阶玩家。 | [`smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models`](https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models) |

---

### 2.2 NSFW 无审核破限核心：`Qwen3-VL-32B-Ultra-Heretic` 编码器替换法

- **为什么原版会有审查？**
  `MiniMax-H3` 使用了一颗高达 32B 参数的 **`Qwen3-VL-32B`** 多模态视觉语言模型作为提示词改写与视觉条件编码器（Text/Vision Conditioning Encoder），官方原版编码器内置了严格的安全对齐过滤。
- **如何彻底解除审查（Uncensored / Abliterated）？**
  无需修改 33B 视频扩散主干，只需将 `ComfyUI/models/text_encoders/` 下的官方编码器替换为社区去审查破限版 **`Qwen3-VL-32B-Ultra-Heretic`**，并在 ComfyUI 的 `CLIPLoader` 节点中选择该文件（`type` 设为 `minimax`），即可彻底解除文本提示词与输入垫图的安全过滤！

| 破限编码器版本 | 精度与体积 | 硬件适配与特点 | HuggingFace 仓库地址 |
| :--- | :--- | :--- | :--- |
| **Ultra-Heretic INT8-ConvRot**<br>*(满血无损破限版)* | INT8 (~33 GB) | 0~49 层全破限条件编码 + 生成尾层，提示词理解与画面细节还原度最高；在 **G4 (96GB)** 上可与 33B 主模型同时 100% 显存常驻。 | • [`OTMFLY/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot`](https://huggingface.co/OTMFLY/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot)<br>• [`ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot`](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot) |
| **Heretic NVFP4**<br>*(Blackwell 原生极速版)* | NVFP4 (~15.7 GB) | 体积缩小一半，且完美触发 Colab **G4 (RTX PRO 6000 Blackwell `sm_120`) 原生 FP4 Tensor Core (4 PFLOPS)** 硬件加速！ | • [`Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4`](https://huggingface.co/Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4) |

---

## 3. `MiniMax-H3-Turbo` 步数蒸馏加速版详解（提速 5 倍 / 省 75% CCU）

`MiniMax-H3-Turbo` 是针对 H3 主干蒸馏出的**加速 LoRA 插件**（存放在 `ComfyUI/models/loras/`），可与上面的 **Ultra-Heretic 无审核破限编码器 100% 同时叠加使用**。

### 3.1 原版 vs Turbo 步数与耗时对比（以 G4 96GB 显卡为例）

| 模式 | 采样步数 (Steps) | 调度器 (Scheduler) | 单条耗时 | 画质与适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **标准原版 (无 Turbo)** | 25 ~ 50 步 | `normal` / `simple` | ~3 ~ 5 分钟 | 耗时较长，消耗 CCU 较多 |
| **Turbo 4-Step** | **4 步** | `simple` | **~20 ~ 25 秒** | 极致快速，适合快速抽卡测试构图与分镜；大幅度高速动作偶有轻微残影 |
| **Turbo 6~8 Step（🏆 黄金推荐）** | **6 ~ 8 步** | `simple` | **~35 ~ 50 秒** | **速度与画质完美平衡**！彻底消除低步数塑料感与动作拖影，人物皮肤纹理、五官细节与立体声音质逼近原版 30 步，且省下 75%+ 算力费 |

### 3.2 推荐 Turbo LoRA 仓库与避坑铁律
- **推荐仓库**：
  - [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo)（如 `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors`、`minimax_h3_ref2v_turbo_4step...`）
  - `Wuli-art / Larryvrh` 社区优化版（如 `v4_step600_ema` 4/6/8 步系列）
- **🚨 避坑铁律**：
  1. **分支必须严格对齐**：跑 `fl2va` 主模型必须挂 `fl2v_turbo` LoRA；跑 `ref2va` 主模型必须挂 `ref2v_turbo` LoRA，混挂会导致画面模糊崩坏！
  2. **Turbo 模式下严禁开启 `Spectrum` 预测缓存节点**：`Spectrum` 依赖多步迭代建立缓存，在 4~8 步 Turbo 下开启反而会导致画面涂抹变糊；**Turbo 模式只需开启 `SageAttention`（或 `Sol Attention`）即可**。

---

## 4. 社区大佬 4 大高阶优化玩法（人物质感 & 2K 蓝光清晰度）

### 玩法 1：首帧决定上限（Flux.1-dev 无审核出首帧 $\rightarrow$ `MiniMax-H3 fl2va` 图生音视频）
- **痛点**：任何视频模型直接用文字抽卡（T2V），人物五官精致度、皮肤毛孔质感和特定姿势控制都远不如顶级图像模型。
- **大佬标准管线**：
  1. 先用 **Flux.1-dev（挂载写实人像 / NSFW LoRA）** 生成一张构图、光影、人物颜值与皮肤质感满分的 **高清首帧图**（如有需要再生成一张目标动作的 **尾帧图**）；
  2. 将首帧（或首尾帧）输入 `MiniMax-H3 (fl2va)` + `Ultra-Heretic 破限编码器` + `8-Step Turbo LoRA`，并在提示词中明確描述运镜方式、动作细节以及**人物台词/环境声音**；
  3. 生成出来的视频从第 1 帧起就是顶级写真画质，且自带同步立体声！

### 玩法 2：多视角零训练锁脸锁声（`ref2va` + `ComfyUI-MiniMaxH3Mod`）
- **痛点**：想让同一个角色在不同场景、不同机位下连续出镜，但不想费时费力训练人物 LoRA。
- **解决方案**：安装 `ComfyUI-MiniMaxH3Mod` 节点，切到 `Refmode`，直接塞入 **3~4 张该角色不同角度的清晰参考图 + 1 段参考音轨**。模型原生通过多模态条件把角色的五官、体型甚至说话/呼吸声线牢牢锁死。

### 玩法 3：长视频不断片无限续接 + 消除色偏（`Extender` + `Color Match`）
- **痛点**：单次生成只有 4~15 秒，直接用尾帧硬接下一段容易出现动作顿挫或越往后颜色越灰/越艳（色偏累积）。
- **解决方案**：
  1. 使用 `ComfyUI_MiniMax_H3_Extender` 节点，将上一段视频末尾的潜空间（Latent）与运动上下文直接传入下一段采样器（设置 `denoise ≈ 0.8` 锚定人物与动态连贯性）；
  2. 在解码输出后串联 `ComfyUI-KJNodes` 中的 **`Color Match`（色彩匹配）** 节点，强制将后续片段的色调分布对齐至首帧原图，实现几十秒长视频零色偏无缝衔接。

### 玩法 4：二阶段 2K 蓝光超分放大（先 768p 极速出片 $\rightarrow$ `SeedVR2` / `4x-UltraSharp` 放大）
- **痛点**：直接在 2K 分辨率下硬跑 33B 视频扩散模型既慢又极其消耗算力。
- **解决方案**：
  1. **第一阶段（保动作与声音）**：在标准分辨率（`1344 × 768` 768p）下配合 **Turbo LoRA** 快速生成动作流畅、声画同步的视频；
  2. **第二阶段（保 2K 毛孔级画质）**：在工作流末尾接上视频超分节点（如 **`SeedVR2`**、**`LTX Video Upscaler`** 或 **`UpscaleWithModel (4x-UltraSharp / Real-ESRGAN)`**），几十秒内直接将画面超分至 **2K（2432 × 1344）**，彻底洗掉涂抹感，发丝与皮肤纹理清晰可见。

### 玩法 5：🚨 彻底消除“AI 油腻感 / 塑料蜡像高光”的黄金四法则（实测必看！）
如果直接拿 `minimax_h3_fl2va` 裸跑低分辨率纯文生视频（T2V），人物额头、苹果肌、鼻梁和肩膀极易出现**像抹了凡士林/猪油一样的过曝塑料高光与十级磨皮感**。实测必须同时执行以下 **4 条去油铁律**，画质将发生质的飞跃：

1. **铁律 1：严禁裸跑 T2V！必须使用「单反级哑光高清首帧 (`first_frame`)」驱动 `fl2va` 的 I2V 模式**
   - `fl2va` 本质是首尾帧图生音视频模型。先用图像模型生成一张 **Sony A7R5 / 哈苏 85mm f/1.4、Kodak Portra 400 胶片感、真实细腻毛孔与柔和哑光肤质（Matte Skin, Zero Oily Shine）的 `1344 × 768` 首帧图**，通过 `LoadImage` 喂给 `MiniMaxH3ImageToVideo.first_frame`，从第 1 帧锁死真实毛孔与自然小痣纹理。
2. **铁律 2：必须跑满原生 `1344 × 768`（103 万像素 768p）画布**
   - 低分辨率（如 `864 × 480` 仅 41 万像素）会把面部高频毛孔细节全部压缩糊成油光色块；切回官方原生 `1344 × 768` 后，发丝、睫毛与皮肤纹理瞬间清晰。
3. **铁律 3：串联 `MiniMaxH3SigmaShift (12.0 / 3.0)` + 将 Turbo LoRA 权重降至 `0.85`（跑 `10` 步）**
   - `8-Step Turbo LoRA` 在权重 `1.0` 满格且无 SigmaShift 时会强行拉爆局部对比度与高光（蒸馏过曝）。将 `strength_model` 设为 **`0.85`**、采样步数设为 **`10` 步**，并在模型后串联 **`MiniMaxH3SigmaShift (shift_video=12.0, shift_audio=3.0)`**，高光过曝立刻消失，光影过渡如丝般顺滑。
4. **铁律 4：提示词严禁出现“发光/水光雷区词”**
   - ❌ **绝对禁用词**：`glowing skin`、`dewy skin`、`porcelain shiny skin`、`glistening skin`（视频模型会直接翻译成满脸抹油）。
   - ✅ **标准去油词**：`Realistic raw 85mm DSLR photography, Sony A7R5, Kodak Portra 400 film look, natural soft matte skin texture with fine pores, soft diffused warm ambient light, zero oily shine, no harsh specular highlights on face`。

#### 附：实测满血去油 768p I2V 声画同出 ComfyUI API 标准工作流 (`prompt` JSON)
```json
{
  "127": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "weight_dtype": "default"}},
  "134": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["127", 0], "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "strength_model": 0.85}},
  "135": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["134", 0], "shift_video": 12.0, "shift_audio": 3.0}},
  "128": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors", "type": "minimax", "device": "default"}},
  "119": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
  "120": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
  "110": {"class_type": "LoadImage", "inputs": {"image": "first_frame_1344x768.png"}},
  "131": {
    "class_type": "MiniMaxH3ImageToVideo",
    "inputs": {
      "clip": ["128", 0],
      "vae": ["119", 0],
      "first_frame": ["110", 0],
      "prompt": "Realistic raw 85mm DSLR photography, Sony A7R5, Kodak Portra 400 film look, natural soft matte skin texture with fine pores, soft diffused warm ambient light, zero oily shine, no harsh specular highlights on face...",
      "width": 1344,
      "height": 768,
      "length": 124
    }
  },
  "126": {"class_type": "BasicGuider", "inputs": {"model": ["135", 0], "conditioning": ["131", 0]}},
  "123": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
  "124": {"class_type": "BasicScheduler", "inputs": {"model": ["135", 0], "scheduler": "simple", "steps": 10, "denoise": 1.0}},
  "129": {"class_type": "RandomNoise", "inputs": {"noise_seed": 20269902}},
  "125": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["129", 0], "guider": ["126", 0], "sampler": ["123", 0], "sigmas": ["124", 0], "latent_image": ["131", 1]}},
  "122": {"class_type": "VAEDecode", "inputs": {"samples": ["125", 0], "vae": ["119", 0]}},
  "121": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["125", 0], "vae": ["120", 0]}},
  "140": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x-UltraSharp.pth"}},
  "141": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["140", 0], "image": ["122", 0]}},
  "142": {"class_type": "ImageScale", "inputs": {"image": ["141", 0], "upscale_method": "lanczos", "width": 2560, "height": 1440, "crop": "disabled"}},
  "130": {"class_type": "CreateVideo", "inputs": {"images": ["142", 0], "audio": ["121", 0], "fps": 24.0}},
  "92": {"class_type": "SaveVideo", "inputs": {"video": ["130", 0], "filename_prefix": "video/MiniMax_H3_Pro_2K_1440p", "format": "mp4", "codec": "h264"}}
}
```
> 💡 **768p 与 2K 灵活切换技巧**：日常抽卡测试时，将 `130 (CreateVideo)` 的 `images` 直接连到 `["122", 0]`（直出 `1344 × 768`，耗时 `~111s`）；挑中满意的种子后，将 `130` 的 `images` 改连到 `["142", 0]`（串联 `140 -> 141 -> 142`），ComfyUI 会 **100% 命中前序节点缓存（0 秒跳过 33B 采样）**，仅用 `~89s` 即可直接输出 **2K QHD (`2560 × 1440`, 10.4 Mbps)** 蓝光成片！

---

## 5. Google AI Ultra 零浪费全自动部署与运行 SOP

利用 `colab-ultra-workstation` 的跨档位流水线，按以下两步实现 **0 CCU 部署、满血 G4 渲染**：

### 第一步：0 CCU/h 免费 CPU 全自动落盘（仅首次执行一次）
```bash
# 1. 启动免费 CPU 实例并零浏览器挂载 20TB/30TB 云盘
colab new -s h3-setup
colab dfs mount -s h3-setup

# 2. 运行一键部署脚本，在云盘 /content/drive/MyDrive/AI_Workstation/ComfyUI 中安装 ComfyUI、插件与 12 个核心权重 (74.90 GB)，
#    并自动通过 Google Drive v3 API 轮询确认 12/12 文件 100% 异步落盘云端后再关机
colab run scripts/setup_minimax_h3_comfyui.py -s h3-setup

# 3. 确认云端全部落盘后销毁 CPU 实例
colab stop -s h3-setup
```

### 第二步：G4 (96GB Blackwell) 90 秒双通道 NVMe 极速冷启动与显存常驻渲染（随用随开，跑完即焚）
> 💡 **备料省钱铁律**：点火 G4 之前，先在本地（`0 CCU`）把今天要拍的所有 `1344 × 768` 哑光首帧大图和台词提示词全部备好；点火后通过 `nohup` 后台运行下方双通道 NVMe 极速脚本（实测仅需 **75.4 秒** 即可将 `41.42 GB` 核心权重以 **463 MB/s** 拉入本地 NVMe SSD，再花 **~16 秒** 以 `2.5 GB/s` 灌入 96GB 显存），随后批量连发渲染！

```bash
# 1. 零浏览器启动 96GB 显存的 G4 Blackwell 实例（8.71 CCU/h）并静默挂载云盘（~9.5s）
https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 no_proxy="localhost,127.0.0.1" colab new -s h3-work --gpu G4
https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 no_proxy="localhost,127.0.0.1" colab dfs mount -s h3-work

# 2. 后台脱机启动双通道 NVMe 极速脚本（75.4s 拉满 41.42GB 至本地 NVMe SSD + 启动 ComfyUI --highvram --disable-mmap + Cloudflare 隧道）
B64=$(base64 -i scripts/launch_g4_comfyui_fast.py | tr -d '\n')
echo "import base64, subprocess; from pathlib import Path; Path('/tmp/launch_g4_comfyui_fast.py').write_bytes(base64.b64decode('$B64')); subprocess.Popen('nohup python3 -u /tmp/launch_g4_comfyui_fast.py > /tmp/fast_boot.log 2>&1 &', shell=True); print('STARTED')" | \
  https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 no_proxy="localhost,127.0.0.1" perl -e 'alarm 20; exec @ARGV' colab exec -s h3-work

# 3. 渲染结束立即销毁所有实例停止计费，并核验 CCU 余额为 0.00/hr
https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 no_proxy="localhost,127.0.0.1" colab stop -s h3-work
https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 no_proxy="localhost,127.0.0.1" colab usage
```

---

## 6. 🚨 实战踩坑与避坑总表（8 大血泪教训汇总）

| 序号 | 坑点现象 | 根本原因 | 标准解决方案 |
| :---: | :--- | :--- | :--- |
| **1** | **首次 `colab dfs consent` 授权后挂载报 `403 ACCESS_TOKEN_SCOPE_INSUFFICIENT`** | Google OAuth 细粒度授权页面中，「查看、编辑、创建和删除您的所有 Google Drive 文件」复选框**默认未勾选**。 | 授权页面必须手动勾选 Drive 复选框；或直接用 `--scopes "https://www.googleapis.com/auth/drive"` 单权限强制重签。 |
| **2** | **本地执行 `colab` 或 `dfs_creds` 卡死 / OAuth `127.0.0.1` 回调失败** | 本地终端未走代理无法访问 `googleapis.com`，或开启代理但未排除 `localhost` 导致本地回调被代理劫持。 | 统一加上：`https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 no_proxy="localhost,127.0.0.1"`。 |
| **3** | **大模型 `shutil.move` 进 `/content/drive` 后立刻 `colab stop` 导致文件丢失** | DriveFS 采用本地缓存（`/root/.config/Google/DriveFS`）+ 后台异步上传机制，`mv` 返回仅代表写入本地缓存，立即销毁容器会中断后台上传。 | 关机前必须调用 Google Drive v3 API (`files.list`) 确认所有文件已在云端可见且 `size > 100000` 后再执行 `colab stop`。 |
| **4** | **ComfyUI 直接从 `/content/drive` 加载 `41.42 GB` 模型极慢（`mmap` 25 MB/s 耗 28 分钟；单流顺序读 62.7 MB/s 耗 11 分钟；纯 Drive API 限速 140 MB/s 耗 5 分钟）** | Google Drive FUSE 对 `mmap` 4KB 随机读限速 ~25 MB/s，对单文件句柄顺序读限速 ~62.7 MB/s，且 Google Drive v3 API 按单账号限速 ~140 MB/s。 | 直接使用 `scripts/launch_g4_comfyui_fast.py` 的 **「双通道并发引擎（Drive v3 API `150 MB/s` + HF CDN `450 MB/s`）」**：实测仅需 **75.4 秒（均速 463 MB/s）** 即可把全部 `41.42 GB` 权重落入本地 NVMe SSD (`/content/fast_models`)，ComfyUI 再从本地 NVMe 以 **`2.5 GB/s`（约 16 秒）** 读入 96GB 显存并永久常驻（`--highvram --disable-mmap`）。 |
| **5** | **生成的视频人物满脸“猪油高光 / 塑料蜡像感”** | 拿 `fl2va`（首尾帧模型）裸跑低分辨率 `480p` 纯文生视频 (T2V)，且提示词写了 `glowing/dewy skin`、Turbo LoRA 开满 `1.0` 导致对比度过曝。 | 严格执行 **4.5 节去油四法则**：单反级哑光高清首帧 (`first_frame`) + 原生 `1344×768` + `MiniMaxH3SigmaShift (12.0/3.0)` + `Turbo 0.85 (10步)`。 |
| **6** | **在浏览器手动点“断开连接”后，`colab usage` 仍显示 `Active assignments: 1` (`0.26/hr`)** | 浏览器端普通断开或切换运行时会在服务端残留未命名的 `HIGH_RAM` CPU 幽灵实例（`name="?"`），普通 `colab stop` 因无会话名无法关闭。 | 运行 `SKILL.md` 4.4 节中的 **`state.client.unassign(a.endpoint)` 全量清场指令**，遍历服务端 `assignments` 强杀所有残留实例，确保 `Usage rate: 0.00/hr`。 |
| **7** | **先开着 G4 (`8.71 CCU/h`) 再去现场想剧本、生首帧底图、调工作流，导致高价显卡大量时间空转浪费** | 把“0 算力的本地备料阶段（生成/挑选 `1344×768` 哑光首帧图、编写台词与 JSON）”和“G4 渲染阶段”混在一起，导致 G4 开机后大半时间在等人工决策和生图。 | 严格执行 **「备料在本地 (0 CCU)，点火即连发」**：必须在 G4 开机前先于本地备齐所有满意的 `1344×768` 哑光首帧底图与提示词队列；G4 点火后仅跑 `launch_g4_comfyui_fast.py`（90 秒极速冷启动）+ 批量连发渲染（110s/条），跑完立即关机。 |
| **8** | **直接用 `colab exec -f` 跑长耗时任务时偶发 `RuntimeError: Connection was lost.` 且本地进程无限假死挂起** | `jupyter_kernel_client` 的 WebSocket 在长耗时阻塞期间闪断后，后台守护线程未清理退出，导致本地 CLI 假死挂住。 | 所有超过 15 秒的远程任务一律通过 **`subprocess.Popen('nohup python3 -u ... > /tmp/task.log 2>&1 &', shell=True)`** 脱机启动（1 秒返回），配合 `perl -e 'alarm 20; exec @ARGV' colab exec` 非阻塞轮询日志。 |

