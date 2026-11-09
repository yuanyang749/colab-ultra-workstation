# 💃 工作流 04：9:16 竖屏 · 抖音高颜值美女/二次元 Coser 活人感热舞工作流

> **独立隔离目录说明：** 本目录专用于 **「9:16 竖屏（`768×1344` 原生 → `2160×3836` 4K）· 单图驱动 + `<Audio 1>` 真实 BGM 踩点 + 动作链库（Action-Chain Library）拼装 + 1:1 摄影机同步推拉」** 的抖音/TikTok 爆款美女热舞生成。

---

## 📂 目录结构一览

```text
04_douyin_beauty_dance/
├── README.md                              # 本说明文档
├── dance_config.json                      # ⭐ 舞蹈编排与音画同步配置文件（含一阶段 + 二阶段 Tessera 2K 多档位配置）
├── workflow_douyin_dance_2seg.json        # 一阶段：ComfyUI 9:16 竖屏一镜到底 API 节点图模板（768×1344）
├── workflow_tessera_2k_refine.json        # 二阶段：ComfyUI-Tessera 2K (1536×2688) 潜空间分块精修节点图模板
├── run_dance.py                           # 一阶段：云端 G4 极速执行器（支持 8s/15s/20s/30s+ 动态时长与自动鼓点对齐）
├── run_tessera_2k.py                      # 二阶段：云端 G4 / 本地 32G/24G 可插拔 Tessera 2K 精修执行器
├── upscale_local_mac_2k.py                # 二阶段（本地零算力）：Mac Apple Silicon 原生 Real-ESRGAN + CAS 2K 超分脚本
├── build_character_sheet.py               # 角色资产工具：「3/4 侧脸大头 + 盖小头正/侧/背全身三视图」自动化合成器
├── open_dance_bgm_kit/                    # 🎵 开源抖音/女团/爵士热舞 BGM 一键下载、无损拼接与鼓点分析套件
│   ├── README.md                          # 四大开源舞曲库直达链接与使用手册
│   ├── fetch_dance_bgm.py                 # 开源舞曲下载、短视频音频扒取与 BPM 鼓点测算脚本
│   ├── sources_catalog.json               # 结构化开源舞曲资源索引表（FineDance / AIST++ / PopDanceSet）
│   └── tracks/                            # 已预置的 6 首 35~36s 44.1kHz 无损热舞母带 WAV
├── references/                            # 核心动作链标准件库与官方参考工作流
│   ├── dance-motion-chain-h3.md           # 2,767 行完整动作链库（Rev.1.4，含 A宅舞/B抖舞/C古风/E运镜/K扩展包）
│   └── h3-motion-transfer-workflow.json   # 动作迁移与多段续接参考工作流
└── sample_inputs/                         # 示例输入锚点（首帧图 + 特写图 + 相位对齐分段 WAV）
    ├── ref_realgirl_v3_2k.png             # 9:16 竖屏首帧全身视觉权威锚点（<Picture 1>）
    ├── ref_realgirl_v3_closeup_9x16.png   # 9:16 竖屏半身特写锚点（<Picture 2>）
    ├── polite_greeting_master_30s.wav     # 0.000s~29.250s 30秒母带无损音轨（默认 4 段 30s 1:1 X 原片编舞）
    ├── polite_greeting_seg01_8s.wav       # Seg 01 音频切片（0.000s ~ 8.000s）
    ├── polite_greeting_seg02_8s.wav       # Seg 02 音频切片（7.083s ~ 15.083s，含 22 帧重叠区相位对齐）
    ├── polite_greeting_seg03_8s.wav       # Seg 03 音频切片（14.167s ~ 22.167s，含 22 帧重叠区相位对齐）
    ├── polite_greeting_seg04_8s.wav       # Seg 04 音频切片（21.250s ~ 29.250s，含 22 帧重叠区相位对齐）
    ├── master_bgm_15s.wav                 # 0.000s~15.083s 15秒母带无损音轨（可选 `--mode=15s`）
    ├── seg01_bgm_8s.wav                   # 15s 模式 Seg 01 音频切片
    └── seg02_bgm_8s.wav                   # 15s 模式 Seg 02 音频切片
```

---

## 🔥 核心五大铁律（源自 `dance-motion-chain-h3.md` Rev.1.4 & 30s 1:1 X 原片复刻）

1. **双图视觉权威 + 零外貌累赘提示词（Zero-Appearance Prompting）**：
   - `<Picture 1>`（全身首帧）与 `<Picture 2>`（腰上中近景）100% 统管人物五官、肤色、发型、服装与背景，提示词正文零外貌重定义，将 `Qwen3-VL-32B` 注意力 100% 集中于肢体动力链与运镜同步。
2. **30s 四段（`702 帧 @ 24fps = 29.25s`）1:1 X 原片逐秒动作链编排（默认配置）**：
   - **持续层（Sustained Layer）**：全程保持 `A1 内八膝弹跳` + `B1 胯部八字引擎`（每拍 `0.48s` 软膝回弹，杜绝站桩）；
   - **`Seg 01 (0.00s–8.00s)`**：`A12 背手待机` → `K44 挥手问候` → **`第1次 1:1 镜头推近至 <Picture 2> 中近景 + A11 叉腰伸指戳镜 + K8 掩口偷笑眨眼`** → `1:1 镜头拉回 <Picture 1> 全身` → `B10 顶胯斜扫 + B12 翻腕` → `A3 胸前交替滚拳`；
   - **`Seg 02 (7.08s–15.08s)`**：`K44 双手胸前十指交握摇摆顶胯` → `A5 右手额头敬礼落点下巴轻触` → `K45 左手叉腰 + 右手耳侧上下脉冲转 45° 右上斜举` → `K58 右手掌心朝上前伸献吻顶胯`；
   - **`Seg 03 (14.17s–22.17s)`**：`A10 右手举过头顶挥手 + 单脚轻跳左脚跟后踢` → **`第2次 1:1 镜头推近至 <Picture 2> 中近景 + A15 胸前双臂交叉比 X`** → `K59 双手食指点太阳穴/脸颊甜笑` → `双臂水平一字展臂 + 1:1 镜头同步拉回 <Picture 1> 全身` → `K51 双臂高举环抱头顶展肘摇胯`；
   - **`Seg 04 (21.25s–29.25s)`**：`K46 左手叉腰 + 右手掌心朝上向右侧平展` → `A15 双臂水平 T 字展臂 → 屈膝捧脸点肩 → 二次 T 字展臂蓄力` → **`K47/A9 360° 原地快速平转甩裙（frame 132 180° 背影长发裙摆水平飞扬 → frame 150 正面回落）`** → `右手叉腰 + 左臂笔直高举指天` → `A16 双手叉腰 + 右歪头定格收尾`。
3. **跨段音频 22 帧相位对齐 + 无损原曲回铺**：
   - 每段 `192 帧 (8.000s)`，`MiniMaxH3MotionContext` 锁定前 `22 帧 (0.9167s)` 重叠区，步进严格为 `170 帧 = 7.0833s`（`50` 个潜空间 Token $\equiv 0 \pmod 5$），最终拼接自动回铺无损 `polite_greeting_master_30s.wav`。
4. **Stage-2 二阶段 2K (`1536×2688`) 精修「强制交互确认门禁」与四档方案矩阵**：
   - **🚨 强制交互铁律**：当用户要求“精修”、“超分”或“2K 高清化”时，**Agent 严禁直接开机或默认启动全幅精修**！必须先向用户提问（优先调用 `ask_question` 工具或列出清晰选项），展示以下四档方案的耗时与 CCU 预估，**等用户明确选择后再执行对应 `--profile`**：
     1. **`--profile local_cas`（推荐日常首选 · `0.00 CCU` · 约 25 秒）**：本地 Mac Apple Silicon 调用 `upscale_local_mac_2k.py --mode cas`（高精度 `Lanczos + AMD FidelityFX CAS + Unsharp`），零云端消耗、100% 保留原片五官肤质（另有 `--profile local_esrgan` 本地 M 系列 GPU 神经网络超分，约 2.5 分钟，`0.00 CCU`）。
     2. **`--profile g4_turbo_face`（极速云端 AI 脸部/上半身精修 · 30s 约 `2.0 CCU` / ~12 分钟 · 省 89% 算力）**：G4 仅重绘上半身与脸部 `Tile 0 (1536×1088)` + `segment_frames=124` 单窗直出（共 6 轮采样，下半身 Lanczos 2K 放大）。
     3. **`--profile g4_fast_face`（标准云端 AI 脸部/上半身精修 · 30s 约 `5.8 CCU` / ~34 分钟 · 省 69% 算力）**：G4 仅重绘上半身与脸部 `Tile 0 (1536×1088)` + `segment_frames=56` 精细滑窗（共 18 轮采样，省去腿部/地板无效算力）。
     4. **`--profile g4_optimal`（全幅满血云端 Tessera 2K 母带级精修 · 30s 约 `18.75 CCU` / ~102 分钟）**：G4 上、中、下 3 个 `1536×1088` 分块 × `56 帧` 滑窗全量重采样（共 54 轮采样，全身含裙摆/白袜/地板均做 2K 扩散重绘）。
   - 脚本层门禁保护：直接无参运行 `python3 run_tessera_2k.py` 会主动拦截并打印四档方案对比表，防止误烧算力。
