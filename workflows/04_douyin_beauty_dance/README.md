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
    ├── master_bgm_15s.wav                 # 0.000s~15.083s 15秒母带无损音轨（用于合成终片 1:1 回铺）
    ├── seg01_bgm_8s.wav                   # Seg 01 音频切片（0.000s ~ 8.000s）
    └── seg02_bgm_8s.wav                   # Seg 02 音频切片（7.083s ~ 15.083s，含 22 帧重叠区相位对齐）
```

---

## 🔥 核心五大铁律（源自 `dance-motion-chain-h3.md` Rev.1.4）

1. **单图唯一视觉权威（Zero-Appearance Prompting）**：
   - `<Picture 1>` 100% 统管人物五官、发型、服装与背景，提示词正文零外貌累赘描写，将 `Qwen3-VL-32B` 注意力 100% 集中于肢体动力链与运镜同步。
2. **三层结构动作链（Three-Layer Action Chain）**：
   - **持续层（Sustained Layer）**：全程保持 `A1 内八膝弹跳` + `B1 胯部八字引擎`（每拍 `0.48s` 软膝回弹，杜绝站桩）；
   - **动链层（Action Chains）**：按拍拼接 `A12 背手待机` → `A11 叉腰戳镜` → `K8 掩口偷笑` → `B10 顶胯斜切` → `A3 胸前交替冲拳` → `K45 对角举臂后踢` → `A5 敬礼歪头` → `K47/A9 360° 原地平转甩裙` → `A4+A16 比心叉腰终格`；
   - **微表情层（Micro-expression Layer）**：反拍歪头、眨眼（wink）、跟唱歌词口型。
3. **舞蹈动作 × 摄影机运镜 1:1 实时同步（Rule #7）**：
   - 伸手戳镜头（`1.22s–2.04s`）时摄影机 1:1 同步从全身推近至腰上中近景；收手展臂（`2.64s–3.12s`）时摄影机 1:1 同步拉回全身！
4. **跨段音频 22 帧相位对齐 + 无损原曲回铺**：
   - 第 2 段 `MiniMaxH3MotionContext` 锁定前 `22 帧 (0.9167s)` 重叠区，因此 `seg02_bgm_8s.wav` 精确从 `8.000 - 22/24 = 7.0833s` 起切，确保跨段鼓点零错位；最终拼接时自动回铺 `44.1kHz` 无损 `master_bgm_15s.wav`。
5. **旋转预算与静止明写（Rule #4）**：
   - 15 秒内仅安排 1 次 360° 原地平转高光（位于 75% 进度处），片尾明确写出 `holds two full beats of stillness` 锁死甜笑收尾定格。
