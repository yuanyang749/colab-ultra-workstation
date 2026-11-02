# MiniMax-H3 视频生成、双擎云端 TTS 与潜变量续跑标准作业程序 (Zero-Burn SOP)

> **适用场景**：当用户使用 Colab G4 (96GB Blackwell) 运行 `MiniMax-H3` 视频生成、多镜头长视频接力、武打/舞蹈一镜到底、独立云端 TTS 配音合成（`Qwen3-TTS` + `IndexTTS-2`）或潜变量断点续跑时，Agent 必须加载并严格遵循本规范。

## 目录 (Table of Contents)

1. [铁律 A：备料在本地 0 CCU，点火即连发](#1-铁律-a备料在本地-0-ccu点火即连发)
2. [铁律 B：三路全并发 NVMe 极速预热（破除 FUSE 限速墙与小文件风暴）](#2-铁律-b三路全并发-nvme-极速预热破除-fuse-限速墙与小文件风暴)
3. [铁律 C：长耗时任务强制 `nohup` 脱机启动 + `alarm` 超时轮询](#3-铁律-c长耗时任务强制-nohup-脱机启动--alarm-超时轮询)
4. [铁律 D：`3秒~15秒` 全谱系弹性分镜时长矩阵 `F = 17k + 5` 与 V3 ProSeam 武指暗切](#4-铁律-d3秒15秒-全谱系弹性分镜时长矩阵-f--17k--5-与-v3-proseam-武指暗切)
5. [铁律 E：独立云端双引擎 TTS 极速工作流（`Qwen3-TTS-1.7B` + `IndexTTS-2`，零 ComfyUI 依赖）](#5-铁律-e独立云端双引擎-tts-极速工作流qwen3-tts-17b--indextts-2零-comfyui-依赖)
6. [铁律 F：`MiniMax-H3 Ref2VA` 通用短剧零穿帮与 `H3 Join Match` 零起雾黄金准则](#6-铁律-fminimax-h3-ref2va-通用短剧零穿帮与-h3-join-match-零起雾黄金准则)
7. [铁律 G：分段原生潜变量快照 `SaveLatent / LoadLatent` 零损耗断点续跑 + 本地 0 CCU 自动字幕](#7-铁律-g分段原生潜变量快照-savelatent--loadlatent-零损耗断点续跑--本地-0-ccu-自动字幕)

---

## 1. 铁律 A：备料在本地 0 CCU，点火即连发

严禁先开着 G4 (`10.05 CCU/h`) 再去现场想剧本、生成首帧底图或调试工作流。所有耗时的人工选图、`1344×768` 哑光首帧图生成与提示词编排必须在 **G4 点火前（0 CCU）** 全部在本地备齐。

---

## 2. 铁律 B：三路全并发 NVMe 极速预热（破除 FUSE 限速墙与小文件风暴）

实测 FUSE `mmap` 仅 `~25 MB/s`、FUSE 单流顺序读仅 `~62.7 MB/s`、FUSE `rsync` 遍历 3,000+ 个 `.py` 小文件需卡顿 60s+，且若误用云盘旧版脚本串行全量拉取 `66.67 GB` 会拖慢至 6 分钟。在 G4 上必须优先上传并执行 [`scripts/launch_g4_comfyui_fast.py`](../scripts/launch_g4_comfyui_fast.py) (`v2.2`，启动时自动覆盖更新云盘旧脚本)：

1. **按需精准预热**：默认 `--profile scheme_a` 仅预热当前管线必需的 **`42.33 GB`** 权重（跳过未使用的 `fl2va 19GB` 等冗余文件）；
2. **三路全并发引擎**：开启「Drive v3 API `150 MB/s` + HF CDN `350~450 MB/s` || 环境构建」，仅需 **`~75.4 秒`（均速 `463 MB/s`）** 即可将核心权重落入本地 NVMe SSD (`/content/fast_models`)；
3. **单文件归档秒级解压**：通过云盘单文件归档包 `custom_nodes_bundle.tar.gz` **`1.5 秒`** 直接解压 13 套插件至本地 NVMe（彻底跳过 FUSE 小文件遍历），再由 ComfyUI (`--disable-mmap --highvram`) 以 **`2.5 GB/s`（约 16 秒）** 读入并永久锁死在 96GB 显存中（第 2 条起 **0 秒** 读盘）。

---

## 3. 铁律 C：长耗时任务强制 `nohup` 脱机启动 + `alarm` 超时轮询

严禁直接用 `colab exec -f` 阻塞运行超过 15 秒的脚本（`wsclient.py` 偶发 `RuntimeError: Connection was lost.` 会导致本地 CLI 守护线程无限假死）。一律用 `subprocess.Popen('nohup python3 -u ... > /tmp/task.log 2>&1 &', shell=True)` 1 秒脱机拉起，再用 `perl -e 'alarm 20; exec @ARGV' colab exec` 轮询日志。

---

## 4. 铁律 D：`3秒~15秒` 全谱系弹性分镜时长矩阵 `F = 17k + 5` 与 V3 ProSeam 武指暗切

- **严禁将每个片段机械锁死为 5 秒**！凭借 G4 (96GB) + `NVFP4` 编码器释放的 49GB 激活值显存空间，所有分镜可在 `F(k) = 17k + 5` 合法帧数表内按剧情弹性设定（且不同时长分段可任意通过 `MotionContext(22帧)` 无缝混合续接，优先选 $k \equiv 2 \pmod 3$ 零音频误差黄金帧数）：
- **G4 四档弹性 `length` 选型矩阵**：
  1. **① 快切/反应短镜档（`3.0~3.75秒`）**：`length = 73` (`3.04s`) 或 **`90` ★ (`3.75s`)**，用于快节奏反应特写、拔刀起势、悬疑闪回；
  2. **② 标准对白主力档（`5.0~6.6秒`）**：`length = 124` (`5.17s`)、**`141` ★ (`5.88s`，续接裁 22 帧后净 `4.96s`)** 或 `158` (`6.58s`)，用于单句/双句核心台词交锋，规避长序列 $\mathcal{O}(L^2)$ 算力开销；
  3. **③ 中长叙事/走位档（`8.0~10.1秒`）**：**`length = 192` ★ (`8.00s`)** 或 **`243` ★ (`10.13s`)**，用于边走边聊（Walk-and-Talk）跟拍、长独白、第一回合过招；
  4. **④ 动作/武打/舞蹈一镜到底长镜档（`12.2~15.1秒` 极限长镜或 `192f + 192f` V3 ProSeam 无痕暗切）**：**`length = 294` ★ (`12.25s`)**、**`345` ★ (`14.38s`)**、**`362` (`15.08s`)** 或 **`192f + 192f` (V3 ProSeam 8s+8s 黄金画质无痕接力)**。
- **V3 ProSeam 武指技巧性暗切五法则**（编写武戏/舞蹈长镜头时，严禁照搬文戏的 `static shot`、严禁在贴身缠抱半途硬拆拼接、严禁写 `four-punch blitz` 高频碎拳）：
  1. 将 `facial-realism-closeup` 特写 LoRA 从 `0.50` 降至 **`0.15`**，`BUNNY_H3_ActionLogic_Bridge_V2` 的 `alpha` 从 `0.10` 提至 **`0.22`**，采用 `28mm` 膝上广角动态手持跟拍；
  2. 从 `00:00.000` 零延迟直接开打，按 **每 `2.0 秒` 一拍（8 秒 4 拍）** 编排，取消文戏 `1.1s holding` 定格；
  3. **上半段结尾（`6.0~8.0s`）必须设计「正蹬踹退/击退呼吸点」**将对手踹退至笼边/墙边拉开 1.5~2 米干净间距，使 `MotionContext(22帧)` 交接处四肢 100% 物理分离（实测 `join_match=0.1243`，肉眼 0 拼接痕迹）；
  4. 下半段开头采用**手持镜头借势推近（`28mm` $\to$ `35mm`）+ 对手借铁笼反弹扑击 + 单步俯身下潜闪避 + 单手精准扣腕 + 寸拳急停**，配合 `140-BPM` 强打击乐节拍驱动，连续多抽 100% 零穿模。

---

## 5. 铁律 E：独立云端双引擎 TTS 极速工作流（`Qwen3-TTS-1.7B` + `IndexTTS-2`，零 ComfyUI 依赖）

- **严禁为了制作/调试 TTS 配音而启动庞大的 ComfyUI 及拉取 42GB 视频大模型！**
- 制作或微调配音时，一律调用独立云端双引擎语音工作流 [`scripts/launch_cloud_tts_studio.py`](../scripts/launch_cloud_tts_studio.py)（本地一键遥控器 [`scripts/run_cloud_tts_remote.py`](../scripts/run_cloud_tts_remote.py)）：
  1. **零 ComfyUI 依赖 + 双引擎显存常驻（`127.0.0.1:8199`）**：不克隆 ComfyUI、不装 15 个视频节点、不拉取 42GB 视频权重。将 **`Qwen3-TTS-12Hz-1.7B` (`VoiceDesign` + `Base`)** 与 **B站 `IndexTTS-2` (`CUDA FP16`)** 同时持久化缓存于 20TB 云盘 `/content/drive/MyDrive/AI_Models/TTS_Cache/`，开机通过 `rsync` 极速同步至本地 NVMe 并双双锁死显存常驻。
  2. **三大合成模式（`--mode hybrid | qwen | indextts`）**：
     - `hybrid`（默认黄金双剑合璧）：先用 `Qwen3-TTS-VoiceDesign` + **基频守卫（$F_0$ Pitch Guard）** 从零捏出 100% 非动漫的电影级成熟声线种子（36岁林薇强制锁定 `160–205 Hz` 女中音区，淘汰 `>215 Hz` 少女/动漫音色；42岁陈默锁定 `90–130 Hz` 深沉男中音），再将该种子喂给 **`IndexTTS-2`**，利用 `IndexTTS-2` 的 **8 维情感向量解耦（`emo_vector`）与真人呼吸气口控制** 合成对白；
     - `qwen`：全链路使用 `Qwen3-TTS-VoiceDesign` + `Base` 合成；
     - `indextts`：直接使用 `IndexTTS-2` 克隆 `/content/drive/MyDrive/AI_Models/TTS_VoiceBank/` 中的自定义真人干声。
  3. **一键同步至 ComfyUI 输入区与本地试听**：常驻服务每次重跑单句仅需 **~2.5 秒**，自动生成与正片台词 0 重叠的 `voice_*_seed.wav` 及避开 Seg 2 前 14 帧（`0.68s` 安全区）的 `seg01/02_guide_audio.wav`，同步写入 `/content/drive/MyDrive/AI_Workstation/ComfyUI/input/` 并拉回本地 `tts_takes/` 供用户即时试听。

---

## 6. 铁律 F：`MiniMax-H3 Ref2VA` 通用短剧零穿帮与 `H3 Join Match` 零起雾黄金准则

- **防起雾发灰**：`MiniMaxH3JoinMatch` 严禁使用默认的 `tuning="auto", strength=1.0, match_spread=True, match_detail=True`（会在人物动作或构图变化时压缩 31%+ 对比度导致画面如蒙灰雾），必须显式设为 **`tuning="manual", strength=0.35, min_excess=2.5, match_spread=False, match_detail=False`**。
- **通用短剧防穿帮五大规范（对齐官方 `h3-prompt-writing/references/ref-en.txt` & `base-en.txt`）**：
  1. 仅向 `MiniMaxH3ReferenceToVideo` 传入纯净角色肖像（`<Picture 1>`, `<Picture 2>`），严禁传入未绑定独立 `<Subject N>` 的合成参考图（否则 `0.0s–0.5s` 必出静态图闪帧跳切）；
  2. `subject_definitions` 末行必须单独声明 `<Audio 1> is the voice-timbre reference for <Subject N> (Sx)...`，`summary` 以 `[reference generation + audio reference]` 开头，台词以 `<d>[Mandarin]...</d>` 包裹；
  3. **一镜到底 vs 多机位硬切双轨路由**：同机位连续镜头开启 `MotionContext (22帧)` 并在 `00:00.000–00:01.100` 设置闭口微动气闸舱；若剧情需要**正反打（OTS）、全景切单人特写或跨场景转场**，该切镜段必须**断开 `MotionContext`（`clip_index=0`）**从第 0 帧干净建立新机位，靠 `<Picture N>` + `<Audio 1>` + `180度轴线南侧坐标系` 保证 100% 连戏；
  4. **大幅度动作位移与机位/景别耦合定律（防切头/防出画）**：当角色执行**站起（`stands up`）、坐下（`sits down`）、走位（`walks`）或打斗**时，在中近景下必须同步搭配相机跟移补偿（如 `the camera tilts up and pulls out with small amplitude at slow speed, dynamically reframing to keep the subject's full head inside the frame with comfortable headroom`），或改用中全景/全景（`medium-full / wide shot`）；若刻意锁死固定中近景（`static MCU/Medium Shot`），则将动作约束在当前画幅平面内；
  5. 按段单说话人门控挂载 `32kHz Stereo` 原生自锚定声纹母本 + `T8mars` 情绪解耦协议，实现跨段 `+7.6 ~ +8.2 dB` 真实情绪爆发。

---

## 7. 铁律 G：分段原生潜变量快照 `SaveLatent / LoadLatent` 零损耗断点续跑 + 本地 0 CCU 自动字幕

- **严禁直接用 `VHS_LoadVideo` 读取 `.mp4` 反向重编码去补拍中间单段一镜到底镜头**（`.mp4` 已丢失 `VAEDecode` 前的 141 帧原始 `AV_LATENT` 张量分布，`MiniMaxH3JoinMatch` 无法做统计对齐，极易引发机位/背景跳变）；**也严禁为了改第 3 段而从头重跑第 1、2 段白烧算力**！
- 一律使用 [`scripts/resume_from_latent_checkpoint.py`](../scripts/resume_from_latent_checkpoint.py)：
  1. **首次全链生成自动存快照（`--inject-checkpoints`）**：在每一段末尾并联 ComfyUI 原生 `SaveLatent` 节点，每段自动将未解码的原始 `AV_LATENT`（仅 `~12MB`）存至 `output/latents/`；
  2. **单段/局部重拍 0.1 秒无损续跑（`--resume-from N`）**：自动剪除前 $N-1$ 段采样节点，将上游输出节点原位替换为 `LoadLatent` 直接加载 `Seg02_raw_av.latent` 喂给 `MiniMaxH3MotionContext` 与 `MiniMaxH3JoinMatch`，实现 **100% 逐比特潜空间一镜到底 + 节省 50%~75% 算力时间**。
- **本地 0 CCU 声学对齐字幕与人物出场铭牌**：严禁在扩散提示词中要求模型直接生成内嵌中文字幕（必出笔画乱码，提示词须声明 `no subtitles`）。成片回传本地后，调用 [`scripts/auto_subtitle_video.py`](../scripts/auto_subtitle_video.py) 自动分析每段人声包络起止点与句中换气停顿（10ms 精度，过滤拍桌脉冲），一键生成 `.srt` 外挂字幕及带柔和出入场微动效（`0.38s` Alpha 淡入淡出 + `18px` 横向轻滑）的人物介绍铭牌与 2 倍超采样电影级内嵌硬字幕。
