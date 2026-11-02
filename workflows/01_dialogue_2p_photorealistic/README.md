# 🎬 工作流 01：真人写实 · 双角色多段连续对手戏（2-Person Photorealistic Dialogue Drama）

> **隔离目录说明：** 本目录专用于 **「真人写实风格 · 双人同框/正反打多段连续对话文戏」**（锁死真实人物长相 + 锁死各自原生音色 + 22帧潜空间无缝续接 + 5MB 单段 3 分钟手术刀重修）。
> 后续新增的 **「02 高燃格斗动作专精工作流」** 与 **「03 长镜头古风/探戈舞蹈工作流」** 将分别存放在平级的 `02_action_mma_combat/` 与 `03_dance_choreography/` 目录中，参数、脚本与素材完全隔离、互不污染。

---

## 📂 目录结构一览

```text
01_dialogue_2p_photorealistic/
├── README.md                        # 本说明文档（新手 3 步快速上手）
├── drama_config.json                # ⭐ 新手唯一需要修改的配置文件（40 行搞定换人/换声/改台词/尾帧锁）
├── workflow_v9_final.json           # ComfyUI 底层 4 段式 API 节点图模板（无需手动改）
├── run_drama.py                     # 云端 G4 执行器（自动同步素材、支持 --only-seg N 单段热插拔）
├── auto_subtitle.py                 # 本地 0 CCU 声学包络自动对齐字幕 + 首段呼吸动效人物铭牌压制器
└── sample_inputs/                   # 示例输入锚点（1344×768 定妆照 + 5秒原生人声种子 WAV + 尾帧锁图）
    ├── ref_linwei_codex_real_1344x768.png
    ├── ref_linwei_8s_profile_1344x768.png
    ├── ref_chenmo_codex_real_1344x768.png
    ├── seg3_tail_lock_1344x768.png
    ├── native_linwei_seed_stereo.wav
    └── native_chenmo_seed_stereo.wav
```

---

## 🚀 新手 3 步快速上手（换自己的角色与剧本）

你**完全不需要**打开上千行的 `workflow_v9_final.json` 去找节点 ID，只需修改 **`drama_config.json`** 即可：

### 第一步：准备你的 4 个基础素材（放入 `sample_inputs/` 或云盘 `ComfyUI/input/`）
1. **左席角色（`role1`）定妆照**：`1344 × 768` PNG（建议真实电影感剧照，不带塑料美颜）；
2. **右席角色（`role2`）定妆照**：`1344 × 768` PNG；
3. **左席角色 5 秒原生人声种子 WAV**：`native_role1.wav`（仅用于锚定物理音色，台词内容不限）；
4. **右席角色 5 秒原生人声种子 WAV**：`native_role2.wav`。

### 第二步：修改 `drama_config.json`（30 秒改完）
打开 [`drama_config.json`](./drama_config.json)：
- 在 `characters.role1` / `role2` 中填入你的图片文件名、音频文件名，以及首段左下/右上角毛玻璃铭牌要显示的姓名与头衔（如果不想显示铭牌，把 `"enable_intro_cards": false` 即可）；
- 在 `segments` 数组中直接修改每段的说话人（`"speaker": "role1"` 或 `"role2"`）和中文台词（`"line": "..."`）；
- **首次跑新剧时**，把每段的 `"tail_lock_image"` 保持为 `null`（脚本会自动旁路尾帧锁节点，绝不会因为找不到截图报错）。

### 第三步：一键跑片 & 单段重修 & 自动加字幕

**1. 首次跑满 4 段（20 秒）：**
```bash
python3 run_drama.py --config drama_config.json --workflow workflow_v9_final.json
```

**2. 如果第 3 段表情/台词不满意，或最后两秒轻微转头，单独重修第 3 段（仅需 3 分钟，立省 74% 时间与 CCU）：**
```bash
# 普通换 Seed 或改台词重修第 3 段：
python3 run_drama.py --config drama_config.json --workflow workflow_v9_final.json --only-seg 3

# 如果想用尾帧夹具（TailLock）100% 焊死第 3 段最后两秒的脸部朝向：
python3 run_drama.py --config drama_config.json --workflow workflow_v9_final.json \
  --only-seg 3 --tail-lock-seg 3 --tail-lock-img my_seg3_anchor_1344x768.png
```

**3. 本地 Mac 3 秒自动对齐声学字幕 + 人物出入场动效铭牌（0 CCU）：**
```bash
python3 auto_subtitle.py --config drama_config.json --root /path/to/videos
```
*(如果不需要首段人物介绍铭牌，加上 `--no-cards` 即可；也可以直接用 `--left-card "姓名|年龄|头衔1|头衔2"` 临时覆盖)*

---

## 🎨 进阶：切换画风或调整动作幅度

- **切换到动漫 / 3D / 非写实风格**：
  在 `drama_config.json` 中将 `"style_mode"` 改为 `"stylized"`（或命令行传 `--style-mode stylized`），脚本会自动将真人皮肤毛孔去油 LoRA 权重归零。
- **增加肢体动作幅度**：
  文戏默认 `"bunny_alpha": 0.10` 以保护细腻面部微表情；若剧本包含较大肢体互动，可在配置中调高至 `0.20 ~ 0.25`。

---

## 🧭 导演开拍前铁律：物理空间连续性与三维视线轴线校验（必检清单）

> **核心经验沉淀（避免因空间逻辑断层或视线朝向错位导致返工）：**
> 视频大模型会严格遵从你传入的**关键帧构图（`frame_locks`）**与**参考人像朝向（`ref_roles`）**。绝大多数镜头违和感并非模型故障，而是分镜提示词与关键帧缺失了**现实三维空间的物理连续性**：

1. **禁止“瞬移式”新角色入场（Entry / Motion Continuity）**：
   - 当封闭空间（如会议室）有新角色（第三人）介入时，绝不能上一镜还是两人独处，下一镜新角色直接坐在桌前喝茶。
   - **标准物理过渡三连镜**：
     1. **入场镜（Door / Entry）**：新角色推门迈步进入空间的物理动作镜头；
     2. **反应与转头镜（Reaction & Gaze Shift）**：场内原有角色听到动静，视线与身体从彼此对视**转向新角色所在方位**（首帧锁原朝向 `frame_idx: 0`，中间帧与尾帧锁转向后的新朝向 `frame_idx: N, end`）；
     3. **落座/站定镜（Settle & Speak）**：新角色走到主位坐下或站定（注意：**落座动作必须在开口说话前完成**，禁止在说话中途插入闭嘴关键帧）。
2. **三维视线轴线与关键帧朝向强绑定（Eyeline & Keyframe Axis Matching）**：
   - 当角色 A 对角色 B 说话时，关键帧与参考图应为**面向彼此的侧脸/斜侧脸**（如 `role1` 侧脸锚点 + 双人互视锁帧）。
   - 一旦对话目标切换为**主位的第三人 C（如董事长）**，绝不能继续沿用 A 与 B 侧脸互看的双人关键帧或纯侧脸参考图（否则模型会让角色一边看着对方一边喊第三人的名字）。
   - **底层机制深度复盘（`ref_images` 与 `guide_keyframes` 的潜空间拔河效应）**：
     - 在 MiniMax-H3 (LTX-2 19B R2V) 中，`ref_images`（`<Picture 1>`, `<Picture 2>`）**不仅是人脸 ID 锚点，更是未锁帧中间时段的默认景别与头部朝向锚点**；而 `MiniMaxH3AddGuide`（`guide_keyframes`）仅锚定指定的单个潜空间帧（`frame_idx // 8`）。
     - **复现规律**：若首尾 `guide_keyframes`（如 `frame_idx: 0` 与 `153`，间隔 `> 80 帧`）给的是「双人朝向主位中景」，但 `ref_images` 传入的却是「单人朝向侧面的近景特写」（如 `ref_linwei_codex_real` 朝右看、`ref_chenmo_codex_real` 单人特写朝左看），则在 `1.0s ~ 5.0s` 的中间未锁帧区间，`ref_images` 会把画面强行拉回单人特写（如 V2 `Seg 08`）或侧脸对视（如 V2 `Seg 09`），到了尾帧又跳回双人中景，形成段内跳切！
     - **根治铁律**：
       1. **`ref_images` 必须与当前镜头的景别及视线轴线完全同构**：若当前镜头是双人朝向主位中景，直接将 `two_shot_facing_chairman` 作为主参考图，或使用真正朝向正前方的单人定妆照（禁止拿朝左/朝右看的单人特写图去配双人正对镜头的关键帧）；
       2. **长镜头（`> 120 帧`）的关键帧跨度控制在 `≤ 75 帧`（约 `3.0s`）以内**（如 V2 `Seg 06` 在 `frame_idx: 42` 与 `119` 之间跨度为 `77 帧`，全程稳如磐石零跳切）。
3. **动态帧长适配台词与动作（Dynamic Segment Length）**：
   - 打破全片机械固定 `5.0s (120f)` 的限制：纯动作反应镜可用 `4.0s (96f)`；带“落座+台词”或长句交锋的镜头扩充至 `6.5s (156f~158f)`；高潮总结陈词给足 `8.0s (192f)`，保证语速自然沉稳（`≤ 3.6 字/秒`）。
