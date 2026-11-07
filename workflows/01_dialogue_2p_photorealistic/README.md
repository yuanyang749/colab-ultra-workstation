# 🎬 工作流 01：真人写实 · 双角色多段连续对手戏（2-Person Photorealistic Dialogue Drama）

本目录是明确选用的旧 V9 专项模板，先读 [当前 SOP](../../references/minimax_h3_zero_waste_sop.md)。它不是当前“每段身份图 + 第 0 帧”的独立分段基线；真人题材不会自动选择本模板。下文的时长、重拍速度和“锁死”效果只作为旧案例描述，验收仍以实际输出为准。生产配置在项目副本中修改。

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

## 开拍前安排空间、表演与引导

- 按剧本交代角色入场、注意力转移与站定状态；可用一个镜头完成，也可按节奏切镜，不强制固定三镜结构或禁止边走边说。
- 先确认谁看谁、角色位置、道具和相机轴线，再检查首帧与提示词一致。听者可以闭嘴回应；避免为了固定身份冻结表演。
- 身份参考与场景引导分工明确。不要为了纠正视线直接把双人场景图替代所有独立身份图；当前独立分段基线把场景单图只接 Add Guide。
- 如使用本目录的多关键帧或 TailLock，检查所选图的实际节点 schema、索引映射及完整对白；额外引导不能保证精确朝向或无跳切。不要套用其他模型的 `frame_idx // 8` 或固定间隔公式。
- 时长按对白、动作与反应确定，使用所选 H3 图要求的 `17k+5` 帧，如 90、124、158、192；24 fps 下分别约 3.75、5.17、6.58、8 秒。保留完整台词，提示时间不是硬调度。
- 后段承接上一段实际状态。仅本图确实采用相应潜变量重叠时使用续接裁帧；检查接缝两侧的体型、姿态、视线、速度和声音。
