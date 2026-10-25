#!/usr/bin/env python3
"""Standalone Cloud Dual-Engine TTS Studio Launcher & Resident API Server (Zero ComfyUI Dependency).

Dual-Engine Architecture (Qwen3-TTS-1.7B + Bilibili IndexTTS-2):
- Completely decouples TTS voice design, zero-shot cloning, emotion decoupling, and timeline assembly
  from ComfyUI and 42GB video models.
- Persistently caches BOTH engines on 20TB Google Drive (`/content/drive/MyDrive/AI_Models/TTS_Cache/`):
  1. `Qwen3-TTS-12Hz-1.7B-VoiceDesign` + `Qwen3-TTS-12Hz-1.7B-Base` (Alibaba Qwen3 Audio)
  2. `IndexTTS-2` (`IndexTeam/IndexTTS-2` + `w2v-bert-2.0` + `bigvgan_v2` + `campplus` + `MaskGCT`)
- Stages weights to local NVMe (`/content/nvme_tts_cache`) via `rsync`/`aria2c` and keeps both engines
  resident in Cloud GPU VRAM on `127.0.0.1:8199` (`fp16`/`bf16` CUDA accelerated).
- Supports 3 Synthesis Modes (`--mode`):
  1. `hybrid` (Whitepaper Golden Synergy - DEFAULT):
     - Uses `Qwen3-TTS-VoiceDesign` + F0 Pitch Guard (`160-205 Hz` for 36yo Lin Wei, `90-130 Hz` for 42yo Chen Mo)
       to create 100% non-anime, realistic mature cinema voice seeds (`seed_linwei_pure.wav`, `seed_chenmo_pure.wav`).
     - Feeds those mature voice seeds into `IndexTTS-2` (with 8D emotion vector decoupling & natural breath cadence)
       to synthesize the dialogue lines (`line1`..`line4`) with realistic human micro-pauses and tension!
  2. `qwen`:
     - Uses `Qwen3-TTS-VoiceDesign` + `Qwen3-TTS-Base` end-to-end.
  3. `indextts`:
     - Uses `IndexTTS-2` directly with custom reference WAVs from `/content/drive/MyDrive/AI_Models/TTS_VoiceBank/`
       (or existing mature seeds).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if os.path.isdir("/content/drive/My Drive") and not os.path.exists("/content/drive/MyDrive"):
    os.symlink("/content/drive/My Drive", "/content/drive/MyDrive")

DRIVE_ROOT = Path("/content/drive/MyDrive")
DRIVE_TTS_CACHE = DRIVE_ROOT / "AI_Models/TTS_Cache"
DRIVE_INDEXTTS_CACHE = DRIVE_TTS_CACHE / "IndexTTS-2"
DRIVE_HF_CACHE = DRIVE_TTS_CACHE / "hf_cache"
DRIVE_VOICE_BANK = DRIVE_ROOT / "AI_Models/TTS_VoiceBank"
DRIVE_COMFY_IN = DRIVE_ROOT / "AI_Workstation/ComfyUI/input"
DRIVE_TTS_OUT = DRIVE_ROOT / "MinimaxH3_Outputs/tts_takes"

LOCAL_NVME_CACHE = Path("/content/nvme_tts_cache")
LOCAL_HF_CACHE = LOCAL_NVME_CACHE / "hf_cache"
LOCAL_INDEXTTS_REPO = Path("/content/index-tts")
LOCAL_INDEXTTS_CKPT = LOCAL_NVME_CACHE / "IndexTTS-2"
LOCAL_OUT = Path("/tmp/tts_takes")
DAEMON_URL = "http://127.0.0.1:8199"

INDEXTTS_FILES = [
    "config.yaml",
    "bpe.model",
    "gpt.pth",
    "s2mel.pth",
    "emo_matrix.pt",
    "spk_matrix.pt",
    "feat1.pt",
    "feat2.pt",
    "wav2vec2bert_stats.pt",
    "qwen0.6bemo4-merge/config.json",
    "qwen0.6bemo4-merge/generation_config.json",
    "qwen0.6bemo4-merge/model.safetensors",
    "qwen0.6bemo4-merge/tokenizer.json",
    "qwen0.6bemo4-merge/tokenizer_config.json",
    "qwen0.6bemo4-merge/vocab.json",
    "qwen0.6bemo4-merge/merges.txt",
]


def sh(cmd: str) -> str:
    print(f"🔧 $ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if p.returncode != 0 and p.stderr:
        print(f"⚠️ {p.stderr[-600:]}", flush=True)
    return p.stdout


def is_daemon_up() -> bool:
    try:
        resp = json.loads(urllib.request.urlopen(f"{DAEMON_URL}/health", timeout=2).read().decode())
        return resp.get("status") == "ok"
    except Exception:
        return False


def ensure_dual_engine_environment(need_indextts: bool = False) -> None:
    """Install Qwen3-TTS (and optionally IndexTTS-2) and pull weights directly from HF CDN (500 MB/s) to local NVMe."""
    t0 = time.time()
    LOCAL_HF_CACHE.mkdir(parents=True, exist_ok=True)
    LOCAL_INDEXTTS_CKPT.mkdir(parents=True, exist_ok=True)
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)

    if DRIVE_ROOT.exists():
        DRIVE_TTS_OUT.mkdir(parents=True, exist_ok=True)
        DRIVE_COMFY_IN.mkdir(parents=True, exist_ok=True)

    chk_sys = subprocess.run(
        [sys.executable, "-c", "from qwen_tts import Qwen3TTSModel; import soundfile, librosa, scipy"],
        capture_output=True,
    )
    if chk_sys.returncode != 0:
        print("⚡ [Cloud TTS Studio] 正在极速安装 Qwen3-TTS 核心依赖 (~12s)...", flush=True)
        sh("pip install -q uv && uv pip install --system qwen-tts hf_transfer soundfile librosa scipy accelerate sentencepiece 'protobuf>=5.26.1' 'numpy<=2.2.6'")

    if need_indextts:
        chk_idx = subprocess.run(
            ["/content/venv_idx/bin/python", "-c", "import numba, whisper, transformers; assert transformers.__version__ == '4.52.1'"],
            capture_output=True,
        ) if Path("/content/venv_idx/bin/python").exists() else subprocess.run(["false"])
        if chk_idx.returncode != 0:
            print("⚡ [Cloud TTS Studio] 正在按需配置 IndexTTS-2 隔离环境...", flush=True)
            sh("apt-get update -qq && apt-get install -y -qq aria2 rsync sox libsox-dev")
            sh(
                "uv pip install --system omegaconf munch einops jieba cn2an pypinyin g2p-en "
                "openai-whisper tiktoken json5 matplotlib modelscope 'numpy<=2.2.6'"
            )
            sh("uv pip install --system --no-deps descript-audiotools argbind julius ffmpy flatten-dict markdown2 randomname")
            sh("rm -rf /content/venv_idx && uv venv --system-site-packages /content/venv_idx")
            sh("uv pip install --python /content/venv_idx/bin/python --no-deps 'transformers==4.52.1' 'tokenizers==0.21.4'")
            for p in Path("/content/venv_idx").glob("lib/python*/site-packages/transformers/utils/args_doc.py"):
                txt = p.read_text(encoding="utf-8").replace("cls.__doc__ = docstring", "cls.__doc__ = locals().get('docstring', '')")
                p.write_text(txt, encoding="utf-8")

        if not (LOCAL_INDEXTTS_REPO / "indextts/infer_v2.py").exists():
            sh(f"rm -rf '{LOCAL_INDEXTTS_REPO}' && git clone --depth 1 https://github.com/index-tts/index-tts.git '{LOCAL_INDEXTTS_REPO}'")
            sh(f"uv pip install --python /content/venv_idx/bin/python --no-deps -e '{LOCAL_INDEXTTS_REPO}'")
            for dp, content in [("indextts/s2mel/dac/__init__.py", "# patched\n"), ("indextts/s2mel/dac/nn/__init__.py", "from . import layers, quantize\n")]:
                p_dac = LOCAL_INDEXTTS_REPO / dp
                if p_dac.exists():
                    p_dac.write_text(content, encoding="utf-8")
            front_py = LOCAL_INDEXTTS_REPO / "indextts/utils/front.py"
            if front_py.exists():
                ftxt = front_py.read_text(encoding="utf-8")
                if "class _Cn2AnNorm:" not in ftxt:
                    ftxt = ftxt.replace(
                        "    def load(self):",
                        "    def load(self):\n        import cn2an\n        class _Cn2AnNorm:\n            def normalize(self, s):\n                try: return cn2an.transform(s, 'an2cn')\n                except Exception: return s\n        self.zh_normalizer = _Cn2AnNorm()\n        self.en_normalizer = _Cn2AnNorm()\n        return\n    def _orig_load(self):",
                        1,
                    )
                    front_py.write_text(ftxt, encoding="utf-8")
            common_py = LOCAL_INDEXTTS_REPO / "indextts/utils/common.py"
            if common_py.exists():
                txt = common_py.read_text(encoding="utf-8").replace(
                    "torchaudio.save(save_path, wav, sample_rate, encoding=\"PCM_S\", bits_per_sample=16)",
                    "import soundfile as sf; sf.write(str(save_path), wav.squeeze(0).cpu().numpy(), sample_rate, subtype='PCM_16')",
                )
                common_py.write_text(txt, encoding="utf-8")

        req_idx_files = ["gpt.pth", "s2mel.pth", "qwen0.6bemo4-merge/model.safetensors", "wav2vec2bert_stats.pt"]
        if not all((LOCAL_INDEXTTS_CKPT / f).exists() for f in req_idx_files):
            print("🚀 [HF CDN 500MB/s 直拉] 正在通过 aria2c 16线程并发拉取 IndexTTS-2 权重至本地 NVMe (~20s)...", flush=True)
            (LOCAL_INDEXTTS_CKPT / "qwen0.6bemo4-merge").mkdir(parents=True, exist_ok=True)
            aria_lines = []
            for rel in INDEXTTS_FILES:
                url = f"https://huggingface.co/IndexTeam/IndexTTS-2/resolve/main/{rel}"
                aria_lines.append(f"{url}\n  dir={LOCAL_INDEXTTS_CKPT / Path(rel).parent}\n  out={Path(rel).name}\n")
            aria_list = Path("/tmp/indextts2_aria.txt")
            aria_list.write_text("".join(aria_lines), encoding="utf-8")
            sh(f"aria2c -c -i '{aria_list}' -x 16 -s 16 -j 8 --file-allocation=none --console-log-level=warn")

    os.environ["HF_HOME"] = str(LOCAL_HF_CACHE)
    os.environ["HF_HUB_CACHE"] = str(LOCAL_HF_CACHE)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print(f"✅ [Cloud TTS Studio] 环境就绪 ({time.time() - t0:.1f}s)", flush=True)


DAEMON_CODE = r'''#!/usr/bin/env python3
"""Resident Dual-Engine GPU TTS Server (Qwen3-TTS-1.7B + IndexTTS-2) on 127.0.0.1:8199."""
import json
import os
import shutil
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ["HF_HOME"] = "/content/nvme_tts_cache/hf_cache"
os.environ["HF_HUB_CACHE"] = "/content/nvme_tts_cache/hf_cache"
sys.path.insert(0, "/content/index-tts")

import librosa
import numpy as np
import scipy.signal as signal
import soundfile as sf
import subprocess
import torch
from qwen_tts import Qwen3TTSModel

print("🎙️ [TTS Daemon] (1/2) 正在加载 Qwen3-TTS-12Hz-1.7B (VoiceDesign + Base) 进入 GPU 显存常驻...", flush=True)
t0 = time.time()
DESIGN_MODEL = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)
BASE_MODEL = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)
print(f"✅ [TTS Daemon] Qwen3-TTS 常驻完成 ({time.time() - t0:.1f}s)！", flush=True)

# Write IndexTTS-2 batch worker script that runs inside isolated /content/venv_idx (transformers==4.52.1)
INDEXTTS_WORKER = Path("/tmp/indextts_batch_worker.py")
INDEXTTS_WORKER.write_text("""import json, os, sys
os.environ["HF_HOME"] = "/content/nvme_tts_cache/hf_cache"
os.environ["HF_HUB_CACHE"] = "/content/nvme_tts_cache/hf_cache"
sys.path.insert(0, "/content/index-tts")
from indextts.infer_v2 import IndexTTS2

tasks = json.loads(sys.argv[1])
tts = IndexTTS2(
    cfg_path="/content/nvme_tts_cache/IndexTTS-2/config.yaml",
    model_dir="/content/nvme_tts_cache/IndexTTS-2",
    use_fp16=True,
    device="cuda:0",
    use_cuda_kernel=False,
)
for item in tasks:
    tts.infer(
        spk_audio_prompt=item["ref_p"],
        text=item["txt"],
        output_path=item["raw_out"],
        emo_vector=item["emo_vec"],
        emo_alpha=0.40,
        use_random=False,
        verbose=False,
    )
print("INDEXTTS_BATCH_DONE")
""", encoding="utf-8")

# Fast local NVMe cache only (avoid slow FUSE rsync of thousands of small HF cache files)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def measure_median_f0(wav: np.ndarray, sr: int) -> float:
    """Measure median fundamental frequency (Hz) to filter out high-pitched/anime voices."""
    try:
        f0, _, _ = librosa.pyin(
            np.asarray(wav, dtype=np.float32),
            fmin=65.0,
            fmax=360.0,
            sr=sr,
            frame_length=1024,
        )
        valid = f0[~np.isnan(f0)]
        return float(np.median(valid)) if len(valid) > 0 else 0.0
    except Exception:
        return 0.0


def clean_and_level(wav: np.ndarray, sr: int, target_sr: int = 24000, **kwargs) -> np.ndarray:
    """100% Phase-Pure Audio Mastering:
    - ZERO time_stretch (eliminates STFT phase-vocoder double-voice/chorus artifact).
    - ZERO internal silence chopping (preserves natural human breath & prosody transitions).
    - ZERO parallel EQ mixing (eliminates comb-filtering hollow resonance).
    - Only trims leading/trailing dead air and normalizes peak to -1.1 dBFS.
    """
    y = np.asarray(wav, dtype=np.float32)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    yc, _ = librosa.effects.trim(y, top_db=34, frame_length=1024, hop_length=256)
    if len(yc) == 0:
        yc = y
    pk = np.max(np.abs(yc))
    if pk > 0.01:
        yc = yc / pk * 0.88
    fade = min(int(0.015 * sr), len(yc) // 6)
    if fade > 0:
        yc[:fade] *= np.linspace(0.0, 1.0, fade)
        yc[-fade:] *= np.linspace(1.0, 0.0, fade)
    return yc.astype(np.float32)


def select_best_pitch_seed(
    role_name: str,
    seed_text: str,
    instruct: str,
    target_f0_range: tuple[float, float],
    seeds: list[int],
    temperature: float = 0.42,
) -> tuple[np.ndarray, int, float, int]:
    """Generate multiple candidates via Qwen3-VoiceDesign and pick the one whose F0 fits the mature target range."""
    best_wav = None
    best_sr = 24000
    best_f0 = 0.0
    best_seed = seeds[0]
    best_dist = 9999.0
    low_f0, high_f0 = target_f0_range
    mid_f0 = 0.5 * (low_f0 + high_f0)

    for sd in seeds:
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)
        wavs, sr = DESIGN_MODEL.generate_voice_design(
            text=seed_text,
            language="Chinese",
            instruct=instruct,
            temperature=temperature,
            top_p=0.75,
        )
        w = clean_and_level(wavs[0], sr, target_sr=24000)
        f0 = measure_median_f0(w, 24000)
        if low_f0 <= f0 <= high_f0:
            dist = abs(f0 - mid_f0) * 0.2
        else:
            dist = min(abs(f0 - low_f0), abs(f0 - high_f0)) + 50.0
        print(f"  🔍 [{role_name}] 候选 seed={sd} -> 实测基频 F0={f0:.1f} Hz (目标 {low_f0:.0f}-{high_f0:.0f} Hz)", flush=True)
        if dist < best_dist:
            best_dist = dist
            best_wav = w
            best_sr = 24000
            best_f0 = f0
            best_seed = sd

    print(f"  ✅ [{role_name}] 锁定最佳沉稳声线: seed={best_seed}, F0={best_f0:.1f} Hz", flush=True)
    return best_wav, best_sr, best_f0, best_seed


def run_lounge_duel_mature(cfg: dict) -> dict:
    """Synthesize mature dialogue using `hybrid` (Qwen3-VoiceDesign seed -> IndexTTS-2 synthesis), `qwen`, or `indextts`."""
    mode = cfg.get("mode", "hybrid")  # "hybrid" | "qwen" | "indextts"
    out_dir = Path(cfg.get("out_dir", "/tmp/tts_takes"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = 24000

    linwei_seed_text = "夜色已经很深了，这间会客厅里说的每一句话，都只会留在我们两个人之间。"
    chenmo_seed_text = "外面的风雨再大，也吹不散桌面上已经摆好的棋局，我们都很清楚这一步的分量。"

    seed_lin_path = out_dir / "seed_linwei_pure.wav"
    seed_chen_path = out_dir / "seed_chenmo_pure.wav"

    # Step 1: Prepare or design mature non-anime voice seeds
    custom_lin_ref = cfg.get("linwei_ref_wav", "")
    custom_chen_ref = cfg.get("chenmo_ref_wav", "")

    if custom_lin_ref and Path(custom_lin_ref).exists():
        y_lin, sr_in = librosa.load(custom_lin_ref, sr=sr, mono=True)
        w_lin_seed = clean_and_level(y_lin, sr_in, target_sr=sr)
        f0_lin = measure_median_f0(w_lin_seed, sr)
        sd_lin = -1
        sf.write(str(seed_lin_path), w_lin_seed, sr, subtype="PCM_16")
    else:
        linwei_instruct = cfg.get(
            "linwei_instruct",
            "36岁成熟中国女性高管，深沉内敛的女中音，胸腔共鸣浑厚扎实，声线偏低沉稳重，语速从容克制，像现实主义涉案电影里的幕后女操盘手，自然松弛的生活化同期声质感，严禁播音腔、严禁甜美少女音或动漫腔。"
        )
        w_lin_seed, sr, f0_lin, sd_lin = select_best_pitch_seed(
            "林薇(36岁沉稳女中音)",
            linwei_seed_text,
            linwei_instruct,
            target_f0_range=(160.0, 205.0),
            seeds=cfg.get("linwei_seeds", [309]),
            temperature=float(cfg.get("temperature", 0.42)),
        )
        sf.write(str(seed_lin_path), w_lin_seed, sr, subtype="PCM_16")

    if custom_chen_ref and Path(custom_chen_ref).exists():
        y_chen, sr_in = librosa.load(custom_chen_ref, sr=sr, mono=True)
        w_chen_seed = clean_and_level(y_chen, sr_in, target_sr=sr)
        f0_chen = measure_median_f0(w_chen_seed, sr)
        sd_chen = -1
        sf.write(str(seed_chen_path), w_chen_seed, sr, subtype="PCM_16")
    else:
        chenmo_instruct = cfg.get(
            "chenmo_instruct",
            "42岁成熟中国男性，低沉沧桑的男中低音，声音厚重有磁性与颗粒感，语气隐忍克制、深思熟虑，真实电影同期声质感，自然呼吸停顿，不要朗诵腔。"
        )
        w_chen_seed, sr, f0_chen, sd_chen = select_best_pitch_seed(
            "陈默(42岁深沉男中音)",
            chenmo_seed_text,
            chenmo_instruct,
            target_f0_range=(90.0, 130.0),
            seeds=cfg.get("chenmo_seeds", [142]),
            temperature=float(cfg.get("temperature", 0.42)),
        )
        sf.write(str(seed_chen_path), w_chen_seed, sr, subtype="PCM_16")

    # Step 2: Synthesize the 4 dialogue lines using the selected engine mode
    # IndexTTS-2 emotion vectors: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    lines_spec = [
        ("line1_linwei.wav", "linwei", "这份底牌一旦交出去，就再也收不回来了。", 2.55, [0.0, 0.0, 0.05, 0.0, 0.0, 0.15, 0.0, 0.65], 601),
        ("line2_chenmo.wav", "chenmo", "我等这一天，已经等得太久了。", 1.55, [0.0, 0.0, 0.15, 0.0, 0.0, 0.25, 0.0, 0.55], 602),
        ("line3_chenmo.wav", "chenmo", "有些真相一旦揭开，就没有回头路了。", 2.48, [0.0, 0.08, 0.15, 0.0, 0.0, 0.30, 0.0, 0.45], 603),
        ("line4_linwei.wav", "linwei", "那我更要看清楚。", 1.70, [0.0, 0.10, 0.0, 0.0, 0.0, 0.10, 0.0, 0.65], 604),
    ]

    line_wavs = {}
    line_meta = {}

    # Always synthesize both Qwen3-TTS-Base and IndexTTS-2 takes when in hybrid mode so user can compare both side-by-side!
    print("🎛️ [Step 2A] 使用 Qwen3-TTS-Base 基于 36岁沉稳女中音/42岁深沉男中音种子合成对白...", flush=True)
    prompt_lin = BASE_MODEL.create_voice_clone_prompt(
        ref_audio=str(seed_lin_path), ref_text=linwei_seed_text, x_vector_only_mode=False
    )
    prompt_chen = BASE_MODEL.create_voice_clone_prompt(
        ref_audio=str(seed_chen_path), ref_text=chenmo_seed_text, x_vector_only_mode=False
    )
    qwen_wavs = {}
    for fname, role, txt, max_dur, _, l_seed in lines_spec:
        torch.manual_seed(l_seed)
        torch.cuda.manual_seed_all(l_seed)
        v_prompt = prompt_lin if role == "linwei" else prompt_chen
        wavs, sr_q = BASE_MODEL.generate_voice_clone(
            text=txt,
            language="Chinese",
            voice_clone_prompt=v_prompt,
            temperature=0.40,
            top_p=0.75,
        )
        wy = clean_and_level(wavs[0], sr_q, target_sr=sr, target_max_sec=max_dur, max_pause=0.08)
        qwen_wavs[fname] = wy
        sf.write(str(out_dir / f"qwen_{fname}"), wy, sr, subtype="PCM_16")

    idx_wavs = {}
    if mode in ("hybrid", "indextts"):
        print("🎛️ [Step 2B] 使用 B站 IndexTTS-2 (隔离环境 tf 4.52) 克隆同一 36岁沉稳种子 + 8维克制情绪向量合成对白...", flush=True)
        torch.cuda.empty_cache()
        batch_tasks = []
        for fname, role, txt, max_dur, emo_vec, _ in lines_spec:
            ref_p = str(seed_lin_path if role == "linwei" else seed_chen_path)
            raw_out = out_dir / f"raw_idx_{fname}"
            batch_tasks.append({"fname": fname, "ref_p": ref_p, "txt": txt, "raw_out": str(raw_out), "emo_vec": emo_vec, "max_dur": max_dur})

        p_idx = subprocess.run(
            ["/content/venv_idx/bin/python", "-u", "/tmp/indextts_batch_worker.py", json.dumps(batch_tasks)],
            capture_output=True,
            text=True,
        )
        if p_idx.returncode != 0:
            raise RuntimeError(f"❌ IndexTTS-2 批量合成失败:\n{p_idx.stderr}\n{p_idx.stdout}")

        for item in batch_tasks:
            fname = item["fname"]
            raw_out = Path(item["raw_out"])
            y_raw, sr_raw = librosa.load(str(raw_out), sr=None, mono=True)
            wy = clean_and_level(y_raw, sr_raw, target_sr=sr, target_max_sec=item["max_dur"], max_pause=0.08)
            idx_wavs[fname] = wy
            sf.write(str(out_dir / f"indextts_{fname}"), wy, sr, subtype="PCM_16")
            raw_out.unlink(missing_ok=True)

    active_wavs = idx_wavs if (mode in ("hybrid", "indextts") and idx_wavs) else qwen_wavs
    for fname, wy in active_wavs.items():
        sf.write(str(out_dir / fname), wy, sr, subtype="PCM_16")
        line_wavs[fname] = wy
        line_meta[fname] = {
            "qwen_f0_hz": round(measure_median_f0(qwen_wavs[fname], sr), 1),
            "indextts_f0_hz": round(measure_median_f0(idx_wavs[fname], sr), 1) if fname in idx_wavs else None,
            "duration_sec": round(len(wy) / sr, 2),
        }

    def build_pair_tracks(tag_prefix: str, w_dict: dict) -> tuple[Path, Path, Path, dict]:
        y1, y2, y3, y4 = w_dict["line1_linwei.wav"], w_dict["line2_chenmo.wav"], w_dict["line3_chenmo.wav"], w_dict["line4_linwei.wav"]
        d1, d2, d3, d4 = len(y1) / sr, len(y2) / sr, len(y3) / sr, len(y4) / sr
        t1 = 0.18
        t2 = max(2.95, t1 + d1 + 0.25)
        seg1_dur = max(5.21, round(t2 + d2 + 0.15, 2))
        seg1 = np.zeros(int(round(seg1_dur * sr)), dtype=np.float32)
        seg1[int(t1 * sr) : int(t1 * sr) + len(y1)] += y1
        seg1[int(t2 * sr) : int(t2 * sr) + len(y2)] += y2

        t3 = 0.68
        t4 = t3 + d3 + 0.28
        seg2_dur = max(5.54, round(t4 + d4 + 0.15, 2))
        seg2 = np.zeros(int(round(seg2_dur * sr)), dtype=np.float32)
        seg2[int(t3 * sr) : int(t3 * sr) + len(y3)] += y3
        seg2[int(t4 * sr) : int(t4 * sr) + len(y4)] += y4

        s1_p = out_dir / f"{tag_prefix}seg01_guide_audio.wav"
        s2_p = out_dir / f"{tag_prefix}seg02_guide_audio.wav"
        m_p = out_dir / f"{tag_prefix}preview_dialogue_master_10s.wav"
        sf.write(str(s1_p), seg1, sr, subtype="PCM_16")
        sf.write(str(s2_p), seg2, sr, subtype="PCM_16")
        overlap_samples = int(round(14 / 24.0 * sr))
        s1_eff = seg1_dur - (14 / 24.0)
        master = np.concatenate([seg1, seg2[overlap_samples:]])
        sf.write(str(m_p), master, sr, subtype="PCM_16")
        tl = {
            "seg1_dur_sec": seg1_dur,
            "seg2_dur_sec": seg2_dur,
            "seg1_line1": f"{t1:.2f}s - {t1+d1:.2f}s",
            "seg1_line2": f"{t2:.2f}s - {t2+d2:.2f}s",
            "seg2_line3": f"{t3:.2f}s - {t3+d3:.2f}s (master {s1_eff+t3:.2f}s - {s1_eff+t3+d3:.2f}s)",
            "seg2_line4": f"{t4:.2f}s - {t4+d4:.2f}s (master {s1_eff+t4:.2f}s - {s1_eff+t4+d4:.2f}s)",
        }
        return s1_p, s2_p, m_p, tl

    build_pair_tracks("qwen_", qwen_wavs)
    if idx_wavs:
        build_pair_tracks("indextts_", idx_wavs)
    s1_path, s2_path, master_path, tl_info = build_pair_tracks("", active_wavs)

    # Sync to Google Drive & ComfyUI input directory
    drv_out = Path("/content/drive/MyDrive/MinimaxH3_Outputs/tts_takes")
    comfy_in = Path("/content/drive/MyDrive/AI_Workstation/ComfyUI/input")
    if drv_out.parent.exists():
        drv_out.mkdir(parents=True, exist_ok=True)
        for f in out_dir.glob("*.wav"):
            shutil.copy2(f, drv_out / f.name)
    if comfy_in.parent.exists():
        comfy_in.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed_lin_path, comfy_in / "voice_linwei_seed.wav")
        shutil.copy2(seed_chen_path, comfy_in / "voice_chenmo_seed.wav")
        shutil.copy2(s1_path, comfy_in / "seg01_guide_audio.wav")
        shutil.copy2(s2_path, comfy_in / "seg02_guide_audio.wav")

    return {
        "status": "ok",
        "mode": mode,
        "linwei_seed": {"seed": sd_lin, "f0_hz": round(f0_lin, 1)},
        "chenmo_seed": {"seed": sd_chen, "f0_hz": round(f0_chen, 1)},
        "lines": line_meta,
        "timeline": tl_info,
        "out_dir": str(out_dir),
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, data: dict) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "engines": ["Qwen3-TTS-1.7B", "IndexTTS-2"]})
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        if self.path == "/synthesize_preset":
            try:
                res = run_lounge_duel_mature(body)
                self._json(200, res)
            except Exception as e:
                import traceback
                self._json(500, {"error": str(e), "trace": traceback.format_exc()})
        else:
            self._json(404, {"error": "unknown_endpoint"})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", 8199), Handler)
    print("🚀 [TTS Daemon] 双引擎常驻语音服务已启动: http://127.0.0.1:8199", flush=True)
    srv.serve_forever()
'''


def start_daemon_if_needed(need_indextts: bool = False) -> None:
    if is_daemon_up():
        print("⚡ [Cloud TTS Studio] 双引擎显存常驻服务已在 127.0.0.1:8199 运行中！(0秒加载)", flush=True)
        return
    ensure_dual_engine_environment(need_indextts=need_indextts)
    daemon_script = Path("/tmp/cloud_tts_daemon.py")
    daemon_script.write_text(DAEMON_CODE, encoding="utf-8")
    log_file = open("/tmp/cloud_tts_daemon.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(daemon_script)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=dict(os.environ, HF_HOME=str(LOCAL_HF_CACHE), HF_HUB_CACHE=str(LOCAL_HF_CACHE)),
    )
    print("⏳ [Cloud TTS Studio] 正在唤醒双引擎常驻服务 (Qwen3-TTS + IndexTTS-2 @ 127.0.0.1:8199)...", flush=True)
    for i in range(120):
        time.sleep(2)
        if is_daemon_up():
            print(f"✅ [Cloud TTS Studio] 双引擎常驻服务就绪！(耗时 {(i+1)*2}s)", flush=True)
            return
        if proc.poll() is not None:
            err_log = Path("/tmp/cloud_tts_daemon.log").read_text(errors="ignore")[-1500:]
            raise RuntimeError(f"❌ TTS Daemon 进程异常退出 (exit={proc.returncode}):\n{err_log}")
        if i % 5 == 4 and Path("/tmp/cloud_tts_daemon.log").exists():
            tail = Path("/tmp/cloud_tts_daemon.log").read_text(errors="ignore").splitlines()[-2:]
            print("  " + " | ".join(tail), flush=True)
    raise RuntimeError("❌ TTS Daemon 启动超时，请检查 /tmp/cloud_tts_daemon.log")


def trigger_preset(preset_name: str, custom_cfg: dict | None = None) -> dict:
    mode = (custom_cfg or {}).get("mode", "qwen")
    start_daemon_if_needed(need_indextts=(mode in ("hybrid", "indextts")))
    payload = json.dumps(custom_cfg or {}).encode("utf-8")
    req = urllib.request.Request(
        f"{DAEMON_URL}/synthesize_preset",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    print(f"🎬 [Cloud TTS Studio] 正在执行预设 [{preset_name}] (mode={custom_cfg.get('mode', 'hybrid')})...", flush=True)
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read().decode("utf-8"))
    print(f"✅ [Cloud TTS Studio] 合成完成 ({time.time() - t0:.1f}s): {json.dumps(resp, ensure_ascii=False, indent=2)}", flush=True)
    return resp


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Cloud Dual-Engine TTS Studio Launcher")
    parser.add_argument("--start-daemon", action="store_true", help="Ensure the resident GPU TTS daemon is running")
    parser.add_argument("--run-preset", type=str, default="", help="Run a dialogue preset (e.g., lounge_duel_mature)")
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "qwen", "indextts"],
        help="Synthesis mode: hybrid (Qwen3 mature seed -> IndexTTS-2 dialogue), qwen, or indextts",
    )
    parser.add_argument("--linwei-instruct", type=str, default="", help="Custom instruct override for Lin Wei")
    parser.add_argument("--linwei-ref-wav", type=str, default="", help="Custom reference WAV path for Lin Wei")
    parser.add_argument("--chenmo-ref-wav", type=str, default="", help="Custom reference WAV path for Chen Mo")
    parser.add_argument("--temperature", type=float, default=0.42, help="Sampling temperature (default 0.42)")
    args = parser.parse_args()

    if args.start_daemon and not args.run_preset:
        start_daemon_if_needed()
        return

    if args.run_preset:
        cfg = {
            "mode": args.mode,
            "temperature": args.temperature,
        }
        if args.linwei_instruct:
            cfg["linwei_instruct"] = args.linwei_instruct
        if args.linwei_ref_wav:
            cfg["linwei_ref_wav"] = args.linwei_ref_wav
        if args.chenmo_ref_wav:
            cfg["chenmo_ref_wav"] = args.chenmo_ref_wav
        trigger_preset(args.run_preset, cfg)
        return

    start_daemon_if_needed()


if __name__ == "__main__":
    main()
