#!/usr/bin/env python3
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

DRIVE_COMFY = Path("/content/drive/MyDrive/AI_Workstation/ComfyUI")
FAST_MODELS = Path("/content/fast_models")
LOCAL_COMFY = Path("/content/ComfyUI")

# Pluggable Stage-2 2K Refinement Profiles (Requires explicit user scheme selection!)
PROFILES = {
    "local_cas": {
        "engine": "local_mac",
        "mode": "cas",
        "cas_strength": 0.65,
        "est_ccu_30s": "0.00 CCU (~25s on Mac)",
        "suffix": "Local2K_CAS",
    },
    "local_esrgan": {
        "engine": "local_mac",
        "mode": "esrgan",
        "cas_strength": 0.45,
        "est_ccu_30s": "0.00 CCU (~2.5min on Mac M-series GPU)",
        "suffix": "Local2K_ESRGAN",
    },
    "g4_turbo_face": {
        "engine": "cloud_tessera",
        "spatial_tile": "custom",
        "tile_width": 1536,
        "tile_height": 1088,
        "segment_frames": 124,
        "tile_select": "0",
        "tile_limit": 1,
        "lms_lora_strength": 0.42,
        "guide_release": 0.55,
        "est_ccu_30s": "~2.0 CCU (~12min on G4, 6 passes, Upper-Body/Face Tile 0 only)",
        "suffix": "Tessera2K_TurboFace",
    },
    "g4_fast_face": {
        "engine": "cloud_tessera",
        "spatial_tile": "custom",
        "tile_width": 1536,
        "tile_height": 1088,
        "segment_frames": 56,
        "tile_select": "0",
        "tile_limit": 1,
        "lms_lora_strength": 0.42,
        "guide_release": 0.55,
        "est_ccu_30s": "~5.8 CCU (~34min on G4, 18 passes, Upper-Body/Face Tile 0 only)",
        "suffix": "Tessera2K_FastFace",
    },
    "g4_optimal": {
        "engine": "cloud_tessera",
        "spatial_tile": "custom",
        "tile_width": 1536,
        "tile_height": 1088,
        "segment_frames": 56,
        "tile_select": "",
        "tile_limit": 0,
        "lms_lora_strength": 0.42,
        "guide_release": 0.55,
        "est_ccu_30s": "~18.75 CCU (~102min on G4, 54 passes, Full-Frame 3 Tiles)",
        "suffix": "Tessera2K",
    },
    "local_32g": {
        "engine": "cloud_tessera",
        "spatial_tile": "med_vram",
        "tile_width": 768,
        "tile_height": 768,
        "segment_frames": 28,
        "tile_select": "0",
        "tile_limit": 1,
        "lms_lora_strength": 0.42,
        "guide_release": 0.55,
        "est_ccu_30s": "0.00 CCU (Local 32GB RTX 5090)",
        "suffix": "Tessera2K_Local32G",
    },
    "local_24g": {
        "engine": "cloud_tessera",
        "spatial_tile": "low_vram",
        "tile_width": 640,
        "tile_height": 640,
        "segment_frames": 28,
        "tile_select": "0",
        "tile_limit": 1,
        "lms_lora_strength": 0.42,
        "guide_release": 0.55,
        "est_ccu_30s": "0.00 CCU (Local 24GB RTX 3090/4090)",
        "suffix": "Tessera2K_Local24G",
    },
}

ACTIVE_PROFILE_NAME = os.environ.get("TESSERA_PROFILE", "").strip()
INPUT_OVERRIDE = None
OUTPUT_OVERRIDE = None
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--profile" and i + 2 <= len(sys.argv):
        ACTIVE_PROFILE_NAME = sys.argv[i + 2]
    elif arg.startswith("--profile="):
        ACTIVE_PROFILE_NAME = arg.split("=", 1)[1]
    elif arg in ("--input", "-i") and i + 2 <= len(sys.argv):
        INPUT_OVERRIDE = sys.argv[i + 2]
    elif arg in ("--output", "-o") and i + 2 <= len(sys.argv):
        OUTPUT_OVERRIDE = sys.argv[i + 2]

if not ACTIVE_PROFILE_NAME or ACTIVE_PROFILE_NAME not in PROFILES:
    print(
        "\n🛑 [MANDATORY REFINEMENT GATE] 未指定二阶段精修方案 (--profile)！\n"
        "为防止误跑高消耗云端全幅精修，Agent 必须先向用户提问确认使用以下哪种精修方案：\n"
        "  1. --profile local_cas      : [推荐日常首选 | 0.00 CCU | ~25秒] 本地 Mac Lanczos + FidelityFX CAS 自适应锐化 2K (1536x2688)\n"
        "  2. --profile local_esrgan   : [本地神经网络 | 0.00 CCU | ~2.5分钟] 本地 Mac M系列 GPU Real-ESRGAN + CAS 混合 2K\n"
        "  3. --profile g4_turbo_face  : [极速云端脸部精修 | 30s约 2.0 CCU (~12分钟) | 省89%算力] 仅重绘上半身/脸部 Tile 0 + 124帧单窗直出 (6轮采样)\n"
        "  4. --profile g4_fast_face   : [标准云端脸部精修 | 30s约 5.8 CCU (~34分钟) | 省69%算力] 仅重绘上半身/脸部 Tile 0 + 56帧滑窗 (18轮采样)\n"
        "  5. --profile g4_optimal     : [全幅满血母带精修 | 30s约 18.75 CCU (~102分钟)] 全画幅上中下3分块 × 56帧滑窗 (54轮采样)\n",
        flush=True,
    )
    sys.exit(2)

ACTIVE_PROFILE = PROFILES[ACTIVE_PROFILE_NAME]
print(f"⚙️ Active Stage-2 Refinement Scheme: {ACTIVE_PROFILE_NAME} -> {ACTIVE_PROFILE}", flush=True)

if ACTIVE_PROFILE.get("engine") == "local_mac":
    local_script = Path(__file__).resolve().parent / "upscale_local_mac_2k.py"
    if not INPUT_OVERRIDE:
        default_in = Path(__file__).resolve().parent / "outputs" / "V15_30s_XOrig" / "DouyinDance_RealGirlV3_30s_V15_XOrig_OfficialBase.mp4"
        INPUT_OVERRIDE = str(default_in)
    if not OUTPUT_OVERRIDE:
        in_p = Path(INPUT_OVERRIDE)
        stem = in_p.stem.replace("_OfficialBase", "")
        OUTPUT_OVERRIDE = str(in_p.parent / f"{stem}_{ACTIVE_PROFILE['suffix']}_1536x2688.mp4")
    cmd = [
        sys.executable,
        str(local_script),
        "--input", str(INPUT_OVERRIDE),
        "--output", str(OUTPUT_OVERRIDE),
        "--mode", ACTIVE_PROFILE["mode"],
        "--cas-strength", str(ACTIVE_PROFILE.get("cas_strength", 0.65)),
    ]
    print(f"🚀 Executing 0-CCU Local Mac 2K Refinement: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    sys.exit(0)


def run_cmd(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout:
        print(p.stdout, flush=True)
    if p.stderr:
        print(p.stderr, flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}")


# 1. Fast bootstrap ComfyUI on G4
if not LOCAL_COMFY.exists():
    print("🚀 Running launch_g4_comfyui_fast.py...", flush=True)
    run_cmd([sys.executable, "/content/drive/MyDrive/AI_Workstation/scripts/launch_g4_comfyui_fast.py"])

for sub in ["diffusion_models", "loras", "vae", "text_encoders"]:
    (FAST_MODELS / sub).mkdir(parents=True, exist_ok=True)
    (LOCAL_COMFY / "models" / sub).mkdir(parents=True, exist_ok=True)
    (DRIVE_COMFY / "models" / sub).mkdir(parents=True, exist_ok=True)

# 2. Ensure base model + LMS LoRA + TaoMate 3-step LoRA
base_unet = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
l_base = LOCAL_COMFY / "models" / "diffusion_models" / base_unet
d_base = DRIVE_COMFY / "models" / "diffusion_models" / base_unet
f_base = FAST_MODELS / "diffusion_models" / base_unet
if not (l_base.exists() and l_base.stat().st_size > 15_000_000_000):
    if f_base.exists() and f_base.stat().st_size < 15_000_000_000:
        f_base.unlink()
    if d_base.exists() and d_base.stat().st_size > 15_000_000_000:
        print(f"📦 Copying {base_unet} from Drive to NVMe...", flush=True)
        shutil.copy2(d_base, f_base)
    elif not (f_base.exists() and f_base.stat().st_size > 15_000_000_000):
        url = f"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/{base_unet}"
        print(f"⬇️ Downloading {base_unet} from Comfy-Org/MiniMax-H3 via aria2c...", flush=True)
        run_cmd(["aria2c", "-x16", "-s16", "-k1M", "--console-log-level=warn", "-d", str(FAST_MODELS / "diffusion_models"), "-o", base_unet, url])
    if not l_base.exists():
        os.symlink(f_base, l_base)

lms_lora = "minimax_h3_lms_v1.0_r64.safetensors"
l_lms = LOCAL_COMFY / "models" / "loras" / lms_lora
d_lms = DRIVE_COMFY / "models" / "loras" / lms_lora
f_lms = FAST_MODELS / "loras" / lms_lora
if not (l_lms.exists() and l_lms.stat().st_size > 500_000_000):
    if d_lms.exists() and d_lms.stat().st_size > 500_000_000:
        shutil.copy2(d_lms, f_lms)
    else:
        url = f"https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/resolve/main/loras/{lms_lora}"
        run_cmd(["curl", "-sL", "--fail", "-o", str(f_lms), url])
        shutil.copy2(f_lms, d_lms)
    if not l_lms.exists():
        os.symlink(f_lms, l_lms)

taomate_lora = "TaoMate-H3-3step-ComfyUI.safetensors"
l_tao = LOCAL_COMFY / "models" / "loras" / taomate_lora
d_tao = DRIVE_COMFY / "models" / "loras" / taomate_lora
f_tao = FAST_MODELS / "loras" / taomate_lora
if not (l_tao.exists() and l_tao.stat().st_size > 500_000_000):
    if d_tao.exists() and d_tao.stat().st_size > 500_000_000:
        print(f"📦 Copying {taomate_lora} from Drive to NVMe...", flush=True)
        shutil.copy2(d_tao, f_tao)
    else:
        url = f"https://huggingface.co/CZMartin22/TaoMate-H3-3step-ComfyUI/resolve/main/{taomate_lora}"
        print(f"⬇️ Downloading {taomate_lora} from HuggingFace...", flush=True)
        run_cmd(["curl", "-sL", "--fail", "-o", str(f_tao), url])
        shutil.copy2(f_tao, d_tao)
    if not l_tao.exists():
        os.symlink(f_tao, l_tao)

# 3. Install ComfyUI-Tessera + patch _px_shift + inject TesseraSigmas
tess_dir = LOCAL_COMFY / "custom_nodes" / "ComfyUI-Tessera"
drv_tess = DRIVE_COMFY / "custom_nodes" / "ComfyUI-Tessera"
if not tess_dir.exists():
    if drv_tess.exists():
        shutil.copytree(drv_tess, tess_dir)
    else:
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/hashu786/ComfyUI-Tessera", str(tess_dir)])

nodes_py = tess_dir / "tessera" / "nodes.py"
nodes_txt = nodes_py.read_text()
old_prep = """    def prep(x):
        g = torch.nn.functional.avg_pool2d(
            x.float().mean(dim=(0, 3))[None, None], k, k)[0, 0]
        wy = torch.hann_window(g.shape[0], periodic=False).view(-1, 1)
        wx = torch.hann_window(g.shape[1], periodic=False).view(1, -1)
        return (g - g.mean()) * (wy * wx)"""
new_prep = """    def prep(x):
        g = torch.nn.functional.avg_pool2d(
            x.detach().cpu().float().mean(dim=(0, 3))[None, None], k, k)[0, 0]
        wy = torch.hann_window(g.shape[0], periodic=False, device=g.device).view(-1, 1)
        wx = torch.hann_window(g.shape[1], periodic=False, device=g.device).view(1, -1)
        return (g - g.mean()) * (wy * wx)"""
if old_prep in nodes_txt:
    nodes_txt = nodes_txt.replace(old_prep, new_prep)

if "TesseraSigmas" not in nodes_txt:
    nodes_txt += """

class TesseraSigmas:
    CATEGORY = "Tessera"
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "run"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sigmas": ("STRING", {"default": "1.0, 0.7, 0.4, 0.0"})}}
    def run(self, sigmas):
        vals = [float(x.strip()) for x in sigmas.replace(";", ",").split(",") if x.strip()]
        return (torch.tensor(vals, dtype=torch.float32),)

NODE_CLASS_MAPPINGS["TesseraSigmas"] = TesseraSigmas
NODE_DISPLAY_NAME_MAPPINGS["TesseraSigmas"] = "Tessera Sigmas"
"""
nodes_py.write_text(nodes_txt)

# 4. Stage inputs & slice Stage-1 base video into exact 124-frame (17*7+5) segments:
inp_dir = LOCAL_COMFY / "input"
inp_dir.mkdir(parents=True, exist_ok=True)

mode_30s = "--mode=15s" not in sys.argv
prof_suffix = ACTIVE_PROFILE.get("suffix", "Tessera2K")
v15_xorig_cand = DRIVE_COMFY / "output" / "20261031_DouyinDance_CatGirl_15s" / "V15_30s_XOrig" / "DouyinDance_RealGirlV3_30s_V15_XOrig_OfficialBase.mp4"
v15_30s_cand = DRIVE_COMFY / "output" / "20261031_DouyinDance_CatGirl_15s" / "V15_30s_Full" / "DouyinDance_RealGirlV3_30s_V15_Full_OfficialBase.mp4"
v15_15s_cand = DRIVE_COMFY / "output" / "20261031_DouyinDance_CatGirl_15s" / "V15_S1_OfficialBase" / "DouyinDance_RealGirlV3_15s_V15_S1_OfficialBase.mp4"

if mode_30s and v15_xorig_cand.exists():
    v15_src = v15_xorig_cand
    bgm_filename = "polite_greeting_master_30s.wav"
    out_subdir_name = "V15_30s_XOrig"
    out_2k_name = f"DouyinDance_RealGirlV3_30s_V15_XOrig_{prof_suffix}_1536x2688.mp4"
    # 702 frames (29.25s @ 24fps) covered by 6 chunks of 124 frames:
    seg_defs = [
        (1, 0, 124),
        (2, 119, 124),
        (3, 238, 124),
        (4, 357, 124),
        (5, 476, 124),
        (6, 578, 124),
    ]
elif mode_30s and v15_30s_cand.exists():
    v15_src = v15_30s_cand
    bgm_filename = "polite_greeting_master_30s.wav"
    out_subdir_name = "V15_30s_Full"
    out_2k_name = f"DouyinDance_RealGirlV3_30s_V15_Full_{prof_suffix}_1536x2688.mp4"
    seg_defs = [
        (1, 0, 124),
        (2, 119, 124),
        (3, 238, 124),
        (4, 357, 124),
        (5, 476, 124),
        (6, 578, 124),
    ]
else:
    v15_src = v15_15s_cand
    bgm_filename = "master_bgm_15s.wav"
    out_subdir_name = "V15_S1_Tessera2K"
    out_2k_name = f"DouyinDance_RealGirlV3_15s_V15_S1_{prof_suffix}_1536x2688.mp4"
    # 362 frames (15.083s @ 24fps) covered by 3 chunks of 124 frames:
    seg_defs = [
        (1, 0, 124),
        (2, 119, 124),
        (3, 238, 124),
    ]

print(f"🎞️ Tessera 2K Source: {v15_src} ({len(seg_defs)} x 124f chunks)", flush=True)
shutil.copy2(v15_src, inp_dir / "v15_s1_full.mp4")

for fn in ["ref_realgirl_v3_2k.png", bgm_filename]:
    for cand in [
        DRIVE_COMFY / "input" / fn,
        Path("/content/drive/MyDrive/AI_Workstation/workflows/04_douyin_beauty_dance/inputs") / fn,
        Path("/tmp") / fn,
    ]:
        if cand.exists():
            shutil.copy2(cand, inp_dir / fn)
            break

for sid, f_start, f_cnt in seg_defs:
    seg_mp4 = inp_dir / f"v15_s1_seg{sid}.mp4"
    t_ss = f_start / 24.0
    t_end = (f_start + f_cnt) / 24.0
    run_cmd([
        "ffmpeg", "-y", "-i", str(inp_dir / "v15_s1_full.mp4"),
        "-vf", f"fps=24,trim=start_frame={f_start}:end_frame={f_start + f_cnt},setpts=PTS-STARTPTS",
        "-af", f"atrim={t_ss:.6f}:{t_end:.6f},asetpts=PTS-STARTPTS",
        "-c:v", "libx264", "-crf", "12", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        str(seg_mp4),
    ])

# 5. Start ComfyUI WITHOUT --highvram so Qwen3-VL 32B unloads cleanly after conditioning, leaving full 96GB VRAM for UNET + VAE
log_path = Path("/tmp/comfyui_tessera.log")
tessera_ready = False
try:
    info = json.loads(urllib.request.urlopen("http://127.0.0.1:8188/object_info/TesseraRefine", timeout=2).read().decode())
    tessera_ready = "TesseraRefine" in info
except Exception:
    tessera_ready = False

if not tessera_ready:
    print("🔄 Starting ComfyUI on 127.0.0.1:8188 (--disable-mmap --fast)...", flush=True)
    subprocess.run(["pkill", "-f", "python.*main.py"], check=False)
    time.sleep(2)
    log_f = open(log_path, "w")
    subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188", "--disable-mmap", "--fast"],
        cwd=str(LOCAL_COMFY),
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    for _ in range(90):
        try:
            urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
            break
        except Exception:
            time.sleep(2)
else:
    print("⚡ ComfyUI (TesseraRefine) already warm on 127.0.0.1:8188!", flush=True)


def run_segment_tessera(sid: int, seg_mp4_name: str, anchor_img_name: str) -> Path:
    prefix = f"Tessera2K_V15_S1_Seg{sid:02d}"
    wf = {
      "1": {
        "class_type": "UNETLoader",
        "inputs": {
          "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
          "weight_dtype": "default"
        }
      },
      "3": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
          "model": ["1", 0],
          "lora_name": "TaoMate-H3-3step-ComfyUI.safetensors",
          "strength_model": 1.0
        }
      },
      "4": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
          "model": ["3", 0],
          "lora_name": "minimax_h3_lms_v1.0_r64.safetensors",
          "strength_model": ACTIVE_PROFILE["lms_lora_strength"]
        }
      },
      "6": {
        "class_type": "CLIPLoader",
        "inputs": {
          "clip_name": "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
          "type": "minimax",
          "device": "default"
        }
      },
      "7": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "minimax_h3_video_vae_fp16.safetensors"
        }
      },
      "8": {
        "class_type": "VAELoader",
        "inputs": {
          "vae_name": "minimax_h3_audio_vae_fp32.safetensors"
        }
      },
      "10": {
        "class_type": "LoadVideo",
        "inputs": {
          "file": seg_mp4_name
        }
      },
      "11": {
        "class_type": "GetVideoComponents",
        "inputs": {
          "video": ["10", 0]
        }
      },
      "12": {
        "class_type": "LoadImage",
        "inputs": {
          "image": anchor_img_name
        }
      },
      "14": {
        "class_type": "KSamplerSelect",
        "inputs": {
          "sampler_name": "er_sde"
        }
      },
      "15": {
        "class_type": "KSamplerSelect",
        "inputs": {
          "sampler_name": "euler"
        }
      },
      "16": {
        "class_type": "TesseraSigmas",
        "inputs": {
          "sigmas": "1.0, 0.7, 0.4, 0.0"
        }
      },
      "22": {
        "class_type": "MiniMaxH3ReferenceToVideo",
        "inputs": {
          "clip": ["6", 0],
          "vae": ["7", 0],
          "audio_vae": ["8", 0],
          "width": 1536,
          "height": 2688,
          "length": 124,
          "ref_image_size": "max",
          "ref_videos.ref_video_0": ["11", 0],
          "prompt": "subject_definitions:\n<Video 1>: the reference video. Its picture, motion, framing, timing and content are the target video's content.\n<Picture 1> is a reference of a high resolution upscaled version of the first frame of the video. \n\nretention_analysis:\n<Video 1>: fully_preserved - <Video 1> is reproduced one to one as the target video. Every frame matches the source.\n\nEnhance this video with sharp, crisp details while preserving a natural photorealistic appearance."
        }
      },
      "23": {
        "class_type": "ConditioningZeroOut",
        "inputs": {
          "conditioning": ["22", 0]
        }
      },
      "24": {
        "class_type": "TesseraEncode",
        "inputs": {
          "images": ["11", 0],
          "vae": ["7", 0],
          "positive": ["22", 0],
          "upscale_by": 2.0,
          "upscale_method": "lanczos",
          "segment_frames": ACTIVE_PROFILE["segment_frames"],
          "anchor": ["12", 0],
          "anchor_strength": 1.0,
          "anchor_spread": 4
        }
      },
      "25": {
        "class_type": "VAEEncodeAudio",
        "inputs": {
          "audio": ["11", 1],
          "vae": ["8", 0]
        }
      },
      "26": {
        "class_type": "SolidMask",
        "inputs": {
          "value": 1.0,
          "width": 32,
          "height": 32
        }
      },
      "27": {
        "class_type": "SetLatentNoiseMask",
        "inputs": {
          "samples": ["25", 0],
          "mask": ["26", 0]
        }
      },
      "28": {
        "class_type": "LTXVConcatAVLatent",
        "inputs": {
          "video_latent": ["24", 1],
          "audio_latent": ["27", 0]
        }
      },
      "29": {
        "class_type": "TesseraArgs",
        "inputs": {
          "video_ref_crop": "per_tile",
          "image_ref_crop": "per_tile",
          "guide_noise": 0.0,
          "guide_noise_band": "full",
          "guide_noise_t": "per_frame",
          "guide_release": ACTIVE_PROFILE["guide_release"],
          "guide_inherit": True,
          "hero_guide_noise": -1.0,
          "motion_adapt": True,
          "motion_release_floor": 0.3,
          "fine_below_sigma": 0.75,
          "tile_select": ACTIVE_PROFILE["tile_select"],
          "tile_limit": ACTIVE_PROFILE["tile_limit"],
          "flat_skip": 4.0
        }
      },
      "30": {
        "class_type": "TesseraRefine",
        "inputs": {
          "model": ["4", 0],
          "positive": ["24", 0],
          "negative": ["23", 0],
          "latent": ["28", 0],
          "vae": ["7", 0],
          "sampler": ["14", 0],
          "sampler_fine": ["15", 0],
          "sigmas": ["16", 0],
          "args": ["29", 0],
          "spatial_tile": ACTIVE_PROFILE["spatial_tile"],
          "tile_width": ACTIVE_PROFILE["tile_width"],
          "tile_height": ACTIVE_PROFILE["tile_height"],
          "temporal_tile": "off",
          "temporal_size": 124,
          "temporal_overlap": 4,
          "context_border": 64,
          "seam_overlap": 32,
          "hero_min": 896,
          "hero_fit": "median",
          "cfg": 1.0,
          "seed": 786110 + sid,
          "seed_mode": "global",
          "pixel_out": True,
          "console_progress": "overall",
          "prompt": "subject_definitions:\n<Video 1>: the reference video. Its picture, motion, framing, timing and content are the target video's content.\n<Picture 1> is a reference of a high resolution upscaled version of the first frame of the video. \n\nretention_analysis:\n<Video 1>: fully_preserved - <Video 1> is reproduced one to one as the target video. Every frame matches the source.\n\nEnhance this video with sharp, crisp details while preserving a natural photorealistic appearance."
        }
      },
      "32": {
        "class_type": "CreateVideo",
        "inputs": {
          "images": ["30", 2],
          "audio": ["11", 1],
          "fps": 24.0
        }
      },
      "33": {
        "class_type": "SaveVideo",
        "inputs": {
          "video": ["32", 0],
          "filename_prefix": prefix,
          "format": "mp4",
          "codec": "h264"
        }
      }
    }

    req = urllib.request.Request(
        "http://127.0.0.1:8188/prompt",
        data=json.dumps({"prompt": wf}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
    pid = resp["prompt_id"]
    print(f"\n🎬 [Seg {sid}/{len(seg_defs)}] TesseraRefine 2K (1536x2688, 124f) Queued! prompt_id={pid}", flush=True)
    t0 = time.time()
    last_line = ""
    while True:
        time.sleep(4)
        try:
            hist = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid}", timeout=5).read().decode())
            if pid in hist:
                st = hist[pid].get("status", {})
                if st.get("status_str") == "error":
                    print(log_path.read_text()[-4000:], flush=True)
                    raise RuntimeError(f"Execution error: {st}")
                if st.get("completed", False) or hist[pid].get("outputs"):
                    break
        except urllib.error.URLError:
            pass
        lines = [ln.strip() for ln in log_path.read_text(errors="ignore").splitlines() if "[Tessera" in ln or "it/s" in ln or "100%" in ln]
        if lines and lines[-1] != last_line:
            last_line = lines[-1]
            print(f"  [Seg {sid} | {int(time.time()-t0)}s] {last_line[-140:]}", flush=True)

    cands = sorted((LOCAL_COMFY / "output").glob(f"{prefix}*.mp4"), key=lambda p: p.stat().st_mtime)
    assert cands, f"No output MP4 found for {prefix}!"
    out_p = cands[-1]
    print(f"✅ [Seg {sid}/{len(seg_defs)}] Done in {time.time()-t0:.1f}s -> {out_p.name} ({out_p.stat().st_size/1e6:.2f} MB)", flush=True)
    return out_p


out_drv_dir = DRIVE_COMFY / "output" / "20261031_DouyinDance_CatGirl_15s" / out_subdir_name
out_drv_dir.mkdir(parents=True, exist_ok=True)

seg_outputs = []
cur_anchor = "ref_realgirl_v3_2k.png"
for idx, (sid, f_start, f_cnt) in enumerate(seg_defs):
    cached_seg = out_drv_dir / f"{prof_suffix}_Seg{sid:02d}.mp4"
    if cached_seg.exists() and cached_seg.stat().st_size > 2_000_000:
        print(f"⚡ Reusing completed 2K segment: {cached_seg.name} ({cached_seg.stat().st_size/1e6:.2f} MB)", flush=True)
        out_mp4 = cached_seg
    else:
        out_mp4 = run_segment_tessera(sid, f"v15_s1_seg{sid}.mp4", cur_anchor)
        shutil.copy2(out_mp4, cached_seg)
    seg_outputs.append(out_mp4)
    if idx + 1 < len(seg_defs):
        next_f_start = seg_defs[idx + 1][1]
        local_anchor_frame = next_f_start - f_start
        next_anchor = inp_dir / f"anchor_seg{sid+1}.png"
        run_cmd([
            "ffmpeg", "-y", "-i", str(out_mp4),
            "-vf", f"select=eq(n\\,{local_anchor_frame})", "-frames:v", "1",
            str(next_anchor),
        ])
        cur_anchor = next_anchor.name

# Stitch all refined 2K (1536x2688) 124-frame segments with exact crossfades + lossless master BGM
final_mp4 = out_drv_dir / out_2k_name
tmp_final_mp4 = Path("/tmp") / out_2k_name
bgm_wav = inp_dir / bgm_filename

ff_inputs = []
for p in seg_outputs:
    ff_inputs.extend(["-i", str(p)])
ff_inputs.extend(["-i", str(bgm_wav)])

xfade_parts = []
prev_label = "[0:v]"
for i in range(1, len(seg_defs)):
    prev_end_f = seg_defs[i - 1][1] + seg_defs[i - 1][2]
    cur_start_f = seg_defs[i][1]
    ov_s = (prev_end_f - cur_start_f) / 24.0
    off_s = cur_start_f / 24.0
    out_label = "[vout]" if i == len(seg_defs) - 1 else f"[v0{i}]"
    xfade_parts.append(f"{prev_label}[{i}:v]xfade=transition=fade:duration={ov_s:.6f}:offset={off_s:.6f}{out_label}")
    prev_label = out_label

run_cmd([
    "ffmpeg", "-y",
    *ff_inputs,
    "-filter_complex", ";".join(xfade_parts),
    "-map", "[vout]", "-map", f"{len(seg_outputs)}:a:0",
    "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "256k", "-shortest",
    str(tmp_final_mp4),
])
shutil.copy2(tmp_final_mp4, final_mp4)
os.sync()
print(f"🎉 ALL {len(seg_defs)} SEGMENTS STITCHED & SYNCED TO DRIVE:\n  Raw Tessera 2K: {final_mp4} ({final_mp4.stat().st_size / 1e6:.2f} MB)", flush=True)
