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

# Pluggable hardware profiles (Default: g4_optimal for 96GB RTX PRO 6000 Blackwell)
PROFILES = {
    "g4_optimal": {
        "spatial_tile": "custom",
        "tile_width": 1536,
        "tile_height": 1088,
        "segment_frames": 56,
        "tile_select": "",
        "tile_limit": 0,
        "lms_lora_strength": 0.42,
        "guide_release": 0.55,
    },
    "g4_fast_face": {
        "spatial_tile": "custom",
        "tile_width": 1536,
        "tile_height": 1088,
        "segment_frames": 56,
        "tile_select": "0",
        "tile_limit": 1,
        "lms_lora_strength": 0.42,
        "guide_release": 0.55,
    },
    "local_32g": {
        "spatial_tile": "med_vram",
        "tile_width": 768,
        "tile_height": 768,
        "segment_frames": 28,
        "tile_select": "0",
        "tile_limit": 1,
        "lms_lora_strength": 0.42,
        "guide_release": 0.55,
    },
    "local_24g": {
        "spatial_tile": "low_vram",
        "tile_width": 640,
        "tile_height": 640,
        "segment_frames": 28,
        "tile_select": "0",
        "tile_limit": 1,
        "lms_lora_strength": 0.42,
        "guide_release": 0.55,
    },
}
ACTIVE_PROFILE_NAME = os.environ.get("TESSERA_PROFILE", "g4_optimal")
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--profile" and i + 2 < len(sys.argv):
        ACTIVE_PROFILE_NAME = sys.argv[i + 2]
ACTIVE_PROFILE = PROFILES[ACTIVE_PROFILE_NAME]
print(f"⚙️ Active Stage-2 Tessera Profile: {ACTIVE_PROFILE_NAME} -> {ACTIVE_PROFILE}", flush=True)


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

# 4. Stage inputs & slice V15_S1 (362f) into 3 exact 124-frame (17*7+5) segments with 5-frame overlap:
# Seg 1: frames 0..123 (0.000s - 5.1667s)
# Seg 2: frames 119..242 (4.9583s - 10.1250s)
# Seg 3: frames 238..361 (9.9167s - 15.0833s)
inp_dir = LOCAL_COMFY / "input"
inp_dir.mkdir(parents=True, exist_ok=True)
v15_src = DRIVE_COMFY / "output" / "20261031_DouyinDance_CatGirl_15s" / "V15_S1_OfficialBase" / "DouyinDance_RealGirlV3_15s_V15_S1_OfficialBase.mp4"
shutil.copy2(v15_src, inp_dir / "v15_s1_full.mp4")

for fn in ["ref_realgirl_v3_2k.png", "master_bgm_15s.wav"]:
    for cand in [
        DRIVE_COMFY / "input" / fn,
        Path("/content/drive/MyDrive/AI_Workstation/workflows/04_douyin_beauty_dance/inputs") / fn,
    ]:
        if cand.exists():
            shutil.copy2(cand, inp_dir / fn)
            break

seg_defs = [
    (1, 0, 124),
    (2, 119, 124),
    (3, 238, 124),
]
for sid, f_start, f_cnt in seg_defs:
    seg_mp4 = inp_dir / f"v15_s1_seg{sid}.mp4"
    t_ss = f_start / 24.0
    t_dur = f_cnt / 24.0
    run_cmd([
        "ffmpeg", "-y", "-i", str(inp_dir / "v15_s1_full.mp4"),
        "-ss", f"{t_ss:.6f}", "-t", f"{t_dur:.6f}",
        "-vf", f"fps=24,trim=start_frame=0:end_frame={f_cnt},setpts=PTS-STARTPTS",
        "-af", f"atrim=0:{t_dur:.6f},asetpts=PTS-STARTPTS",
        "-c:v", "libx264", "-crf", "12", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        str(seg_mp4),
    ])

# 5. Restart ComfyUI WITHOUT --highvram so Qwen3-VL 32B unloads cleanly after conditioning, leaving full 96GB VRAM for UNET + VAE
print("🔄 Starting ComfyUI on 127.0.0.1:8188 (--disable-mmap --fast)...", flush=True)
subprocess.run(["pkill", "-f", "python.*main.py"], check=False)
time.sleep(2)
log_path = Path("/tmp/comfyui_tessera.log")
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
    print(f"\n🎬 [Seg {sid}/3] TesseraRefine 2K (1536x2688, 124f) Queued! prompt_id={pid}", flush=True)
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
    print(f"✅ [Seg {sid}/3] Done in {time.time()-t0:.1f}s -> {out_p.name} ({out_p.stat().st_size/1e6:.2f} MB)", flush=True)
    return out_p


seg_outputs = []
cur_anchor = "ref_realgirl_v3_2k.png"
for sid, f_start, f_cnt in seg_defs:
    out_mp4 = run_segment_tessera(sid, f"v15_s1_seg{sid}.mp4", cur_anchor)
    seg_outputs.append(out_mp4)
    if sid < 3:
        # Extract frame 119 (the exact start frame of the next segment!) as the 2K anchor for the next segment
        next_anchor = inp_dir / f"anchor_seg{sid+1}.png"
        run_cmd([
            "ffmpeg", "-y", "-i", str(out_mp4),
            "-vf", "select=eq(n\\,119)", "-frames:v", "1",
            str(next_anchor),
        ])
        cur_anchor = next_anchor.name

# Stitch the 3 refined 2K (1536x2688) 124-frame segments with 5-frame (0.208333s) crossfades + lossless master_bgm_15s.wav
out_drv_dir = DRIVE_COMFY / "output" / "20261031_DouyinDance_CatGirl_15s" / "V15_S1_Tessera2K"
out_drv_dir.mkdir(parents=True, exist_ok=True)
final_mp4 = out_drv_dir / "DouyinDance_RealGirlV3_15s_V15_S1_Tessera2K_1536x2688.mp4"
bgm_wav = inp_dir / "master_bgm_15s.wav"

ov = 5.0 / 24.0  # 0.208333s
off1 = (124 - 5) / 24.0  # 4.958333s
off2 = (124 - 5 + 124 - 5) / 24.0  # 9.916667s

run_cmd([
    "ffmpeg", "-y",
    "-i", str(seg_outputs[0]),
    "-i", str(seg_outputs[1]),
    "-i", str(seg_outputs[2]),
    "-i", str(bgm_wav),
    "-filter_complex",
    f"[0:v][1:v]xfade=transition=fade:duration={ov:.6f}:offset={off1:.6f}[v01];"
    f"[v01][2:v]xfade=transition=fade:duration={ov:.6f}:offset={off2:.6f}[vout]",
    "-map", "[vout]", "-map", "3:a:0",
    "-c:v", "libx264", "-crf", "17", "-preset", "fast", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "192k", "-shortest",
    str(final_mp4),
])
os.sync()
print(f"🎉 ALL 3 SEGMENTS STITCHED & SYNCED TO DRIVE: {final_mp4} ({final_mp4.stat().st_size / 1e6:.2f} MB)", flush=True)
