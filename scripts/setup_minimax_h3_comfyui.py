#!/usr/bin/env python3
"""Zero-CCU CPU setup script for ComfyUI + MiniMax-H3 Uncensored & Turbo Workstation on 20TB/30TB Google Drive.

Run via:
  colab exec -s h3-setup --timeout 3600 -f scripts/setup_minimax_h3_comfyui.py
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

if os.path.isdir("/content/drive/My Drive") and not os.path.exists("/content/drive/MyDrive"):
    os.symlink("/content/drive/My Drive", "/content/drive/MyDrive")

DRIVE_ROOT = Path("/content/drive/MyDrive/AI_Workstation")
COMFY_DIR = DRIVE_ROOT / "ComfyUI"
TMP_DIR = Path("/tmp/h3_dl")
TMP_DIR.mkdir(parents=True, exist_ok=True)


def run(cmd: str, cwd: str | None = None) -> int:
    print(f"🔧 $ {cmd}", flush=True)
    proc = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    if proc.stdout.strip():
        print(proc.stdout.strip()[-800:], flush=True)
    if proc.returncode != 0 and proc.stderr.strip():
        print(f"⚠️ STDERR:\n{proc.stderr.strip()[-800:]}", flush=True)
    return proc.returncode


print("=== [Step 1/4] 安装 aria2 高速下载引擎 ===")
run("apt-get update -qq && apt-get install -y -qq aria2")

print("\n=== [Step 2/4] 在 20TB/30TB 云盘部署 ComfyUI 本体 ===")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
if not (COMFY_DIR / "main.py").exists():
    run(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git '{COMFY_DIR}'")
else:
    print(f"✅ ComfyUI 已存在 ({COMFY_DIR})，正在拉取最新更新...")
    run("git pull", cwd=str(COMFY_DIR))

print("\n=== [Step 3/4] 部署核心画质与视频工作流插件 (Custom Nodes) ===")
CN_DIR = COMFY_DIR / "custom_nodes"
CN_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_NODES = {
    "ComfyUI-Manager": "https://github.com/ltdrdata/ComfyUI-Manager.git",
    "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes.git",
    "ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
}

for name, repo_url in CUSTOM_NODES.items():
    target = CN_DIR / name
    if not target.exists():
        run(f"git clone --depth 1 {repo_url} '{target}'")
    else:
        print(f"✅ 插件已就绪: {name}", flush=True)

MODEL_SUBDIRS = (
    "diffusion_models",
    "text_encoders",
    "vae",
    "loras",
    "embeddings",
    "model_patches",
    "upscale_models",
)
for sub in MODEL_SUBDIRS:
    (COMFY_DIR / "models" / sub).mkdir(parents=True, exist_ok=True)

print("\n=== [Step 4/4] 高速拉取 MiniMax-H3 主模型 + Ultra-Heretic 破限编码器 + Turbo 加速包 ===")

DOWNLOAD_MANIFEST = [
    # 1. 音视频双 VAE (Comfy-Org/MiniMax-H3)
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors",
        COMFY_DIR / "models/vae/minimax_h3_audio_vae_fp32.safetensors",
    ),
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors",
        COMFY_DIR / "models/vae/minimax_h3_video_vae_fp16.safetensors",
    ),
    # 2. Turbo 步数蒸馏加速 LoRA (8-Step 黄金版 & 4-Step 极速版)
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        COMFY_DIR / "models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    ),
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        COMFY_DIR / "models/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    ),
    # 3. 官方运镜与特效 Embeddings (Kiss Camera / Bullet Time 等 10 款)
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/embeddings/minimaxh3_kiss_camera.safetensors",
        COMFY_DIR / "models/embeddings/minimaxh3_kiss_camera.safetensors",
    ),
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/embeddings/minimaxh3_bullet_time.safetensors",
        COMFY_DIR / "models/embeddings/minimaxh3_bullet_time.safetensors",
    ),
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/embeddings/minimaxh3_truman_show.safetensors",
        COMFY_DIR / "models/embeddings/minimaxh3_truman_show.safetensors",
    ),
    # 4. 二阶段 2K 画质超分模型 (4x-UltraSharp)
    (
        "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
        COMFY_DIR / "models/upscale_models/4x-UltraSharp.pth",
    ),
    # 5. 无审核破限编码器 A: Blackwell G4 原生 NVFP4 极速破限版 (14.61 GB)
    (
        "https://huggingface.co/Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4/resolve/main/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
        COMFY_DIR / "models/text_encoders/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
    ),
    # 6. 无审核破限编码器 B: Ultra-Heretic INT8-ConvRot 满血破限版 (24.55 GB + 7.09 GB Tail)
    (
        "https://huggingface.co/OTMFLY/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot/resolve/main/qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors",
        COMFY_DIR / "models/text_encoders/qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors",
    ),
    (
        "https://huggingface.co/OTMFLY/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot/resolve/main/qwen3vl_32b_minimax_h3_generation_tail_50_63_int8_convrot.safetensors",
        COMFY_DIR / "models/text_encoders/qwen3vl_32b_minimax_h3_generation_tail_50_63_int8_convrot.safetensors",
    ),
    # 7. 主模型: MiniMax-H3 fl2va Pruned FP8 Scaled (19.52 GB, 官方最高画质首尾帧/图生/文生音视频)
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        COMFY_DIR / "models/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    ),
]

for idx, (url, dest) in enumerate(DOWNLOAD_MANIFEST, 1):
    if dest.exists() and dest.stat().st_size > 1024 * 100:
        sz_gb = dest.stat().st_size / (1024**3)
        print(f"⚡ [{idx}/{len(DOWNLOAD_MANIFEST)}] 已存在于云盘，跳过: {dest.name} ({sz_gb:.2f} GB)", flush=True)
        continue

    t0 = time.time()
    tmp_file = TMP_DIR / dest.name
    print(f"⬇️ [{idx}/{len(DOWNLOAD_MANIFEST)}] 正在高速下载: {dest.name} ...", flush=True)
    cmd = (
        f"aria2c -x 16 -s 16 -k 1M --file-allocation=none --continue=true "
        f"--console-log-level=error --summary-interval=0 "
        f"-d '{TMP_DIR}' -o '{dest.name}' '{url}'"
    )
    rc = run(cmd)
    if rc != 0 or not tmp_file.exists():
        raise RuntimeError(f"❌ 下载失败: {dest.name} (rc={rc})")

    sz_gb = tmp_file.stat().st_size / (1024**3)
    dl_sec = max(0.1, time.time() - t0)
    print(f"📦 下载完成 ({sz_gb:.2f} GB, {dl_sec:.1f}s, {sz_gb*1024/dl_sec:.0f} MB/s)，正在同步至 20TB 云盘: {dest} ...", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_file), str(dest))
    print(f"✅ [{idx}/{len(DOWNLOAD_MANIFEST)}] 已写入挂载点: {dest.name} ({dest.stat().st_size / (1024**3):.2f} GB)", flush=True)

print("\n=== [Step 5/5] 等待 DriveFS 将后台缓存 100% 刷入 Google Drive 真云端 ===")
import json, urllib.parse, urllib.request
expected_large = {dest.name for _, dest in DOWNLOAD_MANIFEST if dest.stat().st_size > 100000}

while True:
    try:
        req_tok = urllib.request.Request(
            "http://127.0.0.1:8012/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        tok = json.loads(urllib.request.urlopen(req_tok, timeout=5).read().decode())["access_token"]
        params = urllib.parse.urlencode({
            "q": 'trashed = false and (name contains ".safetensors" or name contains ".pth")',
            "fields": "files(name,size)",
            "pageSize": 50,
        })
        req_files = urllib.request.Request(
            f"https://www.googleapis.com/drive/v3/files?{params}",
            headers={"Authorization": f"Bearer {tok}"},
        )
        files = json.loads(urllib.request.urlopen(req_files, timeout=10).read().decode()).get("files", [])
        synced = {f["name"] for f in files if f["name"] in expected_large and int(f.get("size", 0)) > 100000}
        pending = sorted(expected_large - synced)
        print(f"☁️ 云端落盘校验: {len(synced)}/{len(expected_large)} 已同步至 Google Drive 云端 | 待刷完: {pending}", flush=True)
        if not pending:
            break
    except Exception as e:
        print(f"⚠️ 校验重试: {e}", flush=True)
    time.sleep(10)

print("\n🎉🎉 全部 MiniMax-H3 无审核破限 + Turbo 黄金底座已 100% 落盘至 20TB Google Drive 云端！")

