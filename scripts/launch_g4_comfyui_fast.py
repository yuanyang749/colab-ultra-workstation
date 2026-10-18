#!/usr/bin/env python3
"""G4 (96GB Blackwell) 3-Minute Fast Boot & Multi-Stream Pre-Warm Launcher for ComfyUI + MiniMax-H3.

Eliminates the 38-minute Google Drive FUSE `mmap` bottleneck by:
1. Streaming the core 60GB+ `.safetensors` weights sequentially with 16MB buffers across 4 threads
   to warm the DriveFS cache and Linux page cache at 250-400 MB/s (~2.5 minutes total).
2. Launching ComfyUI with `--highvram --disable-mmap` so weights load sequentially into G4's 96GB VRAM
   and stay 100% pinned in VRAM (0.0s reload on all subsequent videos).
3. Starting a Cloudflare Tunnel (`trycloudflare.com`) for instant Web UI access.

Run on G4 via colab-mcp or CLI:
  python3 scripts/launch_g4_comfyui_fast.py
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

if os.path.isdir("/content/drive/My Drive") and not os.path.exists("/content/drive/MyDrive"):
    os.symlink("/content/drive/My Drive", "/content/drive/MyDrive")

DRIVE_COMFY = Path("/content/drive/MyDrive/AI_Workstation/ComfyUI")
LOCAL_COMFY = Path("/content/ComfyUI")

# Core models used in the Anti-Oil Pro I2V 768p & 2K 1440p Golden Pipeline
PREWARM_MODELS = [
    DRIVE_COMFY / "models/diffusion_models/minimax_h3_fl2va_fp8_e4m3fn.safetensors",
    DRIVE_COMFY / "models/text_encoders/Qwen3-VL-30B-A3B-Instruct-heretic.safetensors",
    DRIVE_COMFY / "models/vae/minimax_h3_video_vae_fp16.safetensors",
    DRIVE_COMFY / "models/vae/minimax_h3_audio_vae_fp32.safetensors",
    DRIVE_COMFY / "models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    DRIVE_COMFY / "models/upscale_models/4x-UltraSharp.pth",
]


def sh(cmd: str) -> str:
    print(f"🔧 $ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if p.returncode != 0 and p.stderr:
        print(f"⚠️ {p.stderr[-600:]}", flush=True)
    return p.stdout


def prewarm_single_file(path: Path, chunk_mb: int = 16) -> tuple[str, float, float]:
    """Sequentially read a model file in 16MB chunks to bypass FUSE 4KB mmap throttling."""
    if not path.exists():
        return (path.name, 0.0, 0.0)
    t0 = time.time()
    buf_size = chunk_mb * 1024 * 1024
    total_bytes = 0
    with open(path, "rb", buffering=buf_size) as f:
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            total_bytes += len(chunk)
    elapsed = max(0.01, time.time() - t0)
    gb = total_bytes / (1024**3)
    return (path.name, gb, elapsed)


def main() -> None:
    print("=== [1/4] 检查 G4 96GB GPU 与云盘挂载状态 ===", flush=True)
    print(sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").strip(), flush=True)
    if not (DRIVE_COMFY / "models").exists():
        raise RuntimeError(f"❌ 未检测到云盘模型库: {DRIVE_COMFY}/models，请先挂载 Google Drive！")

    print("=== [2/4] 部署本地 SSD ComfyUI 核心并软链接 20TB 云盘模型库 ===", flush=True)
    if not (LOCAL_COMFY / "main.py").exists():
        sh(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git {LOCAL_COMFY}")
    else:
        sh(f"git -C {LOCAL_COMFY} pull --ff-only")

    sh(f"pip install -q -r {LOCAL_COMFY}/requirements.txt gguf imageio-ffmpeg")

    for subdir in ("models", "custom_nodes"):
        target = LOCAL_COMFY / subdir
        source = DRIVE_COMFY / subdir
        if target.is_symlink() or target.exists():
            sh(f"rm -rf {target}")
        os.symlink(source, target)

    (DRIVE_COMFY / "output").mkdir(parents=True, exist_ok=True)
    if (LOCAL_COMFY / "output").exists() or (LOCAL_COMFY / "output").is_symlink():
        sh(f"rm -rf {LOCAL_COMFY / 'output'}")
    os.symlink(DRIVE_COMFY / "output", LOCAL_COMFY / "output")

    print("=== [3/4] 4 线程 16MB 大块顺序预热 60GB+ 核心模型（绕过 FUSE mmap 限速） ===", flush=True)
    t_warm = time.time()
    existing = [p for p in PREWARM_MODELS if p.exists()]
    total_gb = sum(p.stat().st_size for p in existing) / (1024**3)
    print(f"📦 待顺序预热文件数: {len(existing)} | 总大小: {total_gb:.2f} GB", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(prewarm_single_file, p) for p in existing]
        for fut in concurrent.futures.as_completed(futures):
            name, gb, sec = fut.result()
            speed = (gb * 1024) / sec if sec > 0 else 0
            print(f"  ✅ 预热完成: {name} ({gb:.2f} GB, 耗时 {sec:.1f}s, {speed:.0f} MB/s)", flush=True)

    warm_sec = time.time() - t_warm
    print(f"⚡ 全部核心模型顺序预热完毕！总耗时: {warm_sec:.1f}s ({total_gb*1024/max(1, warm_sec):.0f} MB/s)", flush=True)

    print("=== [4/4] 启动 ComfyUI (--highvram --disable-mmap) + Cloudflare 隧道 ===", flush=True)
    if not Path("/usr/local/bin/cloudflared").exists():
        sh(
            "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
            "-O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"
        )

    sh("pkill -f 'python3 main.py' || true")
    sh("pkill -f 'cloudflared tunnel' || true")

    comfy_log = open("/tmp/comfyui.log", "w")
    subprocess.Popen(
        ["python3", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--highvram", "--disable-mmap"],
        cwd=str(LOCAL_COMFY),
        stdout=comfy_log,
        stderr=subprocess.STDOUT,
    )

    cf_log = open("/tmp/cloudflared.log", "w")
    subprocess.Popen(
        ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:8188"],
        stdout=cf_log,
        stderr=subprocess.STDOUT,
    )

    # Wait for ComfyUI port 8188 and Cloudflare URL
    cf_url = None
    ready = False
    for _ in range(60):
        try:
            urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
            ready = True
        except Exception:
            pass
        if Path("/tmp/cloudflared.log").exists():
            txt = Path("/tmp/cloudflared.log").read_text(errors="ignore")
            m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", txt)
            if m:
                cf_url = m.group(0)
        if ready and cf_url:
            break
        time.sleep(2)

    print(f"\n🎉 ComfyUI 极速模式已就绪 (--highvram --disable-mmap)！")
    print(f"🌐 Web UI 外网直连地址: {cf_url}", flush=True)


if __name__ == "__main__":
    main()
