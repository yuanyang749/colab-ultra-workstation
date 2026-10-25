#!/usr/bin/env python3
"""Zero-CCU CPU setup script for ComfyUI + MiniMax-H3 v2.1 Industrial Workflow (Scheme A + B + Qwen3-TTS + IndexTTS-2.5) on 20TB/30TB Google Drive.

Run via detached nohup on a 0.00 CCU CPU Colab session.
"""

import json
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
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


print("=== [Step 1/6] 安装 aria2 / unzip 与 huggingface_hub 高速下载引擎 ===")
run("apt-get update -qq && apt-get install -y -qq aria2 unzip")
run("pip install -q -U huggingface_hub hf_transfer")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print("\n=== [Step 2/6] 检查 20TB/30TB 云盘 ComfyUI 本体 ===")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
if not (COMFY_DIR / "main.py").exists():
    run(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git '{COMFY_DIR}'")
else:
    print(f"⚡ ComfyUI 本体已就绪 ({COMFY_DIR})，跳过 FUSE git pull。", flush=True)

print("\n=== [Step 3/6] 部署 v2.1 工业级双轨工作流与开源 TTS 核心插件 (12 Custom Nodes) ===")
CN_DIR = COMFY_DIR / "custom_nodes"
CN_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_NODES = {
    "ComfyUI-Manager": "https://github.com/ltdrdata/ComfyUI-Manager.git",
    "ComfyUI-KJNodes": "https://github.com/kijai/ComfyUI-KJNodes.git",
    "ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
    "ComfyUI-MiniMaxH3-Easy": "https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy.git",
    "ComfyUI-H3-Motion-Context": "https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context.git",
    "BUNNY_H3_Conditioning_Bridge": "https://github.com/aa335615543-ux/BUNNY_H3_Conditioning_Bridge.git",
    "Comfyui_Minimax_h3_latent_Upscaler": "https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git",
    "ComfyUI-MiniMaxH3_LatentUpscaler": "https://github.com/TripleHeadedMonkey/ComfyUI-MiniMaxH3_LatentUpscaler.git",
    "ComfyUI_MiniMaxH3_Director": "https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director.git",
    "ComfyUI-Impact-Pack": "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
    "ComfyUI-Qwen-TTS": "https://github.com/flybirdxx/ComfyUI-Qwen-TTS.git",
    "ComfyUI-Index-TTS": "https://github.com/chenpipi0807/ComfyUI-Index-TTS.git",
}

for name, repo_url in CUSTOM_NODES.items():
    target = CN_DIR / name
    if not (target / "__init__.py").exists() and not (target / "nodes.py").exists():
        # 先克隆到本地 /tmp 并剔除 .git 碎片对象，再移动到 DriveFS，速度提升 20 倍
        tmp_repo = TMP_DIR / f"repo_{name}"
        shutil.rmtree(tmp_repo, ignore_errors=True)
        rc = run(f"git clone --depth 1 {repo_url} '{tmp_repo}'")
        if rc == 0 and tmp_repo.exists():
            shutil.rmtree(tmp_repo / ".git", ignore_errors=True)
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(tmp_repo), str(target))
            print(f"✅ 插件克隆并落盘成功: {name}", flush=True)
    else:
        print(f"⚡ 插件已就绪: {name}", flush=True)

# 额外解压 speach1sdef178/MiniMax-H3-X2-Detail-VAE 附带的 ComfyUI-H3-X2-Detailed 节点
x2_node_dir = CN_DIR / "ComfyUI-H3-X2-Detailed"
if not x2_node_dir.exists():
    zip_tmp = TMP_DIR / "ComfyUI-H3-X2-Detailed.zip"
    run(
        f"aria2c -x 8 -s 8 --console-log-level=error -d '{TMP_DIR}' -o 'ComfyUI-H3-X2-Detailed.zip' "
        "'https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE/resolve/main/custom_node/ComfyUI-H3-X2-Detailed.zip'"
    )
    if zip_tmp.exists():
        run(f"unzip -qo '{zip_tmp}' -d '{CN_DIR}'")
        zip_tmp.unlink(missing_ok=True)
        print("✅ 插件解压成功: ComfyUI-H3-X2-Detailed", flush=True)

MODEL_SUBDIRS = (
    "diffusion_models",
    "text_encoders",
    "vae",
    "loras",
    "embeddings",
    "model_patches",
    "latent_upscale_models",
    "upscale_models",
    "qwen-tts",
    "IndexTTS-2.5",
)
for sub in MODEL_SUBDIRS:
    (COMFY_DIR / "models" / sub).mkdir(parents=True, exist_ok=True)
(CN_DIR / "BUNNY_H3_Conditioning_Bridge" / "models").mkdir(parents=True, exist_ok=True)

print("\n=== [Step 4/6] 高速拉取 MiniMax-H3 v2.1 工业级单文件模型权重 ===")

DOWNLOAD_MANIFEST = [
    # 1. 音视频双 VAE + X2-Detail 2倍细节直出 VAE
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors",
        COMFY_DIR / "models/vae/minimax_h3_audio_vae_fp32.safetensors",
    ),
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors",
        COMFY_DIR / "models/vae/minimax_h3_video_vae_fp16.safetensors",
    ),
    (
        "https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE/resolve/main/MiniMax-H3-X2-Detail-v1.safetensors",
        COMFY_DIR / "models/vae/MiniMax-H3-X2-Detail-v1.safetensors",
    ),
    # 2. Turbo 步数蒸馏加速 LoRA + 双写实去油光 LoRA
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        COMFY_DIR / "models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    ),
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        COMFY_DIR / "models/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    ),
    (
        "https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA/resolve/main/h3-realism-people-t2v-i2v-r2v.safetensors",
        COMFY_DIR / "models/loras/h3-realism-people-t2v-i2v-r2v.safetensors",
    ),
    (
        "https://huggingface.co/prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp/resolve/main/minimax-h3-facial-realism-closeup-cp2000.safetensors",
        COMFY_DIR / "models/loras/minimax-h3-facial-realism-closeup-cp2000.safetensors",
    ),
    # 3. 官方运镜 Embeddings
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
    # 4. 3D 潜空间放大权重 + 4x-UltraSharp
    (
        "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
        COMFY_DIR / "models/latent_upscale_models/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
    ),
    (
        "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
        COMFY_DIR / "models/upscale_models/4x-UltraSharp.pth",
    ),
    # 5. BUNNY H3 1024-d 多人动作与防串台语义桥 (V1 & V2)
    (
        "https://huggingface.co/JOKER141/BUNNY_H3_Conditioning_Bridge/resolve/main/BUNNY_H3_ActionLogic_Bridge_V1.safetensors",
        CN_DIR / "BUNNY_H3_Conditioning_Bridge/models/BUNNY_H3_ActionLogic_Bridge_V1.safetensors",
    ),
    (
        "https://huggingface.co/JOKER141/BUNNY_H3_Conditioning_Bridge/resolve/main/BUNNY_H3_ActionLogic_Bridge_V2.safetensors",
        CN_DIR / "BUNNY_H3_Conditioning_Bridge/models/BUNNY_H3_ActionLogic_Bridge_V2.safetensors",
    ),
    # 6. 无审核破限编码器 A: Blackwell G4 原生 NVFP4 极速破限版 (14.61 GB)
    (
        "https://huggingface.co/Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4/resolve/main/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
        COMFY_DIR / "models/text_encoders/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
    ),
    # 7. 无审核破限编码器 B: Ultra-Heretic INT8-ConvRot 满血破限版 (24.55 GB + 7.09 GB Tail)
    (
        "https://huggingface.co/OTMFLY/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot/resolve/main/qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors",
        COMFY_DIR / "models/text_encoders/qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors",
    ),
    (
        "https://huggingface.co/OTMFLY/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot/resolve/main/qwen3vl_32b_minimax_h3_generation_tail_50_63_int8_convrot.safetensors",
        COMFY_DIR / "models/text_encoders/qwen3vl_32b_minimax_h3_generation_tail_50_63_int8_convrot.safetensors",
    ),
    # 8. 主模型 A: MiniMax-H3 fl2va Pruned FP8 Scaled (19.52 GB)
    (
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        COMFY_DIR / "models/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    ),
    # 9. 主模型 B: Singularity v1.3 写实去油光 + 8步加速 + Ref2VA 融合底模 (19.53 GB)
    (
        "https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/resolve/main/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
        COMFY_DIR / "models/diffusion_models/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
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
    print(
        f"📦 下载完成 ({sz_gb:.2f} GB, {dl_sec:.1f}s, {sz_gb*1024/dl_sec:.0f} MB/s)，正在同步至 20TB 云盘: {dest} ...",
        flush=True,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_file), str(dest))
    print(f"✅ [{idx}/{len(DOWNLOAD_MANIFEST)}] 已写入挂载点: {dest.name} ({dest.stat().st_size / (1024**3):.2f} GB)", flush=True)

# 同步一份 BUNNY Bridge 权重到 models/model_patches 以防双路径扫描
for b_name in ("BUNNY_H3_ActionLogic_Bridge_V1.safetensors", "BUNNY_H3_ActionLogic_Bridge_V2.safetensors"):
    src_b = CN_DIR / "BUNNY_H3_Conditioning_Bridge/models" / b_name
    dst_b = COMFY_DIR / "models/model_patches" / b_name
    if src_b.exists() and not dst_b.exists():
        shutil.copy2(src_b, dst_b)

print("\n=== [Step 5/6] 高速拉取 2026 双王牌开源 TTS (Qwen3-TTS-1.7B + IndexTTS-2.5) 完整模型库 ===")
from huggingface_hub import hf_hub_download, snapshot_download

QWEN_TTS_REPOS = [
    ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", COMFY_DIR / "models/qwen-tts/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", COMFY_DIR / "models/qwen-tts/Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    ("Qwen/Qwen3-TTS-Tokenizer-12Hz", COMFY_DIR / "models/qwen-tts/Qwen/Qwen3-TTS-Tokenizer-12Hz"),
]

for repo_id, drive_target in QWEN_TTS_REPOS:
    if (drive_target / "model.safetensors").exists() and (drive_target / "model.safetensors").stat().st_size > 100_000_000:
        print(f"⚡ Qwen3-TTS 已存在于云盘，跳过: {repo_id}", flush=True)
        continue
    t0 = time.time()
    tmp_sub = TMP_DIR / repo_id.replace("/", "__")
    shutil.rmtree(tmp_sub, ignore_errors=True)
    print(f"⬇️ 正在高速拉取 {repo_id} 至本地 SSD...", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=str(tmp_sub))
    shutil.rmtree(tmp_sub / ".cache", ignore_errors=True)
    drive_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(drive_target, ignore_errors=True)
    shutil.move(str(tmp_sub), str(drive_target))
    print(f"✅ {repo_id} 已同步至云盘 ({time.time() - t0:.1f}s)", flush=True)

# IndexTTS-2.5 完整组件下载
INDEX_DIR = COMFY_DIR / "models/IndexTTS-2.5"
if (INDEX_DIR / "gpt.pth").exists() and (INDEX_DIR / "s2mel.pth").exists() and (INDEX_DIR / "bigvgan/bigvgan_generator.pt").exists():
    print("⚡ IndexTTS-2.5 已存在于云盘，跳过。", flush=True)
else:
    t0 = time.time()
    tmp_idx = TMP_DIR / "IndexTTS-2.5"
    shutil.rmtree(tmp_idx, ignore_errors=True)
    tmp_idx.mkdir(parents=True, exist_ok=True)
    print("⬇️ 正在高速拉取 IndexTTS-2.5 全套权重至本地 SSD...", flush=True)
    snapshot_download(
        repo_id="IndexTeam/IndexTTS-2.5",
        allow_patterns=[
            "config.yaml",
            "codec.pth",
            "feat1.pt",
            "feat2.pt",
            "gpt.pth",
            "s2mel.pth",
            "wav2vec2bert_stats.pt",
            "multilingual_zh_ja_yue_char_del.tiktoken",
            "qwen0.6bemo4-merge/*",
        ],
        local_dir=str(tmp_idx),
    )
    hf_hub_download(repo_id="funasr/campplus", filename="campplus_cn_common.bin", local_dir=str(tmp_idx))
    snapshot_download(
        repo_id="facebook/w2v-bert-2.0",
        allow_patterns=["config.json", "preprocessor_config.json", "conformer_shaw.pt", "model.safetensors"],
        local_dir=str(tmp_idx / "w2v-bert-2.0"),
    )
    snapshot_download(
        repo_id="nvidia/bigvgan_v2_22khz_80band_256x",
        allow_patterns=["config.json", "bigvgan_generator.pt"],
        local_dir=str(tmp_idx / "bigvgan"),
    )
    for cache_dir in tmp_idx.rglob(".cache"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    shutil.move(str(tmp_idx), str(INDEX_DIR))
    print(f"✅ IndexTTS-2.5 全套组件已同步至云盘 ({time.time() - t0:.1f}s)", flush=True)

print("\n=== [Step 6/6] 等待 DriveFS 将后台缓存 100% 刷入 Google Drive 真云端 ===")
expected_large = {dest.name for _, dest in DOWNLOAD_MANIFEST if dest.stat().st_size > 100000}
expected_large.update({"gpt.pth", "s2mel.pth", "bigvgan_generator.pt"})

while True:
    try:
        req_tok = urllib.request.Request(
            "http://127.0.0.1:8012/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        tok = json.loads(urllib.request.urlopen(req_tok, timeout=5).read().decode())["access_token"]
        params = urllib.parse.urlencode(
            {
                "q": 'trashed = false and (name contains ".safetensors" or name contains ".pth" or name contains ".pt")',
                "fields": "files(name,size)",
                "pageSize": 100,
            }
        )
        req_files = urllib.request.Request(
            f"https://www.googleapis.com/drive/v3/files?{params}",
            headers={"Authorization": f"Bearer {tok}"},
        )
        files = json.loads(urllib.request.urlopen(req_files, timeout=10).read().decode()).get("files", [])
        synced = {f["name"] for f in files if f["name"] in expected_large and int(f.get("size", 0)) > 100000}
        pending = sorted(expected_large - synced)
        print(
            f"☁️ 云端落盘校验: {len(synced)}/{len(expected_large)} 已同步至 Google Drive 云端 | 待刷完: {pending}",
            flush=True,
        )
        if not pending:
            break
    except Exception as e:
        print(f"⚠️ 校验重试: {e}", flush=True)
    time.sleep(10)

print("\n🎉🎉 全部 MiniMax-H3 v2.1 工业级双轨工作流 + Qwen3-TTS + IndexTTS-2.5 已 100% 落盘至 20TB Google Drive 云端！")
