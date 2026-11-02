#!/usr/bin/env python3
"""
Step 1 (Colab Free CPU Node - 0.00 CCU/h):
Download DeepSeek-V4.1-Flash-1017 (748B Total / 552B MoE + 196B Conditional Engram) 2-bit GGUF shards
and the DSpark MTP drafter directly into 20TB/30TB Google Drive (`/content/drive/MyDrive/AI_Models/DeepSeek_V41_G4`).

Supported Flagship 2-bit Variants:
  1. `unsloth-ud-iq2xxs` (`unsloth/DeepSeek-V4.1-Flash-1017-GGUF` -> `UD-IQ2_XXS`):
     - 5 MoE Backbone Shards (`00001-of-00006` .. `00005-of-00006`, 165.30 GB total, Unsloth Dynamic V3.1 importance-matrix protected)
     - 1 Conditional Engram Shard (`00006-of-00006`, 110.60 GB; defaults to `--engram-mode sparse-header` which writes only the 128 KB GGUF header in 1s!)
  2. `abiray-gsq-iq2xxs` (`Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF`):
     - 4 MoE Backbone Shards (`00001-of-00004` .. `00004-of-00004`, 138.59 GB total, GSQ-RCO 100.0% recovery + abliterated)
     - 1 Conditional Engram Shard (`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`, 110.60 GB)
  3. `both`:
     - Pre-stages both flagship 2-bit models onto Google Drive so you can hot-swap between them on G4 in ~22 seconds.
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request

DRIVE_V41_DIR = "/content/drive/MyDrive/AI_Models/DeepSeek_V41_G4"

VARIANTS = {
    "unsloth-ud-iq2xxs": {
        "repo": "unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
        "subdir": "unsloth",
        "hf_prefix": "UD-IQ2_XXS/",
        "backbone_shards": [
            "UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf",  # 5.78 MB metadata
            "UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00002-of-00006.gguf",  # 49.96 GB
            "UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00003-of-00006.gguf",  # 49.95 GB
            "UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00004-of-00006.gguf",  # 48.97 GB
            "UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00005-of-00006.gguf",  # 16.42 GB (Total: 165.30 GB)
        ],
        "engram_shard": "UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00006-of-00006.gguf",
    },
    "abiray-gsq-iq2xxs": {
        "repo": "Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF",
        "subdir": "abiray_gsq_uncensored",
        "hf_prefix": "",
        "backbone_shards": [
            "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf",  # 39.08 GB
            "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00004.gguf",  # 39.39 GB
            "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00004.gguf",  # 39.25 GB
            "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00004.gguf",  # 20.87 GB (Total: 138.59 GB)
        ],
        "engram_shard": "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf",
    },
}


def ensure_deps():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub[cli,hf_transfer]"],
        check=True,
    )
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def create_sparse_engram_header(repo: str, shard_rel_path: str, dst_dir: str):
    """Download only the first 128 KB GGUF header of the 110.6 GB Engram shard so Delta/llama.cpp can zero-fill or mmap without downloading 110.6 GB."""
    url = f"https://huggingface.co/{repo}/resolve/main/{shard_rel_path}"
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=20) as resp:
        cl = resp.headers.get("X-Linked-Size") or resp.headers.get("Content-Length") or "0"
        resolved_url = resp.url
        total_size = int(cl)

    dst_path = os.path.join(dst_dir, shard_rel_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    req_range = urllib.request.Request(resolved_url, headers={"Range": "bytes=0-131071"})
    with urllib.request.urlopen(req_range, timeout=20) as resp:
        hdr = resp.read()
    with open(dst_path, "wb") as f:
        f.write(hdr)
        f.truncate(total_size)
    print(f"  [SPARSE ENGRAM READY] {dst_path} (128 KB GGUF header written, logical size {total_size/1e9:.2f} GB)", flush=True)


def download_variant(variant_key: str, engram_mode: str = "sparse-header"):
    v = VARIANTS[variant_key]
    target_dir = os.path.join(DRIVE_V41_DIR, v["subdir"])
    os.makedirs(target_dir, exist_ok=True)

    print(f"\n=== Downloading [{variant_key}] ({v['repo']}) -> {target_dir} ===", flush=True)
    for idx, shard in enumerate(v["backbone_shards"], 1):
        dst_file = os.path.join(target_dir, shard)
        if os.path.exists(dst_file) and os.path.getsize(dst_file) > 1024 * 1024:
            print(f"  [{idx}/{len(v['backbone_shards'])}] [SKIP] Already exists: {shard} ({os.path.getsize(dst_file)/1e9:.2f} GB)", flush=True)
            continue
        t0 = time.time()
        print(f"  [{idx}/{len(v['backbone_shards'])}] Downloading {shard}...", flush=True)
        subprocess.run(
            [
                "hf",
                "download",
                v["repo"],
                shard,
                "--local-dir",
                target_dir,
            ],
            check=True,
        )
        sz_gb = os.path.getsize(dst_file) / 1e9 if os.path.exists(dst_file) else 0
        dt = max(0.1, time.time() - t0)
        print(f"  [DONE] {shard}: {sz_gb:.2f} GB in {dt:.1f}s ({sz_gb*1000/dt:.0f} MB/s)", flush=True)

    if engram_mode == "full":
        print(f"  [ENGRAM FULL] Downloading 110.6 GB Engram shard ({v['engram_shard']})...", flush=True)
        subprocess.run(
            [
                "hf",
                "download",
                v["repo"],
                v["engram_shard"],
                "--local-dir",
                target_dir,
            ],
            check=True,
        )
    else:
        create_sparse_engram_header(v["repo"], v["engram_shard"], target_dir)


def download_shared_sidecars():
    os.makedirs(DRIVE_V41_DIR, exist_ok=True)
    dspark_dst = os.path.join(DRIVE_V41_DIR, "V41-1017-MTP-iq2xxs.gguf")
    if not os.path.exists(dspark_dst) or os.path.getsize(dspark_dst) < 1e9:
        print("\n=== Downloading DSpark MTP Speculative Drafter (4.22 GB) ===", flush=True)
        subprocess.run(
            [
                "hf",
                "download",
                "CjangCjengh/DeepSeek-V4.1-Flash-1017-MTP-DSpark-GGUF",
                "V41-1017-MTP-iq2xxs.gguf",
                "--local-dir",
                DRIVE_V41_DIR,
            ],
            check=True,
        )
    else:
        print(f"\n[SKIP] DSpark MTP Drafter already present at {dspark_dst}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Step 1: Zero-CCU CPU Download of DeepSeek-V4.1-Flash 2-bit GGUF Shards to Google Drive")
    parser.add_argument(
        "--variant",
        choices=["unsloth-ud-iq2xxs", "abiray-gsq-iq2xxs", "both"],
        default="both",
        help="Which 2-bit model variant(s) to pre-stage onto Google Drive (default: both)",
    )
    parser.add_argument(
        "--engram-mode",
        choices=["sparse-header", "full"],
        default="sparse-header",
        help="'sparse-header' (default): writes 128 KB GGUF header for zero-Swap 552B MoE inference; 'full': downloads the 110.6 GB Engram table",
    )
    args = parser.parse_args()

    if os.path.isdir("/content/drive/My Drive") and not os.path.exists("/content/drive/MyDrive"):
        os.symlink("/content/drive/My Drive", "/content/drive/MyDrive")
    if not os.path.exists("/content/drive/MyDrive"):
        sys.exit("[ERROR] Google Drive is not mounted at /content/drive/MyDrive! Run `mcp_dfs_mount.py` first.")

    ensure_deps()
    keys = ["unsloth-ud-iq2xxs", "abiray-gsq-iq2xxs"] if args.variant == "both" else [args.variant]
    for k in keys:
        download_variant(k, engram_mode=args.engram_mode)
    download_shared_sidecars()
    print("\n✅ All requested DeepSeek-V4.1-Flash 2-bit shards and DSpark sidecar are staged on Google Drive!")


if __name__ == "__main__":
    main()
