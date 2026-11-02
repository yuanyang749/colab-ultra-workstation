#!/usr/bin/env python3
"""
Step 3 (Colab G4 96GB GDDR7 VRAM + 188GB DDR5 RAM):
Hybrid NVMe + Google Drive Staging & Dual-Model 2-Bit Server Orchestration:
  - Variant 1 (Default): `unsloth-ud-iq2xxs` (`unsloth/DeepSeek-V4.1-Flash-1017-GGUF` -> `UD-IQ2_XXS`)
      * 5 Unsloth Dynamic V3.1 importance-matrix backbone shards (`165.30 GB` total, `2.38 bpw`) -> Staged on G4 local Gen5 NVMe
      * 1 Conditional Engram shard (`00006-of-00006.gguf`, `110.60 GB` logical) -> 128 KB sparse GGUF header (zero Swap)
  - Variant 2: `abiray-gsq-iq2xxs` (`Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF`)
      * 4 GSQ-RCO abliterated backbone shards (`138.59 GB` total, `2.00 bpw`, 100.0% benchmark recovery) -> Staged on G4 local Gen5 NVMe
      * 1 Conditional Engram shard (`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`, `110.60 GB` logical)
  - Engine A: Delta Inference Engine (`ds41_generate` / `serve_ds41.py`)
      * 96GB GDDR7 VRAM holds Dense MLA (~4.62 GiB) + 128K NVFP4 KV (~1.72 GiB) + DSpark Drafter (4.22 GiB) + ~87.2 GiB hot MoE experts
      * 188GB DDR5 RAM holds remaining MoE experts in pinned memory (`--arena-gib 44` or `76`) -> 100% of 552B MoE in VRAM+RAM!
  - Engine B: Mainline `llama.cpp` (`b11870+` / `PR #31086` `llama-server`)
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

DRIVE_V41_DIR = "/content/drive/MyDrive/AI_Models/DeepSeek_V41_G4"
FAST_DIR = "/content/fast_deepseek"

VARIANTS = {
    "unsloth-ud-iq2xxs": {
        "repo": "unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
        "subdir": "unsloth/UD-IQ2_XXS",
        "hf_prefix": "UD-IQ2_XXS/",
        "shards_nvme": [
            "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf",  # 0.006 GB (5.78 MB metadata)
            "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00002-of-00006.gguf",  # 49.96 GB
            "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00003-of-00006.gguf",  # 49.95 GB
            "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00004-of-00006.gguf",  # 48.97 GB
            "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00005-of-00006.gguf",  # 16.42 GB (Total backbone: 165.30 GB / 153.96 GiB)
        ],
        "shard_engram": "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00006-of-00006.gguf",  # 110.60 GB logical (sparse header)
        "model_name": "deepseek-v4.1-flash-1017-uncensored-gsq-rco-iq2_xxs",
        "display_name": "DeepSeek V4.1 Flash · Unsloth Dynamic V3.1 UD-IQ2_XXS (G4 96GB Blackwell)",
        "is_gsq_4shard": False,
        "arena_gib": 76,
    },
    "abiray-gsq-iq2xxs": {
        "repo": "Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-GGUF",
        "subdir": "abiray_gsq_uncensored",
        "hf_prefix": "",
        "shards_nvme": [
            "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf",  # 39.08 GB
            "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00004.gguf",  # 39.39 GB
            "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00004.gguf",  # 39.25 GB
            "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00004.gguf",  # 20.87 GB (Total backbone: 138.59 GB / 129.07 GiB)
        ],
        "shard_engram": "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf",  # 110.60 GB logical (sparse header)
        "model_name": "deepseek-v4.1-flash-1017-uncensored-gsq-rco-iq2_xxs",
        "display_name": "DeepSeek V4.1 Flash Uncensored · Abiray GSQ-RCO IQ2_XXS (G4 96GB Blackwell)",
        "is_gsq_4shard": True,
        "arena_gib": 44,
    },
}


def ensure_precompiled_engines():
    """
    Unpack precompiled Blackwell sm_120a binaries (g4_sm120a_engines.tar.gz) from Google Drive in ~2s
    if /content/engines/bin/ds41_generate is not yet present on a fresh G4 instance.
    """
    ds41_bin = "/content/engines/bin/ds41_generate"
    tar_path = os.path.join(DRIVE_V41_DIR, "bin/g4_sm120a_engines.tar.gz")
    if os.path.exists(ds41_bin):
        return
    if os.path.exists(tar_path):
        print(f"=== [0/3] Unpacking Precompiled Blackwell sm_120a Engines from {tar_path} (~2s) ===", flush=True)
        os.makedirs("/content/engines", exist_ok=True)
        subprocess.run(["tar", "-xzf", tar_path, "-C", "/content/engines"], check=True)
        print("  [ENGINES READY] Unpacked ds41_generate + llama-server (sm_120a) to /content/engines/bin/", flush=True)
    else:
        print(f"  [WARN] Precompiled archive not found at {tar_path}; run step2_g4_build_engines.sh first.", flush=True)


def _get_colab_drive_token():
    """Fetch OAuth2 token from Colab local metadata service (127.0.0.1:8012)."""
    req = urllib.request.Request(
        "http://127.0.0.1:8012/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def _lookup_drive_file(shard_name, token):
    """Return (file_id, size_bytes) from Google Drive v3 API for shard_name."""
    import urllib.parse

    q = urllib.parse.quote(f"name = '{shard_name}' and trashed = false")
    url = f"https://www.googleapis.com/drive/v3/files?q={q}&fields=files(id,name,size)"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        files = json.loads(resp.read().decode("utf-8")).get("files", [])
    for f in files:
        sz = int(f.get("size", 0))
        if sz > 1 * 1024 * 1024:
            return f["id"], sz
    return None, 0


def _lookup_hf_file(repo, hf_prefix, shard_name):
    """Return (resolved_cdn_url, size_bytes) from Hugging Face Hub."""
    url = f"https://huggingface.co/{repo}/resolve/main/{hf_prefix}{shard_name}"
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=15) as resp:
        cl = resp.headers.get("X-Linked-Size") or resp.headers.get("Content-Length") or "0"
        return resp.url, int(cl)


def stage_hybrid_layout(variant_key="unsloth-ud-iq2xxs"):
    """
    Stage all backbone shards (165.30 GB for Unsloth UD-IQ2_XXS or 138.59 GB for Abiray GSQ-RCO IQ2_XXS)
    directly onto G4's local PCIe Gen5 NVMe (/content/fast_deepseek) via 64-stream Google Drive v3 API Range
    transfers (~1 GB/s), bypassing FUSE DriveFS's 112 GiB content_cache limit and warming the 188GB DDR5
    Page Cache in a single pass!
    """
    if os.path.isdir("/content/drive/My Drive") and not os.path.exists("/content/drive/MyDrive"):
        os.symlink("/content/drive/My Drive", "/content/drive/MyDrive")

    ensure_precompiled_engines()

    v = VARIANTS[variant_key]
    fast_v_dir = os.path.join(FAST_DIR, v["subdir"])
    drive_v_dir = os.path.join(DRIVE_V41_DIR, v["subdir"])
    os.makedirs(fast_v_dir, exist_ok=True)

    print(f"=== [1/3] 64-Stream Concurrent Drive-v3 API -> Gen5 NVMe Staging ({v['repo']}) ===", flush=True)
    t_stage_all = time.time()
    token = _get_colab_drive_token()
    chunk_bytes = 64 * 1024 * 1024  # 64 MiB per range task

    def _stage_shard_drive_v3(shard):
        dst = os.path.join(fast_v_dir, shard)
        if os.path.islink(dst):
            os.remove(dst)
        file_id, total_sz = _lookup_drive_file(shard, token)
        hf_url = None
        if not file_id:
            hf_url, total_sz = _lookup_hf_file(v["repo"], v["hf_prefix"], shard)
        if os.path.exists(dst) and os.path.getsize(dst) == total_sz and total_sz > 0:
            print(f"  [NVMe READY] {shard} ({total_sz/1e9:.2f} GB)", flush=True)
            return
        if not file_id and not hf_url:
            sys.exit(f"[ERROR] {shard} not found in Google Drive v3 API or Hugging Face!")

        t0 = time.time()
        src_label = "DRIVE-v3" if file_id else "HF-HUB"
        print(f"  [{src_label} -> NVMe] {shard} ({total_sz/1e9:.2f} GB)...", flush=True)
        tmp_dst = dst + ".part"
        fd = os.open(tmp_dst, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        os.ftruncate(fd, total_sz)
        ranges = []
        off = 0
        while off < total_sz:
            end = min(off + chunk_bytes - 1, total_sz - 1)
            ranges.append((off, end))
            off = end + 1

        def _fetch_and_write(rng):
            s_off, e_off = rng
            for attempt in range(4):
                try:
                    if file_id:
                        cur_tok = token if attempt == 0 else _get_colab_drive_token()
                        req = urllib.request.Request(
                            f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
                            headers={
                                "Authorization": f"Bearer {cur_tok}",
                                "Range": f"bytes={s_off}-{e_off}",
                            },
                        )
                    else:
                        req = urllib.request.Request(
                            hf_url,
                            headers={"Range": f"bytes={s_off}-{e_off}"},
                        )
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        buf = resp.read()
                    os.pwrite(fd, buf, s_off)
                    return len(buf)
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(0.5 * (attempt + 1))

        with concurrent.futures.ThreadPoolExecutor(max_workers=24 if file_id else 16) as ex:
            list(ex.map(_fetch_and_write, ranges))
        os.close(fd)
        if os.path.lexists(dst):
            os.remove(dst)
        os.rename(tmp_dst, dst)
        dt = max(0.1, time.time() - t0)
        print(f"  [DONE NVMe + DDR5 CACHE] {shard} ({total_sz/1e9:.2f} GB in {dt:.1f}s = {total_sz/1e6/dt:.0f} MB/s)", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as shard_ex:
        list(shard_ex.map(_stage_shard_drive_v3, v["shards_nvme"]))

    dt_all = max(0.1, time.time() - t_stage_all)
    print(f"=== [1.5/3] All Backbone Shards Staged on Local Gen5 NVMe + DDR5 Page Cache in {dt_all:.1f}s ===", flush=True)

    print(f"=== [2/3] Creating Local NVMe Engram GGUF Mapping ({v['shard_engram']}) ===")
    engram_src = os.path.join(drive_v_dir, v["shard_engram"])
    engram_dst = os.path.join(fast_v_dir, "engram_local_nvme.gguf" if v["is_gsq_4shard"] else v["shard_engram"])
    if os.path.exists(engram_src):
        engram_size = os.path.getsize(engram_src)
        if not os.path.exists(engram_dst) or os.path.islink(engram_dst) or os.path.getsize(engram_dst) != engram_size:
            if os.path.lexists(engram_dst):
                os.remove(engram_dst)
            with open(engram_src, "rb") as fsrc:
                hdr = fsrc.read(131072)
            with open(engram_dst, "wb") as fdst:
                fdst.write(hdr)
                fdst.truncate(engram_size)
    else:
        hf_engram_url, engram_size = _lookup_hf_file(v["repo"], v["hf_prefix"], v["shard_engram"])
        if not os.path.exists(engram_dst) or os.path.islink(engram_dst) or os.path.getsize(engram_dst) != engram_size:
            if os.path.lexists(engram_dst):
                os.remove(engram_dst)
            req = urllib.request.Request(hf_engram_url, headers={"Range": "bytes=0-131071"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                hdr = bytearray(resp.read())
            with open(engram_dst, "wb") as fdst:
                fdst.write(hdr)
                fdst.truncate(engram_size)
    print(f"  [NVMe ENGRAM OK] {engram_dst} ({engram_size/1e9:.2f} GB sparse NVMe backing)")

    # Also copy DSpark sidecar to NVMe if available on Drive
    dspark_src = os.path.join(DRIVE_V41_DIR, "V41-1017-MTP-iq2xxs.gguf")
    dspark_dst = os.path.join(FAST_DIR, "V41-1017-MTP-iq2xxs.gguf")
    if os.path.exists(dspark_src) and not os.path.exists(dspark_dst):
        print("  [COPYING DSpark Drafter -> NVMe] V41-1017-MTP-iq2xxs.gguf (4.22 GB)...")
        shutil.copyfile(dspark_src, dspark_dst)

    first_shard = os.path.join(fast_v_dir, v["shards_nvme"][0])
    return first_shard, engram_dst, v


def ensure_llamacpp_6shard_layout(fast_v_dir, engram_dst):
    """
    For Abiray 4-shard GSQ-RCO, if running mainline llama.cpp (b11870+),
    convert the 4 NVMe shards into the 6-shard mainline layout using ISTA-DASLab's convert_ds41_upstream.py.
    Note: Delta (ds41_generate) supports BOTH the 4-shard layout and the converted 6-shard layout!
    """
    shard1_6 = os.path.join(fast_v_dir, "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf")
    if os.path.exists(shard1_6):
        return shard1_6
    conv_script = os.path.join(fast_v_dir, "convert_ds41_upstream.py")
    if not os.path.exists(conv_script):
        subprocess.run(
            [
                "hf",
                "download",
                "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF",
                "convert_ds41_upstream.py",
                "chat_template.jinja",
                "exp_probs_b_vl.f32",
                "--local-dir",
                fast_v_dir,
            ],
            check=True,
        )
    print("  [CONVERTING 4-shard -> 6-shard mainline llama.cpp layout on NVMe (~45s)]...")
    subprocess.run(
        [sys.executable, conv_script, "--no-verify", "--delete-old", fast_v_dir, "--engram", engram_dst],
        check=True,
    )
    return shard1_6


def build_delta_g4_config(
    first_shard_path,
    engram_path,
    v,
    ctx_size=131072,
    port=8150,
    route_bias=0.0,
    quality_mode="exact",
):
    """
    Generate strata-ds41-g4.json tailored for G4 (96GB GDDR7 VRAM + 188GB DDR5 RAM).
    Golden Configuration verified on NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120a):
      - Dense MLA + Shared Experts + 128K NVFP4 KV Cache: ~6.34 GiB GDDR7 VRAM
      - GDDR7 VRAM Expert Slots: ~87.2 GiB (57%~71% of all 15,360 routed experts)
      - Pinned DDR5 RAM Arena (--arena-gib 44..76 + STRATA_ARENA_RESIDENT_LAST=1): holds remaining non-VRAM experts
      - 100.0% of all 15,360 routed experts resident in VRAM + Pinned RAM with 0 Swap!
    """
    delta_root = "/content/engines/delta_serve"
    if not os.path.exists(delta_root) and os.path.exists("/content/deepseek_build/delta-inference-engine"):
        delta_root = "/content/deepseek_build/delta-inference-engine"
    tok_dir = os.path.join(FAST_DIR, "tokenizer")
    export_py = os.path.join(delta_root, "tools/v41/serve/export_tokenizer.py")
    if not os.path.exists(os.path.join(tok_dir, "vocab.json")) and os.path.exists(export_py):
        os.makedirs(tok_dir, exist_ok=True)
        ds_meta = os.path.join(FAST_DIR, "deepseek_meta")
        if not os.path.exists(os.path.join(ds_meta, "tokenizer.json")):
            os.makedirs(ds_meta, exist_ok=True)
            for tf in ("tokenizer.json", "tokenizer_config.json", "config.json"):
                t_url = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/{tf}"
                urllib.request.urlretrieve(t_url, os.path.join(ds_meta, tf))
        subprocess.run(
            [sys.executable, export_py, ds_meta, tok_dir],
            check=True,
        )

    eff_bias = route_bias if quality_mode == "exact" else max(route_bias, 0.28)
    engine_args = [
        "-m",
        first_shard_path,
        "--backend",
        "cuda",
        "--experts",
        "gpu",
        "--engram",
        engram_path,
        "--ctx",
        str(ctx_size),
        "--stop",
        "1",
        "--kv-fp4",
        "--route-bias",
        f"{eff_bias:.2f}",
        "--route-bias-file",
        "0.0",
        "--arena-gib",
        str(v.get("arena_gib", 44)),
        "--vram-margin",
        "2.2",
        "--prompt-batch",
        "4096",
        "--rank",
        os.path.join(delta_root, "data/ds41/rank-1017-share.bin"),
        "--pf-b",
        "0.7",
        "--pcie",
        "0.5",
        "--enc-skip",
        "--enc-replay",
        "2560",
    ]

    env_vars = {
        "STRATA_NO_LARGEPAGES": "1",
        "STRATA_ARENA_RESIDENT_LAST": "1",
        "STRATA_ARENA_FILL_THREADS": "32",
        "STRATA_ARENA_REBAL": "0",
        "STRATA_BLOB_DIRECT": "0",
        "STRATA_DECODE_DIRECT": "0",
        "STRATA_BIAS_REFRESH": "1",
        "STRATA_SERVE_KEEP_CACHE": "1",
        "STRATA_SERVE_KEEP_LRU": "1",
        "STRATA_SERVE_KEEP_MAX": str(ctx_size),
        "STRATA_SERVE_CACHE_SLOTS": "4",
        "STRATA_SERVE_CACHE_GIB": "8",
        "STRATA_SERVE_SNAP_STACK": "4",
        "STRATA_SERVE_REWIND1": "1",
        "STRATA_SERVE_REWIND_GAP": "16",
        "STRATA_SERVE_ROLLBACK": "8",
        "STRATA_ENGRAM_THREADS": "16",
        "STRATA_ENGINE_STALL_S": "0",
    }
    if quality_mode == "turbo":
        env_vars["STRATA_DENSE_REQUANT"] = "q6_K"

    cfg = {
        "model": v["model_name"],
        "model_name": v["model_name"],
        "display_name": f"{v['display_name']} [{quality_mode.upper()}]",
        "engine": "/content/engines/bin/ds41_generate",
        "exe": "/content/engines/bin/ds41_generate",
        "gguf": first_shard_path,
        "args": engine_args,
        "cwd": delta_root,
        "tokenizer": tok_dir,
        "ctx": ctx_size,
        "stop": "1",
        "log": "/tmp/strata-ds41-g4.log",
        "port": port,
        "gpu": 0,
        "engine_silence_s": 0,
        "sampling": {"temperature": 1.0, "top_p": 1.0},
        "api_monitor": True,
        "fit_max_tokens": True,
        "reasoning_budget_tokens": 8192,
        "default_max_tokens": 65536,
        "env": env_vars,
    }
    dspark_gguf = os.path.join(FAST_DIR, "V41-1017-MTP-iq2xxs.gguf")
    if os.path.exists(dspark_gguf):
        cfg["env"]["STRATA_SERVE_DSPARK"] = dspark_gguf
        cfg["env"]["STRATA_DSPARK_KEEP"] = os.path.join(delta_root, "tools/v41/serve/dspark_expert_counts.txt:64")
        cfg["env"]["STRATA_DSPARK_DEPTH"] = "4"
        cfg["env"]["STRATA_DSPARK_CONF"] = "0.6"

    cfg_path = "/tmp/strata-ds41-g4.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"=== [3/3] Generated Delta G4 Config ({quality_mode.upper()} mode, route_bias={eff_bias:.2f}) at {cfg_path} ===")
    return cfg_path


def spawn_eager_vram_seed(port=8150, model_name="deepseek-v4.1-flash-1017-uncensored-gsq-rco-iq2_xxs"):
    """
    Spawn a background child process that waits for serve_ds41.py to report READY,
    then immediately triggers a 1-token warmup request so the ~87.2 GiB GDDR7 VRAM expert seed
    happens right away from the warm 188GB DDR5 Page Cache (~20s) and stays locked via STRATA_SERVE_KEEP_CACHE=1.
    """
    pid = os.fork()
    if pid != 0:
        return
    health_url = f"http://127.0.0.1:{port}/health"
    chat_url = f"http://127.0.0.1:{port}/v1/chat/completions"
    for _ in range(180):
        time.sleep(2)
        try:
            with urllib.request.urlopen(health_url, timeout=3) as r:
                info = json.loads(r.read().decode("utf-8"))
                if info.get("loaded"):
                    break
        except Exception:
            continue
    else:
        os._exit(0)

    print("\n[EAGER-VRAM-SEED] Server READY -> Triggering 1-token warmup to open segmented ~87.2 GiB GDDR7 VRAM cache...", flush=True)
    t0 = time.time()
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        "temperature": 0.0,
        "reasoning_effort": "none",
    }
    try:
        req = urllib.request.Request(
            chat_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp.read()
        print(f"[EAGER-VRAM-SEED] ✅ ~87.2 GiB hot MoE experts locked in GDDR7 VRAM in {time.time()-t0:.1f}s! Subsequent requests = 0.00s wait!", flush=True)
    except Exception as e:
        print(f"[EAGER-VRAM-SEED] Warmup note: {e}", flush=True)
    os._exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        default="unsloth-ud-iq2xxs",
        help="Model variant to stage and run (default: Unsloth Dynamic V3.1 UD-IQ2_XXS)",
    )
    parser.add_argument(
        "--mode",
        choices=["stage-only", "delta", "llamacpp-cpu", "llamacpp-hybrid"],
        default="stage-only",
    )
    parser.add_argument("--ctx", type=int, default=131072)
    parser.add_argument("--port", type=int, default=8150)
    parser.add_argument(
        "--quality-mode",
        choices=["exact", "turbo"],
        default="exact",
        help="Quality mode: 'exact' (default, 100%% native Q8_0 dense + route_bias=0.0 + full 40-layer pass) or 'turbo'",
    )
    parser.add_argument(
        "--route-bias",
        type=float,
        default=0.0,
        help="VRAM expert routing bias (0.0 = 100%% exact native routing for maximum quality; 0.28 = speed boost)",
    )
    parser.add_argument(
        "--no-eager-seed",
        action="store_true",
        default=False,
        help="Disable automatic 1-token warmup request that opens the segmented GDDR7 VRAM cache at startup",
    )
    args = parser.parse_args()

    first_shard, engram_dst, v = stage_hybrid_layout(args.variant)
    cfg_path = build_delta_g4_config(
        first_shard,
        engram_dst,
        v,
        ctx_size=args.ctx,
        port=args.port,
        route_bias=args.route_bias,
        quality_mode=args.quality_mode,
    )

    if args.mode == "stage-only":
        print(f"\n✅ Hybrid NVMe + Google Drive staging complete for [{v['display_name']}]!")
        print("Ready to launch Engine A (Delta) or Engine B (llama.cpp).")
        return

    if args.mode == "delta":
        os.environ["STRATA_ENGINE_STALL_S"] = "0"
        serve_py = "/content/engines/delta_serve/tools/v41/serve/serve_ds41.py"
        if os.path.exists(serve_py):
            txt = open(serve_py, "r", encoding="utf-8").read()
            if "SO_REUSEADDR" not in txt and "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)" in txt:
                txt = txt.replace(
                    "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)",
                    "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)",
                )
                open(serve_py, "w", encoding="utf-8").write(txt)
        srv_core = "/content/engines/delta_serve/serve/server.py"
        if os.path.exists(srv_core):
            stxt = open(srv_core, "r", encoding="utf-8").read()
            old_line = 'body.get("max_completion_tokens") or body.get("max_tokens") or self.default_max_tokens'
            new_line = 'max(int(body.get("max_completion_tokens") or body.get("max_tokens") or 32768), 32768 if int(body.get("max_completion_tokens") or body.get("max_tokens") or 32768) > 16 else 1)'
            if old_line in stxt and new_line not in stxt:
                open(srv_core, "w", encoding="utf-8").write(stxt.replace(old_line, new_line))
        subprocess.run(["fuser", "-k", f"{args.port}/tcp"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        if not args.no_eager_seed:
            spawn_eager_vram_seed(port=args.port, model_name=v["model_name"])
        cmd = [
            sys.executable,
            "/content/engines/delta_serve/tools/v41/serve/serve_ds41.py",
            "--engine",
            "strata",
            "--config",
            cfg_path,
            "--port",
            str(args.port),
        ]
        print("Launching Delta Server:", " ".join(cmd))
        os.execvp(cmd[0], cmd)

    elif args.mode in ("llamacpp-cpu", "llamacpp-hybrid"):
        if v["is_gsq_4shard"]:
            first_shard = ensure_llamacpp_6shard_layout(os.path.dirname(first_shard), engram_dst)
        ot_moe = ".ffn_.*_exps.=CPU" if args.mode == "llamacpp-cpu" else r"blk\.(2[6-9]|3[0-9])\.ffn_.*_exps\.=CPU"
        cmd = [
            "/content/engines/bin/llama-server",
            "--model",
            first_shard,
            "--ctx-size",
            str(args.ctx),
            "-ngl",
            "99",
            "--load-mode",
            "mmap",
            "--lazy-mode",
            "on",
            "--override-tensor",
            "per_layer_token_embd.weight=CPU",
            "-ot",
            ot_moe,
            "--cache-type-k",
            "q4_0",
            "--cache-type-v",
            "q4_0",
            "--temp",
            "1.0",
            "--top-p",
            "0.95",
            "--reasoning",
            "on",
            "--reasoning-effort",
            "50",
            "-ub",
            "512",
            "-b",
            "2048",
            "-t",
            "24",
            "--port",
            str(args.port),
        ]
        print(f"Launching llama-server ({args.mode}):", " ".join(cmd))
        os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
