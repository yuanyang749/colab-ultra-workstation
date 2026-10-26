#!/usr/bin/env python3
"""End-to-End G4 (96GB Blackwell) Runner for MiniMax-H3 v2.5 20-Second (4-Segment) Drama:
Version 5 (V5_ProPrompt_ZeroGlitch):
- Strictly adheres to Official MiniMax-H3 Full-Reference Mode (Ref2VA) Rewrite Guide (`ref-en.txt`) & NikoDemon80 Chaining Rules:
  1. Pure 2-Image Identity Reference (`<Picture 1>` -> `<Subject 1>`, `<Picture 2>` -> `<Subject 2>`), removing `<Picture 3>` to eliminate opening cut glitches.
  2. Explicit `<Audio 1>` definition in `subject_definitions` and `[reference generation + audio reference]` task-type prefix in `summary`.
  3. Consistent `The camera holds a static shot in a balanced medium two-shot` across all 4 clips + 1.10s Airlock hold at the head of Chained Clips 2-4.
  4. Seated upper-torso forward lean + palm press/strike (`remaining seated`, `comfortable headroom`) instead of `rises from chair`, preventing headless out-of-frame cropping.
  5. `MiniMaxH3JoinMatch` configured with `match_spread=False, match_detail=False, strength=0.35, min_excess=2.5` to eliminate contrast crushing & hazy/foggy VAE decoding.
"""

import copy
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
LOCAL_COMFY = Path("/content/ComfyUI")
API_URL = "http://127.0.0.1:8188"


def is_comfy_up() -> bool:
    try:
        urllib.request.urlopen(f"{API_URL}/system_stats", timeout=3)
        return True
    except Exception:
        return False


def ensure_comfy_running() -> None:
    if is_comfy_up():
        print("⚡ ComfyUI 已在 127.0.0.1:8188 运行中！", flush=True)
        return
    launcher = Path("/tmp/launch_g4_comfyui_fast.py")
    if not launcher.exists():
        launcher = Path("/content/drive/MyDrive/AI_Workstation/scripts/launch_g4_comfyui_fast.py")
    print("🚀 正在启动 launch_g4_comfyui_fast.py (v2.2 双通道 NVMe 极速常驻模式)...", flush=True)
    subprocess.run([sys.executable, "-u", str(launcher), "--profile", "scheme_a"], check=True)
    for _ in range(60):
        if is_comfy_up():
            return
        time.sleep(2)
    raise RuntimeError("❌ ComfyUI 启动超时，请检查 /tmp/comfyui.log")


def sync_inputs_to_local_ssd() -> None:
    src_in = DRIVE_COMFY / "input"
    dst_in = LOCAL_COMFY / "input"
    dst_in.mkdir(parents=True, exist_ok=True)
    required = (
        "ref_linwei_codex_real_1344x768.png",
        "ref_linwei_8s_profile_1344x768.png",
        "ref_chenmo_codex_real_1344x768.png",
        "native_linwei_seed_stereo.wav",
        "native_chenmo_seed_stereo.wav",
    )
    for fname in required:
        candidates = [Path("/tmp") / fname, src_in / fname]
        found = next((p for p in candidates if p.exists()), None)
        if not found:
            raise FileNotFoundError(f"❌ 缺少输入锚点文件: {fname}")
        d = dst_in / fname
        shutil.copy2(found, d)
        if src_in.exists() and found.resolve() != (src_in / fname).resolve():
            shutil.copy2(found, src_in / fname)
        print(f"✅ 输入锚点已就位本地 SSD: {fname} ({d.stat().st_size / 1024:.1f} KB)", flush=True)

    bunny_dir = LOCAL_COMFY / "custom_nodes/BUNNY_H3_Conditioning_Bridge/models"
    bunny_dir.mkdir(parents=True, exist_ok=True)
    for bname in ("BUNNY_H3_ActionLogic_Bridge_V1.safetensors", "BUNNY_H3_ActionLogic_Bridge_V2.safetensors"):
        patch_src = DRIVE_COMFY / "models/model_patches" / bname
        if patch_src.exists() and not (bunny_dir / bname).exists():
            shutil.copy2(patch_src, bunny_dir / bname)


def submit_and_wait(workflow: dict, label: str) -> dict:
    payload = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{API_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"❌ 提交工作流失败 ({label}): HTTP {e.code} -> {err_body}") from e

    if resp.get("node_errors"):
        raise RuntimeError(f"❌ 节点参数校验失败 ({label}): {json.dumps(resp['node_errors'], ensure_ascii=False)}")

    prompt_id = resp["prompt_id"]
    print(f"🎬 [{label}] 已提交队列 | prompt_id={prompt_id}", flush=True)
    t0 = time.time()
    last_log_len = 0

    while True:
        time.sleep(4)
        if Path("/tmp/comfyui.log").exists():
            log_txt = Path("/tmp/comfyui.log").read_text(errors="ignore")
            if len(log_txt) > last_log_len:
                new_lines = [
                    l.strip()
                    for l in log_txt[last_log_len:].splitlines()
                    if any(k in l for k in ["%", "it/s", "s/it", "Prompt executed", "Error", "Exception", "MiniMax", " join match", "BUNNY"])
                ]
                if new_lines:
                    print(f"  [{time.time()-t0:.0f}s] " + " | ".join(new_lines[-2:]), flush=True)
                last_log_len = len(log_txt)

        hist_raw = urllib.request.urlopen(f"{API_URL}/history/{prompt_id}", timeout=10).read().decode()
        hist = json.loads(hist_raw)
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"❌ 执行报错 ({label}): {json.dumps(status, ensure_ascii=False)}")
            print(f"✅ [{label}] 渲染完成！耗时: {time.time() - t0:.1f}s", flush=True)
            return entry


def stitch_four_segments(prefix: str, out_mp4_name: str) -> Path:
    out_dirs = [LOCAL_COMFY / "output", DRIVE_COMFY / "output"]
    segs = []
    for idx in (1, 2, 3, 4):
        seg_prefix = f"{prefix}_Seg0{idx}"
        candidates = []
        for od in out_dirs:
            if od.exists():
                candidates.extend(od.glob(f"{seg_prefix}*.mp4"))
        candidates.sort(key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise RuntimeError(f"❌ 未找到分镜输出文件: {seg_prefix}*.mp4")
        local_seg = Path(f"/tmp/{seg_prefix}.mp4")
        shutil.copy2(candidates[-1], local_seg)
        segs.append(local_seg)

    concat_list = Path(f"/tmp/concat_{prefix}.txt")
    concat_list.write_text("".join(f"file '{s}'\n" for s in segs))

    master_tmp = Path(f"/tmp/{out_mp4_name}")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            str(master_tmp),
        ],
        check=True,
    )
    out_drv = DRIVE_COMFY / "output" / out_mp4_name
    out_drv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(master_tmp, out_drv)
    print(f"🎬 20s 四段式超清母带合成并同步至 Google Drive 完毕: {out_drv} ({master_tmp.stat().st_size / (1024**2):.2f} MB)", flush=True)
    return master_tmp


def patch_bypass_bunny_if_needed(wf: dict, obj_info: dict) -> None:
    if "BunnyH3ConditioningBridge" not in obj_info:
        print("⚠️ BunnyH3ConditioningBridge 未注册，自动旁路直连 conditioning", flush=True)
        wf["26"]["inputs"]["conditioning"] = ["20", 0]
        wf["42"]["inputs"]["conditioning"] = ["40", 0]
        wf["62"]["inputs"]["conditioning"] = ["60", 0]
        wf["82"]["inputs"]["conditioning"] = ["80", 0]
        for nid in ("21", "41", "61", "81"):
            wf.pop(nid, None)


def inject_latent_checkpoints_and_slice(
    wf: dict,
    obj_info: dict,
    prefix: str,
    start_seg: int = 1,
    end_seg: int = 4,
) -> None:
    """Attach H3SaveAVLatent checkpoints (~5MB each on Drive) and optionally slice graph to run only [start_seg..end_seg]."""
    seg_latent_nodes = {1: "27", 2: "48", 3: "68", 4: "88"}
    seg_node_ranges = {
        1: range(20, 40),
        2: range(40, 60),
        3: range(60, 80),
        4: range(80, 100),
    }
    latent_dir = DRIVE_COMFY / "output" / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    has_ckpt_nodes = "H3SaveAVLatent" in obj_info and "H3LoadAVLatent" in obj_info
    if not has_ckpt_nodes:
        print("⚠️ H3SaveAVLatent/H3LoadAVLatent 未注册，执行完整四段图。", flush=True)
        return

    # 1. Attach H3SaveAVLatent to every active segment [start_seg..end_seg]
    for s in range(start_seg, end_seg + 1):
        lat_nid = seg_latent_nodes[s]
        if lat_nid in wf:
            save_nid = f"90{s}"
            wf[save_nid] = {
                "class_type": "H3SaveAVLatent",
                "inputs": {
                    "latent": [lat_nid, 0],
                    "checkpoint_name": f"{prefix}_Seg0{s}_av_latent.pt",
                },
            }

    # 2. If start_seg > 1, check if upstream latent checkpoint exists on Drive for 0.05s hot-resume
    if start_seg > 1:
        prev_s = start_seg - 1
        ckpt_name = f"{prefix}_Seg0{prev_s}_av_latent.pt"
        ckpt_path = latent_dir / ckpt_name
        if ckpt_path.exists():
            print(f"⚡ 命中云盘潜变量断点存档: {ckpt_path}，自动跳过前 {prev_s} 段 (耗时 0.05s)！", flush=True)
            for s_drop in range(1, start_seg):
                for nid_int in seg_node_ranges[s_drop]:
                    # Keep shared scheduler/sampler nodes 22, 23, 24!
                    if nid_int in (22, 23, 24):
                        continue
                    wf.pop(str(nid_int), None)
            prev_lat_nid = seg_latent_nodes[prev_s]
            wf[prev_lat_nid] = {
                "class_type": "H3LoadAVLatent",
                "inputs": {
                    "checkpoint_name": ckpt_name,
                },
            }
        else:
            print(f"⚠️ 未找到云盘断点 {ckpt_path}，将从第 1 段起算并首次生成全部断点存档。", flush=True)

    # 3. If end_seg < 4, prune downstream segments so we only render up to end_seg
    if end_seg < 4:
        for s_drop in range(end_seg + 1, 5):
            for nid_int in seg_node_ranges[s_drop]:
                wf.pop(str(nid_int), None)
            wf.pop(f"90{s_drop}", None)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="G4 MiniMax-H3 4-Segment Runner with 5MB Latent Checkpoint Hot-Swap")
    parser.add_argument("--workflow", default="/tmp/minimax_h3_v25_4seg_pro_prompt_workflow.json")
    parser.add_argument("--prefix", default="OfficeDuel_V7_8sLock")
    parser.add_argument("--out-mp4", default="MiniMax_H3_v25_20s_V7_8sLock.mp4")
    parser.add_argument("--only-seg", type=int, choices=[1, 2, 3, 4], default=None, help="Hot-swap only segment N (3 min)")
    parser.add_argument("--start-seg", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--end-seg", type=int, choices=[1, 2, 3, 4], default=4)
    args = parser.parse_args()

    start_s = args.only_seg if args.only_seg else args.start_seg
    end_s = args.only_seg if args.only_seg else args.end_seg

    t_start = time.time()
    ensure_comfy_running()
    sync_inputs_to_local_ssd()

    wf_path = Path(args.workflow)
    wf_v5 = copy.deepcopy(json.loads(wf_path.read_text(encoding="utf-8")))
    obj_info = json.loads(urllib.request.urlopen(f"{API_URL}/object_info", timeout=15).read().decode())
    patch_bypass_bunny_if_needed(wf_v5, obj_info)
    inject_latent_checkpoints_and_slice(wf_v5, obj_info, args.prefix, start_seg=start_s, end_seg=end_s)

    submit_and_wait(wf_v5, f"MiniMax-H3 Seg[{start_s}..{end_s}] ({args.prefix})")
    stitch_four_segments(args.prefix, args.out_mp4)

    print(f"🎉 ALL_V25_DONE (总用时 {time.time() - t_start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
