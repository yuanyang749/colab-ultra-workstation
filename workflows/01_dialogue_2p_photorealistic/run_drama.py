#!/usr/bin/env python3
"""Beginner-Friendly & Fully Parameterized G4 (96GB Blackwell) Runner for MiniMax-H3 v2.5 Multi-Segment Drama:
Features:
1. Zero hardcoded filenames: Dynamically scans all `LoadImage` & `LoadAudio` nodes in the workflow JSON.
2. Beginner Config (`--config drama_config_example.json`) & CLI Overrides:
   - Swap character photos (`role1` / `role2`), native voice WAVs, speaking order (`speaker`), and Mandarin lines (`<d>[Mandarin]...</d>`) without editing the 1,000-line ComfyUI graph.
   - Switch between `"photorealistic"` (realism LoRAs enabled) and `"stylized"` / `"anime"` (realism LoRAs bypassed).
   - Adjust `bunny_alpha` (`0.10` for subtle facial dialogue vs `0.25+` for dynamic action/dance).
3. Smart Tail-Lock Clamp (`MiniMaxH3AddGuide`):
   - Automatically bypasses tail-lock nodes on a fresh first run if the tail-lock frame PNG does not exist yet.
   - Can dynamically inject a dual-frame `MiniMaxH3AddGuide` clamp (`frame_idx=[107, 136]`) into ANY segment via config or `--tail-lock-seg N --tail-lock-img <png>`.
4. 5MB Native `AV_LATENT` Checkpoint Hot-Swap (`--only-seg N` / `--start-seg` / `--end-seg`):
   - Re-renders a single segment in ~3 minutes (0.05s latent resume from Google Drive) without rerunning the whole video.
"""

import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

if os.path.isdir("/content/drive/My Drive") and not os.path.exists("/content/drive/MyDrive"):
    os.symlink("/content/drive/My Drive", "/content/drive/MyDrive")

DRIVE_COMFY = Path("/content/drive/MyDrive/AI_Workstation/ComfyUI")
LOCAL_COMFY = Path("/content/ComfyUI")
API_URL = "http://127.0.0.1:8188"

def get_default_seg_meta() -> Dict[int, Dict]:
    return {
        1: {"r2v": "20", "bunny": "21", "ctx": None, "noise": "25", "guider": "26", "sampler": "27", "out_lat": "27", "save": "35", "range": range(20, 40)},
        2: {"r2v": "40", "bunny": "41", "ctx": "42", "noise": "45", "guider": "46", "sampler": "47", "out_lat": "48", "save": "55", "range": range(40, 60)},
        3: {"r2v": "60", "bunny": "61", "ctx": "62", "noise": "65", "guider": "66", "sampler": "67", "out_lat": "68", "save": "75", "range": range(60, 80)},
        4: {"r2v": "80", "bunny": "81", "ctx": "82", "noise": "85", "guider": "86", "sampler": "87", "out_lat": "88", "save": "95", "range": range(80, 100)},
    }


SEG_META = get_default_seg_meta()


def ensure_segment_slots_in_workflow(wf: dict, max_seg_id: int) -> None:
    """Dynamically prune (<4) or clone (>4) segment slots in the workflow to match `max_seg_id`."""
    global SEG_META
    SEG_META = get_default_seg_meta()
    for s in list(SEG_META.keys()):
        if s > max_seg_id:
            for nid_int in SEG_META[s]["range"]:
                wf.pop(str(nid_int), None)
            wf.pop(str(900 + s), None)
            SEG_META.pop(s, None)
    for s in range(5, max_seg_id + 1):
        base = s * 20
        prev_out_lat = SEG_META[s - 1]["out_lat"]
        old_to_new = {str(80 + offset): str(base + offset) for offset in range(20)}
        for offset in range(20):
            src_nid = str(80 + offset)
            if src_nid in wf:
                dst_nid = str(base + offset)
                node_copy = copy.deepcopy(wf[src_nid])
                for k, v in list(node_copy.get("inputs", {}).items()):
                    if isinstance(v, list) and len(v) == 2:
                        ref_s = str(v[0])
                        if ref_s in old_to_new:
                            node_copy["inputs"][k] = [old_to_new[ref_s], v[1]]
                        elif ref_s in ("68", "67"):
                            node_copy["inputs"][k] = [prev_out_lat, v[1]]
                wf[dst_nid] = node_copy
        SEG_META[s] = {
            "r2v": str(base),
            "bunny": str(base + 1),
            "ctx": str(base + 2),
            "noise": str(base + 5),
            "guider": str(base + 6),
            "sampler": str(base + 7),
            "out_lat": str(base + 8),
            "save": str(base + 15),
            "range": range(base, base + 20),
        }


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


def find_asset_file(fname: str) -> Optional[Path]:
    src_in = DRIVE_COMFY / "input"
    dst_in = LOCAL_COMFY / "input"
    candidates = [Path("/tmp") / fname, dst_in / fname, src_in / fname, Path.cwd() / fname]
    return next((p for p in candidates if p.exists()), None)


def remove_seg3_default_taillock(wf: dict) -> None:
    """Unhook hardcoded nodes 661/662/663 from Segment 3 if present."""
    for nid in ("661", "662", "663"):
        wf.pop(nid, None)
    if "66" in wf and "62" in wf:
        wf["66"]["inputs"]["conditioning"] = ["62", 0]
    if "67" in wf and "60" in wf:
        wf["67"]["inputs"]["latent_image"] = ["60", 1]


def inject_segment_taillock(
    wf: dict,
    seg_id: int,
    image_name: str,
    frames: List[int] = [107, 136],
    strengths: Optional[List[float]] = None,
    keyframes: Optional[List[Dict]] = None,
) -> None:
    """Dynamically inject single-image or multi-image MiniMaxH3AddGuide keyframe sequence into segment `seg_id`."""
    meta = SEG_META[seg_id]
    r2v_nid = meta["r2v"]
    guider_nid = meta["guider"]
    sampler_nid = meta["sampler"]
    upstream_cond_nid = meta["ctx"] or meta["bunny"] or r2v_nid

    if r2v_nid not in wf or guider_nid not in wf or sampler_nid not in wf:
        return

    base_id = 600 + seg_id * 20
    if keyframes:
        kf_list = keyframes
    else:
        kf_list = [{"image": image_name, "frame_idx": int(f)} for f in frames]

    img_to_nid: Dict[str, str] = {}
    next_offset = 1
    for kf in kf_list:
        im_name = kf["image"]
        if im_name not in img_to_nid:
            img_nid = str(base_id + next_offset)
            next_offset += 1
            wf[img_nid] = {
                "class_type": "LoadImage",
                "inputs": {"image": im_name},
            }
            img_to_nid[im_name] = img_nid

    prev_cond = [upstream_cond_nid, 0]
    for kf in kf_list:
        g_nid = str(base_id + next_offset)
        next_offset += 1
        wf[g_nid] = {
            "class_type": "MiniMaxH3AddGuide",
            "inputs": {
                "positive": prev_cond,
                "vae": ["7", 0],
                "latent": [r2v_nid, 1],
                "image": [img_to_nid[kf["image"]], 0],
                "frame_idx": int(kf["frame_idx"]),
            },
        }
        prev_cond = [g_nid, 0]

    wf[guider_nid]["inputs"]["conditioning"] = prev_cond
    wf[sampler_nid]["inputs"]["latent_image"] = [r2v_nid, 1]
    print(f"🔒 已为第 {seg_id} 段挂载多关键帧潜空间引导 (AddGuide): {kf_list}", flush=True)


def apply_config_and_cli_overrides(wf: dict, args) -> str:
    """Apply beginner-friendly JSON config and CLI flags to the workflow dict."""
    cfg = {}
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            print(f"📋 已加载剧本配置文件: {cfg_path}", flush=True)

    prefix = args.prefix or cfg.get("project_prefix") or "OfficeDuel_V9_Final"

    # 1. Style mode & LoRA strengths (photorealistic vs stylized/anime)
    style_mode = args.style_mode or cfg.get("style_mode", "photorealistic")
    if style_mode in ("stylized", "anime", "none"):
        if "2" in wf:
            wf["2"]["inputs"]["strength_model"] = 0.0
        if "3" in wf:
            wf["3"]["inputs"]["strength_model"] = 0.0
        print(f"🎨 画风模式={style_mode}: 已自动关闭真人皮肤去油 LoRA (节点 2 & 3 权重设为 0)", flush=True)
    else:
        if "realism_lora_strength" in cfg and "2" in wf:
            wf["2"]["inputs"]["strength_model"] = float(cfg["realism_lora_strength"])
        if "closeup_lora_strength" in cfg and "3" in wf:
            wf["3"]["inputs"]["strength_model"] = float(cfg["closeup_lora_strength"])

    # 2. BUNNY ActionLogic Bridge alpha
    bunny_alpha = args.bunny_alpha if args.bunny_alpha is not None else cfg.get("bunny_alpha")
    if bunny_alpha is not None:
        for nid in ("21", "41", "61", "81"):
            if nid in wf:
                wf[nid]["inputs"]["alpha"] = float(bunny_alpha)

    # 3. Character Reference Images & Voice Seeds (supports role1, role2, role3)
    chars = cfg.get("characters", {})
    r1 = chars.get("role1", {})
    r2 = chars.get("role2", {})
    r3 = chars.get("role3", {})

    r1_img1 = args.role1_img or r1.get("image_seg1")
    r1_img_chain = args.role1_chained_img or r1.get("image_chained") or r1_img1
    r2_img = args.role2_img or r2.get("image_seg1")
    r3_img = r3.get("image_seg1")
    r1_voice = args.role1_voice or r1.get("voice_wav")
    r2_voice = args.role2_voice or r2.get("voice_wav")
    r3_voice = r3.get("voice_wav")

    if r1_img1 and "10" in wf:
        wf["10"]["inputs"]["image"] = r1_img1
    if r1_img_chain and "14" in wf:
        wf["14"]["inputs"]["image"] = r1_img_chain
    if r2_img and "11" in wf:
        wf["11"]["inputs"]["image"] = r2_img
    if r1_voice and "12" in wf:
        wf["12"]["inputs"]["audio"] = r1_voice
    if r2_voice and "13" in wf:
        wf["13"]["inputs"]["audio"] = r2_voice
    if r3_img:
        wf["15"] = {"class_type": "LoadImage", "inputs": {"image": r3_img}}
    if r3_voice:
        wf["16"] = {"class_type": "LoadAudio", "inputs": {"audio": r3_voice}}

    # 4. Reset hardcoded Seg 3 TailLock first, then re-apply from config/CLI if asset exists
    default_seg3_lock_img = wf.get("661", {}).get("inputs", {}).get("image")
    remove_seg3_default_taillock(wf)

    # 5. Per-segment overrides (speaker, dialogue line, custom prompt, seed, tail_lock, SaveVideo prefix)
    seg_cfgs = {int(s["seg_id"]): s for s in cfg.get("segments", []) if "seg_id" in s}
    max_cfg_seg = max(seg_cfgs.keys()) if seg_cfgs else 4
    ensure_segment_slots_in_workflow(wf, max_cfg_seg)
    if bunny_alpha is not None:
        for meta_b in SEG_META.values():
            if meta_b["bunny"] in wf:
                wf[meta_b["bunny"]]["inputs"]["alpha"] = float(bunny_alpha)
    cli_speakers = [s.strip() for s in args.speakers.split(",")] if args.speakers else []

    for seg_id, meta in SEG_META.items():
        scfg = seg_cfgs.get(seg_id, {})
        r2v_nid = meta["r2v"]
        noise_nid = meta["noise"]
        save_nid = meta["save"]

        out_seg_num = int(scfg.get("output_seg_id", seg_id))
        if save_nid in wf:
            wf[save_nid]["inputs"]["filename_prefix"] = f"{prefix}_Seg{out_seg_num:02d}"

        # Seed
        if "seed" in scfg and noise_nid in wf:
            wf[noise_nid]["inputs"]["noise_seed"] = int(scfg["seed"])

        # Speaker voice wiring ("role1" -> "12", "role2" -> "13", "role3" -> "16")
        spk = cli_speakers[seg_id - 1] if len(cli_speakers) >= seg_id else scfg.get("speaker")
        if spk and r2v_nid in wf:
            spk_str = str(spk).lower()
            if spk_str in ("1", "role1", "s1", "female", "left"):
                voice_node = "12"
            elif spk_str in ("3", "role3", "s3", "chairman", "boss") and "16" in wf:
                voice_node = "16"
            else:
                voice_node = "13"
            wf[r2v_nid]["inputs"]["ref_audios.ref_audio_0"] = [voice_node, 0]

        # Optional per-segment ref_images_custom override (direct filenames) or ref_roles override
        ref_imgs_custom = scfg.get("ref_images_custom")
        ref_roles = scfg.get("ref_roles")
        if ref_imgs_custom and r2v_nid in wf:
            wf[r2v_nid]["inputs"].pop("ref_images.ref_image_0", None)
            wf[r2v_nid]["inputs"].pop("ref_images.ref_image_1", None)
            for r_idx, img_name in enumerate(ref_imgs_custom[:2]):
                custom_load_nid = str(seg_id * 100 + 80 + r_idx)
                wf[custom_load_nid] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
                wf[r2v_nid]["inputs"][f"ref_images.ref_image_{r_idx}"] = [custom_load_nid, 0]
        elif ref_roles and r2v_nid in wf:
            role_to_img_node = {
                "role1": "14" if (seg_id > 1 and "14" in wf) else "10",
                "role1_front": "10",
                "role1_profile": "14" if "14" in wf else "10",
                "role2": "11",
                "role3": "15",
            }
            wf[r2v_nid]["inputs"].pop("ref_images.ref_image_0", None)
            wf[r2v_nid]["inputs"].pop("ref_images.ref_image_1", None)
            for r_idx, r_key in enumerate(ref_roles[:2]):
                img_node = role_to_img_node.get(r_key)
                if img_node and img_node in wf:
                    wf[r2v_nid]["inputs"][f"ref_images.ref_image_{r_idx}"] = [img_node, 0]

        # Prompt or Mandarin line replacement
        if r2v_nid in wf:
            if scfg.get("prompt"):
                wf[r2v_nid]["inputs"]["prompt"] = scfg["prompt"]
            elif scfg.get("line"):
                old_prompt = wf[r2v_nid]["inputs"]["prompt"]
                new_line = scfg["line"].strip()
                wf[r2v_nid]["inputs"]["prompt"] = re.sub(
                    r"<d>\[Mandarin\].*?</d>",
                    f"<d>[Mandarin]{new_line}</d>",
                    old_prompt,
                    count=1,
                )

        # Custom frame length override (must satisfy 17k + 5)
        if "length" in scfg and r2v_nid in wf:
            wf[r2v_nid]["inputs"]["length"] = int(scfg["length"])

        # Optional: disable MotionContext/JoinMatch for clean camera cuts (e.g. close-up cut in Seg 3 or prop cut in Seg 4)
        if scfg.get("use_motion_context") is False and seg_id > 1:
            ctx_nid = meta["ctx"]
            jm_nid = meta["out_lat"]
            guider_nid = meta["guider"]
            sampler_nid = meta["sampler"]
            sep_nid = str(seg_id * 20 + 10)   # "50", "70", "90"
            trim_nid = str(seg_id * 20 + 13)  # "53", "73", "93"
            if r2v_nid in wf:
                wf[r2v_nid]["inputs"]["length"] = int(scfg.get("length", 124))
            if guider_nid in wf:
                wf[guider_nid]["inputs"]["conditioning"] = [meta["bunny"], 0]
            if sep_nid in wf:
                wf[sep_nid]["inputs"]["av_latent"] = [sampler_nid, 0]
            if trim_nid in wf:
                wf[trim_nid]["inputs"]["trim_frames"] = 0
            if ctx_nid:
                wf.pop(ctx_nid, None)
            if jm_nid and jm_nid != sampler_nid:
                wf.pop(jm_nid, None)
                # Rewire any downstream MotionContext / JoinMatch nodes that pointed to jm_nid -> sampler_nid
                for _dnid, dnode in wf.items():
                    dinputs = dnode.get("inputs", {})
                    for dk, dv in list(dinputs.items()):
                        if isinstance(dv, list) and len(dv) == 2 and str(dv[0]) == str(jm_nid):
                            dinputs[dk] = [sampler_nid, dv[1]]
            meta["ctx"] = None
            meta["out_lat"] = sampler_nid
            print(f"🎬 第 {seg_id} 段已设为独立切镜模式 (use_motion_context=False, length={wf[r2v_nid]['inputs']['length']})", flush=True)

        # TailLock / Keyframe Guide injection (supports multi-image guide_keyframes or single-image tail_lock_image)
        guide_kfs = scfg.get("guide_keyframes")
        if guide_kfs:
            valid_kfs = [kf for kf in guide_kfs if find_asset_file(kf["image"]) is not None]
            if len(valid_kfs) == len(guide_kfs):
                inject_segment_taillock(wf, seg_id, valid_kfs[0]["image"], keyframes=valid_kfs)
            else:
                print(f"⚠️ 第 {seg_id} 段部分 guide_keyframes 图片未找到，跳过多关键帧引导。", flush=True)
        else:
            t_img = None
            t_frames = scfg.get("tail_lock_frames", [107, 136])
            t_strengths = scfg.get("tail_lock_strengths", [0.90, 0.95])
            if args.tail_lock_seg == seg_id and args.tail_lock_img:
                t_img = args.tail_lock_img
            elif "tail_lock_image" in scfg:
                t_img = scfg["tail_lock_image"]
            elif seg_id == 3 and default_seg3_lock_img and not cfg:
                t_img = default_seg3_lock_img

            if t_img:
                if find_asset_file(t_img) is not None:
                    inject_segment_taillock(wf, seg_id, t_img, t_frames, t_strengths)
                else:
                    print(
                        f"ℹ️ 第 {seg_id} 段未找到尾帧锁图片 {t_img}（首次跑片无需尾帧锁），已自动切换为标准无锁模式。",
                        flush=True,
                    )

    return prefix


def sync_inputs_to_local_ssd(wf: dict) -> None:
    """Dynamically scan all active LoadImage and LoadAudio nodes in `wf` and stage them to local NVMe SSD."""
    src_in = DRIVE_COMFY / "input"
    dst_in = LOCAL_COMFY / "input"
    dst_in.mkdir(parents=True, exist_ok=True)

    needed_files = []
    for nid, node in wf.items():
        ctype = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if ctype == "LoadImage" and isinstance(inputs.get("image"), str):
            needed_files.append(inputs["image"])
        elif ctype == "LoadAudio" and isinstance(inputs.get("audio"), str):
            needed_files.append(inputs["audio"])

    for fname in dict.fromkeys(needed_files):
        found = find_asset_file(fname)
        if not found:
            raise FileNotFoundError(
                f"❌ 缺少工作流所需的输入文件: {fname}\n"
                f"   请将该文件放到 /tmp/{fname} 或云盘 {src_in}/{fname}"
            )
        d = dst_in / fname
        if found.resolve() != d.resolve():
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


def safe_copy_to_drive_with_history(src: Path, dst_drv: Path) -> None:
    """Copy `src` to `dst_drv` on Google Drive while preserving any existing file as `_hist_NNN`."""
    dst_drv.parent.mkdir(parents=True, exist_ok=True)
    if dst_drv.exists() and src.resolve() != dst_drv.resolve():
        # Archive existing file to _hist_001, _hist_002, ...
        idx = 1
        while True:
            hist_p = dst_drv.with_name(f"{dst_drv.stem}_hist_{idx:03d}{dst_drv.suffix}")
            if not hist_p.exists():
                shutil.copy2(dst_drv, hist_p)
                print(f"📦 已自动归档云端历史版本: {hist_p.name}", flush=True)
                break
            idx += 1
    if src.resolve() != dst_drv.resolve():
        shutil.copy2(src, dst_drv)


def stitch_available_segments(
    prefix: str,
    out_mp4_name: str,
    expected_segs: List[int] = [1, 2, 3, 4],
    output_subdir: Optional[str] = None,
) -> Optional[Path]:
    drv_out_dir = (DRIVE_COMFY / "output" / output_subdir) if output_subdir else (DRIVE_COMFY / "output")
    drv_out_dir.mkdir(parents=True, exist_ok=True)
    out_dirs = [LOCAL_COMFY / "output", drv_out_dir, DRIVE_COMFY / "output"]
    segs = []
    for idx in expected_segs:
        seg_prefix = f"{prefix}_Seg{idx:02d}"
        candidates = []
        for od in out_dirs:
            if od.exists():
                candidates.extend([p for p in od.glob(f"{seg_prefix}*.mp4") if "_hist_" not in p.name])
        candidates.sort(key=lambda p: p.stat().st_mtime)
        if not candidates:
            print(f"ℹ️ 暂未找到分镜 {seg_prefix}*.mp4，跳过全片母带拼接。", flush=True)
            return None
        local_seg = Path(f"/tmp/{seg_prefix}.mp4")
        shutil.copy2(candidates[-1], local_seg)
        drv_seg = drv_out_dir / f"{seg_prefix}.mp4"
        safe_copy_to_drive_with_history(candidates[-1], drv_seg)
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
    out_drv = drv_out_dir / out_mp4_name
    safe_copy_to_drive_with_history(master_tmp, out_drv)
    print(f"🎬 多段超清母带合成并同步至 Google Drive 专属目录完毕: {out_drv} ({master_tmp.stat().st_size / (1024**2):.2f} MB)", flush=True)
    return master_tmp


def patch_bypass_bunny_if_needed(wf: dict, obj_info: dict) -> None:
    if "BunnyH3ConditioningBridge" not in obj_info:
        print("⚠️ BunnyH3ConditioningBridge 未注册，自动旁路直连 conditioning", flush=True)
        bunny_to_r2v = {m["bunny"]: m["r2v"] for m in SEG_META.values()}
        for nid, node in wf.items():
            inputs = node.get("inputs", {})
            for k, v in list(inputs.items()):
                if isinstance(v, list) and len(v) == 2 and str(v[0]) in bunny_to_r2v:
                    inputs[k] = [bunny_to_r2v[str(v[0])], v[1]]
        for b_nid in bunny_to_r2v:
            wf.pop(b_nid, None)


def inject_latent_checkpoints_and_slice(
    wf: dict,
    obj_info: dict,
    prefix: str,
    start_seg: int = 1,
    end_seg: int = 4,
) -> None:
    """Attach H3SaveAVLatent checkpoints (~5MB each on Drive) and optionally slice graph to run only [start_seg..end_seg]."""
    max_s = max(SEG_META.keys())
    for old_save in [f"90{i}" for i in range(1, max_s + 1)]:
        wf.pop(old_save, None)
    seg_latent_nodes = {s: m["out_lat"] for s, m in SEG_META.items()}
    seg_node_ranges = {s: list(m["range"]) + [600 + s * 20 + i for i in range(1, 20)] for s, m in SEG_META.items()}
    latent_dir = DRIVE_COMFY / "output" / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    has_ckpt_nodes = "H3SaveAVLatent" in obj_info and "H3LoadAVLatent" in obj_info
    if has_ckpt_nodes:
        for s in range(start_seg, end_seg + 1):
            lat_nid = seg_latent_nodes[s]
            if lat_nid in wf:
                save_nid = f"90{s}"
                wf[save_nid] = {
                    "class_type": "H3SaveAVLatent",
                    "inputs": {
                        "latent": [lat_nid, 0],
                        "checkpoint_name": f"{prefix}_Seg{s:02d}_av_latent.pt",
                    },
                }
    else:
        print("⚠️ H3SaveAVLatent/H3LoadAVLatent 未注册，跳过潜变量断点落盘。", flush=True)

    if start_seg > 1:
        needs_prev_latent = SEG_META[start_seg]["ctx"] is not None
        prev_s = start_seg - 1
        ckpt_name = f"{prefix}_Seg{prev_s:02d}_av_latent.pt"
        ckpt_path = latent_dir / ckpt_name
        if not needs_prev_latent:
            print(f"⚡ 第 {start_seg} 段为独立切镜模式 (无上游潜空间依赖)，直接剥离前 {prev_s} 段独立极速渲染！", flush=True)
            for s_drop in range(1, start_seg):
                for nid_int in seg_node_ranges[s_drop]:
                    if nid_int in (22, 23, 24):
                        continue
                    wf.pop(str(nid_int), None)
                wf.pop(f"90{s_drop}", None)
        elif has_ckpt_nodes and ckpt_path.exists():
            print(f"⚡ 命中云盘潜变量断点存档: {ckpt_path}，自动跳过前 {prev_s} 段 (耗时 0.05s)！", flush=True)
            for s_drop in range(1, start_seg):
                for nid_int in seg_node_ranges[s_drop]:
                    if nid_int in (22, 23, 24):
                        continue
                    wf.pop(str(nid_int), None)
                wf.pop(f"90{s_drop}", None)
            prev_lat_nid = seg_latent_nodes[prev_s]
            wf[prev_lat_nid] = {
                "class_type": "H3LoadAVLatent",
                "inputs": {
                    "checkpoint_name": ckpt_name,
                },
            }
        else:
            print(f"⚠️ 未找到云盘断点 {ckpt_path}，将从第 1 段起算并首次生成全部断点存档。", flush=True)

    if end_seg < max_s:
        for s_drop in range(end_seg + 1, max_s + 1):
            for nid_int in seg_node_ranges[s_drop]:
                wf.pop(str(nid_int), None)
            wf.pop(f"90{s_drop}", None)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Beginner-Friendly G4 MiniMax-H3 Multi-Segment Runner with 5MB Latent Hot-Swap")
    parser.add_argument("--config", default=None, help="Path to beginner drama_config.json (overrides characters, lines, speakers, tail-locks)")
    parser.add_argument("--workflow", default="/tmp/minimax_h3_v25_4seg_v9_final_workflow.json")
    parser.add_argument("--prefix", default=None, help="Output filename prefix (e.g. OfficeDuel_V9_Final)")
    parser.add_argument("--out-mp4", default=None, help="Stitched master MP4 filename")
    parser.add_argument("--style-mode", choices=["photorealistic", "stylized", "anime", "none"], default=None)
    parser.add_argument("--bunny-alpha", type=float, default=None, help="BUNNY bridge alpha (0.10 for dialogue, 0.25 for action)")
    parser.add_argument("--role1-img", default=None, help="Override Role 1 (left character) reference image")
    parser.add_argument("--role1-chained-img", default=None, help="Override Role 1 chained profile image for Seg 2..4")
    parser.add_argument("--role2-img", default=None, help="Override Role 2 (right character) reference image")
    parser.add_argument("--role1-voice", default=None, help="Override Role 1 reference voice WAV")
    parser.add_argument("--role2-voice", default=None, help="Override Role 2 reference voice WAV")
    parser.add_argument("--speakers", default=None, help="Comma-separated speaking order per segment, e.g. '1,2,1,2' or '2,1,2,1'")
    parser.add_argument("--tail-lock-seg", type=int, default=None, help="Apply AddGuide tail-lock clamp to segment N")
    parser.add_argument("--tail-lock-img", default=None, help="Anchor frame PNG filename for --tail-lock-seg")
    parser.add_argument("--only-seg", type=int, default=None, help="Hot-swap only segment N (~3 min)")
    parser.add_argument("--start-seg", type=int, default=1)
    parser.add_argument("--end-seg", type=int, default=None)
    args = parser.parse_args()

    t_start = time.time()
    ensure_comfy_running()

    wf_path = Path(args.workflow)
    if not wf_path.exists():
        alt = Path("/tmp/minimax_h3_v25_4seg_pro_prompt_workflow.json")
        if alt.exists():
            wf_path = alt
    wf = copy.deepcopy(json.loads(wf_path.read_text(encoding="utf-8")))

    prefix = apply_config_and_cli_overrides(wf, args)
    max_s = max(SEG_META.keys())
    start_s = args.only_seg if args.only_seg else args.start_seg
    end_s = args.only_seg if args.only_seg else (args.end_seg or max_s)
    cfg_dict = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config and Path(args.config).exists() else {}
    out_mp4 = args.out_mp4 or cfg_dict.get("output_mp4") or f"{prefix}_Master.mp4"
    expected_segs = cfg_dict.get("expected_segs", [1, 2, 3, 4])
    output_subdir = cfg_dict.get("output_subdir")

    obj_info = json.loads(urllib.request.urlopen(f"{API_URL}/object_info", timeout=15).read().decode())
    patch_bypass_bunny_if_needed(wf, obj_info)
    inject_latent_checkpoints_and_slice(wf, obj_info, prefix, start_seg=start_s, end_seg=end_s)

    # Stage only the active LoadImage / LoadAudio files required by the sliced graph
    sync_inputs_to_local_ssd(wf)

    submit_and_wait(wf, f"MiniMax-H3 Seg[{start_s}..{end_s}] ({prefix})")
    stitch_available_segments(prefix, out_mp4, expected_segs=expected_segs, output_subdir=output_subdir)

    print(f"🎉 ALL_V25_DONE (总用时 {time.time() - t_start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
