#!/usr/bin/env python3
"""G4 (96GB Blackwell) Runner for Workflow 04: 9:16 Vertical Douyin Beauty Dance (MiniMax-H3 v2.5)
Features:
1. Native 9:16 Vertical Resolution (`768x1344` @ 24fps) with configurable `width` & `height`.
2. Phase-Aligned Per-Segment BGM (`audio_wav` per segment) + Automatic Lossless Master BGM Layback (`master_bgm_wav`).
3. Action-Chain & 1:1 Camera Sync Prompting + Multi-Point Keyframe Guide (`guide_keyframes` / `MiniMaxH3AddGuide`).
4. 5MB Native `AV_LATENT` Checkpoint Hot-Swap (`--only-seg N` / `--start-seg` / `--end-seg`).
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
    }


SEG_META = get_default_seg_meta()


def ensure_segment_slots_in_workflow(wf: dict, max_seg_id: int) -> None:
    """Dynamically prune or clone segment slots in the workflow to match `max_seg_id`."""
    global SEG_META
    SEG_META = get_default_seg_meta()
    for s in list(SEG_META.keys()):
        if s > max_seg_id:
            for nid_int in SEG_META[s]["range"]:
                wf.pop(str(nid_int), None)
            wf.pop(str(900 + s), None)
            SEG_META.pop(s, None)
    for s in range(3, max_seg_id + 1):
        base = s * 20
        prev_out_lat = SEG_META[s - 1]["out_lat"]
        old_to_new = {str(40 + offset): str(base + offset) for offset in range(20)}
        for offset in range(20):
            src_nid = str(40 + offset)
            if src_nid in wf:
                dst_nid = str(base + offset)
                node_copy = copy.deepcopy(wf[src_nid])
                for k, v in list(node_copy.get("inputs", {}).items()):
                    if isinstance(v, list) and len(v) == 2:
                        ref_s = str(v[0])
                        if ref_s in old_to_new:
                            node_copy["inputs"][k] = [old_to_new[ref_s], v[1]]
                        elif ref_s in ("27", "28"):
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
    p_direct = Path(fname)
    if p_direct.is_absolute() and p_direct.exists():
        return p_direct
    src_in = DRIVE_COMFY / "input"
    dst_in = LOCAL_COMFY / "input"
    script_dir = Path(globals().get("__file__", "/tmp/run_dance.py")).resolve().parent
    candidates = [
        Path("/tmp") / fname,
        dst_in / fname,
        src_in / fname,
        Path.cwd() / fname,
        script_dir / fname,
        script_dir / "open_dance_bgm_kit" / "tracks" / fname,
        script_dir / "bgm_library" / fname,
        script_dir / "sample_inputs" / fname,
        script_dir / "inputs" / fname,
        script_dir.parents[2] / "local_debug_workspace" / "01_douyin_dance_catgirl_15s" / "inputs" / fname,
    ]
    return next((p for p in candidates if p.exists()), None)


def build_middle_segment_prompt(seg_id: int, total_segs: int, seg_dur_s: float, bpm: int, beat_s: float, onsets_local: List[float]) -> str:
    """Generate a universal Zero-Appearance middle-segment prompt (for 20s/30s+ multi-segment dances) synced to local audio onsets."""
    def pick(target: float, lo: float, hi: float) -> str:
        cands = [t for t in onsets_local if lo <= t <= hi]
        val = min(cands, key=lambda x: abs(x - target)) if cands else min(seg_dur_s - 0.1, target)
        return f"00:0{val:05.3f}"

    acc_str = ", ".join(f"{t:.2f}s" for t in onsets_local[:12]) if onsets_local else "every musical beat"
    w1 = pick(2.40, 2.00, 2.80)
    a1 = pick(1.35, 1.05, 1.65)
    a2 = pick(1.85, 1.65, 2.20)
    w2 = pick(4.80, 4.30, 5.20)
    a3 = pick(3.10, 2.80, 3.50)
    a4 = pick(4.05, 3.60, 4.45)
    a5 = pick(5.50, 5.10, 5.95)
    a6 = pick(6.40, 6.00, 6.80)
    end_s = f"00:0{seg_dur_s:05.3f}"

    return f"""For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced for identity, skin tone, outfit, and environment.

subject_definitions:
<Subject 1> is the exact same performer shown in <Picture 1>. <Picture 1> is the sole and absolute visual authority for <Subject 1>'s facial identity, warm natural skin tone, hairstyle, body proportions, outfit, footwear, and the surrounding physical environment — nothing about appearance, clothing, or background is redefined in words.
<Audio 1> is the target musical audio track for Segment {seg_id:02d} of {total_segs:02d} (detected tempo ~{bpm} BPM, beat interval ~{beat_s:.2f}s; key musical accents at {acc_str}).

summary:
[reference generation + audio reuse] Seamlessly continuing the unbroken vertical 9:16 live-action dance long take into Segment {seg_id:02d}, <Subject 1> flows from the incoming motion bridge into energetic diagonal arm reaches with playful heel kicks, side-to-side hip-pop pendulum sweeps, inward wrist rolls, and rhythmic chest-level accents synchronized to <Audio 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - Strictly preserve 100% of the exact facial identity, natural skin tone, hairstyle, outfit, footwear, and background environment from <Picture 1>. Do NOT reset to a static pose at 0.000s; continue seamlessly from the incoming motion context.
<Audio 1>: fully_copy - Reused 1:1 as the complete final audio track; synchronize every bounce and dance accent directly to <Audio 1>.

detailed_description:
r34l1sm, Facial Realism, vertical 9:16 live-action smartphone dance video matching the exact lighting, environment, outfit, and subject identity of <Picture 1>, consistent natural skin tone throughout, crisp 1/200s shutter motion clarity, authentic skin texture, natural fabric and hair physics.
[Shot 1] SUSTAINED LAYER (continuous through 00:00.000–{end_s}): <Subject 1> maintains the elastic knee-bounce groove on every musical beat of <Audio 1> (~{beat_s:.2f}s cycle, Chain A1 + B1 engine), keeping the full head and full body clearly inside the vertical frame with an unchanging sweet natural smile.
00:00.000–00:00.917 (Seamless Motion Bridge): Holding the exact incoming full-body framing and body momentum from the end of the previous segment, <Subject 1> completes a soft knee rebound on the beat.
00:00.917–{w1} (CHAIN K45 + A7 — Diagonal Reach & Side Point Hip-Pop): During 00:00.917–{w1}, <Subject 1> reaches the right arm diagonally upward on the {a1} beat while kicking the left heel playfully back, then extends the left arm sideways with a crisp left hip pop on the {a2} beat as the garment hem sways.
{w1}–{w2} (CHAIN B10 + B12 — Pendulum Hip-Swings & Wrist Rolls): During {w1}–{w2}, <Subject 1> sweeps both forearms diagonally right with a right hip pop on the {a3} beat, mirrors a left hip pop on the {a4} beat, and rolls both wrists inward at chest height before flipping both palms open.
{w2}–{end_s} (CHAIN A2 + A3 — Cat-Paw Shoulder Bounce & Transition Hook): During {w2}–{end_s}, <Subject 1> lifts both hands near shoulder height in a playful bounce on the {a5} and {a6} beats, then gathers both hands smoothly to chest height with weight loaded onto the left foot, ready to launch seamlessly into the next segment.

overall_soundscape:
N/A — the reused <Audio 1> is the complete final audio track.

non_diegetic_music:
N/A — <Audio 1> is directly reused 1:1 as the complete final audio track."""


def auto_prepare_audio_and_beat_tokens(
    bgm_source: Optional[str],
    cfg: dict,
    fps: int = 24,
    target_duration: Optional[float] = None,
) -> Dict[str, str]:
    """Automatically detect audio duration (8s/15s/20s/30s+), plan N segments with 22-frame overlap, slice WAVs, and extract beat tokens."""
    import math
    import struct
    import wave

    out_dir = LOCAL_COMFY / "input" if LOCAL_COMFY.exists() else Path("/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)

    master_name = cfg.get("master_bgm_wav", "master_bgm_15s.wav")
    src_p = find_asset_file(bgm_source) if bgm_source else find_asset_file(master_name)
    if bgm_source and not src_p:
        raise FileNotFoundError(f"❌ 未找到指定的音频文件: {bgm_source}")

    # Determine target duration in seconds
    overlap_frames = 22
    overlap_s = overlap_frames / float(fps)  # 0.9167s

    if src_p and (bgm_source or target_duration is not None):
        probe_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(src_p),
        ]
        probed_dur = float(subprocess.check_output(probe_cmd).decode().strip() or "15.0833")
        desired_dur = float(target_duration) if target_duration else probed_dur
        desired_dur = max(4.0, min(desired_dur, probed_dur))

        # Plan segment lengths strictly on the MiniMax-H3 17k+5 Group-Phase Ladder (Tessera Law r69: FRAME_PER_TOKEN=(1,4,4,4,4))
        # Valid 17k+5 frame counts: [56, 73, 90, 107, 124, 141, 158, 175, 192]
        # Standard segment = 192f (17*11+5 = 8.000s), overlap = 22f (17*1+5 = 0.9167s), stride = 170f (17*10 -> 50 latent tokens == 0 mod 5)
        ladder_17k5 = [56, 73, 90, 107, 124, 141, 158, 175, 192]
        def snap_17k5(target_frames: float) -> int:
            return min(ladder_17k5, key=lambda x: abs(x - target_frames))

        seg_plan = []  # list of (seg_id, start_s, dur_s, length_frames)
        if desired_dur <= 8.10:
            f_len = snap_17k5(desired_dur * fps)
            seg_plan.append((1, 0.0, round(f_len / float(fps), 4), f_len))
        else:
            std_frames = 192
            std_dur = std_frames / float(fps)  # 8.000s
            cur_start = 0.0
            s_idx = 1
            while True:
                rem = desired_dur - cur_start
                if s_idx > 1 and rem <= 8.15:
                    f_len = snap_17k5(rem * fps)
                    seg_plan.append((s_idx, round(cur_start, 4), round(f_len / float(fps), 4), f_len))
                    break
                seg_plan.append((s_idx, round(cur_start, 4), std_dur, std_frames))
                cur_start += std_dur - overlap_s  # 7.0833s exact 50-latent-token stride
                s_idx += 1
                if cur_start >= desired_dur - 0.25:
                    break

        total_planned_dur = seg_plan[-1][1] + seg_plan[-1][2]
        stem = Path(src_p).stem
        master_name = f"{stem}_master_{int(round(total_planned_dur))}s.wav"
        master_p = out_dir / master_name
        subprocess.check_call(
            ["ffmpeg", "-y", "-ss", "0.000", "-t", f"{total_planned_dur:.4f}", "-i", str(src_p), "-ar", "44100", "-ac", "2", str(master_p)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cfg["master_bgm_wav"] = master_name
        cfg["expected_segs"] = [sp[0] for sp in seg_plan]

        # Slice per-segment WAVs and dynamically expand/trim cfg["segments"]
        orig_segs = cfg.get("segments", [])
        seg1_tmpl = copy.deepcopy(orig_segs[0]) if orig_segs else {"seg_id": 1, "length": 192, "seed": 2026110111}
        seg_last_tmpl = copy.deepcopy(orig_segs[-1]) if len(orig_segs) >= 2 else copy.deepcopy(seg1_tmpl)
        new_segs = []
        n_total = len(seg_plan)
        for s_id, st_s, d_s, f_len in seg_plan:
            seg_wav_name = f"{stem}_seg{s_id:02d}_{f_len}f.wav"
            seg_wav_p = out_dir / seg_wav_name
            subprocess.check_call(
                ["ffmpeg", "-y", "-ss", f"{st_s:.4f}", "-t", f"{d_s:.4f}", "-i", str(master_p), "-ar", "44100", "-ac", "2", str(seg_wav_p)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if s_id == 1:
                sc = copy.deepcopy(seg1_tmpl)
            elif s_id == n_total:
                sc = copy.deepcopy(seg_last_tmpl)
            else:
                sc = copy.deepcopy(seg_last_tmpl)
                sc.pop("guide_keyframes", None)
                sc["prompt"] = "__AUTO_MIDDLE_PROMPT__"
            sc["seg_id"] = s_id
            sc["length"] = f_len
            sc["audio_wav"] = seg_wav_name
            sc["start_sec"] = st_s
            sc["dur_sec"] = d_s
            sc["seed"] = int(seg1_tmpl.get("seed", 2026110110)) + s_id - 1
            new_segs.append(sc)
        cfg["segments"] = new_segs
        print(
            f"🎵 已自动适配音频时长 ({desired_dur:.2f}s -> 规划 {n_total} 段共 {total_planned_dur:.2f}s): "
            + ", ".join(f"Seg{s[0]}[{s[1]:.2f}s..{s[1]+s[2]:.2f}s/{s[3]}f]" for s in seg_plan),
            flush=True,
        )

    wav_p = find_asset_file(cfg.get("master_bgm_wav", master_name))
    onsets_s: List[float] = []
    bpm = 125
    if wav_p and wav_p.exists():
        try:
            with wave.open(str(wav_p), "rb") as wf_wav:
                sr = wf_wav.getframerate()
                nch = wf_wav.getnchannels()
                nframes = wf_wav.getnframes()
                raw = struct.unpack(f"<{nframes * nch}h", wf_wav.readframes(nframes))
            hop = max(1, (sr // fps) * nch)
            n_vframes = len(raw) // hop
            env = []
            for i in range(n_vframes):
                chunk = raw[i * hop : (i + 1) * hop : nch]
                rms = math.sqrt(sum(x * x for x in chunk) / max(1, len(chunk)))
                env.append(rms)
            onset = [max(0.0, env[i] - env[i - 1]) if i > 0 else 0.0 for i in range(len(env))]
            thresh = sorted(onset)[int(len(onset) * 0.72)] if onset else 0.0
            onsets_s = [
                round(i / float(fps), 3)
                for i in range(2, len(onset) - 2)
                if onset[i] == max(onset[i - 2 : i + 3]) and onset[i] > thresh
            ]
            best_lag = int(round(fps * 60.0 / 125.0))
            best_score = -1.0
            for cand_bpm in range(88, 152, 2):
                lag = max(6, int(round(fps * 60.0 / cand_bpm)))
                score = sum(onset[i] * onset[i - lag] for i in range(lag, len(onset)))
                if score > best_score:
                    best_score = score
                    best_lag = lag
            bpm = int(round(60.0 * fps / best_lag))
        except Exception as exc:
            print(f"⚠️ 自动节拍分析回退默认 125 BPM: {exc}", flush=True)

    beat_s = round(60.0 / max(60, bpm), 2)

    # Populate middle segment prompts if N >= 3
    segs_list = cfg.get("segments", [])
    n_total = len(segs_list)
    for sc in segs_list:
        if sc.get("prompt") == "__AUTO_MIDDLE_PROMPT__":
            st_s = float(sc.get("start_sec", 7.083))
            d_s = float(sc.get("dur_sec", 8.0))
            local_onsets = [round(t - st_s, 3) for t in onsets_s if (st_s + 0.4) <= t <= (st_s + d_s - 0.1)]
            sc["prompt"] = build_middle_segment_prompt(int(sc["seg_id"]), n_total, d_s, bpm, beat_s, local_onsets)

    last_offset = float(segs_list[-1].get("start_sec", 7.083)) if len(segs_list) >= 2 else 7.083
    last_dur = float(segs_list[-1].get("dur_sec", 8.0)) if len(segs_list) >= 2 else 8.0
    scale_last = last_dur / 8.0

    def snap(target: float, lo: float, hi: float) -> str:
        cands = [t for t in onsets_s if lo <= t <= hi]
        val = min(cands, key=lambda x: abs(x - target)) if cands else target
        return f"00:0{val:05.3f}" if val < 10.0 else f"00:{val:06.3f}"

    def snap_seg_last(target_local: float, lo_local: float, hi_local: float) -> str:
        t_scaled = max(0.95, min(last_dur - 0.15, target_local * scale_last))
        lo_s = max(0.92, lo_local * scale_last)
        hi_s = min(last_dur - 0.1, hi_local * scale_last)
        cands = [round(t - last_offset, 3) for t in onsets_s if (lo_s + last_offset) <= t <= (hi_s + last_offset)]
        val = min(cands, key=lambda x: abs(x - t_scaled)) if cands else t_scaled
        return f"00:0{val:05.3f}"

    s1_list = [f"{t:.2f}s" for t in onsets_s if 0.4 <= t <= 7.95][:12]
    s2_list = [f"{t - last_offset:.2f}s" for t in onsets_s if (last_offset + 0.4) <= t <= (last_offset + last_dur - 0.1)][:12]

    tokens = {
        "bpm": str(bpm),
        "beat_s": f"{beat_s:.2f}",
        "s1_accents": ", ".join(s1_list) if s1_list else "every musical beat",
        "s2_accents": ", ".join(s2_list) if s2_list else "every musical beat",
        "t_s1_a1": snap(0.720, 0.50, 0.90),
        "t_s1_a2": snap(1.220, 1.00, 1.45),
        "t_s1_w1_end": snap(1.220, 1.00, 1.45),
        "t_s1_a3": snap(2.040, 1.80, 2.25),
        "t_s1_a4": snap(2.640, 2.45, 2.85),
        "t_s1_w2_end": snap(3.120, 2.95, 3.35),
        "t_s1_a5": snap(3.620, 3.45, 3.90),
        "t_s1_a6": snap(4.560, 4.35, 4.85),
        "t_s1_a7": snap(5.520, 5.30, 5.75),
        "t_s1_w3_end": snap(6.000, 5.80, 6.20),
        "t_s1_a8": snap(6.440, 6.30, 6.65),
        "t_s1_a9": snap(6.960, 6.80, 7.15),
        "t_s1_a10": snap(7.480, 7.30, 7.68),
        "t_s1_a11": snap(7.920, 7.75, 7.98),
        "t_s2_a1": snap_seg_last(1.300, 1.10, 1.55),
        "t_s2_a2": snap_seg_last(1.780, 1.60, 2.05),
        "t_s2_mid1": snap_seg_last(2.360, 2.15, 2.55),
        "t_s2_a3": snap_seg_last(2.760, 2.60, 2.98),
        "t_s2_a4": snap_seg_last(3.220, 3.05, 3.45),
        "t_s2_w1_end": snap_seg_last(3.760, 3.55, 3.98),
        "t_s2_spin_start": snap_seg_last(4.350, 4.10, 4.60),
        "t_s2_spin_apex": snap_seg_last(5.083, 4.85, 5.30),
        "t_s2_spin_land": snap_seg_last(5.750, 5.50, 5.95),
        "t_s2_w2_end": snap_seg_last(6.000, 5.85, 6.20),
        "t_s2_freeze": snap_seg_last(7.040, 6.80, 7.30),
        "t_s2_end": f"00:0{last_dur:05.3f}",
    }
    print(f"🥁 自动节拍对齐完成: 实测 ~{bpm} BPM (单拍 {beat_s:.2f}s) | 总段数={n_total}", flush=True)
    return tokens


def inject_segment_taillock(
    wf: dict,
    seg_id: int,
    image_name: str,
    frames: List[int] = [107, 136],
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
    kf_list = keyframes if keyframes else [{"image": image_name, "frame_idx": int(f)} for f in frames]

    img_to_nid: Dict[str, str] = {}
    next_offset = 1
    for kf in kf_list:
        im_name = kf["image"]
        if im_name not in img_to_nid:
            img_nid = str(base_id + next_offset)
            next_offset += 1
            wf[img_nid] = {"class_type": "LoadImage", "inputs": {"image": im_name}}
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
    cfg = {}
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            print(f"📋 已加载舞蹈配置文件: {cfg_path}", flush=True)

    prefix = args.prefix or cfg.get("project_prefix") or "DouyinDance_15s_V1"
    width = int(cfg.get("width", 768))
    height = int(cfg.get("height", 1344))

    if "unet_name" in cfg and "1" in wf:
        wf["1"]["inputs"]["unet_name"] = cfg["unet_name"]
    if "turbo_lora_name" in cfg and "4" in wf:
        wf["4"]["inputs"]["lora_name"] = cfg["turbo_lora_name"]

    style_mode = args.style_mode or cfg.get("style_mode", "photorealistic")
    if style_mode in ("stylized", "anime", "none"):
        if "2" in wf:
            wf["2"]["inputs"]["strength_model"] = 0.0
        if "3" in wf:
            wf["3"]["inputs"]["strength_model"] = 0.0
    else:
        if "realism_lora_strength" in cfg and "2" in wf:
            wf["2"]["inputs"]["strength_model"] = float(cfg["realism_lora_strength"])
        if "closeup_lora_strength" in cfg and "3" in wf:
            wf["3"]["inputs"]["strength_model"] = float(cfg["closeup_lora_strength"])

    if "5" in wf:
        if "shift_video" in cfg:
            wf["5"]["inputs"]["shift_video"] = float(cfg["shift_video"])
        if "shift_audio" in cfg:
            wf["5"]["inputs"]["shift_audio"] = float(cfg["shift_audio"])

    # Optional Tessera / Alissonerdx Optimizations: MiniMaxChunkFeedForward + LMS Detail LoRA
    prev_model_nid = "4"
    if cfg.get("chunk_ffn", False):
        has_chunk_ffn = False
        try:
            info = json.loads(
                urllib.request.urlopen(f"{COMFY_URL}/object_info/MiniMaxChunkFeedForward", timeout=2).read().decode()
            )
            has_chunk_ffn = "MiniMaxChunkFeedForward" in info
        except Exception:
            has_chunk_ffn = False
        if has_chunk_ffn:
            wf["19"] = {
                "class_type": "MiniMaxChunkFeedForward",
                "inputs": {
                    "model": [prev_model_nid, 0],
                    "chunk_size": int(cfg.get("chunk_ffn_size", 2)),
                    "min_seq_len": int(cfg.get("chunk_ffn_min_seq", 4096)),
                },
            }
            prev_model_nid = "19"
        else:
            print("ℹ️ 当前 ComfyUI 未内置 MiniMaxChunkFeedForward 节点（96GB 显存无需 FFN 分块），自动跳过 Node #19")
    lms_strength = float(cfg.get("lms_lora_strength", 0.0))
    if lms_strength > 0.0:
        wf["18"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": [prev_model_nid, 0],
                "lora_name": cfg.get("lms_lora_name", "minimax_h3_lms_v1.0_r64.safetensors"),
                "strength_model": lms_strength,
            },
        }
        prev_model_nid = "18"
    if "5" in wf and prev_model_nid != "4":
        wf["5"]["inputs"]["model"] = [prev_model_nid, 0]

    if "15" in wf and "sigmas_tail_strength" in cfg:
        wf["15"]["inputs"]["strength"] = float(cfg["sigmas_tail_strength"])

    if "48" in wf:
        if "join_match_strength" in cfg:
            wf["48"]["inputs"]["strength"] = float(cfg["join_match_strength"])
        if "join_match_min_excess" in cfg:
            wf["48"]["inputs"]["min_excess"] = float(cfg["join_match_min_excess"])
            wf["48"]["inputs"]["min_excess_spread"] = float(cfg["join_match_min_excess"])
        if "join_match_spread" in cfg:
            wf["48"]["inputs"]["match_spread"] = bool(cfg["join_match_spread"])

    bunny_alpha = args.bunny_alpha if args.bunny_alpha is not None else cfg.get("bunny_alpha", 0.22)
    disable_ref_audio = bool(cfg.get("disable_ref_audio", False))

    # Universal Audio Beat-Grid & Phase-Aligned Slicer
    beat_tokens = auto_prepare_audio_and_beat_tokens(
        getattr(args, "bgm", None),
        cfg,
        fps=int(cfg.get("fps", 24)),
        target_duration=getattr(args, "duration", None),
    )

    # Universal Character Image Override (--image / --closeup-image)
    cli_img = getattr(args, "image", None)
    cli_closeup = getattr(args, "closeup_image", None)
    if cli_img:
        img_p = find_asset_file(cli_img)
        if not img_p:
            raise FileNotFoundError(f"❌ 未找到指定的人物参考图: {cli_img}")
        img_name = img_p.name
        ref_list = [img_name]
        if cli_closeup:
            cu_p = find_asset_file(cli_closeup)
            if cu_p:
                ref_list.append(cu_p.name)
        for sc in cfg.get("segments", []):
            sc["ref_images_custom"] = list(ref_list)
            if int(sc.get("seg_id", 0)) == 1:
                sc["guide_keyframes"] = [{"image": img_name, "frame_idx": 0}]
            else:
                sc.pop("guide_keyframes", None)

    seg_cfgs = {int(s["seg_id"]): s for s in cfg.get("segments", []) if "seg_id" in s}
    max_cfg_seg = max(seg_cfgs.keys()) if seg_cfgs else 2
    ensure_segment_slots_in_workflow(wf, max_cfg_seg)
    args._runtime_cfg = cfg

    for seg_id, meta in SEG_META.items():
        jm_nid = meta.get("out_lat")
        if seg_id >= 2 and jm_nid and jm_nid in wf and wf[jm_nid].get("class_type") == "MiniMaxH3JoinMatch":
            if "join_match_strength" in cfg:
                wf[jm_nid]["inputs"]["strength"] = float(cfg["join_match_strength"])
            if "join_match_min_excess" in cfg:
                wf[jm_nid]["inputs"]["min_excess"] = float(cfg["join_match_min_excess"])
                wf[jm_nid]["inputs"]["min_excess_spread"] = float(cfg["join_match_min_excess"])
            if "join_match_spread" in cfg:
                wf[jm_nid]["inputs"]["match_spread"] = bool(cfg["join_match_spread"])

    for seg_id, meta in SEG_META.items():
        scfg = seg_cfgs.get(seg_id, {})
        r2v_nid = meta["r2v"]
        bunny_nid = meta["bunny"]
        noise_nid = meta["noise"]
        save_nid = meta["save"]

        if bunny_nid in wf and bunny_alpha is not None:
            wf[bunny_nid]["inputs"]["alpha"] = float(bunny_alpha)

        if r2v_nid in wf:
            wf[r2v_nid]["inputs"]["width"] = width
            wf[r2v_nid]["inputs"]["height"] = height
            if "length" in scfg:
                wf[r2v_nid]["inputs"]["length"] = int(scfg["length"])
            raw_prompt = scfg.get("prompt") or wf[r2v_nid]["inputs"].get("prompt", "")
            if raw_prompt:
                try:
                    raw_prompt = raw_prompt.format(**beat_tokens)
                except KeyError:
                    pass
                wf[r2v_nid]["inputs"]["prompt"] = raw_prompt

        if save_nid in wf:
            out_seg_num = int(scfg.get("output_seg_id", seg_id))
            wf[save_nid]["inputs"]["filename_prefix"] = f"{prefix}_Seg{out_seg_num:02d}"

        if "seed" in scfg and noise_nid in wf:
            wf[noise_nid]["inputs"]["noise_seed"] = int(scfg["seed"])

        # Per-segment audio_wav wiring
        aud_wav = scfg.get("audio_wav")
        if disable_ref_audio and r2v_nid in wf:
            for k in list(wf[r2v_nid]["inputs"].keys()):
                if k.startswith("ref_audios."):
                    wf[r2v_nid]["inputs"].pop(k, None)
        elif aud_wav and r2v_nid in wf:
            aud_load_nid = str(seg_id * 100 + 90)
            wf[aud_load_nid] = {"class_type": "LoadAudio", "inputs": {"audio": aud_wav}}
            wf[r2v_nid]["inputs"]["ref_audios.ref_audio_0"] = [aud_load_nid, 0]

        # Per-segment ref_images_custom wiring
        ref_imgs_custom = scfg.get("ref_images_custom")
        if ref_imgs_custom and r2v_nid in wf:
            for k in list(wf[r2v_nid]["inputs"].keys()):
                if k.startswith("ref_images."):
                    wf[r2v_nid]["inputs"].pop(k, None)
            for r_idx, img_name in enumerate(ref_imgs_custom[:2]):
                custom_load_nid = str(seg_id * 100 + 80 + r_idx)
                wf[custom_load_nid] = {"class_type": "LoadImage", "inputs": {"image": img_name}}
                wf[r2v_nid]["inputs"][f"ref_images.ref_image_{r_idx}"] = [custom_load_nid, 0]

        # Optional keyframe guides
        guide_kfs = scfg.get("guide_keyframes")
        if guide_kfs:
            valid_kfs = [kf for kf in guide_kfs if find_asset_file(kf["image"]) is not None]
            if len(valid_kfs) == len(guide_kfs):
                inject_segment_taillock(wf, seg_id, valid_kfs[0]["image"], keyframes=valid_kfs)
        elif scfg.get("tail_lock_image"):
            t_img = scfg["tail_lock_image"]
            t_frames = scfg.get("tail_lock_frames", [107, 136])
            if find_asset_file(t_img) is not None:
                inject_segment_taillock(wf, seg_id, t_img, t_frames)

    # Remove unused default LoadImage/LoadAudio nodes 10, 12, 13 if no active node references them
    referenced_nids = set()
    for node in wf.values():
        for v in node.get("inputs", {}).values():
            if isinstance(v, list) and len(v) == 2:
                referenced_nids.add(str(v[0]))
    for default_nid in ("10", "12", "13"):
        if default_nid not in referenced_nids:
            wf.pop(default_nid, None)

    return prefix


def sync_inputs_to_local_ssd(wf: dict, extra_files: Optional[List[str]] = None) -> None:
    src_in = DRIVE_COMFY / "input"
    dst_in = LOCAL_COMFY / "input"
    dst_in.mkdir(parents=True, exist_ok=True)

    needed_files = list(extra_files or [])
    for node in wf.values():
        ctype = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if ctype == "LoadImage" and isinstance(inputs.get("image"), str):
            needed_files.append(inputs["image"])
        elif ctype == "LoadAudio" and isinstance(inputs.get("audio"), str):
            needed_files.append(inputs["audio"])

    for fname in dict.fromkeys(needed_files):
        found = find_asset_file(fname)
        if not found:
            raise FileNotFoundError(f"❌ 缺少工作流所需的输入文件: {fname}")
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

    # Ensure all diffusion_models and loras on Drive are symlinked into LOCAL_COMFY
    for sub in ("diffusion_models", "loras", "text_encoders", "vae"):
        drv_m = DRIVE_COMFY / "models" / sub
        loc_m = LOCAL_COMFY / "models" / sub
        if drv_m.exists() and loc_m.exists():
            for f in drv_m.iterdir():
                dst_f = loc_m / f.name
                if f.is_file() and not dst_f.exists():
                    os.symlink(f, dst_f)

    # Auto-provision minimax_h3_ref2va_pruned_int8_convrot.safetensors if requested by workflow
    req_unet = wf.get("1", {}).get("inputs", {}).get("unet_name", "")
    if req_unet == "minimax_h3_ref2va_pruned_int8_convrot.safetensors":
        l_base = LOCAL_COMFY / "models" / "diffusion_models" / req_unet
        d_base = DRIVE_COMFY / "models" / "diffusion_models" / req_unet
        f_base = Path("/content/fast_models/diffusion_models") / req_unet
        f_base.parent.mkdir(parents=True, exist_ok=True)
        if not (l_base.exists() and l_base.stat().st_size > 15_000_000_000):
            if d_base.exists() and d_base.stat().st_size > 15_000_000_000:
                print(f"📦 正在从云盘拷贝 {req_unet} 至本地 NVMe...", flush=True)
                shutil.copy2(d_base, f_base)
            elif not (f_base.exists() and f_base.stat().st_size > 15_000_000_000):
                url = f"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/{req_unet}"
                print(f"⬇️ 正在高速拉取官方基底模型 {req_unet} 至本地 NVMe...", flush=True)
                subprocess.run(["aria2c", "-x16", "-s16", "-k1M", "--console-log-level=warn", "-d", str(f_base.parent), "-o", req_unet, url], check=True)
            if l_base.is_symlink() or l_base.exists():
                l_base.unlink()
            os.symlink(f_base, l_base)


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
    dst_drv.parent.mkdir(parents=True, exist_ok=True)
    if dst_drv.exists() and src.resolve() != dst_drv.resolve():
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
    expected_segs: List[int] = [1, 2],
    output_subdir: Optional[str] = None,
    master_bgm_wav: Optional[str] = None,
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

    raw_master_tmp = Path(f"/tmp/{Path(out_mp4_name).stem}_VAEAudio.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            str(raw_master_tmp),
        ],
        check=True,
    )
    safe_copy_to_drive_with_history(raw_master_tmp, drv_out_dir / raw_master_tmp.name)

    master_tmp = Path(f"/tmp/{out_mp4_name}")
    bgm_file = find_asset_file(master_bgm_wav) if master_bgm_wav else None
    if bgm_file and bgm_file.exists():
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(raw_master_tmp),
                "-i", str(bgm_file),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "256k",
                "-shortest",
                str(master_tmp),
            ],
            check=True,
        )
        print(f"🎵 已将无损满血原声 BGM ({bgm_file.name}) 1:1 回铺至母带！", flush=True)
    else:
        shutil.copy2(raw_master_tmp, master_tmp)

    out_drv = drv_out_dir / out_mp4_name
    safe_copy_to_drive_with_history(master_tmp, out_drv)
    print(f"🎬 竖屏舞蹈母带合成并同步至 Google Drive 完毕: {out_drv} ({master_tmp.stat().st_size / (1024**2):.2f} MB)", flush=True)
    return master_tmp


def patch_bypass_bunny_if_needed(wf: dict, obj_info: dict) -> None:
    if "BunnyH3ConditioningBridge" not in obj_info:
        bunny_to_r2v = {m["bunny"]: m["r2v"] for m in SEG_META.values()}
        for node in wf.values():
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
    end_seg: int = 2,
) -> None:
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

    if start_seg > 1:
        needs_prev_latent = SEG_META[start_seg]["ctx"] is not None
        prev_s = start_seg - 1
        ckpt_name = f"{prefix}_Seg{prev_s:02d}_av_latent.pt"
        ckpt_path = latent_dir / ckpt_name
        if not needs_prev_latent:
            for s_drop in range(1, start_seg):
                for nid_int in seg_node_ranges[s_drop]:
                    wf.pop(str(nid_int), None)
                wf.pop(f"90{s_drop}", None)
        elif has_ckpt_nodes and ckpt_path.exists():
            print(f"⚡ 命中云盘潜变量断点存档: {ckpt_path}，自动跳过前 {prev_s} 段 (耗时 0.05s)！", flush=True)
            for s_drop in range(1, start_seg):
                for nid_int in seg_node_ranges[s_drop]:
                    wf.pop(str(nid_int), None)
                wf.pop(f"90{s_drop}", None)
            prev_lat_nid = seg_latent_nodes[prev_s]
            wf[prev_lat_nid] = {
                "class_type": "H3LoadAVLatent",
                "inputs": {"checkpoint_name": ckpt_name},
            }

    if end_seg < max_s:
        for s_drop in range(end_seg + 1, max_s + 1):
            for nid_int in seg_node_ranges[s_drop]:
                wf.pop(str(nid_int), None)
            wf.pop(f"90{s_drop}", None)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="G4 MiniMax-H3 9:16 Vertical Douyin Dance Runner")
    parser.add_argument("--config", default="/tmp/dance_config.json")
    parser.add_argument("--workflow", default="/tmp/workflow_douyin_dance_2seg.json")
    parser.add_argument("--image", default=None, help="Universal override: path/filename of any character opening reference image (<Picture 1>)")
    parser.add_argument("--closeup-image", default=None, help="Optional medium close-up reference image (<Picture 2>)")
    parser.add_argument("--bgm", default=None, help="Universal override: path/filename of any BGM audio file (auto-sliced & beat-detected)")
    parser.add_argument("--duration", type=float, default=None, help="Target video duration in seconds (e.g. 15, 20, 30); defaults to BGM duration")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--out-mp4", default=None)
    parser.add_argument("--style-mode", choices=["photorealistic", "stylized", "anime", "none"], default=None)
    parser.add_argument("--bunny-alpha", type=float, default=None)
    parser.add_argument("--only-seg", type=int, default=None)
    parser.add_argument("--start-seg", type=int, default=1)
    parser.add_argument("--end-seg", type=int, default=None)
    args, _ = parser.parse_known_args()

    t_start = time.time()
    ensure_comfy_running()

    wf_path = Path(args.workflow)
    wf = copy.deepcopy(json.loads(wf_path.read_text(encoding="utf-8")))

    prefix = apply_config_and_cli_overrides(wf, args)
    max_s = max(SEG_META.keys())
    start_s = args.only_seg if args.only_seg else args.start_seg
    end_s = args.only_seg if args.only_seg else (args.end_seg or max_s)
    cfg_dict = getattr(args, "_runtime_cfg", None) or (
        json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config and Path(args.config).exists() else {}
    )
    out_mp4 = args.out_mp4 or cfg_dict.get("output_mp4") or f"{prefix}_Master.mp4"
    expected_segs = cfg_dict.get("expected_segs", list(range(1, max_s + 1)))
    output_subdir = cfg_dict.get("output_subdir")
    master_bgm_wav = cfg_dict.get("master_bgm_wav")

    obj_info = json.loads(urllib.request.urlopen(f"{API_URL}/object_info", timeout=15).read().decode())
    patch_bypass_bunny_if_needed(wf, obj_info)
    inject_latent_checkpoints_and_slice(wf, obj_info, prefix, start_seg=start_s, end_seg=end_s)

    extra_files = [master_bgm_wav] if master_bgm_wav else []
    sync_inputs_to_local_ssd(wf, extra_files=extra_files)

    submit_and_wait(wf, f"MiniMax-H3 Dance Seg[{start_s}..{end_s}] ({prefix})")
    stitch_available_segments(
        prefix,
        out_mp4,
        expected_segs=expected_segs,
        output_subdir=output_subdir,
        master_bgm_wav=master_bgm_wav,
    )

    print(f"🎉 ALL_DANCE_DONE (总用时 {time.time() - t_start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
