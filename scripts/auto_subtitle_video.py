#!/usr/bin/env python3
"""Automatic Acoustic-Aligned Cinema Subtitle Generator (SRT + Burned-in MP4).

Features:
1. Automatically analyzes the audio track of each segment (or full video) using
   vocal-band RMS energy envelope detection (excluding short impact transients like table slaps)
   to pinpoint exact speech_start and speech_end timestamps to 10ms precision.
2. Automatically splits two-clause sentences at the natural acoustic breath pause (local energy minimum)
   so each clause appears in sync with the actor's actual lip movement.
3. Outputs both a standard `.srt` file and a cinema-grade burned-in `.mp4` rendered with
   macOS PingFang SC / Noto Sans CJK (anti-aliased 2x supersampled typography with subtle film shadow).
"""

import argparse
import array
import math
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


DEFAULT_SCRIPT_LINES = [
    ("林薇", "这份底牌一旦交出去，就再也收不回来了。"),
    ("陈默", "我等这一天，已经等得太久了。"),
    ("林薇", "你知不知道这样做，会毁掉我们所有人！"),
    ("陈默", "有些真相一旦揭开，就没有回头路了！"),
]


def load_wav_mono(video_path: Path, sr: int = 16000) -> Tuple[List[float], float]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-ac", "1", "-ar", str(sr),
                "-af", "highpass=f=130,lowpass=f=3600",
                str(tmp_wav),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with wave.open(str(tmp_wav), "rb") as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            samples = array.array("h", raw)
            audio = [s / 32768.0 for s in samples]
            duration = n_frames / float(sr)
            return audio, duration
    finally:
        tmp_wav.unlink(missing_ok=True)


def detect_speech_window_and_pause(
    audio: List[float],
    sr: int,
    duration: float,
    text: str,
    min_start: float = 0.40,
    max_end_margin: float = 0.25,
) -> List[Tuple[float, float, str]]:
    """Detect exact speech onset, offset, and mid-sentence clause split from vocal energy envelope."""
    hop_ms = 20
    hop = int(sr * hop_ms / 1000)
    win = hop * 2
    n_hops = max(1, (len(audio) - win) // hop)
    rms = []
    for i in range(n_hops):
        chunk = audio[i * hop : i * hop + win]
        mean_sq = sum(x * x for x in chunk) / max(1, len(chunk))
        rms.append(math.sqrt(mean_sq + 1e-12))
    times = [i * (hop_ms / 1000.0) for i in range(n_hops)]

    # Smooth with a 100ms moving average window to reject brief palm-slap impulses (<80ms)
    rms_smooth = []
    for i in range(n_hops):
        lo = max(0, i - 2)
        hi = min(n_hops, i + 3)
        rms_smooth.append(sum(rms[lo:hi]) / (hi - lo))

    valid_indices = [
        i for i, t in enumerate(times)
        if min_start <= t <= max(min_start + 1.0, duration - max_end_margin)
    ]
    sorted_all = sorted(rms_smooth)
    noise_floor = sorted_all[int(len(sorted_all) * 0.20)] if sorted_all else 0.0
    valid_vals = sorted(rms_smooth[i] for i in valid_indices) if valid_indices else sorted_all
    peak_val = valid_vals[int(len(valid_vals) * 0.95)] if valid_vals else 0.1
    thresh = noise_floor + 0.18 * (peak_val - noise_floor)

    active_idx = [i for i in valid_indices if rms_smooth[i] >= thresh]
    if len(active_idx) >= 2:
        t_start = max(0.35, times[active_idx[0]] - 0.06)
        t_end = min(duration - 0.12, times[active_idx[-1]] + 0.14)
    else:
        t_start, t_end = 0.85, min(duration - 0.30, 4.25)

    if t_end - t_start < 1.5:
        t_start = min(t_start, 0.90)
        t_end = max(t_end, min(duration - 0.25, 4.10))

    # Check if we should split into two clauses at comma '，'
    if "，" in text:
        parts = [p.strip() for p in text.split("，", 1) if p.strip()]
        if len(parts) == 2:
            c1 = parts[0]
            c2 = parts[1].rstrip("。.")
            ratio = len(c1) / float(len(c1) + len(c2))
            expected_split = t_start + (t_end - t_start) * ratio
            search_indices = [
                i for i, t in enumerate(times)
                if (expected_split - 0.45) <= t <= (expected_split + 0.45)
            ]
            if search_indices:
                min_idx = min(search_indices, key=lambda idx: rms_smooth[idx])
                t_split = times[min_idx]
            else:
                t_split = expected_split
            t_split = max(t_start + 0.80, min(t_end - 0.80, t_split))
            return [
                (t_start, t_split - 0.02, c1),
                (t_split + 0.02, t_end, c2),
            ]

    clean_txt = text.rstrip("。.")
    return [(t_start, t_end, clean_txt)]


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hrs = ms // 3600000
    ms %= 3600000
    mins = ms // 60000
    ms %= 60000
    secs = ms // 1000
    ms %= 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def load_cjk_font(size: int) -> ImageFont.FreeTypeFont:
    font_candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    ]
    for fp in font_candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=size, index=0)
            except Exception:
                return ImageFont.truetype(fp, size=size)
    return ImageFont.load_default()


def render_subtitle_overlay_png(text: str, width: int, height: int, out_png: Path) -> None:
    """Render a 2x supersampled transparent PNG subtitle overlay with cinema stroke and drop shadow."""
    scale = 2
    W, H = width * scale, height * scale
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_size = int(height * 0.046 * scale)  # ~35px at 768p -> crisp cinema subtitle size
    font = load_cjk_font(font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (W - tw) // 2
    y = int(H * 0.885) - th // 2

    # Soft translucent pill backing for maximum legibility over any background
    pad_x, pad_y = int(22 * scale), int(10 * scale)
    draw.rounded_rectangle(
        [x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y + int(4 * scale)],
        radius=int(10 * scale),
        fill=(10, 14, 20, 125),
    )

    # Subtle dark outline + crisp warm-white text
    stroke_w = int(2.5 * scale)
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(252, 252, 250, 255),
        stroke_width=stroke_w,
        stroke_fill=(0, 0, 0, 235),
    )

    img_down = img.resize((width, height), Image.Resampling.LANCZOS)
    img_down.save(out_png)


def render_character_intro_overlay_png(
    width: int,
    height: int,
    out_png: Path,
    mode: str = "both",
) -> None:
    """Render cinema-grade character introduction badges (Lower-Third / Side Callouts) in Segment 1."""
    scale = 2
    W, H = width * scale, height * scale
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font_name = load_cjk_font(int(27 * scale))
    font_en = load_cjk_font(int(12 * scale))
    font_role = load_cjk_font(int(15 * scale))

    def draw_card(x: int, y: int, name_zh: str, name_en: str, role_line1: str, role_line2: str, align_right: bool = False):
        card_w = int(268 * scale)
        card_h = int(84 * scale)
        # Translucent dark slate glass backing
        draw.rounded_rectangle(
            [x, y, x + card_w, y + card_h],
            radius=int(8 * scale),
            fill=(12, 16, 22, 178),
            outline=(255, 255, 255, 45),
            width=max(1, scale // 2),
        )
        # Gold/amber cinematic vertical accent bar
        bar_w = int(4 * scale)
        if not align_right:
            draw.rounded_rectangle(
                [x, y + int(10 * scale), x + bar_w, y + card_h - int(10 * scale)],
                radius=int(2 * scale),
                fill=(218, 165, 68, 255),
            )
            tx = x + int(16 * scale)
        else:
            draw.rounded_rectangle(
                [x + card_w - bar_w, y + int(10 * scale), x + card_w, y + card_h - int(10 * scale)],
                radius=int(2 * scale),
                fill=(96, 165, 250, 255),
            )
            tx = x + int(16 * scale)

        # Name + English tag
        draw.text(
            (tx, y + int(10 * scale)),
            name_zh,
            font=font_name,
            fill=(250, 250, 248, 255),
            stroke_width=scale,
            stroke_fill=(0, 0, 0, 180),
        )
        nb = draw.textbbox((tx, y + int(10 * scale)), name_zh, font=font_name)
        draw.text(
            (nb[2] + int(10 * scale), y + int(21 * scale)),
            name_en,
            font=font_en,
            fill=(190, 200, 215, 220),
        )
        # Role description
        draw.text(
            (tx, y + int(45 * scale)),
            role_line1,
            font=font_role,
            fill=(228, 232, 238, 245),
        )
        if role_line2:
            draw.text(
                (tx, y + int(63 * scale)),
                role_line2,
                font=font_en,
                fill=(165, 175, 190, 215),
            )

    if mode in ("linwei", "both"):
        # Lower-left over Lin Wei's dark blazer shoulder (well clear of her face & bottom subtitle)
        draw_card(
            x=int(46 * scale),
            y=int(536 * scale),
            name_zh="林 薇",
            name_en="LIN WEI · 35岁",
            role_line1="高级法律合伙人 · 特别合规调查组",
            role_line2="握有境外空壳公司三千万底账核心证据",
            align_right=False,
        )

    if mode in ("chenmo", "both"):
        # Upper-right beside Chen Mo in the clean frosted-glass negative space (clear of his face & the tablet)
        draw_card(
            x=int(760 * scale),
            y=int(180 * scale),
            name_zh="陈 默",
            name_en="CHEN MO · 40岁",
            role_line1="集团执行副总裁 · 核心资本操盘手",
            role_line2="瑞士匿名转账幕后签字人",
            align_right=True,
        )

    img_down = img.resize((width, height), Image.Resampling.LANCZOS)
    img_down.save(out_png)


def build_subtitles_for_four_segments(
    seg_paths: List[Path],
    master_video: Path,
    out_video: Path,
    out_srt: Path,
    script_lines: List[Tuple[str, str]] = DEFAULT_SCRIPT_LINES,
) -> List[Tuple[float, float, str, str]]:
    cues: List[Tuple[float, float, str, str]] = []
    offset = 0.0

    for idx, seg_path in enumerate(seg_paths):
        speaker, line = script_lines[idx]
        audio, dur = load_wav_mono(seg_path)
        # Skip the palm press at 0.85s-0.95s in Seg 3 & Seg 4 by setting min_start=1.02s for Seg 2..4
        min_s = 0.45 if idx == 0 else 1.02
        sub_cues = detect_speech_window_and_pause(audio, 16000, dur, line, min_start=min_s)
        for st, et, txt in sub_cues:
            cues.append((offset + st, offset + et, speaker, txt))
        offset += dur

    # Write standard .srt file
    srt_lines = []
    for i, (st, et, spk, txt) in enumerate(cues, 1):
        srt_lines.append(f"{i}\n{format_srt_time(st)} --> {format_srt_time(et)}\n{txt}\n")
    out_srt.write_text("\n".join(srt_lines), encoding="utf-8")

    # Render transparent overlay PNGs for character intro cards + each subtitle cue and composite with ffmpeg
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        inputs = ["-i", str(master_video)]
        filter_parts = []
        last_v = "[0:v]"
        in_idx = 1

        # 1. Character Introduction Badges in Segment 1 with smooth 0.38s fade-in/out + 18px gentle slide-in
        intro_lw_png = tmp_dir / "intro_linwei.png"
        intro_cm_png = tmp_dir / "intro_chenmo.png"
        render_character_intro_overlay_png(1344, 768, intro_lw_png, mode="linwei")
        render_character_intro_overlay_png(1344, 768, intro_cm_png, mode="chenmo")

        inputs.extend(["-loop", "1", "-t", "5.0", "-i", str(intro_lw_png), "-loop", "1", "-t", "5.0", "-i", str(intro_cm_png)])
        lw_idx = in_idx
        in_idx += 1
        cm_idx = in_idx
        in_idx += 1

        filter_parts.append(
            f"[{lw_idx}:v]format=rgba,fade=t=in:st=0.40:d=0.38:alpha=1,fade=t=out:st=3.97:d=0.38:alpha=1[lw_a];"
            f"{last_v}[lw_a]overlay=x='if(lt(t,0.78),-18*(1-(t-0.40)/0.38),0)':y=0:enable='between(t,0.40,4.35)'[v_lw]"
        )
        last_v = "[v_lw]"

        filter_parts.append(
            f"[{cm_idx}:v]format=rgba,fade=t=in:st=0.85:d=0.38:alpha=1,fade=t=out:st=4.07:d=0.38:alpha=1[cm_a];"
            f"{last_v}[cm_a]overlay=x='if(lt(t,1.23),18*(1-(t-0.85)/0.38),0)':y=0:enable='between(t,0.85,4.45)'[v_cm]"
        )
        last_v = "[v_cm]"

        # 2. Acoustic-aligned dialogue subtitles with soft 0.10s fade-in/out
        for i, (st, et, _spk, txt) in enumerate(cues):
            png_path = tmp_dir / f"sub_{i:02d}.png"
            render_subtitle_overlay_png(txt, 1344, 768, png_path)
            inputs.extend(["-loop", "1", "-t", f"{et + 0.5:.2f}", "-i", str(png_path)])
            sub_a = f"[sa{in_idx}]"
            out_v = f"[v{in_idx}]"
            fd = 0.10
            fout_st = max(st + fd, et - fd)
            filter_parts.append(
                f"[{in_idx}:v]format=rgba,fade=t=in:st={st:.3f}:d={fd:.2f}:alpha=1,fade=t=out:st={fout_st:.3f}:d={fd:.2f}:alpha=1{sub_a};"
                f"{last_v}{sub_a}overlay=0:0:enable='between(t,{st:.3f},{et:.3f})'{out_v}"
            )
            last_v = out_v
            in_idx += 1

        filter_complex = ";".join(filter_parts)
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", last_v,
            "-map", "0:a",
            "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(out_video),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return cues


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/Users/delroy/Documents/Playground/test001")
    parser.add_argument("--seg-prefix", default="OfficeDuel_V6_OneShot")
    parser.add_argument("--master", default="MiniMax_H3_v25_20s_V6_OneShot_HeadTurn.mp4")
    parser.add_argument("--out", default="MiniMax_H3_v25_20s_V6_OneShot_Subtitled.mp4")
    parser.add_argument("--srt", default="MiniMax_H3_v25_20s_V6_OneShot.srt")
    args = parser.parse_args()

    root = Path(args.root)
    segs = [root / f"{args.seg_prefix}_Seg0{i}.mp4" for i in (1, 2, 3, 4)]
    cues = build_subtitles_for_four_segments(
        segs,
        root / args.master,
        root / args.out,
        root / args.srt,
    )
    print(f"✅ 自动声学对齐字幕生成完成: {root / args.out} & {root / args.srt}")
    for st, et, spk, txt in cues:
        print(f"  [{st:05.2f}s -> {et:05.2f}s] ({spk}) {txt}")
