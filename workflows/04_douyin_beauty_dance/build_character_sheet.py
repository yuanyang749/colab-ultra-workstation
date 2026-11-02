#!/usr/bin/env python3
"""
自动化「防变脸人物资产卡 (Character Sheet)」生成器 (基于 codex-gen-image)

输入任意一张人物全身参考图 (--input ref_fullbody.png)，自动调用 codex-gen-image 生成：
1. 3/4 侧脸高清大头特写 (提供鼻梁高度、颧骨与下颌线 3D 纵深及高清五官 Token)
2. 盖住小头的正/侧/背全身穿搭三视图 (屏蔽远景模糊小脸干扰，提供 360° 服装、发型长度与身材比例)
3. 自动拼合为：
   - 9:16 竖版资产卡 (*_charsheet_9x16_2k.png，专供 MiniMax-H3 9:16 竖屏视频 <Picture 2> 100% 满像素喂入)
   - 16:9 横版资产总览卡 (*_charsheet_16x9_2k.png，方便直观审阅与展示)
"""
import argparse
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

CODEX_GEN = Path.home() / ".agents/skills/codex-gen-image/scripts/codex_gen_image.py"


def generate_character_sheet(ref_img: Path, out_prefix: Path):
    head_out = out_prefix.parent / f"{out_prefix.name}_3q_head_2k.png"
    body_out = out_prefix.parent / f"{out_prefix.name}_3view_body_2k.png"
    sheet_9x16_out = out_prefix.parent / f"{out_prefix.name}_charsheet_9x16_2k.png"
    sheet_16x9_out = out_prefix.parent / f"{out_prefix.name}_charsheet_16x9_2k.png"

    prompt_head = (
        "Raw unedited iPhone 16 Pro live-action close-up portrait of the EXACT same person from the reference image, "
        "captured at a 3/4 side-angle view (head turned 35 degrees to show three-quarter facial profile, clear 3D nose bridge contour, "
        "cheekbone curve, and natural jawline). Preserve 100% of exact facial identity, gentle natural expression, warm natural skin texture "
        "with soft pores (not over-sharpened), hairstyle, and upper collar of the exact outfit in the reference image."
    )
    prompt_body = (
        "A clean photorealistic full-body 3-view character turnaround reference sheet (three full-body standing figures side by side: "
        "1. Front view, 2. 90-degree Side profile view, 3. 180-degree Back view) of the exact same person and exact same outfit from the "
        "reference image. Each of the three full-body figures has a clean neutral grey circular privacy disc neatly covering the small face "
        "area above the chin so the focus is 100% on the body proportions, hair length from front/side/back, and exact 360-degree outfit details."
    )

    if not head_out.exists():
        subprocess.run(
            [sys.executable, str(CODEX_GEN), "-r", str(ref_img), "-p", prompt_head, "-ar", "3:4", "--res", "2k", "-o", str(head_out), "--json"],
            check=True,
        )
    if not body_out.exists():
        subprocess.run(
            [sys.executable, str(CODEX_GEN), "-r", str(ref_img), "-p", prompt_body, "-ar", "16:9", "--res", "2k", "-o", str(body_out), "--json"],
            check=True,
        )

    im_head = Image.open(head_out).convert("RGB")
    im_body = Image.open(body_out).convert("RGB")
    bw, bh = im_body.size
    body_tight = im_body.crop((int(bw * 0.08), 0, int(bw * 0.92), bh))

    def paste_contain_seamless(canvas, src, box):
        x0, y0, w, h = box
        contained = ImageOps.contain(src, (w, h), method=Image.LANCZOS)
        cw, ch = contained.size
        ox, oy = x0 + (w - cw) // 2, y0 + (h - ch) // 2
        if oy > y0:
            top_strip = contained.crop((0, 0, cw, 4)).resize((w, oy - y0), Image.BILINEAR)
            bot_strip = contained.crop((0, ch - 4, cw, ch)).resize((w, (y0 + h) - (oy + ch)), Image.BILINEAR)
            canvas.paste(top_strip, (x0, y0))
            canvas.paste(bot_strip, (x0, oy + ch))
        if ox > x0:
            left_strip = contained.crop((0, 0, 4, ch)).resize((ox - x0, h), Image.BILINEAR)
            right_strip = contained.crop((cw - 4, 0, cw, ch)).resize(((x0 + w) - (ox + cw), h), Image.BILINEAR)
            canvas.paste(left_strip, (x0, y0))
            canvas.paste(right_strip, (ox + cw, y0))
        canvas.paste(contained, (ox, oy))

    # 1. 9:16 Vertical Sheet (1536 x 2688)
    w9, h9 = 1536, 2688
    top_h = 1580
    bot_h = h9 - top_h
    sheet_9x16 = Image.new("RGB", (w9, h9))
    sheet_9x16.paste(ImageOps.fit(im_head, (w9, top_h), method=Image.LANCZOS, centering=(0.5, 0.38)), (0, 0))
    paste_contain_seamless(sheet_9x16, body_tight, (0, top_h, w9, bot_h))
    ImageDraw.Draw(sheet_9x16).rectangle([0, top_h - 2, w9, top_h + 2], fill=(240, 240, 242))
    sheet_9x16.save(sheet_9x16_out, optimize=True)

    # 2. 16:9 Horizontal Sheet (2400 x 1350)
    w16, h16 = 2400, 1350
    left_w = 880
    right_w = w16 - left_w
    sheet_16x9 = Image.new("RGB", (w16, h16))
    sheet_16x9.paste(ImageOps.fit(im_head, (left_w, h16), method=Image.LANCZOS, centering=(0.5, 0.42)), (0, 0))
    paste_contain_seamless(sheet_16x9, body_tight, (left_w, 0, right_w, h16))
    ImageDraw.Draw(sheet_16x9).rectangle([left_w - 2, 0, left_w + 2, h16], fill=(240, 240, 242))
    sheet_16x9.save(sheet_16x9_out, optimize=True)

    print(f"✅ Saved 9:16 Character Sheet -> {sheet_9x16_out}")
    print(f"✅ Saved 16:9 Character Sheet -> {sheet_16x9_out}")


def main():
    parser = argparse.ArgumentParser(description="Build 3/4 Head + Covered-Head 3-View Character Sheet")
    parser.add_argument("--input", "-i", required=True, help="Input full-body reference image")
    parser.add_argument("--out-prefix", "-o", required=True, help="Output path prefix")
    args = parser.parse_args()
    generate_character_sheet(Path(args.input).resolve(), Path(args.out_prefix).resolve())


if __name__ == "__main__":
    main()
