#!/usr/bin/env python3
"""Lossless Latent-Space Checkpoint Injector & Partial-Chain Resume Builder for MiniMax-H3.

Solves the multi-segment re-shoot dilemma:
- Instead of decoding/re-encoding `.mp4` via `VHS_LoadVideo` (which loses raw pre-VAE latent statistics
  needed by `MiniMaxH3JoinMatch` and causes camera/background drift), OR re-running Segments 1..N-1 from scratch:
- `inject_save_latent_checkpoints(wf)` attaches ComfyUI native `SaveLatent` nodes (`29`, `49`, `69`, `89`)
  to every segment's raw `AV_LATENT` output (`Node 27`, `Node 48`, `Node 68`, `Node 88`).
- `build_partial_resume_workflow(wf, start_segment=3)` prunes earlier segments (`1..start_segment-1`)
  and replaces the upstream latent output node (e.g., `Node 48` for Segment 3) with a `LoadLatent` node
  reading the exact saved `.latent` tensor from disk in 0.1s!
"""

import argparse
import copy
import json
from pathlib import Path


SEGMENT_LATENT_OUTPUT_NODES = {
    1: ("27", "latents/OfficeDuel_Seg01_raw_av"),
    2: ("48", "latents/OfficeDuel_Seg02_raw_av"),
    3: ("68", "latents/OfficeDuel_Seg03_raw_av"),
    4: ("88", "latents/OfficeDuel_Seg04_raw_av"),
}

SEGMENT_NODE_GROUPS = {
    1: ["20", "21", "25", "26", "27", "29", "30", "31", "32", "34", "35"],
    2: ["40", "41", "42", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55"],
    3: ["60", "61", "62", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75"],
    4: ["80", "81", "82", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95"],
}


def inject_save_latent_checkpoints(wf: dict, prefix: str = "OfficeDuel_V6_OneShot") -> dict:
    """Attach native SaveLatent nodes to each segment's raw AV_LATENT output."""
    out = copy.deepcopy(wf)
    save_map = [
        ("29", "27", f"latents/{prefix}_Seg01_raw_av"),
        ("49", "48", f"latents/{prefix}_Seg02_raw_av"),
        ("69", "68", f"latents/{prefix}_Seg03_raw_av"),
        ("89", "88", f"latents/{prefix}_Seg04_raw_av"),
    ]
    for save_nid, src_nid, fname_prefix in save_map:
        if src_nid in out:
            out[save_nid] = {
                "class_type": "SaveLatent",
                "inputs": {
                    "samples": [src_nid, 0],
                    "filename_prefix": fname_prefix,
                },
            }
    return out


def build_partial_resume_workflow(
    wf: dict,
    start_segment: int,
    end_segment: int = 4,
    latent_filename: str = "OfficeDuel_V6_OneShot_Seg02_raw_av_00001_.latent",
) -> dict:
    """Build a partial workflow that resumes in 100% lossless raw AV_LATENT space from `start_segment`."""
    if start_segment not in (2, 3, 4):
        raise ValueError("start_segment must be 2, 3, or 4")

    out = copy.deepcopy(wf)
    prev_seg = start_segment - 1
    prev_latent_nid, _ = SEGMENT_LATENT_OUTPUT_NODES[prev_seg]

    # Prune all nodes belonging to segments < start_segment and > end_segment
    for seg_idx, nids in SEGMENT_NODE_GROUPS.items():
        if seg_idx < start_segment or seg_idx > end_segment:
            for nid in nids:
                out.pop(nid, None)

    # Replace the upstream segment's latent output node ID with a native LoadLatent node
    # (Notice: LoadLatent reads from ComfyUI/input/, so the runner symlinks output/latents/*.latent into input/)
    out[prev_latent_nid] = {
        "class_type": "LoadLatent",
        "inputs": {
            "latent": latent_filename,
        },
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject SaveLatent checkpoints or build partial resume workflow")
    parser.add_argument("--workflow", required=True, help="Path to 4-segment workflow JSON")
    parser.add_argument("--inject-checkpoints", action="store_true", help="Inject SaveLatent nodes into workflow")
    parser.add_argument("--resume-from", type=int, choices=[2, 3, 4], help="Generate partial workflow starting from segment N")
    parser.add_argument("--end-segment", type=int, default=4, choices=[2, 3, 4], help="Last segment to run")
    parser.add_argument("--latent-file", default="OfficeDuel_V6_OneShot_Seg02_raw_av_00001_.latent")
    parser.add_argument("--out", help="Output JSON path (defaults to in-place for --inject-checkpoints)")
    args = parser.parse_args()

    wf_path = Path(args.workflow)
    wf = json.loads(wf_path.read_text(encoding="utf-8"))

    if args.inject_checkpoints:
        wf = inject_save_latent_checkpoints(wf)
        dst = Path(args.out) if args.out else wf_path
        dst.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ 已注入 4 段 SaveLatent 潜空间快照节点 -> {dst}")

    if args.resume_from:
        partial = build_partial_resume_workflow(wf, args.resume_from, args.end_segment, args.latent_file)
        dst = Path(args.out) if args.out else wf_path.with_name(f"{wf_path.stem}_resume_seg{args.resume_from}to{args.end_segment}.json")
        dst.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ 已生成从第 {args.resume_from} 段至第 {args.end_segment} 段的无损潜空间断点续跑工作流 -> {dst}")


if __name__ == "__main__":
    main()
