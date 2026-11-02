#!/usr/bin/env python3
"""
Run Simon Willison's "Pelican Riding a Bicycle" Interactive SVG+JS Benchmark
against the local G4 DeepSeek-V4.1-Flash-1017 OpenAI Gateway (`http://127.0.0.1:8150/v1/chat/completions`),
measure prompt/reasoning/generation tokens & tok/s, and extract the standalone HTML artifact.
"""
import argparse
import json
import os
import re
import time
import urllib.request

PROMPT = (
    "Create a single self-contained HTML file (including inline CSS, SVG, and JavaScript) that renders "
    "an animated, detailed SVG illustration of a pelican riding a bicycle.\n"
    "Requirements:\n"
    "1. Detailed pelican anatomy: large characteristic throat pouch (gular pouch), long beak, expressive eye, "
    "feathered body, wings holding the handlebars, and webbed feet pedaling on the pedals.\n"
    "2. Complete bicycle geometry: diamond frame, spoked wheels that rotate smoothly, spinning pedals/crankset, "
    "chainstay, saddle, and handlebars.\n"
    "3. Smooth animation loop (CSS keyframes or requestAnimationFrame): rotating wheels/spokes, pedaling legs "
    "with knee/ankle articulation, subtle body bobbing, pouch jiggle, flapping wing feathers, and scrolling "
    "background scenery (clouds, road, parallax hills).\n"
    "4. Interactive controls bar at the bottom: Speed slider (0.2x to 3.0x), Day/Sunset/Night theme toggle, "
    "and a 'Squawk / Catch Fish' button with a fun visual animation.\n"
    "Return ONLY the complete `<!DOCTYPE html>...</html>` code inside a ```html code block."
)


def main():
    parser = argparse.ArgumentParser(description="Run Pelican Riding a Bicycle SVG+JS Benchmark on G4 DeepSeek-V4.1-Flash")
    parser.add_argument("--port", type=int, default=8150, help="Local OpenAI gateway port (default: 8150)")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v4.1-flash-1017-uncensored-gsq-rco-iq2_xxs",
        help="Model alias configured on serve_ds41.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pelican_output.html",
        help="Output path for the extracted HTML file",
    )
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 24576,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    print(f"Sending Pelican SVG+JS Benchmark request to {url} ...", flush=True)
    t0 = time.time()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    dt = time.time() - t0

    usage = res.get("usage", {})
    msg = res["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""

    comp_toks = usage.get("completion_tokens", 0)
    prompt_toks = usage.get("prompt_tokens", 0)
    print(
        f"✅ Completed in {dt:.2f}s | Prompt: {prompt_toks} toks | Generated: {comp_toks} toks "
        f"({comp_toks/max(0.1,dt):.2f} tok/s e2e)",
        flush=True,
    )
    print(f"Reasoning length: {len(reasoning)} chars | Content length: {len(content)} chars", flush=True)

    m = re.search(r"```html\s*(<!DOCTYPE html>.*?</html>)\s*```", content, flags=re.S | re.I)
    if not m:
        m = re.search(r"(<!DOCTYPE html>.*?</html>)", content, flags=re.S | re.I)
    html_code = m.group(1) if m else content

    out_path = os.path.abspath(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_code)
    print(f"Saved HTML ({len(html_code)} bytes) to {out_path}", flush=True)


if __name__ == "__main__":
    main()
