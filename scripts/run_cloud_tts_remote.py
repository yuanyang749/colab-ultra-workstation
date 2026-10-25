#!/usr/bin/env python3
"""Remote Controller for Standalone Cloud Dual-Engine TTS Studio (Qwen3-TTS-1.7B + IndexTTS-2, Zero ComfyUI Dependency).

Follows Colab Ultra Workstation Iron Rules:
- Uses `colab exec -s <session> -f <temp_file>` + `nohup` detached execution + `alarm 20` log polling (Iron Rule C).
- Uses Dual-Track CCU Meter (`colab_meter.py start` / `stop`).
- Pulls all synthesized `.wav` files back to local `/Users/delroy/Documents/Playground/test001/tts_takes/`.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/Users/delroy/Documents/Playground/test001")
SCRIPTS_DIR = ROOT / "colab-ultra-workstation/scripts"
LOCAL_TAKES_DIR = ROOT / "tts_takes"
ARTIFACT_DIR = Path("/Users/delroy/.gemini/antigravity/brain/d4ba0448-57f4-477a-b8cc-1cb22bab61d2")
COLAB_BIN = os.path.expanduser("~/.local/bin/colab")
COLAB_PY = os.path.expanduser("~/.local/share/uv/tools/google-colab-cli/bin/python")

ENV = dict(os.environ)
ENV.setdefault("https_proxy", "http://127.0.0.1:7890")
ENV.setdefault("http_proxy", "http://127.0.0.1:7890")
ENV.setdefault("no_proxy", "localhost,127.0.0.1")


def colab_exec_short(session: str, py_code: str, timeout_sec: int = 30, check: bool = False, retries: int = 3) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(py_code)
        tmp_path = tf.name
    try:
        for attempt in range(retries):
            cmd = ["perl", "-e", f"alarm {timeout_sec}; exec @ARGV", COLAB_BIN, "exec", "-s", session, "-f", tmp_path]
            r = subprocess.run(cmd, text=True, capture_output=True, env=ENV)
            if r.returncode == 0:
                return r.stdout
            if attempt < retries - 1:
                time.sleep(2.5)
                continue
            if check:
                raise RuntimeError(f"❌ colab exec 失败 (exit={r.returncode}):\n{r.stderr}\n{r.stdout}")
            if r.stderr:
                print(f"⚠️ [colab exec] {r.stderr.strip()[-300:]}", flush=True)
        return r.stdout
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def mount_drive_with_retry(session_name: str, retries: int = 3) -> None:
    for i in range(retries):
        chk = colab_exec_short(
            session_name,
            'import os; print("MOUNTED_OK" if os.path.isdir("/content/drive/My Drive") or os.path.isdir("/content/drive/MyDrive") else "NOT_MOUNTED")',
        )
        if "MOUNTED_OK" in chk:
            print("✅ Google Drive 已挂载就绪！", flush=True)
            return
        print(f"📂 正在静默挂载 20TB Google Drive (尝试 {i+1}/{retries})...", flush=True)
        r = subprocess.run([COLAB_BIN, "dfs", "mount", "-s", session_name], env=ENV)
        if r.returncode == 0:
            return
        time.sleep(3)
    raise RuntimeError("❌ Google Drive 挂载失败")


def ensure_cloud_session(gpu: str = "L4", session_name: str = "tts-studio") -> str:
    """Reuse active Colab GPU session or spin up a lightweight one."""
    code = """
import json, time
from colab_cli.common import state
for i in range(4):
    try:
        sessions, assignments = state.sync_sessions()
        print(json.dumps({"assignments": [{"endpoint": a.endpoint} for a in assignments],
                          "sessions": [{"name": s.name, "endpoint": s.endpoint} for s in sessions.values()]}))
        break
    except Exception as e:
        if i == 3: raise
        time.sleep(2)
"""
    r = subprocess.run([COLAB_PY, "-c", code], text=True, capture_output=True, check=True, env=ENV)
    info = json.loads(r.stdout.strip().splitlines()[-1])
    if info["assignments"]:
        active_ep = info["assignments"][0]["endpoint"]
        matched = [s["name"] for s in info["sessions"] if s["endpoint"] == active_ep]
        sname = matched[0] if matched else session_name
        print(f"⚡ 复用已开启的云端实例: {sname} ({active_ep})", flush=True)
        mount_drive_with_retry(sname)
        return sname

    print(f"🚀 当前无活跃实例，正在启动独立双引擎 TTS 云端实例 (规格: {gpu.upper()})...", flush=True)
    subprocess.run([COLAB_BIN, "new", "-s", session_name, "--gpu", gpu.upper()], check=True, env=ENV)
    subprocess.run([COLAB_PY, str(SCRIPTS_DIR / "colab_meter.py"), "start", "-s", session_name, "-g", gpu.upper()], check=False, env=ENV)
    mount_drive_with_retry(session_name)
    return session_name


def push_and_run_tts(session: str, mode: str = "hybrid", linwei_instruct: str = "", temperature: float = 0.42) -> None:
    studio_src = (SCRIPTS_DIR / "launch_cloud_tts_studio.py").read_bytes()
    b64_src = base64.b64encode(studio_src).decode("ascii")

    # 1. Upload script and launch via detached nohup (Iron Rule C)
    launch_py = f"""
import base64, json, subprocess, sys
from pathlib import Path

raw = base64.b64decode("{b64_src}")
Path("/tmp/launch_cloud_tts_studio.py").write_bytes(raw)
drv_script = Path("/content/drive/MyDrive/AI_Workstation/scripts/launch_cloud_tts_studio.py")
if drv_script.parent.exists():
    drv_script.write_bytes(raw)

Path("/tmp/tts_job_done.flag").unlink(missing_ok=True)
Path("/tmp/tts_job_err.flag").unlink(missing_ok=True)

runner = '''import subprocess, sys
from pathlib import Path
cmd = [
    sys.executable, "-u", "/tmp/launch_cloud_tts_studio.py",
    "--run-preset", "lounge_duel_mature",
    "--mode", {json.dumps(mode)},
    "--temperature", "{temperature}",
]
lin_inst = {json.dumps(linwei_instruct)}
if lin_inst:
    cmd.extend(["--linwei-instruct", lin_inst])
p = subprocess.run(cmd)
if p.returncode == 0:
    Path("/tmp/tts_job_done.flag").write_text("OK")
else:
    Path("/tmp/tts_job_err.flag").write_text(str(p.returncode))
'''
Path("/tmp/run_tts_job.py").write_text(runner)
subprocess.Popen("nohup python3 -u /tmp/run_tts_job.py > /tmp/tts_job.log 2>&1 &", shell=True)
print("LAUNCHED_DETACHED_OK")
"""
    out = colab_exec_short(session, launch_py, timeout_sec=25, check=True)
    if "LAUNCHED_DETACHED_OK" not in out:
        raise RuntimeError(f"❌ 远端启动任务失败:\n{out}")
    print(f"🎙️ 云端独立双引擎 TTS 任务已脱机启动 (mode={mode})", flush=True)

    # 2. Poll progress every 6 seconds (max 100 polls = 10 min cap)
    t0 = time.time()
    last_lines = []
    consec_failures = 0
    for _ in range(100):
        time.sleep(6)
        poll_py = """
import json
from pathlib import Path
log_txt = Path("/tmp/tts_job.log").read_text(errors="ignore") if Path("/tmp/tts_job.log").exists() else ""
d_txt = Path("/tmp/cloud_tts_daemon.log").read_text(errors="ignore") if Path("/tmp/cloud_tts_daemon.log").exists() else ""
state = "DONE" if Path("/tmp/tts_job_done.flag").exists() else ("ERR" if Path("/tmp/tts_job_err.flag").exists() else "RUNNING")
print("POLL_JSON:" + json.dumps({"state": state, "job_tail": log_txt.splitlines()[-4:], "daemon_tail": d_txt.splitlines()[-3:]}))
"""
        p_out = colab_exec_short(session, poll_py, timeout_sec=25)
        if "POLL_JSON:" not in p_out:
            consec_failures += 1
            if consec_failures >= 5:
                raise RuntimeError("❌ 连续 5 次轮询未响应，已中止防卡死。")
            continue
        consec_failures = 0
        data = json.loads(p_out.split("POLL_JSON:", 1)[1].splitlines()[0])
        combined = data["daemon_tail"] + data["job_tail"]
        if combined != last_lines:
            print(f"  [{time.time()-t0:.0f}s] " + " | ".join(combined[-3:]), flush=True)
            last_lines = combined
        if data["state"] == "DONE":
            print(f"✅ 云端双引擎 TTS 合成完成！总耗时: {time.time()-t0:.1f}s", flush=True)
            for l in data["job_tail"]:
                print("  " + l, flush=True)
            return
        elif data["state"] == "ERR":
            full_err = colab_exec_short(
                session,
                'from pathlib import Path; print(Path("/tmp/tts_job.log").read_text()[-1500:]); print(Path("/tmp/cloud_tts_daemon.log").read_text()[-1500:] if Path("/tmp/cloud_tts_daemon.log").exists() else "")',
                timeout_sec=25,
            )
            raise RuntimeError(f"❌ 云端 TTS 任务异常:\n{full_err}")
    raise TimeoutError("❌ 云端 TTS 任务超过最大等待时间。")


def pull_wavs_to_local(session: str) -> None:
    LOCAL_TAKES_DIR.mkdir(parents=True, exist_ok=True)
    remote_pack = """
import base64, json
from pathlib import Path
out = {}
for p in sorted(Path("/tmp/tts_takes").glob("*.wav")):
    out[p.name] = base64.b64encode(p.read_bytes()).decode("ascii")
print("TTS_PACK_START:" + json.dumps(out) + ":TTS_PACK_END")
"""
    stdout = colab_exec_short(session, remote_pack, timeout_sec=35, check=True)
    if "TTS_PACK_START:" not in stdout:
        raise RuntimeError(f"❌ 未能从云端提取音频包:\n{stdout}")
    payload = stdout.split("TTS_PACK_START:", 1)[1].split(":TTS_PACK_END", 1)[0]
    files_dict = json.loads(payload)
    for fname, b64_data in files_dict.items():
        raw_bytes = base64.b64decode(b64_data)
        dst = LOCAL_TAKES_DIR / fname
        dst.write_bytes(raw_bytes)
        if ARTIFACT_DIR.exists():
            (ARTIFACT_DIR / fname).write_bytes(raw_bytes)
        print(f"📥 已拉取至本地: {dst.name} ({len(raw_bytes) / 1024:.1f} KB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Standalone Cloud Dual-Engine TTS Studio from Local Mac")
    parser.add_argument("--gpu", type=str, default="L4", help="GPU tier if starting a new session (T4 / L4 / G4)")
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "qwen", "indextts"],
        help="Synthesis mode: hybrid (Qwen3 mature seed -> IndexTTS-2 dialogue), qwen, or indextts",
    )
    parser.add_argument("--linwei-instruct", type=str, default="", help="Custom voice design prompt for Lin Wei")
    parser.add_argument("--temperature", type=float, default=0.42, help="Sampling temperature (default: 0.42)")
    parser.add_argument("--stop-after", action="store_true", help="Immediately stop cloud instance after pulling WAVs")
    args = parser.parse_args()

    sname = ensure_cloud_session(gpu=args.gpu)
    try:
        push_and_run_tts(session=sname, mode=args.mode, linwei_instruct=args.linwei_instruct, temperature=args.temperature)
        pull_wavs_to_local(session=sname)
    finally:
        if args.stop_after:
            print("🛑 正在自动释放云端 TTS 实例...", flush=True)
            subprocess.run([COLAB_PY, str(SCRIPTS_DIR / "colab_meter.py"), "stop", "-s", sname], check=False, env=ENV)


if __name__ == "__main__":
    main()
