#!/usr/bin/env python3
"""Generate zero-browser Google Drive mount payload for colab-mcp sessions.

Usage:
  1. Generate mount payload (inject & run via colab-mcp):
     python3 scripts/mcp_dfs_mount.py
  2. Generate credential-free verification cell (update_cell after mount succeeds):
     python3 scripts/mcp_dfs_mount.py --sanitize
"""

import argparse
import base64
import json
import os
import sys

STORE = os.path.expanduser("~/.config/colab-cli/dfs_credentials.json")

SANITIZED_CELL = '''# === [Cell 2] 20TB/30TB Google Drive 零弹窗挂载状态与目录初始化 (凭证已安全擦除) ===
import sys
sys.path.insert(0, "/content/drive/MyDrive/AI_Workstation")
from workstation_init import ensure_drive_mounted

DIRS = ensure_drive_mounted("/content/drive")
'''

MOUNT_TEMPLATE = '''# === [Cell 2] colab-dfs-ext 零浏览器静默挂载 Google Drive & 目录初始化 ===
import base64, http.server, json, os, shutil, socket, subprocess, sys, time
import urllib.parse, urllib.request
from pathlib import Path

DFS_B64 = "__DFS_B64__"

CFG = json.loads(base64.b64decode(DFS_B64).decode())
MP, PORT, TIMEOUT = CFG["mountpoint"], int(CFG["port"]), int(CFG["timeout"])
DRIVE_DIR, DRIVE = "/opt/google/drive", "/opt/google/drive/drive"
SHIM_CFG, SHIM_SRC, SHIM_LOG = "/tmp/dfs_creds.json", "/tmp/dfs_creds_shim.py", "/tmp/dfs_creds_shim.log"
DFS_LOG = "/root/.config/Google/DriveFS/Logs/drive_fs.txt"

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")

def mounted():
    return os.path.isdir(os.path.join(MP, "My Drive")) or os.path.isdir(os.path.join(MP, "MyDrive"))

if not mounted():
    try:
        import psutil
        for _p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if _p.info["name"] == "drive" or "dfs_creds_shim" in " ".join(_p.info["cmdline"] or []):
                    _p.kill()
            except Exception:
                pass
    except ImportError:
        sh('pkill -9 -x drive; pkill -9 -f "[d]fs_creds_shim.py"')
    time.sleep(1)
    sh(f'umount -f "{MP}" 2>/dev/null; umount "{MP}" 2>/dev/null')
    if os.path.isdir(MP):
        sh(f'rm -rf "{MP}"')
    os.makedirs(MP, exist_ok=True)
    if CFG.get("purge_state", True):
        sh('rm -rf /root/.config/Google/DriveFS')

    SHIM = r\'\'\'import http.server, json, os, threading, time, urllib.parse, urllib.request
CFG = json.load(open("/tmp/dfs_creds.json"))
LOCK = threading.Lock()
CACHE = {"tok": CFG.get("access_token"), "until": CFG.get("expires_at", 0)}

def mint():
    now = time.time()
    with LOCK:
        if CACHE["tok"] and CACHE["until"] - 120 > now:
            return CACHE["tok"]
        payload = {"grant_type": "refresh_token", "refresh_token": CFG["refresh_token"], "client_id": CFG["client_id"]}
        if CFG.get("client_secret"):
            payload["client_secret"] = CFG["client_secret"]
        req = urllib.request.Request(
            CFG.get("token_uri", "https://oauth2.googleapis.com/token"),
            urllib.parse.urlencode(payload).encode(),
            {"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
        CACHE["tok"] = data["access_token"]
        CACHE["until"] = now + int(data.get("expires_in", 3599))
        return CACHE["tok"]

class H(http.server.BaseHTTPRequestHandler):
    def _send(self, body, ctype="text/plain"):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Metadata-Flavor", "Google")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path
        try:
            if p.endswith("/instance/service-accounts/default/token"):
                self._send(json.dumps({
                    "access_token": mint(),
                    "expires_in": max(60, int(CACHE["until"] - time.time())),
                    "token_type": "Bearer",
                    "scope": " ".join(CFG.get("scopes", [])),
                }), "application/json")
            elif p.endswith("/guest-attributes/auth/user-id"):
                self._send(CFG["user_id"])
            elif p.endswith("/email"):
                self._send(CFG.get("email", ""))
            elif p.endswith("/guest-attributes/auth/scopes"):
                self._send(" ".join(CFG.get("scopes", [])))
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
        except Exception as exc:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(exc).encode())
        with open("/tmp/dfs_creds_shim.log", "a") as f:
            f.write(time.strftime("%H:%M:%S ") + p + "\\n")

    def log_message(self, *a):
        pass

srv = http.server.ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("SHIM_PORT", "8012"))), H)
srv.serve_forever()
\'\'\'
    def port_free(p):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return True
            except OSError:
                return False

    PORT = next((p for p in range(PORT, PORT + 30) if port_free(p)), 8012)
    json.dump(CFG, open(SHIM_CFG, "w"))
    os.chmod(SHIM_CFG, 0o600)
    open(SHIM_SRC, "w").write(SHIM)
    subprocess.Popen([sys.executable, SHIM_SRC], stdout=open(SHIM_LOG, "w"),
                     stderr=subprocess.STDOUT, start_new_session=True,
                     env=dict(os.environ, SHIM_PORT=str(PORT)))
    probe = f"curl -sS -m 5 -H 'Metadata-Flavor: Google' http://127.0.0.1:{PORT}/computeMetadata/v1/instance/service-accounts/default/token"
    for _ in range(20):
        out = sh(probe)
        if '"access_token"' in out:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError(f"Shim failed to start: {out} | log: {sh('cat ' + SHIM_LOG)}")

    features = (f"crash_throttle_percentage:100,fuse_max_background:1000,max_read_qps:1000,"
                f"max_write_qps:1000,max_operation_batch_size:15,max_parallel_push_task_instances:10,"
                f"opendir_timeout_ms:{TIMEOUT * 1000},virtual_folders_omit_spaces:true,read_only_mode:{'true' if CFG.get('readonly') else 'false'}")
    prefs = f"trusted_root_certs_file_path:{DRIVE_DIR}/roots.pem,feature_flag_restart_seconds:129600,mount_point_path:{MP}"
    auth_flag = f"--metadata_server_auth_uri=127.0.0.1:{PORT}/computeMetadata/v1"
    cmd = [DRIVE, f"--features={features}", auth_flag, f"--preferences={prefs}", MP]
    proc = subprocess.Popen(cmd, stdout=open("/tmp/dfs_drive.log", "w"), stderr=subprocess.STDOUT,
                            start_new_session=True,
                            env={"HOME": os.environ.get("HOME", "/root"), "FUSE_DEV_NAME": "/dev/fuse",
                                 "PATH": "/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin:."})
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        if mounted():
            break
        if proc.poll() is not None:
            err_detail = sh(f'grep -E "E \\\\[|CELLO|CANNOT_|Status|error|Error|Failed" "{DFS_LOG}" 2>/dev/null | tail -n 35')
            raise RuntimeError(f"DriveFS exited early (rc={proc.poll()}):\\n{err_detail}")
        time.sleep(1)

if os.path.isdir(os.path.join(MP, "My Drive")) and not os.path.exists(os.path.join(MP, "MyDrive")):
    try:
        os.symlink(os.path.join(MP, "My Drive"), os.path.join(MP, "MyDrive"))
    except Exception:
        pass

req = urllib.request.Request(
    f"http://127.0.0.1:{PORT}/computeMetadata/v1/instance/service-accounts/default/token",
    headers={"Metadata-Flavor": "Google"}
)
tok = json.load(urllib.request.urlopen(req, timeout=5))["access_token"]
about_req = urllib.request.Request(
    "https://www.googleapis.com/drive/v3/about?fields=user(emailAddress,displayName),storageQuota",
    headers={"Authorization": f"Bearer {tok}"}
)
about = json.load(urllib.request.urlopen(about_req, timeout=10))
quota = about.get("storageQuota", {})
limit_b, usage_b = int(quota.get("limit", 0)), int(quota.get("usage", 0))
free_b = max(0, limit_b - usage_b)
print(f"🎉 [colab-dfs-ext 零弹窗挂载成功] 账号: {about['user']['emailAddress']} (Shim Port: {PORT})")
print(f"☁️ Google Drive 云端真实容量: 总计 {limit_b / (1024**4):.2f} TB | 已用 {usage_b / (1024**3):.2f} GB | 剩余 {free_b / (1024**4):.2f} TB")

WORKSTATION_ROOT = Path("/content/drive/MyDrive/AI_Workstation")
for sub in ("models", "loras", "hf_cache", "outputs"):
    p = WORKSTATION_ROOT / sub
    p.mkdir(parents=True, exist_ok=True)
    print(f"📁 [{sub}] 就绪: {p}")
os.environ["HF_HOME"] = str(WORKSTATION_ROOT / "hf_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(WORKSTATION_ROOT / "hf_cache" / "hub")
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mountpoint", default="/content/drive")
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--readonly", action="store_true")
    ap.add_argument("--sanitize", action="store_true", help="Print credential-free verification cell")
    args = ap.parse_args()

    if args.sanitize:
        print(SANITIZED_CELL)
        return

    if not os.path.exists(STORE):
        sys.exit(f"Missing {STORE}; run `colab dfs consent` first.")
    data = json.load(open(STORE))
    cfg = {
        "mode": "shim",
        "mountpoint": args.mountpoint,
        "port": args.port,
        "force": False,
        "timeout": args.timeout,
        "readonly": args.readonly,
        "refresh_token": data["refresh_token"],
        "client_id": data["client_id"],
        "client_secret": data.get("client_secret", ""),
        "token_uri": data.get("token_uri", "https://oauth2.googleapis.com/token"),
        "scopes": data.get("scopes", ["https://www.googleapis.com/auth/drive"]),
        "user_id": data["user_id"],
        "email": data.get("email", ""),
        "drive_extra": [],
        "purge_state": True,
    }
    b64 = base64.b64encode(json.dumps(cfg).encode()).decode()
    print(MOUNT_TEMPLATE.replace("__DFS_B64__", b64))


if __name__ == "__main__":
    main()
