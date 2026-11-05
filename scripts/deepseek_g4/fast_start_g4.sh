#!/usr/bin/env bash
# ==============================================================================
# One-Command Daily Fast-Boot Launcher for Colab G4 (RTX PRO 6000 Blackwell 96GB)
# Supported Flagship Variants:
#   - ista-coder-iq2xxs : ISTA-DASLab GSQ-RCO-Coder (72.12 GB Backbone 100% VRAM + 110.60 GB Full IQ4_NL Engram 100% DDR5, 54.63 tok/s) [DEFAULT]
#   - unsloth-ud-iq2xxs : Unsloth Dynamic V3.1 UD-IQ2_XXS (165.30 GB, 47.41 tok/s)
#   - abiray-gsq-iq2xxs : Abiray Uncensored GSQ-RCO IQ2_XXS (138.59 GB, 51.49 tok/s)
#   - unsloth-ud-iq1m   : Unsloth UD-IQ1_M (155.83 GB)
#
# Built-in Anti-Disconnect Protection:
#   - Automatically starts a local 10-minute Jupyter Kernel Anti-Idle Keepalive
#     daemon (`colab_keepalive_<session>`) on `start` (or `keepalive`) and
#     cleanly terminates it on `stop`, preventing Colab's 30-60 min Jupyter
#     idle reaper from reclaiming the GPU while chatting over API Key / Tunnel.
# ==============================================================================
set -euo pipefail

export https_proxy="${https_proxy:-http://127.0.0.1:7890}"
export http_proxy="${http_proxy:-http://127.0.0.1:7890}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"

SESSION="${SESSION:-ds_g4}"
VARIANT="${VARIANT:-ista-coder-iq2xxs}"
QUALITY_MODE="${QUALITY_MODE:-exact}"
CTX_SIZE="${CTX_SIZE:-131072}"
PORT="${PORT:-8150}"
GATE_PORT="$((PORT + 10000))"
DOMAIN_FILE="${HOME}/.config/colab-cli/colab_model_domain.txt"
TOKEN_FILE="${HOME}/.config/colab-cli/colab_model_tunnel_token.txt"
API_KEY_FILE="${HOME}/.config/colab-cli/colab_model_api_key.txt"
PUBLIC_DOMAIN="${PUBLIC_DOMAIN:-$( [[ -s "$DOMAIN_FILE" ]] && tr -d '[:space:]' < "$DOMAIN_FILE" || echo "your-gateway.example.com" )}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../colab_meter.py" ]]; then
  METER_PY="${SCRIPT_DIR}/../colab_meter.py"
else
  METER_PY="${HOME}/.agents/skills/colab-ultra-workstation/scripts/colab_meter.py"
fi
COLAB_BIN="$(command -v colab || echo "${HOME}/.local/bin/colab")"
COLAB_PY="$(head -n 1 "${COLAB_BIN}" | cut -c 3-)"
EXTRA_ARGS=()

ensure_local_api_key() {
  mkdir -p "${HOME}/.config/colab-cli"
  if [[ ! -s "$API_KEY_FILE" ]]; then
    python3 -c 'import secrets; print("sk-g4-" + secrets.token_hex(24))' > "$API_KEY_FILE"
    chmod 600 "$API_KEY_FILE"
  fi
}

start_keepalive_daemon() {
  if [[ -f "/tmp/colab_keepalive_${SESSION}.pid" ]]; then
    kill "$(cat "/tmp/colab_keepalive_${SESSION}.pid")" 2>/dev/null || true
  fi
  pkill -f "colab_keepalive_${SESSION}" 2>/dev/null || true
  nohup bash -c "exec -a colab_keepalive_${SESSION} bash -c 'while true; do sleep 600; echo \"1\" | perl -e \"alarm 20; exec @ARGV\" \"${COLAB_BIN}\" exec -s ${SESSION} >/dev/null 2>&1 || true; done'" >/tmp/colab_keepalive_${SESSION}.log 2>&1 &
  echo $! > "/tmp/colab_keepalive_${SESSION}.pid"
  echo ">>> Local Colab 10-min Anti-Idle Keepalive Daemon started for [${SESSION}] (PID $(cat "/tmp/colab_keepalive_${SESSION}.pid"))"
}

stop_keepalive_daemon() {
  echo ">>> Stopping local Colab keepalive daemon for [$SESSION]..."
  if [[ -f "/tmp/colab_keepalive_${SESSION}.pid" ]]; then
    kill "$(cat "/tmp/colab_keepalive_${SESSION}.pid")" 2>/dev/null || true
    rm -f "/tmp/colab_keepalive_${SESSION}.pid"
  fi
  pkill -f "colab_keepalive_${SESSION}" 2>/dev/null || true
}

usage() {
  cat <<EOF
Usage:
  $(basename "$0") start [--variant ista-coder-iq2xxs|unsloth-ud-iq2xxs|abiray-gsq-iq2xxs] [--vision] [--quality exact|turbo] [--ctx 131072] [--auth-caddy]
  $(basename "$0") switch [--variant ista-coder-iq2xxs|unsloth-ud-iq2xxs|abiray-gsq-iq2xxs] [--vision] [--quality exact|turbo] [--ctx 131072] [--auth-caddy]
  $(basename "$0") tunnel [--port 8150]                                       # Attach Protected HTTPS Gateway to any running model on G4
  $(basename "$0") keepalive [--session ds_g4]                                # Start/restart local 10-min Jupyter anti-idle heartbeat daemon
  $(basename "$0") key [--rotate]                                             # Show (in your terminal) or rotate your secret API Key
  $(basename "$0") status                                                     # Check server health, HTTPS gateway & GPU/RAM telemetry
  $(basename "$0") stop                                                       # Stop keepalive + G4 immediately & print CCU bill
EOF
}

ACTION="${1:-start}"
shift || true

ROTATE_KEY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant|-v) VARIANT="$2"; shift 2 ;;
    ista-coder-iq2xxs|ista-coder-engram|unsloth-ud-iq2xxs|abiray-gsq-iq2xxs|unsloth-ud-iq1m)
      VARIANT="$1"; shift ;;
    --session|-s) SESSION="$2"; shift 2 ;;
    --quality|--quality-mode|-q) QUALITY_MODE="$2"; shift 2 ;;
    --ctx)        CTX_SIZE="$2"; shift 2 ;;
    --port|-p)    PORT="$2"; GATE_PORT="$((PORT + 10000))"; shift 2 ;;
    --vision|--mmproj) EXTRA_ARGS+=("--vision"); shift ;;
    --auth-caddy) EXTRA_ARGS+=("--auth-caddy"); shift ;;
    --tunnel)     EXTRA_ARGS+=("--tunnel" "${2:-quick}"); shift 2 ;;
    --rotate)     ROTATE_KEY=1; shift ;;
    *)            echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

ensure_local_api_key

if [[ "$ACTION" == "key" ]]; then
  if [[ "$ROTATE_KEY" -eq 1 ]]; then
    python3 -c 'import secrets; print("sk-g4-" + secrets.token_hex(24))' > "$API_KEY_FILE"
    chmod 600 "$API_KEY_FILE"
    echo "✅ Generated a new 192-bit API key in $API_KEY_FILE"
  fi
  echo "🔑 Your secret API Key (stored in $API_KEY_FILE):"
  cat "$API_KEY_FILE"
  exit 0
fi

if [[ "$ACTION" == "keepalive" ]]; then
  start_keepalive_daemon
  exit 0
fi

launch_cf_tunnel_on_remote() {
  if [[ ! -s "$TOKEN_FILE" ]]; then
    echo "⚠️ Cloudflare Tunnel token not found at $TOKEN_FILE; skipping public gateway."
    return 0
  fi
  local CF_TOKEN API_KEY
  CF_TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
  API_KEY="$(tr -d '[:space:]' < "$API_KEY_FILE")"
  echo ">>> Starting Bearer-Auth Gate (127.0.0.1:${GATE_PORT} -> 127.0.0.1:${PORT}) + Cloudflare Tunnel..."
  cat <<EOF | perl -e 'alarm 35; exec @ARGV' "${COLAB_BIN}" exec -s "$SESSION"
import os, shutil, subprocess

drive_bin = "/content/drive/MyDrive/AI_Models/DeepSeek_G4_Cache/cloudflared"
local_bin = "/usr/local/bin/cloudflared"

if not os.path.exists(local_bin):
    if os.path.exists(drive_bin) and os.path.getsize(drive_bin) > 10_000_000:
        shutil.copy2(drive_bin, local_bin)
        os.chmod(local_bin, 0o755)
    else:
        subprocess.run(
            "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
            shell=True, check=True
        )
        if os.path.exists(os.path.dirname(drive_bin)):
            shutil.copy2(local_bin, drive_bin)

gate_code = '''#!/usr/bin/env python3
import hmac, http.client, os, socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer

EXPECTED_KEY = os.environ["COLAB_GATE_API_KEY"].strip()
UPSTREAM_PORT = int(os.environ.get("COLAB_UPSTREAM_PORT", "8150"))
GATE_PORT = int(os.environ.get("COLAB_GATE_PORT", "18150"))

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class AuthProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *args):
        pass

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Key, x-api-key, anthropic-version")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _check_auth(self):
        auth = self.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if not token:
            token = (self.headers.get("x-api-key") or self.headers.get("X-API-Key") or "").strip()
        if not token or not hmac.compare_digest(token, EXPECTED_KEY):
            body = b'{"error":{"message":"Unauthorized: valid Bearer API key required","type":"authentication_error","code":401}}'
            self.send_response(401)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False
        return True

    def _proxy(self, method):
        if not self._check_auth():
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        req_body = self.rfile.read(length) if length > 0 else None
        conn = http.client.HTTPConnection("127.0.0.1", UPSTREAM_PORT, timeout=900)
        try:
            fwd_headers = {
                k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "connection", "transfer-encoding")
            }
            conn.request(method, self.path, body=req_body, headers=fwd_headers)
            resp = conn.getresponse()
            if resp.status == 400 and req_body:
                err_Preview = resp.read()
                import json, re
                m = re.search(rb"at most (\\d+) here", err_Preview)
                if m and int(m.group(1)) > 0:
                    room = int(m.group(1))
                    try:
                        j = json.loads(req_body.decode("utf-8"))
                        if "max_tokens" in j:
                            j["max_tokens"] = room
                        if "max_completion_tokens" in j:
                            j["max_completion_tokens"] = room
                        new_body = json.dumps(j).encode("utf-8")
                        fwd_headers["Content-Length"] = str(len(new_body))
                        conn.close()
                        conn = http.client.HTTPConnection("127.0.0.1", UPSTREAM_PORT, timeout=900)
                        conn.request(method, self.path, body=new_body, headers=fwd_headers)
                        resp = conn.getresponse()
                    except Exception:
                        self.send_response(400)
                        self._send_cors()
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(err_Preview)))
                        self.end_headers()
                        self.wfile.write(err_Preview)
                        return
                else:
                    self.send_response(400)
                    self._send_cors()
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_Preview)))
                    self.end_headers()
                    self.wfile.write(err_Preview)
                    return
            self.send_response(resp.status)
            self._send_cors()
            for k, v in resp.getheaders():
                if k.lower() in ("transfer-encoding", "connection", "access-control-allow-origin"):
                    continue
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = resp.read1(4096) if hasattr(resp, "read1") else resp.read(256)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception as e:
            err = f'{{"error":{{"message":"Upstream G4 model server is still loading weights ({e})","code":503}}}}'.encode()
            try:
                self.send_response(503)
                self._send_cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
            except Exception:
                pass
        finally:
            conn.close()

    def do_GET(self):
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")

if __name__ == "__main__":
    ThreadedHTTPServer(("127.0.0.1", GATE_PORT), AuthProxyHandler).serve_forever()
'''
with open("/tmp/cf_auth_gate.py", "w") as f:
    f.write(gate_code)

subprocess.run("pkill -f '/tmp/cf_auth_gate.py' || true", shell=True)
subprocess.run("pkill -f 'cloudflared tunnel' || true", shell=True)

env = os.environ.copy()
env["COLAB_GATE_API_KEY"] = "${API_KEY}"
env["COLAB_UPSTREAM_PORT"] = "${PORT}"
env["COLAB_GATE_PORT"] = "${GATE_PORT}"

subprocess.Popen(
    ["python3", "-u", "/tmp/cf_auth_gate.py"],
    stdout=open("/tmp/cf_auth_gate.log", "w"),
    stderr=subprocess.STDOUT,
    env=env,
    start_new_session=True,
)

cf_env = os.environ.copy()
cf_env["TUNNEL_TOKEN"] = "${CF_TOKEN}"
subprocess.Popen(
    [local_bin, "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:${GATE_PORT}", "run"],
    stdout=open("/tmp/cf_tunnel.log", "w"),
    stderr=subprocess.STDOUT,
    env=cf_env,
    start_new_session=True,
)
print("Protected Cloudflare Gateway active (Bearer API Key enforced on port ${GATE_PORT} -> ${PORT})")
EOF
}

if [[ "$ACTION" == "stop" ]]; then
  stop_keepalive_daemon
  echo ">>> Stopping Colab G4 session [$SESSION] and releasing GPU..."
  "${COLAB_BIN}" stop -s "$SESSION" || true
  "$COLAB_PY" "$METER_PY" stop --session "$SESSION" || true
  "$COLAB_PY" "$METER_PY" report || true
  exit 0
fi

if [[ "$ACTION" == "tunnel" ]]; then
  launch_cf_tunnel_on_remote
  start_keepalive_daemon
  exit 0
fi

if [[ "$ACTION" == "switch" ]]; then
  echo "=== [1/2] Uploading Step 3 Launcher & Freeing Non-Active NVMe Shards on [$SESSION] ==="
  "${COLAB_BIN}" upload -s "$SESSION" "${SCRIPT_DIR}/step3_g4_hybrid_stage_and_benchmark.py" /content/step3_g4_hybrid_stage_and_benchmark.py
  cat <<EOF | perl -e 'alarm 25; exec @ARGV' "${COLAB_BIN}" exec -s "$SESSION"
import glob, os, subprocess, time
subprocess.run("pkill -9 -f 'serve_ds41.py|ds41_generate' || true", shell=True)
time.sleep(1.5)
variant = "${VARIANT}"
if variant in ("ista-coder-iq2xxs", "ista-coder-engram"):
    target_sub = "ista_daslab_coder"
elif variant == "unsloth-ud-iq2xxs":
    target_sub = "unsloth_v31/UD-IQ2_XXS"
elif variant == "unsloth-ud-iq1m":
    target_sub = "unsloth/UD-IQ1_M"
else:
    target_sub = "abiray_gsq_uncensored"
for p in glob.glob("/content/fast_deepseek/**/*.gguf", recursive=True):
    if target_sub and target_sub not in p and "V41-1017-MTP" not in p and "engram" not in p.lower():
        os.remove(p)
cmd = (
    "nohup python3 -u /content/step3_g4_hybrid_stage_and_benchmark.py "
    "--variant ${VARIANT} --mode delta --quality-mode ${QUALITY_MODE} --ctx ${CTX_SIZE} --port ${PORT} ${EXTRA_ARGS[*]:-} "
    "> /tmp/delta_g4_server.log 2>&1 < /dev/null &"
)
subprocess.run(cmd, shell=True, check=True)
print("Hot-switching G4 session [${SESSION}] to variant [${VARIANT}] -> logging to /tmp/delta_g4_server.log")
EOF
  exit 0
fi

if [[ "$ACTION" == "status" ]]; then
  local_key="$(tr -d '[:space:]' < "$API_KEY_FILE")"
  echo "=== Public Gateway Check (https://<configured-domain>/health with Bearer Key) ==="
  curl -sS --max-time 5 -H "Authorization: Bearer ${local_key}" "https://${PUBLIC_DOMAIN}/health" || echo "(Gateway still warming up or server loading)"
  echo ""
  echo "=== Local Keepalive Daemon Status ==="
  if [[ -f "/tmp/colab_keepalive_${SESSION}.pid" ]] && kill -0 "$(cat "/tmp/colab_keepalive_${SESSION}.pid")" 2>/dev/null; then
    echo "🟢 Keepalive daemon running for [${SESSION}] (PID $(cat "/tmp/colab_keepalive_${SESSION}.pid"), 10-min heartbeat)"
  else
    echo "⚪ Keepalive daemon not running for [${SESSION}] (run '$(basename "$0") keepalive -s ${SESSION}' to start)"
  fi
  echo ""
  echo "=== Colab G4 Internal Status ==="
  perl -e 'alarm 25; exec @ARGV' bash -c "echo '!curl -s http://127.0.0.1:${PORT}/health; echo; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader; tail -n 15 /tmp/delta_g4_server.log 2>/dev/null' | \"${COLAB_BIN}\" exec -s ${SESSION}"
  exit 0
fi

echo "=== [1/5] Allocating Colab G4 (RTX PRO 6000 Blackwell 96GB + 188GB DDR5) ==="
"$COLAB_PY" "$METER_PY" start --session "$SESSION" --gpu G4 || true
"${COLAB_BIN}" new -s "$SESSION" --gpu G4

echo "=== [2/5] Zero-Browser Mounting Google Drive (20TB/30TB) ==="
"${COLAB_BIN}" dfs mount -s "$SESSION"

echo "=== [3/5] Uploading Optimized Step 3 Launcher (${QUALITY_MODE} mode) ==="
"${COLAB_BIN}" upload -s "$SESSION" "${SCRIPT_DIR}/step3_g4_hybrid_stage_and_benchmark.py" /content/step3_g4_hybrid_stage_and_benchmark.py

echo "=== [4/5] Starting Unpack (2s) + NVMe & DDR5 Page Cache Warmup + Eager VRAM/RAM Seed ==="
cat <<EOF | perl -e 'alarm 25; exec @ARGV' "${COLAB_BIN}" exec -s "$SESSION"
import subprocess
cmd = (
    "nohup python3 -u /content/step3_g4_hybrid_stage_and_benchmark.py "
    "--variant ${VARIANT} --mode delta --quality-mode ${QUALITY_MODE} --ctx ${CTX_SIZE} --port ${PORT} ${EXTRA_ARGS[*]:-} "
    "> /tmp/delta_g4_server.log 2>&1 < /dev/null &"
)
subprocess.run(cmd, shell=True, check=True)
print("Launched background fast-boot pipeline (${VARIANT}) -> logging to /tmp/delta_g4_server.log")
EOF

echo "=== [5/5] Attaching Permanent Protected HTTPS Gateway & 10-Min Anti-Idle Keepalive ==="
launch_cf_tunnel_on_remote
start_keepalive_daemon

echo ""
echo "✅ Fast-boot pipeline & Cloudflare HTTPS Gateway are running on session [${SESSION}]!"
echo "   • Variant                : ${VARIANT} (${QUALITY_MODE} mode)"
echo "   • Anti-Idle Keepalive    : Active (10-min colab exec heartbeat)"
echo "   • Check progress         : $(basename "$0") status"
echo "   • View/rotate API Key    : $(basename "$0") key"
echo "   • Stop & release         : $(basename "$0") stop"
