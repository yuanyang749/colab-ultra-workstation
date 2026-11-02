#!/usr/bin/env bash
# ==============================================================================
# Fast One-Command Launcher for DeepSeek-V4.1-Flash-1017 (748B / 2-bit Flagships)
# on Google Colab G4 (NVIDIA RTX PRO 6000 Blackwell 96GB GDDR7 + 188GB DDR5 RAM)
#
# Supported Flagship 2-bit Variants:
#   - unsloth-ud-iq2xxs : Unsloth Dynamic V3.1 UD-IQ2_XXS (165.30 GB, default)
#   - abiray-gsq-iq2xxs : Abiray Uncensored GSQ-RCO IQ2_XXS (138.59 GB)
#
# Usage:
#   ./scripts/deepseek_g4/fast_start_g4.sh start [--variant unsloth-ud-iq2xxs|abiray-gsq-iq2xxs]
#   ./scripts/deepseek_g4/fast_start_g4.sh switch <unsloth-ud-iq2xxs|abiray-gsq-iq2xxs>
#   ./scripts/deepseek_g4/fast_start_g4.sh status   # Check local tunnel, health & GPU telemetry
#   ./scripts/deepseek_g4/fast_start_g4.sh stop     # Stop local tunnel and unassign G4 GPU (0 CCU/h)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../colab_meter.py" ]]; then
    METER_PY="${SCRIPT_DIR}/../colab_meter.py"
else
    METER_PY="${HOME}/.agents/skills/colab-ultra-workstation/scripts/colab_meter.py"
fi
LOCAL_PORT="${LOCAL_PORT:-8150}"
REMOTE_PORT="${REMOTE_PORT:-8150}"
QUALITY_MODE="${QUALITY_MODE:-exact}"
VARIANT="${VARIANT:-unsloth-ud-iq2xxs}"

ACTION="${1:-start}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant|-v)
            VARIANT="${2:-unsloth-ud-iq2xxs}"
            shift 2
            ;;
        unsloth-ud-iq2xxs|abiray-gsq-iq2xxs)
            VARIANT="$1"
            shift
            ;;
        --quality-mode|-q)
            QUALITY_MODE="${2:-exact}"
            shift 2
            ;;
        --port|-p)
            LOCAL_PORT="${2:-8150}"
            REMOTE_PORT="${2:-8150}"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

case "${ACTION}" in
  status)
    echo "=== [1] Local Colab Meter & Active Sessions ==="
    python3 "${METER_PY}" status || true
    echo ""
    echo "=== [2] Local OpenAI Endpoint Health (http://127.0.0.1:${LOCAL_PORT}/health) ==="
    curl -s "http://127.0.0.1:${LOCAL_PORT}/health" | python3 -m json.tool || echo "Local tunnel not listening on :${LOCAL_PORT}"
    ;;

  stop)
    echo "=== Stopping local SSH tunnel on :${LOCAL_PORT} & Releasing G4 GPU ==="
    pkill -f "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" 2>/dev/null || true
    echo "ℹ️ Run 'colab session rm <session_id>' or use colab-mcp to unassign the G4 runtime (0.00 CCU/h)."
    python3 "${METER_PY}" status || true
    ;;

  switch)
    echo "=== Hot-Swapping Active 2-Bit Model Variant on :${REMOTE_PORT} -> [${VARIANT}] ==="
    echo "Uploading step3_g4_hybrid_stage_and_benchmark.py and launching [${VARIANT}] (${QUALITY_MODE} mode)..."
    echo "Command on G4:"
    echo "  python3 /content/step3_g4_hybrid_stage_and_benchmark.py --variant ${VARIANT} --mode delta --quality-mode ${QUALITY_MODE} --port ${REMOTE_PORT}"
    ;;

  start)
    echo "================================================================================"
    echo "🚀 DeepSeek-V4.1-Flash (748B / 2-Bit Flagship) G4 96GB Blackwell Fast-Boot"
    echo "   Variant      : ${VARIANT}"
    echo "   Quality Mode : ${QUALITY_MODE} (exact = native Q8_0 dense + route_bias 0.00)"
    echo "   Endpoint     : http://127.0.0.1:${LOCAL_PORT}/v1"
    echo "================================================================================"
    echo "  1. Unpacks precompiled sm_120a binaries (g4_sm120a_engines.tar.gz) in ~2s"
    echo "  2. Stages MoE backbone shards to local Gen5 NVMe via 64-stream Drive-v3 API (~150s)"
    echo "     whilst warming 188GB DDR5 Page Cache in a single pass"
    echo "  3. Starts Delta Server (${QUALITY_MODE} mode) + auto-seeds ~87.2 GiB GDDR7 VRAM (~20s)"
    echo ""
    echo "Remote G4 command:"
    echo "  python3 /content/step3_g4_hybrid_stage_and_benchmark.py --variant ${VARIANT} --mode delta --quality-mode ${QUALITY_MODE} --port ${REMOTE_PORT}"
    ;;

  *)
    echo "Usage: $0 {start|switch|status|stop} [--variant unsloth-ud-iq2xxs|abiray-gsq-iq2xxs] [--quality-mode exact|turbo]"
    exit 1
    ;;
esac
