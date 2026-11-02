#!/usr/bin/env bash
# ==============================================================================
# Step 2 (Colab G4 GPU Node - NVIDIA RTX PRO 6000 Blackwell 96GB GDDR7):
# Build BOTH Inference Engines natively for Blackwell sm_120a and archive the
# precompiled binaries to Google Drive (`g4_sm120a_engines.tar.gz`) for 2s boot:
#   1. Delta Inference Engine (`ds41_generate` + `serve_ds41.py`)
#      -> Native Blackwell NVFP4 (`sm_120a`) + Pinned DDR5 RAM Expert Arena
#      -> Includes G4-96GB segmented GDDR7 allocation (`cudaMalloc` chunks)
#   2. Mainline llama.cpp (`llama-server` / `llama-cli`, `b11870+` / `PR #31086`)
#      -> Native `DEEPSEEK3` + Conditional Engram + Hybrid GPU/CPU `-ot`
# ==============================================================================
set -euo pipefail

V41_DIR="/content/drive/MyDrive/AI_Models/DeepSeek_V41_G4"
BUILD_ROOT="/content/deepseek_build"
INSTALL_BIN="/content/engines/bin"
ARCHIVE_TAR="${V41_DIR}/bin/g4_sm120a_engines.tar.gz"
mkdir -p "${BUILD_ROOT}" "${INSTALL_BIN}" "${V41_DIR}/bin"

echo "=== [0/3] GPU & CUDA Check ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
nvcc --version

# ------------------------------------------------------------------------------
# Engine A: Delta Inference Engine (Blackwell sm_120a NVFP4 + DSpark MTP)
# ------------------------------------------------------------------------------
if [[ -x "${V41_DIR}/bin/ds41_generate" ]]; then
    echo "=== [1/3] Restoring cached Delta binary (ds41_generate) from Google Drive ==="
    cp -f "${V41_DIR}/bin/ds41_generate" "${INSTALL_BIN}/ds41_generate"
    chmod +x "${INSTALL_BIN}/ds41_generate"
else
    echo "=== [1/3] Compiling Delta Inference Engine (Blackwell sm_120a) ==="
    cd "${BUILD_ROOT}"
    if [[ ! -d "delta-inference-engine" ]]; then
        git clone --depth 1 https://github.com/serialx/delta-inference-engine.git
    fi
    cd delta-inference-engine
    make ds41_generate CUDA_ARCH="-gencode arch=compute_120a,code=sm_120a" -j"$(nproc)"
    cp -f build/ds41_generate "${INSTALL_BIN}/ds41_generate"
    cp -f build/ds41_generate "${V41_DIR}/bin/ds41_generate"
fi

# Sync Delta serve tools into /content/engines/delta_serve for portable packaging
if [[ -d "${BUILD_ROOT}/delta-inference-engine" ]]; then
    mkdir -p /content/engines/delta_serve
    rsync -a --exclude='.git' --exclude='build' "${BUILD_ROOT}/delta-inference-engine/" /content/engines/delta_serve/
fi

# ------------------------------------------------------------------------------
# Engine B: Mainline llama.cpp (b11870+ with DeepSeek-V4.1 + Engram support)
# ------------------------------------------------------------------------------
if [[ -x "${V41_DIR}/bin/llama-server" && -x "${V41_DIR}/bin/llama-cli" ]]; then
    echo "=== [2/3] Restoring cached llama.cpp binaries from Google Drive ==="
    cp -f "${V41_DIR}/bin/llama-server" "${INSTALL_BIN}/llama-server"
    cp -f "${V41_DIR}/bin/llama-cli" "${INSTALL_BIN}/llama-cli"
    cp -f "${V41_DIR}/bin/"*.so* "${INSTALL_BIN}/" 2>/dev/null || true
    chmod +x "${INSTALL_BIN}/llama-server" "${INSTALL_BIN}/llama-cli"
else
    echo "=== [2/3] Compiling Mainline llama.cpp (CUDA sm_120a) ==="
    cd "${BUILD_ROOT}"
    if [[ ! -d "llama.cpp" ]]; then
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
    fi
    cd llama.cpp
    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="120a" \
        -DGGML_NATIVE=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release --target llama-server llama-cli -j"$(nproc)"
    cp -f build/bin/llama-server "${INSTALL_BIN}/llama-server"
    cp -f build/bin/llama-cli "${INSTALL_BIN}/llama-cli"
    cp -f build/bin/*.so* "${INSTALL_BIN}/" 2>/dev/null || true
    cp -f build/bin/llama-server "${V41_DIR}/bin/llama-server"
    cp -f build/bin/llama-cli "${V41_DIR}/bin/llama-cli"
    cp -f build/bin/*.so* "${V41_DIR}/bin/" 2>/dev/null || true
fi

export LD_LIBRARY_PATH="${INSTALL_BIN}:${LD_LIBRARY_PATH:-}"

echo "=== [3/3] Archiving Precompiled Engines to ${ARCHIVE_TAR} (for 2s unpack on future G4 boots) ==="
tar -czf "${ARCHIVE_TAR}" -C /content/engines .
ls -lh "${INSTALL_BIN}" "${ARCHIVE_TAR}"
echo "✅ Both Blackwell sm_120a engines are ready at ${INSTALL_BIN} and archived at ${ARCHIVE_TAR}!"
