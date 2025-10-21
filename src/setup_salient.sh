#!/usr/bin/env bash
# file: setup_salient.sh
set -Eeuo pipefail

echo "[setup_salient] start"

# --- 0) Idempotency on the node - if already built, we're done
STAMP="/opt/ml/code/.salientpp_built"
if [[ -f "${STAMP}" ]]; then
  echo "[setup_salient] already built, skipping."
  exit 0
fi

# --- 1) System Packages
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  git build-essential cmake ninja-build libmetis-dev ca-certificates pciutils
rm -rf /var/lib/apt/lists/*

# --- 2) CUDA/PyTorch Version Detection
PYTORCH_VER=$(python -c 'import torch, re; print(re.sub(r"\+.*$", "", torch.__version__))')
CUDA_VER=$(python -c 'import torch; print(torch.version.cuda or "")')
echo "[setup_salient] PyTorch=${PYTORCH_VER}, CUDA=${CUDA_VER}"

# --- 3) PyG Installation matched to CUDA
if [[ -z "${CUDA_VER}" ]]; then
  echo "[setup_salient] ERROR: SALIENT++ requires a GPU environment (CUDA not found)."
  exit 1
fi
PYG_INDEX_BASE="https://data.pyg.org/whl/torch-${PYTORCH_VER}+cu"
if [[ "${CUDA_VER}" == "12.1" ]]; then
  PYG_INDEX="${PYG_INDEX_BASE}121.html"
elif [[ "${CUDA_VER}" == "11.8" ]]; then
  PYG_INDEX="${PYG_INDEX_BASE}118.html"
else
  PYG_INDEX=""
fi

if [[ -n "${PYG_INDEX}" ]]; then
  echo "[setup_salient] Installing PyG for CUDA ${CUDA_VER} from ${PYG_INDEX}"
  pip install --no-cache-dir torch-geometric torch-sparse torch-scatter -f "${PYG_INDEX}"
else
  echo "[setup_salient] WARN: untested CUDA=${CUDA_VER}. Trying vanilla wheels..."
  pip install --no-cache-dir torch-geometric
fi
python -c 'import torch, torch_geometric; print(f"[setup_salient] PyG OK. torch: {torch.__version__}, CUDA: {torch.version.cuda}")'

# --- 4) Download and build Salient++
# --- FIX: Changed path to install from the correct 'fast_sampler' subdirectory.
SALIENT_DIR="/opt/ml/code/SALIENT_plusplus"
if [[ ! -d "${SALIENT_DIR}" ]]; then
  git clone --depth=1 https://github.com/MITIBMxGraph/SALIENT_plusplus.git "${SALIENT_DIR}"
fi

INSTALL_DIR="${SALIENT_DIR}/fast_sampler"

if [[ ! -d "${INSTALL_DIR}" ]]; then
    echo "[setup_salient] ERROR: The expected 'fast_sampler' directory does not exist."
    exit 1
fi

cd "${INSTALL_DIR}"

if [[ -f "setup.py" ]]; then
  echo "[setup_salient] Found setup.py. Running 'pip install .'..."
  # Use pip for a more robust installation
  pip install .
else
  echo "[setup_salient] ERROR: no setup.py found in ${INSTALL_DIR}."
  exit 1
fi

# --- 5) Validate import
# --- FIX: The correct module to import is 'fast_sampler'.
python - <<'PY'
try:
    import fast_sampler
    print("[setup_salient] import fast_sampler OK")
except Exception as e:
    print(f"[setup_salient] FAILED to import library: {e}")
    import sys
    sys.exit(1)
PY

# Create the stamp file to prevent re-running
touch "${STAMP}"

echo "[setup_salient] success"