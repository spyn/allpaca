#!/usr/bin/env bash
# gfx906 Triton kernel lab. Not used by Engine::generate.
#
# - Python 3.12 venv (host 3.13/3.14 has no ROCm 6.2 torch wheels)
# - torch from https://download.pytorch.org/whl/rocm6.2  (not 6.3 / torch 2.9)
# - clone nlzy/triton-gfx906 @ v3.5.0+gfx906
# - HIP headers/hipcc from scripts/fetch-rocm-hip-sdk.sh (already staged under
#   ~/.cache/allpaca/rocm-sdk). Does not apt-install or upgrade host ROCm.
# - Does not pull vllm-gfx906.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
export TRITON_CACHE_DIR="${HERE}/.triton-cache"
export MAX_JOBS="${MAX_JOBS:-8}"

HIP_PREFIX="${ALLPACA_ROCM_SDK:-${HOME}/.cache/allpaca}/rocm-sdk/opt/rocm-6.2.4"
if [[ ! -x "${HIP_PREFIX}/bin/hipcc" || ! -f "${HIP_PREFIX}/include/hip/hip_runtime.h" ]]; then
  echo "staging HIP 6.2.4 SDK via scripts/fetch-rocm-hip-sdk.sh"
  "${REPO}/scripts/fetch-rocm-hip-sdk.sh"
fi
export HIP_PATH="${HIP_PREFIX}"
export ROCM_PATH="${HIP_PREFIX}"
export PATH="${HIP_PATH}/bin:${HIP_PATH}/lib/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/rocm/lib:${HIP_PATH}/lib:${LD_LIBRARY_PATH:-}"
export CPATH="${HIP_PATH}/include:${CPATH:-}"
export LIBRARY_PATH="${HIP_PATH}/lib:/opt/rocm/lib:${LIBRARY_PATH:-}"

mkdir -p "${HERE}/out" "${HERE}/wheels" "${TRITON_CACHE_DIR}" "${HERE}/src"

UV="${UV:-${HOME}/.local/bin/uv}"
if [[ ! -x "${UV}" ]]; then
  echo "uv not found at ${UV}; install https://docs.astral.sh/uv/ or set UV=" >&2
  exit 1
fi

if [[ ! -x "${HERE}/.venv/bin/python" ]]; then
  "${UV}" python install 3.12
  # --seed ships pip; uv venvs are otherwise pip-less (python -m pip fails).
  "${UV}" venv "${HERE}/.venv" --python 3.12 --seed
fi
# shellcheck disable=SC1091
source "${HERE}/.venv/bin/activate"
python -c "import sys; assert sys.version_info[:2]==(3,12), sys.version"

PY="${HERE}/.venv/bin/python"
uv_pip() { "${UV}" pip install --python "${PY}" "$@"; }

uv_pip -U pip setuptools wheel ninja cmake pybind11

if ! python -c "import torch; assert torch.version.hip" 2>/dev/null; then
  uv_pip torch --index-url https://download.pytorch.org/whl/rocm6.2
fi
python - <<'PY'
import torch
print("torch", torch.__version__, "hip", torch.version.hip, "cuda_is_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu0", torch.cuda.get_device_name(0))
PY

# Torch's rocm6.2 wheel pulls stock `pytorch-triton-rocm`. That import succeeds
# and would skip the gfx906 fork — uninstall it first.
"${UV}" pip uninstall --python "${PY}" -y triton pytorch-triton-rocm || true

SRC="${HERE}/src/triton-gfx906"
if [[ ! -d "${SRC}/.git" ]]; then
  git clone --depth 1 --branch "v3.5.0+gfx906" https://github.com/nlzy/triton-gfx906.git "${SRC}"
fi
if [[ -f "${SRC}/python/requirements.txt" ]]; then
  uv_pip -r "${SRC}/python/requirements.txt"
fi

# Triton vendors its own LLVM (GCN5 / v_dot2 patches). Do not point LLVM_SYSPATH
# at ROCm 6.2 clang — that would skip the fork's codegen.
if ! python -c "import triton; print('triton', triton.__version__)" 2>/dev/null; then
  echo "building triton-gfx906 (MAX_JOBS=${MAX_JOBS}) — this can take a long time"
  (cd "${SRC}" && "${UV}" pip install --python "${PY}" --no-build-isolation -e .)
fi
python -c "import triton; print('triton', triton.__version__, triton.__file__)"
echo "setup ok HIP_PATH=${HIP_PATH}"
