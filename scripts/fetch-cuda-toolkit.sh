#!/usr/bin/env bash
# Stage a user-local CUDA 11.4 toolkit for Tesla K80 / driver 470 boxes (no sudo).
#
# Why 11.4: NVIDIA driver 470 advertises CUDA 11.4 max. CUDA 12 dropped Kepler
# (sm_37 / Tesla K80) entirely. CUDA 11.7+ runtimes need a newer driver than 470.
#
# Pulls the small CUDA *redistributable* archives (nvcc + cudart + cccl + nvrtc),
# not the multi-GiB runfile. Layout:
#   ${ALLPACA_CUDA_SDK:-$HOME/.cache/allpaca/cuda-sdk/cuda-11.4}
#
# Then build with:
#   export CUDA_PATH="$HOME/.cache/allpaca/cuda-sdk/cuda-11.4"
#   export PATH="$CUDA_PATH/bin:$PATH"
#   export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
#   # Ubuntu 22.04: CUDA 11.4 prefers g++-10
#   export CXX=g++-10
#   cargo test -p allpaca-cuda --features cuda
#
# Override arches with CUDA_ARCHES=sm_37 (auto-detects K80 by name).
set -euo pipefail

CACHE="${ALLPACA_CUDA_CACHE:-${HOME}/.cache/allpaca}"
DEB_DIR="${CACHE}/cuda-debs"
SDK_ROOT="${ALLPACA_CUDA_SDK:-${CACHE}/cuda-sdk/cuda-11.4}"
REDIST_BASE="https://developer.download.nvidia.com/compute/cuda/redist"
REDIST_JSON="${REDIST_BASE}/redistrib_11.4.4.json"

# Component keys from redistrib_11.4.4.json (linux-x86_64).
COMPONENTS=(cuda_nvcc cuda_cudart cuda_cccl cuda_nvrtc)

mkdir -p "${DEB_DIR}" "${SDK_ROOT}"

if [[ -x "${SDK_ROOT}/bin/nvcc" && -f "${SDK_ROOT}/include/cuda_runtime.h" ]]; then
  echo "CUDA toolkit already present at ${SDK_ROOT}"
  "${SDK_ROOT}/bin/nvcc" --version
  exit 0
fi

echo "Fetching CUDA 11.4.4 redistributable manifest…"
MANIFEST="${DEB_DIR}/redistrib_11.4.4.json"
curl -fsSL -o "${MANIFEST}" "${REDIST_JSON}"

python3 - "${MANIFEST}" "${DEB_DIR}" "${SDK_ROOT}" "${REDIST_BASE}" "${COMPONENTS[@]}" <<'PY'
import json, os, sys, tarfile, urllib.request

manifest_path, deb_dir, sdk_root, base = sys.argv[1:5]
components = sys.argv[5:]
with open(manifest_path) as f:
    data = json.load(f)

os.makedirs(sdk_root, exist_ok=True)
for name in components:
    entry = data[name]["linux-x86_64"]
    rel = entry["relative_path"]
    url = f"{base}/{rel}"
    fname = os.path.basename(rel)
    dest = os.path.join(deb_dir, fname)
    if not os.path.isfile(dest) or os.path.getsize(dest) < 1024:
        print(f"Downloading {fname} ({entry.get('size', '?')} bytes)…")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"Using cached {fname}")
    print(f"Extracting {fname} → {sdk_root}")
    with tarfile.open(dest, "r:xz") as tar:
        # Archives unpack to <name>-archive/{bin,include,lib,...}
        members = tar.getmembers()
        # Strip the top-level archive directory.
        for m in members:
            parts = m.name.split("/", 1)
            if len(parts) < 2:
                continue
            m.name = parts[1]
            target = os.path.join(sdk_root, m.name)
            # LICENSE / README from earlier packs are often mode 444; make writable
            # before overwrite across redistributable merges.
            if os.path.isfile(target) and not os.access(target, os.W_OK):
                os.chmod(target, 0o644)
            tar.extract(m, path=sdk_root)
print("done")
PY

# Prefer lib64 layout (nvcc / cargo link search).
if [[ -d "${SDK_ROOT}/lib" && ! -d "${SDK_ROOT}/lib64" ]]; then
  ln -sfn lib "${SDK_ROOT}/lib64"
fi
# Some redist layouts put shared libs under lib/x86_64-linux-gnu or lib64 already.
if [[ -d "${SDK_ROOT}/lib/x86_64-linux-gnu" && ! -e "${SDK_ROOT}/lib64/libcudart.so" ]]; then
  mkdir -p "${SDK_ROOT}/lib64"
  ln -sfn ../lib/x86_64-linux-gnu/libcudart.so* "${SDK_ROOT}/lib64/" 2>/dev/null || true
fi

if [[ ! -x "${SDK_ROOT}/bin/nvcc" ]]; then
  echo "error: toolkit install incomplete at ${SDK_ROOT}" >&2
  ls -la "${SDK_ROOT}" >&2 || true
  exit 1
fi

cat <<EOF

CUDA 11.4 toolkit ready at:
  ${SDK_ROOT}

Add to your shell (or pass inline):
  export CUDA_PATH="${SDK_ROOT}"
  export PATH="\${CUDA_PATH}/bin:\${PATH}"
  export LD_LIBRARY_PATH="\${CUDA_PATH}/lib64:\${LD_LIBRARY_PATH:-}"
  # Ubuntu 22.04: CUDA 11.4 prefers g++-10
  #   sudo apt install g++-10 && export CXX=g++-10

Smoke:
  cargo test -p allpaca-cuda --features cuda

Tesla K80 is sm_37; build.rs maps the product name automatically (or set CUDA_ARCHES=sm_37).

EOF

"${SDK_ROOT}/bin/nvcc" --version
