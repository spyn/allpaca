#!/usr/bin/env bash
# Stage a user-local HIP compiler next to a runtime-only /opt/rocm (no sudo).
# Home / MI50 boxes: gfx906 device-libs already ship in the runtime package.
#
# Do not `apt install` these debs on Debian 13 — Ubuntu noble packages want
# libstdc++-11-dev, which Trixie does not ship. Download + extract only.
set -euo pipefail

CACHE="${ALLPACA_ROCM_SDK:-${HOME}/.cache/allpaca}"
DEB_DIR="${CACHE}/rocm-debs"
SDK_ROOT="${CACHE}/rocm-sdk"
PREFIX="${SDK_ROOT}/opt/rocm-6.2.4"
PACKAGES=(hipcc6.2.4 hip-dev6.2.4 rocm-llvm6.2.4 hsa-rocr-dev6.2.4)

mkdir -p "${DEB_DIR}" "${SDK_ROOT}"
(
  cd "${DEB_DIR}"
  apt-get download "${PACKAGES[@]}"
  for deb in *.deb; do
    dpkg-deb -x "${deb}" "${SDK_ROOT}"
  done
)

if [[ -d /opt/rocm/amdgcn ]]; then
  ln -sfn /opt/rocm/amdgcn "${PREFIX}/amdgcn"
fi
if [[ -d /opt/rocm/lib ]]; then
  mkdir -p "${PREFIX}/lib"
  for lib in /opt/rocm/lib/libamdhip64.so* /opt/rocm/lib/libhsa-runtime64.so* \
             /opt/rocm/lib/libamd_comgr.so* /opt/rocm/lib/libhiprtc.so* \
             /opt/rocm/lib/libhiprtc-builtins.so*; do
    [[ -e "${lib}" ]] || continue
    ln -sfn "${lib}" "${PREFIX}/lib/$(basename "${lib}")"
  done
fi
if [[ -x "${PREFIX}/lib/llvm/bin/llvm-ar" ]]; then
  ln -sfn "${PREFIX}/lib/llvm/bin/llvm-ar" "${PREFIX}/bin/llvm-ar"
fi

cat <<EOF
HIP SDK staged at ${PREFIX}
Export if build.rs does not pick it up automatically:

  export HIP_PATH=${PREFIX}
  export ROCM_PATH=${PREFIX}

Then: cargo test --workspace --features hip
EOF
