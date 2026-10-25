#!/usr/bin/env bash
# HIP BF16 NT GEMM microbench vs Triton gfx906 sidecar. Kernel lab only.
#
#   ./run.sh              # HIP always; Triton if .venv is built
#   ./run.sh --hip-only
#   ./run.sh --triton-only
#   ./run.sh --dump-hsaco # copy Triton cache into ./hsaco (also auto if Triton wins)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
HIP_PREFIX="${ALLPACA_ROCM_SDK:-${HOME}/.cache/allpaca}/rocm-sdk/opt/rocm-6.2.4"
export HIP_PATH="${HIP_PATH:-${HIP_PREFIX}}"
export ROCM_PATH="${ROCM_PATH:-${HIP_PATH}}"
export PATH="${HIP_PATH}/bin:${HIP_PATH}/lib/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/rocm/lib:${HIP_PATH}/lib:${LD_LIBRARY_PATH:-}"
export TRITON_CACHE_DIR="${HERE}/.triton-cache"

HIP_ONLY=0
TRITON_ONLY=0
DUMP=0
for arg in "$@"; do
  case "${arg}" in
    --help|-h)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    --hip-only) HIP_ONLY=1 ;;
    --triton-only) TRITON_ONLY=1 ;;
    --dump-hsaco) DUMP=1 ;;
    *) echo "unknown arg: ${arg}" >&2; exit 2 ;;
  esac
done

mkdir -p "${HERE}/out"
HIP_JSON="${HERE}/out/hip_gemm.json"
TRITON_JSON="${HERE}/out/triton_gemm.json"

if [[ "${TRITON_ONLY}" -eq 0 ]]; then
  echo "== HIP gemm_nt_bf16 (release, ALLPACA_SIDECAR_GEMM=1) =="
  ALLPACA_SIDECAR_GEMM=1 ALLPACA_SIDECAR_OUT="${HIP_JSON}" \
    cargo test -p allpaca-hip --features hip --release --offline --test sidecar_gemm \
      -- --nocapture --test-threads=1
fi

if [[ "${HIP_ONLY}" -eq 1 ]]; then
  echo "hip-only done: ${HIP_JSON}"
  exit 0
fi

PY="${HERE}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Triton venv missing; run ${HERE}/setup.sh" >&2
  exit 1
fi
if ! "${PY}" -c "import triton, torch" 2>/dev/null; then
  echo "triton/torch not importable; run ${HERE}/setup.sh" >&2
  exit 1
fi

DUMP_ARGS=()
if [[ "${DUMP}" -eq 1 ]]; then
  DUMP_ARGS+=(--dump-hsaco)
fi

echo "== Triton gemm_nt_bf16 =="
"${PY}" "${HERE}/bench_bf16_nt.py" --out "${TRITON_JSON}" "${DUMP_ARGS[@]+"${DUMP_ARGS[@]}"}"
echo "== Triton silu_mul =="
"${PY}" "${HERE}/bench_swiglu.py" --out "${HERE}/out/triton_swiglu.json"

if [[ -f "${HIP_JSON}" && -f "${TRITON_JSON}" ]]; then
  "${PY}" - "${HIP_JSON}" "${TRITON_JSON}" "${HERE}/hsaco" <<'PY'
import json, sys
from pathlib import Path
hip = json.loads(Path(sys.argv[1]).read_text())
tri = json.loads(Path(sys.argv[2]).read_text())
hsaco = Path(sys.argv[3])
print("\n== compare HIP vs Triton (kernel-only us_mean) ==")
wins = 0
by_h = {(r["m"], r["n"], r["k"]): r for r in hip.get("shapes", [])}
for r in tri.get("shapes", []):
    key = (r["m"], r["n"], r["k"])
    h = by_h.get(key)
    if not h:
        continue
    ht, tt = h["us_mean"], r["us_mean"]
    ratio = ht / tt if tt else float("inf")
    winner = "triton" if tt < ht else "hip"
    if winner == "triton":
        wins += 1
    print(f"  {key[0]}x{key[1]}x{key[2]}  hip={ht:.1f}us ({h['gflops']:.1f} GF)  "
          f"triton={tt:.1f}us ({r['gflops']:.1f} GF)  {winner}  {ratio:.2f}x")
print(f"triton_faster_shapes={wins}/{len(tri.get('shapes', []))}")
if wins == 0:
    print("Triton did not win a shape — not dumping HSACO into hip/")
    raise SystemExit(0)
print("Triton won at least one shape. Cache dump is opt-in (--dump-hsaco); "
      "do not vendor into hip/ until a follow-up.")
PY
fi
