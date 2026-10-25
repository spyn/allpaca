#!/usr/bin/env python3
"""Fused SwiGLU: y = silu(gate) * up, f32 device buffers.

Kernel lab only (not MXFP4 dequant — that stays the in-tree 17-byte HIP path).
Shapes: M in {1,32,40} tokens, N=2048 (shexp intermediate).

  .venv/bin/python bench_swiglu.py [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TRITON_CACHE_DIR", str(HERE / ".triton-cache"))

SHAPES = ((1, 2048), (32, 2048), (40, 2048), (1, 4096), (32, 4096))


def _kernel():
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 256}, num_warps=4),
            triton.Config({"BLOCK": 1024}, num_warps=4),
            triton.Config({"BLOCK": 2048}, num_warps=8),
        ],
        key=["N"],
    )
    @triton.jit
    def silu_mul(gate_ptr, up_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        g = tl.load(gate_ptr + offs, mask=mask, other=0.0)
        u = tl.load(up_ptr + offs, mask=mask, other=0.0)
        # silu(x) = x * sigmoid(x); avoid tl.sigmoid for 3.5 portability
        y = (g * (1.0 / (1.0 + tl.exp(-g)))) * u
        tl.store(out_ptr + offs, y, mask=mask)

    return silu_mul


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", type=Path, default=HERE / "out" / "triton_swiglu.json")
    args = p.parse_args()

    try:
        import torch
        import triton
    except ImportError as e:
        print(f"missing import: {e} (run setup.sh)", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is false", file=sys.stderr)
        return 2

    silu_mul = _kernel()
    rows = []
    device = "cuda:0"
    for m, n in SHAPES:
        numel = m * n
        gate = torch.empty(numel, dtype=torch.float32, device=device)
        up = torch.empty(numel, dtype=torch.float32, device=device)
        out = torch.empty(numel, dtype=torch.float32, device=device)
        gate.uniform_(-2.0, 2.0)
        up.uniform_(-0.5, 0.5)
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK"]),)  # noqa: E731
        for _ in range(args.warmup):
            silu_mul[grid](gate, up, out, numel)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            silu_mul[grid](gate, up, out, numel)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) * 1e6 / args.iters
        ref = torch.nn.functional.silu(gate) * up
        max_abs = (out - ref).abs().max().item()
        rows.append({
            "m": m, "n": n, "numel": numel,
            "us_mean": round(us, 3),
            "max_abs_err": max_abs,
        })
        print(f"triton silu_mul {m}x{n}  {us:.1f} us  err={max_abs:.3e}")

    payload = {
        "backend": "triton-gfx906",
        "kernel": "silu_mul",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "warmup": args.warmup,
        "iters": args.iters,
        "shapes": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("SIDECAR_JSON:" + json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
