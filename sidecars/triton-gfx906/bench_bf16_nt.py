#!/usr/bin/env python3
"""Device-timed BF16 NT GEMM: C[m,n] = A[m,k]_f32 @ dequant(B[n,k]_bf16).T

Matches allpaca HIP `gemm_nt_bf16` layout (not a generate() path).
DSV4-Flash shexp shapes: gate/up N=2048 K=4096; down N=4096 K=2048.

  .venv/bin/python bench_bf16_nt.py [--dump-hsaco] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TRITON_CACHE_DIR", str(HERE / ".triton-cache"))

# DSV4-Flash dense-ish shexp (Unsloth 6/256, ff=2048, dim=4096)
SHAPES = (
    (1, 2048, 4096),
    (32, 2048, 4096),
    (40, 2048, 4096),
    (1, 4096, 2048),
    (32, 4096, 2048),
)


def _kernel():
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def gemm_nt_bf16(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_off = k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] + k_off < K),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] + k_off < K),
                other=0.0,
            )
            # HIP path: dequant BF16 B to f32, then FMA. tl.dot on f32 is the
            # same math; gfx906 has no BF16 MFMA.
            acc += tl.dot(a, tl.trans(b.to(tl.float32)))
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    return gemm_nt_bf16


def _dump_hsaco(dest: Path) -> list[str]:
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    if not cache.is_dir():
        return copied
    for pat in ("**/*.hsaco", "**/*.so", "**/*.cobin", "**/*.bin"):
        for src in cache.glob(pat):
            if src.is_file() and src.stat().st_size > 0:
                rel = src.relative_to(cache)
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out)
                copied.append(str(rel))
    return copied


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--dump-hsaco", action="store_true")
    p.add_argument("--out", type=Path, default=HERE / "out" / "triton_gemm.json")
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

    gemm_nt_bf16 = _kernel()
    rows = []
    device = "cuda:0"
    for m, n, k in SHAPES:
        a = torch.empty((m, k), dtype=torch.float32, device=device)
        b = torch.empty((n, k), dtype=torch.bfloat16, device=device)
        c = torch.empty((m, n), dtype=torch.float32, device=device)
        a.uniform_(-0.5, 0.5)
        b.uniform_(-0.5, 0.5)
        grid = lambda meta: (  # noqa: E731
            triton.cdiv(m, meta["BLOCK_M"]),
            triton.cdiv(n, meta["BLOCK_N"]),
        )
        for _ in range(args.warmup):
            gemm_nt_bf16[grid](
                a, b, c, m, n, k,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            gemm_nt_bf16[grid](
                a, b, c, m, n, k,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
            )
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) * 1e6 / args.iters
        gflops = (2.0 * m * n * k) / (us * 1e-6) / 1e9
        ref = a @ b.float().T
        max_abs = (c - ref).abs().max().item()
        rows.append({
            "m": m, "n": n, "k": k,
            "us_mean": round(us, 3),
            "gflops": round(gflops, 2),
            "max_abs_err": max_abs,
        })
        print(f"triton gemm_nt_bf16 {m}x{n}x{k}  {us:.1f} us  {gflops:.1f} GFLOP/s  err={max_abs:.3e}")

    payload = {
        "backend": "triton-gfx906",
        "kernel": "gemm_nt_bf16",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "warmup": args.warmup,
        "iters": args.iters,
        "shapes": rows,
    }
    if args.dump_hsaco:
        copied = _dump_hsaco(HERE / "hsaco")
        payload["hsaco_copied"] = copied[:32]
        payload["hsaco_n"] = len(copied)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    line = "SIDECAR_JSON:" + json.dumps(payload, separators=(",", ":"))
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
