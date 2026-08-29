# ADR 0002: No Third-Party GPU Runtime or Kernel Library in the Engine

## Status

Accepted

## Context

The ROCm.AI release prompted a review of whether allpaca should adopt AMD-adjacent
Rust and Python components rather than keep growing its own HIP layer. Two
candidates came up repeatedly:

1. **IronAccelerator** (`ironaccelerator` on crates.io) — a Rust HIP/CUDA FFI crate
   whose headline feature is a thread-local `MemPool` that recycles small device
   allocations.
2. **AITER** — AMD's fused-kernel library (Composable Kernel / ASM / Triton) used
   by vLLM and SGLang on Instinct, including `gemm_a4w4` for MXFP4.

Both would sit underneath or beside the layer ADR 0001 reserves for a flat
`extern "C"` kernel ABI.

## Decision

**Neither is adopted.** `allpaca-hip` plus `hip/src/kernels.cpp` remain the only
GPU runtime and kernel source, and no third-party GPU crate enters the dependency
graph.

## Rationale

**IronAccelerator.** Its `MemPool` is CUDA-only; the ROCm status is HIP wrappers
present, MemPool equivalent absent, no AMD GPU in its CI. The pool also solves a
problem allpaca does not have: the decode path does not churn allocations.
`HipWorkspace` holds scratch and decode/dequant buffers for the process lifetime,
weights go H2D once at load, and expert misses copy into pre-sized LRU slots
(`bulk_fill_experts`). A per-allocation saving of tens of nanoseconds cannot move
inter-token latency when the count of allocations per token is zero.

**AITER.** It is not a stable C ABI. Binding it would pull Python, JIT and
Composable Kernel into the kernel layer, which contradicts ADR 0001. Its published
1.08–1.2x figures are a framework-level delta against untuned vLLM, not a portable
gain for custom GEMV, and its FP4 path targets MI350 matrix cores while allpaca
targets consumer RDNA (RX 9070 + 9060 XT).

## Consequences

- Every dtype allpaca executes needs a kernel in `hip/src/kernels.cpp`. This is
  already true for Q8_0, Q4_K, F16, and now ROCmFP4.
- AITER stays useful as a **reading** reference for kernel algorithms. It is not
  linked, vendored, or invoked.
- If small-buffer churn ever appears (KV page turnover under many concurrent
  requests), the answer is a local power-of-two freelist inside `allpaca-hip` over
  the existing `allpaca_hip_malloc`, still behind the C ABI — not a dependency.
- Managed-memory APIs (`hipMemPrefetchAsync_v2`, `hipMemAdvise_v2`) are rejected on
  the same grounds: they advise unified memory, while allpaca's working-set model
  is explicit `hipMalloc` plus H2D over a pin-prefix, ring and expert LRU. The
  async APIs allpaca does take are the explicit ones: `hipMemcpyAsync` and
  `hipMemcpyBatchAsync` on a dedicated H2D stream.
