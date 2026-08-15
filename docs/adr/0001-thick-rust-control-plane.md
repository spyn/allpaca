# ADR 0001: Thick Rust Control Plane with Thin C++ Kernel ABI

## Status

Accepted

## Context

The engine combines a scheduling/memory-management layer with GPU compute kernels. We must decide where the Rust/C++ boundary lies.

Two options:

1. **Thick Rust** — Rust owns scheduler, memory tier bookkeeping, token forest, constraint engine. C++ exposes a flat kernel ABI (`allpaca_gemm`, `allpaca_attention`, `allpaca_copy_async`).

2. **Thick C++** — C++ owns the inference graph and memory. Rust sends high-level commands ("run prefill on device 0").

## Decision

**Thick Rust (Option 1).**

Rust owns all control flow, data structures, and policy logic. C++/HIP/CUDA provides only numerical kernels behind a flat `extern "C"` interface.

## Consequences

**Benefits:**
- Scheduler complexity (page tables, tree search, async coordination) stays in memory-safe code
- Easier to test control logic without GPU
- Clearer ownership model — Rust manages lifetimes, C++ kernels are stateless

**Costs:**
- Every kernel interaction crosses FFI — must carefully manage pointer validity
- Cannot use C++ RAII or smart pointers across the boundary
- HIP/CUDA stream and event handles must be opaque to Rust and tracked manually
