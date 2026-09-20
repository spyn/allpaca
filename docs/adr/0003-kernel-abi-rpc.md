# ADR 0003: Remote Devices Are a Kernel-ABI RPC Backend

## Status

Accepted

## Context

llama.cpp exposes remote GPUs by serializing ggml graphs over TCP (`GRAPH_COMPUTE`).
Allpaca has a thick Rust control plane (ADR 0001) and a pointer-based C kernel ABI,
not a graph IR. We still need to pool VRAM across machines so a model that does not
fit one box can run.

## Decision

Remote accelerators are **another device**, reached by a versioned kernel-ABI RPC:

- Coordinator (`allpaca-server`) owns scheduling, KV policy, speculation, and HTTP.
- Worker (`allpaca-rpc-server`) only `alloc` / `H2D` / `D2H` / batched `LaunchList`.
- Buffer handles are generation-counted IDs, never raw device pointers on the wire.
- Layers are split contiguously by advertised memory (or `--tensor-split`).
- HIP and CUDA workers ship the same `KernelOp` ABI. Coordinators treat either
  backend as a remote device after HELLO (`RpcBackendKind::Hip` / `Cuda`).
- Tensor-parallel / row-split, worker-to-worker RDMA, and replica load-balance are out of v1.

## Consequences

- Decode that already fits on one machine will not get faster; RPC trades latency for capacity.
- Activation host-bounce at device boundaries is recorded in `rpc_activation_bytes`.
- HELLO major must match on every machine. Non-loopback worker binds require `--insecure-bind` and a shared HELLO `--token` / `ALLPACA_RPC_TOKEN`. Default bind is localhost.
