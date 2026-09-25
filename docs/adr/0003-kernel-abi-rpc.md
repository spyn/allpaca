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
- Worker (`allpaca-rpc-server`) only `alloc` / `H2D` / `D2H` / `PutTensor` / batched `LaunchList` / `RunSpan`.
- Buffer handles are generation-counted IDs, never raw device pointers on the wire.
- Weights cross the wire as packed GGUF tensors (`PutTensor`: ggml type, dims, raw bytes). The worker stores those bytes. `GemmNtQuant` reads the slot's ggml type and runs the existing quant GEMM (CPU dequant, or HIP `gemm_nt_*_device`, including ROCmFP4). Activations at a span edge stay F32. A large GGUF is not expanded to host F32 to feed RPC.
- Compute stays `LaunchList` / `RunSpan`. This is not llama.cpp `GRAPH_COMPUTE` and not a ggml graph IR.
- A remote layer is placed only when the worker can run it: dense attention + FFN (including Qwen3.5 gated Q, per-head Q/K norm, and partial RoPE) and a ggml type that worker's GEMM accepts. GDN blocks stay on the local device. A dense launch is never substituted for a GDN block. When a HIP resident model is loaded, remote layers are omitted from the local upload and the device loop bounces the activation around each remote span.
- Layers are split contiguously by advertised memory (or `--tensor-split`). The first split weight is the local GPU pool, subdivided across local HIP devices by VRAM. Later weights are RPC endpoints. Consecutive placed layers on one worker are one `RunSpan`.
- HIP and CUDA workers ship the same `KernelOp` ABI. Coordinators treat either
  backend as a remote device after HELLO (`RpcBackendKind::Hip` / `Cuda`). CUDA workers reject `GemmNtQuant` until they grow quant GEMM.
- Tensor-parallel / row-split, worker-to-worker RDMA, and replica load-balance are out of v1.

## Consequences

- Decode that already fits on one machine will not get faster; RPC trades latency for capacity.
- Activation host-bounce at device boundaries is recorded in `rpc_activation_bytes`.
- HELLO major must match on every machine. `KERNEL_ABI_REV` must match too (rev 2 is `PutTensor` + `GemmNtQuant`). Non-loopback worker binds require `--insecure-bind` and a shared HELLO `--token` / `ALLPACA_RPC_TOKEN`. Default bind is localhost.
- If the plan names remote owners but no layer was placed, decode stays on the local device.
