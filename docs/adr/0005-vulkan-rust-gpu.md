# ADR 0005: Optional Vulkan Backend via rust-gpu

## Status

Accepted

## Context

Allpaca's production GPU path is first-party HIP kernels behind a flat C ABI
(ADR 0001, ADR 0002). CUDA is an optional F32 decode scaffold with the same
shape: Cargo feature, stub when the toolkit is missing, concrete
`CudaWorkspace` rather than a shared GPU trait.

A portable GPU option is useful on machines that have Vulkan but not ROCm or
CUDA. [rust-gpu](https://rust-gpu.github.io/) compiles `no_std` Rust compute
shaders to SPIR-V. That is a compiler, not a kernel library.

## Decision

Add an **optional** Vulkan backend:

- Cargo feature `vulkan` on `allpaca-vulkan`, `allpaca-core`, and
  `allpaca-server`.
- First-party kernels in `crates/allpaca-shaders`, compiled at build time by
  rust-gpu (`cargo-gpu-install` / `spirv-builder`) to SPIR-V.
- Host dispatch with `ash` + `gpu-allocator` in `crates/allpaca-vulkan`.
- Engine integration copies the CUDA scaffold: `DeviceKind::VulkanGpu`,
  `gpu_backend = "vulkan"` / `vulkan:N`, `VulkanWorkspace`, TinyModel resident
  F32 decode. No HIP kernel parity, no RPC workers, no wgpu.

**Auto selection:** HIP if present, else CUDA, else Vulkan, else CPU.

**Toolchain:** the workspace stays on its current stable/nightly pin.
`allpaca-shaders` is **not** a workspace member. Only the nested rust-gpu
build uses rust-gpu's pinned nightly.

**ABI:** HIP/CUDA kernels are C++ so they cross a C ABI. Vulkan kernels are
already Rust compiled to SPIR-V; `VulkanWorkspace` methods are the ABI. No
extra `extern "C"` wrapper.

**ADR 0002:** rust-gpu is a compiler (like hipcc). `ash` is thin Vulkan FFI
(like the HIP runtime). Kernels remain first-party. No third-party fused-kernel
library is linked.

## Consequences

- `--features vulkan` always compiles. If shader compile fails or no Vulkan
  device exists, `device_count() == 0` and the engine uses CPU.
- First rust-gpu build compiles `rustc_codegen_spirv` and is slow; CI without
  the feature is unchanged.
- rust-gpu has no first-class f16 ALU and is not a CUDA/HIP replacement.
  Quant GEMV, SSM/GDN/MLA, and packed prefill stay HIP-first.
- Workspace `[profile.*.build-override] opt-level = 3` keeps the codegen
  backend from building in debug.
