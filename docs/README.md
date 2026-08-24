# Heterogeneous Inference Engine Research Bundle

This bundle contains the deep-research assessment and implementation plan for a new local inference engine aimed at heterogeneous CPU/GPU/accelerator systems.

## Files

- `PRD.md` — full PRD suitable for importing into an implementation workflow.
- `SPECS.md` — technical specifications, API contracts, and interface definitions.
- `EXECUTIVE_RESEARCH.md` — validation of the original opportunity list and corrected technical conclusions.
- `OPPORTUNITY_DESIGN.md` — detailed architecture and implementation notes for each opportunity.
- `CODE_EXAMPLES.md` — C++/HIP/Python examples for the core building blocks.
- `BENCHMARK_SPEC.md` — experimental design, workload matrix and acceptance gates.

## Core design decision

The engine should be a **heterogeneous scheduler + memory-fabric runtime** rather than another generic single-device transformer implementation.

## Important corrections

- No programmable "GPU -> CPU L3" DMA destination on ordinary discrete GPUs.
- No transparent 32 GB pooled VRAM just because two 16 GB GPUs are installed.
- No generic safe early-exit abort in an arbitrary pretrained transformer.
- No 1F1B backward-pass schedule for pure inference.
- No software guarantee of Infinity Cache residency.
- No generic LLM inference on a Coral Edge TPU.
- NVIDIA GPUDirect Storage is not assumed for an AMD desktop; the engine uses a backend-neutral direct-I/O interface.

## Primary sources

- AMD HIP multi-GPU P2P: https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/hip_runtime_api/multi_device.html
- AMD RX 9070 specs: https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070.html
- AMD RX 9060 XT 16 GB specs: https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9060xt.html
- llama.cpp speculative decoding: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- vLLM disaggregated prefill: https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_prefill.md
- FreeToken (edge MoE, `q*` miss split, FTW layout): https://arxiv.org/abs/2608.16157 — https://github.com/FlashML-org/FreeToken
- kimi-k3-in-c (packed trunk, pin-prefix + ring, expert LRU): https://github.com/FareedKhan-dev/kimi-k3-in-c
- SpecInfer: https://arxiv.org/abs/2305.09781
- Medusa: https://arxiv.org/abs/2401.10774
- LayerSkip: https://arxiv.org/abs/2404.16710
- LIA + CXL: https://deepmind.google/research/publications/81986/
- PIM/PNM for long-context inference: https://arxiv.org/abs/2511.00321
- HPIM: https://arxiv.org/abs/2509.12993
- Microsoft BitNet: https://github.com/microsoft/BitNet
- Coral Edge TPU: https://www.coral.withgoogle.com/docs/edgetpu/inference/
- NVIDIA GPUDirect Storage (reference architecture, not AMD MVP dependency): https://docs.nvidia.com/gpudirect-storage/overview-guide/
