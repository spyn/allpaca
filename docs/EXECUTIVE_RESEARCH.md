# Heterogeneous Local Inference Engine — Deep Research Review

**Research date:** 2026-08-16  
**Implementation Path:** Greenfield Rust (Control Plane) + C++/HIP/CUDA/OpenVINO (Kernels)  
**Target reference platform:** Ryzen 7 5800X-class 8C/16T host, 80+ GB DDR4, Radeon RX 9070 16 GB + Radeon RX 9060 XT 16 GB, optional Google Coral Edge TPU.

## Executive conclusion

The original opportunity list was heavily biased toward rack-scale enterprise systems (CXL 3.0, AVX-512, transparent pooling). To build a truly viable _local_ engine across diverse desktop and mobile hardware, we must pivot from datacenter assumptions to **dynamic, hardware-aware execution routing**.

The original opportunity list mixes four different categories:

1. **Already implemented in leading engines:** speculative decoding, n-gram speculation, tree verification, disaggregated prefill/decode, paged/prefix KV caching, structured generation, and CPU/GPU overlap.
2. **Feasible but under-integrated on a heterogeneous desktop:** asymmetric dual-GPU scheduling, explicit host-memory model/expert staging, multi-tier KV caching, asynchronous CPU-side constrained decoding, and modality-specific GPU residency.
3. **Model/training dependent research features:** true early exit, self-speculative LayerSkip-style execution, custom tree drafters, and native ternary kernels.
4. **Future hardware research:** CXL memory/PNM/PIM, HBM-PIM, silicon photonics, and true accelerator-coherent memory fabrics.

The engine cannot assume a monolithic execution strategy. It must explicitly handle three distinct architectural realities discovered during research:

1. **Asymmetric Desktops:** Mismatched GPUs (e.g., RX 9070 + RX 9060 XT) require independent prefill/decode pipelines and discrete speculative draft routing.
2. **Symmetric Legacy Workstations:** Systems with dual identical GPUs (e.g., Dual MI50s) bottleneck on older PCIe 3.0 buses and must fall back to traditional Tensor Parallelism.
3. **Modern Mobile Architectures:** Devices like the Dell Pro Max pair small discrete VRAM (6GB RTX PRO 500) with massive, high-bandwidth LPDDR5X system RAM and integrated NPUs (Intel AI Boost). These require "Extreme VRAM Tiering" and NPU task delegation.

## Critical corrections

The most important correction is that **GPU logits cannot generally be DMA'd "directly into the CPU L3 cache" as a software-controlled destination**. PCIe/CXL can provide coherent memory semantics in supported systems, but ordinary discrete GPUs do not expose the CPU cache as a programmable DMA target. The practical substitute is a pinned, cache-conscious host ring buffer with asynchronous device-to-host copies, or a future CXL Type-2 accelerator design.

A second correction is that **32 GB of dual-GPU VRAM is not one automatically pooled 32 GB address space**. HIP supports peer-to-peer access where the topology and driver expose it, but allocation, placement and communication remain explicit. AMD documents P2P GPU access and notes that otherwise copies can fall back through host staging. [AMD HIP multi-device documentation](https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/hip_runtime_api/multi_device.html)

A third correction is that **1F1B is a training/pipeline-parallelism term**; an inference engine should instead implement a decode/prefill pipeline with microbatching and double buffering. Backward-pass terminology should not be copied into a pure generation scheduler.

## Dynamic hardware adaptability

### CPU instruction set realities (AVX2 vs. AVX-512)

The proposed BitNet 1.58-bit ternary acceleration initially assumed AVX-512 VNNI. However, target processors like the Ryzen 7 5800X (Zen 3) and Intel Core Ultra 7 265H (Arrow Lake-H) only support 256-bit registers (AVX2 / AVX-VNNI). The engine's low-level math kernels must natively fall back to AVX2 intrinsics, which prevents crashes but slightly limits theoretical CPU throughput compared to enterprise Xeon chips.

### Replacing CXL 3.0 with Smart Access Memory (SAM)

Local desktops do not support coherent CXL 3.0 memory pools. The engine must utilize PCIe 4.0 Smart Access Memory (Resizable BAR) for zero-copy host RAM paging. The engine will explicitly treat the system's large DDR4/LPDDR5X RAM as a "warm" hierarchical KV-cache tier.

### NPU and Edge TPU offloading

Modern local hardware includes underutilized INT8 silicon.

- **Desktop:** The Google Coral TPU (4 TOPS) can run lightweight sequence classifiers via `libcoral`.
- **Mobile:** The Intel AI Boost NPU (13 TOPS) can run reasoning monitors or Markov chains via OpenVINO or DirectML.

The engine must abstract these behind a generic `AuxiliaryDevice` API rather than hardcoding for one TPU.

### GPU symmetry dictates pipeline

The Rust control plane must query GPU symmetry on boot.

- If `Bandwidth(GPU_A) >> Bandwidth(GPU_B)`: Engage Asymmetric Pipeline. Use the slower GPU purely for speculative token trees (Token Forest) or multimodal encoding.
- If `Bandwidth(GPU_A) == Bandwidth(GPU_B)` (and PCIe is a bottleneck): Engage Symmetric Pipeline. Split the target model uniformly across both GPUs.

## Opportunity matrix

| Opportunity | Status in current engines/research | Target engine decision | Feasibility here |
|---|---|---|---|
| Draft-model speculative decoding | Productized in llama.cpp/vLLM/SGLang | Implement baseline first | High |
| n-gram speculation | Productized in llama.cpp | Keep as zero-cost fallback | High |
| Tree speculative verification | Research/product components exist | Build heterogeneous tree scheduler | High/Medium |
| EAGLE/MTP/self-speculation | Increasingly productized | Plugin architecture | Medium |
| Prefill/decode disaggregation | Productized in vLLM ecosystems | Implement local asymmetric variant | High |
| CPU/GPU overlap + async sampling | Common pattern | Make first-class scheduler primitive | High |
| Early exit | Research/model-dependent | Only for trained early-exit models | Medium/Low |
| Host RAM weight/expert staging | Established in offload systems | Implement page cache + async prefetch | High |
| NVMe KV/weight tier | Research/throughput-oriented systems | Optional cold tier | Medium |
| "GPU directly to L3" logits | Not a normal PCIe GPU facility | Reject literal design; replace with ring-buffer DMA | Not applicable |
| GPU P2P DMA | Supported by HIP when topology permits | Detect and exploit | High |
| GPU memory pooling as transparent 32 GB | Not transparent | Explicit device-local allocators | High |
| Coral Edge TPU as generic LLM draft engine | Wrong abstraction; op-limited TFLite | Restrict to small compiled classifiers | Medium |
| Coral structured-output arbiter | Not inherently useful | CPU SIMD is usually preferable | Low |
| Multimodal GPU partitioning | Feasible | Permanent modality residency | High |
| Infinity Cache software "locking" | Not directly exposed | Tune kernels, do not assume residency | Medium |
| Native ternary/BitNet backend | Real and advancing | Dedicated model backend | Medium |
| CXL memory tier | Real server hardware/research | Architecture extension, not desktop MVP | Low here |
| HBM-PIM/PNM | Research/accelerator products | Future backend interface | Not on desktop |
| Photonic compute/interconnect | Hardware research/productization | Future accelerator abstraction | Not on desktop |

## Key research findings

### 1. Speculative decoding is no longer an untouched opportunity

llama.cpp currently supports draft-model speculation, n-gram variants, EAGLE3, MTP and configurable speculative-device placement. Its server and common APIs expose draft-model controls directly. This means a new engine should not waste effort cloning a basic linear drafter; its differentiator should be **policy selection, heterogeneous placement, acceptance-aware scheduling, and multi-path/tree orchestration**.

Sources: [llama.cpp speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md), [llama.cpp CLI options](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md).

### 2. Tree verification is a legitimate high-value extension

SpecInfer and Medusa demonstrated tree-based candidate generation/verification. The value is not "parallel branches by themselves"; it is reducing sequential target-model decode steps while preserving target-model semantics. [SpecInfer](https://arxiv.org/abs/2305.09781), [Medusa](https://arxiv.org/abs/2401.10774).

The engine should therefore use a **Token Forest** abstraction with:

- node token IDs,
- parent index,
- depth,
- cumulative draft score,
- attention-mask ancestry,
- source/drafter ID,
- expected verification cost.

### 3. Disaggregated prefill/decode is already mainstream enough to be a design primitive

vLLM explicitly documents disaggregated prefill, including KV transfer connectors and remote decode instances. It also notes that the main benefit is tail inter-token latency control rather than automatic throughput increase. The new engine should therefore differentiate **throughput objectives** from **tail-latency objectives** in its scheduler. [vLLM disaggregated prefill](https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_prefill.md).

### 4. Early exit must be model-aware

LayerSkip shows that useful early exit requires models trained for early exits/self-speculation. A generic runtime cannot safely look at entropy halfway through an arbitrary model, abort the remaining layers, and assume the result is equivalent. [LayerSkip paper](https://arxiv.org/abs/2404.16710), [official implementation](https://github.com/facebookresearch/LayerSkip).

Therefore the runtime should expose an `EarlyExitPolicy`, but only enable it when the model manifest declares a valid exit head/calibration scheme.

### 5. Host/NVMe tiering is real; latency matters

FlexGen demonstrated explicit allocation across GPU, CPU and disk, with overlap-aware scheduling. More recent work has characterized NVMe traffic for model and KV offload. This supports an explicit **three-tier page cache**, but NVMe should be treated as a cold-capacity extension rather than pretending it behaves like VRAM. [FlexGen](https://github.com/tomaarsen/FlexGen), [CHEOPS 2025 study](https://research.ibm.com/publications/an-io-characterizing-study-of-offloading-llm-models-and-kv-caches-to-nvme-ssd).

### 6. CXL is highly relevant architecturally, but not for this motherboard as a software toggle

Recent research such as LIA and CXL-enabled KV-cache systems shows real gains from CXL memory and cooperative CPU/accelerator execution. CXL.cache/CXL.mem provide coherent protocols for suitable hardware. That does **not** imply the current Radeon/AM4 desktop can turn system DDR into transparent coherent GPU HBM or address CPU L3 as a DMA buffer. [LIA](https://deepmind.google/research/publications/81986/), [Linux CXL protocol documentation](https://www.kernel.org/doc/html/latest/driver-api/cxl/devices/device-types.html).

### 7. PIM is credible research, but should be an accelerator abstraction

Recent work proposes heterogeneous SRAM-PIM/HBM-PIM or CXL-PNM systems for LLM inference. These are architectural research platforms, not software capabilities that can be switched on in a desktop GPU. [HPIM](https://arxiv.org/abs/2509.12993), [CXL-PNM KV management](https://arxiv.org/abs/2511.00321).

### 8. BitNet is now a practical software research target

Microsoft's bitnet.cpp is an official inference framework with CPU and GPU kernels for native ~1.58-bit models, including an official 2B/4T model. The right opportunity is not "make ordinary FP16 weights ternary at runtime"; it is a dedicated model/backend path for weights trained in the ternary format. [bitnet.cpp](https://github.com/microsoft/BitNet), [BitNet 2B/4T model](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T).

### 9. Coral is useful as a side accelerator, not as a magical LLM co-processor

The Edge TPU executes compiled TensorFlow Lite models and only supports a documented operation subset; unsupported operators fall back to CPU. This makes it suitable for small classifiers, wake-word models, image detectors and other fixed INT8 graphs. It is not a drop-in general LLM draft-model accelerator. [Coral inference overview](https://www.coral.withgoogle.com/docs/edgetpu/inference/), [supported operations](https://www.coral.withgoogle.com/docs/edgetpu/models-intro/).

### 10. The Radeon pair is strongly asymmetric

AMD lists the RX 9070 at 16 GB, 256-bit, up to 640 GB/s and 64 MB Infinity Cache; the RX 9060 XT 16 GB is 128-bit, up to 320 GB/s and 32 MB Infinity Cache. The engine should therefore prefer **9070 = target / main decoder** and **9060 XT = draft / vision / auxiliary work**, unless a benchmark demonstrates a better split. [RX 9070](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070.html), [RX 9060 XT 16 GB](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9060xt.html).

## Recommended MVP order

1. **Rust Control Plane:** Build the hardware discovery engine (Detect CPU, GPUs, memory buses, and NPUs).
2. **Explicit Memory Manager:** Create the VRAM -> Pinned DRAM -> NVMe hierarchy struct.
3. **Asymmetric Target/Draft Pipeline:** Implement the logic for the Desktop profile (RX 9070 + 9060 XT).
4. **Symmetric Fallback Pipeline:** Implement standard Tensor Parallelism for the Workstation profile (MI50s).
5. **Extreme Tiering Pipeline:** Implement aggressive LPDDR5X paging for the Mobile profile (6GB RTX 500).
6. Token Forest + tree verification.
7. Prefill/decode separation inside one machine.
8. Pinned host-memory ring buffers and async sampling/masking.
9. Host RAM model/expert page cache.
10. Hierarchical KV cache: VRAM -> pinned/regular DDR -> NVMe.
11. Multimodal placement.
12. Structured output scheduler.
13. Early-exit/self-speculation only for compatible checkpoints.
14. **Auxiliary NPU Abstraction:** Implement OpenVINO and `libcoral` plugins for background intent/markov classification.
15. **BitNet Backend:** Implement ternary kernels with strict AVX2/AVX-VNNI fallbacks.
16. CXL/PIM simulator/plugin interface.
17. Photonic/remote-accelerator interface.

## What should be removed or rewritten from the original proposal

- Replace "direct-to-L3" with "asynchronous pinned-memory logits transport".
- Replace "VRAM pool" with "explicit cross-device memory fabric".
- Replace "CPU sends CUDA abort interrupt" with "model-specific early-exit graph path".
- Replace "1F1B inference" with "inference microbatch pipeline / double buffering".
- Replace "Infinity Cache clamp" with "cache-aware kernel tiling and measured cache locality".
- Replace "Coral as generic JSON arbiter" with "Coral fixed-function sidecar inference plugin".
- Replace "GPUDirect Storage on Radeon" with "backend-neutral direct-I/O interface; NVIDIA GDS implementation is optional, AMD desktop MVP uses async host staging".
