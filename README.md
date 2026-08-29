# Allpaca

Research inference engine with a **thick Rust control plane** and a **thin C/HIP kernel ABI**.
It treats the whole machine — AMD GPUs, host DRAM, NVMe, and CPU — as one elastic serving
platform, not a single undersized VRAM pool.

OpenAI-compatible HTTP server · GGUF / SafeTensors / ONNX weight import · CPU always works ·
optional HIP on AMD ROCm.

**License:** [MIT](LICENSE)

---

## Why Allpaca exists

Edge MoE research (notably [FreeToken](https://arxiv.org/abs/2608.16157)) showed that consumer
PCs can serve sparse models by co-scheduling GPU, CPU, and host RAM. That line of work left
several gaps that matter on real desktops — especially **AMD / Windows** boxes with mismatched
GPUs. Allpaca fills those gaps on a Rust/HIP stack (see [`docs/OPPORTUNITY_DESIGN.md`](docs/OPPORTUNITY_DESIGN.md)
and [`docs/GEM_RESEARCH.md`](docs/GEM_RESEARCH.md)).

### Opportunities that were missing — now in-tree

| Gap (FreeToken / community engines) | Allpaca response |
|-------------------------------------|------------------|
| CUDA / NVIDIA lock-in | First-class **AMD HIP** kernels; CPU fallback always; no third-party GPU runtime ([ADR 0002](docs/adr/0002-no-third-party-gpu-runtime.md)) |
| No asymmetric dual-GPU draft | **Target on fast GPU, draft on slow GPU**; host-staged tokens only |
| Dense cyclic scan thrashes LRU | **Pin-prefix + ring** dense trunk (kimi-style policy) |
| Static MoE expert placement | Host-truth expert pool + **GPU LRU** + measured **`q* ≈ m·B_P/B_H`** miss split |
| Semantic anchors only | **Radix KV** + optional **semantic recurrent anchors** |
| No Windows DirectStorage path | **`dstorage.dll` COM bind** + aligned Direct I/O → pinned H2D (true NVMe→VRAM DMA still roadmap) |
| Single-request hybrid sync barrier | **Micro-batch** partner `poll` + **layer-stagger** CPU experts during GPU forward |
| Reactive VRAM only | **Elastic expert↔KV rebalance**, attention-trim cold KV, **`gpu.util` / `gpu.vram`** from `hipMemGetInfo` (no AMD SMI on Windows) |
| Narrow GGUF quants | **F16 / Q4_0 / Q8_0 / Q4_K / Q5_K / Q6_K** + **ROCmFP4 (`ggml_type_id=100`)** CPU dequant + fused HIP GEMV |
| MoE-only / no hybrid SSM | **`nemotron_h_moe` schedule**: selective-SSM stub + attention + MoE; real ROCmFP4 opt-in via `ALLPACA_NEMOTRON=1` |
| Proprietary weight banks only | **GGUF** primary + **packed layer-major / `ALLPFTW1`** trunk for streaming |
| No continuous batching story | Fair **`Engine::poll`**, chunked prefill, multi-request backpressure |
| Speculative decode absent or linear-only | N-gram + draft-model; **linear packed verify**; optional **tree / Token Forest** |
| Structured output bolted on | CPU **JSON / JSON Schema / Lark** via `llguidance` |
| Early-exit unused | Manifest **self-speculate early exit** when enabled |
| NUMA-blind host experts | **NUMA first-touch** on the host expert pool |

Still out of MVP (documented, not claimed): Coral/NPU backends, BitNet, CUDA backend, production
Mamba-2 kernels, CXL/PIM, and full DirectStorage GPU DMA.

---

## Features (quick map)

- **Control plane:** scheduling, KV tiers (VRAM / pinned DRAM / optional NVMe), weight working set, speculation, constraints — all Rust ([ADR 0001](docs/adr/0001-thick-rust-control-plane.md)).
- **Kernels:** `hip/` C ABI (`allpaca_gemm*`, attention, RoPE, RMSNorm, …); `allpaca-cpu` AVX2 fallbacks.
- **Formats:** GGUF (Llama-family + hybrid Nemotron), SafeTensors + HF sidecars, ONNX weight-export.
- **Server:** `POST /v1/chat/completions`, `GET /v1/models`, `GET /stats`.
- **Config:** TOML + CLI; CLI overrides file; file overrides compiled defaults (`Config::compiled_defaults`).

---

## Build

Requires a recent stable Rust toolchain (`rustup`).

### CPU (default)

```bash
cargo build --workspace
cargo test --workspace
cargo build -p allpaca-server --release
```

### HIP / AMD GPU (optional)

Needs the HIP/ROCm SDK. On Windows the crate probes `HIP_PATH`, `ROCM_PATH`, and
`C:\Program Files\AMD\ROCm\*`. MSVC (`vcvars64`) is required so `hipcc` finds the C++ STL.
Without an SDK, `--features hip` still compiles (stub; GPU tests skip).

```bash
# Linux / Git Bash
cargo test --workspace --features hip
cargo run -p allpaca-server --features hip -- --synthetic --port 8080
```

```bat
REM Windows (example: ROCm 7.1 + RX 9070 / 9060 XT)
set HIP_PATH=C:\Program Files\AMD\ROCm\7.1
set ROCM_PATH=C:\Program Files\AMD\ROCm\7.1
cargo test --workspace --features hip
```

Kernels live in `hip/` (CMake + hipcc). `allpaca-hip/build.rs` compiles them when the SDK is found.

---

## Run

### Synthetic smoke (no weights)

```bash
cargo run -p allpaca-server -- --synthetic --cpu-only --host 127.0.0.1 --port 8080
```

```bash
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/stats
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"synthetic","messages":[{"role":"user","content":"hello"}],"max_tokens":8}'
```

### GGUF on CPU or HIP

```bash
# Tiny / medium GGUF on CPU
cargo run -p allpaca-server -- --cpu-only --model path/to/model.gguf --port 8080

# Real ~7B/8B on HIP (do not commit weights)
cargo run -p allpaca-server --features hip -- --model %ALLPACA_7B_GGUF%
```

Optional draft (second GPU): `--draft-model`, `model.draft_path`, or `ALLPACA_DRAFT_GGUF`.

Gated HIP soaks (when weights are present locally):

| Env | Purpose |
|-----|---------|
| `ALLPACA_7B_GGUF` / `ALLPACA_GGUF` | Real Llama-family generate |
| `ALLPACA_DRAFT_GGUF` | Speculative draft GGUF |
| `ALLPACA_Q4K_GGUF` / `ALLPACA_Q4KM_GGUF` | Q4_K / mixed Q4_K_M |
| `ALLPACA_NEMOTRON=1` | Opt-in 30B ROCmFP4 Nemotron generate |
| `ALLPACA_QWEN=1` | Opt-in Qwen35 ROCmFP4 generate (`models/Qwen3.8-27B-…`) |
| `ALLPACA_ROCMFP4_GGUF` / `ALLPACA_QWEN_GGUF` | Explicit ROCmFP4 GGUF path (probe always discovers under `models/`) |
| `ALLPACA_DEMO=1` | Asymmetric draft vs greedy tok/s |
| `ALLPACA_OVERNIGHT=1` | Long draft soak |

### Bench helper

```bash
cargo run -p allpaca-bench -- --tokens 32 --prompt "hello world"
```

### Useful CLI flags (`allpaca-server`)

| Flag | Meaning |
|------|---------|
| `--config PATH` | TOML config |
| `--model PATH` | Target weights |
| `--draft-model PATH` | Draft weights |
| `--host` / `--port` | Bind address (default loopback; use `0.0.0.0` in Docker) |
| `--cpu-only` | Force CPU backend |
| `--synthetic` | Built-in tiny model |
| `--speculative true\|false` | Toggle speculation |
| `--pipeline-mode asymmetric\|symmetric\|mobile_tiered` | Feature-flag profile |
| `--target-device` / `--draft-device` | Device selectors |

---

## Docker

CPU-only image (HIP stays on the metal host — ROCm passthrough is not the default story).

```bash
docker build -t allpaca:cpu .
docker run --rm -p 8080:8080 allpaca:cpu
# → synthetic server on http://127.0.0.1:8080
```

Compose:

```bash
docker compose up --build
```

Mount a GGUF:

```bash
docker run --rm -p 8080:8080 \
  -v /path/to/models:/models:ro \
  allpaca:cpu \
  --cpu-only --host 0.0.0.0 --port 8080 --model /models/model.gguf
```

Or edit `docker-compose.yml` `command` / `volumes` accordingly.

---

## Crate layout

| Crate / path | Role |
|--------------|------|
| `allpaca-types` | Config, devices, metrics, errors |
| `allpaca-core` | Engine, scheduler, KV / weight tiers, draft, DS bridge |
| `allpaca-gguf` | GGUF parse, tokenizer extract, ROCmFP4 host path |
| `allpaca-safetensors` / `allpaca-onnx` | Alternate importers |
| `allpaca-hip` | HIP FFI (`--features hip`) |
| `allpaca-cpu` | CPU kernels + dequant |
| `allpaca-server` | OpenAI-compatible HTTP |
| `tools/bench` | Lightweight bench binary |
| `hip/` | C/HIP kernel ABI sources |

---

## Configuration & docs

- Compiled defaults: `allpaca-types` `Config::compiled_defaults`
- Domain glossary / MVP scope: [`CONTEXT.md`](CONTEXT.md)
- Opportunity → implementation notes: [`docs/OPPORTUNITY_DESIGN.md`](docs/OPPORTUNITY_DESIGN.md)
- Benchmark field contract: [`docs/BENCHMARK_SPEC.md`](docs/BENCHMARK_SPEC.md)
- Architecture decisions: [`docs/adr/`](docs/adr/)

---

## Sources & acknowledgements

Policies and formats we studied (and re-implemented on our ABI — we do **not** vendor their runtimes):

| Source | What we took |
|--------|----------------|
| [FreeToken](https://arxiv.org/abs/2608.16157) ([FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)) | Edge MoE: host expert pool, GPU LRU slots, `q*` PCIe/CPU miss split, elastic VRAM, FTW-style banks, semantic anchors |
| kimi-k3-in-c / related dense-offload work | Dense **pin-prefix + ring** (never LRU a cyclic layer walk); packed trunk I/O discipline |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) / [ggml](https://github.com/ggml-org/ggml) | GGUF layout, K-quant block sizes, community quant vocabulary |
| ROCmFP4 / ggml type 100 (AMD community ROCmFP4 GGUF work) | 18 B / 32-elem dual-scale blocks; CPU dequant + fused HIP GEMV |
| [AMD ROCm / HIP](https://rocm.docs.amd.com/) | Multi-device, P2P vs host staging, hipBLASLt where linked |
| [Microsoft DirectStorage](https://learn.microsoft.com/en-us/windows/win32/direct3d12/directstorage) | Windows NVMe staging API surface (`DStorageGetFactory`) |
| [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) | Vocab / BPE encode-decode |
| [llguidance](https://github.com/guidance-ai/llguidance) | Constrained JSON Schema / Lark decoding |
| Speculative decoding literature (draft-then-verify, tree/forest verify) | N-gram + draft-model providers; packed linear / optional tree verify |
| Attention cache work (H₂O / StreamingLLM-style cold trim ideas) | Soft **attention-trim** of cold VRAM KV for expert headroom |

Internal research notes that drove the cut list: [`docs/EXECUTIVE_RESEARCH.md`](docs/EXECUTIVE_RESEARCH.md), [`docs/GEM_RESEARCH.md`](docs/GEM_RESEARCH.md).

---

## License

Copyright (c) 2026 Allpaca contributors. Released under the [MIT License](LICENSE).
