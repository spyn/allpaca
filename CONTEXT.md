# Allpaca Context

## MVP Scope

**In:**
- GGUF, SafeTensors, and ONNX (weight-export) model loading via `tokenizers` crate
- Single-request generation (one active at a time)
- Asymmetric GPU scheduling (target on fast GPU, draft on slow GPU)
- NGram + DraftModel speculation providers
- VRAM + Pinned DRAM KV tiering (2-tier)
- CPU sampling with JSON mode constraints
- HIP backend (AMD GPU)
- CPU fallback for all kernels
- OpenAI-compatible `/v1/chat/completions` server
- TOML config + CLI flags
- `tracing` logs + `/stats` metrics endpoint

**Out (fast-follow):**
- Tree speculation
- NVMe cold tier
- Multi-request continuous batching
- Full grammar/schema support
- CUDA backend
- Auxiliary accelerators (Coral, NPU)
- Multimodal (vision/audio)
- Prometheus/OpenTelemetry export

**Demo:** Load 7B GGUF, asymmetric speculative decoding on RX 9070 + 9060 XT, show speedup vs single-GPU, KV spill to DRAM when VRAM fills.

## Glossary

| Term | Definition |
|------|------------|
| **Engine** | The core Rust library (`allpaca-core`) that manages scheduling, memory tiers, and device orchestration. Embeddable, no network boundary. |
| **Server** | A thin binary wrapper around the Engine that exposes an OpenAI-compatible HTTP API. Optional deployment mode. |
| **Model Format** | On-disk representation of weights, tokenizer, and metadata. Engine detects GGUF, SafeTensors, or ONNX from the weight-file extension. SafeTensors and ONNX use HuggingFace-style `config.json` + vocab `tokenizer.json` sidecars. ONNX is a weight-export importer (initializers into the CPU transformer), not a general ONNX runtime. Eventual goal is a custom tiered-memory-optimized format. |
| **Pipeline Mode** | A detected hardware profile (Asymmetric, Symmetric, MobileTiered) that sets default feature flags. Auto-detected at boot, overridable via config. |
| **Feature Flag** | An independent, composable behavior (e.g., asymmetric drafting, DRAM tiering, NPU offload). Pipeline modes are shorthand for common feature flag combinations, not monolithic profiles. |
| **Control Plane** | Rust code that owns all scheduling, memory bookkeeping, token forest construction, and constraint logic. Memory-safe, runs on CPU. |
| **Kernel ABI** | A flat C interface (`extern "C"`) exposing GPU/CPU compute primitives. C++/HIP/CUDA implements the kernels; Rust calls them. No logic in kernels beyond numerics. |
| **Draft Provider** | A pluggable source of speculative token candidates. MVP includes NGramProvider (CPU trie) and DraftModelProvider (secondary GPU model). Tree speculation is Phase 2. |
| **Target Model** | The authoritative model that verifies draft tokens. Runs on the fastest available GPU. |
| **Draft Model** | A smaller/faster model that proposes candidate tokens for speculative decoding. Runs on secondary GPU or CPU. |
| **KV Tier** | A memory region where KV cache pages can reside. MVP supports two tiers: VRAM (hot) and Pinned DRAM (warm). NVMe cold tier is post-MVP. |
| **KV Page** | A fixed-size block of KV cache data that can migrate between tiers. Unit of residency tracking and prefetch. |
| **Auxiliary Accelerator** | A sidecar compute device (Coral TPU, Intel NPU) for non-LLM workloads like classifiers. Post-MVP; trait interface may exist but no implementations ship initially. |
| **Constraint Mask** | A bitset of allowed tokens produced by the CPU grammar engine, applied to GPU logits before sampling. Enables structured output. |
| **Structured Output** | Constrained generation (JSON, schemas, grammars). MVP supports JSON mode only via hardcoded grammar. Full `llguidance` integration is post-MVP. |
| **GPU Backend** | Real HIP kernels (GEMM, attention, hipMalloc, H2D/D2H/P2P copy) behind `--features hip` when ROCm is present. `Engine::generate` runs target forwards on HIP devices and reports `backend = "hip"` / `gpu_forwards`. Without the feature, without an SDK, or with 0 devices / `cpu_only`, the CPU fallback is used. CUDA is post-MVP. |
| **CPU Fallback** | A slow but functional kernel implementation using AVX2 intrinsics. Required for every operation so the engine runs without GPU. |
| **Tokenizer** | Converts text ↔ tokens. Uses HuggingFace `tokenizers` crate. Vocab comes from GGUF metadata or a SafeTensors/ONNX sidecar JSON; decoupled from weight layout. |
| **Request** | A generation task with its own KV cache (`KvHandle` / KV Pages) and decode state. Multiple Requests can be active up to `engine.max_active_requests` (compiled default 1). A saturated scheduler returns Transient `RequestBusy` backpressure. |
| **Telemetry** | Internal metrics (TTFT, ITL, acceptance rate, page migrations) stored in memory, exposed via `/stats` JSON endpoint. Uses `tracing` crate for structured logs. Prometheus/OpenTelemetry export is post-MVP. |
| **Multimodal** | Vision/audio encoder support. Post-MVP; deferred until text-only asymmetric scheduling is proven. |
| **Transfer Path** | The mechanism for moving tensors between devices. At boot the engine queries `hipDeviceCanAccessPeer`, enables both directions (`hipDeviceEnablePeerAccess`), re-queries, then benchmarks that enable+peer-copy against host staging (GPU→DRAM→GPU). `P2p` is chosen only when enable+copy succeed and measured P2P is not slower than host staging by more than 10% (`P2P_MAX_SLOWER_FRACTION`). Host staging is the fallback when P2P is unavailable or slower. Never assume P2P is faster. |
| **Configuration** | Engine settings loaded from TOML file, overridable by CLI flags. Compiled defaults for unspecified values. No env vars or runtime API in MVP. |

## Crate Structure

| Crate | Purpose |
|-------|---------|
| `allpaca-types` | Shared types (DeviceInfo, KvPage, Config, etc.) |
| `allpaca-core` | Engine, scheduler, memory tiers, token forest, draft providers |
| `allpaca-gguf` | GGUF model parsing and loading |
| `allpaca-safetensors` | SafeTensors weight parsing (F32 / F16→F32) and sidecar config/tokenizer |
| `allpaca-onnx` | ONNX weight-export loading (initializer import; unsupported ops are Fatal) |
| `allpaca-hip` | HIP kernel FFI bindings |
| `allpaca-cpu` | CPU fallback kernels (AVX2) |
| `allpaca-server` | HTTP server wrapper (OpenAI-compatible API) |

## Error Classes

| Class | Behavior | Examples |
|-------|----------|----------|
| **Fatal** | Return error to caller, cannot continue | Model load failure, OOM, corrupt GGUF/SafeTensors/ONNX, unsupported dtype/op, invalid UTF-8 |
| **Degraded** | Disable feature, continue slower | Draft GPU OOM → disable speculation; P2P fails → use host staging |
| **Transient** | Retry with backpressure | Page migration contention, kernel timeout |

## Testing Strategy

| Tier | What | When |
|------|------|------|
| **Unit tests** | Pure Rust logic (scheduler, token forest, page tables) | Every commit |
| **Integration tests** | Full engine with CPU backend (`allpaca-cpu`) | Every commit |
| **GPU tests** | Actual HIP kernels on AMD hardware | Self-hosted runner or manual pre-release |

## Library API Layers

| Method | Use case |
|--------|----------|
| `engine.generate(prompt, params)` | One-shot blocking generation |
| `engine.generate_stream(prompt, params)` | Async iterator of tokens (used by server) |
| `engine.begin_request()` / `engine.step()` | Low-level control for custom loops |

## Key Dependencies

| Dependency | Purpose |
|------------|---------|
| `tokio` | Async runtime for I/O, channels, spawn_blocking |
| `tokenizers` | Tokenization (HuggingFace crate) |
| `serde` / `toml` | Configuration deserialization |
| `tracing` | Structured logging |

## Sampling

Sampling (temperature, top-k, top-p, grammar masking, RNG) happens on **CPU**. Logits are copied async from GPU to pinned DRAM ring buffer. GPU sampling is a post-MVP optimization.

## Server API (MVP)

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | Chat generation (streaming and non-streaming) |
| `GET /v1/models` | List loaded model info |
| `GET /stats` | Internal metrics JSON (non-OpenAI) |
