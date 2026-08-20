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
| **Draft Provider** | A pluggable source of speculative token candidates. MVP includes NGramProvider (CPU trie) and DraftModelProvider (secondary GPU model). Token Forest (tree) speculation is a Feature Flag (`speculative.tree.enabled` / `features.tree_speculation`), off by default. When enabled, packed verify runs one target eval with an ancestry attention mask; on HIP the mask is uploaded and applied on the target GPU (`tree_gpu_verifies`). Linear draft verify packs the proposed span in one Target Model forward (`linear_packed_verifies`) and rewinds rejected KV `filled`; it is used when the tree flag is off. |
| **Target Model** | The authoritative model that verifies draft tokens. Runs on the fastest available GPU. |
| **Draft Model** | A smaller/faster model that proposes candidate tokens for speculative decoding. Weights and draft KV stay resident on the secondary GPU (slow device) when two HIP GPUs exist; CPU fallback on 1 GPU / `cpu_only`. Draft HIP GEMM uses the same packed-dtype dispatch as the Target Model (`ResidentModel::gemm_nt`: F16/Q8/Q4_K/F32). Draft DenseKv keeps the accepted prefix: `propose_on` truncates to the shared token prefix and forwards only the suffix (plus a last-token logits refresh), instead of replaying the full context every step. Only draft token ids (and short logits) cross to the target path — never full weights or KV. Path comes from `--draft-model` / `model.draft_path` / `ALLPACA_DRAFT_GGUF` (env fills `draft_path` at `Engine::new` when unset). Gated HIP tests discover a local small GGUF under `ALLPACA_DRAFT_GGUF`, repo `models/Llama-3.2-*-Instruct-*.gguf` (same-family, preferred over Qwen), or HF cache dirs (Llama-3.2-1B/3B if present; `models--ggml-org--Qwen2.5-Coder-0.5B-Q8_0-GGUF` including hash blobs without a `.gguf` suffix). Missing draft is Degraded; a tiny written draft GGUF is enough to wire the Engine path. |
| **KV Tier** | A memory region where KV cache pages can reside. MVP supports two tiers: VRAM (hot) and Pinned DRAM (warm). NVMe cold tier is post-MVP. Target pages are `hipMalloc` on the fast GPU. Draft DenseKv is `hipMalloc` on the slow GPU when two HIP devices exist (`draft_kv_bytes_on_device`, counted in `kv_bytes_on_device`); host Vecs on 1 GPU / `cpu_only`. |
| **KV Page** | A fixed-size block of KV cache data that can migrate between tiers. Unit of residency tracking and prefetch. |
| **Auxiliary Accelerator** | A sidecar compute device (Coral TPU, Intel NPU) for non-LLM workloads like classifiers. Post-MVP; trait interface may exist but no implementations ship initially. |
| **Constraint Mask** | A bitset of allowed tokens produced by the CPU grammar engine, applied to GPU logits before sampling. Enables structured output. |
| **Structured Output** | Constrained generation (JSON, schemas, grammars). MVP supports JSON mode only via hardcoded grammar. Full `llguidance` integration is post-MVP. |
| **GPU Backend** | Real HIP kernels (GEMM, attention, RoPE, RMSNorm, SiLU, residual add, embed gather, Q8_0 dequant, Q4_K fused GEMV, F16 GEMV, hipMalloc, H2D/D2H/P2P copy) behind `--features hip` when ROCm is present. Weights are uploaded once to the target GPU (draft weights to the second GPU); decode GEMM uses device weight pointers (`weight_h2d_bytes` grows at load only). F32 lm_head GEMV (~128k vocab) uses four columns per warp with float4 loads. Decode Q8 GEMV uses eight columns per warp and 16 warps per block, with 36-byte column-major Q8 blocks and the A row staged in shared memory so columns do not re-read it from VRAM. An integer A×Q8 GEMV (quantize A tiles, int32 MAC, ggml vec_dot_q8-style) exists as `gemm_nt_q8_0_intdot` but is not the decode default: on Groq 8B Q8_0 it did not beat float A × Q8 (memory-bound on weight loads). A 32+scale split / wider Q8 load pass did not beat that 36-byte kernel either, so decode stays on the 36-byte layout. GGUF Q4_K / Q4_K_S / Q4_K_M super-blocks (256 elems, 144 bytes) dequantize on CPU for load and tiny generate; large Q4_K GEMM weights stay packed on device and use a fused HIP GEMV (`gemm_nt_q4_k`) that does not change the Q8_0 decode default. Small Q4_K tensors and `token_embd` dequant to F32 at load. Q5_K (176-byte) and Q6_K (210-byte) super-blocks dequantize on CPU at load so mixed Q4_K_M files (Q6_K on `output` / `attn_v` / `ffn_down`) are not Fatal; they materialize to F32 (no fused HIP GEMV). Small GGUF F16 tensors (n≤65536, except `token_embd`) materialize to F32 at load. Large F16 GEMM weights, `output.weight`, and `token_embd` stay packed on device (`gemm_nt_f16` / `embed_gather_f16`; decode GEMV M≤8, eight columns per warp; tiled M>8). Tied models reuse the embed pointer for lm_head. Other K-quants (Q2_K / Q3_K) and legacy quants (Q4_1 / Q5_0 / Q5_1 / Q8_1) remain Fatal. Decode fuses residual+RMSNorm and Q/K RoPE into one launch each; decode RMSNorm is a single-block reduction. Prefill RMSNorm of N token rows is one launch (`allpaca_rmsnorm_rows`, one block per row). Prefill Q/K RoPE of N tokens is one launch (`allpaca_rope_qk_rows`; position is `start_pos + token`). Prefill writes packed K/V into HIP pages as one D2D copy per page run (not per token). Prefill embed gather of N tokens is one launch (`allpaca_embed_gather_rows` / `allpaca_embed_gather_q8_0_rows` / `allpaca_embed_gather_f16_rows`; token ids H2D once per chunk). Linear speculative verify of a draft span is one Target Model forward (`linear_packed_verifies`; lm_head M=N, then KV `filled` rewind of the rejected suffix). Small Q8_0 tensors (norms, n≤65536) dequant to F32 at load so decode does not stage them per token. A gated 7B/8B Llama GGUF keeps Q8_0 tensors quantized on device. Embeddings stay 36-byte linear blocks (F32 scale + aligned qs) for gather. GEMM weights are the same 36-byte column-major blocks, staged at load via n,k-aware `stage_q8_0_gemv`. F32 GEMM with M>8 uses hipBLASLt when the ROCm SDK links and Tensile heuristic returns an algo (column-major NT mapping inside `kernels.cpp`; rocBLAS still fallback on miss). Decode GEMV (M≤8) stays on custom kernels: hipBLASLt matched rocBLAS in losing on skinny lm_head (~378 ms vs ~0.5 ms custom). Q8 decode (M≤8) stays on the 36-byte GEMV; Q8 prefill (M>8) uses a tiled custom GEMM that dequants B tiles into shared memory so weight blocks are reused across A rows. There is no rocBLAS/hipBLASLt Q8 path. Prompt prefill on HIP runs each `prefill.chunk_tokens` chunk as one forward with M=N GEMM (hipBLASLt when N>8 on F32; tiled Q8 when N>8 on Q8_0; tiled F16 when N>8 on packed F16); decode stays per-token (`forward_token` / M=1 custom GEMV + Q8 GEMV + F16 GEMV). Host Q8 match tests still pass packed 34-byte GGUF bytes into the public API. Sequential target and draft decode keep hidden state in `hipMalloc` across layers; attention reads resident device KV in-place (a single target HIP page is used without a D2D pack; draft DenseKv; no D2H gather / H2D QKV). Decode/prefill attention is a warp-per-head online-softmax kernel with coalesced (float4) KV reads, not a 3-pass one-thread-per-head loop. `Engine::generate` reports `backend = "hip"`, `gpu_forwards`, `weight_bytes_on_device`, `kv_bytes_on_device`, `draft_kv_bytes_on_device`, `elementwise_gpu_ops`, `residual_gpu_ops`, `activation_h2d_bytes`, and `activation_d2h_bytes`. Decode activation H2D stays flat; D2H is logits-only for CPU sampling (one async pinned D2H per token, waited before the next sample; draft logits/tokens may host-stage to the target). Unconstrained `step` does not clone the vocab each token. Without the feature, without an SDK, or with 0 devices / `cpu_only`, the CPU fallback is used. CUDA is post-MVP. |
| **CPU Fallback** | A slow but functional kernel implementation using AVX2 intrinsics. Required for every operation so the engine runs without GPU. |
| **Tokenizer** | Converts text ↔ tokens. Uses HuggingFace `tokenizers` crate. Vocab comes from GGUF metadata or a SafeTensors/ONNX sidecar JSON; decoupled from weight layout. GGUF `tokenizer.ggml.model = gpt2` (Llama-3) encodes with BPE plus a ByteLevel pre-tokenizer/decoder (`add_prefix_space = false`) so spaces become `Ġ` pieces before merges; WordLevel+Whitespace is only the non-gpt2 fallback. Decode maps GPT-2 `Ġ` (U+0120) and SentencePiece `▁` (U+2581) to spaces, and `<0xHH>` byte tokens to bytes. |
| **Request** | A generation task with its own KV cache (`KvHandle` / KV Pages) and decode state. Multiple Requests can be active up to `engine.max_active_requests` (compiled default 1). A saturated scheduler returns Transient `RequestBusy` backpressure. Prompt prefill runs in chunks of `prefill.chunk_tokens` (compiled default 512); `begin_request` admits and processes one chunk, then `step` continues (`StepResult::Prefill`, `prefill_chunks`) so a second Request can admit or decode between chunks. On HIP a chunk of N>1 tokens is one forward with M=N GEMM (hipBLASLt when N>8 on F32; tiled Q8 when N>8 on Q8_0; tiled F16 when N>8 on packed F16); decode of each new token stays M=1. |
| **Telemetry** | Internal metrics (TTFT, ITL, acceptance rate, page migrations, HIP decode profile buckets `q8_gemm_us` / `attention_us` / `lm_head_us` / `other_gpu_us` / `host_us`) stored in memory, exposed via `/stats` JSON endpoint. On HIP, `host_us` is the ITL remainder after GPU kernel events so the profile sum tracks token spacing. Uses `tracing` crate for structured logs. Prometheus/OpenTelemetry export is post-MVP. |
| **Multimodal** | Vision/audio encoder support. Post-MVP; deferred until text-only asymmetric scheduling is proven. |
| **Transfer Path** | The mechanism for moving tensors between devices. At boot the engine queries `hipDeviceCanAccessPeer`, enables both directions (`hipDeviceEnablePeerAccess`), re-queries, then benchmarks that enable+peer-copy against host staging (GPU→DRAM→GPU). `P2p` is chosen only when enable+copy succeed and measured P2P is not slower than host staging by more than 10% (`P2P_MAX_SLOWER_FRACTION`). Host staging is the fallback when P2P is unavailable or slower. Never assume P2P is faster. Speculative generate does **not** require P2P: target stays on the fast GPU, draft stays on the slow GPU, and only token-scale payloads are host-staged (`host_staging_bytes` / `draft_tokens_copied`; `weight_bytes_migrated` stays 0). |
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
| **7B HIP generate** | Gated integration: `Engine` loads a real Llama-family ~7B/8B GGUF on HIP and `generate()` runs. Skips unless `ALLPACA_7B_GGUF` / `ALLPACA_GGUF` (or a discovered local file) is present. Weights are **not** checked into git. Optional draft uses `ALLPACA_DRAFT_GGUF`. Optional Q4_K uses `ALLPACA_Q4K_GGUF` (or `ALLPACA_7B_GGUF` when that file is Q4_K). Optional mixed Q4_K_M uses `ALLPACA_Q4KM_GGUF` (skips 27B/31B-class files). | Manual / self-hosted when a GGUF is available |

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

Sampling (temperature, top-k, top-p, grammar masking, RNG) happens on **CPU**. On HIP, decode hidden state stays device-resident for the whole sequential target forward; logits are copied once per token from GPU to a vocab-sized pinned buffer (async, then waited before the next sample). GPU sampling is a post-MVP optimization.

## Server API (MVP)

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | Chat generation (streaming and non-streaming) |
| `GET /v1/models` | List loaded model info |
| `GET /stats` | Internal metrics JSON (non-OpenAI) |
