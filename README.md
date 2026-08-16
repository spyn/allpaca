# Allpaca

Research inference engine with a thick Rust control plane and a thin C/HIP kernel ABI. The MVP runs a CPU backend end-to-end (tokenize → decode → sample) and exposes an OpenAI-compatible HTTP server.

## Build

CPU-only (default; no ROCm required):

```bash
cargo build --workspace
cargo test --workspace
```

Optional HIP path (AMD GPU; `generate()` uses HIP GEMM + attention):

```bash
# Needs the HIP/ROCm SDK. On Windows the crate probes
# HIP_PATH, ROCM_PATH, and C:\Program Files\AMD\ROCm\*.
# MSVC (vcvars64) is required so hipcc can find the C++ STL.
# Without an SDK, `--features hip` still compiles (stub; GPU tests skip).

cargo test --workspace --features hip
cargo run -p allpaca-server --features hip -- --synthetic --port 8080
```

Windows example (ROCm 7.1 + RX 9070 / 9060 XT):

```bat
set HIP_PATH=C:\Program Files\AMD\ROCm\7.1
set ROCM_PATH=C:\Program Files\AMD\ROCm\7.1
cargo test --workspace --features hip
```

Kernels live in `hip/` (CMake + hipcc). `allpaca-hip/build.rs` compiles them when the SDK is found. GPU-gated tests are `#[cfg(feature = "hip")]` and return if `hipGetDeviceCount() == 0`.

## Run (CPU-only)

Built-in tiny synthetic model (no GGUF file):

```bash
cargo run -p allpaca-server -- --synthetic --cpu-only --port 8080
```

Then:

```bash
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/stats
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"synthetic","messages":[{"role":"user","content":"hello"}],"max_tokens":8}'
```

Load a GGUF (`--model` / `model.path`; missing file is Fatal). Tiny F32/Q4_0/Q8_0 files run on CPU. A real Llama-family ~7B/8B GGUF is a **gated HIP** path (Q8_0 stays quantized on device so a 16GB card does not OOM):

```bash
# CPU tiny / synthetic GGUF
cargo run -p allpaca-server -- --config allpaca.toml --model path/to/model.gguf --cpu-only

# Real 7B-class GGUF on HIP (set the path; do not commit weights)
set ALLPACA_7B_GGUF=E:\path\to\Llama-3-8B-Q8_0.gguf
cargo test -p allpaca-core --features hip --test hip_runtime hip_generate_real_7b -- --nocapture
cargo run -p allpaca-server --features hip -- --model %ALLPACA_7B_GGUF%
```

Optional speculative draft (second GPU) uses `--draft-model` / `ALLPACA_DRAFT_GGUF`. If that file is absent, tests skip.

CLI flags override the TOML file; the file overrides compiled defaults.

Lightweight bench:

```bash
cargo run -p allpaca-bench -- --tokens 32 --prompt "hello world"
```

## Crate layout

| Crate | Role |
|-------|------|
| `allpaca-types` | DeviceInfo, KvPage, Config, PipelineMode, FeatureFlags, errors |
| `allpaca-core` | Engine, scheduler, KV tiers, draft providers |
| `allpaca-gguf` | GGUF parse / tokenizer extract |
| `allpaca-hip` | HIP FFI (feature `hip`) |
| `allpaca-cpu` | CPU fallback kernels |
| `allpaca-server` | `POST /v1/chat/completions`, `GET /v1/models`, `GET /stats` |

`hip/` holds the C ABI (`allpaca_gemm`, `allpaca_attention`, `allpaca_copy_async`). Kernels contain no scheduling logic.

## Configuration

See compiled defaults in `allpaca-types` (`Config::compiled_defaults`) and CONTEXT.md. MVP fields: engine devices, speculative providers (ngram + draft model), 2-tier KV (VRAM + pinned DRAM), CPU sampling, JSON mode.
