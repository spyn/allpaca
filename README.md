# Allpaca

Research inference engine with a thick Rust control plane and a thin C/HIP kernel ABI. The MVP runs a CPU backend end-to-end (tokenize → decode → sample) and exposes an OpenAI-compatible HTTP server.

## Build

CPU-only (default; no ROCm required):

```bash
cargo build --workspace
cargo test --workspace
```

Optional HIP bindings (needs ROCm + a built `liballpaca_kernels`):

```bash
cmake -S hip -B hip/build
cmake --build hip/build
cargo build -p allpaca-server --features hip
```

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

Load a GGUF (F32 weights only in this MVP):

```bash
cargo run -p allpaca-server -- --config allpaca.toml --model path/to/model.gguf --cpu-only
```

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
