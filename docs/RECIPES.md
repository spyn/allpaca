# Operator recipes

Copy-paste commands for this tree. Measured numbers live in [`BENCHMARK.md`](BENCHMARK.md); do not invent them here. Experimental design stays in [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md).

Always `source ~/.cargo/env` on this host (rustup). HIP tests:

```bash
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
# unset ALLPACA_QWEN ALLPACA_NEMOTRON ALLPACA_OVERNIGHT for the default suite
```

Do not weaken the five HIP functional asserts. Do not apt-install Ubuntu HIP debs on Debian 13.

## Quality generate (Perth gate)

Shared HIP probe prompt (default when `ALLPACA_QWEN_PROMPT` is unset):

```text
<|im_start|>user
What is the capitol of Western Australia<|im_end|>
<|im_start|>assistant
```

The completion must contain `perth` (case-insensitive) and at least 4 unique tokens.

```bash
# Nemotron Lightning 30B-A3B ROCmFP4 (absolute path — cargo test cwd is the crate)
export ALLPACA_NEMOTRON=1
export ALLPACA_ROCMFP4_GGUF="$PWD/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-COHERENT.gguf"
export ALLPACA_QWEN_N=32
cargo test -p allpaca-core --features hip --test hip_runtime \
  hip_rocmfp4_qwen_probe_and_optional_generate -- --nocapture --test-threads=1
```

Qwen 27B ROCmFP4 (same Perth gate; GDN asserts stay Qwen-only):

```bash
unset ALLPACA_ROCMFP4_GGUF
export ALLPACA_QWEN=1
cargo test -p allpaca-core --features hip --test hip_runtime \
  hip_rocmfp4_qwen_probe_and_optional_generate -- --nocapture --test-threads=1
```

Sidecar `mtp-*.gguf` is unused. In-file `nextn_predict_layers` is the engine MTP path.

## Default HIP suite (no overnight / no 30B)

```bash
unset ALLPACA_QWEN ALLPACA_NEMOTRON ALLPACA_OVERNIGHT
cargo test -p allpaca-core --features hip --test hip_runtime -- --test-threads=1
```

## Tiny Lightning (CPU + HIP)

CPU: sigmoid + ReLU² + shared expert fixture (no 17 GiB weights):

```bash
cargo test -p allpaca-core \
  nemotron_lightning_shexp_sets_routing_flags_and_generates \
  -- --nocapture
```

HIP batched prefill (one `gpu_forward` per prompt chunk; tokens must match CPU):

```bash
cargo test -p allpaca-core --features hip \
  hip_lightning_shexp_long_prompt_batches_prefill -- --nocapture
```

## Dual-GPU overnight (9070 + 9060 box only)

Different host from sunny / dual gfx906. Keep both GPUs visible.

```bat
set ALLPACA_OVERNIGHT=1
set ALLPACA_OVERNIGHT_9060=1
cargo test -p allpaca-core --features hip --test hip_runtime hip_overnight_llama32_f16_on_second_gpu_when_env_set -- --nocapture
```

Short smoke: `ALLPACA_OVERNIGHT_ROUNDS=1` and `ALLPACA_OVERNIGHT_TOKENS=16`.

## Vision LayerNorm (HIP)

Mean/var LN (not RMSNorm) for ViT ln1/ln2 and CLIP `ln_pre`. Kernel must match CPU per row; resident ViT encode must match CPU tokens.

```bash
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
cargo test -p allpaca-hip --features hip tests::layernorm_rows_matches_cpu_per_row -- --exact
cargo test -p allpaca-core --features hip --lib vision::tests::vit_block_hip_encode_matches_cpu -- --exact
```

## Token-parallel MoE (HIP prefill)

Routed experts: one GEMM per unique expert with `M = tokens that selected it`. Shared expert: `M = n_tok`. HIP tokens must match CPU; `gpu_forwards` still one per prompt chunk.

```bash
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
cargo test -p allpaca-core --features hip \
  hip_lightning_shexp_long_prompt_batches_prefill -- --nocapture
```

## ChatML (no `<think>` on Nemotron)

`/v1/chat/completions` uses ChatML when the vocab has `<|im_start|>` / `<|im_end|>`. Those pieces are default stop tokens. Do not open `<think>` on Nemotron.

Nemotron `hello` is not a quality gate: ChatML without think role-loops `hello user`; NVIDIA’s closed `<think></think>` prefix stops that loop but still does not greet. Use the Perth ChatML prompt (absolute `ALLPACA_ROCMFP4_GGUF`) instead.

## HuggingFace Hub (SafeTensors / GGUF)

```bash
# Dense Llama-family SafeTensors (F32/F16/BF16). Gated models: export HF_TOKEN=...
cargo run -p allpaca-server -- --model Qwen/Qwen2.5-0.5B-Instruct --cpu-only

# Snapshot dir already on disk (config.json + model.safetensors + tokenizer.json)
cargo run -p allpaca-server -- --model ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<sha>

# One GGUF in a multi-file Hub repo
cargo run -p allpaca-server -- \
  --model bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q8_0.gguf
```

Cache: `ALLPACA_HF_CACHE` / `HF_HUB_CACHE` / `$HF_HOME/hub` / `~/.cache/huggingface/hub`. Offline: `HF_HUB_OFFLINE=1` (cache hit only). Mixtral-style MoE SafeTensors is not loaded yet.

## DeepSeek-V4-Flash + DSpark

Unsloth `DeepSeek-V4-Flash-0731` is a split GGUF (`UD-Q8_K_XL/*-00001-of-00005.gguf`, architecture `deepseek4`). Point `--model` at shard 00001, the `UD-Q8_K_XL` directory, or the Hub snapshot. UD-Q8_K_XL is Unsloth Dynamic (BF16 dense + MXFP4 experts), not ggml Q8_0. HIP decode uses packed BF16 MLA/shexp/lm_head GEMV (gfx906 expands to F32) and packed MXFP4 expert GEMV with an expert LRU plus optional GPU1 layer prefill buffer. Generic GGUF→F32 HIP upload is skipped. Do not F32-materialize the 149 GB file. `--cpu-only` keeps the CPU gemv path (correctness / tiny F32 fixtures).

DSpark is a **sidecar**, not in-file NextN: Unsloth puts `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` (~10.9 GB) at the **snapshot root** (one copy for every quant folder) and the BF16 sibling under `dspark/`. Markov `w1`/`w2` stay packed BF16 (no 132 MB F32 materialize at load). On HIP the w2 GEMV runs on the draft GPU when present. Point `--speculative-providers dspark` or `--draft-model` at that file. Missing sidecar warns and continues (`degraded`). HuggingFace CLI timeouts are usually this extra download (`HF_HUB_ENABLE_HF_TRANSFER=1` / a long resume), not allpaca’s HTTP client.

```bash
# Target already on disk (example cache layout)
export HF_HOME=/zepool/cache/huggingface
SNAP="$HF_HOME/hub/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/snapshots/<sha>"
cargo run -p allpaca-server --release --features hip -- \
  --model "$SNAP/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf"

# CPU-only correctness path
cargo run -p allpaca-server --release -- --cpu-only \
  --model "$SNAP/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf"

# Sidecar (~10.9 GB) — snapshot root, not the UD-Q8_K_XL folder
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
  --include '*dspark-DeepSeek-V4-Flash-0731-Q8_0*'
# --speculative-providers dspark
# or --draft-model "$SNAP/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
```

Opt-in real-weight smoke (gated; one token is slow on CPU, faster on HIP):

```bash
ALLPACA_DSV4=1 ALLPACA_DSV4_GENERATE=1 cargo test -p allpaca-core --release --offline --lib unsloth_dsv4 -- --nocapture
ALLPACA_DSV4=1 ALLPACA_DSV4_HIP=1 ALLPACA_DSV4_GENERATE=1 \
  cargo test -p allpaca-core --features hip --release --offline --lib unsloth_dsv4 -- --nocapture
```

Tiny fixtures (no 149 GB weights):

```bash
cargo test -p allpaca-core tiny_deepseek4_cpu_generates tiny_dspark_sidecar_is_used_when_present -- --nocapture
cargo test -p allpaca-core --features hip --offline --test hip_runtime tiny_deepseek4 -- --nocapture
```

## Ideas still out (next slices)

- Packed decode Q/K/V/O C feeding attention (lm_head skip on greedy HIP cohort landed).
- CUDA 7B / MoE / hybrid parity (dense TinyModel CUDA decode + CUDA KV pages landed).
- Real CXL.mem / PIM (simulators only).
