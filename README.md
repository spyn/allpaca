# Allpaca

Research inference engine with a **thick Rust control plane** and a **thin C/HIP kernel ABI**.
It treats the whole machine — AMD GPUs, host DRAM, NVMe, and CPU — as one elastic serving
platform, not a single undersized VRAM pool.

OpenAI-compatible HTTP server · GGUF / HuggingFace SafeTensors / ONNX weight import · Hub download · CPU always works ·
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
| No Windows DirectStorage path | Stable **DirectStorage 1.3** file→D3D12 DMA into opt-in HIP-imported buffers; ordinary weight uploads stay plain `hipMalloc` + H2D; aligned host-staged fallback |
| Single-request hybrid sync barrier | **Micro-batch** partner `poll` + **layer-stagger** CPU experts during GPU forward |
| Reactive VRAM only | **Elastic expert↔KV rebalance**, attention-trim cold KV, **`gpu.util` / `gpu.vram`** from `hipMemGetInfo` (no AMD SMI on Windows) |
| Narrow GGUF quants | **F16 / BF16 / I32 / MXFP4 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q8_1 / Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / IQ4_NL / IQ4_XS / TQ1_0 / TQ2_0** + **ROCmFP4 (`ggml_type_id=100`)** CPU dequant + fused HIP GEMV where wired (IQ4_* dequant to F32 at load; no fused HIP GEMV yet). Split `*-NNNNN-of-MMMMM.gguf` catalogs merge. |
| MoE-only / no hybrid SSM | **`nemotron_h_moe` schedule**: full Mamba-2 SSD + attention + MoE on CPU/HIP (incl. Lightning gate-less `up`/`down`); real ROCmFP4 opt-in via `ALLPACA_NEMOTRON=1` |
| Proprietary weight banks only | **GGUF** primary + **packed layer-major / `ALLPFTW1`** trunk for streaming |
| No continuous batching story | Fair **`Engine::poll`**, chunked prefill, multi-request backpressure |
| Speculative decode absent or linear-only | N-gram + draft-model + **MTP** + **DSpark** (`dspark` / `dflash`); **linear packed verify**; optional **tree / Token Forest** (overlaps MTP draft-ahead) |
| Structured output bolted on | CPU **JSON / JSON Schema / Lark** via `llguidance` |
| Early-exit unused | Manifest **`allpaca.early_exit.*`** + optional dedicated **`exit_head`** tensors; self-speculate when heads absent |
| NUMA-blind host experts | **NUMA first-touch** on the host expert pool |

Still out of MVP (documented, not claimed): full CUDA engine parity, CXL/PIM.
Coral/NPU backends are dropped.

---

## Features (quick map)

- **Control plane:** scheduling, KV tiers (VRAM / pinned DRAM / optional NVMe), weight working set, speculation, constraints — all Rust ([ADR 0001](docs/adr/0001-thick-rust-control-plane.md)).
- **Kernels:** `hip/` C ABI (`allpaca_gemm*`, attention, RoPE, RMSNorm, …); `allpaca-cpu` AVX2 fallbacks.
- **Formats:** GGUF (Llama-family + hybrid Nemotron + **DeepSeek-V4 `deepseek4`**, including split shards and BF16/MXFP4), HuggingFace SafeTensors (`config.json` + `tokenizer.json`, F32/F16/BF16, sharded `index.json`) with Hub `org/name` download, ONNX weight-export.
- **Server:** `POST /v1/chat/completions`, `POST /v1/systemone` (decision models), `GET /v1/models`, `GET /stats`.
- **Config:** TOML + CLI; CLI overrides file; file overrides compiled defaults (`Config::compiled_defaults`).

---

## Build

Requires a recent stable Rust toolchain (`rustup`, 1.88+). Debian's `rustc` 1.85
cannot resolve current workspace crates (`icu_*`, `llguidance`, `zip`). Put
`$HOME/.cargo/bin` first on `PATH`.

### CPU (default)

```bash
cargo build --workspace
cargo test --workspace
cargo build -p allpaca-server --release
```

### HIP / AMD GPU (optional)

Needs the HIP/ROCm **compiler** SDK (`hipcc`, `hip_runtime.h`), not only the
runtime in `/opt/rocm`. On Linux the crate probes `HIP_PATH`, `ROCM_PATH`,
`/opt/rocm*`, and `~/.cache/allpaca/rocm-sdk/opt/rocm-*`. On Windows it probes
`C:\Program Files\AMD\ROCm\*`. MSVC (`vcvars64`) is required so `hipcc` finds
the C++ STL. Without an SDK, `--features hip` still compiles (stub; GPU tests skip).

Offload ISAs default to live `rocminfo` GPUs when present (this box: **gfx906**
for MI50 / MI60 / Radeon VII). Override with `AMDGPU_TARGETS=gfx906,gfx900`.

#### Linux compiler SDK (Debian 13 / runtime-only `/opt/rocm`)

A runtime-only `/opt/rocm` is completed **without root** by
`scripts/fetch-rocm-hip-sdk.sh` (ROCm 6.2.4 `hipcc` + LLVM extracted into
`~/.cache/allpaca/rocm-sdk/opt/rocm-6.2.4`, then symlinked to the `/opt/rocm`
runtime `.so` files). `build.rs` picks that cache path up automatically.

**Do not** `sudo apt install hipcc6.2.4 hip-dev6.2.4 rocm-llvm6.2.4 hsa-rocr-dev6.2.4`
on Debian 13 (Trixie). Those are Ubuntu noble debs and depend on
`libstdc++-11-dev` / `libgcc-11-dev`, which Trixie does not ship (gcc-14 only).
The fetch script uses `apt-get download` + `dpkg-deb -x` so the same debs never
have to be installed.

New shells need rustup on `PATH`:

```bash
source ~/.cargo/env   # once per session until bashrc is reloaded
```

Debian's distro `rustc` 1.85 cannot build this workspace. Use rustup 1.88+.

#### Dual-GPU P2P (Linux kernel)

Stock Debian kernels often ship **`# CONFIG_PCI_P2PDMA is not set`**. On gfx906
that makes `hipDeviceCanAccessPeer` return 0 and `hipDeviceEnablePeerAccess`
fail (`invalid device ordinal`) even when **`hipMemcpyPeer` still copies**.

A custom kernel with `CONFIG_PCI_P2PDMA=y` is what KFD needs for the formal
enable path. After reboot:

```bash
grep CONFIG_PCI_P2PDMA /boot/config-$(uname -r)
# expect: CONFIG_PCI_P2PDMA=y
```

Optional cmdline if enable still fails (IOMMU groups split the two cards):
`intel_iommu=on iommu=pt`, or `intel_iommu=off`. There is no xGMI cable on a
typical dual-MI50 desktop; `use_xgmi_p2p` does not apply.

The engine does **not** require EnablePeerAccess. At boot it:

1. tries enable both directions (best-effort),
2. treats a successful 64 KiB `hipMemcpyPeer` as capable,
3. benches 4 MiB peer-copy vs host staging (GPU→DRAM→GPU),
4. selects `P2p` only when the copy worked **and** measured P2P is not more
   than 10% slower than host (`P2P_MAX_SLOWER_FRACTION`).

Never assume P2P is faster. Speculative generate does not require P2P: target
and draft stay on their GPUs; only token-scale payloads move.

After a P2PDMA kernel lands, re-measure (look for `p2p_enabled` plus `p2p_gbps`):

```bash
source ~/.cargo/env
cargo test -p allpaca-core --features hip --test hip_runtime \
  transfer_path_is_measured -- --nocapture
```

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

### CUDA / NVIDIA GPU (optional scaffold)

Needs the CUDA toolkit (`nvcc`). Probes `CUDA_PATH` / `CUDA_HOME`, PATH `nvcc`,
`~/.cache/allpaca/cuda-sdk/cuda-*` (from `scripts/fetch-cuda-toolkit.sh`),
`/usr/local/cuda*`, and
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*`. Without a toolkit,
`--features cuda` still compiles (stub). Without an NVIDIA driver/GPU,
`device_count()` is 0 — you can still compile and link on a toolkit-only machine.

**GPU arches (dual toolchain):** `ALLPACA_CUDA_ARCH` / `CUDA_ARCHES` /
`CUDAARCHS` / `TORCH_CUDA_ARCH_LIST`, else live `nvidia-smi` (driver 470 rejects
the `compute_cap` query, so those boxes skip this step), else a toolkit-aware
fat default: **CUDA ≤11.4** includes `sm_35` and `sm_37`; **CUDA 12+** defaults
to **sm_86** / **sm_89** / **sm_120** and filters Kepler out.
Attention uses all-lane `__shfl_xor` (pre-Volta) on `sm < 70`.

Example (modern): `ALLPACA_CUDA_ARCH=sm_86,sm_89,sm_120 cargo test -p allpaca-cuda --features cuda`.

#### Tesla K80 / driver 470 (Kepler)

K80 is **sm_37** and Quadro K6000 is **sm_35**; a `sm_37` binary fails on the
K6000 with `rc=3`. Driver 470 advertises CUDA **11.4** max; CUDA **12** removed
Kepler. Stage a matching toolkit without root:

```bash
./scripts/fetch-cuda-toolkit.sh   # ~50 MiB redistributables → ~/.cache/allpaca/cuda-sdk/cuda-11.4
# Ubuntu 22.04: CUDA 11.4 prefers g++-10
sudo apt install g++-10
export CUDA_PATH="$HOME/.cache/allpaca/cuda-sdk/cuda-11.4"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
export CXX=g++-10
ALLPACA_CUDA_ARCH=sm_37 cargo test -p allpaca-cuda --features cuda
```

```bash
cargo test -p allpaca-cuda --features cuda
```

Kernels live in `cuda/`. Symbols are `allpaca_cuda_*` so HIP and CUDA can coexist.
TinyModel F32 decode (KV pages + device attention, tiny F32/Q4_K/F16 GGUF)
runs on CUDA when a device is present. 7B packed-quant still HIP.

Kepler (K6000 `sm_35`, K80 `sm_37`) needs CUDA ≤11.4 (`scripts/fetch-cuda-toolkit.sh`); both are in the fat default only on CUDA ≤11.x toolkits.

Dual K80 (two compute devices in one board) shows up as two ordinals; prefer `cpu_only` coordinator + `cuda:0` worker (or CUDA coordinator + CPU worker) to avoid two concurrent CUDA contexts.

Dense TinyModel CUDA decode, KV pages, RPC `KernelOp` workers, P2P probe, memset,
and theoretical `bandwidth_gbps` are in-tree. **TMA / tensor-core GEMM is not:**
`allpaca_cuda_gemm_nt` is still the F32 path that also runs on Ampere (`sm_86`).
Do not enable a Blackwell-only kernel on this branch until a host with `sm_120`
has run the checklist below.

### CUDA GPU host checklist (3060 or Blackwell laptop)

Toolkit-only machines (`nvcc` but no `nvidia-smi` / driver) compile and skip
device tests (`device_count() == 0`). On a box with a driver, the same tests
**must not skip**. Confirm the live arch first:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
# RTX 3060 12GB → compute_cap 8.6 (sm_86), ~360 GB/s theoretical
# Blackwell GeForce laptop → compute_cap 12.0 (sm_120)
```

```bash
# Default fat binary already includes sm_86, sm_89, sm_120.
cargo test -p allpaca-cuda --features cuda -- --nocapture
cargo test -p allpaca-core --features cuda --test cuda_runtime -- --nocapture
cargo test -p allpaca-rpc --features cuda -- --nocapture

# Two-process CUDA RPC worker (needs a device; skip unless env is set):
ALLPACA_RPC_CUDA=1 cargo test -p allpaca-rpc-server --features cuda -- gated_cuda_two_process -- --nocapture

# Optional: faster compile for one SKU
ALLPACA_CUDA_ARCH=sm_86 cargo test -p allpaca-cuda --features cuda
ALLPACA_CUDA_ARCH=sm_120 cargo test -p allpaca-cuda --features cuda
```

Gates that should **fail** (not skip) when a GPU is present:

| Check | Where |
| --- | --- |
| `device_sm_arch` is `sm_86` / `sm_89` / `sm_120` | `allpaca-cuda` |
| `bandwidth_gbps` in (10, 50000) | `enumerate_devices_reports_theoretical_bandwidth_when_present` |
| P2P same-device roundtrip; two-GPU peer list | `memcpy_peer_*` |
| memset / pin / batch H2D | `allpaca-cuda` |
| Engine `backend=cuda`, MobileTiered, VRAM stats, `gpu_util>0` | `cuda_runtime` |
| RPC GEMM / residual / SiLU / embed / clear | `allpaca-rpc` loopback_cuda_* |

Blackwell TMA follow-up (other host, later slice): keep the F32 GEMM for `sm_86`,
add a `sm_90+` / `sm_120` path behind an arch guard, and extend
`gemm_nt_matches_cpu_when_device_present` plus `loopback_cuda_gemm_nt` so both
SKUs stay bit-close to CPU F32. Do not drop Ampere from the fat binary.

Server: `cargo run -p allpaca-rpc-server --features cuda -- --device cuda:0`.
Engine: `gpu_backend = "cuda"` (see `cuda_runtime` synthetic config).

### Vulkan (optional rust-gpu scaffold)

Needs a Vulkan 1.3 loader/device. `--features vulkan` compiles first-party
compute shaders with [rust-gpu](https://rust-gpu.github.io/) (nested nightly via
`cargo-gpu-install`) and dispatches them with `ash`. Without a successful SPIR-V
build or GPU, `device_count()` is 0 and the crate still compiles (stub).

GGUF F16 and Q8_0 GEMM weights stay packed on device (rust-gpu unpack + F32 MAC).
When `glslangValidator` is on `PATH` / `GLSLANG_VALIDATOR` / `VULKAN_SDK` (or
`C:\VulkanSDK\*\Bin` on Windows) and the GPU has `shaderFloat16`, F16 GEMV uses
a GLSL native-`float16_t` pipe instead. Q4_K is still host-dequant then F32.

```bash
cargo test -p allpaca-vulkan --features vulkan
cargo test -p allpaca-core --features vulkan --test vulkan_runtime
```

Set `ALLPACA_SKIP_VULKAN_SHADERS=1` to skip the rust-gpu nested build (stub
only). Auto backend order is HIP, then CUDA, then Vulkan, then CPU. Explicit
`gpu_backend = "vulkan"` / `vulkan:N` selects Vulkan. This is a TinyModel
scaffold (KV pages + device attention, tiny F32 / packed F16/Q8_0 / host-dequant
Q4_K GGUF), not HIP Q4_K / SSM / GDN / MLA parity
([ADR 0005](docs/adr/0005-vulkan-rust-gpu.md)).

---

## Run

### Synthetic smoke (no weights)

```bash
cargo run -p allpaca-server -- --synthetic --cpu-only --host 127.0.0.1 --port 8080 --ui-password secret
```

Open `http://127.0.0.1:8080` and sign in. If `--ui-password` / `[http] password` is omitted, the server prints `webui password: …` once to stderr (not tracing / `GET /v1/logs`). The same secret is `Authorization: Bearer` for API clients.

Headless on a LAN: `--host 0.0.0.0` plus a password. There is no TLS in v1.

```bash
curl http://127.0.0.1:8080/v1/models -H "authorization: Bearer secret"
curl http://127.0.0.1:8080/stats -H "authorization: Bearer secret"
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "content-type: application/json" \
  -H "authorization: Bearer secret" \
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
| `ALLPACA_QWEN=1` | Opt-in Qwen35 ROCmFP4 generate. Discovers `models/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF/Qwen3.8-27B-Q4_0_ROCMFP4_COHERENT.gguf` (ggml type 100; in-file `qwen35.nextn_predict_layers`). Unsloth `Qwen3.8-27B-UD-Q5_K_XL.gguf` loads (IQ4_NL / IQ4_XS dequant to F32 at load). Sidecar `mtp-*.gguf` files are unused; MTP tests load NextN from the same ROCmFP4 GGUF. |
| `ALLPACA_QWEN_N` | Decode length for the opt-in Qwen tests (default 32; use `8` for a short HIP smoke). |
| `ALLPACA_ROCMFP4_GGUF` / `ALLPACA_QWEN_GGUF` | Explicit ROCmFP4 GGUF path (probe always discovers under `models/`) |
| `ALLPACA_DEMO=1` | Asymmetric draft vs greedy tok/s |
| `ALLPACA_OVERNIGHT=1` | Long draft soak |
| `ALLPACA_OVERNIGHT_9060=1` | Dual-GPU bakeoff: optional HIP:1 soak (prefers Q8) |
| `ALLPACA_OVERNIGHT_9060_Q8=0` | Force F16 for the 9060 soak leg |

#### Dual-GPU overnight (Q8 9060) — operator runbook

Reference dual-HIP box (RX 9070 target + RX 9060 XT draft/soak). Keep both GPUs visible (do **not** set `HIP_VISIBLE_DEVICES`). Draft / 9060 soak weights resolve via test helpers (`overnight_9060_gguf`, `demo_draft_gguf`): prefer `models/Llama-3.2-3B-Instruct-Q8_0.gguf` when present (then 1B Q8; F16 only if Q8 missing or forced off).

```bat
REM Full dual-GPU bakeoff with Q8 9060 soak (default when Q8 GGUF exists)
set ALLPACA_OVERNIGHT=1
set ALLPACA_OVERNIGHT_9060=1
REM optional: set ALLPACA_OVERNIGHT_9060_Q8=0   REM force F16 9060 soak (~4.5 tps historically)
cargo test -p allpaca-core --features hip --test hip_runtime hip_overnight_llama32_f16_on_second_gpu_when_env_set -- --nocapture > overnight-q8-9060.log 2>&1
```

Short smoke: `ALLPACA_OVERNIGHT_ROUNDS=1` and `ALLPACA_OVERNIGHT_TOKENS=16`.

**Ballpark (Q8 9060 soak, warm rounds):** 9070 greedy ~33 tps · 9060 Q8 greedy ~32 tps · asymmetric spec acceptance ~0.99. Example local artifact name: `overnight-q8-9060.log` (do not commit; see `.gitignore`).

### Bench helper

```bash
cargo run -p allpaca-bench -- --tokens 32 --prompt "hello world"
```

### Useful CLI flags (`allpaca-server`)

| Flag | Meaning |
|------|---------|
| `--config PATH` | TOML config (copy `allpaca.example.toml`) |
| `--model PATH` | Target weights |
| `--draft-model PATH` | Draft weights |
| `--host` / `--port` | Bind address (default loopback; use `0.0.0.0` in Docker) |
| `--ui-password` | Shared WebUI login and Bearer secret |
| `--cpu-only` | Force CPU backend |
| `--synthetic` | Built-in tiny model |
| `--speculative true\|false` | Toggle speculation |
| `--pipeline-mode asymmetric\|symmetric\|mobile_tiered` | Feature-flag profile |
| `--target-device` / `--draft-device` | Device selectors |
| `--rpc host:port,...` | Remote kernel-ABI workers (layer-split) |
| `--rpc-token` | Shared RPC HELLO secret (`ALLPACA_RPC_TOKEN`) |
| `--tensor-split a,b,...` | Relative layer weights (local target, then RPC) |

### RPC workers (multi-machine)

RPC pools VRAM so a model that does not fit one machine can still run. It does **not** make decode faster than a single box that already fits the model. Pin the same allpaca version on every machine (HELLO major must match). Non-loopback worker binds require `--insecure-bind` and `--token` / `ALLPACA_RPC_TOKEN`.

```bash
# worker
cargo run -p allpaca-rpc-server -- --cpu-only -H 127.0.0.1 -p 50052

# coordinator
cargo run -p allpaca-server -- --synthetic --cpu-only --rpc 127.0.0.1:50052 --port 8080
```

HIP worker on a LAN, pooling every GPU in one process: `cargo run -p allpaca-rpc-server --features hip -- --device hip -H 0.0.0.0 -p 50052 --insecure-bind --token "$ALLPACA_RPC_TOKEN"`. `--device hip:N` pins one ordinal. The coordinator uses the same token (`--rpc-token` or `ALLPACA_RPC_TOKEN`) and one `--rpc` endpoint; HELLO lists each GPU, and `--tensor-split` may be one weight per endpoint or one weight per GPU. CUDA worker (same wire ABI, e.g. RTX 3060 / Blackwell laptop): `cargo run -p allpaca-rpc-server --features cuda -- --device cuda:0 -H 0.0.0.0 -p 50052 --insecure-bind --token "$ALLPACA_RPC_TOKEN"`. HIP and CUDA workers are interchangeable after HELLO. A CUDA-only box can still run `--cpu-only` for RAM overflow.

Lab two-NIC procedure (SSH on the management address, `--rpc` on the fast NIC): [docs/rpc-remote-test.md](docs/rpc-remote-test.md).

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
| `allpaca-safetensors` / `allpaca-onnx` | HuggingFace SafeTensors (BF16/F16/F32, shards, Llama-family names) / ONNX importers |
| `allpaca-hip` | HIP FFI (`--features hip`) |
| `allpaca-cuda` | CUDA FFI (`--features cuda`; toolkit builds without a GPU) |
| `allpaca-vulkan` | Vulkan/`ash` host (`--features vulkan`; rust-gpu SPIR-V + optional glslang f16) |
| `allpaca-shaders` | rust-gpu compute crate (not a workspace member; built from `allpaca-vulkan`) |
| `allpaca-cpu` | CPU kernels + dequant |
| `allpaca-server` | OpenAI-compatible HTTP |
| `tools/bench` | Lightweight bench binary |
| `hip/` | C/HIP kernel ABI sources |

---

## HuggingFace models

`--model` / `model.path` accepts a local GGUF/SafeTensors/ONNX file, a snapshot directory (`config.json` + `model.safetensors` or `model.safetensors.index.json` + `tokenizer.json`), or a Hub id:

```bash
cargo run -p allpaca-server -- --model Qwen/Qwen2.5-0.5B-Instruct --cpu-only
# revision: Qwen/Qwen2.5-0.5B-Instruct@main
# one file in a multi-GGUF repo: bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q8_0.gguf
```

Downloads land in the HuggingFace hub cache (`HF_HOME` / `HF_HUB_CACHE` / `ALLPACA_HF_CACHE`). Gated repos need `HF_TOKEN`. Dense Llama / Qwen2 / Mistral / Gemma SafeTensors (F32 / F16 / BF16) load into the engine; Mixtral-style MoE SafeTensors is Fatal for now.

---

## Configuration & docs

- Compiled defaults: `allpaca-types` `Config::compiled_defaults`
- Domain glossary / MVP scope: [`CONTEXT.md`](CONTEXT.md)
- Opportunity → implementation notes: [`docs/OPPORTUNITY_DESIGN.md`](docs/OPPORTUNITY_DESIGN.md)
- Benchmark field contract: [`docs/BENCHMARK_SPEC.md`](docs/BENCHMARK_SPEC.md)
- Measured runs: [`docs/BENCHMARK.md`](docs/BENCHMARK.md) (append a row after every successful generate / transfer)
- Operator recipes: [`docs/RECIPES.md`](docs/RECIPES.md)
- Architecture decisions: [`docs/adr/`](docs/adr/)

---

## Sources & acknowledgements

Policies and formats we studied and re-implemented on our ABI. DirectStorage is the exception:
its redistributable runtime is restored from Microsoft's pinned NuGet package at build time.

| Source | What we took |
|--------|----------------|
| [FreeToken](https://arxiv.org/abs/2608.16157) ([FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)) | Edge MoE: host expert pool, GPU LRU slots, `q*` PCIe/CPU miss split, elastic VRAM, FTW-style banks, semantic anchors |
| kimi-k3-in-c / related dense-offload work | Dense **pin-prefix + ring** (never LRU a cyclic layer walk); packed trunk I/O discipline |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) / [ggml](https://github.com/ggml-org/ggml) | GGUF layout, K-quant block sizes, community quant vocabulary |
| ROCmFP4 / ggml type 100 (AMD community ROCmFP4 GGUF work) | 18 B / 32-elem dual-scale blocks; CPU dequant + fused HIP GEMV |
| [AMD ROCm / HIP](https://rocm.docs.amd.com/) | Multi-device, P2P vs host staging, hipBLASLt where linked |
| [Microsoft DirectStorage](https://learn.microsoft.com/en-us/windows/win32/direct3d12/directstorage) | Stable `Microsoft.Direct3D.DirectStorage` 1.3.0 SDK/runtime; file→D3D12 requests synchronized by a shared fence and imported into HIP |
| [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) | Vocab / BPE encode-decode |
| [Hugging Face Hub](https://huggingface.co/docs/hub/en/models-downloading) + [SafeTensors](https://huggingface.co/docs/safetensors) | `org/name` download, `config.json` / `tokenizer.json`, F32/F16/BF16, sharded `index.json` |
| [llguidance](https://github.com/guidance-ai/llguidance) | Constrained JSON Schema / Lark decoding |
| Speculative decoding literature (draft-then-verify, tree/forest verify) | N-gram + draft-model providers; packed linear / optional tree verify |
| Attention cache work (H₂O / StreamingLLM-style cold trim ideas) | Soft **attention-trim** of cold VRAM KV for expert headroom |

Internal research notes that drove the cut list: [`docs/EXECUTIVE_RESEARCH.md`](docs/EXECUTIVE_RESEARCH.md), [`docs/GEM_RESEARCH.md`](docs/GEM_RESEARCH.md).

---

## License

Copyright (c) 2026 Allpaca contributors. Released under the [MIT License](LICENSE).
