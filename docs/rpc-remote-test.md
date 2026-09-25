# RPC remote test (lab)

Desktop is the coordinator. The MI50 box is the worker. SSH and RPC use different NICs.

| Path | Desktop | Worker (`2× MI50 32GB`, gfx906) |
|------|---------|----------------------------------|
| SSH (management) | `10.1.1.10` | `10.1.1.69` |
| RPC (5 GB/s NIC) | `192.168.69.3` | `192.168.69.2` |

`--rpc` always uses `192.168.69.2`. Do not point it at `10.1.1.69`.

Bind the worker to `192.168.69.2` only. Current `main` refuses that bind unless you pass `--insecure-bind` and a shared HELLO token (`--token` / `ALLPACA_RPC_TOKEN` on the worker, `--rpc-token` / `ALLPACA_RPC_TOKEN` on the coordinator). There is still no TLS. Both checkouts are `~/workspace/allpaca-main-rpc-test` on sunny and this repo, commit `9e7b5dc`.

## Step 0 — Reachability

On the desktop, before the worker is up, port 50052 should refuse. After it is up, the probe should succeed.

```powershell
Test-NetConnection 192.168.69.2 -Port 50052
```

On the worker, allow the desktop fast NIC only:

```bash
sudo ufw allow from 192.168.69.3 to any port 50052 proto tcp
sudo ufw allow from 192.168.69.3 to any port 50053 proto tcp
```

Port 50053 is only for a second GPU worker.

## Step 1 — CPU baseline

SSH in, build, and leave the worker in the foreground.

```bash
ssh spyn@10.1.1.69
cd ~/workspace/allpaca-main-rpc-test
cargo build -p allpaca-rpc-server
cargo run -p allpaca-rpc-server -- --cpu-only --insecure-bind --token "$ALLPACA_RPC_TOKEN" -H 192.168.69.2 -p 50052 -m 256
```

On the desktop, set the same token:

```powershell
cd D:\code\inference\allpaca
$env:ALLPACA_RPC_TOKEN = "<same token>"
cargo run -p allpaca-server -- `
  --synthetic --cpu-only `
  --rpc 192.168.69.2:50052 `
  --rpc-token $env:ALLPACA_RPC_TOKEN `
  --tensor-split 1,1 `
  --port 8080
```

```powershell
curl http://127.0.0.1:8080/stats
```

Pass: generation finishes, `rpc_roundtrips` > 0, `rpc_activation_bytes` > 0, worker log shows `rpc client connected`.

Once the gated test exists and the worker is still running:

```powershell
$env:ALLPACA_RPC_REMOTE = "192.168.69.2:50052"
cargo test -p allpaca-rpc-server gated_remote_synthetic_generate -- --nocapture
```

## Step 2 — HIP on MI50 GPU 0

```bash
ssh spyn@10.1.1.69
/opt/rocm/bin/rocminfo | grep -E "Marketing Name|gfx"
cd ~/workspace/allpaca-main-rpc-test
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
cargo run -p allpaca-rpc-server --features hip -- --device hip:0 --insecure-bind --token "$ALLPACA_RPC_TOKEN" -H 192.168.69.2 -p 50052
```

HELLO should report a Hip backend and about 32 GB. Desktop coordinator command is the same as Step 1. The synthetic coordinator stays `--cpu-only`; remote layers run on the MI50.

```powershell
$env:ALLPACA_RPC_REMOTE = "192.168.69.2:50052"
$env:ALLPACA_RPC_HIP = "1"
cargo test -p allpaca-rpc-server --features hip gated_remote_synthetic_generate -- --nocapture
```

## Step 3 — Optional second MI50

One GPU per `allpaca-rpc-server` process. Second SSH session:

```bash
cargo run -p allpaca-rpc-server --features hip -- --device hip:1 --insecure-bind --token "$ALLPACA_RPC_TOKEN" -H 192.168.69.2 -p 50053
```

```powershell
cargo run -p allpaca-server -- `
  --synthetic --cpu-only `
  --rpc 192.168.69.2:50052,192.168.69.2:50053 `
  --tensor-split 1,1,1 `
  --port 8080
```

Device order: local target, then `rpc[192.168.69.2:50052]`, then `rpc[192.168.69.2:50053]`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Connection refused | Worker down, or probing `10.1.1.69` | Probe `192.168.69.2`; worker `-H 192.168.69.2` |
| Timeout, SSH still works | RPC sent on the 10.1.1.0 NIC | `--rpc 192.168.69.2:50052` only |
| Version / kernel ABI mismatch | Different commits, or a worker still on `KERNEL_ABI_REV` 1 | Rebuild both from the same `HEAD`. Rev 2 is `PutTensor` + `GemmNtQuant` |

## Quantized GGUF

`PutTensor` sends ggml type, dims, and the packed tensor bytes. The worker stores those bytes. `GemmNtQuant` dispatches the existing quant GEMM (CPU dequant, or HIP `gemm_nt_*_device`). Weights are not expanded to F32. Activations at a `RunSpan` edge stay F32.

The HIP-loadable Qwen file is `models/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF` (ggml type 100). `models/Qwen3.8-27B-GGUF` (UD-Q5_K_XL) includes IQ4 blocks, which have no fused HIP GEMV, so a HIP worker will not place those layers.

A remote layer is placed when the block is dense attention + FFN and the worker accepts the ggml type. `--tensor-split`'s first weight is the local GPU pool: it is subdivided across every local HIP device by VRAM, and later weights are RPC endpoints. Qwen3.5 full-attention blocks are gated Q (`[Q|gate]` per head), with Q/K RMSNorm and partial RoPE; those ops run on the worker. The FFN RMSNorm in that GGUF is `post_attention_norm.weight`. GDN blocks stay in the local pool. A gfx906 worker expands type 100 to F32 at `PutTensor`. Each remote full-attention span is a few milliseconds. Decode uses a K-split ROCmFP4 GEMV, and the GDN state update runs one thread per K-lane instead of one thread per head. A dense launch is not a substitute for a GDN block. The local HIP upload skips layers the worker owns, and device decode bounces the activation around each remote span and at local device boundaries. If nothing was placed, decode stays on the local GPUs.
| Worker exits immediately | HIP build without ROCm | `--cpu-only` first; set `LD_LIBRARY_PATH` |
| `rpc_roundtrips` = 0 | No remote layer owners | `--tensor-split 1,1` (or `1,1,1`) |
