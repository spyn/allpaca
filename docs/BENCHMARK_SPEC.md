# Benchmark and Validation Specification

## Objective

Prove which heterogeneous techniques improve **real end-to-end latency, throughput, memory capacity and energy efficiency** on the reference desktop.

## Primary metrics

- Time to first token (TTFT): p50/p95/p99.
- Inter-token latency (ITL): p50/p95/p99.
- Decode tokens/s.
- Prompt tokens/s.
- End-to-end tokens/s.
- Draft acceptance rate.
- Accepted tokens per target-model invocation.
- GPU utilization per device.
- GPU memory residency by tier.
- P2P bandwidth and latency.
- Host DRAM bandwidth consumed.
- NVMe read bandwidth and queue depth.
- KV-cache hit/miss/prefetch accuracy.
- CPU core utilization and scheduler migrations.
- Energy per generated token when power telemetry is available.
- Structured-generation rejection/constraint overhead.

## Required baselines

### B0 — Single GPU
Target model on RX 9070 only; no speculative decoding.

### B1 — Dual GPU naive
Measure whatever existing engine does by default when given both GPUs. This exposes the cost of naive splitting.

### B2 — Asymmetric speculative
RX 9070 target + RX 9060 XT draft.

### B3 — Tree speculation
Same as B2 with a configurable token forest.

### B4 — Async CPU sampling
Compare device-side sampling against pinned host sampling.

### B5 — Host KV tier
VRAM + DRAM paging.

### B6 — Host + NVMe cold tier
Add NVMe and prefetch/eviction.

### B7 — Multimodal split
9060 XT handles vision/audio encoder while 9070 handles the language model.

### B8 — Structured output
Measure grammar/JSON overhead with and without pipelined CPU masks.

### B9 — Model-specific early exit
Only compatible LayerSkip/self-speculative checkpoints.

## Workload matrix

| Workload | Prompt | Output | Concurrency |
|---|---:|---:|---:|
| Short chat | 256 | 128 | 1 |
| Long chat | 8k | 512 | 1 |
| Long context | 32k | 256 | 1 |
| Long context | 64k | 256 | 1 |
| Code generation | 4k | 1k | 1 |
| JSON/structured | 2k | 512 | 1 |
| Agent workload | 8k | 1k | 4 |
| Mixed workload | 1k–16k | 128–2k | 8 |

## Experimental controls

Pin model, quantization, tokenizer, prompt set, decoding parameters and thermal state. Run each measurement enough times to provide stable p50/p95 results. Report warm-cache and cold-cache results separately.

## Acceptance gates

A technique is promoted from experimental to default only if all are satisfied:

1. No measurable correctness regression on deterministic tests.
2. >=10% end-to-end improvement in the target workload, or a demonstrated capacity increase that enables otherwise impossible workloads.
3. p95 latency does not regress by >5% for the selected workload class unless the feature is explicitly throughput-oriented.
4. Failure/disable path is automatic and observable.
5. Metrics identify when the feature is hurting performance.

## Specific hypotheses

### H1 — Asymmetric drafting
RX 9060 XT draft + RX 9070 target will beat CPU drafting for sufficiently small draft models.

### H2 — Tree verification
For workloads with high draft predictability, tree verification will improve accepted tokens per target invocation.

### H3 — Host KV tier
A DRAM tier will materially extend usable context while preserving significantly better latency than NVMe.

### H4 — NVMe tier
NVMe will be valuable primarily as cold capacity / cache persistence, not as a per-token hot path.

### H5 — CPU structured masks
CPU-side constraint work can be hidden under target-model compute for large enough GPU batches or asynchronous decoding windows.

### H6 — 9060 XT modality split
Keeping vision/audio on the 9060 XT will improve target-model stability under concurrent multimodal requests.

## Telemetry schema

```json
{
  "request_id": "...",
  "step": 1234,
  "ttft_us": 0,
  "itl_us": 0,
  "draft_proposed": 8,
  "draft_accepted": 6,
  "target_batch": 7,
  "gpu": {
    "9070": {"util": 0.0, "vram": 0},
    "9060xt": {"util": 0.0, "vram": 0}
  },
  "kv": {
    "vram_pages": 0,
    "dram_pages": 0,
    "nvme_pages": 0,
    "prefetch_hit": 0.0
  }
}
```

## Reporting

Every benchmark run must produce:

- raw JSONL metrics,
- CSV summary,
- HTML dashboard,
- configuration manifest,
- git commit hash,
- model checksum,
- ROCm/driver version,
- operating-system build,
- topology report.

## Important comparison rule

Never report a speculative speedup only as "tokens/s". Also report:

`accepted_tokens_per_target_eval = accepted / target_forward_calls`

and end-to-end TTFT/ITL. A larger draft tree can increase target batch work without improving wall-clock latency.
