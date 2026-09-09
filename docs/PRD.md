# PRD — Heterogeneous Research Inference Engine

**Status:** Proposed  
**Date:** 2026-08-15  
**Primary objective:** Build a research-grade local inference engine that treats heterogeneous compute and memory resources as a single schedulable system while preserving model correctness and providing hard performance evidence for every optimization.

## 1. Problem statement

Existing inference engines are highly optimized within a device class, but heterogeneous desktop hardware is often handled as a collection of independent devices. The reference system combines two materially different Radeon GPUs, substantial host memory and a many-core CPU. The opportunity is to improve latency, capacity and concurrency through explicit scheduling and memory tiering.

The product should **not** try to reproduce every capability of vLLM, SGLang or llama.cpp. It should provide an experimental runtime where unusual device combinations and memory policies can be tested rapidly and measured rigorously.

## 2. Product thesis

> The main differentiator is not a new transformer kernel. It is a **hardware-aware inference scheduler** capable of overlapping compute, data movement, sampling, constraints, speculation and memory residency across heterogeneous devices.

## 3. Goals

### P0 goals

- Detect CPU/GPU/NVMe topology and capabilities.
- Explicitly manage multiple HIP GPUs.
- Implement asymmetric target/draft execution.
- Implement adaptive speculative decoding.
- Implement token-tree construction and verification.
- Separate prefill and decode scheduling.
- Implement pinned host ring buffers for asynchronous CPU work.
- Implement VRAM -> DRAM KV tiering.
- Implement model/expert page residency and prefetch (dense pin-prefix + ring; MoE host-pool + GPU LRU; measured `q*` miss split). Weights are a working set, not KV pages.
- Provide structured-output/grammar scheduling.
- Provide detailed telemetry and benchmark output.

### P1 goals

- NVMe cold KV/weight tier (direct I/O under the host-resident pool; not OS page cache).
- Elastic VRAM split between KV pages and expert/weight slots at scheduler safe points.
- Multimodal device placement.
- Model-aware early exit and self-speculation.
- BitNet/ternary backend (CPU TQ done; HIP ternary GEMV next).
- CUDA backend scaffold → engine parity.
- Automated policy optimization.

### P2 research goals

- CXL memory backend.
- PNM/PIM simulator backend.
- Remote accelerator / photonic interconnect abstraction.
- Learned scheduler.

## 4. Non-goals

- Transparent 32 GB VRAM pooling.
- Arbitrary GPU-to-L3 cache writes.
- Generic mid-graph aborts on unmodified transformer checkpoints.
- Treating NVMe as hot VRAM.
- Coral / Edge TPU / NPU backends (dropped from scope).
- Claiming photonic/PIM performance without target hardware.

## 5. User stories

### US-01 — asymmetric generation
As an operator, I can put the target model on the faster RX 9070 and the draft model on the RX 9060 XT without manual process orchestration.

### US-02 — adaptive speculation
As an operator, the engine automatically adjusts draft depth based on acceptance rate, target queue pressure and latency SLA.

### US-03 — long-context serving
As an operator, I can configure a tiered KV cache so that older pages move to host DRAM and cold pages can be persisted on NVMe.

### US-07 — models larger than VRAM
As an operator, I can run a dense or MoE checkpoint whose weights exceed GPU memory. Always-hot tensors stay on the target GPU; streamable FFN/experts move through a pinned DRAM ring (and later NVMe). Output tokens match the fully-resident path. More RAM/VRAM only changes speed.

### US-04 — tree verification
As a researcher, I can enable a token forest and compare linear vs tree speculative verification using the same target model.

### US-05 — structured generation
As an application developer, I can use JSON/grammar constraints without blocking the target GPU whenever the CPU can compute the next constraint state in parallel.

### US-06 — modality separation
As an application developer, I can keep vision/audio encoders resident on the 9060 XT while the 9070 remains dedicated to the language model.

## 6. High-level architecture

```text
                    +-----------------------+
                    |   Request Gateway     |
                    +-----------+-----------+
                                |
                       +--------v---------+
                       | Policy Scheduler |
                       +--+-----+-----+---+
                          |     |     |
                 +--------+  +--+--+  +--------+
                 |           |     |           |
          +------v------+ +--v--+ +v---------+ |
          | Prefill GPU | |Draft| |CPU Logic | |
          |   RX 9070   | |9060 | | sampler  | |
          +------+------+
                 |
             KV Handle
                 |
          +------v-------------------------------+
          |       Target Decode / Verify         |
          | RX 9070 + Tree Attention / MTP/EAGLE |
          +------+-----------------------+--------+
                 |                       |
          +------v------+         +------v-------+
          | DRAM KV Tier|         | NVMe Cold Tier|
          +-------------+         +--------------+

          +--------------------------------------+
          | Weight working set (not KV)          |
          | VRAM pin-prefix / expert LRU slots   |
          | DRAM ring + host expert pool         |
          | NVMe packed trunk / FTW banks        |
          +--------------------------------------+
```

## 7. Core modules

### 7.1 Device manager

Responsibilities:

- enumerate HIP devices (and CUDA inventory when `--features cuda`),
- detect P2P capability,
- measure PCIe/P2P transfer paths,
- track free memory,
- expose device health and load.

API:

```cpp
struct DeviceManager {
    std::vector<DeviceInfo> enumerate();
    bool can_p2p(int a, int b);
    BandwidthBenchmark benchmark_p2p(int a, int b, size_t bytes);
};
```

### 7.2 Execution graph manager

Represent model execution as a graph of immutable compute stages and movable data dependencies.

Stages:

- tokenize,
- prefill layer group,
- draft,
- target verify,
- sampling,
- grammar transition,
- KV migration,
- multimodal encoder.

Every stage exposes estimated compute and transfer costs.

### 7.3 Scheduler

The scheduler chooses:

- device,
- queue,
- microbatch size,
- speculative depth,
- KV residency,
- prefetch distance,
- CPU affinity.

Use a cost function:

`score = SLA_weight * predicted_latency + transfer_penalty + queue_penalty + memory_penalty - reuse_bonus`

The first implementation should be heuristic. Later iterations may learn the cost model from telemetry.

### 7.4 Token Forest

Support:

- breadth-limited tree,
- depth-limited tree,
- probability thresholding,
- multiple draft providers,
- ancestry-mask generation,
- acceptance accounting.

### 7.5 KV tier manager

Implement page granularity. A page should be independently migratable, reference-counted and observable.

Required state machine:

```text
VRAM <-> DRAM <-> NVMe
  \                   /
   +---- prefetch ----+
```

No synchronous cold-page fault is allowed in the critical decode path except under an explicit emergency policy.

### 7.6 Sampling pipeline

Stages:

1. logits production,
2. optional async D2H,
3. temperature/top-k/top-p,
4. grammar filtering,
5. RNG sampling,
6. token publication.

Maintain an optional device-side path for comparison.

### 7.7 Auxiliary accelerator API

Dropped from scope (no Coral/NPU). Historical note only: a sidecar `AuxiliaryDevice` trait was considered for fixed-function classifiers; it is not implemented.

## 8. Technical requirements

### TR-01 — Correctness

Target-model sampling must be equivalent to the selected baseline within the chosen stochastic tolerance. Speculative paths must preserve the target distribution according to the selected speculative algorithm.

### TR-02 — Determinism mode

Provide a mode with fixed RNG seed, fixed scheduling policy and deterministic kernel selection where the backend permits it.

### TR-03 — P2P detection

The engine must refuse to assume direct GPU P2P until capability and benchmark validation succeeds.

### TR-04 — Memory accounting

All VRAM, DRAM and NVMe residency must be tracked per request, per KV page, and per weight/expert slot.

### TR-05 — Backpressure

A full KV tier or saturated transfer queue must propagate backpressure to the scheduler instead of causing unbounded allocations.

### TR-06 — Telemetry

Export OpenTelemetry/Prometheus-compatible counters for latency, queue depth, acceptance, cache residency, migration and device utilization.

### TR-07 — Fault handling

If a side accelerator fails, inference falls back to a CPU/GPU implementation without terminating the request unless policy forbids fallback.

## 9. Implementation phases

### Phase 0 — instrumentation and topology

Deliverables:

- device inventory,
- P2P test,
- host-memory bandwidth test,
- latency histogram framework,
- baseline single-GPU engine.

Exit gate: reproducible baseline metrics.

### Phase 1 — asymmetric speculation

Deliverables:

- draft-provider interface,
- RX 9060 XT draft placement,
- target verify on RX 9070,
- adaptive depth,
- acceptance telemetry.

Exit gate: >=10% improvement in at least one target workload without >5% p95 regression.

### Phase 2 — token forest

Deliverables:

- tree builder,
- ancestry mask representation,
- batched target verification,
- adaptive branch budget.

Exit gate: improved accepted tokens/target invocation and end-to-end speedup on high-predictability prompts.

### Phase 3 — pipelined prefill/decode

Deliverables:

- independent prefill/decode queues,
- chunked prefill,
- overlapping request admission,
- KV handles.

Exit gate: improved p95 ITL under mixed prefill/decode load.

### Phase 4 — memory tiers

Deliverables:

- VRAM page allocator,
- DRAM page cache,
- prefetch thread,
- eviction policy,
- optional NVMe KV tier,
- packed layer-major / expert-bank layout,
- dense pin-prefix + 1–2 ring slots (cyclic layers; not LRU),
- host-truth expert pool + GPU LRU slots + `q*` miss split (profiled `B_P`, `B_H`),
- full-layer prefill double-buffering for MoE,
- byte-identical tokens vs full-resident at every memory budget.

Exit gate: context capacity increase and no catastrophic tail latency for configured hot-set workloads; a model that does not fit VRAM generates the same greedy tokens as the resident path.

### Phase 5 — multimodal

Deliverables:

- encoder placement API (`EncoderPlacement` / `VisionEncoder` / `NoopVisionEncoder`, `multimodal.vision_device` → `Placement.vision_device`) — **landed**;
- encode + soft-prompt inject (`ImageInput` / `VisionTokens`, `begin_request_multimodal` / `generate_multimodal` / `generate_stream_multimodal`, prefill `soft_x`) — **landed**;
- OpenAI `image_url` content parts for **local file paths** and **`data:` base64** — **landed**;
- http(s) image fetch + real vision/audio residency on 9060 XT — **deferred**.

Exit gate: improved concurrency/VRAM stability under multimodal workload.

### Phase 6 — model-aware early exit

Deliverables:

- model manifest extension,
- exit-head interface,
- self-speculative verification mode,
- calibration data.

Exit gate: correctness preserved and measurable speedup on compatible checkpoints.

### Phase 7 — low-bit and future hardware

Deliverables:

- BitNet backend,
- CXL memory abstraction (`MemoryTierBackend` trait + `PinnedDramBackend` / `CxlMemBackend` stub) — **trait In; hardware Out**,
- PIM simulator backend (`PimBackend` stub) — **trait In; hardware Out**,
- remote accelerator protocol.

Exit gate: backend interfaces compile and benchmark harness can simulate transfer/compute characteristics.

## 10. Configuration example

```yaml
engine:
  scheduler: adaptive
  target_device: hip:0
  draft_device: hip:1
  cpu_threads: 6

speculative:
  enabled: true
  providers: [draft_model, ngram]
  max_depth: 12
  min_depth: 2
  adaptive: true
  tree:
    enabled: true
    max_nodes: 32

prefill:
  chunk_tokens: 512
  overlap_decode: true

kv:
  page_tokens: 128
  hot_fraction: 0.75
  warm: dram
  cold: nvme
  prefetch_distance: 4

weights:
  always_hot: [embed, attn, norm, lm_head, shared_expert]
  stream: [ffn, routed_expert]
  trunk_gb: 8          # pinned prefix + ring (dense / shared)
  cache_gb: 8          # GPU expert LRU + host pool budget
  ring_slots: 2        # granted only if they fit trunk_gb
  miss_policy: qstar   # qstar | fill | cpu
  packed_layout: auto  # auto | gguf | trunk | ftw
  pin_after_fill: true

sampling:
  mode: async_host
  pinned_ring_slots: 8

structured_output:
  enabled: true
  engine: llguidance
  async: true

multimodal:
  vision_device: hip:1
```

## 11. Observability

Required dashboards:

1. End-to-end latency.
2. GPU utilization and memory by device.
3. Speculative acceptance waterfall.
4. KV tier occupancy and migration.
5. Weight working-set occupancy (pinned layers, ring hits, expert LRU hits/misses, `q*` fill vs CPU).
6. P2P/host transfer bandwidth (`B_P`) and CPU expert bandwidth (`B_H`).
7. Scheduler queue depth.
8. CPU core affinity and utilization.
9. Structured-output overhead.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| P2P topology is slower than host staging | Runtime benchmark + policy fallback |
| Draft model contention | Adaptive disabling / queue-aware policy |
| Tree verification overhead dominates | Branch budget and acceptance-aware pruning |
| NVMe paging destroys latency | NVMe limited to cold tier; prefetch only; dense trunk uses pin-prefix not LRU |
| LRU on cyclic dense layers | Pin prefix + ring; LRU only for sparse experts |
| Weight cache in KvTierManager | Separate working-set manager; host pool is source of truth |
| CUDA-graph host callbacks on HIP | Keep `q*` in Rust; streams/events; device-side victims later if sync shows up |
| Early exit degrades quality | Compatible-checkpoint gating + verification |
| ROCm kernel instability | Kernel capability registry and fallback implementation |
| Memory fragmentation | Page allocators and compaction |
| Driver reset / device loss | Request migration to fallback backend |

## 13. Success criteria

The MVP is successful when it can demonstrate, on the reference platform:

- one target model using RX 9070 and an independent draft/model workload on RX 9060 XT;
- measurable speculative acceptance telemetry;
- measurable end-to-end benefit from at least one adaptive speculative policy;
- context capacity beyond VRAM-only limits using DRAM KV tiering;
- a weights-over-VRAM path whose greedy tokens match the fully-resident path;
- no host-staging surprise where P2P is expected;
- structured generation overlapping with GPU compute;
- reproducible benchmarks with automated reports;
- all experimental features switchable independently.

## 14. Research principles

1. **Never call a capacity tier a performance tier.** DRAM/NVMe extend capacity but do not become VRAM.
2. **GPU weight residency is a speed dial, not a correctness gate.** Host (or NVMe) is source of truth when the model does not fit.
3. **Match cache policy to access pattern.** Cyclic dense layers: pin prefix + ring. Sparse experts: LRU + measured miss split. Never LRU a cyclic scan.
4. **Measure topology before optimizing transport.** PCIe tree placement can dominate. Profile `B_P` and `B_H` on the machine, as with P2P vs host staging.
5. **Optimize end-to-end latency, not isolated kernel throughput.**
6. **Treat speculative decoding as a control problem.** Acceptance, cost and queue pressure must jointly determine policy.
7. **Keep hardware research behind stable interfaces.** PIM/CXL/photonic ideas should not leak into core transformer code.
8. **Prefer reversible experiments.** Every optimization should have an A/B switch and telemetry.
