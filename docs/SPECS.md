# Technical Specifications

This document consolidates technical requirements, API contracts, and interface specifications extracted from the PRD and design documents.

## Technical Requirements

### TR-01 — Correctness

Target-model sampling must be equivalent to the selected baseline within the chosen stochastic tolerance. Speculative paths must preserve the target distribution according to the selected speculative algorithm.

### TR-02 — Determinism Mode

Provide a mode with fixed RNG seed, fixed scheduling policy and deterministic kernel selection where the backend permits it.

### TR-03 — P2P Detection

The engine must refuse to assume direct GPU P2P until capability and benchmark validation succeeds.

### TR-04 — Memory Accounting

All VRAM, DRAM and NVMe residency must be tracked per request and per page.

### TR-05 — Backpressure

A full KV tier or saturated transfer queue must propagate backpressure to the scheduler instead of causing unbounded allocations.

### TR-06 — Telemetry

Export OpenTelemetry/Prometheus-compatible counters for latency, queue depth, acceptance, cache residency, migration and device utilization.

### TR-07 — Fault Handling

If a side accelerator fails, inference falls back to a CPU/GPU implementation without terminating the request unless policy forbids fallback.

## Core Data Structures

### Device Information

```cpp
struct DeviceInfo {
    int id;
    enum class Kind { CPU, HIP_GPU, EDGE_TPU, NVME } kind;
    size_t memory_bytes;
    double bandwidth_gbps;
    std::vector<int> peers;
};
```

### Task Cost Model

```cpp
struct TaskCost {
    double compute_us;
    double h2d_us;
    double d2h_us;
    double p2p_us;
    size_t bytes_read;
    size_t bytes_written;
};
```

### Placement Decision

```cpp
struct Placement {
    int target_device;
    int draft_device;
    int vision_device;
    enum class KvTier { VRAM, DRAM, NVME } warm_kv;
};
```

### Pipeline Mode (Rust)

```rust
pub enum PipelineMode {
    Asymmetric,      // e.g., RX 9070 + 9060 XT
    Symmetric,       // e.g., Dual MI50s
    MobileTiered,    // e.g., 6GB RTX 500 + Intel NPU
}
```

## API Contracts

### Device Manager

```cpp
struct DeviceManager {
    std::vector<DeviceInfo> enumerate();
    bool can_p2p(int a, int b);
    BandwidthBenchmark benchmark_p2p(int a, int b, size_t bytes);
};
```

### Draft Provider Interface

```cpp
class DraftProvider {
public:
    virtual ~DraftProvider() = default;
    virtual std::vector<int> propose(
        const TokenContext& ctx,
        int max_tokens,
        float temperature) = 0;
};
```

Required implementations:
- `DraftModelProvider`
- `NGramProvider`
- `EagleProvider`
- `MTPProvider`
- `TreeSearchProvider`
- `HybridProvider`

### Auxiliary Accelerator Interface

```cpp
class AuxiliaryDevice {
public:
    virtual ~AuxiliaryDevice() = default;
    virtual std::string name() const = 0;
    virtual bool supports(const ModelDescriptor&) const = 0;
    virtual Future<Result> submit(const TensorView&) = 0;
};
```

### KV Tier Interface

```cpp
class KvTier {
public:
    virtual ~KvTier() = default;
    virtual bool contains(uint64_t page) const = 0;
    virtual void prefetch(uint64_t page) = 0;
    virtual void evict(uint64_t page) = 0;
};

class CompositeKvCache {
public:
    KvTier& hot;   // VRAM
    KvTier& warm;  // Pinned DRAM
    KvTier& cold;  // NVMe
};
```

### Memory Tier Backend (Future-Proof)

```cpp
class MemoryTierBackend {
public:
    virtual ~MemoryTierBackend() = default;
    virtual ReadTicket prefetch(PageId p, void* dst, size_t bytes) = 0;
    virtual bool ready(ReadTicket t) const = 0;
    virtual void release(PageId p) = 0;
};
```

Backend implementations:
- `VramBackend`
- `PinnedDramBackend`
- `NvmeBackend`
- future `CxlMemBackend`
- future `PimBackend`

### Quantization Backend Interface

```cpp
class QuantKernelBackend {
public:
    virtual bool supports(ModelArch arch, QuantType q) const = 0;
    virtual void matvec(const Tensor& x,
                        const Tensor& w,
                        Tensor& y,
                        Stream stream) = 0;
};
```

## KV Cache Specification

### Page Structure

```cpp
enum class KvResidency { VRAM, DRAM, NVME };

struct KvPage {
    PageId id;
    KvResidency tier;
    uint32_t refcount;
    uint64_t last_access;
    float predicted_reuse;
    DeviceBuffer location;
};
```

### KV Handle

```cpp
struct KvHandle {
    uint64_t request_id;
    uint64_t epoch;
    size_t token_count;
    GpuAddress key_base;
    GpuAddress value_base;
    std::vector<PageId> pages;
};
```

### Tier State Machine

```text
VRAM <-> DRAM <-> NVMe
  \                   /
   +---- prefetch ----+
```

### Priority Scoring

```
priority = predicted_reuse * recency_weight * layer_weight - transfer_cost
```

## Token Forest Specification

### Node Structure

```cpp
struct TreeNode {
    int token_id;
    int parent;
    uint16_t depth;
    float logprob;
    uint32_t flags;
};

struct TokenForest {
    std::vector<TreeNode> nodes;
    std::vector<uint32_t> first_child;
    std::vector<uint32_t> next_sibling;
};
```

### Token Forest Requirements

- breadth-limited tree
- depth-limited tree
- probability thresholding
- multiple draft providers
- ancestry-mask generation
- acceptance accounting

## Async Logits Transport

### Slot Structure

```cpp
struct LogitSlot {
    alignas(64) float *ptr;
    size_t vocab;
    std::atomic<uint32_t> state; // FREE, GPU_WRITING, READY, CPU_READING
};
```

### State Transitions

1. `FREE` → `GPU_WRITING`: GPU claims slot
2. `GPU_WRITING` → `READY`: DMA complete (event recorded)
3. `READY` → `CPU_READING`: CPU claims for sampling
4. `CPU_READING` → `FREE`: Sampling complete

## Structured Decoding

### Constraint State

```cpp
struct ConstraintState {
    uint64_t state_id;
    const uint64_t* allowed_bitset;
    size_t words;
};
```

### Token Mask (AVX2-Friendly)

```cpp
struct TokenMask {
    const uint64_t* words;  // 64-token words
    size_t count;
};
```

## Early Exit Specification

```cpp
struct EarlyExitSpec {
    bool supported;
    std::vector<int> exit_layers;
    float min_confidence;
    float min_calibration_margin;
};
```

Only enable when model manifest declares valid exit head/calibration scheme.

## Scheduler Cost Function

```
score = SLA_weight * predicted_latency + transfer_penalty + queue_penalty + memory_penalty - reuse_bonus
```

The scheduler chooses:
- device
- queue
- microbatch size
- speculative depth
- KV residency
- prefetch distance
- CPU affinity

## Hardware Constraints

### CPU Instruction Sets

| Processor | Max Vector Width | Notes |
|-----------|-----------------|-------|
| Ryzen 7 5800X (Zen 3) | 256-bit (AVX2) | No AVX-512 |
| Intel Core Ultra 7 265H (Arrow Lake-H) | 256-bit (AVX-VNNI) | No AVX-512 |
| Enterprise Xeon | 512-bit (AVX-512) | Not assumed |

### GPU Asymmetry

| GPU | Memory | Bus Width | Bandwidth | Infinity Cache |
|-----|--------|-----------|-----------|----------------|
| RX 9070 | 16 GB | 256-bit | 640 GB/s | 64 MB |
| RX 9060 XT | 16 GB | 128-bit | 320 GB/s | 32 MB |

**Recommended placement:** 9070 = target/main decoder, 9060 XT = draft/vision/auxiliary

### Memory Hierarchy

| Tier | Latency | Bandwidth | Capacity | Use Case |
|------|---------|-----------|----------|----------|
| VRAM | ~100ns | 320-640 GB/s | 16 GB | Hot KV, active weights |
| Pinned DRAM | ~100ns | 50-100 GB/s | 64+ GB | Warm KV, prompt cache |
| NVMe | ~10μs | 5-7 GB/s | TB-scale | Cold KV, weight persistence |

### P2P Constraints

- P2P must be validated at runtime: query `hipDeviceCanAccessPeer`, enable both directions with `hipDeviceEnablePeerAccess`, then re-query
- Benchmark enable+peer-copy vs host staging for each transfer size
- Never assume P2P is faster without measurement; prefer P2P only when it is not more than 10% slower than host staging
