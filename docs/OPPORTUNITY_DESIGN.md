# Opportunity Design and Implementation Notes

## 1. Heterogeneous scheduler

### Goal
Treat CPU, GPU0, GPU1, Coral, host DRAM and NVMe as explicit resources with measurable transfer costs.

### Core data structures

```cpp
struct DeviceInfo {
    int id;
    enum class Kind { CPU, HIP_GPU, EDGE_TPU, NVME } kind;
    size_t memory_bytes;
    double bandwidth_gbps;
    std::vector<int> peers;
};

struct TaskCost {
    double compute_us;
    double h2d_us;
    double d2h_us;
    double p2p_us;
    size_t bytes_read;
    size_t bytes_written;
};

struct Placement {
    int target_device;
    int draft_device;
    int vision_device;
    enum class KvTier { VRAM, DRAM, NVME } warm_kv;
};
```

Use online measurements rather than static theoretical bandwidth. The scheduler should periodically benchmark or ingest telemetry for:

- token latency,
- queue depth,
- GPU busy time,
- P2P bandwidth,
- host-copy latency,
- KV hit ratio,
- draft acceptance,
- page-fault/prefetch misses.

## 2. Asymmetric target/draft inference

### Recommended placement

- RX 9070: target model.
- RX 9060 XT: draft model, vision encoder, audio encoder, or secondary request.
- CPU: scheduler, tokenizer, trie/n-gram lookup, structured-output state machine.
- Coral: independent fixed-function classifier only when useful.

The scheduler must support a fallback where the draft runs on CPU when the 9060 XT is needed by a vision/audio pipeline.

### Draft interface

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

Implement providers:

- `DraftModelProvider`
- `NGramProvider`
- `EagleProvider`
- `MTPProvider`
- `TreeSearchProvider`
- `HybridProvider`

## 3. Token Forest / tree speculation

The engine should represent candidate branches independently from the target model graph.

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

### Build phase

```cpp
TokenForest build_tree(const TokenContext& ctx, int beam, int depth) {
    TokenForest tree;
    // CPU threads explore separate child frontiers.
    // Use a lock-free or sharded priority queue for frontier expansion.
    return tree;
}
```

### Verification phase

For every target-model graph invocation, flatten all tree nodes into a packed batch and construct an ancestry attention mask:

```python
def tree_attention_mask(parent):
    n = len(parent)
    mask = [[False] * n for _ in range(n)]
    for i in range(n):
        j = i
        while j >= 0:
            mask[i][j] = True
            j = parent[j]
    return mask
```

In production, replace the Python mask with a compact CSR/bitset representation and construct GPU attention metadata directly.

## 4. Disaggregated prefill/decode on one host

Instead of a network boundary, use two execution lanes:

```text
Request -> Prefill Queue -> GPU compute lane
                        -> KV handle -> Decode Queue -> Target GPU
                                                  ^
                              Draft / speculative lane
```

A KV handle contains:

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

Use chunked prefill to avoid one large prompt monopolizing the decoder.

## 5. Async host logits transport

### Correct replacement for "direct to L3"

Allocate a pinned host ring buffer and issue asynchronous device-to-host copies. The CPU sampler consumes entries only after a lightweight event/fence.

```cpp
struct LogitSlot {
    alignas(64) float *ptr;
    size_t vocab;
    std::atomic<uint32_t> state; // FREE, GPU_WRITING, READY, CPU_READING
};

// HIP pseudocode
hipHostMalloc(&slot.ptr, vocab * sizeof(float), hipHostMallocDefault);
hipMemcpyAsync(
    slot.ptr,
    gpu_logits,
    vocab * sizeof(float),
    hipMemcpyDeviceToHost,
    copy_stream);
hipEventRecord(copy_done, copy_stream);
```

The CPU thread waits on the event, then performs SIMD masking/sampling. Avoid assuming the data is resident in L3; measure cache behavior.

## 6. CPU-side structured decoding

Keep the grammar engine CPU-resident and produce a compact allowed-token representation.

```cpp
struct ConstraintState {
    uint64_t state_id;
    const uint64_t* allowed_bitset;
    size_t words;
};

void apply_constraints(float* logits,
                       const ConstraintState& s,
                       size_t vocab) {
    const size_t words = (vocab + 63) / 64;
    for (size_t w = 0; w < words; ++w) {
        uint64_t valid = s.allowed_bitset[w];
        uint64_t invalid = ~valid;
        while (invalid) {
            unsigned bit = __builtin_ctzll(invalid);
            size_t id = w * 64 + bit;
            if (id < vocab) logits[id] = -INFINITY;
            invalid &= invalid - 1;
        }
    }
}
```

For large vocabularies, prefer a candidate-index list plus top-k pruning to avoid scanning every token on every step.

## 7. Hierarchical KV cache

Use page-level residency rather than token-level malloc/free.

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

### Policy

Score a page by:

`priority = predicted_reuse * recency_weight * layer_weight - transfer_cost`

- VRAM: hot pages.
- DRAM: warm pages and prompt caches.
- NVMe: cold, rarely reused pages.

Do not synchronously page fault during a decode token if it can be avoided. Prefetch one or more pages ahead based on the attention window and reuse model.

## 8. MoE expert residency

For MoE models, model each expert as a cacheable page group.

```cpp
struct ExpertStats {
    uint64_t hits;
    uint64_t misses;
    float ema_prob;
};

std::vector<int> choose_prefetch_experts(
    const RouterPrediction& rp,
    const Cache& cache,
    size_t budget_bytes) {
    // Rank by predicted probability / bytes and prefetch top experts.
    return {};
}
```

The win comes from predicting likely experts early enough to overlap DMA with current-layer compute.

## 9. Model-specific early exit

Only enable on a compatible checkpoint.

```cpp
struct EarlyExitSpec {
    bool supported;
    std::vector<int> exit_layers;
    float min_confidence;
    float min_calibration_margin;
};

bool accept_exit(const ExitPrediction& p, const EarlyExitSpec& s) {
    return s.supported &&
           p.confidence >= s.min_confidence &&
           p.margin >= s.min_calibration_margin;
}
```

Prefer a self-speculative verification path over a destructive mid-graph abort when correctness requirements are strict.

## 10. Coral Edge TPU plugin

Use the Coral only for a compiled TensorFlow Lite side model.

```python
from tflite_runtime.interpreter import Interpreter, load_delegate

interpreter = Interpreter(
    model_path="sidecar_int8.tflite",
    experimental_delegates=[load_delegate("libedgetpu.so.1")],
)
interpreter.allocate_tensors()

# Sidecar input is deliberately tiny: e.g. classifier state/features.
```

Suggested workloads:

- wake-word detection,
- document/intent classifier,
- fixed schema/state classifier,
- vision pre-filter.

Do not route the main LLM through Edge TPU unless a deliberately compiled compatible model exists.

## 11. BitNet backend

Implement a backend boundary instead of changing the generic GEMM path:

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

The BitNet backend can use packed ternary weights and lookup/addition-based kernels. Do not convert ordinary Q4/Q8 weights into ternary at inference time.

## 12. Cache-aware AMD kernels

The RX 9070 has 64 MB Infinity Cache and the RX 9060 XT has 32 MB. The runtime cannot command cache residency as if it were a software-managed SRAM. Instead:

- choose tile shapes from profiler data,
- fuse dequantization + matrix-vector operations,
- minimize redundant reads,
- align strides,
- measure cache hit/miss counters where the profiling stack exposes them.

Do not hard-code "all intermediate matrices must fit in Infinity Cache."

## 13. CXL/PIM future interface

Define future memory backends now:

```cpp
class MemoryTierBackend {
public:
    virtual ~MemoryTierBackend() = default;
    virtual ReadTicket prefetch(PageId p, void* dst, size_t bytes) = 0;
    virtual bool ready(ReadTicket t) const = 0;
    virtual void release(PageId p) = 0;
};
```

Backends:

- `VramBackend`
- `PinnedDramBackend`
- `NvmeBackend`
- future `CxlMemBackend`
- future `PimBackend`

This lets the scheduler evolve without embedding hardware assumptions into model code.
