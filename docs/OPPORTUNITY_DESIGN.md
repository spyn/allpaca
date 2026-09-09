# Opportunity Design and Implementation Notes

## 1. Heterogeneous scheduler

### Goal
Treat CPU, GPU0, GPU1, host DRAM and NVMe as explicit resources with measurable transfer costs.

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

## 8. Weight working set (dense + MoE)

Weights and KV are different reuse patterns. Do **not** put either cache in `KvTierManager`.

- KV pages: decode-hot, random-ish reuse, already VRAM ↔ pinned DRAM (NVMe cold is post-MVP).
- Dense layers: walked 0…N-1 every token. That is a cyclic scan.
- Routed experts: sparse, overlapping across consecutive tokens. That is an LRU working set.

Host (or NVMe) is the source of truth whenever the full model does not fit. GPU residency is a speed dial. Same tokens at every budget; only the clock changes.

Two 2026 systems already solve halves of this. Steal the policies, not their runtimes (kimi-k3-in-c is CPU-only C99; FreeToken is CUDA/Python on NVIDIA). Allpaca stays thick-Rust + HIP.

### 8.1 Dense trunk — pin prefix + ring (from kimi-k3-in-c)

A cyclic layer walk is the pathological case for LRU: by the time layer 0 comes around again it is the least-recently-used slot, so hit rate is exactly zero. Pin the first *k* layers that fit the VRAM/DRAM budget. Stream the rest through 1–2 ring slots. Prefetch layer *L+1* into the spare slot while *L* GEMMs.

```cpp
// Slot size = largest *unpinned* layer, not the largest layer in the file.
// Two ring slots only if they fit the printed budget (second slot is a request).
struct DenseTrunkPlan {
    int pinned_prefix;     // layers [0, pinned_prefix)
    size_t ring_slots;     // 1 or 2
    size_t slot_bytes;     // align_up(max remaining layer, DIRECT_ALIGN)
};

// GPU shape: NVMe -> pinned DRAM ring -> H2D into a device slot.
// Overlap k3_trunk_prefetch(L+1) with compute(L).
```

Repack GGUF/safetensors into a layer-major file (one aligned sequential read per layer) before streaming. Refuse to pack a layer whose tensors are not a contiguous run — a gap would drag unrelated bytes. This is the custom tiered-memory-optimized format already named in CONTEXT.

Always-hot vs streamable (two budgets, like `--trunk-gb` / `--cache-gb`):

- always-hot on VRAM: embed, norms, attention, lm_head, shared experts
- streamable: dense FFN (and routed experts, §8.2)

Do not LRU this path. Do not treat GGUF mmap as the cache; a cyclic scan thrashes the OS page cache. Prefer aligned direct I/O into hugepage/pinned arenas. Slot states need `EMPTY | INFLIGHT | key` so a prefetch cannot alias a live buffer. Sum trunk + model + cache + KV + scratch against `MemAvailable` and free VRAM *before* allocating; quote measured peak RSS, not the plan.

### 8.2 Routed experts — host pool + GPU LRU + q* (from FreeToken)

FreeToken ([paper](https://arxiv.org/abs/2608.16157), [code](https://github.com/FlashML-org/FreeToken)) is the closest analogue: edge-native MoE serving that treats GPU, CPU, host RAM and PCIe as one elastic platform. Non-expert weights stay GPU-resident. The complete expert pool stays in host memory. Remaining VRAM is a single LRU of **complete** `(layer, expert)` slots shared across all MoE layers.

Prefill activates almost the whole expert set. Do not fetch those experts on demand. Double-buffer at **full-layer** granularity: while the GPU computes layer *l*, a transfer stream loads *all* experts of *l+1*. When two full layers do not fit, fall back to on-demand rather than oversubscribe VRAM. Surviving prefill slots seed decode.

Decode routing has short-range locality. A global LRU following the router beats static load-time placement (llama.cpp) and prefill-frozen hot sets (KTransformers) at equal capacity. Caching cannot eliminate misses. Serve residual misses with a measured-bandwidth split, not "always DMA" or "always CPU":

```
B_P = measured pinned expert H2D bandwidth
B_H = measured CPU expert-kernel bandwidth (same host DRAM)
q*  ≈ m * B_P / B_H     // of m unique misses, fill q*, execute the rest in place on CPU
```

Always fill at least one miss so the cache keeps warming. CPU and GPU partial sums merge exactly — no router change, no expert substitution, no precision drop. Profile `B_P` and `B_H` on the deployed machine the same way we already benchmark P2P vs host staging. An RX 9070 desktop and a dual-channel laptop will not share a ratio.

On HIP, keep the split in the Rust control plane with streams/events. Do not require CUDA-graph host-function capture (NVIDIA-specific). Device-side victim selection in one pass is worth a later kernel if host sync shows up in the decode profile.

Elastic VRAM: at a scheduler safe point, rebuild the expert-slot pool vs KV pages under a new budget without restarting or reloading the host pool. GPU memory affects speed, never correctness.

Bootstrap: read experts into their **final host layout**, then pin. Pinning empty buffers first faults in and zeroes gigabytes you are about to overwrite. First request may run with a cold cache.

FTW-style banks (flattened `l*E+e` as the leading dimension, one identity for GPU and CPU kernels) are the MoE half of the packed format.

If `hipHostRegister` / DMA pin fails on a platform, fall back to CPU MoE: only activations, routing metadata and aggregated outputs cross the PCI boundary.

Router-prediction prefetch (`choose_prefetch_experts`) is optional on top of LRU+q*. It is not the miss policy. Trace-replay cache sizing (kimi's `sim_cache.py`) is how `--cache-gb` gets picked, not guessed.

### 8.3 Semantic anchors (FreeToken, when hybrid attention exists)

Agent harnesses edit context at special-token boundaries (thinking blocks, tool calls, turns). Checkpoint recurrent state (KDA / DeltaNet / SWA) at those anchors, LRU-evict independently of the KV pool, and re-prefill only the new suffix. Full-attention KV can keep the existing radix/prefix reuse. Not needed for the current dense GGUF path; needed the moment a Kimi-K3 / Qwen3.6-style hybrid checkpoint is in scope.

### 8.4 What not to copy

- kimi-k3-in-c CPU-only MXFP4 kernels, or 26 s/token as a product posture. Allpaca's HIP contract is the opposite until streaming is on: `decode_does_not_h2d_weights`.
- FreeToken's Python/CUDA stack, CUDA graphs, or NVIDIA-only kernels. Allpaca is MIT Rust/HIP; steal policy, rewrite on our ABI.
- LRU on the dense trunk.
- Stuffing weight blobs into `KvTierManager`.
- Trusting mmap as the working-set cache.
- Static expert placement chosen at load time.

### 8.5 Allpaca mapping

Today: `ResidentModel::upload_gguf` copies the whole packed blob into `DeviceWeights`. mmap is how the file is opened, not a cache. If that upload fails, GEMM degrades to per-op H2D — that is not a working set.

Ship in this order:

1. Packed layer-major (or FTW-bank) layout + pin-prefix + 1–2 host/device ring slots for dense FFN that does not fit VRAM. Byte-identical vs full-resident at every budget.
2. Host-truth expert pool + GPU LRU slots + `q*` miss split, with full-layer prefill double-buffering.
3. NVMe as the layer/expert backing store under the host pool (P1). Direct I/O, not page-cache.
4. Elastic VRAM split (expert slots vs KV pages) at scheduler safe points.
5. Semantic recurrent-state anchors when a hybrid-attention MoE is loaded.

Keep asymmetric draft/target on two HIP GPUs. FreeToken is single-GPU + CPU; that is complementary, not a replacement. Draft weights can stay fully resident on the slow GPU while the target streams.

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

## 10. BitNet backend

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

The BitNet backend can use packed ternary weights and lookup/addition-based kernels. Do not convert ordinary Q4/Q8 weights into ternary at inference time. HIP ternary GEMV is the GPU follow-on after CPU TQ1_0/TQ2_0 load.

## 11. Cache-aware AMD kernels

The RX 9070 has 64 MB Infinity Cache and the RX 9060 XT has 32 MB. The runtime cannot command cache residency as if it were a software-managed SRAM. Instead:

- choose tile shapes from profiler data,
- fuse dequantization + matrix-vector operations,
- minimize redundant reads,
- align strides,
- measure cache hit/miss counters where the profiling stack exposes them.

Do not hard-code "all intermediate matrices must fit in Infinity Cache."

## 12. CXL/PIM future interface

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
