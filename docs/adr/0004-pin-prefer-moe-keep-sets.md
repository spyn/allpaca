# ADR 0004: Pin-prefer MoE keep-sets

## Status

Accepted

## Context

Flash Spark's tune screen picks a **keep-set** of routed experts so a machine that cannot hold every expert still chooses *what it is good at*. That recipe **prunes**: experts outside the set are not routable.

Allpaca's weight working set ([`docs/OPPORTUNITY_DESIGN.md`](../OPPORTUNITY_DESIGN.md) §8.2) treats GPU expert slots as a speed dial. The host pool (or NVMe) remains source of truth; `q*` splits residual misses between PCIe fill and CPU. §8.4 warns against freezing the whole expert map at load time.

Operators still need a way to spend a limited GPU cache on the experts that matter for a job (late layers, a router group, or a measured topic catalog).

## Decision

**Pin-prefer keep-sets**, not prune.

- `allpaca-server --tune` writes `[tunes."<model spec>"]` (profile, groups, context, `cache_bytes`, compact `keep_experts`).
- On load, those ids occupy reserved GPU LRU slots and are never chosen as victims.
- Every expert stays routable. Misses still follow `miss_policy` (`qstar` / fill / CPU).
- Without a routing-trace catalog, groups are architecture-native (router groups, layer bands) plus budget profiles. Language topics (`python`, `html`) appear only when a sidecar catalog exists.

This is a reserved speed-dial, not a second placement policy and not a reversal of §8.4.

## Consequences

**Benefits:**
- The TUI and the engine share the same budget arithmetic.
- Existing `q*` / host-pool paths keep working for cold experts.
- A later prune mode can be added without changing the saved config shape.

**Costs:**
- Pin-prefer improves hit rate for the selected job; it does not stop the router from activating cold experts (quality is the checkpoint's, speed is the cache's).
- Keep-set size must fit `cache_bytes` or load sizes the LRU up to the pin floor.
