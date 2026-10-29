use allpaca_types::{AllpacaError, FatalError};

/// Per-layer owner index into the split device list (local target first, then RPC).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPlan {
    pub owners: Vec<usize>,
}

impl LayerPlan {
    pub fn all_local(n_layers: usize) -> Self {
        Self {
            owners: vec![0; n_layers],
        }
    }

    pub fn owner(&self, layer: usize) -> usize {
        self.owners.get(layer).copied().unwrap_or(0)
    }

    pub fn has_remote(&self) -> bool {
        self.owners.iter().any(|&o| o != 0)
    }

    /// Contiguous runs of `(owner, start, end)` (end exclusive).
    pub fn runs(&self) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        if self.owners.is_empty() {
            return out;
        }
        let mut owner = self.owners[0];
        let mut start = 0usize;
        for (i, &o) in self.owners.iter().enumerate().skip(1) {
            if o != owner {
                out.push((owner, start, i));
                owner = o;
                start = i;
            }
        }
        out.push((owner, start, self.owners.len()));
        out
    }
}

fn portion(weight: f64, mems: &[u64]) -> Vec<f64> {
    let mems = if mems.is_empty() { vec![1] } else { mems.to_vec() };
    let sum = mems.iter().sum::<u64>().max(1) as f64;
    mems.iter()
        .map(|mem| weight * (*mem as f64) / sum)
        .collect()
}

/// Expand `--tensor-split` across local GPUs, then each worker's GPUs.
///
/// `split[0]` is the local pool (every local HIP device, by VRAM). Each later
/// entry is one RPC endpoint. When `split` already has one weight per device
/// (every local GPU + every remote GPU), it is returned as-is.
pub fn expand_rpc_split(
    split: &[f64],
    local_mems: &[u64],
    remote_groups: &[Vec<u64>],
) -> Result<Vec<f64>, AllpacaError> {
    let n_local = local_mems.len().max(1);
    let n_remote: usize = remote_groups.iter().map(|g| g.len().max(1)).sum();
    let n_dev = n_local + n_remote;
    if split.len() == n_dev {
        return Ok(split.to_vec());
    }
    if split.len() != 1 + remote_groups.len() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "tensor_split has {} entries but {} split devices (1 local pool + {} RPC endpoints)",
            split.len(),
            n_dev,
            remote_groups.len()
        ))));
    }
    let mut out = Vec::with_capacity(n_dev);
    out.extend(portion(split[0], local_mems));
    for (group, weight) in remote_groups.iter().zip(split.iter().skip(1)) {
        out.extend(portion(*weight, group));
    }
    Ok(out)
}

/// Assign contiguous layer ranges by relative weights (VRAM or `--tensor-split`).
pub fn plan_layer_owners(
    n_layers: usize,
    device_bytes: &[u64],
    tensor_split: Option<&[f64]>,
) -> Result<LayerPlan, AllpacaError> {
    if n_layers == 0 {
        return Ok(LayerPlan { owners: Vec::new() });
    }
    if device_bytes.is_empty() {
        return Ok(LayerPlan::all_local(n_layers));
    }
    let n_dev = device_bytes.len();
    let weights: Vec<f64> = if let Some(split) = tensor_split {
        if split.len() != n_dev {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "tensor_split has {} entries but {} split devices",
                split.len(),
                n_dev
            ))));
        }
        split.to_vec()
    } else {
        device_bytes.iter().map(|b| (*b as f64).max(1.0)).collect()
    };
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Ok(LayerPlan::all_local(n_layers));
    }
    let mut owners = vec![0usize; n_layers];
    let mut acc = 0.0f64;
    let mut cuts = Vec::with_capacity(n_dev);
    for w in &weights {
        acc += *w;
        cuts.push((acc / total) * n_layers as f64);
    }
    for (layer, slot) in owners.iter_mut().enumerate() {
        let mut owner = n_dev - 1;
        for (j, cut) in cuts.iter().enumerate() {
            if (layer as f64) < *cut {
                owner = j;
                break;
            }
        }
        *slot = owner;
    }
    Ok(LayerPlan { owners })
}

/// Packed weights fit in the local GPU pool (KV / scratch headroom reserved).
/// RPC is then overflow-only instead of taking a mandatory layer cut.
pub fn packed_fits_local(packed_bytes: u64, local_mems: &[u64]) -> bool {
    const HEADROOM: f64 = 0.82;
    let local: u64 = local_mems.iter().copied().sum();
    packed_bytes > 0 && packed_bytes <= ((local as f64) * HEADROOM) as u64
}

/// Local-only multi-GPU plan: one contiguous layer range per device, filled in
/// device order (target GPU first).
///
/// `layer_bytes[i]` is layer i's VRAM footprint (weights plus its KV share) and
/// `budgets[d]` is what device d can hold for layers. Decode walks layers in
/// order, so a byte parked on the faster target GPU costs less than one on the
/// peer, and contiguous ranges keep the activation hand-off to one hop per
/// boundary. A later device only takes what does not fit the earlier ones.
/// Layers that fit nowhere land on the last device.
pub fn plan_local_fill(layer_bytes: &[u64], budgets: &[u64]) -> LayerPlan {
    let n = layer_bytes.len();
    if n == 0 || budgets.len() < 2 {
        return LayerPlan::all_local(n);
    }
    let mut owners = vec![0usize; n];
    let mut dev = 0usize;
    let mut used = 0u64;
    for (slot, &bytes) in owners.iter_mut().zip(layer_bytes) {
        while dev + 1 < budgets.len() && used + bytes > budgets[dev] {
            dev += 1;
            used = 0;
        }
        *slot = dev;
        used += bytes;
    }
    LayerPlan { owners }
}

/// Local-only two-GPU plan: the peer owns a contiguous *suffix*.
///
/// Used when every layer is full-attn (no GDN). Hybrid Qwen uses
/// [`plan_local_hybrid_peer`] instead.
pub fn plan_local_peer_window(n_layers: usize, local_mems: &[u64]) -> LayerPlan {
    if n_layers == 0 || local_mems.len() < 2 {
        return LayerPlan::all_local(n_layers);
    }
    let m0 = local_mems[0].max(1) as f64;
    let m1 = local_mems[1].max(1) as f64;
    let total = m0 + m1;
    let gpu0_n = ((n_layers as f64) * m0 / total * 1.25).round() as usize;
    let gpu0_n = gpu0_n.clamp((n_layers + 1) / 2, n_layers.saturating_sub(1));
    let mut owners = vec![0usize; n_layers];
    for slot in owners.iter_mut().skip(gpu0_n) {
        *slot = 1;
    }
    LayerPlan { owners }
}

/// Local-only hybrid plan: GPU1 gets the proven middle window plus later GDN
/// up to half the layers, so both cards stay near 12 GB.
///
/// Later full-attn stays on GPU0. Dumping every later GDN on GPU1 packed the
/// 9060 to 14.4/16 GB (garbage). Parking them all on GPU0 packed the 9070
/// the same way. Cap GPU1 at `n/2` layers: middle window + as many later
/// GDN blocks as fit.
pub fn plan_local_hybrid_peer(
    n_layers: usize,
    local_mems: &[u64],
    is_full_attn: &[bool],
) -> LayerPlan {
    if n_layers == 0 || local_mems.len() < 2 {
        return LayerPlan::all_local(n_layers);
    }
    let m0 = local_mems[0].max(1) as f64;
    let m1 = local_mems[1].max(1) as f64;
    let total = m0 + m1;
    let peer_n = ((n_layers as f64) * m1 / total / 2.0).round() as usize;
    let peer_n = peer_n.clamp(1, n_layers.saturating_sub(1));
    let start = ((n_layers as f64) * m0 / total / 2.0).round() as usize;
    let start = start.min(n_layers.saturating_sub(peer_n));
    let end = start + peer_n;
    let cap = (n_layers / 2).max(end);
    let mut owners = vec![0usize; n_layers];
    let mut peer = 0usize;
    for (i, slot) in owners.iter_mut().enumerate() {
        if i >= start && i < end {
            *slot = 1;
            peer += 1;
        } else if i >= end && peer < cap && !is_full_attn.get(i).copied().unwrap_or(true) {
            *slot = 1;
            peer += 1;
        }
    }
    LayerPlan { owners }
}

/// Keep full-attention layers on the target HIP device.
///
/// A VRAM-even local split puts later Qwen full-attn blocks on the peer GPU.
/// That path has produced garbage tokens; GDN on the peer is the proven split.
#[allow(dead_code)]
pub fn pin_full_attn_to_target(layer_device: &mut [i32], target: i32, is_full_attn: &[bool]) {
    for (slot, want) in layer_device.iter_mut().zip(is_full_attn.iter().copied()) {
        if want && *slot >= 0 && *slot != target {
            *slot = target;
        }
    }
}

/// Expert-parallel MoE placement: every layer body (attention or GDN, router,
/// shared expert) runs on a local GPU; routed-expert blocks fill what the
/// bodies leave on the GPU that owns their layer, and the rest go to RPC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPlan {
    /// Local GPU slot per layer, in contiguous runs.
    pub layers: LayerPlan,
    /// The layer's experts sit beside its body; `false` means an RPC worker.
    pub experts_local: Vec<bool>,
}

/// Where the operator wants one layer's routed experts (`rpc.expert_homes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpertPin {
    /// Local while the layer's GPU has room, else a worker.
    #[default]
    Auto,
    /// Beside the layer body on its local GPU.
    Local,
    /// On an RPC worker.
    Remote,
}

/// Split layers across local GPUs in proportion to their budgets, so each
/// GPU keeps room for experts next to its own layer bodies, then keep experts
/// local in layer order while they fit.
#[cfg(test)]
pub fn plan_expert_parallel(body: &[u64], experts: &[u64], budgets: &[u64]) -> ExpertPlan {
    plan_expert_parallel_pinned(body, experts, budgets, &[])
}

/// [`plan_expert_parallel`] with per-layer pins (missing entries are `Auto`).
/// Layers pinned local take their GPU's room first, in layer order; one that
/// does not fit is left remote, as an unpinned one would be. Layers pinned
/// remote never take local room. Layer ownership does not move.
pub fn plan_expert_parallel_pinned(
    body: &[u64],
    experts: &[u64],
    budgets: &[u64],
    pins: &[ExpertPin],
) -> ExpertPlan {
    let n = body.len();
    let total: Vec<u64> = body.iter().zip(experts).map(|(b, e)| b + e).collect();
    let need: u64 = total.iter().sum();
    let have: u64 = budgets.iter().sum::<u64>().max(1);
    let layers = if budgets.len() < 2 {
        LayerPlan::all_local(n)
    } else {
        let scale = need.max(have) as f64 / have as f64;
        let scaled: Vec<u64> = budgets
            .iter()
            .map(|&b| (b as f64 * scale).ceil() as u64)
            .collect();
        plan_local_fill(&total, &scaled)
    };
    let mut left: Vec<u64> = budgets.to_vec();
    if left.is_empty() {
        left.push(0);
    }
    for (l, &b) in body.iter().enumerate() {
        let o = layers.owner(l).min(left.len() - 1);
        left[o] = left[o].saturating_sub(b);
    }
    let pin = |l: usize| pins.get(l).copied().unwrap_or_default();
    let mut experts_local = vec![false; n];
    for want in [ExpertPin::Local, ExpertPin::Auto] {
        for (l, &e) in experts.iter().enumerate() {
            if pin(l) != want {
                continue;
            }
            let o = layers.owner(l).min(left.len() - 1);
            if e <= left[o] {
                left[o] -= e;
                experts_local[l] = true;
            }
        }
    }
    ExpertPlan {
        layers,
        experts_local,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_parallel_keeps_bodies_local_and_overflows_experts() {
        // Four layers of body 1 + experts 10 on two GPUs of 15: each GPU owns
        // two bodies and keeps one layer's experts; the other two go remote.
        let plan = plan_expert_parallel(&[1; 4], &[10; 4], &[15, 15]);
        assert_eq!(plan.layers.owners, vec![0, 0, 1, 1]);
        assert_eq!(plan.experts_local, vec![true, false, true, false]);
        // Everything fits: all local, first GPU filled first.
        let plan = plan_expert_parallel(&[1; 4], &[10; 4], &[100, 100]);
        assert_eq!(plan.experts_local, vec![true; 4]);
        assert_eq!(plan.layers.owners, vec![0, 0, 0, 0]);
    }

    #[test]
    fn expert_pins_claim_local_room_first_and_keep_remote_layers_out() {
        use ExpertPin::{Auto, Local, Remote};
        // One GPU with room for two of four layers' experts.
        let (body, experts, budgets) = ([1u64; 4], [10u64; 4], [24u64]);
        let auto = plan_expert_parallel(&body, &experts, &budgets);
        assert_eq!(auto.experts_local, vec![true, true, false, false]);
        // Pin the last layer local and the first remote: 3 then 1 are local.
        let pins = [Remote, Auto, Auto, Local];
        let plan = plan_expert_parallel_pinned(&body, &experts, &budgets, &pins);
        assert_eq!(plan.experts_local, vec![false, true, false, true]);
        assert_eq!(plan.layers, auto.layers, "pins do not move layer bodies");
        // More local pins than room: the later ones stay remote.
        let plan = plan_expert_parallel_pinned(&body, &experts, &budgets, &[Local; 4]);
        assert_eq!(plan.experts_local, vec![true, true, false, false]);
    }

    #[test]
    fn single_device_all_local() {
        let plan = plan_layer_owners(4, &[8 << 30], None).unwrap();
        assert_eq!(plan.owners, vec![0, 0, 0, 0]);
        assert!(!plan.has_remote());
    }

    #[test]
    fn even_two_devices() {
        let plan = plan_layer_owners(2, &[8 << 30, 8 << 30], None).unwrap();
        assert_eq!(plan.owners, vec![0, 1]);
        assert!(plan.has_remote());
        assert_eq!(plan.runs(), vec![(0, 0, 1), (1, 1, 2)]);
    }

    #[test]
    fn tensor_split_overrides_memory() {
        let plan = plan_layer_owners(4, &[1, 100], Some(&[3.0, 1.0])).unwrap();
        assert_eq!(plan.owners, vec![0, 0, 0, 1]);
    }

    #[test]
    fn split_len_mismatch_is_fatal() {
        assert!(plan_layer_owners(2, &[1, 1], Some(&[1.0])).is_err());
    }

    #[test]
    fn endpoint_split_spreads_across_worker_gpus() {
        let split = expand_rpc_split(&[1.0, 2.0], &[16], &[vec![8, 24]]).unwrap();
        assert_eq!(split.len(), 3);
        assert!((split[0] - 1.0).abs() < 1e-9);
        assert!((split[1] - 0.5).abs() < 1e-9);
        assert!((split[2] - 1.5).abs() < 1e-9);
        let already = expand_rpc_split(&[1.0, 2.0, 3.0], &[16], &[vec![8, 24]]).unwrap();
        assert_eq!(already, vec![1.0, 2.0, 3.0]);
        assert!(expand_rpc_split(&[1.0], &[16], &[vec![8, 24]]).is_err());
    }

    #[test]
    fn local_pool_splits_before_rpc() {
        let split = expand_rpc_split(&[1.0, 1.0], &[16, 16], &[vec![32, 32]]).unwrap();
        assert_eq!(split.len(), 4);
        assert!((split[0] - 0.5).abs() < 1e-9);
        assert!((split[1] - 0.5).abs() < 1e-9);
        assert!((split[2] - 0.5).abs() < 1e-9);
        assert!((split[3] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn packed_27b_fits_two_16gb_local_gpus() {
        let local = [16u64 << 30, 16u64 << 30];
        assert!(packed_fits_local(20u64 << 30, &local));
        assert!(!packed_fits_local(40u64 << 30, &local));
        assert!(!packed_fits_local(0, &local));
    }

    #[test]
    fn pin_full_attn_moves_peer_attention_to_target() {
        let mut devices = vec![0, 1, 1, 1, 0];
        let attn = vec![true, false, true, false, false];
        pin_full_attn_to_target(&mut devices, 0, &attn);
        assert_eq!(devices, vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn local_hybrid_peer_keeps_later_attn_on_target() {
        let attn: Vec<bool> = (0..64).map(|i| i % 4 == 0).collect();
        let plan = plan_local_hybrid_peer(64, &[16 << 30, 16 << 30], &attn);
        assert_eq!(&plan.owners[..16], &[0; 16]);
        assert_eq!(&plan.owners[16..32], &[1; 16]);
        for i in 32..64 {
            if attn[i] {
                assert_eq!(plan.owners[i], 0, "later attn {i}");
            }
        }
        assert_eq!(plan.owners.iter().filter(|&&o| o == 0).count(), 32);
        assert_eq!(plan.owners.iter().filter(|&&o| o == 1).count(), 32);
        let later_attn_on_peer = (32..64).filter(|&i| attn[i] && plan.owners[i] == 1).count();
        assert_eq!(later_attn_on_peer, 0);
    }

    #[test]
    fn local_fill_keeps_everything_on_target_when_it_fits() {
        let plan = plan_local_fill(&[10; 8], &[100, 100]);
        assert_eq!(plan.owners, vec![0; 8]);
        assert!(!plan.has_remote());
    }

    #[test]
    fn local_fill_spills_contiguous_suffix_to_peer() {
        let plan = plan_local_fill(&[10; 8], &[35, 100]);
        assert_eq!(plan.owners, vec![0, 0, 0, 1, 1, 1, 1, 1]);
        assert_eq!(plan.runs(), vec![(0, 0, 3), (1, 3, 8)]);
    }

    #[test]
    fn local_fill_overflow_lands_on_last_device() {
        let plan = plan_local_fill(&[10; 6], &[20, 20]);
        assert_eq!(plan.owners, vec![0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn local_fill_uneven_layers_respect_budget() {
        // Heavier later layers must not push the target past its budget.
        let plan = plan_local_fill(&[5, 5, 30, 5], &[20, 100]);
        assert_eq!(plan.owners, vec![0, 0, 1, 1]);
    }

    #[test]
    fn local_peer_window_single_device_stays_target() {
        let plan = plan_local_peer_window(8, &[16 << 30]);
        assert_eq!(plan.owners, vec![0; 8]);
    }
}
