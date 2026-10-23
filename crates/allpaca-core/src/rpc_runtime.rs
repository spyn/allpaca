use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use allpaca_gguf::{GgufFile, HybridLayerKind};
use allpaca_rpc::{
    BufId, KernelOp, LaunchItem, LaunchList, MoeFfnRequest, RpcBackendKind, RpcSession,
};
use allpaca_types::{AllpacaError, DeviceKind, FatalError, RPC_DEVICE_ID_BASE};

use crate::compute::ComputeDev;
use crate::layer_split::{
    expand_rpc_split, packed_fits_local, plan_layer_owners, plan_local_hybrid_peer,
    plan_local_peer_window, LayerPlan,
};
use crate::model::{LayerWeights, ModelConfig, ResidentModel, TinyModel};

/// Connected RPC workers plus the layer assignment used at decode.
pub struct RpcRuntime {
    sessions: Mutex<Vec<RpcSession>>,
    /// Worker `host:port` per session index (readable while a session is busy).
    endpoints: Vec<String>,
    /// Each session's call counters, readable while the session is busy.
    counters: Vec<(Arc<allpaca_rpc::CallCounters>, Arc<allpaca_rpc::CallCounters>)>,
    plan: Mutex<LayerPlan>,
    remote: Mutex<Vec<Option<RemoteLayer>>>,
    /// Expert-parallel MoE: layers whose routed experts live on a worker.
    experts: Mutex<Vec<Option<RemoteExperts>>>,
    /// HIP device for each local layer. `-1` means the layer was placed on RPC.
    layer_device: Mutex<Vec<i32>>,
    /// Second local GPU: device id, workspace, and the layers that live there.
    local_peer: Mutex<Option<(i32, Arc<allpaca_hip::HipWorkspace>, Arc<ResidentModel>)>>,
    /// `engine.local_vram_budget_mib`: per-slot layer budgets that replace the
    /// measured ones (0 or missing = measured).
    local_budget_mib: Mutex<Vec<u64>>,
    /// `rpc.expert_homes`: operator pins for layers' routed experts.
    expert_pins: Mutex<std::collections::BTreeMap<usize, String>>,
    /// Expert-parallel layout of the loaded model: bytes of each layer's
    /// routed experts and the worker GPUs they may be placed on.
    expert_layout: Mutex<(Vec<u64>, Vec<String>)>,
    /// Bytes of expert slots put on each (session, worker GPU) by
    /// [`Self::put_expert_slots`].
    slot_bytes: Mutex<std::collections::HashMap<(usize, u32), u64>>,
    /// Last worker GPU stats and when they were read.
    gpu_stats: Mutex<(Option<std::time::Instant>, Vec<RemoteGpu>)>,
    /// `rpc.prefill_rows`: rows per batched span call (0 = per token).
    prefill_rows: std::sync::atomic::AtomicUsize,
    /// Batched-prefill activations per (session, worker device), shared by
    /// every remote layer on that device.
    row_scratch: Mutex<std::collections::HashMap<(usize, u32), RowScratch>>,
    pub roundtrips: AtomicU64,
    pub activation_bytes: AtomicU64,
    pub h2d_bytes: AtomicU64,
}

impl std::fmt::Debug for RpcRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RpcRuntime")
            .field("sessions", &self.session_count())
            .field("remote_layers", &self.plan_has_remote())
            .field("roundtrips", &self.roundtrips.load(Ordering::Relaxed))
            .finish()
    }
}

/// Activation buffers for `rows` prompt rows on one worker GPU, sized for the
/// widest remote layer there. Batched launch lists run in these; the
/// per-layer single-row buffers stay for decode.
#[derive(Debug, Clone, Copy)]
struct RowScratch {
    rows: usize,
    x: BufId,
    norm_out: BufId,
    q_proj: BufId,
    q: BufId,
    q_gate: BufId,
    k: BufId,
    v: BufId,
    attn_out: BufId,
    hidden: BufId,
    gate: BufId,
    up: BufId,
}

/// How long a worker GPU reading is served before it is refreshed.
const GPU_STATS_MAX_AGE: std::time::Duration = std::time::Duration::from_secs(2);

/// Longest a stats refresh waits for the worker sessions to be free.
const GPU_STATS_LOCK_WAIT: std::time::Duration = std::time::Duration::from_millis(60);

/// One GPU on an RPC worker, for the operator dashboard. `None` = the
/// worker's platform does not report it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RemoteGpu {
    /// Worker `host:port`.
    pub endpoint: String,
    /// Device index on that worker.
    pub index: u32,
    pub name: String,
    /// Device-wide VRAM in use (all processes).
    pub mem_used_bytes: u64,
    pub mem_total_bytes: u64,
    pub util_pct: Option<u32>,
    pub temp_c: Option<u32>,
    pub power_w: Option<u32>,
}

/// Where one layer's body runs (see [`RpcRuntime::layer_home`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerHome {
    Local(i32),
    Remote { endpoint: String, device: u32 },
}

/// One layer's routed experts on a worker GPU (`MoeFfn`).
#[derive(Clone, Copy)]
struct RemoteExperts {
    session: usize,
    /// Worker GPU holding the experts.
    device: u32,
    req: MoeFfnRequest,
    /// Send rows as int8 blocks: the worker's fused K-quant kernels quantize
    /// to exactly that, so it is lossless and a quarter of the bytes.
    q8: bool,
}

#[derive(Clone, Copy)]
struct RemoteLayer {
    session: usize,
    /// Worker GPU holding this layer's buffers.
    device: u32,
    attn_norm: BufId,
    wq: BufId,
    wk: BufId,
    wv: BufId,
    wo: BufId,
    ffn_norm: BufId,
    w_gate: BufId,
    w_up: BufId,
    w_down: BufId,
    x: BufId,
    norm_out: BufId,
    q: BufId,
    k: BufId,
    v: BufId,
    attn_out: BufId,
    hidden: BufId,
    gate: BufId,
    up: BufId,
    k_cache: BufId,
    v_cache: BufId,
    dim: usize,
    kv_dim: usize,
    ff_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    theta: f32,
    max_seq: usize,
    /// Weight GEMMs are `GemmNtQuant` against packed GGUF bytes.
    packed: bool,
    /// Q projection output length. Equals `dim` for the square F32 path.
    q_len: usize,
    /// `wo` K. Equals `dim` for the square F32 path.
    o_in: usize,
    /// Q RoPE head size. `0` means use `head_dim`.
    q_head_dim: usize,
    /// Qwen3.5 full attention: Q GEMM is per-head `[Q|gate]`.
    gated: bool,
    attn_dim: usize,
    /// Destination of the Q GEMM. Equals `q` when `gated` is false.
    q_proj: BufId,
    q_gate: BufId,
    q_norm: BufId,
    k_norm: BufId,
    has_q_norm: bool,
    has_k_norm: bool,
    /// `0` uses full-head RoPE. Otherwise partial / sectioned RoPE.
    n_rot: usize,
    rope_sections: [i32; 4],
}

/// VRAM a local GPU keeps back from layer weights: HIP context, decode and
/// prefill workspaces, hipBLASLt scratch, and desktop churn on a display GPU.
fn local_reserve_bytes(total: u64) -> u64 {
    (768u64 << 20) + total / 32
}

/// Place GGUF layers on the local GPUs by measured free VRAM and per-layer bytes.
///
/// The target GPU (first) also carries lm_head / `output_norm`; the second
/// keeps room for the draft model. Free VRAM is
/// read now, not the card's total: on Windows a display GPU can have several
/// GiB taken by the desktop, and WDDM silently spills past-capacity
/// allocations to shared system memory, which reads over PCIe at decode.
/// `engine.local_vram_budget_mib` entry for `slot`, in bytes; `None` keeps
/// the measured budget.
fn pinned_budget(pinned_mib: &[u64], slot: usize) -> Option<u64> {
    pinned_mib.get(slot).copied().filter(|m| *m > 0).map(|m| m << 20)
}

fn plan_local_by_bytes(
    model: &TinyModel,
    file: &GgufFile,
    local_ids: &[i32],
    is_full_attn: &[bool],
    kv_vram_bytes: u64,
    draft_bytes: u64,
    pinned_mib: &[u64],
) -> LayerPlan {
    let (mut layer_bytes, globals) = crate::model::gguf_device_bytes(file, &model.config);
    let n_attn = is_full_attn.iter().filter(|a| **a).count().max(1) as u64;
    let kv_share = kv_vram_bytes / n_attn;
    for (bytes, attn) in layer_bytes.iter_mut().zip(is_full_attn) {
        if *attn {
            *bytes += kv_share;
        }
    }
    let mut budgets = Vec::with_capacity(local_ids.len());
    for (slot, dev) in local_ids.iter().copied().enumerate() {
        let (hip_free, total) = allpaca_hip::device_memory_info(dev).unwrap_or((0, 0));
        let free = allpaca_hip::device_local_headroom(dev).map_or(hip_free, |h| h.min(hip_free));
        let mut budget = free.saturating_sub(local_reserve_bytes(total));
        if slot == 0 {
            budget = budget.saturating_sub(globals);
        } else if slot == 1 {
            budget = budget.saturating_sub(draft_bytes);
        }
        let pinned = pinned_budget(pinned_mib, slot);
        if let Some(b) = pinned {
            budget = b;
        }
        tracing::info!(
            hip = dev,
            free_mib = free >> 20,
            total_mib = total >> 20,
            budget_mib = budget >> 20,
            pinned = pinned.is_some(),
            "local GPU pool budget"
        );
        budgets.push(budget);
    }
    let plan = crate::layer_split::plan_local_fill(&layer_bytes, &budgets);
    let need: u64 = layer_bytes.iter().sum::<u64>() + globals;
    let have: u64 = budgets.iter().sum::<u64>() + globals;
    if need > have {
        tracing::warn!(
            need_mib = need >> 20,
            budget_mib = have >> 20,
            "GGUF layers exceed local GPU budgets; the last GPU is oversubscribed"
        );
    }
    for (owner, start, end) in plan.runs() {
        let bytes: u64 = layer_bytes[start..end].iter().sum();
        tracing::info!(
            hip = local_ids.get(owner).copied().unwrap_or(-1),
            layers = format!("{start}..{end}"),
            mib = bytes >> 20,
            "local-only HIP pool: contiguous layer range"
        );
    }
    plan
}

impl RpcRuntime {
    pub fn connect(endpoints: &[String], token: Option<&str>) -> Result<Self, AllpacaError> {
        let mut sessions = Vec::new();
        let token = token.unwrap_or("");
        for ep in endpoints {
            tracing::info!(endpoint = %ep, "rpc HELLO");
            let session = RpcSession::connect_addr_with_token(ep, token).map_err(map_rpc)?;
            if !session.same_build() {
                tracing::warn!(
                    endpoint = %ep,
                    worker = session.build_id().unwrap_or("unknown"),
                    coordinator = allpaca_rpc::BUILD_ID,
                    "rpc worker was built from other sources than this coordinator; rebuild it (scripts/deploy-worker.sh)"
                );
            }
            sessions.push(session);
        }
        let counters = sessions.iter().map(RpcSession::counters).collect();
        Ok(Self {
            sessions: Mutex::new(sessions),
            endpoints: endpoints.to_vec(),
            counters,
            plan: Mutex::new(LayerPlan::all_local(0)),
            remote: Mutex::new(Vec::new()),
            experts: Mutex::new(Vec::new()),
            layer_device: Mutex::new(Vec::new()),
            local_budget_mib: Mutex::new(Vec::new()),
            expert_pins: Mutex::new(std::collections::BTreeMap::new()),
            expert_layout: Mutex::new((Vec::new(), Vec::new())),
            slot_bytes: Mutex::new(std::collections::HashMap::new()),
            gpu_stats: Mutex::new((None, Vec::new())),
            prefill_rows: std::sync::atomic::AtomicUsize::new(128),
            row_scratch: Mutex::new(std::collections::HashMap::new()),
            local_peer: Mutex::new(None),
            roundtrips: AtomicU64::new(0),
            activation_bytes: AtomicU64::new(0),
            h2d_bytes: AtomicU64::new(0),
        })
    }

    pub fn apply_hello_to_inventory(
        &self,
        devices: &mut Vec<allpaca_types::DeviceInfo>,
    ) -> Result<(), AllpacaError> {
        let sessions = self.sessions.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into()))
        })?;
        let endpoints: Vec<String> = devices
            .iter()
            .filter(|d| d.is_rpc())
            .filter_map(|d| d.rpc_endpoint().map(str::to_string))
            .collect();
        devices.retain(|d| !d.is_rpc());
        let mut id = RPC_DEVICE_ID_BASE;
        for (i, session) in sessions.iter().enumerate() {
            let hello = session.hello();
            let endpoint = endpoints.get(i).map(String::as_str).unwrap_or("rpc");
            let kind = match hello.backend {
                RpcBackendKind::Hip => DeviceKind::HipGpu,
                RpcBackendKind::Cuda => DeviceKind::CudaGpu,
                RpcBackendKind::Cpu => DeviceKind::Cpu,
            };
            let mems = hello.device_memories();
            for (dev_i, mem) in mems.iter().copied().enumerate() {
                let mut dev = allpaca_types::DeviceInfo::rpc(id, endpoint, mem, kind);
                if mems.len() > 1 {
                    dev.name = format!("rpc[{endpoint}#{dev_i}]");
                }
                devices.push(dev);
                id += 1;
            }
        }
        Ok(())
    }

    pub fn plan_has_remote(&self) -> bool {
        self.plan.lock().map(|p| p.has_remote()).unwrap_or(false)
    }

    /// Calls to each worker since connect, in endpoint order. Never waits on a
    /// busy session.
    pub fn worker_calls(&self) -> Vec<allpaca_types::RpcWorkerCalls> {
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        self.endpoints
            .iter()
            .zip(&self.counters)
            .map(|(endpoint, (calls, ext))| allpaca_types::RpcWorkerCalls {
                endpoint: endpoint.clone(),
                calls: load(&calls.calls),
                wall_us: load(&calls.wall_us),
                bytes_out: load(&calls.bytes_out),
                bytes_in: load(&calls.bytes_in),
                ext_calls: load(&ext.calls),
                ext_wait_us: load(&ext.wall_us),
                ext_bytes_out: load(&ext.bytes_out),
                ext_bytes_in: load(&ext.bytes_in),
            })
            .collect()
    }

    /// A layer was uploaded to a worker. A split plan alone does not count:
    /// large GGUFs skip host-F32 layers and stay on the local device.
    pub fn placed_remote(&self) -> bool {
        self.remote
            .lock()
            .map(|layers| layers.iter().any(|layer| layer.is_some()))
            .unwrap_or(false)
    }

    pub fn session_count(&self) -> usize {
        self.sessions.lock().map(|s| s.len()).unwrap_or(0)
    }

    /// Every worker GPU's live stats, at most [`GPU_STATS_MAX_AGE`] old when
    /// the sessions are free. Never waits on a session: while one is busy (a
    /// weight upload, a decode round trip) the last reading is returned.
    pub fn remote_gpu_stats(&self) -> Vec<RemoteGpu> {
        let Ok(mut cache) = self.gpu_stats.lock() else {
            return Vec::new();
        };
        let fresh = cache.0.is_some_and(|t| t.elapsed() < GPU_STATS_MAX_AGE);
        if !fresh {
            // During decode the sessions are busy in back-to-back round
            // trips: wait briefly for a gap between two of them. A weight
            // upload holds them for seconds, and then this gives up.
            let deadline = std::time::Instant::now() + GPU_STATS_LOCK_WAIT;
            let sessions = loop {
                match self.sessions.try_lock() {
                    Ok(g) => break Some(g),
                    Err(_) if std::time::Instant::now() < deadline => {
                        std::thread::sleep(std::time::Duration::from_micros(200));
                    }
                    Err(_) => break None,
                }
            };
            if let Some(mut sessions) = sessions {
                let mut out = Vec::new();
                for s in sessions.iter_mut() {
                    let endpoint = s.endpoint().to_string();
                    let Ok(stats) = s.device_stats() else {
                        continue;
                    };
                    for (index, d) in stats.into_iter().enumerate() {
                        let known = |v: i32| u32::try_from(v).ok();
                        out.push(RemoteGpu {
                            endpoint: endpoint.clone(),
                            index: index as u32,
                            name: d.name,
                            mem_used_bytes: d.mem_used,
                            mem_total_bytes: d.mem_total,
                            util_pct: known(d.util_pct),
                            temp_c: known(d.temp_c),
                            power_w: known(d.power_w),
                        });
                    }
                }
                *cache = (Some(std::time::Instant::now()), out);
            }
        }
        cache.1.clone()
    }

    /// Pin the per-GPU layer budgets (`engine.local_vram_budget_mib`).
    pub fn set_local_budgets_mib(&self, mib: &[u64]) {
        if let Ok(mut b) = self.local_budget_mib.lock() {
            *b = mib.to_vec();
        }
    }

    /// Operator pins for layers' routed experts (`rpc.expert_homes`), read
    /// by the next [`Self::prepare`]. Keys that are not layer numbers are
    /// ignored.
    pub fn set_expert_homes(&self, homes: &std::collections::BTreeMap<String, String>) {
        if let Ok(mut pins) = self.expert_pins.lock() {
            *pins = homes
                .iter()
                .filter_map(|(layer, home)| Some((layer.trim().parse().ok()?, home.trim().to_string())))
                .collect();
        }
    }

    fn local_budgets_mib(&self) -> Vec<u64> {
        self.local_budget_mib.lock().map(|b| b.clone()).unwrap_or_default()
    }

    /// HIP device that runs a local layer. `-1` when the layer is on RPC.
    pub fn local_device(&self, layer: usize) -> i32 {
        self.layer_device
            .lock()
            .ok()
            .and_then(|d| d.get(layer).copied())
            .unwrap_or(0)
    }

    /// Move every local layer on HIP `from` to HIP `to` (the second GPU's
    /// layers were not uploaded there, so the target GPU runs them).
    pub fn fold_local_device(&self, from: i32, to: i32) {
        if let Ok(mut devs) = self.layer_device.lock() {
            for d in devs.iter_mut().filter(|d| **d == from) {
                *d = to;
            }
        }
    }

    /// Local layers occupy more than one HIP device.
    pub fn splits_local(&self) -> bool {
        let Ok(devs) = self.layer_device.lock() else {
            return false;
        };
        let mut first = None;
        for d in devs.iter().copied() {
            if d < 0 {
                continue;
            }
            match first {
                None => first = Some(d),
                Some(found) if found != d => return true,
                _ => {}
            }
        }
        false
    }

    /// Install the second local GPU's workspace and the layers uploaded there.
    pub fn set_local_peer(
        &self,
        device: i32,
        hip: Arc<allpaca_hip::HipWorkspace>,
        resident: Arc<ResidentModel>,
    ) {
        if let Ok(mut slot) = self.local_peer.lock() {
            *slot = Some((device, hip, resident));
        }
    }

    pub fn local_peer(
        &self,
    ) -> Option<(i32, Arc<allpaca_hip::HipWorkspace>, Arc<ResidentModel>)> {
        self.local_peer.lock().ok().and_then(|slot| slot.clone())
    }

    /// Split devices: local GPU pool first, then each RPC GPU.
    ///
    /// `kv_vram_bytes` is the KV VRAM budget; attention layers reserve a share
    /// of it on whichever local GPU they land. `draft_bytes` stays free on the
    /// second local GPU, where the draft model is uploaded.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &self,
        model: &TinyModel,
        local_devices: &[(i32, u64)],
        tensor_split: Option<&[f64]>,
        max_seq: usize,
        gguf: Option<&GgufFile>,
        kv_vram_bytes: u64,
        draft_bytes: u64,
    ) -> Result<(), AllpacaError> {
        let local_devices = if local_devices.is_empty() {
            vec![(0, 1u64 << 30)]
        } else {
            local_devices.to_vec()
        };
        let n_local = local_devices.len();
        let mut device_bytes = Vec::with_capacity(n_local);
        let mut local_ids = Vec::with_capacity(n_local);
        for (dev, mem) in &local_devices {
            local_ids.push(*dev);
            device_bytes.push((*mem).max(1));
        }
        let local_mems: Vec<u64> = device_bytes.clone();
        let mut owners = Vec::new();
        let groups = {
            let sessions = self.sessions.lock().map_err(|_| {
                AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into()))
            })?;
            let mut groups = Vec::new();
            for (session_idx, s) in sessions.iter().enumerate() {
                let mems = s.hello().device_memories();
                for (device, mem) in mems.iter().copied().enumerate() {
                    let mem = mem.max(1);
                    device_bytes.push(mem);
                    owners.push((session_idx, device as u32));
                }
                groups.push(mems);
            }
            groups
        };
        if let Some(file) = gguf {
            if !groups.is_empty() && tensor_split.is_none() && has_stacked_experts(file) {
                return self.prepare_expert_parallel(
                    model,
                    file,
                    &local_ids,
                    &owners,
                    kv_vram_bytes,
                    draft_bytes,
                );
            }
        }
        let expanded = match tensor_split {
            Some(split) => Some(expand_rpc_split(split, &local_mems, &groups)?),
            None => None,
        };
        let packed_bytes = gguf
            .map(|file| file.tensors.iter().map(|t| t.nbytes() as u64).sum::<u64>())
            .unwrap_or(0);
        // GPU before GPU-over-RPC: weights that fit the local pool stay there
        // even when RPC workers are configured (an explicit tensor_split
        // still decides).
        let fits_local = packed_fits_local(packed_bytes, &local_mems) && tensor_split.is_none();
        if fits_local {
            tracing::info!(
                packed_bytes,
                local_vram = local_mems.iter().copied().sum::<u64>(),
                rpc = !groups.is_empty(),
                "packed weights fit in the local GPU pool"
            );
        }
        let plan = if fits_local && n_local == 1 {
            plan_layer_owners(model.config.n_layers, &device_bytes[..1], None)?
        } else if (groups.is_empty() || fits_local) && n_local > 1 && tensor_split.is_none() {
            let attn: Vec<bool> = model
                .layers
                .iter()
                .map(|l| l.block == HybridLayerKind::Attention)
                .collect();
            if let Some(file) = gguf {
                plan_local_by_bytes(
                    model,
                    file,
                    &local_ids,
                    &attn,
                    kv_vram_bytes,
                    draft_bytes,
                    &self.local_budgets_mib(),
                )
            } else {
                let hybrid = attn.iter().any(|full| !*full);
                let window = if hybrid {
                    plan_local_hybrid_peer(model.config.n_layers, &local_mems, &attn)
                } else {
                    plan_local_peer_window(model.config.n_layers, &local_mems)
                };
                tracing::info!(
                    peer_layers = window.owners.iter().filter(|&&o| o != 0).count(),
                    hybrid,
                    "local-only HIP pool: hybrid peer (middle window + later GDN)"
                );
                window
            }
        } else {
            plan_layer_owners(model.config.n_layers, &device_bytes, expanded.as_deref())?
        };
        let mut remote = vec![None; model.layers.len()];
        let mut layer_device = vec![local_ids[0]; model.layers.len()];
        let mut local_load = vec![0u64; n_local];
        let cfg = &model.config;
        let max_seq = max_seq.max(1);
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            let owner = plan.owner(layer_idx);
            if owner < n_local {
                layer_device[layer_idx] = local_ids[owner];
                local_load[owner] += 1;
                continue;
            }
            let (session_idx, device) = owners.get(owner - n_local).copied().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::InvalidConfig(
                    "rpc layer owner has no device".into(),
                ))
            })?;
            if let Some(file) = gguf {
                if let Some(uploaded) =
                    self.upload_packed_layer(file, layer_idx, session_idx, device, cfg, max_seq)?
                {
                    remote[layer_idx] = Some(uploaded);
                    layer_device[layer_idx] = -1;
                    continue;
                }
            }
            if is_dense_attn_ffn(layer) {
                let uploaded = self.upload_layer(session_idx, device, cfg, layer, max_seq)?;
                remote[layer_idx] = Some(uploaded);
                layer_device[layer_idx] = -1;
                continue;
            }
            let pick = 0usize;
            layer_device[layer_idx] = local_ids[pick];
            local_load[pick] += 1;
            tracing::warn!(
                layer_idx,
                device = local_ids[pick],
                "rpc owner skipped; layer stays on the local target GPU"
            );
        }
        let placed = remote.iter().any(|layer| layer.is_some());
        let remote_planned = plan.owners.iter().any(|&o| o >= n_local);
        if remote_planned && !placed {
            tracing::warn!(
                "rpc layer plan names remote owners but no dense layer was uploaded; decode stays on the local GPUs"
            );
        }
        for (slot, id) in local_ids.iter().copied().enumerate() {
            let layers = layer_device.iter().filter(|d| **d == id).count();
            tracing::info!(
                slot,
                hip = id,
                layers,
                "local GPU pool occupancy"
            );
        }
        let rpc_layers = layer_device.iter().filter(|d| **d < 0).count();
        tracing::info!(
            local_devices = n_local,
            rpc_layers,
            remote_layers = placed,
            "layer split: local GPU pool, then RPC if any"
        );
        *self.plan.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc plan lock".into()))
        })? = plan;
        *self.remote.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc remote lock".into()))
        })? = remote;
        // Scratch is sized for the layers placed before; size it again lazily.
        if let Ok(mut scratch) = self.row_scratch.lock() {
            scratch.clear();
        }
        *self.layer_device.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc layer device lock".into()))
        })? = layer_device;
        Ok(())
    }

    /// Expert-parallel MoE: every layer body stays on the local GPUs, which
    /// then fill with routed experts to ~92% of VRAM; the remaining layers'
    /// experts go to worker GPUs by free memory and run there as `MoeFfn`.
    fn prepare_expert_parallel(
        &self,
        model: &TinyModel,
        file: &GgufFile,
        local_ids: &[i32],
        owners: &[(usize, u32)],
        kv_vram_bytes: u64,
        draft_bytes: u64,
    ) -> Result<(), AllpacaError> {
        let cfg = &model.config;
        let n = model.layers.len();
        let (mut body, experts, globals) = crate::model::gguf_device_bytes_split(file, cfg);
        let is_attn: Vec<bool> = model
            .layers
            .iter()
            .map(|l| l.block == HybridLayerKind::Attention)
            .collect();
        let kv_share = kv_vram_bytes / is_attn.iter().filter(|a| **a).count().max(1) as u64;
        for (b, attn) in body.iter_mut().zip(&is_attn) {
            if *attn {
                *b += kv_share;
            }
        }
        let pinned_mib = self.local_budgets_mib();
        let mut budgets = Vec::with_capacity(local_ids.len());
        for (slot, dev) in local_ids.iter().copied().enumerate() {
            let (hip_free, total) = allpaca_hip::device_memory_info(dev).unwrap_or((0, 0));
            let free = allpaca_hip::device_local_headroom(dev).map_or(hip_free, |h| h.min(hip_free));
            // Keep ~8% of the card free: workspaces, hipBLASLt, desktop churn.
            let reserve = local_reserve_bytes(total).max(total / 100 * 8);
            let mut budget = free.saturating_sub(reserve);
            if slot == 0 {
                budget = budget.saturating_sub(globals);
            } else if slot == 1 {
                budget = budget.saturating_sub(draft_bytes);
            }
            let pinned = pinned_budget(&pinned_mib, slot);
            if let Some(b) = pinned {
                budget = b;
            }
            tracing::info!(
                hip = dev,
                free_mib = free >> 20,
                total_mib = total >> 20,
                budget_mib = budget >> 20,
                pinned = pinned.is_some(),
                "expert-parallel: local GPU budget"
            );
            budgets.push(budget);
        }
        // Operator pins: "local", or "rpc:<endpoint>/<gpu>" naming a worker
        // GPU this run has. Anything else is placed automatically.
        use crate::layer_split::ExpertPin;
        let targets: Vec<String> = owners
            .iter()
            .map(|&(s, d)| format!("rpc:{}/{d}", self.endpoints.get(s).map_or("", String::as_str)))
            .collect();
        let wanted = self.expert_pins.lock().map(|p| p.clone()).unwrap_or_default();
        let mut pins = vec![ExpertPin::Auto; n];
        let mut pinned_owner: Vec<Option<usize>> = vec![None; n];
        for (&l, home) in &wanted {
            if l >= n {
                tracing::warn!(layer = l, "expert-parallel: pin names a layer this model does not have");
            } else if home == "local" {
                pins[l] = ExpertPin::Local;
            } else if let Some(o) = targets.iter().position(|t| t == home) {
                pins[l] = ExpertPin::Remote;
                pinned_owner[l] = Some(o);
            } else {
                tracing::warn!(layer = l, home = %home, "expert-parallel: pinned device is not connected; placing automatically");
            }
        }
        let plan = crate::layer_split::plan_expert_parallel_pinned(&body, &experts, &budgets, &pins);
        for l in 0..n {
            if pins[l] == ExpertPin::Local && !plan.experts_local[l] && experts[l] > 0 {
                tracing::warn!(
                    layer = l,
                    mib = experts[l] >> 20,
                    "expert-parallel: experts pinned local do not fit their GPU; using a worker"
                );
            }
        }
        let mut layer_device = vec![local_ids[0]; n];
        for (l, dev) in layer_device.iter_mut().enumerate() {
            *dev = local_ids[plan.layers.owner(l).min(local_ids.len() - 1)];
        }
        // Worker GPU free bytes, filled in order: consecutive remote layers
        // share a GPU. An MI50 left idle ~2 ms between calls drops its clocks
        // and runs the next MoeFfn ~2x slower; alternating GPUs per layer
        // idled each one that long every token.
        let mut worker_free: Vec<u64> = {
            let sessions = self.sessions.lock().map_err(|_| {
                AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into()))
            })?;
            owners
                .iter()
                .map(|&(s, d)| {
                    let mems = sessions[s].hello().device_memories();
                    let mem = mems.get(d as usize).copied().unwrap_or(0);
                    // Leave the worker ~5% for MoeFfn scratch.
                    mem.saturating_sub(mem / 20)
                })
                .collect()
        };
        // Pinned layers take their worker GPU's room first; one that does
        // not fit there is placed like an unpinned layer.
        let remote = |l: usize| !plan.experts_local[l] && experts[l] > 0;
        let mut picks: Vec<Option<usize>> = vec![None; n];
        for l in (0..n).filter(|&l| remote(l)) {
            if let Some(o) = pinned_owner[l] {
                if worker_free[o] >= experts[l] {
                    worker_free[o] -= experts[l];
                    picks[l] = Some(o);
                } else {
                    tracing::warn!(
                        layer = l,
                        home = %targets[o],
                        mib = experts[l] >> 20,
                        "expert-parallel: pinned worker GPU is full; placing automatically"
                    );
                }
            }
        }
        let mut remote_experts = vec![None; n];
        let mut remote_bytes = 0u64;
        for l in 0..n {
            if !remote(l) {
                continue;
            }
            let pick = match picks[l] {
                Some(o) => o,
                None => {
                    let o = (0..owners.len())
                        .find(|&o| worker_free[o] >= experts[l])
                        .ok_or_else(|| {
                            AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                                "expert-parallel: layer {l} experts ({} MiB) fit no local or RPC GPU",
                                experts[l] >> 20
                            )))
                        })?;
                    worker_free[o] -= experts[l];
                    o
                }
            };
            let (session, device) = owners[pick];
            remote_experts[l] = Some(self.upload_experts(file, cfg, l, session, device)?);
            remote_bytes += experts[l];
            tracing::info!(
                layer = l,
                session,
                device,
                mib = experts[l] >> 20,
                "expert-parallel: layer experts uploaded to rpc"
            );
        }
        for (slot, id) in local_ids.iter().copied().enumerate() {
            let on = |l: &usize| layer_device[*l] == id;
            let body_mib: u64 = (0..n).filter(on).map(|l| body[l]).sum::<u64>() >> 20;
            let expert_mib: u64 = (0..n)
                .filter(|l| on(l) && plan.experts_local[*l])
                .map(|l| experts[l])
                .sum::<u64>()
                >> 20;
            tracing::info!(
                slot,
                hip = id,
                layers = (0..n).filter(on).count(),
                body_mib,
                expert_mib,
                budget_mib = budgets[slot] >> 20,
                "expert-parallel: local GPU plan"
            );
        }
        tracing::info!(
            remote_expert_layers = remote_experts.iter().filter(|e| e.is_some()).count(),
            remote_mib = remote_bytes >> 20,
            "expert-parallel: routed experts on rpc"
        );
        let lock = |e| AllpacaError::Fatal(FatalError::InvalidConfig(e));
        *self.plan.lock().map_err(|_| lock("rpc plan lock".into()))? = plan.layers;
        *self.remote.lock().map_err(|_| lock("rpc remote lock".into()))? = vec![None; n];
        *self.experts.lock().map_err(|_| lock("rpc experts lock".into()))? = remote_experts;
        *self.expert_layout.lock().map_err(|_| lock("rpc expert layout lock".into()))? =
            (experts, targets);
        *self.layer_device.lock().map_err(|_| lock("rpc layer device lock".into()))? =
            layer_device;
        Ok(())
    }

    /// `PutTensor` a layer's stacked `ffn_{gate,up,down}_exps` onto one worker GPU.
    fn upload_experts(
        &self,
        file: &GgufFile,
        cfg: &ModelConfig,
        layer: usize,
        session: usize,
        device: u32,
    ) -> Result<RemoteExperts, AllpacaError> {
        let blk = cfg.gguf_blk(layer);
        let mut sessions = self.sessions.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into()))
        })?;
        let s = &mut sessions[session];
        let backend = s.hello().backend;
        let mut put = |part: &str| -> Result<BufId, AllpacaError> {
            let name = format!("blk.{blk}.ffn_{part}_exps.weight");
            let info = file.tensor(&name).ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!("missing tensor {name}")))
            })?;
            if !worker_accepts_type(backend, info.ggml_type_id) {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "rpc worker has no gemm for {name} (ggml type {})",
                    info.ggml_type_id
                ))));
            }
            let dims: Vec<u32> = info.dims.iter().map(|&d| d as u32).collect();
            let bytes = file.tensor_bytes(&name)?;
            self.h2d_bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
            s.put_tensor(info.ggml_type_id, &dims, bytes, Some(device))
                .map_err(map_rpc)
        };
        let (gate, up, down) = (put("gate")?, put("up")?, put("down")?);
        // What the worker's fused MoE kernels take (they read int8 rows):
        // K-quants, Q8_0 with gate and up both Q8_0, or MoESQ experts.
        let ty = |part: &str| {
            file.tensor(&format!("blk.{blk}.ffn_{part}_exps.weight"))
                .map_or(0, |t| t.ggml_type_id)
        };
        let (g, u, d) = (ty("gate"), ty("up"), ty("down"));
        // Only the HIP worker's kernels have the Q8_0 case.
        let q8_0_ok = matches!(backend, RpcBackendKind::Hip) || ![g, u, d].contains(&8);
        // MoESQ paired-4:8 experts (200) are fused only as a full set, and
        // only on the HIP worker.
        let p48 = [g, u, d] == [200, 200, 200] && matches!(backend, RpcBackendKind::Hip);
        let kquant = p48
            || ([g, u, d].iter().all(|t| matches!(t, 8 | 12..=14))
                && (g == 8) == (u == 8)
                && q8_0_ok);
        let fused = matches!(backend, RpcBackendKind::Hip | RpcBackendKind::Cuda)
            && cfg.dim % 256 == 0
            && cfg.ff_dim % 256 == 0;
        Ok(RemoteExperts {
            q8: kquant && fused,
            session,
            device,
            req: MoeFfnRequest {
                gate,
                up,
                down,
                n_experts: cfg.n_experts as u32,
                dim: cfg.dim as u32,
                ff: cfg.ff_dim as u32,
                // Sunny's 5 GbE card sits in a Gen2 x1 slot (~2 Gbit/s): the
                // reply is the larger half of every call, so ship it as F16.
                f16_out: true,
            },
        })
    }

    /// The first worker (any backend) that serves the extension `name`.
    pub fn session_serving(&self, name: &str) -> Option<usize> {
        let sessions = self.sessions.lock().ok()?;
        sessions.iter().position(|session| session.has_extension(name))
    }

    /// The first HIP worker that serves the extension `name`: its session
    /// and the free bytes of its GPUs.
    pub fn extension_host(&self, name: &str) -> Option<(usize, Vec<u64>)> {
        let sessions = self.sessions.lock().ok()?;
        sessions.iter().enumerate().find_map(|(s, session)| {
            let hip = matches!(session.hello().backend, RpcBackendKind::Hip);
            (hip && session.has_extension(name)).then(|| (s, session.hello().device_memories()))
        })
    }

    /// Run `f` with worker `session` to itself: no other call reaches that
    /// worker until `f` returns. `f` must not use this runtime (it holds
    /// the sessions).
    pub fn with_session<T>(
        &self,
        session: usize,
        f: impl FnOnce(&mut RpcSession) -> Result<T, AllpacaError>,
    ) -> Result<T, AllpacaError> {
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|_| AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into())))?;
        let session = sessions
            .get_mut(session)
            .ok_or_else(|| AllpacaError::Fatal(FatalError::InvalidConfig(format!("no rpc session {session}"))))?;
        f(session)
    }

    /// One call to the extension `name` of worker `session`.
    pub fn ext(&self, session: usize, name: &str, head: &[u8], data: &[u8]) -> Result<Vec<u8>, AllpacaError> {
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|_| AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into())))?;
        let session = sessions
            .get_mut(session)
            .ok_or_else(|| AllpacaError::Fatal(FatalError::InvalidConfig(format!("no rpc session {session}"))))?;
        let reply = session.ext(name, head, data).map_err(map_rpc)?;
        self.h2d_bytes.fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(reply)
    }

    /// Put one layer's routed experts on a HIP worker GPU as a buffer of
    /// Q2_0 slots (gate | up | down per expert); [`Self::moe_ffn`] then runs
    /// them. Worker GPUs fill in order, so consecutive layers share one.
    /// `false` when no HIP worker has room left.
    pub fn put_expert_slots(
        &self,
        layer: usize,
        slots: &[u8],
        n_experts: usize,
        dim: usize,
        ff: usize,
    ) -> Result<bool, AllpacaError> {
        let lock = || AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into()));
        let mut sessions = self.sessions.lock().map_err(|_| lock())?;
        let mut placed = self.slot_bytes.lock().map_err(|_| lock())?;
        let need = slots.len() as u64;
        let home = sessions.iter().enumerate().find_map(|(s, session)| {
            if !matches!(session.hello().backend, RpcBackendKind::Hip) {
                return None;
            }
            session.hello().device_memories().iter().enumerate().find_map(|(d, &mem)| {
                // Leave the worker ~5% and 256 MB for MoeFfn scratch.
                let room = mem.saturating_sub(mem / 20 + (256 << 20));
                let used = placed.get(&(s, d as u32)).copied().unwrap_or(0);
                (room >= used + need).then_some((s, d as u32))
            })
        });
        let Some((session, device)) = home else {
            return Ok(false);
        };
        let pool = sessions[session]
            .put_tensor(42, &[(slots.len() / n_experts.max(1)) as u32, n_experts as u32], slots, Some(device))
            .map_err(map_rpc)?;
        *placed.entry((session, device)).or_default() += need;
        self.h2d_bytes.fetch_add(need, Ordering::Relaxed);
        let mut experts = self.experts.lock().map_err(|_| lock())?;
        if experts.len() <= layer {
            experts.resize(layer + 1, None);
        }
        experts[layer] = Some(RemoteExperts {
            session,
            device,
            // Rows go as int8 blocks unless asked otherwise: a quarter of
            // the bytes, and float rows of this size have stalled on
            // retransmits.
            q8: dim % 32 == 0 && std::env::var_os("ALLPACA_Q4X_RPC_F32").is_none(),
            req: MoeFfnRequest {
                gate: pool,
                up: pool,
                down: pool,
                n_experts: n_experts as u32,
                dim: dim as u32,
                ff: ff as u32,
                f16_out: true,
            },
        });
        Ok(true)
    }

    /// Where `layer_idx` runs: a worker GPU, or the local HIP device.
    pub fn layer_home(&self, layer_idx: usize) -> LayerHome {
        let remote = self
            .remote
            .lock()
            .ok()
            .and_then(|r| r.get(layer_idx).and_then(|x| x.as_ref()).map(|l| (l.session, l.device)));
        match remote {
            Some((session, device)) => LayerHome::Remote {
                endpoint: self.endpoints.get(session).cloned().unwrap_or_default(),
                device,
            },
            None => LayerHome::Local(self.local_device(layer_idx)),
        }
    }

    /// The worker holding this layer's routed experts, when they are remote.
    /// The worker (`host:port`) and GPU holding this layer's routed experts;
    /// `None` when they are local.
    pub fn experts_home(&self, layer_idx: usize) -> Option<(String, u32)> {
        let e = self.experts.lock().ok()?;
        let ex = e.get(layer_idx)?.as_ref()?;
        Some((self.endpoints.get(ex.session).cloned().unwrap_or_default(), ex.device))
    }

    /// With expert-parallel placement: each layer's routed-expert bytes and
    /// the worker GPUs (`rpc:<host:port>/<gpu>`) a layer may be pinned to.
    /// `None` for other placements.
    pub fn expert_layout(&self) -> Option<(Vec<u64>, Vec<String>)> {
        let layout = self.expert_layout.lock().ok()?;
        (!layout.0.is_empty()).then(|| layout.clone())
    }

    /// This layer's routed experts run on an RPC worker.
    pub fn experts_remote(&self, layer_idx: usize) -> bool {
        self.experts
            .lock()
            .ok()
            .is_some_and(|e| e.get(layer_idx).is_some_and(Option::is_some))
    }

    /// Routing-weighted sum of each row's experts, computed on the worker that
    /// holds this layer's experts. `x` is `[rows][dim]` post-norm hidden
    /// states and `routes[r]` row `r`'s `(expert, weight)`. `None` when the
    /// layer's experts are local.
    pub fn moe_ffn(
        &self,
        layer_idx: usize,
        x: &[f32],
        routes: &[Vec<(usize, f32)>],
    ) -> Result<Option<Vec<f32>>, AllpacaError> {
        let home = self
            .experts
            .lock()
            .ok()
            .and_then(|e| e.get(layer_idx).copied().flatten());
        let Some(home) = home else {
            return Ok(None);
        };
        let pairs: Vec<(u32, u32, f32)> = routes
            .iter()
            .enumerate()
            .flat_map(|(r, route)| route.iter().map(move |&(e, w)| (r as u32, e as u32, w)))
            .collect();
        let out = {
            let mut sessions = self.sessions.lock().map_err(|_| {
                AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into()))
            })?;
            sessions[home.session]
                .moe_ffn(home.req, x, home.q8, &pairs)
                .map_err(map_rpc)?
        };
        self.roundtrips.fetch_add(1, Ordering::Relaxed);
        self.activation_bytes
            .fetch_add(((x.len() + out.len()) * 4) as u64, Ordering::Relaxed);
        Ok(Some(out))
    }

    fn upload_layer(
        &self,
        session_idx: usize,
        device: u32,
        cfg: &ModelConfig,
        layer: &LayerWeights,
        max_seq: usize,
    ) -> Result<RemoteLayer, AllpacaError> {
        let mut sessions = self.sessions.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into()))
        })?;
        let session = sessions.get_mut(session_idx).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc session missing".into()))
        })?;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        let ff = cfg.ff_dim;
        let alloc_f32 = |s: &mut RpcSession, n: usize| -> Result<BufId, AllpacaError> {
            s.alloc_on((n * 4) as u64, Some(device)).map_err(map_rpc)
        };
        let put = |s: &mut RpcSession, id: BufId, data: &[f32]| -> Result<(), AllpacaError> {
            s.h2d(id, 0, &f32_bytes(data)).map_err(map_rpc)
        };
        let attn_norm = alloc_f32(session, dim)?;
        put(session, attn_norm, &layer.attn_norm)?;
        let wq = alloc_f32(session, layer.wq.len())?;
        put(session, wq, &layer.wq)?;
        let wk = alloc_f32(session, layer.wk.len())?;
        put(session, wk, &layer.wk)?;
        let wv = alloc_f32(session, layer.wv.len())?;
        put(session, wv, &layer.wv)?;
        let wo = alloc_f32(session, layer.wo.len())?;
        put(session, wo, &layer.wo)?;
        let ffn_norm = alloc_f32(session, dim)?;
        put(session, ffn_norm, &layer.ffn_norm)?;
        let w_gate = alloc_f32(session, layer.w_gate.len())?;
        put(session, w_gate, &layer.w_gate)?;
        let w_up = alloc_f32(session, layer.w_up.len())?;
        put(session, w_up, &layer.w_up)?;
        let w_down = alloc_f32(session, layer.w_down.len())?;
        put(session, w_down, &layer.w_down)?;
        let x = alloc_f32(session, dim)?;
        let norm_out = alloc_f32(session, dim)?;
        let q = alloc_f32(session, dim)?;
        let k = alloc_f32(session, kv_dim)?;
        let v = alloc_f32(session, kv_dim)?;
        let attn_out = alloc_f32(session, dim)?;
        let hidden = alloc_f32(session, dim)?;
        let gate = alloc_f32(session, ff)?;
        let up = alloc_f32(session, ff)?;
        let k_cache = alloc_f32(session, max_seq * kv_dim)?;
        let v_cache = alloc_f32(session, max_seq * kv_dim)?;
        let h2d = (layer.attn_norm.len()
            + layer.wq.len()
            + layer.wk.len()
            + layer.wv.len()
            + layer.wo.len()
            + layer.ffn_norm.len()
            + layer.w_gate.len()
            + layer.w_up.len()
            + layer.w_down.len())
            * 4;
        self.h2d_bytes.fetch_add(h2d as u64, Ordering::Relaxed);
        Ok(RemoteLayer {
            session: session_idx,
            device,
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
            x,
            norm_out,
            q,
            k,
            v,
            attn_out,
            hidden,
            gate,
            up,
            k_cache,
            v_cache,
            dim,
            kv_dim,
            ff_dim: ff,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim,
            eps: cfg.eps,
            theta: cfg.rope_theta,
            max_seq,
            packed: false,
            q_len: dim,
            o_in: dim,
            q_head_dim: 0,
            gated: false,
            attn_dim: dim,
            q_proj: q,
            q_gate: q,
            q_norm: q,
            k_norm: q,
            has_q_norm: false,
            has_k_norm: false,
            n_rot: 0,
            rope_sections: [0; 4],
        })
    }

    /// Upload one dense GGUF block as packed tensors. `None` when the block is
    /// GDN, MoE, split-head, or a ggml type this worker cannot GEMM.
    fn upload_packed_layer(
        &self,
        file: &GgufFile,
        layer_idx: usize,
        session_idx: usize,
        device: u32,
        cfg: &ModelConfig,
        max_seq: usize,
    ) -> Result<Option<RemoteLayer>, AllpacaError> {
        let blk = cfg.gguf_blk(layer_idx);
        let name = |suffix: &str| format!("blk.{blk}.{suffix}");
        if cfg.headwise_attn_gate
            || file.tensor(&name("attn_q.weight")).is_none()
            || file.tensor(&name("ffn_gate.weight")).is_none()
            || file.tensor(&name("ffn_gate_exps.weight")).is_some()
            || file.tensor(&name("ffn_up_exps.weight")).is_some()
        {
            return Ok(None);
        }
        let mut sessions = self.sessions.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into()))
        })?;
        let session = sessions.get_mut(session_idx).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc session missing".into()))
        })?;
        let backend = session.hello().backend;
        let weight_names = [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ];
        let mut infos = Vec::with_capacity(weight_names.len());
        for suffix in weight_names {
            let Some(info) = file.tensor(&name(suffix)).cloned() else {
                return Ok(None);
            };
            if !worker_accepts_type(backend, info.ggml_type_id) {
                tracing::warn!(
                    layer_idx,
                    tensor = suffix,
                    ggml_type = info.ggml_type_id,
                    "rpc packed layer skipped; worker has no gemm for this ggml type"
                );
                return Ok(None);
            }
            infos.push(info);
        }
        let dim = cfg.dim;
        let Some((q_n, q_k)) = gemm_nk(&infos[0]) else {
            return Ok(None);
        };
        let Some((k_n, k_k)) = gemm_nk(&infos[1]) else {
            return Ok(None);
        };
        let Some((v_n, v_k)) = gemm_nk(&infos[2]) else {
            return Ok(None);
        };
        let Some((o_n, o_k)) = gemm_nk(&infos[3]) else {
            return Ok(None);
        };
        let Some((gate_n, gate_k)) = gemm_nk(&infos[4]) else {
            return Ok(None);
        };
        let Some((up_n, up_k)) = gemm_nk(&infos[5]) else {
            return Ok(None);
        };
        let Some((down_n, down_k)) = gemm_nk(&infos[6]) else {
            return Ok(None);
        };
        let al = crate::model::attn_layout(
            cfg,
            infos[0].n_elements() as usize,
            infos[1].n_elements() as usize,
            infos[3].n_elements() as usize,
        );
        // The same choice the local path makes: NeoX pairing (partial or
        // full, `RopePartial`) only where `neox_rot_dims` says so; otherwise
        // the standard interleaved rotation (`Rope`). Llama GGUFs set
        // rope.dimension_count == head_dim and are still standard.
        let n_rot = cfg.neox_rot_dims(layer_idx, al.head_dim).unwrap_or(0);
        if q_k != dim
            || k_k != dim
            || v_k != dim
            || o_n != dim
            || gate_k != dim
            || up_k != dim
            || up_n != gate_n
            || down_n != dim
            || down_k != gate_n
            || k_n != v_n
            || al.q_out != q_n
            || al.kv_dim != k_n
            || al.wo_k != o_k
            || al.head_dim == 0
            || al.attn_dim % al.head_dim != 0
            || k_n % al.head_dim != 0
            || (n_rot > 0 && (n_rot > al.head_dim || n_rot % 2 != 0))
            // The worker's attention always rotates Q/K.
            || cfg.rope_none
        {
            tracing::warn!(layer_idx, "rpc packed layer skipped; tensor shape is not dense attn+ffn");
            return Ok(None);
        }
        let attn_norm = file.tensor_f32(&name("attn_norm.weight"))?;
        // Qwen GGUF stores the FFN RMSNorm as `post_attention_norm`.
        let ffn_norm = if file.tensor(&name("ffn_norm.weight")).is_some() {
            file.tensor_f32(&name("ffn_norm.weight"))?
        } else {
            file.tensor_f32(&name("post_attention_norm.weight"))?
        };
        if attn_norm.len() != dim || ffn_norm.len() != dim {
            tracing::warn!(layer_idx, "rpc packed layer skipped; norm rank is not dim");
            return Ok(None);
        }
        let alloc_f32 = |s: &mut RpcSession, n: usize| -> Result<BufId, AllpacaError> {
            s.alloc_on((n * 4) as u64, Some(device)).map_err(map_rpc)
        };
        let put_f32 = |s: &mut RpcSession, id: BufId, data: &[f32]| -> Result<(), AllpacaError> {
            s.h2d(id, 0, &f32_bytes(data)).map_err(map_rpc)
        };
        let attn_norm_id = alloc_f32(session, dim)?;
        put_f32(session, attn_norm_id, &attn_norm)?;
        let ffn_norm_id = alloc_f32(session, dim)?;
        put_f32(session, ffn_norm_id, &ffn_norm)?;
        let q_norm_name = name("attn_q_norm.weight");
        let k_norm_name = name("attn_k_norm.weight");
        let q_norm_f = if file.tensor(&q_norm_name).is_some() {
            Some(file.tensor_f32(&q_norm_name)?)
        } else {
            None
        };
        let k_norm_f = if file.tensor(&k_norm_name).is_some() {
            Some(file.tensor_f32(&k_norm_name)?)
        } else {
            None
        };
        if q_norm_f.as_ref().is_some_and(|w| w.len() != al.head_dim)
            || k_norm_f.as_ref().is_some_and(|w| w.len() != al.head_dim)
        {
            tracing::warn!(layer_idx, "rpc packed layer skipped; q/k norm is not head_dim");
            return Ok(None);
        }
        let put_w = |s: &mut RpcSession,
                     info: &allpaca_gguf::TensorInfo|
         -> Result<(BufId, usize), AllpacaError> {
            let bytes = file.tensor_bytes(&info.name)?;
            let mut dims = Vec::with_capacity(info.dims.len());
            for d in &info.dims {
                dims.push(u32::try_from(*d).map_err(|_| {
                    AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "tensor {} dim does not fit u32",
                        info.name
                    )))
                })?);
            }
            let id = s
                .put_tensor(info.ggml_type_id, &dims, bytes, Some(device))
                .map_err(map_rpc)?;
            Ok((id, bytes.len()))
        };
        let (wq, q_bytes) = put_w(session, &infos[0])?;
        let (wk, k_bytes) = put_w(session, &infos[1])?;
        let (wv, v_bytes) = put_w(session, &infos[2])?;
        let (wo, o_bytes) = put_w(session, &infos[3])?;
        let (w_gate, gate_bytes) = put_w(session, &infos[4])?;
        let (w_up, up_bytes) = put_w(session, &infos[5])?;
        let (w_down, down_bytes) = put_w(session, &infos[6])?;
        let packed_bytes = q_bytes + k_bytes + v_bytes + o_bytes + gate_bytes + up_bytes + down_bytes;
        let x = alloc_f32(session, dim)?;
        let norm_out = alloc_f32(session, dim)?;
        let q_proj = alloc_f32(session, q_n)?;
        let q = alloc_f32(session, if al.gated { al.attn_dim } else { q_n })?;
        let q_gate = alloc_f32(session, if al.gated { al.attn_dim } else { 1 })?;
        let q_norm = if let Some(w) = &q_norm_f {
            let id = alloc_f32(session, w.len())?;
            put_f32(session, id, w)?;
            id
        } else {
            q
        };
        let k = alloc_f32(session, k_n)?;
        let k_norm = if let Some(w) = &k_norm_f {
            let id = alloc_f32(session, w.len())?;
            put_f32(session, id, w)?;
            id
        } else {
            k
        };
        let v = alloc_f32(session, k_n)?;
        let attn_out = alloc_f32(session, o_k)?;
        let hidden = alloc_f32(session, dim)?;
        let gate = alloc_f32(session, gate_n)?;
        let up = alloc_f32(session, gate_n)?;
        let k_cache = alloc_f32(session, max_seq * k_n)?;
        let v_cache = alloc_f32(session, max_seq * k_n)?;
        let norm_bytes = (attn_norm.len()
            + ffn_norm.len()
            + q_norm_f.as_ref().map(|w| w.len()).unwrap_or(0)
            + k_norm_f.as_ref().map(|w| w.len()).unwrap_or(0))
            * 4;
        self.h2d_bytes
            .fetch_add((packed_bytes + norm_bytes) as u64, Ordering::Relaxed);
        tracing::info!(
            layer_idx,
            blk,
            ggml_type = infos[0].ggml_type_id,
            "rpc packed dense layer uploaded"
        );
        Ok(Some(RemoteLayer {
            session: session_idx,
            device,
            attn_norm: attn_norm_id,
            wq,
            wk,
            wv,
            wo,
            ffn_norm: ffn_norm_id,
            w_gate,
            w_up,
            w_down,
            x,
            norm_out,
            q,
            k,
            v,
            attn_out,
            hidden,
            gate,
            up,
            k_cache,
            v_cache,
            dim,
            kv_dim: k_n,
            ff_dim: gate_n,
            n_heads: al.n_heads,
            n_kv_heads: al.n_kv_heads,
            head_dim: al.head_dim,
            eps: cfg.eps,
            theta: cfg.rope_theta_for_layer(layer_idx),
            max_seq,
            packed: true,
            q_len: q_n,
            o_in: o_k,
            q_head_dim: al.head_dim,
            gated: al.gated,
            attn_dim: al.attn_dim,
            q_proj,
            q_gate,
            q_norm,
            k_norm,
            has_q_norm: q_norm_f.is_some(),
            has_k_norm: k_norm_f.is_some(),
            n_rot,
            rope_sections: cfg.rope_sections,
        }))
    }

    pub fn is_remote(&self, layer_idx: usize) -> bool {
        self.remote_session(layer_idx).is_some()
    }

    /// First layer of a contiguous same-worker remote run.
    pub fn is_span_start(&self, layer_idx: usize) -> bool {
        let Some(session) = self.remote_session(layer_idx) else {
            return false;
        };
        self.remote_session(layer_idx.wrapping_sub(1)) != Some(session)
    }

    fn remote_session(&self, layer_idx: usize) -> Option<usize> {
        self.remote
            .lock()
            .ok()
            .and_then(|r| r.get(layer_idx).and_then(|x| x.as_ref()).map(|l| l.session))
    }

    /// Rows per batched span call (`rpc.prefill_rows`, 0 = one token per call).
    pub fn set_prefill_rows(&self, rows: usize) {
        self.prefill_rows.store(rows, Ordering::Relaxed);
    }

    /// The contiguous same-worker run of remote layers starting at `layer_idx`.
    fn span_from(&self, layer_idx: usize) -> Result<Vec<RemoteLayer>, AllpacaError> {
        let guard = self.remote.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc remote lock".into()))
        })?;
        let session = guard
            .get(layer_idx)
            .and_then(|layer| layer.as_ref())
            .map(|layer| layer.session)
            .ok_or_else(|| AllpacaError::Fatal(FatalError::InvalidConfig("rpc layer missing".into())))?;
        let mut span = Vec::new();
        for idx in layer_idx..guard.len() {
            match guard.get(idx).and_then(|layer| layer.as_ref()) {
                Some(layer) if layer.session == session => span.push(*layer),
                _ => break,
            }
        }
        Ok(span)
    }

    /// Run the remote span starting at `layer_idx` over `x`, which holds
    /// `x.len() / dim` prompt rows at consecutive positions from `start_pos`.
    /// Rows go to the worker `rpc.prefill_rows` at a time, one round trip per
    /// chunk; with `prefill_rows = 0`, or a layer the batched list cannot
    /// express, it falls back to one round trip per row. A call on any other
    /// layer of the span is a no-op.
    pub fn run_rows(
        &self,
        layer_idx: usize,
        start_pos: usize,
        dim: usize,
        x: &mut [f32],
    ) -> Result<(), AllpacaError> {
        if !self.is_span_start(layer_idx) || dim == 0 {
            return Ok(());
        }
        let n_rows = x.len() / dim;
        let cap = self.prefill_rows.load(Ordering::Relaxed);
        let span = self.span_from(layer_idx)?;
        if n_rows <= 1 || cap == 0 || span.iter().any(|l| !rows_supported(l)) {
            for (t, row) in x.chunks_exact_mut(dim).enumerate() {
                self.run_layer(layer_idx, start_pos + t, row)?;
            }
            return Ok(());
        }
        if let Some(layer) = span.iter().find(|l| start_pos + n_rows > l.max_seq) {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "context position {} is past the remote layers' KV capacity of {} tokens; raise rpc.max_seq (0 sizes it from kv.vram_bytes)",
                start_pos + n_rows - 1,
                layer.max_seq
            ))));
        }
        let mut sessions = self.sessions.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into()))
        })?;
        let session_idx = span[0].session;
        let session = sessions.get_mut(session_idx).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc session missing".into()))
        })?;
        let mut scratch = Vec::with_capacity(span.len());
        for layer in &span {
            scratch.push(self.row_scratch_for(session, session_idx, layer.device, cap)?);
        }
        for (c, chunk) in x.chunks_mut(cap * dim).enumerate() {
            let rows = chunk.len() / dim;
            let pos = start_pos + c * cap;
            let xs: Vec<BufId> = scratch.iter().map(|s| s.x).collect();
            let lists: Vec<LaunchList> = span
                .iter()
                .zip(&scratch)
                .map(|(layer, s)| layer_launch_rows(layer, s, pos, rows))
                .collect();
            let activation = f32_bytes(chunk);
            self.activation_bytes
                .fetch_add(activation.len() as u64, Ordering::Relaxed);
            let out = session.run_span(&xs, &lists, &activation).map_err(map_rpc)?;
            self.activation_bytes
                .fetch_add(out.len() as u64, Ordering::Relaxed);
            self.roundtrips.fetch_add(1, Ordering::Relaxed);
            if out.len() != chunk.len() * 4 {
                return Err(AllpacaError::Fatal(FatalError::Io(format!(
                    "rpc run_span returned {} bytes for {} rows",
                    out.len(),
                    rows
                ))));
            }
            from_f32_bytes(&out, chunk)?;
        }
        Ok(())
    }

    /// The row scratch on `device` of session `session_idx`, allocated on
    /// first use for the widest remote layer placed there.
    fn row_scratch_for(
        &self,
        session: &mut RpcSession,
        session_idx: usize,
        device: u32,
        rows: usize,
    ) -> Result<RowScratch, AllpacaError> {
        let mut map = self.row_scratch.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc row scratch lock".into()))
        })?;
        if let Some(s) = map.get(&(session_idx, device)).filter(|s| s.rows == rows) {
            return Ok(*s);
        }
        if let Some(old) = map.remove(&(session_idx, device)) {
            for id in [
                old.x, old.norm_out, old.q_proj, old.q, old.q_gate, old.k, old.v, old.attn_out,
                old.hidden, old.gate, old.up,
            ] {
                let _ = session.free(id);
            }
        }
        let layers: Vec<RemoteLayer> = self
            .remote
            .lock()
            .map_err(|_| AllpacaError::Fatal(FatalError::InvalidConfig("rpc remote lock".into())))?
            .iter()
            .flatten()
            .filter(|l| l.session == session_idx && l.device == device)
            .copied()
            .collect();
        let widest = |f: &dyn Fn(&RemoteLayer) -> usize| layers.iter().map(f).max().unwrap_or(1).max(1);
        let mut alloc = |width: usize| -> Result<BufId, AllpacaError> {
            session
                .alloc_on((width * rows * 4) as u64, Some(device))
                .map_err(map_rpc)
        };
        let s = RowScratch {
            rows,
            x: alloc(widest(&|l| l.dim))?,
            norm_out: alloc(widest(&|l| l.dim))?,
            q_proj: alloc(widest(&|l| l.q_len))?,
            q: alloc(widest(&|l| if l.gated { l.attn_dim } else { l.q_len }))?,
            q_gate: alloc(widest(&|l| if l.gated { l.attn_dim } else { 1 }))?,
            k: alloc(widest(&|l| l.kv_dim))?,
            v: alloc(widest(&|l| l.kv_dim))?,
            attn_out: alloc(widest(&|l| l.o_in))?,
            hidden: alloc(widest(&|l| l.dim))?,
            gate: alloc(widest(&|l| l.ff_dim))?,
            up: alloc(widest(&|l| l.ff_dim))?,
        };
        tracing::info!(session = session_idx, device, rows, "rpc prefill row scratch allocated");
        map.insert((session_idx, device), s);
        Ok(s)
    }

    /// Run one remote layer. A call on the first layer of a contiguous same-worker
    /// run executes that whole span in one RPC. Later layers in the span are no-ops;
    /// `x` already holds the span output.
    pub fn run_layer(
        &self,
        layer_idx: usize,
        pos: usize,
        x: &mut [f32],
    ) -> Result<(), AllpacaError> {
        if !self.is_span_start(layer_idx) {
            return Ok(());
        }
        let span = {
            let guard = self.remote.lock().map_err(|_| {
                AllpacaError::Fatal(FatalError::InvalidConfig("rpc remote lock".into()))
            })?;
            let session = guard
                .get(layer_idx)
                .and_then(|layer| layer.as_ref())
                .map(|layer| layer.session)
                .ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::InvalidConfig("rpc layer missing".into()))
                })?;
            let mut span = Vec::new();
            for idx in layer_idx..guard.len() {
                match guard.get(idx).and_then(|layer| layer.as_ref()) {
                    Some(layer) if layer.session == session => span.push(*layer),
                    _ => break,
                }
            }
            span
        };
        if span.is_empty() {
            return Ok(());
        }
        let mut lists = Vec::with_capacity(span.len());
        let mut xs = Vec::with_capacity(span.len());
        for layer in &span {
            if pos >= layer.max_seq {
                return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                    "context position {pos} is past the remote layers' KV capacity of {} tokens; raise rpc.max_seq (0 sizes it from kv.vram_bytes)",
                    layer.max_seq
                ))));
            }
            xs.push(layer.x);
            lists.push(layer_launch(layer, pos));
        }
        let mut sessions = self.sessions.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into()))
        })?;
        let session = sessions.get_mut(span[0].session).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc session missing".into()))
        })?;
        let _ = ComputeDev::Rpc(session);
        let activation = f32_bytes(x);
        self.activation_bytes
            .fetch_add(activation.len() as u64, Ordering::Relaxed);
        let out = session
            .run_span(&xs, &lists, &activation)
            .map_err(map_rpc)?;
        self.activation_bytes
            .fetch_add(out.len() as u64, Ordering::Relaxed);
        self.roundtrips.fetch_add(1, Ordering::Relaxed);
        from_f32_bytes(&out, x)?;
        Ok(())
    }
}

/// Whether [`layer_launch_rows`] can express this layer: full-head RoPE
/// rotates Q and K with one head size.
fn rows_supported(layer: &RemoteLayer) -> bool {
    layer.n_rot > 0 || layer.q_head_dim == 0 || layer.q_head_dim == layer.head_dim
}

/// [`layer_launch`] for `rows` prompt rows at positions `pos..pos + rows`,
/// in the device's row scratch. Each row's K/V lands in the layer's cache at
/// its position, and attention is causal over the cache, so the result
/// matches running the rows one at a time.
fn layer_launch_rows(layer: &RemoteLayer, s: &RowScratch, pos: usize, rows: usize) -> LaunchList {
    let r = rows as i32;
    let packed = layer.packed;
    let q_dst = if layer.gated { s.q_proj } else { s.q };
    let kv_row = layer.kv_dim * 4;
    let mut items = vec![
        rms_rows(s.x, layer.attn_norm, s.norm_out, rows, layer.dim, layer.eps),
        gemm(packed, s.norm_out, layer.wq, q_dst, r, layer.q_len as i32, layer.dim as i32),
        gemm(packed, s.norm_out, layer.wk, s.k, r, layer.kv_dim as i32, layer.dim as i32),
        gemm(packed, s.norm_out, layer.wv, s.v, r, layer.kv_dim as i32, layer.dim as i32),
    ];
    if layer.gated {
        items.push(LaunchItem {
            op: KernelOp::DeinterleaveQGate,
            bufs: vec![s.q_proj, s.q, s.q_gate],
            i32s: vec![(layer.n_heads * rows) as i32, layer.head_dim as i32],
            f32s: vec![],
        });
    }
    if layer.has_q_norm {
        items.push(LaunchItem {
            op: KernelOp::RmsNormHeads,
            bufs: vec![s.q, layer.q_norm],
            i32s: vec![(layer.n_heads * rows) as i32, layer.head_dim as i32],
            f32s: vec![layer.eps],
        });
    }
    if layer.has_k_norm {
        items.push(LaunchItem {
            op: KernelOp::RmsNormHeads,
            bufs: vec![s.k, layer.k_norm],
            i32s: vec![(layer.n_kv_heads * rows) as i32, layer.head_dim as i32],
            f32s: vec![layer.eps],
        });
    }
    let rope_i32 = vec![
        r,
        layer.n_heads as i32,
        layer.n_kv_heads as i32,
        layer.head_dim as i32,
        pos as i32,
    ];
    if layer.n_rot > 0 {
        let mut i32s = rope_i32;
        i32s.push(layer.n_rot as i32);
        items.push(LaunchItem {
            op: KernelOp::RopePartialRows,
            bufs: vec![s.q, s.k],
            i32s,
            f32s: vec![
                layer.theta,
                layer.rope_sections[0] as f32,
                layer.rope_sections[1] as f32,
                layer.rope_sections[2] as f32,
                layer.rope_sections[3] as f32,
            ],
        });
    } else {
        items.push(LaunchItem {
            op: KernelOp::RopeRows,
            bufs: vec![s.q, s.k],
            i32s: rope_i32,
            f32s: vec![layer.theta],
        });
    }
    items.push(memcpy_at(layer.k_cache, s.k, (pos * kv_row) as i32, (rows * kv_row) as i32));
    items.push(memcpy_at(layer.v_cache, s.v, (pos * kv_row) as i32, (rows * kv_row) as i32));
    let mut attention = attn(
        s.q,
        layer.k_cache,
        layer.v_cache,
        s.attn_out,
        pos + rows,
        layer.n_heads,
        layer.n_kv_heads,
        layer.head_dim,
    );
    attention.i32s[0] = r;
    items.push(attention);
    if layer.gated {
        items.push(LaunchItem {
            op: KernelOp::SigmoidMul,
            bufs: vec![s.attn_out, s.q_gate],
            i32s: vec![(layer.attn_dim * rows) as i32],
            f32s: vec![],
        });
    }
    items.push(gemm(packed, s.attn_out, layer.wo, s.hidden, r, layer.dim as i32, layer.o_in as i32));
    items.push(residual_n(s.x, s.hidden, rows * layer.dim));
    items.push(rms_rows(s.x, layer.ffn_norm, s.norm_out, rows, layer.dim, layer.eps));
    items.push(gemm(packed, s.norm_out, layer.w_gate, s.gate, r, layer.ff_dim as i32, layer.dim as i32));
    items.push(gemm(packed, s.norm_out, layer.w_up, s.up, r, layer.ff_dim as i32, layer.dim as i32));
    items.push(LaunchItem {
        op: KernelOp::SiluMul,
        bufs: vec![s.gate, s.up],
        i32s: vec![(rows * layer.ff_dim) as i32],
        f32s: vec![],
    });
    items.push(gemm(packed, s.gate, layer.w_down, s.hidden, r, layer.dim as i32, layer.ff_dim as i32));
    items.push(residual_n(s.x, s.hidden, rows * layer.dim));
    LaunchList { items }
}

fn rms_rows(x: BufId, w: BufId, out: BufId, rows: usize, dim: usize, eps: f32) -> LaunchItem {
    LaunchItem {
        op: KernelOp::RmsNormRows,
        bufs: vec![x, w, out],
        i32s: vec![rows as i32, dim as i32],
        f32s: vec![eps],
    }
}

fn residual_n(dst: BufId, src: BufId, n: usize) -> LaunchItem {
    LaunchItem {
        op: KernelOp::ResidualAdd,
        bufs: vec![dst, src],
        i32s: vec![n as i32],
        f32s: vec![],
    }
}

fn layer_launch(layer: &RemoteLayer, pos: usize) -> LaunchList {
    let n_kv = pos + 1;
    let kv_off = (pos * layer.kv_dim * 4) as i32;
    let kv_bytes = (layer.kv_dim * 4) as i32;
    let q_head = if layer.q_head_dim == 0 {
        layer.head_dim
    } else {
        layer.q_head_dim
    };
    let packed = layer.packed;
    let q_dst = if layer.gated { layer.q_proj } else { layer.q };
    let mut items = vec![
        rms(layer.x, layer.attn_norm, layer.norm_out, layer.eps),
        gemm(
            packed,
            layer.norm_out,
            layer.wq,
            q_dst,
            1,
            layer.q_len as i32,
            layer.dim as i32,
        ),
        gemm(
            packed,
            layer.norm_out,
            layer.wk,
            layer.k,
            1,
            layer.kv_dim as i32,
            layer.dim as i32,
        ),
        gemm(
            packed,
            layer.norm_out,
            layer.wv,
            layer.v,
            1,
            layer.kv_dim as i32,
            layer.dim as i32,
        ),
    ];
    if layer.gated {
        items.push(LaunchItem {
            op: KernelOp::DeinterleaveQGate,
            bufs: vec![layer.q_proj, layer.q, layer.q_gate],
            i32s: vec![layer.n_heads as i32, layer.head_dim as i32],
            f32s: vec![],
        });
    }
    if layer.has_q_norm {
        items.push(LaunchItem {
            op: KernelOp::RmsNormHeads,
            bufs: vec![layer.q, layer.q_norm],
            i32s: vec![layer.n_heads as i32, layer.head_dim as i32],
            f32s: vec![layer.eps],
        });
    }
    if layer.has_k_norm {
        items.push(LaunchItem {
            op: KernelOp::RmsNormHeads,
            bufs: vec![layer.k, layer.k_norm],
            i32s: vec![layer.n_kv_heads as i32, layer.head_dim as i32],
            f32s: vec![layer.eps],
        });
    }
    if layer.n_rot > 0 {
        items.push(LaunchItem {
            op: KernelOp::RopePartial,
            bufs: vec![layer.q, layer.k],
            i32s: vec![
                layer.n_heads as i32,
                layer.n_kv_heads as i32,
                layer.head_dim as i32,
                pos as i32,
                layer.n_rot as i32,
            ],
            f32s: vec![
                layer.theta,
                layer.rope_sections[0] as f32,
                layer.rope_sections[1] as f32,
                layer.rope_sections[2] as f32,
                layer.rope_sections[3] as f32,
            ],
        });
    } else {
        items.push(rope_op(layer.q, layer.n_heads, q_head, pos, layer.theta));
        items.push(rope_op(
            layer.k,
            layer.n_kv_heads,
            layer.head_dim,
            pos,
            layer.theta,
        ));
    }
    items.push(memcpy_at(layer.k_cache, layer.k, kv_off, kv_bytes));
    items.push(memcpy_at(layer.v_cache, layer.v, kv_off, kv_bytes));
    items.push(attn(
        layer.q,
        layer.k_cache,
        layer.v_cache,
        layer.attn_out,
        n_kv,
        layer.n_heads,
        layer.n_kv_heads,
        layer.head_dim,
    ));
    if layer.gated {
        items.push(LaunchItem {
            op: KernelOp::SigmoidMul,
            bufs: vec![layer.attn_out, layer.q_gate],
            i32s: vec![layer.attn_dim as i32],
            f32s: vec![],
        });
    }
    items.push(gemm(
        packed,
        layer.attn_out,
        layer.wo,
        layer.hidden,
        1,
        layer.dim as i32,
        layer.o_in as i32,
    ));
    items.push(residual(layer.x, layer.hidden));
    items.push(rms(layer.x, layer.ffn_norm, layer.norm_out, layer.eps));
    items.push(gemm(
        packed,
        layer.norm_out,
        layer.w_gate,
        layer.gate,
        1,
        layer.ff_dim as i32,
        layer.dim as i32,
    ));
    items.push(gemm(
        packed,
        layer.norm_out,
        layer.w_up,
        layer.up,
        1,
        layer.ff_dim as i32,
        layer.dim as i32,
    ));
    items.push(silu(layer.gate, layer.up));
    items.push(gemm(
        packed,
        layer.gate,
        layer.w_down,
        layer.hidden,
        1,
        layer.dim as i32,
        layer.ff_dim as i32,
    ));
    items.push(residual(layer.x, layer.hidden));
    LaunchList { items }
}

fn gemm_nk(info: &allpaca_gguf::TensorInfo) -> Option<(usize, usize)> {
    let k = *info.dims.first()? as usize;
    if k == 0 {
        return None;
    }
    let n_el = info.n_elements() as usize;
    if n_el % k != 0 {
        return None;
    }
    Some((n_el / k, k))
}

fn worker_accepts_type(backend: RpcBackendKind, ty: u32) -> bool {
    let hip = matches!(ty, 0 | 1 | 8 | 12 | 13 | 14 | 30 | 34 | 35 | 39 | 100 | 101);
    match backend {
        // 200 = MoESQ paired-4:8 experts; only the HIP worker has kernels for it.
        RpcBackendKind::Hip => hip || ty == 200,
        RpcBackendKind::Cpu => hip || matches!(ty, 2 | 3 | 6 | 7 | 9 | 10 | 11 | 20 | 23),
        // F32, Q8_0 and the int8 K-quant GEMV; ROCmFP4 and the rest stay off
        // CUDA workers.
        RpcBackendKind::Cuda => matches!(ty, 0 | 8 | 12 | 13 | 14),
    }
}

fn has_stacked_experts(file: &GgufFile) -> bool {
    file.tensors
        .iter()
        .any(|t| crate::model::is_stacked_expert_tensor(&t.name))
}

fn is_dense_attn_ffn(layer: &LayerWeights) -> bool {
    !layer.wq.is_empty() && !layer.w_gate.is_empty() && layer.experts.is_empty()
}

pub(crate) fn map_rpc(e: allpaca_rpc::RpcError) -> AllpacaError {
    AllpacaError::Fatal(FatalError::Io(e.to_string()))
}

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn from_f32_bytes(bytes: &[u8], dst: &mut [f32]) -> Result<(), AllpacaError> {
    let n = dst.len().min(bytes.len() / 4);
    for i in 0..n {
        dst[i] = f32::from_le_bytes(
            bytes[i * 4..i * 4 + 4]
                .try_into()
                .map_err(|_| AllpacaError::Fatal(FatalError::Io("rpc d2h".into())))?,
        );
    }
    Ok(())
}

fn rms(x: BufId, w: BufId, out: BufId, eps: f32) -> LaunchItem {
    LaunchItem {
        op: KernelOp::RmsNorm,
        bufs: vec![x, w, out],
        i32s: vec![],
        f32s: vec![eps],
    }
}

fn gemm(packed: bool, a: BufId, b: BufId, c: BufId, m: i32, n: i32, k: i32) -> LaunchItem {
    LaunchItem {
        op: if packed {
            KernelOp::GemmNtQuant
        } else {
            KernelOp::GemmNt
        },
        bufs: vec![a, b, c],
        i32s: vec![m, n, k],
        f32s: vec![],
    }
}

fn rope_op(x: BufId, n_heads: usize, head_dim: usize, pos: usize, theta: f32) -> LaunchItem {
    LaunchItem {
        op: KernelOp::Rope,
        bufs: vec![x],
        i32s: vec![n_heads as i32, head_dim as i32, pos as i32],
        f32s: vec![theta],
    }
}

fn attn(
    q: BufId,
    k: BufId,
    v: BufId,
    out: BufId,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> LaunchItem {
    LaunchItem {
        op: KernelOp::Attention,
        bufs: vec![q, k, v, out],
        i32s: vec![
            1,
            n_kv as i32,
            n_heads as i32,
            n_kv_heads as i32,
            head_dim as i32,
            1,
        ],
        f32s: vec![],
    }
}

fn residual(dst: BufId, src: BufId) -> LaunchItem {
    LaunchItem {
        op: KernelOp::ResidualAdd,
        bufs: vec![dst, src],
        i32s: vec![],
        f32s: vec![],
    }
}

fn silu(gate: BufId, up: BufId) -> LaunchItem {
    LaunchItem {
        op: KernelOp::SiluMul,
        bufs: vec![gate, up],
        i32s: vec![],
        f32s: vec![],
    }
}

fn memcpy_at(dst: BufId, src: BufId, dst_off: i32, nbytes: i32) -> LaunchItem {
    LaunchItem {
        op: KernelOp::Memcpy,
        bufs: vec![dst, src],
        i32s: vec![dst_off, nbytes],
        f32s: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::pinned_budget;

    #[test]
    fn pinned_budget_overrides_only_set_slots() {
        assert_eq!(pinned_budget(&[], 0), None);
        assert_eq!(pinned_budget(&[8332, 0], 0), Some(8332 << 20));
        assert_eq!(pinned_budget(&[8332, 0], 1), None);
        assert_eq!(pinned_budget(&[8332], 1), None);
    }
}
