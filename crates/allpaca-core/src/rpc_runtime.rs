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
    plan: Mutex<LayerPlan>,
    remote: Mutex<Vec<Option<RemoteLayer>>>,
    /// Expert-parallel MoE: layers whose routed experts live on a worker.
    experts: Mutex<Vec<Option<RemoteExperts>>>,
    /// HIP device for each local layer. `-1` means the layer was placed on RPC.
    layer_device: Mutex<Vec<i32>>,
    /// Second local GPU: device id, workspace, and the layers that live there.
    local_peer: Mutex<Option<(i32, Arc<allpaca_hip::HipWorkspace>, Arc<ResidentModel>)>>,
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

/// One layer's routed experts on a worker GPU (`MoeFfn`).
#[derive(Clone, Copy)]
struct RemoteExperts {
    session: usize,
    req: MoeFfnRequest,
    /// Send rows as int8 blocks: the worker's fused K-quant kernels quantize
    /// to exactly that, so it is lossless and a quarter of the bytes.
    q8: bool,
}

#[derive(Clone, Copy)]
struct RemoteLayer {
    session: usize,
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
fn plan_local_by_bytes(
    model: &TinyModel,
    file: &GgufFile,
    local_ids: &[i32],
    is_full_attn: &[bool],
    kv_vram_bytes: u64,
    draft_bytes: u64,
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
        tracing::info!(
            hip = dev,
            free_mib = free >> 20,
            total_mib = total >> 20,
            budget_mib = budget >> 20,
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
            sessions.push(session);
        }
        Ok(Self {
            sessions: Mutex::new(sessions),
            plan: Mutex::new(LayerPlan::all_local(0)),
            remote: Mutex::new(Vec::new()),
            experts: Mutex::new(Vec::new()),
            layer_device: Mutex::new(Vec::new()),
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
                plan_local_by_bytes(model, file, &local_ids, &attn, kv_vram_bytes, draft_bytes)
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
            tracing::info!(
                hip = dev,
                free_mib = free >> 20,
                total_mib = total >> 20,
                budget_mib = budget >> 20,
                "expert-parallel: local GPU budget"
            );
            budgets.push(budget);
        }
        let plan = crate::layer_split::plan_expert_parallel(&body, &experts, &budgets);
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
        let mut remote_experts = vec![None; n];
        let mut remote_bytes = 0u64;
        for l in 0..n {
            if plan.experts_local[l] || experts[l] == 0 {
                continue;
            }
            let pick = (0..owners.len())
                .find(|&o| worker_free[o] >= experts[l])
                .ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                        "expert-parallel: layer {l} experts ({} MiB) fit no local or RPC GPU",
                        experts[l] >> 20
                    )))
                })?;
            worker_free[pick] -= experts[l];
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
        let kquant = ["gate", "up", "down"].iter().all(|part| {
            file.tensor(&format!("blk.{blk}.ffn_{part}_exps.weight"))
                .is_some_and(|t| matches!(t.ggml_type_id, 12..=14))
        });
        let fused = backend == RpcBackendKind::Hip && cfg.dim % 256 == 0 && cfg.ff_dim % 256 == 0;
        Ok(RemoteExperts {
            q8: kquant && fused,
            session,
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
        let n_rot = cfg.rope_dims_for_layer(layer_idx);
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
                    "rpc kv pos {pos} >= max_seq {}",
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
        RpcBackendKind::Hip => hip,
        RpcBackendKind::Cpu => hip || matches!(ty, 2 | 3 | 6 | 7 | 9 | 10 | 11 | 20 | 23),
        RpcBackendKind::Cuda => false,
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

fn map_rpc(e: allpaca_rpc::RpcError) -> AllpacaError {
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
