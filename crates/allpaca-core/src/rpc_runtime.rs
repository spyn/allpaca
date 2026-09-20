use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use allpaca_rpc::{BufId, KernelOp, LaunchItem, LaunchList, RpcBackendKind, RpcSession};
use allpaca_types::{AllpacaError, DeviceKind, FatalError, RPC_DEVICE_ID_BASE};

use crate::compute::ComputeDev;
use crate::layer_split::{plan_layer_owners, LayerPlan};
use crate::model::{LayerWeights, ModelConfig, TinyModel};

/// Connected RPC workers plus the layer assignment used at decode.
pub struct RpcRuntime {
    sessions: Mutex<Vec<RpcSession>>,
    plan: Mutex<LayerPlan>,
    remote: Mutex<Vec<Option<RemoteLayer>>>,
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
            roundtrips: AtomicU64::new(0),
            activation_bytes: AtomicU64::new(0),
            h2d_bytes: AtomicU64::new(0),
        })
    }

    pub fn apply_hello_to_inventory(
        &self,
        devices: &mut [allpaca_types::DeviceInfo],
    ) -> Result<(), AllpacaError> {
        let sessions = self.sessions.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into()))
        })?;
        for (i, session) in sessions.iter().enumerate() {
            let id = RPC_DEVICE_ID_BASE + i as i32;
            if let Some(dev) = devices.iter_mut().find(|d| d.id == id) {
                let hello = session.hello();
                dev.memory_bytes = hello.memory_bytes;
                dev.kind = match hello.backend {
                    RpcBackendKind::Hip => DeviceKind::HipGpu,
                    RpcBackendKind::Cuda => DeviceKind::CudaGpu,
                    RpcBackendKind::Cpu => DeviceKind::Cpu,
                };
            }
        }
        Ok(())
    }

    pub fn plan_has_remote(&self) -> bool {
        self.plan.lock().map(|p| p.has_remote()).unwrap_or(false)
    }

    pub fn session_count(&self) -> usize {
        self.sessions.lock().map(|s| s.len()).unwrap_or(0)
    }

    /// Split devices: local target (index 0) then each RPC session.
    pub fn prepare(
        &self,
        model: &TinyModel,
        local_bytes: u64,
        tensor_split: Option<&[f64]>,
        max_seq: usize,
    ) -> Result<(), AllpacaError> {
        let mut device_bytes = vec![local_bytes.max(1)];
        {
            let sessions = self.sessions.lock().map_err(|_| {
                AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into()))
            })?;
            for s in sessions.iter() {
                device_bytes.push(s.hello().memory_bytes.max(1));
            }
        }
        let plan = plan_layer_owners(model.config.n_layers, &device_bytes, tensor_split)?;
        let mut remote = vec![None; model.layers.len()];
        let cfg = &model.config;
        let max_seq = max_seq.max(1);
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            let owner = plan.owner(layer_idx);
            if owner == 0 {
                continue;
            }
            if !is_dense_attn_ffn(layer) {
                tracing::warn!(layer_idx, "rpc owner skipped; layer is not dense attn+ffn");
                continue;
            }
            let session_idx = owner - 1;
            let uploaded = self.upload_layer(session_idx, cfg, layer, max_seq)?;
            remote[layer_idx] = Some(uploaded);
        }
        *self.plan.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc plan lock".into()))
        })? = plan;
        *self.remote.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc remote lock".into()))
        })? = remote;
        Ok(())
    }

    fn upload_layer(
        &self,
        session_idx: usize,
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
            s.alloc((n * 4) as u64).map_err(map_rpc)
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
        })
    }

    pub fn is_remote(&self, layer_idx: usize) -> bool {
        self.remote
            .lock()
            .ok()
            .and_then(|r| r.get(layer_idx).and_then(|x| x.as_ref()).map(|_| ()))
            .is_some()
    }

    /// Run one remote layer; hidden state `x` is host-side f32.
    pub fn run_layer(
        &self,
        layer_idx: usize,
        pos: usize,
        x: &mut [f32],
    ) -> Result<(), AllpacaError> {
        let guard = self.remote.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc remote lock".into()))
        })?;
        let Some(Some(layer)) = guard.get(layer_idx) else {
            return Ok(());
        };
        if pos >= layer.max_seq {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "rpc kv pos {pos} >= max_seq {}",
                layer.max_seq
            ))));
        }
        let layer = RemoteLayer {
            session: layer.session,
            attn_norm: layer.attn_norm,
            wq: layer.wq,
            wk: layer.wk,
            wv: layer.wv,
            wo: layer.wo,
            ffn_norm: layer.ffn_norm,
            w_gate: layer.w_gate,
            w_up: layer.w_up,
            w_down: layer.w_down,
            x: layer.x,
            norm_out: layer.norm_out,
            q: layer.q,
            k: layer.k,
            v: layer.v,
            attn_out: layer.attn_out,
            hidden: layer.hidden,
            gate: layer.gate,
            up: layer.up,
            k_cache: layer.k_cache,
            v_cache: layer.v_cache,
            dim: layer.dim,
            kv_dim: layer.kv_dim,
            ff_dim: layer.ff_dim,
            n_heads: layer.n_heads,
            n_kv_heads: layer.n_kv_heads,
            head_dim: layer.head_dim,
            eps: layer.eps,
            theta: layer.theta,
            max_seq: layer.max_seq,
        };
        drop(guard);
        let mut sessions = self.sessions.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc session lock".into()))
        })?;
        let session = sessions.get_mut(layer.session).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig("rpc session missing".into()))
        })?;
        let _ = ComputeDev::Rpc(session);
        session.h2d(layer.x, 0, &f32_bytes(x)).map_err(map_rpc)?;
        self.activation_bytes
            .fetch_add((x.len() * 4) as u64, Ordering::Relaxed);
        let n_kv = pos + 1;
        let kv_off = (pos * layer.kv_dim * 4) as i32;
        let kv_bytes = (layer.kv_dim * 4) as i32;
        let list = LaunchList {
            items: vec![
                rms(layer.x, layer.attn_norm, layer.norm_out, layer.eps),
                gemm(
                    layer.norm_out,
                    layer.wq,
                    layer.q,
                    1,
                    layer.dim as i32,
                    layer.dim as i32,
                ),
                gemm(
                    layer.norm_out,
                    layer.wk,
                    layer.k,
                    1,
                    layer.kv_dim as i32,
                    layer.dim as i32,
                ),
                gemm(
                    layer.norm_out,
                    layer.wv,
                    layer.v,
                    1,
                    layer.kv_dim as i32,
                    layer.dim as i32,
                ),
                rope_op(layer.q, layer.n_heads, layer.head_dim, pos, layer.theta),
                rope_op(layer.k, layer.n_kv_heads, layer.head_dim, pos, layer.theta),
                // Device-resident KV slot write (no host bounce).
                memcpy_at(layer.k_cache, layer.k, kv_off, kv_bytes),
                memcpy_at(layer.v_cache, layer.v, kv_off, kv_bytes),
                attn(
                    layer.q,
                    layer.k_cache,
                    layer.v_cache,
                    layer.attn_out,
                    n_kv,
                    layer.n_heads,
                    layer.n_kv_heads,
                    layer.head_dim,
                ),
                gemm(
                    layer.attn_out,
                    layer.wo,
                    layer.hidden,
                    1,
                    layer.dim as i32,
                    layer.dim as i32,
                ),
                residual(layer.x, layer.hidden),
                rms(layer.x, layer.ffn_norm, layer.norm_out, layer.eps),
                gemm(
                    layer.norm_out,
                    layer.w_gate,
                    layer.gate,
                    1,
                    layer.ff_dim as i32,
                    layer.dim as i32,
                ),
                gemm(
                    layer.norm_out,
                    layer.w_up,
                    layer.up,
                    1,
                    layer.ff_dim as i32,
                    layer.dim as i32,
                ),
                silu(layer.gate, layer.up),
                gemm(
                    layer.gate,
                    layer.w_down,
                    layer.hidden,
                    1,
                    layer.dim as i32,
                    layer.ff_dim as i32,
                ),
                residual(layer.x, layer.hidden),
            ],
        };
        session.launch(&list).map_err(map_rpc)?;
        session.sync().map_err(map_rpc)?;
        let out = session
            .d2h(layer.x, 0, (layer.dim * 4) as u64)
            .map_err(map_rpc)?;
        self.activation_bytes
            .fetch_add(out.len() as u64, Ordering::Relaxed);
        self.roundtrips.fetch_add(1, Ordering::Relaxed);
        from_f32_bytes(&out, x)?;
        Ok(())
    }
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

fn gemm(a: BufId, b: BufId, c: BufId, m: i32, n: i32, k: i32) -> LaunchItem {
    LaunchItem {
        op: KernelOp::GemmNt,
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
