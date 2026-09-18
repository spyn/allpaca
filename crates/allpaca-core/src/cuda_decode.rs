//! CUDA decode path: dense, tiny MoE, and host-step SSM hybrid on CudaWorkspace.

use super::{
    gather_kv, softcap_inplace, topk_route, LayerWeights, ModelConfig, TinyModel, WeightRef,
};
use crate::anchors::RecurrentState;
use crate::deltanet::{
    conv1d_step, decay_from_alpha, expand_k_heads, gated_delta_step, rmsnorm_silu_gate,
};
use crate::edge0::mla::mla_attend_cpu;
use crate::kv_tier::KvTierManager;
use crate::ssm::mamba2_step;
use crate::telemetry::Metrics;
use allpaca_cpu::{attention_softcap, matmul_nt};
use allpaca_cuda::{CudaDeviceWeights, CudaWorkspace};
use allpaca_gguf::HybridLayerKind;
use allpaca_types::{AllpacaError, FatalError};
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
struct CudaExpertOff {
    w_gate: WeightRef,
    w_up: WeightRef,
    w_down: WeightRef,
}

struct CudaLayerOff {
    attn_norm: WeightRef,
    wq: WeightRef,
    wk: WeightRef,
    wv: WeightRef,
    wo: WeightRef,
    ffn_norm: WeightRef,
    w_gate: WeightRef,
    w_up: WeightRef,
    w_down: WeightRef,
    experts: Vec<CudaExpertOff>,
    shared_expert: CudaExpertOff,
}

/// F32 weights packed once onto a CUDA device (embed + layers + experts + norm + lm_head).
pub struct CudaResidentModel {
    weights: CudaDeviceWeights,
    packed_h2d: u64,
    embed: WeightRef,
    layers: Vec<CudaLayerOff>,
    norm: WeightRef,
    lm_head: WeightRef,
}

impl CudaResidentModel {
    pub fn upload(model: &TinyModel, device: i32) -> Result<Self, AllpacaError> {
        cuda_reject_unsupported_layers(model)?;
        let mut slices: Vec<&[f32]> = Vec::new();
        let mut off = 0usize;
        fn push<'a>(slices: &mut Vec<&'a [f32]>, off: &mut usize, s: &'a [f32]) -> WeightRef {
            let o = *off;
            slices.push(s);
            *off += s.len() * 4;
            WeightRef::f32(o, s.len())
        }
        let embed = push(&mut slices, &mut off, model.embed.as_slice());
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let attn_norm = push(&mut slices, &mut off, layer.attn_norm.as_slice());
            let wq = push(&mut slices, &mut off, layer.wq.as_slice());
            let wk = push(&mut slices, &mut off, layer.wk.as_slice());
            let wv = push(&mut slices, &mut off, layer.wv.as_slice());
            let wo = push(&mut slices, &mut off, layer.wo.as_slice());
            let ffn_norm = if layer.ffn_norm.is_empty() {
                WeightRef::f32(0, 0)
            } else {
                push(&mut slices, &mut off, layer.ffn_norm.as_slice())
            };
            let (w_gate, w_up, w_down) = if layer.w_gate.is_empty() {
                (
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                )
            } else {
                (
                    push(&mut slices, &mut off, layer.w_gate.as_slice()),
                    push(&mut slices, &mut off, layer.w_up.as_slice()),
                    push(&mut slices, &mut off, layer.w_down.as_slice()),
                )
            };
            let mut experts = Vec::with_capacity(layer.experts.len());
            for expert in &layer.experts {
                experts.push(CudaExpertOff {
                    w_gate: push(&mut slices, &mut off, expert.w_gate.as_slice()),
                    w_up: push(&mut slices, &mut off, expert.w_up.as_slice()),
                    w_down: push(&mut slices, &mut off, expert.w_down.as_slice()),
                });
            }
            let shared_expert = if layer.shared_expert.w_up.is_empty() {
                CudaExpertOff {
                    w_gate: WeightRef::f32(0, 0),
                    w_up: WeightRef::f32(0, 0),
                    w_down: WeightRef::f32(0, 0),
                }
            } else {
                CudaExpertOff {
                    w_gate: if layer.shared_expert.w_gate.is_empty() {
                        WeightRef::f32(0, 0)
                    } else {
                        push(&mut slices, &mut off, layer.shared_expert.w_gate.as_slice())
                    },
                    w_up: push(&mut slices, &mut off, layer.shared_expert.w_up.as_slice()),
                    w_down: push(&mut slices, &mut off, layer.shared_expert.w_down.as_slice()),
                }
            };
            layers.push(CudaLayerOff {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
                experts,
                shared_expert,
            });
        }
        let norm = push(&mut slices, &mut off, model.norm.as_slice());
        let lm_head = push(&mut slices, &mut off, model.lm_head.as_slice());
        let (weights, packed_h2d) = CudaDeviceWeights::upload(device, &slices)?;
        Ok(Self {
            weights,
            packed_h2d,
            embed,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn bytes(&self) -> u64 {
        self.weights.bytes() as u64
    }

    pub fn packed_h2d_bytes(&self) -> u64 {
        self.packed_h2d
    }

    fn ptr(&self, w: WeightRef) -> Result<*const f32, AllpacaError> {
        if w.n == 0 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "empty CUDA weight ref".into(),
            )));
        }
        Ok(self.weights.ptr_f32(w.off)? as *const f32)
    }
}

/// Attention / tiny MoE / host-step SSM, GDN, or MLA.
fn cuda_reject_unsupported_layers(model: &TinyModel) -> Result<(), AllpacaError> {
    for (i, layer) in model.layers.iter().enumerate() {
        if let Some(gdn) = &layer.gdn {
            if gdn.w_qkv.is_empty() {
                return Err(AllpacaError::Degraded(
                    allpaca_types::DegradedError::FeatureDisabled(format!(
                        "CUDA resident: layer {i} GDN lacks host projections"
                    )),
                ));
            }
        }
    }
    Ok(())
}

impl TinyModel {
    /// One-token decode using CUDA-resident weights + CudaWorkspace ops
    /// (dense, tiny MoE, or host-step SSM / GDN / MLA hybrid).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token_cuda(
        &self,
        token: u32,
        pos: usize,
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
        cuda: &CudaWorkspace,
        resident: &CudaResidentModel,
        soft_x: Option<&[f32]>,
        mut recurrent: Option<&mut RecurrentState>,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        cuda_reject_unsupported_layers(self)?;
        if self.rpc.as_ref().is_some_and(|r| r.plan_has_remote()) {
            return Err(AllpacaError::Degraded(
                allpaca_types::DegradedError::FeatureDisabled(
                    "CUDA decode: RPC layer-split unsupported".into(),
                ),
            ));
        }
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        let vocab = cfg.vocab_size;
        if logits.len() < vocab {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "logits buffer smaller than vocab".into(),
            )));
        }
        if soft_x.is_none() && token as usize >= vocab {
            return Err(AllpacaError::Fatal(FatalError::Tokenizer(format!(
                "token {token} out of vocab"
            ))));
        }

        struct ProfileGuard<'a> {
            cuda: &'a CudaWorkspace,
        }
        impl Drop for ProfileGuard<'_> {
            fn drop(&mut self) {
                self.cuda.end_decode_profile();
            }
        }
        cuda.begin_decode_profile();
        let _profile = ProfileGuard { cuda };

        let mut x = vec![0.0f32; dim];
        if let Some(x0) = soft_x {
            if x0.len() != dim {
                return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                    "soft embed len {} != dim {dim}",
                    x0.len()
                ))));
            }
            x.copy_from_slice(x0);
        } else {
            cuda.embed_gather_dev_table(
                resident.ptr(resident.embed)?,
                token as i32,
                &mut x,
                dim as i32,
            )?;
            metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);
        }

        let mut scratch = vec![0.0f32; dim.max(cfg.ff_dim).max(kv_dim)];
        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; dim];
        let mut hidden = vec![0.0f32; dim];

        for (layer_idx, host_layer) in self.layers.iter().enumerate() {
            let layer = &resident.layers[layer_idx];
            if host_layer.gdn.is_some() {
                apply_gdn_cuda(
                    cfg,
                    layer_idx,
                    host_layer,
                    layer,
                    resident,
                    cuda,
                    &mut x,
                    &mut scratch,
                    &mut hidden,
                    metrics,
                    recurrent.as_deref_mut(),
                )?;
                continue;
            }
            if host_layer.block == HybridLayerKind::Ssm {
                apply_ssm_cuda(
                    cfg,
                    layer_idx,
                    host_layer,
                    layer,
                    resident,
                    cuda,
                    &mut x,
                    &mut scratch,
                    &mut hidden,
                    metrics,
                    recurrent.as_deref_mut(),
                )?;
                continue;
            }
            if matches!(
                host_layer.block,
                HybridLayerKind::DenseFfn | HybridLayerKind::Moe
            ) {
                // FFN / MoE-only hybrid block: no attention QKV.
                apply_ffn_cuda(
                    cfg,
                    layer_idx,
                    host_layer,
                    layer,
                    resident,
                    cuda,
                    &mut x,
                    &mut scratch,
                    &mut hidden,
                    metrics,
                )?;
                continue;
            }
            if host_layer.block != HybridLayerKind::Attention {
                return Err(AllpacaError::Degraded(
                    allpaca_types::DegradedError::FeatureDisabled(
                        "CUDA decode: attention, SSM, DenseFfn, or MoE blocks only".into(),
                    ),
                ));
            }
            cuda.rmsnorm_host_x_dev_w(
                &x,
                resident.ptr(layer.attn_norm)?,
                &mut scratch[..dim],
                dim as i32,
                cfg.eps,
            )?;
            metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);

            if host_layer.mla.is_some() {
                apply_mla_cuda(
                    self,
                    cfg,
                    layer_idx,
                    host_layer,
                    layer,
                    resident,
                    cuda,
                    &mut x,
                    &mut scratch,
                    &mut hidden,
                    metrics,
                )?;
                continue;
            }

            cuda.gemm_nt_host_a_dev_b(
                &scratch[..dim],
                resident.ptr(layer.wq)?,
                &mut q,
                1,
                dim as i32,
                dim as i32,
            )?;
            cuda.gemm_nt_host_a_dev_b(
                &scratch[..dim],
                resident.ptr(layer.wk)?,
                &mut k,
                1,
                kv_dim as i32,
                dim as i32,
            )?;
            cuda.gemm_nt_host_a_dev_b(
                &scratch[..dim],
                resident.ptr(layer.wv)?,
                &mut v,
                1,
                kv_dim as i32,
                dim as i32,
            )?;

            // RoPE on device (interleaved heads, same layout as HIP allpaca_rope).
            cuda.rope(&mut q, cfg.n_heads as i32, head_dim as i32, pos as i32, cfg.rope_theta)?;
            cuda.rope(
                &mut k,
                cfg.n_kv_heads as i32,
                head_dim as i32,
                pos as i32,
                cfg.rope_theta,
            )?;
            metrics.elementwise_gpu_ops.fetch_add(2, Ordering::Relaxed);

            let page_tokens = kv
                .get(page_ids[layer_idx][0])
                .map(|p| p.capacity_tokens)
                .unwrap_or(128);
            let page_index = pos / page_tokens as usize;
            let slot = (pos % page_tokens as usize) as u32;
            let page_id = page_ids[layer_idx][page_index];
            kv.write_slot(page_id, slot, cfg.n_kv_heads, head_dim, &k, &v)?;
            kv.touch(page_id, 1.0);

            // Softcap path is host-only until CUDA kernels grow a softcap arg
            // (same split as HIP `attention_dispatch`).
            if cfg.attn_logit_softcap > 0.0 {
                let (k_all, v_all) = gather_kv(kv, &page_ids[layer_idx], cfg.n_kv_heads, head_dim)?;
                let n_kv = k_all.len() / kv_dim.max(1);
                attention_softcap(
                    &q,
                    &k_all,
                    &v_all,
                    &mut attn_out,
                    1,
                    n_kv,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    head_dim,
                    true,
                    cfg.attn_logit_softcap,
                )?;
            } else if let Some((k_dev, v_dev, n_kv)) =
                kv.cuda_single_page_view(&page_ids[layer_idx], cfg.n_kv_heads, head_dim)
            {
                cuda.attention_host_q_dev_kv(
                    &q,
                    k_dev,
                    v_dev,
                    &mut attn_out,
                    1,
                    n_kv as i32,
                    cfg.n_heads as i32,
                    cfg.n_kv_heads as i32,
                    head_dim as i32,
                    true,
                )?;
            } else {
                let (k_all, v_all) = gather_kv(kv, &page_ids[layer_idx], cfg.n_kv_heads, head_dim)?;
                let n_kv = k_all.len() / kv_dim.max(1);
                cuda.attention(
                    &q,
                    &k_all,
                    &v_all,
                    &mut attn_out,
                    1,
                    n_kv as i32,
                    cfg.n_heads as i32,
                    cfg.n_kv_heads as i32,
                    head_dim as i32,
                    true,
                )?;
            }

            cuda.gemm_nt_host_a_dev_b(
                &attn_out,
                resident.ptr(layer.wo)?,
                &mut hidden,
                1,
                dim as i32,
                dim as i32,
            )?;
            cuda.residual_add_host(&mut x, &hidden)?;
            metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);

            apply_ffn_cuda(
                cfg,
                layer_idx,
                host_layer,
                layer,
                resident,
                cuda,
                &mut x,
                &mut scratch,
                &mut hidden,
                metrics,
            )?;
        }

        cuda.rmsnorm_host_x_dev_w(
            &x,
            resident.ptr(resident.norm)?,
            &mut scratch[..dim],
            dim as i32,
            cfg.eps,
        )?;
        metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);
        cuda.gemm_nt_host_a_dev_b(
            &scratch[..dim],
            resident.ptr(resident.lm_head)?,
            &mut logits[..vocab],
            1,
            vocab as i32,
            dim as i32,
        )?;
        softcap_inplace(&mut logits[..vocab], cfg.final_logit_softcap);

        metrics.gpu_forwards.fetch_add(1, Ordering::Relaxed);
        metrics.target_gpu_forwards.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_mla_cuda(
    model: &TinyModel,
    cfg: &ModelConfig,
    layer_idx: usize,
    host: &LayerWeights,
    layer: &CudaLayerOff,
    resident: &CudaResidentModel,
    cuda: &CudaWorkspace,
    x: &mut [f32],
    scratch: &mut [f32],
    hidden: &mut [f32],
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    let dim = cfg.dim;
    let mla_w = host.mla.as_ref().ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "CUDA MLA layer {layer_idx} missing host weights"
        )))
    })?;
    if let Some(cache) = &model.mla_cache {
        let mut g = cache.lock().unwrap_or_else(|e| e.into_inner());
        if g.len() <= layer_idx {
            g.resize(layer_idx + 1, (Vec::new(), Vec::new()));
        }
        let (ckv, krope) = &mut g[layer_idx];
        mla_attend_cpu(
            &scratch[..dim],
            mla_w,
            ckv,
            krope,
            cfg.rope_theta,
            cfg.eps,
            hidden,
        )?;
    } else {
        let mut ckv = Vec::new();
        let mut krope = Vec::new();
        mla_attend_cpu(
            &scratch[..dim],
            mla_w,
            &mut ckv,
            &mut krope,
            cfg.rope_theta,
            cfg.eps,
            hidden,
        )?;
    }
    cuda.residual_add_host(x, hidden)?;
    metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);
    apply_ffn_cuda(
        cfg,
        layer_idx,
        host,
        layer,
        resident,
        cuda,
        x,
        scratch,
        hidden,
        metrics,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_gdn_cuda(
    cfg: &ModelConfig,
    layer_idx: usize,
    host: &LayerWeights,
    layer: &CudaLayerOff,
    resident: &CudaResidentModel,
    cuda: &CudaWorkspace,
    x: &mut [f32],
    scratch: &mut [f32],
    hidden: &mut [f32],
    metrics: &Metrics,
    recurrent: Option<&mut RecurrentState>,
) -> Result<(), AllpacaError> {
    let dim = cfg.dim;
    let gdn = host.gdn.as_ref().ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "CUDA GDN layer {layer_idx} missing host weights"
        )))
    })?;
    let dims = &gdn.dims;
    if gdn.w_qkv.len() < dims.qkv_dim.saturating_mul(dim)
        || gdn.w_z.len() < dims.value_dim.saturating_mul(dim)
        || gdn.w_alpha.len() < dims.n_v_heads.saturating_mul(dim)
        || gdn.w_beta.len() < dims.n_v_heads.saturating_mul(dim)
        || gdn.w_out.len() < dim.saturating_mul(dims.value_dim)
    {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "CUDA GDN layer {layer_idx} projection shapes incomplete"
        ))));
    }
    if layer.attn_norm.n == 0 {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "CUDA GDN layer {layer_idx} missing attn_norm"
        ))));
    }
    cuda.rmsnorm_host_x_dev_w(
        x,
        resident.ptr(layer.attn_norm)?,
        &mut scratch[..dim],
        dim as i32,
        cfg.eps,
    )?;
    metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);

    let mut qkv = vec![0.0f32; dims.qkv_dim];
    let mut z = vec![0.0f32; dims.value_dim];
    let mut alpha = vec![0.0f32; dims.n_v_heads];
    let mut beta = vec![0.0f32; dims.n_v_heads];
    matmul_nt(&scratch[..dim], &gdn.w_qkv, &mut qkv, 1, dims.qkv_dim, dim).map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "CUDA GDN qkv gemm layer {layer_idx}: {e}"
        )))
    })?;
    matmul_nt(&scratch[..dim], &gdn.w_z, &mut z, 1, dims.value_dim, dim).map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "CUDA GDN z gemm layer {layer_idx}: {e}"
        )))
    })?;
    matmul_nt(
        &scratch[..dim],
        &gdn.w_alpha,
        &mut alpha,
        1,
        dims.n_v_heads,
        dim,
    )
    .map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "CUDA GDN alpha gemm layer {layer_idx}: {e}"
        )))
    })?;
    matmul_nt(
        &scratch[..dim],
        &gdn.w_beta,
        &mut beta,
        1,
        dims.n_v_heads,
        dim,
    )
    .map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "CUDA GDN beta gemm layer {layer_idx}: {e}"
        )))
    })?;

    let state = recurrent.ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(
            "CUDA GDN layer requires recurrent state".into(),
        ))
    })?;
    state.ensure_gdn(layer_idx, dims);

    let mut conv_out = vec![0.0f32; dims.qkv_dim];
    conv1d_step(
        &qkv,
        &gdn.conv1d,
        &mut state.gdn_conv[layer_idx],
        &mut conv_out,
        dims.conv_kernel,
    );
    let q_raw = &conv_out[..dims.key_dim];
    let k_raw = &conv_out[dims.key_dim..dims.key_dim * 2];
    let v_raw = &conv_out[dims.key_dim * 2..];
    let mut q_exp = vec![0.0f32; dims.n_v_heads * dims.head_k];
    let mut k_exp = vec![0.0f32; dims.n_v_heads * dims.head_k];
    expand_k_heads(q_raw, &mut q_exp, dims.n_k_heads, dims.n_v_heads, dims.head_k);
    expand_k_heads(k_raw, &mut k_exp, dims.n_k_heads, dims.n_v_heads, dims.head_k);
    let mut decay = vec![0.0f32; dims.n_v_heads];
    decay_from_alpha(&alpha, &gdn.dt_bias, &gdn.a, &mut decay);
    let mut y = vec![0.0f32; dims.value_dim];
    gated_delta_step(
        &mut q_exp,
        &mut k_exp,
        v_raw,
        &decay,
        &beta,
        &mut state.gdn_state[layer_idx],
        &mut y,
        dims.n_v_heads,
        dims.head_v,
        dims.head_k,
    );
    let mut gated = vec![0.0f32; dims.value_dim];
    rmsnorm_silu_gate(&y, &z, &gdn.norm, &mut gated, dims.head_v, cfg.eps);
    matmul_nt(&gated, &gdn.w_out, hidden, 1, dim, dims.value_dim).map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "CUDA GDN out gemm layer {layer_idx}: {e}"
        )))
    })?;
    cuda.residual_add_host(x, hidden)?;
    metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);

    if !host.w_gate.is_empty() || !host.experts.is_empty() {
        apply_ffn_cuda(
            cfg,
            layer_idx,
            host,
            layer,
            resident,
            cuda,
            x,
            scratch,
            hidden,
            metrics,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_ssm_cuda(
    cfg: &ModelConfig,
    layer_idx: usize,
    host: &LayerWeights,
    layer: &CudaLayerOff,
    resident: &CudaResidentModel,
    cuda: &CudaWorkspace,
    x: &mut [f32],
    scratch: &mut [f32],
    hidden: &mut [f32],
    metrics: &Metrics,
    recurrent: Option<&mut RecurrentState>,
) -> Result<(), AllpacaError> {
    let dim = cfg.dim;
    let ssm = host.ssm.as_ref().ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "CUDA SSM layer {layer_idx} missing Mamba-2 weights"
        )))
    })?;
    if layer.attn_norm.n == 0 {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "CUDA SSM layer {layer_idx} missing attn_norm"
        ))));
    }
    cuda.rmsnorm_host_x_dev_w(
        x,
        resident.ptr(layer.attn_norm)?,
        &mut scratch[..dim],
        dim as i32,
        cfg.eps,
    )?;
    metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);

    let state = recurrent.ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(
            "CUDA SSM layer requires recurrent state".into(),
        ))
    })?;
    state.ensure_ssm(
        layer_idx,
        ssm.dims.state_floats(),
        ssm.dims.conv_state_floats(),
    );
    mamba2_step(
        &scratch[..dim],
        &mut state.ssm_h[layer_idx],
        &mut state.ssm_conv[layer_idx],
        ssm,
        cfg.eps,
        hidden,
    )
    .map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "CUDA SSM layer {layer_idx}: {e}"
        )))
    })?;
    cuda.residual_add_host(x, hidden)?;
    metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);

    if !host.w_gate.is_empty() || !host.experts.is_empty() {
        apply_ffn_cuda(
            cfg,
            layer_idx,
            host,
            layer,
            resident,
            cuda,
            x,
            scratch,
            hidden,
            metrics,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_ffn_cuda(
    cfg: &ModelConfig,
    layer_idx: usize,
    host: &LayerWeights,
    layer: &CudaLayerOff,
    resident: &CudaResidentModel,
    cuda: &CudaWorkspace,
    x: &mut [f32],
    scratch: &mut [f32],
    hidden: &mut [f32],
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    let use_moe = cfg.n_experts > 0
        && !host.experts.is_empty()
        && !matches!(host.block, HybridLayerKind::DenseFfn)
        && layer_idx >= cfg.leading_dense_blocks;
    if use_moe {
        return apply_moe_cuda(cfg, host, layer, resident, cuda, x, scratch, metrics);
    }

    let dim = cfg.dim;
    let ff = cfg.ff_dim;
    if host.ffn_norm.is_empty() {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(
            "CUDA FFN missing ffn_norm".into(),
        )));
    }
    cuda.rmsnorm_host_x_dev_w(
        x,
        resident.ptr(layer.ffn_norm)?,
        &mut scratch[..dim],
        dim as i32,
        cfg.eps,
    )?;
    metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);

    let mut gate = vec![0.0f32; ff];
    let mut up = vec![0.0f32; ff];
    cuda.gemm_nt_host_a_dev_b(
        &scratch[..dim],
        resident.ptr(layer.w_gate)?,
        &mut gate,
        1,
        ff as i32,
        dim as i32,
    )?;
    cuda.gemm_nt_host_a_dev_b(
        &scratch[..dim],
        resident.ptr(layer.w_up)?,
        &mut up,
        1,
        ff as i32,
        dim as i32,
    )?;
    cuda.silu_mul_host(&mut gate, &up)?;
    metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);
    cuda.gemm_nt_host_a_dev_b(
        &gate,
        resident.ptr(layer.w_down)?,
        hidden,
        1,
        dim as i32,
        ff as i32,
    )?;
    cuda.residual_add_host(x, hidden)?;
    metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_moe_cuda(
    cfg: &ModelConfig,
    host: &LayerWeights,
    layer: &CudaLayerOff,
    resident: &CudaResidentModel,
    cuda: &CudaWorkspace,
    x: &mut [f32],
    scratch: &mut [f32],
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    let dim = cfg.dim;
    let ff = cfg.ff_dim;
    if host.ffn_norm.is_empty() || layer.ffn_norm.n == 0 {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(
            "CUDA MoE missing ffn_norm".into(),
        )));
    }
    if layer.experts.len() < cfg.n_experts {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(
            "CUDA MoE resident experts shorter than n_experts".into(),
        )));
    }
    cuda.rmsnorm_host_x_dev_w(
        x,
        resident.ptr(layer.ffn_norm)?,
        &mut scratch[..dim],
        dim as i32,
        cfg.eps,
    )?;
    metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);

    let n_exp = cfg.n_experts.max(1);
    let n_act = cfg.n_active.max(1).min(n_exp);
    let routed = topk_route(&scratch[..dim], &host.w_router, n_exp, n_act, cfg);

    let mut acc = vec![0.0f32; dim];
    let mut expert_out = vec![0.0f32; dim];
    let mut gate = vec![0.0f32; ff];
    let mut up = vec![0.0f32; ff];
    for &(e, w) in &routed {
        let ex = &layer.experts[e];
        cuda.gemm_nt_host_a_dev_b(
            &scratch[..dim],
            resident.ptr(ex.w_gate)?,
            &mut gate,
            1,
            ff as i32,
            dim as i32,
        )?;
        cuda.gemm_nt_host_a_dev_b(
            &scratch[..dim],
            resident.ptr(ex.w_up)?,
            &mut up,
            1,
            ff as i32,
            dim as i32,
        )?;
        cuda.silu_mul_host(&mut gate, &up)?;
        metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);
        cuda.gemm_nt_host_a_dev_b(
            &gate,
            resident.ptr(ex.w_down)?,
            &mut expert_out,
            1,
            dim as i32,
            ff as i32,
        )?;
        for i in 0..dim {
            acc[i] += w * expert_out[i];
        }
    }

    if layer.shared_expert.w_up.n > 0 {
        let sff = if cfg.shared_ff_dim > 0 {
            cfg.shared_ff_dim
        } else {
            layer.shared_expert.w_up.n / dim.max(1)
        };
        let mut shared_gate = vec![0.0f32; sff];
        let mut shared_up = vec![0.0f32; sff];
        if layer.shared_expert.w_gate.n > 0 {
            cuda.gemm_nt_host_a_dev_b(
                &scratch[..dim],
                resident.ptr(layer.shared_expert.w_gate)?,
                &mut shared_gate,
                1,
                sff as i32,
                dim as i32,
            )?;
            cuda.gemm_nt_host_a_dev_b(
                &scratch[..dim],
                resident.ptr(layer.shared_expert.w_up)?,
                &mut shared_up,
                1,
                sff as i32,
                dim as i32,
            )?;
            cuda.silu_mul_host(&mut shared_gate, &shared_up)?;
            metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);
            cuda.gemm_nt_host_a_dev_b(
                &shared_gate,
                resident.ptr(layer.shared_expert.w_down)?,
                &mut expert_out,
                1,
                dim as i32,
                sff as i32,
            )?;
        } else {
            // Gate-less shared expert: act(up) @ down (Lightning ReLU² or SiLU).
            cuda.gemm_nt_host_a_dev_b(
                &scratch[..dim],
                resident.ptr(layer.shared_expert.w_up)?,
                &mut shared_up,
                1,
                sff as i32,
                dim as i32,
            )?;
            if cfg.moe_relu_sqr {
                for v in &mut shared_up {
                    *v = super::relu_sqr(*v);
                }
            } else {
                for v in &mut shared_up {
                    *v = *v * (1.0 / (1.0 + (-*v).exp()));
                }
            }
            cuda.gemm_nt_host_a_dev_b(
                &shared_up,
                resident.ptr(layer.shared_expert.w_down)?,
                &mut expert_out,
                1,
                dim as i32,
                sff as i32,
            )?;
        }
        for i in 0..dim {
            acc[i] += expert_out[i];
        }
    }

    cuda.residual_add_host(x, &acc)?;
    metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);
    Ok(())
}
