//! Narrow CUDA decode path: dense TinyModel token forwards on CudaWorkspace.

use super::{gather_kv, LayerWeights, ModelConfig, TinyModel, WeightRef};
use crate::kv_tier::KvTierManager;
use crate::telemetry::Metrics;
use allpaca_cpu::{attention, rope, silu};
use allpaca_cuda::{CudaDeviceWeights, CudaWorkspace};
use allpaca_gguf::HybridLayerKind;
use allpaca_types::{AllpacaError, FatalError};
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
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
}

/// Dense F32 weights packed once onto a CUDA device (embed + layers + norm + lm_head).
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
        if model.config.n_experts > 0 || model.config.recurrent {
            return Err(AllpacaError::Degraded(
                allpaca_types::DegradedError::FeatureDisabled(
                    "CUDA resident upload: dense attention TinyModel only".into(),
                ),
            ));
        }
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
            layers.push(CudaLayerOff {
                attn_norm: push(&mut slices, &mut off, layer.attn_norm.as_slice()),
                wq: push(&mut slices, &mut off, layer.wq.as_slice()),
                wk: push(&mut slices, &mut off, layer.wk.as_slice()),
                wv: push(&mut slices, &mut off, layer.wv.as_slice()),
                wo: push(&mut slices, &mut off, layer.wo.as_slice()),
                ffn_norm: push(&mut slices, &mut off, layer.ffn_norm.as_slice()),
                w_gate: push(&mut slices, &mut off, layer.w_gate.as_slice()),
                w_up: push(&mut slices, &mut off, layer.w_up.as_slice()),
                w_down: push(&mut slices, &mut off, layer.w_down.as_slice()),
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

impl TinyModel {
    /// One-token dense decode using CUDA-resident weights + CudaWorkspace ops.
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
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        if cfg.n_experts > 0 || cfg.recurrent {
            return Err(AllpacaError::Degraded(
                allpaca_types::DegradedError::FeatureDisabled(
                    "CUDA decode: dense attention only".into(),
                ),
            ));
        }
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
            metrics
                .elementwise_gpu_ops
                .fetch_add(1, Ordering::Relaxed);
        }

        let mut scratch = vec![0.0f32; dim.max(cfg.ff_dim).max(kv_dim)];
        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; dim];
        let mut hidden = vec![0.0f32; dim];

        for (layer_idx, host_layer) in self.layers.iter().enumerate() {
            if host_layer.block != HybridLayerKind::Attention {
                return Err(AllpacaError::Degraded(
                    allpaca_types::DegradedError::FeatureDisabled(
                        "CUDA decode: attention blocks only".into(),
                    ),
                ));
            }
            let layer = &resident.layers[layer_idx];
            cuda.rmsnorm_host_x_dev_w(
                &x,
                resident.ptr(layer.attn_norm)?,
                &mut scratch[..dim],
                dim as i32,
                cfg.eps,
            )?;
            metrics
                .elementwise_gpu_ops
                .fetch_add(1, Ordering::Relaxed);

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

            // RoPE on host (device rope available; host keeps KV slot writes simple).
            for h in 0..cfg.n_heads {
                rope(&mut q[h * head_dim..(h + 1) * head_dim], pos, cfg.rope_theta)?;
            }
            for h in 0..cfg.n_kv_heads {
                rope(&mut k[h * head_dim..(h + 1) * head_dim], pos, cfg.rope_theta)?;
            }

            let page_tokens = kv
                .get(page_ids[layer_idx][0])
                .map(|p| p.capacity_tokens)
                .unwrap_or(128);
            let page_index = pos / page_tokens as usize;
            let slot = (pos % page_tokens as usize) as u32;
            let page_id = page_ids[layer_idx][page_index];
            kv.write_slot(page_id, slot, cfg.n_kv_heads, head_dim, &k, &v)?;
            kv.touch(page_id, 1.0);

            let (k_all, v_all) = gather_kv(kv, &page_ids[layer_idx], cfg.n_kv_heads, head_dim)?;
            let n_kv = k_all.len() / kv_dim;
            attention(
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
            )?;

            cuda.gemm_nt_host_a_dev_b(
                &attn_out,
                resident.ptr(layer.wo)?,
                &mut hidden,
                1,
                dim as i32,
                dim as i32,
            )?;
            for i in 0..dim {
                x[i] += hidden[i];
            }

            apply_ffn_cuda(
                cfg,
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
        metrics
            .elementwise_gpu_ops
            .fetch_add(1, Ordering::Relaxed);
        cuda.gemm_nt_host_a_dev_b(
            &scratch[..dim],
            resident.ptr(resident.lm_head)?,
            &mut logits[..vocab],
            1,
            vocab as i32,
            dim as i32,
        )?;

        metrics
            .gpu_forwards
            .fetch_add(1, Ordering::Relaxed);
        metrics
            .target_gpu_forwards
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_ffn_cuda(
    cfg: &ModelConfig,
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
    let ff = cfg.ff_dim;
    let norm_w = if host.ffn_norm.is_empty() {
        // Should not happen for synthetic dense; fall back to ones on host for rmsnorm weight upload path.
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(
            "CUDA FFN missing ffn_norm".into(),
        )));
    } else {
        layer.ffn_norm
    };
    cuda.rmsnorm_host_x_dev_w(x, resident.ptr(norm_w)?, &mut scratch[..dim], dim as i32, cfg.eps)?;
    metrics
        .elementwise_gpu_ops
        .fetch_add(1, Ordering::Relaxed);

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
    for i in 0..ff {
        gate[i] = silu(gate[i]) * up[i];
    }
    cuda.gemm_nt_host_a_dev_b(
        &gate,
        resident.ptr(layer.w_down)?,
        hidden,
        1,
        dim as i32,
        ff as i32,
    )?;
    for i in 0..dim {
        x[i] += hidden[i];
    }
    Ok(())
}
