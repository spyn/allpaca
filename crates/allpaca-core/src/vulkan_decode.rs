//! Vulkan decode path: dense + synthetic MoE TinyModel token forwards on VulkanWorkspace.

use super::{gather_kv, topk_route, LayerWeights, ModelConfig, TinyModel, WeightRef};
use crate::kv_tier::KvTierManager;
use crate::telemetry::Metrics;
use allpaca_cpu::{rope, silu};
use allpaca_gguf::HybridLayerKind;
use allpaca_types::{AllpacaError, FatalError};
use allpaca_vulkan::{DeviceSlice, VulkanDeviceWeights, VulkanWorkspace};
use std::sync::atomic::Ordering;

fn vulkan_decode_blocks_ok(model: &TinyModel) -> bool {
    if model.dsv4.is_some() {
        return false;
    }
    model.layers.iter().all(|l| {
        l.ssm.is_none()
            && l.gdn.is_none()
            && l.mla.is_none()
            && matches!(
                l.block,
                HybridLayerKind::Attention | HybridLayerKind::DenseFfn | HybridLayerKind::Moe
            )
    })
}

fn align256(x: usize) -> usize {
    x.div_ceil(256) * 256
}

#[derive(Clone, Copy)]
struct VulkanExpertOff {
    w_gate: WeightRef,
    w_up: WeightRef,
    w_down: WeightRef,
}

struct VulkanLayerOff {
    attn_norm: WeightRef,
    wq: WeightRef,
    wk: WeightRef,
    wv: WeightRef,
    wo: WeightRef,
    ffn_norm: WeightRef,
    w_gate: WeightRef,
    w_up: WeightRef,
    w_down: WeightRef,
    w_router: WeightRef,
    experts: Vec<VulkanExpertOff>,
}

/// Dense F32 weights packed once onto a Vulkan device (embed + layers + norm + lm_head).
pub struct VulkanResidentModel {
    weights: VulkanDeviceWeights,
    packed_h2d: u64,
    embed: WeightRef,
    layers: Vec<VulkanLayerOff>,
    norm: WeightRef,
    lm_head: WeightRef,
}

impl VulkanResidentModel {
    pub fn upload(model: &TinyModel, ws: &VulkanWorkspace) -> Result<Self, AllpacaError> {
        if !vulkan_decode_blocks_ok(model) {
            return Err(AllpacaError::Degraded(
                allpaca_types::DegradedError::FeatureDisabled(
                    "Vulkan resident upload: attention/dense-FFN/MoE TinyModel only (no SSM/GDN/MLA/DSV4)".into(),
                ),
            ));
        }
        let mut slices: Vec<&[f32]> = Vec::new();
        let mut off = 0usize;
        fn push<'a>(slices: &mut Vec<&'a [f32]>, off: &mut usize, s: &'a [f32]) -> WeightRef {
            let o = *off;
            slices.push(s);
            *off = align256(*off + s.len() * 4);
            WeightRef::f32(o, s.len())
        }
        let embed = push(&mut slices, &mut off, model.embed.as_slice());
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            layers.push(VulkanLayerOff {
                attn_norm: push(&mut slices, &mut off, layer.attn_norm.as_slice()),
                wq: push(&mut slices, &mut off, layer.wq.as_slice()),
                wk: push(&mut slices, &mut off, layer.wk.as_slice()),
                wv: push(&mut slices, &mut off, layer.wv.as_slice()),
                wo: push(&mut slices, &mut off, layer.wo.as_slice()),
                ffn_norm: push(&mut slices, &mut off, layer.ffn_norm.as_slice()),
                w_gate: push(&mut slices, &mut off, layer.w_gate.as_slice()),
                w_up: push(&mut slices, &mut off, layer.w_up.as_slice()),
                w_down: push(&mut slices, &mut off, layer.w_down.as_slice()),
                w_router: push(&mut slices, &mut off, layer.w_router.as_slice()),
                experts: layer
                    .experts
                    .iter()
                    .map(|e| VulkanExpertOff {
                        w_gate: push(&mut slices, &mut off, e.w_gate.as_slice()),
                        w_up: push(&mut slices, &mut off, e.w_up.as_slice()),
                        w_down: push(&mut slices, &mut off, e.w_down.as_slice()),
                    })
                    .collect(),
            });
        }
        let norm = push(&mut slices, &mut off, model.norm.as_slice());
        let lm_head = push(&mut slices, &mut off, model.lm_head.as_slice());
        let (weights, packed_h2d) = VulkanDeviceWeights::upload(ws, &slices)?;
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

    fn slice(&self, w: WeightRef) -> Result<DeviceSlice, AllpacaError> {
        if w.n == 0 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "empty Vulkan weight ref".into(),
            )));
        }
        self.weights.slice_f32(w.off, w.n)
    }
}

impl TinyModel {
    /// One-token dense decode using Vulkan-resident weights + VulkanWorkspace ops.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token_vulkan(
        &self,
        token: u32,
        pos: usize,
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
        vulkan: &VulkanWorkspace,
        resident: &VulkanResidentModel,
        soft_x: Option<&[f32]>,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        if !vulkan_decode_blocks_ok(self) {
            return Err(AllpacaError::Degraded(
                allpaca_types::DegradedError::FeatureDisabled(
                    "Vulkan decode: attention/dense-FFN/MoE only (no SSM/GDN/MLA)".into(),
                ),
            ));
        }
        if self.rpc.as_ref().is_some_and(|r| r.plan_has_remote()) {
            return Err(AllpacaError::Degraded(
                allpaca_types::DegradedError::FeatureDisabled(
                    "Vulkan decode: RPC layer-split unsupported".into(),
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
            vulkan.embed_gather_dev_table(
                resident.slice(resident.embed)?,
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
            if matches!(
                host_layer.block,
                HybridLayerKind::DenseFfn | HybridLayerKind::Moe
            ) {
                apply_ffn_vulkan(
                    cfg,
                    host_layer,
                    layer,
                    resident,
                    vulkan,
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
                        "Vulkan decode: attention blocks only".into(),
                    ),
                ));
            }
            vulkan.rmsnorm_host_x_dev_w(
                &x,
                resident.slice(layer.attn_norm)?,
                &mut scratch[..dim],
                dim as i32,
                cfg.eps,
            )?;
            metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);

            vulkan.gemm_nt_host_a_dev_b(
                &scratch[..dim],
                resident.slice(layer.wq)?,
                &mut q,
                1,
                dim as i32,
                dim as i32,
            )?;
            vulkan.gemm_nt_host_a_dev_b(
                &scratch[..dim],
                resident.slice(layer.wk)?,
                &mut k,
                1,
                kv_dim as i32,
                dim as i32,
            )?;
            vulkan.gemm_nt_host_a_dev_b(
                &scratch[..dim],
                resident.slice(layer.wv)?,
                &mut v,
                1,
                kv_dim as i32,
                dim as i32,
            )?;

            for h in 0..cfg.n_heads {
                rope(
                    &mut q[h * head_dim..(h + 1) * head_dim],
                    pos,
                    cfg.rope_theta,
                )?;
            }
            for h in 0..cfg.n_kv_heads {
                rope(
                    &mut k[h * head_dim..(h + 1) * head_dim],
                    pos,
                    cfg.rope_theta,
                )?;
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

            if let Some((k_dev, v_dev, n_kv)) =
                kv.vulkan_single_page_view(&page_ids[layer_idx], cfg.n_kv_heads, head_dim)
            {
                vulkan.attention_host_q_dev_kv(
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
                let n_kv = k_all.len() / kv_dim;
                vulkan.attention(
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

            vulkan.gemm_nt_host_a_dev_b(
                &attn_out,
                resident.slice(layer.wo)?,
                &mut hidden,
                1,
                dim as i32,
                dim as i32,
            )?;
            for i in 0..dim {
                x[i] += hidden[i];
            }

            apply_ffn_vulkan(
                cfg,
                host_layer,
                layer,
                resident,
                vulkan,
                &mut x,
                &mut scratch,
                &mut hidden,
                metrics,
            )?;
        }

        vulkan.rmsnorm_host_x_dev_w(
            &x,
            resident.slice(resident.norm)?,
            &mut scratch[..dim],
            dim as i32,
            cfg.eps,
        )?;
        metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);
        vulkan.gemm_nt_host_a_dev_b(
            &scratch[..dim],
            resident.slice(resident.lm_head)?,
            &mut logits[..vocab],
            1,
            vocab as i32,
            dim as i32,
        )?;

        metrics.gpu_forwards.fetch_add(1, Ordering::Relaxed);
        metrics.target_gpu_forwards.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_ffn_vulkan(
    cfg: &ModelConfig,
    host: &LayerWeights,
    layer: &VulkanLayerOff,
    resident: &VulkanResidentModel,
    vulkan: &VulkanWorkspace,
    x: &mut [f32],
    scratch: &mut [f32],
    hidden: &mut [f32],
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    if cfg.n_experts > 0 && !host.experts.is_empty() {
        return apply_moe_vulkan(cfg, host, layer, resident, vulkan, x, scratch, metrics);
    }
    let dim = cfg.dim;
    let ff = cfg.ff_dim;
    if host.ffn_norm.is_empty() {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(
            "Vulkan FFN missing ffn_norm".into(),
        )));
    }
    vulkan.rmsnorm_host_x_dev_w(
        x,
        resident.slice(layer.ffn_norm)?,
        &mut scratch[..dim],
        dim as i32,
        cfg.eps,
    )?;
    metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);

    let mut gate = vec![0.0f32; ff];
    let mut up = vec![0.0f32; ff];
    vulkan.gemm_nt_host_a_dev_b(
        &scratch[..dim],
        resident.slice(layer.w_gate)?,
        &mut gate,
        1,
        ff as i32,
        dim as i32,
    )?;
    vulkan.gemm_nt_host_a_dev_b(
        &scratch[..dim],
        resident.slice(layer.w_up)?,
        &mut up,
        1,
        ff as i32,
        dim as i32,
    )?;
    for i in 0..ff {
        gate[i] = silu(gate[i]) * up[i];
    }
    vulkan.gemm_nt_host_a_dev_b(
        &gate,
        resident.slice(layer.w_down)?,
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

#[allow(clippy::too_many_arguments)]
fn apply_moe_vulkan(
    cfg: &ModelConfig,
    host: &LayerWeights,
    layer: &VulkanLayerOff,
    resident: &VulkanResidentModel,
    vulkan: &VulkanWorkspace,
    x: &mut [f32],
    scratch: &mut [f32],
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    let dim = cfg.dim;
    let ff = cfg.ff_dim;
    if host.ffn_norm.is_empty() || host.w_router.is_empty() {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(
            "Vulkan MoE missing ffn_norm or w_router".into(),
        )));
    }
    if layer.w_router.n != 0 && layer.w_router.n != host.w_router.len() {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(
            "Vulkan MoE router size mismatch".into(),
        )));
    }
    vulkan.rmsnorm_host_x_dev_w(
        x,
        resident.slice(layer.ffn_norm)?,
        &mut scratch[..dim],
        dim as i32,
        cfg.eps,
    )?;
    metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);
    let n_exp = cfg.n_experts.max(1);
    let n_act = cfg.n_active.max(1).min(n_exp);
    let routed = topk_route(&scratch[..dim], &host.w_router, n_exp, n_act, cfg);
    let mut acc = vec![0.0f32; dim];
    let mut gate = vec![0.0f32; ff];
    let mut up = vec![0.0f32; ff];
    let mut down = vec![0.0f32; dim];
    for (eid, weight) in routed {
        let Some(ex) = layer.experts.get(eid) else {
            continue;
        };
        vulkan.gemm_nt_host_a_dev_b(
            &scratch[..dim],
            resident.slice(ex.w_gate)?,
            &mut gate,
            1,
            ff as i32,
            dim as i32,
        )?;
        vulkan.gemm_nt_host_a_dev_b(
            &scratch[..dim],
            resident.slice(ex.w_up)?,
            &mut up,
            1,
            ff as i32,
            dim as i32,
        )?;
        for i in 0..ff {
            gate[i] = silu(gate[i]) * up[i];
        }
        vulkan.gemm_nt_host_a_dev_b(
            &gate,
            resident.slice(ex.w_down)?,
            &mut down,
            1,
            dim as i32,
            ff as i32,
        )?;
        for i in 0..dim {
            acc[i] += weight * down[i];
        }
    }
    for i in 0..dim {
        x[i] += acc[i];
    }
    Ok(())
}
