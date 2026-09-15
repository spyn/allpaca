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

    /// Dense Llama-family GGUF → device pack. Small tensors dequant to F32; large
/// Q8_0 GEMM weights and `token_embd` stay as 36-byte staged blocks so 7B-class
/// models fit K80 VRAM (~12GiB) instead of exploding to ~32GiB F32.
pub fn upload_gguf(
        file: &allpaca_gguf::GgufFile,
        cfg: &ModelConfig,
        device: i32,
    ) -> Result<Self, AllpacaError> {
        if cfg.n_experts > 0 || cfg.recurrent {
            return Err(AllpacaError::Degraded(
                allpaca_types::DegradedError::FeatureDisabled(
                    "CUDA GGUF upload: dense attention only (no MoE/hybrid yet)".into(),
                ),
            ));
        }
        let mut owned: Vec<Vec<u8>> = Vec::new();
        let mut off = 0usize;
        let mut push_bytes =
            |owned: &mut Vec<Vec<u8>>, off: &mut usize, data: Vec<u8>, kind: super::WeightKind, n: usize| -> WeightRef {
                if kind == super::WeightKind::Q8_0 && *off % allpaca_cuda::Q8_STAGED_BLOCK != 0 {
                    let pad = allpaca_cuda::Q8_STAGED_BLOCK - (*off % allpaca_cuda::Q8_STAGED_BLOCK);
                    owned.push(vec![0u8; pad]);
                    *off += pad;
                }
                let o = *off;
                *off += data.len();
                owned.push(data);
                match kind {
                    super::WeightKind::Q8_0 => WeightRef::q8_0(o, n),
                    _ => WeightRef::f32(o, n),
                }
            };
        let take_tensor = |file: &allpaca_gguf::GgufFile,
                           name: &str,
                           gemv: bool|
         -> Result<(Vec<u8>, super::WeightKind, usize), AllpacaError> {
            let info = file.tensor(name).ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "CUDA GGUF upload missing {name}"
                )))
            })?;
            let n = info.n_elements() as usize;
            match info.ggml_type {
                allpaca_gguf::GgmlType::Q8_0 if n > 65_536 => {
                    let packed = file.tensor_bytes(name).map_err(|e| {
                        AllpacaError::Fatal(FatalError::ModelLoad(format!(
                            "CUDA GGUF Q8 bytes {name}: {e}"
                        )))
                    })?;
                    if name == "token_embd.weight" || !gemv {
                        let staged = allpaca_cuda::stage_q8_0_blocks(packed).map_err(|e| {
                            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                "CUDA Q8_0 stage {name}: {e}"
                            )))
                        })?;
                        Ok((staged, super::WeightKind::Q8_0, n))
                    } else {
                        let k = *info.dims.first().unwrap_or(&1) as usize;
                        if k == 0 || n % k != 0 {
                            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                "Q8_0 GEMV {name} dims {:?} do not tile n={n}",
                                info.dims
                            ))));
                        }
                        let n_rows = n / k;
                        let staged =
                            allpaca_cuda::stage_q8_0_gemv(packed, n_rows, k).map_err(|e| {
                                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                    "CUDA Q8_0 GEMV stage {name}: {e}"
                                )))
                            })?;
                        Ok((staged, super::WeightKind::Q8_0, n))
                    }
                }
                _ => {
                    let f32s = file.tensor_f32(name).map_err(|e| {
                        AllpacaError::Fatal(FatalError::ModelLoad(format!(
                            "CUDA GGUF upload missing {name}: {e}"
                        )))
                    })?;
                    let n = f32s.len();
                    let mut bytes = Vec::with_capacity(n * 4);
                    for v in f32s {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    Ok((bytes, super::WeightKind::F32, n))
                }
            }
        };

        let (embed_bytes, embed_kind, embed_n) = take_tensor(file, "token_embd.weight", false)?;
        let embed = push_bytes(&mut owned, &mut off, embed_bytes, embed_kind, embed_n);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for local in 0..cfg.n_layers {
            let i = cfg.gguf_blk(local);
            let take = |suffix: &str, gemv: bool| {
                take_tensor(file, &format!("blk.{i}.{suffix}"), gemv)
            };
            let (b, k, n) = take("attn_norm.weight", false)?;
            let attn_norm = push_bytes(&mut owned, &mut off, b, k, n);
            let (b, k, n) = take("attn_q.weight", true)?;
            let wq = push_bytes(&mut owned, &mut off, b, k, n);
            let (b, k, n) = take("attn_k.weight", true)?;
            let wk = push_bytes(&mut owned, &mut off, b, k, n);
            let (b, k, n) = take("attn_v.weight", true)?;
            let wv = push_bytes(&mut owned, &mut off, b, k, n);
            let (b, k, n) = take("attn_output.weight", true)?;
            let wo = push_bytes(&mut owned, &mut off, b, k, n);
            let (b, k, n) = take("ffn_norm.weight", false)?;
            let ffn_norm = push_bytes(&mut owned, &mut off, b, k, n);
            let (b, k, n) = take("ffn_gate.weight", true)?;
            let w_gate = push_bytes(&mut owned, &mut off, b, k, n);
            let (b, k, n) = take("ffn_up.weight", true)?;
            let w_up = push_bytes(&mut owned, &mut off, b, k, n);
            let (b, k, n) = take("ffn_down.weight", true)?;
            let w_down = push_bytes(&mut owned, &mut off, b, k, n);
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
            });
        }
        let (norm_bytes, norm_kind, norm_n) = take_tensor(file, "output_norm.weight", false)?;
        let norm = push_bytes(&mut owned, &mut off, norm_bytes, norm_kind, norm_n);
        let lm_head = if file.tensor("output.weight").is_some() {
            let (b, k, n) = take_tensor(file, "output.weight", true)?;
            push_bytes(&mut owned, &mut off, b, k, n)
        } else {
            // Tied embedding / lm_head: reuse embed byte offset.
            embed
        };
        let slices: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
        let (weights, packed_h2d) = CudaDeviceWeights::upload_bytes(device, &slices)?;
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
        if w.kind != super::WeightKind::F32 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "CUDA F32 ptr requested for non-F32 weight".into(),
            )));
        }
        Ok(self.weights.ptr_f32(w.off)? as *const f32)
    }

    fn byte_ptr(&self, w: WeightRef) -> Result<*const u8, AllpacaError> {
        if w.n == 0 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "empty CUDA weight ref".into(),
            )));
        }
        Ok(self.weights.byte_ptr_at(w.off)? as *const u8)
    }

    fn gemm_host_a(
        &self,
        cuda: &CudaWorkspace,
        a: &[f32],
        w: WeightRef,
        c: &mut [f32],
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        match w.kind {
            super::WeightKind::F32 => {
                cuda.gemm_nt_host_a_dev_b(a, self.ptr(w)?, c, m, n, k)
            }
            super::WeightKind::Q8_0 => {
                cuda.gemm_nt_host_a_dev_b_q8(a, self.byte_ptr(w)?, c, m, n, k)
            }
            _ => Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "CUDA decode: unsupported weight kind for GEMM".into(),
            ))),
        }
    }

    fn embed_gather(
        &self,
        cuda: &CudaWorkspace,
        token: i32,
        out: &mut [f32],
        dim: i32,
    ) -> Result<(), AllpacaError> {
        match self.embed.kind {
            super::WeightKind::F32 => {
                cuda.embed_gather_dev_table(self.ptr(self.embed)?, token, out, dim)
            }
            super::WeightKind::Q8_0 => {
                cuda.embed_gather_q8_dev_table(self.byte_ptr(self.embed)?, token, out, dim)
            }
            _ => Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "CUDA decode: unsupported embed kind".into(),
            ))),
        }
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
            resident.embed_gather(cuda, token as i32, &mut x, dim as i32)?;
            metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);
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
            metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);

            resident.gemm_host_a(
                cuda,
                &scratch[..dim],
                layer.wq,
                &mut q,
                1,
                dim as i32,
                dim as i32,
            )?;
            resident.gemm_host_a(
                cuda,
                &scratch[..dim],
                layer.wk,
                &mut k,
                1,
                kv_dim as i32,
                dim as i32,
            )?;
            resident.gemm_host_a(
                cuda,
                &scratch[..dim],
                layer.wv,
                &mut v,
                1,
                kv_dim as i32,
                dim as i32,
            )?;

            // RoPE on host (device rope available; host keeps KV slot writes simple).
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
            }

            resident.gemm_host_a(
                cuda,
                &attn_out,
                layer.wo,
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
        metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);
        resident.gemm_host_a(
            cuda,
            &scratch[..dim],
            resident.lm_head,
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
    cuda.rmsnorm_host_x_dev_w(
        x,
        resident.ptr(norm_w)?,
        &mut scratch[..dim],
        dim as i32,
        cfg.eps,
    )?;
    metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);

    let mut gate = vec![0.0f32; ff];
    let mut up = vec![0.0f32; ff];
    resident.gemm_host_a(
        cuda,
        &scratch[..dim],
        layer.w_gate,
        &mut gate,
        1,
        ff as i32,
        dim as i32,
    )?;
    resident.gemm_host_a(
        cuda,
        &scratch[..dim],
        layer.w_up,
        &mut up,
        1,
        ff as i32,
        dim as i32,
    )?;
    for i in 0..ff {
        gate[i] = silu(gate[i]) * up[i];
    }
    resident.gemm_host_a(
        cuda,
        &gate,
        layer.w_down,
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
