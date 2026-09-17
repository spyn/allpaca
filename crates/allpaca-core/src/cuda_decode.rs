//! Narrow CUDA decode path: TinyModel token forwards on CudaWorkspace
//! (dense + synthetic MoE; attention + host Mamba-2 / GDN hybrid).

use super::{topk_route, GdnHostWeights, LayerWeights, ModelConfig, TinyModel, WeightRef};
use crate::anchors::RecurrentState;
use crate::deltanet::{
    conv1d_step, decay_from_alpha, expand_k_heads, gated_delta_step_eps, rmsnorm_silu_gate, sigmoid,
};
use crate::kv_tier::KvTierManager;
use crate::ssm::mamba2_step;
use crate::telemetry::Metrics;
use allpaca_cpu::silu;
use allpaca_cuda::{CudaDeviceWeights, CudaWorkspace};
use allpaca_gguf::HybridLayerKind;
use allpaca_types::{AllpacaError, FatalError};
use std::sync::atomic::Ordering;

#[derive(Clone)]
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
    /// Host copy of router logits weights for top-k when `LayerWeights.w_router` is empty
    /// (large GGUF skeleton path).
    router_host: Vec<f32>,
    /// Per-expert `[w_gate, w_up, w_down]` when `n_experts > 0`.
    experts: Vec<[WeightRef; 3]>,
    /// GDN projections (`attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_out`).
    w_qkv: WeightRef,
    w_z: WeightRef,
    w_alpha: WeightRef,
    w_beta: WeightRef,
    w_ssm_out: WeightRef,
}

/// Device-resident weights for CUDA decode (embed + layers + experts + norm + lm_head).
pub struct CudaResidentModel {
    weights: CudaDeviceWeights,
    packed_h2d: u64,
    embed: WeightRef,
    layers: Vec<CudaLayerOff>,
    norm: WeightRef,
    lm_head: WeightRef,
}

fn cuda_supports_blocks(model: &TinyModel) -> Result<(), AllpacaError> {
    for (i, layer) in model.layers.iter().enumerate() {
        match layer.block {
            HybridLayerKind::Attention
            | HybridLayerKind::Ssm
            | HybridLayerKind::DenseFfn
            | HybridLayerKind::Moe => {}
            other => {
                return Err(AllpacaError::Degraded(
                    allpaca_types::DegradedError::FeatureDisabled(format!(
                        "CUDA resident upload: layer {i} is {other:?}; GDN later"
                    )),
                ));
            }
        }
        if layer.block == HybridLayerKind::Ssm && layer.ssm.is_none() && layer.gdn.is_none() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "CUDA upload: SSM/GDN layer {i} missing host recurrent weights"
            ))));
        }
        if let Some(gdn) = layer.gdn.as_ref() {
            let host_qkv = !gdn.w_qkv.is_empty();
            let host_out = !gdn.w_out.is_empty();
            if host_qkv != host_out {
                return Err(AllpacaError::Degraded(
                    allpaca_types::DegradedError::FeatureDisabled(
                        "CUDA GDN host projections incomplete".into(),
                    ),
                ));
            }
        }
    }
    Ok(())
}

impl CudaResidentModel {
    pub fn upload(model: &TinyModel, device: i32) -> Result<Self, AllpacaError> {
        cuda_supports_blocks(model)?;
        let mut slices: Vec<&[f32]> = Vec::new();
        let mut off = 0usize;
        fn push<'a>(slices: &mut Vec<&'a [f32]>, off: &mut usize, s: &'a [f32]) -> WeightRef {
            if s.is_empty() {
                return WeightRef::f32(0, 0);
            }
            let o = *off;
            slices.push(s);
            *off += s.len() * 4;
            WeightRef::f32(o, s.len())
        }
        let embed = push(&mut slices, &mut off, model.embed.as_slice());
        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let mut experts = Vec::with_capacity(layer.experts.len());
            for expert in &layer.experts {
                experts.push([
                    push(&mut slices, &mut off, expert.w_gate.as_slice()),
                    push(&mut slices, &mut off, expert.w_up.as_slice()),
                    push(&mut slices, &mut off, expert.w_down.as_slice()),
                ]);
            }
            let gdn = layer.gdn.as_ref();
            let empty: &[f32] = &[];
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
                router_host: layer.w_router.clone(),
                experts,
                w_qkv: push(
                    &mut slices,
                    &mut off,
                    gdn.map(|g| g.w_qkv.as_slice()).unwrap_or(empty),
                ),
                w_z: push(
                    &mut slices,
                    &mut off,
                    gdn.map(|g| g.w_z.as_slice()).unwrap_or(empty),
                ),
                w_alpha: push(
                    &mut slices,
                    &mut off,
                    gdn.map(|g| g.w_alpha.as_slice()).unwrap_or(empty),
                ),
                w_beta: push(
                    &mut slices,
                    &mut off,
                    gdn.map(|g| g.w_beta.as_slice()).unwrap_or(empty),
                ),
                w_ssm_out: push(
                    &mut slices,
                    &mut off,
                    gdn.map(|g| g.w_out.as_slice()).unwrap_or(empty),
                ),
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

    /// Llama-family GGUF → device pack. Small tensors dequant to F32; large Q8_0
    /// GEMM weights / `token_embd` stay as 36-byte staged blocks. Mixtral-style
    /// stacked experts (`ffn_*_exps`) upload and split into per-expert GEMV refs.
    pub fn upload_gguf(
        file: &allpaca_gguf::GgufFile,
        cfg: &ModelConfig,
        device: i32,
    ) -> Result<Self, AllpacaError> {
        // Recurrent GGUFs upload attention (+ MoE + GDN projections). Mamba-2
        // extras and GDN scalars/conv stay on host (`SsmLayerWeights` /
        // `GdnHostWeights`).

        let mut owned: Vec<Vec<u8>> = Vec::new();
        let mut off = 0usize;
        let push_bytes = |owned: &mut Vec<Vec<u8>>,
                          off: &mut usize,
                          data: Vec<u8>,
                          kind: super::WeightKind,
                          n: usize|
         -> WeightRef {
            if data.is_empty() || n == 0 {
                return WeightRef::f32(0, 0);
            }
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
        let try_tensor = |file: &allpaca_gguf::GgufFile, name: &str, gemv: bool| {
            if file.tensor(name).is_none() {
                Ok(None)
            } else {
                take_tensor(file, name, gemv).map(Some)
            }
        };
        let split_f32_experts =
            |w: WeightRef, n_experts: usize, what: &str| -> Result<Vec<WeightRef>, AllpacaError> {
                if n_experts == 0 || w.n == 0 || w.n % n_experts != 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "CUDA {what} n={} not divisible by {n_experts}",
                        w.n
                    ))));
                }
                let n_per = w.n / n_experts;
                let stride = n_per.saturating_mul(4);
                Ok((0..n_experts)
                    .map(|e| WeightRef::f32(w.off + e * stride, n_per))
                    .collect())
            };
        let split_q8_staged_experts =
            |w: WeightRef, n_experts: usize, what: &str| -> Result<Vec<WeightRef>, AllpacaError> {
                if n_experts == 0 || w.n == 0 || w.n % n_experts != 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "CUDA {what} n={} not divisible by {n_experts}",
                        w.n
                    ))));
                }
                let n_per = w.n / n_experts;
                if n_per % allpaca_cpu::Q8_0_BLOCK_ELEMS != 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "CUDA {what} Q8 expert elems {n_per} not aligned to {}",
                        allpaca_cpu::Q8_0_BLOCK_ELEMS
                    ))));
                }
                let stride = (n_per / allpaca_cpu::Q8_0_BLOCK_ELEMS)
                    .saturating_mul(allpaca_cuda::Q8_STAGED_BLOCK);
                Ok((0..n_experts)
                    .map(|e| WeightRef::q8_0(w.off + e * stride, n_per))
                    .collect())
            };

        let (embed_bytes, embed_kind, embed_n) = take_tensor(file, "token_embd.weight", false)?;
        let embed = push_bytes(&mut owned, &mut off, embed_bytes, embed_kind, embed_n);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for local in 0..cfg.n_layers {
            let i = cfg.gguf_blk(local);
            let take =
                |suffix: &str, gemv: bool| take_tensor(file, &format!("blk.{i}.{suffix}"), gemv);
            let (b, k, n) = take("attn_norm.weight", false)?;
            let attn_norm = push_bytes(&mut owned, &mut off, b, k, n);
            let gdn = file.tensor(&format!("blk.{i}.attn_qkv.weight")).is_some()
                && file.tensor(&format!("blk.{i}.ssm_out.weight")).is_some()
                && file.tensor(&format!("blk.{i}.attn_q.weight")).is_none();
            let (wq, wk, wv, wo) = if gdn {
                (
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                )
            } else {
                let (b, k, n) = take("attn_q.weight", true)?;
                let wq = push_bytes(&mut owned, &mut off, b, k, n);
                let (b, k, n) = take("attn_k.weight", true)?;
                let wk = push_bytes(&mut owned, &mut off, b, k, n);
                let (b, k, n) = take("attn_v.weight", true)?;
                let wv = push_bytes(&mut owned, &mut off, b, k, n);
                let (b, k, n) = take("attn_output.weight", true)?;
                let wo = push_bytes(&mut owned, &mut off, b, k, n);
                (wq, wk, wv, wo)
            };
            let ffn_norm = if let Some((b, k, n)) =
                try_tensor(file, &format!("blk.{i}.ffn_norm.weight"), false)?
            {
                push_bytes(&mut owned, &mut off, b, k, n)
            } else {
                WeightRef::f32(0, 0)
            };

            let moe = cfg.n_experts > 0
                && (file
                    .tensor(&format!("blk.{i}.ffn_gate_inp.weight"))
                    .is_some()
                    || file.tensor(&format!("blk.{i}.ffn_router.weight")).is_some()
                    || file
                        .tensor(&format!("blk.{i}.ffn_gate_exps.weight"))
                        .is_some()
                    || file
                        .tensor(&format!("blk.{i}.ffn_up_exps.weight"))
                        .is_some());

            let (w_gate, w_up, w_down, router_host, experts) = if moe {
                let n_exp = cfg.n_experts.max(1);
                let router_name = if file
                    .tensor(&format!("blk.{i}.ffn_gate_inp.weight"))
                    .is_some()
                {
                    format!("blk.{i}.ffn_gate_inp.weight")
                } else {
                    format!("blk.{i}.ffn_router.weight")
                };
                let router_host = file.tensor_f32(&router_name).map_err(|e| {
                    AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "CUDA MoE router {router_name}: {e}"
                    )))
                })?;
                let stacked_gate = format!("blk.{i}.ffn_gate_exps.weight");
                let stacked_up = format!("blk.{i}.ffn_up_exps.weight");
                let experts = if file.tensor(&stacked_gate).is_some()
                    || file.tensor(&stacked_up).is_some()
                {
                    // Stacked experts: F32 dequant or always-pack Q8_0 GEMV (K80 VRAM).
                    let push_stacked = |owned: &mut Vec<Vec<u8>>,
                                        off: &mut usize,
                                        name: &str|
                     -> Result<WeightRef, AllpacaError> {
                        let info = file.tensor(name).ok_or_else(|| {
                            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                "CUDA MoE missing {name}"
                            )))
                        })?;
                        let n = info.n_elements() as usize;
                        if info.ggml_type == allpaca_gguf::GgmlType::Q8_0 {
                            let packed = file.tensor_bytes(name).map_err(|e| {
                                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                    "CUDA MoE Q8 bytes {name}: {e}"
                                )))
                            })?;
                            let k = *info.dims.first().unwrap_or(&1) as usize;
                            if k == 0 || n % k != 0 {
                                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                    "Q8_0 MoE {name} dims {:?} do not tile n={n}",
                                    info.dims
                                ))));
                            }
                            let n_rows = n / k;
                            let staged =
                                allpaca_cuda::stage_q8_0_gemv(packed, n_rows, k).map_err(|e| {
                                    AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                        "CUDA Q8_0 MoE stage {name}: {e}"
                                    )))
                                })?;
                            Ok(push_bytes(owned, off, staged, super::WeightKind::Q8_0, n))
                        } else {
                            let f32s = file.tensor_f32(name).map_err(|e| {
                                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                    "CUDA MoE {name}: {e}"
                                )))
                            })?;
                            let n = f32s.len();
                            let mut bytes = Vec::with_capacity(n * 4);
                            for v in f32s {
                                bytes.extend_from_slice(&v.to_le_bytes());
                            }
                            Ok(push_bytes(owned, off, bytes, super::WeightKind::F32, n))
                        }
                    };
                    let gate_w = if file.tensor(&stacked_gate).is_some() {
                        Some(push_stacked(&mut owned, &mut off, &stacked_gate)?)
                    } else {
                        None
                    };
                    let up_w = push_stacked(&mut owned, &mut off, &stacked_up)?;
                    let down_w = push_stacked(
                        &mut owned,
                        &mut off,
                        &format!("blk.{i}.ffn_down_exps.weight"),
                    )?;
                    let split =
                        |w: WeightRef, what: &str| -> Result<Vec<WeightRef>, AllpacaError> {
                            match w.kind {
                                super::WeightKind::Q8_0 => split_q8_staged_experts(w, n_exp, what),
                                _ => split_f32_experts(w, n_exp, what),
                            }
                        };
                    let ups = split(up_w, "ffn_up_exps")?;
                    let downs = split(down_w, "ffn_down_exps")?;
                    let gates = if let Some(g) = gate_w {
                        split(g, "ffn_gate_exps")?
                    } else {
                        vec![WeightRef::f32(0, 0); n_exp]
                    };
                    (0..n_exp).map(|e| [gates[e], ups[e], downs[e]]).collect()
                } else {
                    let mut ex = Vec::with_capacity(n_exp);
                    for e in 0..n_exp {
                        let gate_name = format!("blk.{i}.ffn_gate.{e}.weight");
                        let w_gate = if let Some((b, k, n)) = try_tensor(file, &gate_name, true)? {
                            push_bytes(&mut owned, &mut off, b, k, n)
                        } else {
                            WeightRef::f32(0, 0)
                        };
                        let (b, k, n) =
                            take_tensor(file, &format!("blk.{i}.ffn_up.{e}.weight"), true)?;
                        let w_up = push_bytes(&mut owned, &mut off, b, k, n);
                        let (b, k, n) =
                            take_tensor(file, &format!("blk.{i}.ffn_down.{e}.weight"), true)?;
                        let w_down = push_bytes(&mut owned, &mut off, b, k, n);
                        ex.push([w_gate, w_up, w_down]);
                    }
                    ex
                };
                (
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    router_host,
                    experts,
                )
            } else if file.tensor(&format!("blk.{i}.ffn_gate.weight")).is_some() {
                let (b, k, n) = take("ffn_gate.weight", true)?;
                let w_gate = push_bytes(&mut owned, &mut off, b, k, n);
                let (b, k, n) = take("ffn_up.weight", true)?;
                let w_up = push_bytes(&mut owned, &mut off, b, k, n);
                let (b, k, n) = take("ffn_down.weight", true)?;
                let w_down = push_bytes(&mut owned, &mut off, b, k, n);
                (w_gate, w_up, w_down, Vec::new(), Vec::new())
            } else {
                (
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    Vec::new(),
                    Vec::new(),
                )
            };

            let (w_qkv, w_z, w_alpha, w_beta, w_ssm_out) = if gdn {
                let push_t = |owned: &mut Vec<Vec<u8>>,
                              off: &mut usize,
                              suffix: &str,
                              gemv: bool|
                 -> Result<WeightRef, AllpacaError> {
                    let (b, k, n) = take_tensor(file, &format!("blk.{i}.{suffix}"), gemv)?;
                    Ok(push_bytes(owned, off, b, k, n))
                };
                (
                    push_t(&mut owned, &mut off, "attn_qkv.weight", true)?,
                    push_t(&mut owned, &mut off, "attn_gate.weight", true)?,
                    push_t(&mut owned, &mut off, "ssm_alpha.weight", true)?,
                    push_t(&mut owned, &mut off, "ssm_beta.weight", true)?,
                    push_t(&mut owned, &mut off, "ssm_out.weight", true)?,
                )
            } else {
                (
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                )
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
                router_host,
                experts,
                w_qkv,
                w_z,
                w_alpha,
                w_beta,
                w_ssm_out,
            });
        }
        let (norm_bytes, norm_kind, norm_n) = take_tensor(file, "output_norm.weight", false)?;
        let norm = push_bytes(&mut owned, &mut off, norm_bytes, norm_kind, norm_n);
        let lm_head = if file.tensor("output.weight").is_some() {
            let (b, k, n) = take_tensor(file, "output.weight", true)?;
            push_bytes(&mut owned, &mut off, b, k, n)
        } else {
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

    /// True when GDN projection GEMMs are device-resident.
    pub fn has_gdn_device_projections(&self) -> bool {
        self.layers
            .iter()
            .any(|l| l.w_qkv.n > 0 && l.w_ssm_out.n > 0)
    }

    /// True when any GDN projection was staged as packed Q8_0 (n>65536).
    pub fn has_q8_gdn_projections(&self) -> bool {
        self.layers.iter().any(|l| {
            [l.w_qkv, l.w_z, l.w_alpha, l.w_beta, l.w_ssm_out]
                .iter()
                .any(|w| w.kind == super::WeightKind::Q8_0 && w.n > 0)
        })
    }

    /// True when any MoE expert GEMM weight was staged as packed Q8_0.
    pub fn has_q8_moe_experts(&self) -> bool {
        self.layers.iter().any(|l| {
            l.experts.iter().any(|e| {
                e.iter()
                    .any(|w| w.kind == super::WeightKind::Q8_0 && w.n > 0)
            })
        })
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
            super::WeightKind::F32 => cuda.gemm_nt_host_a_dev_b(a, self.ptr(w)?, c, m, n, k),
            super::WeightKind::Q8_0 => {
                cuda.gemm_nt_host_a_dev_b_q8(a, self.byte_ptr(w)?, c, m, n, k)
            }
            _ => Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "CUDA decode: unsupported weight kind for GEMM".into(),
            ))),
        }
    }

    fn gemm_dev_a(
        &self,
        cuda: &CudaWorkspace,
        a: *const f32,
        w: WeightRef,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        if a.is_null() || c.is_null() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "CUDA decode: null GEMM activation pointer".into(),
            )));
        }
        match w.kind {
            super::WeightKind::F32 => cuda.gemm_nt_dev(a, self.ptr(w)?, c, m, n, k),
            super::WeightKind::Q8_0 => cuda.gemm_nt_q8_dev(a, self.byte_ptr(w)?, c, m, n, k),
            _ => Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "CUDA decode: unsupported weight kind for device GEMM".into(),
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
    /// One-token dense / MoE / SSM-hybrid decode using CUDA-resident weights.
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
        for (i, layer) in self.layers.iter().enumerate() {
            match layer.block {
                HybridLayerKind::Attention
                | HybridLayerKind::Ssm
                | HybridLayerKind::DenseFfn
                | HybridLayerKind::Moe => {}
                other => {
                    return Err(AllpacaError::Degraded(
                        allpaca_types::DegradedError::FeatureDisabled(format!(
                            "CUDA decode: layer {i} is {other:?}; GDN later"
                        )),
                    ));
                }
            }
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
        let mut hidden = vec![0.0f32; dim];
        let acts = cuda.decode_acts(dim, kv_dim)?;

        for (layer_idx, host_layer) in self.layers.iter().enumerate() {
            if let Some(rpc) = &self.rpc {
                if rpc.is_remote(layer_idx) {
                    rpc.run_layer(layer_idx, pos, &mut x)?;
                    continue;
                }
            }
            let layer = &resident.layers[layer_idx];
            if matches!(
                host_layer.block,
                HybridLayerKind::Ssm | HybridLayerKind::DenseFfn | HybridLayerKind::Moe
            ) && layer.wq.n == 0
            {
                if host_layer.block == HybridLayerKind::Ssm {
                    if let Some(gdn) = host_layer.gdn.as_ref() {
                        apply_gdn_cuda(
                            cfg,
                            layer_idx,
                            gdn,
                            layer,
                            resident,
                            cuda,
                            &mut x,
                            &mut scratch,
                            &mut hidden,
                            recurrent.as_deref_mut(),
                            metrics,
                        )?;
                    } else {
                        let ssm = host_layer.ssm.as_ref().ok_or_else(|| {
                            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                "CUDA SSM layer {layer_idx} missing host Mamba-2 weights"
                            )))
                        })?;
                        let norm_ptr = if layer.attn_norm.n > 0 {
                            resident.ptr(layer.attn_norm)?
                        } else {
                            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                "CUDA SSM layer {layer_idx} missing attn_norm"
                            ))));
                        };
                        cuda.rmsnorm_host_x_dev_w(
                            &x,
                            norm_ptr,
                            &mut scratch[..dim],
                            dim as i32,
                            cfg.eps,
                        )?;
                        metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);
                        let state = recurrent.as_deref_mut().ok_or_else(|| {
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
                            &mut hidden,
                        )
                        .map_err(|e| {
                            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                "CUDA SSM layer {layer_idx}: {e}"
                            )))
                        })?;
                        for i in 0..dim {
                            x[i] += hidden[i];
                        }
                    }
                }
                if !host_layer.w_gate.is_empty()
                    || !host_layer.experts.is_empty()
                    || !layer.experts.is_empty()
                {
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
                continue;
            }

            if host_layer.block != HybridLayerKind::Attention {
                return Err(AllpacaError::Degraded(
                    allpaca_types::DegradedError::FeatureDisabled(format!(
                        "CUDA decode: unexpected block {:?} at layer {layer_idx}",
                        host_layer.block
                    )),
                ));
            }
            cuda.copy_h2d_f32(acts.x, &x)?;
            cuda.rmsnorm_dev(
                acts.x,
                resident.ptr(layer.attn_norm)?,
                acts.norm,
                dim as i32,
                cfg.eps,
            )?;
            metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);

            resident.gemm_dev_a(cuda, acts.norm, layer.wq, acts.q, 1, dim as i32, dim as i32)?;
            resident.gemm_dev_a(
                cuda,
                acts.norm,
                layer.wk,
                acts.k,
                1,
                kv_dim as i32,
                dim as i32,
            )?;
            resident.gemm_dev_a(
                cuda,
                acts.norm,
                layer.wv,
                acts.v,
                1,
                kv_dim as i32,
                dim as i32,
            )?;

            cuda.rope_dev(
                acts.q,
                cfg.n_heads as i32,
                head_dim as i32,
                pos as i32,
                cfg.rope_theta,
            )?;
            cuda.rope_dev(
                acts.k,
                cfg.n_kv_heads as i32,
                head_dim as i32,
                pos as i32,
                cfg.rope_theta,
            )?;
            metrics.elementwise_gpu_ops.fetch_add(1, Ordering::Relaxed);

            let page_tokens = kv
                .get(page_ids[layer_idx][0])
                .map(|p| p.capacity_tokens)
                .unwrap_or(128);
            let page_index = pos / page_tokens as usize;
            let slot = (pos % page_tokens as usize) as u32;
            let page_id = page_ids[layer_idx][page_index];
            kv.write_slot_from_device(page_id, slot, cfg.n_kv_heads, head_dim, acts.k, acts.v)?;
            kv.touch(page_id, 1.0);

            let n_kv_max = pos + 1;
            let (dest_k, dest_v) = cuda.kv_scratch_pair_after(acts.bytes, n_kv_max, kv_dim)?;
            let (k_dev, v_dev, n_kv) = kv.cuda_kv_views(
                &page_ids[layer_idx],
                cfg.n_kv_heads,
                head_dim,
                dest_k,
                dest_v,
                n_kv_max,
            )?;
            cuda.attention_dev(
                acts.q,
                k_dev,
                v_dev,
                acts.attn,
                1,
                n_kv as i32,
                cfg.n_heads as i32,
                cfg.n_kv_heads as i32,
                head_dim as i32,
                true,
            )?;

            resident.gemm_dev_a(
                cuda, acts.attn, layer.wo, acts.proj, 1, dim as i32, dim as i32,
            )?;
            cuda.download_f32(acts.proj, &mut hidden)?;
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

fn matvec_rows(weight: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(weight.len(), rows.saturating_mul(cols));
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    for r in 0..rows {
        let row = &weight[r * cols..(r + 1) * cols];
        out[r] = row.iter().zip(x.iter()).map(|(w, v)| w * v).sum();
    }
}

/// Host Gated DeltaNet (HIP fallback layout) after CUDA rmsnorm.
#[allow(clippy::too_many_arguments)]
fn apply_gdn_cuda(
    cfg: &ModelConfig,
    layer_idx: usize,
    gdn: &GdnHostWeights,
    layer: &CudaLayerOff,
    resident: &CudaResidentModel,
    cuda: &CudaWorkspace,
    x: &mut [f32],
    scratch: &mut [f32],
    hidden: &mut [f32],
    recurrent: Option<&mut RecurrentState>,
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    let dim = cfg.dim;
    let d = gdn.dims;
    let device_proj = layer.w_qkv.n > 0 && layer.w_ssm_out.n > 0;
    if !device_proj
        && (gdn.w_qkv.len() != d.qkv_dim * dim
            || gdn.w_z.len() != d.value_dim * dim
            || gdn.w_alpha.len() != d.n_v_heads * dim
            || gdn.w_beta.len() != d.n_v_heads * dim
            || gdn.w_out.len() != dim * d.value_dim)
    {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "CUDA GDN layer {layer_idx} missing host and device projections"
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

    let mut qkv = vec![0.0f32; d.qkv_dim];
    let mut z = vec![0.0f32; d.value_dim];
    let mut alpha = vec![0.0f32; d.n_v_heads];
    let mut beta = vec![0.0f32; d.n_v_heads];
    if device_proj {
        resident.gemm_host_a(
            cuda,
            &scratch[..dim],
            layer.w_qkv,
            &mut qkv,
            1,
            d.qkv_dim as i32,
            dim as i32,
        )?;
        resident.gemm_host_a(
            cuda,
            &scratch[..dim],
            layer.w_z,
            &mut z,
            1,
            d.value_dim as i32,
            dim as i32,
        )?;
        resident.gemm_host_a(
            cuda,
            &scratch[..dim],
            layer.w_alpha,
            &mut alpha,
            1,
            d.n_v_heads as i32,
            dim as i32,
        )?;
        resident.gemm_host_a(
            cuda,
            &scratch[..dim],
            layer.w_beta,
            &mut beta,
            1,
            d.n_v_heads as i32,
            dim as i32,
        )?;
    } else {
        matvec_rows(&gdn.w_qkv, d.qkv_dim, dim, &scratch[..dim], &mut qkv);
        matvec_rows(&gdn.w_z, d.value_dim, dim, &scratch[..dim], &mut z);
        matvec_rows(&gdn.w_alpha, d.n_v_heads, dim, &scratch[..dim], &mut alpha);
        matvec_rows(&gdn.w_beta, d.n_v_heads, dim, &scratch[..dim], &mut beta);
    }

    let state = recurrent.ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(
            "CUDA GDN layer requires recurrent state".into(),
        ))
    })?;
    state.ensure_gdn(layer_idx, &d);

    let mut conv_out = vec![0.0f32; d.qkv_dim];
    conv1d_step(
        &qkv,
        &gdn.conv1d,
        &mut state.gdn_conv[layer_idx],
        &mut conv_out,
        d.conv_kernel,
    );
    let q_raw = &conv_out[..d.key_dim];
    let k_raw = &conv_out[d.key_dim..d.key_dim * 2];
    let v_raw = &conv_out[d.key_dim * 2..];
    let mut q_exp = vec![0.0f32; d.n_v_heads * d.head_k];
    let mut k_exp = vec![0.0f32; d.n_v_heads * d.head_k];
    expand_k_heads(q_raw, &mut q_exp, d.n_k_heads, d.n_v_heads, d.head_k);
    expand_k_heads(k_raw, &mut k_exp, d.n_k_heads, d.n_v_heads, d.head_k);
    let mut decay = vec![0.0f32; d.n_v_heads];
    decay_from_alpha(&alpha, &gdn.dt_bias, &gdn.a, &mut decay);
    for b in &mut beta {
        *b = sigmoid(*b);
    }
    let mut y = vec![0.0f32; d.value_dim];
    gated_delta_step_eps(
        &mut q_exp,
        &mut k_exp,
        v_raw,
        &decay,
        &beta,
        &mut state.gdn_state[layer_idx],
        &mut y,
        d.n_v_heads,
        d.head_v,
        d.head_k,
        cfg.eps,
    );
    let mut gated = vec![0.0f32; d.value_dim];
    rmsnorm_silu_gate(&y, &z, &gdn.norm, &mut gated, d.head_v, cfg.eps);
    if hidden.len() < dim {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(
            "CUDA GDN hidden buffer too small".into(),
        )));
    }
    if device_proj {
        resident.gemm_host_a(
            cuda,
            &gated,
            layer.w_ssm_out,
            &mut hidden[..dim],
            1,
            dim as i32,
            d.value_dim as i32,
        )?;
    } else {
        matvec_rows(&gdn.w_out, dim, d.value_dim, &gated, &mut hidden[..dim]);
    }
    for i in 0..dim {
        x[i] += hidden[i];
    }
    Ok(())
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

    if !layer.experts.is_empty() {
        let n_exp = cfg.n_experts.max(layer.experts.len());
        let n_act = cfg.n_active.max(1).min(n_exp);
        let router = if !host.w_router.is_empty() {
            host.w_router.as_slice()
        } else if !layer.router_host.is_empty() {
            layer.router_host.as_slice()
        } else {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "CUDA MoE missing router weights".into(),
            )));
        };
        let routed = topk_route(&scratch[..dim], router, n_exp, n_act, cfg);
        let mut acc = vec![0.0f32; dim];
        let mut expert_out = vec![0.0f32; dim];
        let mut gate = vec![0.0f32; ff];
        let mut up = vec![0.0f32; ff];
        for &(e, w) in &routed {
            let Some([w_gate, w_up, w_down]) = layer.experts.get(e).copied() else {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "CUDA MoE missing expert {e}"
                ))));
            };
            resident.gemm_host_a(
                cuda,
                &scratch[..dim],
                w_up,
                &mut up,
                1,
                ff as i32,
                dim as i32,
            )?;
            if w_gate.n > 0 {
                resident.gemm_host_a(
                    cuda,
                    &scratch[..dim],
                    w_gate,
                    &mut gate,
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
                    w_down,
                    &mut expert_out,
                    1,
                    dim as i32,
                    ff as i32,
                )?;
            } else {
                for v in &mut up {
                    *v = silu(*v);
                }
                resident.gemm_host_a(
                    cuda,
                    &up,
                    w_down,
                    &mut expert_out,
                    1,
                    dim as i32,
                    ff as i32,
                )?;
            }
            for i in 0..dim {
                acc[i] += w * expert_out[i];
            }
        }
        for i in 0..dim {
            x[i] += acc[i];
        }
        return Ok(());
    }

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
    resident.gemm_host_a(cuda, &gate, layer.w_down, hidden, 1, dim as i32, ff as i32)?;
    for i in 0..dim {
        x[i] += hidden[i];
    }
    Ok(())
}
