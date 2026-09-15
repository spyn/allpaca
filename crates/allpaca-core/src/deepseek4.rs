//! DeepSeek-V4-Flash (`deepseek4`) host decode: MLA + hyper-connections + MoE.
//!
//! CSA/HCA compressors are not applied: each token stores the 512-d latent KV
//! and attends densely. That is enough for short-context load/use; long-context
//! sparsity still needs the indexer path.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use allpaca_cpu::{attention, matmul_nt, rmsnorm, rope_offset, rope_offset_back};
use allpaca_gguf::GgufFile;
use allpaca_types::{AllpacaError, FatalError};

use crate::model::{
    expert_mlp, expert_mlp_rows, gather_kv, topk_from_logits, DenseKv, ModelConfig, TinyModel,
};
use crate::telemetry::Metrics;
use crate::weight_set::LayerFfn;
use crate::KvTierManager;

#[derive(Debug, Clone)]
pub struct Dsv4Spec {
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_group_count: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: u32,
    pub hc_eps: f32,
    pub hash_layer_count: usize,
    pub rope_offset: usize,
}

impl Dsv4Spec {
    pub fn from_gguf(file: &GgufFile, cfg: &ModelConfig) -> Self {
        let arch = file.architecture().unwrap_or("deepseek4");
        let q_lora = file
            .meta_u32(&format!("{arch}.attention.q_lora_rank"))
            .unwrap_or(1024) as usize;
        let o_groups = file
            .meta_u32(&format!("{arch}.attention.output_group_count"))
            .unwrap_or(8)
            .max(1) as usize;
        let o_lora = file
            .meta_u32(&format!("{arch}.attention.output_lora_rank"))
            .unwrap_or(q_lora as u32) as usize;
        let hc = file
            .meta_u32(&format!("{arch}.hyper_connection.count"))
            .unwrap_or(1)
            .max(1) as usize;
        let head_dim = cfg.head_dim();
        let n_rot = if cfg.rope_dims > 0 {
            cfg.rope_dims
        } else {
            head_dim
        };
        Self {
            q_lora_rank: q_lora,
            o_lora_rank: o_lora,
            o_group_count: o_groups,
            hc_mult: hc,
            hc_sinkhorn_iters: file
                .meta_u32(&format!("{arch}.hyper_connection.sinkhorn_iterations"))
                .unwrap_or(20)
                .max(1),
            hc_eps: file
                .meta_f32(&format!("{arch}.hyper_connection.epsilon"))
                .unwrap_or(1e-6),
            hash_layer_count: file
                .meta_u32(&format!("{arch}.hash_layer_count"))
                .unwrap_or(0) as usize,
            rope_offset: head_dim.saturating_sub(n_rot),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MlaHost {
    pub q_a: Vec<f32>,
    pub q_a_norm: Vec<f32>,
    pub q_b: Vec<f32>,
    pub kv: Vec<f32>,
    pub kv_norm: Vec<f32>,
    pub o_a: Vec<f32>,
    pub o_b: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct HcMix {
    pub fn_w: Vec<f32>,
    pub base: Vec<f32>,
    pub scale: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Dsv4LayerMeta {
    pub sinks: Vec<f32>,
    pub hc_attn: HcMix,
    pub hc_ffn: HcMix,
    pub tid2eid: Vec<i32>,
    pub exp_probs_b: Vec<f32>,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
}

struct Dsv4Cache {
    /// One decoded MLA layer at a time (full-stack F32 MLA does not fit RAM).
    mla: Option<(usize, MlaHost)>,
    shexp: Option<(usize, LayerFfn)>,
    lm_head: Option<Vec<f32>>,
}

pub struct Dsv4Weights {
    pub spec: Dsv4Spec,
    pub layers: Vec<Dsv4LayerMeta>,
    pub hc_head_fn: Vec<f32>,
    pub hc_head_base: Vec<f32>,
    pub hc_head_scale: f32,
    cache: Mutex<Dsv4Cache>,
    pub(crate) gguf: Option<Arc<GgufFile>>,
    pub(crate) gpu: Mutex<Option<crate::dsv4_gpu::Dsv4Gpu>>,
}

impl Dsv4Weights {
    pub fn from_gguf(
        file: &GgufFile,
        cfg: &ModelConfig,
        materialize_mla: bool,
    ) -> Result<Self, AllpacaError> {
        let spec = Dsv4Spec::from_gguf(file, cfg);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let blk = cfg.gguf_blk(i);
            if materialize_mla || file.tensor(&format!("blk.{blk}.attn_q_a.weight")).is_some() {
                for suffix in [
                    "attn_q_a.weight",
                    "attn_q_b.weight",
                    "attn_kv.weight",
                    "attn_output_a.weight",
                    "attn_output_b.weight",
                ] {
                    let name = format!("blk.{blk}.{suffix}");
                    if file.tensor(&name).is_none() {
                        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                            "missing DeepSeek-V4 tensor {name}"
                        ))));
                    }
                }
            }
            layers.push(Dsv4LayerMeta {
                sinks: optional_f32(file, &format!("blk.{blk}.attn_sinks.weight"))?,
                hc_attn: load_hc_mix(file, blk, "attn")?,
                hc_ffn: load_hc_mix(file, blk, "ffn")?,
                tid2eid: if file
                    .tensor(&format!("blk.{blk}.ffn_gate_tid2eid.weight"))
                    .is_some()
                {
                    file.tensor_i32(&format!("blk.{blk}.ffn_gate_tid2eid.weight"))?
                } else {
                    Vec::new()
                },
                exp_probs_b: optional_f32(file, &format!("blk.{blk}.exp_probs_b.bias"))?,
                attn_norm: optional_f32(file, &format!("blk.{blk}.attn_norm.weight"))?,
                ffn_norm: optional_f32(file, &format!("blk.{blk}.ffn_norm.weight"))?,
            });
        }
        let _ = materialize_mla;
        let hc_head_fn = optional_f32(file, "output_hc_fn.weight")?;
        let hc_head_base = optional_f32(file, "output_hc_base.weight")?;
        let hc_head_scale = file
            .tensor_f32("output_hc_scale.weight")
            .ok()
            .and_then(|v| v.first().copied())
            .unwrap_or(1.0);
        Ok(Self {
            spec,
            layers,
            hc_head_fn,
            hc_head_base,
            hc_head_scale,
            cache: Mutex::new(Dsv4Cache {
                mla: None,
                shexp: None,
                lm_head: None,
            }),
            gguf: None,
            gpu: Mutex::new(None),
        })
    }

    pub fn attach_gguf(&mut self, file: Arc<GgufFile>) {
        self.gguf = Some(file);
    }
}

impl std::fmt::Debug for Dsv4Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dsv4Weights")
            .field("spec", &self.spec)
            .field("n_layers", &self.layers.len())
            .finish_non_exhaustive()
    }
}

fn optional_f32(file: &GgufFile, name: &str) -> Result<Vec<f32>, AllpacaError> {
    if file.tensor(name).is_some() {
        file.tensor_f32(name)
    } else {
        Ok(Vec::new())
    }
}

fn load_hc_mix(file: &GgufFile, layer: usize, kind: &str) -> Result<HcMix, AllpacaError> {
    Ok(HcMix {
        fn_w: optional_f32(file, &format!("blk.{layer}.hc_{kind}_fn.weight"))?,
        base: optional_f32(file, &format!("blk.{layer}.hc_{kind}_base.weight"))?,
        scale: optional_f32(file, &format!("blk.{layer}.hc_{kind}_scale.weight"))?,
    })
}

fn load_mla(file: &GgufFile, layer: usize) -> Result<MlaHost, AllpacaError> {
    let kv_norm = if file
        .tensor(&format!("blk.{layer}.attn_kv_a_norm.weight"))
        .is_some()
    {
        file.tensor_f32(&format!("blk.{layer}.attn_kv_a_norm.weight"))?
    } else {
        optional_f32(file, &format!("blk.{layer}.attn_kv_norm.weight"))?
    };
    Ok(MlaHost {
        q_a: file.tensor_f32(&format!("blk.{layer}.attn_q_a.weight"))?,
        q_a_norm: file.tensor_f32(&format!("blk.{layer}.attn_q_a_norm.weight"))?,
        q_b: file.tensor_f32(&format!("blk.{layer}.attn_q_b.weight"))?,
        kv: file.tensor_f32(&format!("blk.{layer}.attn_kv.weight"))?,
        kv_norm,
        o_a: file.tensor_f32(&format!("blk.{layer}.attn_output_a.weight"))?,
        o_b: file.tensor_f32(&format!("blk.{layer}.attn_output_b.weight"))?,
    })
}

fn load_shexp(file: &GgufFile, layer: usize) -> Result<LayerFfn, AllpacaError> {
    Ok(LayerFfn {
        ffn_norm: Vec::new(),
        w_gate: file.tensor_f32(&format!("blk.{layer}.ffn_gate_shexp.weight"))?,
        w_up: file.tensor_f32(&format!("blk.{layer}.ffn_up_shexp.weight"))?,
        w_down: file.tensor_f32(&format!("blk.{layer}.ffn_down_shexp.weight"))?,
    })
}

pub fn file_is_deepseek4(file: &GgufFile) -> bool {
    match file.architecture() {
        Some("deepseek4") | Some("deepseek_v4") => true,
        Some("dflash" | "dspark" | "deepseek_v4_flash_dspark_draft") => false,
        _ => file.tensor("blk.0.attn_q_a.weight").is_some(),
    }
}

fn sqrt_softplus(x: f32) -> f32 {
    let sp = if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    };
    sp.max(0.0).sqrt()
}

fn sigmoid(v: f32) -> f32 {
    if v >= 0.0 {
        let z = (-v).exp();
        1.0 / (1.0 + z)
    } else {
        let z = v.exp();
        z / (1.0 + z)
    }
}

/// HC pre: mix `hc` residual streams into one `dim` activation.
fn hc_pre(
    streams: &[Vec<f32>],
    mix: &HcMix,
    spec: &Dsv4Spec,
    dim: usize,
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), AllpacaError> {
    let hc = spec.hc_mult.max(1);
    if mix.fn_w.is_empty() || streams.len() != hc {
        let mean = mean_streams(streams, dim);
        return Ok((mean, vec![1.0; hc], vec![0.0; hc * hc]));
    }
    let hc_dim = dim * hc;
    let mut flat = vec![0.0f32; hc_dim];
    for (s, row) in streams.iter().enumerate() {
        let n = row.len().min(dim);
        flat[s * dim..s * dim + n].copy_from_slice(&row[..n]);
    }
    let mut normed = vec![0.0f32; hc_dim];
    rmsnorm(&flat, &vec![1.0; hc_dim], eps, &mut normed)?;
    let mix_dim = (2 + hc) * hc;
    let mut mixes = vec![0.0f32; mix_dim];
    matmul_nt(&normed, &mix.fn_w, &mut mixes, 1, mix_dim, hc_dim)?;
    let scale_pre = mix.scale.first().copied().unwrap_or(1.0);
    let scale_post = mix.scale.get(1).copied().unwrap_or(1.0);
    let scale_comb = mix.scale.get(2).copied().unwrap_or(1.0);
    let mut pre = vec![0.0f32; hc];
    let mut post = vec![0.0f32; hc];
    for i in 0..hc {
        let b = mix.base.get(i).copied().unwrap_or(0.0);
        pre[i] = sigmoid(mixes[i] * scale_pre + b) + spec.hc_eps;
        let b2 = mix.base.get(hc + i).copied().unwrap_or(0.0);
        post[i] = sigmoid(mixes[hc + i] * scale_post + b2) * 2.0;
    }
    let mut comb = vec![0.0f32; hc * hc];
    for i in 0..hc * hc {
        let b = mix.base.get(2 * hc + i).copied().unwrap_or(0.0);
        comb[i] = mixes[2 * hc + i] * scale_comb + b;
    }
    sinkhorn(&mut comb, hc, spec.hc_sinkhorn_iters, spec.hc_eps);
    let mut out = vec![0.0f32; dim];
    for (s, row) in streams.iter().enumerate() {
        let w = pre[s];
        for i in 0..dim.min(row.len()) {
            out[i] += row[i] * w;
        }
    }
    Ok((out, post, comb))
}

fn hc_post(
    x: &[f32],
    residual: &[Vec<f32>],
    post: &[f32],
    comb: &[f32],
    spec: &Dsv4Spec,
    dim: usize,
) -> Vec<Vec<f32>> {
    let hc = spec.hc_mult.max(1);
    let mut out = vec![vec![0.0f32; dim]; hc];
    for dst in 0..hc {
        let pw = post.get(dst).copied().unwrap_or(1.0);
        for i in 0..dim.min(x.len()) {
            out[dst][i] = x[i] * pw;
        }
        for src in 0..hc {
            let w = comb.get(dst + src * hc).copied().unwrap_or(0.0);
            if let Some(row) = residual.get(src) {
                for i in 0..dim.min(row.len()) {
                    out[dst][i] += row[i] * w;
                }
            }
        }
    }
    out
}

fn hc_head(
    streams: &[Vec<f32>],
    fn_w: &[f32],
    base: &[f32],
    scale: f32,
    spec: &Dsv4Spec,
    dim: usize,
    eps: f32,
) -> Result<Vec<f32>, AllpacaError> {
    let hc = spec.hc_mult.max(1);
    if fn_w.is_empty() {
        return Ok(mean_streams(streams, dim));
    }
    let hc_dim = dim * hc;
    let mut flat = vec![0.0f32; hc_dim];
    for (s, row) in streams.iter().enumerate() {
        let n = row.len().min(dim);
        flat[s * dim..s * dim + n].copy_from_slice(&row[..n]);
    }
    let mut normed = vec![0.0f32; hc_dim];
    rmsnorm(&flat, &vec![1.0; hc_dim], eps, &mut normed)?;
    let mut mixes = vec![0.0f32; hc];
    matmul_nt(&normed, fn_w, &mut mixes, 1, hc, hc_dim)?;
    let mut pre = vec![0.0f32; hc];
    for i in 0..hc {
        let b = base.get(i).copied().unwrap_or(0.0);
        pre[i] = sigmoid(mixes[i] * scale + b) + spec.hc_eps;
    }
    let mut out = vec![0.0f32; dim];
    for (s, row) in streams.iter().enumerate() {
        let w = pre[s];
        for i in 0..dim.min(row.len()) {
            out[i] += row[i] * w;
        }
    }
    Ok(out)
}

fn mean_streams(streams: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    if streams.is_empty() {
        return out;
    }
    let inv = 1.0 / streams.len() as f32;
    for row in streams {
        for i in 0..dim.min(row.len()) {
            out[i] += row[i] * inv;
        }
    }
    out
}

fn sinkhorn(comb: &mut [f32], hc: usize, iters: u32, eps: f32) {
    // softmax over dest (rows) for each src column? llama.cpp softmax on [dst, src]
    // then col-norm, then (iters-1)×(row-norm, col-norm).
    for src in 0..hc {
        let mut max = f32::NEG_INFINITY;
        for dst in 0..hc {
            max = max.max(comb[dst + src * hc]);
        }
        let mut sum = 0.0f32;
        for dst in 0..hc {
            let e = (comb[dst + src * hc] - max).exp();
            comb[dst + src * hc] = e;
            sum += e;
        }
        let sum = sum.max(1e-9);
        for dst in 0..hc {
            comb[dst + src * hc] = comb[dst + src * hc] / sum + eps;
        }
    }
    let norm_cols = |c: &mut [f32]| {
        for src in 0..hc {
            let mut sum = eps;
            for dst in 0..hc {
                sum += c[dst + src * hc];
            }
            for dst in 0..hc {
                c[dst + src * hc] /= sum;
            }
        }
    };
    let norm_rows = |c: &mut [f32]| {
        for dst in 0..hc {
            let mut sum = eps;
            for src in 0..hc {
                sum += c[dst + src * hc];
            }
            for src in 0..hc {
                c[dst + src * hc] /= sum;
            }
        }
    };
    norm_cols(comb);
    for _ in 1..iters {
        norm_rows(comb);
        norm_cols(comb);
    }
}

fn timed_rmsnorm(
    x: &[f32],
    w: &[f32],
    eps: f32,
    out: &mut [f32],
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    let t = Instant::now();
    let r = rmsnorm(x, w, eps, out);
    metrics.dsv4_norm_ns.fetch_add(
        t.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    r
}

fn attention_sinks(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sinks: &[f32],
    out: &mut [f32],
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<(), AllpacaError> {
    if sinks.is_empty() {
        return attention(q, k, v, out, 1, n_kv, n_heads, n_kv_heads, head_dim, true);
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_dim = n_kv_heads * head_dim;
    out.fill(0.0);
    for h in 0..n_heads {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let kv_h = (h * n_kv_heads) / n_heads.max(1);
        let sink = sinks.get(h).copied().unwrap_or(0.0);
        let mut scores = vec![0.0f32; n_kv];
        let mut max = sink;
        for t in 0..n_kv {
            let kh = &k[t * kv_dim + kv_h * head_dim..t * kv_dim + (kv_h + 1) * head_dim];
            let mut s = 0.0f32;
            for i in 0..head_dim {
                s += qh[i] * kh[i];
            }
            s *= scale;
            scores[t] = s;
            if s > max {
                max = s;
            }
        }
        let mut acc = (sink - max).exp();
        for s in &mut scores {
            *s = (*s - max).exp();
            acc += *s;
        }
        let inv = 1.0 / acc.max(1e-9);
        let oh = &mut out[h * head_dim..(h + 1) * head_dim];
        for t in 0..n_kv {
            let w = scores[t] * inv;
            let vh = &v[t * kv_dim + kv_h * head_dim..t * kv_dim + (kv_h + 1) * head_dim];
            for i in 0..head_dim {
                oh[i] += w * vh[i];
            }
        }
    }
    Ok(())
}

fn gemv(x: &[f32], w: &[f32], out: &mut [f32], n: usize, k: usize) -> Result<(), AllpacaError> {
    matmul_nt(x, w, out, 1, n, k)
}

fn gather_embed(file: &GgufFile, token: u32, dim: usize) -> Result<Vec<f32>, AllpacaError> {
    let info = file.tensor("token_embd.weight").ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad("missing token_embd.weight".into()))
    })?;
    let n_vocab = if info.dims.len() >= 2 {
        info.dims[1] as usize
    } else {
        info.n_elements() as usize / dim.max(1)
    };
    if token as usize >= n_vocab {
        return Err(AllpacaError::Fatal(FatalError::Tokenizer(format!(
            "token {token} out of vocab {n_vocab}"
        ))));
    }
    let bytes = file.tensor_bytes_of(info)?;
    let mut out = vec![0.0f32; dim];
    match info.ggml_type {
        allpaca_gguf::GgmlType::BF16 => {
            let off = token as usize * dim * 2;
            if off + dim * 2 > bytes.len() {
                return Err(AllpacaError::Fatal(FatalError::CorruptGguf(
                    "embed BF16 row OOB".into(),
                )));
            }
            for i in 0..dim {
                let h = u16::from_le_bytes([bytes[off + i * 2], bytes[off + i * 2 + 1]]);
                out[i] = allpaca_cpu::bf16_to_f32(h);
            }
        }
        allpaca_gguf::GgmlType::F16 => {
            let off = token as usize * dim * 2;
            for i in 0..dim {
                let h = u16::from_le_bytes([bytes[off + i * 2], bytes[off + i * 2 + 1]]);
                out[i] = allpaca_cpu::f16_to_f32(h);
            }
        }
        allpaca_gguf::GgmlType::F32 => {
            let off = token as usize * dim * 4;
            for i in 0..dim {
                let o = off + i * 4;
                out[i] = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
            }
        }
        _ => {
            let all = file.tensor_f32("token_embd.weight")?;
            let start = token as usize * dim;
            out.copy_from_slice(&all[start..start + dim]);
        }
    }
    Ok(out)
}

fn with_mla<R>(
    dsv4: &Dsv4Weights,
    layer: usize,
    blk: usize,
    norms_only: bool,
    f: impl FnOnce(&MlaHost) -> Result<R, AllpacaError>,
) -> Result<R, AllpacaError> {
    let mut cache = dsv4
        .cache
        .lock()
        .map_err(|_| AllpacaError::Fatal(FatalError::ModelLoad("dsv4 cache poisoned".into())))?;
    let hit = cache
        .mla
        .as_ref()
        .is_some_and(|(i, m)| *i == layer && (!norms_only || m.q_a.is_empty()));
    if !hit {
        let file = dsv4.gguf.as_ref().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(
                "DeepSeek-V4 MLA not materialized and no GGUF handle".into(),
            ))
        })?;
        cache.mla = Some((
            layer,
            if norms_only {
                load_mla_norms(file, blk)?
            } else {
                load_mla(file, blk)?
            },
        ));
    }
    let mla = &cache.mla.as_ref().expect("MLA just loaded").1;
    f(mla)
}

fn load_mla_norms(file: &GgufFile, layer: usize) -> Result<MlaHost, AllpacaError> {
    let kv_norm = if file
        .tensor(&format!("blk.{layer}.attn_kv_a_norm.weight"))
        .is_some()
    {
        file.tensor_f32(&format!("blk.{layer}.attn_kv_a_norm.weight"))?
    } else {
        optional_f32(file, &format!("blk.{layer}.attn_kv_norm.weight"))?
    };
    Ok(MlaHost {
        q_a: Vec::new(),
        q_a_norm: file
            .tensor_f32(&format!("blk.{layer}.attn_q_a_norm.weight"))
            .unwrap_or_default(),
        q_b: Vec::new(),
        kv: Vec::new(),
        kv_norm,
        o_a: Vec::new(),
        o_b: Vec::new(),
    })
}

fn dsv4_gemv(
    gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
    name: &str,
    x: &[f32],
    cpu_w: &[f32],
    out: &mut [f32],
    n: usize,
    k: usize,
    elem_off: usize,
    metrics: Option<&Metrics>,
) -> Result<(), AllpacaError> {
    dsv4_gemm(gpu, file, name, x, cpu_w, out, n, k, elem_off, 1, metrics)
}

#[allow(clippy::too_many_arguments)]
fn dsv4_gemm(
    gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
    name: &str,
    x: &[f32],
    cpu_w: &[f32],
    out: &mut [f32],
    n: usize,
    k: usize,
    elem_off: usize,
    m: usize,
    metrics: Option<&Metrics>,
) -> Result<(), AllpacaError> {
    if m == 0 {
        return Ok(());
    }
    if let (Some(gpu), Some(file)) = (gpu, file) {
        if gpu
            .gemm_bf16(file, name, x, out, n, k, elem_off, m, metrics)
            .is_ok()
        {
            return Ok(());
        }
    }
    if cpu_w.len() >= n.saturating_mul(k) {
        return matmul_nt(x, cpu_w, out, m, n, k);
    }
    if let Some(file) = file {
        if file.tensor(name).is_some() {
            let all = file.tensor_f32(name)?;
            let start = elem_off;
            let end = start.saturating_add(n.saturating_mul(k));
            if end <= all.len() {
                return matmul_nt(x, &all[start..end], out, m, n, k);
            }
        }
    }
    Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
        "no weights for {name}"
    ))))
}

struct Dsv4FfnPrep {
    fcur: Vec<f32>,
    fpost: Vec<f32>,
    fcomb: Vec<f32>,
    picks: Vec<(usize, f32)>,
}

#[allow(clippy::too_many_arguments)]
fn dsv4_attn_and_route(
    model: &TinyModel,
    dsv4: &Dsv4Weights,
    streams: &[Vec<f32>],
    token: u32,
    pos: usize,
    layer_idx: usize,
    page_ids: &[Vec<u64>],
    kv: &mut KvTierManager,
    metrics: &Metrics,
    gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
    hip: Option<&allpaca_hip::HipWorkspace>,
    skip_mla: bool,
    prefill: bool,
) -> Result<(Vec<Vec<f32>>, Dsv4FfnPrep), AllpacaError> {
    let cfg = &model.config;
    let spec = &dsv4.spec;
    let dim = cfg.dim;
    let eps = cfg.eps;
    let head_dim = cfg.head_dim();
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads.max(1);
    let kv_dim = n_kv_heads * head_dim;
    let q_out = n_heads * head_dim;
    let meta = &dsv4.layers[layer_idx];
    let host = &model.layers[layer_idx];
    let mut gpu = gpu;
    let t_hc = Instant::now();
    let (mut cur, post, comb) = hc_pre(streams, &meta.hc_attn, spec, dim, eps)?;
    metrics
        .dsv4_hc_ns
        .fetch_add(t_hc.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let attn_norm = if !meta.attn_norm.is_empty() {
        &meta.attn_norm
    } else {
        &host.attn_norm
    };
    if !attn_norm.is_empty() {
        let mut n = vec![0.0f32; dim];
        timed_rmsnorm(&cur, attn_norm, eps, &mut n, metrics)?;
        cur = n;
    }
    let blk = cfg.gguf_blk(layer_idx);
    if crate::dsv4_gpu::want_full_layer_prefetch(prefill, cfg) {
        if let (Some(g), Some(f)) = (gpu.as_mut(), file) {
            let _ = g.prefetch_layer_experts(f, blk, cfg, Some(metrics));
        }
    }
    let attn_out = with_mla(dsv4, layer_idx, blk, skip_mla, |mla| {
        let mut q = vec![0.0f32; q_out];
        let mut used_gpu_q = false;
        if let (Some(g), Some(f)) = (gpu.as_deref_mut(), file) {
            if let Ok(gq) = g.mla_q_proj(
                f,
                blk,
                &cur,
                &mla.q_a_norm,
                eps,
                spec.q_lora_rank,
                q_out,
                dim,
                Some(metrics),
            ) {
                q = gq;
                used_gpu_q = true;
            }
        }
        if !used_gpu_q {
            let mut qr = vec![0.0f32; spec.q_lora_rank];
            dsv4_gemv(
                gpu.as_deref_mut(),
                file,
                &format!("blk.{blk}.attn_q_a.weight"),
                &cur,
                &mla.q_a,
                &mut qr,
                spec.q_lora_rank,
                dim,
                0,
                Some(metrics),
            )?;
            if !mla.q_a_norm.is_empty() {
                let mut n = vec![0.0f32; spec.q_lora_rank];
                timed_rmsnorm(&qr, &mla.q_a_norm, eps, &mut n, metrics)?;
                qr = n;
            }
            dsv4_gemv(
                gpu.as_deref_mut(),
                file,
                &format!("blk.{blk}.attn_q_b.weight"),
                &qr,
                &mla.q_b,
                &mut q,
                q_out,
                spec.q_lora_rank,
                0,
                Some(metrics),
            )?;
        }
        for h in 0..n_heads {
            let row = &mut q[h * head_dim..(h + 1) * head_dim];
            let mut n = vec![0.0f32; head_dim];
            timed_rmsnorm(row, &vec![1.0; head_dim], eps, &mut n, metrics)?;
            row.copy_from_slice(&n);
            rope_offset(row, pos, cfg.rope_theta, cfg.rope_dims, spec.rope_offset)?;
        }
        let mut kv_lat = vec![0.0f32; head_dim];
        let mut used_gpu_kv = false;
        if let (Some(g), Some(f)) = (gpu.as_deref_mut(), file) {
            if let Ok(kv_g) = g.mla_kv_proj(
                f,
                blk,
                &cur,
                &mla.kv_norm,
                eps,
                head_dim,
                dim,
                Some(metrics),
            ) {
                kv_lat = kv_g;
                used_gpu_kv = true;
            }
        }
        if !used_gpu_kv {
            dsv4_gemv(
                gpu.as_deref_mut(),
                file,
                &format!("blk.{blk}.attn_kv.weight"),
                &cur,
                &mla.kv,
                &mut kv_lat,
                head_dim,
                dim,
                0,
                Some(metrics),
            )?;
            if !mla.kv_norm.is_empty() {
                let mut n = vec![0.0f32; head_dim];
                timed_rmsnorm(&kv_lat, &mla.kv_norm, eps, &mut n, metrics)?;
                kv_lat = n;
            }
        }
        rope_offset(
            &mut kv_lat,
            pos,
            cfg.rope_theta,
            cfg.rope_dims,
            spec.rope_offset,
        )?;
        let page_tokens = kv
            .get(page_ids[layer_idx][0])
            .map(|p| p.capacity_tokens)
            .unwrap_or(128);
        let page_index = pos / page_tokens as usize;
        let slot = (pos % page_tokens as usize) as u32;
        let page_id = page_ids[layer_idx][page_index];
        kv.write_slot(page_id, slot, n_kv_heads, head_dim, &kv_lat, &kv_lat)?;
        kv.touch(page_id, 1.0);
        let (k_all, v_all) = gather_kv(kv, &page_ids[layer_idx], n_kv_heads, head_dim)?;
        let n_kv = k_all.len() / kv_dim.max(1);
        let mut attn_raw = vec![0.0f32; q_out];
        let t_attn = Instant::now();
        let mut gpu_attn = false;
        if skip_mla {
            if let Some(g) = gpu.as_deref_mut() {
                if g.attention_sinks(
                    &q,
                    &k_all,
                    &v_all,
                    &meta.sinks,
                    &mut attn_raw,
                    1,
                    n_kv as i32,
                    n_heads as i32,
                    n_kv_heads as i32,
                    head_dim as i32,
                    true,
                )
                .is_ok()
                {
                    gpu_attn = true;
                }
            } else if let Some(ws) = hip {
                if ws
                    .attention_sinks(
                        &q,
                        &k_all,
                        &v_all,
                        &meta.sinks,
                        &mut attn_raw,
                        1,
                        n_kv as i32,
                        n_heads as i32,
                        n_kv_heads as i32,
                        head_dim as i32,
                        true,
                    )
                    .is_ok()
                {
                    gpu_attn = true;
                }
            }
        }
        if !gpu_attn {
            attention_sinks(
                &q,
                &k_all,
                &v_all,
                &meta.sinks,
                &mut attn_raw,
                n_kv,
                n_heads,
                n_kv_heads,
                head_dim,
            )?;
        }
        metrics
            .dsv4_attn_ns
            .fetch_add(t_attn.elapsed().as_nanos() as u64, Ordering::Relaxed);
        for h in 0..n_heads {
            let row = &mut attn_raw[h * head_dim..(h + 1) * head_dim];
            rope_offset_back(row, pos, cfg.rope_theta, cfg.rope_dims, spec.rope_offset)?;
        }
        let groups = spec.o_group_count.max(1);
        let heads_g = (n_heads / groups).max(1);
        let o_group_dim = heads_g * head_dim;
        let oa_out = spec.o_lora_rank * groups;
        let per = spec.o_lora_rank.max(1);
        let mut oa = vec![0.0f32; oa_out];
        for g in 0..groups {
            let mut slice = vec![0.0f32; o_group_dim];
            for hg in 0..heads_g {
                let src = (g * heads_g + hg) * head_dim;
                let dst = hg * head_dim;
                if src + head_dim <= attn_raw.len() && dst + head_dim <= slice.len() {
                    slice[dst..dst + head_dim].copy_from_slice(&attn_raw[src..src + head_dim]);
                }
            }
            let w_off = g.saturating_mul(per).saturating_mul(o_group_dim);
            let w_end = w_off + per * o_group_dim;
            if !mla.o_a.is_empty() && w_end > mla.o_a.len() {
                return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                    "attn_output_a grouped slice OOB".into(),
                )));
            }
            let cpu = if mla.o_a.is_empty() {
                &[][..]
            } else {
                &mla.o_a[w_off..w_end]
            };
            dsv4_gemv(
                gpu.as_deref_mut(),
                file,
                &format!("blk.{blk}.attn_output_a.weight"),
                &slice,
                cpu,
                &mut oa[g * per..(g + 1) * per],
                per,
                o_group_dim,
                w_off,
                Some(metrics),
            )?;
        }
        let mut attn_out = vec![0.0f32; dim];
        dsv4_gemv(
            gpu.as_deref_mut(),
            file,
            &format!("blk.{blk}.attn_output_b.weight"),
            &oa,
            &mla.o_b,
            &mut attn_out,
            dim,
            oa_out,
            0,
            Some(metrics),
        )?;
        Ok(attn_out)
    })?;
    let t_hc = Instant::now();
    let streams = hc_post(&attn_out, streams, &post, &comb, spec, dim);
    metrics
        .dsv4_hc_ns
        .fetch_add(t_hc.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let t_hc = Instant::now();
    let (mut fcur, fpost, fcomb) = hc_pre(&streams, &meta.hc_ffn, spec, dim, eps)?;
    metrics
        .dsv4_hc_ns
        .fetch_add(t_hc.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let ffn_norm = if !meta.ffn_norm.is_empty() {
        &meta.ffn_norm
    } else {
        &host.ffn_norm
    };
    if !ffn_norm.is_empty() {
        let mut n = vec![0.0f32; dim];
        timed_rmsnorm(&fcur, ffn_norm, eps, &mut n, metrics)?;
        fcur = n;
    }
    let t_route = Instant::now();
    let picks = route_experts(
        &fcur,
        host,
        meta,
        cfg,
        spec,
        token,
        layer_idx,
        gpu.as_deref_mut(),
        file,
        Some(metrics),
    )?;
    metrics
        .dsv4_router_ns
        .fetch_add(t_route.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok((
        streams,
        Dsv4FfnPrep {
            fcur,
            fpost,
            fcomb,
            picks,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn dsv4_mla_q_rows(
    gpu: &mut Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
    mla: &MlaHost,
    curs: &[f32],
    blk: usize,
    n_tok: usize,
    q_lora: usize,
    q_out: usize,
    dim: usize,
    eps: f32,
    metrics: &Metrics,
) -> Result<Vec<f32>, AllpacaError> {
    let mut qr = vec![0.0f32; n_tok.saturating_mul(q_lora)];
    dsv4_gemm(
        gpu.as_deref_mut(),
        file,
        &format!("blk.{blk}.attn_q_a.weight"),
        curs,
        &mla.q_a,
        &mut qr,
        q_lora,
        dim,
        0,
        n_tok,
        Some(metrics),
    )?;
    if !mla.q_a_norm.is_empty() {
        for t in 0..n_tok {
            let row = &qr[t * q_lora..(t + 1) * q_lora];
            let mut n = vec![0.0f32; q_lora];
            timed_rmsnorm(row, &mla.q_a_norm, eps, &mut n, metrics)?;
            qr[t * q_lora..(t + 1) * q_lora].copy_from_slice(&n);
        }
    }
    let mut q = vec![0.0f32; n_tok.saturating_mul(q_out)];
    dsv4_gemm(
        gpu.as_deref_mut(),
        file,
        &format!("blk.{blk}.attn_q_b.weight"),
        &qr,
        &mla.q_b,
        &mut q,
        q_out,
        q_lora,
        0,
        n_tok,
        Some(metrics),
    )?;
    Ok(q)
}

#[allow(clippy::too_many_arguments)]
fn dsv4_mla_kv_rows(
    gpu: &mut Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
    mla: &MlaHost,
    curs: &[f32],
    blk: usize,
    n_tok: usize,
    head_dim: usize,
    dim: usize,
    eps: f32,
    metrics: &Metrics,
) -> Result<Vec<f32>, AllpacaError> {
    let mut kv = vec![0.0f32; n_tok.saturating_mul(head_dim)];
    dsv4_gemm(
        gpu.as_deref_mut(),
        file,
        &format!("blk.{blk}.attn_kv.weight"),
        curs,
        &mla.kv,
        &mut kv,
        head_dim,
        dim,
        0,
        n_tok,
        Some(metrics),
    )?;
    if !mla.kv_norm.is_empty() {
        for t in 0..n_tok {
            let row = &kv[t * head_dim..(t + 1) * head_dim];
            let mut n = vec![0.0f32; head_dim];
            timed_rmsnorm(row, &mla.kv_norm, eps, &mut n, metrics)?;
            kv[t * head_dim..(t + 1) * head_dim].copy_from_slice(&n);
        }
    }
    Ok(kv)
}

#[allow(clippy::too_many_arguments)]
fn dsv4_mla_o_rows(
    gpu: &mut Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
    mla: &MlaHost,
    attn_raws: &[f32],
    blk: usize,
    n_tok: usize,
    q_out: usize,
    n_heads: usize,
    head_dim: usize,
    spec: &Dsv4Spec,
    dim: usize,
    metrics: &Metrics,
) -> Result<Vec<f32>, AllpacaError> {
    let groups = spec.o_group_count.max(1);
    let heads_g = (n_heads / groups).max(1);
    let o_group_dim = heads_g.saturating_mul(head_dim);
    let per = spec.o_lora_rank.max(1);
    let oa_out = per.saturating_mul(groups);
    let mut oa = vec![0.0f32; n_tok.saturating_mul(oa_out)];
    for g in 0..groups {
        let mut xs = vec![0.0f32; n_tok.saturating_mul(o_group_dim)];
        for t in 0..n_tok {
            let src = t
                .saturating_mul(q_out)
                .saturating_add(g.saturating_mul(o_group_dim));
            let dst = t.saturating_mul(o_group_dim);
            if src + o_group_dim <= attn_raws.len() && dst + o_group_dim <= xs.len() {
                xs[dst..dst + o_group_dim].copy_from_slice(&attn_raws[src..src + o_group_dim]);
            }
        }
        let w_off = g.saturating_mul(per).saturating_mul(o_group_dim);
        let w_end = w_off + per * o_group_dim;
        if !mla.o_a.is_empty() && w_end > mla.o_a.len() {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "attn_output_a grouped slice OOB".into(),
            )));
        }
        let cpu = if mla.o_a.is_empty() {
            &[][..]
        } else {
            &mla.o_a[w_off..w_end]
        };
        let mut ys = vec![0.0f32; n_tok.saturating_mul(per)];
        dsv4_gemm(
            gpu.as_deref_mut(),
            file,
            &format!("blk.{blk}.attn_output_a.weight"),
            &xs,
            cpu,
            &mut ys,
            per,
            o_group_dim,
            w_off,
            n_tok,
            Some(metrics),
        )?;
        for t in 0..n_tok {
            let src = t.saturating_mul(per);
            let dst = t
                .saturating_mul(oa_out)
                .saturating_add(g.saturating_mul(per));
            oa[dst..dst + per].copy_from_slice(&ys[src..src + per]);
        }
    }
    let mut attn_out = vec![0.0f32; n_tok.saturating_mul(dim)];
    dsv4_gemm(
        gpu.as_deref_mut(),
        file,
        &format!("blk.{blk}.attn_output_b.weight"),
        &oa,
        &mla.o_b,
        &mut attn_out,
        dim,
        oa_out,
        0,
        n_tok,
        Some(metrics),
    )?;
    Ok(attn_out)
}

#[allow(clippy::too_many_arguments)]
fn dsv4_attn_and_route_rows(
    model: &TinyModel,
    dsv4: &Dsv4Weights,
    states: &mut [Vec<Vec<f32>>],
    tokens: &[u32],
    start_pos: usize,
    layer_idx: usize,
    page_ids: &[Vec<u64>],
    kv: &mut KvTierManager,
    metrics: &Metrics,
    gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
    hip: Option<&allpaca_hip::HipWorkspace>,
    skip_mla: bool,
    prefill: bool,
) -> Result<Vec<Dsv4FfnPrep>, AllpacaError> {
    let n_tok = tokens.len();
    if n_tok == 0 {
        return Ok(Vec::new());
    }
    if n_tok != states.len() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dsv4 chunk attn token count mismatch".into(),
        )));
    }
    let cfg = &model.config;
    let spec = &dsv4.spec;
    let dim = cfg.dim;
    let eps = cfg.eps;
    let head_dim = cfg.head_dim();
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads.max(1);
    let kv_dim = n_kv_heads.saturating_mul(head_dim);
    let q_out = n_heads.saturating_mul(head_dim);
    let meta = &dsv4.layers[layer_idx];
    let host = &model.layers[layer_idx];
    let mut gpu = gpu;
    let blk = cfg.gguf_blk(layer_idx);
    if crate::dsv4_gpu::want_full_layer_prefetch(prefill, cfg) {
        if let (Some(g), Some(f)) = (gpu.as_mut(), file) {
            let _ = g.prefetch_layer_experts(f, blk, cfg, Some(metrics));
        }
    }
    let attn_norm = if !meta.attn_norm.is_empty() {
        &meta.attn_norm
    } else {
        &host.attn_norm
    };
    let mut curs = vec![0.0f32; n_tok.saturating_mul(dim)];
    let mut posts = Vec::with_capacity(n_tok);
    let mut combs = Vec::with_capacity(n_tok);
    let t_hc = Instant::now();
    for t in 0..n_tok {
        let (mut cur, post, comb) = hc_pre(&states[t], &meta.hc_attn, spec, dim, eps)?;
        if !attn_norm.is_empty() {
            let mut n = vec![0.0f32; dim];
            timed_rmsnorm(&cur, attn_norm, eps, &mut n, metrics)?;
            cur = n;
        }
        let off = t.saturating_mul(dim);
        let n = cur.len().min(dim);
        curs[off..off + n].copy_from_slice(&cur[..n]);
        posts.push(post);
        combs.push(comb);
    }
    metrics
        .dsv4_hc_ns
        .fetch_add(t_hc.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let attn_outs = with_mla(dsv4, layer_idx, blk, skip_mla, |mla| {
        let mut q = dsv4_mla_q_rows(
            &mut gpu,
            file,
            mla,
            &curs,
            blk,
            n_tok,
            spec.q_lora_rank,
            q_out,
            dim,
            eps,
            metrics,
        )?;
        let ones = vec![1.0f32; head_dim];
        for t in 0..n_tok {
            let pos = start_pos + t;
            let qrow = &mut q[t * q_out..(t + 1) * q_out];
            for h in 0..n_heads {
                let row = &mut qrow[h * head_dim..(h + 1) * head_dim];
                let mut n = vec![0.0f32; head_dim];
                timed_rmsnorm(row, &ones, eps, &mut n, metrics)?;
                row.copy_from_slice(&n);
                rope_offset(row, pos, cfg.rope_theta, cfg.rope_dims, spec.rope_offset)?;
            }
        }
        let mut kv_lat = dsv4_mla_kv_rows(
            &mut gpu, file, mla, &curs, blk, n_tok, head_dim, dim, eps, metrics,
        )?;
        let mut attn_raws = vec![0.0f32; n_tok.saturating_mul(q_out)];
        for t in 0..n_tok {
            let pos = start_pos + t;
            let kv_row = &mut kv_lat[t * head_dim..(t + 1) * head_dim];
            rope_offset(kv_row, pos, cfg.rope_theta, cfg.rope_dims, spec.rope_offset)?;
            let page_tokens = kv
                .get(page_ids[layer_idx][0])
                .map(|p| p.capacity_tokens)
                .unwrap_or(128);
            let page_index = pos / page_tokens as usize;
            let slot = (pos % page_tokens as usize) as u32;
            let page_id = page_ids[layer_idx][page_index];
            kv.write_slot(page_id, slot, n_kv_heads, head_dim, kv_row, kv_row)?;
            kv.touch(page_id, 1.0);
            let (k_all, v_all) = gather_kv(kv, &page_ids[layer_idx], n_kv_heads, head_dim)?;
            let n_kv = k_all.len() / kv_dim.max(1);
            let qrow = &q[t * q_out..(t + 1) * q_out];
            let attn_raw = &mut attn_raws[t * q_out..(t + 1) * q_out];
            let t_attn = Instant::now();
            let mut gpu_attn = false;
            if skip_mla {
                if let Some(g) = gpu.as_deref_mut() {
                    if g.attention_sinks(
                        qrow,
                        &k_all,
                        &v_all,
                        &meta.sinks,
                        attn_raw,
                        1,
                        n_kv as i32,
                        n_heads as i32,
                        n_kv_heads as i32,
                        head_dim as i32,
                        true,
                    )
                    .is_ok()
                    {
                        gpu_attn = true;
                    }
                } else if let Some(ws) = hip {
                    if ws
                        .attention_sinks(
                            qrow,
                            &k_all,
                            &v_all,
                            &meta.sinks,
                            attn_raw,
                            1,
                            n_kv as i32,
                            n_heads as i32,
                            n_kv_heads as i32,
                            head_dim as i32,
                            true,
                        )
                        .is_ok()
                    {
                        gpu_attn = true;
                    }
                }
            }
            if !gpu_attn {
                attention_sinks(
                    qrow,
                    &k_all,
                    &v_all,
                    &meta.sinks,
                    attn_raw,
                    n_kv,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                )?;
            }
            metrics
                .dsv4_attn_ns
                .fetch_add(t_attn.elapsed().as_nanos() as u64, Ordering::Relaxed);
            for h in 0..n_heads {
                let row = &mut attn_raw[h * head_dim..(h + 1) * head_dim];
                rope_offset_back(row, pos, cfg.rope_theta, cfg.rope_dims, spec.rope_offset)?;
            }
        }
        dsv4_mla_o_rows(
            &mut gpu, file, mla, &attn_raws, blk, n_tok, q_out, n_heads, head_dim, spec, dim,
            metrics,
        )
    })?;
    let ffn_norm = if !meta.ffn_norm.is_empty() {
        &meta.ffn_norm
    } else {
        &host.ffn_norm
    };
    let mut preps = Vec::with_capacity(n_tok);
    let mut fcurs: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
    let mut fposts: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
    let mut fcombs: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
    for t in 0..n_tok {
        let attn_out = &attn_outs[t * dim..(t + 1) * dim];
        let t_hc = Instant::now();
        let streams = hc_post(attn_out, &states[t], &posts[t], &combs[t], spec, dim);
        metrics
            .dsv4_hc_ns
            .fetch_add(t_hc.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let t_hc = Instant::now();
        let (mut fcur, fpost, fcomb) = hc_pre(&streams, &meta.hc_ffn, spec, dim, eps)?;
        metrics
            .dsv4_hc_ns
            .fetch_add(t_hc.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if !ffn_norm.is_empty() {
            let mut n = vec![0.0f32; dim];
            timed_rmsnorm(&fcur, ffn_norm, eps, &mut n, metrics)?;
            fcur = n;
        }
        states[t] = streams;
        fcurs.push(fcur);
        fposts.push(fpost);
        fcombs.push(fcomb);
    }
    let mut xs = vec![0.0f32; n_tok.saturating_mul(dim)];
    for (t, fcur) in fcurs.iter().enumerate() {
        let off = t.saturating_mul(dim);
        let n = fcur.len().min(dim);
        xs[off..off + n].copy_from_slice(&fcur[..n]);
    }
    let t_route = Instant::now();
    let routes = route_experts_rows(
        &xs,
        n_tok,
        tokens,
        host,
        meta,
        cfg,
        spec,
        layer_idx,
        gpu.as_deref_mut(),
        file,
        Some(metrics),
    )?;
    metrics
        .dsv4_router_ns
        .fetch_add(t_route.elapsed().as_nanos() as u64, Ordering::Relaxed);
    for (((fcur, fpost), fcomb), picks) in fcurs.into_iter().zip(fposts).zip(fcombs).zip(routes) {
        preps.push(Dsv4FfnPrep {
            fcur,
            fpost,
            fcomb,
            picks,
        });
    }
    Ok(preps)
}

#[allow(clippy::too_many_arguments)]
fn dsv4_shexp(
    model: &TinyModel,
    dsv4: &Dsv4Weights,
    fcur: &[f32],
    layer_idx: usize,
    metrics: &Metrics,
    gpu: &mut Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
    skip_shexp: bool,
) -> Result<Vec<f32>, AllpacaError> {
    let cfg = &model.config;
    let dim = cfg.dim;
    let host = &model.layers[layer_idx];
    let blk = cfg.gguf_blk(layer_idx);
    let mut shexp_y = vec![0.0f32; dim];
    if !host.shared_expert.w_up.is_empty() {
        let sff = if cfg.shared_ff_dim > 0 {
            cfg.shared_ff_dim
        } else {
            host.shared_expert.w_up.len() / dim.max(1)
        };
        expert_mlp(fcur, &host.shared_expert, sff, dim, &mut shexp_y, false)?;
        return Ok(shexp_y);
    }
    let sff_hint = if cfg.shared_ff_dim > 0 {
        cfg.shared_ff_dim
    } else {
        cfg.ff_dim.max(1)
    };
    let mut fused = false;
    if let (Some(g), Some(f)) = (gpu.as_mut(), file) {
        if let Ok(y) = g.shexp_swiglu(f, blk, fcur, dim, sff_hint, Some(metrics)) {
            shexp_y = y;
            fused = true;
        }
    }
    if fused {
        return Ok(shexp_y);
    }
    with_shexp(dsv4, layer_idx, blk, skip_shexp, |shexp| {
        let sff = if cfg.shared_ff_dim > 0 {
            cfg.shared_ff_dim
        } else if !shexp.w_up.is_empty() {
            shexp.w_up.len() / dim.max(1)
        } else {
            cfg.ff_dim.max(1)
        };
        let mut up = vec![0.0f32; sff];
        dsv4_gemv(
            gpu.as_deref_mut(),
            file,
            &format!("blk.{blk}.ffn_up_shexp.weight"),
            fcur,
            &shexp.w_up,
            &mut up,
            sff,
            dim,
            0,
            Some(metrics),
        )?;
        if !shexp.w_gate.is_empty()
            || file.is_some_and(|f| {
                f.tensor(&format!("blk.{blk}.ffn_gate_shexp.weight"))
                    .is_some()
            })
        {
            let mut gate = vec![0.0f32; sff];
            dsv4_gemv(
                gpu.as_deref_mut(),
                file,
                &format!("blk.{blk}.ffn_gate_shexp.weight"),
                fcur,
                &shexp.w_gate,
                &mut gate,
                sff,
                dim,
                0,
                Some(metrics),
            )?;
            let t_silu = Instant::now();
            for i in 0..sff {
                up[i] = allpaca_cpu::silu(gate[i]) * up[i];
            }
            metrics
                .dsv4_silu_ns
                .fetch_add(t_silu.elapsed().as_nanos() as u64, Ordering::Relaxed);
        } else {
            let t_silu = Instant::now();
            for v in &mut up {
                *v = allpaca_cpu::silu(*v);
            }
            metrics
                .dsv4_silu_ns
                .fetch_add(t_silu.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        dsv4_gemv(
            gpu.as_deref_mut(),
            file,
            &format!("blk.{blk}.ffn_down_shexp.weight"),
            &up,
            &shexp.w_down,
            &mut shexp_y,
            dim,
            sff,
            0,
            Some(metrics),
        )?;
        Ok(())
    })?;
    Ok(shexp_y)
}

#[allow(clippy::too_many_arguments)]
fn dsv4_apply_moe(
    model: &TinyModel,
    dsv4: &Dsv4Weights,
    streams: &[Vec<f32>],
    prep: Dsv4FfnPrep,
    layer_idx: usize,
    metrics: &Metrics,
    gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
    skip_shexp: bool,
    warmup: bool,
) -> Result<Vec<Vec<f32>>, AllpacaError> {
    let cfg = &model.config;
    let spec = &dsv4.spec;
    let dim = cfg.dim;
    let host = &model.layers[layer_idx];
    let blk = cfg.gguf_blk(layer_idx);
    let mut gpu = gpu;
    let Dsv4FfnPrep {
        fcur,
        fpost,
        fcomb,
        picks,
    } = prep;
    if warmup {
        let ids: Vec<usize> = picks.iter().map(|(e, _)| *e).collect();
        if let (Some(g), Some(f)) = (gpu.as_mut(), file) {
            let _ = g.warmup_experts(f, blk, &ids, cfg, Some(metrics));
        }
    }
    let shexp_y = dsv4_shexp(
        model, dsv4, &fcur, layer_idx, metrics, &mut gpu, file, skip_shexp,
    )?;
    if let Some(g) = gpu.as_mut() {
        g.join_expert_h2d();
    }
    let fill = gpu
        .as_ref()
        .map(|g| if g.has_expert_slots() { picks.len() } else { 0 })
        .unwrap_or(0);
    let mut routed = vec![0.0f32; dim];
    for (i, (e, w)) in picks.into_iter().enumerate() {
        let mut y = vec![0.0f32; dim];
        let mut did = false;
        if i < fill {
            if let (Some(g), Some(f)) = (gpu.as_mut(), file) {
                if g.expert_mlp_gpu(f, blk, e, cfg, &fcur, &mut y, Some(metrics))
                    .is_ok()
                {
                    did = true;
                }
            }
        }
        if !did {
            let ffn = if !host.experts.is_empty() {
                model.host_expert(layer_idx, e)?
            } else if let Some(f) = file {
                load_mxfp4_expert(f, blk, e, cfg.n_experts, dim, cfg.ff_dim)?
            } else {
                continue;
            };
            expert_mlp(&fcur, &ffn, cfg.ff_dim, dim, &mut y, false)?;
        }
        for i in 0..dim {
            routed[i] += y[i] * w;
        }
    }
    if let Some(g) = gpu.as_mut() {
        g.finish_h2d_keep();
    }
    for i in 0..dim {
        routed[i] += shexp_y[i];
    }
    let t_hc = Instant::now();
    let streams = hc_post(&routed, streams, &fpost, &fcomb, spec, dim);
    metrics
        .dsv4_hc_ns
        .fetch_add(t_hc.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(streams)
}

fn scatter_expert_rows(routed: &mut [f32], y: &[f32], group: &[(usize, f32)], dim: usize) {
    for (i, &(t, w)) in group.iter().enumerate() {
        let src = i.saturating_mul(dim);
        let dst = t.saturating_mul(dim);
        if src + dim > y.len() || dst + dim > routed.len() {
            continue;
        }
        for j in 0..dim {
            routed[dst + j] += y[src + j] * w;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dsv4_apply_moe_rows(
    model: &TinyModel,
    dsv4: &Dsv4Weights,
    states: &mut [Vec<Vec<f32>>],
    preps: Vec<Dsv4FfnPrep>,
    layer_idx: usize,
    metrics: &Metrics,
    gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
    skip_shexp: bool,
) -> Result<(), AllpacaError> {
    let n_tok = preps.len();
    if n_tok == 0 {
        return Ok(());
    }
    if n_tok != states.len() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dsv4 chunk moe token count mismatch".into(),
        )));
    }
    let cfg = &model.config;
    let spec = &dsv4.spec;
    let dim = cfg.dim;
    let host = &model.layers[layer_idx];
    let blk = cfg.gguf_blk(layer_idx);
    let mut gpu = gpu;
    let mut xs = vec![0.0f32; n_tok.saturating_mul(dim)];
    for (t, prep) in preps.iter().enumerate() {
        let dst = t.saturating_mul(dim);
        let n = prep.fcur.len().min(dim);
        xs[dst..dst + n].copy_from_slice(&prep.fcur[..n]);
    }
    let mut shexp_y = vec![0.0f32; n_tok.saturating_mul(dim)];
    let mut shexp_done = false;
    if !host.shared_expert.w_up.is_empty() {
        let sff = if cfg.shared_ff_dim > 0 {
            cfg.shared_ff_dim
        } else {
            host.shared_expert.w_up.len() / dim.max(1)
        };
        expert_mlp_rows(
            &xs,
            &host.shared_expert,
            sff,
            dim,
            n_tok,
            &mut shexp_y,
            false,
        )?;
        shexp_done = true;
    } else {
        let sff_hint = if cfg.shared_ff_dim > 0 {
            cfg.shared_ff_dim
        } else {
            cfg.ff_dim.max(1)
        };
        if let (Some(g), Some(f)) = (gpu.as_mut(), file) {
            if let Ok(y) = g.shexp_swiglu_rows(f, blk, &xs, n_tok, dim, sff_hint, Some(metrics)) {
                shexp_y = y;
                shexp_done = true;
            }
        }
    }
    if !shexp_done {
        for t in 0..n_tok {
            let row = &xs[t * dim..(t + 1) * dim];
            let y = dsv4_shexp(
                model, dsv4, row, layer_idx, metrics, &mut gpu, file, skip_shexp,
            )?;
            let n = y.len().min(dim);
            shexp_y[t * dim..t * dim + n].copy_from_slice(&y[..n]);
        }
    }
    if let Some(g) = gpu.as_mut() {
        g.join_expert_h2d();
    }
    let gpu_slots = gpu.as_ref().is_some_and(|g| g.has_expert_slots());
    let mut groups: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
    for (t, prep) in preps.iter().enumerate() {
        for &(e, w) in &prep.picks {
            groups.entry(e).or_default().push((t, w));
        }
    }
    let mut routed = vec![0.0f32; n_tok.saturating_mul(dim)];
    let (hip0, hip1) = match gpu.as_ref() {
        Some(g) => {
            let (h0, h1) = g.hip_pair();
            (Some(h0), h1)
        }
        None => (None, None),
    };
    let dev0 = hip0.as_ref().map(|h| h.device);
    let dev1 = hip1.as_ref().map(|h| h.device);
    let mut jobs0 = Vec::new();
    let mut jobs1 = Vec::new();
    let mut cpu_work: Vec<(usize, Vec<f32>, Vec<(usize, f32)>)> = Vec::new();
    for (e, group) in groups {
        let m = group.len();
        let mut gx = vec![0.0f32; m.saturating_mul(dim)];
        for (i, &(t, _)) in group.iter().enumerate() {
            let src = t.saturating_mul(dim);
            let dst = i.saturating_mul(dim);
            gx[dst..dst + dim].copy_from_slice(&xs[src..src + dim]);
        }
        let mut job = None;
        if gpu_slots {
            if let Some(g) = gpu.as_ref() {
                if let Some((ptr, device)) = g.resident_expert(layer_idx, e) {
                    metrics.expert_lru_hits.fetch_add(1, Ordering::Relaxed);
                    job = Some((ptr as usize, device));
                }
            }
        }
        match job {
            Some((ptr, device)) if Some(device) == dev0 => {
                jobs0.push(crate::dsv4_gpu::ExpertSwigluJob {
                    expert: e,
                    ptr,
                    x: gx,
                    m,
                    group,
                });
            }
            Some((ptr, device)) if Some(device) == dev1 => {
                jobs1.push(crate::dsv4_gpu::ExpertSwigluJob {
                    expert: e,
                    ptr,
                    x: gx,
                    m,
                    group,
                });
            }
            _ => cpu_work.push((e, gx, group)),
        }
    }
    let ff = cfg.ff_dim.max(1);
    let mut gpu_out: Vec<(Vec<f32>, Vec<(usize, f32)>)> = Vec::new();
    let take_fail = |failed: Vec<crate::dsv4_gpu::ExpertSwigluJob>,
                     cpu_work: &mut Vec<(usize, Vec<f32>, Vec<(usize, f32)>)>| {
        for job in failed {
            cpu_work.push((job.expert, job.x, job.group));
        }
    };
    if !jobs1.is_empty() {
        if let (Some(h0), Some(h1)) = (hip0.as_ref(), hip1.as_ref()) {
            std::thread::scope(|s| {
                let j0 = s.spawn(|| {
                    crate::dsv4_gpu::run_expert_swiglu_jobs(h0, jobs0, dim, ff, Some(metrics))
                });
                let j1 = s.spawn(|| {
                    crate::dsv4_gpu::run_expert_swiglu_jobs(h1, jobs1, dim, ff, Some(metrics))
                });
                match j0.join() {
                    Ok((rows, fail)) => {
                        gpu_out.extend(rows);
                        take_fail(fail, &mut cpu_work);
                    }
                    Err(_) => tracing::warn!("GPU0 expert SwiGLU thread panicked"),
                }
                match j1.join() {
                    Ok((rows, fail)) => {
                        gpu_out.extend(rows);
                        take_fail(fail, &mut cpu_work);
                    }
                    Err(_) => tracing::warn!("GPU1 expert SwiGLU thread panicked"),
                }
            });
            if let Some(g) = gpu.as_mut() {
                g.used.store(true, Ordering::Relaxed);
            }
            let _ = hip0.as_ref().map(|h| h.bind());
        } else {
            take_fail(jobs0, &mut cpu_work);
            take_fail(jobs1, &mut cpu_work);
        }
    } else if !jobs0.is_empty() {
        if let Some(h0) = hip0.as_ref() {
            let (rows, fail) =
                crate::dsv4_gpu::run_expert_swiglu_jobs(h0, jobs0, dim, ff, Some(metrics));
            gpu_out.extend(rows);
            take_fail(fail, &mut cpu_work);
            if let Some(g) = gpu.as_mut() {
                g.used.store(true, Ordering::Relaxed);
            }
        } else {
            take_fail(jobs0, &mut cpu_work);
        }
    }
    for (y, group) in gpu_out {
        scatter_expert_rows(&mut routed, &y, &group, dim);
    }
    for (e, gx, group) in cpu_work {
        let m = group.len();
        let mut gy = vec![0.0f32; m.saturating_mul(dim)];
        let mut did = false;
        if gpu_slots {
            if let (Some(g), Some(f)) = (gpu.as_mut(), file) {
                if g.expert_mlp_gpu_rows(f, blk, e, cfg, &gx, &mut gy, m, Some(metrics))
                    .is_ok()
                {
                    did = true;
                }
            }
        }
        if !did {
            let ffn = if !host.experts.is_empty() {
                model.host_expert(layer_idx, e)?
            } else if let Some(f) = file {
                load_mxfp4_expert(f, blk, e, cfg.n_experts, dim, cfg.ff_dim)?
            } else {
                continue;
            };
            expert_mlp_rows(&gx, &ffn, cfg.ff_dim, dim, m, &mut gy, false)?;
        }
        scatter_expert_rows(&mut routed, &gy, &group, dim);
    }
    if let Some(g) = gpu.as_mut() {
        g.finish_h2d_keep();
    }
    let t_hc = Instant::now();
    for t in 0..n_tok {
        let mut y = vec![0.0f32; dim];
        let off = t.saturating_mul(dim);
        for j in 0..dim {
            y[j] = routed[off + j] + shexp_y[off + j];
        }
        states[t] = hc_post(&y, &states[t], &preps[t].fpost, &preps[t].fcomb, spec, dim);
    }
    metrics
        .dsv4_hc_ns
        .fetch_add(t_hc.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(())
}

fn dsv4_lm_head(
    model: &TinyModel,
    dsv4: &Dsv4Weights,
    streams: &[Vec<f32>],
    logits: &mut [f32],
    metrics: &Metrics,
    gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
) -> Result<bool, AllpacaError> {
    let cfg = &model.config;
    let spec = &dsv4.spec;
    let dim = cfg.dim;
    let eps = cfg.eps;
    let t_hc = Instant::now();
    let mut x = hc_head(
        streams,
        &dsv4.hc_head_fn,
        &dsv4.hc_head_base,
        dsv4.hc_head_scale,
        spec,
        dim,
        eps,
    )?;
    metrics
        .dsv4_hc_ns
        .fetch_add(t_hc.elapsed().as_nanos() as u64, Ordering::Relaxed);
    if !model.norm.is_empty() {
        let mut n = vec![0.0f32; dim];
        timed_rmsnorm(&x, &model.norm, eps, &mut n, metrics)?;
        x = n;
    } else if let Some(file) = &dsv4.gguf {
        if let Ok(norm) = file.tensor_f32("output_norm.weight") {
            let mut n = vec![0.0f32; dim];
            timed_rmsnorm(&x, &norm, eps, &mut n, metrics)?;
            x = n;
        }
    }
    if logits.len() < cfg.vocab_size {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "logits buffer smaller than vocab".into(),
        )));
    }
    let mut used_gpu_head = false;
    let mut gpu = gpu;
    if let (Some(g), Some(f)) = (gpu.as_mut(), file) {
        if g.gemv_bf16(
            f,
            "output.weight",
            &x,
            &mut logits[..cfg.vocab_size],
            cfg.vocab_size,
            dim,
            0,
            Some(metrics),
        )
        .is_ok()
        {
            used_gpu_head = true;
        }
    }
    if !used_gpu_head {
        if !model.lm_head.is_empty() {
            gemv(&x, &model.lm_head, logits, cfg.vocab_size, dim)?;
        } else {
            let mut cache = dsv4.cache.lock().map_err(|_| {
                AllpacaError::Fatal(FatalError::ModelLoad("dsv4 cache poisoned".into()))
            })?;
            if cache.lm_head.is_none() {
                if let Some(file) = &dsv4.gguf {
                    cache.lm_head = Some(file.tensor_f32("output.weight")?);
                }
            }
            let head = cache.lm_head.as_ref().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad("DeepSeek-V4 lm_head missing".into()))
            })?;
            gemv(&x, head, logits, cfg.vocab_size, dim)?;
        }
    }
    Ok(gpu.as_ref().is_some_and(|g| g.used.load(Ordering::Relaxed)))
}

fn with_shexp<R>(
    dsv4: &Dsv4Weights,
    layer: usize,
    blk: usize,
    skip_weights: bool,
    f: impl FnOnce(&LayerFfn) -> Result<R, AllpacaError>,
) -> Result<R, AllpacaError> {
    if skip_weights {
        return f(&LayerFfn::default());
    }
    let mut cache = dsv4
        .cache
        .lock()
        .map_err(|_| AllpacaError::Fatal(FatalError::ModelLoad("dsv4 cache poisoned".into())))?;
    let hit = cache.shexp.as_ref().is_some_and(|(i, _)| *i == layer);
    if !hit {
        let loaded = if let Some(file) = &dsv4.gguf {
            load_shexp(file, blk)
        } else {
            Ok(LayerFfn::default())
        };
        cache.shexp = Some((layer, loaded.unwrap_or_default()));
    }
    let shexp = &cache.shexp.as_ref().expect("shexp just loaded").1;
    f(shexp)
}

pub(crate) fn load_mxfp4_expert(
    file: &GgufFile,
    layer: usize,
    expert: usize,
    n_experts: usize,
    dim: usize,
    ff: usize,
) -> Result<LayerFfn, AllpacaError> {
    let load = |name: &str| -> Result<Vec<f32>, AllpacaError> {
        let info = file
            .tensor(name)
            .ok_or_else(|| AllpacaError::Fatal(FatalError::ModelLoad(format!("missing {name}"))))?;
        let n = info.n_elements() as usize;
        if n_experts == 0 || n % n_experts != 0 {
            return file.tensor_f32(name);
        }
        let per = n / n_experts;
        if info.ggml_type == allpaca_gguf::GgmlType::MXFP4 {
            let bytes = file.tensor_bytes_of(info)?;
            let stride = per
                .div_ceil(allpaca_cpu::MXFP4_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::MXFP4_BLOCK_BYTES);
            let start = expert.saturating_mul(stride);
            let slice = bytes.get(start..start + stride).ok_or_else(|| {
                AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                    "{name} expert {expert} OOB"
                )))
            })?;
            let mut out = vec![0.0f32; per];
            allpaca_cpu::dequant_mxfp4(slice, per, &mut out)?;
            return Ok(out);
        }
        let all = file.tensor_f32(name)?;
        Ok(all[expert * per..(expert + 1) * per].to_vec())
    };
    let gate = load(&format!("blk.{layer}.ffn_gate_exps.weight"))?;
    let up = load(&format!("blk.{layer}.ffn_up_exps.weight"))?;
    let down = load(&format!("blk.{layer}.ffn_down_exps.weight"))?;
    if gate.len() != dim * ff && !gate.is_empty() {
        // stacked layout may be [in, ff] per expert
    }
    let _ = (dim, ff);
    Ok(LayerFfn {
        ffn_norm: Vec::new(),
        w_gate: gate,
        w_up: up,
        w_down: down,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn forward_dsv4_token(
    model: &TinyModel,
    token: u32,
    pos: usize,
    page_ids: &[Vec<u64>],
    kv: &mut KvTierManager,
    logits: &mut [f32],
    metrics: &Metrics,
    _dense: Option<&mut DenseKv>,
    hip: Option<&allpaca_hip::HipWorkspace>,
    prefill: bool,
) -> Result<(), AllpacaError> {
    let dsv4 = model.dsv4.as_ref().ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad("missing DeepSeek-V4 weights".into()))
    })?;
    let cfg = &model.config;
    let dim = cfg.dim;
    let hc = dsv4.spec.hc_mult.max(1);

    let t_embed = Instant::now();
    let x0 = if !model.embed.is_empty() {
        model.embed[token as usize * dim..(token as usize + 1) * dim].to_vec()
    } else if let Some(file) = &dsv4.gguf {
        gather_embed(file, token, dim)?
    } else {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(
            "DeepSeek-V4 embed missing".into(),
        )));
    };
    metrics
        .dsv4_embed_ns
        .fetch_add(t_embed.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let mut streams = vec![x0.clone(); hc];
    let file = dsv4.gguf.as_deref();
    let mut gpu_guard = dsv4.gpu.lock().ok();
    let mut gpu = gpu_guard.as_mut().and_then(|g| g.as_mut());
    let skip_mla = gpu.as_ref().is_some()
        && file.is_some_and(|f| {
            f.tensor("blk.0.attn_q_a.weight")
                .is_some_and(|t| t.ggml_type == allpaca_gguf::GgmlType::BF16)
        });
    let skip_shexp = gpu.as_ref().is_some()
        && file.is_some_and(|f| {
            f.tensor("blk.0.ffn_up_shexp.weight")
                .is_some_and(|t| t.ggml_type == allpaca_gguf::GgmlType::BF16)
        });

    for layer_idx in 0..cfg.n_layers {
        tracing::debug!(layer_idx, n_layers = cfg.n_layers, "dsv4 forward layer");
        let (next, prep) = dsv4_attn_and_route(
            model,
            dsv4,
            &streams,
            token,
            pos,
            layer_idx,
            page_ids,
            kv,
            metrics,
            gpu.as_deref_mut(),
            file,
            hip,
            skip_mla,
            prefill,
        )?;
        streams = dsv4_apply_moe(
            model,
            dsv4,
            &next,
            prep,
            layer_idx,
            metrics,
            gpu.as_deref_mut(),
            file,
            skip_shexp,
            true,
        )?;
    }
    let used = dsv4_lm_head(
        model,
        dsv4,
        &streams,
        logits,
        metrics,
        gpu.as_deref_mut(),
        file,
    )?;
    if used {
        metrics.gpu_forwards.fetch_add(1, Ordering::Relaxed);
        metrics.target_gpu_forwards.fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn forward_dsv4_tokens(
    model: &TinyModel,
    tokens: &[u32],
    start_pos: usize,
    page_ids: &[Vec<u64>],
    kv: &mut KvTierManager,
    logits: &mut [f32],
    metrics: &Metrics,
    hip: Option<&allpaca_hip::HipWorkspace>,
    write_logits: bool,
) -> Result<(), AllpacaError> {
    if tokens.len() <= 1 {
        if let Some(&tok) = tokens.first() {
            return forward_dsv4_token(
                model, tok, start_pos, page_ids, kv, logits, metrics, None, hip, true,
            );
        }
        return Ok(());
    }
    let dsv4 = model.dsv4.as_ref().ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad("missing DeepSeek-V4 weights".into()))
    })?;
    let cfg = &model.config;
    let dim = cfg.dim;
    let hc = dsv4.spec.hc_mult.max(1);
    let file = dsv4.gguf.as_deref();
    let mut gpu_guard = dsv4.gpu.lock().ok();
    let mut gpu = gpu_guard.as_mut().and_then(|g| g.as_mut());
    let skip_mla = gpu.as_ref().is_some()
        && file.is_some_and(|f| {
            f.tensor("blk.0.attn_q_a.weight")
                .is_some_and(|t| t.ggml_type == allpaca_gguf::GgmlType::BF16)
        });
    let skip_shexp = gpu.as_ref().is_some()
        && file.is_some_and(|f| {
            f.tensor("blk.0.ffn_up_shexp.weight")
                .is_some_and(|t| t.ggml_type == allpaca_gguf::GgmlType::BF16)
        });
    let mut states: Vec<Vec<Vec<f32>>> = Vec::with_capacity(tokens.len());
    for &token in tokens {
        let t_embed = Instant::now();
        let x0 = if !model.embed.is_empty() {
            model.embed[token as usize * dim..(token as usize + 1) * dim].to_vec()
        } else if let Some(file) = file {
            gather_embed(file, token, dim)?
        } else {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "DeepSeek-V4 embed missing".into(),
            )));
        };
        metrics
            .dsv4_embed_ns
            .fetch_add(t_embed.elapsed().as_nanos() as u64, Ordering::Relaxed);
        states.push(vec![x0; hc]);
    }
    for layer_idx in 0..cfg.n_layers {
        tracing::debug!(layer_idx, n_layers = cfg.n_layers, "dsv4 chunk layer");
        let blk = cfg.gguf_blk(layer_idx);
        let preps = dsv4_attn_and_route_rows(
            model,
            dsv4,
            &mut states,
            tokens,
            start_pos,
            layer_idx,
            page_ids,
            kv,
            metrics,
            gpu.as_deref_mut(),
            file,
            hip,
            skip_mla,
            true,
        )?;
        let mut union = Vec::new();
        for prep in &preps {
            for (e, _) in &prep.picks {
                if !union.contains(e) {
                    union.push(*e);
                }
            }
        }
        if let (Some(g), Some(f)) = (gpu.as_mut(), file) {
            let _ = g.warmup_experts(f, blk, &union, cfg, Some(metrics));
        }
        dsv4_apply_moe_rows(
            model,
            dsv4,
            &mut states,
            preps,
            layer_idx,
            metrics,
            gpu.as_deref_mut(),
            file,
            skip_shexp,
        )?;
    }
    if write_logits {
        if let Some(last) = states.last() {
            let used = dsv4_lm_head(model, dsv4, last, logits, metrics, gpu.as_deref_mut(), file)?;
            if used {
                metrics.gpu_forwards.fetch_add(1, Ordering::Relaxed);
                metrics.target_gpu_forwards.fetch_add(1, Ordering::Relaxed);
            }
        }
    } else if gpu.as_ref().is_some_and(|g| g.used.load(Ordering::Relaxed)) {
        metrics.gpu_forwards.fetch_add(1, Ordering::Relaxed);
        metrics.target_gpu_forwards.fetch_add(1, Ordering::Relaxed);
    }
    Ok(())
}

fn route_experts(
    x: &[f32],
    host: &crate::model::LayerWeights,
    meta: &Dsv4LayerMeta,
    cfg: &ModelConfig,
    spec: &Dsv4Spec,
    token: u32,
    layer_idx: usize,
    gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
    metrics: Option<&Metrics>,
) -> Result<Vec<(usize, f32)>, AllpacaError> {
    if cfg.n_experts == 0 {
        return Ok(Vec::new());
    }
    if layer_idx < spec.hash_layer_count && !meta.tid2eid.is_empty() {
        let n_act = cfg.n_active.max(1);
        let off = token as usize * n_act;
        if off + n_act <= meta.tid2eid.len() {
            let mut picks = Vec::new();
            let w = 1.0 / n_act as f32 * cfg.expert_weights_scale.max(1e-6);
            for k in 0..n_act {
                let e = meta.tid2eid[off + k];
                if e >= 0 {
                    picks.push((e as usize, w));
                }
            }
            if !picks.is_empty() {
                return Ok(picks);
            }
        }
    }
    let dim = x.len();
    let mut scores = vec![0.0f32; cfg.n_experts];
    let mut have_logits = false;
    if host.w_router.len() >= cfg.n_experts.saturating_mul(dim) {
        for e in 0..cfg.n_experts {
            let row = &host.w_router[e * dim..(e + 1) * dim];
            let mut s = 0.0f32;
            for i in 0..dim {
                s += x[i] * row[i];
            }
            scores[e] = s;
        }
        have_logits = true;
    } else if let (Some(gpu), Some(file)) = (gpu, file) {
        let blk = cfg.gguf_blk(layer_idx);
        let name = format!("blk.{blk}.ffn_gate_inp.weight");
        if gpu
            .gemv_bf16(file, &name, x, &mut scores, cfg.n_experts, dim, 0, metrics)
            .is_ok()
        {
            have_logits = true;
        }
    }
    if !have_logits {
        return Ok(Vec::new());
    }
    if !meta.exp_probs_b.is_empty() {
        for e in 0..cfg.n_experts {
            scores[e] += meta.exp_probs_b.get(e).copied().unwrap_or(0.0);
        }
    }
    Ok(topk_from_logits(&mut scores, cfg.n_active, cfg))
}

#[allow(clippy::too_many_arguments)]
fn route_experts_rows(
    xs: &[f32],
    n_tok: usize,
    tokens: &[u32],
    host: &crate::model::LayerWeights,
    meta: &Dsv4LayerMeta,
    cfg: &ModelConfig,
    spec: &Dsv4Spec,
    layer_idx: usize,
    gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
    metrics: Option<&Metrics>,
) -> Result<Vec<Vec<(usize, f32)>>, AllpacaError> {
    if n_tok == 0 {
        return Ok(Vec::new());
    }
    if n_tok == 1 {
        let tok = tokens.first().copied().unwrap_or(0);
        let picks = route_experts(
            &xs[..cfg.dim.min(xs.len())],
            host,
            meta,
            cfg,
            spec,
            tok,
            layer_idx,
            gpu,
            file,
            metrics,
        )?;
        return Ok(vec![picks]);
    }
    if cfg.n_experts == 0 {
        return Ok(vec![Vec::new(); n_tok]);
    }
    if layer_idx < spec.hash_layer_count && !meta.tid2eid.is_empty() {
        let mut out = Vec::with_capacity(n_tok);
        for t in 0..n_tok {
            let tok = tokens.get(t).copied().unwrap_or(0);
            let dim = cfg.dim;
            let row = &xs[t * dim..(t + 1) * dim];
            out.push(route_experts(
                row, host, meta, cfg, spec, tok, layer_idx, None, file, metrics,
            )?);
        }
        return Ok(out);
    }
    let dim = cfg.dim;
    let n_exp = cfg.n_experts;
    let mut scores = vec![0.0f32; n_tok.saturating_mul(n_exp)];
    let gemm_ok = dsv4_gemm(
        gpu,
        file,
        &format!("blk.{}.ffn_gate_inp.weight", cfg.gguf_blk(layer_idx)),
        xs,
        &host.w_router,
        &mut scores,
        n_exp,
        dim,
        0,
        n_tok,
        metrics,
    )
    .is_ok();
    if !gemm_ok {
        let mut out = Vec::with_capacity(n_tok);
        for t in 0..n_tok {
            let tok = tokens.get(t).copied().unwrap_or(0);
            let row = &xs[t * dim..(t + 1) * dim];
            out.push(route_experts(
                row, host, meta, cfg, spec, tok, layer_idx, None, file, metrics,
            )?);
        }
        return Ok(out);
    }
    if !meta.exp_probs_b.is_empty() {
        for t in 0..n_tok {
            let off = t.saturating_mul(n_exp);
            for e in 0..n_exp {
                scores[off + e] += meta.exp_probs_b.get(e).copied().unwrap_or(0.0);
            }
        }
    }
    let mut out = Vec::with_capacity(n_tok);
    for t in 0..n_tok {
        let off = t.saturating_mul(n_exp);
        let mut row = scores[off..off + n_exp].to_vec();
        out.push(topk_from_logits(&mut row, cfg.n_active, cfg));
    }
    Ok(out)
}

pub fn write_tiny_deepseek4_gguf(path: &std::path::Path) {
    use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
    use std::collections::HashMap;

    let dim = 16usize;
    let vocab = 8usize;
    let n_heads = 2usize;
    let head_dim = 8usize;
    let q_lora = 8usize;
    let o_lora = 8usize;
    let groups = 2usize;
    let hc = 2usize;
    let n_experts = 4usize;
    let ff = 16usize;
    let q_out = n_heads * head_dim;
    let oa_out = o_lora * groups;
    let o_group_dim = (n_heads / groups) * head_dim;
    let hc_dim = dim * hc;
    let mix_dim = (2 + hc) * hc;
    let w = 0.02f32;

    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::String("deepseek4".into()),
    );
    meta.insert("general.name".into(), GgufValue::String("tiny-dsv4".into()));
    meta.insert(
        "deepseek4.vocab_size".into(),
        GgufValue::Uint32(vocab as u32),
    );
    meta.insert(
        "deepseek4.embedding_length".into(),
        GgufValue::Uint32(dim as u32),
    );
    meta.insert("deepseek4.block_count".into(), GgufValue::Uint32(1));
    meta.insert(
        "deepseek4.attention.head_count".into(),
        GgufValue::Uint32(n_heads as u32),
    );
    meta.insert(
        "deepseek4.attention.head_count_kv".into(),
        GgufValue::Uint32(1),
    );
    meta.insert(
        "deepseek4.attention.key_length".into(),
        GgufValue::Uint32(head_dim as u32),
    );
    meta.insert(
        "deepseek4.attention.value_length".into(),
        GgufValue::Uint32(head_dim as u32),
    );
    meta.insert(
        "deepseek4.rope.dimension_count".into(),
        GgufValue::Uint32(4),
    );
    meta.insert(
        "deepseek4.attention.q_lora_rank".into(),
        GgufValue::Uint32(q_lora as u32),
    );
    meta.insert(
        "deepseek4.attention.output_lora_rank".into(),
        GgufValue::Uint32(o_lora as u32),
    );
    meta.insert(
        "deepseek4.attention.output_group_count".into(),
        GgufValue::Uint32(groups as u32),
    );
    meta.insert(
        "deepseek4.hyper_connection.count".into(),
        GgufValue::Uint32(hc as u32),
    );
    meta.insert(
        "deepseek4.hyper_connection.sinkhorn_iterations".into(),
        GgufValue::Uint32(4),
    );
    meta.insert(
        "deepseek4.hyper_connection.epsilon".into(),
        GgufValue::Float32(1e-6),
    );
    meta.insert("deepseek4.hash_layer_count".into(), GgufValue::Uint32(0));
    meta.insert(
        "deepseek4.expert_count".into(),
        GgufValue::Uint32(n_experts as u32),
    );
    meta.insert("deepseek4.expert_used_count".into(), GgufValue::Uint32(2));
    meta.insert("deepseek4.expert_shared_count".into(), GgufValue::Uint32(1));
    meta.insert(
        "deepseek4.expert_feed_forward_length".into(),
        GgufValue::Uint32(ff as u32),
    );
    meta.insert(
        "deepseek4.expert_shared_feed_forward_length".into(),
        GgufValue::Uint32(ff as u32),
    );
    meta.insert(
        "deepseek4.expert_weights_norm".into(),
        GgufValue::Bool(true),
    );
    meta.insert(
        "deepseek4.expert_weights_scale".into(),
        GgufValue::Float32(1.5),
    );
    meta.insert("deepseek4.expert_gating_func".into(), GgufValue::Uint32(4));
    meta.insert(
        "tokenizer.ggml.model".into(),
        GgufValue::String("wordlevel".into()),
    );
    meta.insert(
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array(
            ["<unk>", "<s>", "</s>", "hi", "a", "b", "c", "d"]
                .into_iter()
                .map(|t| GgufValue::String(t.into()))
                .collect(),
        ),
    );
    meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
    meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));

    let owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
        (
            "token_embd.weight".into(),
            vec![vocab as u64, dim as u64],
            vec![w; vocab * dim],
        ),
        (
            "blk.0.attn_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ),
        (
            "blk.0.attn_q_a.weight".into(),
            vec![q_lora as u64, dim as u64],
            vec![w; q_lora * dim],
        ),
        (
            "blk.0.attn_q_a_norm.weight".into(),
            vec![q_lora as u64],
            vec![1.0; q_lora],
        ),
        (
            "blk.0.attn_q_b.weight".into(),
            vec![q_out as u64, q_lora as u64],
            vec![w; q_out * q_lora],
        ),
        (
            "blk.0.attn_kv.weight".into(),
            vec![head_dim as u64, dim as u64],
            vec![w; head_dim * dim],
        ),
        (
            "blk.0.attn_kv_a_norm.weight".into(),
            vec![head_dim as u64],
            vec![1.0; head_dim],
        ),
        (
            "blk.0.attn_output_a.weight".into(),
            vec![o_group_dim as u64, oa_out as u64],
            vec![w; o_group_dim * oa_out],
        ),
        (
            "blk.0.attn_output_b.weight".into(),
            vec![dim as u64, oa_out as u64],
            vec![w; dim * oa_out],
        ),
        (
            "blk.0.attn_sinks.weight".into(),
            vec![n_heads as u64],
            vec![0.0; n_heads],
        ),
        (
            "blk.0.ffn_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ),
        (
            "blk.0.hc_attn_fn.weight".into(),
            vec![hc_dim as u64, mix_dim as u64],
            vec![w; mix_dim * hc_dim],
        ),
        (
            "blk.0.hc_attn_base.weight".into(),
            vec![(2 * hc + hc * hc) as u64],
            vec![0.0; 2 * hc + hc * hc],
        ),
        (
            "blk.0.hc_attn_scale.weight".into(),
            vec![3],
            vec![1.0, 1.0, 1.0],
        ),
        (
            "blk.0.hc_ffn_fn.weight".into(),
            vec![hc_dim as u64, mix_dim as u64],
            vec![w; mix_dim * hc_dim],
        ),
        (
            "blk.0.hc_ffn_base.weight".into(),
            vec![(2 * hc + hc * hc) as u64],
            vec![0.0; 2 * hc + hc * hc],
        ),
        (
            "blk.0.hc_ffn_scale.weight".into(),
            vec![3],
            vec![1.0, 1.0, 1.0],
        ),
        (
            "blk.0.ffn_gate_inp.weight".into(),
            vec![n_experts as u64, dim as u64],
            vec![w; n_experts * dim],
        ),
        (
            "blk.0.ffn_gate_exps.weight".into(),
            vec![(n_experts * ff) as u64, dim as u64],
            vec![w; n_experts * ff * dim],
        ),
        (
            "blk.0.ffn_up_exps.weight".into(),
            vec![(n_experts * ff) as u64, dim as u64],
            vec![w; n_experts * ff * dim],
        ),
        (
            "blk.0.ffn_down_exps.weight".into(),
            vec![(n_experts * dim) as u64, ff as u64],
            vec![w; n_experts * dim * ff],
        ),
        (
            "blk.0.ffn_gate_shexp.weight".into(),
            vec![ff as u64, dim as u64],
            vec![w; ff * dim],
        ),
        (
            "blk.0.ffn_up_shexp.weight".into(),
            vec![ff as u64, dim as u64],
            vec![w; ff * dim],
        ),
        (
            "blk.0.ffn_down_shexp.weight".into(),
            vec![dim as u64, ff as u64],
            vec![w; dim * ff],
        ),
        (
            "output_hc_fn.weight".into(),
            vec![hc_dim as u64, hc as u64],
            vec![w; hc * hc_dim],
        ),
        (
            "output_hc_base.weight".into(),
            vec![hc as u64],
            vec![0.0; hc],
        ),
        ("output_hc_scale.weight".into(), vec![1], vec![1.0]),
        (
            "output_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ),
        (
            "output.weight".into(),
            vec![vocab as u64, dim as u64],
            vec![w; vocab * dim],
        ),
    ];
    let views: Vec<GgufWriteTensor<'_>> = owned
        .iter()
        .map(|(name, dims, data)| GgufWriteTensor {
            name: name.clone(),
            dims: dims.clone(),
            data,
        })
        .collect();
    write_gguf(path, &meta, &views).expect("write tiny deepseek4");
}

/// Tiny DeepSeek-V4 GGUF with BF16 MLA/shexp/lm_head and MXFP4 experts (`k % 32 == 0`).
/// HIP GEMV is used only for those dtypes; F32 fixtures stay on the CPU path.
pub fn write_tiny_deepseek4_packed_gguf(path: &std::path::Path) {
    use allpaca_gguf::{
        quantize_bf16, write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor, GgufWriteTensor,
    };
    use std::collections::HashMap;

    let dim = 32usize;
    let vocab = 8usize;
    let n_heads = 2usize;
    let head_dim = 16usize;
    let q_lora = 16usize;
    let o_lora = 16usize;
    let groups = 2usize;
    let hc = 2usize;
    let n_experts = 4usize;
    let ff = 32usize;
    let q_out = n_heads * head_dim;
    let oa_out = o_lora * groups;
    let o_group_dim = (n_heads / groups) * head_dim;
    let hc_dim = dim * hc;
    let mix_dim = (2 + hc) * hc;
    let w = 0.02f32;

    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::String("deepseek4".into()),
    );
    meta.insert(
        "general.name".into(),
        GgufValue::String("tiny-dsv4-packed".into()),
    );
    meta.insert(
        "deepseek4.vocab_size".into(),
        GgufValue::Uint32(vocab as u32),
    );
    meta.insert(
        "deepseek4.embedding_length".into(),
        GgufValue::Uint32(dim as u32),
    );
    meta.insert("deepseek4.block_count".into(), GgufValue::Uint32(1));
    meta.insert(
        "deepseek4.attention.head_count".into(),
        GgufValue::Uint32(n_heads as u32),
    );
    meta.insert(
        "deepseek4.attention.head_count_kv".into(),
        GgufValue::Uint32(1),
    );
    meta.insert(
        "deepseek4.attention.key_length".into(),
        GgufValue::Uint32(head_dim as u32),
    );
    meta.insert(
        "deepseek4.attention.value_length".into(),
        GgufValue::Uint32(head_dim as u32),
    );
    meta.insert(
        "deepseek4.rope.dimension_count".into(),
        GgufValue::Uint32(8),
    );
    meta.insert(
        "deepseek4.attention.q_lora_rank".into(),
        GgufValue::Uint32(q_lora as u32),
    );
    meta.insert(
        "deepseek4.attention.output_lora_rank".into(),
        GgufValue::Uint32(o_lora as u32),
    );
    meta.insert(
        "deepseek4.attention.output_group_count".into(),
        GgufValue::Uint32(groups as u32),
    );
    meta.insert(
        "deepseek4.hyper_connection.count".into(),
        GgufValue::Uint32(hc as u32),
    );
    meta.insert(
        "deepseek4.hyper_connection.sinkhorn_iterations".into(),
        GgufValue::Uint32(4),
    );
    meta.insert(
        "deepseek4.hyper_connection.epsilon".into(),
        GgufValue::Float32(1e-6),
    );
    meta.insert("deepseek4.hash_layer_count".into(), GgufValue::Uint32(0));
    meta.insert(
        "deepseek4.expert_count".into(),
        GgufValue::Uint32(n_experts as u32),
    );
    meta.insert("deepseek4.expert_used_count".into(), GgufValue::Uint32(2));
    meta.insert("deepseek4.expert_shared_count".into(), GgufValue::Uint32(1));
    meta.insert(
        "deepseek4.expert_feed_forward_length".into(),
        GgufValue::Uint32(ff as u32),
    );
    meta.insert(
        "deepseek4.expert_shared_feed_forward_length".into(),
        GgufValue::Uint32(ff as u32),
    );
    meta.insert(
        "deepseek4.expert_weights_norm".into(),
        GgufValue::Bool(true),
    );
    meta.insert(
        "deepseek4.expert_weights_scale".into(),
        GgufValue::Float32(1.5),
    );
    meta.insert("deepseek4.expert_gating_func".into(), GgufValue::Uint32(4));
    meta.insert(
        "tokenizer.ggml.model".into(),
        GgufValue::String("wordlevel".into()),
    );
    meta.insert(
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array(
            ["<unk>", "<s>", "</s>", "hi", "a", "b", "c", "d"]
                .into_iter()
                .map(|t| GgufValue::String(t.into()))
                .collect(),
        ),
    );
    meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
    meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));

    let owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
        (
            "token_embd.weight".into(),
            vec![vocab as u64, dim as u64],
            vec![w; vocab * dim],
        ),
        (
            "blk.0.attn_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ),
        (
            "blk.0.attn_q_a.weight".into(),
            vec![q_lora as u64, dim as u64],
            vec![w; q_lora * dim],
        ),
        (
            "blk.0.attn_q_a_norm.weight".into(),
            vec![q_lora as u64],
            vec![1.0; q_lora],
        ),
        (
            "blk.0.attn_q_b.weight".into(),
            vec![q_out as u64, q_lora as u64],
            vec![w; q_out * q_lora],
        ),
        (
            "blk.0.attn_kv.weight".into(),
            vec![head_dim as u64, dim as u64],
            vec![w; head_dim * dim],
        ),
        (
            "blk.0.attn_kv_a_norm.weight".into(),
            vec![head_dim as u64],
            vec![1.0; head_dim],
        ),
        (
            "blk.0.attn_output_a.weight".into(),
            vec![o_group_dim as u64, oa_out as u64],
            vec![w; o_group_dim * oa_out],
        ),
        (
            "blk.0.attn_output_b.weight".into(),
            vec![dim as u64, oa_out as u64],
            vec![w; dim * oa_out],
        ),
        (
            "blk.0.attn_sinks.weight".into(),
            vec![n_heads as u64],
            vec![0.0; n_heads],
        ),
        (
            "blk.0.ffn_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ),
        (
            "blk.0.hc_attn_fn.weight".into(),
            vec![hc_dim as u64, mix_dim as u64],
            vec![w; mix_dim * hc_dim],
        ),
        (
            "blk.0.hc_attn_base.weight".into(),
            vec![(2 * hc + hc * hc) as u64],
            vec![0.0; 2 * hc + hc * hc],
        ),
        (
            "blk.0.hc_attn_scale.weight".into(),
            vec![3],
            vec![1.0, 1.0, 1.0],
        ),
        (
            "blk.0.hc_ffn_fn.weight".into(),
            vec![hc_dim as u64, mix_dim as u64],
            vec![w; mix_dim * hc_dim],
        ),
        (
            "blk.0.hc_ffn_base.weight".into(),
            vec![(2 * hc + hc * hc) as u64],
            vec![0.0; 2 * hc + hc * hc],
        ),
        (
            "blk.0.hc_ffn_scale.weight".into(),
            vec![3],
            vec![1.0, 1.0, 1.0],
        ),
        (
            "blk.0.ffn_gate_inp.weight".into(),
            vec![n_experts as u64, dim as u64],
            vec![w; n_experts * dim],
        ),
        (
            "blk.0.ffn_gate_exps.weight".into(),
            vec![(n_experts * ff) as u64, dim as u64],
            vec![w; n_experts * ff * dim],
        ),
        (
            "blk.0.ffn_up_exps.weight".into(),
            vec![(n_experts * ff) as u64, dim as u64],
            vec![w; n_experts * ff * dim],
        ),
        (
            "blk.0.ffn_down_exps.weight".into(),
            vec![(n_experts * dim) as u64, ff as u64],
            vec![w; n_experts * dim * ff],
        ),
        (
            "blk.0.ffn_gate_shexp.weight".into(),
            vec![ff as u64, dim as u64],
            vec![w; ff * dim],
        ),
        (
            "blk.0.ffn_up_shexp.weight".into(),
            vec![ff as u64, dim as u64],
            vec![w; ff * dim],
        ),
        (
            "blk.0.ffn_down_shexp.weight".into(),
            vec![dim as u64, ff as u64],
            vec![w; dim * ff],
        ),
        (
            "output_hc_fn.weight".into(),
            vec![hc_dim as u64, hc as u64],
            vec![w; hc * hc_dim],
        ),
        (
            "output_hc_base.weight".into(),
            vec![hc as u64],
            vec![0.0; hc],
        ),
        ("output_hc_scale.weight".into(), vec![1], vec![1.0]),
        (
            "output_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ),
        (
            "output.weight".into(),
            vec![vocab as u64, dim as u64],
            vec![w; vocab * dim],
        ),
    ];
    let bf16_names = [
        "blk.0.attn_q_a.weight",
        "blk.0.attn_q_b.weight",
        "blk.0.attn_kv.weight",
        "blk.0.attn_output_a.weight",
        "blk.0.attn_output_b.weight",
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_up_shexp.weight",
        "blk.0.ffn_down_shexp.weight",
        "output.weight",
    ];
    let mxfp4_names = [
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
    ];
    let mut f32_tensors = Vec::new();
    let mut packed_store: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)> = Vec::new();
    for (name, dims, data) in &owned {
        if bf16_names.contains(&name.as_str()) {
            packed_store.push((
                name.clone(),
                dims.clone(),
                GgmlType::BF16,
                quantize_bf16(data),
            ));
        } else if mxfp4_names.contains(&name.as_str()) {
            packed_store.push((
                name.clone(),
                dims.clone(),
                GgmlType::MXFP4,
                allpaca_cpu::quantize_mxfp4(data),
            ));
        } else {
            f32_tensors.push(GgufWriteTensor {
                name: name.clone(),
                dims: dims.clone(),
                data,
            });
        }
    }
    let quant_tensors: Vec<GgufWriteQuantTensor<'_>> = packed_store
        .iter()
        .map(|(name, dims, ty, bytes)| GgufWriteQuantTensor {
            name: name.clone(),
            dims: dims.clone(),
            ggml_type: *ty,
            bytes,
        })
        .collect();
    write_gguf_quant(path, &meta, &f32_tensors, &quant_tensors)
        .expect("write packed tiny deepseek4");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinkhorn_rows_and_cols_normalized() {
        let hc = 2;
        let mut comb = vec![1.0f32, 2.0, 3.0, 4.0];
        sinkhorn(&mut comb, hc, 8, 1e-6);
        for dst in 0..hc {
            let sum: f32 = (0..hc).map(|src| comb[dst + src * hc]).sum();
            assert!((sum - 1.0).abs() < 0.15, "row {dst} sum={sum}");
        }
    }

    #[test]
    fn sqrt_softplus_positive() {
        assert!(sqrt_softplus(0.0) >= 0.0);
        assert!(sqrt_softplus(4.0) > sqrt_softplus(0.0));
    }

    #[test]
    fn tiny_deepseek4_gguf_loads_mla_flags() {
        let path = std::env::temp_dir().join(format!(
            "allpaca_tiny_dsv4-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_tiny_deepseek4_gguf(&path);
        let file = std::sync::Arc::new(GgufFile::open(&path).expect("open tiny dsv4"));
        assert!(file_is_deepseek4(&file));
        let cfg = ModelConfig::from_gguf(&file).expect("cfg");
        assert!(cfg.moe_sqrt_softplus);
        assert!(!cfg.moe_relu_sqr);
        assert!(!cfg.moe_sigmoid_gating);
        assert_eq!(cfg.rope_offset, 4);
        let mut model = TinyModel::from_gguf(&file).expect("model");
        model.bind_gguf(file);
        assert!(model.dsv4.is_some());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tiny_deepseek4_chunk_prefill_matches_sequential() {
        let path = std::env::temp_dir().join(format!(
            "allpaca_tiny_dsv4_chunk-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_tiny_deepseek4_gguf(&path);
        let file = std::sync::Arc::new(GgufFile::open(&path).expect("open tiny dsv4"));
        let mut model = TinyModel::from_gguf(&file).expect("model");
        model.bind_gguf(file);
        let cfg = &model.config;
        let metrics = crate::telemetry::Metrics::default();
        let toks = [3u32, 4];
        let alloc = |kv: &mut KvTierManager| {
            let mut pages = Vec::new();
            for layer in 0..cfg.n_layers {
                let id = kv
                    .alloc_page(
                        layer as u32,
                        0,
                        32,
                        cfg.n_kv_heads.max(1),
                        cfg.head_dim(),
                        &metrics,
                    )
                    .expect("kv page");
                pages.push(vec![id]);
            }
            pages
        };
        let mut kv_seq = KvTierManager::new(8 << 20, 8 << 20, 0.5);
        let pages_seq = alloc(&mut kv_seq);
        let mut logits_seq = vec![0.0f32; cfg.vocab_size];
        for (i, &tok) in toks.iter().enumerate() {
            super::forward_dsv4_token(
                &model,
                tok,
                i,
                &pages_seq,
                &mut kv_seq,
                &mut logits_seq,
                &metrics,
                None,
                None,
                true,
            )
            .expect("sequential token");
        }
        let mut kv_chunk = KvTierManager::new(8 << 20, 8 << 20, 0.5);
        let pages_chunk = alloc(&mut kv_chunk);
        let mut logits_chunk = vec![0.0f32; cfg.vocab_size];
        super::forward_dsv4_tokens(
            &model,
            &toks,
            0,
            &pages_chunk,
            &mut kv_chunk,
            &mut logits_chunk,
            &metrics,
            None,
            true,
        )
        .expect("chunk tokens");
        for (i, (a, b)) in logits_seq.iter().zip(logits_chunk.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "logit {i} seq={a} chunk={b}");
        }
        let _ = std::fs::remove_file(&path);
    }
}
