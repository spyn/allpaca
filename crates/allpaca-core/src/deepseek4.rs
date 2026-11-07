//! DeepSeek-V4-Flash (`deepseek4`) host decode: MLA + hyper-connections + MoE.
//!
//! What a token attends to (sliding window of raw keys, compressed keys, the
//! indexer) is in [`crate::dsv4_attn`].

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use allpaca_cpu::{attention, matmul_nt, rmsnorm, rope_offset, rope_offset_back};
use allpaca_gguf::GgufFile;
use allpaca_types::{AllpacaError, FatalError};

use crate::model::{
    expert_mlp_rows, topk_from_logits, DenseKv, ModelConfig, TinyModel,
};
use crate::telemetry::Metrics;
use crate::weight_set::LayerFfn;
use crate::dsv4_attn::{Dsv4AttnSpec, Dsv4Memory};
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
    pub attn: Dsv4AttnSpec,
    /// Per GGUF block: routed experts cap their gate at this before SiLU
    /// and their up projection to plus or minus it (empty or 0 = no cap).
    pub swiglu_clamp: Vec<f32>,
    /// The same cap for the shared expert (`swiglu_clamp_shexp`).
    pub swiglu_clamp_shexp: Vec<f32>,
}

/// The cap a per-block clamp list sets for `blk` (0 = none).
fn limit_at(limits: &[f32], blk: usize) -> f32 {
    match limits.get(blk) {
        Some(limit) if *limit > 1.0e-6 => *limit,
        _ => 0.0,
    }
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
            attn: Dsv4AttnSpec::from_gguf(file, arch, head_dim, n_rot, cfg.rope_theta, cfg.eps),
            swiglu_clamp: file
                .metadata
                .get(&format!("{arch}.swiglu_clamp_exp"))
                .and_then(|v| v.as_f32_array())
                .unwrap_or_default(),
            swiglu_clamp_shexp: file
                .metadata
                .get(&format!("{arch}.swiglu_clamp_shexp"))
                .and_then(|v| v.as_f32_array())
                .unwrap_or_default(),
        }
    }

    /// The cap for the routed experts of GGUF block `blk` (0 = none).
    pub fn swiglu_limit(&self, blk: usize) -> f32 {
        limit_at(&self.swiglu_clamp, blk)
    }

    /// The cap for the shared expert of GGUF block `blk` (0 = none).
    /// DeepSeek-V4 builds its shared expert with the routed experts'
    /// `swiglu_limit` (inference/model.py, `MoE.shared_experts`).
    pub fn shexp_swiglu_limit(&self, blk: usize) -> f32 {
        limit_at(&self.swiglu_clamp_shexp, blk)
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
    /// Compressor gate bias per position in a group and key norm; empty
    /// for a layer that keeps raw keys only.
    pub comp_ape: Vec<f32>,
    pub comp_norm: Vec<f32>,
    pub index_ape: Vec<f32>,
    pub index_norm: Vec<f32>,
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
    /// Host copies of dense tensors the GPU does not hold, decoded on first use.
    host_dense: Mutex<BTreeMap<String, Arc<Vec<f32>>>>,
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
                comp_ape: optional_f32(file, &format!("blk.{blk}.attn_compressor_ape.weight"))?,
                comp_norm: optional_f32(file, &format!("blk.{blk}.attn_compressor_norm.weight"))?,
                index_ape: optional_f32(file, &format!("blk.{blk}.indexer_compressor_ape.weight"))?,
                index_norm: optional_f32(file, &format!("blk.{blk}.indexer_compressor_norm.weight"))?,
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
            host_dense: Mutex::new(BTreeMap::new()),
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

#[cfg(test)]
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

fn rmsnorm_unit(x: &[f32], eps: f32, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() != x.len() || x.is_empty() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "rmsnorm_unit length mismatch".into(),
        )));
    }
    let mut ss = 0.0f32;
    for v in x {
        ss += *v * *v;
    }
    let inv = (ss / x.len() as f32 + eps).sqrt().recip();
    for i in 0..x.len() {
        out[i] = x[i] * inv;
    }
    Ok(())
}

/// Packed HC residual: `[n_tok][stream][dim]` in one allocation.
struct HcBatch {
    data: Vec<f32>,
    scratch: Vec<f32>,
    n_tok: usize,
    hc: usize,
    dim: usize,
}

impl HcBatch {
    fn new(n_tok: usize, hc: usize, dim: usize) -> Self {
        let n = n_tok.saturating_mul(hc).saturating_mul(dim);
        Self {
            data: vec![0.0f32; n],
            scratch: vec![0.0f32; n],
            n_tok,
            hc: hc.max(1),
            dim,
        }
    }

    fn fill_token_all_streams(&mut self, t: usize, x0: &[f32]) {
        let dim = self.dim;
        let n = x0.len().min(dim);
        for s in 0..self.hc {
            let off = t
                .saturating_mul(self.hc)
                .saturating_add(s)
                .saturating_mul(dim);
            if off + n <= self.data.len() {
                self.data[off..off + n].copy_from_slice(&x0[..n]);
            }
        }
    }

    fn last_streams(&self) -> Vec<Vec<f32>> {
        let t = self.n_tok.saturating_sub(1);
        let dim = self.dim;
        (0..self.hc)
            .map(|s| {
                let off = t
                    .saturating_mul(self.hc)
                    .saturating_add(s)
                    .saturating_mul(dim);
                if off + dim <= self.data.len() {
                    self.data[off..off + dim].to_vec()
                } else {
                    vec![0.0f32; dim]
                }
            })
            .collect()
    }
}

#[cfg(test)]
fn pack_hc_states(states: &[Vec<Vec<f32>>], hc: usize, dim: usize) -> Vec<f32> {
    let n_tok = states.len();
    let mut packed = vec![0.0f32; n_tok.saturating_mul(hc).saturating_mul(dim)];
    for (t, streams) in states.iter().enumerate() {
        for (s, row) in streams.iter().enumerate().take(hc) {
            let n = row.len().min(dim);
            let off = t.saturating_mul(hc).saturating_add(s).saturating_mul(dim);
            packed[off..off + n].copy_from_slice(&row[..n]);
        }
    }
    packed
}

fn unpack_hc_states(packed: &[f32], n_tok: usize, hc: usize, dim: usize) -> Vec<Vec<Vec<f32>>> {
    let mut states = Vec::with_capacity(n_tok);
    for t in 0..n_tok {
        let mut streams = Vec::with_capacity(hc);
        for s in 0..hc {
            let off = t.saturating_mul(hc).saturating_add(s).saturating_mul(dim);
            let mut row = vec![0.0f32; dim];
            if off < packed.len() {
                let n = dim.min(packed.len().saturating_sub(off));
                row[..n].copy_from_slice(&packed[off..off + n]);
            }
            streams.push(row);
        }
        states.push(streams);
    }
    states
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
    rmsnorm_unit(&flat, eps, &mut normed)?;
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
    hc_mix_streams(&mut out, streams, &pre, dim);
    Ok((out, post, comb))
}

/// Token-parallel [`hc_pre`]: one mix GEMM at `M=n_tok`, per-token Sinkhorn.
#[cfg(test)]
fn hc_pre_rows(
    states: &[Vec<Vec<f32>>],
    mix: &HcMix,
    spec: &Dsv4Spec,
    dim: usize,
    eps: f32,
) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>), AllpacaError> {
    let hc = spec.hc_mult.max(1);
    let mut packed = pack_hc_states(states, hc, dim);
    hc_pre_packed(
        &mut packed,
        states.len(),
        mix,
        spec,
        dim,
        eps,
        None,
        None,
        None,
        None,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn hc_pre_packed(
    packed: &mut [f32],
    n_tok: usize,
    mix: &HcMix,
    spec: &Dsv4Spec,
    dim: usize,
    eps: f32,
    mut gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    mix_name: Option<&str>,
    norm_w: Option<&[f32]>,
    metrics: Option<&Metrics>,
    copy_curs: bool,
) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>), AllpacaError> {
    let hc = spec.hc_mult.max(1);
    if n_tok == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let hc_dim = dim.saturating_mul(hc);
    let need = n_tok.saturating_mul(hc_dim);
    if mix.fn_w.is_empty() || packed.len() < need {
        let states = unpack_hc_states(packed, n_tok, hc, dim);
        let mut curs = vec![0.0f32; n_tok.saturating_mul(dim)];
        let mut posts = Vec::with_capacity(n_tok);
        let mut combs = Vec::with_capacity(n_tok);
        for (t, streams) in states.iter().enumerate() {
            let (cur, post, comb) = hc_pre(streams, mix, spec, dim, eps)?;
            let n = cur.len().min(dim);
            curs[t * dim..t * dim + n].copy_from_slice(&cur[..n]);
            posts.push(post);
            combs.push(comb);
        }
        apply_curs_norm(
            &mut curs,
            n_tok,
            dim,
            eps,
            gpu.as_deref_mut(),
            norm_w,
            metrics,
        )?;
        return Ok((curs, posts, combs));
    }
    let mix_dim = (2 + hc).saturating_mul(hc);
    let mut mixes = vec![0.0f32; n_tok.saturating_mul(mix_dim)];
    let mut used_gpu = false;
    if let Some(g) = gpu.as_mut() {
        used_gpu = g
            .hc_mix_logits(
                packed, &mix.fn_w, mix_name, n_tok, hc_dim, mix_dim, eps, &mut mixes,
            )
            .is_ok();
        if !used_gpu {
            let _ = g.download_hc_packed(packed);
        }
    }
    if !used_gpu {
        let mut normed = vec![0.0f32; need];
        for t in 0..n_tok {
            let a = t.saturating_mul(hc_dim);
            rmsnorm_unit(&packed[a..a + hc_dim], eps, &mut normed[a..a + hc_dim])?;
        }
        matmul_nt(&normed, &mix.fn_w, &mut mixes, n_tok, mix_dim, hc_dim)?;
    }
    let scale_pre = mix.scale.first().copied().unwrap_or(1.0);
    let scale_post = mix.scale.get(1).copied().unwrap_or(1.0);
    let scale_comb = mix.scale.get(2).copied().unwrap_or(1.0);
    let mut curs = vec![0.0f32; n_tok.saturating_mul(dim)];
    let mut posts = Vec::with_capacity(n_tok);
    let mut combs = Vec::with_capacity(n_tok);
    let mut pres = vec![0.0f32; n_tok.saturating_mul(hc)];
    for t in 0..n_tok {
        let row = &mixes[t * mix_dim..(t + 1) * mix_dim];
        let mut pre = vec![0.0f32; hc];
        let mut post = vec![0.0f32; hc];
        for i in 0..hc {
            let b = mix.base.get(i).copied().unwrap_or(0.0);
            pre[i] = sigmoid(row[i] * scale_pre + b) + spec.hc_eps;
            let b2 = mix.base.get(hc + i).copied().unwrap_or(0.0);
            post[i] = sigmoid(row[hc + i] * scale_post + b2) * 2.0;
        }
        pres[t * hc..t * hc + hc].copy_from_slice(&pre);
        let mut comb = vec![0.0f32; hc.saturating_mul(hc)];
        for i in 0..hc.saturating_mul(hc) {
            let b = mix.base.get(2 * hc + i).copied().unwrap_or(0.0);
            comb[i] = row[2 * hc + i] * scale_comb + b;
        }
        sinkhorn(&mut comb, hc, spec.hc_sinkhorn_iters, spec.hc_eps);
        posts.push(post);
        combs.push(comb);
    }
    let mut mixed_gpu = false;
    if used_gpu {
        if let Some(g) = gpu.as_mut() {
            mixed_gpu = g
                .hc_mix_pre(&pres, n_tok, hc, dim, &mut curs, norm_w, eps, copy_curs)
                .is_ok();
            if !mixed_gpu {
                let _ = g.download_hc_packed(packed);
            }
        }
    }
    if !mixed_gpu {
        for t in 0..n_tok {
            let dst = &mut curs[t * dim..(t + 1) * dim];
            let tok = &packed[t * hc_dim..(t + 1) * hc_dim];
            hc_mix_packed(dst, tok, &pres[t * hc..t * hc + hc], hc, dim);
        }
    }
    let gpu_normed = mixed_gpu && norm_w.is_some_and(|w| w.len() >= dim);
    if !gpu_normed {
        if !copy_curs {
            if let Some(g) = gpu.as_mut() {
                g.ensure_hc_curs_host(&mut curs)?;
            }
        }
        apply_curs_norm(
            &mut curs,
            n_tok,
            dim,
            eps,
            gpu.as_deref_mut(),
            norm_w,
            metrics,
        )?;
    }
    Ok((curs, posts, combs))
}

fn apply_curs_norm(
    curs: &mut [f32],
    n_tok: usize,
    dim: usize,
    eps: f32,
    gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    norm_w: Option<&[f32]>,
    metrics: Option<&Metrics>,
) -> Result<(), AllpacaError> {
    let Some(w) = norm_w.filter(|w| w.len() >= dim) else {
        return Ok(());
    };
    if n_tok == 0 {
        return Ok(());
    }
    if let Some(m) = metrics {
        rmsnorm_rows_maybe_gpu(gpu, curs, w, n_tok, dim, eps, m)?;
        return Ok(());
    }
    let mut tmp = vec![0.0f32; dim];
    for t in 0..n_tok {
        let row = &mut curs[t * dim..(t + 1) * dim];
        rmsnorm(row, w, eps, &mut tmp)?;
        row.copy_from_slice(&tmp);
    }
    Ok(())
}

fn ensure_hc_buf(buf: &mut Vec<Vec<f32>>, hc: usize, dim: usize) {
    if buf.len() != hc {
        *buf = vec![vec![0.0f32; dim]; hc];
        return;
    }
    for row in buf.iter_mut() {
        if row.len() != dim {
            row.resize(dim, 0.0);
        }
    }
}

/// Weighted sum of HC streams into one `dim` vector (fused for `hc` 2/4).
fn hc_mix_streams(dst: &mut [f32], streams: &[Vec<f32>], weights: &[f32], dim: usize) {
    let n = dim.min(dst.len());
    dst[..n].fill(0.0);
    if streams.len() >= 4
        && weights.len() >= 4
        && streams[0].len() >= n
        && streams[1].len() >= n
        && streams[2].len() >= n
        && streams[3].len() >= n
    {
        let (w0, w1, w2, w3) = (weights[0], weights[1], weights[2], weights[3]);
        let s0 = &streams[0][..n];
        let s1 = &streams[1][..n];
        let s2 = &streams[2][..n];
        let s3 = &streams[3][..n];
        for i in 0..n {
            dst[i] = s0[i] * w0 + s1[i] * w1 + s2[i] * w2 + s3[i] * w3;
        }
        return;
    }
    if streams.len() >= 2 && weights.len() >= 2 && streams[0].len() >= n && streams[1].len() >= n {
        let (w0, w1) = (weights[0], weights[1]);
        let s0 = &streams[0][..n];
        let s1 = &streams[1][..n];
        for i in 0..n {
            dst[i] = s0[i] * w0 + s1[i] * w1;
        }
        for (s, stream) in streams.iter().enumerate().skip(2) {
            let w = weights.get(s).copied().unwrap_or(0.0);
            let n2 = n.min(stream.len());
            for i in 0..n2 {
                dst[i] += stream[i] * w;
            }
        }
        return;
    }
    for (s, stream) in streams.iter().enumerate() {
        let w = weights.get(s).copied().unwrap_or(0.0);
        let n2 = n.min(stream.len());
        for i in 0..n2 {
            dst[i] += stream[i] * w;
        }
    }
}

fn hc_mix_packed(dst: &mut [f32], token_hc: &[f32], weights: &[f32], hc: usize, dim: usize) {
    let n = dim.min(dst.len());
    dst[..n].fill(0.0);
    if hc >= 4 && weights.len() >= 4 && token_hc.len() >= 4 * dim {
        let (w0, w1, w2, w3) = (weights[0], weights[1], weights[2], weights[3]);
        let s0 = &token_hc[0..n];
        let s1 = &token_hc[dim..dim + n];
        let s2 = &token_hc[2 * dim..2 * dim + n];
        let s3 = &token_hc[3 * dim..3 * dim + n];
        for i in 0..n {
            dst[i] = s0[i] * w0 + s1[i] * w1 + s2[i] * w2 + s3[i] * w3;
        }
        return;
    }
    if hc >= 2 && weights.len() >= 2 && token_hc.len() >= 2 * dim {
        let (w0, w1) = (weights[0], weights[1]);
        let s0 = &token_hc[0..n];
        let s1 = &token_hc[dim..dim + n];
        for i in 0..n {
            dst[i] = s0[i] * w0 + s1[i] * w1;
        }
        for s in 2..hc {
            let w = weights.get(s).copied().unwrap_or(0.0);
            let off = s.saturating_mul(dim);
            let n2 = n.min(token_hc.len().saturating_sub(off));
            for i in 0..n2 {
                dst[i] += token_hc[off + i] * w;
            }
        }
        return;
    }
    for s in 0..hc {
        let w = weights.get(s).copied().unwrap_or(0.0);
        let off = s.saturating_mul(dim);
        let n2 = n.min(token_hc.len().saturating_sub(off));
        for i in 0..n2 {
            dst[i] += token_hc[off + i] * w;
        }
    }
}

fn hc_post_into(
    x: &[f32],
    residual: &[Vec<f32>],
    post: &[f32],
    comb: &[f32],
    spec: &Dsv4Spec,
    dim: usize,
    out: &mut Vec<Vec<f32>>,
) {
    let hc = spec.hc_mult.max(1);
    ensure_hc_buf(out, hc, dim);
    let n = dim.min(x.len());
    for row in out.iter_mut().take(hc) {
        for v in row.iter_mut().skip(n) {
            *v = 0.0;
        }
    }
    if hc == 4
        && residual.len() >= 4
        && residual[0].len() >= n
        && residual[1].len() >= n
        && residual[2].len() >= n
        && residual[3].len() >= n
        && out.len() >= 4
    {
        let r0 = &residual[0][..n];
        let r1 = &residual[1][..n];
        let r2 = &residual[2][..n];
        let r3 = &residual[3][..n];
        let xn = &x[..n];
        let pw0 = post.first().copied().unwrap_or(1.0);
        let pw1 = post.get(1).copied().unwrap_or(1.0);
        let pw2 = post.get(2).copied().unwrap_or(1.0);
        let pw3 = post.get(3).copied().unwrap_or(1.0);
        let w00 = comb.first().copied().unwrap_or(0.0);
        let w10 = comb.get(1).copied().unwrap_or(0.0);
        let w20 = comb.get(2).copied().unwrap_or(0.0);
        let w30 = comb.get(3).copied().unwrap_or(0.0);
        let w01 = comb.get(4).copied().unwrap_or(0.0);
        let w11 = comb.get(5).copied().unwrap_or(0.0);
        let w21 = comb.get(6).copied().unwrap_or(0.0);
        let w31 = comb.get(7).copied().unwrap_or(0.0);
        let w02 = comb.get(8).copied().unwrap_or(0.0);
        let w12 = comb.get(9).copied().unwrap_or(0.0);
        let w22 = comb.get(10).copied().unwrap_or(0.0);
        let w32 = comb.get(11).copied().unwrap_or(0.0);
        let w03 = comb.get(12).copied().unwrap_or(0.0);
        let w13 = comb.get(13).copied().unwrap_or(0.0);
        let w23 = comb.get(14).copied().unwrap_or(0.0);
        let w33 = comb.get(15).copied().unwrap_or(0.0);
        let (s0, rest) = out.split_at_mut(1);
        let (s1, rest) = rest.split_at_mut(1);
        let (s2, rest) = rest.split_at_mut(1);
        let o0 = &mut s0[0][..n];
        let o1 = &mut s1[0][..n];
        let o2 = &mut s2[0][..n];
        let o3 = &mut rest[0][..n];
        for i in 0..n {
            let xv = xn[i];
            let a = r0[i];
            let b = r1[i];
            let c = r2[i];
            let d = r3[i];
            o0[i] = xv * pw0 + a * w00 + b * w01 + c * w02 + d * w03;
            o1[i] = xv * pw1 + a * w10 + b * w11 + c * w12 + d * w13;
            o2[i] = xv * pw2 + a * w20 + b * w21 + c * w22 + d * w23;
            o3[i] = xv * pw3 + a * w30 + b * w31 + c * w32 + d * w33;
        }
        return;
    }
    if hc == 2
        && residual.len() >= 2
        && residual[0].len() >= n
        && residual[1].len() >= n
        && out.len() >= 2
    {
        let r0 = &residual[0][..n];
        let r1 = &residual[1][..n];
        let xn = &x[..n];
        let pw0 = post.first().copied().unwrap_or(1.0);
        let pw1 = post.get(1).copied().unwrap_or(1.0);
        let w00 = comb.first().copied().unwrap_or(0.0);
        let w10 = comb.get(1).copied().unwrap_or(0.0);
        let w01 = comb.get(2).copied().unwrap_or(0.0);
        let w11 = comb.get(3).copied().unwrap_or(0.0);
        let (s0, rest) = out.split_at_mut(1);
        let o0 = &mut s0[0][..n];
        let o1 = &mut rest[0][..n];
        for i in 0..n {
            let xv = xn[i];
            let a = r0[i];
            let b = r1[i];
            o0[i] = xv * pw0 + a * w00 + b * w01;
            o1[i] = xv * pw1 + a * w10 + b * w11;
        }
        return;
    }
    for dst in 0..hc {
        let pw = post.get(dst).copied().unwrap_or(1.0);
        let o = &mut out[dst][..n];
        let xn = &x[..n];
        for i in 0..n {
            o[i] = xn[i] * pw;
        }
        for src in 0..hc {
            let w = comb.get(dst + src * hc).copied().unwrap_or(0.0);
            if let Some(row) = residual.get(src) {
                let m = n.min(row.len());
                for i in 0..m {
                    o[i] += row[i] * w;
                }
            }
        }
    }
}

fn hc_post(
    x: &[f32],
    residual: &[Vec<f32>],
    post: &[f32],
    comb: &[f32],
    spec: &Dsv4Spec,
    dim: usize,
) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    hc_post_into(x, residual, post, comb, spec, dim, &mut out);
    out
}

/// Packed [`hc_post`]: one scratch buffer, swapped into `states` (no per-token alloc).
#[cfg(test)]
fn hc_post_rows(
    xs: &[f32],
    states: &mut [Vec<Vec<f32>>],
    posts: &[Vec<f32>],
    combs: &[Vec<f32>],
    spec: &Dsv4Spec,
    dim: usize,
) {
    hc_post_rows_with(xs, states, spec, dim, |t| {
        (
            posts.get(t).map(Vec::as_slice).unwrap_or(&[]),
            combs.get(t).map(Vec::as_slice).unwrap_or(&[]),
        )
    });
}

#[cfg(test)]
fn hc_post_rows_with<'p>(
    xs: &[f32],
    states: &mut [Vec<Vec<f32>>],
    spec: &Dsv4Spec,
    dim: usize,
    mut post_comb: impl FnMut(usize) -> (&'p [f32], &'p [f32]),
) {
    let hc = spec.hc_mult.max(1);
    let mut scratch = vec![vec![0.0f32; dim]; hc];
    for t in 0..states.len() {
        let start = t.saturating_mul(dim);
        let x = if start < xs.len() {
            &xs[start..xs.len().min(start.saturating_add(dim))]
        } else {
            &[]
        };
        let (post, comb) = post_comb(t);
        hc_post_into(x, &states[t], post, comb, spec, dim, &mut scratch);
        std::mem::swap(&mut states[t], &mut scratch);
    }
}

fn hc_post_into_flat(
    x: &[f32],
    residual: &[f32],
    post: &[f32],
    comb: &[f32],
    spec: &Dsv4Spec,
    dim: usize,
    out: &mut [f32],
) {
    let hc = spec.hc_mult.max(1);
    let n = dim.min(x.len());
    let need = hc.saturating_mul(dim);
    if residual.len() < need || out.len() < need {
        out.fill(0.0);
        return;
    }
    for dst in 0..hc {
        for i in n..dim {
            out[dst * dim + i] = 0.0;
        }
    }
    if hc == 4 {
        let r0 = &residual[0..n];
        let r1 = &residual[dim..dim + n];
        let r2 = &residual[2 * dim..2 * dim + n];
        let r3 = &residual[3 * dim..3 * dim + n];
        let xn = &x[..n];
        let pw0 = post.first().copied().unwrap_or(1.0);
        let pw1 = post.get(1).copied().unwrap_or(1.0);
        let pw2 = post.get(2).copied().unwrap_or(1.0);
        let pw3 = post.get(3).copied().unwrap_or(1.0);
        let w00 = comb.first().copied().unwrap_or(0.0);
        let w10 = comb.get(1).copied().unwrap_or(0.0);
        let w20 = comb.get(2).copied().unwrap_or(0.0);
        let w30 = comb.get(3).copied().unwrap_or(0.0);
        let w01 = comb.get(4).copied().unwrap_or(0.0);
        let w11 = comb.get(5).copied().unwrap_or(0.0);
        let w21 = comb.get(6).copied().unwrap_or(0.0);
        let w31 = comb.get(7).copied().unwrap_or(0.0);
        let w02 = comb.get(8).copied().unwrap_or(0.0);
        let w12 = comb.get(9).copied().unwrap_or(0.0);
        let w22 = comb.get(10).copied().unwrap_or(0.0);
        let w32 = comb.get(11).copied().unwrap_or(0.0);
        let w03 = comb.get(12).copied().unwrap_or(0.0);
        let w13 = comb.get(13).copied().unwrap_or(0.0);
        let w23 = comb.get(14).copied().unwrap_or(0.0);
        let w33 = comb.get(15).copied().unwrap_or(0.0);
        let (o0, rest) = out.split_at_mut(dim);
        let (o1, rest) = rest.split_at_mut(dim);
        let (o2, rest) = rest.split_at_mut(dim);
        let o0 = &mut o0[..n];
        let o1 = &mut o1[..n];
        let o2 = &mut o2[..n];
        let o3 = &mut rest[..n];
        for i in 0..n {
            let xv = xn[i];
            let a = r0[i];
            let b = r1[i];
            let c = r2[i];
            let d = r3[i];
            o0[i] = xv * pw0 + a * w00 + b * w01 + c * w02 + d * w03;
            o1[i] = xv * pw1 + a * w10 + b * w11 + c * w12 + d * w13;
            o2[i] = xv * pw2 + a * w20 + b * w21 + c * w22 + d * w23;
            o3[i] = xv * pw3 + a * w30 + b * w31 + c * w32 + d * w33;
        }
        return;
    }
    if hc == 2 {
        let r0 = &residual[0..n];
        let r1 = &residual[dim..dim + n];
        let xn = &x[..n];
        let pw0 = post.first().copied().unwrap_or(1.0);
        let pw1 = post.get(1).copied().unwrap_or(1.0);
        let w00 = comb.first().copied().unwrap_or(0.0);
        let w10 = comb.get(1).copied().unwrap_or(0.0);
        let w01 = comb.get(2).copied().unwrap_or(0.0);
        let w11 = comb.get(3).copied().unwrap_or(0.0);
        let (o0, rest) = out.split_at_mut(dim);
        let o0 = &mut o0[..n];
        let o1 = &mut rest[..n];
        for i in 0..n {
            let xv = xn[i];
            let a = r0[i];
            let b = r1[i];
            o0[i] = xv * pw0 + a * w00 + b * w01;
            o1[i] = xv * pw1 + a * w10 + b * w11;
        }
        return;
    }
    for dst in 0..hc {
        let pw = post.get(dst).copied().unwrap_or(1.0);
        let o = &mut out[dst * dim..dst * dim + n];
        let xn = &x[..n];
        for i in 0..n {
            o[i] = xn[i] * pw;
        }
        for src in 0..hc {
            let w = comb.get(dst + src * hc).copied().unwrap_or(0.0);
            let off = src.saturating_mul(dim);
            let m = n.min(residual.len().saturating_sub(off));
            for i in 0..m {
                o[i] += residual[off + i] * w;
            }
        }
    }
}

fn hc_post_batch<'p>(
    xs: &[f32],
    batch: &mut HcBatch,
    spec: &Dsv4Spec,
    dim: usize,
    mut gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    mut post_comb: impl FnMut(usize) -> (&'p [f32], &'p [f32]),
    xs_dev: Option<*const f32>,
) -> bool {
    let hc = spec.hc_mult.max(1);
    let hcd = hc.saturating_mul(dim);
    let n_tok = batch.n_tok;
    let mut posts = vec![0.0f32; n_tok.saturating_mul(hc)];
    let mut combs = vec![0.0f32; n_tok.saturating_mul(hc).saturating_mul(hc)];
    for t in 0..n_tok {
        let (post, comb) = post_comb(t);
        let po = t.saturating_mul(hc);
        let co = t.saturating_mul(hc).saturating_mul(hc);
        for i in 0..hc.min(post.len()) {
            posts[po + i] = post[i];
        }
        for i in 0..(hc.saturating_mul(hc)).min(comb.len()) {
            combs[co + i] = comb[i];
        }
    }
    if let Some(g) = gpu.as_mut() {
        if g.hc_post_packed(&batch.data, xs, &posts, &combs, n_tok, hc, dim, xs_dev)
            .is_ok()
        {
            return true;
        }
        if xs.len() >= n_tok.saturating_mul(dim) {
            let _ = g.download_hc_packed(&mut batch.data);
            g.invalidate_hc_device();
        }
    }
    if xs.len() < n_tok.saturating_mul(dim) {
        return false;
    }
    for t in 0..n_tok {
        let start = t.saturating_mul(dim);
        let x = if start < xs.len() {
            &xs[start..xs.len().min(start.saturating_add(dim))]
        } else {
            &[]
        };
        let post = &posts[t.saturating_mul(hc)..t.saturating_mul(hc).saturating_add(hc)];
        let comb = &combs[t * hc * hc..(t + 1) * hc * hc];
        let off = t.saturating_mul(hcd);
        let residual = &batch.data[off..off + hcd];
        let out = &mut batch.scratch[off..off + hcd];
        hc_post_into_flat(x, residual, post, comb, spec, dim, out);
    }
    std::mem::swap(&mut batch.data, &mut batch.scratch);
    true
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
    rmsnorm_unit(&flat, eps, &mut normed)?;
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

fn rmsnorm_rows_maybe_gpu(
    gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    rows: &mut [f32],
    weight: &[f32],
    n_tok: usize,
    dim: usize,
    eps: f32,
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    if n_tok == 0 || weight.is_empty() {
        return Ok(());
    }
    let need = n_tok.saturating_mul(dim);
    if rows.len() < need {
        return Ok(());
    }
    let mut tmp = vec![0.0f32; need];
    let t = Instant::now();
    if let Some(g) = gpu {
        if g.rmsnorm_rows(rows, weight, &mut tmp, n_tok, dim, eps)
            .is_ok()
        {
            rows[..need].copy_from_slice(&tmp);
            metrics.dsv4_norm_ns.fetch_add(
                t.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            return Ok(());
        }
    }
    drop(tmp);
    let mut row_tmp = vec![0.0f32; dim];
    for i in 0..n_tok {
        let row = &mut rows[i * dim..(i + 1) * dim];
        timed_rmsnorm(row, weight, eps, &mut row_tmp, metrics)?;
        row.copy_from_slice(&row_tmp);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn attention_sinks(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sinks: &[f32],
    out: &mut [f32],
    n_q: usize,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<(), AllpacaError> {
    if sinks.is_empty() {
        return attention(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, true);
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_dim = n_kv_heads.saturating_mul(head_dim);
    let q_stride = n_heads.saturating_mul(head_dim);
    let prefix = n_kv.saturating_sub(n_q);
    out.fill(0.0);
    let mut scores = vec![0.0f32; n_kv];
    for qi in 0..n_q {
        let t_end = (qi + prefix + 1).min(n_kv);
        let q_base = qi.saturating_mul(q_stride);
        for h in 0..n_heads {
            let qh = &q[q_base + h * head_dim..q_base + (h + 1) * head_dim];
            let kv_h = (h * n_kv_heads) / n_heads.max(1);
            let sink = sinks.get(h).copied().unwrap_or(0.0);
            let mut max = sink;
            for t in 0..t_end {
                let k_off = t * kv_dim + kv_h * head_dim;
                let kh = &k[k_off..k_off + head_dim];
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
            for s in &mut scores[..t_end] {
                *s = (*s - max).exp();
                acc += *s;
            }
            let inv = 1.0 / acc.max(1e-9);
            let oh = &mut out[q_base + h * head_dim..q_base + (h + 1) * head_dim];
            for t in 0..t_end {
                let w = scores[t] * inv;
                let v_off = t * kv_dim + kv_h * head_dim;
                let vh = &v[v_off..v_off + head_dim];
                for i in 0..head_dim {
                    oh[i] += w * vh[i];
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dsv4_attention_sinks_maybe_gpu(
    skip_mla: bool,
    gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    hip: Option<&allpaca_hip::HipWorkspace>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sinks: &[f32],
    out: &mut [f32],
    n_q: usize,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<(), AllpacaError> {
    if skip_mla {
        let n_q_i = n_q as i32;
        let n_kv_i = n_kv as i32;
        let n_heads_i = n_heads as i32;
        let n_kv_heads_i = n_kv_heads as i32;
        let head_dim_i = head_dim as i32;
        if let Some(g) = gpu {
            if g.attention_sinks(
                q,
                k,
                v,
                sinks,
                out,
                n_q_i,
                n_kv_i,
                n_heads_i,
                n_kv_heads_i,
                head_dim_i,
                true,
            )
            .is_ok()
            {
                return Ok(());
            }
        } else if let Some(ws) = hip {
            if ws
                .attention_sinks(
                    q,
                    k,
                    v,
                    sinks,
                    out,
                    n_q_i,
                    n_kv_i,
                    n_heads_i,
                    n_kv_heads_i,
                    head_dim_i,
                    true,
                )
                .is_ok()
            {
                return Ok(());
            }
        }
    }
    attention_sinks(
        q, k, v, sinks, out, n_q, n_kv, n_heads, n_kv_heads, head_dim,
    )
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
        allpaca_gguf::GgmlType::Q8_0 if dim % 32 == 0 => {
            // One row: `dim / 32` blocks of an f16 scale and 32 int8.
            let row = dim / 32 * 34;
            let off = token as usize * row;
            let blocks = bytes.get(off..off + row).ok_or_else(|| {
                AllpacaError::Fatal(FatalError::CorruptGguf("embed Q8_0 row OOB".into()))
            })?;
            for (b, blk) in blocks.chunks_exact(34).enumerate() {
                let d = allpaca_cpu::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for (i, q) in blk[2..].iter().enumerate() {
                    out[b * 32 + i] = d * f32::from(*q as i8);
                }
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

/// Dense tensor types the HIP path keeps resident and multiplies on the
/// GPU (BF16 as it is, Q8_0 as staged int8 blocks).
fn gpu_dense_type(t: allpaca_gguf::GgmlType) -> bool {
    matches!(t, allpaca_gguf::GgmlType::BF16 | allpaca_gguf::GgmlType::Q8_0)
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

/// Rotate one key or query head to `pos`: the layer's own frequencies when
/// it compresses, the model's plain RoPE otherwise.
fn dsv4_rope(
    cfg: &ModelConfig,
    spec: &Dsv4Spec,
    ratio: usize,
    row: &mut [f32],
    pos: usize,
    inverse: bool,
) -> Result<(), AllpacaError> {
    if ratio != 0 {
        spec.attn.rope(ratio).rotate(row, pos, inverse);
        Ok(())
    } else if inverse {
        rope_offset_back(row, pos, cfg.rope_theta, cfg.rope_dims, spec.rope_offset)
    } else {
        rope_offset(row, pos, cfg.rope_theta, cfg.rope_dims, spec.rope_offset)
    }
}

/// Raw keys of positions `lo..hi` from a layer's KV pages.
fn gather_raw_keys(
    kv: &KvTierManager,
    pages: &[u64],
    lo: usize,
    hi: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, AllpacaError> {
    let kv_dim = n_kv_heads * head_dim;
    let mut out = Vec::with_capacity(hi.saturating_sub(lo) * kv_dim);
    if hi <= lo {
        return Ok(out);
    }
    let cap = pages
        .first()
        .and_then(|id| kv.get(*id))
        .map(|p| p.capacity_tokens as usize)
        .unwrap_or(128)
        .max(1);
    for page in lo / cap..=(hi - 1) / cap {
        let id = pages.get(page).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "DeepSeek-V4 KV page {page} not allocated"
            )))
        })?;
        let (k, _) = kv.read_kv(*id, n_kv_heads, head_dim)?;
        let base = page * cap;
        let (a, b) = (lo.max(base) - base, hi.min(base + cap) - base);
        let rows = k.get(a * kv_dim..b * kv_dim).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "DeepSeek-V4 KV page {page} holds {} keys, need {b}",
                k.len() / kv_dim.max(1)
            )))
        })?;
        out.extend_from_slice(rows);
    }
    Ok(out)
}

/// `m` rows of `x` times the dense tensor `name` (`n` outputs of `k`
/// inputs): on the GPU when it holds the tensor, else from a host copy.
#[allow(clippy::too_many_arguments)]
fn dsv4_dense_rows(
    dsv4: &Dsv4Weights,
    gpu: Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
    name: &str,
    x: &[f32],
    n: usize,
    k: usize,
    m: usize,
    metrics: &Metrics,
) -> Result<Vec<f32>, AllpacaError> {
    let mut out = vec![0.0f32; m * n];
    if let (Some(gpu), Some(file)) = (gpu, file) {
        if gpu
            .gemm_bf16(file, name, x, &mut out, n, k, 0, m, Some(metrics))
            .is_ok()
        {
            return Ok(out);
        }
    }
    let file = file.ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!("no weights for {name}")))
    })?;
    let w = {
        let mut cache = dsv4.host_dense.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::ModelLoad("dsv4 dense cache poisoned".into()))
        })?;
        match cache.get(name) {
            Some(w) => w.clone(),
            None => {
                let w = Arc::new(file.tensor_f32(name)?);
                cache.insert(name.to_string(), w.clone());
                w
            }
        }
    };
    if w.len() < n * k {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "{name} holds {} values, need {}",
            w.len(),
            n * k
        ))));
    }
    matmul_nt(x, &w[..n * k], &mut out, m, n, k)?;
    Ok(out)
}

/// What a layer's tokens may attend to, once [`dsv4_absorb`] has taken them in.
struct Dsv4Keys {
    /// Raw keys from position `raw_start` up to the forward's last token.
    raw: Vec<f32>,
    raw_start: usize,
    /// Set once the indexer has to choose among the compressed keys.
    index_query: Option<crate::dsv4_attn::IndexQuery>,
}

/// Take a forward's tokens into a layer's attention memory. Their raw keys
/// are already in the KV pages; this runs the layer's compressors on `curs`
/// (the layer's normed input, on the host) and gathers the raw keys the
/// tokens may see.
#[allow(clippy::too_many_arguments)]
fn dsv4_absorb(
    model: &TinyModel,
    dsv4: &Dsv4Weights,
    layer_idx: usize,
    ratio: usize,
    curs: &[f32],
    q_a_norm: &[f32],
    start_pos: usize,
    n_tok: usize,
    pages: &[u64],
    kv: &KvTierManager,
    seq: &mut Dsv4Memory,
    gpu: &mut Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    file: Option<&GgufFile>,
    metrics: &Metrics,
) -> Result<Dsv4Keys, AllpacaError> {
    use crate::dsv4_attn::{IndexQuery, PoolCfg};
    let cfg = &model.config;
    let spec = &dsv4.spec;
    let attn = &spec.attn;
    let meta = &dsv4.layers[layer_idx];
    let (dim, eps, head_dim) = (cfg.dim, cfg.eps, cfg.head_dim());
    let blk = cfg.gguf_blk(layer_idx);
    let end = start_pos + n_tok;
    let raw_start = attn.raw_lo(start_pos);
    let raw = gather_raw_keys(kv, pages, raw_start, end, cfg.n_kv_heads.max(1), head_dim)?;
    let mut index_query = None;
    if ratio != 0 {
        let rope = attn.rope(ratio);
        let mut rows = |name: &str, x: &[f32], n: usize, k: usize| {
            dsv4_dense_rows(
                dsv4,
                gpu.as_deref_mut(),
                file,
                &format!("blk.{blk}.{name}.weight"),
                x,
                n,
                k,
                n_tok,
                metrics,
            )
        };
        let pool = PoolCfg {
            ratio,
            head: head_dim,
            ape: &meta.comp_ape,
            norm: &meta.comp_norm,
            eps,
            rope,
            fp8_nope: attn.nope,
            index_qat: false,
        };
        let kv_rows = rows("attn_compressor_kv", curs, pool.width(), dim)?;
        let gate_rows = rows("attn_compressor_gate", curs, pool.width(), dim)?;
        if ratio == 4 && !meta.index_ape.is_empty() {
            let index_pool = PoolCfg {
                ratio,
                head: attn.index_dim,
                ape: &meta.index_ape,
                norm: &meta.index_norm,
                eps,
                rope,
                fp8_nope: 0,
                index_qat: true,
            };
            let index_kv = rows("indexer_compressor_kv", curs, index_pool.width(), dim)?;
            let index_gate = rows("indexer_compressor_gate", curs, index_pool.width(), dim)?;
            seq.absorb(
                layer_idx,
                start_pos,
                n_tok,
                (&pool, &kv_rows, &gate_rows),
                Some((&index_pool, &index_kv, &index_gate)),
            )?;
            // The indexer only chooses once more keys are visible than it keeps.
            if attn.top_k > 0 && end / ratio > attn.top_k {
                let rank = spec.q_lora_rank;
                let mut qr = rows("attn_q_a", curs, rank, dim)?;
                if !q_a_norm.is_empty() {
                    let mut normed = vec![0.0f32; rank];
                    for row in qr.chunks_mut(rank) {
                        rmsnorm(row, q_a_norm, eps, &mut normed)?;
                        row.copy_from_slice(&normed);
                    }
                }
                let (heads, idim) = (attn.index_heads, attn.index_dim);
                let mut q = rows("indexer.attn_q_b", &qr, heads * idim, rank)?;
                for (t, tok) in q.chunks_mut(heads * idim).enumerate() {
                    for head in tok.chunks_mut(idim) {
                        rope.rotate(head, start_pos + t, false);
                        crate::dsv4_attn::index_qat(head);
                    }
                }
                index_query = Some(IndexQuery {
                    q,
                    head_weights: rows("indexer.proj", curs, heads, dim)?,
                });
            }
        } else {
            seq.absorb(layer_idx, start_pos, n_tok, (&pool, &kv_rows, &gate_rows), None)?;
        }
    }
    Ok(Dsv4Keys {
        raw,
        raw_start,
        index_query,
    })
}

/// The keys the `t`-th token of a forward attends to, into `out`.
#[allow(clippy::too_many_arguments)]
fn dsv4_token_keys(
    model: &TinyModel,
    dsv4: &Dsv4Weights,
    layer_idx: usize,
    ratio: usize,
    keys: &Dsv4Keys,
    start_pos: usize,
    t: usize,
    seq: &Dsv4Memory,
    out: &mut Vec<f32>,
) -> Result<(), AllpacaError> {
    let attn = &dsv4.spec.attn;
    let (heads, idim) = (attn.index_heads, attn.index_dim);
    let index_query = keys.index_query.as_ref().map(|iq| {
        (
            &iq.q[t * heads * idim..(t + 1) * heads * idim],
            &iq.head_weights[t * heads..(t + 1) * heads],
        )
    });
    seq.keys(
        attn,
        layer_idx,
        ratio,
        start_pos + t,
        model.config.head_dim(),
        &keys.raw,
        keys.raw_start,
        index_query,
        out,
    )
}

/// Attention on the host for tokens that each see their own set of keys:
/// `q` holds their rotated queries. Returns the result rotated back.
#[allow(clippy::too_many_arguments)]
fn dsv4_memory_attention(
    model: &TinyModel,
    dsv4: &Dsv4Weights,
    layer_idx: usize,
    ratio: usize,
    keys: &Dsv4Keys,
    q: &[f32],
    start_pos: usize,
    n_tok: usize,
    seq: &Dsv4Memory,
    gpu: &mut Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    hip: Option<&allpaca_hip::HipWorkspace>,
    skip_mla: bool,
) -> Result<Vec<f32>, AllpacaError> {
    let cfg = &model.config;
    let meta = &dsv4.layers[layer_idx];
    let (n_heads, head_dim) = (cfg.n_heads, cfg.head_dim());
    let q_out = n_heads * head_dim;
    let rope = dsv4.spec.attn.rope(ratio);
    let mut out = vec![0.0f32; n_tok * q_out];
    let mut seen = Vec::new();
    for (t, (q, out)) in q.chunks(q_out).zip(out.chunks_mut(q_out)).enumerate() {
        dsv4_token_keys(model, dsv4, layer_idx, ratio, keys, start_pos, t, seq, &mut seen)?;
        if skip_mla {
            dsv4_attention_sinks_maybe_gpu(
                skip_mla,
                gpu.as_deref_mut(),
                hip,
                q,
                &seen,
                &seen,
                &meta.sinks,
                out,
                1,
                seen.len() / head_dim.max(1),
                n_heads,
                cfg.n_kv_heads.max(1),
                head_dim,
            )?;
        } else {
            crate::dsv4_attn::attend(q, &seen, &meta.sinks, n_heads, head_dim, out)?;
        }
        for head in out.chunks_mut(head_dim) {
            rope.rotate(head, start_pos + t, true);
        }
    }
    Ok(out)
}

/// A routed expert on host weights: SwiGLU with the gate capped at
/// `limit` before SiLU and the up projection to plus or minus `limit`
/// (0 = no cap).
fn dsv4_expert_rows(
    x: &[f32],
    ffn: &LayerFfn,
    ff_dim: usize,
    dim: usize,
    m: usize,
    out: &mut [f32],
    limit: f32,
) -> Result<(), AllpacaError> {
    if limit <= 0.0 || ffn.w_gate.is_empty() {
        return expert_mlp_rows(x, ffn, ff_dim, dim, m, out, false);
    }
    let mut up = vec![0.0f32; m * ff_dim];
    let mut gate = vec![0.0f32; m * ff_dim];
    matmul_nt(x, &ffn.w_up, &mut up, m, ff_dim, dim)?;
    matmul_nt(x, &ffn.w_gate, &mut gate, m, ff_dim, dim)?;
    for (g, u) in gate.iter_mut().zip(&up) {
        *g = capped_swiglu(*g, *u, limit);
    }
    matmul_nt(&gate, &ffn.w_down, out, m, dim, ff_dim)
}

/// SwiGLU with DeepSeek-V4's cap (`limit` > 0): the gate is capped at
/// `limit` before SiLU and the up projection clamped to plus or minus it.
fn capped_swiglu(gate: f32, up: f32, limit: f32) -> f32 {
    let g = gate.min(limit);
    g / (1.0 + (-g).exp()) * up.clamp(-limit, limit)
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
    seq: &mut Dsv4Memory,
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
    let ratio = spec.attn.ratio(blk);
    let masked = spec.attn.needs_mask(ratio, pos, 1);
    if crate::dsv4_gpu::want_full_layer_prefetch(prefill, cfg) {
        if let (Some(g), Some(f)) = (gpu.as_mut(), file) {
            let _ = g.prefetch_layer_experts(f, blk, cfg, Some(metrics));
        }
    }
    let attn_out = with_mla(dsv4, layer_idx, blk, skip_mla, |mla| {
        let mut q = vec![0.0f32; q_out];
        let mut have_host_q = false;
        let dq = if !meta.sinks.is_empty() && !masked {
            match (gpu.as_deref_mut(), file) {
                (Some(g), Some(f)) => g
                    .mla_q_norm_rope_device(
                        f,
                        blk,
                        &cur,
                        &mla.q_a_norm,
                        eps,
                        spec.q_lora_rank,
                        q_out,
                        dim,
                        1,
                        n_heads,
                        head_dim,
                        pos,
                        cfg.rope_theta,
                        None,
                        cfg.rope_dims,
                        spec.rope_offset,
                        Some(metrics),
                    )
                    .ok(),
                _ => None,
            }
        } else {
            None
        };
        if dq.is_none() {
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
                dsv4_rope(cfg, spec, ratio, row, pos, false)?;
            }
            have_host_q = true;
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
        dsv4_rope(cfg, spec, ratio, &mut kv_lat, pos, false)?;
        crate::dsv4_attn::fp8_quantize(&mut kv_lat, spec.rope_offset);
        let page_tokens = kv
            .get(page_ids[layer_idx][0])
            .map(|p| p.capacity_tokens)
            .unwrap_or(128);
        let page_index = pos / page_tokens as usize;
        let slot = (pos % page_tokens as usize) as u32;
        let page_id = page_ids[layer_idx][page_index];
        kv.write_slot(page_id, slot, n_kv_heads, head_dim, &kv_lat, &kv_lat)?;
        kv.touch(page_id, 1.0);
        // The masked path reads the keys it needs itself.
        let k_all = if masked {
            Vec::new()
        } else {
            let lo = spec.attn.raw_lo(pos);
            gather_raw_keys(kv, &page_ids[layer_idx], lo, pos + 1, n_kv_heads, head_dim)?
        };
        let n_kv = k_all.len() / kv_dim.max(1);
        let mut attn_raw = Vec::new();
        let t_attn = Instant::now();
        let dattn = match (dq, gpu.as_deref_mut()) {
            (Some(p), Some(g)) => g
                .attention_rope_back_from_device_q(
                    p,
                    &k_all,
                    &k_all,
                    &meta.sinks,
                    1,
                    n_kv,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    pos,
                    cfg.rope_theta,
                    None,
                    cfg.rope_dims,
                    spec.rope_offset,
                )
                .ok(),
            _ => None,
        };
        if masked {
            let keys = dsv4_absorb(
                model,
                dsv4,
                layer_idx,
                ratio,
                &cur,
                &mla.q_a_norm,
                pos,
                1,
                &page_ids[layer_idx],
                kv,
                seq,
                &mut gpu,
                file,
                metrics,
            )?;
            attn_raw = dsv4_memory_attention(
                model, dsv4, layer_idx, ratio, &keys, &q, pos, 1, seq, &mut gpu, hip, skip_mla,
            )?;
        } else if dattn.is_none() {
            attn_raw = vec![0.0f32; q_out];
            let attn_ok = match (dq, gpu.as_deref_mut()) {
                (Some(p), Some(g)) => g
                    .attention_sinks_from_device_q(
                        p,
                        &k_all,
                        &k_all,
                        &meta.sinks,
                        &mut attn_raw,
                        1,
                        n_kv as i32,
                        n_heads as i32,
                        n_kv_heads as i32,
                        head_dim as i32,
                        true,
                    )
                    .is_ok(),
                _ => false,
            };
            if !attn_ok {
                if !have_host_q {
                    if let (Some(p), Some(g)) = (dq, gpu.as_deref_mut()) {
                        g.copy_d2h_f32(p, &mut q)?;
                    } else {
                        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                            "MLA Q is empty after device-path miss".into(),
                        )));
                    }
                }
                dsv4_attention_sinks_maybe_gpu(
                    skip_mla,
                    gpu.as_deref_mut(),
                    hip,
                    &q,
                    &k_all,
                    &k_all,
                    &meta.sinks,
                    &mut attn_raw,
                    1,
                    n_kv,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                )?;
            }
            for h in 0..n_heads {
                let row = &mut attn_raw[h * head_dim..(h + 1) * head_dim];
                rope_offset_back(row, pos, cfg.rope_theta, cfg.rope_dims, spec.rope_offset)?;
            }
        }
        metrics
            .dsv4_attn_ns
            .fetch_add(t_attn.elapsed().as_nanos() as u64, Ordering::Relaxed);
        dsv4_mla_o_rows(
            &mut gpu, file, mla, &attn_raw, dattn, blk, 1, q_out, n_heads, head_dim, spec, dim,
            metrics,
        )
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
    if let (Some(g), Some(f)) = (gpu.as_deref_mut(), file) {
        if let Ok(q) = g.mla_q_rows(
            f,
            blk,
            curs,
            &mla.q_a_norm,
            eps,
            q_lora,
            q_out,
            dim,
            n_tok,
            Some(metrics),
        ) {
            return Ok(q);
        }
    }
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
    dattn: Option<*const f32>,
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
    if let (Some(g), Some(f)) = (gpu.as_deref_mut(), file) {
        if let Some(p) = dattn {
            if let Ok(out) = g.mla_o_rows_from_device(
                f,
                blk,
                p,
                n_tok,
                q_out,
                n_heads,
                head_dim,
                groups,
                per,
                dim,
                Some(metrics),
                true,
            ) {
                return Ok(out);
            }
        }
        if !attn_raws.is_empty() {
            if let Ok(out) = g.mla_o_rows(
                f,
                blk,
                attn_raws,
                n_tok,
                q_out,
                n_heads,
                head_dim,
                groups,
                per,
                dim,
                Some(metrics),
                true,
            ) {
                return Ok(out);
            }
        }
    }
    let owned = if attn_raws.is_empty() {
        if let (Some(p), Some(g)) = (dattn, gpu.as_deref_mut()) {
            let mut buf = vec![0.0f32; n_tok.saturating_mul(q_out)];
            g.copy_d2h_f32(p, &mut buf)?;
            Some(buf)
        } else {
            None
        }
    } else {
        None
    };
    let attn_raws = owned.as_deref().unwrap_or(attn_raws);
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
    states: &mut HcBatch,
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
    seq: &mut Dsv4Memory,
) -> Result<Vec<Dsv4FfnPrep>, AllpacaError> {
    let n_tok = tokens.len();
    if n_tok == 0 {
        return Ok(Vec::new());
    }
    if n_tok != states.n_tok {
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
    let ratio = spec.attn.ratio(blk);
    let masked = spec.attn.needs_mask(ratio, start_pos, n_tok);
    // Several tokens with their own keys each attend on the host; one
    // token's keys go to the device as a single run.
    let host_attn = masked && n_tok > 1;
    let (rope_theta, rope_yarn) = spec.attn.rope(ratio).device();
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
    let t_hc = Instant::now();
    let attn_mix = format!("blk.{blk}.hc_attn_fn.weight");
    let (mut curs, posts, combs) = hc_pre_packed(
        &mut states.data,
        n_tok,
        &meta.hc_attn,
        spec,
        dim,
        eps,
        gpu.as_deref_mut(),
        Some(attn_mix.as_str()),
        Some(attn_norm),
        Some(metrics),
        false,
    )?;
    metrics
        .dsv4_hc_ns
        .fetch_add(t_hc.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let attn_outs = with_mla(dsv4, layer_idx, blk, skip_mla, |mla| {
        let mut q = Vec::new();
        let dq = if !meta.sinks.is_empty() && !host_attn {
            match (gpu.as_deref_mut(), file) {
                (Some(g), Some(f)) => g
                    .mla_q_norm_rope_device(
                        f,
                        blk,
                        &curs,
                        &mla.q_a_norm,
                        eps,
                        spec.q_lora_rank,
                        q_out,
                        dim,
                        n_tok,
                        n_heads,
                        head_dim,
                        start_pos,
                        rope_theta,
                        rope_yarn,
                        cfg.rope_dims,
                        spec.rope_offset,
                        Some(metrics),
                    )
                    .ok(),
                _ => None,
            }
        } else {
            None
        };
        if dq.is_none() {
            if let Some(g) = gpu.as_deref_mut() {
                g.ensure_hc_curs_host(&mut curs)?;
            }
            q = dsv4_mla_q_rows(
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
            rmsnorm_rows_maybe_gpu(
                gpu.as_deref_mut(),
                &mut q,
                &ones,
                n_tok.saturating_mul(n_heads),
                head_dim,
                eps,
                metrics,
            )?;
            for t in 0..n_tok {
                let pos = start_pos + t;
                let qrow = &mut q[t * q_out..(t + 1) * q_out];
                for h in 0..n_heads {
                    let row = &mut qrow[h * head_dim..(h + 1) * head_dim];
                    dsv4_rope(cfg, spec, ratio, row, pos, false)?;
                }
            }
        }
        let mut kv_lat = dsv4_mla_kv_rows(
            &mut gpu, file, mla, &curs, blk, n_tok, head_dim, dim, eps, metrics,
        )?;
        let page_tokens = kv
            .get(page_ids[layer_idx][0])
            .map(|p| p.capacity_tokens)
            .unwrap_or(128);
        for t in 0..n_tok {
            let pos = start_pos + t;
            let kv_row = &mut kv_lat[t * head_dim..(t + 1) * head_dim];
            dsv4_rope(cfg, spec, ratio, kv_row, pos, false)?;
            crate::dsv4_attn::fp8_quantize(kv_row, spec.rope_offset);
            let page_index = pos / page_tokens as usize;
            let slot = (pos % page_tokens as usize) as u32;
            let page_id = page_ids[layer_idx][page_index];
            kv.write_slot(page_id, slot, n_kv_heads, head_dim, kv_row, kv_row)?;
            kv.touch(page_id, 1.0);
        }
        let memory_keys = if masked {
            if let Some(g) = gpu.as_deref_mut() {
                g.ensure_hc_curs_host(&mut curs)?;
            }
            Some(dsv4_absorb(
                model,
                dsv4,
                layer_idx,
                ratio,
                &curs,
                &mla.q_a_norm,
                start_pos,
                n_tok,
                &page_ids[layer_idx],
                kv,
                seq,
                &mut gpu,
                file,
                metrics,
            )?)
        } else {
            None
        };
        let k_all = match &memory_keys {
            Some(_) if host_attn => Vec::new(),
            Some(keys) => {
                let mut seen = Vec::new();
                dsv4_token_keys(model, dsv4, layer_idx, ratio, keys, start_pos, 0, seq, &mut seen)?;
                seen
            }
            None => gather_raw_keys(
                kv,
                &page_ids[layer_idx],
                spec.attn.raw_lo(start_pos),
                start_pos + n_tok,
                n_kv_heads,
                head_dim,
            )?,
        };
        let n_kv = k_all.len() / kv_dim.max(1);
        let mut attn_raws = Vec::new();
        let t_attn = Instant::now();
        let dattn = match (dq, gpu.as_deref_mut()) {
            (Some(p), Some(g)) => g
                .attention_rope_back_from_device_q(
                    p,
                    &k_all,
                    &k_all,
                    &meta.sinks,
                    n_tok,
                    n_kv,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    start_pos,
                    rope_theta,
                    rope_yarn,
                    cfg.rope_dims,
                    spec.rope_offset,
                )
                .ok(),
            _ => None,
        };
        if let (true, Some(keys)) = (host_attn, &memory_keys) {
            attn_raws = dsv4_memory_attention(
                model, dsv4, layer_idx, ratio, keys, &q, start_pos, n_tok, seq, &mut gpu, hip,
                skip_mla,
            )?;
        } else if dattn.is_none() {
            if let Some(g) = gpu.as_deref_mut() {
                g.ensure_hc_curs_host(&mut curs)?;
            }
            attn_raws = vec![0.0f32; n_tok.saturating_mul(q_out)];
            let attn_ok = match (dq, gpu.as_deref_mut()) {
                (Some(p), Some(g)) => g
                    .attention_sinks_from_device_q(
                        p,
                        &k_all,
                        &k_all,
                        &meta.sinks,
                        &mut attn_raws,
                        n_tok as i32,
                        n_kv as i32,
                        n_heads as i32,
                        n_kv_heads as i32,
                        head_dim as i32,
                        true,
                    )
                    .is_ok(),
                _ => false,
            };
            if !attn_ok {
                if q.is_empty() {
                    if let (Some(p), Some(g)) = (dq, gpu.as_deref_mut()) {
                        q = vec![0.0f32; n_tok.saturating_mul(q_out)];
                        g.copy_d2h_f32(p, &mut q)?;
                    } else {
                        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                            "MLA Q is empty after device-path miss".into(),
                        )));
                    }
                }
                dsv4_attention_sinks_maybe_gpu(
                    skip_mla,
                    gpu.as_deref_mut(),
                    hip,
                    &q,
                    &k_all,
                    &k_all,
                    &meta.sinks,
                    &mut attn_raws,
                    n_tok,
                    n_kv,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                )?;
            }
            for t in 0..n_tok {
                let pos = start_pos + t;
                let attn_raw = &mut attn_raws[t * q_out..(t + 1) * q_out];
                for h in 0..n_heads {
                    let row = &mut attn_raw[h * head_dim..(h + 1) * head_dim];
                    dsv4_rope(cfg, spec, ratio, row, pos, true)?;
                }
            }
        }
        metrics
            .dsv4_attn_ns
            .fetch_add(t_attn.elapsed().as_nanos() as u64, Ordering::Relaxed);
        dsv4_mla_o_rows(
            &mut gpu, file, mla, &attn_raws, dattn, blk, n_tok, q_out, n_heads, head_dim, spec,
            dim, metrics,
        )
    })?;
    let ffn_norm = if !meta.ffn_norm.is_empty() {
        &meta.ffn_norm
    } else {
        &host.ffn_norm
    };
    let t_hc = Instant::now();
    let xs_dev = gpu.as_ref().and_then(|g| g.last_o_device());
    let _ = hc_post_batch(
        &attn_outs,
        states,
        spec,
        dim,
        gpu.as_deref_mut(),
        |t| {
            (
                posts.get(t).map(Vec::as_slice).unwrap_or(&[]),
                combs.get(t).map(Vec::as_slice).unwrap_or(&[]),
            )
        },
        xs_dev,
    );
    let (xs, fposts, fcombs) = {
        let ffn_mix = format!("blk.{blk}.hc_ffn_fn.weight");
        hc_pre_packed(
            &mut states.data,
            n_tok,
            &meta.hc_ffn,
            spec,
            dim,
            eps,
            gpu.as_deref_mut(),
            Some(ffn_mix.as_str()),
            Some(ffn_norm),
            Some(metrics),
            false,
        )?
    };
    metrics
        .dsv4_hc_ns
        .fetch_add(t_hc.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let mut fcurs: Vec<Vec<f32>> = Vec::with_capacity(n_tok);
    for t in 0..n_tok {
        fcurs.push(xs[t * dim..(t + 1) * dim].to_vec());
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
    let mut preps = Vec::with_capacity(n_tok);
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
    let limit = dsv4.spec.shexp_swiglu_limit(blk);
    let mut shexp_y = vec![0.0f32; dim];
    if !host.shared_expert.w_up.is_empty() {
        let sff = if cfg.shared_ff_dim > 0 {
            cfg.shared_ff_dim
        } else {
            host.shared_expert.w_up.len() / dim.max(1)
        };
        dsv4_expert_rows(fcur, &host.shared_expert, sff, dim, 1, &mut shexp_y, limit)?;
        return Ok(shexp_y);
    }
    let sff_hint = if cfg.shared_ff_dim > 0 {
        cfg.shared_ff_dim
    } else {
        cfg.ff_dim.max(1)
    };
    let mut fused = false;
    if let (Some(g), Some(f)) = (gpu.as_mut(), file) {
        if let Ok(y) = g.shexp_swiglu(f, blk, fcur, dim, sff_hint, limit, Some(metrics)) {
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
            for (u, g) in up.iter_mut().zip(&gate) {
                *u = if limit > 0.0 {
                    capped_swiglu(*g, *u, limit)
                } else {
                    allpaca_cpu::silu(*g) * *u
                };
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
    let limit = spec.swiglu_limit(blk);
    if let Some(g) = gpu.as_deref() {
        g.set_swiglu_limit(limit);
    }
    let Dsv4FfnPrep {
        fcur,
        fpost,
        fcomb,
        picks,
    } = prep;
    // Hash-routed layers list experts in table order, not by score: no tail.
    let picks = if model.route_tail.enabled && layer_idx >= spec.hash_layer_count {
        crate::route_tail::filter_picks(
            std::slice::from_ref(&picks),
            |e| {
                gpu.as_mut().is_some_and(|g| {
                    g.cpu_miss() && g.resident_expert(blk, e).is_none() && !g.admits(blk, e)
                })
            },
            model.route_tail,
        )
        .into_iter()
        .next()
        .unwrap_or(picks)
    } else {
        picks
    };
    // Experts that stay out of the GPU caches run on the CPU, beside the
    // GPU work.
    let mut cpu_work: Vec<(usize, Vec<f32>, Vec<(usize, f32)>)> = Vec::new();
    let picks: Vec<(usize, f32)> = match (gpu.as_mut(), file) {
        (Some(g), Some(_)) if g.cpu_miss() => picks
            .into_iter()
            .filter(|&(e, w)| {
                g.note_route(blk, e);
                let on_gpu = g.resident_expert(blk, e).is_some() || g.admits(blk, e);
                if !on_gpu {
                    cpu_work.push((e, fcur.clone(), vec![(0, w)]));
                }
                on_gpu
            })
            .collect(),
        _ => picks,
    };
    let ids: Vec<usize> = picks.iter().map(|(e, _)| *e).collect();
    if warmup {
        if let (Some(g), Some(f)) = (gpu.as_mut(), file) {
            if let Ok(plan) = g.warmup_experts(f, blk, &ids, cfg, Some(metrics)) {
                if !plan.is_empty() {
                    let inf = g.submit_copy_plan(&plan, Some(metrics));
                    g.mark_expert_h2d(inf.0, inf.1);
                    if inf.0 || inf.1 {
                        metrics
                            .expert_prefill_overlaps
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    }
    // With `shexp_early` the shared expert waits until the CPU-miss experts
    // have started, so the card runs it beside them; its sum is added last
    // either way.
    let mut shexp_y = Vec::new();
    if !model.shexp_early {
        shexp_y = dsv4_shexp(model, dsv4, &fcur, layer_idx, metrics, &mut gpu, file, skip_shexp)?;
    }
    let fill = gpu
        .as_ref()
        .map(|g| if g.has_expert_slots() { picks.len() } else { 0 })
        .unwrap_or(0);
    let mut routed = vec![0.0f32; dim];
    let cpu_sum = std::thread::scope(|s| -> Result<Option<Vec<f32>>, AllpacaError> {
    let cpu_jh = match file {
        Some(f) if !cpu_work.is_empty() => {
            let work = &cpu_work;
            let ram = gpu.as_ref().and_then(|g| g.expert_ram());
            Some(s.spawn(move || dsv4_cpu_experts(f, blk, cfg, work, 1, limit, metrics, ram.as_deref())))
        }
        _ => None,
    };
    if model.shexp_early {
        shexp_y = dsv4_shexp(model, dsv4, &fcur, layer_idx, metrics, &mut gpu, file, skip_shexp)?;
    }
    let mut deferred: Vec<(usize, usize, f32)> = Vec::new();
    for (i, (e, w)) in picks.into_iter().enumerate() {
        if gpu.as_ref().is_some_and(|g| g.expert_copy_pending(blk, e)) {
            deferred.push((i, e, w));
            continue;
        }
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
            dsv4_expert_rows(&fcur, &ffn, cfg.ff_dim, dim, 1, &mut y, limit)?;
        }
        for j in 0..dim {
            routed[j] += y[j] * w;
        }
    }
    if let Some(g) = gpu.as_mut() {
        g.join_expert_h2d();
    }
    for (i, e, w) in deferred {
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
            dsv4_expert_rows(&fcur, &ffn, cfg.ff_dim, dim, 1, &mut y, limit)?;
        }
        for j in 0..dim {
            routed[j] += y[j] * w;
        }
    }
    match cpu_jh.map(|jh| jh.join()) {
        Some(Ok(sum)) => sum.map(Some),
        Some(Err(_)) => Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "CPU expert thread panicked".into(),
        ))),
        None => Ok(None),
    }
    })?;
    if let Some(sum) = cpu_sum {
        add_acc_slice(&mut routed, &sum);
    }
    if let Some(g) = gpu.as_mut() {
        g.finish_h2d_keep();
    }
    for i in 0..dim {
        routed[i] += shexp_y[i];
    }
    let next_blk = (layer_idx + 1 < cfg.n_layers).then_some(cfg.gguf_blk(layer_idx + 1));
    let prefetch = if crate::dsv4_gpu::want_hint_prefetch(cfg) {
        match (gpu.as_mut(), file, next_blk) {
            (Some(g), Some(f), Some(nl)) => g
                .begin_hint_prefetch(f, nl, &ids, cfg, Some(metrics), blk)
                .ok(),
            _ => None,
        }
    } else {
        None
    };
    let t_hc = Instant::now();
    let mut inflight = (false, false);
    let streams = std::thread::scope(|s| {
        let pref_jh = match (gpu.as_ref(), prefetch.as_ref()) {
            (Some(g), Some(plan)) if !plan.is_empty() => {
                Some(s.spawn(|| g.submit_copy_plan(plan, Some(metrics))))
            }
            _ => None,
        };
        let streams = hc_post(&routed, streams, &fpost, &fcomb, spec, dim);
        if let Some(jh) = pref_jh {
            match jh.join() {
                Ok(pair) => inflight = pair,
                Err(_) => tracing::warn!("L+1 expert prefetch H2D panicked"),
            }
        }
        streams
    });
    if let Some(g) = gpu.as_mut() {
        g.mark_expert_h2d(inflight.0, inflight.1);
        if inflight.0 || inflight.1 {
            metrics
                .expert_prefill_overlaps
                .fetch_add(1, Ordering::Relaxed);
        }
        g.unpin_layer(blk);
    }
    metrics
        .dsv4_hc_ns
        .fetch_add(t_hc.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(streams)
}

/// Routed experts that are not in a GPU cache, computed on the CPU from the
/// mapped MXFP4 file. `work` is `(expert, its m rows of dim, [(token, weight)])`;
/// the result is the weighted sum per token, `n_tok * dim`.
fn dsv4_cpu_experts(
    file: &GgufFile,
    blk: usize,
    cfg: &ModelConfig,
    work: &[(usize, Vec<f32>, Vec<(usize, f32)>)],
    n_tok: usize,
    limit: f32,
    metrics: &Metrics,
    ram: Option<&Mutex<crate::dsv4_ram::ExpertRam>>,
) -> Result<Vec<f32>, AllpacaError> {
    let (dim, ff) = (cfg.dim, cfg.ff_dim.max(1));
    let t = Instant::now();
    // Through the RAM tier (cached, else read from the file in large direct
    // reads) when there is one; else through the mapped file.
    let spans: Option<Vec<(usize, Vec<crate::dsv4_ram::FileSpan>)>> = ram.and_then(|_| {
        work.iter()
            .map(|(e, _, _)| crate::dsv4_gpu::expert_mxfp4_spans(file, blk, *e, cfg).map(|s| (*e, s)))
            .collect()
    });
    // The tier is locked for the fetch only: fetched bytes stay where they
    // are until this thread's next fetch, and the read-ahead worker, which
    // needs the lock, never touches them.
    let fetched: Option<Vec<(usize, usize)>> = match (ram, spans.as_ref()) {
        (Some(r), Some(spans)) => {
            let mut tier = r.lock().unwrap_or_else(|e| e.into_inner());
            let locs = tier.fetch(blk, spans, metrics)?;
            Some(
                locs.iter()
                    .map(|loc| {
                        let b = tier.bytes(*loc);
                        (b.as_ptr() as usize, b.len())
                    })
                    .collect(),
            )
        }
        _ => None,
    };
    let mut parts = Vec::with_capacity(work.len());
    if fetched.is_none() {
        for (e, _, _) in work {
            parts.push(crate::dsv4_gpu::expert_mxfp4_parts(file, blk, *e, cfg)?);
        }
    }
    let experts: Vec<allpaca_cpu::Mxfp4Expert<'_>> = match (fetched.as_ref(), spans.as_ref()) {
        (Some(fetched), Some(spans)) => work
            .iter()
            .zip(fetched.iter().zip(spans))
            .map(|((_, x, group), (&(ptr, len), (_, s)))| {
                // SAFETY: see above; `len` is one packed expert.
                let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
                let (gate, rest) = bytes.split_at(s[0].len);
                let (up, down) = rest.split_at(s[1].len);
                allpaca_cpu::Mxfp4Expert {
                    gate,
                    up,
                    down,
                    x,
                    m: group.len(),
                }
            })
            .collect(),
        _ => work
            .iter()
            .zip(&parts)
            .map(|((_, x, group), p)| allpaca_cpu::Mxfp4Expert {
                gate: p[0].1,
                up: p[1].1,
                down: p[2].1,
                x,
                m: group.len(),
            })
            .collect(),
    };
    // Half the logical cores: the rest drive the GPUs and the copies.
    let threads = std::thread::available_parallelism().map_or(1, |n| (n.get() / 2).max(1));
    let ys = allpaca_cpu::mxfp4_swiglu_experts(&experts, dim, ff, limit, threads);
    let mut sum = vec![0.0f32; n_tok.saturating_mul(dim)];
    for (y, (_, _, group)) in ys.iter().zip(work) {
        scatter_expert_rows(&mut sum, y, group, dim);
    }
    metrics.expert_lru_misses.fetch_add(work.len() as u64, Ordering::Relaxed);
    metrics.expert_cpu_runs.fetch_add(work.len() as u64, Ordering::Relaxed);
    metrics
        .dsv4_cpu_expert_ns
        .fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(sum)
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

fn add_acc_slice(routed: &mut [f32], acc: &[f32]) {
    for (dst, src) in routed.iter_mut().zip(acc.iter()) {
        *dst += *src;
    }
}

fn add_moe_acc(
    hip: Option<&std::sync::Arc<allpaca_hip::HipWorkspace>>,
    routed: &mut [f32],
    n_tok: usize,
    dim: usize,
) {
    let Some(h) = hip else {
        return;
    };
    if !h.moe_active() {
        return;
    }
    let mut acc = vec![0.0f32; n_tok.saturating_mul(dim)];
    if h.moe_finish(&mut acc).is_err() {
        tracing::warn!("device moe accumulate download failed");
        return;
    }
    add_acc_slice(routed, &acc);
}

fn take_wait_job(
    expert: usize,
    jobs0: &mut Vec<crate::dsv4_gpu::ExpertSwigluJob>,
    jobs1: &mut Vec<crate::dsv4_gpu::ExpertSwigluJob>,
    buf0: &mut Vec<crate::dsv4_gpu::ExpertSwigluJob>,
    buf1: &mut Vec<crate::dsv4_gpu::ExpertSwigluJob>,
) -> Option<bool> {
    if let Some(i) = jobs0.iter().position(|j| j.expert == expert) {
        buf0.push(jobs0.swap_remove(i));
        Some(false)
    } else if let Some(i) = jobs1.iter().position(|j| j.expert == expert) {
        buf1.push(jobs1.swap_remove(i));
        Some(true)
    } else {
        None
    }
}

fn wait_miss_ready_bufs(gpu: Option<&crate::dsv4_gpu::Dsv4Gpu>, ev0: &[usize], ev1: &[usize]) {
    let Some(g) = gpu else {
        return;
    };
    for &idx in ev0 {
        if let Err(e) = g.wait_miss_ready(false, idx) {
            tracing::warn!(error = %e, "GPU0 miss ready wait failed");
        }
    }
    for &idx in ev1 {
        if let Err(e) = g.wait_miss_ready(true, idx) {
            tracing::warn!(error = %e, "GPU1 miss ready wait failed");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_expert_job_pair(
    hip0: Option<&std::sync::Arc<allpaca_hip::HipWorkspace>>,
    hip1: Option<&std::sync::Arc<allpaca_hip::HipWorkspace>>,
    jobs0: Vec<crate::dsv4_gpu::ExpertSwigluJob>,
    jobs1: Vec<crate::dsv4_gpu::ExpertSwigluJob>,
    dim: usize,
    ff: usize,
    metrics: &Metrics,
    gpu_out: &mut Vec<(Vec<f32>, Vec<(usize, f32)>)>,
    cpu_work: &mut Vec<(usize, Vec<f32>, Vec<(usize, f32)>)>,
) -> bool {
    crate::dsv4_gpu::dbg_sync(4, hip0.map(|h| &**h), hip1.map(|h| &**h));
    let take_fail = |failed: Vec<crate::dsv4_gpu::ExpertSwigluJob>,
                     cpu_work: &mut Vec<(usize, Vec<f32>, Vec<(usize, f32)>)>| {
        for job in failed {
            cpu_work.push((job.expert, job.x, job.group));
        }
    };
    if !jobs1.is_empty() {
        if let (Some(h0), Some(h1)) = (hip0, hip1) {
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
                        take_fail(fail, cpu_work);
                    }
                    Err(_) => tracing::warn!("GPU0 expert SwiGLU thread panicked"),
                }
                match j1.join() {
                    Ok((rows, fail)) => {
                        gpu_out.extend(rows);
                        take_fail(fail, cpu_work);
                    }
                    Err(_) => tracing::warn!("GPU1 expert SwiGLU thread panicked"),
                }
            });
            return true;
        }
        take_fail(jobs0, cpu_work);
        take_fail(jobs1, cpu_work);
        return false;
    }
    if !jobs0.is_empty() {
        if let Some(h0) = hip0 {
            let (rows, fail) =
                crate::dsv4_gpu::run_expert_swiglu_jobs(h0, jobs0, dim, ff, Some(metrics));
            gpu_out.extend(rows);
            take_fail(fail, cpu_work);
            return true;
        }
        take_fail(jobs0, cpu_work);
    }
    false
}

#[allow(clippy::too_many_arguments)]
/// CPU shexp loads are skipped when the BF16 tensors are already resident.
/// Device accumulate still runs from `moe_x`.
fn should_run_device_shexp(weights_resident: bool, moe_active: bool) -> bool {
    weights_resident && moe_active
}

fn dsv4_apply_moe_rows(
    model: &TinyModel,
    dsv4: &Dsv4Weights,
    states: &mut HcBatch,
    mut preps: Vec<Dsv4FfnPrep>,
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
    if n_tok != states.n_tok {
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
    let limit = spec.swiglu_limit(blk);
    let shexp_limit = spec.shexp_swiglu_limit(blk);
    if let Some(g) = gpu.as_deref() {
        g.set_swiglu_limit(limit);
    }
    let mut xs = vec![0.0f32; n_tok.saturating_mul(dim)];
    for (t, prep) in preps.iter().enumerate() {
        let dst = t.saturating_mul(dim);
        let n = prep.fcur.len().min(dim);
        xs[dst..dst + n].copy_from_slice(&prep.fcur[..n]);
    }
    let sff_hint = if cfg.shared_ff_dim > 0 {
        cfg.shared_ff_dim
    } else {
        cfg.ff_dim.max(1)
    };
    let mut shexp_y = vec![0.0f32; n_tok.saturating_mul(dim)];
    let mut shexp_done = false;
    if !host.shared_expert.w_up.is_empty() {
        let sff = if cfg.shared_ff_dim > 0 {
            cfg.shared_ff_dim
        } else {
            host.shared_expert.w_up.len() / dim.max(1)
        };
        dsv4_expert_rows(
            &xs,
            &host.shared_expert,
            sff,
            dim,
            n_tok,
            &mut shexp_y,
            shexp_limit,
        )?;
        shexp_done = true;
    }
    if model.route_tail.enabled && layer_idx >= dsv4.spec.hash_layer_count {
        let ranked: Vec<Vec<(usize, f32)>> = preps.iter().map(|p| p.picks.clone()).collect();
        let filtered = crate::route_tail::filter_picks(
            &ranked,
            |e| {
                gpu.as_mut().is_some_and(|g| {
                    g.cpu_miss() && g.resident_expert(blk, e).is_none() && !g.admits(blk, e)
                })
            },
            model.route_tail,
        );
        for (prep, picks) in preps.iter_mut().zip(filtered) {
            prep.picks = picks;
        }
    }
    let hint: Vec<usize> = {
        let mut u = Vec::new();
        for prep in &preps {
            for (e, _) in &prep.picks {
                if !u.contains(e) {
                    u.push(*e);
                }
            }
        }
        u
    };
    // Experts that stay out of the GPU caches run on the CPU instead.
    let cpu_miss = file.is_some() && gpu.as_ref().is_some_and(|g| g.cpu_miss());
    let hint: Vec<usize> = match gpu.as_mut() {
        Some(g) if cpu_miss => {
            for prep in &preps {
                for (e, _) in &prep.picks {
                    g.note_route(blk, *e);
                }
            }
            hint.into_iter()
                .filter(|&e| g.resident_expert(blk, e).is_some() || g.admits(blk, e))
                .collect()
        }
        _ => hint,
    };
    let warmup_plan = match (gpu.as_mut(), file) {
        (Some(g), Some(f)) => g.warmup_experts(f, blk, &hint, cfg, Some(metrics)).ok(),
        _ => None,
    };
    // The expert inputs may live on the GPU only (`xs` is then stale): the
    // CPU experts need them on the host.
    if cpu_miss {
        if let Some(g) = gpu.as_ref() {
            let on_cpu = preps
                .iter()
                .flat_map(|p| p.picks.iter())
                .any(|(e, _)| g.resident_expert(blk, *e).is_none());
            if on_cpu {
                if let Some((ptr, n)) = g.hc_curs_device() {
                    if n == xs.len() {
                        g.copy_d2h_f32(ptr, &mut xs)?;
                    }
                }
                // While this layer computes, read the next layer's cold
                // experts. The first layers route by token id, not by
                // their input.
                let next = layer_idx + 1;
                if next < cfg.n_layers
                    && next >= spec.hash_layer_count
                    && n_tok <= crate::dsv4_ram::AHEAD_MAX_ROWS
                {
                    g.read_ahead(cfg.gguf_blk(next), &xs);
                }
            }
        }
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
    let mut jobs0_ready = Vec::new();
    let mut jobs0_wait = Vec::new();
    let mut jobs1_ready = Vec::new();
    let mut jobs1_wait = Vec::new();
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
        let pending = gpu.as_ref().is_some_and(|g| g.expert_copy_pending(blk, e));
        if gpu_slots {
            if let Some(g) = gpu.as_ref() {
                if let Some((ptr, device)) = g.resident_expert(blk, e) {
                    metrics.expert_lru_hits.fetch_add(1, Ordering::Relaxed);
                    job = Some((ptr as usize, device));
                }
            }
        }
        match job {
            Some((ptr, device)) if Some(device) == dev0 => {
                let sw = crate::dsv4_gpu::ExpertSwigluJob {
                    expert: e,
                    ptr,
                    x: gx,
                    m,
                    group,
                };
                if pending {
                    jobs0_wait.push(sw);
                } else {
                    jobs0_ready.push(sw);
                }
            }
            Some((ptr, device)) if Some(device) == dev1 => {
                let sw = crate::dsv4_gpu::ExpertSwigluJob {
                    expert: e,
                    ptr,
                    x: gx,
                    m,
                    group,
                };
                if pending {
                    jobs1_wait.push(sw);
                } else {
                    jobs1_ready.push(sw);
                }
            }
            _ => cpu_work.push((e, gx, group)),
        }
    }
    if gpu.is_some() {
        if let Some(h) = hip0.as_ref() {
            match gpu.as_ref().and_then(|g| g.hc_curs_device()) {
                Some((ptr, n)) if n == xs.len() => {
                    let _ = h.moe_begin_from_device(ptr, n_tok, dim);
                }
                _ => {
                    let _ = h.moe_begin(&xs, n_tok, dim);
                }
            }
        }
    }
    if !jobs1_ready.is_empty() || !jobs1_wait.is_empty() {
        if let Some(h) = hip1.as_ref() {
            let peer = match gpu.as_ref().and_then(|g| g.hc_curs_device()) {
                Some((ptr, n)) if n == xs.len() => {
                    let src_dev = hip0.as_ref().map(|h0| h0.device).unwrap_or(0);
                    h.moe_begin_from_peer(ptr, src_dev, n_tok, dim).is_ok()
                }
                _ => false,
            };
            if !peer {
                let _ = h.moe_begin(&xs, n_tok, dim);
            }
        }
    }
    let cpu_now = if cpu_miss {
        std::mem::take(&mut cpu_work)
    } else {
        Vec::new()
    };
    let mut cpu_sum: Option<Vec<f32>> = None;
    let mut cpu_err: Option<AllpacaError> = None;
    let ff = cfg.ff_dim.max(1);
    let mut gpu_out: Vec<(Vec<f32>, Vec<(usize, f32)>)> = Vec::new();
    let mut inflight = (false, false);
    let mut gpu_compute = false;
    let mut shexp_dev = false;
    let (tx, rx) = std::sync::mpsc::channel();
    let mut bounced = false;
    std::thread::scope(|s| {
        let cpu_jh = match file {
            Some(f) if !cpu_now.is_empty() => {
                let work = &cpu_now;
                let ram = gpu.as_ref().and_then(|g| g.expert_ram());
                Some(s.spawn(move || dsv4_cpu_experts(f, blk, cfg, work, n_tok, limit, metrics, ram.as_deref())))
            }
            _ => None,
        };
        let bounce = match (gpu.as_ref(), warmup_plan.as_ref()) {
            (Some(g), Some(plan)) if !plan.is_empty() => {
                bounced = true;
                let tx = tx.clone();
                Some(s.spawn(move || g.submit_copy_plan_notifying(plan, Some(metrics), Some(tx))))
            }
            _ => None,
        };
        drop(tx);
        if !shexp_done {
            if let Some(g) = gpu.as_ref() {
                if let Some((gate, up, down)) = g.shexp_resident_ptrs(blk, dim, sff_hint) {
                    let moe_live = hip0.as_ref().is_some_and(|h| h.moe_active());
                    if should_run_device_shexp(true, moe_live) {
                        match g.shexp_accumulate_into_moe(
                            n_tok,
                            dim,
                            sff_hint,
                            gate,
                            up,
                            down,
                            shexp_limit,
                            Some(metrics),
                        ) {
                            Ok(()) => {
                                shexp_done = true;
                                shexp_dev = true;
                            }
                            Err(e) => {
                                tracing::debug!(error = %e, "shexp device accumulate failed");
                            }
                        }
                    }
                    if !shexp_done {
                        if let Ok(y) = g.shexp_swiglu_with_ptrs(
                            &xs,
                            n_tok,
                            dim,
                            sff_hint,
                            gate,
                            up,
                            down,
                            shexp_limit,
                            Some(metrics),
                        ) {
                            shexp_y = y;
                            shexp_done = true;
                        }
                    }
                }
            }
        }
        gpu_compute = run_expert_job_pair(
            hip0.as_ref(),
            hip1.as_ref(),
            jobs0_ready,
            jobs1_ready,
            dim,
            ff,
            metrics,
            &mut gpu_out,
            &mut cpu_work,
        );
        let mut buf0 = Vec::new();
        let mut buf1 = Vec::new();
        let mut ev0 = Vec::new();
        let mut ev1 = Vec::new();
        loop {
            let (expert, idx) = match rx.recv() {
                Ok(pair) => pair,
                Err(_) => break,
            };
            match take_wait_job(
                expert,
                &mut jobs0_wait,
                &mut jobs1_wait,
                &mut buf0,
                &mut buf1,
            ) {
                Some(false) => ev0.push(idx),
                Some(true) => ev1.push(idx),
                None => {}
            }
            while let Ok((e, eidx)) = rx.try_recv() {
                match take_wait_job(e, &mut jobs0_wait, &mut jobs1_wait, &mut buf0, &mut buf1) {
                    Some(false) => ev0.push(eidx),
                    Some(true) => ev1.push(eidx),
                    None => {}
                }
            }
            wait_miss_ready_bufs(gpu.as_deref(), &ev0, &ev1);
            ev0.clear();
            ev1.clear();
            gpu_compute |= run_expert_job_pair(
                hip0.as_ref(),
                hip1.as_ref(),
                std::mem::take(&mut buf0),
                std::mem::take(&mut buf1),
                dim,
                ff,
                metrics,
                &mut gpu_out,
                &mut cpu_work,
            );
        }
        if let Some(jh) = bounce {
            match jh.join() {
                Ok(pair) => inflight = pair,
                Err(_) => tracing::warn!("expert miss H2D thread panicked"),
            }
        }
        match cpu_jh.map(|jh| jh.join()) {
            Some(Ok(Ok(sum))) => cpu_sum = Some(sum),
            Some(Ok(Err(e))) => cpu_err = Some(e),
            Some(Err(_)) => {
                cpu_err = Some(AllpacaError::Fatal(FatalError::InvalidConfig(
                    "CPU expert thread panicked".into(),
                )))
            }
            None => {}
        }
    });
    if let Some(e) = cpu_err {
        return Err(e);
    }
    if !shexp_done {
        if let (Some(g), Some(f)) = (gpu.as_mut(), file) {
            if let Ok(y) =
                g.shexp_swiglu_rows(f, blk, &xs, n_tok, dim, sff_hint, shexp_limit, Some(metrics))
            {
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
        g.mark_expert_h2d(inflight.0, inflight.1);
        if bounced || inflight.0 || inflight.1 {
            metrics
                .expert_prefill_overlaps
                .fetch_add(1, Ordering::Relaxed);
        }
        g.join_expert_h2d();
    }
    gpu_compute |= run_expert_job_pair(
        hip0.as_ref(),
        hip1.as_ref(),
        jobs0_wait,
        jobs1_wait,
        dim,
        ff,
        metrics,
        &mut gpu_out,
        &mut cpu_work,
    );
    let next_blk = (layer_idx + 1 < cfg.n_layers).then_some(cfg.gguf_blk(layer_idx + 1));
    let prefetch = if crate::dsv4_gpu::want_hint_prefetch(cfg) {
        match (gpu.as_mut(), file, next_blk) {
            (Some(g), Some(f), Some(nl)) => g
                .begin_hint_prefetch(f, nl, &hint, cfg, Some(metrics), blk)
                .ok(),
            _ => None,
        }
    } else {
        None
    };
    if gpu_compute {
        if let Some(g) = gpu.as_mut() {
            g.used.store(true, Ordering::Relaxed);
        }
        let _ = hip0.as_ref().map(|h| h.bind());
    }
    if let Some(h0) = hip0.as_ref() {
        if h0.moe_active() {
            let _ = h0.moe_join_compute();
        }
    }
    let mut gpu1_host: Option<Vec<f32>> = None;
    if !gpu.as_ref().is_some_and(|g| g.fold_gpu1_moe_into_gpu0()) {
        if let Some(h1) = hip1.as_ref() {
            if h1.moe_active() {
                let mut tmp = vec![0.0f32; n_tok.saturating_mul(dim)];
                if h1.moe_finish(&mut tmp).is_ok() {
                    gpu1_host = Some(tmp);
                }
            }
        }
    }
    let leftover = !gpu_out.is_empty() || !cpu_work.is_empty();
    let mut device_hc = false;
    let t_hc = Instant::now();
    if !leftover {
        if let Some(h0) = hip0.as_ref() {
            if h0.moe_active() {
                let mut folded = true;
                if let Some(tmp) = gpu1_host.as_ref() {
                    if h0.moe_add_from_host(tmp).is_ok() {
                        gpu1_host = None;
                    } else {
                        folded = false;
                    }
                }
                if folded {
                    if let Some(sum) = cpu_sum.as_ref() {
                        if h0.moe_add_from_host(sum).is_ok() {
                            cpu_sum = None;
                        } else {
                            folded = false;
                        }
                    }
                }
                if folded && !shexp_dev {
                    if h0.moe_add_from_host(&shexp_y).is_ok() {
                        shexp_dev = true;
                    } else {
                        folded = false;
                    }
                }
                if folded {
                    if let Ok(acc) = h0.moe_acc_device() {
                        if hc_post_batch(
                            &[],
                            states,
                            spec,
                            dim,
                            gpu.as_deref_mut(),
                            |t| (preps[t].fpost.as_slice(), preps[t].fcomb.as_slice()),
                            Some(acc as *const f32),
                        ) {
                            h0.moe_release();
                            device_hc = true;
                        }
                    }
                }
            }
        }
    }
    if !device_hc {
        add_moe_acc(hip0.as_ref(), &mut routed, n_tok, dim);
        if let Some(tmp) = gpu1_host.as_ref() {
            add_acc_slice(&mut routed, tmp);
        }
        if let Some(sum) = cpu_sum.as_ref() {
            add_acc_slice(&mut routed, sum);
        }
        let _ = hip0.as_ref().map(|h| h.bind());
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
                dsv4_expert_rows(&gx, &ffn, cfg.ff_dim, dim, m, &mut gy, limit)?;
            }
            scatter_expert_rows(&mut routed, &gy, &group, dim);
        }
        if !shexp_dev {
            add_acc_slice(&mut routed, &shexp_y);
        }
        let _ = hc_post_batch(
            &routed,
            states,
            spec,
            dim,
            gpu.as_deref_mut(),
            |t| (preps[t].fpost.as_slice(), preps[t].fcomb.as_slice()),
            None,
        );
    }
    let mut inflight = (false, false);
    if let (Some(g), Some(plan)) = (gpu.as_ref(), prefetch.as_ref()) {
        if !plan.is_empty() {
            inflight = g.submit_copy_plan(plan, Some(metrics));
        }
    }
    if let Some(g) = gpu.as_mut() {
        g.finish_h2d_keep();
        g.unpin_layer(blk);
        g.mark_expert_h2d(inflight.0, inflight.1);
        if inflight.0 || inflight.1 {
            metrics
                .expert_prefill_overlaps
                .fetch_add(1, Ordering::Relaxed);
        }
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
    if std::env::var("ALLPACA_DSV4_NO_CURS_FIX").is_err() {
        if let Some(g) = gpu.as_deref_mut() {
            g.invalidate_hc_curs();
        }
    }
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

fn dsv4_packed_gpu_live(model: &TinyModel) -> bool {
    let Some(dsv4) = model.dsv4.as_ref() else {
        return false;
    };
    let Ok(g) = dsv4.gpu.lock() else {
        return false;
    };
    g.as_ref().is_some()
        && dsv4.gguf.as_ref().is_some_and(|f| {
            f.tensor("blk.0.attn_q_a.weight")
                .is_some_and(|t| gpu_dense_type(t.ggml_type))
        })
}

/// The sequence's attention memory, or a fresh one for a forward that
/// starts a sequence. A model with compressed layers cannot continue a
/// sequence without the memory of what came before.
fn sequence_memory<'a>(
    spec: &Dsv4Spec,
    seq: Option<&'a mut Dsv4Memory>,
    start_pos: usize,
    fresh: &'a mut Dsv4Memory,
) -> Result<&'a mut Dsv4Memory, AllpacaError> {
    match seq {
        Some(seq) => Ok(seq),
        None if start_pos == 0 || !spec.attn.compresses() => Ok(fresh),
        None => Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "DeepSeek-V4 forward at position {start_pos} without the sequence's attention memory"
        )))),
    }
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
    seq: Option<&mut Dsv4Memory>,
) -> Result<(), AllpacaError> {
    if dsv4_packed_gpu_live(model) {
        return forward_dsv4_tokens(
            model,
            &[token],
            pos,
            page_ids,
            kv,
            logits,
            metrics,
            hip,
            true,
            prefill,
            seq,
        );
    }
    let dsv4 = model.dsv4.as_ref().ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad("missing DeepSeek-V4 weights".into()))
    })?;
    let cfg = &model.config;
    let dim = cfg.dim;
    let hc = dsv4.spec.hc_mult.max(1);
    let mut fresh = Dsv4Memory::default();
    let seq = sequence_memory(&dsv4.spec, seq, pos, &mut fresh)?;

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
    if let Some(g) = gpu.as_mut() {
        g.invalidate_hc_device();
    }
    let skip_mla = gpu.as_ref().is_some()
        && file.is_some_and(|f| {
            f.tensor("blk.0.attn_q_a.weight")
                .is_some_and(|t| gpu_dense_type(t.ggml_type))
        });
    let skip_shexp = gpu.as_ref().is_some()
        && file.is_some_and(|f| {
            f.tensor("blk.0.ffn_up_shexp.weight")
                .is_some_and(|t| gpu_dense_type(t.ggml_type))
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
            seq,
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

/// P21 debug: `ALLPACA_DSV4_LAYER_HASH=1` downloads the packed hidden state after
/// every layer of a chunk and prints `layer_hash pos layer n fnv max_abs nonfinite`
/// to stderr, to find the first layer where two runs (or a corrupted step) differ.
/// The download waits for the device, so it changes timing. Off by default.
fn debug_layer_hash(
    gpu: &mut Option<&mut crate::dsv4_gpu::Dsv4Gpu>,
    states: &mut HcBatch,
    start_pos: usize,
    layer_idx: usize,
) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("ALLPACA_DSV4_LAYER_HASH").is_ok_and(|v| v == "1")) {
        return;
    }
    let Some(g) = gpu.as_mut() else { return };
    let mut scratch = vec![0.0f32; states.data.len()];
    if g.download_hc_packed(&mut scratch).is_err() {
        return;
    }
    let mut h: u64 = 0xcbf29ce484222325;
    let mut max_abs = 0.0f32;
    let mut nonfinite = 0usize;
    for v in &scratch {
        for b in v.to_bits().to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
        if v.is_finite() {
            max_abs = max_abs.max(v.abs());
        } else {
            nonfinite += 1;
        }
    }
    eprintln!(
        "layer_hash {} {} {} {:016x} {:.4} {}",
        start_pos,
        layer_idx,
        states.n_tok,
        h,
        max_abs,
        nonfinite
    );
}

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
    prefill: bool,
    seq: Option<&mut Dsv4Memory>,
) -> Result<(), AllpacaError> {
    if tokens.is_empty() {
        return Ok(());
    }
    let dsv4 = model.dsv4.as_ref().ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad("missing DeepSeek-V4 weights".into()))
    })?;
    let mut fresh = Dsv4Memory::default();
    let seq = sequence_memory(&dsv4.spec, seq, start_pos, &mut fresh)?;
    let cfg = &model.config;
    let dim = cfg.dim;
    let hc = dsv4.spec.hc_mult.max(1);
    let file = dsv4.gguf.as_deref();
    let mut gpu_guard = dsv4.gpu.lock().ok();
    let mut gpu = gpu_guard.as_mut().and_then(|g| g.as_mut());
    if let Some(g) = gpu.as_mut() {
        g.invalidate_hc_device();
    }
    let skip_mla = gpu.as_ref().is_some()
        && file.is_some_and(|f| {
            f.tensor("blk.0.attn_q_a.weight")
                .is_some_and(|t| gpu_dense_type(t.ggml_type))
        });
    let skip_shexp = gpu.as_ref().is_some()
        && file.is_some_and(|f| {
            f.tensor("blk.0.ffn_up_shexp.weight")
                .is_some_and(|t| gpu_dense_type(t.ggml_type))
        });
    let mut states = HcBatch::new(tokens.len(), hc, dim);
    for (t, &token) in tokens.iter().enumerate() {
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
        states.fill_token_all_streams(t, &x0);
    }
    for layer_idx in 0..cfg.n_layers {
        tracing::debug!(layer_idx, n_layers = cfg.n_layers, "dsv4 chunk layer");
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
            prefill,
            seq,
        )?;
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
        debug_layer_hash(&mut gpu, &mut states, start_pos, layer_idx);
    }
    if write_logits {
        if states.n_tok > 0 {
            if let Some(g) = gpu.as_mut() {
                let _ = g.download_hc_packed(&mut states.data);
            }
            let last = states.last_streams();
            let used = dsv4_lm_head(
                model,
                dsv4,
                &last,
                logits,
                metrics,
                gpu.as_deref_mut(),
                file,
            )?;
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
    Ok(topk_from_logits(&mut scores, cfg.n_active, cfg, None, &[]))
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
        out.push(topk_from_logits(&mut row, cfg.n_active, cfg, None, &[]));
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

/// What [`write_tiny_deepseek4_ctx_gguf`] varies.
pub struct TinyDsv4Ctx {
    /// Raw keys a token attends to (0 = all).
    pub window: u32,
    /// One layer per entry: 0, 4 or 128.
    pub compress_ratios: Vec<u32>,
    /// Compressed keys the indexer keeps (0 = all).
    pub top_k: u32,
    /// `swiglu_clamp_exp` of every layer (0 = the key is left out).
    pub swiglu_clamp: f32,
    /// `swiglu_clamp_shexp` of every layer (0 = the key is left out).
    pub swiglu_clamp_shexp: f32,
}

/// Tiny DeepSeek-V4 GGUF for attention-memory tests: every weight differs
/// (seeded noise), so tokens and positions are told apart, and layers have
/// the compressors and indexer their ratio calls for.
pub fn write_tiny_deepseek4_ctx_gguf(path: &std::path::Path, opts: &TinyDsv4Ctx) {
    use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
    use std::collections::HashMap;

    let (dim, vocab, n_heads, head_dim, n_rot) = (16usize, 8usize, 2usize, 8usize, 4u32);
    let (q_lora, o_lora, groups, hc) = (8usize, 8usize, 2usize, 2usize);
    let (n_experts, ff) = (4usize, 16usize);
    let (index_heads, index_dim) = (2usize, 8usize);
    let q_out = n_heads * head_dim;
    let oa_out = o_lora * groups;
    let o_group_dim = (n_heads / groups) * head_dim;
    let hc_dim = dim * hc;
    let mix_dim = (2 + hc) * hc;

    let mut meta = HashMap::new();
    let u = |v: usize| GgufValue::Uint32(v as u32);
    for (key, value) in [
        ("vocab_size", u(vocab)),
        ("embedding_length", u(dim)),
        ("block_count", u(opts.compress_ratios.len())),
        ("attention.head_count", u(n_heads)),
        ("attention.head_count_kv", u(1)),
        ("attention.key_length", u(head_dim)),
        ("attention.value_length", u(head_dim)),
        ("rope.dimension_count", GgufValue::Uint32(n_rot)),
        ("rope.scaling.type", GgufValue::String("yarn".into())),
        ("rope.scaling.factor", GgufValue::Float32(4.0)),
        ("rope.scaling.original_context_length", u(64)),
        ("attention.compress_rope_freq_base", GgufValue::Float32(100.0)),
        ("attention.sliding_window", GgufValue::Uint32(opts.window)),
        (
            "attention.compress_ratios",
            GgufValue::Array(opts.compress_ratios.iter().map(|r| GgufValue::Uint32(*r)).collect()),
        ),
        ("attention.indexer.head_count", u(index_heads)),
        ("attention.indexer.key_length", u(index_dim)),
        ("attention.indexer.top_k", GgufValue::Uint32(opts.top_k)),
        ("attention.q_lora_rank", u(q_lora)),
        ("attention.output_lora_rank", u(o_lora)),
        ("attention.output_group_count", u(groups)),
        ("hyper_connection.count", u(hc)),
        ("hyper_connection.sinkhorn_iterations", u(4)),
        ("hyper_connection.epsilon", GgufValue::Float32(1e-6)),
        ("hash_layer_count", u(0)),
        ("expert_count", u(n_experts)),
        ("expert_used_count", u(2)),
        ("expert_shared_count", u(1)),
        ("expert_feed_forward_length", u(ff)),
        ("expert_shared_feed_forward_length", u(ff)),
        ("expert_weights_norm", GgufValue::Bool(true)),
        ("expert_weights_scale", GgufValue::Float32(1.5)),
        ("expert_gating_func", u(4)),
    ] {
        meta.insert(format!("deepseek4.{key}"), value);
    }
    meta.insert(
        "general.architecture".into(),
        GgufValue::String("deepseek4".into()),
    );
    meta.insert("general.name".into(), GgufValue::String("tiny-dsv4-ctx".into()));
    for (key, limit) in [
        ("swiglu_clamp_exp", opts.swiglu_clamp),
        ("swiglu_clamp_shexp", opts.swiglu_clamp_shexp),
    ] {
        if limit > 0.0 {
            meta.insert(
                format!("deepseek4.{key}"),
                GgufValue::Array(
                    opts.compress_ratios
                        .iter()
                        .map(|_| GgufValue::Float32(limit))
                        .collect(),
                ),
            );
        }
    }
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

    let mut seed = 0x9e37_79b9u32;
    let mut noise = |n: usize, amp: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    };
    let mut owned: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    let mut add = |name: String, dims: &[usize], data: Vec<f32>| {
        owned.push((name, dims.iter().map(|d| *d as u64).collect(), data));
    };
    add("token_embd.weight".into(), &[vocab, dim], noise(vocab * dim, 1.0));
    for (blk, &ratio) in opts.compress_ratios.iter().enumerate() {
        let ratio = ratio as usize;
        let mut t = |name: &str, dims: &[usize], data: Vec<f32>| {
            add(format!("blk.{blk}.{name}.weight"), dims, data);
        };
        t("attn_norm", &[dim], vec![1.0; dim]);
        t("attn_q_a", &[q_lora, dim], noise(q_lora * dim, 0.5));
        t("attn_q_a_norm", &[q_lora], vec![1.0; q_lora]);
        t("attn_q_b", &[q_out, q_lora], noise(q_out * q_lora, 0.5));
        t("attn_kv", &[head_dim, dim], noise(head_dim * dim, 0.5));
        t("attn_kv_a_norm", &[head_dim], vec![1.0; head_dim]);
        t("attn_output_a", &[o_group_dim, oa_out], noise(o_group_dim * oa_out, 0.5));
        t("attn_output_b", &[dim, oa_out], noise(dim * oa_out, 0.5));
        t("attn_sinks", &[n_heads], noise(n_heads, 0.5));
        t("ffn_norm", &[dim], vec![1.0; dim]);
        for kind in ["attn", "ffn"] {
            t(&format!("hc_{kind}_fn"), &[hc_dim, mix_dim], noise(mix_dim * hc_dim, 0.1));
            t(&format!("hc_{kind}_base"), &[2 * hc + hc * hc], vec![0.0; 2 * hc + hc * hc]);
            t(&format!("hc_{kind}_scale"), &[3], vec![1.0; 3]);
        }
        t("ffn_gate_inp", &[n_experts, dim], noise(n_experts * dim, 0.5));
        t("ffn_gate_exps", &[n_experts * ff, dim], noise(n_experts * ff * dim, 0.3));
        t("ffn_up_exps", &[n_experts * ff, dim], noise(n_experts * ff * dim, 0.3));
        t("ffn_down_exps", &[n_experts * dim, ff], noise(n_experts * dim * ff, 0.3));
        t("ffn_gate_shexp", &[ff, dim], noise(ff * dim, 0.3));
        t("ffn_up_shexp", &[ff, dim], noise(ff * dim, 0.3));
        t("ffn_down_shexp", &[dim, ff], noise(dim * ff, 0.3));
        if ratio == 0 {
            continue;
        }
        // Ratio-4 compressors project to two keys' width (group overlap).
        let coff = if ratio == 4 { 2 } else { 1 };
        let mut compressor = |prefix: &str, head: usize| {
            let width = coff * head;
            t(&format!("{prefix}_kv"), &[width, dim], noise(width * dim, 0.5));
            t(&format!("{prefix}_gate"), &[width, dim], noise(width * dim, 0.5));
            t(&format!("{prefix}_ape"), &[ratio, width], noise(ratio * width, 0.5));
            t(&format!("{prefix}_norm"), &[head], vec![1.0; head]);
        };
        compressor("attn_compressor", head_dim);
        if ratio == 4 {
            compressor("indexer_compressor", index_dim);
            t(
                "indexer.attn_q_b",
                &[index_heads * index_dim, q_lora],
                noise(index_heads * index_dim * q_lora, 0.5),
            );
            t("indexer.proj", &[index_heads, dim], noise(index_heads * dim, 0.5));
        }
    }
    add("output_hc_fn.weight".into(), &[hc_dim, hc], noise(hc * hc_dim, 0.1));
    add("output_hc_base.weight".into(), &[hc], vec![0.0; hc]);
    add("output_hc_scale.weight".into(), &[1], vec![1.0]);
    add("output_norm.weight".into(), &[dim], vec![1.0; dim]);
    add("output.weight".into(), &[vocab, dim], noise(vocab * dim, 0.5));
    let views: Vec<GgufWriteTensor<'_>> = owned
        .iter()
        .map(|(name, dims, data)| GgufWriteTensor {
            name: name.clone(),
            dims: dims.clone(),
            data,
        })
        .collect();
    write_gguf(path, &meta, &views).expect("write tiny deepseek4 ctx");
}

/// Tiny DeepSeek-V4 GGUF with BF16 MLA/shexp/lm_head and MXFP4 experts (`k % 32 == 0`).
/// HIP GEMV is used only for those dtypes; F32 fixtures stay on the CPU path.
pub fn write_tiny_deepseek4_packed_gguf(path: &std::path::Path) {
    write_tiny_deepseek4_packed(path, false);
}

/// The packed tiny fixture with its dense tensors as Q8_0 where a row is a
/// whole number of 32-value blocks (as in Unsloth's UD-Q4_K_XL), BF16
/// elsewhere.
pub fn write_tiny_deepseek4_q8_gguf(path: &std::path::Path) {
    write_tiny_deepseek4_packed(path, true);
}

fn write_tiny_deepseek4_packed(path: &std::path::Path, q8_dense: bool) {
    use allpaca_gguf::{
        quantize_bf16, quantize_q8_0, write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor,
        GgufWriteTensor,
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
            if q8_dense && dims.len() == 2 && dims[0] % 32 == 0 {
                packed_store.push((name.clone(), dims.clone(), GgmlType::Q8_0, quantize_q8_0(data)));
                continue;
            }
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

    /// Real weights (`ALLPACA_DSV4_GGUF` = first shard): the packed CPU expert
    /// kernel against dequantise-then-F32 on experts taken from the file.
    #[test]
    fn cpu_packed_experts_match_dequantised_on_real_weights() {
        let Ok(path) = std::env::var("ALLPACA_DSV4_GGUF") else {
            return;
        };
        let file = GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let cfg = ModelConfig::from_gguf(&file).expect("config");
        let (dim, ff) = (cfg.dim, cfg.ff_dim);
        let metrics = Metrics::default();
        let slot = 3 * (dim * ff / 32 * 17);
        let ram = Mutex::new(
            crate::dsv4_ram::ExpertRam::new(file.shard_paths().expect("paths"), slot, slot / 3, 3 * slot as u64)
                .expect("ram tier")
                .expect("slots"),
        );
        for (blk, expert, m) in [(0usize, 3usize, 1usize), (5, 17, 1), (20, 255, 2), (cfg.n_layers - 1, 128, 1)] {
            let x: Vec<f32> = (0..m * dim)
                .map(|i| (((i * 2654435761usize) >> 7) % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            let ffn = load_mxfp4_expert(&file, blk, expert, cfg.n_experts, dim, ff).expect("dequant");
            let mut want = vec![0.0f32; m * dim];
            expert_mlp_rows(&x, &ffn, ff, dim, m, &mut want, false).expect("reference");
            let group: Vec<(usize, f32)> = (0..m).map(|t| (t, 1.0)).collect();
            let work = [(expert, x, group)];
            let got = dsv4_cpu_experts(&file, blk, &cfg, &work, m, 0.0, &metrics, None).expect("packed");
            // The same expert through the RAM tier: read from the file, then cached.
            for pass in 0..2 {
                let tiered = dsv4_cpu_experts(&file, blk, &cfg, &work, m, 0.0, &metrics, Some(&ram)).expect("tiered");
                assert_eq!(tiered, got, "blk {blk} expert {expert} pass {pass}");
            }
            let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
            let worst = got.iter().zip(&want).fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
            eprintln!("blk {blk} expert {expert} m {m}: |y| max {scale:.4} worst diff {worst:.3e}");
            assert!(worst <= 1e-3 * scale, "blk {blk} expert {expert}: worst {worst} of {scale}");
        }
    }

    #[test]
    fn device_shexp_runs_when_cpu_shexp_loader_is_skipped() {
        let skip_cpu_shexp = true;
        assert!(
            skip_cpu_shexp && should_run_device_shexp(true, true),
            "resident BF16 shexp still accumulates from device moe_x"
        );
        assert!(!should_run_device_shexp(false, true));
        assert!(!should_run_device_shexp(true, false));
    }

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
    fn hc_pre_rows_matches_per_token() {
        let spec = Dsv4Spec {
            q_lora_rank: 8,
            o_lora_rank: 8,
            o_group_count: 2,
            hc_mult: 2,
            hc_sinkhorn_iters: 4,
            hc_eps: 1e-6,
            hash_layer_count: 0,
            rope_offset: 0,
            attn: Default::default(),
            swiglu_clamp: Vec::new(),
            swiglu_clamp_shexp: Vec::new(),
        };
        let dim = 8;
        let hc = spec.hc_mult;
        let hc_dim = dim * hc;
        let mix_dim = (2 + hc) * hc;
        let mix = HcMix {
            fn_w: (0..mix_dim * hc_dim)
                .map(|i| 0.01 * (i as f32 % 7.0))
                .collect(),
            base: vec![0.0; 2 * hc + hc * hc],
            scale: vec![1.0, 1.0, 1.0],
        };
        let states: Vec<Vec<Vec<f32>>> = (0..3)
            .map(|t| {
                (0..hc)
                    .map(|s| (0..dim).map(|i| 0.1 * (t + s + i) as f32).collect())
                    .collect()
            })
            .collect();
        let (curs, posts, combs) = hc_pre_rows(&states, &mix, &spec, dim, 1e-5).expect("rows");
        for (t, streams) in states.iter().enumerate() {
            let (cur, post, comb) = hc_pre(streams, &mix, &spec, dim, 1e-5).expect("one");
            let row = &curs[t * dim..(t + 1) * dim];
            for i in 0..dim {
                assert!(
                    (row[i] - cur[i]).abs() < 1e-5,
                    "token {t} dim {i} rows={} seq={}",
                    row[i],
                    cur[i]
                );
            }
            assert_eq!(post.len(), posts[t].len());
            for (a, b) in posts[t].iter().zip(post.iter()) {
                assert!((a - b).abs() < 1e-5);
            }
            for (a, b) in combs[t].iter().zip(comb.iter()) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    fn hc_post_ref(
        x: &[f32],
        residual: &[Vec<f32>],
        post: &[f32],
        comb: &[f32],
        hc: usize,
        dim: usize,
    ) -> Vec<Vec<f32>> {
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

    #[test]
    fn hc_post_rows_matches_per_token() {
        for hc in [2usize, 4] {
            let spec = Dsv4Spec {
                q_lora_rank: 8,
                o_lora_rank: 8,
                o_group_count: 2,
                hc_mult: hc,
                hc_sinkhorn_iters: 4,
                hc_eps: 1e-6,
                hash_layer_count: 0,
                rope_offset: 0,
                attn: Default::default(),
                swiglu_clamp: Vec::new(),
                swiglu_clamp_shexp: Vec::new(),
            };
            let dim = 8;
            let n_tok = 3;
            let xs: Vec<f32> = (0..n_tok * dim).map(|i| 0.01 * i as f32).collect();
            let posts: Vec<Vec<f32>> = (0..n_tok)
                .map(|t| (0..hc).map(|i| 0.2 + 0.1 * (t + i) as f32).collect())
                .collect();
            let combs: Vec<Vec<f32>> = (0..n_tok)
                .map(|t| (0..hc * hc).map(|i| 0.05 * ((t + i) % 5) as f32).collect())
                .collect();
            let mut states: Vec<Vec<Vec<f32>>> = (0..n_tok)
                .map(|t| {
                    (0..hc)
                        .map(|s| (0..dim).map(|i| 0.1 * (t + s + i) as f32).collect())
                        .collect()
                })
                .collect();
            let expect: Vec<Vec<Vec<f32>>> = states
                .iter()
                .enumerate()
                .map(|(t, residual)| {
                    hc_post_ref(
                        &xs[t * dim..(t + 1) * dim],
                        residual,
                        &posts[t],
                        &combs[t],
                        hc,
                        dim,
                    )
                })
                .collect();
            hc_post_rows(&xs, &mut states, &posts, &combs, &spec, dim);
            for t in 0..n_tok {
                assert_eq!(states[t].len(), expect[t].len());
                for s in 0..hc {
                    for i in 0..dim {
                        let a = states[t][s][i];
                        let b = expect[t][s][i];
                        assert!(
                            (a - b).abs() < 1e-5,
                            "hc={hc} token {t} stream {s} dim {i} rows={a} seq={b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn hc_post_batch_matches_rows() {
        for hc in [2usize, 4] {
            let spec = Dsv4Spec {
                q_lora_rank: 8,
                o_lora_rank: 8,
                o_group_count: 2,
                hc_mult: hc,
                hc_sinkhorn_iters: 4,
                hc_eps: 1e-6,
                hash_layer_count: 0,
                rope_offset: 0,
                attn: Default::default(),
                swiglu_clamp: Vec::new(),
                swiglu_clamp_shexp: Vec::new(),
            };
            let dim = 8;
            let n_tok = 3;
            let xs: Vec<f32> = (0..n_tok * dim).map(|i| 0.01 * i as f32).collect();
            let posts: Vec<Vec<f32>> = (0..n_tok)
                .map(|t| (0..hc).map(|i| 0.2 + 0.1 * (t + i) as f32).collect())
                .collect();
            let combs: Vec<Vec<f32>> = (0..n_tok)
                .map(|t| (0..hc * hc).map(|i| 0.05 * ((t + i) % 5) as f32).collect())
                .collect();
            let nested: Vec<Vec<Vec<f32>>> = (0..n_tok)
                .map(|t| {
                    (0..hc)
                        .map(|s| (0..dim).map(|i| 0.1 * (t + s + i) as f32).collect())
                        .collect()
                })
                .collect();
            let mut rows = nested.clone();
            hc_post_rows(&xs, &mut rows, &posts, &combs, &spec, dim);
            let mut batch = HcBatch::new(n_tok, hc, dim);
            batch.data = pack_hc_states(&nested, hc, dim);
            let _ = hc_post_batch(
                &xs,
                &mut batch,
                &spec,
                dim,
                None,
                |t| (posts[t].as_slice(), combs[t].as_slice()),
                None,
            );
            let got = unpack_hc_states(&batch.data, n_tok, hc, dim);
            for t in 0..n_tok {
                for s in 0..hc {
                    for i in 0..dim {
                        let a = got[t][s][i];
                        let b = rows[t][s][i];
                        assert!(
                            (a - b).abs() < 1e-5,
                            "hc={hc} token {t} stream {s} dim {i} batch={a} rows={b}"
                        );
                    }
                }
            }
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
    fn attention_sinks_batched_causal_matches_growing_kv() {
        let n_q = 3usize;
        let prefix = 2usize;
        let n_kv = prefix + n_q;
        let n_heads = 4usize;
        let n_kv_heads = 2usize;
        let head_dim = 8usize;
        let qn = n_q * n_heads * head_dim;
        let kn = n_kv * n_kv_heads * head_dim;
        let q: Vec<f32> = (0..qn).map(|i| 0.02 * i as f32 - 0.3).collect();
        let k: Vec<f32> = (0..kn).map(|i| 0.01 * i as f32 - 0.2).collect();
        let v: Vec<f32> = (0..kn).map(|i| 0.015 * i as f32 - 0.1).collect();
        let sinks: Vec<f32> = (0..n_heads).map(|h| 0.1 * h as f32 - 0.4).collect();
        let mut batched = vec![0.0f32; qn];
        super::attention_sinks(
            &q,
            &k,
            &v,
            &sinks,
            &mut batched,
            n_q,
            n_kv,
            n_heads,
            n_kv_heads,
            head_dim,
        )
        .expect("batched sinks");
        let mut seq = vec![0.0f32; qn];
        let kv_stride = n_kv_heads * head_dim;
        let q_stride = n_heads * head_dim;
        for qi in 0..n_q {
            let t_end = prefix + qi + 1;
            let q_off = qi * q_stride;
            super::attention_sinks(
                &q[q_off..q_off + q_stride],
                &k[..t_end * kv_stride],
                &v[..t_end * kv_stride],
                &sinks,
                &mut seq[q_off..q_off + q_stride],
                1,
                t_end,
                n_heads,
                n_kv_heads,
                head_dim,
            )
            .expect("sequential sinks");
        }
        for i in 0..qn {
            let a = batched[i];
            let b = seq[i];
            assert!((a - b).abs() < 1e-5, "idx {i} batched={a} sequential={b}");
        }
    }

    #[test]
    fn tiny_deepseek4_swiglu_clamp_caps_routed_experts() {
        // First-token logits of the same weights under different clamps.
        let logits = |swiglu_clamp: f32| {
            let path = std::env::temp_dir().join(format!(
                "allpaca_tiny_dsv4_clamp-{}-{}.gguf",
                std::process::id(),
                swiglu_clamp.to_bits()
            ));
            write_tiny_deepseek4_ctx_gguf(
                &path,
                &TinyDsv4Ctx {
                    window: 0,
                    compress_ratios: vec![0],
                    top_k: 0,
                    swiglu_clamp,
                    swiglu_clamp_shexp: 0.0,
                },
            );
            let file = std::sync::Arc::new(GgufFile::open(&path).expect("open tiny dsv4"));
            let mut model = TinyModel::from_gguf(&file).expect("model");
            model.bind_gguf(file);
            let _ = std::fs::remove_file(&path);
            let cfg = &model.config;
            let metrics = crate::telemetry::Metrics::default();
            let mut kv = KvTierManager::new(8 << 20, 8 << 20, 0.5);
            let pages = vec![vec![kv
                .alloc_page(0, 0, 32, cfg.n_kv_heads.max(1), cfg.head_dim(), &metrics)
                .expect("kv page")]];
            let mut out = vec![0.0f32; cfg.vocab_size];
            super::forward_dsv4_token(&model, 3, 0, &pages, &mut kv, &mut out, &metrics, None, None, true, None)
                .expect("forward");
            out
        };
        let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        let free = logits(0.0);
        // A limit no activation reaches changes nothing; a tight one does.
        assert!(diff(&free, &logits(1.0e6)) < 1e-6);
        assert!(diff(&free, &logits(1.0e-3)) > 1e-4);
    }

    #[test]
    fn tiny_deepseek4_shared_expert_follows_swiglu_clamp_shexp() {
        // DeepSeek-V4 inference/model.py: `shared_experts = Expert(...,
        // swiglu_limit=args.swiglu_limit)`, and Expert.forward caps the gate at
        // the limit before SiLU and clamps up to +-limit. Reference values come
        // straight from that formula on the fixture's shared-expert weights.
        let limit = 0.05f32;
        let path = std::env::temp_dir().join(format!(
            "allpaca_tiny_dsv4_shexp_clamp-{}.gguf",
            std::process::id()
        ));
        write_tiny_deepseek4_ctx_gguf(
            &path,
            &TinyDsv4Ctx {
                window: 0,
                compress_ratios: vec![0],
                top_k: 0,
                swiglu_clamp: 0.0,
                swiglu_clamp_shexp: limit,
            },
        );
        let file = std::sync::Arc::new(GgufFile::open(&path).expect("open tiny dsv4"));
        let mut model = TinyModel::from_gguf(&file).expect("model");
        model.bind_gguf(file.clone());
        let _ = std::fs::remove_file(&path);
        let w = |name: &str| {
            file.tensor_f32(&format!("blk.0.{name}.weight"))
                .expect("shared expert tensor")
        };
        let (gate_w, up_w, down_w) = (w("ffn_gate_shexp"), w("ffn_up_shexp"), w("ffn_down_shexp"));
        let dim = model.config.dim;
        let ff = gate_w.len() / dim;
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin() * 2.0).collect();
        let dot = |w: &[f32], row: usize, v: &[f32]| -> f32 {
            w[row * v.len()..(row + 1) * v.len()]
                .iter()
                .zip(v)
                .map(|(a, b)| a * b)
                .sum()
        };
        let gates: Vec<f32> = (0..ff).map(|r| dot(&gate_w, r, &x)).collect();
        assert!(
            gates.iter().any(|g| g.abs() > limit),
            "fixture never reaches the cap"
        );
        let act: Vec<f32> = (0..ff)
            .map(|r| {
                let g = gates[r].min(limit);
                let u = dot(&up_w, r, &x).clamp(-limit, limit);
                g / (1.0 + (-g).exp()) * u
            })
            .collect();
        let want: Vec<f32> = (0..dim).map(|r| dot(&down_w, r, &act)).collect();
        let dsv4 = model.dsv4.as_ref().expect("dsv4 weights");
        let metrics = crate::telemetry::Metrics::default();
        let got = super::dsv4_shexp(&model, dsv4, &x, 0, &metrics, &mut None, Some(&file), false)
            .expect("shared expert");
        let diff = want
            .iter()
            .zip(&got)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff < 1e-6, "shared expert differs from the capped SwiGLU by {diff}");
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
                None,
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
            true,
            None,
        )
        .expect("chunk tokens");
        for (i, (a, b)) in logits_seq.iter().zip(logits_chunk.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "logit {i} seq={a} chunk={b}");
        }
        let _ = std::fs::remove_file(&path);
    }
}
