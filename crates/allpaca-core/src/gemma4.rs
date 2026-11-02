//! `gemma4` (Gemma 4 E2B/E4B, Nemotron-3.5-Bolt-E4B): follows llama.cpp's
//! `src/models/gemma4.cpp`.
//!
//! - The token embedding is scaled by `sqrt(dim)`. Each block is the Gemma
//!   sandwich: `h = x + norm(attn(norm(x)))`, `x = h + norm(ffn(norm(h)))`
//!   with a tanh-GELU gated FFN, then the per-layer embedding (PLE) term,
//!   then the whole block output times the block's `layer_output_scale`.
//! - Attention has Q/K RMS norms, a weightless RMS norm on V, NeoX rope and
//!   a score scale of 1 (no `1/sqrt(head_dim)`).
//! - Sliding-window layers (head dim 256, window 512, their own rope base)
//!   and global layers (head dim 512, proportional rope via `rope_freqs`)
//!   alternate by `attention.sliding_window_pattern`.
//! - The last `attention.shared_kv_layers` blocks write no K/V: they attend
//!   over the last earlier block of the same kind (llama.cpp's KV reuse).
//! - Tied output head with a final logit softcap.
//!
//! The model keeps its K/V in its own per-sequence memory ([`Gemma4Memory`])
//! instead of the engine's paged tiers: the two head dims and the shared
//! layers do not fit that layout.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use allpaca_cpu::matmul_nt;
use allpaca_gguf::{dequant_packed, GgufFile, GgufValue};
use allpaca_types::{AllpacaError, FatalError};

use crate::model::TinyModel;

pub(crate) fn load_err(msg: String) -> AllpacaError {
    AllpacaError::Fatal(FatalError::ModelLoad(msg))
}

pub fn file_is_gemma4(file: &GgufFile) -> bool {
    file.architecture() == Some("gemma4")
}

/// Shapes and constants of a gemma4 GGUF.
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4Spec {
    pub n_layers: usize,
    pub dim: usize,
    pub vocab: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// FFN width per layer.
    pub ff: Vec<usize>,
    pub head_dim: usize,
    pub head_dim_swa: usize,
    pub rope_theta: f32,
    pub rope_theta_swa: f32,
    pub window: usize,
    pub swa: Vec<bool>,
    /// Blocks `0..kv_layers` write K/V; the rest reuse an earlier block's.
    pub kv_layers: usize,
    pub ple_dim: usize,
    pub softcap: f32,
    pub eps: f32,
}

fn meta_u32_or_first(file: &GgufFile, key: &str) -> Option<usize> {
    file.meta_u32(key).map(|v| v as usize).or_else(|| {
        file.metadata
            .get(key)
            .and_then(GgufValue::as_u32_array)
            .and_then(|a| a.first().copied())
            .map(|v| v as usize)
    })
}

impl Gemma4Spec {
    pub fn from_gguf(file: &GgufFile) -> Result<Self, AllpacaError> {
        let a = "gemma4";
        let u = |k: &str| meta_u32_or_first(file, &format!("{a}.{k}"));
        let need = |k: &str| u(k).ok_or_else(|| load_err(format!("gemma4: missing {a}.{k}")));
        let n_layers = need("block_count")?;
        let dim = need("embedding_length")?;
        let vocab = file
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(GgufValue::as_string_array)
            .map(|t| t.len())
            .or_else(|| u("vocab_size"))
            .ok_or_else(|| load_err("gemma4: no vocabulary".into()))?;
        let n_heads = need("attention.head_count")?;
        let n_kv_heads = u("attention.head_count_kv").unwrap_or(n_heads);
        let head_dim = need("attention.key_length")?;
        let head_dim_swa = u("attention.key_length_swa").unwrap_or(head_dim);
        let swa: Vec<bool> = match file.metadata.get(&format!("{a}.attention.sliding_window_pattern")) {
            Some(GgufValue::Array(items)) => items.iter().map(|v| v.as_bool().unwrap_or(false)).collect(),
            _ => vec![false; n_layers],
        };
        if swa.len() != n_layers {
            return Err(load_err(format!("gemma4: sliding_window_pattern has {} entries for {n_layers} blocks", swa.len())));
        }
        let shared = u("attention.shared_kv_layers").unwrap_or(0);
        let kv_layers = n_layers.saturating_sub(shared);
        if shared > 0 && kv_layers < 2 {
            return Err(load_err("gemma4: shared K/V needs two blocks that write K/V".into()));
        }
        let ff = (0..n_layers)
            .map(|l| {
                file.tensor(&format!("blk.{l}.ffn_gate.weight"))
                    .and_then(|t| t.dims.get(1).copied())
                    .map(|v| v as usize)
                    .ok_or_else(|| load_err(format!("gemma4: blk.{l}.ffn_gate.weight missing")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Rope turns whole heads (the global blocks' unrotated pairs come
        // from rope_freqs, not from a shorter rope width).
        for (key, hd) in [("rope.dimension_count", head_dim), ("rope.dimension_count_swa", head_dim_swa)] {
            if let Some(n) = u(key).filter(|&n| n != hd) {
                return Err(load_err(format!("gemma4: {a}.{key} = {n}, but the head dim is {hd}")));
            }
        }
        let f = |k: &str| file.meta_f32(&format!("{a}.{k}"));
        let rope_theta = f("rope.freq_base").unwrap_or(1_000_000.0);
        Ok(Self {
            n_layers,
            dim,
            vocab,
            n_heads,
            n_kv_heads,
            ff,
            head_dim,
            head_dim_swa,
            rope_theta,
            rope_theta_swa: f("rope.freq_base_swa").unwrap_or(rope_theta),
            window: u("attention.sliding_window").unwrap_or(512),
            swa,
            kv_layers,
            ple_dim: u("embedding_length_per_layer_input").unwrap_or(0),
            softcap: f("final_logit_softcapping").unwrap_or(0.0),
            eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
        })
    }

    pub fn head_dim_of(&self, layer: usize) -> usize {
        if self.swa[layer] {
            self.head_dim_swa
        } else {
            self.head_dim
        }
    }

    pub fn theta_of(&self, layer: usize) -> f32 {
        if self.swa[layer] {
            self.rope_theta_swa
        } else {
            self.rope_theta
        }
    }

    /// Floats of one token's K (or V) at `layer`.
    pub fn kv_row(&self, layer: usize) -> usize {
        self.n_kv_heads * self.head_dim_of(layer)
    }

    /// Block whose K/V `layer` attends over (llama.cpp's reuse callback):
    /// itself, or for a shared block the last K/V block of its kind.
    pub fn kv_source(&self, layer: usize) -> usize {
        if layer < self.kv_layers {
            return layer;
        }
        (0..self.kv_layers)
            .rev()
            .find(|&l| self.swa[l] == self.swa[layer])
            .unwrap_or(self.kv_layers - 1)
    }

    /// First key position a query at `pos` sees in `layer`.
    pub fn first_visible(&self, layer: usize, pos: usize) -> usize {
        if self.swa[layer] {
            (pos + 1).saturating_sub(self.window)
        } else {
            0
        }
    }

    /// Float bytes of K plus V per token over every block that writes them.
    pub fn kv_bytes_per_token(&self) -> usize {
        (0..self.kv_layers).map(|l| 2 * 4 * self.kv_row(l)).sum()
    }
}

/// Weights read from the file: small tensors decoded to F32 once, the
/// embedding tables one row at a time.
pub struct Gemma4Weights {
    pub spec: Gemma4Spec,
    file: Option<Arc<GgufFile>>,
    dense: Mutex<HashMap<String, Arc<Vec<f32>>>>,
    /// The device chain, once attached.
    pub(crate) gpu: Mutex<Option<crate::gemma4_gpu::Gemma4Gpu>>,
}

impl std::fmt::Debug for Gemma4Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemma4Weights").field("spec", &self.spec).finish_non_exhaustive()
    }
}

impl Gemma4Weights {
    pub fn from_gguf(file: &GgufFile) -> Result<Self, AllpacaError> {
        Ok(Self {
            spec: Gemma4Spec::from_gguf(file)?,
            file: None,
            dense: Mutex::new(HashMap::new()),
            gpu: Mutex::new(None),
        })
    }

    pub fn attach_gguf(&mut self, file: Arc<GgufFile>) {
        self.file = Some(file);
    }

    pub(crate) fn file(&self) -> Result<&GgufFile, AllpacaError> {
        self.file.as_deref().ok_or_else(|| load_err("gemma4 weights have no GGUF handle".into()))
    }

    pub(crate) fn file_arc(&self) -> Option<Arc<GgufFile>> {
        self.file.clone()
    }

    fn has(&self, name: &str) -> bool {
        self.file.as_ref().is_some_and(|f| f.tensor(name).is_some())
    }

    /// A whole tensor as F32, decoded once.
    fn w(&self, name: &str) -> Result<Arc<Vec<f32>>, AllpacaError> {
        let mut cache = self.dense.lock().map_err(|_| load_err("gemma4 weight cache poisoned".into()))?;
        if let Some(w) = cache.get(name) {
            return Ok(w.clone());
        }
        let w = Arc::new(self.file()?.tensor_f32(name)?);
        cache.insert(name.to_string(), w.clone());
        Ok(w)
    }

    /// Row `row` of `row_len` weights, decoded from the packed tensor.
    pub(crate) fn row(&self, name: &str, row: usize, row_len: usize) -> Result<Vec<f32>, AllpacaError> {
        let file = self.file()?;
        let info = file.tensor(name).ok_or_else(|| load_err(format!("gemma4: missing tensor {name}")))?;
        let bytes_of = |weights: usize| {
            allpaca_gguf::TensorInfo {
                dims: vec![weights as u64],
                ..info.clone()
            }
            .nbytes()
        };
        let (skip, take) = (bytes_of(row * row_len), bytes_of(row_len));
        if (row + 1) * row_len > info.n_elements() as usize || skip + take != bytes_of((row + 1) * row_len) {
            return Err(load_err(format!("gemma4: {name} row {row} of {row_len} is not whole blocks")));
        }
        let bytes = &file.tensor_bytes(name)?[skip..skip + take];
        let mut out = vec![0.0f32; row_len];
        dequant_packed(info.ggml_type, bytes, row_len, &mut out)?;
        Ok(out)
    }

    fn head_name(&self) -> &'static str {
        if self.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        }
    }

    /// The scaled token embedding of `token`.
    pub(crate) fn embed(&self, token: u32) -> Result<Vec<f32>, AllpacaError> {
        let dim = self.spec.dim;
        let mut x = self.row("token_embd.weight", token as usize, dim)?;
        let s = (dim as f32).sqrt();
        x.iter_mut().for_each(|v| *v *= s);
        Ok(x)
    }

    /// The token's per-layer embedding rows times `sqrt(ple_dim)`
    /// (`[n_layers][ple_dim]`).
    pub(crate) fn ple_token(&self, token: u32) -> Result<Vec<f32>, AllpacaError> {
        let sp = &self.spec;
        let mut t = self.row("per_layer_token_embd.weight", token as usize, sp.n_layers * sp.ple_dim)?;
        let s = (sp.ple_dim as f32).sqrt();
        t.iter_mut().for_each(|v| *v *= s);
        Ok(t)
    }
}

/// Debug (`ALLPACA_G4_DUMP=1`): the L2 norm and first values of a row.
pub(crate) fn dump_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ALLPACA_G4_DUMP").as_deref() == Ok("1"))
}

pub(crate) fn dump(tag: &str, layer: usize, row: &[f32]) {
    if dump_on() {
        let n = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!("G4 {tag} l{layer} norm={n:.4} head={:?}", &row[..4.min(row.len())]);
    }
}

pub fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

fn rmsnorm_into(x: &mut [f32], w: Option<&[f32]>, eps: f32) {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv = (ss / x.len() as f32 + eps).sqrt().recip();
    match w {
        Some(w) => x.iter_mut().zip(w).for_each(|(v, g)| *v = *v * inv * g),
        None => x.iter_mut().for_each(|v| *v *= inv),
    }
}

fn gemv(x: &[f32], w: &[f32], n: usize) -> Result<Vec<f32>, AllpacaError> {
    let mut out = vec![0.0f32; n];
    matmul_nt(x, w, &mut out, 1, n, x.len())?;
    Ok(out)
}

/// NeoX rope over a head: pairs `(i, i + hd/2)`, angle
/// `pos * theta^(-2i/hd) / ff[i]`.
pub fn rope_neox_ff(head: &mut [f32], pos: usize, theta: f32, ff: Option<&[f32]>) {
    let half = head.len() / 2;
    for i in 0..half {
        let mut freq = theta.powf(-(i as f32) / half as f32);
        if let Some(ff) = ff {
            freq /= ff[i];
        }
        let angle = pos as f32 * freq;
        let (sn, cs) = angle.sin_cos();
        let (a, b) = (head[i], head[i + half]);
        head[i] = a * cs - b * sn;
        head[i + half] = a * sn + b * cs;
    }
}

/// One query row over keys `lo..=pos` (rows of `n_kv * hd`), score scale 1.
#[allow(clippy::too_many_arguments)]
fn attend(q: &[f32], keys: &[f32], values: &[f32], lo: usize, pos: usize, n_heads: usize, n_kv: usize, hd: usize) -> Vec<f32> {
    let row = n_kv * hd;
    let mut out = vec![0.0f32; n_heads * hd];
    let mut scores = vec![0.0f32; pos + 1 - lo];
    for h in 0..n_heads {
        let kh = h / (n_heads / n_kv);
        let qh = &q[h * hd..(h + 1) * hd];
        for (j, s) in scores.iter_mut().enumerate() {
            let k = &keys[(lo + j) * row + kh * hd..(lo + j) * row + (kh + 1) * hd];
            *s = qh.iter().zip(k).map(|(a, b)| a * b).sum();
        }
        let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut l = 0.0f32;
        scores.iter_mut().for_each(|s| {
            *s = (*s - m).exp();
            l += *s;
        });
        let oh = &mut out[h * hd..(h + 1) * hd];
        for (j, &p) in scores.iter().enumerate() {
            let v = &values[(lo + j) * row + kh * hd..(lo + j) * row + (kh + 1) * hd];
            oh.iter_mut().zip(v).for_each(|(o, v)| *o += p / l * v);
        }
    }
    out
}

/// One sequence's state: the tokens read and, per block that writes them,
/// its keys and values (one row per token).
#[derive(Debug, Clone, Default)]
pub struct Gemma4Memory {
    tokens: usize,
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// The same on the card when the forward runs there.
    pub(crate) device: Option<crate::gemma4_gpu::Gemma4DeviceKv>,
}

impl Gemma4Memory {
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Forget every token from `n` on (a rejected draft span).
    pub fn truncate(&mut self, n: usize, spec: &Gemma4Spec) {
        if n >= self.tokens {
            return;
        }
        for (l, (k, v)) in self.kv.iter_mut().enumerate() {
            let row = spec.kv_row(l);
            k.truncate(n * row);
            v.truncate(n * row);
        }
        self.tokens = n;
    }

    pub(crate) fn set_tokens(&mut self, n: usize) {
        self.tokens = n;
    }

    #[allow(dead_code)] // Host KV size; no caller reads it yet.
    pub(crate) fn host_bytes(&self) -> usize {
        self.kv.iter().map(|(k, v)| (k.len() + v.len()) * 4).sum()
    }
}

/// The per-layer inputs of one token: `(proj(x0) normed + ple_token) / sqrt(2)`,
/// `[n_layers][ple_dim]`.
fn ple_inputs(w: &Gemma4Weights, token: u32, x0: &[f32]) -> Result<Vec<f32>, AllpacaError> {
    let sp = &w.spec;
    let (pd, nl) = (sp.ple_dim, sp.n_layers);
    let mut proj = gemv(x0, &w.w("per_layer_model_proj.weight")?, nl * pd)?;
    let s = (sp.dim as f32).sqrt().recip();
    proj.iter_mut().for_each(|v| *v *= s);
    let norm = w.w("per_layer_proj_norm.weight")?;
    for chunk in proj.chunks_mut(pd) {
        rmsnorm_into(chunk, Some(&norm), sp.eps);
    }
    let tok = w.ple_token(token)?;
    let r2 = std::f32::consts::FRAC_1_SQRT_2;
    Ok(proj.iter().zip(&tok).map(|(p, t)| (p + t) * r2).collect())
}

/// Host forward of one token at `pos`; `logits` gets its (softcapped) logits.
pub(crate) fn forward_host(
    model: &TinyModel,
    w: &Gemma4Weights,
    token: u32,
    pos: usize,
    logits: &mut [f32],
    mem: &mut Gemma4Memory,
) -> Result<(), AllpacaError> {
    let sp = &w.spec;
    let (dim, eps) = (sp.dim, sp.eps);
    if mem.tokens != pos {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "gemma4 sequence memory holds {} tokens, forward is at position {pos}",
            mem.tokens
        ))));
    }
    mem.kv.resize_with(sp.kv_layers, Default::default);
    let mut x = w.embed(token)?;
    let ple = if sp.ple_dim > 0 { ple_inputs(w, token, &x)? } else { Vec::new() };
    let rope_ff = if w.has("rope_freqs.weight") { Some(w.w("rope_freqs.weight")?) } else { None };
    for l in 0..sp.n_layers {
        if let Some(k) = model.layer_capture.slot(l) {
            model.layer_capture.record(k, &[(pos, token)], &x);
        }
        let name = |s: &str| format!("blk.{l}.{s}");
        let hd = sp.head_dim_of(l);
        let theta = sp.theta_of(l);
        let ff = if sp.swa[l] { None } else { rope_ff.as_deref().map(|v| v.as_slice()) };
        let mut xn = x.clone();
        rmsnorm_into(&mut xn, Some(&w.w(&name("attn_norm.weight"))?), eps);
        let mut q = gemv(&xn, &w.w(&name("attn_q.weight"))?, sp.n_heads * hd)?;
        let qn = w.w(&name("attn_q_norm.weight"))?;
        for head in q.chunks_mut(hd) {
            rmsnorm_into(head, Some(&qn), eps);
            rope_neox_ff(head, pos, theta, ff);
        }
        if l < sp.kv_layers {
            let mut k = gemv(&xn, &w.w(&name("attn_k.weight"))?, sp.kv_row(l))?;
            let mut v = if w.has(&name("attn_v.weight")) {
                gemv(&xn, &w.w(&name("attn_v.weight"))?, sp.kv_row(l))?
            } else {
                k.clone()
            };
            let kn = w.w(&name("attn_k_norm.weight"))?;
            for head in k.chunks_mut(hd) {
                rmsnorm_into(head, Some(&kn), eps);
                rope_neox_ff(head, pos, theta, ff);
            }
            for head in v.chunks_mut(hd) {
                rmsnorm_into(head, None, eps);
            }
            mem.kv[l].0.extend_from_slice(&k);
            mem.kv[l].1.extend_from_slice(&v);
        }
        let src = sp.kv_source(l);
        let (keys, values) = &mem.kv[src];
        let attn = attend(&q, keys, values, sp.first_visible(l, pos), pos, sp.n_heads, sp.n_kv_heads, hd);
        dump("q", l, &q);
        dump("attn", l, &attn);
        let mut o = gemv(&attn, &w.w(&name("attn_output.weight"))?, dim)?;
        rmsnorm_into(&mut o, Some(&w.w(&name("post_attention_norm.weight"))?), eps);
        x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
        let mut f = x.clone();
        rmsnorm_into(&mut f, Some(&w.w(&name("ffn_norm.weight"))?), eps);
        let mut g = gemv(&f, &w.w(&name("ffn_gate.weight"))?, sp.ff[l])?;
        let u = gemv(&f, &w.w(&name("ffn_up.weight"))?, sp.ff[l])?;
        g.iter_mut().zip(&u).for_each(|(g, u)| *g = gelu_tanh(*g) * u);
        let mut d = gemv(&g, &w.w(&name("ffn_down.weight"))?, dim)?;
        rmsnorm_into(&mut d, Some(&w.w(&name("post_ffw_norm.weight"))?), eps);
        x.iter_mut().zip(&d).for_each(|(a, b)| *a += b);
        dump("ffn", l, &x);
        if sp.ple_dim > 0 {
            let pd = sp.ple_dim;
            let mut pg = gemv(&x, &w.w(&name("inp_gate.weight"))?, pd)?;
            pg.iter_mut().zip(&ple[l * pd..(l + 1) * pd]).for_each(|(g, p)| *g = gelu_tanh(*g) * p);
            let mut pp = gemv(&pg, &w.w(&name("proj.weight"))?, dim)?;
            rmsnorm_into(&mut pp, Some(&w.w(&name("post_norm.weight"))?), eps);
            x.iter_mut().zip(&pp).for_each(|(a, b)| *a += b);
        }
        if w.has(&name("layer_output_scale.weight")) {
            let s = w.w(&name("layer_output_scale.weight"))?[0];
            x.iter_mut().for_each(|v| *v *= s);
        }
    }
    mem.tokens = pos + 1;
    rmsnorm_into(&mut x, Some(&w.w("output_norm.weight")?), eps);
    matmul_nt(&x, &w.w(w.head_name())?, &mut logits[..sp.vocab], 1, sp.vocab, dim)?;
    softcap(&mut logits[..sp.vocab], sp.softcap);
    Ok(())
}

pub(crate) fn softcap(logits: &mut [f32], cap: f32) {
    if cap > 0.0 {
        logits.iter_mut().for_each(|v| *v = cap * (*v / cap).tanh());
    }
}

/// Forward `tokens` from `start_pos` (on the card when the device chain is
/// attached, else on the host one token at a time). `logits` ends with the
/// last token's; `row_logits`, when given, gets every token's.
pub(crate) fn forward_tokens(
    model: &TinyModel,
    tokens: &[u32],
    start_pos: usize,
    logits: &mut [f32],
    mut row_logits: Option<&mut [f32]>,
    mem: &mut Gemma4Memory,
) -> Result<(), AllpacaError> {
    let w = model.g4.as_ref().ok_or_else(|| load_err("missing gemma4 weights".into()))?;
    let vocab = w.spec.vocab;
    if row_logits.as_ref().is_some_and(|r| r.len() < tokens.len() * vocab) || logits.len() < vocab {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "gemma4 logits buffer is smaller than the rows asked for".into(),
        )));
    }
    let gpu = w.gpu.lock().map_err(|_| load_err("gemma4 gpu lock poisoned".into()))?;
    if let Some(gpu) = gpu.as_ref() {
        return gpu.forward(model, w, tokens, start_pos, logits, row_logits, mem);
    }
    drop(gpu);
    for (i, &t) in tokens.iter().enumerate() {
        forward_host(model, w, t, start_pos + i, logits, mem)?;
        if let Some(rows) = row_logits.as_deref_mut() {
            rows[i * vocab..(i + 1) * vocab].copy_from_slice(&logits[..vocab]);
        }
    }
    Ok(())
}

/// Several sequences' rows (`segs` in order over `tokens`), each run as its
/// own pass, so every row equals its solo run. `row_logits` gets every row's.
pub(crate) fn forward_batch(
    model: &TinyModel,
    tokens: &[u32],
    segs: &mut [crate::model::SeqRows<'_>],
    row_logits: &mut [f32],
) -> Result<(), AllpacaError> {
    let vocab = model.config.vocab_size;
    let mut last = vec![0.0f32; vocab];
    let mut at = 0usize;
    for seg in segs.iter_mut() {
        let mem = &mut seg
            .recurrent
            .as_deref_mut()
            .ok_or_else(|| load_err("gemma4 needs the sequence's state".into()))?
            .g4;
        let n = seg.n_rows;
        if at + n > tokens.len() || row_logits.len() < (at + n) * vocab {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "gemma4 batch: rows, tokens and logits do not line up".into(),
            )));
        }
        forward_tokens(model, &tokens[at..at + n], seg.start_pos, &mut last, Some(&mut row_logits[at * vocab..(at + n) * vocab]), mem)?;
        at += n;
    }
    Ok(())
}

/// Put the device chain for `model` on `hip`'s card.
pub(crate) fn attach_gpu(model: &TinyModel, hip: Arc<allpaca_hip::HipWorkspace>) -> Result<(), AllpacaError> {
    let w = model.g4.as_ref().ok_or_else(|| load_err("missing gemma4 weights".into()))?;
    let file = w.file_arc().ok_or_else(|| load_err("gemma4 weights have no GGUF handle".into()))?;
    let gpu = crate::gemma4_gpu::Gemma4Gpu::load(&file, &w.spec, hip)?;
    *w.gpu.lock().map_err(|_| load_err("gemma4 gpu lock poisoned".into()))? = Some(gpu);
    Ok(())
}

/// Write a tiny gemma4 GGUF for tests: `n_layers` blocks, the last
/// `shared` sharing K/V, SWA on all but every third block, PLE, softcap.
#[doc(hidden)]
pub fn write_tiny_gemma4_gguf(path: &std::path::Path, shared: usize) {
    use allpaca_gguf::{write_gguf_quant, GgmlType, GgufWriteQuantTensor, GgufWriteTensor};
    let (dim, ff, vocab, n_layers, n_heads, n_kv, hd, hd_swa, pd) = (32usize, 64usize, 40usize, 6usize, 2usize, 1usize, 16usize, 8usize, 4usize);
    let swa: Vec<bool> = (0..n_layers).map(|l| l % 3 != 2).collect();
    let mut meta = HashMap::new();
    let u = |v: usize| GgufValue::Uint32(v as u32);
    for (k, v) in [
        ("block_count", u(n_layers)),
        ("context_length", u(4096)),
        ("embedding_length", u(dim)),
        ("feed_forward_length", u(ff)),
        ("attention.head_count", u(n_heads)),
        ("attention.head_count_kv", u(n_kv)),
        ("attention.key_length", u(hd)),
        ("attention.value_length", u(hd)),
        ("attention.key_length_swa", u(hd_swa)),
        ("attention.value_length_swa", u(hd_swa)),
        ("attention.sliding_window", u(5)),
        ("attention.shared_kv_layers", u(shared)),
        ("attention.layer_norm_rms_epsilon", GgufValue::Float32(1e-6)),
        ("rope.freq_base", GgufValue::Float32(1_000_000.0)),
        ("rope.freq_base_swa", GgufValue::Float32(10_000.0)),
        ("final_logit_softcapping", GgufValue::Float32(30.0)),
        ("embedding_length_per_layer_input", u(pd)),
        ("attention.sliding_window_pattern", GgufValue::Array(swa.iter().map(|&b| GgufValue::Uint32(u32::from(b))).collect())),
    ] {
        meta.insert(format!("gemma4.{k}"), v);
    }
    meta.insert("general.architecture".into(), GgufValue::String("gemma4".into()));
    meta.insert("general.name".into(), GgufValue::String("tiny-gemma4".into()));
    meta.insert("tokenizer.ggml.model".into(), GgufValue::String("wordlevel".into()));
    meta.insert(
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array((0..vocab).map(|i| GgufValue::String(format!("t{i}"))).collect()),
    );
    let mut seed = 0x9e37_79b9_u64;
    let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0 * scale
            })
            .collect()
    };
    let mut f32s: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    let mut quant: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)> = Vec::new();
    let norm = |rnd: &mut dyn FnMut(usize, f32) -> Vec<f32>, n: usize| rnd(n, 0.3).into_iter().map(|v| 1.0 + v).collect::<Vec<f32>>();
    f32s.push(("token_embd.weight".into(), vec![dim as u64, vocab as u64], rnd(dim * vocab, 0.5)));
    f32s.push(("per_layer_token_embd.weight".into(), vec![(n_layers * pd) as u64, vocab as u64], rnd(n_layers * pd * vocab, 0.5)));
    f32s.push(("per_layer_model_proj.weight".into(), vec![dim as u64, (n_layers * pd) as u64], rnd(dim * n_layers * pd, 0.3)));
    f32s.push(("per_layer_proj_norm.weight".into(), vec![pd as u64], norm(&mut rnd, pd)));
    f32s.push(("output_norm.weight".into(), vec![dim as u64], norm(&mut rnd, dim)));
    let mut ffs: Vec<f32> = vec![1.0; hd / 4];
    ffs.extend(vec![1e30; hd / 2 - hd / 4]);
    f32s.push(("rope_freqs.weight".into(), vec![(hd / 2) as u64], ffs));
    for l in 0..n_layers {
        let h = if swa[l] { hd_swa } else { hd };
        let t = |s: &str| format!("blk.{l}.{s}");
        let d = |a: usize, b: usize| vec![a as u64, b as u64];
        f32s.push((t("attn_norm.weight"), vec![dim as u64], norm(&mut rnd, dim)));
        f32s.push((t("attn_q.weight"), d(dim, n_heads * h), rnd(dim * n_heads * h, 0.4)));
        f32s.push((t("attn_k.weight"), d(dim, n_kv * h), rnd(dim * n_kv * h, 0.4)));
        f32s.push((t("attn_v.weight"), d(dim, n_kv * h), rnd(dim * n_kv * h, 0.4)));
        f32s.push((t("attn_q_norm.weight"), vec![h as u64], norm(&mut rnd, h)));
        f32s.push((t("attn_k_norm.weight"), vec![h as u64], norm(&mut rnd, h)));
        f32s.push((t("attn_output.weight"), d(n_heads * h, dim), rnd(n_heads * h * dim, 0.4)));
        f32s.push((t("post_attention_norm.weight"), vec![dim as u64], norm(&mut rnd, dim)));
        f32s.push((t("ffn_norm.weight"), vec![dim as u64], norm(&mut rnd, dim)));
        // A mix of encodings, as in the real file.
        let gate = rnd(dim * ff, 0.4);
        quant.push((t("ffn_gate.weight"), d(dim, ff), GgmlType::Q8_0, allpaca_gguf::quantize_q8_0(&gate)));
        f32s.push((t("ffn_up.weight"), d(dim, ff), rnd(dim * ff, 0.4)));
        let down = rnd(ff * dim, 0.3);
        quant.push((t("ffn_down.weight"), d(ff, dim), GgmlType::BF16, allpaca_gguf::quantize_bf16(&down)));
        f32s.push((t("post_ffw_norm.weight"), vec![dim as u64], norm(&mut rnd, dim)));
        f32s.push((t("inp_gate.weight"), d(dim, pd), rnd(dim * pd, 0.4)));
        f32s.push((t("proj.weight"), d(pd, dim), rnd(pd * dim, 0.4)));
        f32s.push((t("post_norm.weight"), vec![dim as u64], norm(&mut rnd, dim)));
        f32s.push((t("layer_output_scale.weight"), vec![1], vec![0.5 + 0.1 * l as f32]));
    }
    let ft: Vec<GgufWriteTensor<'_>> = f32s
        .iter()
        .map(|(name, dims, data)| GgufWriteTensor {
            name: name.clone(),
            dims: dims.clone(),
            data,
        })
        .collect();
    let qt: Vec<GgufWriteQuantTensor<'_>> = quant
        .iter()
        .map(|(name, dims, ty, bytes)| GgufWriteQuantTensor {
            name: name.clone(),
            dims: dims.clone(),
            ggml_type: *ty,
            bytes,
        })
        .collect();
    write_gguf_quant(path, &meta, &ft, &qt).expect("write tiny gemma4");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(shared: usize, tag: &str) -> TinyModel {
        let path = std::env::temp_dir().join(format!("allpaca_g4_{tag}_{}.gguf", std::process::id()));
        write_tiny_gemma4_gguf(&path, shared);
        let file = Arc::new(GgufFile::open(&path).unwrap());
        let mut model = TinyModel::from_gguf(&file).unwrap();
        model.bind_gguf(file);
        model
    }

    fn greedy(model: &TinyModel, prompt: &[u32], n: usize) -> (Vec<u32>, Vec<f32>) {
        let mut mem = Gemma4Memory::default();
        let vocab = model.config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        forward_tokens(model, prompt, 0, &mut logits, None, &mut mem).unwrap();
        let mut out = Vec::new();
        for _ in 0..n {
            let t = crate::sampling::argmax(&logits);
            out.push(t);
            forward_tokens(model, &[t], mem.tokens(), &mut logits, None, &mut mem).unwrap();
        }
        (out, logits)
    }

    #[test]
    fn spec_reads_shared_kv_sources_and_dims() {
        let model = tiny(2, "spec");
        let sp = &model.g4.as_ref().unwrap().spec;
        assert_eq!(sp.kv_layers, 4);
        // Blocks 4 (SWA) and 5 (global) reuse the last K/V block of their kind.
        assert_eq!((sp.kv_source(4), sp.kv_source(5)), (3, 2));
        assert_eq!((sp.head_dim_of(0), sp.head_dim_of(2)), (8, 16));
        assert_eq!(sp.first_visible(0, 10), 6);
        assert_eq!(sp.first_visible(2, 10), 0);
        assert!(model.config.recurrent, "the sequence state carries the K/V");
    }

    #[test]
    fn shared_blocks_ignore_their_own_kv_weights() {
        let model = tiny(2, "shared_a");
        let (a, la) = greedy(&model, &[1, 5, 9, 3, 7, 2, 11, 4], 6);
        // Scramble the K/V projections of the shared blocks: nothing reads them.
        let path = std::env::temp_dir().join(format!("allpaca_g4_shared_b_{}.gguf", std::process::id()));
        write_tiny_gemma4_gguf(&path, 2);
        let file = Arc::new(GgufFile::open(&path).unwrap());
        let mut model_b = TinyModel::from_gguf(&file).unwrap();
        model_b.bind_gguf(file);
        {
            let w = model_b.g4.as_ref().unwrap();
            let mut cache = w.dense.lock().unwrap();
            for l in 4..6 {
                for t in ["attn_k.weight", "attn_v.weight", "attn_k_norm.weight"] {
                    let name = format!("blk.{l}.{t}");
                    let n = w.file().unwrap().tensor(&name).unwrap().n_elements() as usize;
                    cache.insert(name, Arc::new(vec![7.0; n]));
                }
            }
        }
        let (b, lb) = greedy(&model_b, &[1, 5, 9, 3, 7, 2, 11, 4], 6);
        assert_eq!(a, b);
        assert_eq!(la, lb);
        // Without sharing those weights matter.
        let unshared = tiny(0, "shared_c");
        assert_ne!(greedy(&unshared, &[1, 5, 9, 3, 7, 2, 11, 4], 6).1, la);
    }

    #[test]
    fn logits_are_softcapped_and_memory_truncates() {
        let model = tiny(2, "cap");
        let sp = model.g4.as_ref().unwrap().spec.clone();
        let mut mem = Gemma4Memory::default();
        let mut logits = vec![0.0f32; sp.vocab];
        forward_tokens(&model, &[1, 2, 3, 4], 0, &mut logits, None, &mut mem).unwrap();
        assert!(logits.iter().all(|v| v.abs() <= 30.0));
        let after4 = logits.clone();
        forward_tokens(&model, &[9, 9], 4, &mut logits, None, &mut mem).unwrap();
        mem.truncate(4, &sp);
        assert_eq!(mem.tokens(), 4);
        forward_tokens(&model, &[5], 4, &mut logits, None, &mut mem).unwrap();
        let mut fresh = Gemma4Memory::default();
        let mut l2 = vec![0.0f32; sp.vocab];
        forward_tokens(&model, &[1, 2, 3, 4, 5], 0, &mut l2, None, &mut fresh).unwrap();
        assert_eq!(logits, l2, "a truncated memory resumes as if the rejected rows never ran");
        assert_ne!(after4, l2);
    }

    #[test]
    fn sliding_window_hides_old_tokens_from_swa_blocks() {
        // Only the SWA blocks see a token 6 back through attention; with the
        // window at 5 a change there reaches the output only through the
        // global block. Check the reference attention's range directly.
        let sp = tiny(0, "swa").g4.as_ref().unwrap().spec.clone();
        assert_eq!(sp.window, 5);
        let q = vec![1.0f32; 8];
        let keys: Vec<f32> = (0..8 * 8).map(|i| (i / 8) as f32 * 0.1).collect();
        let mut values = vec![0.0f32; 8 * 8];
        values[..8].iter_mut().for_each(|v| *v = 100.0);
        let lo = sp.first_visible(0, 7);
        let out = attend(&q, &keys, &values, lo, 7, 1, 1, 8);
        assert!(out.iter().all(|v| *v == 0.0), "position 0 is outside the window of 7");
    }

    fn on_gpu(model: &TinyModel) -> bool {
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            eprintln!("gemma4 GPU test skipped: no HIP device");
            return false;
        };
        attach_gpu(model, Arc::new(hip)).map_err(|e| eprintln!("gemma4 attach failed: {e}")).is_ok()
    }

    fn prompt(n: usize) -> Vec<u32> {
        (0..n).map(|i| ((i * 7 + 3) % 37) as u32 + 1).collect()
    }

    #[test]
    fn gpu_forward_matches_the_host_forward() {
        let host = tiny(2, "gpu_host");
        let gpu = tiny(2, "gpu_dev");
        if !on_gpu(&gpu) {
            return;
        }
        let toks = prompt(300);
        let vocab = host.config.vocab_size;
        let (mut mh, mut md) = (Gemma4Memory::default(), Gemma4Memory::default());
        let (mut rh, mut rd) = (vec![0.0f32; toks.len() * vocab], vec![0.0f32; toks.len() * vocab]);
        let mut last = vec![0.0f32; vocab];
        forward_tokens(&host, &toks, 0, &mut last, Some(&mut rh), &mut mh).unwrap();
        // Prefill in two passes (one past MAX_ROWS), then decode rows.
        forward_tokens(&gpu, &toks[..280], 0, &mut last, Some(&mut rd[..280 * vocab]), &mut md).unwrap();
        for (i, &t) in toks.iter().enumerate().skip(280) {
            forward_tokens(&gpu, &[t], i, &mut last, Some(&mut rd[i * vocab..(i + 1) * vocab]), &mut md).unwrap();
        }
        let worst = rh.iter().zip(&rd).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 2e-2, "device logits differ from the host by {worst}");
        for r in [0, 5, 130, 279, 299] {
            let row = |v: &[f32]| crate::sampling::argmax(&v[r * vocab..(r + 1) * vocab]);
            assert_eq!(row(&rh), row(&rd), "row {r}");
        }
    }

    #[test]
    fn gpu_verify_rows_equal_decode_rows_bit_for_bit() {
        let model = tiny(2, "gpu_exact");
        if !on_gpu(&model) {
            return;
        }
        let vocab = model.config.vocab_size;
        let toks = prompt(300);
        let mut last = vec![0.0f32; vocab];
        // Positions 250..258 and 380..388 straddle the 128-key chunks and
        // move the SWA window start.
        // The window's chunk-aligned start moves inside both spans (window 5:
        // it is 128 up to position 259 and 256 from 260; 256 up to 387, 384 from 388).
        for prefix in [255usize, 385] {
            let toks: Vec<u32> = toks.iter().cycle().take(prefix + 8).copied().collect();
            let mut a = Gemma4Memory::default();
            forward_tokens(&model, &toks[..prefix], 0, &mut last, None, &mut a).unwrap();
            let mut b = a.clone();
            let mut one = vec![0.0f32; 8 * vocab];
            for i in 0..8 {
                forward_tokens(&model, &[toks[prefix + i]], prefix + i, &mut last, Some(&mut one[i * vocab..(i + 1) * vocab]), &mut a).unwrap();
            }
            let mut packed = vec![0.0f32; 8 * vocab];
            forward_tokens(&model, &toks[prefix..], prefix, &mut last, Some(&mut packed), &mut b).unwrap();
            assert_eq!(one, packed, "an 8-row pass at {prefix} must equal 8 decode steps");
            // A rejected tail: keep 3 rows, then go on from there.
            b.truncate(prefix + 3, &model.g4.as_ref().unwrap().spec);
            let mut again = vec![0.0f32; vocab];
            forward_tokens(&model, &[toks[prefix + 3]], prefix + 3, &mut last, Some(&mut again), &mut b).unwrap();
            assert_eq!(&again[..], &one[3 * vocab..4 * vocab]);
        }

        let prefix = 255;
        let toks: Vec<u32> = toks.iter().cycle().take(prefix + 17).copied().collect();
        let mut a = Gemma4Memory::default();
        forward_tokens(&model, &toks[..prefix], 0, &mut last, None, &mut a).unwrap();
        let mut b = a.clone();
        let mut one = vec![0.0f32; 17 * vocab];
        for i in 0..17 {
            forward_tokens(&model, &[toks[prefix + i]], prefix + i, &mut last, Some(&mut one[i * vocab..(i + 1) * vocab]), &mut a).unwrap();
        }
        let mut packed = vec![0.0f32; 17 * vocab];
        forward_tokens(&model, &toks[prefix..], prefix, &mut last, Some(&mut packed), &mut b).unwrap();
        assert_eq!(one, packed, "a 17-row verify must remain row-exact");
    }

    /// The real model on hip0 against llama.cpp's greedy tokens
    /// (`allpaca-research/p16/ref_hip.json`). Loads 5 GB: opt in with
    /// `ALLPACA_GEMMA4_REF=1` (and hold the GPU lock).
    #[test]
    fn real_model_greedy_tokens_match_llamacpp() {
        if std::env::var("ALLPACA_GEMMA4_REF").as_deref() != Ok("1") {
            return;
        }
        let path = std::env::var("ALLPACA_GEMMA4_GGUF").unwrap_or_else(|_| "D:/code/inference/allpaca/models/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF/NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf".into());
        let path = std::path::PathBuf::from(path);
        let refs: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string("D:/code/inference/allpaca-research/p16/ref_hip.json").unwrap()).unwrap();
        let refs_vk: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string("D:/code/inference/allpaca-research/p16/ref_vk.json").unwrap()).unwrap();
        let file = Arc::new(GgufFile::open(&path).unwrap());
        let mut model = TinyModel::from_gguf(&file).unwrap();
        model.bind_gguf(file);
        let dev = std::env::var("ALLPACA_GEMMA4_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        if std::env::var("ALLPACA_GEMMA4_HOST").as_deref() != Ok("1") {
            attach_gpu(&model, Arc::new(allpaca_hip::HipWorkspace::new(dev, 8 << 20, 256).unwrap())).unwrap();
        }
        let ids = |v: &serde_json::Value| -> Vec<u32> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect() };
        let vocab = model.config.vocab_size;
        if std::env::var("ALLPACA_G4_MATCHECK").as_deref() == Ok("1") {
            let hip = allpaca_hip::HipWorkspace::new(dev, 8 << 20, 256).unwrap();
            let f = model.g4.as_ref().unwrap().file().unwrap();
            for name in ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_output.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.0.inp_gate.weight", "blk.0.proj.weight", "per_layer_model_proj.weight", "blk.24.attn_k.weight"] {
                for m in [1usize, 3] {
                    let e = crate::gemma4_gpu::matrix_error(f, name, &hip, m, 1.0).unwrap();
                    eprintln!("matcheck {name} {:?} m={m} rel_err={e:.5}", f.tensor(name).unwrap().ggml_type);
                }
            }
            return;
        }
        if let Ok(probe) = std::env::var("ALLPACA_GEMMA4_PROBE") {
            // Top-10 probabilities after a prompt of comma-separated ids,
            // prefilled at once and one row at a time.
            let prompt: Vec<u32> = probe.split(',').map(|v| v.trim().parse().unwrap()).collect();
            for step in [false, true] {
                let mut mem = Gemma4Memory::default();
                let mut logits = vec![0.0f32; vocab];
                if step {
                    for (i, &t) in prompt.iter().enumerate() {
                        forward_tokens(&model, &[t], i, &mut logits, None, &mut mem).unwrap();
                    }
                } else {
                    forward_tokens(&model, &prompt, 0, &mut logits, None, &mut mem).unwrap();
                }
                let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let z: f32 = logits.iter().map(|v| (v - m).exp()).sum();
                let mut idx: Vec<usize> = (0..vocab).collect();
                idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
                let top: Vec<(usize, f32)> = idx[..10].iter().map(|&i| (i, (logits[i] - m).exp() / z)).collect();
                eprintln!("probe step={step}: {top:?}");
            }
            return;
        }
        for case in ["p512", "p4k"] {
            let prompt = ids(&refs[case]["prompt"]);
            let want = ids(&refs[case]["gen"]);
            let mut mem = Gemma4Memory::default();
            let mut logits = vec![0.0f32; vocab];
            let t0 = std::time::Instant::now();
            forward_tokens(&model, &prompt, 0, &mut logits, None, &mut mem).unwrap();
            let prefill = t0.elapsed();
            let mut got = Vec::new();
            let t1 = std::time::Instant::now();
            for _ in 0..want.len() {
                let t = crate::sampling::argmax(&logits);
                got.push(t);
                forward_tokens(&model, &[t], mem.tokens(), &mut logits, None, &mut mem).unwrap();
            }
            let decode = t1.elapsed();
            // Two llama.cpp backends (CPU build 10190, Vulkan b11852) agree
            // with each other only for a few tokens: greedy streams fork at
            // near-ties. Allpaca must agree with one of them at least as long
            // as they agree with each other, and the text must stay coherent.
            let vk = ids(&refs_vk[case]["gen"]);
            let run = |a: &[u32], b: &[u32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
            let (cpu, vulkan, between) = (run(&got, &want), run(&got, &vk), run(&want, &vk));
            eprintln!(
                "{case}: {} prompt tokens, prefill {prefill:?}, decode {:.1} tok/s, agree with llama.cpp CPU {cpu}/{n}, Vulkan {vulkan}/{n} (CPU vs Vulkan {between}/{n}); got {got:?}",
                prompt.len(),
                want.len() as f64 / decode.as_secs_f64(),
                n = want.len()
            );
            assert!(cpu.max(vulkan) >= between, "{case}");
        }
    }

    fn gpu_of(model: &TinyModel) -> std::sync::MutexGuard<'_, Option<crate::gemma4_gpu::Gemma4Gpu>> {
        model.g4.as_ref().unwrap().gpu.lock().unwrap()
    }

    /// Prefill `n` tokens in passes of `pass`, then `steps` decode rows;
    /// every row's logits.
    fn run_rows(model: &TinyModel, n: usize, pass: usize, steps: usize) -> (Vec<f32>, Gemma4Memory) {
        let vocab = model.config.vocab_size;
        let toks = prompt(n + steps);
        let mut mem = Gemma4Memory::default();
        let mut rows = vec![0.0f32; (n + steps) * vocab];
        let mut last = vec![0.0f32; vocab];
        let mut at = 0;
        while at < n + steps {
            let len = if at < n { pass.min(n - at) } else { 1 };
            forward_tokens(model, &toks[at..at + len], at, &mut last, Some(&mut rows[at * vocab..(at + len) * vocab]), &mut mem).unwrap();
            at += len;
        }
        (rows, mem)
    }

    #[test]
    fn gpu_swa_buffer_shift_keeps_the_bits() {
        let shifted = tiny(2, "gpu_shift_a");
        let wide = tiny(2, "gpu_shift_b");
        if !on_gpu(&shifted) || !on_gpu(&wide) {
            return;
        }
        gpu_of(&wide).as_mut().unwrap().set_swa_rows(1 << 14);
        let (a, ma) = run_rows(&shifted, 1500, 200, 40);
        let (b, mb) = run_rows(&wide, 1500, 200, 40);
        let (da, db) = (ma.device.as_ref().unwrap(), mb.device.as_ref().unwrap());
        assert!(da.swa_base() > 0, "the default SWA buffer must have moved");
        assert_eq!(db.swa_base(), 0);
        assert_eq!(a, b, "moving the SWA rows must not change a bit");
        // A rejected tail after a move: truncate 8 and redo them.
        let mut m2 = ma.clone();
        let sp = shifted.g4.as_ref().unwrap().spec.clone();
        m2.truncate(1532, &sp);
        let vocab = shifted.config.vocab_size;
        let toks = prompt(1540);
        let mut redo = vec![0.0f32; 8 * vocab];
        let mut last = vec![0.0f32; vocab];
        forward_tokens(&shifted, &toks[1532..], 1532, &mut last, Some(&mut redo), &mut m2).unwrap();
        assert_eq!(&redo[..], &a[1532 * vocab..1540 * vocab]);
    }

    #[test]
    fn gpu_long_prompt_keeps_swa_memory_bounded() {
        let model = tiny(2, "gpu_long");
        if !on_gpu(&model) {
            return;
        }
        let sp = model.g4.as_ref().unwrap().spec.clone();
        let toks = prompt(40_000);
        let mut mem = Gemma4Memory::default();
        let mut last = vec![0.0f32; model.config.vocab_size];
        forward_tokens(&model, &toks, 0, &mut last, None, &mut mem).unwrap();
        let kv = mem.device.as_ref().unwrap();
        let swa_rows = crate::gemma4_gpu::Gemma4DeviceKv::swa_rows_for(&sp);
        let cap = 40_000usize.div_ceil(4096) * 4096;
        let want: usize = (0..sp.kv_layers)
            .map(|l| 2 * 4 * sp.kv_row(l) * if sp.swa[l] { swa_rows } else { cap })
            .sum::<usize>()
            + 4 * swa_rows * (0..sp.kv_layers).filter(|&l| sp.swa[l]).map(|l| sp.kv_row(l)).max().unwrap();
        assert_eq!(kv.device_bytes() as usize, want, "SWA buffers stay {swa_rows} rows; globals grow in 4096-token steps");
        assert!(kv.swa_base() > 39_000 - swa_rows);
        assert!(crate::gemma4_gpu::kv_device_bytes() >= kv.device_bytes());
        assert!(last.iter().all(|v| v.is_finite()));
    }

    fn device_vec(hip: &allpaca_hip::HipWorkspace, v: &[f32]) -> allpaca_hip::DeviceWeights {
        let w = allpaca_hip::DeviceWeights::alloc(hip.device, v.len() * 4).unwrap();
        hip.copy_h2d_f32(w.ptr_at(0) as *mut f32, v).unwrap();
        w
    }

    fn from_device(hip: &allpaca_hip::HipWorkspace, w: &allpaca_hip::DeviceWeights, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        hip.copy_d2h_f32(w.ptr_at(0), &mut out).unwrap();
        out
    }

    fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs() / b.abs().max(1.0)).fold(0.0f32, f32::max);
        assert!(worst <= tol, "{what}: worst relative error {worst}");
    }

    #[test]
    fn gpu_kernels_match_the_host_at_gemma4_head_dims() {
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            eprintln!("gemma4 GPU test skipped: no HIP device");
            return;
        };
        let val = |i: usize, salt: usize| (((i * 7919 + salt) % 997) as f32 - 498.0) / 97.0;
        for hd in [256usize, 512] {
            let (rows, heads) = (5usize, 3usize);
            let x: Vec<f32> = (0..rows * heads * hd).map(|i| val(i, hd)).collect();
            // Proportional rope: a quarter of the pairs turn.
            let ff: Vec<f32> = (0..hd / 2).map(|i| if i < hd / 8 { 1.0 } else { 1e30 }).collect();
            for (theta, ffs) in [(10_000.0f32, None), (1_000_000.0, Some(&ff))] {
                let d = device_vec(&hip, &x);
                let fd = ffs.map(|f| device_vec(&hip, f));
                allpaca_hip::rope_neox_ff_rows_device(d.ptr_at(0) as *mut f32, rows, heads, hd, 777, theta, fd.as_ref().map_or(std::ptr::null(), |f| f.ptr_at(0))).unwrap();
                let mut want = x.clone();
                for (r, row) in want.chunks_mut(heads * hd).enumerate() {
                    for head in row.chunks_mut(hd) {
                        rope_neox_ff(head, 777 + r, theta, ffs.map(|f| f.as_slice()));
                    }
                }
                close(&from_device(&hip, &d, x.len()), &want, 1e-3, &format!("rope hd {hd}"));
            }
            let up: Vec<f32> = (0..x.len()).map(|i| val(i, 3)).collect();
            let (g, u) = (device_vec(&hip, &x), device_vec(&hip, &up));
            allpaca_hip::gelu_tanh_mul_device(g.ptr_at(0) as *mut f32, u.ptr_at(0), x.len()).unwrap();
            let want: Vec<f32> = x.iter().zip(&up).map(|(a, b)| gelu_tanh(*a) * b).collect();
            close(&from_device(&hip, &g, x.len()), &want, 1e-5, "gelu_tanh_mul");
            let big: Vec<f32> = x.iter().map(|v| v * 20.0).collect();
            let c = device_vec(&hip, &big);
            allpaca_hip::softcap_device(c.ptr_at(0) as *mut f32, big.len(), 30.0).unwrap();
            let mut want = big.clone();
            softcap(&mut want, 30.0);
            close(&from_device(&hip, &c, big.len()), &want, 1e-5, "softcap");
            // Attention at this head dim (the 512 case is the widest LDS
            // tile): causal over 300 keys, 8 query heads on 2 KV heads,
            // queries pre-scaled by sqrt(hd) as the chain does.
            let (n_q, n_kv, h, kvh) = (4usize, 300usize, 8usize, 2usize);
            let q: Vec<f32> = (0..n_q * h * hd).map(|i| val(i, 11) * 0.05).collect();
            let k: Vec<f32> = (0..n_kv * kvh * hd).map(|i| val(i, 13) * 0.05).collect();
            let v: Vec<f32> = (0..n_kv * kvh * hd).map(|i| val(i, 17)).collect();
            let s = (hd as f32).sqrt();
            let qs: Vec<f32> = q.iter().map(|x| x * s).collect();
            let (qd, kd, vd) = (device_vec(&hip, &qs), device_vec(&hip, &k), device_vec(&hip, &v));
            let od = device_vec(&hip, &vec![0.0; n_q * h * hd]);
            hip.attention_device(qd.ptr_at(0), kd.ptr_at(0), vd.ptr_at(0), od.ptr_at(0) as *mut f32, n_q as i32, n_kv as i32, h as i32, kvh as i32, hd as i32, true).unwrap();
            let got = from_device(&hip, &od, n_q * h * hd);
            for r in 0..n_q {
                let pos = n_kv - n_q + r;
                let want = attend(&q[r * h * hd..(r + 1) * h * hd], &k, &v, 0, pos, h, kvh, hd);
                close(&got[r * h * hd..(r + 1) * h * hd], &want, 1e-4, &format!("attention hd {hd} row {r}"));
            }
        }
    }

    #[test]
    fn rope_with_huge_freq_factor_leaves_the_pair_alone() {
        let mut head: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let orig = head.clone();
        rope_neox_ff(&mut head, 37, 10_000.0, Some(&[1.0, 1e30, 1e30, 1e30]));
        assert_eq!(&head[1..4], &orig[1..4]);
        assert_eq!(&head[5..8], &orig[5..8]);
        assert_ne!(head[0], orig[0]);
    }
}
