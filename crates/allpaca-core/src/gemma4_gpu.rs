//! `gemma4` on HIP: a pass of rows runs as one chain of kernels on one card,
//! with the sequence's keys and values in its own device buffers
//! ([`Gemma4DeviceKv`]). The token and per-layer embedding rows are read on
//! the host (the PLE table is IQ4_NL, 1.4 GB) and uploaded per pass.
//!
//! Exactness: a pass of up to [`EXACT_ROWS`] rows (decode, a draft verify)
//! runs under [`allpaca_hip::ExactRows`], so each row equals that row run
//! alone. Attention keeps the same bits too: global blocks use the causal
//! kernel over keys from position 0, SWA blocks the masked kernel over keys
//! from a 128-aligned start, so the kernels' 128-key chunks fall on the same
//! absolute positions whatever the pass, and the masked keys add nothing.

use std::sync::Arc;

use allpaca_gguf::{GgmlType, GgufFile};
use allpaca_hip::{memcpy, set_device, DeviceWeights, HipWorkspace, MEMCPY_D2D};
use allpaca_types::{AllpacaError, DegradedError};

use crate::gemma4::{load_err, softcap, Gemma4Memory, Gemma4Spec, Gemma4Weights};
use crate::model::TinyModel;

/// Rows per device pass; longer prompts run as several.
pub(crate) const MAX_ROWS: usize = 256;
/// Passes up to this many rows are row-exact (decode and verify).
pub(crate) const EXACT_ROWS: usize = 16;
/// Output-head rows per GEMM (the logits buffer holds this many).
const HEAD_ROWS: usize = 16;
/// The attention kernels' key chunk; SWA key windows start on a multiple.
const ATTN_CHUNK: usize = 128;
/// Tokens of K/V a sequence starts with on the card; it doubles from there.
const KV_START: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    F32,
    F16,
    Bf16,
    Q8,
    Q4K,
    Q5K,
    Q6K,
}

/// A matrix `[n, k]` on the card in an encoding a GEMM kernel reads.
struct Mat {
    kind: Kind,
    w: DeviceWeights,
    n: usize,
    k: usize,
    row_bytes: usize,
}

impl Mat {
    fn load(file: &GgufFile, name: &str, device: i32) -> Result<Self, AllpacaError> {
        let info = file.tensor(name).ok_or_else(|| load_err(format!("gemma4: missing tensor {name}")))?;
        let k = info.dims.first().copied().unwrap_or(1) as usize;
        let n = info.dims.get(1).copied().unwrap_or(1) as usize;
        let raw = file.tensor_bytes_of(info)?;
        let (kind, bytes) = match info.ggml_type {
            GgmlType::F32 => (Kind::F32, raw.to_vec()),
            GgmlType::F16 => (Kind::F16, raw.to_vec()),
            GgmlType::BF16 => (Kind::Bf16, raw.to_vec()),
            GgmlType::Q8_0 => (Kind::Q8, allpaca_hip::stage_q8_0_blocks(raw)?),
            GgmlType::Q4_K => (Kind::Q4K, raw.to_vec()),
            GgmlType::Q5_K => (Kind::Q5K, raw.to_vec()),
            GgmlType::Q6_K => (Kind::Q6K, raw.to_vec()),
            _ => {
                // Any other encoding: requantised to Q8_0 (rows are whole blocks).
                let q = allpaca_gguf::quantize_q8_0(&file.tensor_f32(name)?);
                (Kind::Q8, allpaca_hip::stage_q8_0_blocks(&q)?)
            }
        };
        let row_bytes = bytes.len() / n.max(1);
        Ok(Self {
            kind,
            w: DeviceWeights::upload_bytes(device, &[&bytes])?,
            n,
            k,
            row_bytes,
        })
    }

    /// `c[m, count] = a[m, k] · rows first..first+count`.
    fn gemm_rows(&self, hip: &HipWorkspace, a: *const f32, c: *mut f32, m: usize, first: usize, count: usize) -> Result<(), AllpacaError> {
        let b = self.w.byte_ptr_at(first * self.row_bytes);
        let (m, n, k) = (m as i32, count as i32, self.k as i32);
        match self.kind {
            Kind::F32 => hip.gemm_nt_dev(a, b.cast(), c, m, n, k),
            Kind::F16 => hip.gemm_nt_f16_dev(a, b.cast(), c, m, n, k),
            Kind::Bf16 => hip.gemm_nt_bf16_dev(a, b.cast(), c, m, n, k),
            Kind::Q8 => hip.gemm_nt_q8_dev(a, b, c, m, n, k),
            Kind::Q4K => hip.gemm_nt_q4_k_dev(a, b, c, m, n, k),
            Kind::Q5K => hip.gemm_nt_q5_k_dev(a, b, c, m, n, k),
            Kind::Q6K => hip.gemm_nt_q6_k_dev(a, b, c, m, n, k),
        }
    }

    fn gemm(&self, hip: &HipWorkspace, a: *const f32, c: *mut f32, m: usize) -> Result<(), AllpacaError> {
        self.gemm_rows(hip, a, c, m, 0, self.n)
    }

    fn bytes(&self) -> u64 {
        self.w.bytes()
    }
}

fn vec_f32(file: &GgufFile, name: &str, device: i32) -> Result<DeviceWeights, AllpacaError> {
    let v = file.tensor_f32(name)?;
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    DeviceWeights::upload_bytes(device, &[&bytes])
}

struct Layer {
    attn_norm: DeviceWeights,
    q_norm: DeviceWeights,
    k_norm: Option<DeviceWeights>,
    post_attn: DeviceWeights,
    ffn_norm: DeviceWeights,
    post_ffw: DeviceWeights,
    post_norm: Option<DeviceWeights>,
    out_scale: Option<f32>,
    wq: Mat,
    wk: Option<Mat>,
    wv: Option<Mat>,
    wo: Mat,
    gate: Mat,
    up: Mat,
    down: Mat,
    inp_gate: Option<Mat>,
    proj: Option<Mat>,
}

/// Device scratch for one pass of up to [`MAX_ROWS`] rows.
struct Scratch {
    _buf: DeviceWeights,
    x: *mut f32,
    xn: *mut f32,
    q: *mut f32,
    attn: *mut f32,
    o: *mut f32,
    g: *mut f32,
    u: *mut f32,
    pg: *mut f32,
    /// Per-layer inputs, `[n_layers][rows][ple_dim]`.
    ple: *mut f32,
    ple_tok: *mut f32,
    logits: *mut f32,
    mask: *mut u8,
}

// SAFETY: raw pointers into `_buf`, which lives as long as the scratch; the
// device chain is only driven under the weights' GPU lock.
unsafe impl Send for Scratch {}

/// One sequence's keys and values on the card, per block that writes them.
pub struct Gemma4DeviceKv {
    device: i32,
    k: Vec<DeviceWeights>,
    v: Vec<DeviceWeights>,
    /// Tokens each buffer holds room for.
    cap: usize,
}

impl std::fmt::Debug for Gemma4DeviceKv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemma4DeviceKv").field("cap", &self.cap).finish_non_exhaustive()
    }
}

fn copy_of(src: &DeviceWeights, bytes: usize) -> Result<DeviceWeights, AllpacaError> {
    let dst = DeviceWeights::alloc(src.device(), bytes.max(src.bytes() as usize))?;
    set_device(src.device())?;
    memcpy(dst.ptr_at(0).cast_mut().cast(), src.ptr_at(0).cast(), src.bytes() as usize, MEMCPY_D2D)?;
    Ok(dst)
}

impl Clone for Gemma4DeviceKv {
    /// A deep copy on the same card.
    fn clone(&self) -> Self {
        let dup = |w: &DeviceWeights| copy_of(w, 0).expect("copy gemma4 device K/V");
        Self {
            device: self.device,
            k: self.k.iter().map(dup).collect(),
            v: self.v.iter().map(dup).collect(),
            cap: self.cap,
        }
    }
}

impl Gemma4DeviceKv {
    fn new(spec: &Gemma4Spec, device: i32, cap: usize) -> Result<Self, AllpacaError> {
        let mut k = Vec::with_capacity(spec.kv_layers);
        let mut v = Vec::with_capacity(spec.kv_layers);
        for l in 0..spec.kv_layers {
            let bytes = cap * spec.kv_row(l) * 4;
            k.push(DeviceWeights::alloc(device, bytes)?);
            v.push(DeviceWeights::alloc(device, bytes)?);
        }
        Ok(Self { device, k, v, cap })
    }

    /// Room for `tokens`, keeping what the buffers hold.
    fn reserve(&mut self, spec: &Gemma4Spec, tokens: usize) -> Result<(), AllpacaError> {
        if tokens <= self.cap {
            return Ok(());
        }
        let cap = tokens.max(self.cap * 2);
        for l in 0..spec.kv_layers {
            let bytes = cap * spec.kv_row(l) * 4;
            self.k[l] = copy_of(&self.k[l], bytes)?;
            self.v[l] = copy_of(&self.v[l], bytes)?;
        }
        self.cap = cap;
        Ok(())
    }

    pub(crate) fn device_bytes(&self) -> u64 {
        self.k.iter().chain(&self.v).map(DeviceWeights::bytes).sum()
    }
}

pub struct Gemma4Gpu {
    hip: Arc<HipWorkspace>,
    layers: Vec<Layer>,
    ple_proj: Option<Mat>,
    ple_norm: Option<DeviceWeights>,
    out_norm: DeviceWeights,
    head: Mat,
    rope_ff: Option<DeviceWeights>,
    /// Ones, the gain of V's weightless RMS norm.
    ones: DeviceWeights,
    scratch: Scratch,
    weight_bytes: u64,
}

impl Gemma4Gpu {
    pub fn load(file: &GgufFile, spec: &Gemma4Spec, hip: Arc<HipWorkspace>) -> Result<Self, AllpacaError> {
        let dev = hip.device;
        hip.bind()?;
        let opt_vec = |name: &str| -> Result<Option<DeviceWeights>, AllpacaError> {
            file.tensor(name).map(|_| vec_f32(file, name, dev)).transpose()
        };
        let opt_mat = |name: &str| -> Result<Option<Mat>, AllpacaError> {
            file.tensor(name).map(|_| Mat::load(file, name, dev)).transpose()
        };
        let mut layers = Vec::with_capacity(spec.n_layers);
        for l in 0..spec.n_layers {
            let t = |s: &str| format!("blk.{l}.{s}");
            let owns_kv = l < spec.kv_layers;
            let scale = match file.tensor(&t("layer_output_scale.weight")) {
                Some(_) => Some(file.tensor_f32(&t("layer_output_scale.weight"))?[0]),
                None => None,
            };
            layers.push(Layer {
                attn_norm: vec_f32(file, &t("attn_norm.weight"), dev)?,
                q_norm: vec_f32(file, &t("attn_q_norm.weight"), dev)?,
                k_norm: if owns_kv { Some(vec_f32(file, &t("attn_k_norm.weight"), dev)?) } else { None },
                post_attn: vec_f32(file, &t("post_attention_norm.weight"), dev)?,
                ffn_norm: vec_f32(file, &t("ffn_norm.weight"), dev)?,
                post_ffw: vec_f32(file, &t("post_ffw_norm.weight"), dev)?,
                post_norm: opt_vec(&t("post_norm.weight"))?,
                out_scale: scale,
                wq: Mat::load(file, &t("attn_q.weight"), dev)?,
                wk: if owns_kv { Some(Mat::load(file, &t("attn_k.weight"), dev)?) } else { None },
                wv: if owns_kv { opt_mat(&t("attn_v.weight"))? } else { None },
                wo: Mat::load(file, &t("attn_output.weight"), dev)?,
                gate: Mat::load(file, &t("ffn_gate.weight"), dev)?,
                up: Mat::load(file, &t("ffn_up.weight"), dev)?,
                down: Mat::load(file, &t("ffn_down.weight"), dev)?,
                inp_gate: opt_mat(&t("inp_gate.weight"))?,
                proj: opt_mat(&t("proj.weight"))?,
            });
        }
        let head_name = if file.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
        let head = Mat::load(file, head_name, dev)?;
        let ple_proj = if spec.ple_dim > 0 { Some(Mat::load(file, "per_layer_model_proj.weight", dev)?) } else { None };
        let ple_norm = if spec.ple_dim > 0 { Some(vec_f32(file, "per_layer_proj_norm.weight", dev)?) } else { None };
        let out_norm = vec_f32(file, "output_norm.weight", dev)?;
        let rope_ff = opt_vec("rope_freqs.weight")?;
        let scratch = Self::scratch(spec, dev)?;
        let one_bytes: Vec<u8> = (0..spec.head_dim.max(spec.head_dim_swa)).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let ones = DeviceWeights::upload_bytes(dev, &[&one_bytes])?;
        let mut weight_bytes = head.bytes() + ple_proj.as_ref().map_or(0, Mat::bytes);
        for l in &layers {
            for m in [Some(&l.wq), l.wk.as_ref(), l.wv.as_ref(), Some(&l.wo), Some(&l.gate), Some(&l.up), Some(&l.down), l.inp_gate.as_ref(), l.proj.as_ref()]
                .into_iter()
                .flatten()
            {
                weight_bytes += m.bytes();
            }
        }
        tracing::info!(device = dev, weight_mib = weight_bytes >> 20, layers = spec.n_layers, "gemma4 device chain loaded");
        Ok(Self {
            hip,
            layers,
            ple_proj,
            ple_norm,
            out_norm,
            head,
            rope_ff,
            ones,
            scratch,
            weight_bytes,
        })
    }

    fn scratch(spec: &Gemma4Spec, dev: i32) -> Result<Scratch, AllpacaError> {
        let r = MAX_ROWS;
        let q_max = spec.n_heads * spec.head_dim.max(spec.head_dim_swa);
        let ff_max = spec.ff.iter().copied().max().unwrap_or(0);
        let mask_cols = spec.window + r + ATTN_CHUNK;
        let sizes = [
            r * spec.dim,
            r * spec.dim,
            r * q_max,
            r * q_max,
            r * spec.dim,
            r * ff_max,
            r * ff_max,
            r * spec.ple_dim.max(1),
            spec.n_layers * r * spec.ple_dim.max(1),
            spec.n_layers * r * spec.ple_dim.max(1),
            HEAD_ROWS * spec.vocab,
            (r * mask_cols).div_ceil(4),
        ];
        let align = |n: usize| n.div_ceil(64) * 64;
        let total: usize = sizes.iter().map(|&n| align(n)).sum();
        let buf = DeviceWeights::alloc(dev, total * 4)?;
        let mut off = 0usize;
        let mut views = [std::ptr::null_mut::<f32>(); 12];
        for (v, &n) in views.iter_mut().zip(&sizes) {
            *v = buf.ptr_at(off * 4) as *mut f32;
            off += align(n);
        }
        let [x, xn, q, attn, o, g, u, pg, ple, ple_tok, logits, mask] = views;
        Ok(Scratch {
            _buf: buf,
            x,
            xn,
            q,
            attn,
            o,
            g,
            u,
            pg,
            ple,
            ple_tok,
            logits,
            mask: mask.cast(),
        })
    }

    pub(crate) fn device(&self) -> i32 {
        self.hip.device
    }

    pub(crate) fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Forward `tokens` from `start_pos`, in passes of at most [`MAX_ROWS`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        model: &TinyModel,
        w: &Gemma4Weights,
        tokens: &[u32],
        start_pos: usize,
        logits: &mut [f32],
        mut row_logits: Option<&mut [f32]>,
        mem: &mut Gemma4Memory,
    ) -> Result<(), AllpacaError> {
        let vocab = w.spec.vocab;
        if mem.tokens() != start_pos {
            return Err(AllpacaError::Fatal(allpaca_types::FatalError::InvalidConfig(format!(
                "gemma4 sequence memory holds {} tokens, forward is at position {start_pos}",
                mem.tokens()
            ))));
        }
        for (pass, chunk) in tokens.chunks(MAX_ROWS).enumerate() {
            let first = pass * MAX_ROWS;
            let rows = row_logits.as_deref_mut().map(|r| &mut r[first * vocab..(first + chunk.len()) * vocab]);
            self.pass(model, w, chunk, start_pos + first, &mut logits[..vocab], rows, mem)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn pass(
        &self,
        model: &TinyModel,
        w: &Gemma4Weights,
        tokens: &[u32],
        start_pos: usize,
        logits: &mut [f32],
        row_logits: Option<&mut [f32]>,
        mem: &mut Gemma4Memory,
    ) -> Result<(), AllpacaError> {
        let sp = &w.spec;
        let hip = &*self.hip;
        let s = &self.scratch;
        let m = tokens.len();
        let (dim, eps, pd) = (sp.dim, sp.eps, sp.ple_dim);
        let end = start_pos + m;
        hip.bind()?;
        let _exact = (m <= EXACT_ROWS).then(allpaca_hip::ExactRows::enable);
        if mem.device.as_ref().is_some_and(|d| d.device != hip.device) {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "gemma4 sequence K/V is on another card".into(),
            )));
        }
        if mem.device.is_none() {
            mem.device = Some(Gemma4DeviceKv::new(sp, hip.device, KV_START.max(end))?);
        }
        let kv = mem.device.as_mut().expect("device K/V");
        kv.reserve(sp, end)?;

        // Embeddings on the host: the scaled token rows and the PLE rows,
        // the latter laid out [layer][row][ple_dim].
        let mut x0 = Vec::with_capacity(m * dim);
        let mut ple_tok = vec![0.0f32; sp.n_layers * m * pd];
        for (r, &t) in tokens.iter().enumerate() {
            x0.extend_from_slice(&w.embed(t)?);
            if pd > 0 {
                let row = w.ple_token(t)?;
                for l in 0..sp.n_layers {
                    ple_tok[(l * m + r) * pd..(l * m + r + 1) * pd].copy_from_slice(&row[l * pd..(l + 1) * pd]);
                }
            }
        }
        hip.copy_h2d_f32(s.x, &x0)?;
        if let (Some(proj), Some(norm)) = (&self.ple_proj, &self.ple_norm) {
            hip.copy_h2d_f32(s.ple_tok, &ple_tok)?;
            for l in 0..sp.n_layers {
                // SAFETY: `ple` holds n_layers * MAX_ROWS * pd floats.
                let dst = unsafe { s.ple.add(l * m * pd) };
                proj.gemm_rows(hip, s.x, dst, m, l * pd, pd)?;
            }
            let n = sp.n_layers * m * pd;
            hip.scale_f32(s.ple, n, (dim as f32).sqrt().recip())?;
            hip.rmsnorm_device_rows(s.ple, norm.ptr_at(0), s.ple, (sp.n_layers * m) as i32, pd as i32, eps)?;
            hip.residual_add(s.ple, s.ple_tok, n as i32)?;
            hip.scale_f32(s.ple, n, std::f32::consts::FRAC_1_SQRT_2)?;
        }

        // The SWA key window of this pass: from a chunk-aligned start up to
        // the last row, with a mask per row.
        let swa_lo = (start_pos + 1).saturating_sub(sp.window) / ATTN_CHUNK * ATTN_CHUNK;
        let swa_n = end - swa_lo;
        if sp.swa.iter().any(|&b| b) {
            let mut mask = vec![0u8; m * swa_n];
            for r in 0..m {
                let p = start_pos + r;
                let lo = (p + 1).saturating_sub(sp.window);
                for j in lo..=p {
                    mask[r * swa_n + (j - swa_lo)] = 1;
                }
            }
            allpaca_hip::memcpy(s.mask, mask.as_ptr(), mask.len(), allpaca_hip::MEMCPY_H2D)?;
        }

        let rows: Vec<(usize, u32)> = tokens.iter().enumerate().map(|(r, &t)| (start_pos + r, t)).collect();
        let (mi, di) = (m as i32, dim as i32);
        for (l, ly) in self.layers.iter().enumerate() {
            if let Some(k) = model.layer_capture.slot(l) {
                let mut host = vec![0.0f32; m * dim];
                hip.copy_d2h_f32(s.x, &mut host)?;
                model.layer_capture.record(k, &rows, &host);
            }
            let hd = sp.head_dim_of(l);
            let theta = sp.theta_of(l);
            let ff = if sp.swa[l] { std::ptr::null() } else { self.rope_ff.as_ref().map_or(std::ptr::null(), |f| f.ptr_at(0)) };
            let kv_row = sp.kv_row(l);
            hip.rmsnorm_device_rows(s.x, ly.attn_norm.ptr_at(0), s.xn, mi, di, eps)?;
            ly.wq.gemm(hip, s.xn, s.q, m)?;
            hip.rmsnorm_heads_device(s.q, ly.q_norm.ptr_at(0), (m * sp.n_heads) as i32, hd as i32, eps)?;
            allpaca_hip::rope_neox_ff_rows_device(s.q, m, sp.n_heads, hd, start_pos, theta, ff)?;
            // The kernels scale scores by 1/sqrt(hd); gemma4 uses 1.
            hip.scale_f32(s.q, m * sp.n_heads * hd, (hd as f32).sqrt())?;
            if let (Some(wk), Some(kn)) = (&ly.wk, &ly.k_norm) {
                let kd = kv.k[l].ptr_at(start_pos * kv_row * 4) as *mut f32;
                let vd = kv.v[l].ptr_at(start_pos * kv_row * 4) as *mut f32;
                wk.gemm(hip, s.xn, kd, m)?;
                match &ly.wv {
                    Some(wv) => wv.gemm(hip, s.xn, vd, m)?,
                    None => hip.copy_d2d_f32(vd, kd, m * kv_row)?,
                }
                hip.rmsnorm_heads_device(kd, kn.ptr_at(0), (m * sp.n_kv_heads) as i32, hd as i32, eps)?;
                // V: RMS norm without a gain.
                self.rmsnorm_plain(vd, m * sp.n_kv_heads, hd, eps)?;
                allpaca_hip::rope_neox_ff_rows_device(kd, m, sp.n_kv_heads, hd, start_pos, theta, ff)?;
            }
            let src = sp.kv_source(l);
            let (kb, vb) = (&kv.k[src], &kv.v[src]);
            let (h, kvh) = (sp.n_heads as i32, sp.n_kv_heads as i32);
            if sp.swa[l] {
                hip.attention_device_masked(
                    s.q,
                    kb.ptr_at(swa_lo * kv_row * 4),
                    vb.ptr_at(swa_lo * kv_row * 4),
                    s.attn,
                    mi,
                    swa_n as i32,
                    h,
                    kvh,
                    hd as i32,
                    s.mask,
                )?;
            } else {
                hip.attention_device(s.q, kb.ptr_at(0), vb.ptr_at(0), s.attn, mi, end as i32, h, kvh, hd as i32, true)?;
            }
            if crate::gemma4::dump_on() {
                let qd = sp.n_heads * hd;
                let mut h = vec![0.0f32; m * qd];
                hip.copy_d2h_f32(s.q, &mut h)?;
                let sc = (hd as f32).sqrt();
                let q: Vec<f32> = h[(m - 1) * qd..].iter().map(|v| v / sc).collect();
                crate::gemma4::dump("q", l, &q);
                hip.copy_d2h_f32(s.attn, &mut h)?;
                crate::gemma4::dump("attn", l, &h[(m - 1) * qd..]);
            }
            ly.wo.gemm(hip, s.attn, s.o, m)?;
            hip.rmsnorm_device_rows(s.o, ly.post_attn.ptr_at(0), s.o, mi, di, eps)?;
            hip.residual_add(s.x, s.o, (m * dim) as i32)?;
            hip.rmsnorm_device_rows(s.x, ly.ffn_norm.ptr_at(0), s.xn, mi, di, eps)?;
            ly.gate.gemm(hip, s.xn, s.g, m)?;
            ly.up.gemm(hip, s.xn, s.u, m)?;
            allpaca_hip::gelu_tanh_mul_device(s.g, s.u, m * sp.ff[l])?;
            ly.down.gemm(hip, s.g, s.o, m)?;
            hip.rmsnorm_device_rows(s.o, ly.post_ffw.ptr_at(0), s.o, mi, di, eps)?;
            hip.residual_add(s.x, s.o, (m * dim) as i32)?;
            if crate::gemma4::dump_on() {
                let mut h = vec![0.0f32; m * dim];
                hip.copy_d2h_f32(s.x, &mut h)?;
                crate::gemma4::dump("ffn", l, &h[(m - 1) * dim..]);
            }
            if let (Some(gate), Some(proj), Some(pn)) = (&ly.inp_gate, &ly.proj, &ly.post_norm) {
                gate.gemm(hip, s.x, s.pg, m)?;
                // SAFETY: layer `l`'s rows of the PLE inputs.
                let inp = unsafe { s.ple.add(l * m * pd) };
                allpaca_hip::gelu_tanh_mul_device(s.pg, inp, m * pd)?;
                proj.gemm(hip, s.pg, s.o, m)?;
                hip.rmsnorm_device_rows(s.o, pn.ptr_at(0), s.o, mi, di, eps)?;
                hip.residual_add(s.x, s.o, (m * dim) as i32)?;
            }
            if let Some(scale) = ly.out_scale {
                hip.scale_f32(s.x, m * dim, scale)?;
            }
        }
        mem.set_tokens(end);
        hip.rmsnorm_device_rows(s.x, self.out_norm.ptr_at(0), s.xn, mi, di, eps)?;
        let vocab = sp.vocab;
        let head_rows = |first: usize, n: usize, out: &mut [f32]| -> Result<(), AllpacaError> {
            // SAFETY: rows first..first+n of the normed hidden.
            let a = unsafe { s.xn.add(first * dim) };
            let _scope = hip.lm_head_scope();
            self.head.gemm(hip, a, s.logits, n)?;
            hip.copy_d2h_f32(s.logits, &mut out[..n * vocab])?;
            softcap(&mut out[..n * vocab], sp.softcap);
            Ok(())
        };
        match row_logits {
            Some(rows) => {
                let mut first = 0;
                while first < m {
                    let n = (m - first).min(HEAD_ROWS);
                    head_rows(first, n, &mut rows[first * vocab..(first + n) * vocab])?;
                    first += n;
                }
                logits[..vocab].copy_from_slice(&rows[(m - 1) * vocab..m * vocab]);
            }
            None => head_rows(m - 1, 1, logits)?,
        }
        Ok(())
    }

    /// RMS norm without a gain over `rows` rows of `n` (device, in place).
    fn rmsnorm_plain(&self, x: *mut f32, rows: usize, n: usize, eps: f32) -> Result<(), AllpacaError> {
        self.hip.rmsnorm_heads_device(x, self.ones.ptr_at(0), rows as i32, n as i32, eps)
    }
}

/// Test hook: the largest |device - host| / max|host| of `m` rows through
/// tensor `name` on `hip`'s card (host: the F32 tensor).
#[cfg(test)]
pub(crate) fn matrix_error(file: &GgufFile, name: &str, hip: &HipWorkspace, m: usize, scale: f32) -> Result<f32, AllpacaError> {
    let mat = Mat::load(file, name, hip.device)?;
    let w = file.tensor_f32(name)?;
    let x: Vec<f32> = (0..m * mat.k).map(|i| ((i * 2654435761usize % 1000) as f32 / 500.0 - 1.0) * scale).collect();
    let mut want = vec![0.0f32; m * mat.n];
    allpaca_cpu::matmul_nt(&x, &w, &mut want, m, mat.n, mat.k)?;
    let a = DeviceWeights::alloc(hip.device, x.len() * 4)?;
    let c = DeviceWeights::alloc(hip.device, want.len() * 4)?;
    hip.copy_h2d_f32(a.ptr_at(0) as *mut f32, &x)?;
    let _exact = allpaca_hip::ExactRows::enable();
    mat.gemm(hip, a.ptr_at(0), c.ptr_at(0) as *mut f32, m)?;
    let mut got = vec![0.0f32; want.len()];
    hip.copy_d2h_f32(c.ptr_at(0), &mut got)?;
    let amax = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    Ok(got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max) / amax)
}
