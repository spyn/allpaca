//! DFlash2 block-diffusion draft provider (Inco AI / z-lab, llama.cpp PR #27342).
//!
//! The target exposes the residual stream entering `dflash.target_layers` for
//! every row it forwards ([`LayerCapture`]). Those features are fused (`fc` +
//! `enc.output_norm`) and projected into each draft layer's K/V ring, one slot
//! per target position (sliding window `W`). A draft pass runs the block
//! `[anchor, MASK x (B-1)]` non-causally over the ring plus itself, takes the
//! top-k candidates per position from the target lm_head, and a low-rank
//! selector walks one path through them.
//!
//! Each draft layer wraps attention and FFN in two-tap dynamic convolutions
//! along the block: `out[t] = w0[t] * h[t] + w1[t] * h[t-1]`, with per-group
//! weights projected from the layer's normed input plus a static base.
//!
//! Plain DFlash (llama.cpp `draft-dflash`, e.g. the Nemotron-3.5-Bolt
//! drafter) is the same pass without the convolutions and the selector:
//! the block's tokens embedded times `dflash.embedding_scale`, Qwen3-style
//! layers, and the target head's top-1 at each position.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use allpaca_gguf::{GgmlType, GgufFile};
use allpaca_hip::{DeviceWeights, HipWorkspace};
use allpaca_types::{AllpacaError, DegradedError, FatalError};

use crate::draft::DraftProvider;

/// Slots kept beyond the window: verify rows land before they are known to
/// be accepted and must not evict positions the next query still sees.
const RING_SLACK: usize = 64;
/// Feature rows fused per `fc` GEMM.
const INJECT_CHUNK: usize = 256;

#[derive(Debug, Clone, PartialEq)]
pub struct Dflash2Spec {
    pub n_layers: usize,
    pub dim: usize,
    pub ff: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub theta: f32,
    pub block_size: usize,
    pub conv_kernel: usize,
    pub conv_group: usize,
    pub selector_rank: usize,
    pub top_k: usize,
    pub window: usize,
    /// Target layers whose *input* feeds the draft, in feature order.
    pub target_layers: Vec<usize>,
    pub mask_token: u32,
    /// Plain DFlash: no dynamic convolutions, no selector.
    pub plain: bool,
    /// Scale on the block's token embeddings (`dflash.embedding_scale`).
    pub embedding_scale: f32,
    /// Tokens in the draft's own vocabulary, when it carries one.
    pub vocab: Option<usize>,
}

impl Dflash2Spec {
    pub fn from_gguf(file: &GgufFile) -> Option<Self> {
        let plain = file.tensor("selector_hidden.weight").is_none();
        // A plain DFlash draft has none of DSpark's Markov or hyper-connection
        // tensors and none of DFlash2's convolutions.
        if plain
            && (file.architecture() != Some("dflash")
                || file.tensor("fc.weight").is_none()
                || file.tensor("markov_w1.weight").is_some()
                || file.tensor("hc_head_fn.weight").is_some()
                || file.tensor("blk.0.attn_conv_proj.weight").is_some()
                || file.tensor("blk.0.attn_q.weight").is_none())
        {
            return None;
        }
        let u = |k: &str| file.meta_u32(&format!("dflash.{k}")).map(|v| v as usize);
        let dim = u("embedding_length")?;
        let n_heads = u("attention.head_count")?;
        let head_dim = u("attention.key_length").unwrap_or(dim / n_heads.max(1));
        let target_layers: Vec<usize> = file
            .metadata
            .get("dflash.target_layers")
            .and_then(allpaca_gguf::GgufValue::as_u32_array)?
            .into_iter()
            .map(|v| v as usize)
            .collect();
        let spec = Self {
            n_layers: u("block_count")?,
            dim,
            ff: u("feed_forward_length")?,
            n_heads,
            n_kv_heads: u("attention.head_count_kv").unwrap_or(n_heads),
            head_dim,
            eps: file
                .meta_f32("dflash.attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-6),
            theta: file.meta_f32("dflash.rope.freq_base").unwrap_or(10_000.0),
            block_size: u("block_size").unwrap_or(8),
            conv_kernel: u("conv_kernel_size").unwrap_or(2),
            conv_group: u("conv_group_size").unwrap_or(16),
            selector_rank: u("selector_rank").unwrap_or(256),
            top_k: if plain { 1 } else { u("selector_top_k").unwrap_or(16) },
            window: u("attention.sliding_window").unwrap_or(2048),
            target_layers,
            mask_token: file.meta_u32("tokenizer.ggml.mask_token_id")?,
            plain,
            embedding_scale: file.meta_f32("dflash.embedding_scale").unwrap_or(1.0),
            vocab: match file.metadata.get("tokenizer.ggml.tokens") {
                Some(allpaca_gguf::GgufValue::Array(items)) => Some(items.len()),
                _ => None,
            },
        };
        ((plain || spec.dim % spec.conv_group == 0)
            && spec.block_size >= 2
            && !spec.target_layers.is_empty()
            && spec.top_k > 0)
            .then_some(spec)
    }

    pub fn feature_dim(&self) -> usize {
        self.target_layers.len() * self.dim
    }

    fn conv_proj(&self) -> usize {
        2 * self.conv_kernel * (self.dim / self.conv_group)
    }

    fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    pub fn ring_slots(&self) -> usize {
        self.window + RING_SLACK
    }
}

/// What a file nominated as a DSpark sidecar turns out to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SidecarKind {
    /// A DSpark sidecar for this target.
    Dspark,
    /// A block draft (plain DFlash or DFlash2): attaches only through
    /// `providers = ["dflash1"]` / `["dflash2"]`.
    BlockDraft { plain: bool },
    /// Built for another model (width or vocabulary differ).
    Mismatch,
}

/// Classify a nominated sidecar by its content, for a target of hidden
/// width `dim` and `vocab` tokens.
pub(crate) fn classify_sidecar(file: &GgufFile, dim: usize, vocab: usize) -> SidecarKind {
    if let Some(spec) = Dflash2Spec::from_gguf(file) {
        return SidecarKind::BlockDraft { plain: spec.plain };
    }
    if allpaca_gguf::sidecar_matches_target(file, dim, vocab) {
        SidecarKind::Dspark
    } else {
        SidecarKind::Mismatch
    }
}

/// Whether a block draft fits the configured providers and the target
/// (hidden width `dim`, `vocab` tokens).
pub(crate) fn check_block_draft(
    spec: &Dflash2Spec,
    providers: &[allpaca_types::DraftProviderKind],
    dim: usize,
    vocab: usize,
) -> Result<(), AllpacaError> {
    use allpaca_types::DraftProviderKind as K;
    let (kind, provider) = if spec.plain { ("plain DFlash", K::Dflash1) } else { ("DFlash2", K::Dflash2) };
    let bad = |msg: String| Err(AllpacaError::Fatal(FatalError::InvalidConfig(msg)));
    if !providers.contains(&provider) {
        return bad(format!(
            "the draft is a {kind} draft; use providers = [\"{}\"]",
            if spec.plain { "dflash1" } else { "dflash2" }
        ));
    }
    if spec.dim != dim {
        return bad(format!("{kind} draft width {} != target {dim}", spec.dim));
    }
    if spec.vocab.is_some_and(|v| v != vocab) {
        return bad(format!("{kind} draft vocabulary {} != target {vocab}", spec.vocab.unwrap_or(0)));
    }
    if spec.mask_token as usize >= vocab {
        return bad(format!("{kind} draft mask token {} is outside the target's {vocab} tokens", spec.mask_token));
    }
    Ok(())
}

/// One target row's features: the residual entering each captured layer.
#[derive(Debug, Clone)]
pub struct CapturedRow {
    pub pos: usize,
    pub token: u32,
    pub feat: Vec<f32>,
}

#[derive(Debug, Default)]
struct CaptureBatch {
    rows: Vec<(usize, u32)>,
    feats: Vec<f32>,
    next: usize,
}

#[derive(Debug, Default)]
struct CaptureState {
    layers: Vec<usize>,
    dim: usize,
    max_rows: usize,
    cur: Option<CaptureBatch>,
    done: VecDeque<CapturedRow>,
    reset: bool,
}

/// Target-side tap: forwards call [`Self::slot`] at every layer start and
/// [`Self::record`] the rows' residual when it is a captured layer.
#[derive(Debug, Default)]
pub struct LayerCapture {
    enabled: AtomicBool,
    state: Mutex<CaptureState>,
}

impl LayerCapture {
    pub fn enable(&self, layers: Vec<usize>, dim: usize, max_rows: usize) {
        let mut s = self.lock();
        *s = CaptureState {
            layers,
            dim,
            max_rows: max_rows.max(1),
            reset: true,
            ..CaptureState::default()
        };
        self.enabled.store(true, Ordering::Release);
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Feature slot of `layer`, or `None` when it is not captured.
    pub fn slot(&self, layer: usize) -> Option<usize> {
        if !self.is_enabled() {
            return None;
        }
        self.lock().layers.iter().position(|&l| l == layer)
    }

    /// `x` holds `rows.len()` residual rows entering captured slot `k`.
    pub fn record(&self, k: usize, rows: &[(usize, u32)], x: &[f32]) {
        let mut s = self.lock();
        let dim = s.dim;
        let n_layers = s.layers.len();
        if x.len() < rows.len() * dim || k >= n_layers {
            s.cur = None;
            return;
        }
        if k == 0 {
            s.cur = Some(CaptureBatch {
                rows: rows.to_vec(),
                feats: vec![0.0; rows.len() * n_layers * dim],
                next: 0,
            });
        }
        let Some(mut b) = s.cur.take() else {
            return;
        };
        if b.next != k || b.rows != rows {
            return;
        }
        let stride = n_layers * dim;
        for r in 0..rows.len() {
            b.feats[r * stride + k * dim..r * stride + (k + 1) * dim]
                .copy_from_slice(&x[r * dim..(r + 1) * dim]);
        }
        b.next += 1;
        if b.next < n_layers {
            s.cur = Some(b);
            return;
        }
        for (r, &(pos, token)) in b.rows.iter().enumerate() {
            s.done.push_back(CapturedRow {
                pos,
                token,
                feat: b.feats[r * stride..(r + 1) * stride].to_vec(),
            });
        }
        while s.done.len() > s.max_rows {
            s.done.pop_front();
        }
    }

    /// Rows from another sequence (a multi-request batch): what the ring
    /// holds no longer describes one context.
    pub fn invalidate(&self) {
        if !self.is_enabled() {
            return;
        }
        let mut s = self.lock();
        s.cur = None;
        s.done.clear();
        s.reset = true;
    }

    /// Completed rows in forward order, and whether the ring must be cleared first.
    pub fn drain(&self) -> (bool, Vec<CapturedRow>) {
        let mut s = self.lock();
        let reset = std::mem::take(&mut s.reset);
        (reset, s.done.drain(..).collect())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CaptureState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Target embeddings and lm_head, shared with the draft (DFlash2 ships neither).
pub trait TargetHead: Send {
    fn embed_rows(&self, tokens: &[u32]) -> Result<Vec<f32>, AllpacaError>;
    /// The `k` largest logits (id, logit), largest first, of each output-normed
    /// hidden row in `hidden[rows, dim]`.
    fn topk_rows(
        &self,
        hidden: &[f32],
        rows: usize,
        k: usize,
    ) -> Result<Vec<Vec<(usize, f32)>>, AllpacaError>;
}

/// Logits per GPU top-k chunk; each chunk returns its own top `k`.
pub(crate) const TOPK_CHUNK: usize = 2048;
/// Largest `k` the head's candidate buffers hold.
pub(crate) const TOPK_MAX: usize = 32;

/// [`TargetHead`] over the target's resident weights on its own GPU.
pub struct ResidentHead {
    hip: Arc<HipWorkspace>,
    resident: Arc<crate::model::ResidentModel>,
    dim: usize,
    vocab: usize,
    rows: usize,
    a: DeviceWeights,
    out: DeviceWeights,
    cand_v: DeviceWeights,
    cand_i: DeviceWeights,
}

impl ResidentHead {
    pub fn new(
        hip: Arc<HipWorkspace>,
        resident: Arc<crate::model::ResidentModel>,
        dim: usize,
        vocab: usize,
        rows: usize,
    ) -> Result<Self, AllpacaError> {
        let a = DeviceWeights::alloc(hip.device, rows * dim * 4)?;
        let out = DeviceWeights::alloc(hip.device, rows * vocab.max(dim) * 4)?;
        let cands = rows * vocab.div_ceil(TOPK_CHUNK) * TOPK_MAX * 4;
        let cand_v = DeviceWeights::alloc(hip.device, cands)?;
        let cand_i = DeviceWeights::alloc(hip.device, cands)?;
        Ok(Self {
            hip,
            resident,
            dim,
            vocab,
            rows,
            a,
            out,
            cand_v,
            cand_i,
        })
    }

    fn check(&self, rows: usize) -> Result<(), AllpacaError> {
        if rows == 0 || rows > self.rows {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(format!(
                "DFlash2 head: {rows} rows (max {})",
                self.rows
            ))));
        }
        Ok(())
    }
}

impl TargetHead for ResidentHead {
    fn embed_rows(&self, tokens: &[u32]) -> Result<Vec<f32>, AllpacaError> {
        self.check(tokens.len())?;
        self.hip.bind()?;
        let out = self.out.ptr_at(0) as *mut f32;
        self.resident.embed_rows_device(&self.hip, tokens, out, self.dim)?;
        let mut host = vec![0.0f32; tokens.len() * self.dim];
        self.hip.copy_d2h_f32(out, &mut host)?;
        Ok(host)
    }

    fn topk_rows(
        &self,
        hidden: &[f32],
        rows: usize,
        k: usize,
    ) -> Result<Vec<Vec<(usize, f32)>>, AllpacaError> {
        self.check(rows)?;
        let k = k.clamp(1, TOPK_MAX);
        self.hip.bind()?;
        let a = self.a.ptr_at(0) as *mut f32;
        let out = self.out.ptr_at(0) as *mut f32;
        self.hip.copy_h2d_f32(a, &hidden[..rows * self.dim])?;
        self.resident
            .lm_head_rows_device(&self.hip, a, out, rows, self.vocab, self.dim)?;
        // Per-chunk top-k on the GPU; the host merges ~1k candidates a row
        // instead of copying back and scanning the whole vocab.
        let chunks = self.vocab.div_ceil(TOPK_CHUNK);
        allpaca_hip::topk_chunks_device(
            out,
            rows,
            self.vocab,
            TOPK_CHUNK,
            k,
            self.cand_v.ptr_at(0) as *mut f32,
            self.cand_i.ptr_at(0) as *mut i32,
        )?;
        let n = rows * chunks * k;
        let mut v = vec![0.0f32; n];
        let mut i = vec![0.0f32; n];
        self.hip.copy_d2h_f32(self.cand_v.ptr_at(0), &mut v)?;
        self.hip.copy_d2h_f32(self.cand_i.ptr_at(0), &mut i)?;
        Ok((0..rows)
            .map(|r| {
                let vr = &v[r * chunks * k..(r + 1) * chunks * k];
                let ir = &i[r * chunks * k..(r + 1) * chunks * k];
                top_k_desc(vr, k)
                    .into_iter()
                    .filter_map(|j| {
                        let id = ir[j].to_bits() as i32;
                        (id >= 0).then_some((id as usize, vr[j]))
                    })
                    .collect()
            })
            .collect())
    }
}

/// Two-tap dynamic convolution along each block (llama.cpp `build_dflash2_conv`);
/// the host reference for `dflash_conv_kernel`.
/// `h` is `[rows, dim]` with `rows` a multiple of `block`; `dynamic` is
/// `[rows, 2 * kernel * groups]`; `base` is `[2][kernel][dim]`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn dynamic_conv(
    h: &[f32],
    dynamic: &[f32],
    base: &[f32],
    side: usize,
    rows: usize,
    block: usize,
    dim: usize,
    kernel: usize,
    group: usize,
) -> Vec<f32> {
    let groups = dim / group;
    let proj = 2 * kernel * groups;
    let mut out = vec![0.0f32; rows * dim];
    for t in 0..rows {
        let in_block = t % block;
        let d = &dynamic[t * proj..(t + 1) * proj];
        for tap in 0..kernel.min(in_block + 1) {
            let src = &h[(t - tap) * dim..(t - tap + 1) * dim];
            let coeff = &d[groups * (tap + kernel * side)..groups * (tap + kernel * side + 1)];
            let b = &base[dim * (tap + kernel * side)..dim * (tap + kernel * side + 1)];
            let o = &mut out[t * dim..(t + 1) * dim];
            for c in 0..dim {
                o[c] += (coeff[c / group] + b[c]) * src[c];
            }
        }
    }
    out
}

fn debug_enabled() -> bool {
    std::env::var("ALLPACA_DFLASH2_DEBUG").is_ok_and(|v| v == "1")
}

/// Indices of the `k` largest values, largest first (ties keep the lower index).
pub fn top_k_desc(row: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(row.len());
    let mut best: Vec<usize> = Vec::with_capacity(k + 1);
    for (i, &v) in row.iter().enumerate() {
        if best.len() == k && v <= row[best[k - 1]] {
            continue;
        }
        let at = best.partition_point(|&j| row[j] >= v);
        best.insert(at, i);
        best.truncate(k);
    }
    best
}

/// Q8_0 table row (`rank` values) of token `tok` in a `[rank, vocab]` tensor.
fn q8_0_row(bytes: &[u8], tok: usize, rank: usize, out: &mut [f32]) -> bool {
    let blocks = rank / 32;
    let start = tok * blocks * 34;
    if rank % 32 != 0 || start + blocks * 34 > bytes.len() {
        return false;
    }
    allpaca_cpu::dequant_q8_0(&bytes[start..start + blocks * 34], rank, out).is_ok()
}

/// Host copy of a `[rank, vocab]` selector table, gathered one token row at a time.
struct SelectorTable {
    kind: GgmlType,
    bytes: Vec<u8>,
    rank: usize,
}

impl SelectorTable {
    fn load(file: &GgufFile, name: &str, rank: usize) -> Result<Self, AllpacaError> {
        let info = file.tensor(name).ok_or_else(|| missing(name))?;
        let kind = info.ggml_type;
        let bytes = match kind {
            GgmlType::Q8_0 | GgmlType::F32 => file.tensor_bytes_of(info)?.to_vec(),
            _ => {
                // Any other type: materialize once as F32.
                let f = file.tensor_f32(name)?;
                return Ok(Self {
                    kind: GgmlType::F32,
                    bytes: f.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    rank,
                });
            }
        };
        Ok(Self { kind, bytes, rank })
    }

    fn row(&self, tok: u32, out: &mut [f32]) -> bool {
        let tok = tok as usize;
        match self.kind {
            GgmlType::Q8_0 => q8_0_row(&self.bytes, tok, self.rank, out),
            _ => {
                let start = tok * self.rank * 4;
                if start + self.rank * 4 > self.bytes.len() {
                    return false;
                }
                for (r, o) in out.iter_mut().take(self.rank).enumerate() {
                    let b = &self.bytes[start + r * 4..start + r * 4 + 4];
                    *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
                true
            }
        }
    }
}

/// Greedy walk through the candidate lattice (llama.cpp `draft()` for DFlash2).
/// `cands[i]` / `unary[i]` are block position `i + 1`'s top-k ids and logits,
/// `gates[i]` its selector gate (`rank`).
fn walk_selector(
    anchor: u32,
    cands: &[Vec<usize>],
    unary: &[Vec<f32>],
    gates: &[Vec<f32>],
    prev: &SelectorTable,
    next: &SelectorTable,
    rank: usize,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(cands.len());
    let mut pred_tok = anchor;
    let mut p = vec![0.0f32; rank];
    let mut s = vec![0.0f32; rank];
    for i in 0..cands.len() {
        if !prev.row(pred_tok, &mut p) {
            break;
        }
        for (pv, g) in p.iter_mut().zip(&gates[i]) {
            *pv *= g;
        }
        let mut best = (f32::NEG_INFINITY, 0usize);
        for (j, &c) in cands[i].iter().enumerate() {
            if !next.row(c as u32, &mut s) {
                continue;
            }
            let score = s.iter().zip(&p).map(|(a, b)| a * b).sum::<f32>() + unary[i][j];
            if score > best.0 {
                best = (score, j);
            }
        }
        if best.0 == f32::NEG_INFINITY {
            break;
        }
        pred_tok = cands[i][best.1] as u32;
        out.push(pred_tok);
    }
    out
}

/// Ring slot bookkeeping: which target position each K/V slot holds, and the
/// token committed there.
#[derive(Debug, Clone, Default)]
struct RingIndex {
    slot_pos: Vec<Option<usize>>,
    tokens: Vec<u32>,
}

impl RingIndex {
    fn new(slots: usize) -> Self {
        Self {
            slot_pos: vec![None; slots],
            tokens: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.slot_pos.iter_mut().for_each(|s| *s = None);
        self.tokens.clear();
    }

    fn slot_of(&self, pos: usize) -> usize {
        pos % self.slot_pos.len()
    }

    fn put(&mut self, pos: usize, token: u32) -> usize {
        let slot = self.slot_of(pos);
        self.slot_pos[slot] = Some(pos);
        if self.tokens.len() <= pos {
            self.tokens.resize(pos + 1, u32::MAX);
        }
        self.tokens[pos] = token;
        slot
    }

    /// Drops every slot from the first position where the ring disagrees
    /// with `ctx` (a new request, a rejected verify row, or a deferred
    /// correction the target has not forwarded yet). The anchor `ctx.last()`
    /// usually has been forwarded; its row stays for the next step.
    fn sync_prefix(&mut self, ctx: &[u32]) -> usize {
        let n = ctx.len().min(self.tokens.len());
        let agree = (0..n).take_while(|&p| self.tokens[p] == ctx[p]).count();
        if agree < self.tokens.len() {
            self.tokens.truncate(agree);
            for s in self.slot_pos.iter_mut() {
                if s.is_some_and(|p| p >= agree) {
                    *s = None;
                }
            }
        }
        agree
    }

    /// Slots `0..used()` hold everything written so far.
    fn used(&self) -> usize {
        self.slot_pos.iter().rposition(Option::is_some).map_or(0, |s| s + 1)
    }

    /// `[block, block + used()]` attention mask for queries at `anchor..`:
    /// the block itself, then context slots before the anchor inside the window.
    fn mask(&self, anchor: usize, block: usize, window: usize) -> Vec<u8> {
        let used = self.used();
        let n_kv = block + used;
        let mut m = vec![0u8; block * n_kv];
        for i in 0..block {
            let q = anchor + i;
            let row = &mut m[i * n_kv..(i + 1) * n_kv];
            row[..block].iter_mut().for_each(|v| *v = 1);
            for (s, p) in self.slot_pos[..used].iter().enumerate() {
                if let Some(p) = *p {
                    if p < anchor && q - p < window {
                        row[block + s] = 1;
                    }
                }
            }
        }
        m
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct LayerOff {
    attn_norm: usize,
    attn_conv_base: usize,
    attn_conv_proj: usize,
    wq: usize,
    wk: usize,
    wv: usize,
    q_norm: usize,
    k_norm: usize,
    wo: usize,
    ffn_norm: usize,
    ffn_conv_base: usize,
    ffn_conv_proj: usize,
    w_gate: usize,
    w_up: usize,
    w_down: usize,
}

/// Device scratch: one allocation, carved into named views.
struct Scratch {
    _buf: DeviceWeights,
    x: *mut f32,
    nn: *mut f32,
    dynamic: *mut f32,
    q: *mut f32,
    attn: *mut f32,
    o: *mut f32,
    gate: *mut f32,
    up: *mut f32,
    sel: *mut f32,
    feat: *mut f32,
    g: *mut f32,
    k: *mut f32,
    v: *mut f32,
    tmp: *mut f32,
}

// SAFETY: raw pointers into `_buf`, which lives as long as `Scratch`; the
// provider is only driven from one thread at a time.
unsafe impl Send for Scratch {}

impl Scratch {
    fn new(device: i32, spec: &Dflash2Spec) -> Result<Self, AllpacaError> {
        let b = spec.block_size;
        let r = INJECT_CHUNK;
        let sizes = [
            b * spec.dim,
            b * spec.dim,
            b * spec.conv_proj(),
            b * spec.q_dim(),
            b * spec.q_dim(),
            b * spec.dim,
            b * spec.ff,
            b * spec.ff,
            b * spec.selector_rank,
            r * spec.feature_dim(),
            r * spec.dim,
            r * spec.kv_dim(),
            r * spec.kv_dim(),
            b * spec.dim,
        ];
        let align = |n: usize| n.div_ceil(64) * 64;
        let total: usize = sizes.iter().map(|&n| align(n)).sum();
        let buf = DeviceWeights::alloc(device, total * 4)?;
        let mut off = 0usize;
        let mut views = [std::ptr::null_mut::<f32>(); 14];
        for (v, &n) in views.iter_mut().zip(&sizes) {
            *v = buf.ptr_at(off * 4) as *mut f32;
            off += align(n);
        }
        let [x, nn, dynamic, q, attn, o, gate, up, sel, feat, g, k, v, tmp] = views;
        Ok(Self {
            _buf: buf,
            x,
            nn,
            dynamic,
            q,
            attn,
            o,
            gate,
            up,
            sel,
            feat,
            g,
            k,
            v,
            tmp,
        })
    }
}

pub struct Dflash2Provider {
    spec: Dflash2Spec,
    hip: Arc<HipWorkspace>,
    weights: DeviceWeights,
    fc: usize,
    enc_norm: usize,
    out_norm: usize,
    sel_hidden: usize,
    layers: Vec<LayerOff>,
    /// The selector's tables (DFlash2 only).
    sel: Option<(SelectorTable, SelectorTable)>,
    ring_k: Vec<DeviceWeights>,
    ring_v: Vec<DeviceWeights>,
    ring: RingIndex,
    scratch: Scratch,
    capture: Arc<LayerCapture>,
    head: Box<dyn TargetHead>,
    pub stats: Dflash2Stats,
}

#[derive(Debug, Clone, Default)]
pub struct Dflash2Stats {
    pub injected_rows: u64,
    pub drafts: u64,
    pub draft_us: u64,
    pub ctx_slots: u64,
}

fn missing(name: &str) -> AllpacaError {
    AllpacaError::Fatal(FatalError::ModelLoad(format!("DFlash2 tensor {name} missing")))
}

impl Dflash2Provider {
    pub fn load(
        file: &GgufFile,
        hip: Arc<HipWorkspace>,
        capture: Arc<LayerCapture>,
        head: Box<dyn TargetHead>,
    ) -> Result<Self, AllpacaError> {
        let spec = Dflash2Spec::from_gguf(file).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad("not a DFlash2 draft GGUF".into()))
        })?;
        let device = hip.device;
        // Pack every device tensor into one buffer, 256-byte aligned.
        let mut parts: Vec<Vec<u8>> = Vec::new();
        let mut off = 0usize;
        let mut push = |bytes: Vec<u8>| -> usize {
            let at = off;
            let pad = bytes.len().div_ceil(256) * 256 - bytes.len();
            off += bytes.len() + pad;
            let mut b = bytes;
            b.extend(std::iter::repeat_n(0u8, pad));
            parts.push(b);
            at
        };
        let q8 = |name: &str| -> Result<Vec<u8>, AllpacaError> {
            let info = file.tensor(name).ok_or_else(|| missing(name))?;
            if info.ggml_type != GgmlType::Q8_0 {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(format!(
                    "DFlash2 {name} is {:?}; only the Q8_0 draft is supported",
                    info.ggml_type
                ))));
            }
            // Device Q8_0 is the 36-byte staged layout (F32 scale, aligned qs).
            allpaca_hip::stage_q8_0_blocks(file.tensor_bytes_of(info)?)
        };
        let f32b = |name: &str| -> Result<Vec<u8>, AllpacaError> {
            Ok(file
                .tensor_f32(name)?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        };
        let fc = push(q8("fc.weight")?);
        let enc_norm = push(f32b("enc.output_norm.weight")?);
        let out_norm = push(f32b("output_norm.weight")?);
        let plain = spec.plain;
        let sel_hidden = if plain { 0 } else { push(q8("selector_hidden.weight")?) };
        let mut layers = Vec::with_capacity(spec.n_layers);
        for i in 0..spec.n_layers {
            let t = |s: &str| format!("blk.{i}.{s}");
            let attn_norm = push(f32b(&t("attn_norm.weight"))?);
            let (attn_conv_base, attn_conv_proj) = if plain {
                (0, 0)
            } else {
                (push(f32b(&t("attn_conv_base"))?), push(q8(&t("attn_conv_proj.weight"))?))
            };
            layers.push(LayerOff {
                attn_norm,
                attn_conv_base,
                attn_conv_proj,
                wq: push(q8(&t("attn_q.weight"))?),
                wk: push(q8(&t("attn_k.weight"))?),
                wv: push(q8(&t("attn_v.weight"))?),
                q_norm: push(f32b(&t("attn_q_norm.weight"))?),
                k_norm: push(f32b(&t("attn_k_norm.weight"))?),
                wo: push(q8(&t("attn_output.weight"))?),
                ffn_norm: push(f32b(&t("ffn_norm.weight"))?),
                ffn_conv_base: if plain { 0 } else { push(f32b(&t("ffn_conv_base"))?) },
                ffn_conv_proj: if plain { 0 } else { push(q8(&t("ffn_conv_proj.weight"))?) },
                w_gate: push(q8(&t("ffn_gate.weight"))?),
                w_up: push(q8(&t("ffn_up.weight"))?),
                w_down: push(q8(&t("ffn_down.weight"))?),
            });
        }
        let refs: Vec<&[u8]> = parts.iter().map(|p| p.as_slice()).collect();
        let weights = DeviceWeights::upload_bytes(device, &refs)?;
        drop(parts);
        let rank = spec.selector_rank;
        let sel = if plain {
            None
        } else {
            Some((
                SelectorTable::load(file, "selector_predecessor.weight", rank)?,
                SelectorTable::load(file, "selector_successor.weight", rank)?,
            ))
        };
        let slots = spec.ring_slots();
        let ring_bytes = (slots + spec.block_size) * spec.kv_dim() * 4;
        let mut ring_k = Vec::with_capacity(spec.n_layers);
        let mut ring_v = Vec::with_capacity(spec.n_layers);
        // Zeroed: masked-out slots still enter the weighted V sum as 0 * v.
        let zeros = vec![0u8; ring_bytes];
        for _ in 0..spec.n_layers {
            ring_k.push(DeviceWeights::upload_bytes(device, &[&zeros])?);
            ring_v.push(DeviceWeights::upload_bytes(device, &[&zeros])?);
        }
        let scratch = Scratch::new(device, &spec)?;
        capture.enable(spec.target_layers.clone(), spec.dim, slots);
        tracing::info!(
            device,
            weight_mib = weights.bytes() >> 20,
            layers = spec.n_layers,
            block = spec.block_size,
            window = spec.window,
            plain = spec.plain,
            target_layers = ?spec.target_layers,
            "DFlash draft loaded"
        );
        Ok(Self {
            ring: RingIndex::new(slots),
            spec,
            hip,
            weights,
            fc,
            enc_norm,
            out_norm,
            sel_hidden,
            layers,
            sel,
            ring_k,
            ring_v,
            scratch,
            capture,
            head,
            stats: Dflash2Stats::default(),
        })
    }

    pub fn spec(&self) -> &Dflash2Spec {
        &self.spec
    }

    /// HIP device the draft runs on.
    pub fn device(&self) -> i32 {
        self.hip.device
    }

    fn w(&self, off: usize) -> *const f32 {
        self.weights.ptr_at(off)
    }

    fn wb(&self, off: usize) -> *const u8 {
        self.weights.byte_ptr_at(off)
    }

    /// Ring buffer row `row`: rows `0..block` are the draft block's own K/V,
    /// context slot `s` is row `block + s`.
    fn ring_row(buf: &DeviceWeights, row: usize, kv_dim: usize) -> *mut f32 {
        buf.ptr_at(row * kv_dim * 4) as *mut f32
    }

    /// Fuse captured target rows into every layer's K/V ring.
    fn inject(&mut self) -> Result<(), AllpacaError> {
        let (reset, rows) = self.capture.drain();
        if reset {
            self.ring.clear();
        }
        if rows.is_empty() {
            return Ok(());
        }
        let sp = self.spec.clone();
        let hip = Arc::clone(&self.hip);
        hip.bind()?;
        let s = &self.scratch;
        let fdim = sp.feature_dim();
        let kv = sp.kv_dim();
        let mut i = 0;
        while i < rows.len() {
            // A run of consecutive positions, at most one chunk.
            let mut n = 1;
            while i + n < rows.len()
                && n < INJECT_CHUNK
                && rows[i + n].pos == rows[i + n - 1].pos + 1
            {
                n += 1;
            }
            let run = &rows[i..i + n];
            let mut host = Vec::with_capacity(n * fdim);
            for r in run {
                host.extend_from_slice(&r.feat);
            }
            hip.copy_h2d_f32(s.feat, &host)?;
            hip.gemm_nt_q8_dev(s.feat, self.wb(self.fc), s.g, n as i32, sp.dim as i32, fdim as i32)?;
            hip.rmsnorm_device_rows(s.g, self.w(self.enc_norm), s.g, n as i32, sp.dim as i32, sp.eps)?;
            for (l, lo) in self.layers.iter().enumerate() {
                hip.gemm_nt_q8_dev(s.g, self.wb(lo.wk), s.k, n as i32, kv as i32, sp.dim as i32)?;
                hip.gemm_nt_q8_dev(s.g, self.wb(lo.wv), s.v, n as i32, kv as i32, sp.dim as i32)?;
                hip.rmsnorm_heads_device(
                    s.k,
                    self.w(lo.k_norm),
                    (n * sp.n_kv_heads) as i32,
                    sp.head_dim as i32,
                    sp.eps,
                )?;
                allpaca_hip::rope_neox_rows_device(
                    s.k,
                    n,
                    sp.n_kv_heads,
                    sp.head_dim,
                    run[0].pos,
                    sp.theta,
                )?;
                // Copy into the ring, splitting where the slot index wraps.
                let mut j = 0;
                while j < n {
                    let slot = self.ring.slot_of(run[j].pos);
                    let len = (n - j).min(self.ring.slot_pos.len() - slot);
                    // SAFETY: `k`/`v` hold `n * kv` floats.
                    let (ks, vs) = unsafe { (s.k.add(j * kv), s.v.add(j * kv)) };
                    let row = sp.block_size + slot;
                    hip.copy_d2d_f32(Self::ring_row(&self.ring_k[l], row, kv), ks, len * kv)?;
                    hip.copy_d2d_f32(Self::ring_row(&self.ring_v[l], row, kv), vs, len * kv)?;
                    j += len;
                }
            }
            for r in run {
                self.ring.put(r.pos, r.token);
            }
            self.stats.injected_rows += n as u64;
            i += n;
        }
        Ok(())
    }

    /// In-place dynamic conv of the block rows at `dev`.
    fn conv(&self, dev: *mut f32, base: usize, side: usize) -> Result<(), AllpacaError> {
        let sp = &self.spec;
        let b = sp.block_size;
        let s = &self.scratch;
        allpaca_hip::dflash_conv_device(
            dev,
            s.dynamic,
            self.w(base),
            s.tmp,
            b,
            b,
            sp.dim,
            sp.conv_kernel,
            sp.conv_group,
            side,
        )?;
        self.hip.copy_d2d_f32(dev, s.tmp, b * sp.dim)
    }

    /// Debug: NaN count and a few values of a device buffer.
    fn probe(&self, name: &str, ptr: *const f32, n: usize) {
        if !std::env::var("ALLPACA_DFLASH2_PROBE").is_ok_and(|v| v == "1") {
            return;
        }
        let mut h = vec![0.0f32; n];
        if self.hip.copy_d2h_f32(ptr, &mut h).is_ok() {
            let nan = h.iter().filter(|v| !v.is_finite()).count();
            let amax = h.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            eprintln!("DFLASH2 probe {name}: nonfinite={nan}/{n} amax={amax} head={:?}", &h[..4.min(n)]);
        }
    }

    /// One block pass; returns up to `block - 1` draft tokens after `anchor`.
    fn draft_block(&mut self, anchor: u32, anchor_pos: usize) -> Result<Vec<u32>, AllpacaError> {
        let sp = self.spec.clone();
        let hip = Arc::clone(&self.hip);
        let b = sp.block_size;
        let dim = sp.dim;
        let kv = sp.kv_dim();
        let n_kv = b + self.ring.used();
        let mut block_tokens = vec![sp.mask_token; b];
        block_tokens[0] = anchor;
        let t0 = std::time::Instant::now();
        let mut x0 = self.head.embed_rows(&block_tokens)?;
        if sp.embedding_scale != 1.0 {
            x0.iter_mut().for_each(|v| *v *= sp.embedding_scale);
        }
        let t_embed = t0.elapsed();
        hip.bind()?;
        let s = &self.scratch;
        hip.copy_h2d_f32(s.x, &x0)?;
        self.probe("x0", s.x, b * dim);
        let mask = self.ring.mask(anchor_pos, b, sp.window);
        self.stats.ctx_slots = mask[b..n_kv].iter().map(|&v| u64::from(v)).sum();
        let proj = sp.conv_proj();
        let (m, d) = (b as i32, dim as i32);
        for (l, lo) in self.layers.iter().enumerate() {
            hip.rmsnorm_device_rows(s.x, self.w(lo.attn_norm), s.nn, m, d, sp.eps)?;
            if !sp.plain {
                hip.gemm_nt_q8_dev(s.nn, self.wb(lo.attn_conv_proj), s.dynamic, m, proj as i32, d)?;
                if l == 0 {
                    self.probe("dyn", s.dynamic, b * proj);
                }
                self.conv(s.nn, lo.attn_conv_base, 0)?;
                if l == 0 {
                    self.probe("conv_in", s.nn, b * dim);
                }
            }
            let kb = Self::ring_row(&self.ring_k[l], 0, kv);
            let vb = Self::ring_row(&self.ring_v[l], 0, kv);
            hip.gemm_nt_q8_dev(s.nn, self.wb(lo.wq), s.q, m, sp.q_dim() as i32, d)?;
            hip.gemm_nt_q8_dev(s.nn, self.wb(lo.wk), kb, m, kv as i32, d)?;
            hip.gemm_nt_q8_dev(s.nn, self.wb(lo.wv), vb, m, kv as i32, d)?;
            hip.rmsnorm_heads_device(
                s.q,
                self.w(lo.q_norm),
                (b * sp.n_heads) as i32,
                sp.head_dim as i32,
                sp.eps,
            )?;
            hip.rmsnorm_heads_device(
                kb,
                self.w(lo.k_norm),
                (b * sp.n_kv_heads) as i32,
                sp.head_dim as i32,
                sp.eps,
            )?;
            // Qwen-family NeoX pairs (llama.cpp rope type NEOX for the DFlash arch).
            allpaca_hip::rope_neox_rows_device(s.q, b, sp.n_heads, sp.head_dim, anchor_pos, sp.theta)?;
            allpaca_hip::rope_neox_rows_device(kb, b, sp.n_kv_heads, sp.head_dim, anchor_pos, sp.theta)?;
            hip.attention_device_masked_upload(
                s.q,
                self.ring_k[l].ptr_at(0),
                self.ring_v[l].ptr_at(0),
                s.attn,
                m,
                n_kv as i32,
                sp.n_heads as i32,
                sp.n_kv_heads as i32,
                sp.head_dim as i32,
                &mask,
            )?;
            if l == 0 {
                self.probe("q", s.q, b * sp.q_dim());
                self.probe("kb", kb, b * kv);
                self.probe("ctx_k", self.ring_k[l].ptr_at(0), 4 * kv);
                self.probe("attn", s.attn, b * sp.q_dim());
            }
            hip.gemm_nt_q8_dev(s.attn, self.wb(lo.wo), s.o, m, d, sp.q_dim() as i32)?;
            if !sp.plain {
                self.conv(s.o, lo.attn_conv_base, 1)?;
            }
            hip.residual_add(s.x, s.o, (b * dim) as i32)?;

            hip.rmsnorm_device_rows(s.x, self.w(lo.ffn_norm), s.nn, m, d, sp.eps)?;
            if !sp.plain {
                hip.gemm_nt_q8_dev(s.nn, self.wb(lo.ffn_conv_proj), s.dynamic, m, proj as i32, d)?;
                self.conv(s.nn, lo.ffn_conv_base, 0)?;
            }
            hip.gemm_nt_q8_dev(s.nn, self.wb(lo.w_gate), s.gate, m, sp.ff as i32, d)?;
            hip.gemm_nt_q8_dev(s.nn, self.wb(lo.w_up), s.up, m, sp.ff as i32, d)?;
            hip.silu_mul_device(s.gate, s.up, (b * sp.ff) as i32)?;
            hip.gemm_nt_q8_dev(s.gate, self.wb(lo.w_down), s.o, m, d, sp.ff as i32)?;
            if !sp.plain {
                self.conv(s.o, lo.ffn_conv_base, 1)?;
            }
            hip.residual_add(s.x, s.o, (b * dim) as i32)?;
            self.probe("layer_out", s.x, b * dim);
        }
        hip.rmsnorm_device_rows(s.x, self.w(self.out_norm), s.nn, m, d, sp.eps)?;
        let Some((sel_prev, sel_next)) = self.sel.as_ref() else {
            // Plain DFlash: the head's top-1 at positions 1.. (greedy).
            let mut hidden = vec![0.0f32; b * dim];
            hip.copy_d2h_f32(s.nn, &mut hidden)?;
            let top = self.head.topk_rows(&hidden[dim..], b - 1, 1)?;
            return Ok(top.into_iter().filter_map(|row| row.first().map(|&(id, _)| id as u32)).collect());
        };
        let rank = sp.selector_rank;
        hip.gemm_nt_q8_dev(s.nn, self.wb(self.sel_hidden), s.sel, m, rank as i32, d)?;
        let mut hidden = vec![0.0f32; b * dim];
        hip.copy_d2h_f32(s.nn, &mut hidden)?;
        let t_layers = t0.elapsed();
        let mut gate = vec![0.0f32; b * rank];
        hip.copy_d2h_f32(s.sel, &mut gate)?;
        // Positions 1.. only: the anchor's logits are the target's job.
        let top = self.head.topk_rows(&hidden[dim..], b - 1, sp.top_k)?;
        let t_head = t0.elapsed();
        let mut cands = Vec::with_capacity(b - 1);
        let mut unary = Vec::with_capacity(b - 1);
        let mut gates = Vec::with_capacity(b - 1);
        for (i, row) in top.into_iter().enumerate() {
            cands.push(row.iter().map(|&(id, _)| id).collect::<Vec<usize>>());
            unary.push(row.iter().map(|&(_, v)| v).collect::<Vec<f32>>());
            gates.push(gate[(i + 1) * rank..(i + 2) * rank].to_vec());
        }
        let t_topk = t0.elapsed();
        if debug_enabled() {
            eprintln!(
                "DFLASH2 timing embed={:?} layers={:?} head={:?} topk={:?}",
                t_embed, t_layers, t_head, t_topk
            );
            eprintln!(
                "DFLASH2 anchor={anchor} pos={anchor_pos} hidden0={:?} gate0={:?} cands1={:?} unary1={:?}",
                &hidden[dim..dim + 4],
                &gate[rank..rank + 4],
                &cands[0][..cands[0].len().min(4)],
                &unary[0][..unary[0].len().min(4)]
            );
        }
        Ok(walk_selector(
            anchor,
            &cands,
            &unary,
            &gates,
            sel_prev,
            sel_next,
            rank,
        ))
    }
}

impl DraftProvider for Dflash2Provider {
    fn name(&self) -> &'static str {
        if self.spec.plain {
            "dflash"
        } else {
            "dflash2"
        }
    }

    fn observe(&mut self, _tokens: &[u32]) {}

    fn propose(
        &mut self,
        ctx: &[u32],
        max_tokens: usize,
        _temperature: f32,
    ) -> Result<Vec<u32>, AllpacaError> {
        let Some(&anchor) = ctx.last() else {
            return Ok(Vec::new());
        };
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        let started = std::time::Instant::now();
        self.inject()?;
        let before = self.ring.tokens.len();
        let ring_before = self.ring.tokens[ctx.len().saturating_sub(4).min(before)..].to_vec();
        let agree = self.ring.sync_prefix(ctx);
        if debug_enabled() {
            let n = ctx.len();
            let lo = n.saturating_sub(4);
            eprintln!(
                "DFLASH2 ctx_len={n} ring_tokens={before} agree={agree} ctx_tail={:?} ring_before={ring_before:?} ring_tail={:?} slots={:?}",
                &ctx[lo..],
                &self.ring.tokens[lo.min(self.ring.tokens.len())..],
                self.ring
                    .slot_pos
                    .iter()
                    .flatten()
                    .filter(|&&p| p + 6 >= n)
                    .collect::<Vec<_>>()
            );
        }
        let mut out = self.draft_block(anchor, ctx.len() - 1)?;
        out.truncate(max_tokens);
        self.stats.drafts += 1;
        self.stats.draft_us += started.elapsed().as_micros() as u64;
        if self.stats.drafts % 32 == 1 {
            tracing::info!(
                drafts = self.stats.drafts,
                injected_rows = self.stats.injected_rows,
                ctx_slots = self.stats.ctx_slots,
                anchor_pos = ctx.len() - 1,
                mean_draft_ms = self.stats.draft_us as f64 / self.stats.drafts as f64 / 1e3,
                last = ?out,
                "DFlash2 draft"
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_conv_mixes_the_previous_row_within_a_block_only() {
        // dim 4, group 2 -> 2 groups; kernel 2; two blocks of 2 rows.
        let (dim, group, kernel, block, rows) = (4, 2, 2, 2, 4);
        let groups = dim / group;
        let proj = 2 * kernel * groups;
        let h: Vec<f32> = (0..rows * dim).map(|i| (i + 1) as f32).collect();
        // Side 1 coefficients: tap0 = 1 per group, tap1 = 10 per group; side 0 zero.
        let mut dynamic = vec![0.0f32; rows * proj];
        for t in 0..rows {
            for g in 0..groups {
                dynamic[t * proj + g + groups * (0 + kernel)] = 1.0;
                dynamic[t * proj + g + groups * (1 + kernel)] = 10.0;
            }
        }
        // Base adds 0.5 on side 1 tap 0 for channel 3 only.
        let mut base = vec![0.0f32; 2 * kernel * dim];
        base[3 + dim * kernel] = 0.5;
        let out = dynamic_conv(&h, &dynamic, &base, 1, rows, block, dim, kernel, group);
        // Row 0 starts a block: tap 0 only.
        assert_eq!(&out[0..4], &[1.0, 2.0, 3.0, 4.0 * 1.5]);
        // Row 1: h1 + 10 * h0.
        assert_eq!(&out[4..8], &[5.0 + 10.0, 6.0 + 20.0, 7.0 + 30.0, 8.0 * 1.5 + 40.0]);
        // Row 2 starts the second block: no reach back into row 1.
        assert_eq!(&out[8..12], &[9.0, 10.0, 11.0, 12.0 * 1.5]);
        // Side 0 has zero dynamic coefficients and zero base.
        let zero = dynamic_conv(&h, &dynamic, &base, 0, rows, block, dim, kernel, group);
        assert!(zero.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn dflash_conv_kernel_matches_the_host_reference() {
        let Ok(hip) = HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        let (dim, group, kernel, block, rows) = (64, 16, 2, 8, 16);
        let proj = 2 * kernel * (dim / group);
        let val = |i: usize, salt: usize| (((i * 7919 + salt) % 97) as f32 - 48.0) / 17.0;
        let h: Vec<f32> = (0..rows * dim).map(|i| val(i, 1)).collect();
        let dynamic: Vec<f32> = (0..rows * proj).map(|i| val(i, 2)).collect();
        let base: Vec<f32> = (0..2 * kernel * dim).map(|i| val(i, 3)).collect();
        let up = |v: &[f32]| {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            DeviceWeights::upload_bytes(hip.device, &[&bytes]).unwrap()
        };
        let (hd, dd, bd, od) = (up(&h), up(&dynamic), up(&base), up(&vec![0.0; rows * dim]));
        for side in 0..2 {
            allpaca_hip::dflash_conv_device(
                hd.ptr_at(0),
                dd.ptr_at(0),
                bd.ptr_at(0),
                od.ptr_at(0) as *mut f32,
                rows,
                block,
                dim,
                kernel,
                group,
                side,
            )
            .unwrap();
            let mut got = vec![0.0f32; rows * dim];
            hip.copy_d2h_f32(od.ptr_at(0), &mut got).unwrap();
            let want = dynamic_conv(&h, &dynamic, &base, side, rows, block, dim, kernel, group);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() <= 1e-5 * w.abs().max(1.0), "side {side}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn topk_chunks_kernel_merges_to_the_host_top_k() {
        let Ok(hip) = HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        // 3 rows of 5000 (a ragged last chunk), with repeated values for ties.
        let (rows, n, chunk, k) = (3usize, 5000usize, 2048usize, 16usize);
        let x: Vec<f32> = (0..rows * n)
            .map(|i| ((i * 2654435761usize) % 1009) as f32 / 7.0)
            .collect();
        let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let xd = DeviceWeights::upload_bytes(hip.device, &[&bytes]).unwrap();
        let chunks = n.div_ceil(chunk);
        let vd = DeviceWeights::alloc(hip.device, rows * chunks * k * 4).unwrap();
        let id = DeviceWeights::alloc(hip.device, rows * chunks * k * 4).unwrap();
        allpaca_hip::topk_chunks_device(
            xd.ptr_at(0),
            rows,
            n,
            chunk,
            k,
            vd.ptr_at(0) as *mut f32,
            id.ptr_at(0) as *mut i32,
        )
        .unwrap();
        let mut v = vec![0.0f32; rows * chunks * k];
        let mut ib = vec![0.0f32; rows * chunks * k];
        hip.copy_d2h_f32(vd.ptr_at(0), &mut v).unwrap();
        hip.copy_d2h_f32(id.ptr_at(0), &mut ib).unwrap();
        for r in 0..rows {
            let vr = &v[r * chunks * k..(r + 1) * chunks * k];
            let ir = &ib[r * chunks * k..(r + 1) * chunks * k];
            let got: Vec<usize> = top_k_desc(vr, k)
                .into_iter()
                .map(|j| ir[j].to_bits() as i32 as usize)
                .collect();
            assert_eq!(got, top_k_desc(&x[r * n..(r + 1) * n], k), "row {r}");
        }
    }

    /// A metadata-only draft GGUF: `tensors` named (shape [1]), dim 64.
    fn sidecar_file(tag: &str, tensors: &[&str], vocab: usize, dim: u32) -> GgufFile {
        use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
        let path = std::env::temp_dir().join(format!("allpaca_sidecar_{tag}_{}.gguf", std::process::id()));
        let mut meta = std::collections::HashMap::new();
        meta.insert("general.architecture".into(), GgufValue::String("dflash".into()));
        let u = |v: u32| GgufValue::Uint32(v);
        for (k, v) in [
            ("block_count", u(1)),
            ("embedding_length", u(dim)),
            ("feed_forward_length", u(128)),
            ("attention.head_count", u(4)),
            ("attention.head_count_kv", u(2)),
            ("block_size", u(8)),
        ] {
            meta.insert(format!("dflash.{k}"), v);
        }
        meta.insert("dflash.target_layers".into(), GgufValue::Array(vec![u(1), u(2)]));
        meta.insert("tokenizer.ggml.mask_token_id".into(), u(4));
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array((0..vocab).map(|i| GgufValue::String(format!("t{i}"))).collect()),
        );
        let one = [0.0f32];
        let ts: Vec<GgufWriteTensor<'_>> = tensors
            .iter()
            .map(|n| GgufWriteTensor { name: (*n).into(), dims: vec![1], data: &one })
            .collect();
        write_gguf(&path, &meta, &ts).unwrap();
        GgufFile::open(&path).unwrap()
    }

    #[test]
    fn nominated_sidecars_are_classified_by_content() {
        let plain = sidecar_file("plain", &["fc.weight", "blk.0.attn_q.weight"], 32, 64);
        let dflash2 = sidecar_file("d2", &["fc.weight", "selector_hidden.weight", "blk.0.attn_q.weight"], 32, 64);
        let dspark = sidecar_file("dspark", &["fc.weight", "markov_w1.weight", "blk.0.attn_q.weight"], 32, 64);
        // A DeepSeek DSpark (hyper-connections) named like a DFlash file.
        let dsv4 = sidecar_file("dsv4_named_dflash", &["fc.weight", "hc_head_fn.weight", "blk.0.attn_q.weight"], 32, 64);
        assert_eq!(classify_sidecar(&plain, 64, 32), SidecarKind::BlockDraft { plain: true });
        assert_eq!(classify_sidecar(&dflash2, 64, 32), SidecarKind::BlockDraft { plain: false });
        assert_eq!(classify_sidecar(&dspark, 64, 32), SidecarKind::Dspark);
        assert_eq!(classify_sidecar(&dsv4, 64, 32), SidecarKind::Dspark);
        assert_eq!(classify_sidecar(&dspark, 128, 32), SidecarKind::Mismatch);
        assert_eq!(classify_sidecar(&dsv4, 64, 33), SidecarKind::Mismatch);
    }

    #[test]
    fn block_draft_attach_checks_provider_shape_and_vocab() {
        use allpaca_types::DraftProviderKind as K;
        let plain = Dflash2Spec::from_gguf(&sidecar_file("chk_plain", &["fc.weight", "blk.0.attn_q.weight"], 32, 64)).unwrap();
        let d2 = Dflash2Spec::from_gguf(&sidecar_file("chk_d2", &["fc.weight", "selector_hidden.weight", "blk.0.attn_q.weight"], 32, 64)).unwrap();
        assert!(check_block_draft(&plain, &[K::Dflash1], 64, 32).is_ok());
        assert!(check_block_draft(&plain, &[K::Dflash2], 64, 32).is_err(), "a plain draft needs dflash1");
        assert!(check_block_draft(&d2, &[K::Dflash1], 64, 32).is_err(), "a DFlash2 draft needs dflash2");
        // Listing both accepts either kind.
        assert!(check_block_draft(&d2, &[K::Dflash1, K::Dflash2], 64, 32).is_ok());
        assert!(check_block_draft(&plain, &[K::Dflash1, K::Dflash2], 64, 32).is_ok());
        assert!(check_block_draft(&plain, &[K::Dflash1], 128, 32).is_err(), "width");
        assert!(check_block_draft(&plain, &[K::Dflash1], 64, 40).is_err(), "vocabulary");
        let mut tiny = plain.clone();
        tiny.vocab = None;
        assert!(check_block_draft(&tiny, &[K::Dflash1], 64, 4).is_err(), "mask token 4 outside 4 tokens");
    }

    #[test]
    fn top_k_desc_orders_largest_first() {
        let row = [0.1, 3.0, -1.0, 3.0, 2.0, 7.0];
        assert_eq!(top_k_desc(&row, 3), vec![5, 1, 3]);
        assert_eq!(top_k_desc(&row, 10).len(), row.len());
    }

    #[test]
    fn capture_keeps_rows_only_when_every_layer_reported() {
        let cap = LayerCapture::default();
        assert_eq!(cap.slot(3), None, "disabled capture taps nothing");
        cap.enable(vec![3, 7], 2, 16);
        assert_eq!(cap.slot(7), Some(1));
        let rows = [(10usize, 5u32), (11, 6)];
        cap.record(0, &rows, &[1.0, 2.0, 3.0, 4.0]);
        let (reset, done) = cap.drain();
        assert!(reset, "enable starts from an empty ring");
        assert!(done.is_empty(), "half-captured rows are not complete");
        cap.record(1, &rows, &[5.0, 6.0, 7.0, 8.0]);
        let (_, done) = cap.drain();
        assert_eq!(done.len(), 2);
        assert_eq!(done[0].pos, 10);
        assert_eq!(done[0].feat, vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(done[1].feat, vec![3.0, 4.0, 7.0, 8.0]);
        // A layer arriving without slot 0 (a remote layer in between) is dropped.
        cap.record(1, &rows, &[0.0; 4]);
        assert!(cap.drain().1.is_empty());
    }

    #[test]
    fn capture_caps_pending_rows_to_the_newest() {
        let cap = LayerCapture::default();
        cap.enable(vec![0], 1, 3);
        let rows: Vec<(usize, u32)> = (0..5).map(|p| (p, p as u32)).collect();
        cap.record(0, &rows, &[0.0, 1.0, 2.0, 3.0, 4.0]);
        let (_, done) = cap.drain();
        assert_eq!(done.iter().map(|r| r.pos).collect::<Vec<_>>(), vec![2, 3, 4]);
    }

    #[test]
    fn ring_mask_sees_committed_context_inside_the_window_and_the_whole_block() {
        let mut ring = RingIndex::new(8);
        for p in 0..6 {
            ring.put(p, 100 + p as u32);
        }
        // Anchor at 4: positions 4 and 5 are unverified rows, not context.
        let m = ring.mask(4, 2, 3);
        // Two block columns, then the 6 slots written so far.
        let n_kv = 2 + 6;
        assert_eq!(m.len(), 2 * n_kv);
        // q = 4, window 3: the block, then positions 2 and 3.
        assert_eq!(&m[..n_kv], &[1, 1, 0, 0, 1, 1, 0, 0]);
        // q = 5: position 3 only.
        assert_eq!(&m[n_kv..], &[1, 1, 0, 0, 0, 1, 0, 0]);
    }

    #[test]
    fn ring_prefix_sync_drops_slots_past_the_first_disagreement() {
        let mut ring = RingIndex::new(8);
        for p in 0..5 {
            ring.put(p, p as u32);
        }
        // New context diverges at position 2; the last entry is the anchor.
        let agree = ring.sync_prefix(&[0, 1, 9, 3, 4, 5]);
        assert_eq!(agree, 2);
        assert_eq!(ring.slot_pos[..5], [Some(0), Some(1), None, None, None]);
        // A matching context keeps everything it covers.
        let mut ring = RingIndex::new(8);
        for p in 0..3 {
            ring.put(p, p as u32);
        }
        assert_eq!(ring.sync_prefix(&[0, 1, 2, 7]), 3);
        assert_eq!(ring.slot_pos[2], Some(2));
    }

    #[test]
    fn ring_slots_wrap_by_position() {
        let mut ring = RingIndex::new(4);
        assert_eq!(ring.put(5, 1), 1);
        assert_eq!(ring.slot_pos[1], Some(5));
    }

    fn table(rank: usize, rows: &[Vec<f32>]) -> SelectorTable {
        SelectorTable {
            kind: GgmlType::F32,
            bytes: rows.iter().flatten().flat_map(|v| v.to_le_bytes()).collect(),
            rank,
        }
    }

    #[test]
    fn selector_walk_follows_the_pairwise_score_not_just_the_logits() {
        // Vocab 4, rank 2. Candidate 1 has the higher logit at position 1, but
        // the anchor (token 0) pairs strongly with candidate 2.
        let prev = table(2, &[vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 1.0]]);
        let next = table(2, &[vec![0.0, 0.0], vec![0.0, 0.0], vec![5.0, 0.0], vec![0.0, 5.0]]);
        let cands = vec![vec![1, 2], vec![3, 1]];
        let unary = vec![vec![1.0, 0.0], vec![0.0, 0.5]];
        let gates = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let path = walk_selector(0, &cands, &unary, &gates, &prev, &next, 2);
        // Pos 1: cand 1 -> 0 + 1.0; cand 2 -> 5*1 + 0 = 5 -> token 2.
        // Pos 2 (pred 2, prev row [0,1]): cand 3 -> 5; cand 1 -> 0.5 -> token 3.
        assert_eq!(path, vec![2, 3]);
    }

    #[test]
    fn q8_0_row_reads_one_token_of_a_rank_by_vocab_table() {
        // Two tokens, rank 32: one block each, scale 0.5 (f16 0x3800).
        let mut bytes = Vec::new();
        for tok in 0..2u8 {
            bytes.extend_from_slice(&0x3800u16.to_le_bytes());
            bytes.extend((0..32).map(|i| (i as i8 - tok as i8 * 2) as u8));
        }
        let mut out = vec![0.0f32; 32];
        assert!(q8_0_row(&bytes, 1, 32, &mut out));
        assert_eq!(out[0], -1.0);
        assert_eq!(out[31], 14.5);
        assert!(!q8_0_row(&bytes, 2, 32, &mut out));
    }
}
