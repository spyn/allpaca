//! `qwen4exp` on HIP: a token runs as one chain of kernels on the card that
//! holds its block, from the residual streams to the block's output. The
//! host forward in `qwen4exp.rs` drives it and keeps what needs the host:
//! the token and n-gram embeddings, the expert routing (the router's scores
//! come back once per block) and the experts that are not on a card.
//!
//! Matrices sit on their block's card in the file encoding where a kernel
//! reads it (F32, F16, BF16, Q2_0, Q3_K to Q6_K, Q8_0) and requantised to
//! Q8_0 otherwise (the float kernels are an order of magnitude slower).
//! Routed experts are whole blocks of Q2_0 slots (gate, up, down per
//! expert). Blocks are split in three runs: the first on card 0 with their
//! experts, the next on card 1 with theirs, the rest on card 0 with their
//! experts on the host.

use std::collections::HashMap;
use std::sync::Arc;

use allpaca_gguf::{GgmlType, GgufFile, TensorInfo};
use allpaca_hip::{
    device_local_headroom, device_memory_info, device_memset, memcpy, q2_0_expert_slot_bytes, q2_0_moe_ffn_device,
    q4x, set_device, silu_scale_device, DeviceWeights, HipWorkspace, MEMCPY_D2D,
};
use allpaca_types::{AllpacaError, FatalError};

use crate::deltanet::GdnDims;
use crate::model::{ModelConfig, TinyModel};

/// Matrices of a block that run on the GPU, by tensor name after `blk.N.`.
const BLOCK_MATRICES: [&str; 22] = [
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_out.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "hc_attn_down.weight",
    "hc_attn_up.weight",
    "hc_attn_inject.weight",
    "hc_ffn_down.weight",
    "hc_ffn_up.weight",
    "hc_ffn_inject.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_inp_shexp.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
    "ple_key.weight",
    "ple_value.weight",
];
/// F32 vectors of a block the kernels read.
const BLOCK_VECTORS: [&str; 8] = [
    "hc_attn_norm.weight",
    "hc_ffn_norm.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "ssm_norm.weight",
    "ssm_conv1d.weight",
    "ssm_dt.bias",
    "ssm_a",
];
const GLOBAL_MATRICES: [&str; 3] = ["output.weight", "output_hc_down.weight", "output_hc_up.weight"];
const GLOBAL_VECTORS: [&str; 1] = ["output_hc_norm.weight"];
/// VRAM left free on a card after the blocks are placed.
const CARD_RESERVE: u64 = 1536 * 1024 * 1024;
/// Most routed experts one kernel call takes.
const MAX_PICKS: usize = 32;
/// Tokens a full-attention block's key and value buffers start with.
const KV_START_TOKENS: usize = 256;

fn gpu_err(msg: String) -> AllpacaError {
    AllpacaError::Fatal(FatalError::ModelLoad(msg))
}

/// The block a tensor belongs to and its name inside the block.
fn block_of(name: &str) -> Option<(usize, &str)> {
    let (blk, suffix) = name.strip_prefix("blk.")?.split_once('.')?;
    Some((blk.parse().ok()?, suffix))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Encoding {
    F32,
    F16,
    Bf16,
    Q2_0,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    /// Staged Q8_0 blocks ([`allpaca_hip::stage_q8_0_gemv`]).
    Q8,
}

/// The encoding a kernel reads `info` in, if any: K-quants need rows of
/// whole 256-weight super-blocks, Q2_0 whole 64-weight blocks.
fn native_encoding(info: &TensorInfo, k: usize) -> Option<Encoding> {
    match info.ggml_type {
        GgmlType::F32 => Some(Encoding::F32),
        GgmlType::F16 => Some(Encoding::F16),
        GgmlType::BF16 => Some(Encoding::Bf16),
        GgmlType::Q2_0 if k % 64 == 0 => Some(Encoding::Q2_0),
        GgmlType::Q3_K if k % 256 == 0 => Some(Encoding::Q3K),
        GgmlType::Q4_K if k % 256 == 0 => Some(Encoding::Q4K),
        GgmlType::Q5_K if k % 256 == 0 => Some(Encoding::Q5K),
        GgmlType::Q6_K if k % 256 == 0 => Some(Encoding::Q6K),
        _ => None,
    }
}

/// Input and output widths of a matrix tensor (a vector is one row).
fn shape(info: &TensorInfo) -> (usize, usize) {
    let k = info.dims[0] as usize;
    (k, info.dims.get(1).map_or(1, |n| *n as usize))
}

/// Device bytes of a matrix in the encoding it is uploaded in.
fn matrix_bytes(info: &TensorInfo) -> u64 {
    let (k, n) = shape(info);
    match native_encoding(info, k) {
        Some(_) => info.nbytes() as u64,
        None if k % 32 == 0 => (n * k / 32 * 36) as u64,
        None => (2 * k * n) as u64,
    }
}

/// A matrix in the encoding it runs in on the device.
fn encode(file: &GgufFile, info: &TensorInfo) -> Result<(Encoding, Vec<u8>), AllpacaError> {
    let (k, n) = shape(info);
    if let Some(encoding) = native_encoding(info, k) {
        return Ok((encoding, file.tensor_bytes(&info.name)?.to_vec()));
    }
    if k % 32 != 0 {
        let halves = file.tensor_f32(&info.name)?;
        return Ok((Encoding::F16, halves.iter().flat_map(|v| allpaca_cpu::f32_to_f16(*v).to_le_bytes()).collect()));
    }
    let staged = if info.ggml_type == GgmlType::Q8_0 {
        allpaca_hip::stage_q8_0_gemv(file.tensor_bytes(&info.name)?, n, k)?
    } else {
        allpaca_hip::stage_q8_0_gemv(&allpaca_cpu::quantize_q8_0(&file.tensor_f32(&info.name)?), n, k)?
    };
    Ok((Encoding::Q8, staged))
}

struct Matrix {
    weights: DeviceWeights,
    encoding: Encoding,
    /// Output and input widths.
    n: usize,
    k: usize,
}

/// Float offsets of a card's activation buffer.
#[derive(Debug, Clone, Copy, Default)]
struct Act {
    streams: usize,
    xn: usize,
    gate: usize,
    low: usize,
    /// The mixed block input, directly followed by the router's scores.
    x: usize,
    scores: usize,
    inject: usize,
    /// Five scratch rows of `wide` floats.
    t: [usize; 5],
    out: usize,
    expert_h: usize,
    routed: usize,
    shared: usize,
    shared_gate: usize,
    shared_up: usize,
    shared_open: usize,
    /// Host products: input, then outputs (the logits among them).
    io_in: usize,
    io_out: usize,
    io_out_floats: usize,
    total: usize,
}

struct Card {
    hip: Arc<HipWorkspace>,
    act: DeviceWeights,
    /// A block's routed experts as Q2_0 slots, slot `e` = expert `e`.
    pools: HashMap<usize, DeviceWeights>,
}

impl Card {
    fn at(&self, float: usize) -> *mut f32 {
        self.act.ptr_at(float * 4).cast_mut()
    }
}

/// Which card runs each block and how many blocks have their experts there.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Placement {
    block_card: Vec<usize>,
    /// Blocks with experts on card 0 (the first ones) and on card 1 (the next).
    expert_blocks: [usize; 2],
}

fn free_bytes(device: i32) -> u64 {
    let (free, _) = device_memory_info(device).unwrap_or((0, 0));
    device_local_headroom(device).map_or(free, |h| h.min(free))
}

/// Split `n_blocks` blocks over the cards: `asked` (expert blocks per card)
/// when given, else as many as fit. `fixed` is card 0's bytes outside the
/// blocks, `dense` a block's matrices, `experts` a block's expert pool
/// (zero when the experts cannot go on a card).
fn place(n_blocks: usize, budgets: &[u64], asked: Option<&[usize]>, fixed: u64, dense: u64, experts: u64) -> Placement {
    let two = budgets.len() > 1;
    let (first, second) = match asked {
        Some(asked) => {
            let second = if two { asked.get(1).copied().unwrap_or(0).min(n_blocks) } else { 0 };
            (asked.first().copied().unwrap_or(0).min(n_blocks - second), second)
        }
        None if experts == 0 => (0, 0),
        None => {
            let second = if two { (budgets[1] / (dense + experts)) as usize } else { 0 }.min(n_blocks);
            let rest = (n_blocks - second) as u64 * dense + fixed;
            (((budgets[0].saturating_sub(rest) / experts) as usize).min(n_blocks - second), second)
        }
    };
    let block_card = (0..n_blocks).map(|b| usize::from(b >= first && b < first + second)).collect();
    Placement {
        block_card,
        expert_blocks: [first, second],
    }
}

/// One sequence's state on the cards: the recurrent blocks' states and conv
/// histories, the attention blocks' keys and values.
#[derive(Default)]
pub struct Qwen4DeviceState {
    blocks: Vec<BlockState>,
}

#[derive(Default)]
struct BlockState {
    state: Option<DeviceWeights>,
    conv: Option<DeviceWeights>,
    k: Option<DeviceWeights>,
    v: Option<DeviceWeights>,
    kv_tokens: usize,
}

fn zeroed(device: i32, floats: usize) -> Result<DeviceWeights, AllpacaError> {
    let buf = DeviceWeights::alloc(device, floats * 4)?;
    set_device(device)?;
    device_memset(buf.ptr_at(0).cast_mut().cast(), 0, floats * 4)?;
    Ok(buf)
}

fn copy_of(src: &DeviceWeights, bytes: usize) -> Result<DeviceWeights, AllpacaError> {
    let dst = DeviceWeights::alloc(src.device(), bytes.max(src.bytes() as usize))?;
    set_device(src.device())?;
    memcpy(dst.ptr_at(0).cast_mut().cast(), src.ptr_at(0).cast(), src.bytes() as usize, MEMCPY_D2D)?;
    Ok(dst)
}

impl Clone for Qwen4DeviceState {
    /// A deep copy on the same cards.
    fn clone(&self) -> Self {
        let dup = |w: &Option<DeviceWeights>| w.as_ref().map(|w| copy_of(w, 0).expect("copy qwen4exp device state"));
        Self {
            blocks: self
                .blocks
                .iter()
                .map(|b| BlockState {
                    state: dup(&b.state),
                    conv: dup(&b.conv),
                    k: dup(&b.k),
                    v: dup(&b.v),
                    kv_tokens: b.kv_tokens,
                })
                .collect(),
        }
    }
}

impl std::fmt::Debug for Qwen4DeviceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen4DeviceState").field("blocks", &self.blocks.len()).finish()
    }
}

pub struct Qwen4Gpu {
    cards: Vec<Card>,
    placement: Placement,
    matrices: HashMap<String, Matrix>,
    vectors: HashMap<String, DeviceWeights>,
    act: Act,
    hc: usize,
    dim: usize,
    ff: usize,
    n_experts: usize,
    eps: f32,
}

impl std::fmt::Debug for Qwen4Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen4Gpu")
            .field("matrices", &self.matrices.len())
            .field("placement", &self.placement)
            .finish_non_exhaustive()
    }
}

impl Qwen4Gpu {
    /// Upload the model to `hip` and `second`. `expert_blocks` fixes how
    /// many blocks have their experts on each card; `None` reads
    /// `ALLPACA_Q4X_EXPERT_LAYERS` ("9,21") or fits as many as there is room for.
    pub fn build(
        file: &GgufFile,
        cfg: &ModelConfig,
        hc: usize,
        hip: Arc<HipWorkspace>,
        second: Option<Arc<HipWorkspace>>,
        expert_blocks: Option<&[usize]>,
    ) -> Result<Self, AllpacaError> {
        let (dim, ff, n_experts) = (cfg.dim, cfg.ff_dim.max(1), cfg.n_experts);
        let hips: Vec<Arc<HipWorkspace>> = std::iter::once(hip).chain(second).collect();
        let on_gpu = |t: &&TensorInfo| match block_of(&t.name) {
            Some((_, suffix)) => BLOCK_MATRICES.contains(&suffix),
            None => GLOBAL_MATRICES.contains(&t.name.as_str()),
        };
        let wanted: Vec<&TensorInfo> = file.tensors.iter().filter(on_gpu).filter(|t| t.dims.len() <= 2).collect();

        // Sizes: the widest rows for the activation buffer, bytes for the placement.
        let (mut wide, mut low, mut shared_ff, mut io_k, mut io_n) = (dim, 1, 1, hc * dim, cfg.vocab_size.max(hc * dim + dim));
        let mut block_dense = vec![0u64; cfg.n_layers];
        let mut fixed = 0u64;
        for info in &wanted {
            let (k, n) = shape(info);
            match block_of(&info.name) {
                Some((blk, suffix)) => {
                    if let Some(bytes) = block_dense.get_mut(blk) {
                        *bytes += matrix_bytes(info);
                    }
                    match suffix {
                        "hc_attn_down.weight" | "hc_ffn_down.weight" => low = low.max(n),
                        "ffn_gate_shexp.weight" => shared_ff = shared_ff.max(n),
                        "hc_attn_up.weight" | "hc_ffn_up.weight" | "hc_attn_inject.weight" | "hc_ffn_inject.weight"
                        | "ffn_gate_inp.weight" => {}
                        "ple_key.weight" | "ple_value.weight" => (io_k, io_n) = (io_k.max(k), io_n.max(hc * dim + dim)),
                        _ => wide = wide.max(n).max(k),
                    }
                }
                None => {
                    fixed += matrix_bytes(info);
                    if info.name.ends_with("_down.weight") {
                        low = low.max(n);
                    }
                }
            }
        }
        let mut act = Act::default();
        let mut at = 0usize;
        let mut take = |floats: usize| {
            let start = at;
            at += floats;
            start
        };
        act.streams = take(hc * dim);
        act.xn = take(hc * dim);
        act.gate = take(hc * dim);
        act.low = take(low);
        act.x = take(dim);
        act.scores = take(n_experts.max(1));
        act.inject = take(hc);
        act.t = [take(wide), take(wide), take(wide), take(wide), take(wide)];
        act.out = take(dim);
        act.expert_h = take(MAX_PICKS * ff);
        act.routed = take(dim);
        act.shared = take(dim);
        act.shared_gate = take(shared_ff);
        act.shared_up = take(shared_ff);
        act.shared_open = take(1);
        act.io_in = take(io_k);
        act.io_out = take(io_n);
        act.io_out_floats = io_n;
        act.total = at;

        // Routed experts go on a card only as Q2_0 rows of whole blocks.
        let slot = q2_0_expert_slot_bytes(dim, ff);
        let packed = n_experts > 0
            && dim % 64 == 0
            && ff % 64 == 0
            && (0..cfg.n_layers).all(|layer| {
                ["gate", "up", "down"].iter().all(|part| {
                    file.tensor(&format!("blk.{}.ffn_{part}_exps.weight", cfg.gguf_blk(layer)))
                        .is_some_and(|t| t.ggml_type == GgmlType::Q2_0 && t.nbytes() == n_experts * slot / 3)
                })
            });
        let pool_bytes = if packed { (n_experts * slot) as u64 } else { 0 };
        let budgets: Vec<u64> = hips.iter().map(|h| free_bytes(h.device).saturating_sub(CARD_RESERVE)).collect();
        let from_env: Option<Vec<usize>> = std::env::var("ALLPACA_Q4X_EXPERT_LAYERS")
            .ok()
            .map(|v| v.split(',').filter_map(|n| n.trim().parse().ok()).collect());
        let asked = expert_blocks.or(from_env.as_deref()).filter(|_| packed);
        let dense = block_dense.iter().copied().max().unwrap_or(0);
        let placement = place(
            cfg.n_layers,
            &budgets,
            asked,
            fixed + (act.total * 4 * hips.len()) as u64,
            dense,
            pool_bytes,
        );

        let mut cards = Vec::new();
        for hip in hips {
            let act = DeviceWeights::alloc(hip.device, act.total * 4)?;
            cards.push(Card {
                hip,
                act,
                pools: HashMap::new(),
            });
        }
        let card_of = |name: &str| block_of(name).map_or(0, |(blk, _)| placement.block_card.get(blk).copied().unwrap_or(0));
        let mut matrices = HashMap::new();
        let mut dense_bytes = 0u64;
        for info in &wanted {
            let (k, n) = shape(info);
            let device = cards[card_of(&info.name)].hip.device;
            let (encoding, bytes) = encode(file, info)?;
            let weights = DeviceWeights::upload_bytes(device, &[&bytes])?;
            dense_bytes += weights.bytes();
            matrices.insert(info.name.clone(), Matrix { weights, encoding, n, k });
        }
        let mut vectors = HashMap::new();
        for info in &file.tensors {
            let wanted = match block_of(&info.name) {
                Some((_, suffix)) => BLOCK_VECTORS.contains(&suffix),
                None => GLOBAL_VECTORS.contains(&info.name.as_str()),
            };
            if wanted {
                let values = file.tensor_f32(&info.name)?;
                let device = cards[card_of(&info.name)].hip.device;
                vectors.insert(info.name.clone(), DeviceWeights::upload(device, &[&values])?);
            }
        }
        let mut gpu = Self {
            cards,
            placement,
            matrices,
            vectors,
            act,
            hc,
            dim,
            ff,
            n_experts,
            eps: cfg.eps,
        };
        let expert_bytes = gpu.upload_experts(file, cfg, slot)?;
        tracing::info!(
            matrices = gpu.matrices.len(),
            dense_mb = dense_bytes >> 20,
            expert_blocks = ?gpu.placement.expert_blocks,
            expert_mb = expert_bytes >> 20,
            "qwen4exp: matrices and routed experts on HIP"
        );
        Ok(gpu)
    }

    /// Blocks whose routed experts are on each card.
    #[cfg(test)]
    pub fn expert_layers(&self) -> Vec<Vec<usize>> {
        self.cards
            .iter()
            .map(|c| {
                let mut blocks: Vec<usize> = c.pools.keys().copied().collect();
                blocks.sort_unstable();
                blocks
            })
            .collect()
    }

    fn upload_experts(&mut self, file: &GgufFile, cfg: &ModelConfig, slot: usize) -> Result<u64, AllpacaError> {
        let matrix = slot / 3;
        let with_experts = self.placement.expert_blocks[0] + self.placement.expert_blocks[1];
        let mut total = 0u64;
        for layer in 0..with_experts.min(cfg.n_layers) {
            let blk = cfg.gguf_blk(layer);
            let card = self.placement.block_card[blk];
            let names = ["gate", "up", "down"].map(|part| format!("blk.{blk}.ffn_{part}_exps.weight"));
            let parts = [
                file.tensor_bytes(&names[0])?,
                file.tensor_bytes(&names[1])?,
                file.tensor_bytes(&names[2])?,
            ];
            let mut host = vec![0u8; self.n_experts * slot];
            for (e, dst) in host.chunks_exact_mut(slot).enumerate() {
                for (part, dst) in parts.iter().zip(dst.chunks_exact_mut(matrix)) {
                    dst.copy_from_slice(&part[e * matrix..(e + 1) * matrix]);
                }
            }
            let pool = DeviceWeights::upload_bytes(self.cards[card].hip.device, &[&host])?;
            total += pool.bytes();
            self.cards[card].pools.insert(blk, pool);
        }
        Ok(total)
    }

    pub fn has(&self, name: &str) -> bool {
        self.matrices.contains_key(name)
    }

    pub fn has_experts(&self, blk: usize) -> bool {
        self.cards.iter().any(|c| c.pools.contains_key(&blk))
    }

    /// The card that runs block `blk`.
    pub fn card_of(&self, blk: usize) -> usize {
        self.placement.block_card.get(blk).copied().unwrap_or(0)
    }

    fn matrix(&self, name: &str, k: usize) -> Result<&Matrix, AllpacaError> {
        let m = self
            .matrices
            .get(name)
            .ok_or_else(|| gpu_err(format!("qwen4exp: {name} is not on the GPU")))?;
        if m.k != k {
            return Err(gpu_err(format!("qwen4exp: {name} takes {} inputs, got {k}", m.k)));
        }
        Ok(m)
    }

    fn vector(&self, name: &str) -> Result<*const f32, AllpacaError> {
        self.vectors
            .get(name)
            .map(|v| v.ptr_at(0))
            .ok_or_else(|| gpu_err(format!("qwen4exp: {name} is not on the GPU")))
    }

    /// One row times `m` on `card`.
    fn launch(&self, card: &Card, m: &Matrix, a: *const f32, c: *mut f32) -> Result<(), AllpacaError> {
        let hip = &card.hip;
        let (n, k) = (m.n as i32, m.k as i32);
        let b = m.weights.byte_ptr_at(0);
        match m.encoding {
            Encoding::F32 => hip.gemm_nt_dev(a, b.cast(), c, 1, n, k),
            Encoding::F16 => hip.gemm_nt_f16_dev(a, b.cast(), c, 1, n, k),
            Encoding::Bf16 => hip.gemm_nt_bf16_dev(a, b.cast(), c, 1, n, k),
            Encoding::Q2_0 => hip.gemm_nt_q2_0_dev(a, b, c, 1, n, k),
            Encoding::Q3K => hip.gemm_nt_q3_k_dev(a, b, c, 1, n, k),
            Encoding::Q4K => hip.gemm_nt_q4_k_dev(a, b, c, 1, n, k),
            Encoding::Q5K => hip.gemm_nt_q5_k_dev(a, b, c, 1, n, k),
            Encoding::Q6K => hip.gemm_nt_q6_k_dev(a, b, c, 1, n, k),
            Encoding::Q8 => hip.gemm_nt_q8_dev(a, b, c, 1, n, k),
        }
    }

    /// `x` times each named matrix of one card: one upload and one download.
    pub fn products(&self, names: &[String], x: &[f32]) -> Result<Vec<Vec<f32>>, AllpacaError> {
        let card = &self.cards[names.first().and_then(|n| block_of(n)).map_or(0, |(blk, _)| self.card_of(blk))];
        let mats: Vec<&Matrix> = names.iter().map(|n| self.matrix(n, x.len())).collect::<Result<_, _>>()?;
        let total: usize = mats.iter().map(|m| m.n).sum();
        if x.len() > self.act.io_out - self.act.io_in || total > self.act.io_out_floats {
            return Err(gpu_err(format!("qwen4exp: products of {} inputs and {total} outputs do not fit", x.len())));
        }
        card.hip.copy_h2d_f32(card.at(self.act.io_in), x)?;
        let mut at = self.act.io_out;
        for m in &mats {
            self.launch(card, m, card.at(self.act.io_in), card.at(at))?;
            at += m.n;
        }
        let mut all = vec![0.0f32; total];
        card.hip.copy_d2h_f32(card.at(self.act.io_out), &mut all)?;
        let mut rest = all.as_slice();
        Ok(mats
            .iter()
            .map(|m| {
                let (head, tail) = rest.split_at(m.n);
                rest = tail;
                head.to_vec()
            })
            .collect())
    }

    /// Wait for everything queued on `card`.
    pub fn sync(&self, card: usize) -> Result<(), AllpacaError> {
        self.cards[card].hip.bind()?;
        allpaca_hip::stream_synchronize()
    }

    pub fn set_streams(&self, card: usize, streams: &[f32]) -> Result<(), AllpacaError> {
        let card = &self.cards[card];
        card.hip.copy_h2d_f32(card.at(self.act.streams), streams)
    }

    pub fn get_streams(&self, card: usize, streams: &mut [f32]) -> Result<(), AllpacaError> {
        let card = &self.cards[card];
        card.hip.copy_d2h_f32(card.at(self.act.streams), streams)
    }

    /// Collapse the card's streams into the block input through the mixer
    /// `prefix` (`blk.N.hc_attn`, `output_hc`); `inject` also computes the
    /// logits [`Self::combine`] weights the block's output with.
    pub fn mix(&self, card: usize, prefix: &str, inject: bool) -> Result<(), AllpacaError> {
        let (card, act, hc_dim) = (&self.cards[card], &self.act, self.hc * self.dim);
        let (hc, dim) = (self.hc as i32, self.dim as i32);
        let down = self.matrix(&format!("{prefix}_down.weight"), hc_dim)?;
        let up = self.matrix(&format!("{prefix}_up.weight"), down.n)?;
        let gains = self.vector(&format!("{prefix}_norm.weight"))?;
        card.hip.bind()?;
        q4x::stream_norm(card.at(act.streams), gains, card.at(act.xn), hc, dim, self.eps)?;
        self.launch(card, down, card.at(act.xn), card.at(act.low))?;
        silu_scale_device(card.at(act.low), down.n as i32, 1.0 / self.hc as f32)?;
        self.launch(card, up, card.at(act.low), card.at(act.gate))?;
        card.hip.bind()?;
        q4x::gated_mean(card.at(act.xn), card.at(act.gate), card.at(act.x), hc, dim)?;
        if inject {
            let w = self.matrix(&format!("{prefix}_inject.weight"), hc_dim)?;
            self.launch(card, w, card.at(act.xn), card.at(act.inject))?;
        }
        Ok(())
    }

    /// Add the block's output to the card's streams, weighted by the logits
    /// of the last [`Self::mix`].
    pub fn combine(&self, card: usize) -> Result<(), AllpacaError> {
        let (card, act) = (&self.cards[card], &self.act);
        card.hip.bind()?;
        q4x::combine(
            card.at(act.streams),
            card.at(act.out),
            card.at(act.inject),
            self.hc as i32,
            self.dim as i32,
        )
    }

    fn block_state<'a>(&self, state: &'a mut Qwen4DeviceState, blk: usize) -> &'a mut BlockState {
        if state.blocks.len() <= blk {
            state.blocks.resize_with(blk + 1, BlockState::default);
        }
        &mut state.blocks[blk]
    }

    /// Gated DeltaNet block `blk` on the mixed input: one recurrent step
    /// with a sigmoid output gate.
    pub fn gdn(&self, blk: usize, dims: &GdnDims, state: &mut Qwen4DeviceState) -> Result<(), AllpacaError> {
        let (card, act, dim) = (&self.cards[self.card_of(blk)], &self.act, self.dim);
        let device = card.hip.device;
        let name = |s: &str| format!("blk.{blk}.{s}");
        let state = self.block_state(state, blk);
        if state.state.is_none() {
            state.state = Some(zeroed(device, dims.state_floats())?);
            state.conv = Some(zeroed(device, dims.conv_state_floats().max(1))?);
        }
        let (rec, conv) = match (&state.state, &state.conv) {
            (Some(rec), Some(conv)) => (rec.ptr_at(0).cast_mut(), conv.ptr_at(0).cast_mut()),
            _ => return Err(gpu_err(format!("qwen4exp: block {blk} has no recurrent state"))),
        };
        let [qkv, z, alpha, beta, y] = act.t.map(|t| card.at(t));
        let x = card.at(act.x);
        self.launch(card, self.matrix(&name("attn_qkv.weight"), dim)?, x, qkv)?;
        self.launch(card, self.matrix(&name("attn_gate.weight"), dim)?, x, z)?;
        self.launch(card, self.matrix(&name("ssm_alpha.weight"), dim)?, x, alpha)?;
        self.launch(card, self.matrix(&name("ssm_beta.weight"), dim)?, x, beta)?;
        card.hip.gdn_layer_step(
            qkv,
            alpha,
            beta,
            self.vector(&name("ssm_conv1d.weight"))?,
            conv,
            self.vector(&name("ssm_dt.bias"))?,
            self.vector(&name("ssm_a"))?,
            rec,
            y,
            dims.n_v_heads as i32,
            dims.n_k_heads as i32,
            dims.head_v as i32,
            dims.head_k as i32,
            dims.conv_kernel as i32,
            self.eps,
            false,
        )?;
        card.hip.rmsnorm_heads_device(
            y,
            self.vector(&name("ssm_norm.weight"))?,
            dims.n_v_heads as i32,
            dims.head_v as i32,
            self.eps,
        )?;
        card.hip.sigmoid_mul_device(y, z, dims.value_dim as i32)?;
        self.launch(card, self.matrix(&name("ssm_out.weight"), dims.value_dim)?, y, card.at(act.out))
    }

    /// Full-attention block `blk` on the mixed input at position `pos`:
    /// gated queries, normed and rotated keys, attention over the sequence.
    pub fn attention(
        &self,
        blk: usize,
        cfg: &ModelConfig,
        pos: usize,
        state: &mut Qwen4DeviceState,
    ) -> Result<(), AllpacaError> {
        let (card, act, dim) = (&self.cards[self.card_of(blk)], &self.act, self.dim);
        let hip = &card.hip;
        let (head_dim, n_heads, n_kv) = (cfg.head_dim(), cfg.n_heads, cfg.n_kv_heads.max(1));
        let (attn, kv) = (n_heads * head_dim, n_kv * head_dim);
        let name = |s: &str| format!("blk.{blk}.{s}");
        let state = self.block_state(state, blk);
        if state.kv_tokens <= pos {
            let tokens = (2 * state.kv_tokens).max(KV_START_TOKENS).max(pos + 1);
            for buf in [&mut state.k, &mut state.v] {
                *buf = Some(match buf.as_ref() {
                    Some(old) => copy_of(old, tokens * kv * 4)?,
                    None => DeviceWeights::alloc(hip.device, tokens * kv * 4)?,
                });
            }
            state.kv_tokens = tokens;
        }
        let (keys, values) = match (&state.k, &state.v) {
            (Some(k), Some(v)) => (k.ptr_at(0).cast_mut(), v.ptr_at(0).cast_mut()),
            _ => return Err(gpu_err(format!("qwen4exp: block {blk} has no key buffer"))),
        };
        let [q_full, q, gate, k, v] = act.t.map(|t| card.at(t));
        let x = card.at(act.x);
        self.launch(card, self.matrix(&name("attn_q.weight"), dim)?, x, q_full)?;
        self.launch(card, self.matrix(&name("attn_k.weight"), dim)?, x, k)?;
        self.launch(card, self.matrix(&name("attn_v.weight"), dim)?, x, v)?;
        hip.deinterleave_q_gate_device(q_full, q, gate, n_heads as i32, head_dim as i32)?;
        hip.rmsnorm_heads_device(q, self.vector(&name("attn_q_norm.weight"))?, n_heads as i32, head_dim as i32, self.eps)?;
        hip.rmsnorm_heads_device(k, self.vector(&name("attn_k_norm.weight"))?, n_kv as i32, head_dim as i32, self.eps)?;
        let n_rot = if cfg.rope_dims > 0 { cfg.rope_dims } else { head_dim };
        hip.rope_imrope_qk_device(
            q,
            k,
            n_heads as i32,
            n_kv as i32,
            head_dim as i32,
            n_rot as i32,
            [pos as i32; 3],
            cfg.rope_theta,
            cfg.rope_sections,
        )?;
        // SAFETY: the buffers hold `kv_tokens > pos` rows of `kv` floats.
        let (key_row, value_row) = unsafe { (keys.add(pos * kv), values.add(pos * kv)) };
        hip.copy_d2d_f32(key_row, k, kv)?;
        hip.copy_d2d_f32(value_row, v, kv)?;
        // The interleaved queries are spent: their row takes the output.
        hip.attention_device(
            q,
            keys,
            values,
            q_full,
            1,
            (pos + 1) as i32,
            n_heads as i32,
            n_kv as i32,
            head_dim as i32,
            true,
        )?;
        hip.sigmoid_mul_device(q_full, gate, attn as i32)?;
        self.launch(card, self.matrix(&name("attn_output.weight"), attn)?, q_full, card.at(act.out))
    }

    /// Start block `blk`'s feed-forward on the mixed input: the router's
    /// scores come back in `scores`, the shared expert runs on the card.
    /// `x` also gets the mixed input when the experts are not on a card.
    pub fn ffn_route(&self, blk: usize, scores: &mut [f32], x: &mut Vec<f32>) -> Result<(), AllpacaError> {
        let (card, act, dim) = (&self.cards[self.card_of(blk)], &self.act, self.dim);
        let name = |s: &str| format!("blk.{blk}.{s}");
        let input = card.at(act.x);
        self.launch(card, self.matrix(&name("ffn_gate_inp.weight"), dim)?, input, card.at(act.scores))?;
        if self.has(&name("ffn_up_shexp.weight")) {
            let gate = self.matrix(&name("ffn_gate_shexp.weight"), dim)?;
            let (g, u) = (card.at(act.shared_gate), card.at(act.shared_up));
            self.launch(card, gate, input, g)?;
            self.launch(card, self.matrix(&name("ffn_up_shexp.weight"), dim)?, input, u)?;
            card.hip.silu_mul_device(g, u, gate.n as i32)?;
            self.launch(card, self.matrix(&name("ffn_down_shexp.weight"), gate.n)?, g, card.at(act.shared))?;
            self.launch(card, self.matrix(&name("ffn_gate_inp_shexp.weight"), dim)?, input, card.at(act.shared_open))?;
        }
        if card.pools.contains_key(&blk) {
            x.clear();
            return card.hip.copy_d2h_f32(card.at(act.scores), &mut scores[..self.n_experts]);
        }
        // The input and the scores are neighbours: one download.
        let mut both = vec![0.0f32; dim + self.n_experts];
        card.hip.copy_d2h_f32(input, &mut both)?;
        scores[..self.n_experts].copy_from_slice(&both[dim..]);
        both.truncate(dim);
        *x = both;
        Ok(())
    }

    /// Finish block `blk`'s feed-forward into the block output: the routed
    /// experts `picks` from the card's slots, or `host_routed` (their sum,
    /// computed on the host), plus the gated shared expert.
    pub fn ffn_finish(&self, blk: usize, picks: &[(usize, f32)], host_routed: Option<&[f32]>) -> Result<(), AllpacaError> {
        let (card, act) = (&self.cards[self.card_of(blk)], &self.act);
        let routed = card.at(act.routed);
        match (card.pools.get(&blk), host_routed) {
            (_, Some(sum)) => card.hip.copy_h2d_f32(routed, sum)?,
            (Some(pool), None) if !picks.is_empty() && picks.len() <= MAX_PICKS => {
                let slots: Vec<i32> = picks.iter().map(|(e, _)| *e as i32).collect();
                let weights: Vec<f32> = picks.iter().map(|(_, w)| *w).collect();
                card.hip.bind()?;
                q2_0_moe_ffn_device(
                    card.at(act.x),
                    pool.byte_ptr_at(0),
                    &slots,
                    &weights,
                    card.at(act.expert_h),
                    routed,
                    self.dim as i32,
                    self.ff as i32,
                )?;
            }
            _ => return Err(gpu_err(format!("qwen4exp: block {blk} has no routed experts to run"))),
        }
        let shared = self.has(&format!("blk.{blk}.ffn_up_shexp.weight"));
        card.hip.bind()?;
        q4x::ffn_sum(
            routed,
            if shared { card.at(act.shared).cast_const() } else { std::ptr::null() },
            if shared { card.at(act.shared_open).cast_const() } else { std::ptr::null() },
            card.at(act.out),
            self.dim as i32,
        )
    }

    /// The logits of card 0's mixed input through the output matrix `name`.
    pub fn head(&self, name: &str, logits: &mut [f32]) -> Result<(), AllpacaError> {
        let (card, act) = (&self.cards[0], &self.act);
        let m = self.matrix(name, self.dim)?;
        if m.n > act.io_out_floats || logits.len() < m.n {
            return Err(gpu_err(format!("qwen4exp: {} logits do not fit", m.n)));
        }
        self.launch(card, m, card.at(act.x), card.at(act.io_out))?;
        card.hip.copy_d2h_f32(card.at(act.io_out), &mut logits[..m.n])
    }
}

/// Build the GPU working set of a loaded `qwen4exp` model.
pub fn attach(
    model: &TinyModel,
    hip: &Arc<HipWorkspace>,
    second: Option<&Arc<HipWorkspace>>,
    expert_blocks: Option<&[usize]>,
) -> Result<(), AllpacaError> {
    let q4 = model.q4x.as_ref().ok_or_else(|| gpu_err("missing qwen4exp weights".into()))?;
    let gpu = Qwen4Gpu::build(q4.gguf()?, &model.config, q4.spec.hc, hip.clone(), second.cloned(), expert_blocks)?;
    *q4.gpu.lock().map_err(|_| gpu_err("qwen4exp gpu lock poisoned".into()))? = Some(gpu);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    #[test]
    fn blocks_fill_the_second_card_first_and_the_rest_keep_their_experts_on_the_host() {
        // 48 blocks of 0.1 GB matrices and 0.7 GB experts; card 0 has 9 GB
        // after 1 GB of fixed data, card 1 has 16 GB.
        let p = place(48, &[9 * GB, 16 * GB], None, GB, GB / 10, 7 * GB / 10);
        assert_eq!(p.expert_blocks, [7, 20]);
        assert_eq!(p.block_card[..7], [0; 7]);
        assert_eq!(p.block_card[7..27], [1; 20]);
        assert_eq!(p.block_card[27..], [0; 21]);
        // One card: every block's matrices are there, experts as far as they fit.
        let p = place(48, &[9 * GB], None, GB, GB / 10, 7 * GB / 10);
        assert_eq!((p.expert_blocks, p.block_card.iter().sum::<usize>()), ([4, 0], 0));
        // Experts that cannot go on a card: matrices only.
        assert_eq!(place(4, &[9 * GB, 9 * GB], None, 0, GB, 0).expert_blocks, [0, 0]);
        // An explicit split is taken as given, within the block count.
        let p = place(4, &[GB, GB], Some(&[1, 2]), 0, GB, GB);
        assert_eq!((p.expert_blocks, p.block_card), ([1, 2], vec![0, 1, 1, 0]));
        assert_eq!(place(4, &[GB, GB], Some(&[9, 9]), 0, GB, GB).expert_blocks, [0, 4]);
    }
}
