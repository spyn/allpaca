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

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

use allpaca_gguf::{GgmlType, GgufFile, TensorInfo};
use allpaca_hip::{
    device_local_headroom, device_memory_info, device_memset, memcpy, q2_0_expert_slot_bytes, q2_0_moe_ffn_device,
    q4x, set_device, silu_scale_device, DeviceWeights, HipWorkspace, MEMCPY_D2D, MEMCPY_D2H, MEMCPY_H2D,
};
use allpaca_types::{AllpacaError, FatalError};

use crate::deltanet::GdnDims;
use crate::model::{ModelConfig, TinyModel};
use crate::qsa::{QsaKeys, QsaShape};

/// Matrices of a block that run on the GPU, by tensor name after `blk.N.`.
const BLOCK_MATRICES: [&str; 24] = [
    "indexer.q_proj.weight",
    "indexer.k_proj.weight",
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
/// F32 vectors of a block the host reads (the indexer's norm gains).
const HOST_VECTORS: [&str; 2] = ["indexer.q_norm.weight", "indexer.k_norm.weight"];
const GLOBAL_MATRICES: [&str; 3] = ["output.weight", "output_hc_down.weight", "output_hc_up.weight"];
const GLOBAL_VECTORS: [&str; 1] = ["output_hc_norm.weight"];
/// VRAM left free on a card after the blocks are placed.
const CARD_RESERVE: u64 = 1536 * 1024 * 1024;
/// VRAM a worker's GPU keeps free when the coordinator counts the blocks it
/// can host, and when the worker places them.
const WORKER_RESERVE: u64 = 1024 * 1024 * 1024;
const HOST_RESERVE: u64 = 512 * 1024 * 1024;
/// Most routed experts one kernel call takes.
const MAX_PICKS: usize = 32;
/// Most tokens of a sequence one pass takes.
pub const MAX_ROWS: usize = 16;
/// Tokens a full-attention block's key and value buffers start with.
const KV_START_TOKENS: usize = 256;

/// Launches a profiled matrix product is repeated for.
const PROFILE_REPEATS: u32 = 16;

/// Calls and seconds per matrix kind (encoding, shape, rows), filled when
/// `ALLPACA_Q4X_PROFILE` is set.
fn profile() -> Option<&'static Mutex<BTreeMap<String, (u64, f64)>>> {
    static PROFILE: OnceLock<Option<Mutex<BTreeMap<String, (u64, f64)>>>> = OnceLock::new();
    PROFILE
        .get_or_init(|| std::env::var_os("ALLPACA_Q4X_PROFILE").map(|_| Mutex::default()))
        .as_ref()
}

/// The matrix products of the passes so far, slowest kind first:
/// `(kind, calls, seconds)`. Empty unless `ALLPACA_Q4X_PROFILE` is set.
pub fn profile_report() -> Vec<(String, u64, f64)> {
    let mut kinds: Vec<_> = profile()
        .map(|p| p.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|(k, v)| (k.clone(), v.0, v.1)).collect())
        .unwrap_or_default();
    kinds.sort_by(|a, b| b.2.total_cmp(&a.2));
    kinds
}

fn gpu_err(msg: String) -> AllpacaError {
    AllpacaError::Fatal(FatalError::ModelLoad(msg))
}

/// The block a tensor belongs to and its name inside the block.
fn block_of(name: &str) -> Option<(usize, &str)> {
    let (blk, suffix) = name.strip_prefix("blk.")?.split_once('.')?;
    Some((blk.parse().ok()?, suffix))
}

/// What a full-attention block needs of the model's shape.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct AttnShape {
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub rope_dims: usize,
    pub rope_theta: f32,
    pub rope_sections: [i32; 4],
    /// The indexer of the blocks that have one: their attention turns
    /// sparse past its budget ([`crate::qsa`]).
    pub qsa: Option<QsaShape>,
}

impl AttnShape {
    pub fn of(cfg: &ModelConfig, qsa: Option<QsaShape>) -> Self {
        Self {
            head_dim: cfg.head_dim(),
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            rope_dims: cfg.rope_dims,
            rope_theta: cfg.rope_theta,
            rope_sections: cfg.rope_sections,
            qsa,
        }
    }
}

/// What running blocks without the host needs besides their weights.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ChainShape {
    /// Routed experts per token and the scale of their weights.
    pub n_active: usize,
    pub scale: f32,
    pub gdn: Option<GdnDims>,
    pub attn: AttnShape,
}

/// A block an RPC worker hosts.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct HostedBlock {
    pub blk: usize,
    /// Gated DeltaNet, else full attention.
    pub recurrent: bool,
    /// Device bytes of its matrices and routed experts.
    pub bytes: u64,
}

/// The blocks an RPC worker hosts and the sizes their chain needs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HostedSpec {
    pub hc: usize,
    pub dim: usize,
    pub ff: usize,
    pub n_experts: usize,
    pub eps: f32,
    /// Blocks of the whole model.
    pub n_blocks: usize,
    sizes: ActSizes,
    /// In the order they run.
    pub blocks: Vec<HostedBlock>,
    pub shape: ChainShape,
}

/// Row widths the activation buffer is laid out for.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
struct ActSizes {
    /// Widest scratch row, the mixers' low dimension, the shared expert's width.
    wide: usize,
    low: usize,
    shared_ff: usize,
    /// Host products: input and output floats.
    io_k: usize,
    io_n: usize,
}

/// A card's activation buffer for `MAX_ROWS` rows. Every part starts on 64
/// bytes: some kernels pick their variant by the alignment of their input,
/// and a token's logits must not depend on how the buffer happens to be
/// laid out.
fn layout(hc: usize, dim: usize, ff: usize, n_experts: usize, sizes: &ActSizes) -> Act {
    let mut act = Act::default();
    let mut at = 0usize;
    let mut take = |floats: usize| {
        let start = at;
        at += floats.next_multiple_of(16);
        start
    };
    let (r, wide) = (MAX_ROWS, MAX_ROWS * sizes.wide);
    act.streams = take(r * hc * dim);
    act.xn = take(r * hc * dim);
    act.gate = take(r * hc * dim);
    act.low = take(r * sizes.low);
    act.x = take(r * (dim + n_experts.max(1)));
    act.inject = take(r * hc);
    act.t = [take(wide), take(wide), take(wide), take(wide), take(wide)];
    act.out = take(r * dim);
    act.expert_h = take(r * MAX_PICKS * ff);
    act.picks = take(r * q4x::PICKS_BYTES / 4);
    act.routed = take(r * dim);
    act.shared = take(r * dim);
    act.shared_gate = take(r * sizes.shared_ff);
    act.shared_up = take(r * sizes.shared_ff);
    act.shared_open = take(r);
    act.io_in = take(sizes.io_k);
    act.io_out = take(sizes.io_n);
    act.io_out_floats = sizes.io_n;
    act.total = at;
    act
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

impl Encoding {
    const ALL: [Self; 9] =
        [Self::F32, Self::F16, Self::Bf16, Self::Q2_0, Self::Q3K, Self::Q4K, Self::Q5K, Self::Q6K, Self::Q8];

    /// Its number on the wire.
    fn code(self) -> u8 {
        Self::ALL.iter().position(|e| *e == self).unwrap_or(0) as u8
    }

    fn from_code(code: u8) -> Option<Self> {
        Self::ALL.get(code as usize).copied()
    }
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

/// Float offsets of a card's activation buffer; each part holds
/// [`MAX_ROWS`] rows.
#[derive(Debug, Clone, Copy, Default)]
struct Act {
    streams: usize,
    xn: usize,
    gate: usize,
    low: usize,
    /// The mixed block input, then the router's scores right after the
    /// rows in use (one download brings both).
    x: usize,
    inject: usize,
    /// Five scratch rows of `wide` floats.
    t: [usize; 5],
    out: usize,
    expert_h: usize,
    /// A token's picks when its experts are routed on the card.
    picks: usize,
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
/// (zero when the experts cannot go on a card). The last `remote` blocks
/// are not on these cards at all.
fn place(
    n_blocks: usize,
    budgets: &[u64],
    asked: Option<&[usize]>,
    fixed: u64,
    dense: u64,
    experts: u64,
    remote: usize,
) -> Placement {
    let n_blocks = n_blocks - remote.min(n_blocks);
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
    let block_card = (0..n_blocks + remote).map(|b| usize::from(b >= first && b < first + second)).collect();
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
    /// Per device, once an attention is sparse: the cells a token keeps
    /// (row numbers, then their keys, then their values).
    picked: HashMap<i32, DeviceWeights>,
}

#[derive(Default)]
struct BlockState {
    state: Option<DeviceWeights>,
    conv: Option<DeviceWeights>,
    k: Option<DeviceWeights>,
    v: Option<DeviceWeights>,
    /// Raw indexer keys, one row per token like `k` and `v`.
    ik: Option<DeviceWeights>,
    kv_tokens: usize,
    /// The indexer keys the host has read back to select with.
    qsa: QsaKeys,
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
                    ik: dup(&b.ik),
                    kv_tokens: b.kv_tokens,
                    qsa: b.qsa.clone(),
                })
                .collect(),
            picked: HashMap::new(),
        }
    }
}

/// A sequence's [`Qwen4DeviceState`] read back to the host (for the prompt
/// cache), to be put on the same cards again.
#[derive(Debug, Clone, Default)]
pub struct Qwen4DeviceHost {
    blocks: Vec<BlockHost>,
}

#[derive(Debug, Clone, Default)]
struct BlockHost {
    device: i32,
    state: Option<Vec<f32>>,
    conv: Option<Vec<f32>>,
    /// The rows of `k`, `v` and `ik` that hold tokens.
    k: Option<Vec<f32>>,
    v: Option<Vec<f32>>,
    ik: Option<Vec<f32>>,
    rows: usize,
    qsa: QsaKeys,
}

fn read_back(buf: &Option<DeviceWeights>, floats: usize) -> Result<Option<Vec<f32>>, AllpacaError> {
    let Some(buf) = buf else {
        return Ok(None);
    };
    let floats = floats.min(buf.bytes() as usize / 4);
    let mut out = vec![0.0f32; floats];
    set_device(buf.device())?;
    memcpy(out.as_mut_ptr().cast(), buf.ptr_at(0).cast(), floats * 4, MEMCPY_D2H)?;
    Ok(Some(out))
}

fn put_back(device: i32, values: &Option<Vec<f32>>, floats: usize) -> Result<Option<DeviceWeights>, AllpacaError> {
    let Some(values) = values else {
        return Ok(None);
    };
    let buf = zeroed(device, floats.max(values.len()).max(1))?;
    memcpy(buf.ptr_at(0).cast_mut().cast(), values.as_ptr().cast(), values.len() * 4, MEMCPY_H2D)?;
    Ok(Some(buf))
}

impl Qwen4DeviceState {
    /// Read back what the sequence's first `tokens` tokens left on the cards
    /// (everything queued on them must have finished).
    pub fn to_host(&self, tokens: usize) -> Result<Qwen4DeviceHost, AllpacaError> {
        let blocks = self
            .blocks
            .iter()
            .map(|b| {
                let device = [&b.state, &b.conv, &b.k, &b.v, &b.ik]
                    .into_iter()
                    .find_map(|w| w.as_ref().map(DeviceWeights::device))
                    .unwrap_or(0);
                let rows = tokens.min(b.kv_tokens);
                let used = |w: &Option<DeviceWeights>| {
                    w.as_ref()
                        .map_or(0, |w| w.bytes() as usize / 4 / b.kv_tokens.max(1) * rows)
                };
                Ok(BlockHost {
                    device,
                    state: read_back(&b.state, usize::MAX)?,
                    conv: read_back(&b.conv, usize::MAX)?,
                    k: read_back(&b.k, used(&b.k))?,
                    v: read_back(&b.v, used(&b.v))?,
                    ik: read_back(&b.ik, used(&b.ik))?,
                    rows,
                    qsa: b.qsa.clone(),
                })
            })
            .collect::<Result<_, AllpacaError>>()?;
        Ok(Qwen4DeviceHost { blocks })
    }

    /// Put a state read back by [`Self::to_host`] on its cards again.
    pub fn from_host(host: &Qwen4DeviceHost) -> Result<Self, AllpacaError> {
        let blocks = host
            .blocks
            .iter()
            .map(|b| {
                let capacity = if b.rows == 0 { 0 } else { b.rows.max(KV_START_TOKENS) };
                let rows_of = |v: &Option<Vec<f32>>| {
                    v.as_ref().map_or(0, |v| v.len() / b.rows.max(1) * capacity)
                };
                Ok(BlockState {
                    state: put_back(b.device, &b.state, 0)?,
                    conv: put_back(b.device, &b.conv, 0)?,
                    k: put_back(b.device, &b.k, rows_of(&b.k))?,
                    v: put_back(b.device, &b.v, rows_of(&b.v))?,
                    ik: put_back(b.device, &b.ik, rows_of(&b.ik))?,
                    kv_tokens: capacity,
                    qsa: b.qsa.clone(),
                })
            })
            .collect::<Result<_, AllpacaError>>()?;
        Ok(Self {
            blocks,
            picked: HashMap::new(),
        })
    }
}

impl Qwen4DeviceHost {
    /// Bytes held on the host.
    pub fn host_bytes(&self) -> usize {
        self.blocks
            .iter()
            .map(|b| {
                let len = |v: &Option<Vec<f32>>| v.as_ref().map_or(0, Vec::len);
                (len(&b.state) + len(&b.conv) + len(&b.k) + len(&b.v) + len(&b.ik) + b.qsa.floats()) * 4
            })
            .sum()
    }

    /// Append to a prompt-cache checkpoint file.
    pub fn write_to(&self, out: &mut Vec<u8>) {
        use crate::kv_codec::write_opt_f32s;
        out.extend_from_slice(&(self.blocks.len() as u32).to_le_bytes());
        for b in &self.blocks {
            out.extend_from_slice(&b.device.to_le_bytes());
            out.extend_from_slice(&(b.rows as u64).to_le_bytes());
            for v in [&b.state, &b.conv, &b.k, &b.v, &b.ik] {
                write_opt_f32s(out, v.as_deref());
            }
            b.qsa.write_to(out);
        }
    }

    /// What [`Self::write_to`] wrote.
    pub fn read_from(r: &mut crate::kv_codec::Reader<'_>) -> Option<Self> {
        let n = r.u32()? as usize;
        let mut blocks = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            let device = r.u32()? as i32;
            let rows = usize::try_from(r.u64()?).ok()?;
            blocks.push(BlockHost {
                device,
                rows,
                state: r.opt_f32s()?,
                conv: r.opt_f32s()?,
                k: r.opt_f32s()?,
                v: r.opt_f32s()?,
                ik: r.opt_f32s()?,
                qsa: QsaKeys::read_from(r)?,
            });
        }
        Some(Self { blocks })
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
    /// [`HOST_VECTORS`] by tensor name.
    host_vectors: HashMap<String, Vec<f32>>,
    act: Act,
    hc: usize,
    dim: usize,
    ff: usize,
    n_experts: usize,
    eps: f32,
    /// The routed experts are Q2_0 rows of whole blocks (they can go on a
    /// card or a worker).
    packed: bool,
    /// Blocks whose routed experts are on an RPC worker.
    remote: Vec<usize>,
    sizes: ActSizes,
    /// Device bytes of each block's matrices.
    block_dense: Vec<u64>,
    /// Blocks left off the cards for an RPC worker to host whole, and the
    /// worker's end once it has them.
    to_host: Vec<usize>,
    hosted: Option<Arc<crate::qwen4_remote::RemoteBlocks>>,
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
    /// `worker` is an RPC worker that can host whole blocks: the free bytes
    /// of its GPUs and the first block it may take. Blocks without their
    /// experts on a card are then left for it ([`Self::host_blocks`]).
    pub fn build(
        file: &GgufFile,
        cfg: &ModelConfig,
        hc: usize,
        hip: Arc<HipWorkspace>,
        second: Option<Arc<HipWorkspace>>,
        expert_blocks: Option<&[usize]>,
        worker: Option<(&[u64], usize)>,
    ) -> Result<Self, AllpacaError> {
        let (dim, ff, n_experts) = (cfg.dim, cfg.ff_dim.max(1), cfg.n_experts);
        let hips: Vec<Arc<HipWorkspace>> = std::iter::once(hip).chain(second).collect();
        let on_gpu = |t: &&TensorInfo| match block_of(&t.name) {
            Some((_, suffix)) => BLOCK_MATRICES.contains(&suffix),
            None => GLOBAL_MATRICES.contains(&t.name.as_str()),
        };
        let wanted: Vec<&TensorInfo> = file.tensors.iter().filter(on_gpu).filter(|t| t.dims.len() <= 2).collect();

        // Sizes: the widest rows for the activation buffer, bytes for the placement.
        let (mut wide, mut low, mut shared_ff, mut io_k, mut io_n) = (dim, 1, 1, hc * dim, hc * dim + dim);
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
                        | "ffn_gate_inp.weight" | "indexer.q_proj.weight" | "indexer.k_proj.weight" => {}
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
        let sizes = ActSizes {
            wide,
            low,
            shared_ff,
            io_k,
            io_n: io_n.max(MAX_ROWS * cfg.vocab_size),
        };
        let act = layout(hc, dim, ff, n_experts, &sizes);

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
        let fixed = fixed + (act.total * 4 * hips.len()) as u64;
        // Blocks a worker can host: whole ones, from `first` on, as many as
        // its GPUs hold. Take as few as leave every other block its experts
        // on a card.
        let room = match worker.filter(|_| packed) {
            Some((free, first)) => free
                .iter()
                .map(|bytes| (bytes.saturating_sub(WORKER_RESERVE) / (dense + pool_bytes).max(1)) as usize)
                .sum::<usize>()
                .min(cfg.n_layers.saturating_sub(first)),
            None => 0,
        };
        let fits = |remote: usize| {
            let p = place(cfg.n_layers, &budgets, asked, fixed, dense, pool_bytes, remote);
            p.expert_blocks[0] + p.expert_blocks[1] + remote >= cfg.n_layers
        };
        let remote = (0..room).find(|r| fits(*r)).unwrap_or(room);
        let placement = place(cfg.n_layers, &budgets, asked, fixed, dense, pool_bytes, remote);
        let to_host: Vec<usize> = (cfg.n_layers - remote..cfg.n_layers).map(|layer| cfg.gguf_blk(layer)).collect();
        let hosted_block = |name: &str| block_of(name).is_some_and(|(blk, _)| to_host.contains(&blk));

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
        for info in wanted.iter().filter(|info| !hosted_block(&info.name)) {
            let (k, n) = shape(info);
            let device = cards[card_of(&info.name)].hip.device;
            let (encoding, bytes) = encode(file, info)?;
            let weights = DeviceWeights::upload_bytes(device, &[&bytes])?;
            dense_bytes += weights.bytes();
            matrices.insert(info.name.clone(), Matrix { weights, encoding, n, k });
        }
        let mut vectors = HashMap::new();
        let mut host_vectors = HashMap::new();
        for info in &file.tensors {
            let wanted = match block_of(&info.name) {
                Some((_, suffix)) => BLOCK_VECTORS.contains(&suffix),
                None => GLOBAL_VECTORS.contains(&info.name.as_str()),
            };
            if wanted && !hosted_block(&info.name) {
                let values = file.tensor_f32(&info.name)?;
                let device = cards[card_of(&info.name)].hip.device;
                vectors.insert(info.name.clone(), DeviceWeights::upload(device, &[&values])?);
            }
            let for_host = block_of(&info.name).is_some_and(|(_, suffix)| HOST_VECTORS.contains(&suffix));
            if for_host && !hosted_block(&info.name) {
                host_vectors.insert(info.name.clone(), file.tensor_f32(&info.name)?);
            }
        }
        let mut gpu = Self {
            cards,
            placement,
            matrices,
            vectors,
            host_vectors,
            act,
            hc,
            dim,
            ff,
            n_experts,
            eps: cfg.eps,
            packed,
            remote: Vec::new(),
            sizes,
            block_dense,
            to_host,
            hosted: None,
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
            let host = self.expert_slots(file, blk, slot)?;
            let pool = DeviceWeights::upload_bytes(self.cards[card].hip.device, &[&host])?;
            total += pool.bytes();
            self.cards[card].pools.insert(blk, pool);
        }
        Ok(total)
    }

    /// Block `blk`'s routed experts as slots: gate, up and down of expert
    /// `e` together, where the file stacks each of the three by expert.
    fn expert_slots(&self, file: &GgufFile, blk: usize, slot: usize) -> Result<Vec<u8>, AllpacaError> {
        let matrix = slot / 3;
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
        Ok(host)
    }

    /// Give the blocks left off the cards to worker `session` of `rt`:
    /// their matrices, vectors and routed experts, straight from the file.
    pub fn host_blocks(
        &mut self,
        file: &GgufFile,
        cfg: &ModelConfig,
        gdn: Option<GdnDims>,
        recurrent: &dyn Fn(usize) -> bool,
        qsa: Option<QsaShape>,
        rt: &Arc<crate::rpc_runtime::RpcRuntime>,
        session: usize,
    ) -> Result<(), AllpacaError> {
        use crate::qwen4_remote::{RemoteBlocks, Weight};
        let Some(scale) = crate::model::plain_softmax_routing(cfg).filter(|_| !self.to_host.is_empty()) else {
            return Ok(());
        };
        let slot = q2_0_expert_slot_bytes(self.dim, self.ff);
        let spec = HostedSpec {
            hc: self.hc,
            dim: self.dim,
            ff: self.ff,
            n_experts: self.n_experts,
            eps: self.eps,
            n_blocks: cfg.n_layers,
            // The worker multiplies nothing for the host.
            sizes: ActSizes { io_k: 16, io_n: 16, ..self.sizes },
            blocks: self
                .to_host
                .iter()
                .map(|&blk| HostedBlock {
                    blk,
                    recurrent: recurrent(blk),
                    bytes: self.block_dense.get(blk).copied().unwrap_or(0) + (self.n_experts * slot) as u64,
                })
                .collect(),
            shape: ChainShape {
                n_active: cfg.n_active,
                scale,
                gdn,
                attn: AttnShape::of(cfg, qsa),
            },
        };
        let host = RemoteBlocks::begin(rt, session, &spec)?;
        let mut bytes = 0u64;
        for info in &file.tensors {
            let Some((blk, suffix)) = block_of(&info.name) else {
                continue;
            };
            if !self.to_host.contains(&blk) {
                continue;
            }
            let name = info.name.clone();
            if BLOCK_MATRICES.contains(&suffix) && info.dims.len() <= 2 {
                let (k, n) = shape(info);
                let (encoding, data) = encode(file, info)?;
                bytes += data.len() as u64;
                host.send_weight(Weight::Matrix { name, encoding: encoding.code(), n, k }, &data)?;
            } else if BLOCK_VECTORS.contains(&suffix) || HOST_VECTORS.contains(&suffix) {
                let data: Vec<u8> = file.tensor_f32(&info.name)?.iter().flat_map(|v| v.to_le_bytes()).collect();
                host.send_weight(Weight::Vector { name }, &data)?;
            }
        }
        for &blk in &self.to_host {
            let slots = self.expert_slots(file, blk, slot)?;
            bytes += slots.len() as u64;
            host.send_weight(Weight::Pool { blk }, &slots)?;
        }
        tracing::info!(
            blocks = ?self.to_host,
            mb = bytes >> 20,
            from_worker_cache_mb = host.cache_bytes.load(std::sync::atomic::Ordering::Relaxed) >> 20,
            "qwen4exp: whole blocks on an RPC worker"
        );
        self.hosted = Some(Arc::new(host));
        Ok(())
    }

    /// The worker hosting this model's last blocks, when one does.
    pub fn hosted(&self) -> Option<&Arc<crate::qwen4_remote::RemoteBlocks>> {
        self.hosted.as_ref()
    }

    /// Blocks an RPC worker hosts whole.
    #[cfg(test)]
    pub fn hosted_blocks(&self) -> Vec<usize> {
        self.hosted.as_ref().map(|h| h.blocks().to_vec()).unwrap_or_default()
    }

    /// The worker's end: an empty chain for `spec`'s blocks on this
    /// machine's cards, each block on the first card with room for it. The
    /// weights follow ([`Self::add_matrix`], [`Self::add_vector`], [`Self::add_pool`]).
    pub fn for_worker(spec: &HostedSpec, hips: Vec<Arc<HipWorkspace>>) -> Result<Self, AllpacaError> {
        let act = layout(spec.hc, spec.dim, spec.ff, spec.n_experts, &spec.sizes);
        let mut room: Vec<u64> =
            hips.iter().map(|h| free_bytes(h.device).saturating_sub(HOST_RESERVE + (act.total * 4) as u64)).collect();
        let mut block_card = vec![0usize; spec.n_blocks];
        let mut card = 0usize;
        for block in &spec.blocks {
            while room.get(card).is_some_and(|free| *free < block.bytes) {
                card += 1;
            }
            let (Some(free), Some(home)) = (room.get_mut(card), block_card.get_mut(block.blk)) else {
                return Err(gpu_err(format!("qwen4exp: no room to host block {}", block.blk)));
            };
            *free -= block.bytes;
            *home = card;
        }
        let mut cards = Vec::new();
        for hip in hips {
            let act = DeviceWeights::alloc(hip.device, act.total * 4)?;
            cards.push(Card {
                hip,
                act,
                pools: HashMap::new(),
            });
        }
        Ok(Self {
            cards,
            placement: Placement {
                block_card,
                expert_blocks: [0, 0],
            },
            matrices: HashMap::new(),
            vectors: HashMap::new(),
            host_vectors: HashMap::new(),
            act,
            hc: spec.hc,
            dim: spec.dim,
            ff: spec.ff,
            n_experts: spec.n_experts,
            eps: spec.eps,
            packed: true,
            remote: Vec::new(),
            sizes: spec.sizes,
            block_dense: Vec::new(),
            to_host: Vec::new(),
            hosted: None,
        })
    }

    /// The card of each of `spec`'s blocks.
    pub fn hosted_cards(&self, spec: &HostedSpec) -> Vec<usize> {
        spec.blocks.iter().map(|b| self.card_of(b.blk)).collect()
    }

    fn device_of(&self, name: &str) -> i32 {
        self.cards[block_of(name).map_or(0, |(blk, _)| self.card_of(blk))].hip.device
    }

    /// A matrix in the encoding `code` ([`Encoding::code`]) its kernel reads.
    pub fn add_matrix(&mut self, name: &str, code: u8, n: usize, k: usize, bytes: &[u8]) -> Result<(), AllpacaError> {
        let encoding = Encoding::from_code(code).ok_or_else(|| gpu_err(format!("qwen4exp: {name} has encoding {code}")))?;
        let weights = DeviceWeights::upload_bytes(self.device_of(name), &[bytes])?;
        self.matrices.insert(name.to_string(), Matrix { weights, encoding, n, k });
        Ok(())
    }

    pub fn add_vector(&mut self, name: &str, values: &[f32]) -> Result<(), AllpacaError> {
        if block_of(name).is_some_and(|(_, suffix)| HOST_VECTORS.contains(&suffix)) {
            self.host_vectors.insert(name.to_string(), values.to_vec());
            return Ok(());
        }
        let weights = DeviceWeights::upload(self.device_of(name), &[values])?;
        self.vectors.insert(name.to_string(), weights);
        Ok(())
    }

    /// Block `blk`'s routed experts as Q2_0 slots.
    pub fn add_pool(&mut self, blk: usize, slots: &[u8]) -> Result<(), AllpacaError> {
        if slots.len() != self.n_experts * q2_0_expert_slot_bytes(self.dim, self.ff) {
            return Err(gpu_err(format!("qwen4exp: block {blk}'s expert slots are {} bytes", slots.len())));
        }
        let card = self.card_of(blk);
        let pool = DeviceWeights::upload_bytes(self.cards[card].hip.device, &[slots])?;
        self.cards[card].pools.insert(blk, pool);
        Ok(())
    }

    /// Run `blocks` in order on the streams of `rows` tokens from `pos`, in
    /// place, without the host: each block routes on its card.
    pub fn run_blocks(
        &self,
        blocks: &[HostedBlock],
        shape: &ChainShape,
        pos: usize,
        rows: usize,
        state: &mut Qwen4DeviceState,
        streams: &mut [f32],
    ) -> Result<(), AllpacaError> {
        let Some(first) = blocks.first() else {
            return Ok(());
        };
        // A token's result must not depend on how many share its pass.
        let _exact = allpaca_hip::ExactRows::enable();
        let mut card = self.card_of(first.blk);
        self.set_streams(card, streams)?;
        let mut x = Vec::new();
        for block in blocks {
            let (blk, home) = (block.blk, self.card_of(block.blk));
            if home != card {
                self.get_streams(card, streams)?;
                self.set_streams(home, streams)?;
                card = home;
            }
            self.mix(card, &format!("blk.{blk}.hc_attn"), true, rows)?;
            if block.recurrent {
                let dims = shape.gdn.ok_or_else(|| gpu_err("qwen4exp: no Gated DeltaNet shape".into()))?;
                self.gdn(blk, &dims, state, rows)?;
            } else {
                self.attention(blk, &shape.attn, pos, state, rows)?;
            }
            self.combine(card, rows)?;
            self.mix(card, &format!("blk.{blk}.hc_ffn"), true, rows)?;
            self.ffn_route(blk, rows, None, &mut x)?;
            self.ffn_finish_routed(blk, rows, shape.n_active, shape.scale)?;
            self.combine(card, rows)?;
        }
        self.get_streams(card, streams)
    }

    /// Put the routed experts of the blocks that have none on a card on
    /// `rpc`'s workers, as far as those have room; the rest stay on the host.
    pub fn send_experts(
        &mut self,
        file: &GgufFile,
        cfg: &ModelConfig,
        rpc: &crate::rpc_runtime::RpcRuntime,
    ) -> Result<(), AllpacaError> {
        if !self.packed || rpc.session_count() == 0 {
            return Ok(());
        }
        let slot = q2_0_expert_slot_bytes(self.dim, self.ff);
        let mut bytes = 0u64;
        for layer in 0..cfg.n_layers {
            let blk = cfg.gguf_blk(layer);
            if self.has_experts(blk) {
                continue;
            }
            let slots = self.expert_slots(file, blk, slot)?;
            if !rpc.put_expert_slots(layer, &slots, self.n_experts, self.dim, self.ff)? {
                break;
            }
            bytes += slots.len() as u64;
            self.remote.push(blk);
        }
        tracing::info!(blocks = self.remote.len(), expert_mb = bytes >> 20, "qwen4exp: routed experts on RPC workers");
        Ok(())
    }

    /// Block `blk`'s routed experts are on an RPC worker.
    pub fn remote_experts(&self, blk: usize) -> bool {
        self.remote.contains(&blk)
    }

    /// Blocks whose routed experts are on an RPC worker.
    #[cfg(test)]
    pub fn remote_expert_layers(&self) -> Vec<usize> {
        self.remote.clone()
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

    /// `rows` rows times `m` on `card`. With the profile on, the product is
    /// timed alone on the card, over [`PROFILE_REPEATS`] launches.
    fn launch(&self, card: &Card, m: &Matrix, a: *const f32, c: *mut f32, rows: usize) -> Result<(), AllpacaError> {
        let Some(profile) = profile() else {
            return self.launch_once(card, m, a, c, rows);
        };
        self.step(card, "one matrix alone", || self.launch_once(card, m, a, c, rows))?;
        card.hip.bind()?;
        allpaca_hip::stream_synchronize()?;
        let clock = std::time::Instant::now();
        for _ in 0..PROFILE_REPEATS {
            self.launch_once(card, m, a, c, rows)?;
        }
        card.hip.bind()?;
        allpaca_hip::stream_synchronize()?;
        let seconds = clock.elapsed().as_secs_f64() / f64::from(PROFILE_REPEATS);
        let kind = format!("{:?} {}x{} rows {rows}", m.encoding, m.n, m.k);
        let mut kinds = profile.lock().unwrap_or_else(|e| e.into_inner());
        let entry = kinds.entry(kind).or_default();
        *entry = (entry.0 + 1, entry.1 + seconds);
        Ok(())
    }

    /// Run one kernel launch `f`; with the profile on, alone on the card
    /// and timed under `kind` (the time includes the wait for it to end).
    fn step(&self, card: &Card, kind: &str, f: impl FnOnce() -> Result<(), AllpacaError>) -> Result<(), AllpacaError> {
        let Some(profile) = profile() else {
            return f();
        };
        card.hip.bind()?;
        allpaca_hip::stream_synchronize()?;
        let clock = std::time::Instant::now();
        f()?;
        card.hip.bind()?;
        allpaca_hip::stream_synchronize()?;
        let seconds = clock.elapsed().as_secs_f64();
        let mut kinds = profile.lock().unwrap_or_else(|e| e.into_inner());
        let entry = kinds.entry(format!("kernel {kind}")).or_default();
        *entry = (entry.0 + 1, entry.1 + seconds);
        Ok(())
    }

    fn launch_once(&self, card: &Card, m: &Matrix, a: *const f32, c: *mut f32, rows: usize) -> Result<(), AllpacaError> {
        let hip = &card.hip;
        let (r, n, k) = (rows as i32, m.n as i32, m.k as i32);
        let b = m.weights.byte_ptr_at(0);
        match m.encoding {
            Encoding::F32 => hip.gemm_nt_dev(a, b.cast(), c, r, n, k),
            Encoding::F16 => hip.gemm_nt_f16_dev(a, b.cast(), c, r, n, k),
            Encoding::Bf16 => hip.gemm_nt_bf16_dev(a, b.cast(), c, r, n, k),
            Encoding::Q2_0 => hip.gemm_nt_q2_0_dev(a, b, c, r, n, k),
            Encoding::Q3K => hip.gemm_nt_q3_k_dev(a, b, c, r, n, k),
            Encoding::Q4K => hip.gemm_nt_q4_k_dev(a, b, c, r, n, k),
            Encoding::Q5K => hip.gemm_nt_q5_k_dev(a, b, c, r, n, k),
            Encoding::Q6K => hip.gemm_nt_q6_k_dev(a, b, c, r, n, k),
            Encoding::Q8 => hip.gemm_nt_q8_dev(a, b, c, r, n, k),
        }
    }

    fn check_rows(&self, rows: usize) -> Result<(), AllpacaError> {
        if rows == 0 || rows > MAX_ROWS {
            return Err(gpu_err(format!("qwen4exp: a pass takes 1 to {MAX_ROWS} tokens, got {rows}")));
        }
        Ok(())
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
            self.launch(card, m, card.at(self.act.io_in), card.at(at), 1)?;
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

    /// Wait for everything queued on every card.
    pub fn sync_all(&self) -> Result<(), AllpacaError> {
        (0..self.cards.len()).try_for_each(|card| self.sync(card))
    }

    /// The streams of the pass's tokens, token-major, to `card`.
    pub fn set_streams(&self, card: usize, streams: &[f32]) -> Result<(), AllpacaError> {
        self.check_rows(streams.len() / (self.hc * self.dim))?;
        let card = &self.cards[card];
        card.hip.copy_h2d_f32(card.at(self.act.streams), streams)
    }

    pub fn get_streams(&self, card: usize, streams: &mut [f32]) -> Result<(), AllpacaError> {
        self.check_rows(streams.len() / (self.hc * self.dim))?;
        let card = &self.cards[card];
        card.hip.copy_d2h_f32(card.at(self.act.streams), streams)
    }

    /// Collapse the card's streams into the block inputs of `rows` tokens
    /// through the mixer `prefix` (`blk.N.hc_attn`, `output_hc`); `inject`
    /// also computes the logits [`Self::combine`] weights the block's
    /// output with.
    pub fn mix(&self, card: usize, prefix: &str, inject: bool, rows: usize) -> Result<(), AllpacaError> {
        self.check_rows(rows)?;
        let (card, act, hc_dim) = (&self.cards[card], &self.act, self.hc * self.dim);
        let (r, hc, dim) = (rows as i32, self.hc as i32, self.dim as i32);
        let down = self.matrix(&format!("{prefix}_down.weight"), hc_dim)?;
        let up = self.matrix(&format!("{prefix}_up.weight"), down.n)?;
        let gains = self.vector(&format!("{prefix}_norm.weight"))?;
        card.hip.bind()?;
        self.step(card, "stream_norm", || q4x::stream_norm(card.at(act.streams), gains, card.at(act.xn), r, hc, dim, self.eps))?;
        self.launch(card, down, card.at(act.xn), card.at(act.low), rows)?;
        self.step(card, "silu_scale", || silu_scale_device(card.at(act.low), (rows * down.n) as i32, 1.0 / self.hc as f32))?;
        self.launch(card, up, card.at(act.low), card.at(act.gate), rows)?;
        card.hip.bind()?;
        self.step(card, "gated_mean", || q4x::gated_mean(card.at(act.xn), card.at(act.gate), card.at(act.x), r, hc, dim))?;
        if inject {
            let w = self.matrix(&format!("{prefix}_inject.weight"), hc_dim)?;
            self.launch(card, w, card.at(act.xn), card.at(act.inject), rows)?;
        }
        Ok(())
    }

    /// Add the block's outputs to the card's streams, weighted by the
    /// logits of the last [`Self::mix`].
    pub fn combine(&self, card: usize, rows: usize) -> Result<(), AllpacaError> {
        let (card, act) = (&self.cards[card], &self.act);
        card.hip.bind()?;
        self.step(card, "combine", || {
            q4x::combine(
                card.at(act.streams),
                card.at(act.out),
                card.at(act.inject),
                rows as i32,
                self.hc as i32,
                self.dim as i32,
            )
        })
    }

    fn block_state<'a>(&self, state: &'a mut Qwen4DeviceState, blk: usize) -> &'a mut BlockState {
        if state.blocks.len() <= blk {
            state.blocks.resize_with(blk + 1, BlockState::default);
        }
        &mut state.blocks[blk]
    }

    /// Gated DeltaNet block `blk` on the mixed inputs of `rows` tokens: the
    /// products for all of them at once, then one recurrent step per token,
    /// with a sigmoid output gate.
    pub fn gdn(&self, blk: usize, dims: &GdnDims, state: &mut Qwen4DeviceState, rows: usize) -> Result<(), AllpacaError> {
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
        self.launch(card, self.matrix(&name("attn_qkv.weight"), dim)?, x, qkv, rows)?;
        self.launch(card, self.matrix(&name("attn_gate.weight"), dim)?, x, z, rows)?;
        self.launch(card, self.matrix(&name("ssm_alpha.weight"), dim)?, x, alpha, rows)?;
        self.launch(card, self.matrix(&name("ssm_beta.weight"), dim)?, x, beta, rows)?;
        let (conv_w, dt_bias, a) = (
            self.vector(&name("ssm_conv1d.weight"))?,
            self.vector(&name("ssm_dt.bias"))?,
            self.vector(&name("ssm_a"))?,
        );
        for r in 0..rows {
            // SAFETY: each scratch row holds MAX_ROWS rows of at least these widths.
            let (qkv, alpha, beta, y) = unsafe {
                (
                    qkv.add(r * dims.qkv_dim),
                    alpha.add(r * dims.n_v_heads),
                    beta.add(r * dims.n_v_heads),
                    y.add(r * dims.value_dim),
                )
            };
            self.step(card, "gdn_layer_step", || {
                card.hip.gdn_layer_step(
                    qkv,
                    alpha,
                    beta,
                    conv_w,
                    conv,
                    dt_bias,
                    a,
                    rec,
                    y,
                    dims.n_v_heads as i32,
                    dims.n_k_heads as i32,
                    dims.head_v as i32,
                    dims.head_k as i32,
                    dims.conv_kernel as i32,
                    self.eps,
                    false,
                )
            })?;
        }
        let gains = self.vector(&name("ssm_norm.weight"))?;
        self.step(card, "gdn rmsnorm_heads", || {
            card.hip.rmsnorm_heads_device(y, gains, (rows * dims.n_v_heads) as i32, dims.head_v as i32, self.eps)
        })?;
        self.step(card, "gdn sigmoid_mul", || card.hip.sigmoid_mul_device(y, z, (rows * dims.value_dim) as i32))?;
        self.launch(card, self.matrix(&name("ssm_out.weight"), dims.value_dim)?, y, card.at(act.out), rows)
    }

    fn host_vector(&self, name: &str) -> Result<&[f32], AllpacaError> {
        self.host_vectors
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| gpu_err(format!("qwen4exp: {name} is not held on the host")))
    }

    /// Full-attention block `blk` on the mixed inputs of `rows` tokens from
    /// position `pos`: gated queries, normed and rotated keys, each token
    /// attending to the sequence up to itself, or, past its indexer's
    /// budget, to the cells the indexer keeps.
    pub fn attention(
        &self,
        blk: usize,
        cfg: &AttnShape,
        pos: usize,
        state: &mut Qwen4DeviceState,
        rows: usize,
    ) -> Result<(), AllpacaError> {
        let (card, act, dim) = (&self.cards[self.card_of(blk)], &self.act, self.dim);
        let hip = &card.hip;
        let (head_dim, n_heads, n_kv) = (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads.max(1));
        let (attn, kv) = (n_heads * head_dim, n_kv * head_dim);
        let name = |s: &str| format!("blk.{blk}.{s}");
        let qsa = cfg.qsa.filter(|_| self.matrices.contains_key(&name("indexer.k_proj.weight")));
        if qsa.is_some_and(|shape| shape.heads * shape.dim > dim) {
            return Err(gpu_err(format!("qwen4exp: block {blk}'s indexer queries are wider than the model")));
        }
        let sparse = qsa.filter(|shape| shape.is_sparse(pos + rows - 1));
        // Row numbers, keys and values of the cells a token keeps.
        let picked = match &sparse {
            Some(shape) => {
                let cells = shape.max_cells().next_multiple_of(16);
                let buf = match state.picked.entry(hip.device) {
                    std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(DeviceWeights::alloc(hip.device, cells * (1 + 2 * kv) * 4)?)
                    }
                };
                let at = |floats: usize| buf.ptr_at(floats * 4).cast_mut();
                Some((at(0), at(cells), at(cells * (1 + kv))))
            }
            None => None,
        };
        let state = self.block_state(state, blk);
        if state.kv_tokens < pos + rows {
            let tokens = (2 * state.kv_tokens).max(KV_START_TOKENS).max(pos + rows);
            let key_dim = qsa.map_or(0, |shape| shape.dim);
            for (buf, width) in [(&mut state.k, kv), (&mut state.v, kv), (&mut state.ik, key_dim)] {
                if width == 0 {
                    continue;
                }
                *buf = Some(match buf.as_ref() {
                    Some(old) => copy_of(old, tokens * width * 4)?,
                    None => DeviceWeights::alloc(hip.device, tokens * width * 4)?,
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
        self.launch(card, self.matrix(&name("attn_q.weight"), dim)?, x, q_full, rows)?;
        self.launch(card, self.matrix(&name("attn_k.weight"), dim)?, x, k, rows)?;
        self.launch(card, self.matrix(&name("attn_v.weight"), dim)?, x, v, rows)?;
        let (all_heads, all_kv) = ((rows * n_heads) as i32, (rows * n_kv) as i32);
        hip.deinterleave_q_gate_device(q_full, q, gate, all_heads, head_dim as i32)?;
        hip.rmsnorm_heads_device(q, self.vector(&name("attn_q_norm.weight"))?, all_heads, head_dim as i32, self.eps)?;
        hip.rmsnorm_heads_device(k, self.vector(&name("attn_k_norm.weight"))?, all_kv, head_dim as i32, self.eps)?;
        let n_rot = if cfg.rope_dims > 0 { cfg.rope_dims } else { head_dim };
        for r in 0..rows {
            // SAFETY: the scratch rows hold `rows` rows of `attn` and `kv` floats.
            let (q, k) = unsafe { (q.add(r * attn), k.add(r * kv)) };
            hip.rope_imrope_qk_device(
                q,
                k,
                n_heads as i32,
                n_kv as i32,
                head_dim as i32,
                n_rot as i32,
                [(pos + r) as i32; 3],
                cfg.rope_theta,
                cfg.rope_sections,
            )?;
        }
        // SAFETY: the buffers hold `kv_tokens >= pos + rows` rows of `kv` floats.
        let (key_rows, value_rows) = unsafe { (keys.add(pos * kv), values.add(pos * kv)) };
        hip.copy_d2d_f32(key_rows, k, rows * kv)?;
        hip.copy_d2d_f32(value_rows, v, rows * kv)?;
        // The indexer's raw keys stay on the card until a token needs them.
        // Its products use rows the feed-forward fills later.
        let (index_q, index_k) = (card.at(act.routed), card.at(act.shared));
        if let Some(shape) = &qsa {
            let raw = state.ik.as_ref().ok_or_else(|| gpu_err(format!("qwen4exp: block {blk} has no indexer keys")))?;
            self.launch(card, self.matrix(&name("indexer.k_proj.weight"), dim)?, x, index_k, rows)?;
            hip.copy_d2d_f32(raw.ptr_at(pos * shape.dim * 4).cast_mut(), index_k, rows * shape.dim)?;
        }
        // The interleaved queries are spent: their rows take the output.
        let attend = |q: *const f32, keys: *const f32, values: *const f32, out: *mut f32, rows: usize, seq: usize| {
            self.step(card, "attention", || {
                hip.attention_device(
                    q,
                    keys,
                    values,
                    out,
                    rows as i32,
                    seq as i32,
                    n_heads as i32,
                    n_kv as i32,
                    head_dim as i32,
                    true,
                )
            })
        };
        match (&sparse, picked) {
            (Some(shape), Some((rows_at, keys_at, values_at))) => {
                self.launch(card, self.matrix(&name("indexer.q_proj.weight"), dim)?, x, index_q, rows)?;
                // The keys the host has not read yet, then the queries.
                let have = state.qsa.tokens(shape);
                let raw = state.ik.as_ref().ok_or_else(|| gpu_err(format!("qwen4exp: block {blk} has no indexer keys")))?;
                let mut fresh = vec![0.0f32; (pos + rows - have) * shape.dim];
                hip.copy_d2h_f32(raw.ptr_at(have * shape.dim * 4), &mut fresh)?;
                state.qsa.push(shape, &fresh, self.host_vector(&name("indexer.k_norm.weight"))?)?;
                let width = shape.heads * shape.dim;
                let mut queries = vec![0.0f32; rows * width];
                hip.copy_d2h_f32(index_q, &mut queries)?;
                let q_norm = self.host_vector(&name("indexer.q_norm.weight"))?;
                // One token at a time: each keeps its own cells.
                for (r, query) in queries.chunks_mut(width).enumerate() {
                    // SAFETY: the scratch rows hold `rows` rows of `attn` floats.
                    let (q, out) = unsafe { (q.add(r * attn), q_full.add(r * attn)) };
                    match state.qsa.select(shape, pos + r, query, q_norm)? {
                        None => attend(q, keys, values, out, 1, pos + r + 1)?,
                        Some(cells) => {
                            // SAFETY: row numbers are plain 32-bit values, copied as such.
                            let bits = unsafe { std::slice::from_raw_parts(cells.as_ptr().cast::<f32>(), cells.len()) };
                            hip.copy_h2d_f32(rows_at, bits)?;
                            let n = cells.len() as i32;
                            hip.gather_rows_device(keys_at, keys, rows_at.cast(), n, kv as i32)?;
                            hip.gather_rows_device(values_at, values, rows_at.cast(), n, kv as i32)?;
                            attend(q, keys_at, values_at, out, 1, cells.len())?;
                        }
                    }
                }
            }
            _ => attend(q, keys, values, q_full, rows, pos + rows)?,
        }
        hip.sigmoid_mul_device(q_full, gate, (rows * attn) as i32)?;
        self.launch(card, self.matrix(&name("attn_output.weight"), attn)?, q_full, card.at(act.out), rows)
    }

    /// Whether block `blk`'s routed experts can be picked on its card
    /// ([`Self::ffn_finish_routed`]).
    pub fn routes_on_card(&self, blk: usize, n_active: usize) -> bool {
        self.has_experts(blk) && self.n_experts <= 1024 && (1..=MAX_PICKS.min(self.n_experts)).contains(&n_active)
    }

    /// Start block `blk`'s feed-forward on the mixed inputs of `rows`
    /// tokens: the shared expert runs on the card, and the router's scores
    /// come back in `scores` (a row per token) unless it is `None` (the
    /// card will route, [`Self::ffn_finish_routed`]). `x` also gets the
    /// mixed inputs when the experts are not on a card.
    pub fn ffn_route(
        &self,
        blk: usize,
        rows: usize,
        scores: Option<&mut [f32]>,
        x: &mut Vec<f32>,
    ) -> Result<(), AllpacaError> {
        let (card, act, dim) = (&self.cards[self.card_of(blk)], &self.act, self.dim);
        let name = |s: &str| format!("blk.{blk}.{s}");
        let (input, n_scores) = (card.at(act.x), rows * self.n_experts);
        // SAFETY: the input part holds MAX_ROWS rows of inputs and of scores.
        let device_scores = unsafe { input.add(rows * dim) };
        self.launch(card, self.matrix(&name("ffn_gate_inp.weight"), dim)?, input, device_scores, rows)?;
        if self.has(&name("ffn_up_shexp.weight")) {
            let gate = self.matrix(&name("ffn_gate_shexp.weight"), dim)?;
            let (g, u) = (card.at(act.shared_gate), card.at(act.shared_up));
            self.launch(card, gate, input, g, rows)?;
            self.launch(card, self.matrix(&name("ffn_up_shexp.weight"), dim)?, input, u, rows)?;
            self.step(card, "silu_mul", || card.hip.silu_mul_device(g, u, (rows * gate.n) as i32))?;
            self.launch(card, self.matrix(&name("ffn_down_shexp.weight"), gate.n)?, g, card.at(act.shared), rows)?;
            let open = self.matrix(&name("ffn_gate_inp_shexp.weight"), dim)?;
            self.launch(card, open, input, card.at(act.shared_open), rows)?;
        }
        x.clear();
        let Some(scores) = scores else {
            return match card.pools.contains_key(&blk) {
                true => Ok(()),
                false => Err(gpu_err(format!("qwen4exp: block {blk} has no experts on its card to route to"))),
            };
        };
        if card.pools.contains_key(&blk) {
            return self.step(card, "download scores", || card.hip.copy_d2h_f32(device_scores, &mut scores[..n_scores]));
        }
        x.resize(rows * dim + n_scores, 0.0);
        self.step(card, "download block input and scores", || card.hip.copy_d2h_f32(input, x))?;
        scores[..n_scores].copy_from_slice(&x[rows * dim..]);
        x.truncate(rows * dim);
        Ok(())
    }

    /// Finish block `blk`'s feed-forward with the routing on the card:
    /// each token's `n_active` best experts by the scores of
    /// [`Self::ffn_route`], weights times `scale`, plus the gated shared
    /// expert. Nothing comes back to the host.
    pub fn ffn_finish_routed(&self, blk: usize, rows: usize, n_active: usize, scale: f32) -> Result<(), AllpacaError> {
        let (card, act, dim) = (&self.cards[self.card_of(blk)], &self.act, self.dim);
        self.check_rows(rows)?;
        let pool = match card.pools.get(&blk) {
            Some(pool) if self.routes_on_card(blk, n_active) => pool,
            _ => return Err(gpu_err(format!("qwen4exp: block {blk} cannot route {n_active} experts on its card"))),
        };
        card.hip.bind()?;
        // SAFETY: the input part holds MAX_ROWS rows; the scores follow the inputs in use.
        let scores = unsafe { card.at(act.x).add(rows * dim) };
        // Every token in one launch; each one's result is the same as alone.
        self.step(card, "routed_experts", || {
            q4x::routed_experts(
                card.at(act.x),
                scores,
                pool.byte_ptr_at(0),
                self.n_experts as i32,
                n_active as i32,
                scale,
                card.at(act.picks).cast(),
                card.at(act.expert_h),
                card.at(act.routed),
                dim as i32,
                self.ff as i32,
                rows as i32,
            )
        })?;
        self.ffn_sum(blk, rows)
    }

    /// The block outputs: the routed experts' sums plus the gated shared
    /// expert.
    fn ffn_sum(&self, blk: usize, rows: usize) -> Result<(), AllpacaError> {
        let (card, act, dim) = (&self.cards[self.card_of(blk)], &self.act, self.dim);
        let routed = card.at(act.routed);
        let shared = self.has(&format!("blk.{blk}.ffn_up_shexp.weight"));
        card.hip.bind()?;
        self.step(card, "ffn_sum", || {
            q4x::ffn_sum(
                routed,
                if shared { card.at(act.shared).cast_const() } else { std::ptr::null() },
                if shared { card.at(act.shared_open).cast_const() } else { std::ptr::null() },
                card.at(act.out),
                rows as i32,
                dim as i32,
            )
        })
    }

    /// Finish block `blk`'s feed-forward into the block outputs: each
    /// token's routed experts `picks` from the card's slots, or
    /// `host_routed` (their sums, a row per token, computed on the host),
    /// plus the gated shared expert.
    pub fn ffn_finish(
        &self,
        blk: usize,
        picks: &[Vec<(usize, f32)>],
        host_routed: Option<&[f32]>,
    ) -> Result<(), AllpacaError> {
        let (card, act, dim, rows) = (&self.cards[self.card_of(blk)], &self.act, self.dim, picks.len());
        self.check_rows(rows)?;
        let routed = card.at(act.routed);
        match (card.pools.get(&blk), host_routed) {
            (_, Some(sums)) if sums.len() == rows * dim => {
                self.step(card, "upload expert sums", || card.hip.copy_h2d_f32(routed, sums))?
            }
            (Some(pool), None) if picks.iter().all(|p| !p.is_empty() && p.len() <= MAX_PICKS) => {
                card.hip.bind()?;
                for (r, picks) in picks.iter().enumerate() {
                    let slots: Vec<i32> = picks.iter().map(|(e, _)| *e as i32).collect();
                    let weights: Vec<f32> = picks.iter().map(|(_, w)| *w).collect();
                    // SAFETY: the input and routed parts hold MAX_ROWS rows of `dim` floats.
                    let (x, out) = unsafe { (card.at(act.x).add(r * dim), routed.add(r * dim)) };
                    self.step(card, "q2_0_moe_ffn", || {
                        q2_0_moe_ffn_device(
                            x,
                            pool.byte_ptr_at(0),
                            &slots,
                            &weights,
                            card.at(act.expert_h),
                            out,
                            dim as i32,
                            self.ff as i32,
                        )
                    })?;
                }
            }
            _ => return Err(gpu_err(format!("qwen4exp: block {blk} has no routed experts to run"))),
        }
        self.ffn_sum(blk, rows)
    }

    /// The logits of card 0's mixed inputs through the output matrix
    /// `name`: of the last of `rows` tokens when `logits` holds one row, of
    /// every token when it holds `rows` rows.
    pub fn head(&self, name: &str, rows: usize, logits: &mut [f32]) -> Result<(), AllpacaError> {
        let (card, act) = (&self.cards[0], &self.act);
        let m = self.matrix(name, self.dim)?;
        self.check_rows(rows)?;
        let out_rows = if logits.len() >= rows * m.n { rows } else { 1 };
        if out_rows * m.n > act.io_out_floats || logits.len() < m.n {
            return Err(gpu_err(format!("qwen4exp: {} logits do not fit", out_rows * m.n)));
        }
        // SAFETY: the input part holds `rows` rows of `dim` floats.
        let x = unsafe { card.at(act.x).add((rows - out_rows) * self.dim) };
        self.launch(card, m, x, card.at(act.io_out), out_rows)?;
        card.hip.copy_d2h_f32(card.at(act.io_out), &mut logits[..out_rows * m.n])
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
    // A worker that hosts whole blocks takes those after the last n-gram
    // block (that one needs the host) when the routing can run on a card.
    // `ALLPACA_Q4X_NO_REMOTE_BLOCKS` keeps every block here.
    let worker = model
        .rpc
        .as_ref()
        .filter(|_| std::env::var_os("ALLPACA_Q4X_NO_REMOTE_BLOCKS").is_none())
        .filter(|_| crate::model::plain_softmax_routing(&model.config).is_some())
        .and_then(|rt| rt.extension_host(crate::qwen4_remote::EXT).map(|(session, free)| (rt, session, free)));
    let first_free = q4.spec.ple.as_ref().and_then(|p| p.layers.iter().max()).map_or(0, |last| last + 1);
    let mut gpu = Qwen4Gpu::build(
        q4.gguf()?,
        &model.config,
        q4.spec.hc,
        hip.clone(),
        second.cloned(),
        expert_blocks,
        worker.as_ref().map(|(_, _, free)| (free.as_slice(), first_free)),
    )?;
    if let Some((rt, session, _)) = worker {
        let recurrent = |blk: usize| q4.spec.is_recurrent(blk);
        // One indexer shape serves every block that has one.
        let mut qsa = None;
        for layer in 0..model.config.n_layers {
            if let Some(shape) = crate::qwen4exp::qsa_shape(q4, &model.config, model.config.gguf_blk(layer))? {
                qsa = Some(shape);
                break;
            }
        }
        gpu.host_blocks(q4.gguf()?, &model.config, q4.spec.gdn, &recurrent, qsa, rt, session)?;
    }
    if let Some(rpc) = model.rpc.as_deref().filter(|_| gpu.hosted().is_none()) {
        gpu.send_experts(q4.gguf()?, &model.config, rpc)?;
    }
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
        let p = place(48, &[9 * GB, 16 * GB], None, GB, GB / 10, 7 * GB / 10, 0);
        assert_eq!(p.expert_blocks, [7, 20]);
        assert_eq!(p.block_card[..7], [0; 7]);
        assert_eq!(p.block_card[7..27], [1; 20]);
        assert_eq!(p.block_card[27..], [0; 21]);
        // One card: every block's matrices are there, experts as far as they fit.
        let p = place(48, &[9 * GB], None, GB, GB / 10, 7 * GB / 10, 0);
        assert_eq!((p.expert_blocks, p.block_card.iter().sum::<usize>()), ([4, 0], 0));
        // Experts that cannot go on a card: matrices only.
        assert_eq!(place(4, &[9 * GB, 9 * GB], None, 0, GB, 0, 0).expert_blocks, [0, 0]);
        // An explicit split is taken as given, within the block count.
        let p = place(4, &[GB, GB], Some(&[1, 2]), 0, GB, GB, 0);
        assert_eq!((p.expert_blocks, p.block_card), ([1, 2], vec![0, 1, 1, 0]));
        assert_eq!(place(4, &[GB, GB], Some(&[9, 9]), 0, GB, GB, 0).expert_blocks, [0, 4]);
    }
}
