//! `qwen4exp` blocks run whole on an RPC worker: the worker holds the
//! blocks' matrices and routed experts on its GPUs and each sequence's state
//! for them, so a pass sends the residual streams once and gets them back
//! once, instead of stopping at every block.
//!
//! Both ends are here. The worker serves the extension [`EXT`]
//! ([`worker_extension`]); the coordinator fills it and calls it through
//! [`RemoteBlocks`]. A call's head is a JSON [`Call`], its payload raw bytes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use allpaca_hip::HipWorkspace;
use allpaca_types::{AllpacaError, FatalError};
use serde::{Deserialize, Serialize};

use crate::qwen4_gpu::{HostedSpec, Qwen4DeviceState, Qwen4Gpu};
use crate::rpc_runtime::RpcRuntime;

/// The extension's name in a worker's HELLO.
pub const EXT: &str = "q4x-blocks";

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum Call {
    /// Start over with these blocks; the payloads follow.
    Begin(Box<HostedSpec>),
    /// A matrix in the encoding its kernel reads (payload: its bytes).
    Matrix { name: String, encoding: u8, n: usize, k: usize },
    /// An F32 vector (payload: little-endian floats).
    Vector { name: String },
    /// A block's routed experts as Q2_0 slots (payload).
    Pool { blk: usize },
    /// Run the blocks on `rows` tokens from `pos` of sequence `seq`
    /// (payload and reply: the streams, little-endian floats).
    Run { seq: u64, pos: usize, rows: usize },
    /// Sequence `dst` becomes a copy of `src`.
    CloneSeq { src: u64, dst: u64 },
    DropSeq { seq: u64 },
}

fn err(msg: String) -> AllpacaError {
    AllpacaError::Fatal(FatalError::ModelLoad(msg))
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn bytes_of(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The worker's end: the hosted blocks and each sequence's state.
#[derive(Default)]
struct BlockHost {
    gpu: Option<Qwen4Gpu>,
    spec: Option<HostedSpec>,
    seqs: HashMap<u64, Qwen4DeviceState>,
}

impl BlockHost {
    fn serve(&mut self, head: &[u8], data: &[u8]) -> Result<Vec<u8>, AllpacaError> {
        let call: Call = serde_json::from_slice(head).map_err(|e| err(format!("qwen4exp block call: {e}")))?;
        if let Call::Begin(spec) = call {
            self.seqs.clear();
            self.gpu = None;
            let hips = (0..allpaca_hip::device_count())
                .map(|d| HipWorkspace::new(d, 8 * 1024 * 1024, 256).map(Arc::new))
                .collect::<Result<Vec<_>, _>>()?;
            let gpu = Qwen4Gpu::for_worker(&spec, hips)?;
            tracing::info!(blocks = spec.blocks.len(), cards = ?gpu.hosted_cards(&spec), "qwen4exp: hosting blocks");
            (self.gpu, self.spec) = (Some(gpu), Some(*spec));
            return Ok(Vec::new());
        }
        let (Some(gpu), Some(spec)) = (self.gpu.as_mut(), self.spec.as_ref()) else {
            return Err(err("qwen4exp block call before Begin".into()));
        };
        match call {
            Call::Begin(_) => {}
            Call::Matrix { name, encoding, n, k } => gpu.add_matrix(&name, encoding, n, k, data)?,
            Call::Vector { name } => gpu.add_vector(&name, &floats(data))?,
            Call::Pool { blk } => gpu.add_pool(blk, data)?,
            Call::Run { seq, pos, rows } => {
                let mut streams = floats(data);
                if rows == 0 || streams.len() != rows * spec.hc * spec.dim {
                    return Err(err(format!("qwen4exp block run: {} floats for {rows} rows", streams.len())));
                }
                let state = self.seqs.entry(seq).or_default();
                gpu.run_blocks(&spec.blocks, &spec.shape, pos, rows, state, &mut streams)?;
                return Ok(bytes_of(&streams));
            }
            Call::CloneSeq { src, dst } => match self.seqs.get(&src).cloned() {
                Some(state) => drop(self.seqs.insert(dst, state)),
                None => drop(self.seqs.remove(&dst)),
            },
            Call::DropSeq { seq } => drop(self.seqs.remove(&seq)),
        }
        Ok(Vec::new())
    }
}

impl allpaca_rpc::Extension for BlockHost {
    fn name(&self) -> &str {
        EXT
    }

    fn call(&mut self, head: &[u8], data: &[u8]) -> Result<Vec<u8>, String> {
        self.serve(head, data).map_err(|e| e.to_string())
    }
}

/// The extension a HIP worker serves to host `qwen4exp` blocks.
pub fn worker_extension() -> Box<dyn allpaca_rpc::Extension> {
    Box::<BlockHost>::default()
}

/// The coordinator's end: blocks hosted on one worker.
#[derive(Debug)]
pub struct RemoteBlocks {
    rt: Arc<RpcRuntime>,
    session: usize,
    blks: Vec<usize>,
    next_seq: AtomicU64,
}

impl RemoteBlocks {
    /// Start hosting on `session` of `rt`: the worker forgets what it had.
    pub(crate) fn begin(rt: &Arc<RpcRuntime>, session: usize, spec: &HostedSpec) -> Result<Self, AllpacaError> {
        let host = Self {
            rt: rt.clone(),
            session,
            blks: spec.blocks.iter().map(|b| b.blk).collect(),
            next_seq: AtomicU64::new(1),
        };
        host.call(&Call::Begin(Box::new(spec.clone())), &[])?;
        Ok(host)
    }

    pub(crate) fn call(&self, call: &Call, data: &[u8]) -> Result<Vec<u8>, AllpacaError> {
        let head = serde_json::to_vec(call).map_err(|e| err(format!("qwen4exp block call: {e}")))?;
        self.rt.ext(self.session, EXT, &head, data)
    }

    /// The hosted blocks, in the order they run.
    pub fn blocks(&self) -> &[usize] {
        &self.blks
    }

    /// A new sequence; its state on the worker starts with its first pass.
    pub fn new_seq(self: &Arc<Self>) -> RemoteSeq {
        RemoteSeq {
            host: self.clone(),
            id: self.next_seq.fetch_add(1, Ordering::Relaxed),
            lost: false,
        }
    }

    /// Run the hosted blocks on the streams of `rows` tokens from `pos` of
    /// `seq`, in place: one round trip.
    pub fn run(&self, seq: &RemoteSeq, pos: usize, rows: usize, streams: &mut [f32]) -> Result<(), AllpacaError> {
        if seq.lost {
            return Err(err("qwen4exp: this sequence's state on the worker was lost".into()));
        }
        let reply = self.call(&Call::Run { seq: seq.id, pos, rows }, &bytes_of(streams))?;
        if reply.len() != streams.len() * 4 {
            return Err(err(format!("qwen4exp block run: {} reply bytes", reply.len())));
        }
        streams.copy_from_slice(&floats(&reply));
        self.rt.roundtrips.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// One sequence's state on the worker that hosts blocks. A clone is a copy
/// of the state there; dropping it frees it.
#[derive(Debug)]
pub struct RemoteSeq {
    host: Arc<RemoteBlocks>,
    id: u64,
    /// The copy on the worker failed: the sequence cannot go on.
    lost: bool,
}

impl Clone for RemoteSeq {
    fn clone(&self) -> Self {
        let mut copy = self.host.new_seq();
        copy.lost = self.lost
            || self
                .host
                .call(&Call::CloneSeq { src: self.id, dst: copy.id }, &[])
                .inspect_err(|e| tracing::warn!(error = %e, "qwen4exp: copying a sequence on the worker failed"))
                .is_err();
        copy
    }
}

impl Drop for RemoteSeq {
    fn drop(&mut self) {
        // A dead connection has dropped the state already.
        let _ = self.host.call(&Call::DropSeq { seq: self.id }, &[]);
    }
}
