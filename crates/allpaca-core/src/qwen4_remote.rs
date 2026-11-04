//! `qwen4exp` blocks run whole on an RPC worker: the worker holds the
//! blocks' matrices and routed experts on its GPUs and each sequence's state
//! for them, so a pass sends the residual streams once and gets them back
//! once, instead of stopping at every block.
//!
//! Both ends are here. The worker serves the extension [`EXT`]
//! ([`worker_extension`]); the coordinator fills it and calls it through
//! [`RemoteBlocks`]. A call's head is a JSON [`Call`], its payload raw bytes.
//!
//! A worker started with a cache directory keeps the weights it is sent on
//! its disk under their content key ([`WeightCache`]); the coordinator asks
//! for each by key first and only sends what the worker does not have.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use allpaca_hip::HipWorkspace;
use allpaca_types::{AllpacaError, FatalError};
use serde::{Deserialize, Serialize};

use crate::qwen4_gpu::{HostedSpec, Qwen4DeviceHost, Qwen4DeviceState, Qwen4Gpu};
use crate::rpc_runtime::RpcRuntime;

/// The extension's name in a worker's HELLO.
pub const EXT: &str = "q4x-blocks";

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum Call {
    /// Start over with these blocks; the weights follow. The reply is one
    /// byte: 1 when the worker keeps a weight cache.
    Begin(Box<HostedSpec>),
    /// One weight of `len` bytes with content key `key`. With `probe` the
    /// call has no payload and asks the worker to take the weight from its
    /// cache (reply 1) or say it cannot (reply 0); without, the payload is
    /// the weight.
    Weight { item: Weight, key: u64, len: u64, probe: bool },
    /// Run the blocks on `rows` tokens from `pos` of sequence `seq`
    /// (payload and reply: the streams, little-endian floats).
    Run { seq: u64, pos: usize, rows: usize },
    /// Sequence `dst` becomes a copy of `src`.
    CloneSeq { src: u64, dst: u64 },
    DropSeq { seq: u64 },
    /// Keep a copy of sequence `seq` after its first `tokens` tokens in the
    /// worker's RAM as `id`, for the prompt cache. Reply 1 when kept, 0 when
    /// it does not fit.
    SaveSeq { seq: u64, id: u64, tokens: usize },
    /// Sequence `seq` becomes the copy kept as `id`. Reply 1, or 0 when the
    /// worker no longer has it.
    LoadSeq { id: u64, seq: u64 },
    DropSaved { id: u64 },
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum Weight {
    /// A matrix in the encoding its kernel reads.
    Matrix { name: String, encoding: u8, n: usize, k: usize },
    /// An F32 vector, little-endian floats.
    Vector { name: String },
    /// A block's routed experts as Q2_0 slots.
    Pool { blk: usize },
}

/// A 64-bit content key of `data`: eight bytes a step, so 15 GB of weights
/// take seconds. Not a cryptographic hash; the cache also checks the length.
pub fn weight_key(data: &[u8]) -> u64 {
    const M: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut h = (data.len() as u64).wrapping_mul(M);
    let mut words = data.chunks_exact(8);
    for w in &mut words {
        let w = u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]);
        h = (h ^ w).wrapping_mul(M);
        h ^= h >> 29;
    }
    for (i, b) in words.remainder().iter().enumerate() {
        h = (h ^ (u64::from(*b) << (8 * i))).wrapping_mul(M);
    }
    h ^= h >> 32;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^ (h >> 33)
}

/// Weights on a worker's disk, a file per weight named by its content key
/// and length. Reading one checks both; when the directory would pass its
/// size limit the files used longest ago go first.
#[derive(Debug, Clone)]
pub struct WeightCache {
    dir: PathBuf,
    max_bytes: u64,
}

impl WeightCache {
    pub fn new(dir: impl Into<PathBuf>, max_bytes: u64) -> Self {
        Self {
            dir: dir.into(),
            max_bytes,
        }
    }

    fn path(&self, key: u64, len: u64) -> PathBuf {
        self.dir.join(format!("{key:016x}-{len}.weight"))
    }

    /// The weight with this key and length, when the cache holds it intact.
    pub fn get(&self, key: u64, len: u64) -> Option<Vec<u8>> {
        let path = self.path(key, len);
        let data = std::fs::read(&path).ok()?;
        if data.len() as u64 != len || weight_key(&data) != key {
            tracing::warn!(path = %path.display(), "weight cache: a damaged file was dropped");
            let _ = std::fs::remove_file(&path);
            return None;
        }
        // Reading counts as use for the size limit.
        if let Ok(file) = std::fs::File::options().write(true).open(&path) {
            let _ = file.set_modified(std::time::SystemTime::now());
        }
        Some(data)
    }

    /// Keep `data` under `key`. A failure only costs the next upload.
    pub fn put(&self, key: u64, data: &[u8]) {
        if let Err(e) = self.store(key, data) {
            tracing::warn!(error = %e, dir = %self.dir.display(), "weight cache: could not store a weight");
        }
    }

    fn store(&self, key: u64, data: &[u8]) -> std::io::Result<()> {
        let len = data.len() as u64;
        if len > self.max_bytes {
            return Ok(());
        }
        std::fs::create_dir_all(&self.dir)?;
        self.make_room(len, &self.path(key, len))?;
        let part = self.dir.join(format!("{key:016x}-{len}.part"));
        std::fs::write(&part, data)?;
        std::fs::rename(&part, self.path(key, len))
    }

    /// Delete the weights used longest ago until `incoming` more bytes fit.
    fn make_room(&self, incoming: u64, keep: &Path) -> std::io::Result<()> {
        let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = std::fs::read_dir(&self.dir)?
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "weight") && e.path() != keep)
            .filter_map(|e| {
                let meta = e.metadata().ok()?;
                Some((meta.modified().ok()?, meta.len(), e.path()))
            })
            .collect();
        files.sort();
        let mut total: u64 = files.iter().map(|f| f.1).sum();
        for (_, bytes, path) in files {
            if total + incoming <= self.max_bytes {
                break;
            }
            std::fs::remove_file(&path)?;
            total -= bytes;
        }
        Ok(())
    }
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
    cache: Option<WeightCache>,
    saved: SavedSeqs,
}

/// Sequence states kept in RAM for coordinators' prompt caches, by id; the
/// ones used longest ago go first when `max_bytes` would be passed.
#[derive(Default)]
struct SavedSeqs {
    max_bytes: usize,
    bytes: usize,
    clock: u64,
    states: HashMap<u64, (Qwen4DeviceHost, u64)>,
}

impl SavedSeqs {
    fn put(&mut self, id: u64, state: Qwen4DeviceHost) -> bool {
        let size = state.host_bytes();
        if size > self.max_bytes {
            return false;
        }
        self.drop_id(id);
        while self.bytes + size > self.max_bytes {
            let Some(oldest) = self.states.iter().min_by_key(|(_, (_, used))| *used).map(|(id, _)| *id) else {
                break;
            };
            self.drop_id(oldest);
        }
        self.clock += 1;
        self.bytes += size;
        self.states.insert(id, (state, self.clock));
        true
    }

    fn get(&mut self, id: u64) -> Option<&Qwen4DeviceHost> {
        self.clock += 1;
        let clock = self.clock;
        self.states.get_mut(&id).map(|(state, used)| {
            *used = clock;
            &*state
        })
    }

    fn drop_id(&mut self, id: u64) {
        if let Some((state, _)) = self.states.remove(&id) {
            self.bytes -= state.host_bytes();
        }
    }
}

fn add_weight(gpu: &mut Qwen4Gpu, item: &Weight, data: &[u8]) -> Result<(), AllpacaError> {
    match item {
        Weight::Matrix { name, encoding, n, k } => gpu.add_matrix(name, *encoding, *n, *k, data),
        Weight::Vector { name } => gpu.add_vector(name, &floats(data)),
        Weight::Pool { blk } => gpu.add_pool(*blk, data),
    }
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
            return Ok(vec![u8::from(self.cache.is_some())]);
        }
        let (Some(gpu), Some(spec)) = (self.gpu.as_mut(), self.spec.as_ref()) else {
            return Err(err("qwen4exp block call before Begin".into()));
        };
        match call {
            Call::Begin(_) => {}
            Call::Weight { item, key, len, probe: true } => {
                let Some(cached) = self.cache.as_ref().and_then(|c| c.get(key, len)) else {
                    return Ok(vec![0]);
                };
                add_weight(gpu, &item, &cached)?;
                return Ok(vec![1]);
            }
            Call::Weight { item, key, len, probe: false } => {
                if data.len() as u64 != len {
                    return Err(err(format!("qwen4exp weight: {} bytes sent, {len} named", data.len())));
                }
                add_weight(gpu, &item, data)?;
                if let Some(cache) = &self.cache {
                    cache.put(key, data);
                }
            }
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
            Call::SaveSeq { seq, id, tokens } => {
                let Some(state) = self.seqs.get(&seq) else {
                    return Ok(vec![0]);
                };
                gpu.sync_all()?;
                return Ok(vec![u8::from(self.saved.put(id, state.to_host(tokens)?))]);
            }
            Call::LoadSeq { id, seq } => {
                let Some(saved) = self.saved.get(id) else {
                    return Ok(vec![0]);
                };
                self.seqs.insert(seq, Qwen4DeviceState::from_host(saved)?);
                return Ok(vec![1]);
            }
            Call::DropSaved { id } => self.saved.drop_id(id),
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

/// The extension a HIP worker serves to host `qwen4exp` blocks. With
/// `cache` it keeps the weights it is sent on disk and takes them from
/// there the next time. It keeps no sequences for prompt caches.
pub fn worker_extension(cache: Option<WeightCache>) -> Box<dyn allpaca_rpc::Extension> {
    worker_extension_saving(cache, 0)
}

/// [`worker_extension`] that also keeps up to `saved_bytes` of sequence
/// states in RAM for coordinators' prompt caches ([`RemoteSaved`]).
pub fn worker_extension_saving(cache: Option<WeightCache>, saved_bytes: usize) -> Box<dyn allpaca_rpc::Extension> {
    Box::new(BlockHost {
        cache,
        saved: SavedSeqs {
            max_bytes: saved_bytes,
            ..SavedSeqs::default()
        },
        ..BlockHost::default()
    })
}

/// The coordinator's end: blocks hosted on one worker.
#[derive(Debug)]
pub struct RemoteBlocks {
    rt: Arc<RpcRuntime>,
    session: usize,
    blks: Vec<usize>,
    next_seq: AtomicU64,
    /// The worker keeps a weight cache: ask before sending.
    cached: AtomicBool,
    /// Weight bytes the worker took from its cache instead of the network.
    pub cache_bytes: AtomicU64,
}

impl RemoteBlocks {
    /// Start hosting on `session` of `rt`: the worker forgets what it had.
    pub(crate) fn begin(rt: &Arc<RpcRuntime>, session: usize, spec: &HostedSpec) -> Result<Self, AllpacaError> {
        let host = Self {
            rt: rt.clone(),
            session,
            blks: spec.blocks.iter().map(|b| b.blk).collect(),
            next_seq: AtomicU64::new(1),
            cached: AtomicBool::new(false),
            cache_bytes: AtomicU64::new(0),
        };
        let reply = host.call(&Call::Begin(Box::new(spec.clone())), &[])?;
        host.cached.store(reply.first() == Some(&1), Ordering::Relaxed);
        Ok(host)
    }

    /// Give the worker one weight: from its cache when it has it there,
    /// else over the network.
    pub(crate) fn send_weight(&self, item: Weight, data: &[u8]) -> Result<(), AllpacaError> {
        let (key, len) = (weight_key(data), data.len() as u64);
        let mut call = Call::Weight { item, key, len, probe: true };
        if self.cached.load(Ordering::Relaxed) && self.call(&call, &[])?.first() == Some(&1) {
            self.cache_bytes.fetch_add(len, Ordering::Relaxed);
            return Ok(());
        }
        if let Call::Weight { probe, .. } = &mut call {
            *probe = false;
        }
        self.call(&call, data).map(drop)
    }

    pub(crate) fn call(&self, call: &Call, data: &[u8]) -> Result<Vec<u8>, AllpacaError> {
        let head = serde_json::to_vec(call).map_err(|e| err(format!("qwen4exp block call: {e}")))?;
        self.rt.ext(self.session, EXT, &head, data)
    }

    /// A call whose reply the next step does not use ([`RpcRuntime::ext_post`]).
    fn post(&self, call: &Call) -> Result<allpaca_rpc::Posted, AllpacaError> {
        let head = serde_json::to_vec(call).map_err(|e| err(format!("qwen4exp block call: {e}")))?;
        self.rt.ext_post(self.session, EXT, &head, &[])
    }

    /// The hosted blocks, in the order they run.
    pub fn blocks(&self) -> &[usize] {
        &self.blks
    }

    /// Keep a copy of `seq` after its first `tokens` tokens on the worker,
    /// for the prompt cache. `None` when the worker has no room for it.
    pub fn save(self: &Arc<Self>, seq: &RemoteSeq, tokens: usize) -> Result<Option<RemoteSaved>, AllpacaError> {
        if seq.check().is_err() {
            return Ok(None);
        }
        let id = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let reply = self.call(&Call::SaveSeq { seq: seq.id, id, tokens }, &[])?;
        let saved = (reply.first() == Some(&1)).then(|| RemoteSaved { host: self.clone(), id });
        // The copy this state came from failed: what was kept is nothing.
        Ok(saved.filter(|_| seq.check().is_ok()))
    }

    /// A new sequence that starts from `saved`. `None` when the worker no
    /// longer has it (it made room, or it restarted).
    pub fn load(self: &Arc<Self>, saved: &RemoteSaved) -> Result<Option<RemoteSeq>, AllpacaError> {
        let seq = self.new_seq();
        let reply = self.call(&Call::LoadSeq { id: saved.id, seq: seq.id }, &[])?;
        Ok((reply.first() == Some(&1)).then_some(seq))
    }

    /// A new sequence; its state on the worker starts with its first pass.
    pub fn new_seq(self: &Arc<Self>) -> RemoteSeq {
        RemoteSeq {
            host: self.clone(),
            id: self.next_seq.fetch_add(1, Ordering::Relaxed),
            lost: false,
            copies: Vec::new(),
        }
    }

    /// Run the hosted blocks on the streams of `rows` tokens from `pos` of
    /// `seq`, in place: one round trip.
    pub fn run(&self, seq: &RemoteSeq, pos: usize, rows: usize, streams: &mut [f32]) -> Result<(), AllpacaError> {
        seq.check()?;
        let reply = self.call(&Call::Run { seq: seq.id, pos, rows }, &bytes_of(streams));
        // Waiting for this reply read the replies of the copies posted
        // before it: a failed one means the worker ran this on no state.
        seq.check()?;
        let reply = reply?;
        if reply.len() != streams.len() * 4 {
            return Err(err(format!("qwen4exp block run: {} reply bytes", reply.len())));
        }
        streams.copy_from_slice(&floats(&reply));
        self.rt.roundtrips.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl RemoteBlocks {
    /// Run `f` with the worker to itself, so it can have several passes in
    /// flight: [`Pipe::send`] hands one over and returns at once,
    /// [`Pipe::recv`] waits for the oldest one's result. The worker works
    /// on a pass while the coordinator prepares the next. `f` must not
    /// copy or drop a [`RemoteSeq`] (each is a call of its own).
    pub fn pipeline<T>(&self, f: impl FnOnce(&mut Pipe<'_>) -> Result<T, AllpacaError>) -> Result<T, AllpacaError> {
        self.rt.with_session(self.session, |session| {
            let mut pipe = Pipe {
                session,
                in_flight: std::collections::VecDeque::new(),
                roundtrips: &self.rt.roundtrips,
            };
            let out = f(&mut pipe);
            // Leave nothing unread: the next call would take it for its reply.
            while pipe.in_flight.pop_front().is_some() {
                let _ = pipe.session.ext_recv();
            }
            out
        })
    }
}

/// Bytes of streams that may be in flight to a worker at once
/// ([`RemoteBlocks::pipeline`]): well inside the socket buffers both ends ask for.
pub const PIPE_BYTES: usize = allpaca_rpc::SOCKET_BUFFER / 4;

/// The worker's connection while passes are in flight.
pub struct Pipe<'a> {
    session: &'a mut allpaca_rpc::RpcSession,
    /// Per pass in flight, the posted copies its sequence's state needs.
    in_flight: std::collections::VecDeque<Vec<allpaca_rpc::Posted>>,
    roundtrips: &'a AtomicU64,
}

impl Pipe<'_> {
    /// Hand the worker the streams of `rows` tokens from `pos` of `seq`.
    pub fn send(&mut self, seq: &RemoteSeq, pos: usize, rows: usize, streams: &[f32]) -> Result<(), AllpacaError> {
        seq.check()?;
        let head = serde_json::to_vec(&Call::Run { seq: seq.id, pos, rows })
            .map_err(|e| err(format!("qwen4exp block call: {e}")))?;
        self.session.ext_send(EXT, &head, &bytes_of(streams)).map_err(crate::rpc_runtime::map_rpc)?;
        self.in_flight.push_back(seq.copies.clone());
        Ok(())
    }

    /// The streams of the oldest pass in flight, after the hosted blocks.
    pub fn recv(&mut self, streams: &mut [f32]) -> Result<(), AllpacaError> {
        let Some(copies) = self.in_flight.pop_front() else {
            return Err(err("qwen4exp: no pass is in flight".into()));
        };
        let reply = self.session.ext_recv().map_err(crate::rpc_runtime::map_rpc);
        check_copies(&copies)?;
        let reply = reply?;
        if reply.len() != streams.len() * 4 {
            return Err(err(format!("qwen4exp block run: {} reply bytes", reply.len())));
        }
        streams.copy_from_slice(&floats(&reply));
        self.roundtrips.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// A sequence state kept on the worker that hosts blocks, for the prompt
/// cache ([`RemoteBlocks::save`]). Dropping it frees it there.
#[derive(Debug)]
pub struct RemoteSaved {
    host: Arc<RemoteBlocks>,
    id: u64,
}

impl RemoteSaved {
    /// The worker that keeps it.
    pub fn host(&self) -> &Arc<RemoteBlocks> {
        &self.host
    }
}

impl Drop for RemoteSaved {
    fn drop(&mut self) {
        // Nothing waits for it; a failure only leaves the copy on the worker.
        let _ = self.host.post(&Call::DropSaved { id: self.id });
    }
}

/// Fails when one of the posted copies a sequence's state comes from failed.
fn check_copies(copies: &[allpaca_rpc::Posted]) -> Result<(), AllpacaError> {
    match copies.iter().find_map(allpaca_rpc::Posted::failed) {
        Some(e) => Err(err(format!("qwen4exp: a deferred sequence copy failed on the worker: {e}"))),
        None => Ok(()),
    }
}

/// One sequence's state on the worker that hosts blocks. A clone is a copy
/// of the state there; dropping it frees it.
#[derive(Debug)]
pub struct RemoteSeq {
    host: Arc<RemoteBlocks>,
    id: u64,
    /// The copy on the worker could not be sent: the sequence cannot go on.
    lost: bool,
    /// The posted copies this state comes from (its own and its source's)
    /// whose replies were not yet read as successes: a failure means the
    /// worker has no state for this sequence.
    copies: Vec<allpaca_rpc::Posted>,
}

impl RemoteSeq {
    /// Fails when the state on the worker is known to be missing.
    fn check(&self) -> Result<(), AllpacaError> {
        if self.lost {
            return Err(err("qwen4exp: this sequence's state on the worker was lost".into()));
        }
        check_copies(&self.copies)
    }
}

impl Clone for RemoteSeq {
    /// The copy is posted: the worker makes it before the next call, and
    /// the copy's next pass fails if it could not.
    fn clone(&self) -> Self {
        let mut copy = self.host.new_seq();
        if self.check().is_err() {
            copy.lost = true;
            return copy;
        }
        // With `rpc.defer_replies` off the reply is in already.
        let posted = self.host.post(&Call::CloneSeq { src: self.id, dst: copy.id }).and_then(|posted| match posted.failed() {
            Some(e) => Err(err(e)),
            None => Ok(posted),
        });
        match posted {
            Ok(posted) => {
                copy.copies = self
                    .copies
                    .iter()
                    .filter(|p| p.outcome().is_none())
                    .cloned()
                    .chain(std::iter::once(posted))
                    .collect();
            }
            Err(e) => {
                tracing::warn!(error = %e, "qwen4exp: copying a sequence on the worker failed");
                copy.lost = true;
            }
        }
        copy
    }
}

impl Drop for RemoteSeq {
    fn drop(&mut self) {
        // Nothing waits for it. A dead connection has dropped the state already.
        let _ = self.host.post(&Call::DropSeq { seq: self.id });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A worker serving only `q4x-blocks` with no blocks begun (every
    /// sequence call fails) and the coordinator's end of it.
    fn unbegun_blocks() -> (Arc<RpcRuntime>, Arc<RemoteBlocks>, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let backend = allpaca_rpc::WorkerBackend::Cpu(allpaca_rpc::CpuWorker::new(1 << 20));
            allpaca_rpc::serve_connection_ext(stream, backend, None, vec![worker_extension(None)]).unwrap();
        });
        let rt = Arc::new(RpcRuntime::connect(&[addr.to_string()], None).unwrap());
        let host = Arc::new(RemoteBlocks {
            rt: rt.clone(),
            session: 0,
            blks: Vec::new(),
            next_seq: AtomicU64::new(1),
            cached: AtomicBool::new(false),
            cache_bytes: AtomicU64::new(0),
        });
        (rt, host, worker)
    }

    fn op_calls(rt: &RpcRuntime, op: &str) -> (u64, u64) {
        rt.worker_calls()[0].ops.iter().find(|o| o.op == op).map_or((0, 0), |o| (o.calls, o.posted))
    }

    /// Copying and dropping a sequence's state, and dropping a saved one,
    /// go out without waiting for the reply (Q19). A copy that failed on
    /// the worker fails the copy's next pass, alone or pipelined.
    #[test]
    fn sequence_copies_are_posted_and_a_failed_copy_fails_its_next_pass() {
        let (rt, host, worker) = unbegun_blocks();
        let seq = host.new_seq();
        let copy = seq.clone();
        assert_eq!(op_calls(&rt, "q4x-blocks:CloneSeq"), (0, 1), "the copy is posted");
        let mut streams = vec![0.0f32; 8];
        let error = host.run(&copy, 0, 1, &mut streams).unwrap_err().to_string();
        assert!(
            error.contains("sequence copy failed") && error.contains("before Begin"),
            "the copy's pass reports the failed copy: {error}"
        );
        let other = seq.clone();
        let error = host
            .pipeline(|pipe| {
                pipe.send(&other, 0, 1, &streams)?;
                pipe.recv(&mut streams)
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("sequence copy failed"), "pipelined: {error}");
        drop(RemoteSaved { host: host.clone(), id: 77 });
        drop((copy, other));
        assert_eq!(op_calls(&rt, "q4x-blocks:DropSaved"), (0, 1));
        assert_eq!(op_calls(&rt, "q4x-blocks:DropSeq"), (0, 2));
        // A save of a sequence whose copy failed is no save.
        let copy = seq.clone();
        rt.settle_posted(0).unwrap();
        assert!(host.save(&copy, 1).unwrap().is_none());
        drop((seq, copy, host, rt));
        worker.join().unwrap();
    }
}
