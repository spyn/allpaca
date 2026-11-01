//! Radix / prefix-tree KV reuse (GEM vertical).
//!
//! Stores prompt prefixes with KV+recurrent checkpoints so a later prompt
//! that shares one restores it instead of prefilling it again. Edges hold
//! runs of tokens, so the tree is as deep as its branch points, not its
//! prompts.
//!
//! Caps: node count and a RAM byte budget; the coldest checkpoints (LRU by
//! last hit/insert) go first. With a disk tier, every checkpoint is also
//! written to a file named by its tokens: RAM eviction then only drops the
//! in-memory copy, a lookup reads it back, the files outlive the process, and
//! a disk budget deletes the coldest files.
//!
//! A state checkpoint holds only recurrent state: the keys and values of its
//! prefix come from any longer prompt's checkpoint below it, so a hybrid
//! model can keep its state at several points of a prompt while the keys
//! and values are stored once.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use allpaca_types::{AllpacaError, FatalError, KvCodec};

use crate::anchors::{snapshot_kv, KvRow, RecurrentState};
use crate::kv_codec::{EncodedRows, Reader};
use crate::kv_tier::KvTierManager;
use crate::model::ModelConfig;

/// MTP host KV saved beside a prompt-cache checkpoint.
pub(crate) const DRAFT_MTP: u8 = 1;
/// DFlash2 K/V ring saved beside a prompt-cache checkpoint.
pub(crate) const DRAFT_DFLASH2: u8 = 2;
/// Qwen4exp MTP memory saved beside a prompt-cache checkpoint.
pub(crate) const DRAFT_QWEN4_MTP: u8 = 3;

/// Drafter state for the tokens this checkpoint covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DraftCache {
    pub(crate) kind: u8,
    pub(crate) bytes: Vec<u8>,
}

/// Checkpoint attached to a radix node (prefix of the prompt).
#[derive(Debug, Clone)]
pub struct RadixCheckpoint {
    pub recurrent: RecurrentState,
    /// Per layer, in the forest's [`KvCodec`].
    pub(crate) kv: Vec<EncodedRows>,
    /// No recurrent state: the keys and values of any prefix of this
    /// checkpoint's tokens resume that prefix.
    pub(crate) sliceable: bool,
    /// Only recurrent state: the keys and values come from a longer
    /// prompt's checkpoint below it.
    pub(crate) state_only: bool,
    /// Warmed drafter for this prefix. A shorter cut drops it.
    pub(crate) draft: Option<DraftCache>,
}

impl RadixCheckpoint {
    /// Keys and values per layer, decoded.
    pub(crate) fn kv_rows(&self) -> Vec<Vec<KvRow>> {
        self.kv.iter().map(EncodedRows::decode).collect()
    }

    /// Approximate resident size of this checkpoint (payload + thin header).
    pub fn approx_bytes(&self) -> usize {
        let r = &self.recurrent;
        let rec: usize = [&r.layers, &r.ssm_h, &r.ssm_conv, &r.gdn_state, &r.gdn_conv]
            .into_iter()
            .flatten()
            .map(|l| l.len().saturating_mul(4))
            .sum();
        let host_kv: usize = r.host_kv.iter().map(|(k, v)| (k.len() + v.len()) * 4).sum();
        let kv: usize = self.kv.iter().map(EncodedRows::bytes).sum();
        let draft = self.draft.as_ref().map_or(0, |d| d.bytes.len().saturating_add(8));
        rec.saturating_add(host_kv)
            .saturating_add(r.q4x.host_bytes())
            .saturating_add(kv)
            .saturating_add(draft)
            .saturating_add(64)
    }

    /// The same checkpoint cut to its first `n` tokens.
    fn prefix(&self, n: usize) -> Self {
        Self {
            recurrent: self.recurrent.clone(),
            kv: self.kv.iter().map(|layer| layer.prefix(n)).collect(),
            sliceable: self.sliceable,
            state_only: self.state_only,
            draft: None,
        }
    }
}

/// A checkpoint's file in the disk tier.
#[derive(Debug)]
struct OnDisk {
    name: String,
    bytes: usize,
    sliceable: bool,
    /// A state checkpoint (no keys and values).
    state_only: bool,
}

/// One node of the radix: the end of `edge`, a run of tokens.
#[derive(Debug, Default)]
struct RadixNode {
    /// Tokens from the parent to this node (empty for the root).
    edge: Vec<u32>,
    /// Keyed by the first token of the child's edge.
    children: HashMap<u32, Box<RadixNode>>,
    checkpoint: Option<RadixCheckpoint>,
    on_disk: Option<OnDisk>,
    /// LRU clock when this node's checkpoint was last touched (0 = never / no cp).
    used_at: u64,
}

impl RadixNode {
    fn holds(&self) -> bool {
        self.checkpoint.is_some() || self.on_disk.is_some()
    }

    /// Holds keys and values (not only recurrent state).
    fn has_kv(&self) -> bool {
        self.checkpoint.as_ref().is_some_and(|cp| !cp.state_only)
            || self.on_disk.as_ref().is_some_and(|d| !d.state_only)
    }

    fn sliceable(&self) -> bool {
        self.checkpoint.as_ref().is_some_and(|cp| cp.sliceable)
            || self.on_disk.as_ref().is_some_and(|d| d.sliceable)
    }
}

/// Where the disk tier keeps checkpoint files: a local folder, or a worker
/// ([`crate::prompt_store::RemoteStore`]).
pub(crate) trait CheckpointStore: Send + std::fmt::Debug {
    fn put(&mut self, name: &str, data: &[u8]) -> Result<(), String>;
    /// [`Self::put`] of a buffer the store may keep.
    fn put_owned(&mut self, name: &str, data: Vec<u8>) -> Result<(), String> {
        self.put(name, &data)
    }
    fn get(&mut self, name: &str) -> Option<Vec<u8>>;
    fn remove(&mut self, name: &str);
    /// Every checkpoint held, oldest first.
    fn list(&mut self) -> Result<Vec<Listed>, String>;
}

/// One file a [`CheckpointStore`] holds.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Listed {
    pub(crate) name: String,
    pub(crate) bytes: usize,
    pub(crate) tokens: Vec<u32>,
    pub(crate) flags: u8,
}

/// Checkpoint files in a local folder.
#[derive(Debug)]
pub(crate) struct LocalDir {
    dir: PathBuf,
    /// Text of the folder's `key.txt`, written again if the folder is.
    key: Option<String>,
}

/// The file in a checkpoint folder that names what its checkpoints belong to.
pub(crate) const KEY_FILE: &str = "key.txt";

impl LocalDir {
    pub(crate) fn open(dir: &Path) -> Result<Self, AllpacaError> {
        std::fs::create_dir_all(dir).map_err(|e| io_error(dir, e))?;
        Ok(Self { dir: dir.to_path_buf(), key: None })
    }

    /// Write `text` to the folder's [`KEY_FILE`] now and whenever the folder
    /// has to be made again.
    pub(crate) fn with_key(mut self, text: &str) -> Self {
        if let Err(e) = std::fs::write(self.dir.join(KEY_FILE), text) {
            tracing::warn!(error = %e, dir = %self.dir.display(), "checkpoint folder: key.txt not written");
        }
        self.key = Some(text.to_string());
        self
    }
}

impl CheckpointStore for LocalDir {
    /// Makes the folder again if it is gone (another server's prune of a
    /// folder it judged unused).
    fn put(&mut self, name: &str, data: &[u8]) -> Result<(), String> {
        let file = self.dir.join(name);
        write_file(&file, data)
            .or_else(|e| {
                if self.dir.is_dir() {
                    return Err(e);
                }
                std::fs::create_dir_all(&self.dir)?;
                if let Some(key) = &self.key {
                    std::fs::write(self.dir.join(KEY_FILE), key)?;
                }
                write_file(&file, data)
            })
            .map_err(|e| e.to_string())
    }

    fn get(&mut self, name: &str) -> Option<Vec<u8>> {
        std::fs::read(self.dir.join(name)).ok()
    }

    fn remove(&mut self, name: &str) {
        let _ = std::fs::remove_file(self.dir.join(name));
    }

    fn list(&mut self) -> Result<Vec<Listed>, String> {
        let mut found = Vec::new();
        for entry in std::fs::read_dir(&self.dir).map_err(|e| e.to_string())?.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "ckpt") {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            let Some((tokens, flags)) = read_file_header(&path) else {
                let _ = std::fs::remove_file(&path);
                continue;
            };
            let modified = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
            let name = entry.file_name().to_string_lossy().into_owned();
            found.push((modified, Listed { name, bytes: meta.len() as usize, tokens, flags }));
        }
        found.sort_by_key(|f| f.0);
        Ok(found.into_iter().map(|(_, l)| l).collect())
    }
}

/// Files queued for a [`BackgroundStore`]'s writer.
#[derive(Debug, Default)]
struct Queued {
    /// Name -> (put number, data), until the writer has stored it.
    files: HashMap<String, (u64, std::sync::Arc<Vec<u8>>)>,
    bytes: usize,
    puts: u64,
}

impl Queued {
    fn take(&mut self, name: &str) -> Option<(u64, std::sync::Arc<Vec<u8>>)> {
        let file = self.files.remove(name)?;
        self.bytes -= file.1.len();
        Some(file)
    }
}

#[derive(Debug)]
enum Job {
    Put(String, u64),
    Remove(String),
}

/// Writes to another store on a thread of its own, so a checkpoint costs
/// the request only its encoding. Queued files read back from memory,
/// removals keep their order with the writes, and dropping it waits for
/// the queue.
#[derive(Debug)]
pub(crate) struct BackgroundStore {
    store: std::sync::Arc<std::sync::Mutex<Box<dyn CheckpointStore>>>,
    queued: std::sync::Arc<std::sync::Mutex<Queued>>,
    /// Most bytes waiting; a put past it is refused.
    max_queued: usize,
    jobs: Option<std::sync::mpsc::Sender<Job>>,
    writer: Option<std::thread::JoinHandle<()>>,
}

impl BackgroundStore {
    pub(crate) fn new(store: Box<dyn CheckpointStore>, max_queued: usize) -> Self {
        let store = std::sync::Arc::new(std::sync::Mutex::new(store));
        let queued = std::sync::Arc::new(std::sync::Mutex::new(Queued::default()));
        let (jobs, rx) = std::sync::mpsc::channel::<Job>();
        let writer = {
            let (store, queued) = (store.clone(), queued.clone());
            std::thread::Builder::new()
                .name("prompt-cache-writer".into())
                .spawn(move || {
                    for job in rx {
                        let mut store = store.lock().unwrap_or_else(|e| e.into_inner());
                        match job {
                            Job::Put(name, put) => {
                                // Replaced or removed since: a later job has it.
                                let data = {
                                    let q = queued.lock().unwrap_or_else(|e| e.into_inner());
                                    match q.files.get(&name) {
                                        Some((n, data)) if *n == put => data.clone(),
                                        _ => continue,
                                    }
                                };
                                if let Err(e) = store.put(&name, &data) {
                                    tracing::warn!(error = %e, name, "prompt cache: checkpoint not written");
                                }
                                let mut q = queued.lock().unwrap_or_else(|e| e.into_inner());
                                if q.files.get(&name).is_some_and(|(n, _)| *n == put) {
                                    q.take(&name);
                                }
                            }
                            Job::Remove(name) => store.remove(&name),
                        }
                    }
                })
                .expect("spawn the prompt cache writer")
        };
        Self {
            store,
            queued,
            max_queued,
            jobs: Some(jobs),
            writer: Some(writer),
        }
    }
}

impl CheckpointStore for BackgroundStore {
    fn put(&mut self, name: &str, data: &[u8]) -> Result<(), String> {
        self.put_owned(name, data.to_vec())
    }

    fn put_owned(&mut self, name: &str, data: Vec<u8>) -> Result<(), String> {
        let put = {
            let mut q = self.queued.lock().map_err(|e| e.to_string())?;
            let replaced = q.files.get(name).map_or(0, |f| f.1.len());
            if q.bytes - replaced + data.len() > self.max_queued {
                return Err(format!("{} bytes already queued for writing", q.bytes));
            }
            q.take(name);
            q.puts += 1;
            let put = q.puts;
            q.bytes += data.len();
            q.files.insert(name.to_string(), (put, std::sync::Arc::new(data)));
            put
        };
        self.jobs
            .as_ref()
            .and_then(|jobs| jobs.send(Job::Put(name.to_string(), put)).ok())
            .ok_or_else(|| "the prompt cache writer stopped".to_string())
    }

    fn get(&mut self, name: &str) -> Option<Vec<u8>> {
        if let Some((_, data)) = self.queued.lock().ok()?.files.get(name) {
            return Some(data.as_ref().clone());
        }
        self.store.lock().ok()?.get(name)
    }

    fn remove(&mut self, name: &str) {
        if let Ok(mut q) = self.queued.lock() {
            q.take(name);
        }
        if let Some(jobs) = self.jobs.as_ref() {
            let _ = jobs.send(Job::Remove(name.to_string()));
        }
    }

    fn list(&mut self) -> Result<Vec<Listed>, String> {
        self.store.lock().map_err(|e| e.to_string())?.list()
    }
}

impl Drop for BackgroundStore {
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

/// The disk tier: a store and its budget.
#[derive(Debug)]
struct Disk {
    store: Box<dyn CheckpointStore>,
    /// 0 = unlimited.
    max_bytes: usize,
    bytes: usize,
}

/// Forest of prompt prefixes for longest-common-prefix KV reuse.
#[derive(Debug)]
pub struct KvRadixForest {
    root: RadixNode,
    /// Nodes allocated (excluding root).
    pub nodes: usize,
    /// Soft node cap; 0 = unlimited. Excess evicts the coldest leaves.
    max_nodes: usize,
    /// Soft checkpoint byte budget (RAM); 0 = unlimited.
    max_bytes: usize,
    /// Sum of [`RadixCheckpoint::approx_bytes`] currently held in RAM.
    checkpoint_bytes: usize,
    clock: u64,
    /// Lookups that restored a checkpoint (depth > 0).
    pub hits: u64,
    /// Lookups with no restorable prefix.
    pub misses: u64,
    /// Cumulative tokens skipped via LCP restore.
    pub skip_tokens: u64,
    /// Checkpoints or nodes removed to stay under node/byte caps.
    pub evictions: u64,
    /// How checkpoints hold keys and values.
    codec: KvCodec,
    disk: Option<Disk>,
}

impl Default for KvRadixForest {
    fn default() -> Self {
        Self::with_limits(0, 0)
    }
}

impl KvRadixForest {
    #[cfg(test)]
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn with_max_nodes(max_nodes: usize) -> Self {
        Self::with_limits(max_nodes, 0)
    }

    /// Node cap + checkpoint byte budget (either 0 = unlimited for that axis).
    pub fn with_limits(max_nodes: usize, max_bytes: usize) -> Self {
        Self {
            root: RadixNode::default(),
            nodes: 0,
            max_nodes,
            max_bytes,
            checkpoint_bytes: 0,
            clock: 0,
            hits: 0,
            misses: 0,
            skip_tokens: 0,
            evictions: 0,
            codec: KvCodec::F32,
            disk: None,
        }
    }

    /// Hold checkpoint keys and values in `codec`.
    pub fn with_codec(mut self, codec: KvCodec) -> Self {
        self.codec = codec;
        self
    }

    /// Keep checkpoints in `dir` too; see [`Self::open_disk`].
    #[cfg(test)]
    pub fn with_disk(mut self, dir: &Path, max_bytes: usize) -> Result<Self, AllpacaError> {
        self.open_disk(dir, max_bytes)?;
        Ok(self)
    }

    /// Keep checkpoints in `store` too; see [`Self::open_store`].
    #[cfg(test)]
    pub(crate) fn with_store(mut self, store: Box<dyn CheckpointStore>, max_bytes: usize) -> Result<Self, AllpacaError> {
        self.open_store(store, max_bytes)?;
        Ok(self)
    }

    /// [`Self::open_store`] on the folder `dir`.
    #[cfg(test)]
    pub fn open_disk(&mut self, dir: &Path, max_bytes: usize) -> Result<(), AllpacaError> {
        self.open_store(Box::new(LocalDir::open(dir)?), max_bytes)
    }

    /// Write every checkpoint to `store` as well, read them back when RAM no
    /// longer holds them, and pick up the ones already there. `max_bytes`
    /// caps the files (0 = unlimited).
    pub(crate) fn open_store(&mut self, mut store: Box<dyn CheckpointStore>, max_bytes: usize) -> Result<(), AllpacaError> {
        let found = store.list().map_err(|e| {
            AllpacaError::Fatal(FatalError::InvalidConfig(format!("prompt cache store: {e}")))
        })?;
        self.disk = Some(Disk {
            store,
            max_bytes,
            bytes: 0,
        });
        for listed in found {
            if listed.tokens.is_empty() {
                continue;
            }
            let stamp = self.touch();
            let on_disk = OnDisk {
                name: listed.name,
                bytes: listed.bytes,
                sliceable: listed.flags & FLAG_SLICEABLE != 0,
                state_only: listed.flags & FLAG_STATE_ONLY != 0,
            };
            let node = node_insert(&mut self.root, &mut self.nodes, &listed.tokens);
            node.used_at = stamp;
            let old = node.on_disk.replace(on_disk).map_or(0, |d| d.bytes);
            self.disk_sub(old);
            self.disk_add(listed.bytes);
        }
        self.trim_disk();
        Ok(())
    }

    /// Forget every checkpoint (RAM and tree) and stop using the disk tier;
    /// files stay where they are.
    pub fn reset(&mut self) {
        self.root = RadixNode::default();
        self.nodes = 0;
        self.checkpoint_bytes = 0;
        self.disk = None;
    }

    /// A checkpoint of `kv` (per layer) in this forest's codec.
    pub(crate) fn checkpoint(&self, recurrent: RecurrentState, kv: Vec<Vec<KvRow>>) -> RadixCheckpoint {
        RadixCheckpoint {
            recurrent,
            kv: kv.iter().map(|rows| EncodedRows::encode(self.codec, rows)).collect(),
            sliceable: false,
            state_only: false,
            draft: None,
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// How checkpoints hold keys and values.
    pub fn codec(&self) -> KvCodec {
        self.codec
    }

    pub fn skip_tokens(&self) -> u64 {
        self.skip_tokens
    }

    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    pub fn checkpoint_bytes(&self) -> usize {
        self.checkpoint_bytes
    }

    /// Bytes of checkpoint files in the disk tier.
    #[cfg(test)]
    pub fn disk_bytes(&self) -> usize {
        self.disk.as_ref().map_or(0, |d| d.bytes)
    }

    #[allow(dead_code)]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Evict cold checkpoints until `checkpoint_bytes <= target`. Returns bytes freed.
    pub fn trim_to_bytes(&mut self, target: usize) -> usize {
        let before = self.checkpoint_bytes;
        while self.checkpoint_bytes > target {
            if !self.evict_coldest_checkpoint() {
                break;
            }
            self.evictions += 1;
        }
        before.saturating_sub(self.checkpoint_bytes)
    }

    fn touch(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    fn disk_add(&mut self, bytes: usize) {
        if let Some(d) = self.disk.as_mut() {
            d.bytes = d.bytes.saturating_add(bytes);
        }
    }

    /// Delete `file` from the store.
    fn disk_remove(&mut self, file: &OnDisk) {
        if let Some(d) = self.disk.as_mut() {
            d.store.remove(&file.name);
        }
        self.disk_sub(file.bytes);
    }

    fn disk_sub(&mut self, bytes: usize) {
        if let Some(d) = self.disk.as_mut() {
            d.bytes = d.bytes.saturating_sub(bytes);
        }
    }

    /// Insert `tokens` and attach `cp` at the terminal (replacing an older
    /// checkpoint of the same prefix). Returns the nodes added.
    pub fn insert_checkpoint(&mut self, tokens: &[u32], cp: RadixCheckpoint) -> usize {
        if tokens.is_empty() {
            return 0;
        }
        // A checkpoint with part of its state on a worker stays in RAM.
        let written = self.disk.as_mut().filter(|_| cp.recurrent.q4x.persistable()).and_then(|d| {
            let name = file_name(tokens);
            let data = encode_file(tokens, &cp);
            let bytes = data.len();
            match d.store.put_owned(&name, data) {
                Ok(()) => Some(OnDisk {
                    name,
                    bytes,
                    sliceable: cp.sliceable,
                    state_only: cp.state_only,
                }),
                Err(e) => {
                    tracing::warn!(error = %e, "prompt cache: checkpoint not written to disk");
                    None
                }
            }
        });
        let new_bytes = cp.approx_bytes();
        let stamp = self.touch();
        let before = self.nodes;
        let node = node_insert(&mut self.root, &mut self.nodes, tokens);
        let old_cp = node.checkpoint.replace(cp);
        node.used_at = stamp;
        let (old_file, new_file) = match written {
            Some(w) => {
                let bytes = w.bytes;
                (node.on_disk.replace(w), bytes)
            }
            None => (None, 0),
        };
        let added = self.nodes - before;
        if let Some(old) = old_cp {
            self.checkpoint_bytes = self.checkpoint_bytes.saturating_sub(old.approx_bytes());
        }
        self.checkpoint_bytes = self.checkpoint_bytes.saturating_add(new_bytes);
        if let Some(old) = old_file {
            // Same tokens, same name: the new file replaced it.
            self.disk_sub(old.bytes);
        }
        self.disk_add(new_file);
        self.evict_to_cap();
        self.trim_disk();
        added
    }

    /// Keep the recurrent state at the end of `tokens` (a state checkpoint).
    /// A checkpoint already there with keys and values is left as it is.
    pub fn insert_state(&mut self, tokens: &[u32], recurrent: RecurrentState) {
        if node_at(&self.root, tokens).is_some_and(|n| n.has_kv()) {
            return;
        }
        let cp = RadixCheckpoint {
            recurrent,
            kv: Vec::new(),
            sliceable: false,
            state_only: true,
            draft: None,
        };
        self.insert_checkpoint(tokens, cp);
    }

    fn evict_to_cap(&mut self) {
        if self.max_nodes > 0 {
            while self.nodes > self.max_nodes {
                if !self.evict_coldest_leaf() {
                    break;
                }
                self.evictions += 1;
            }
        }
        if self.max_bytes > 0 {
            while self.checkpoint_bytes > self.max_bytes {
                if !self.evict_coldest_checkpoint() {
                    break;
                }
                self.evictions += 1;
            }
        }
    }

    /// Drop the coldest checkpoint from RAM (its file, if any, stays).
    fn evict_coldest_checkpoint(&mut self) -> bool {
        let Some(path) = coldest(&self.root, |n| n.checkpoint.is_some()) else {
            return false;
        };
        let node = node_at_mut(&mut self.root, &path).expect("coldest path");
        if let Some(cp) = node.checkpoint.take() {
            self.checkpoint_bytes = self.checkpoint_bytes.saturating_sub(cp.approx_bytes());
        }
        self.prune(&path);
        true
    }

    /// Remove the coldest leaf with whatever it holds.
    fn evict_coldest_leaf(&mut self) -> bool {
        let Some(path) = coldest(&self.root, |n| n.children.is_empty()) else {
            return false;
        };
        let node = node_at_mut(&mut self.root, &path).expect("coldest path");
        let cp = node.checkpoint.take();
        let file = node.on_disk.take();
        if let Some(cp) = cp {
            self.checkpoint_bytes = self.checkpoint_bytes.saturating_sub(cp.approx_bytes());
        }
        if let Some(file) = file {
            self.disk_remove(&file);
        }
        self.prune(&path);
        true
    }

    /// Delete the coldest files until the disk tier fits its budget.
    fn trim_disk(&mut self) {
        loop {
            match self.disk.as_ref() {
                Some(d) if d.max_bytes > 0 && d.bytes > d.max_bytes => {}
                _ => return,
            }
            let Some(path) = coldest(&self.root, |n| n.on_disk.is_some()) else {
                return;
            };
            let node = node_at_mut(&mut self.root, &path).expect("coldest path");
            let file = node.on_disk.take().expect("chosen for its file");
            // A file that goes takes its RAM copy with it: the two stay in step.
            let cp = node.checkpoint.take();
            self.disk_remove(&file);
            if let Some(cp) = cp {
                self.checkpoint_bytes = self.checkpoint_bytes.saturating_sub(cp.approx_bytes());
            }
            self.evictions += 1;
            self.prune(&path);
        }
    }

    /// Remove the node at `path` if it holds nothing and has no children,
    /// and fold a parent left with one child and nothing of its own.
    fn prune(&mut self, path: &[u32]) {
        let Some(parent_path) = parent_path(&self.root, path) else {
            return;
        };
        let key = path[parent_path.len()];
        let parent = node_at_mut(&mut self.root, &parent_path).expect("parent of a node");
        let empty = parent
            .children
            .get(&key)
            .is_some_and(|c| c.children.is_empty() && !c.holds());
        let mut removed = 0;
        if empty {
            parent.children.remove(&key);
            removed += 1;
        }
        if !parent_path.is_empty() && !parent.holds() && parent.children.len() == 1 {
            let (_, child) = parent.children.drain().next().expect("one child");
            let child = *child;
            parent.edge.extend_from_slice(&child.edge);
            parent.children = child.children;
            parent.checkpoint = child.checkpoint;
            parent.on_disk = child.on_disk;
            parent.used_at = child.used_at;
            removed += 1;
        }
        self.nodes = self.nodes.saturating_sub(removed);
    }

    /// Snapshot current KV/recurrent and insert under `tokens`.
    pub fn capture(
        &mut self,
        tokens: &[u32],
        recurrent: &RecurrentState,
        kv: &KvTierManager,
        pages: &[Vec<u64>],
        cfg: &ModelConfig,
        draft: Option<DraftCache>,
    ) -> Result<(), AllpacaError> {
        if tokens.is_empty() {
            return Ok(());
        }
        let mut cp = self.checkpoint(recurrent.clone(), snapshot_kv(kv, pages, cfg)?);
        cp.sliceable = !cfg.recurrent;
        cp.draft = draft;
        self.insert_checkpoint(tokens, cp);
        Ok(())
    }

    /// After a batched prefill write, snapshot KV once and attach a checkpoint
    /// at every prefix. Each checkpoint keeps only the first `i` KV rows so a
    /// later mid-prompt edit can restore the LCP without leftover page slots.
    /// HIP device pages are not snapshotted here: `read_kv` would D2H the
    /// growing sequence and break logits-only decode accounting.
    #[allow(dead_code)]
    pub fn capture_prefixes(
        &mut self,
        tokens: &[u32],
        recurrent: &RecurrentState,
        kv: &KvTierManager,
        pages: &[Vec<u64>],
        cfg: &ModelConfig,
    ) -> Result<(), AllpacaError> {
        if tokens.is_empty() {
            return Ok(());
        }
        let mut full = self.checkpoint(recurrent.clone(), snapshot_kv(kv, pages, cfg)?);
        full.sliceable = !cfg.recurrent;
        for n in 1..=tokens.len() {
            let mut cp = full.prefix(n);
            if n == tokens.len() {
                cp.draft = full.draft.clone();
            }
            self.insert_checkpoint(&tokens[..n], cp);
        }
        Ok(())
    }

    /// Longest restorable prefix: deepest node on the walk that has a checkpoint
    /// with `depth <= max_depth` (use `max_depth = tokens.len()-1` to leave logits work).
    #[cfg(test)]
    pub fn lookup_checkpoint(&mut self, tokens: &[u32]) -> Option<(usize, RadixCheckpoint)> {
        self.lookup_checkpoint_capped(tokens, usize::MAX)
    }

    pub fn lookup_checkpoint_capped(
        &mut self,
        tokens: &[u32],
        max_depth: usize,
    ) -> Option<(usize, RadixCheckpoint)> {
        // Walk: every node on the path that holds a checkpoint, and how far
        // the tokens follow the tree at all.
        let mut node = &self.root;
        let mut depth = 0usize;
        let mut held = Vec::new();
        // The node the walk ended at or inside, and the depth above its edge.
        let mut under: Option<(&RadixNode, usize)> = None;
        while depth < tokens.len() {
            let Some(child) = node.children.get(&tokens[depth]) else {
                break;
            };
            let common = child
                .edge
                .iter()
                .zip(&tokens[depth..])
                .take_while(|(a, b)| a == b)
                .count();
            under = Some((child, depth));
            depth += common;
            if common < child.edge.len() {
                break;
            }
            node = child;
            if depth <= max_depth && child.holds() {
                held.push(depth);
            }
        }
        let shared = depth.min(max_depth);
        // Deeper than any checkpoint on the path: a sliceable checkpoint
        // further down (a longer prompt through here) holds this prefix too.
        let mut sliced = None;
        if shared > held.last().copied().unwrap_or(0) {
            if let Some((child, base)) = under {
                if let Some(rest) = sliceable_below(child) {
                    let mut path = tokens[..base].to_vec();
                    path.extend_from_slice(&child.edge);
                    path.extend(rest);
                    sliced = Some(path);
                }
            }
        }
        let mut found = sliced.and_then(|path| Some((shared, self.load(&path)?.prefix(shared))));
        for &d in held.iter().rev() {
            if found.is_some() {
                break;
            }
            found = self.resume_at(&tokens[..d]);
        }
        match found {
            Some((cut, cp)) => {
                self.hits += 1;
                self.skip_tokens += cut as u64;
                Some((cut, cp))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// The checkpoint at `path`, with the keys and values of a state
    /// checkpoint taken from a longer prompt's checkpoint below it.
    fn resume_at(&mut self, path: &[u32]) -> Option<(usize, RadixCheckpoint)> {
        let mut cp = self.load(path)?;
        if cp.state_only {
            let rest = kv_below(node_at(&self.root, path)?)?;
            let mut below = path.to_vec();
            below.extend(rest);
            cp.kv = self.load(&below)?.prefix(path.len()).kv;
            cp.state_only = false;
        }
        Some((path.len(), cp))
    }

    /// The checkpoint at `path`, read back from disk when RAM no longer has
    /// it; touches it for the LRU.
    fn load(&mut self, path: &[u32]) -> Option<RadixCheckpoint> {
        let stamp = self.touch();
        let node = node_at_mut(&mut self.root, path)?;
        node.used_at = stamp;
        if let Some(cp) = node.checkpoint.as_ref() {
            return Some(cp.clone());
        }
        let name = node.on_disk.as_ref()?.name.clone();
        let cp = self
            .disk
            .as_mut()
            .and_then(|d| d.store.get(&name))
            .and_then(|bytes| decode_file(&bytes, path));
        let Some(cp) = cp else {
            tracing::warn!(name, "prompt cache: checkpoint file gone or unreadable");
            if let Some(file) = node.on_disk.take() {
                self.disk_remove(&file);
            }
            self.prune(path);
            return None;
        };
        self.checkpoint_bytes = self.checkpoint_bytes.saturating_add(cp.approx_bytes());
        node.checkpoint = Some(cp.clone());
        self.evict_to_cap();
        Some(cp)
    }

    /// Longest common prefix length (token count), checkpoint or not.
    #[allow(dead_code)]
    pub fn longest_common_prefix(&mut self, tokens: &[u32]) -> usize {
        let mut node = &self.root;
        let mut depth = 0usize;
        while depth < tokens.len() {
            let Some(child) = node.children.get(&tokens[depth]) else {
                break;
            };
            let common = child
                .edge
                .iter()
                .zip(&tokens[depth..])
                .take_while(|(a, b)| a == b)
                .count();
            depth += common;
            if common < child.edge.len() {
                break;
            }
            node = child;
        }
        depth
    }
}

/// The node at the end of `tokens`, creating and splitting nodes on the way.
fn node_insert<'a>(root: &'a mut RadixNode, nodes: &mut usize, tokens: &[u32]) -> &'a mut RadixNode {
    let mut node = root;
    let mut at = 0usize;
    while at < tokens.len() {
        let key = tokens[at];
        let Some(child) = node.children.get_mut(&key) else {
            node.children.insert(
                key,
                Box::new(RadixNode {
                    edge: tokens[at..].to_vec(),
                    ..RadixNode::default()
                }),
            );
            *nodes += 1;
            return node.children.get_mut(&key).expect("just inserted");
        };
        let common = child
            .edge
            .iter()
            .zip(&tokens[at..])
            .take_while(|(a, b)| a == b)
            .count();
        if common < child.edge.len() {
            // Split: a new node for the shared part, the old child below it.
            let mut lower = node.children.remove(&key).expect("checked");
            let upper_edge = lower.edge[..common].to_vec();
            lower.edge.drain(..common);
            let mut upper = Box::new(RadixNode {
                edge: upper_edge,
                ..RadixNode::default()
            });
            upper.children.insert(lower.edge[0], lower);
            node.children.insert(key, upper);
            *nodes += 1;
        }
        at += common;
        node = node.children.get_mut(&key).expect("present");
    }
    node
}

fn node_at<'a>(root: &'a RadixNode, tokens: &[u32]) -> Option<&'a RadixNode> {
    let mut node = root;
    let mut at = 0usize;
    while at < tokens.len() {
        let child = node.children.get(&tokens[at])?;
        let end = at + child.edge.len();
        if end > tokens.len() || child.edge[..] != tokens[at..end] {
            return None;
        }
        at = end;
        node = child;
    }
    Some(node)
}

fn node_at_mut<'a>(root: &'a mut RadixNode, tokens: &[u32]) -> Option<&'a mut RadixNode> {
    let mut node = root;
    let mut at = 0usize;
    while at < tokens.len() {
        let child = node.children.get_mut(&tokens[at])?;
        let end = at + child.edge.len();
        if end > tokens.len() || child.edge[..] != tokens[at..end] {
            return None;
        }
        at = end;
        node = child;
    }
    Some(node)
}

/// The path of the parent of the node at `path`.
fn parent_path(root: &RadixNode, path: &[u32]) -> Option<Vec<u32>> {
    let mut node = root;
    let mut at = 0usize;
    while at < path.len() {
        let child = node.children.get(&path[at])?;
        let end = at + child.edge.len();
        if end == path.len() {
            return Some(path[..at].to_vec());
        }
        at = end;
        node = child;
    }
    None
}

/// The path of the coldest node that `pick` accepts.
fn coldest(root: &RadixNode, pick: impl Fn(&RadixNode) -> bool) -> Option<Vec<u32>> {
    let mut best: Option<(u64, Vec<u32>)> = None;
    let mut path = Vec::new();
    // Depth-first with an explicit stack: (node, path length before its edge).
    let mut stack: Vec<(&RadixNode, usize)> = root.children.values().map(|c| (&**c, 0)).collect();
    while let Some((node, base)) = stack.pop() {
        path.truncate(base);
        path.extend_from_slice(&node.edge);
        if pick(node) && best.as_ref().is_none_or(|(t, _)| node.used_at < *t) {
            best = Some((node.used_at, path.clone()));
        }
        let len = path.len();
        stack.extend(node.children.values().map(|c| (&**c, len)));
    }
    best.map(|(_, p)| p)
}

/// The tokens from the end of `node` down to a sliceable checkpoint at or
/// below it (empty when `node` itself holds one).
fn sliceable_below(node: &RadixNode) -> Option<Vec<u32>> {
    let mut stack = vec![(node, Vec::new())];
    while let Some((n, path)) = stack.pop() {
        if n.sliceable() {
            return Some(path);
        }
        for child in n.children.values() {
            let mut p = path.clone();
            p.extend_from_slice(&child.edge);
            stack.push((&**child, p));
        }
    }
    None
}

/// The tokens from the end of `node` to a checkpoint below it that holds
/// keys and values.
fn kv_below(node: &RadixNode) -> Option<Vec<u32>> {
    let mut stack: Vec<(&RadixNode, Vec<u32>)> = node
        .children
        .values()
        .map(|c| (&**c, c.edge.clone()))
        .collect();
    while let Some((n, path)) = stack.pop() {
        if n.has_kv() {
            return Some(path);
        }
        for child in n.children.values() {
            let mut p = path.clone();
            p.extend_from_slice(&child.edge);
            stack.push((&**child, p));
        }
    }
    None
}

const MAGIC: &[u8; 4] = b"APKC";
/// Checkpoint files. 3 has no drafter trailer; 4 may.
const VERSION: u32 = 4;
const VERSION_V3: u32 = 3;
const FLAG_SLICEABLE: u8 = 1;
const FLAG_STATE_ONLY: u8 = 2;

/// Checkpoint file name: a hash of its tokens.
fn file_name(tokens: &[u32]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{h:016x}.ckpt")
}

fn io_error(path: &Path, e: std::io::Error) -> AllpacaError {
    AllpacaError::Fatal(FatalError::InvalidConfig(format!(
        "prompt cache {}: {e}",
        path.display()
    )))
}

/// Write through a temporary file so a crash never leaves half a checkpoint.
fn write_file(file: &Path, data: &[u8]) -> std::io::Result<()> {
    let tmp = file.with_extension("tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, file)
}

/// `MAGIC`, version, tokens, flags, per-layer keys and values, then the
/// recurrent state (layers, ssm_h, ssm_conv, gdn_state, gdn_conv), the host
/// attention keys and values, and the qwen4exp memory.
fn encode_file(tokens: &[u32], cp: &RadixCheckpoint) -> Vec<u8> {
    let mut out = Vec::with_capacity(cp.approx_bytes() + tokens.len() * 4 + 64);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for t in tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    let mut flags = 0;
    if cp.sliceable {
        flags |= FLAG_SLICEABLE;
    }
    if cp.state_only {
        flags |= FLAG_STATE_ONLY;
    }
    out.push(flags);
    out.extend_from_slice(&(cp.kv.len() as u32).to_le_bytes());
    for layer in &cp.kv {
        layer.write_to(&mut out);
    }
    let r = &cp.recurrent;
    for group in [&r.layers, &r.ssm_h, &r.ssm_conv, &r.gdn_state, &r.gdn_conv] {
        out.extend_from_slice(&(group.len() as u32).to_le_bytes());
        for v in group {
            out.extend_from_slice(&(v.len() as u32).to_le_bytes());
            crate::kv_codec::extend_f32s(&mut out, v);
        }
    }
    out.extend_from_slice(&(r.host_kv.len() as u32).to_le_bytes());
    for (k, v) in &r.host_kv {
        crate::kv_codec::write_f32s(&mut out, k);
        crate::kv_codec::write_f32s(&mut out, v);
    }
    let q4x = r.q4x.tokens() > 0;
    out.push(q4x as u8);
    if q4x {
        r.q4x.write_to(&mut out);
    }
    match &cp.draft {
        None => out.push(0),
        Some(draft) => {
            out.push(1);
            out.push(draft.kind);
            let n = u32::try_from(draft.bytes.len()).expect("draft cache fits in a checkpoint file");
            out.extend_from_slice(&n.to_le_bytes());
            out.extend_from_slice(&draft.bytes);
        }
    }
    out
}

fn read_header(r: &mut Reader<'_>) -> Option<(Vec<u32>, u8, u32)> {
    if r.take(4)? != MAGIC {
        return None;
    }
    let version = r.u32()?;
    if version != VERSION && version != VERSION_V3 {
        return None;
    }
    let n = r.u32()? as usize;
    let tokens = (0..n).map(|_| r.u32()).collect::<Option<Vec<_>>>()?;
    let flags = r.u8()?;
    Some((tokens, flags, version))
}

/// Tokens and flags of the checkpoint file `data`.
pub(crate) fn header_of(data: &[u8]) -> Option<(Vec<u32>, u8)> {
    let (tokens, flags, _) = read_header(&mut Reader(data))?;
    Some((tokens, flags))
}

/// Tokens and flags, reading only the start of the file.
fn read_file_header(path: &Path) -> Option<(Vec<u32>, u8)> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut head = [0u8; 12];
    f.read_exact(&mut head).ok()?;
    let n = u32::from_le_bytes(head[8..12].try_into().ok()?) as usize;
    let mut rest = vec![0u8; n.checked_mul(4)?.checked_add(1)?];
    f.read_exact(&mut rest).ok()?;
    let mut all = head.to_vec();
    all.extend_from_slice(&rest);
    let (tokens, flags, _) = read_header(&mut Reader(&all))?;
    Some((tokens, flags))
}

/// The checkpoint in `bytes`, when it is one and belongs to `tokens`.
fn decode_file(bytes: &[u8], tokens: &[u32]) -> Option<RadixCheckpoint> {
    let mut r = Reader(bytes);
    let (found, flags, version) = read_header(&mut r)?;
    if found != tokens {
        return None;
    }
    let n_layers = r.u32()? as usize;
    let kv = (0..n_layers)
        .map(|_| EncodedRows::read_from(&mut r))
        .collect::<Option<Vec<_>>>()?;
    let mut groups = Vec::with_capacity(5);
    for _ in 0..5 {
        let n = r.u32()? as usize;
        let mut group = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            let len = r.u32()? as usize;
            let raw = r.take(len.checked_mul(4)?)?;
            group.push(
                raw.chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<f32>>(),
            );
        }
        groups.push(group);
    }
    let n_host_kv = r.u32()? as usize;
    let mut host_kv = Vec::with_capacity(n_host_kv.min(1024));
    for _ in 0..n_host_kv {
        host_kv.push((r.f32s()?, r.f32s()?));
    }
    let q4x = match r.u8()? {
        0 => None,
        1 => Some(crate::qwen4exp::Qwen4Memory::read_from(&mut r)?),
        _ => return None,
    };
    let draft = if version >= VERSION {
        match r.u8()? {
            0 => None,
            1 => {
                let kind = r.u8()?;
                let n = r.u32()? as usize;
                Some(DraftCache { kind, bytes: r.take(n)?.to_vec() })
            }
            _ => return None,
        }
    } else {
        None
    };
    if !r.0.is_empty() {
        return None;
    }
    let mut groups = groups.into_iter();
    let mut recurrent = RecurrentState::zeros(0, 0);
    recurrent.layers = groups.next()?;
    recurrent.ssm_h = groups.next()?;
    recurrent.ssm_conv = groups.next()?;
    recurrent.gdn_state = groups.next()?;
    recurrent.gdn_conv = groups.next()?;
    recurrent.host_kv = host_kv;
    if let Some(q4x) = q4x {
        recurrent.q4x = q4x;
    }
    Some(RadixCheckpoint {
        recurrent,
        kv,
        sliceable: flags & FLAG_SLICEABLE != 0,
        state_only: flags & FLAG_STATE_ONLY != 0,
        draft,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_cp() -> RadixCheckpoint {
        KvRadixForest::new().checkpoint(RecurrentState::zeros(1, 4), vec![vec![]])
    }

    fn fat_cp(n: usize) -> RadixCheckpoint {
        KvRadixForest::new().checkpoint(
            RecurrentState::zeros(1, n),
            vec![vec![KvRow {
                k: vec![0.0; n],
                v: vec![0.0; n],
            }]],
        )
    }

    /// A store whose writes take `delay`, holding its files where the test
    /// can see them.
    #[derive(Debug, Clone, Default)]
    struct SlowStore {
        files: std::sync::Arc<std::sync::Mutex<HashMap<String, Vec<u8>>>>,
        delay: std::time::Duration,
    }

    impl CheckpointStore for SlowStore {
        fn put(&mut self, name: &str, data: &[u8]) -> Result<(), String> {
            std::thread::sleep(self.delay);
            self.files.lock().unwrap().insert(name.to_string(), data.to_vec());
            Ok(())
        }

        fn get(&mut self, name: &str) -> Option<Vec<u8>> {
            self.files.lock().unwrap().get(name).cloned()
        }

        fn remove(&mut self, name: &str) {
            self.files.lock().unwrap().remove(name);
        }

        fn list(&mut self) -> Result<Vec<Listed>, String> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn a_local_folder_deleted_under_the_store_comes_back_on_the_next_write() {
        let dir = std::env::temp_dir().join(format!("allpaca_localdir_gone_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = LocalDir::open(&dir).unwrap();
        // Another server pruned it.
        std::fs::remove_dir_all(&dir).unwrap();
        store.put("a.ckpt", b"data").expect("the folder is made again");
        assert_eq!(store.get("a.ckpt").as_deref(), Some(&b"data"[..]));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn background_writes_leave_the_request_path_and_finish_on_drop() {
        let slow = SlowStore { delay: std::time::Duration::from_millis(300), ..Default::default() };
        let files = slow.files.clone();
        let mut store = BackgroundStore::new(Box::new(slow), 1 << 20);
        let started = std::time::Instant::now();
        store.put("a.ckpt", b"first").unwrap();
        store.put("b.ckpt", b"second").unwrap();
        assert!(started.elapsed() < std::time::Duration::from_millis(150), "{:?}", started.elapsed());
        assert_eq!(store.get("b.ckpt").as_deref(), Some(&b"second"[..]), "a queued file reads back");
        store.remove("b.ckpt");
        assert_eq!(store.get("b.ckpt"), None);
        drop(store);
        let files = files.lock().unwrap();
        assert_eq!(files.get("a.ckpt").map(Vec::as_slice), Some(&b"first"[..]), "dropping waits for the queue");
        assert!(!files.contains_key("b.ckpt"), "a removal after its put wins");
    }

    #[test]
    fn background_refuses_writes_past_its_queue_budget() {
        let slow = SlowStore { delay: std::time::Duration::from_millis(300), ..Default::default() };
        let mut store = BackgroundStore::new(Box::new(slow), 8);
        store.put("a.ckpt", b"12345").unwrap();
        assert!(store.put("b.ckpt", b"12345").is_err(), "10 queued bytes over a budget of 8");
    }

    #[test]
    fn insert_builds_nodes_along_prefix() {
        let mut forest = KvRadixForest::new();
        let added = forest.insert_checkpoint(&[1, 2, 3], empty_cp());
        assert_eq!(added, 1, "one edge holds the run");
        assert_eq!(forest.nodes, 1);
        assert_eq!(forest.insert_checkpoint(&[1, 2, 3], empty_cp()), 0);
        assert_eq!(forest.insert_checkpoint(&[1, 2], empty_cp()), 1, "split");
        assert_eq!(forest.insert_checkpoint(&[1, 9], empty_cp()), 2, "split and leaf");
        assert_eq!(forest.nodes, 4);
    }

    #[test]
    fn lookup_restores_shared_prefix_after_mid_edit() {
        let mut forest = KvRadixForest::new();
        forest.insert_checkpoint(&[10, 11], empty_cp());
        forest.insert_checkpoint(&[10, 11, 12, 13], empty_cp());
        let (n, _) = forest
            .lookup_checkpoint(&[10, 11, 99, 100])
            .expect("LCP hello must restore");
        assert_eq!(n, 2);
        assert!(forest.hits >= 1);
        assert!(forest.skip_tokens >= 2);
    }

    #[test]
    fn lookup_miss_on_unrelated_prompt() {
        let mut forest = KvRadixForest::new();
        forest.insert_checkpoint(&[1, 2, 3], empty_cp());
        assert!(forest.lookup_checkpoint(&[9, 8, 7]).is_none());
        assert!(forest.misses >= 1);
    }

    #[test]
    fn branching_shares_parent_and_picks_deepest_checkpoint() {
        let mut forest = KvRadixForest::new();
        forest.insert_checkpoint(&[1, 2], empty_cp());
        forest.insert_checkpoint(&[1, 2, 3], empty_cp());
        forest.insert_checkpoint(&[1, 2, 4], empty_cp());
        let (n, _) = forest.lookup_checkpoint(&[1, 2, 4]).unwrap();
        assert_eq!(n, 3);
        let (n2, _) = forest.lookup_checkpoint(&[1, 2, 9]).unwrap();
        assert_eq!(n2, 2, "partial walk keeps parent checkpoint");
    }

    #[test]
    fn max_nodes_evicts_leaves() {
        let mut forest = KvRadixForest::with_max_nodes(2);
        forest.insert_checkpoint(&[1], empty_cp());
        forest.insert_checkpoint(&[2], empty_cp());
        forest.insert_checkpoint(&[3], empty_cp());
        assert!(forest.nodes <= 2, "nodes={}", forest.nodes);
        assert!(forest.evictions >= 1, "evictions={}", forest.evictions);
    }

    #[test]
    fn byte_budget_evicts_cold_checkpoints() {
        let mut forest = KvRadixForest::with_limits(0, 800);
        forest.insert_checkpoint(&[1], fat_cp(32)); // warm later
        forest.insert_checkpoint(&[2], fat_cp(32)); // cold
        let warm_bytes = forest.checkpoint_bytes();
        assert!(warm_bytes > 800 || warm_bytes > 0);
        // Touch [1] so [2] is colder.
        let _ = forest.lookup_checkpoint(&[1]);
        forest.insert_checkpoint(&[3], fat_cp(32));
        assert!(
            forest.checkpoint_bytes() <= 800,
            "bytes={} over budget",
            forest.checkpoint_bytes()
        );
        assert!(forest.evictions >= 1, "evictions={}", forest.evictions);
        // Hot prefix should still be restorable more often than the cold one.
        assert!(
            forest.lookup_checkpoint(&[1]).is_some() || forest.lookup_checkpoint(&[3]).is_some(),
            "at least one recent checkpoint must survive"
        );
    }

    #[test]
    fn approx_bytes_tracks_insert_replace() {
        let mut forest = KvRadixForest::new();
        forest.insert_checkpoint(&[1], fat_cp(16));
        let a = forest.checkpoint_bytes();
        assert!(a > 0);
        forest.insert_checkpoint(&[1], fat_cp(8));
        let b = forest.checkpoint_bytes();
        assert!(b < a, "replace must shrink tracked bytes: a={a} b={b}");
    }

    #[test]
    fn a_q4_forest_holds_a_prefix_in_a_fraction_of_the_bytes() {
        let rows: Vec<Vec<KvRow>> = (0..3)
            .map(|layer| {
                (0..8)
                    .map(|t| KvRow {
                        k: (0..64).map(|i| ((layer + t * i) % 11) as f32 - 5.0).collect(),
                        v: (0..64).map(|i| ((t + i) % 7) as f32 * 0.5).collect(),
                    })
                    .collect()
            })
            .collect();
        let mut exact = KvRadixForest::new();
        let cp = exact.checkpoint(RecurrentState::zeros(1, 4), rows.clone());
        exact.insert_checkpoint(&[1, 2, 3], cp);
        let mut small = KvRadixForest::new().with_codec(KvCodec::Q4);
        let cp = small.checkpoint(RecurrentState::zeros(1, 4), rows.clone());
        small.insert_checkpoint(&[1, 2, 3], cp);
        assert!(
            small.checkpoint_bytes() * 4 < exact.checkpoint_bytes(),
            "q4 {} vs f32 {}",
            small.checkpoint_bytes(),
            exact.checkpoint_bytes()
        );

        let (_, cp) = exact.lookup_checkpoint(&[1, 2, 3]).unwrap();
        assert_eq!(cp.kv_rows(), rows, "f32 restores exactly");
        let (_, cp) = small.lookup_checkpoint(&[1, 2, 3]).unwrap();
        let back = cp.kv_rows();
        assert_eq!(back.len(), 3);
        for (a, b) in rows.iter().flatten().zip(back.iter().flatten()) {
            for (x, y) in a.k.iter().chain(&a.v).zip(b.k.iter().chain(&b.v)) {
                assert!((x - y).abs() <= 0.4, "{x} vs {y}");
            }
        }
    }

    #[test]
    fn attention_only_checkpoints_serve_any_shared_prefix() {
        let rows = |n: usize| -> Vec<Vec<KvRow>> {
            vec![(0..n)
                .map(|t| KvRow {
                    k: vec![t as f32; 4],
                    v: vec![-(t as f32); 4],
                })
                .collect()]
        };
        let mut forest = KvRadixForest::new();
        let mut cp = forest.checkpoint(RecurrentState::zeros(1, 4), rows(5));
        cp.sliceable = true;
        forest.insert_checkpoint(&[1, 2, 3, 4, 5], cp);
        let (n, cp) = forest.lookup_checkpoint(&[1, 2, 3, 9, 9]).expect("prefix of a stored prompt");
        assert_eq!(n, 3);
        assert_eq!(cp.kv_rows(), rows(3));

        // A recurrent checkpoint only resumes where it was taken.
        let mut hybrid = KvRadixForest::new();
        let cp = hybrid.checkpoint(RecurrentState::zeros(1, 4), rows(5));
        hybrid.insert_checkpoint(&[1, 2, 3, 4, 5], cp);
        assert!(hybrid.lookup_checkpoint(&[1, 2, 3, 9, 9]).is_none());
    }

    fn disk_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join("allpaca_prompt_cache_tests")
            .join(format!("{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn rich_cp(seed: u32) -> RadixCheckpoint {
        let rows = (0..2)
            .map(|layer| {
                (0..3)
                    .map(|t| KvRow {
                        k: (0..8).map(|i| (seed + layer + t * i) as f32 * 0.25).collect(),
                        v: (0..8).map(|i| (seed * 3 + t + i) as f32 * -0.5).collect(),
                    })
                    .collect()
            })
            .collect();
        let mut state = RecurrentState::zeros(2, 4);
        state.gdn_state = vec![vec![seed as f32; 16]; 2];
        state.gdn_conv = vec![vec![1.5; 6]; 2];
        KvRadixForest::new().checkpoint(state, rows)
    }

    #[test]
    fn a_checkpoint_pushed_out_of_ram_comes_back_from_disk() {
        let dir = disk_dir("spill");
        let one = rich_cp(1);
        let budget = one.approx_bytes() * 3 / 2;
        let mut forest = KvRadixForest::with_limits(0, budget).with_disk(&dir, 0).unwrap();
        forest.insert_checkpoint(&[1, 2, 3], one.clone());
        forest.insert_checkpoint(&[4, 5, 6], rich_cp(2));
        assert!(forest.checkpoint_bytes() <= budget, "{}", forest.checkpoint_bytes());
        assert!(forest.disk_bytes() > 0);

        let (n, cp) = forest.lookup_checkpoint(&[1, 2, 3, 7]).expect("back from disk");
        assert_eq!(n, 3);
        assert_eq!(cp.kv_rows(), one.kv_rows());
        assert_eq!(cp.recurrent.gdn_state, one.recurrent.gdn_state);
        assert_eq!(cp.recurrent.gdn_conv, one.recurrent.gdn_conv);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A later turn resumes the drafter that was warmed with this prefix.
    /// The bytes ride in the checkpoint file, so a RAM miss still brings them back.
    #[test]
    fn a_checkpoint_keeps_the_drafter_state_it_was_saved_with() {
        let dir = disk_dir("draft");
        let mut saved = rich_cp(1);
        saved.draft = Some(DraftCache {
            kind: DRAFT_MTP,
            bytes: vec![9, 8, 7, 6],
        });
        let mut forest = KvRadixForest::with_limits(0, 1).with_disk(&dir, 0).unwrap();
        forest.insert_checkpoint(&[1, 2, 3], saved);
        assert_eq!(forest.checkpoint_bytes(), 0, "the checkpoint left RAM");

        let (n, back) = forest.lookup_checkpoint(&[1, 2, 3, 9]).expect("back from disk");
        assert_eq!(n, 3);
        let draft = back.draft.expect("drafter state");
        assert_eq!(draft.kind, DRAFT_MTP);
        assert_eq!(draft.bytes, vec![9, 8, 7, 6]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_shorter_cut_does_not_keep_the_drafter_from_a_longer_prefix() {
        let mut full = rich_cp(1);
        full.draft = Some(DraftCache {
            kind: DRAFT_MTP,
            bytes: vec![1, 2, 3, 4],
        });
        let rows = full.kv[0].rows();
        assert!(rows > 1);
        assert!(full.prefix(rows - 1).draft.is_none());
        assert!(full.prefix(rows).draft.is_none());
    }

    #[test]
    fn checkpoints_on_disk_outlive_the_forest() {
        let dir = disk_dir("restart");
        let one = rich_cp(5);
        {
            let mut forest = KvRadixForest::new().with_disk(&dir, 0).unwrap();
            forest.insert_checkpoint(&[9, 8, 7], one.clone());
        }
        let mut forest = KvRadixForest::new().with_disk(&dir, 0).unwrap();
        let (n, cp) = forest.lookup_checkpoint(&[9, 8, 7, 1]).expect("found after a restart");
        assert_eq!(n, 3);
        assert_eq!(cp.kv_rows(), one.kv_rows());
        assert!(forest.lookup_checkpoint(&[9, 1]).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_disk_budget_drops_the_coldest_files() {
        let dir = disk_dir("budget");
        let size = rich_cp(1).approx_bytes();
        let mut forest = KvRadixForest::with_limits(0, 1)
            .with_disk(&dir, size * 5 / 2)
            .unwrap();
        forest.insert_checkpoint(&[1], rich_cp(1));
        forest.insert_checkpoint(&[2], rich_cp(2));
        let _ = forest.lookup_checkpoint(&[1]);
        forest.insert_checkpoint(&[3], rich_cp(3));
        assert!(forest.disk_bytes() <= size * 5 / 2, "{}", forest.disk_bytes());
        let files = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(files, 2);
        assert!(forest.lookup_checkpoint(&[2]).is_none(), "coldest is gone");
        assert!(forest.lookup_checkpoint(&[1]).is_some());
        assert!(forest.lookup_checkpoint(&[3]).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn state(v: f32) -> RecurrentState {
        let mut s = RecurrentState::zeros(1, 4);
        s.gdn_state = vec![vec![v; 8]];
        s
    }

    fn rows_upto(n: usize) -> Vec<Vec<KvRow>> {
        vec![(0..n)
            .map(|t| KvRow {
                k: vec![t as f32; 4],
                v: vec![-(t as f32); 4],
            })
            .collect()]
    }

    #[test]
    fn a_state_checkpoint_takes_its_keys_from_a_longer_prompt() {
        let dir = disk_dir("state");
        for reopen in [false, true] {
            let mut forest = KvRadixForest::new().with_disk(&dir, 0).unwrap();
            if !reopen {
                let full = forest.checkpoint(state(6.0), rows_upto(6));
                forest.insert_checkpoint(&[1, 2, 3, 4, 5, 6], full);
                forest.insert_state(&[1, 2, 3], state(3.0));
            }
            let (n, cp) = forest
                .lookup_checkpoint(&[1, 2, 3, 9, 9])
                .unwrap_or_else(|| panic!("state checkpoint (reopen {reopen})"));
            assert_eq!(n, 3);
            assert_eq!(cp.recurrent.gdn_state, state(3.0).gdn_state, "its own state");
            assert_eq!(cp.kv_rows(), rows_upto(3), "keys from the longer prompt");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_state_checkpoint_without_keys_below_is_passed_over() {
        let mut forest = KvRadixForest::new();
        let short = forest.checkpoint(state(1.0), rows_upto(1));
        forest.insert_checkpoint(&[1], short);
        forest.insert_state(&[1, 2, 3], state(3.0));
        let (n, cp) = forest.lookup_checkpoint(&[1, 2, 3, 9]).expect("the shorter one");
        assert_eq!(n, 1);
        assert_eq!(cp.recurrent.gdn_state, state(1.0).gdn_state);
    }

    #[test]
    fn a_long_prompt_is_one_node_and_drops_cleanly() {
        let tokens: Vec<u32> = (0..200_000).collect();
        let mut forest = KvRadixForest::new();
        forest.insert_checkpoint(&tokens, empty_cp());
        assert_eq!(forest.nodes, 1);
        let mut edited = tokens.clone();
        edited[150_000] = 7;
        forest.insert_checkpoint(&edited, empty_cp());
        assert_eq!(forest.nodes, 3);
        let (n, _) = forest.lookup_checkpoint(&tokens).unwrap();
        assert_eq!(n, tokens.len());
        drop(forest);
    }

    #[test]
    fn checkpoint_bytes_count_the_whole_recurrent_state() {
        let forest = KvRadixForest::new();
        let bare = forest.checkpoint(RecurrentState::zeros(2, 4), vec![vec![]]).approx_bytes();
        let mut state = RecurrentState::zeros(2, 4);
        state.gdn_state = vec![vec![0.0; 1000]; 2];
        state.gdn_conv = vec![vec![0.0; 100]; 2];
        state.ssm_h = vec![vec![0.0; 10]; 2];
        let full = forest.checkpoint(state, vec![vec![]]).approx_bytes();
        assert_eq!(full - bare, (2 * 1000 + 2 * 100 + 2 * 10) * 4);
    }

    #[test]
    fn trim_to_bytes_frees_cold_checkpoints() {
        let mut forest = KvRadixForest::with_limits(0, 0);
        forest.insert_checkpoint(&[1], fat_cp(64));
        forest.insert_checkpoint(&[2], fat_cp(64));
        forest.insert_checkpoint(&[3], fat_cp(64));
        let before = forest.checkpoint_bytes();
        assert!(before > 200, "before={before}");
        let freed = forest.trim_to_bytes(before / 3);
        assert!(freed > 0, "freed={freed}");
        assert!(
            forest.checkpoint_bytes() <= before / 3 + 64,
            "after={}",
            forest.checkpoint_bytes()
        );
    }
}
