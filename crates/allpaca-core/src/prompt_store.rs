//! Prompt-cache checkpoints kept on an RPC worker: the worker serves the
//! `prompt_cache` extension (a store in its RAM, shared by its connections,
//! so it outlives a coordinator restart) and the coordinator's
//! [`RemoteStore`] puts and gets checkpoint files there, one folder per
//! model and expert mask.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::kv_radix::{header_of, CheckpointStore, Listed};
use crate::rpc_runtime::RpcRuntime;

/// The extension's name in a worker's HELLO.
pub const EXT: &str = "prompt_cache";

const PUT: u8 = 0;
const GET: u8 = 1;
const REMOVE: u8 = 2;
const LIST: u8 = 3;

/// Checkpoint files in a worker's RAM, shared by its connections. Past
/// `max_bytes` (0 = unlimited) the files used longest ago go first.
#[derive(Debug, Default)]
pub struct WorkerStore {
    files: HashMap<String, (Vec<u8>, u64)>,
    bytes: usize,
    max_bytes: usize,
    clock: u64,
}

impl WorkerStore {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            ..Self::default()
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    pub fn put(&mut self, name: &str, data: Vec<u8>) {
        self.remove(name);
        let stamp = self.tick();
        self.bytes += data.len();
        self.files.insert(name.to_string(), (data, stamp));
        while self.max_bytes > 0 && self.bytes > self.max_bytes {
            let Some(oldest) = self.files.iter().min_by_key(|(_, (_, t))| *t).map(|(n, _)| n.clone()) else {
                break;
            };
            self.remove(&oldest);
        }
    }

    pub fn get(&mut self, name: &str) -> Option<&[u8]> {
        let stamp = self.tick();
        let (data, used) = self.files.get_mut(name)?;
        *used = stamp;
        Some(data)
    }

    pub fn remove(&mut self, name: &str) {
        if let Some((data, _)) = self.files.remove(name) {
            self.bytes -= data.len();
        }
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The files under `prefix`, used longest ago first.
    pub fn names(&self, prefix: &str) -> Vec<String> {
        let mut found: Vec<(&String, u64)> = self
            .files
            .iter()
            .filter(|(n, _)| n.starts_with(prefix))
            .map(|(n, (_, t))| (n, *t))
            .collect();
        found.sort_by_key(|f| f.1);
        found.into_iter().map(|(n, _)| n.clone()).collect()
    }

    /// Name, size, flags and tokens of each checkpoint under `prefix`.
    fn list(&self, prefix: &str) -> Vec<u8> {
        let listed: Vec<(String, usize, (Vec<u32>, u8))> = self
            .names(prefix)
            .into_iter()
            .filter_map(|n| {
                let data = &self.files.get(&n)?.0;
                let header = header_of(data)?;
                Some((n, data.len(), header))
            })
            .collect();
        let mut out = Vec::new();
        out.extend_from_slice(&(listed.len() as u32).to_le_bytes());
        for (name, bytes, (tokens, flags)) in listed {
            out.extend_from_slice(&(name.len() as u32).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(bytes as u64).to_le_bytes());
            out.push(flags);
            out.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
            for t in tokens {
                out.extend_from_slice(&t.to_le_bytes());
            }
        }
        out
    }
}

struct StoreExt(Arc<Mutex<WorkerStore>>);

impl allpaca_rpc::Extension for StoreExt {
    fn name(&self) -> &str {
        EXT
    }

    /// `head` is the operation then the file's name; `data` what a put
    /// stores.
    fn call(&mut self, head: &[u8], data: &[u8]) -> Result<Vec<u8>, String> {
        let (&op, name) = head.split_first().ok_or("prompt_cache: empty call")?;
        let name = std::str::from_utf8(name).map_err(|e| format!("prompt_cache: name: {e}"))?;
        let mut store = self.0.lock().map_err(|_| "prompt_cache: store lock")?;
        match op {
            PUT => {
                store.put(name, data.to_vec());
                Ok(Vec::new())
            }
            GET => Ok(match store.get(name) {
                Some(found) => {
                    let mut out = Vec::with_capacity(found.len() + 1);
                    out.push(1);
                    out.extend_from_slice(found);
                    out
                }
                None => vec![0],
            }),
            REMOVE => {
                store.remove(name);
                Ok(Vec::new())
            }
            LIST => Ok(store.list(name)),
            _ => Err(format!("prompt_cache: unknown operation {op}")),
        }
    }
}

/// The worker's end: serves `store` to a connection.
pub fn worker_extension(store: Arc<Mutex<WorkerStore>>) -> Box<dyn allpaca_rpc::Extension> {
    Box::new(StoreExt(store))
}

/// The coordinator's end: checkpoint files in `folder` of the store on
/// worker `session`.
#[derive(Debug)]
pub struct RemoteStore {
    rt: Arc<RpcRuntime>,
    session: usize,
    folder: String,
}

impl RemoteStore {
    pub fn new(rt: Arc<RpcRuntime>, session: usize, folder: &str) -> Self {
        Self {
            rt,
            session,
            folder: folder.to_string(),
        }
    }

    fn call(&self, op: u8, name: &str, data: &[u8]) -> Result<Vec<u8>, String> {
        let mut head = vec![op];
        head.extend_from_slice(self.folder.as_bytes());
        head.push(b'/');
        head.extend_from_slice(name.as_bytes());
        self.rt.ext(self.session, EXT, &head, data).map_err(|e| e.to_string())
    }
}

impl CheckpointStore for RemoteStore {
    fn put(&mut self, name: &str, data: &[u8]) -> Result<(), String> {
        self.call(PUT, name, data).map(|_| ())
    }

    fn get(&mut self, name: &str) -> Option<Vec<u8>> {
        let mut reply = self.call(GET, name, &[]).ok()?;
        (reply.first() == Some(&1)).then(|| reply.split_off(1))
    }

    fn remove(&mut self, name: &str) {
        let _ = self.call(REMOVE, name, &[]);
    }

    fn list(&mut self) -> Result<Vec<Listed>, String> {
        let reply = self.call(LIST, "", &[])?;
        let prefix = format!("{}/", self.folder);
        let mut r = crate::kv_codec::Reader(&reply);
        let bad = || "prompt_cache: bad list reply".to_string();
        let n = r.u32().ok_or_else(bad)? as usize;
        let mut out = Vec::with_capacity(n.min(4096));
        for _ in 0..n {
            let len = r.u32().ok_or_else(bad)? as usize;
            let name = String::from_utf8(r.take(len).ok_or_else(bad)?.to_vec()).map_err(|_| bad())?;
            let bytes = r.u64().ok_or_else(bad)? as usize;
            let flags = r.u8().ok_or_else(bad)?;
            let count = r.u32().ok_or_else(bad)? as usize;
            let tokens = (0..count).map(|_| r.u32()).collect::<Option<Vec<_>>>().ok_or_else(bad)?;
            let name = name.strip_prefix(&prefix).unwrap_or(&name).to_string();
            out.push(Listed { name, bytes, tokens, flags });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anchors::{KvRow, RecurrentState};
    use crate::kv_radix::KvRadixForest;

    fn loopback(store: Arc<Mutex<WorkerStore>>) -> Arc<crate::rpc_runtime::RpcRuntime> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let backend = allpaca_rpc::WorkerBackend::Cpu(allpaca_rpc::CpuWorker::new(1 << 20));
            let _ = allpaca_rpc::serve_connection_ext(stream, backend, None, vec![worker_extension(store)]);
        });
        Arc::new(crate::rpc_runtime::RpcRuntime::connect(&[addr], None).expect("connect"))
    }

    fn rows(n: usize) -> Vec<Vec<KvRow>> {
        vec![(0..n)
            .map(|t| KvRow {
                k: vec![t as f32 + 0.5; 8],
                v: vec![-(t as f32); 8],
            })
            .collect()]
    }

    #[test]
    fn a_forest_keeps_its_checkpoints_on_a_worker() {
        let rt = loopback(Arc::new(Mutex::new(WorkerStore::new(0))));
        let session = rt.session_serving(EXT).expect("the worker serves the store");
        let mut state = RecurrentState::zeros(1, 4);
        state.gdn_state = vec![vec![2.0; 16]];
        let store = |folder: &str| Box::new(RemoteStore::new(rt.clone(), session, folder));
        {
            let mut forest = KvRadixForest::with_limits(0, 1).with_store(store("model-a"), 0).unwrap();
            let cp = forest.checkpoint(state.clone(), rows(3));
            forest.insert_checkpoint(&[1, 2, 3], cp);
            assert_eq!(forest.checkpoint_bytes(), 0, "RAM holds nothing at a 1-byte budget");
            let (n, cp) = forest.lookup_checkpoint(&[1, 2, 3, 9]).expect("from the worker");
            assert_eq!(n, 3);
            assert_eq!(cp.kv_rows(), rows(3));
            assert_eq!(cp.recurrent.gdn_state, state.gdn_state);
        }
        // A coordinator that starts again finds them on the worker.
        let mut forest = KvRadixForest::new().with_store(store("model-a"), 0).unwrap();
        assert!(forest.disk_bytes() > 0);
        assert!(forest.lookup_checkpoint(&[1, 2, 3, 9]).is_some());
        // Another model's folder does not see them.
        let mut other = KvRadixForest::new().with_store(store("model-b"), 0).unwrap();
        assert!(other.lookup_checkpoint(&[1, 2, 3, 9]).is_none());
    }

    #[test]
    fn the_worker_store_drops_the_files_used_longest_ago() {
        let mut store = WorkerStore::new(250);
        store.put("m/a", vec![1; 100]);
        store.put("m/b", vec![2; 100]);
        assert!(store.get("m/a").is_some());
        store.put("m/c", vec![3; 100]);
        assert!(store.get("m/b").is_none(), "b was used longest ago");
        assert!(store.get("m/a").is_some() && store.get("m/c").is_some());
        assert_eq!(store.bytes(), 200);
        store.put("m/a", vec![4; 10]);
        assert_eq!(store.bytes(), 110, "a replaced");
        store.remove("m/c");
        assert_eq!(store.names("m/"), ["m/a"]);
    }
}
