//! Whole Qwen3-Coder-Next layer spans on an RPC worker.

use allpaca_gguf::{write_gguf_quant, GgmlType, GgufFile, GgufValue, GgufWriteQuantTensor};
use allpaca_types::{AllpacaError, FatalError};

pub fn write_span_gguf(
    path: &std::path::Path,
    file: &GgufFile,
    start: usize,
    end: usize,
) -> Result<(), AllpacaError> {
    let cfg = crate::model::ModelConfig::from_gguf(file)?;
    if start >= end || end > cfg.n_layers {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "invalid remote layer span {start}..{end} of {}",
            cfg.n_layers
        ))));
    }
    let arch = file.architecture().unwrap_or("llama").to_string();
    let mut metadata = file.metadata.clone();
    metadata.insert(
        format!("{arch}.block_count"),
        GgufValue::Uint32((end - start) as u32),
    );
    metadata.insert(format!("{arch}.vocab_size"), GgufValue::Uint32(1));
    metadata.insert(
        "allpaca.gguf_layer_base".into(),
        GgufValue::Uint32(start as u32),
    );

    let mut tensors = Vec::new();
    for info in &file.tensors {
        let in_span = info
            .name
            .strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
            .and_then(|(blk, _)| blk.parse::<usize>().ok())
            .is_some_and(|blk| (start..end).contains(&blk));
        if in_span {
            tensors.push(GgufWriteQuantTensor {
                name: info.name.clone(),
                dims: info.dims.clone(),
                ggml_type: info.ggml_type,
                bytes: file.tensor_bytes(&info.name)?,
            });
        }
    }
    let embed = vec![0.0f32; cfg.dim];
    let norm = vec![1.0f32; cfg.dim];
    let head = vec![0.0f32; cfg.dim];
    let f32_bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let (embed_b, norm_b, head_b) = (f32_bytes(&embed), f32_bytes(&norm), f32_bytes(&head));
    for (name, bytes) in [
        ("token_embd.weight", &embed_b),
        ("output_norm.weight", &norm_b),
        ("output.weight", &head_b),
    ] {
        tensors.push(GgufWriteQuantTensor {
            name: name.into(),
            dims: vec![cfg.dim as u64, 1],
            ggml_type: GgmlType::F32,
            bytes,
        });
    }
    write_gguf_quant(path, &metadata, &[], &tensors)
}

struct HostedModel {
    model: crate::model::TinyModel,
    resident: crate::model::ResidentModel,
    hip: allpaca_hip::HipWorkspace,
    metrics: crate::telemetry::Metrics,
}

struct HostedSeq {
    id: u64,
    recurrent: crate::anchors::RecurrentState,
    kv: crate::kv_tier::KvTierManager,
    pages: Vec<Vec<u64>>,
}

impl HostedModel {
    fn load(path: &std::path::Path, device: i32) -> Result<Self, AllpacaError> {
        let file = GgufFile::open(path)?;
        let model = crate::model::TinyModel::from_gguf(&file)?;
        let resident = crate::model::ResidentModel::upload_gguf_peer_layers(
            &file,
            &model.config,
            device,
            &[],
            &[],
        )?;
        let hip = allpaca_hip::HipWorkspace::new(device, 8 * 1024 * 1024, 256)?;
        Ok(Self {
            model,
            resident,
            hip,
            metrics: crate::telemetry::Metrics::default(),
        })
    }

    fn new_seq(&self, id: u64) -> Result<HostedSeq, AllpacaError> {
        let cfg = &self.model.config;
        let mut kv = crate::kv_tier::KvTierManager::with_hip(
            64 << 20,
            64 << 20,
            0.75,
            Some(self.hip.device),
        );
        let keeps = crate::model::kv_layer_mask(&self.model);
        let pages = crate::model::alloc_pages_for_seq_masked(
            &mut kv,
            cfg,
            128,
            1,
            Some(&keeps),
            &self.metrics,
        )?;
        Ok(HostedSeq {
            id,
            recurrent: crate::anchors::RecurrentState::zeros(cfg.n_layers, cfg.dim),
            kv,
            pages,
        })
    }

    fn run(&self, seq: &mut HostedSeq, pos: usize, x: &mut [f32]) -> Result<(), AllpacaError> {
        crate::model::ensure_pages(
            &mut seq.kv,
            &self.model.config,
            128,
            &mut seq.pages,
            pos,
            &self.metrics,
        )?;
        self.model.forward_hidden_token_device(
            pos,
            x,
            &seq.pages,
            &mut seq.kv,
            &self.metrics,
            &self.hip,
            &self.resident,
            &mut seq.recurrent,
            seq.id,
        )
    }

    fn clone_seq(&self, src: &HostedSeq, dst: &mut HostedSeq) -> Result<(), AllpacaError> {
        let rows = crate::anchors::snapshot_kv(&src.kv, &src.pages, &self.model.config)?;
        crate::anchors::restore_kv(&mut dst.kv, &dst.pages, &rows, &self.model.config)?;
        dst.recurrent = src.recurrent.clone();
        Ok(())
    }
}

pub const EXT: &str = "q3n-layers";

#[derive(Debug, serde::Serialize, serde::Deserialize)]
enum Call {
    Begin { key: u64, len: u64, device: i32 },
    Chunk { offset: u64 },
    Load,
    Run { seq: u64, pos: usize },
    CloneSeq { src: u64, dst: u64 },
    DropSeq { seq: u64 },
}

fn err(msg: String) -> AllpacaError {
    AllpacaError::Fatal(FatalError::ModelLoad(msg))
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn bytes_of(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[derive(Default)]
struct SpanHost {
    model: Option<HostedModel>,
    seqs: std::collections::HashMap<u64, HostedSeq>,
    cache: Option<crate::qwen4_remote::WeightCache>,
    upload: Option<(std::path::PathBuf, u64, u64, u64)>,
    device: i32,
}

impl SpanHost {
    fn upload_path(key: u64) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "allpaca-q3n-{}-{key:016x}.gguf",
            std::process::id()
        ))
    }

    fn load_cached(&mut self, key: u64, len: u64) -> Result<bool, AllpacaError> {
        let Some(data) = self.cache.as_ref().and_then(|cache| cache.get(key, len)) else {
            return Ok(false);
        };
        let path = Self::upload_path(key);
        std::fs::write(&path, data).map_err(FatalError::from)?;
        self.model = Some(HostedModel::load(&path, self.device)?);
        Ok(true)
    }

    fn serve(&mut self, head: &[u8], data: &[u8]) -> Result<Vec<u8>, AllpacaError> {
        let call: Call = serde_json::from_slice(head).map_err(|e| err(format!("q3n call: {e}")))?;
        match call {
            Call::Begin { key, len, device } => {
                self.seqs.clear();
                self.model = None;
                self.upload = None;
                self.device = device;
                if self.load_cached(key, len)? {
                    return Ok(vec![1]);
                }
                let path = Self::upload_path(key);
                let _ = std::fs::remove_file(&path);
                self.upload = Some((path, key, len, 0));
                Ok(vec![0])
            }
            Call::Chunk { offset } => {
                let Some((path, _, _, received)) = self.upload.as_mut() else {
                    return Err(err("q3n chunk before Begin".into()));
                };
                if offset != *received {
                    return Err(err(format!(
                        "q3n chunk offset {offset}, expected {received}"
                    )));
                }
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map_err(FatalError::from)?;
                std::io::Write::write_all(&mut file, data).map_err(FatalError::from)?;
                *received += data.len() as u64;
                Ok(Vec::new())
            }
            Call::Load => {
                let Some((path, key, len, received)) = self.upload.take() else {
                    return Err(err("q3n load before Begin".into()));
                };
                if received != len {
                    return Err(err(format!(
                        "q3n upload has {received} bytes, expected {len}"
                    )));
                }
                self.model = Some(HostedModel::load(&path, self.device)?);
                if let Some(cache) = &self.cache {
                    cache.put(key, &std::fs::read(path).map_err(FatalError::from)?);
                }
                Ok(Vec::new())
            }
            Call::Run { seq, pos } => {
                let Some(model) = &self.model else {
                    return Err(err("q3n run before Load".into()));
                };
                let mut x = floats(data);
                if x.len() != model.model.config.dim {
                    return Err(err(format!(
                        "q3n run has {} floats, expected {}",
                        x.len(),
                        model.model.config.dim
                    )));
                }
                let state = match self.seqs.entry(seq) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(model.new_seq(seq)?)
                    }
                };
                model.run(state, pos, &mut x)?;
                Ok(bytes_of(&x))
            }
            Call::CloneSeq { src, dst } => {
                let Some(model) = &self.model else {
                    return Err(err("q3n clone before Load".into()));
                };
                let Some(source) = self.seqs.get(&src) else {
                    self.seqs.remove(&dst);
                    return Ok(Vec::new());
                };
                let mut target = model.new_seq(dst)?;
                model.clone_seq(source, &mut target)?;
                self.seqs.insert(dst, target);
                Ok(Vec::new())
            }
            Call::DropSeq { seq } => {
                self.seqs.remove(&seq);
                Ok(Vec::new())
            }
        }
    }
}

impl allpaca_rpc::Extension for SpanHost {
    fn name(&self) -> &str {
        EXT
    }

    fn call(&mut self, head: &[u8], data: &[u8]) -> Result<Vec<u8>, String> {
        self.serve(head, data).map_err(|e| e.to_string())
    }
}

pub fn worker_extension(
    cache: Option<crate::qwen4_remote::WeightCache>,
) -> Box<dyn allpaca_rpc::Extension> {
    Box::new(SpanHost {
        cache,
        ..SpanHost::default()
    })
}

#[derive(Debug)]
pub struct RemoteSpan {
    rt: std::sync::Weak<crate::rpc_runtime::RpcRuntime>,
    session: usize,
    next_seq: std::sync::atomic::AtomicU64,
}

fn weight_key_file(path: &std::path::Path) -> Result<(u64, u64), AllpacaError> {
    use std::io::Read as _;

    const M: u64 = 0x9e37_79b9_7f4a_7c15;
    let len = std::fs::metadata(path).map_err(FatalError::from)?.len();
    let mut file = std::fs::File::open(path).map_err(FatalError::from)?;
    let mut h = len.wrapping_mul(M);
    let mut word = [0u8; 8];
    loop {
        let n = file.read(&mut word).map_err(FatalError::from)?;
        if n == 0 {
            break;
        }
        if n < 8 {
            for (i, b) in word[..n].iter().enumerate() {
                h = (h ^ (u64::from(*b) << (8 * i))).wrapping_mul(M);
            }
            break;
        }
        h = (h ^ u64::from_le_bytes(word)).wrapping_mul(M);
        h ^= h >> 29;
    }
    h ^= h >> 32;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    Ok((h ^ (h >> 33), len))
}

impl RemoteSpan {
    pub fn from_gguf(
        rt: &std::sync::Arc<crate::rpc_runtime::RpcRuntime>,
        session: usize,
        device: i32,
        path: &std::path::Path,
    ) -> Result<std::sync::Arc<Self>, AllpacaError> {
        let host = std::sync::Arc::new(Self {
            rt: std::sync::Arc::downgrade(rt),
            session,
            next_seq: std::sync::atomic::AtomicU64::new(1),
        });
        let (key, len) = weight_key_file(path)?;
        let cached = host.call(&Call::Begin { key, len, device }, &[])?.first() == Some(&1);
        if !cached {
            use std::io::Read as _;
            let mut file = std::fs::File::open(path).map_err(FatalError::from)?;
            let mut offset = 0u64;
            let mut chunk = vec![0u8; 16 * 1024 * 1024];
            loop {
                let n = file.read(&mut chunk).map_err(FatalError::from)?;
                if n == 0 {
                    break;
                }
                host.call(&Call::Chunk { offset }, &chunk[..n])?;
                offset += n as u64;
            }
            host.call(&Call::Load, &[])?;
        }
        Ok(host)
    }

    fn call(&self, call: &Call, data: &[u8]) -> Result<Vec<u8>, AllpacaError> {
        let head = serde_json::to_vec(call).map_err(|e| err(format!("q3n call: {e}")))?;
        let rt = self
            .rt
            .upgrade()
            .ok_or_else(|| err("q3n runtime was dropped".into()))?;
        rt.ext(self.session, EXT, &head, data)
    }

    pub fn new_seq(self: &std::sync::Arc<Self>) -> RemoteSeq {
        RemoteSeq {
            host: self.clone(),
            id: self
                .next_seq
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            lost: false,
        }
    }

    pub fn run(&self, seq: &RemoteSeq, pos: usize, x: &mut [f32]) -> Result<(), AllpacaError> {
        if seq.lost {
            return Err(err("q3n sequence state was lost".into()));
        }
        let reply = self.call(&Call::Run { seq: seq.id, pos }, &bytes_of(x))?;
        if reply.len() != x.len() * 4 {
            return Err(err(format!("q3n run returned {} bytes", reply.len())));
        }
        x.copy_from_slice(&floats(&reply));
        if let Some(rt) = self.rt.upgrade() {
            rt.roundtrips
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            rt.activation_bytes
                .fetch_add((x.len() * 8) as u64, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct RemoteSeq {
    host: std::sync::Arc<RemoteSpan>,
    id: u64,
    lost: bool,
}

impl Clone for RemoteSeq {
    fn clone(&self) -> Self {
        let mut copy = self.host.new_seq();
        copy.lost = self.lost
            || self
                .host
                .call(
                    &Call::CloneSeq {
                        src: self.id,
                        dst: copy.id,
                    },
                    &[],
                )
                .inspect_err(|e| tracing::warn!(error = %e, "q3n sequence clone failed"))
                .is_err();
        copy
    }
}

impl Drop for RemoteSeq {
    fn drop(&mut self) {
        let _ = self.host.call(&Call::DropSeq { seq: self.id }, &[]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_gguf_keeps_only_the_requested_layers_and_uses_tiny_globals() {
        let dir = std::env::temp_dir().join(format!("allpaca-r3-span-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.gguf");
        crate::calib::tests::write_tiny_qwen35moe_gguf_with(&source, 4, true);
        let source_file = GgufFile::open(&source).unwrap();
        let span = dir.join("span.gguf");
        write_span_gguf(&span, &source_file, 1, 2).unwrap();

        let file = GgufFile::open(&span).unwrap();
        let data = std::fs::read(&span).unwrap();
        assert_eq!(
            weight_key_file(&span).unwrap(),
            (crate::qwen4_remote::weight_key(&data), data.len() as u64)
        );
        assert!(file.tensor("blk.1.attn_q.weight").is_some());
        assert!(file.tensor("blk.0.attn_qkv.weight").is_none());
        assert!(file.tensor("token_embd.weight").is_some());
        assert_eq!(file.tensor("token_embd.weight").unwrap().n_elements(), 16);
        let arch = file.architecture().unwrap();
        assert_eq!(file.meta_u32(&format!("{arch}.block_count")), Some(1));
        assert_eq!(file.meta_u32(&format!("{arch}.vocab_size")), Some(1));
        let mut cfg = crate::model::ModelConfig::from_gguf(&file).unwrap();
        cfg.gguf_layer_base = 1;
        assert_eq!(cfg.n_layers, 1);
        assert_eq!(cfg.gguf_blk(0), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hidden_spans_compose_exactly_like_the_full_forward() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let dir = std::env::temp_dir().join(format!("allpaca-r3-hidden-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.gguf");
        crate::calib::tests::write_tiny_qwen35moe_gguf_with(&source, 4, true);
        let source_file = GgufFile::open(&source).unwrap();
        let model = crate::model::TinyModel::from_gguf(&source_file).unwrap();
        let token = 3usize;
        let dim = model.config.dim;
        let input = model.embed[token * dim..(token + 1) * dim].to_vec();

        let run = |file: &GgufFile, start: usize, input: &[f32]| {
            let mut model = crate::model::TinyModel::from_gguf(file).unwrap();
            model.config.gguf_layer_base = start;
            let hip = allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 256).unwrap();
            let resident = crate::model::ResidentModel::upload_gguf_peer_layers(
                file,
                &model.config,
                0,
                &[],
                &[],
            )
            .unwrap();
            let metrics = crate::telemetry::Metrics::default();
            let mut kv = crate::kv_tier::KvTierManager::with_hip(64 << 20, 64 << 20, 0.75, Some(0));
            let pages = crate::model::alloc_pages_for_seq(&mut kv, &model.config, 128, 1, &metrics)
                .unwrap();
            let mut recurrent = crate::anchors::RecurrentState::zeros(model.config.n_layers, dim);
            let mut x = input.to_vec();
            model
                .forward_hidden_token_device(
                    0,
                    &mut x,
                    &pages,
                    &mut kv,
                    &metrics,
                    &hip,
                    &resident,
                    &mut recurrent,
                    1,
                )
                .unwrap();
            x
        };

        let whole = run(&source_file, 0, &input);
        let first_path = dir.join("first.gguf");
        write_span_gguf(&first_path, &source_file, 0, 1).unwrap();
        let first_file = GgufFile::open(&first_path).unwrap();
        let first = run(&first_file, 0, &input);
        let second_path = dir.join("second.gguf");
        write_span_gguf(&second_path, &source_file, 1, 2).unwrap();
        let second_file = GgufFile::open(&second_path).unwrap();
        let second = run(&second_file, 1, &first);
        assert_eq!(second, whole);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn cloning_a_worker_sequence_restores_the_accepted_state() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let dir = std::env::temp_dir().join(format!("allpaca-r3-clone-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.gguf");
        crate::calib::tests::write_tiny_qwen35moe_gguf_with(&source, 4, true);
        let source_file = GgufFile::open(&source).unwrap();
        let span = dir.join("span.gguf");
        write_span_gguf(&span, &source_file, 1, 2).unwrap();
        let model = HostedModel::load(&span, 0).unwrap();
        let dim = model.model.config.dim;
        let input = source_file.tensor_f32("token_embd.weight").unwrap()[3 * dim..4 * dim].to_vec();
        let mut accepted = model.new_seq(1).unwrap();
        let mut rejected = model.new_seq(2).unwrap();
        let mut reference = model.new_seq(3).unwrap();
        let mut x = input.clone();
        model.run(&mut accepted, 0, &mut x).unwrap();
        model.clone_seq(&accepted, &mut rejected).unwrap();
        model.clone_seq(&accepted, &mut reference).unwrap();
        model.run(&mut rejected, 1, &mut x).unwrap();
        model.clone_seq(&accepted, &mut rejected).unwrap();
        let mut restored = x.clone();
        let mut expected = x.clone();
        model.run(&mut rejected, 1, &mut restored).unwrap();
        model.run(&mut reference, 1, &mut expected).unwrap();
        assert_eq!(restored, expected);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn rpc_span_runs_and_clones_like_the_worker() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let dir = std::env::temp_dir().join(format!("allpaca-r3-rpc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.gguf");
        crate::calib::tests::write_tiny_qwen35moe_gguf_with(&source, 4, true);
        let source_file = GgufFile::open(&source).unwrap();
        let span_path = dir.join("span.gguf");
        write_span_gguf(&span_path, &source_file, 1, 2).unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let backend = allpaca_rpc::WorkerBackend::Hip(allpaca_rpc::HipWorker::new(0).unwrap());
            allpaca_rpc::serve_connection_ext(stream, backend, None, vec![worker_extension(None)])
                .unwrap();
        });
        let rt = std::sync::Arc::new(
            crate::rpc_runtime::RpcRuntime::connect(&[addr.to_string()], None).unwrap(),
        );
        let span = RemoteSpan::from_gguf(&rt, 0, 0, &span_path).unwrap();
        let dim = source_file.meta_u32("qwen35moe.embedding_length").unwrap() as usize;
        let input = source_file.tensor_f32("token_embd.weight").unwrap()[3 * dim..4 * dim].to_vec();
        let seq = span.new_seq();
        let mut remote = input.clone();
        span.run(&seq, 0, &mut remote).unwrap();

        let direct = HostedModel::load(&span_path, 0).unwrap();
        let mut expected_seq = direct.new_seq(99).unwrap();
        let mut expected = input.clone();
        direct.run(&mut expected_seq, 0, &mut expected).unwrap();
        assert_eq!(remote, expected);

        let accepted = seq.clone();
        let mut rejected = input.clone();
        span.run(&seq, 1, &mut rejected).unwrap();
        let mut restored = rejected.clone();
        let mut reference = rejected.clone();
        span.run(&accepted, 1, &mut restored).unwrap();
        direct.run(&mut expected_seq, 1, &mut reference).unwrap();
        assert_eq!(restored, reference);
        drop((seq, accepted));
        drop(span);
        drop(rt);
        worker.join().unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }
}
