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
        let mut model = crate::model::TinyModel::from_gguf(&file)?;
        model.hosted_moe_compat = true;
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

    fn run_rows(&self, seq: &mut HostedSeq, pos: usize, x: &mut [f32]) -> Result<(), AllpacaError> {
        let dim = self.model.config.dim;
        if x.is_empty() || x.len() % dim != 0 {
            return Err(err(format!(
                "q3n run has {} floats, expected a non-empty multiple of {dim}",
                x.len()
            )));
        }
        crate::model::ensure_pages(
            &mut seq.kv,
            &self.model.config,
            128,
            &mut seq.pages,
            pos + x.len() / dim - 1,
            &self.metrics,
        )?;
        self.model.forward_hidden_tokens_device(
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

    /// Make `dst` continue exactly like `src`.
    fn clone_seq(&self, src: &mut HostedSeq, dst: &mut HostedSeq) -> Result<(), AllpacaError> {
        let cfg = &self.model.config;
        // The live GDN state is on the device once a step has run there; the
        // host copy is only what seeded it.
        self.hip.gdn_sync_request_to_host(src.id, &mut src.recurrent.gdn_state)?;
        self.hip.gdn_sync_conv_to_host(src.id, &mut src.recurrent.gdn_conv)?;
        self.hip
            .mamba2_sync_request_to_host(src.id, &mut src.recurrent.ssm_h, &mut src.recurrent.ssm_conv)?;
        let rows = crate::anchors::snapshot_kv(&src.kv, &src.pages, cfg)?;
        if let Some(len) = rows.iter().map(Vec::len).max().filter(|&n| n > 0) {
            crate::model::ensure_pages(&mut dst.kv, cfg, 128, &mut dst.pages, len - 1, &self.metrics)?;
        }
        crate::anchors::restore_kv(&mut dst.kv, &dst.pages, &rows, cfg)?;
        dst.recurrent = src.recurrent.clone();
        // `dst` seeds its device slots from that host copy on its next step.
        self.hip.gdn_clear_request(dst.id);
        Ok(())
    }

    /// Free `seq`'s device GDN slots.
    fn drop_seq(&self, seq: u64) {
        self.hip.gdn_clear_request(seq);
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

impl Drop for SpanHost {
    /// A connection that ends mid-upload leaves no partial file.
    fn drop(&mut self) {
        self.abandon_upload();
    }
}

impl SpanHost {
    /// Forget an unfinished upload and delete its partial file.
    fn abandon_upload(&mut self) {
        if let Some((path, ..)) = self.upload.take() {
            let _ = std::fs::remove_file(path);
        }
    }

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
        let loaded = HostedModel::load(&path, self.device);
        let _ = std::fs::remove_file(&path);
        self.model = Some(loaded?);
        Ok(true)
    }

    fn serve(&mut self, head: &[u8], data: &[u8]) -> Result<Vec<u8>, AllpacaError> {
        let call: Call = serde_json::from_slice(head).map_err(|e| err(format!("q3n call: {e}")))?;
        match call {
            Call::Begin { key, len, device } => {
                self.seqs.clear();
                self.model = None;
                self.abandon_upload();
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
                let Some((path, _, len, received)) = self.upload.as_mut() else {
                    return Err(err("q3n chunk before Begin".into()));
                };
                if offset != *received {
                    return Err(err(format!(
                        "q3n chunk offset {offset}, expected {received}"
                    )));
                }
                if *received > *len || data.len() as u64 > *len - *received {
                    return Err(err(format!(
                        "q3n chunk exceeds the declared upload length of {len} bytes"
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
                    let _ = std::fs::remove_file(&path);
                    return Err(err(format!(
                        "q3n upload has {received} bytes, expected {len}"
                    )));
                }
                let loaded: Result<HostedModel, AllpacaError> = (|| {
                    let model = HostedModel::load(&path, self.device)?;
                    if let Some(cache) = &self.cache {
                        cache.put(key, &std::fs::read(&path).map_err(FatalError::from)?);
                    }
                    Ok(model)
                })();
                let _ = std::fs::remove_file(&path);
                self.model = Some(loaded?);
                Ok(Vec::new())
            }
            Call::Run { seq, pos } => {
                let Some(model) = &self.model else {
                    return Err(err("q3n run before Load".into()));
                };
                let mut x = floats(data);
                let dim = model.model.config.dim;
                if x.is_empty() || x.len() % dim != 0 {
                    return Err(err(format!(
                        "q3n run has {} floats, expected a non-empty multiple of {dim}",
                        x.len()
                    )));
                }
                let state = match self.seqs.entry(seq) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(model.new_seq(seq)?)
                    }
                };
                if x.len() == dim {
                    model.run(state, pos, &mut x)?;
                } else {
                    model.run_rows(state, pos, &mut x)?;
                }
                Ok(bytes_of(&x))
            }
            Call::CloneSeq { src, dst } => {
                let Some(model) = &self.model else {
                    return Err(err("q3n clone before Load".into()));
                };
                let Some(source) = self.seqs.get_mut(&src) else {
                    self.seqs.remove(&dst);
                    model.drop_seq(dst);
                    return Ok(Vec::new());
                };
                let mut target = model.new_seq(dst)?;
                model.clone_seq(source, &mut target)?;
                self.seqs.insert(dst, target);
                Ok(Vec::new())
            }
            Call::DropSeq { seq } => {
                self.seqs.remove(&seq);
                if let Some(model) = &self.model {
                    model.drop_seq(seq);
                }
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
    let mut host = SpanHost::default();
    host.cache = cache;
    Box::new(host)
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
    let mut file = std::io::BufReader::with_capacity(
        16 * 1024 * 1024,
        std::fs::File::open(path).map_err(FatalError::from)?,
    );
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
        self.run_rows(seq, pos, x)
    }

    pub fn run_rows(&self, seq: &RemoteSeq, pos: usize, x: &mut [f32]) -> Result<(), AllpacaError> {
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
    fn abandoned_and_short_uploads_leave_no_files() {
        let call = |c: &Call| serde_json::to_vec(c).unwrap();
        let mut host = SpanHost::default();
        let (first, second) = (0x5e_0001u64, 0x5e_0002u64);
        host.serve(&call(&Call::Begin { key: first, len: 8, device: 0 }), &[]).unwrap();
        host.serve(&call(&Call::Chunk { offset: 0 }), &[1, 2, 3]).unwrap();
        assert!(SpanHost::upload_path(first).exists());
        // A new upload replaces the unfinished one.
        host.serve(&call(&Call::Begin { key: second, len: 8, device: 0 }), &[]).unwrap();
        assert!(!SpanHost::upload_path(first).exists(), "the abandoned partial file is deleted");
        host.serve(&call(&Call::Chunk { offset: 0 }), &[1, 2, 3]).unwrap();
        assert!(SpanHost::upload_path(second).exists());
        // Loading a short upload fails and deletes it.
        assert!(host.serve(&call(&Call::Load), &[]).is_err());
        assert!(!SpanHost::upload_path(second).exists(), "a short upload is deleted");
    }

    #[test]
    fn chunk_upload_cannot_pass_declared_length() {
        let mut host = SpanHost::default();
        let begin = serde_json::to_vec(&Call::Begin {
            key: 17,
            len: 4,
            device: 0,
        })
        .unwrap();
        host.serve(&begin, &[]).unwrap();
        let chunk = serde_json::to_vec(&Call::Chunk { offset: 0 }).unwrap();
        assert!(host.serve(&chunk, &[1, 2, 3, 4, 5]).is_err());
    }

    #[cfg(feature = "hip")]
    #[test]
    fn load_paths_remove_uploaded_gguf() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let dir = std::env::temp_dir().join(format!("allpaca-r3-upload-cleanup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.gguf");
        crate::calib::tests::write_tiny_qwen35moe_gguf_with(&source, 2, true);
        let source_file = GgufFile::open(&source).unwrap();
        let span = dir.join("span.gguf");
        write_span_gguf(&span, &source_file, 0, 1).unwrap();
        let bytes = std::fs::read(&span).unwrap();
        let key = 23u64;
        let len = bytes.len() as u64;

        let mut host = SpanHost::default();
        let begin = serde_json::to_vec(&Call::Begin { key, len, device: 0 }).unwrap();
        host.serve(&begin, &[]).unwrap();
        let chunk = serde_json::to_vec(&Call::Chunk { offset: 0 }).unwrap();
        host.serve(&chunk, &bytes).unwrap();
        assert!(SpanHost::upload_path(key).exists());
        let load = serde_json::to_vec(&Call::Load).unwrap();
        host.serve(&load, &[]).unwrap();
        assert!(!SpanHost::upload_path(key).exists());

        let cache = crate::qwen4_remote::WeightCache::new(&dir, 1 << 30);
        cache.put(key, &bytes);
        let mut host = SpanHost::default();
        host.cache = Some(cache);
        host.load_cached(key, len).unwrap();
        assert!(!SpanHost::upload_path(key).exists());
        let _ = std::fs::remove_dir_all(dir);
    }

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
        model.clone_seq(&mut accepted, &mut rejected).unwrap();
        model.clone_seq(&mut accepted, &mut reference).unwrap();
        model.run(&mut rejected, 1, &mut x).unwrap();
        model.clone_seq(&mut accepted, &mut rejected).unwrap();
        let mut restored = x.clone();
        let mut expected = x.clone();
        model.run(&mut rejected, 1, &mut restored).unwrap();
        model.run(&mut reference, 1, &mut expected).unwrap();
        assert_eq!(restored, expected);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A clone taken mid-sequence carries the live device GDN state and
    /// keys and values past the first KV page.
    #[cfg(feature = "hip")]
    #[test]
    fn a_clone_past_the_first_page_carries_gdn_and_attention_state() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let dir = std::env::temp_dir().join(format!("allpaca-r3-clone-long-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("source.gguf");
        crate::calib::tests::write_tiny_qwen35moe_gguf_with(&source, 4, true);
        let source_file = GgufFile::open(&source).unwrap();
        let span = dir.join("span.gguf");
        // Layer 0 is Gated DeltaNet, layer 1 full attention.
        write_span_gguf(&span, &source_file, 0, 2).unwrap();
        let model = HostedModel::load(&span, 0).unwrap();
        let dim = model.model.config.dim;
        let embed = source_file.tensor_f32("token_embd.weight").unwrap();
        let vocab = embed.len() / dim;
        let input = |pos: usize| embed[(pos % vocab) * dim..(pos % vocab + 1) * dim].to_vec();
        let steps = 140; // past one 128-token page
        let mut live = model.new_seq(1).unwrap();
        let mut reference = model.new_seq(2).unwrap();
        for pos in 0..steps {
            let mut x = input(pos);
            model.run(&mut live, pos, &mut x).unwrap();
            let mut y = input(pos);
            model.run(&mut reference, pos, &mut y).unwrap();
        }
        let mut copy = model.new_seq(3).unwrap();
        model.clone_seq(&mut live, &mut copy).unwrap();
        let mut from_copy = input(steps);
        model.run(&mut copy, steps, &mut from_copy).unwrap();
        let mut from_reference = input(steps);
        model.run(&mut reference, steps, &mut from_reference).unwrap();
        assert_eq!(from_copy, from_reference, "the copy continues like the original");
        let mut from_live = input(steps);
        model.run(&mut live, steps, &mut from_live).unwrap();
        assert_eq!(from_live, from_reference, "cloning leaves the source as it was");
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

    /// A hosted span's multi-row forward (prefill, verify) must round its
    /// routed MoE like single-token decode, which matches a separate
    /// `MoeFfn` reply (host top-k, F16 routed sum). Needs a width the fused
    /// GPU-routed kernels take (256) and stacked Q8_0 experts: the default
    /// tiny model never reached that path, so rows routed on the GPU in F32
    /// and the real model's hosted text drifted from the separate path.
    #[cfg(feature = "hip")]
    #[test]
    fn hosted_rows_round_routed_experts_like_single_token_runs() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let dir = std::env::temp_dir().join(format!("allpaca-r3-compat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f32_src = dir.join("f32.gguf");
        crate::calib::tests::write_tiny_qwen35moe_gguf_sized(&f32_src, 4, false, 256, 32, 256);
        let f32_file = GgufFile::open(&f32_src).unwrap();
        let quant: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)> = f32_file
            .tensors
            .iter()
            .map(|info| {
                if info.name.ends_with("_exps.weight") {
                    let data = f32_file.tensor_f32(&info.name).unwrap();
                    let bytes = allpaca_gguf::quantize_q8_0(&data);
                    (info.name.clone(), info.dims.clone(), GgmlType::Q8_0, bytes)
                } else {
                    let bytes = f32_file.tensor_bytes(&info.name).unwrap().to_vec();
                    (info.name.clone(), info.dims.clone(), info.ggml_type, bytes)
                }
            })
            .collect();
        let tensors: Vec<GgufWriteQuantTensor<'_>> = quant
            .iter()
            .map(|(name, dims, ggml_type, bytes)| GgufWriteQuantTensor {
                name: name.clone(),
                dims: dims.clone(),
                ggml_type: *ggml_type,
                bytes,
            })
            .collect();
        let source = dir.join("q8.gguf");
        write_gguf_quant(&source, &f32_file.metadata, &[], &tensors).unwrap();
        let source_file = GgufFile::open(&source).unwrap();
        let span = dir.join("span.gguf");
        write_span_gguf(&span, &source_file, 0, 2).unwrap();
        let model = HostedModel::load(&span, 0).unwrap();
        assert!(model.model.hosted_moe_compat, "a hosted span must load in compat mode");
        let dim = model.model.config.dim;
        let input = source_file.tensor_f32("token_embd.weight").unwrap()[dim..5 * dim].to_vec();

        let mut rows_seq = model.new_seq(1).unwrap();
        let mut rows = input.clone();
        model.run_rows(&mut rows_seq, 0, &mut rows).unwrap();

        let mut one_seq = model.new_seq(2).unwrap();
        let mut one = input;
        for (t, row) in one.chunks_exact_mut(dim).enumerate() {
            model.run(&mut one_seq, t, row).unwrap();
        }
        assert_eq!(rows, one, "hosted rows must equal hosted single-token runs bit for bit");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn rpc_span_runs_batched_rows_like_the_worker() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let dir = std::env::temp_dir().join(format!("allpaca-r3-rows-{}", std::process::id()));
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
        let embeddings = source_file.tensor_f32("token_embd.weight").unwrap();
        let input = embeddings[dim..5 * dim].to_vec();
        let seq = span.new_seq();
        let mut remote = input.clone();
        span.run_rows(&seq, 0, &mut remote).unwrap();

        let direct = HostedModel::load(&span_path, 0).unwrap();
        let mut expected_seq = direct.new_seq(99).unwrap();
        let mut expected = input;
        direct
            .run_rows(&mut expected_seq, 0, &mut expected)
            .unwrap();
        assert_eq!(remote, expected);

        let mut remote_next = remote.clone();
        let mut expected_next = expected.clone();
        span.run_rows(&seq, 4, &mut remote_next).unwrap();
        direct
            .run_rows(&mut expected_seq, 4, &mut expected_next)
            .unwrap();
        assert_eq!(remote_next, expected_next);
        drop((seq, span, rt));
        worker.join().unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }
}
