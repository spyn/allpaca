//! DSpark (DFlash + Markov head) draft provider.
//!
//! DeepSeek-V4-Flash-0731 ships a sidecar GGUF (`dspark-*.gguf`) instead of
//! in-file NextN/MTP. The Markov head biases each block position from the
//! previous token. Full DFlash denoising still needs the sidecar's MLA stack;
//! this provider always applies the Markov head and, when the sidecar is a
//! tiny host-ready model, runs those layers first.

use std::sync::{Arc, Mutex};

use allpaca_cpu::matmul_nt;
use allpaca_gguf::{GgmlType, GgufFile};
use allpaca_hip::{DeviceWeights, HipWorkspace};
use allpaca_types::AllpacaError;

use crate::draft::DraftProvider;
use crate::model::TinyModel;

#[derive(Debug, Clone)]
pub struct DsparkSpec {
    pub block_size: usize,
    pub markov_rank: usize,
    pub noise_token_id: u32,
    #[allow(dead_code)]
    pub target_layer_ids: Vec<u32>,
    #[allow(dead_code)]
    pub sample_from_anchor: bool,
}

impl DsparkSpec {
    pub fn from_gguf(file: &GgufFile) -> Self {
        let block = file
            .meta_u32("dspark.block_size")
            .or_else(|| file.meta_u32("dflash.block_size"))
            .unwrap_or(5) as usize;
        let rank = file
            .meta_u32("dspark.markov_rank")
            .or_else(|| {
                markov_tensor(file, "w1")
                    .and_then(|name| file.tensor(name))
                    .and_then(|t| t.dims.first().copied().map(|d| d as u32))
            })
            .unwrap_or(256) as usize;
        let noise = file
            .meta_u32("dspark.noise_token_id")
            .or_else(|| file.meta_u32("dflash.mask_token_id"))
            .unwrap_or(0);
        let layers = file
            .metadata
            .get("dspark.target_layer_ids")
            .and_then(allpaca_gguf::GgufValue::as_u32_array)
            .unwrap_or_default();
        let sample_from_anchor = file
            .meta_bool("dflash.sample_from_anchor")
            .or_else(|| file.meta_bool("dspark.sample_from_anchor"))
            .unwrap_or(true);
        Self {
            block_size: block.max(1),
            markov_rank: rank.max(1),
            noise_token_id: noise,
            target_layer_ids: layers,
            sample_from_anchor,
        }
    }
}

pub struct DsparkProvider {
    spec: DsparkSpec,
    /// Tiny F32 sidecars (host-ready tests). Empty when `w1_bf16` is packed.
    markov_w1: Vec<f32>,
    markov_w2: Vec<f32>,
    /// Packed GGUF BF16 bytes for Unsloth Markov (`[rank, vocab]` / `[rank, vocab]` storage).
    w1_bf16: Vec<u8>,
    w2_bf16: Vec<u8>,
    /// CPU fallback for packed w2 (allocated on first CPU Markov, not at load).
    w2_f32: Mutex<Option<Vec<f32>>>,
    vocab: usize,
    /// Optional host-ready sidecar layers (tiny tests).
    draft: Option<TinyModel>,
    hip: Option<Arc<HipWorkspace>>,
    w2_dev: Option<DeviceWeights>,
}

impl DsparkProvider {
    pub fn from_gguf(file: &GgufFile, draft: Option<TinyModel>) -> Result<Self, AllpacaError> {
        let spec = DsparkSpec::from_gguf(file);
        let w1_name = markov_tensor(file, "w1").unwrap_or("");
        let w2_name = markov_tensor(file, "w2").unwrap_or("");
        if w1_name.is_empty() || w2_name.is_empty() {
            return Ok(Self {
                spec,
                markov_w1: Vec::new(),
                markov_w2: Vec::new(),
                w1_bf16: Vec::new(),
                w2_bf16: Vec::new(),
                w2_f32: Mutex::new(None),
                vocab: draft.as_ref().map(|d| d.config.vocab_size).unwrap_or(0),
                draft,
                hip: None,
                w2_dev: None,
            });
        }
        let vocab = file
            .tensor(w1_name)
            .and_then(|t| t.dims.get(1).copied())
            .unwrap_or(0) as usize;
        let vocab = vocab.max(
            file.tensor(w2_name)
                .and_then(|t| t.dims.get(1).copied())
                .unwrap_or(0) as usize,
        );
        let w1_bf16 = packed_bf16(file, w1_name)?;
        let w2_bf16 = packed_bf16(file, w2_name)?;
        let (markov_w1, markov_w2) = if w1_bf16.is_empty() || w2_bf16.is_empty() {
            (file.tensor_f32(w1_name)?, file.tensor_f32(w2_name)?)
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(Self {
            spec,
            markov_w1,
            markov_w2,
            w1_bf16,
            w2_bf16,
            w2_f32: Mutex::new(None),
            vocab: vocab.max(1),
            draft,
            hip: None,
            w2_dev: None,
        })
    }

    pub fn spec(&self) -> &DsparkSpec {
        &self.spec
    }

    pub fn attach_hip(&mut self, hip: Arc<HipWorkspace>) {
        if self.w2_bf16.is_empty() || self.vocab == 0 {
            return;
        }
        let Ok(blob) = DeviceWeights::alloc(hip.device, self.w2_bf16.len()) else {
            return;
        };
        if blob.copy_h2d(0, &self.w2_bf16).is_err() {
            return;
        }
        tracing::info!(bytes = blob.bytes(), "DSpark Markov w2 packed BF16 on HIP");
        self.w2_dev = Some(blob);
        self.hip = Some(hip);
    }

    fn markov_next(&self, prev: u32) -> Option<u32> {
        if self.vocab == 0 {
            return None;
        }
        let rank = self.spec.markov_rank;
        let tok = prev as usize;
        if tok >= self.vocab {
            return None;
        }
        let mut h = vec![0.0f32; rank];
        if !self.w1_bf16.is_empty() {
            if !gather_bf16_row(&self.w1_bf16, tok, rank, &mut h) {
                return None;
            }
        } else if self.markov_w1.len() >= (tok + 1) * rank {
            h.copy_from_slice(&self.markov_w1[tok * rank..(tok + 1) * rank]);
        } else if self.markov_w1.len() >= rank * self.vocab {
            for r in 0..rank {
                h[r] = self.markov_w1[r + tok * rank];
            }
        } else {
            return None;
        }
        let mut logits = vec![0.0f32; self.vocab];
        let gpu_ok = if let (Some(hip), Some(dev)) = (self.hip.as_ref(), self.w2_dev.as_ref()) {
            hip.gemm_nt_bf16_dev_b(
                &h,
                dev.byte_ptr_at(0) as *const u16,
                &mut logits,
                1,
                self.vocab as i32,
                rank as i32,
            )
            .is_ok()
        } else {
            false
        };
        if !gpu_ok {
            if !self.markov_w2.is_empty() {
                if matmul_nt(&h, &self.markov_w2, &mut logits, 1, self.vocab, rank).is_err() {
                    return None;
                }
            } else {
                let mut guard = self.w2_f32.lock().ok()?;
                if guard.is_none() {
                    if self.w2_bf16.is_empty() {
                        return None;
                    }
                    *guard = Some(dequant_bf16_bytes(&self.w2_bf16));
                }
                let w2 = guard.as_ref()?;
                if matmul_nt(&h, w2, &mut logits, 1, self.vocab, rank).is_err() {
                    return None;
                }
            }
        }
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, v) in logits.iter().enumerate() {
            if *v > best_v {
                best_v = *v;
                best = i;
            }
        }
        Some(best as u32)
    }
}

impl DraftProvider for DsparkProvider {
    fn name(&self) -> &'static str {
        "dspark"
    }

    fn observe(&mut self, _tokens: &[u32]) {}

    fn propose(
        &mut self,
        ctx: &[u32],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<u32>, AllpacaError> {
        if ctx.is_empty() {
            return Ok(Vec::new());
        }
        let n_max = max_tokens.min(self.spec.block_size).max(1);
        if let Some(draft) = &self.draft {
            if draft.host_weights_ready() {
                let mut p = crate::draft::DraftModelProvider::new(draft.clone());
                let mut out = p.propose(ctx, n_max, temperature)?;
                if let Some(&last) = ctx.last() {
                    if let Some(m) = self.markov_next(last) {
                        if out.is_empty() {
                            out.push(m);
                        }
                    }
                }
                return Ok(out);
            }
        }
        let mut out = Vec::new();
        let mut prev = *ctx.last().unwrap();
        for _ in 0..n_max {
            let Some(next) = self.markov_next(prev) else {
                break;
            };
            out.push(next);
            prev = next;
        }
        if out.is_empty() && self.spec.noise_token_id != 0 {
            out.push(self.spec.noise_token_id);
        }
        Ok(out)
    }
}

fn packed_bf16(file: &GgufFile, name: &str) -> Result<Vec<u8>, AllpacaError> {
    let Some(info) = file.tensor(name) else {
        return Ok(Vec::new());
    };
    if info.ggml_type != GgmlType::BF16 {
        return Ok(Vec::new());
    }
    Ok(file.tensor_bytes_of(info)?.to_vec())
}

fn gather_bf16_row(bytes: &[u8], tok: usize, rank: usize, h: &mut [f32]) -> bool {
    let start = tok.saturating_mul(rank).saturating_mul(2);
    let need = rank.saturating_mul(2);
    if start.saturating_add(need) > bytes.len() || h.len() < rank {
        return false;
    }
    for r in 0..rank {
        let o = start + r * 2;
        h[r] = allpaca_cpu::bf16_to_f32(u16::from_le_bytes([bytes[o], bytes[o + 1]]));
    }
    true
}

fn dequant_bf16_bytes(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 2;
    let mut out = vec![0.0f32; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let o = i * 2;
        *slot = allpaca_cpu::bf16_to_f32(u16::from_le_bytes([bytes[o], bytes[o + 1]]));
    }
    out
}

fn markov_tensor<'a>(file: &'a GgufFile, which: &str) -> Option<&'a str> {
    let names: [&str; 4] = match which {
        "w2" => [
            "markov_w2.weight",
            "dspark.markov_w2.weight",
            "markov_w2",
            "dspark.markov_w2",
        ],
        _ => [
            "markov_w1.weight",
            "dspark.markov_w1.weight",
            "markov_w1",
            "dspark.markov_w1",
        ],
    };
    names.into_iter().find(|n| file.tensor(n).is_some())
}

pub fn file_is_dspark(file: &GgufFile) -> bool {
    let arch = file.architecture().unwrap_or("");
    arch.contains("dspark")
        || arch.contains("dflash")
        || markov_tensor(file, "w1").is_some()
        || file.meta_u32("dspark.block_size").is_some()
        || file.meta_u32("dflash.block_size").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
    use std::collections::HashMap;

    #[test]
    fn dspark_markov_proposes_from_sidecar() {
        let path = std::env::temp_dir().join("allpaca_dspark_tiny.gguf");
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("dflash".into()),
        );
        meta.insert("dspark.block_size".into(), GgufValue::Uint32(3));
        meta.insert("dspark.markov_rank".into(), GgufValue::Uint32(2));
        meta.insert("llama.vocab_size".into(), GgufValue::Uint32(4));
        meta.insert("llama.embedding_length".into(), GgufValue::Uint32(2));
        meta.insert("llama.block_count".into(), GgufValue::Uint32(1));
        // w1 [rank=2, vocab=4], identity-ish so token 1 → rank slot
        let w1 = vec![
            1.0, 0.0, // tok0
            0.0, 1.0, // tok1
            1.0, 1.0, // tok2
            0.5, 0.5, // tok3
        ];
        // w2 [vocab=4, rank=2]: maps rank [0,1] toward token 2
        let w2 = vec![0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0];
        write_gguf(
            &path,
            &meta,
            &[
                GgufWriteTensor {
                    name: "dspark.markov_w1.weight".into(),
                    dims: vec![2, 4],
                    data: &w1,
                },
                GgufWriteTensor {
                    name: "dspark.markov_w2.weight".into(),
                    dims: vec![2, 4],
                    data: &w2,
                },
            ],
        )
        .unwrap();
        let file = GgufFile::open(&path).unwrap();
        assert!(file_is_dspark(&file));
        let mut p = DsparkProvider::from_gguf(&file, None).unwrap();
        assert_eq!(p.spec().block_size, 3);
        let draft = p.propose(&[1], 2, 0.0).unwrap();
        assert!(!draft.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dspark_markov_reads_llama_cpp_tensor_names() {
        let path = std::env::temp_dir().join("allpaca_dspark_llamacpp_names.gguf");
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("dflash".into()),
        );
        meta.insert("dflash.block_size".into(), GgufValue::Uint32(3));
        let w1 = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
        let w2 = vec![0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0];
        write_gguf(
            &path,
            &meta,
            &[
                GgufWriteTensor {
                    name: "markov_w1.weight".into(),
                    dims: vec![2, 4],
                    data: &w1,
                },
                GgufWriteTensor {
                    name: "markov_w2.weight".into(),
                    dims: vec![2, 4],
                    data: &w2,
                },
            ],
        )
        .unwrap();
        let file = GgufFile::open(&path).unwrap();
        assert!(file_is_dspark(&file));
        let mut p = DsparkProvider::from_gguf(&file, None).unwrap();
        assert_eq!(p.spec().block_size, 3);
        assert_eq!(p.spec().markov_rank, 2);
        let draft = p.propose(&[1], 2, 0.0).unwrap();
        assert!(!draft.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dspark_markov_reads_packed_bf16_without_f32_materialize() {
        use allpaca_gguf::{
            quantize_bf16, write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor,
            GgufWriteTensor,
        };
        let path = std::env::temp_dir().join("allpaca_dspark_bf16.gguf");
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("dflash".into()),
        );
        meta.insert("dspark.block_size".into(), GgufValue::Uint32(3));
        meta.insert("dspark.markov_rank".into(), GgufValue::Uint32(2));
        let w1 = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
        let w2 = vec![0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0];
        let w1_b = quantize_bf16(&w1);
        let w2_b = quantize_bf16(&w2);
        write_gguf_quant(
            &path,
            &meta,
            &[] as &[GgufWriteTensor<'_>],
            &[
                GgufWriteQuantTensor {
                    name: "markov_w1.weight".into(),
                    dims: vec![2, 4],
                    ggml_type: GgmlType::BF16,
                    bytes: &w1_b,
                },
                GgufWriteQuantTensor {
                    name: "markov_w2.weight".into(),
                    dims: vec![2, 4],
                    ggml_type: GgmlType::BF16,
                    bytes: &w2_b,
                },
            ],
        )
        .unwrap();
        let file = GgufFile::open(&path).unwrap();
        let mut p = DsparkProvider::from_gguf(&file, None).unwrap();
        assert!(p.markov_w1.is_empty());
        assert!(!p.w1_bf16.is_empty());
        let draft = p.propose(&[1], 2, 0.0).unwrap();
        assert!(!draft.is_empty());
        let _ = std::fs::remove_file(&path);
    }
}
