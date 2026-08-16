use allpaca_cpu::{attention, attention_with_mask, matmul_nt, rmsnorm, rope, silu};
use allpaca_hip::{DeviceWeights, HipWorkspace};
use crate::forest::TokenForest;
use allpaca_gguf::GgufFile;
use allpaca_onnx::OnnxFile;
use allpaca_safetensors::{HfConfig, SafetensorsFile};
use allpaca_types::{AllpacaError, FatalError};

use crate::kv_tier::KvTierManager;
use crate::telemetry::Metrics;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub ff_dim: usize,
    pub eps: f32,
    pub rope_theta: f32,
}

impl ModelConfig {
    pub fn tiny() -> Self {
        Self {
            vocab_size: 64,
            dim: 32,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            ff_dim: 64,
            eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }

    pub fn draft_tiny() -> Self {
        Self {
            n_layers: 1,
            ..Self::tiny()
        }
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    pub fn from_gguf(file: &GgufFile) -> Result<Self, AllpacaError> {
        let arch = file.architecture().unwrap_or("llama").to_string();
        let vocab = file
            .meta_u32(&format!("{arch}.vocab_size"))
            .or_else(|| {
                file.metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(allpaca_gguf::GgufValue::as_string_array)
                    .map(|t| t.len() as u32)
            })
            .ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad("missing vocab_size".into()))
            })? as usize;
        let dim = file
            .meta_u32(&format!("{arch}.embedding_length"))
            .ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad("missing embedding_length".into()))
            })? as usize;
        let n_layers = file
            .meta_u32(&format!("{arch}.block_count"))
            .ok_or_else(|| AllpacaError::Fatal(FatalError::ModelLoad("missing block_count".into())))?
            as usize;
        let n_heads = file
            .meta_u32(&format!("{arch}.attention.head_count"))
            .unwrap_or(4) as usize;
        let n_kv_heads = file
            .meta_u32(&format!("{arch}.attention.head_count_kv"))
            .unwrap_or(n_heads as u32) as usize;
        let ff_dim = file
            .meta_u32(&format!("{arch}.feed_forward_length"))
            .unwrap_or((dim * 4) as u32) as usize;
        let eps = file
            .meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let rope_theta = file
            .meta_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(10_000.0);
        Ok(Self {
            vocab_size: vocab,
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            ff_dim,
            eps,
            rope_theta,
        })
    }
}

const TENSORS_PER_LAYER: usize = 9;
/// Host-materialize F32 only for tiny GGUFs. A 7B/8B Q8 file is ~32GiB as F32.
const HOST_F32_LIMIT: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WeightKind {
    F32,
    Q8_0,
}

#[derive(Debug, Clone, Copy)]
struct WeightRef {
    off: usize,
    n: usize,
    kind: WeightKind,
}

impl WeightRef {
    fn f32(off: usize, n: usize) -> Self {
        Self {
            off,
            n,
            kind: WeightKind::F32,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LayerOff {
    attn_norm: WeightRef,
    wq: WeightRef,
    wk: WeightRef,
    wv: WeightRef,
    wo: WeightRef,
    ffn_norm: WeightRef,
    w_gate: WeightRef,
    w_up: WeightRef,
    w_down: WeightRef,
}

/// Host-layout map onto a once-uploaded [`DeviceWeights`] blob.
pub struct ResidentModel {
    weights: DeviceWeights,
    embed: WeightRef,
    layers: Vec<LayerOff>,
    norm: WeightRef,
    lm_head: WeightRef,
}

impl ResidentModel {
    pub fn upload(model: &TinyModel, device: i32) -> Result<Self, AllpacaError> {
        let slices = model.weight_slices();
        let mut offsets = Vec::with_capacity(slices.len());
        let mut off = 0usize;
        for s in &slices {
            offsets.push(off);
            off += s.len() * 4;
        }
        let weights = DeviceWeights::upload(device, &slices)?;
        let mut layers = Vec::with_capacity(model.layers.len());
        let mut idx = 1;
        for layer in &model.layers {
            layers.push(LayerOff {
                attn_norm: WeightRef::f32(offsets[idx], layer.attn_norm.len()),
                wq: WeightRef::f32(offsets[idx + 1], layer.wq.len()),
                wk: WeightRef::f32(offsets[idx + 2], layer.wk.len()),
                wv: WeightRef::f32(offsets[idx + 3], layer.wv.len()),
                wo: WeightRef::f32(offsets[idx + 4], layer.wo.len()),
                ffn_norm: WeightRef::f32(offsets[idx + 5], layer.ffn_norm.len()),
                w_gate: WeightRef::f32(offsets[idx + 6], layer.w_gate.len()),
                w_up: WeightRef::f32(offsets[idx + 7], layer.w_up.len()),
                w_down: WeightRef::f32(offsets[idx + 8], layer.w_down.len()),
            });
            idx += TENSORS_PER_LAYER;
        }
        Ok(Self {
            weights,
            embed: WeightRef::f32(offsets[0], model.embed.len()),
            layers,
            norm: WeightRef::f32(offsets[idx], model.norm.len()),
            lm_head: WeightRef::f32(offsets[idx + 1], model.lm_head.len()),
        })
    }

    /// Upload a real GGUF without host-materializing every tensor as F32.
    /// Q8_0 weights stay quantized on device; `output.weight` is dequantized once
    /// so the lm_head GEMM can use a resident F32 matrix.
    pub fn upload_gguf(file: &GgufFile, cfg: &ModelConfig, device: i32) -> Result<Self, AllpacaError> {
        let mut parts: Vec<Vec<u8>> = Vec::new();
        let mut off = 0usize;
        let embed = append_gguf_tensor(file, "token_embd.weight", &mut parts, &mut off)?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(LayerOff {
                attn_norm: append_gguf_tensor(
                    file,
                    &format!("blk.{i}.attn_norm.weight"),
                    &mut parts,
                    &mut off,
                )?,
                wq: append_gguf_tensor(file, &format!("blk.{i}.attn_q.weight"), &mut parts, &mut off)?,
                wk: append_gguf_tensor(file, &format!("blk.{i}.attn_k.weight"), &mut parts, &mut off)?,
                wv: append_gguf_tensor(file, &format!("blk.{i}.attn_v.weight"), &mut parts, &mut off)?,
                wo: append_gguf_tensor(
                    file,
                    &format!("blk.{i}.attn_output.weight"),
                    &mut parts,
                    &mut off,
                )?,
                ffn_norm: append_gguf_tensor(
                    file,
                    &format!("blk.{i}.ffn_norm.weight"),
                    &mut parts,
                    &mut off,
                )?,
                w_gate: append_gguf_tensor(
                    file,
                    &format!("blk.{i}.ffn_gate.weight"),
                    &mut parts,
                    &mut off,
                )?,
                w_up: append_gguf_tensor(file, &format!("blk.{i}.ffn_up.weight"), &mut parts, &mut off)?,
                w_down: append_gguf_tensor(
                    file,
                    &format!("blk.{i}.ffn_down.weight"),
                    &mut parts,
                    &mut off,
                )?,
            });
        }
        let norm = append_gguf_tensor(file, "output_norm.weight", &mut parts, &mut off)?;
        let lm_f32 = match file.tensor_f32("output.weight") {
            Ok(w) => w,
            Err(_) => file.tensor_f32("token_embd.weight")?,
        };
        let mut lm_bytes = Vec::with_capacity(lm_f32.len() * 4);
        for v in &lm_f32 {
            lm_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let lm_head = WeightRef::f32(off, lm_f32.len());
        off += lm_bytes.len();
        let _ = off;
        parts.push(lm_bytes);

        let views: Vec<&[u8]> = parts.iter().map(|p| p.as_slice()).collect();
        let weights = DeviceWeights::upload_bytes(device, &views)?;
        Ok(Self {
            weights,
            embed,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn bytes(&self) -> u64 {
        self.weights.bytes()
    }

    fn f32_or_dequant(
        &self,
        hip: &HipWorkspace,
        w: WeightRef,
    ) -> Result<*const f32, AllpacaError> {
        match w.kind {
            WeightKind::F32 => Ok(self.weights.ptr_at(w.off)),
            WeightKind::Q8_0 => {
                let out = hip.ensure_dequant(w.n)?;
                hip.dequant_q8_0(self.weights.byte_ptr_at(w.off), w.n as i32, out)?;
                Ok(out as *const f32)
            }
        }
    }

    fn gemm_nt(
        &self,
        a: *const f32,
        w: WeightRef,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        match w.kind {
            WeightKind::F32 => allpaca_hip::gemm_nt_device(a, self.weights.ptr_at(w.off), c, m, n, k),
            WeightKind::Q8_0 => {
                allpaca_hip::gemm_nt_q8_0_device(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
            }
        }
    }

    pub fn embed(&self) -> *const f32 {
        self.weights.ptr_at(self.embed.off)
    }

    pub fn attn_norm(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].attn_norm.off)
    }

    pub fn wq(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].wq.off)
    }

    pub fn wk(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].wk.off)
    }

    pub fn wv(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].wv.off)
    }

    pub fn wo(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].wo.off)
    }

    pub fn ffn_norm(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].ffn_norm.off)
    }

    pub fn w_gate(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].w_gate.off)
    }

    pub fn w_up(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].w_up.off)
    }

    pub fn w_down(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].w_down.off)
    }

    pub fn norm(&self) -> *const f32 {
        self.weights.ptr_at(self.norm.off)
    }

    pub fn lm_head(&self) -> *const f32 {
        self.weights.ptr_at(self.lm_head.off)
    }
}

fn empty_layer() -> LayerWeights {
    LayerWeights {
        attn_norm: Vec::new(),
        wq: Vec::new(),
        wk: Vec::new(),
        wv: Vec::new(),
        wo: Vec::new(),
        ffn_norm: Vec::new(),
        w_gate: Vec::new(),
        w_up: Vec::new(),
        w_down: Vec::new(),
    }
}

fn estimated_gguf_f32_bytes(file: &GgufFile, n_layers: usize) -> u64 {
    let mut names = vec![
        "token_embd.weight".to_string(),
        "output_norm.weight".to_string(),
        "output.weight".to_string(),
    ];
    for i in 0..n_layers {
        names.push(format!("blk.{i}.attn_norm.weight"));
        names.push(format!("blk.{i}.attn_q.weight"));
        names.push(format!("blk.{i}.attn_k.weight"));
        names.push(format!("blk.{i}.attn_v.weight"));
        names.push(format!("blk.{i}.attn_output.weight"));
        names.push(format!("blk.{i}.ffn_norm.weight"));
        names.push(format!("blk.{i}.ffn_gate.weight"));
        names.push(format!("blk.{i}.ffn_up.weight"));
        names.push(format!("blk.{i}.ffn_down.weight"));
    }
    names
        .iter()
        .filter_map(|n| file.tensor(n))
        .map(|t| t.n_elements().saturating_mul(4))
        .sum()
}

fn append_gguf_tensor(
    file: &GgufFile,
    name: &str,
    parts: &mut Vec<Vec<u8>>,
    off: &mut usize,
) -> Result<WeightRef, AllpacaError> {
    let info = file.tensor(name).ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!("missing tensor {name}")))
    })?;
    let n = info.n_elements() as usize;
    let (bytes, kind) = match info.ggml_type {
        allpaca_gguf::GgmlType::F32 => (file.tensor_bytes(name)?.to_vec(), WeightKind::F32),
        allpaca_gguf::GgmlType::Q8_0 => (file.tensor_bytes(name)?.to_vec(), WeightKind::Q8_0),
        _ => {
            let f32s = file.tensor_f32(name)?;
            let mut bytes = Vec::with_capacity(f32s.len() * 4);
            for v in &f32s {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            (bytes, WeightKind::F32)
        }
    };
    let w = WeightRef {
        off: *off,
        n,
        kind,
    };
    *off += bytes.len();
    parts.push(bytes);
    Ok(w)
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TinyModel {
    pub config: ModelConfig,
    pub embed: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub norm: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub name: String,
}

impl TinyModel {
    pub fn synthetic(cfg: ModelConfig, seed: u64, name: impl Into<String>) -> Self {
        let mut rng = Lcg::new(seed);
        let dim = cfg.dim;
        let vocab = cfg.vocab_size;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            layers.push(LayerWeights {
                attn_norm: ones(dim),
                wq: rng.vec(dim * dim),
                wk: rng.vec(kv_dim * dim),
                wv: rng.vec(kv_dim * dim),
                wo: rng.vec(dim * dim),
                ffn_norm: ones(dim),
                w_gate: rng.vec(cfg.ff_dim * dim),
                w_up: rng.vec(cfg.ff_dim * dim),
                w_down: rng.vec(dim * cfg.ff_dim),
            });
        }
        Self {
            embed: rng.vec(vocab * dim),
            layers,
            norm: ones(dim),
            lm_head: rng.vec(vocab * dim),
            name: name.into(),
            config: cfg,
        }
    }

    pub fn host_weights_ready(&self) -> bool {
        !self.embed.is_empty()
    }

    pub fn from_gguf(file: &GgufFile) -> Result<Self, AllpacaError> {
        let cfg = ModelConfig::from_gguf(file)?;
        let name = file
            .meta_str("general.name")
            .unwrap_or("gguf")
            .to_string();
        for i in 0..cfg.n_layers {
            for suffix in [
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                let name = format!("blk.{i}.{suffix}");
                if file.tensor(&name).is_none() {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "missing tensor {name}"
                    ))));
                }
            }
        }
        if file.tensor("token_embd.weight").is_none() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "missing tensor token_embd.weight".into(),
            )));
        }
        if file.tensor("output_norm.weight").is_none() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "missing tensor output_norm.weight".into(),
            )));
        }
        let host_bytes = estimated_gguf_f32_bytes(file, cfg.n_layers);
        if host_bytes > HOST_F32_LIMIT {
            return Ok(Self::skeleton(cfg, name));
        }
        let embed = file.tensor_f32("token_embd.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(LayerWeights {
                attn_norm: file.tensor_f32(&format!("blk.{i}.attn_norm.weight"))?,
                wq: file.tensor_f32(&format!("blk.{i}.attn_q.weight"))?,
                wk: file.tensor_f32(&format!("blk.{i}.attn_k.weight"))?,
                wv: file.tensor_f32(&format!("blk.{i}.attn_v.weight"))?,
                wo: file.tensor_f32(&format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: file.tensor_f32(&format!("blk.{i}.ffn_norm.weight"))?,
                w_gate: file.tensor_f32(&format!("blk.{i}.ffn_gate.weight"))?,
                w_up: file.tensor_f32(&format!("blk.{i}.ffn_up.weight"))?,
                w_down: file.tensor_f32(&format!("blk.{i}.ffn_down.weight"))?,
            });
        }
        let norm = file.tensor_f32("output_norm.weight")?;
        let lm_head = match file.tensor_f32("output.weight") {
            Ok(w) => w,
            Err(_) => embed.clone(),
        };
        Ok(Self {
            config: cfg,
            embed,
            layers,
            norm,
            lm_head,
            name,
        })
    }

    fn skeleton(cfg: ModelConfig, name: impl Into<String>) -> Self {
        Self {
            embed: Vec::new(),
            layers: (0..cfg.n_layers).map(|_| empty_layer()).collect(),
            norm: Vec::new(),
            lm_head: Vec::new(),
            name: name.into(),
            config: cfg,
        }
    }

    pub fn from_safetensors(file: &SafetensorsFile, hf: &HfConfig) -> Result<Self, AllpacaError> {
        let n_heads = hf.num_attention_heads.max(1);
        let n_kv_heads = hf.num_key_value_heads.unwrap_or(n_heads).max(1);
        let ff_dim = hf.intermediate_size.unwrap_or(hf.hidden_size * 4);
        let cfg = ModelConfig {
            vocab_size: hf.vocab_size,
            dim: hf.hidden_size,
            n_layers: hf.num_hidden_layers,
            n_heads,
            n_kv_heads,
            ff_dim,
            eps: hf.rms_norm_eps,
            rope_theta: hf.rope_theta,
        };
        let embed = file.tensor_f32("token_embd.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(LayerWeights {
                attn_norm: file.tensor_f32(&format!("blk.{i}.attn_norm.weight"))?,
                wq: file.tensor_f32(&format!("blk.{i}.attn_q.weight"))?,
                wk: file.tensor_f32(&format!("blk.{i}.attn_k.weight"))?,
                wv: file.tensor_f32(&format!("blk.{i}.attn_v.weight"))?,
                wo: file.tensor_f32(&format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: file.tensor_f32(&format!("blk.{i}.ffn_norm.weight"))?,
                w_gate: file.tensor_f32(&format!("blk.{i}.ffn_gate.weight"))?,
                w_up: file.tensor_f32(&format!("blk.{i}.ffn_up.weight"))?,
                w_down: file.tensor_f32(&format!("blk.{i}.ffn_down.weight"))?,
            });
        }
        let norm = file.tensor_f32("output_norm.weight")?;
        let lm_head = match file.tensor_f32("output.weight") {
            Ok(w) => w,
            Err(_) => embed.clone(),
        };
        let name = hf
            .architectures
            .first()
            .cloned()
            .unwrap_or_else(|| "safetensors".into());
        Ok(Self {
            config: cfg,
            embed,
            layers,
            norm,
            lm_head,
            name,
        })
    }

    pub fn from_onnx(file: &OnnxFile, hf: &HfConfig) -> Result<Self, AllpacaError> {
        let n_heads = hf.num_attention_heads.max(1);
        let n_kv_heads = hf.num_key_value_heads.unwrap_or(n_heads).max(1);
        let ff_dim = hf.intermediate_size.unwrap_or(hf.hidden_size * 4);
        let cfg = ModelConfig {
            vocab_size: hf.vocab_size,
            dim: hf.hidden_size,
            n_layers: hf.num_hidden_layers,
            n_heads,
            n_kv_heads,
            ff_dim,
            eps: hf.rms_norm_eps,
            rope_theta: hf.rope_theta,
        };
        let embed = file.tensor_f32("token_embd.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(LayerWeights {
                attn_norm: file.tensor_f32(&format!("blk.{i}.attn_norm.weight"))?,
                wq: file.tensor_f32(&format!("blk.{i}.attn_q.weight"))?,
                wk: file.tensor_f32(&format!("blk.{i}.attn_k.weight"))?,
                wv: file.tensor_f32(&format!("blk.{i}.attn_v.weight"))?,
                wo: file.tensor_f32(&format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: file.tensor_f32(&format!("blk.{i}.ffn_norm.weight"))?,
                w_gate: file.tensor_f32(&format!("blk.{i}.ffn_gate.weight"))?,
                w_up: file.tensor_f32(&format!("blk.{i}.ffn_up.weight"))?,
                w_down: file.tensor_f32(&format!("blk.{i}.ffn_down.weight"))?,
            });
        }
        let norm = file.tensor_f32("output_norm.weight")?;
        let lm_head = match file.tensor_f32("output.weight") {
            Ok(w) => w,
            Err(_) => embed.clone(),
        };
        Ok(Self {
            config: cfg,
            embed,
            layers,
            norm,
            lm_head,
            name: "onnx".into(),
        })
    }

    pub fn weight_bytes(&self) -> u64 {
        let n: usize = self
            .weight_slices()
            .iter()
            .map(|t| t.len())
            .sum();
        (n * 4) as u64
    }

    fn weight_slices(&self) -> Vec<&[f32]> {
        let mut v = Vec::with_capacity(2 + self.layers.len() * TENSORS_PER_LAYER);
        v.push(self.embed.as_slice());
        for layer in &self.layers {
            v.push(layer.attn_norm.as_slice());
            v.push(layer.wq.as_slice());
            v.push(layer.wk.as_slice());
            v.push(layer.wv.as_slice());
            v.push(layer.wo.as_slice());
            v.push(layer.ffn_norm.as_slice());
            v.push(layer.w_gate.as_slice());
            v.push(layer.w_up.as_slice());
            v.push(layer.w_down.as_slice());
        }
        v.push(self.norm.as_slice());
        v.push(self.lm_head.as_slice());
        v
    }

    /// Decode one token at `pos`, writing K/V into paged tiers. Logits written to `logits`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token(
        &self,
        token: u32,
        pos: usize,
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
        hip: Option<&HipWorkspace>,
        resident: Option<&ResidentModel>,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        if token as usize >= cfg.vocab_size {
            return Err(AllpacaError::Fatal(FatalError::Tokenizer(format!(
                "token {token} out of vocab"
            ))));
        }
        if let (Some(hip), Some(resident)) = (hip, resident) {
            match self.forward_token_device(
                token, pos, page_ids, kv, metrics, logits, hip, resident,
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if !self.host_weights_ready() {
                        return Err(e);
                    }
                    tracing::warn!(pos, token, error = %e, "HIP on-device decode failed; host fallback");
                }
            }
        }
        if !self.host_weights_ready() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "large GGUF has no host F32 weights; HIP decode is required".into(),
            )));
        }
        let mut x = self.embed[token as usize * dim..(token as usize + 1) * dim].to_vec();
        let mut scratch = vec![0.0f32; dim.max(cfg.ff_dim).max(kv_dim)];
        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; dim];
        let mut hidden = vec![0.0f32; dim];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            rmsnorm_dispatch(
                hip,
                &x,
                &layer.attn_norm,
                resident.map(|r| r.attn_norm(layer_idx)),
                cfg.eps,
                &mut scratch[..dim],
                Some(metrics),
            )?;
            gemm_nt_dispatch(hip, &scratch[..dim], &layer.wq, resident.map(|r| r.wq(layer_idx)), &mut q, 1, dim, dim)?;
            gemm_nt_dispatch(hip, &scratch[..dim], &layer.wk, resident.map(|r| r.wk(layer_idx)), &mut k, 1, kv_dim, dim)?;
            gemm_nt_dispatch(hip, &scratch[..dim], &layer.wv, resident.map(|r| r.wv(layer_idx)), &mut v, 1, kv_dim, dim)?;
            rope_dispatch(hip, &mut q, cfg.n_heads, head_dim, pos, cfg.rope_theta, Some(metrics))?;
            rope_dispatch(hip, &mut k, cfg.n_kv_heads, head_dim, pos, cfg.rope_theta, Some(metrics))?;

            let page_tokens = kv
                .get(page_ids[layer_idx][0])
                .map(|p| p.capacity_tokens)
                .unwrap_or(128);
            let page_index = pos / page_tokens as usize;
            let slot = (pos % page_tokens as usize) as u32;
            let page_id = page_ids[layer_idx][page_index];
            kv.write_slot(page_id, slot, cfg.n_kv_heads, head_dim, &k, &v)?;
            kv.touch(page_id, 1.0);

            let (k_all, v_all) = gather_kv(kv, &page_ids[layer_idx], cfg.n_kv_heads, head_dim)?;
            let n_kv = k_all.len() / kv_dim;
            attention_dispatch(
                hip,
                &q,
                &k_all,
                &v_all,
                &mut attn_out,
                1,
                n_kv,
                cfg.n_heads,
                cfg.n_kv_heads,
                head_dim,
                true,
            )?;
            gemm_nt_dispatch(hip, &attn_out, &layer.wo, resident.map(|r| r.wo(layer_idx)), &mut hidden, 1, dim, dim)?;
            for i in 0..dim {
                x[i] += hidden[i];
            }

            rmsnorm_dispatch(
                hip,
                &x,
                &layer.ffn_norm,
                resident.map(|r| r.ffn_norm(layer_idx)),
                cfg.eps,
                &mut scratch[..dim],
                Some(metrics),
            )?;
            let mut gate = vec![0.0f32; cfg.ff_dim];
            let mut up = vec![0.0f32; cfg.ff_dim];
            gemm_nt_dispatch(hip, &scratch[..dim], &layer.w_gate, resident.map(|r| r.w_gate(layer_idx)), &mut gate, 1, cfg.ff_dim, dim)?;
            gemm_nt_dispatch(hip, &scratch[..dim], &layer.w_up, resident.map(|r| r.w_up(layer_idx)), &mut up, 1, cfg.ff_dim, dim)?;
            silu_mul_dispatch(hip, &mut gate, &up, Some(metrics))?;
            gemm_nt_dispatch(hip, &gate, &layer.w_down, resident.map(|r| r.w_down(layer_idx)), &mut hidden, 1, dim, cfg.ff_dim)?;
            for i in 0..dim {
                x[i] += hidden[i];
            }
        }

        rmsnorm_dispatch(
            hip,
            &x,
            &self.norm,
            resident.map(|r| r.norm()),
            cfg.eps,
            &mut scratch[..dim],
            Some(metrics),
        )?;
        if logits.len() < cfg.vocab_size {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "logits buffer smaller than vocab".into(),
            )));
        }
        gemm_nt_dispatch(
            hip,
            &scratch[..dim],
            &self.lm_head,
            resident.map(|r| r.lm_head()),
            &mut logits[..cfg.vocab_size],
            1,
            cfg.vocab_size,
            dim,
        )?;
        if hip.is_some() {
            metrics
                .gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            metrics
                .target_gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    /// Sequential target decode with device-resident hidden state and KV.
    #[allow(clippy::too_many_arguments)]
    fn forward_token_device(
        &self,
        token: u32,
        pos: usize,
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
        hip: &HipWorkspace,
        resident: &ResidentModel,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        let vocab = cfg.vocab_size;
        if logits.len() < vocab {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "logits buffer smaller than vocab".into(),
            )));
        }
        let page_tokens = page_ids
            .first()
            .and_then(|p| p.first())
            .and_then(|id| kv.get(*id).map(|p| p.capacity_tokens as usize))
            .unwrap_or(128);
        let max_kv = page_ids
            .iter()
            .map(|layer| layer.len().saturating_mul(page_tokens))
            .max()
            .unwrap_or(pos + 1)
            .max(pos + 1);
        hip.bind()?;
        let buf = hip.ensure_decode(dim, kv_dim, cfg.ff_dim, vocab, max_kv)?;
        match resident.embed.kind {
            WeightKind::F32 => hip.embed_gather(resident.embed(), token, buf.x, dim as i32)?,
            WeightKind::Q8_0 => hip.embed_gather_q8_0(
                resident.weights.byte_ptr_at(resident.embed.off),
                token,
                buf.x,
                dim as i32,
            )?,
        }
        record_elementwise(Some(metrics));

        for (layer_idx, _) in self.layers.iter().enumerate() {
            let layer = &resident.layers[layer_idx];
            hip.rmsnorm_device(
                buf.x,
                resident.f32_or_dequant(hip, layer.attn_norm)?,
                buf.normed,
                dim as i32,
                cfg.eps,
            )?;
            record_elementwise(Some(metrics));
            resident.gemm_nt(buf.normed, layer.wq, buf.q, 1, dim as i32, dim as i32)?;
            resident.gemm_nt(buf.normed, layer.wk, buf.k, 1, kv_dim as i32, dim as i32)?;
            resident.gemm_nt(buf.normed, layer.wv, buf.v, 1, kv_dim as i32, dim as i32)?;
            hip.rope_device(
                buf.q,
                cfg.n_heads as i32,
                head_dim as i32,
                pos as i32,
                cfg.rope_theta,
            )?;
            record_elementwise(Some(metrics));
            hip.rope_device(
                buf.k,
                cfg.n_kv_heads as i32,
                head_dim as i32,
                pos as i32,
                cfg.rope_theta,
            )?;
            record_elementwise(Some(metrics));

            let page_index = pos / page_tokens;
            let slot = (pos % page_tokens) as u32;
            let page_id = page_ids[layer_idx][page_index];
            kv.write_slot_from_device(page_id, slot, cfg.n_kv_heads, head_dim, buf.k, buf.v)?;
            kv.touch(page_id, 1.0);

            let n_kv = kv.pack_kv_device(
                &page_ids[layer_idx],
                cfg.n_kv_heads,
                head_dim,
                buf.packed_k,
                buf.packed_v,
                buf.max_kv,
            )?;
            hip.attention_device(
                buf.q,
                buf.packed_k,
                buf.packed_v,
                buf.attn,
                1,
                n_kv as i32,
                cfg.n_heads as i32,
                cfg.n_kv_heads as i32,
                head_dim as i32,
                true,
            )?;
            resident.gemm_nt(buf.attn, layer.wo, buf.proj, 1, dim as i32, dim as i32)?;
            hip.residual_add(buf.x, buf.proj, dim as i32)?;
            metrics
                .residual_gpu_ops
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            record_elementwise(Some(metrics));

            hip.rmsnorm_device(
                buf.x,
                resident.f32_or_dequant(hip, layer.ffn_norm)?,
                buf.normed,
                dim as i32,
                cfg.eps,
            )?;
            record_elementwise(Some(metrics));
            resident.gemm_nt(
                buf.normed,
                layer.w_gate,
                buf.gate,
                1,
                cfg.ff_dim as i32,
                dim as i32,
            )?;
            resident.gemm_nt(
                buf.normed,
                layer.w_up,
                buf.up,
                1,
                cfg.ff_dim as i32,
                dim as i32,
            )?;
            hip.silu_mul_device(buf.gate, buf.up, cfg.ff_dim as i32)?;
            record_elementwise(Some(metrics));
            resident.gemm_nt(
                buf.gate,
                layer.w_down,
                buf.proj,
                1,
                dim as i32,
                cfg.ff_dim as i32,
            )?;
            hip.residual_add(buf.x, buf.proj, dim as i32)?;
            metrics
                .residual_gpu_ops
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            record_elementwise(Some(metrics));
        }

        hip.rmsnorm_device(
            buf.x,
            resident.f32_or_dequant(hip, resident.norm)?,
            buf.normed,
            dim as i32,
            cfg.eps,
        )?;
        record_elementwise(Some(metrics));
        resident.gemm_nt(
            buf.normed,
            resident.lm_head,
            buf.logits,
            1,
            vocab as i32,
            dim as i32,
        )?;
        hip.publish_logits(buf.logits, &mut logits[..vocab])?;
        metrics
            .gpu_forwards
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        metrics
            .target_gpu_forwards
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Packed Token Forest verify: one target eval with an ancestry attention mask.
    /// HIP stays on the sequential `forward_token` path.
    pub fn forward_forest(
        &self,
        forest: &TokenForest,
        page_ids: &[Vec<u64>],
        kv: &KvTierManager,
        prompt_len: usize,
        node_logits: &mut [Vec<f32>],
    ) -> Result<(), AllpacaError> {
        let n_tree = forest.nodes.len();
        if n_tree == 0 {
            return Ok(());
        }
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        let vocab = cfg.vocab_size;
        let mut xs: Vec<Vec<f32>> = Vec::with_capacity(n_tree);
        for n in &forest.nodes {
            let tok = n.token_id as usize;
            if tok >= cfg.vocab_size {
                return Err(AllpacaError::Fatal(FatalError::Tokenizer(format!(
                    "token {tok} out of vocab"
                ))));
            }
            xs.push(self.embed[tok * dim..(tok + 1) * dim].to_vec());
        }
        let mut scratch = vec![0.0f32; dim.max(cfg.ff_dim).max(kv_dim)];
        let ancestry = forest.ancestry_mask();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (prompt_k, prompt_v) = gather_kv(kv, &page_ids[layer_idx], cfg.n_kv_heads, head_dim)?;
            let n_prompt = prompt_k.len().checked_div(kv_dim).unwrap_or(0);
            let n_kv = n_prompt + n_tree;
            let mut qs = vec![0.0f32; n_tree * dim];
            let mut ks = vec![0.0f32; n_tree * kv_dim];
            let mut vs = vec![0.0f32; n_tree * kv_dim];
            for (i, x) in xs.iter().enumerate() {
                rmsnorm(x, &layer.attn_norm, cfg.eps, &mut scratch[..dim])?;
                matmul_nt(
                    &scratch[..dim],
                    &layer.wq,
                    &mut qs[i * dim..(i + 1) * dim],
                    1,
                    dim,
                    dim,
                )?;
                matmul_nt(
                    &scratch[..dim],
                    &layer.wk,
                    &mut ks[i * kv_dim..(i + 1) * kv_dim],
                    1,
                    kv_dim,
                    dim,
                )?;
                matmul_nt(
                    &scratch[..dim],
                    &layer.wv,
                    &mut vs[i * kv_dim..(i + 1) * kv_dim],
                    1,
                    kv_dim,
                    dim,
                )?;
                let pos = prompt_len + forest.nodes[i].depth as usize - 1;
                for h in 0..cfg.n_heads {
                    rope(&mut qs[i * dim + h * head_dim..i * dim + (h + 1) * head_dim], pos, cfg.rope_theta)?;
                }
                for h in 0..cfg.n_kv_heads {
                    rope(
                        &mut ks[i * kv_dim + h * head_dim..i * kv_dim + (h + 1) * head_dim],
                        pos,
                        cfg.rope_theta,
                    )?;
                }
            }
            let mut k_all = prompt_k;
            k_all.extend_from_slice(&ks);
            let mut v_all = prompt_v;
            v_all.extend_from_slice(&vs);
            let mut mask = vec![false; n_tree * n_kv];
            for i in 0..n_tree {
                for t in 0..n_prompt {
                    mask[i * n_kv + t] = true;
                }
                for j in 0..n_tree {
                    if ancestry[i * n_tree + j] {
                        mask[i * n_kv + n_prompt + j] = true;
                    }
                }
            }
            let mut attn_out = vec![0.0f32; n_tree * dim];
            attention_with_mask(
                &qs,
                &k_all,
                &v_all,
                &mut attn_out,
                n_tree,
                n_kv,
                cfg.n_heads,
                cfg.n_kv_heads,
                head_dim,
                &mask,
            )?;
            for i in 0..n_tree {
                let mut hidden = vec![0.0f32; dim];
                matmul_nt(
                    &attn_out[i * dim..(i + 1) * dim],
                    &layer.wo,
                    &mut hidden,
                    1,
                    dim,
                    dim,
                )?;
                for d in 0..dim {
                    xs[i][d] += hidden[d];
                }
                rmsnorm(&xs[i], &layer.ffn_norm, cfg.eps, &mut scratch[..dim])?;
                let mut gate = vec![0.0f32; cfg.ff_dim];
                let mut up = vec![0.0f32; cfg.ff_dim];
                matmul_nt(&scratch[..dim], &layer.w_gate, &mut gate, 1, cfg.ff_dim, dim)?;
                matmul_nt(&scratch[..dim], &layer.w_up, &mut up, 1, cfg.ff_dim, dim)?;
                for g in 0..cfg.ff_dim {
                    gate[g] = silu(gate[g]) * up[g];
                }
                matmul_nt(&gate, &layer.w_down, &mut hidden, 1, dim, cfg.ff_dim)?;
                for d in 0..dim {
                    xs[i][d] += hidden[d];
                }
            }
        }

        for (i, x) in xs.iter().enumerate() {
            rmsnorm(x, &self.norm, cfg.eps, &mut scratch[..dim])?;
            if node_logits[i].len() < vocab {
                node_logits[i].resize(vocab, 0.0);
            }
            matmul_nt(
                &scratch[..dim],
                &self.lm_head,
                &mut node_logits[i][..vocab],
                1,
                vocab,
                dim,
            )?;
        }
        Ok(())
    }

    /// Draft-path forward with a dense (non-paged) KV cache.
    pub fn forward_dense(
        &self,
        token: u32,
        pos: usize,
        cache: &mut DenseKv,
        logits: &mut [f32],
    ) -> Result<(), AllpacaError> {
        self.forward_dense_on(token, pos, cache, logits, None, None, None)
    }

    /// Draft forward. GEMM/attention run on `hip` when present (resident draft GPU).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_dense_on(
        &self,
        token: u32,
        pos: usize,
        cache: &mut DenseKv,
        logits: &mut [f32],
        hip: Option<&HipWorkspace>,
        resident: Option<&ResidentModel>,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        if let (Some(hip), Some(resident)) = (hip, resident) {
            match self.forward_dense_device(token, pos, cache, logits, hip, resident, metrics) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    tracing::warn!(pos, token, error = %e, "HIP draft on-device decode failed; host fallback");
                }
            }
        }
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        cache.ensure_layers(cfg.n_layers);
        let mut x = self.embed[token as usize * dim..(token as usize + 1) * dim].to_vec();
        let mut normed = vec![0.0f32; dim];
        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; dim];
        let mut hidden = vec![0.0f32; dim];
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            rmsnorm_dispatch(
                hip,
                &x,
                &layer.attn_norm,
                resident.map(|r| r.attn_norm(layer_idx)),
                cfg.eps,
                &mut normed,
                metrics,
            )?;
            gemm_nt_dispatch(hip, &normed, &layer.wq, resident.map(|r| r.wq(layer_idx)), &mut q, 1, dim, dim)?;
            gemm_nt_dispatch(hip, &normed, &layer.wk, resident.map(|r| r.wk(layer_idx)), &mut k, 1, kv_dim, dim)?;
            gemm_nt_dispatch(hip, &normed, &layer.wv, resident.map(|r| r.wv(layer_idx)), &mut v, 1, kv_dim, dim)?;
            rope_dispatch(hip, &mut q, cfg.n_heads, head_dim, pos, cfg.rope_theta, metrics)?;
            rope_dispatch(hip, &mut k, cfg.n_kv_heads, head_dim, pos, cfg.rope_theta, metrics)?;
            cache.layers[layer_idx].k.extend_from_slice(&k);
            cache.layers[layer_idx].v.extend_from_slice(&v);
            let n_kv = cache.layers[layer_idx].k.len() / kv_dim;
            attention_dispatch(
                hip,
                &q,
                &cache.layers[layer_idx].k,
                &cache.layers[layer_idx].v,
                &mut attn_out,
                1,
                n_kv,
                cfg.n_heads,
                cfg.n_kv_heads,
                head_dim,
                true,
            )?;
            gemm_nt_dispatch(hip, &attn_out, &layer.wo, resident.map(|r| r.wo(layer_idx)), &mut hidden, 1, dim, dim)?;
            for i in 0..dim {
                x[i] += hidden[i];
            }
            rmsnorm_dispatch(
                hip,
                &x,
                &layer.ffn_norm,
                resident.map(|r| r.ffn_norm(layer_idx)),
                cfg.eps,
                &mut normed,
                metrics,
            )?;
            let mut gate = vec![0.0f32; cfg.ff_dim];
            let mut up = vec![0.0f32; cfg.ff_dim];
            gemm_nt_dispatch(hip, &normed, &layer.w_gate, resident.map(|r| r.w_gate(layer_idx)), &mut gate, 1, cfg.ff_dim, dim)?;
            gemm_nt_dispatch(hip, &normed, &layer.w_up, resident.map(|r| r.w_up(layer_idx)), &mut up, 1, cfg.ff_dim, dim)?;
            silu_mul_dispatch(hip, &mut gate, &up, metrics)?;
            gemm_nt_dispatch(hip, &gate, &layer.w_down, resident.map(|r| r.w_down(layer_idx)), &mut hidden, 1, dim, cfg.ff_dim)?;
            for i in 0..dim {
                x[i] += hidden[i];
            }
        }
        rmsnorm_dispatch(
            hip,
            &x,
            &self.norm,
            resident.map(|r| r.norm()),
            cfg.eps,
            &mut normed,
            metrics,
        )?;
        gemm_nt_dispatch(
            hip,
            &normed,
            &self.lm_head,
            resident.map(|r| r.lm_head()),
            &mut logits[..cfg.vocab_size],
            1,
            cfg.vocab_size,
            dim,
        )?;
        if hip.is_some() {
            if let Some(metrics) = metrics {
                metrics
                    .gpu_forwards
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                metrics
                    .draft_gpu_forwards
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// Sequential draft decode with device-resident hidden state and DenseKv.
    #[allow(clippy::too_many_arguments)]
    fn forward_dense_device(
        &self,
        token: u32,
        pos: usize,
        cache: &mut DenseKv,
        logits: &mut [f32],
        hip: &HipWorkspace,
        resident: &ResidentModel,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        let vocab = cfg.vocab_size;
        if logits.len() < vocab {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "logits buffer smaller than vocab".into(),
            )));
        }
        hip.bind()?;
        cache.ensure_device(hip.device, cfg.n_layers, kv_dim, pos + 1)?;
        if let Some(metrics) = metrics {
            metrics
                .draft_kv_bytes_on_device
                .store(cache.device_bytes(), std::sync::atomic::Ordering::Relaxed);
        }
        let buf = hip.ensure_decode(dim, kv_dim, cfg.ff_dim, vocab, pos + 1)?;
        hip.embed_gather(resident.embed(), token, buf.x, dim as i32)?;
        record_elementwise(metrics);

        for layer_idx in 0..self.layers.len() {
            hip.rmsnorm_device(
                buf.x,
                resident.attn_norm(layer_idx),
                buf.normed,
                dim as i32,
                cfg.eps,
            )?;
            record_elementwise(metrics);
            allpaca_hip::gemm_nt_device(
                buf.normed,
                resident.wq(layer_idx),
                buf.q,
                1,
                dim as i32,
                dim as i32,
            )?;
            allpaca_hip::gemm_nt_device(
                buf.normed,
                resident.wk(layer_idx),
                buf.k,
                1,
                kv_dim as i32,
                dim as i32,
            )?;
            allpaca_hip::gemm_nt_device(
                buf.normed,
                resident.wv(layer_idx),
                buf.v,
                1,
                kv_dim as i32,
                dim as i32,
            )?;
            hip.rope_device(
                buf.q,
                cfg.n_heads as i32,
                head_dim as i32,
                pos as i32,
                cfg.rope_theta,
            )?;
            record_elementwise(metrics);
            hip.rope_device(
                buf.k,
                cfg.n_kv_heads as i32,
                head_dim as i32,
                pos as i32,
                cfg.rope_theta,
            )?;
            record_elementwise(metrics);
            cache.write_pos_from_device(layer_idx, pos, buf.k, buf.v)?;
            let n_kv = pos + 1;
            hip.attention_device(
                buf.q,
                cache.k_ptr(layer_idx),
                cache.v_ptr(layer_idx),
                buf.attn,
                1,
                n_kv as i32,
                cfg.n_heads as i32,
                cfg.n_kv_heads as i32,
                head_dim as i32,
                true,
            )?;
            allpaca_hip::gemm_nt_device(
                buf.attn,
                resident.wo(layer_idx),
                buf.proj,
                1,
                dim as i32,
                dim as i32,
            )?;
            hip.residual_add(buf.x, buf.proj, dim as i32)?;
            if let Some(metrics) = metrics {
                metrics
                    .residual_gpu_ops
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            record_elementwise(metrics);

            hip.rmsnorm_device(
                buf.x,
                resident.ffn_norm(layer_idx),
                buf.normed,
                dim as i32,
                cfg.eps,
            )?;
            record_elementwise(metrics);
            allpaca_hip::gemm_nt_device(
                buf.normed,
                resident.w_gate(layer_idx),
                buf.gate,
                1,
                cfg.ff_dim as i32,
                dim as i32,
            )?;
            allpaca_hip::gemm_nt_device(
                buf.normed,
                resident.w_up(layer_idx),
                buf.up,
                1,
                cfg.ff_dim as i32,
                dim as i32,
            )?;
            hip.silu_mul_device(buf.gate, buf.up, cfg.ff_dim as i32)?;
            record_elementwise(metrics);
            allpaca_hip::gemm_nt_device(
                buf.gate,
                resident.w_down(layer_idx),
                buf.proj,
                1,
                dim as i32,
                cfg.ff_dim as i32,
            )?;
            hip.residual_add(buf.x, buf.proj, dim as i32)?;
            if let Some(metrics) = metrics {
                metrics
                    .residual_gpu_ops
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            record_elementwise(metrics);
        }

        hip.rmsnorm_device(buf.x, resident.norm(), buf.normed, dim as i32, cfg.eps)?;
        record_elementwise(metrics);
        allpaca_hip::gemm_nt_device(
            buf.normed,
            resident.lm_head(),
            buf.logits,
            1,
            vocab as i32,
            dim as i32,
        )?;
        hip.publish_logits(buf.logits, &mut logits[..vocab])?;
        if let Some(metrics) = metrics {
            metrics
                .gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            metrics
                .draft_gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }
}

struct DeviceLayerKv {
    k: *mut u8,
    v: *mut u8,
    bytes: usize,
}

unsafe impl Send for DeviceLayerKv {}
unsafe impl Sync for DeviceLayerKv {}

impl std::fmt::Debug for DeviceLayerKv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceLayerKv")
            .field("bytes", &self.bytes)
            .finish()
    }
}

#[derive(Debug)]
pub struct DenseKv {
    pub layers: Vec<DenseLayerKv>,
    device: Option<i32>,
    device_layers: Vec<DeviceLayerKv>,
    tokens: usize,
    capacity: usize,
    kv_dim: usize,
}

impl Default for DenseKv {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            device: None,
            device_layers: Vec::new(),
            tokens: 0,
            capacity: 0,
            kv_dim: 0,
        }
    }
}

impl Drop for DenseKv {
    fn drop(&mut self) {
        self.free_device();
    }
}

#[derive(Debug, Default, Clone)]
pub struct DenseLayerKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

impl DenseKv {
    fn ensure_layers(&mut self, n: usize) {
        self.layers.resize_with(n, DenseLayerKv::default);
    }

    pub fn device_bytes(&self) -> u64 {
        self.device_layers.iter().map(|l| (l.bytes * 2) as u64).sum()
    }

    pub fn reset(&mut self) {
        self.tokens = 0;
        for layer in &mut self.layers {
            layer.k.clear();
            layer.v.clear();
        }
    }

    fn free_device(&mut self) {
        for layer in self.device_layers.drain(..) {
            allpaca_hip::device_free(layer.k, layer.bytes);
            allpaca_hip::device_free(layer.v, layer.bytes);
        }
        self.device = None;
        self.capacity = 0;
    }

    fn ensure_device(
        &mut self,
        device: i32,
        n_layers: usize,
        kv_dim: usize,
        min_tokens: usize,
    ) -> Result<(), AllpacaError> {
        allpaca_hip::set_device(device)?;
        self.ensure_layers(n_layers);
        let need = min_tokens.max(1);
        let reuse = self.device == Some(device)
            && self.kv_dim == kv_dim
            && self.device_layers.len() == n_layers
            && self.capacity >= need;
        if reuse {
            return Ok(());
        }
        let new_cap = need.max(self.capacity.saturating_mul(2)).max(8);
        let bytes = new_cap * kv_dim * 4;
        let copy_tokens = if self.device == Some(device) && self.kv_dim == kv_dim {
            self.tokens
        } else {
            0
        };
        let copy_bytes = copy_tokens * kv_dim * 4;
        let mut new_layers: Vec<DeviceLayerKv> = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let k = match allpaca_hip::device_malloc(bytes, device) {
                Ok(p) => p,
                Err(e) => {
                    for layer in new_layers.drain(..) {
                        allpaca_hip::device_free(layer.k, layer.bytes);
                        allpaca_hip::device_free(layer.v, layer.bytes);
                    }
                    return Err(e);
                }
            };
            let v = match allpaca_hip::device_malloc(bytes, device) {
                Ok(p) => p,
                Err(e) => {
                    allpaca_hip::device_free(k, bytes);
                    for layer in new_layers.drain(..) {
                        allpaca_hip::device_free(layer.k, layer.bytes);
                        allpaca_hip::device_free(layer.v, layer.bytes);
                    }
                    return Err(e);
                }
            };
            if copy_bytes > 0 && i < self.device_layers.len() {
                allpaca_hip::memcpy(k, self.device_layers[i].k, copy_bytes, allpaca_hip::MEMCPY_D2D)?;
                allpaca_hip::memcpy(v, self.device_layers[i].v, copy_bytes, allpaca_hip::MEMCPY_D2D)?;
            }
            new_layers.push(DeviceLayerKv { k, v, bytes });
        }
        self.free_device();
        self.device_layers = new_layers;
        self.device = Some(device);
        self.capacity = new_cap;
        self.kv_dim = kv_dim;
        Ok(())
    }

    fn write_pos_from_device(
        &mut self,
        layer: usize,
        pos: usize,
        k_dev: *const f32,
        v_dev: *const f32,
    ) -> Result<(), AllpacaError> {
        if layer >= self.device_layers.len() || k_dev.is_null() || v_dev.is_null() {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                "draft device KV pointer is null".into(),
            )));
        }
        if pos >= self.capacity || self.kv_dim == 0 {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                "draft device KV capacity exceeded".into(),
            )));
        }
        let kv_bytes = self.kv_dim * 4;
        let off = pos * kv_bytes;
        let dest_k = unsafe { self.device_layers[layer].k.add(off) };
        let dest_v = unsafe { self.device_layers[layer].v.add(off) };
        allpaca_hip::memcpy(dest_k, k_dev.cast(), kv_bytes, allpaca_hip::MEMCPY_D2D)?;
        allpaca_hip::memcpy(dest_v, v_dev.cast(), kv_bytes, allpaca_hip::MEMCPY_D2D)?;
        self.tokens = self.tokens.max(pos + 1);
        Ok(())
    }

    fn k_ptr(&self, layer: usize) -> *const f32 {
        self.device_layers
            .get(layer)
            .map(|l| l.k.cast::<f32>() as *const f32)
            .unwrap_or(std::ptr::null())
    }

    fn v_ptr(&self, layer: usize) -> *const f32 {
        self.device_layers
            .get(layer)
            .map(|l| l.v.cast::<f32>() as *const f32)
            .unwrap_or(std::ptr::null())
    }

    pub fn rewind_to(&mut self, tokens: usize, kv_dim: usize) {
        for layer in &mut self.layers {
            let keep = tokens * kv_dim;
            layer.k.truncate(keep);
            layer.v.truncate(keep);
        }
        self.tokens = tokens;
    }
}

fn record_elementwise(metrics: Option<&Metrics>) {
    if let Some(metrics) = metrics {
        metrics
            .elementwise_gpu_ops
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

fn rmsnorm_dispatch(
    hip: Option<&HipWorkspace>,
    x: &[f32],
    weight: &[f32],
    weight_dev: Option<*const f32>,
    eps: f32,
    out: &mut [f32],
    metrics: Option<&Metrics>,
) -> Result<(), AllpacaError> {
    if let Some(hip) = hip {
        if hip.rmsnorm(x, weight, weight_dev, eps, out).is_ok() {
            record_elementwise(metrics);
            return Ok(());
        }
    }
    rmsnorm(x, weight, eps, out)
}

fn rope_dispatch(
    hip: Option<&HipWorkspace>,
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
    metrics: Option<&Metrics>,
) -> Result<(), AllpacaError> {
    if let Some(hip) = hip {
        if hip
            .rope(x, n_heads as i32, head_dim as i32, pos as i32, theta)
            .is_ok()
        {
            record_elementwise(metrics);
            return Ok(());
        }
    }
    for h in 0..n_heads {
        rope(&mut x[h * head_dim..(h + 1) * head_dim], pos, theta)?;
    }
    Ok(())
}

fn silu_mul_dispatch(
    hip: Option<&HipWorkspace>,
    gate: &mut [f32],
    up: &[f32],
    metrics: Option<&Metrics>,
) -> Result<(), AllpacaError> {
    if let Some(hip) = hip {
        if hip.silu_mul(gate, up).is_ok() {
            record_elementwise(metrics);
            return Ok(());
        }
    }
    for i in 0..gate.len() {
        gate[i] = silu(gate[i]) * up[i];
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gemm_nt_dispatch(
    hip: Option<&HipWorkspace>,
    a: &[f32],
    b: &[f32],
    b_dev: Option<*const f32>,
    c: &mut [f32],
    rows: usize,
    n: usize,
    k: usize,
) -> Result<(), AllpacaError> {
    if let (Some(hip), Some(b_dev)) = (hip, b_dev) {
        if hip
            .gemm_nt_dev_b(a, b_dev, c, rows as i32, n as i32, k as i32)
            .is_ok()
        {
            return Ok(());
        }
    }
    if let Some(hip) = hip {
        if hip
            .gemm_nt(a, b, c, rows as i32, n as i32, k as i32)
            .is_ok()
        {
            return Ok(());
        }
    }
    matmul_nt(a, b, c, rows, n, k)
}

#[allow(clippy::too_many_arguments)]
fn attention_dispatch(
    hip: Option<&HipWorkspace>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_q: usize,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Result<(), AllpacaError> {
    if let Some(hip) = hip {
        if hip
            .attention(
                q,
                k,
                v,
                out,
                n_q as i32,
                n_kv as i32,
                n_heads as i32,
                n_kv_heads as i32,
                head_dim as i32,
                causal,
            )
            .is_ok()
        {
            return Ok(());
        }
    }
    attention(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal)
}

fn gather_kv(
    kv: &KvTierManager,
    pages: &[u64],
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), AllpacaError> {
    let mut ks = Vec::new();
    let mut vs = Vec::new();
    for id in pages {
        if kv.get(*id).map(|p| p.filled).unwrap_or(0) == 0 {
            continue;
        }
        let (k, v) = kv.read_kv(*id, n_kv_heads, head_dim)?;
        ks.extend_from_slice(&k);
        vs.extend_from_slice(&v);
    }
    Ok((ks, vs))
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed | 1,
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = ((self.state >> 33) as u32) & 0x007F_FFFF;
        (bits as f32 / 8_388_608.0) * 0.2 - 0.1
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Allocate one page per layer covering `n_tokens` (rounded up to page_tokens).
pub fn alloc_pages_for_seq(
    kv: &mut KvTierManager,
    cfg: &ModelConfig,
    page_tokens: u32,
    n_tokens: usize,
    metrics: &Metrics,
) -> Result<Vec<Vec<u64>>, AllpacaError> {
    let n_pages = n_tokens.div_ceil(page_tokens as usize).max(1);
    let mut pages = Vec::with_capacity(cfg.n_layers);
    for layer in 0..cfg.n_layers {
        let mut layer_pages = Vec::with_capacity(n_pages);
        for p in 0..n_pages {
            layer_pages.push(kv.alloc_page(
                layer as u32,
                p as u32,
                page_tokens,
                cfg.n_kv_heads,
                cfg.head_dim(),
                metrics,
            )?);
        }
        pages.push(layer_pages);
    }
    Ok(pages)
}

pub fn ensure_pages(
    kv: &mut KvTierManager,
    cfg: &ModelConfig,
    page_tokens: u32,
    pages: &mut [Vec<u64>],
    pos: usize,
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    let need = pos / page_tokens as usize;
    for (layer, layer_pages) in pages.iter_mut().enumerate() {
        while layer_pages.len() <= need {
            let idx = layer_pages.len() as u32;
            layer_pages.push(kv.alloc_page(
                layer as u32,
                idx,
                page_tokens,
                cfg.n_kv_heads,
                cfg.head_dim(),
                metrics,
            )?);
        }
    }
    Ok(())
}
