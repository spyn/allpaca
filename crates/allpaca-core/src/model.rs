use allpaca_cpu::{attention, attention_with_mask, matmul_nt, rmsnorm, rope, silu};
use crate::forest::TokenForest;
use allpaca_gguf::GgufFile;
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
        let cfg = ModelConfig {
            vocab_size: vocab,
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            ff_dim,
            eps: 1e-5,
            rope_theta: 10_000.0,
        };
        let embed = file.tensor_f32("token_embd.weight")?;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
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
            name: file
                .meta_str("general.name")
                .unwrap_or("gguf")
                .to_string(),
        })
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

    /// Decode one token at `pos`, writing K/V into paged tiers. Logits written to `logits`.
    pub fn forward_token(
        &self,
        token: u32,
        pos: usize,
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
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
        let mut x = self.embed[token as usize * dim..(token as usize + 1) * dim].to_vec();
        let mut scratch = vec![0.0f32; dim.max(cfg.ff_dim).max(kv_dim)];
        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; dim];
        let mut hidden = vec![0.0f32; dim];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            rmsnorm(&x, &layer.attn_norm, cfg.eps, &mut scratch[..dim])?;
            // TODO: HIP gemm at this call site when a HIP stream is selected.
            matmul_nt(&scratch[..dim], &layer.wq, &mut q, 1, dim, dim)?;
            matmul_nt(&scratch[..dim], &layer.wk, &mut k, 1, kv_dim, dim)?;
            matmul_nt(&scratch[..dim], &layer.wv, &mut v, 1, kv_dim, dim)?;
            for h in 0..cfg.n_heads {
                rope(&mut q[h * head_dim..(h + 1) * head_dim], pos, cfg.rope_theta)?;
            }
            for h in 0..cfg.n_kv_heads {
                rope(&mut k[h * head_dim..(h + 1) * head_dim], pos, cfg.rope_theta)?;
            }

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
            // TODO: HIP attention kernel when the target device is HIP.
            attention(
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
            matmul_nt(&attn_out, &layer.wo, &mut hidden, 1, dim, dim)?;
            for i in 0..dim {
                x[i] += hidden[i];
            }

            rmsnorm(&x, &layer.ffn_norm, cfg.eps, &mut scratch[..dim])?;
            let mut gate = vec![0.0f32; cfg.ff_dim];
            let mut up = vec![0.0f32; cfg.ff_dim];
            matmul_nt(&scratch[..dim], &layer.w_gate, &mut gate, 1, cfg.ff_dim, dim)?;
            matmul_nt(&scratch[..dim], &layer.w_up, &mut up, 1, cfg.ff_dim, dim)?;
            for i in 0..cfg.ff_dim {
                gate[i] = silu(gate[i]) * up[i];
            }
            matmul_nt(&gate, &layer.w_down, &mut hidden, 1, dim, cfg.ff_dim)?;
            for i in 0..dim {
                x[i] += hidden[i];
            }
            let _ = metrics;
        }

        rmsnorm(&x, &self.norm, cfg.eps, &mut scratch[..dim])?;
        if logits.len() < cfg.vocab_size {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "logits buffer smaller than vocab".into(),
            )));
        }
        matmul_nt(
            &scratch[..dim],
            &self.lm_head,
            &mut logits[..cfg.vocab_size],
            1,
            cfg.vocab_size,
            dim,
        )?;
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
            rmsnorm(&x, &layer.attn_norm, cfg.eps, &mut normed)?;
            matmul_nt(&normed, &layer.wq, &mut q, 1, dim, dim)?;
            matmul_nt(&normed, &layer.wk, &mut k, 1, kv_dim, dim)?;
            matmul_nt(&normed, &layer.wv, &mut v, 1, kv_dim, dim)?;
            for h in 0..cfg.n_heads {
                rope(&mut q[h * head_dim..(h + 1) * head_dim], pos, cfg.rope_theta)?;
            }
            for h in 0..cfg.n_kv_heads {
                rope(&mut k[h * head_dim..(h + 1) * head_dim], pos, cfg.rope_theta)?;
            }
            cache.layers[layer_idx].k.extend_from_slice(&k);
            cache.layers[layer_idx].v.extend_from_slice(&v);
            let n_kv = cache.layers[layer_idx].k.len() / kv_dim;
            attention(
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
            matmul_nt(&attn_out, &layer.wo, &mut hidden, 1, dim, dim)?;
            for i in 0..dim {
                x[i] += hidden[i];
            }
            rmsnorm(&x, &layer.ffn_norm, cfg.eps, &mut normed)?;
            let mut gate = vec![0.0f32; cfg.ff_dim];
            let mut up = vec![0.0f32; cfg.ff_dim];
            matmul_nt(&normed, &layer.w_gate, &mut gate, 1, cfg.ff_dim, dim)?;
            matmul_nt(&normed, &layer.w_up, &mut up, 1, cfg.ff_dim, dim)?;
            for i in 0..cfg.ff_dim {
                gate[i] = silu(gate[i]) * up[i];
            }
            matmul_nt(&gate, &layer.w_down, &mut hidden, 1, dim, cfg.ff_dim)?;
            for i in 0..dim {
                x[i] += hidden[i];
            }
        }
        rmsnorm(&x, &self.norm, cfg.eps, &mut normed)?;
        matmul_nt(&normed, &self.lm_head, &mut logits[..cfg.vocab_size], 1, cfg.vocab_size, dim)?;
        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
pub struct DenseKv {
    pub layers: Vec<DenseLayerKv>,
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

    pub fn rewind_to(&mut self, tokens: usize, kv_dim: usize) {
        for layer in &mut self.layers {
            let keep = tokens * kv_dim;
            layer.k.truncate(keep);
            layer.v.truncate(keep);
        }
    }
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
