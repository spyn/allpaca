//! Semantic-anchor cache for recurrent state. Not KV, not the weight working set.
//!
//! Full-attention KV still lives in [`crate::kv_tier::KvTierManager`]. Recurrent
//! layers compress the prefix into one state that cannot be sliced from KV, so
//! checkpoints sit at special-token boundaries (thinking / tool / turn) and
//! recycle with their own LRU.

use std::sync::atomic::{AtomicU64, Ordering};

use allpaca_types::{AllpacaError, FatalError};

use crate::kv_tier::KvTierManager;
use crate::model::ModelConfig;

#[derive(Debug, Clone)]
pub struct RecurrentState {
    pub layers: Vec<Vec<f32>>,
    /// Per-layer selective-SSM hidden (`d_model * d_state`); empty when unused.
    pub ssm_h: Vec<Vec<f32>>,
    /// Per-layer Mamba-2 causal-convolution history.
    pub ssm_conv: Vec<Vec<f32>>,
    /// Per-layer Gated DeltaNet state `[n_v * head_v * head_k]`.
    pub gdn_state: Vec<Vec<f32>>,
    /// Per-layer conv1d history `[(k-1) * qkv_dim]`.
    pub gdn_conv: Vec<Vec<f32>>,
    /// DeepSeek-V4 compressed attention keys and the groups being pooled.
    pub dsv4: crate::dsv4_attn::Dsv4Memory,
    /// qwen4exp: layer states, attention keys, token history and n-gram conv history.
    pub q4x: crate::qwen4exp::Qwen4Memory,
    /// Sensitivity probe: save every layer's input, or resume from saved ones.
    pub trace: Option<crate::sensitivity::LayerTrace>,
    /// Per-layer keys and values of the CPU Qwen3.5 attention
    /// ([`crate::qwen35_host`]); empty when unused.
    pub host_kv: Vec<(Vec<f32>, Vec<f32>)>,
}

impl RecurrentState {
    pub fn zeros(n_layers: usize, dim: usize) -> Self {
        Self {
            layers: vec![vec![0.0; dim]; n_layers],
            ssm_h: vec![Vec::new(); n_layers],
            ssm_conv: vec![Vec::new(); n_layers],
            gdn_state: vec![Vec::new(); n_layers],
            gdn_conv: vec![Vec::new(); n_layers],
            dsv4: Default::default(),
            q4x: Default::default(),
            trace: None,
            host_kv: vec![Default::default(); n_layers],
        }
    }

    pub fn ensure_ssm(&mut self, layer: usize, state_floats: usize, conv_floats: usize) {
        if self.ssm_h.len() <= layer {
            self.ssm_h.resize(layer + 1, Vec::new());
        }
        if self.ssm_conv.len() <= layer {
            self.ssm_conv.resize(layer + 1, Vec::new());
        }
        if self.ssm_h[layer].len() != state_floats {
            self.ssm_h[layer] = vec![0.0; state_floats];
        }
        if self.ssm_conv[layer].len() != conv_floats {
            self.ssm_conv[layer] = vec![0.0; conv_floats];
        }
    }

    pub fn ensure_gdn(&mut self, layer: usize, dims: &crate::deltanet::GdnDims) {
        if self.gdn_state.len() <= layer {
            self.gdn_state.resize(layer + 1, Vec::new());
        }
        if self.gdn_conv.len() <= layer {
            self.gdn_conv.resize(layer + 1, Vec::new());
        }
        let need_s = dims.state_floats();
        if self.gdn_state[layer].len() != need_s {
            self.gdn_state[layer] = vec![0.0; need_s];
        }
        let need_c = dims.conv_state_floats();
        if self.gdn_conv[layer].len() != need_c {
            self.gdn_conv[layer] = vec![0.0; need_c];
        }
    }

    /// Mix compressed history into the residual. Deterministic; not a real KDA kernel.
    pub fn mix_layer(&mut self, layer: usize, x: &mut [f32]) {
        let Some(h) = self.layers.get_mut(layer) else {
            return;
        };
        let n = x.len().min(h.len());
        for i in 0..n {
            h[i] = 0.5 * h[i] + 0.5 * x[i];
            x[i] += h[i];
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct KvRow {
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
}

#[derive(Debug, Clone)]
struct AnchorSnapshot {
    prefix: Vec<u32>,
    recurrent: RecurrentState,
    kv: Vec<Vec<KvRow>>,
}

pub struct AnchorCache {
    n_slots: usize,
    token_ids: Vec<u32>,
    slots: Vec<Option<AnchorSnapshot>>,
    used_at: Vec<u64>,
    clock: u64,
    hits: AtomicU64,
    skips: AtomicU64,
}

impl std::fmt::Debug for AnchorCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnchorCache")
            .field("n_slots", &self.n_slots)
            .field("token_ids", &self.token_ids)
            .finish()
    }
}

impl AnchorCache {
    pub fn new(n_slots: usize, token_ids: Vec<u32>) -> Self {
        let n = if n_slots == 0 || token_ids.is_empty() {
            0
        } else {
            n_slots
        };
        Self {
            n_slots: n,
            token_ids,
            slots: vec![None; n],
            used_at: vec![0; n],
            clock: 0,
            hits: AtomicU64::new(0),
            skips: AtomicU64::new(0),
        }
    }

    pub fn enabled(&self) -> bool {
        self.n_slots > 0
    }

    pub fn is_anchor(&self, token: u32) -> bool {
        self.token_ids.contains(&token)
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn skip_tokens(&self) -> u64 {
        self.skips.load(Ordering::Relaxed)
    }

    pub fn lookup(&mut self, tokens: &[u32]) -> Option<(RecurrentState, Vec<Vec<KvRow>>, usize)> {
        if !self.enabled() || tokens.is_empty() {
            return None;
        }
        let mut best: Option<(usize, usize)> = None;
        for (i, slot) in self.slots.iter().enumerate() {
            let Some(snap) = slot else { continue };
            if snap.prefix.is_empty() || snap.prefix.len() > tokens.len() {
                continue;
            }
            if tokens.starts_with(&snap.prefix) {
                let n = snap.prefix.len();
                if best.is_none_or(|(_, bn)| n > bn) {
                    best = Some((i, n));
                }
            }
        }
        let (i, n) = best?;
        self.clock = self.clock.saturating_add(1);
        self.used_at[i] = self.clock;
        let snap = self.slots[i].as_ref()?;
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.skips.fetch_add(n as u64, Ordering::Relaxed);
        Some((snap.recurrent.clone(), snap.kv.clone(), n))
    }

    pub fn capture(
        &mut self,
        prefix: &[u32],
        recurrent: &RecurrentState,
        kv: &KvTierManager,
        pages: &[Vec<u64>],
        cfg: &ModelConfig,
    ) -> Result<(), AllpacaError> {
        if !self.enabled() || prefix.is_empty() {
            return Ok(());
        }
        let last = *prefix.last().unwrap();
        if !self.is_anchor(last) {
            return Ok(());
        }
        if let Some(existing) = self
            .slots
            .iter()
            .position(|s| s.as_ref().is_some_and(|s| s.prefix == prefix))
        {
            self.clock = self.clock.saturating_add(1);
            self.used_at[existing] = self.clock;
            return Ok(());
        }
        let snap = AnchorSnapshot {
            prefix: prefix.to_vec(),
            recurrent: recurrent.clone(),
            kv: snapshot_kv(kv, pages, cfg)?,
        };
        self.clock = self.clock.saturating_add(1);
        let victim = self
            .slots
            .iter()
            .position(|s| s.is_none())
            .unwrap_or_else(|| {
                self.used_at
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, t)| *t)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            });
        self.slots[victim] = Some(snap);
        self.used_at[victim] = self.clock;
        Ok(())
    }
}

pub fn restore_kv(
    kv: &mut KvTierManager,
    pages: &[Vec<u64>],
    rows: &[Vec<KvRow>],
    cfg: &ModelConfig,
) -> Result<(), AllpacaError> {
    let page_tokens = kv
        .get(pages.first().and_then(|l| l.first()).copied().unwrap_or(0))
        .map(|p| p.capacity_tokens)
        .unwrap_or(128);
    for (layer, layer_rows) in rows.iter().enumerate() {
        let layer_pages = pages.get(layer).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "anchor restore missing KV layer".into(),
            ))
        })?;
        for (pos, row) in layer_rows.iter().enumerate() {
            let page_index = pos / page_tokens as usize;
            let slot = (pos % page_tokens as usize) as u32;
            let page_id = *layer_pages.get(page_index).ok_or_else(|| {
                AllpacaError::Fatal(FatalError::OutOfMemory(
                    "anchor restore missing KV page".into(),
                ))
            })?;
            kv.write_slot(
                page_id,
                slot,
                cfg.n_kv_heads,
                cfg.head_dim(),
                &row.k,
                &row.v,
            )?;
        }
    }
    Ok(())
}

pub(crate) fn snapshot_kv(
    kv: &KvTierManager,
    pages: &[Vec<u64>],
    cfg: &ModelConfig,
) -> Result<Vec<Vec<KvRow>>, AllpacaError> {
    let kv_width = cfg.n_kv_heads * cfg.head_dim();
    let mut out = Vec::with_capacity(pages.len());
    for layer_pages in pages {
        let (mut ks, mut vs) = (Vec::new(), Vec::new());
        for &id in layer_pages {
            if kv.get(id).map_or(0, |p| p.filled) == 0 {
                continue;
            }
            let (k, v) = kv.read_kv_for_cache(id, cfg.n_kv_heads, cfg.head_dim())?;
            ks.extend_from_slice(&k);
            vs.extend_from_slice(&v);
        }
        let n = if kv_width == 0 {
            0
        } else {
            ks.len() / kv_width
        };
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let a = i * kv_width;
            let b = a + kv_width;
            rows.push(KvRow {
                k: ks[a..b].to_vec(),
                v: vs[a..b].to_vec(),
            });
        }
        out.push(rows);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_picks_the_longest_surviving_prefix() {
        let mut cache = AnchorCache::new(2, vec![3]);
        cache.slots[0] = Some(AnchorSnapshot {
            prefix: vec![1, 3],
            recurrent: RecurrentState::zeros(1, 2),
            kv: Vec::new(),
        });
        cache.slots[1] = Some(AnchorSnapshot {
            prefix: vec![1],
            recurrent: RecurrentState::zeros(1, 2),
            kv: Vec::new(),
        });
        let (_, _, n) = cache.lookup(&[1, 3, 4]).expect("must hit");
        assert_eq!(n, 2, "deepest surviving anchor is the longer prefix");
    }
}
