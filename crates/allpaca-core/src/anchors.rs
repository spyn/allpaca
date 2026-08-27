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
}

impl RecurrentState {
    pub fn zeros(n_layers: usize, dim: usize) -> Self {
        Self {
            layers: vec![vec![0.0; dim]; n_layers],
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

#[derive(Debug, Clone)]
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
        let (ks, vs) = crate::model::gather_kv(kv, layer_pages, cfg.n_kv_heads, cfg.head_dim())?;
        let n = if kv_width == 0 { 0 } else { ks.len() / kv_width };
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
