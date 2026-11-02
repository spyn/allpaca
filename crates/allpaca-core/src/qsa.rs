//! QSA, the sparse attention of `qwen4exp` full-attention blocks.
//!
//! A block's earlier tokens are grouped, in sequence order, into pools of
//! `ratio` cells. A small indexer gives every token a raw key; a complete
//! pool gets one pooled key: the mean of its members' raw keys, normed and
//! rotated to its first member's position. A token scores the pools that
//! are complete at its position with its indexer queries and attends to
//! the cells of the best `top_k / ratio` of them plus the incomplete tail
//! (the newest cells that do not fill a pool yet). While no more pools
//! than that exist every cell is kept and the attention is dense.

use allpaca_cpu::rope_imrope_neox;
use allpaca_types::{AllpacaError, FatalError};

use crate::deltanet::rmsnorm_heads;

fn qsa_err(msg: String) -> AllpacaError {
    AllpacaError::Fatal(FatalError::InvalidConfig(msg))
}

/// The indexer's shape and the rotation its keys and queries share with
/// the attention.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct QsaShape {
    /// Cells per pool.
    pub ratio: usize,
    /// Cells kept through the pools (a multiple of `ratio`).
    pub top_k: usize,
    /// Indexer query heads and the width of a key.
    pub heads: usize,
    pub dim: usize,
    pub n_rot: usize,
    pub rope_theta: f32,
    pub rope_sections: [i32; 4],
    pub eps: f32,
}

impl QsaShape {
    /// Pools a token may keep.
    pub fn top_pools(&self) -> usize {
        self.top_k / self.ratio
    }

    /// Whether the token at `rank` (its index in the sequence) sees more
    /// complete pools than it may keep.
    pub fn is_sparse(&self, rank: usize) -> bool {
        (rank + 1) / self.ratio > self.top_pools()
    }

    /// Most cells a token attends to once the attention is sparse.
    pub fn max_cells(&self) -> usize {
        self.top_pools() * self.ratio + self.ratio - 1
    }
}

/// The indexer keys of one block of one sequence.
#[derive(Debug, Clone, Default)]
pub struct QsaKeys {
    /// Raw key per token.
    raw: Vec<f32>,
    /// Pooled key per complete pool.
    pooled: Vec<f32>,
}

impl QsaKeys {
    /// Floats held (raw and pooled).
    pub(crate) fn floats(&self) -> usize {
        self.raw.len() + self.pooled.len()
    }

    /// Append to a prompt-cache checkpoint file.
    pub(crate) fn write_to(&self, out: &mut Vec<u8>) {
        crate::kv_codec::write_f32s(out, &self.raw);
        crate::kv_codec::write_f32s(out, &self.pooled);
    }

    /// What [`Self::write_to`] wrote.
    pub(crate) fn read_from(r: &mut crate::kv_codec::Reader<'_>) -> Option<Self> {
        Some(Self {
            raw: r.f32s()?,
            pooled: r.f32s()?,
        })
    }

    /// Tokens whose raw key is held.
    #[cfg(test)]
    pub fn tokens(&self, shape: &QsaShape) -> usize {
        self.raw.len() / shape.dim
    }

    /// Add the raw keys of the next tokens (`shape.dim` floats each) and
    /// pool what they complete. `k_norm` is the pooled keys' norm gain.
    pub fn push(&mut self, shape: &QsaShape, raw: &[f32], k_norm: &[f32]) -> Result<(), AllpacaError> {
        let (dim, ratio) = (shape.dim, shape.ratio);
        if dim == 0 || ratio < 2 || raw.len() % dim != 0 || k_norm.len() != dim {
            return Err(qsa_err(format!(
                "qsa: {} key floats for width {dim}, ratio {ratio}, norm of {}",
                raw.len(),
                k_norm.len()
            )));
        }
        self.raw.extend_from_slice(raw);
        while (self.pooled.len() / dim + 1) * ratio * dim <= self.raw.len() {
            let pool = self.pooled.len() / dim;
            let members = &self.raw[pool * ratio * dim..(pool + 1) * ratio * dim];
            let mut key = vec![0.0f32; dim];
            for member in members.chunks(dim) {
                for (k, m) in key.iter_mut().zip(member) {
                    *k += m;
                }
            }
            let inv = 1.0 / ratio as f32;
            key.iter_mut().for_each(|k| *k *= inv);
            rmsnorm_heads(&mut key, k_norm, dim, shape.eps);
            rope_imrope_neox(&mut key, pool * ratio, shape.rope_theta, shape.n_rot, shape.rope_sections)?;
            self.pooled.extend_from_slice(&key);
        }
        Ok(())
    }

    /// The cells the token at `rank` attends to, ascending, or `None`
    /// while it keeps them all. `q` is the token's indexer query
    /// projection (`heads * dim` floats, normed and rotated here); the
    /// token's own key must have been pushed.
    pub fn select(
        &self,
        shape: &QsaShape,
        rank: usize,
        q: &mut [f32],
        q_norm: &[f32],
    ) -> Result<Option<Vec<u32>>, AllpacaError> {
        if !shape.is_sparse(rank) {
            return Ok(None);
        }
        let (dim, ratio) = (shape.dim, shape.ratio);
        let visible = (rank + 1) / ratio;
        if q.len() != shape.heads * dim || q_norm.len() != dim || self.pooled.len() < visible * dim {
            return Err(qsa_err(format!(
                "qsa: token {rank} needs {visible} pooled keys and {} query floats, has {} and {}",
                shape.heads * dim,
                self.pooled.len() / dim.max(1),
                q.len()
            )));
        }
        rmsnorm_heads(q, q_norm, dim, shape.eps);
        for head in q.chunks_mut(dim) {
            rope_imrope_neox(head, rank, shape.rope_theta, shape.n_rot, shape.rope_sections)?;
        }
        // A pool's score: its key against every query head, negative
        // products dropped, the rest summed.
        let scale = 1.0 / (dim as f32).sqrt();
        let scores: Vec<f32> = self.pooled[..visible * dim]
            .chunks(dim)
            .map(|key| {
                q.chunks(dim)
                    .map(|head| head.iter().zip(key).map(|(a, b)| a * b).sum::<f32>().max(0.0) * scale)
                    .sum()
            })
            .collect();
        // The best pools; of equal scores the earlier pool wins.
        let mut order: Vec<u32> = (0..visible as u32).collect();
        order.sort_by(|a, b| scores[*b as usize].total_cmp(&scores[*a as usize]).then(a.cmp(b)));
        let mut pools = order[..shape.top_pools()].to_vec();
        pools.sort_unstable();
        let mut cells = Vec::with_capacity(shape.max_cells());
        for pool in pools {
            cells.extend((0..ratio as u32).map(|i| pool * ratio as u32 + i));
        }
        cells.extend((visible * ratio..=rank).map(|c| c as u32));
        Ok(Some(cells))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two cells per pool, two pools kept, keys of two floats that are not
    /// rotated (no rotated pairs at position 0; the others use `n_rot` 2).
    fn shape() -> QsaShape {
        QsaShape {
            ratio: 2,
            top_k: 4,
            heads: 1,
            dim: 2,
            n_rot: 2,
            rope_theta: 10000.0,
            rope_sections: [0; 4],
            eps: 0.0,
        }
    }

    #[test]
    fn a_complete_pool_gets_the_normed_mean_of_its_raw_keys_at_its_first_position() {
        let shape = shape();
        let mut keys = QsaKeys::default();
        keys.push(&shape, &[3.0, 0.0], &[1.0, 1.0]).unwrap();
        assert_eq!((keys.tokens(&shape), keys.pooled.len()), (1, 0), "one cell is no pool yet");
        keys.push(&shape, &[5.0, 6.0], &[1.0, 1.0]).unwrap();
        // Mean (4, 3): RMS sqrt(12.5), and position 0 turns nothing.
        let rms = 12.5f32.sqrt();
        assert!((keys.pooled[0] - 4.0 / rms).abs() < 1e-6 && (keys.pooled[1] - 3.0 / rms).abs() < 1e-6);
        // The second pool starts at position 2: its key is turned by 2 radians.
        keys.push(&shape, &[1.0, 0.0, 1.0, 0.0], &[1.0, 1.0]).unwrap();
        let norm = 2.0f32.sqrt();
        let (sin, cos) = 2.0f32.sin_cos();
        assert!((keys.pooled[2] - norm * cos).abs() < 1e-5 && (keys.pooled[3] - norm * sin).abs() < 1e-5);
    }

    #[test]
    fn every_cell_is_kept_while_the_pools_fit_the_budget() {
        let shape = shape();
        // Ranks 0..=4 see at most two complete pools; rank 5 completes the third.
        assert_eq!((0..7).map(|r| shape.is_sparse(r)).collect::<Vec<_>>(), [false, false, false, false, false, true, true]);
        let mut keys = QsaKeys::default();
        keys.push(&shape, &[1.0; 10], &[1.0, 1.0]).unwrap();
        assert_eq!(keys.select(&shape, 4, &mut [1.0, 0.0], &[1.0, 1.0]).unwrap(), None);
    }

    #[test]
    fn past_the_budget_the_best_scored_pools_and_the_tail_are_kept() {
        let shape = shape();
        // Pooled keys set directly: (1, 0), (0, 1), (0.9, 0.2), (1, 0).
        let keys = QsaKeys {
            raw: vec![0.0; 16],
            pooled: vec![1.0, 0.0, 0.0, 1.0, 0.9, 0.2, 1.0, 0.0],
        };
        // `select` turns the query by its position: these are the queries
        // that end up along the second and the first axis.
        let along = |axis: usize, rank: usize| {
            let (sin, cos) = (rank as f32).sin_cos();
            if axis == 1 {
                [sin, cos]
            } else {
                [cos, -sin]
            }
        };
        // Rank 6 sees three complete pools and a tail of one cell; pool 1
        // scores 1, pool 2 0.2 and pool 0 nothing.
        let picked = keys.select(&shape, 6, &mut along(1, 6), &[1.0, 1.0]).unwrap().expect("sparse");
        assert_eq!(picked, [2, 3, 4, 5, 6], "pools 1 and 2, then the tail cell");
        // Rank 7 completes the fourth pool, so there is no tail: pools 0
        // and 3 score 1, pool 2 0.9.
        let picked = keys.select(&shape, 7, &mut along(0, 7), &[1.0, 1.0]).unwrap().expect("sparse");
        assert_eq!(picked, [0, 1, 6, 7]);
        // Of equal scores the earlier pools are kept: a query pointing away
        // from every key scores them all nothing.
        let (a, b) = (along(0, 7), along(1, 7));
        let mut away = [-(a[0] + b[0]), -(a[1] + b[1])];
        let picked = keys.select(&shape, 7, &mut away, &[1.0, 1.0]).unwrap().expect("sparse");
        assert_eq!(picked, [0, 1, 2, 3]);
    }

    #[test]
    fn a_selection_without_the_pooled_keys_it_needs_is_refused() {
        let shape = shape();
        let mut keys = QsaKeys::default();
        keys.push(&shape, &[1.0; 8], &[1.0, 1.0]).unwrap();
        let err = keys.select(&shape, 5, &mut [1.0, 0.0], &[1.0, 1.0]).expect_err("keys of ranks 4 and 5 are missing");
        assert!(err.to_string().contains("pooled keys"), "{err}");
    }
}
