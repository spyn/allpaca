//! Latent sparse attention (LSA), flat search: the CPU reference.
//!
//! Source: "Latent Sparse Attention" (arXiv 2610.42147) and its released
//! reference, `src/lsa/core/reference.py` and `ward.py` of
//! `Prefrontal-AI/latent-sparse-attention` (Apache-2.0). This is a port of
//! the flat design for a grouped-query model with a partial rotation
//! (Qwen3.5 / Qwen3.8).
//!
//! Every cached key gets an index key: the position-free dims of its
//! key-value heads, side by side, through a per-layer map, at unit length.
//! Keys are clustered in blocks of `cluster_block` keys once a block is
//! complete: consecutive runs of `pool_size` index keys are summed into
//! pools, and Ward's method merges the pools into buckets. A bucket's
//! centroid is its key sum at unit length. A query sums each group's query
//! heads on the same dims, maps and normalises them, scores the centroids of
//! the blocks closed before its own, keeps the `top_p` best buckets and reads
//! their keys best bucket first until it holds `top_k` of them, beside the
//! first `sink` keys and the last `window` keys.
//!
//! Selection only decides which keys the attention reads. It is approximate
//! by design, so any runtime use sits behind `experimental.lsa`.

use std::path::Path;

use allpaca_types::{AllpacaError, FatalError};

fn lsa_err(msg: String) -> AllpacaError {
    AllpacaError::Fatal(FatalError::InvalidConfig(msg))
}

/// The sizes of the flat design. [`LsaShape::paper`] holds the paper's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LsaShape {
    /// Consecutive keys summed into one pool; a pool is never split.
    pub pool_size: usize,
    /// Average keys per bucket.
    pub bucket_size: usize,
    /// Most keys in a bucket, at least twice `bucket_size`.
    pub bucket_cap: usize,
    /// Keys in a cluster block.
    pub cluster_block: usize,
    /// Buckets a query picks.
    pub top_p: usize,
    /// Long-range keys a query reads at most.
    pub top_k: usize,
    /// First keys, always read.
    pub sink: usize,
    /// Most recent keys, always read (the query's own included).
    pub window: usize,
}

impl LsaShape {
    /// The paper's flat design with a long-range budget of `top_k` keys:
    /// pools of 4, buckets of 8 (at most 16) in blocks of 512, `top_k / 4`
    /// buckets picked, 4 sinks and a window of 512.
    pub fn paper(top_k: usize) -> Self {
        Self {
            pool_size: 4,
            bucket_size: 8,
            bucket_cap: 16,
            cluster_block: 512,
            top_p: top_k / 4,
            top_k,
            sink: 4,
            window: 512,
        }
    }

    /// Buckets a cluster block ends with.
    pub fn block_buckets(&self) -> usize {
        self.cluster_block / self.bucket_size
    }

    /// Most keys a query reads.
    pub fn max_keys(&self) -> usize {
        self.sink + self.top_k + self.window
    }

    /// Whether the sizes nest and leave Ward an allowed pair at every step.
    pub fn check(&self) -> Result<(), AllpacaError> {
        let s = self;
        if s.pool_size == 0 || s.bucket_size == 0 || s.cluster_block == 0 || s.top_p == 0 || s.top_k == 0 || s.window == 0 {
            return Err(lsa_err("lsa: pool, bucket, block, top_p, top_k and window sizes must be positive".into()));
        }
        if s.bucket_size % s.pool_size != 0 || s.cluster_block % s.bucket_size != 0 {
            return Err(lsa_err(format!(
                "lsa: pool size {} must divide bucket size {}, which must divide the cluster block {}",
                s.pool_size, s.bucket_size, s.cluster_block
            )));
        }
        if s.bucket_cap < 2 * s.bucket_size {
            return Err(lsa_err(format!(
                "lsa: bucket cap {} is under twice the bucket size {}",
                s.bucket_cap, s.bucket_size
            )));
        }
        if s.window < s.cluster_block {
            // The open block has no buckets: only the window covers it.
            return Err(lsa_err(format!(
                "lsa: window {} is under the cluster block {}, so keys of the open block would be unreachable",
                s.window, s.cluster_block
            )));
        }
        Ok(())
    }
}

/// Where the position-free part of a head sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LsaGeometry {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Leading dims of a head the rotation covers.
    pub n_rot: usize,
}

impl LsaGeometry {
    /// Dims of the key and query input the maps read.
    pub fn key_dim(&self) -> usize {
        self.n_kv_heads * (self.head_dim - self.n_rot)
    }

    /// Whether a head has position-free dims and the heads group evenly.
    pub fn check(&self) -> Result<(), AllpacaError> {
        if self.n_kv_heads == 0 || self.n_heads % self.n_kv_heads != 0 || self.n_rot >= self.head_dim {
            return Err(lsa_err(format!(
                "lsa: {} query heads over {} key heads of {} dims with {} rotated leave nothing position-free to index",
                self.n_heads, self.n_kv_heads, self.head_dim, self.n_rot
            )));
        }
        Ok(())
    }

    /// The key input of one cached key row (`n_kv_heads * head_dim` floats):
    /// every head's dims past its rotated prefix, side by side.
    pub fn key_input(&self, k: &[f32], out: &mut Vec<f32>) {
        out.clear();
        for head in k.chunks(self.head_dim) {
            out.extend_from_slice(&head[self.n_rot..]);
        }
    }

    /// The query input of one query row (`n_heads * head_dim` floats): each
    /// group's query heads summed on the dims past the rotated prefix, in the
    /// slice of the key head the group reads.
    pub fn query_input(&self, q: &[f32], out: &mut Vec<f32>) {
        let free = self.head_dim - self.n_rot;
        let group = self.n_heads / self.n_kv_heads;
        out.clear();
        out.resize(self.n_kv_heads * free, 0.0);
        for (h, head) in q.chunks(self.head_dim).enumerate() {
            let dst = &mut out[(h / group) * free..(h / group + 1) * free];
            for (d, s) in dst.iter_mut().zip(&head[self.n_rot..]) {
                *d += s;
            }
        }
    }
}

/// One layer's index maps, `[key_dim, index_dim]` row-major each.
#[derive(Debug, Clone)]
pub struct LsaMap {
    pub key_dim: usize,
    pub index_dim: usize,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
}

impl LsaMap {
    /// Read layer `layer`'s maps (`<layer>.W_Q_idx`, `<layer>.W_K_idx`) from
    /// a safetensors file made from a released `.pt` map.
    pub fn load(path: &Path, layer: usize, key_dim: usize) -> Result<Self, AllpacaError> {
        let file = allpaca_safetensors::SafetensorsFile::open(path)?;
        let w_q = file.tensor_f32(&format!("{layer}.W_Q_idx"))?;
        let w_k = file.tensor_f32(&format!("{layer}.W_K_idx"))?;
        if w_q.len() != w_k.len() || w_q.is_empty() || w_q.len() % key_dim != 0 {
            return Err(lsa_err(format!(
                "lsa: layer {layer} maps hold {} and {} floats, not a multiple of the key input's {key_dim}",
                w_q.len(),
                w_k.len()
            )));
        }
        Ok(Self {
            key_dim,
            index_dim: w_q.len() / key_dim,
            w_q,
            w_k,
        })
    }

    fn project(&self, w: &[f32], input: &[f32], out: &mut [f32]) {
        out.fill(0.0);
        for (x, row) in input.iter().zip(w.chunks(self.index_dim)) {
            for (o, m) in out.iter_mut().zip(row) {
                *o += x * m;
            }
        }
        let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-30);
        out.iter_mut().for_each(|v| *v /= norm);
    }

    /// The unit index key of a key input (`key_dim` floats).
    pub fn index_key(&self, key_input: &[f32], out: &mut [f32]) {
        self.project(&self.w_k, key_input, out);
    }

    /// The unit index query of a query input (`key_dim` floats).
    pub fn index_query(&self, query_input: &[f32], out: &mut [f32]) {
        self.project(&self.w_q, query_input, out);
    }
}

/// Ward's merge of one cluster block's pools down to `bucket_count` buckets.
///
/// `pools` holds `pool_count` key sums of `dim` floats, each of `pool_size`
/// keys. A bucket's alignment is the norm of its key sum; merging two costs
/// the alignment the union loses. Each round every bucket finds its cheapest
/// partner among those the cap allows (a tie to the lower index) and every
/// pair that is each other's cheapest merges into its lower member, the
/// cheapest pairs first in the round that reaches the count.
///
/// Returns each pool's bucket, numbered in the order of the buckets' first
/// pools.
pub fn ward_merge(
    pools: &[f32],
    dim: usize,
    pool_size: usize,
    bucket_count: usize,
    bucket_cap: usize,
) -> Result<Vec<u32>, AllpacaError> {
    let n = pools.len() / dim.max(1);
    if dim == 0 || pools.len() != n * dim || bucket_count == 0 || bucket_count > n {
        return Err(lsa_err(format!(
            "lsa: {} pool floats of width {dim} cannot make {bucket_count} buckets",
            pools.len()
        )));
    }
    let mut sums = pools.to_vec();
    let mut sizes = vec![pool_size; n];
    let mut alive = vec![true; n];
    let mut parent: Vec<usize> = (0..n).collect();
    let mut live = n;
    let mut gram = vec![0.0f32; n * n];
    let mut best = vec![usize::MAX; n];
    let mut best_cost = vec![f32::INFINITY; n];
    while live > bucket_count {
        for i in 0..n {
            if !alive[i] {
                continue;
            }
            for j in i..n {
                if !alive[j] {
                    continue;
                }
                let g: f32 = sums[i * dim..(i + 1) * dim]
                    .iter()
                    .zip(&sums[j * dim..(j + 1) * dim])
                    .map(|(a, b)| a * b)
                    .sum();
                gram[i * n + j] = g;
                gram[j * n + i] = g;
            }
        }
        for i in 0..n {
            best[i] = usize::MAX;
            best_cost[i] = f32::INFINITY;
            if !alive[i] {
                continue;
            }
            let sq_i = gram[i * n + i];
            let a_i = sq_i.max(0.0).sqrt();
            for j in 0..n {
                if j == i || !alive[j] || sizes[i] + sizes[j] > bucket_cap {
                    continue;
                }
                let sq_j = gram[j * n + j];
                let merged = (sq_i + sq_j + 2.0 * gram[i * n + j]).max(0.0).sqrt();
                let cost = a_i + sq_j.max(0.0).sqrt() - merged;
                if cost < best_cost[i] {
                    best_cost[i] = cost;
                    best[i] = j;
                }
            }
        }
        // Pairs that are each other's cheapest, as (cost, lower member).
        let mut pairs: Vec<(f32, usize)> = (0..n)
            .filter(|&i| best[i] != usize::MAX && best[i] > i && best[best[i]] == i)
            .map(|i| (best_cost[i], i))
            .collect();
        if pairs.is_empty() {
            return Err(lsa_err(format!(
                "lsa: Ward stopped at {live} buckets, short of {bucket_count}, under a cap of {bucket_cap}"
            )));
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        pairs.truncate(live - bucket_count);
        for (_, kept) in pairs {
            let absorbed = best[kept];
            for d in 0..dim {
                sums[kept * dim + d] += sums[absorbed * dim + d];
            }
            sizes[kept] += sizes[absorbed];
            alive[absorbed] = false;
            parent[absorbed] = kept;
            live -= 1;
        }
    }
    let mut number = vec![0u32; n];
    let mut next = 0u32;
    for i in 0..n {
        if alive[i] {
            number[i] = next;
            next += 1;
        }
    }
    Ok((0..n)
        .map(|mut i| {
            while parent[i] != i {
                i = parent[i];
            }
            number[i]
        })
        .collect())
}

/// The index of one layer of one sequence: the buckets of its closed cluster
/// blocks, and the index keys of the block still open.
#[derive(Debug, Clone)]
pub struct LsaIndex {
    shape: LsaShape,
    dim: usize,
    /// Index keys of the open block (fewer than `cluster_block`).
    open: Vec<f32>,
    /// Keys pushed.
    tokens: usize,
    /// Unit centroid per bucket, block after block.
    centroids: Vec<f32>,
    /// Positions of each bucket's keys, ascending, `bucket_cap` wide,
    /// `u32::MAX` past its size.
    members: Vec<u32>,
}

impl LsaIndex {
    pub fn new(shape: LsaShape, index_dim: usize) -> Result<Self, AllpacaError> {
        shape.check()?;
        if index_dim == 0 {
            return Err(lsa_err("lsa: index keys need a width".into()));
        }
        Ok(Self {
            shape,
            dim: index_dim,
            open: Vec::new(),
            tokens: 0,
            centroids: Vec::new(),
            members: Vec::new(),
        })
    }

    pub fn shape(&self) -> &LsaShape {
        &self.shape
    }

    /// Keys pushed.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// Buckets of the closed blocks.
    pub fn buckets(&self) -> usize {
        self.centroids.len() / self.dim
    }

    /// The unit centroids, `buckets() * index_dim` floats.
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Bytes the index holds per cached key once its block is closed
    /// (centroids as f32 and member positions).
    pub fn bytes_per_key(&self) -> f64 {
        (self.dim * 4 + self.shape.bucket_cap * 4) as f64 / self.shape.bucket_size as f64
    }

    /// Drop the keys from position `tokens` on (a rewind). Blocks that lose
    /// a key reopen, so the caller pushes their keys again from
    /// [`Self::tokens`] on.
    pub fn truncate(&mut self, tokens: usize) {
        if tokens >= self.tokens {
            return;
        }
        let blocks = tokens / self.shape.cluster_block;
        let closed = self.buckets() / self.shape.block_buckets();
        if blocks >= closed {
            self.open.truncate((tokens - closed * self.shape.cluster_block) * self.dim);
            self.tokens = tokens;
        } else {
            self.centroids.truncate(blocks * self.shape.block_buckets() * self.dim);
            self.members.truncate(blocks * self.shape.block_buckets() * self.shape.bucket_cap);
            self.open.clear();
            self.tokens = blocks * self.shape.cluster_block;
        }
    }

    /// Add the unit index keys of the next positions (`index_dim` floats
    /// each) and cluster every block they complete.
    pub fn push(&mut self, index_keys: &[f32]) -> Result<(), AllpacaError> {
        if index_keys.len() % self.dim != 0 {
            return Err(lsa_err(format!(
                "lsa: {} index key floats are not rows of {}",
                index_keys.len(),
                self.dim
            )));
        }
        for key in index_keys.chunks(self.dim) {
            self.open.extend_from_slice(key);
            self.tokens += 1;
            if self.open.len() == self.shape.cluster_block * self.dim {
                self.close_block()?;
            }
        }
        Ok(())
    }

    fn close_block(&mut self) -> Result<(), AllpacaError> {
        let (s, dim) = (self.shape, self.dim);
        let pool_count = s.cluster_block / s.pool_size;
        let mut pools = vec![0.0f32; pool_count * dim];
        for (i, key) in self.open.chunks(dim).enumerate() {
            let pool = &mut pools[(i / s.pool_size) * dim..(i / s.pool_size + 1) * dim];
            for (p, k) in pool.iter_mut().zip(key) {
                *p += k;
            }
        }
        let labels = ward_merge(&pools, dim, s.pool_size, s.block_buckets(), s.bucket_cap)?;
        let first_key = (self.tokens - s.cluster_block) as u32;
        let mut sums = vec![0.0f32; s.block_buckets() * dim];
        let mut members = vec![u32::MAX; s.block_buckets() * s.bucket_cap];
        let mut sizes = vec![0usize; s.block_buckets()];
        for (i, key) in self.open.chunks(dim).enumerate() {
            let bucket = labels[i / s.pool_size] as usize;
            for (c, k) in sums[bucket * dim..(bucket + 1) * dim].iter_mut().zip(key) {
                *c += k;
            }
            members[bucket * s.bucket_cap + sizes[bucket]] = first_key + i as u32;
            sizes[bucket] += 1;
        }
        for sum in sums.chunks_mut(dim) {
            let norm = sum.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-30);
            sum.iter_mut().for_each(|v| *v /= norm);
        }
        self.centroids.extend_from_slice(&sums);
        self.members.extend_from_slice(&members);
        self.open.clear();
        Ok(())
    }

    /// Buckets the query at `position` may score: those of the cluster
    /// blocks closed before its own.
    pub fn visible_buckets(&self, position: usize) -> usize {
        (position / self.shape.cluster_block * self.shape.block_buckets()).min(self.buckets())
    }

    /// Whether the query at `position` reads fewer keys than all of them.
    pub fn is_sparse(&self, position: usize) -> bool {
        position + 1 > self.shape.max_keys()
    }

    /// The keys the query at `position` reads, ascending, or `None` while
    /// it reads them all. `index_query` is its unit index query. Keys up to
    /// `position` must have been pushed.
    pub fn select(&self, position: usize, index_query: &[f32]) -> Result<Option<Vec<u32>>, AllpacaError> {
        self.select_budget(position, index_query, self.shape.top_k, self.shape.top_p)
    }

    /// [`Self::select`] with another budget: `top_k` long-range keys from
    /// `top_p` buckets. The buckets do not depend on the budget, so one
    /// index serves any.
    pub fn select_budget(
        &self,
        position: usize,
        index_query: &[f32],
        top_k: usize,
        top_p: usize,
    ) -> Result<Option<Vec<u32>>, AllpacaError> {
        let shape = LsaShape { top_k, top_p, ..self.shape };
        shape.check()?;
        if position + 1 <= shape.max_keys() {
            return Ok(None);
        }
        let visible = self.visible_buckets(position);
        if index_query.len() != self.dim || position >= self.tokens {
            return Err(lsa_err(format!(
                "lsa: query at {position} of {} dims over {} keys of index width {}",
                index_query.len(),
                self.tokens,
                self.dim
            )));
        }
        let scores: Vec<f32> = self.centroids[..visible * self.dim]
            .chunks(self.dim)
            .map(|c| c.iter().zip(index_query).map(|(a, b)| a * b).sum())
            .collect();
        Ok(Some(self.select_scored(position, &scores, &shape)))
    }

    /// The keys of a sparse query from the scores of its visible buckets
    /// (the dot of each centroid with the index query), for a caller that
    /// scored them elsewhere. `shape` carries the budget.
    pub fn select_scored(&self, position: usize, scores: &[f32], shape: &LsaShape) -> Vec<u32> {
        let s = shape;
        // Best score first, a tie to the lower bucket.
        let mut order: Vec<u32> = (0..scores.len() as u32).collect();
        let by_score = |a: &u32, b: &u32| scores[*b as usize].total_cmp(&scores[*a as usize]).then(a.cmp(b));
        if order.len() > s.top_p {
            order.select_nth_unstable_by(s.top_p - 1, by_score);
            order.truncate(s.top_p);
        }
        order.sort_unstable_by(by_score);
        let window_start = (position + 1).saturating_sub(s.window);
        let sinks = s.sink.min(window_start);
        let mut keys: Vec<u32> = Vec::with_capacity(s.max_keys());
        keys.extend(0..sinks as u32);
        let long_start = keys.len();
        'buckets: for bucket in order {
            let row = &self.members[bucket as usize * s.bucket_cap..(bucket as usize + 1) * s.bucket_cap];
            for &key in row {
                if key == u32::MAX {
                    break;
                }
                if (key as usize) < s.sink || key as usize >= window_start {
                    continue;
                }
                if keys.len() - long_start == s.top_k {
                    break 'buckets;
                }
                keys.push(key);
            }
        }
        keys[long_start..].sort_unstable();
        keys.extend(window_start as u32..=position as u32);
        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    /// Pools of 1, buckets of 2 (at most 4), blocks of 8, 2 buckets and at
    /// most 4 long-range keys a query, 1 sink, a window of 8.
    fn small() -> LsaShape {
        LsaShape {
            pool_size: 1,
            bucket_size: 2,
            bucket_cap: 4,
            cluster_block: 8,
            top_p: 2,
            top_k: 4,
            sink: 1,
            window: 8,
        }
    }

    #[test]
    fn the_papers_sizes_nest_and_bad_ones_are_refused() {
        let paper = LsaShape::paper(4096);
        paper.check().unwrap();
        assert_eq!((paper.top_p, paper.block_buckets(), paper.max_keys()), (1024, 64, 4612));
        let mut bad = paper;
        bad.bucket_cap = 15;
        assert!(bad.check().unwrap_err().to_string().contains("cap"));
        let mut bad = paper;
        bad.window = 256;
        assert!(bad.check().unwrap_err().to_string().contains("unreachable"));
        let mut bad = paper;
        bad.pool_size = 3;
        assert!(bad.check().is_err());
    }

    #[test]
    fn the_inputs_are_the_dims_past_the_rotation_with_each_groups_queries_summed() {
        // 4 query heads over 2 key heads of 4 dims, the first 2 rotated.
        let g = LsaGeometry { n_heads: 4, n_kv_heads: 2, head_dim: 4, n_rot: 2 };
        g.check().unwrap();
        assert_eq!(g.key_dim(), 4);
        let mut out = Vec::new();
        g.key_input(&[9.0, 9.0, 1.0, 2.0, 9.0, 9.0, 3.0, 4.0], &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        let q: Vec<f32> = (0..16).map(|i| i as f32).collect();
        g.query_input(&q, &mut out);
        // Heads 0 and 1 read key head 0: dims (2, 3) + (6, 7); heads 2 and 3: (10, 11) + (14, 15).
        assert_eq!(out, [8.0, 10.0, 24.0, 26.0]);
        assert!(LsaGeometry { n_heads: 4, n_kv_heads: 2, head_dim: 4, n_rot: 4 }.check().is_err());
    }

    #[test]
    fn a_map_projects_and_normalises() {
        // key_dim 3 to index_dim 2: rows (1, 0), (0, 1), (1, 1).
        let map = LsaMap {
            key_dim: 3,
            index_dim: 2,
            w_q: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            w_k: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        };
        let mut out = [0.0f32; 2];
        map.index_key(&[3.0, 4.0, 100.0], &mut out);
        assert!((out[0] - 0.6).abs() < 1e-6 && (out[1] - 0.8).abs() < 1e-6);
        map.index_query(&[1.0, 0.0, 1.0], &mut out);
        // (1, 0) + (1, 1) = (2, 1), at unit length.
        let n = 5.0f32.sqrt();
        assert!((out[0] - 2.0 / n).abs() < 1e-6 && (out[1] - 1.0 / n).abs() < 1e-6);
    }

    #[test]
    fn ward_merges_the_aligned_pools_and_numbers_buckets_by_first_pool() {
        // Four unit pools in two directions, interleaved: pools 0 and 2 along
        // x, 1 and 3 along y. Merging an aligned pair costs 1 + 1 - 2 = 0, a
        // crossed pair 2 - sqrt(2).
        let pools = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        assert_eq!(ward_merge(&pools, 2, 1, 2, 2).unwrap(), [0, 1, 0, 1]);
        // Down to one bucket the cap must allow all four.
        assert_eq!(ward_merge(&pools, 2, 1, 1, 4).unwrap(), [0, 0, 0, 0]);
        assert!(ward_merge(&pools, 2, 1, 1, 2).unwrap_err().to_string().contains("stopped"));
    }

    #[test]
    fn ward_stops_at_the_count_with_the_cheapest_pairs_first() {
        // Six pools: (0, 1) nearly aligned (cheapest), (2, 3) aligned less,
        // (4, 5) least. Five buckets take one merge: the cheapest pair.
        let pools: Vec<f32> = [
            unit(&[1.0, 0.0, 0.0]),
            unit(&[1.0, 0.01, 0.0]),
            unit(&[0.0, 1.0, 0.0]),
            unit(&[0.0, 1.0, 0.2]),
            unit(&[0.0, 0.0, 1.0]),
            unit(&[0.5, 0.0, 1.0]),
        ]
        .concat();
        assert_eq!(ward_merge(&pools, 3, 1, 5, 2).unwrap(), [0, 0, 1, 2, 3, 4]);
        assert_eq!(ward_merge(&pools, 3, 1, 4, 2).unwrap(), [0, 0, 1, 1, 2, 3]);
        assert_eq!(ward_merge(&pools, 3, 1, 3, 2).unwrap(), [0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn the_cap_keeps_a_pool_out_of_a_full_bucket() {
        // Three aligned pools of 2 keys under a cap of 4: two merge, the
        // third cannot join them and stays alone with the crossed pool left
        // for the last merge.
        let pools = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        assert_eq!(ward_merge(&pools, 2, 2, 2, 4).unwrap(), [0, 0, 1, 1]);
    }

    /// 16 keys in two blocks of 8. Key i points along axis `axes[i]`.
    fn index_of(axes: &[usize]) -> LsaIndex {
        let mut index = LsaIndex::new(small(), 4).unwrap();
        for &axis in axes {
            let mut key = [0.0f32; 4];
            key[axis] = 1.0;
            index.push(&key).unwrap();
        }
        index
    }

    #[test]
    fn a_block_is_clustered_when_it_completes_and_buckets_hold_its_aligned_keys() {
        let mut index = index_of(&[0, 1, 2, 3, 0, 1, 2]);
        assert_eq!((index.tokens(), index.buckets()), (7, 0), "an open block has no buckets");
        index.push(&[0.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(index.buckets(), 4);
        // Buckets are numbered by first key: axis 0 holds keys 0 and 4, and so on.
        assert_eq!(&index.members[..4], [0, 4, u32::MAX, u32::MAX]);
        assert_eq!(&index.members[12..16], [3, 7, u32::MAX, u32::MAX]);
        assert_eq!(&index.centroids[..4], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(index.bytes_per_key(), (4.0 * 4.0 + 4.0 * 4.0) / 2.0);
    }

    #[test]
    fn a_query_reads_everything_until_its_budget_is_exceeded() {
        let shape = small();
        assert_eq!(shape.max_keys(), 13);
        let index = index_of(&[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
        assert!(!index.is_sparse(12) && index.is_sparse(13));
        assert_eq!(index.select(12, &[1.0, 0.0, 0.0, 0.0]).unwrap(), None);
    }

    #[test]
    fn a_sparse_query_reads_the_sink_its_best_buckets_and_the_window() {
        // Three blocks. The query at 23 sees the buckets of blocks 0 and 1
        // (keys 0..16); its window is 16..=23 and its sink key 0.
        let axes: Vec<usize> = (0..24).map(|i| i % 4).collect();
        let index = index_of(&axes);
        assert_eq!((index.buckets(), index.visible_buckets(23), index.visible_buckets(15)), (12, 8, 4));
        // Along axis 1: buckets 1 (keys 1, 5) and 5 (keys 9, 13) score 1.
        let keys = index.select(23, &[0.0, 1.0, 0.0, 0.0]).unwrap().expect("sparse");
        assert_eq!(keys, [0, 1, 5, 9, 13, 16, 17, 18, 19, 20, 21, 22, 23]);
        // Along axis 0 the best buckets are 0 (keys 0, 4) and 4 (keys 8, 12):
        // key 0 is the sink already and is not counted twice.
        let keys = index.select(23, &[1.0, 0.0, 0.0, 0.0]).unwrap().expect("sparse");
        assert_eq!(keys, [0, 4, 8, 12, 16, 17, 18, 19, 20, 21, 22, 23]);
        // Of equal scores the lower buckets win: a query along nothing the
        // keys hold scores every bucket 0 and picks buckets 0 and 1.
        let keys = index.select(23, &[0.0, 0.0, 0.0, 0.0]).unwrap().expect("sparse");
        assert_eq!(keys[..4], [0, 1, 4, 5]);
    }

    #[test]
    fn the_long_range_budget_cuts_the_last_bucket_short() {
        let mut shape = small();
        shape.top_k = 3;
        let mut index = LsaIndex::new(shape, 4).unwrap();
        for i in 0..24 {
            let mut key = [0.0f32; 4];
            key[i % 4] = 1.0;
            index.push(&key).unwrap();
        }
        // Buckets 1 then 5, best first: keys 1, 5, then 9, and 13 is cut.
        let keys = index.select(23, &[0.0, 1.0, 0.0, 0.0]).unwrap().expect("sparse");
        assert_eq!(keys[..4], [0, 1, 5, 9]);
        assert_eq!(keys.len(), 1 + 3 + 8);
    }

    #[test]
    fn a_rewind_reopens_the_blocks_that_lose_keys() {
        let axes: Vec<usize> = (0..20).map(|i| i % 4).collect();
        let full = index_of(&axes);
        let mut index = full.clone();
        // Inside the open block: only its keys go.
        index.truncate(18);
        assert_eq!((index.tokens(), index.buckets()), (18, 8));
        // Into a closed block: it reopens empty and its keys are pushed again.
        index.truncate(13);
        assert_eq!((index.tokens(), index.buckets()), (8, 4));
        for &axis in &axes[8..] {
            let mut key = [0.0f32; 4];
            key[axis] = 1.0;
            index.push(&key).unwrap();
        }
        assert_eq!((index.centroids.clone(), index.members.clone(), index.open.clone()), (full.centroids, full.members, full.open));
    }

    #[test]
    fn a_query_past_the_pushed_keys_is_refused() {
        let axes: Vec<usize> = (0..16).map(|i| i % 4).collect();
        let index = index_of(&axes);
        let err = index.select(20, &[1.0, 0.0, 0.0, 0.0]).expect_err("keys 16..=20 are missing");
        assert!(err.to_string().contains("over 16 keys"), "{err}");
    }
}
