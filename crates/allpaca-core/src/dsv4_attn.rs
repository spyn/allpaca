//! DeepSeek-V4 attention memory: the keys a token attends to.
//!
//! Every layer attends to a sliding window of raw token keys. A layer with a
//! compression ratio (4 or 128) also attends to compressed keys, one per
//! `ratio` tokens, pooled from the layer's input by a learned softmax gate.
//! Ratio-4 layers pool each group together with the group before it, and
//! once more than `top_k` compressed keys are visible an indexer picks the
//! ones to attend to. Compressed layers rotate with their own RoPE
//! frequencies. Follows llama.cpp's `deepseek4-graph.cpp`.

use allpaca_cpu::{matmul_nt, rmsnorm};
use allpaca_gguf::{GgufFile, GgufValue};
use allpaca_types::{AllpacaError, FatalError};

fn invalid(msg: String) -> AllpacaError {
    AllpacaError::Fatal(FatalError::InvalidConfig(msg))
}

/// Rotation angle per position for each pair of the rotated dims.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RopeTable {
    freqs: Vec<f32>,
    base: f32,
    /// YaRN: the scale of the slow pairs and the pair indices the blend
    /// runs between.
    yarn: Option<[f32; 3]>,
}

impl RopeTable {
    pub fn plain(base: f32, n_rot: usize) -> Self {
        let half = n_rot / 2;
        Self {
            freqs: (0..half)
                .map(|i| 1.0 / base.powf(i as f32 / half as f32))
                .collect(),
            base,
            yarn: None,
        }
    }

    /// YaRN: pairs below the `beta_fast` correction dim keep their frequency,
    /// pairs above the `beta_slow` one are scaled by `scale`, the ones between
    /// are blended. Magnitudes are left alone, as DeepSeek-V4 does.
    pub fn yarn(
        base: f32,
        scale: f32,
        n_ctx_orig: u32,
        beta_fast: f32,
        beta_slow: f32,
        n_rot: usize,
    ) -> Self {
        if scale == 1.0 {
            return Self::plain(base, n_rot);
        }
        if n_ctx_orig == 0 {
            // No blend: every pair is scaled.
            let mut all = Self::plain(base, n_rot);
            all.freqs.iter_mut().for_each(|f| *f *= scale);
            all.yarn = Some([scale, -1.0, -1.0]);
            return all;
        }
        let half = n_rot / 2;
        let corr = |beta: f32| {
            n_rot as f32 * (n_ctx_orig as f32 / (beta * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = corr(beta_fast).floor().max(0.0);
        let high = corr(beta_slow).ceil().min(n_rot as f32 - 1.0);
        Self {
            freqs: (0..half)
                .map(|i| {
                    let extrap = 1.0 / base.powf(i as f32 / half as f32);
                    let ramp = 1.0 - ((i as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
                    extrap * (scale * (1.0 - ramp) + ramp)
                })
                .collect(),
            base,
            yarn: Some([scale, low, high]),
        }
    }

    /// The base and YaRN parameters, for a device kernel that computes the
    /// same frequencies.
    pub fn device(&self) -> (f32, Option<[f32; 3]>) {
        (self.base, self.yarn)
    }

    pub fn n_rot(&self) -> usize {
        self.freqs.len() * 2
    }

    /// Rotate the last `n_rot` values of `row` to `pos` (`inverse` undoes it).
    pub fn rotate(&self, row: &mut [f32], pos: usize, inverse: bool) {
        let n = self.n_rot().min(row.len());
        let start = row.len() - n;
        let tail = &mut row[start..];
        for (i, f) in self.freqs.iter().take(n / 2).enumerate() {
            let (mut sin, cos) = (pos as f32 * f).sin_cos();
            if inverse {
                sin = -sin;
            }
            let (a, b) = (tail[2 * i], tail[2 * i + 1]);
            tail[2 * i] = a * cos - b * sin;
            tail[2 * i + 1] = a * sin + b * cos;
        }
    }
}

/// What the model file says about the attention memory.
#[derive(Debug, Clone, Default)]
pub struct Dsv4AttnSpec {
    /// Raw keys a token attends to, itself included; 0 = all of them.
    pub window: usize,
    /// Compression ratio per GGUF block; 0 = raw keys only.
    pub ratios: Vec<u32>,
    pub index_heads: usize,
    pub index_dim: usize,
    pub top_k: usize,
    pub eps: f32,
    /// Leading key dims that are not rotated (and are stored as FP8).
    pub nope: usize,
    plain: RopeTable,
    compressed: RopeTable,
}

impl Dsv4AttnSpec {
    pub fn from_gguf(
        file: &GgufFile,
        arch: &str,
        head_dim: usize,
        n_rot: usize,
        theta: f32,
        eps: f32,
    ) -> Self {
        let u32_of = |key: &str| file.meta_u32(&format!("{arch}.{key}"));
        let f32_of = |key: &str| file.meta_f32(&format!("{arch}.{key}"));
        let ratios = match file.metadata.get(&format!("{arch}.attention.compress_ratios")) {
            Some(GgufValue::Array(items)) => {
                items.iter().map(|v| v.as_u32().unwrap_or(0)).collect()
            }
            _ => Vec::new(),
        };
        let scaling = file
            .meta_str(&format!("{arch}.rope.scaling.type"))
            .unwrap_or("none");
        let scale = match f32_of("rope.scaling.factor") {
            Some(factor) if scaling != "none" && factor > 0.0 => 1.0 / factor,
            _ => 1.0,
        };
        let base = f32_of("attention.compress_rope_freq_base")
            .filter(|b| *b > 0.0)
            .unwrap_or(theta);
        let compressed = if scaling == "yarn" {
            RopeTable::yarn(
                base,
                scale,
                u32_of("rope.scaling.original_context_length")
                    .or_else(|| u32_of("context_length"))
                    .unwrap_or(0),
                f32_of("rope.scaling.yarn_beta_fast").unwrap_or(32.0),
                f32_of("rope.scaling.yarn_beta_slow").unwrap_or(1.0),
                n_rot,
            )
        } else {
            RopeTable::yarn(base, scale, 0, 32.0, 1.0, n_rot)
        };
        Self {
            window: u32_of("attention.sliding_window").unwrap_or(0) as usize,
            ratios,
            index_heads: u32_of("attention.indexer.head_count").unwrap_or(0) as usize,
            index_dim: u32_of("attention.indexer.key_length").unwrap_or(0) as usize,
            top_k: u32_of("attention.indexer.top_k").unwrap_or(0) as usize,
            eps,
            nope: head_dim.saturating_sub(n_rot),
            plain: RopeTable::plain(theta, n_rot),
            compressed,
        }
    }

    pub fn ratio(&self, blk: usize) -> usize {
        self.ratios.get(blk).copied().unwrap_or(0) as usize
    }

    /// Some layer keeps compressed keys, so a sequence needs a [`Dsv4Memory`].
    pub fn compresses(&self) -> bool {
        self.ratios.iter().any(|r| *r != 0)
    }

    pub fn rope(&self, ratio: usize) -> &RopeTable {
        if ratio == 0 {
            &self.plain
        } else {
            &self.compressed
        }
    }

    /// First raw position the token at `pos` attends to.
    pub fn raw_lo(&self, pos: usize) -> usize {
        if self.window == 0 {
            0
        } else {
            (pos + 1).saturating_sub(self.window)
        }
    }

    /// The token at `pos` in a `n_tok`-token forward from `start_pos` cannot
    /// use plain causal attention over one run of raw keys.
    pub fn needs_mask(&self, ratio: usize, start_pos: usize, n_tok: usize) -> bool {
        ratio != 0 || (n_tok > 1 && self.window > 0 && start_pos + n_tok > self.window)
    }
}

fn e4m3(x: f32) -> f32 {
    let value = |i: i32| {
        let (exp, mant) = ((i >> 3) & 0x0f, (i & 0x07) as f32);
        if exp == 0 {
            mant * 2f32.powi(-9)
        } else {
            (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
        }
    };
    let ax = x.abs().min(448.0);
    let (mut best, mut best_diff) = (0, ax);
    for i in 1..127 {
        let diff = (ax - value(i)).abs();
        if diff < best_diff || (diff == best_diff && i & 1 == 0 && best & 1 != 0) {
            best = i;
            best_diff = diff;
        }
    }
    value(best).copysign(x)
}

/// Round the first `nope` values of a key to FP8 (E4M3) with a power-of-two
/// scale per 64 values, as the model's KV cache stores them.
pub fn fp8_quantize(row: &mut [f32], nope: usize) {
    let nope = nope.min(row.len());
    for block in row[..nope].chunks_mut(64) {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0e-4);
        let scale = 2f32.powi((amax / 448.0).log2().ceil() as i32);
        for v in block {
            *v = e4m3((*v / scale).clamp(-448.0, 448.0)) * scale;
        }
    }
}

/// How one compressor turns `ratio` tokens into a key.
pub struct PoolCfg<'a> {
    pub ratio: usize,
    /// Width of the key it produces.
    pub head: usize,
    /// Gate bias per position in the group, `ratio` rows of the projection width.
    pub ape: &'a [f32],
    pub norm: &'a [f32],
    pub eps: f32,
    pub rope: &'a RopeTable,
    /// FP8-round this many leading dims of the key (0 = keep it exact).
    pub fp8_nope: usize,
}

impl PoolCfg<'_> {
    /// Ratio-4 compressors pool a group together with the one before it.
    fn overlap(&self) -> usize {
        if self.ratio == 4 {
            2
        } else {
            1
        }
    }

    /// Width of the key and gate projections of one token.
    pub fn width(&self) -> usize {
        self.overlap() * self.head
    }
}

/// The tokens a compressor has seen but not finished pooling.
#[derive(Debug, Clone, Default)]
struct Pool {
    kv: Vec<f32>,
    score: Vec<f32>,
}

impl Pool {
    /// Take the token at `pos` (its key and gate projections); returns the
    /// compressed key when the token completes a group.
    fn push(
        &mut self,
        cfg: &PoolCfg<'_>,
        pos: usize,
        kv: &[f32],
        gate: &[f32],
    ) -> Result<Option<Vec<f32>>, AllpacaError> {
        let (ratio, head, width) = (cfg.ratio, cfg.head, cfg.width());
        let rows = cfg.overlap() * ratio;
        if kv.len() != width || gate.len() != width || cfg.ape.len() < ratio * width {
            return Err(invalid(format!(
                "DeepSeek-V4 compressor width {width} does not match its projections"
            )));
        }
        if self.kv.is_empty() {
            self.kv = vec![0.0; rows * width];
            self.score = vec![f32::NEG_INFINITY; rows * width];
        }
        let slot = pos % ratio;
        let row = rows - ratio + slot;
        self.kv[row * width..(row + 1) * width].copy_from_slice(kv);
        let ape = &cfg.ape[slot * width..(slot + 1) * width];
        for ((dst, g), a) in self.score[row * width..(row + 1) * width]
            .iter_mut()
            .zip(gate)
            .zip(ape)
        {
            *dst = g + a;
        }
        if slot + 1 != ratio {
            return Ok(None);
        }
        // Rows of the earlier group use the first `head` columns, the
        // current group the last `head`.
        let col_of = |r: usize| if r < rows - ratio { 0 } else { width - head };
        let mut pooled = vec![0.0f32; head];
        for (c, out) in pooled.iter_mut().enumerate() {
            let at = |r: usize| r * width + col_of(r) + c;
            let max = (0..rows).fold(f32::NEG_INFINITY, |m, r| m.max(self.score[at(r)]));
            let (mut sum, mut acc) = (0.0f32, 0.0f32);
            for r in 0..rows {
                let w = (self.score[at(r)] - max).exp();
                sum += w;
                acc += w * self.kv[at(r)];
            }
            *out = acc / sum;
        }
        if rows > ratio {
            self.kv.copy_within(ratio * width.., 0);
            self.score.copy_within(ratio * width.., 0);
        }
        let mut key = vec![0.0f32; head];
        rmsnorm(&pooled, cfg.norm, cfg.eps, &mut key)?;
        cfg.rope.rotate(&mut key, pos + 1 - ratio, false);
        if cfg.fp8_nope > 0 {
            fp8_quantize(&mut key, cfg.fp8_nope);
        }
        Ok(Some(key))
    }
}

#[derive(Debug, Clone, Default)]
struct LayerMemory {
    /// Tokens this layer has taken in.
    tokens: usize,
    attn: Pool,
    index: Pool,
    /// Compressed keys, `head_dim` each.
    keys: Vec<f32>,
    /// The indexer's view of the same groups, `index_dim` each.
    index_keys: Vec<f32>,
}

/// One sequence's compressed attention memory, for every layer.
#[derive(Debug, Clone, Default)]
pub struct Dsv4Memory {
    layers: Vec<LayerMemory>,
}

impl Dsv4Memory {
    fn layer(&mut self, layer: usize) -> &mut LayerMemory {
        if self.layers.len() <= layer {
            self.layers.resize_with(layer + 1, LayerMemory::default);
        }
        &mut self.layers[layer]
    }

    /// Compressed keys held for `layer`.
    pub fn compressed_keys(&self, layer: usize, head_dim: usize) -> usize {
        self.layers
            .get(layer)
            .map_or(0, |l| l.keys.len() / head_dim.max(1))
    }
}

/// The indexer's inputs for the tokens of one forward.
pub struct IndexQuery {
    /// `index_heads * index_dim` per token, rotated to the token's position.
    pub q: Vec<f32>,
    /// One weight per indexer head per token.
    pub head_weights: Vec<f32>,
}

impl Dsv4Memory {
    /// Take the `n_tok` tokens from `start_pos` into `layer`'s compressors:
    /// `pool` and `index_pool` are each a compressor with its key and gate
    /// projections of those tokens.
    pub fn absorb(
        &mut self,
        layer: usize,
        start_pos: usize,
        n_tok: usize,
        pool: (&PoolCfg<'_>, &[f32], &[f32]),
        index_pool: Option<(&PoolCfg<'_>, &[f32], &[f32])>,
    ) -> Result<(), AllpacaError> {
        let mem = self.layer(layer);
        if mem.tokens != start_pos {
            return Err(invalid(format!(
                "DeepSeek-V4 attention memory of layer {layer} holds {} tokens, forward starts at {start_pos}",
                mem.tokens
            )));
        }
        let (cfg, kv, gate) = pool;
        let width = cfg.width();
        for t in 0..n_tok {
            let at = t * width..(t + 1) * width;
            if let Some(key) = mem.attn.push(cfg, start_pos + t, &kv[at.clone()], &gate[at])? {
                mem.keys.extend_from_slice(&key);
            }
        }
        if let Some((cfg, kv, gate)) = index_pool {
            let width = cfg.width();
            for t in 0..n_tok {
                let at = t * width..(t + 1) * width;
                if let Some(key) = mem.index.push(cfg, start_pos + t, &kv[at.clone()], &gate[at])? {
                    mem.index_keys.extend_from_slice(&key);
                }
            }
        }
        mem.tokens = start_pos + n_tok;
        Ok(())
    }

    /// The keys the token at `pos` attends to, into `keys`: its window of
    /// `raw` keys (which start at position `raw_start`), then the layer's
    /// compressed keys of the groups complete at `pos`; with `index_query`
    /// (the token's indexer query and head weights), only the `top_k` of
    /// those the indexer scores highest.
    #[allow(clippy::too_many_arguments)]
    pub fn keys(
        &self,
        spec: &Dsv4AttnSpec,
        layer: usize,
        ratio: usize,
        pos: usize,
        head_dim: usize,
        raw: &[f32],
        raw_start: usize,
        index_query: Option<(&[f32], &[f32])>,
        keys: &mut Vec<f32>,
    ) -> Result<(), AllpacaError> {
        keys.clear();
        let lo = spec.raw_lo(pos);
        if lo < raw_start || (pos + 1 - raw_start) * head_dim > raw.len() {
            return Err(invalid(format!(
                "DeepSeek-V4 raw keys cover {raw_start}..{}, token {pos} needs {lo}..={pos}",
                raw_start + raw.len() / head_dim.max(1)
            )));
        }
        keys.extend_from_slice(&raw[(lo - raw_start) * head_dim..(pos + 1 - raw_start) * head_dim]);
        if ratio == 0 {
            return Ok(());
        }
        let visible = (pos + 1) / ratio;
        let mem = self.layers.get(layer).filter(|m| visible * head_dim <= m.keys.len());
        let Some(mem) = mem else {
            return Err(invalid(format!(
                "DeepSeek-V4 layer {layer} holds {} compressed keys, token {pos} needs {visible}",
                self.compressed_keys(layer, head_dim)
            )));
        };
        match index_query {
            Some((q, head_weights)) if spec.top_k > 0 && visible > spec.top_k => {
                let dim = spec.index_dim;
                let picked =
                    index_select(q, head_weights, &mem.index_keys[..visible * dim], dim, spec.top_k)?;
                for c in picked {
                    keys.extend_from_slice(&mem.keys[c * head_dim..(c + 1) * head_dim]);
                }
            }
            _ => keys.extend_from_slice(&mem.keys[..visible * head_dim]),
        }
        Ok(())
    }
}

/// The compressed keys the indexer keeps for one token: the `top_k` with the
/// highest score, in key order.
pub fn index_select(
    q: &[f32],
    head_weights: &[f32],
    index_keys: &[f32],
    index_dim: usize,
    top_k: usize,
) -> Result<Vec<usize>, AllpacaError> {
    let n_heads = head_weights.len();
    let n_keys = index_keys.len() / index_dim.max(1);
    let mut dots = vec![0.0f32; n_heads * n_keys];
    matmul_nt(q, index_keys, &mut dots, n_heads, n_keys, index_dim)?;
    let scale = 1.0 / ((index_dim * n_heads) as f32).sqrt();
    let mut scored: Vec<(f32, usize)> = (0..n_keys)
        .map(|c| {
            let s: f32 = (0..n_heads)
                .map(|h| dots[h * n_keys + c].max(0.0) * head_weights[h] * scale)
                .sum();
            (s, c)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut picked: Vec<usize> = scored.into_iter().take(top_k).map(|(_, c)| c).collect();
    picked.sort_unstable();
    Ok(picked)
}

/// One token's attention over `keys` (each key is also its value, shared by
/// every head), with a per-head sink logit that takes weight but adds nothing.
pub fn attend(
    q: &[f32],
    keys: &[f32],
    sinks: &[f32],
    n_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) -> Result<(), AllpacaError> {
    let n_keys = keys.len() / head_dim.max(1);
    let mut scores = vec![0.0f32; n_heads * n_keys];
    matmul_nt(q, keys, &mut scores, n_heads, n_keys, head_dim)?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    out.fill(0.0);
    for h in 0..n_heads {
        let row = &mut scores[h * n_keys..(h + 1) * n_keys];
        let sink = sinks.get(h).copied();
        let mut max = sink.unwrap_or(f32::NEG_INFINITY);
        for s in row.iter_mut() {
            *s *= scale;
            max = max.max(*s);
        }
        let mut sum = sink.map_or(0.0, |s| (s - max).exp());
        for s in row.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        let inv = 1.0 / sum.max(1e-9);
        let oh = &mut out[h * head_dim..(h + 1) * head_dim];
        for (t, w) in row.iter().enumerate() {
            let w = w * inv;
            for (o, k) in oh.iter_mut().zip(&keys[t * head_dim..(t + 1) * head_dim]) {
                *o += w * k;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_rounds_each_block_to_e4m3_at_its_own_scale() {
        // Largest value 448 or below: scale 1, values snap to the E4M3 grid.
        let mut row = vec![0.0f32; 64];
        row[..6].copy_from_slice(&[1.0, 3.3, -3.3, 448.0, 0.001, 300.0]);
        fp8_quantize(&mut row, 64);
        // 3.3 lies between 3.25 and 3.5; 300 between 288 and 320; 0.001 on the
        // subnormal grid of 1/512.
        assert_eq!(&row[..6], &[1.0, 3.25, -3.25, 448.0, 1.0 / 512.0, 288.0]);
        // A small block uses a smaller power-of-two scale, so keeps its detail.
        let mut small = vec![0.0f32; 64];
        small[0] = 0.001;
        small[1] = 0.00033;
        fp8_quantize(&mut small, 64);
        assert!((small[0] - 0.001).abs() < 0.001 / 16.0, "{}", small[0]);
        assert!((small[1] - 0.00033).abs() < 0.00033 / 16.0, "{}", small[1]);
        // Dims past `nope` (the rotated tail) are left exact.
        let mut tail = vec![3.3f32; 72];
        fp8_quantize(&mut tail, 64);
        assert_eq!(tail[63], 3.25);
        assert_eq!(tail[64], 3.3);
    }

    #[test]
    fn yarn_table_keeps_fast_pairs_and_scales_slow_ones() {
        // DeepSeek-V4-Flash: 64 rotated dims, base 160000, factor 16, 64k
        // original context: correction dims 15 and 25.
        let t = RopeTable::yarn(160000.0, 1.0 / 16.0, 65536, 32.0, 1.0, 64);
        let plain = RopeTable::plain(160000.0, 64);
        for i in 0..32 {
            let want = if i <= 15 {
                plain.freqs[i]
            } else if i >= 25 {
                plain.freqs[i] / 16.0
            } else {
                let ramp = 1.0 - (i as f32 - 15.0) / 10.0;
                plain.freqs[i] * ((1.0 - ramp) / 16.0 + ramp)
            };
            assert!((t.freqs[i] - want).abs() <= want * 1e-5, "pair {i}");
        }
        // A rotation and its inverse cancel; dims before the tail are untouched.
        let mut row: Vec<f32> = (0..72).map(|i| i as f32 * 0.1 - 3.0).collect();
        let orig = row.clone();
        t.rotate(&mut row, 1234, false);
        assert_eq!(&row[..8], &orig[..8]);
        assert!((row[8] - orig[8]).abs() > 1e-3);
        t.rotate(&mut row, 1234, true);
        for (a, b) in row.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    /// The reference's prompt-time formula: softmax-pool whole groups at once.
    fn pool_groups(
        kv: &[f32],
        score: &[f32],
        ape: &[f32],
        ratio: usize,
        head: usize,
        n_tok: usize,
    ) -> Vec<Vec<f32>> {
        let coff = if ratio == 4 { 2 } else { 1 };
        let width = coff * head;
        (0..n_tok / ratio)
            .map(|g| {
                (0..head)
                    .map(|c| {
                        let mut items = Vec::new();
                        if coff == 2 && g > 0 {
                            for r in 0..ratio {
                                let t = (g - 1) * ratio + r;
                                items.push((score[t * width + c] + ape[r * width + c], kv[t * width + c]));
                            }
                        }
                        for r in 0..ratio {
                            let t = g * ratio + r;
                            let col = width - head + c;
                            items.push((score[t * width + col] + ape[r * width + col], kv[t * width + col]));
                        }
                        let z: f32 = items.iter().map(|(s, _)| s.exp()).sum();
                        items.iter().map(|(s, v)| s.exp() / z * v).sum()
                    })
                    .collect()
            })
            .collect()
    }

    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn compressor_fed_token_by_token_pools_like_whole_groups() {
        for (ratio, head, n_tok) in [(4usize, 6usize, 14usize), (128, 4, 260)] {
            let width = if ratio == 4 { 2 * head } else { head };
            let kv = noise(n_tok * width, 1);
            let score = noise(n_tok * width, 2);
            let ape = noise(ratio * width, 3);
            let norm = vec![1.0f32; head];
            let rope = RopeTable::plain(10000.0, 2);
            let cfg = PoolCfg {
                ratio,
                head,
                ape: &ape,
                norm: &norm,
                eps: 1e-6,
                rope: &rope,
                fp8_nope: 0,
            };
            let mut pool = Pool::default();
            let mut got = Vec::new();
            for t in 0..n_tok {
                let at = t * width..(t + 1) * width;
                let key = pool.push(&cfg, t, &kv[at.clone()], &score[at]).unwrap();
                // A key appears exactly when a group of `ratio` tokens completes.
                assert_eq!(key.is_some(), (t + 1) % ratio == 0, "token {t}");
                got.extend(key);
            }
            let want = pool_groups(&kv, &score, &ape, ratio, head, n_tok);
            assert_eq!(got.len(), want.len());
            for (g, (key, pooled)) in got.iter().zip(&want).enumerate() {
                let mut expect = vec![0.0f32; head];
                rmsnorm(pooled, &norm, 1e-6, &mut expect).unwrap();
                // Rotated to the position of the group's first token.
                rope.rotate(&mut expect, g * ratio, false);
                for (a, b) in key.iter().zip(&expect) {
                    assert!((a - b).abs() < 1e-5, "ratio {ratio} group {g}: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn indexer_keeps_the_top_scoring_compressed_keys() {
        // Two heads of two dims; keys 0..4 along +x, +y, -x, a weak +x.
        let keys = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.2, 0.0];
        let q = [1.0, 0.0, 0.0, 1.0]; // head 0 looks along x, head 1 along y
        // Head 0 counts three times as much: +x first, then +y, then the weak +x.
        assert_eq!(index_select(&q, &[3.0, 1.0], &keys, 2, 2).unwrap(), vec![0, 1]);
        assert_eq!(index_select(&q, &[3.0, 1.0], &keys, 2, 3).unwrap(), vec![0, 1, 3]);
        // A negative match scores zero (ReLU), it does not count against a key.
        assert_eq!(index_select(&q, &[1.0, 0.0], &keys, 2, 1).unwrap(), vec![0]);
    }

    #[test]
    fn attention_sink_takes_weight_without_adding_a_value() {
        // One head, two keys with equal scores: without a sink the output is
        // their mean; a sink with the same logit takes a third of the weight.
        let keys = [2.0, 0.0, 0.0, 2.0];
        let q = [1.0, 1.0];
        let mut out = [0.0f32; 2];
        attend(&q, &keys, &[], 1, 2, &mut out).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6 && (out[1] - 1.0).abs() < 1e-6);
        let sink = 2.0 / 2f32.sqrt();
        attend(&q, &keys, &[sink], 1, 2, &mut out).unwrap();
        assert!((out[0] - 2.0 / 3.0).abs() < 1e-6, "{}", out[0]);
    }

    use crate::deepseek4::{forward_dsv4_token, forward_dsv4_tokens, write_tiny_deepseek4_ctx_gguf, TinyDsv4Ctx};
    use crate::model::TinyModel;
    use crate::telemetry::Metrics;
    use crate::KvTierManager;

    fn tiny(window: u32, compress_ratios: &[u32], top_k: u32) -> TinyModel {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "allpaca_tiny_dsv4_ctx-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        write_tiny_deepseek4_ctx_gguf(
            &path,
            &TinyDsv4Ctx {
                window,
                compress_ratios: compress_ratios.to_vec(),
                top_k,
                swiglu_clamp: 0.0,
            },
        );
        let file = std::sync::Arc::new(GgufFile::open(&path).expect("open tiny dsv4"));
        let mut model = TinyModel::from_gguf(&file).expect("model");
        model.bind_gguf(file);
        let _ = std::fs::remove_file(&path);
        model
    }

    /// A sequence being forwarded: its KV pages and attention memory.
    struct Run<'m> {
        model: &'m TinyModel,
        kv: KvTierManager,
        pages: Vec<Vec<u64>>,
        memory: Dsv4Memory,
        pos: usize,
        logits: Vec<f32>,
        metrics: Metrics,
    }

    impl<'m> Run<'m> {
        fn new(model: &'m TinyModel) -> Self {
            let cfg = &model.config;
            let metrics = Metrics::default();
            let mut kv = KvTierManager::new(64 << 20, 64 << 20, 0.5);
            let pages = (0..cfg.n_layers)
                .map(|layer| {
                    (0..8)
                        .map(|i| {
                            kv.alloc_page(layer as u32, i, 32, 1, cfg.head_dim(), &metrics)
                                .expect("kv page")
                        })
                        .collect()
                })
                .collect();
            Self {
                model,
                kv,
                pages,
                memory: Dsv4Memory::default(),
                pos: 0,
                logits: vec![0.0; cfg.vocab_size],
                metrics,
            }
        }

        /// Forward the next tokens as one chunk (a single token decodes).
        fn feed(&mut self, tokens: &[u32]) -> Result<(), AllpacaError> {
            if let [token] = tokens {
                forward_dsv4_token(
                    self.model,
                    *token,
                    self.pos,
                    &self.pages,
                    &mut self.kv,
                    &mut self.logits,
                    &self.metrics,
                    None,
                    None,
                    true,
                    Some(&mut self.memory),
                )?;
            } else {
                forward_dsv4_tokens(
                    self.model,
                    tokens,
                    self.pos,
                    &self.pages,
                    &mut self.kv,
                    &mut self.logits,
                    &self.metrics,
                    None,
                    true,
                    true,
                    Some(&mut self.memory),
                )?;
            }
            self.pos += tokens.len();
            Ok(())
        }
    }

    /// Logits after the last token, forwarding `tokens` in chunks of the given sizes.
    fn last_logits(model: &TinyModel, tokens: &[u32], chunks: &[usize]) -> Vec<f32> {
        let mut run = Run::new(model);
        let mut rest = tokens;
        for &n in chunks {
            let (chunk, tail) = rest.split_at(n);
            run.feed(chunk).expect("forward");
            rest = tail;
        }
        assert!(rest.is_empty(), "chunks must cover the tokens");
        run.logits
    }

    fn one_by_one(model: &TinyModel, tokens: &[u32]) -> Vec<f32> {
        last_logits(model, tokens, &vec![1; tokens.len()])
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    #[test]
    fn raw_keys_outside_the_sliding_window_are_not_attended() {
        // One layer, raw keys only: the last token sees itself and the token
        // before it, so what came earlier cannot change its logits.
        let windowed = tiny(2, &[0], 0);
        let a = one_by_one(&windowed, &[3, 4, 5, 6]);
        let b = one_by_one(&windowed, &[7, 3, 5, 6]);
        assert!(max_diff(&a, &b) < 1e-5, "window leaked: {}", max_diff(&a, &b));
        // Without a window the same change is visible.
        let dense = tiny(0, &[0], 0);
        let a = one_by_one(&dense, &[3, 4, 5, 6]);
        let b = one_by_one(&dense, &[7, 3, 5, 6]);
        assert!(max_diff(&a, &b) > 1e-4, "tokens indistinguishable: {}", max_diff(&a, &b));
    }

    #[test]
    fn compressed_keys_carry_tokens_from_beyond_the_window() {
        let model = tiny(2, &[4], 0);
        // Three tokens: no group of four is complete, the first token is
        // outside the window and gone.
        let a = one_by_one(&model, &[3, 4, 5]);
        let b = one_by_one(&model, &[7, 4, 5]);
        assert!(max_diff(&a, &b) < 1e-5, "{}", max_diff(&a, &b));
        // Eight tokens: the first token reaches the last through the
        // compressed key of its group.
        let a = one_by_one(&model, &[3, 4, 5, 6, 7, 3, 4, 5]);
        let b = one_by_one(&model, &[7, 4, 5, 6, 7, 3, 4, 5]);
        assert!(max_diff(&a, &b) > 1e-4, "{}", max_diff(&a, &b));
    }

    #[test]
    fn a_prompt_forwarded_in_chunks_matches_token_by_token() {
        // Raw-only, ratio-4 (with the indexer choosing 2 keys) and ratio-128
        // layers; long enough for a ratio-128 group and several windows.
        let model = tiny(3, &[0, 4, 128, 4], 2);
        let tokens: Vec<u32> = (0..140u32).map(|i| 3 + (i * 7 + i / 5) % 5).collect();
        let want = one_by_one(&model, &tokens);
        assert!(want.iter().all(|v| v.is_finite()));
        for chunks in [vec![140], vec![1, 5, 64, 70], vec![130, 1, 1, 8]] {
            let got = last_logits(&model, &tokens, &chunks);
            assert!(max_diff(&want, &got) < 1e-3, "chunks {chunks:?}: {}", max_diff(&want, &got));
        }
    }

    #[test]
    fn the_indexer_limits_compressed_keys_once_more_than_top_k_are_visible() {
        let all = tiny(1, &[4], 0);
        let top1 = tiny(1, &[4], 1);
        // One compressed key visible: nothing to choose, same result.
        let tokens = [3u32, 4, 5, 6, 7];
        let (a, b) = (one_by_one(&all, &tokens), one_by_one(&top1, &tokens));
        assert!(max_diff(&a, &b) < 1e-5, "{}", max_diff(&a, &b));
        // Three visible, one kept: the result differs from attending to all.
        let tokens = [3u32, 4, 5, 6, 7, 3, 5, 4, 6, 7, 4, 3, 5];
        let (a, b) = (one_by_one(&all, &tokens), one_by_one(&top1, &tokens));
        assert!(max_diff(&a, &b) > 1e-4, "{}", max_diff(&a, &b));
    }

    #[test]
    fn continuing_a_sequence_needs_its_memory() {
        let model = tiny(2, &[4], 0);
        let mut run = Run::new(&model);
        run.feed(&[3, 4, 5, 6, 7]).expect("first tokens");
        // The memory of another sequence position is refused, not misread.
        run.memory = Dsv4Memory::default();
        let err = run.feed(&[3]).expect_err("memory was dropped");
        assert!(err.to_string().contains("attention memory"), "{err}");
    }
}
