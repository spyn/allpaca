//! Measured sensitivity: how far the model's next-token distribution moves
//! when one weight matrix is replaced by a quantised version of itself.
//!
//! A matrix's own error says how well it was quantised, not how much the
//! model cares. [`Probe`] reads calibration texts once with the original
//! weights, then again with one matrix swapped, and reports the mean
//! Kullback-Leibler divergence from the original predictions. That number is
//! comparable across matrices, which is what a bit budget needs.
//!
//! It runs on the CPU forward with every weight decoded to F32 in RAM, and
//! covers the token embedding and each block's dense attention and
//! feed-forward matrices.
//!
//! A swapped matrix cannot change anything ahead of its own block, so the
//! first reading keeps every block's input ([`LayerTrace`]) and a probe of
//! block `b` starts there: the blocks before it are not run again.

use std::path::Path;
use std::sync::Arc;

use allpaca_gguf::{extract_tokenizer, GgufFile};
use allpaca_types::{AllpacaError, FatalError};

use crate::anchors::RecurrentState;
use crate::kv_tier::KvTierManager;
use crate::model::{alloc_pages_for_seq, TinyModel};
use crate::telemetry::Metrics;
use crate::tokenizer::EngineTokenizer;

const PAGE_TOKENS: u32 = 128;

pub struct Probe {
    model: TinyModel,
    /// The output head is the embedding table (no `output.weight`).
    tied_head: bool,
    texts: Vec<Vec<u32>>,
    /// Log-probabilities of the original weights: `positions * vocab`.
    reference: Vec<f32>,
    /// Per text, every block's input with the original weights.
    saved: Vec<Arc<Vec<f32>>>,
    reuse: bool,
    block_runs: usize,
}

/// What the CPU forward does at the top of each block of one text.
#[derive(Debug, Clone)]
pub enum LayerTrace {
    /// Keep each block's input, and the last block's output:
    /// `[position][block or end][dim]`.
    Record { saved: Vec<f32>, ran: usize },
    /// Skip the blocks before `from` and give block `from` its saved input
    /// (`from` = the block count: only the output head runs).
    Resume { from: usize, saved: Arc<Vec<f32>>, ran: usize },
    /// Run block `at` alone: `input` is what it gets (`[position][dim]`),
    /// `output` what it hands on. Nothing else runs, the output head included.
    Block { at: usize, input: Arc<Vec<f32>>, output: Vec<f32> },
}

impl LayerTrace {
    /// At the top of block `layer` of position `pos`, or past the last block
    /// (`layer` = `blocks`): whether to skip the block. `x` is the hidden
    /// state.
    pub(crate) fn enter(&mut self, pos: usize, layer: usize, blocks: usize, x: &mut [f32]) -> bool {
        let at = (pos * (blocks + 1) + layer) * x.len();
        match self {
            Self::Record { saved, ran } => {
                if saved.len() < at + x.len() {
                    saved.resize(at + x.len(), 0.0);
                }
                saved[at..at + x.len()].copy_from_slice(x);
                *ran += usize::from(layer < blocks);
                false
            }
            Self::Resume { from, saved, ran } => {
                if layer < *from {
                    return true;
                }
                if layer == *from {
                    x.copy_from_slice(&saved[at..at + x.len()]);
                }
                *ran += usize::from(layer < blocks);
                false
            }
            Self::Block { at: block, input, output } => {
                let row = pos * x.len();
                if layer == *block {
                    x.copy_from_slice(&input[row..row + x.len()]);
                    return false;
                }
                if layer == *block + 1 {
                    if output.len() < row + x.len() {
                        output.resize(row + x.len(), 0.0);
                    }
                    output[row..row + x.len()].copy_from_slice(x);
                }
                true
            }
        }
    }

    fn ran(&self) -> usize {
        match self {
            Self::Record { ran, .. } | Self::Resume { ran, .. } => *ran,
            Self::Block { .. } => 0,
        }
    }
}

/// Where a probed tensor lives in the loaded model.
#[derive(Clone, Copy)]
enum Slot {
    Embed,
    Head,
    /// Layer and which of q, k, v, output, gate, up, down.
    Layer(usize, usize),
}

fn bad(msg: impl Into<String>) -> AllpacaError {
    AllpacaError::Fatal(FatalError::InvalidConfig(msg.into()))
}

impl Probe {
    /// Load `model` for the CPU and read `texts` (at most `max_tokens` tokens
    /// of each) with its original weights.
    pub fn open(model: &Path, texts: &[&str], max_tokens: usize) -> Result<Self, AllpacaError> {
        let (loaded, tok, tied_head) = load_host(model)?;
        let mut ids = Vec::new();
        for text in texts {
            let mut t = tok.encode(text)?;
            t.truncate(max_tokens);
            // A single token has no context to be sensitive to.
            if t.len() >= 2 {
                ids.push(t);
            }
        }
        if ids.is_empty() {
            return Err(bad("sensitivity probe: no calibration text of two or more tokens"));
        }
        let mut probe = Self {
            model: loaded,
            tied_head,
            texts: ids,
            reference: Vec::new(),
            saved: Vec::new(),
            reuse: true,
            block_runs: 0,
        };
        let record = |_: usize| Some(LayerTrace::Record { saved: Vec::new(), ran: 0 });
        let (reference, traces) = log_probs(&probe.model, &probe.texts, record)?;
        probe.reference = reference;
        probe.saved = traces
            .into_iter()
            .map(|t| match t {
                Some(LayerTrace::Record { saved, .. }) => Arc::new(saved),
                _ => Arc::new(Vec::new()),
            })
            .collect();
        Ok(probe)
    }

    /// Whether a probe starts at the swapped matrix's block (the default) or
    /// reads every text from the first block again. Both give the same number.
    pub fn set_prefix_reuse(&mut self, reuse: bool) {
        self.reuse = reuse;
    }

    /// Blocks run, over all positions, by the last [`Self::divergence`].
    pub fn block_runs(&self) -> usize {
        self.block_runs
    }

    /// Token positions the divergence is averaged over.
    pub fn positions(&self) -> usize {
        self.texts.iter().map(Vec::len).sum()
    }

    /// Whether [`Self::divergence`] can swap GGUF tensor `name`.
    pub fn covers(&self, name: &str) -> bool {
        self.len_of(name).is_some()
    }

    /// Mean divergence of the predictions, in nats per token, with tensor
    /// `name` holding `weights` instead of its own. The model is left as it was.
    pub fn divergence(&mut self, name: &str, weights: &[f32]) -> Result<f64, AllpacaError> {
        let len = self.len_of(name).ok_or_else(|| bad(format!("sensitivity probe: no matrix {name}")))?;
        if weights.len() != len {
            return Err(bad(format!("sensitivity probe: {name} holds {len} weights, got {}", weights.len())));
        }
        // The first block the swap can reach. The embedding is block 0's
        // input, so a swap of it has nothing to start from.
        let from = match self.locate(name) {
            Some(Slot::Layer(local, _)) => Some(local),
            Some(Slot::Head) => Some(self.model.layers.len()),
            _ => None,
        }
        .filter(|_| self.reuse);
        let original = self.put(name, weights.to_vec());
        let saved = &self.saved;
        let trace = |text: usize| {
            Some(match from {
                Some(from) => LayerTrace::Resume { from, saved: saved[text].clone(), ran: 0 },
                None => LayerTrace::Record { saved: Vec::new(), ran: 0 },
            })
        };
        let run = log_probs(&self.model, &self.texts, trace);
        self.put(name, original);
        let (run, traces) = run?;
        self.block_runs = traces.iter().flatten().map(LayerTrace::ran).sum();
        Ok(mean_divergence(&self.reference, &run, self.model.config.vocab_size))
    }

    pub(crate) fn blocks(&self) -> usize {
        self.model.layers.len()
    }

    /// The number block `layer` has in the file's tensor names.
    pub(crate) fn gguf_block(&self, layer: usize) -> usize {
        self.model.config.gguf_blk(layer)
    }

    /// Tensor `name`'s weights as the model holds them now.
    pub(crate) fn weights(&self, name: &str) -> Option<&[f32]> {
        Some(match self.locate(name)? {
            Slot::Embed => &self.model.embed,
            Slot::Head => &self.model.lm_head,
            Slot::Layer(local, kind) => {
                let layer = self.model.layers.get(local)?;
                [&layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.w_gate, &layer.w_up, &layer.w_down][kind]
            }
        })
    }

    /// Put `weights` in tensor `name`'s place for good.
    pub(crate) fn replace(&mut self, name: &str, weights: Vec<f32>) {
        self.put(name, weights);
    }

    /// Per text, what block `layer` got with the original weights: `[position][dim]`.
    pub(crate) fn original_inputs(&self, layer: usize) -> Vec<Arc<Vec<f32>>> {
        let (dim, per) = (self.model.config.dim, self.blocks() + 1);
        self.saved
            .iter()
            .map(|saved| {
                let row = |pos: usize| &saved[(pos * per + layer) * dim..(pos * per + layer + 1) * dim];
                Arc::new((0..saved.len() / (per * dim)).flat_map(|pos| row(pos).iter().copied()).collect())
            })
            .collect()
    }

    /// Per text, what the first block gets with the weights as they are now.
    pub(crate) fn first_inputs(&self) -> Result<Vec<Arc<Vec<f32>>>, AllpacaError> {
        // A full reading: the recording trace keeps every block's input.
        let record = |_: usize| Some(LayerTrace::Record { saved: Vec::new(), ran: 0 });
        let (dim, per) = (self.model.config.dim, self.blocks() + 1);
        let (_, traces) = log_probs(&self.model, &self.texts, record)?;
        Ok(traces
            .into_iter()
            .map(|trace| match trace {
                Some(LayerTrace::Record { saved, .. }) => {
                    Arc::new((0..saved.len() / (per * dim)).flat_map(|pos| saved[pos * per * dim..][..dim].to_vec()).collect())
                }
                _ => Arc::new(Vec::new()),
            })
            .collect())
    }

    /// Run block `layer` alone on `inputs` (per text, `[position][dim]`)
    /// with the weights as they are now. Returns what it hands on, per
    /// text, and the vectors its matrices were fed, per site, over all
    /// texts in order.
    #[allow(clippy::type_complexity)]
    pub(crate) fn block_sites(
        &self,
        layer: usize,
        inputs: &[Arc<Vec<f32>>],
    ) -> Result<(Vec<Arc<Vec<f32>>>, std::collections::HashMap<crate::calib::Site, Vec<f32>>), AllpacaError> {
        if layer >= self.blocks() || inputs.len() != self.texts.len() {
            return Err(bad(format!("probe: no block {layer} or not an input per text")));
        }
        self.model.calib.keep(self.gguf_block(layer));
        // One text after the other, so that the sites' vectors stay in order.
        let mut outputs = Vec::with_capacity(inputs.len());
        let mut failed = None;
        for (tokens, input) in self.texts.iter().zip(inputs) {
            let trace = LayerTrace::Block { at: layer, input: input.clone(), output: Vec::new() };
            match text_log_probs(&self.model, tokens, Some(trace)) {
                Ok((_, Some(LayerTrace::Block { output, .. }))) => outputs.push(Arc::new(output)),
                Ok(_) => failed = Some(bad("probe: the block trace was lost")),
                Err(e) => failed = Some(e),
            }
            if failed.is_some() {
                break;
            }
        }
        let sites = self.model.calib.take_kept();
        failed.map_or(Ok((outputs, sites)), Err)
    }

    fn locate(&self, name: &str) -> Option<Slot> {
        if name == "token_embd.weight" {
            return Some(Slot::Embed);
        }
        if name == "output.weight" {
            return (!self.tied_head).then_some(Slot::Head);
        }
        let mut parts = name.split('.');
        let (blk, index, kind, tail) = (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
        if blk != "blk" || tail != "weight" || parts.next().is_some() {
            return None;
        }
        let index: usize = index.parse().ok()?;
        let cfg = &self.model.config;
        let local = (0..cfg.n_layers).find(|l| cfg.gguf_blk(*l) == index)?;
        let kind = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
            .iter()
            .position(|k| *k == kind)?;
        Some(Slot::Layer(local, kind))
    }

    fn slot(&mut self, at: Slot) -> &mut Vec<f32> {
        match at {
            Slot::Embed => &mut self.model.embed,
            Slot::Head => &mut self.model.lm_head,
            Slot::Layer(local, kind) => {
                let layer = &mut self.model.layers[local];
                match kind {
                    0 => &mut layer.wq,
                    1 => &mut layer.wk,
                    2 => &mut layer.wv,
                    3 => &mut layer.wo,
                    4 => &mut layer.w_gate,
                    5 => &mut layer.w_up,
                    _ => &mut layer.w_down,
                }
            }
        }
    }

    fn len_of(&self, name: &str) -> Option<usize> {
        let len = match self.locate(name)? {
            Slot::Embed => self.model.embed.len(),
            Slot::Head => self.model.lm_head.len(),
            Slot::Layer(local, kind) => {
                let layer = self.model.layers.get(local)?;
                [&layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.w_gate, &layer.w_up, &layer.w_down][kind].len()
            }
        };
        (len > 0).then_some(len)
    }

    /// Put `weights` in tensor `name`'s place and return what was there.
    fn put(&mut self, name: &str, weights: Vec<f32>) -> Vec<f32> {
        if name == "token_embd.weight" && self.tied_head {
            self.model.lm_head = weights.clone();
        }
        match self.locate(name) {
            Some(at) => std::mem::replace(self.slot(at), weights),
            None => weights,
        }
    }
}

/// A model with every weight in F32 for the CPU forward, its tokenizer,
/// and whether its output head is the embedding table.
fn load_host(path: &Path) -> Result<(TinyModel, EngineTokenizer, bool), AllpacaError> {
    let file = Arc::new(GgufFile::open(path)?);
    let tok = EngineTokenizer::from_gguf(extract_tokenizer(&file)?);
    let mut model = TinyModel::from_gguf_host(&file)?;
    if !model.host_weights_ready() {
        return Err(bad("sensitivity probe: this architecture has no CPU forward on F32 weights"));
    }
    let tied_head = file.tensor("output.weight").is_none();
    model.bind_gguf(file);
    Ok((model, tok, tied_head))
}

/// Mean over positions of `KL(reference || now)`, both log-probabilities.
fn mean_divergence(reference: &[f32], now: &[f32], vocab: usize) -> f64 {
    let total: f64 = reference
        .chunks(vocab)
        .zip(now.chunks(vocab))
        .map(|(p, q)| p.iter().zip(q).map(|(lp, lq)| f64::from(lp.exp()) * f64::from(lp - lq)).sum::<f64>())
        .sum();
    (total / (reference.len() / vocab).max(1) as f64).max(0.0)
}

/// Log-probabilities of every position of `texts` (`positions * vocab`),
/// and each text's trace after its run. `trace(i)` is text `i`'s trace.
/// Texts are independent, so they are read on separate threads.
fn log_probs(
    model: &TinyModel,
    texts: &[Vec<u32>],
    trace: impl Fn(usize) -> Option<LayerTrace> + Sync,
) -> Result<(Vec<f32>, Vec<Option<LayerTrace>>), AllpacaError> {
    type Read = Result<(Vec<f32>, Option<LayerTrace>), AllpacaError>;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(texts.len()).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut reads: Vec<Option<Read>> = (0..texts.len()).map(|_| None).collect();
    std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let text = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if text >= texts.len() {
                            return done;
                        }
                        done.push((text, text_log_probs(model, &texts[text], trace(text))));
                    }
                })
            })
            .collect();
        for worker in workers {
            for (text, read) in worker.join().expect("probe thread") {
                reads[text] = Some(read);
            }
        }
    });
    let mut out = Vec::with_capacity(texts.iter().map(Vec::len).sum::<usize>() * model.config.vocab_size);
    let mut traces = Vec::with_capacity(texts.len());
    for read in reads {
        let (log_probs, trace) = read.expect("every text is read")?;
        out.extend(log_probs);
        traces.push(trace);
    }
    Ok((out, traces))
}

/// One text's log-probabilities (`positions * vocab`) and its trace afterwards.
fn text_log_probs(
    model: &TinyModel,
    tokens: &[u32],
    trace: Option<LayerTrace>,
) -> Result<(Vec<f32>, Option<LayerTrace>), AllpacaError> {
    let cfg = &model.config;
    let vocab = cfg.vocab_size;
    let metrics = Metrics::default();
    let mut out = Vec::with_capacity(tokens.len() * vocab);
    let mut logits = vec![0.0f32; vocab];
    let mut kv = KvTierManager::new(1 << 30, 1 << 30, 0.5);
    let pages = alloc_pages_for_seq(&mut kv, cfg, PAGE_TOKENS, tokens.len(), &metrics)?;
    let mut state = RecurrentState::zeros(cfg.n_layers, cfg.dim);
    state.trace = trace;
    for (pos, &token) in tokens.iter().enumerate() {
        model.forward_token(
            token,
            pos,
            &pages,
            &mut kv,
            &metrics,
            &mut logits,
            None,
            None,
            false,
            Some(&mut state),
            None,
            None,
            None,
        )?;
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let norm = max + logits.iter().map(|l| f64::from(l - max).exp()).sum::<f64>().ln() as f32;
        out.extend(logits.iter().map(|l| l - norm));
    }
    Ok((out, state.trace.take()))
}

/// Mean divergence (nats per token) of the predictions of the model at
/// `other` from those of the model at `reference` on `texts`, both on the
/// CPU: how much of the reference a requantised file kept. The texts are
/// tokenised with the reference's tokenizer.
pub fn divergence_between(
    reference: &Path,
    other: &Path,
    texts: &[&str],
    max_tokens: usize,
) -> Result<f64, AllpacaError> {
    let Probe { model, texts, reference, .. } = Probe::open(reference, texts, max_tokens)?;
    let vocab = model.config.vocab_size;
    drop(model);
    let (other, _, _) = load_host(other)?;
    if other.config.vocab_size != vocab {
        return Err(bad(format!(
            "divergence: vocabularies differ ({vocab} and {})",
            other.config.vocab_size
        )));
    }
    Ok(mean_divergence(&reference, &log_probs(&other, &texts, |_| None)?.0, vocab))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calib::tests::write_tiny_calib_gguf;

    /// The probe reports nothing for a matrix's own weights, more the
    /// further the replacement is from them, and leaves the model unchanged.
    #[test]
    fn divergence_grows_with_the_damage_and_the_model_is_restored() {
        let path = std::env::temp_dir().join(format!("allpaca_probe-{}.gguf", std::process::id()));
        write_tiny_calib_gguf(&path, 64, 128);
        let file = GgufFile::open(&path).unwrap();
        let mut probe = Probe::open(&path, &["hello world again hello world", "again hello", "hello"], 16).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(probe.positions() >= 7, "the one-token text is left out: {}", probe.positions());

        for name in [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ] {
            assert!(probe.covers(name), "{name}");
            let w = file.tensor_f32(name).unwrap();
            assert!(probe.divergence(name, &w).unwrap() < 1e-9, "{name}: its own weights");
            let slightly: Vec<f32> = w.iter().enumerate().map(|(i, v)| v * if i % 2 == 0 { 1.05 } else { 0.95 }).collect();
            let zeros = vec![0.0f32; w.len()];
            let (small, large) = (probe.divergence(name, &slightly).unwrap(), probe.divergence(name, &zeros).unwrap());
            assert!(small > 0.0, "{name}: a 5% change moves the output ({small})");
            assert!(large > small, "{name}: zeroed {large} vs 5% {small}");
            assert!(probe.divergence(name, &w).unwrap() < 1e-9, "{name}: restored after probing");
        }

        assert!(!probe.covers("blk.0.attn_norm.weight"));
        assert!(!probe.covers("blk.3.attn_q.weight"));
        assert!(!probe.covers("output.weight"), "this file ties the head to the embedding");
        assert!(probe.divergence("blk.0.attn_norm.weight", &[1.0; 64]).is_err());
        assert!(probe.divergence("blk.0.attn_q.weight", &[0.0; 7]).is_err());
    }

    /// A probe of a later block starts at that block: fewer blocks run, and
    /// the number is the one a full reading gives.
    #[test]
    fn a_probe_starts_at_the_swapped_block_and_gives_the_same_number() {
        use crate::calib::tests::write_tiny_calib_gguf_shape;
        let path = std::env::temp_dir().join(format!("allpaca_probe_reuse-{}.gguf", std::process::id()));
        write_tiny_calib_gguf_shape(&path, 64, 128, 0.3, 3, true);
        let file = GgufFile::open(&path).unwrap();
        let mut probe = Probe::open(&path, &["hello world again hello world", "again hello"], 16).unwrap();
        let _ = std::fs::remove_file(&path);
        let positions = probe.positions();
        // (tensor, blocks run per position when the probe starts at its block)
        let cases = [
            ("token_embd.weight", 3),
            ("blk.0.attn_q.weight", 3),
            ("blk.1.ffn_up.weight", 2),
            ("blk.1.attn_output.weight", 2),
            ("blk.2.attn_v.weight", 1),
            ("blk.2.ffn_down.weight", 1),
            ("output.weight", 0),
        ];
        for (name, blocks) in cases {
            let w = file.tensor_f32(name).unwrap();
            let changed: Vec<f32> = w.iter().enumerate().map(|(i, v)| v * if i % 2 == 0 { 1.2 } else { 0.8 }).collect();
            probe.set_prefix_reuse(false);
            let full = probe.divergence(name, &changed).unwrap();
            assert_eq!(probe.block_runs(), positions * 3, "{name}: a full reading");
            probe.set_prefix_reuse(true);
            let resumed = probe.divergence(name, &changed).unwrap();
            assert_eq!(probe.block_runs(), positions * blocks, "{name}");
            assert!(full > 0.0, "{name}: the change moves the output");
            assert_eq!(resumed.to_bits(), full.to_bits(), "{name}: {resumed} vs {full}");
            assert!(probe.divergence(name, &w).unwrap() < 1e-9, "{name}: restored");
        }
    }

    /// Opt-in timing on a real model:
    /// `ALLPACA_PROBE_MODEL=<gguf> ALLPACA_PROBE_TEXT=<txt> cargo test --release -p allpaca-core --lib real_probe_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_probe_timing() {
        let (Ok(model), Ok(text)) = (std::env::var("ALLPACA_PROBE_MODEL"), std::env::var("ALLPACA_PROBE_TEXT")) else {
            return;
        };
        let tokens: usize = std::env::var("ALLPACA_PROBE_TOKENS").ok().and_then(|t| t.parse().ok()).unwrap_or(16);
        let text = std::fs::read_to_string(text).unwrap();
        let texts: Vec<&str> = text.split("\n\n").filter(|t| !t.trim().is_empty()).collect();
        let file = GgufFile::open(Path::new(&model)).unwrap();
        let t = std::time::Instant::now();
        let mut probe = Probe::open(Path::new(&model), &texts, tokens).unwrap();
        println!("open: {} texts, {} positions, {:.1} s", texts.len(), probe.positions(), t.elapsed().as_secs_f64());
        let blocks = probe.model.config.n_layers;
        let mut names = vec!["token_embd.weight".to_string()];
        for b in [0, blocks / 2, blocks - 1] {
            names.push(format!("blk.{}.ffn_down.weight", probe.model.config.gguf_blk(b)));
        }
        for reuse in [false, true] {
            probe.set_prefix_reuse(reuse);
            for name in &names {
                let w: Vec<f32> = file.tensor_f32(name).unwrap().iter().map(|v| v * 1.05).collect();
                let t = std::time::Instant::now();
                let d = probe.divergence(name, &w).unwrap();
                println!("reuse {reuse:5} {name:28} {:7.2} s  kl {d:.6e}  block runs {}", t.elapsed().as_secs_f64(), probe.block_runs());
            }
        }
    }

    /// A file is no distance from itself, and a 2-bit requantisation of it
    /// is further from it than an 8-bit one.
    #[test]
    fn divergence_between_files_grows_as_bits_drop() {
        use crate::requant::{requantise, Options};
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let path = |tag: &str| dir.join(format!("allpaca_kld_{tag}-{pid}.gguf"));
        let (src, eight, two) = (path("src"), path("q8"), path("q2"));
        write_tiny_calib_gguf(&src, 64, 128);
        let gsq = crate::gsq::GsqConfig { steps: 20, ..Default::default() };
        requantise(&src, &eight, &Options { avg_bits: 8.5, gsq: gsq.clone(), ..Options::default() }).unwrap();
        requantise(&src, &two, &Options { avg_bits: 2.25, gsq, ..Options::default() }).unwrap();
        let texts = ["hello world again hello world", "again again hello"];
        let same = divergence_between(&src, &src, &texts, 32).unwrap();
        let d8 = divergence_between(&src, &eight, &texts, 32).unwrap();
        let d2 = divergence_between(&src, &two, &texts, 32).unwrap();
        for p in [src, eight, two] {
            let _ = std::fs::remove_file(p);
        }
        assert!(same < 1e-9, "{same}");
        assert!(d8 > 0.0 && d8 < d2, "8 bits {d8}, 2 bits {d2}");
    }
}
