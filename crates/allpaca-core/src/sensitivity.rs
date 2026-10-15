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
//! feed-forward matrices. In an MoE block it covers each stacked expert
//! tensor (`ffn_*_exps`, every expert at once) and the shared expert; the
//! routed experts stay packed in RAM and are decoded per routed expert
//! ([`crate::host_experts`]), so a 30B MoE model fits.
//!
//! A swapped matrix cannot change anything ahead of its own block, so the
//! first reading keeps every block's input ([`LayerTrace`]) and a probe of
//! block `b` starts there: the blocks before it are not run again.

use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;

use allpaca_gguf::{extract_tokenizer, GgufFile};
use allpaca_types::{AllpacaError, FatalError};

use crate::anchors::RecurrentState;
use crate::kv_tier::KvTierManager;
use crate::model::{alloc_pages_for_seq, ModelConfig, TinyModel};
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

/// What a swap replaced: weights, or a packed stack's earlier replacement
/// (`None`: the file's own bytes).
enum Held {
    Weights(Vec<f32>),
    Packed(Option<Vec<f32>>),
}

/// Where a probed tensor lives in the loaded model.
#[derive(Clone, Copy)]
enum Slot {
    Embed,
    Head,
    /// Layer and which of q, k, v, output, gate, up, down.
    Layer(usize, usize),
    /// Layer and which of the stacked gate, up, down expert tensors.
    Experts(usize, usize),
    /// Layer and which of the shared expert's gate, up, down.
    Shared(usize, usize),
    /// Layer and which of the gated delta-net projections ([`GDN_PARTS`]).
    Gdn(usize, usize),
}

/// The gated delta-net projections by tensor name, in the order of
/// [`gdn_part`]. A fused `ssm_ba` is reshuffled on load, so it is not here.
const GDN_PARTS: [&str; 5] = ["attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out"];

fn gdn_part(gdn: &crate::model::GdnHostWeights, which: usize) -> &Vec<f32> {
    [&gdn.w_qkv, &gdn.w_z, &gdn.w_alpha, &gdn.w_beta, &gdn.w_out][which]
}

fn gdn_part_mut(gdn: &mut crate::model::GdnHostWeights, which: usize) -> &mut Vec<f32> {
    match which {
        0 => &mut gdn.w_qkv,
        1 => &mut gdn.w_z,
        2 => &mut gdn.w_alpha,
        3 => &mut gdn.w_beta,
        _ => &mut gdn.w_out,
    }
}

/// Which matrix of an expert: gate, up, down.
fn expert_part(ffn: &crate::weight_set::LayerFfn, which: usize) -> &Vec<f32> {
    [&ffn.w_gate, &ffn.w_up, &ffn.w_down][which]
}

fn expert_part_mut(ffn: &mut crate::weight_set::LayerFfn, which: usize) -> &mut Vec<f32> {
    match which {
        0 => &mut ffn.w_gate,
        1 => &mut ffn.w_up,
        _ => &mut ffn.w_down,
    }
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
            Some(Slot::Layer(local, _) | Slot::Experts(local, _) | Slot::Shared(local, _) | Slot::Gdn(local, _)) => {
                Some(local)
            }
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
        self.restore(name, original);
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

    /// Tensor `name`'s weights as the model holds them now (a stacked
    /// expert tensor is put together from its experts).
    pub(crate) fn weights(&self, name: &str) -> Option<Cow<'_, [f32]>> {
        Some(match self.locate(name)? {
            Slot::Embed => Cow::Borrowed(&self.model.embed),
            Slot::Head => Cow::Borrowed(&self.model.lm_head),
            Slot::Layer(local, kind) => {
                let layer = self.model.layers.get(local)?;
                Cow::Borrowed([&layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.w_gate, &layer.w_up, &layer.w_down][kind])
            }
            Slot::Experts(local, which) => match self.packed(local) {
                Some(packed) => Cow::Owned(packed.stack(local, which).ok()?),
                None => {
                    let experts = &self.model.layers.get(local)?.experts;
                    Cow::Owned(experts.iter().flat_map(|e| expert_part(e, which).iter().copied()).collect())
                }
            },
            Slot::Shared(local, which) => Cow::Borrowed(expert_part(&self.model.layers.get(local)?.shared_expert, which)),
            Slot::Gdn(local, which) => Cow::Borrowed(gdn_part(self.model.layers.get(local)?.gdn.as_ref()?, which)),
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
        let parts = ["ffn_gate", "ffn_up", "ffn_down"];
        if let Some(which) = parts.iter().position(|p| kind.strip_suffix("_exps") == Some(p)) {
            return Some(Slot::Experts(local, which));
        }
        if let Some(which) = parts.iter().position(|p| kind.strip_suffix("_shexp") == Some(p)) {
            return Some(Slot::Shared(local, which));
        }
        if let Some(which) = GDN_PARTS.iter().position(|p| *p == kind) {
            return Some(Slot::Gdn(local, which));
        }
        let kind = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
            .iter()
            .position(|k| *k == kind)?;
        Some(Slot::Layer(local, kind))
    }

    /// The one vector a slot is held in; `None` for stacked experts.
    fn slot(&mut self, at: Slot) -> Option<&mut Vec<f32>> {
        Some(match at {
            Slot::Embed => &mut self.model.embed,
            Slot::Head => &mut self.model.lm_head,
            Slot::Experts(..) => return None,
            Slot::Shared(local, which) => expert_part_mut(&mut self.model.layers[local].shared_expert, which),
            Slot::Gdn(local, which) => gdn_part_mut(self.model.layers[local].gdn.as_mut()?, which),
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
        })
    }

    fn len_of(&self, name: &str) -> Option<usize> {
        let len = match self.locate(name)? {
            Slot::Embed => self.model.embed.len(),
            Slot::Head => self.model.lm_head.len(),
            Slot::Layer(local, kind) => {
                let layer = self.model.layers.get(local)?;
                [&layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.w_gate, &layer.w_up, &layer.w_down][kind].len()
            }
            Slot::Experts(local, which) if self.packed(local).is_some() => self.packed(local)?.stack_len(local, which)?,
            Slot::Experts(local, which) => {
                let experts = &self.model.layers.get(local)?.experts;
                // Every expert must hold the matrix for the stack to be swapped.
                let lens: Vec<usize> = experts.iter().map(|e| expert_part(e, which).len()).collect();
                if lens.iter().any(|&l| l == 0) {
                    return None;
                }
                lens.iter().sum()
            }
            Slot::Shared(local, which) => expert_part(&self.model.layers.get(local)?.shared_expert, which).len(),
            Slot::Gdn(local, which) => gdn_part(self.model.layers.get(local)?.gdn.as_ref()?, which).len(),
        };
        (len > 0).then_some(len)
    }

    /// The packed experts of `layer`, when they are kept packed.
    fn packed(&self, layer: usize) -> Option<&crate::host_experts::PackedExperts> {
        self.model.packed_experts.as_deref().filter(|p| p.has_layer(layer))
    }

    /// Put back what [`Self::put`] returned.
    fn restore(&mut self, name: &str, held: Held) {
        match held {
            Held::Weights(w) => {
                self.put(name, w);
            }
            Held::Packed(previous) => {
                if let Some(Slot::Experts(local, which)) = self.locate(name) {
                    if let Some(packed) = self.model.packed_experts.as_mut().and_then(Arc::get_mut) {
                        packed.swap(local, which, previous);
                    }
                }
            }
        }
    }

    /// Put `weights` in tensor `name`'s place and return what was there.
    fn put(&mut self, name: &str, weights: Vec<f32>) -> Held {
        if name == "token_embd.weight" && self.tied_head {
            self.model.lm_head = weights.clone();
        }
        match self.locate(name) {
            Some(Slot::Experts(local, which)) if self.packed(local).is_some() => {
                let packed = self.model.packed_experts.as_mut().and_then(Arc::get_mut).expect("the probe owns its model");
                Held::Packed(packed.swap(local, which, Some(weights)))
            }
            Some(Slot::Experts(local, which)) => {
                let experts = &mut self.model.layers[local].experts;
                let mut old = Vec::with_capacity(weights.len());
                let mut at = 0;
                for e in experts.iter_mut() {
                    let part = expert_part_mut(e, which);
                    let n = part.len();
                    old.extend_from_slice(part);
                    part.copy_from_slice(&weights[at..at + n]);
                    at += n;
                }
                Held::Weights(old)
            }
            Some(at) => Held::Weights(match self.slot(at) {
                Some(slot) => std::mem::replace(slot, weights),
                None => weights,
            }),
            None => Held::Weights(weights),
        }
    }
}

/// A model with every weight in F32 for the CPU forward, its tokenizer,
/// and whether its output head is the embedding table.
pub(crate) fn load_host(path: &Path) -> Result<(TinyModel, EngineTokenizer, bool), AllpacaError> {
    let file = Arc::new(GgufFile::open(path)?);
    let tok = EngineTokenizer::from_gguf(extract_tokenizer(&file)?);
    // Routed experts stay packed: decoded, a 30B MoE model would not fit.
    let mut model = if ModelConfig::from_gguf(&file)?.n_experts > 0 {
        TinyModel::from_gguf_host_packed_experts(file.clone())?
    } else {
        TinyModel::from_gguf_host(&file)?
    };
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
pub(crate) fn text_log_probs(
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
pub(crate) mod tests {
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

    /// In an MoE model the probe swaps a block's stacked expert tensors
    /// (every expert at once) and its shared expert, like any matrix.
    #[test]
    fn the_probe_swaps_stacked_experts_and_the_shared_expert() {
        use crate::calib::tests::write_tiny_calib_gguf_moe;
        let path = std::env::temp_dir().join(format!("allpaca_probe_moe-{}.gguf", std::process::id()));
        write_tiny_calib_gguf_moe(&path, 64, 128, 0.3, 2, false, 4);
        let file = GgufFile::open(&path).unwrap();
        let mut probe = Probe::open(&path, &["hello world again hello world", "again hello"], 16).unwrap();
        let _ = std::fs::remove_file(&path);
        for name in [
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.1.ffn_down_exps.weight",
            "blk.0.ffn_gate_shexp.weight",
            "blk.1.ffn_up_shexp.weight",
            "blk.1.ffn_down_shexp.weight",
        ] {
            assert!(probe.covers(name), "{name}");
            let w = file.tensor_f32(name).unwrap();
            assert_eq!(probe.weights(name).unwrap().as_ref(), w.as_slice(), "{name}: held as in the file");
            assert!(probe.divergence(name, &w).unwrap() < 1e-9, "{name}: its own weights");
            let zeros = vec![0.0f32; w.len()];
            assert!(probe.divergence(name, &zeros).unwrap() > 0.0, "{name}: zeroed moves the output");
            assert!(probe.divergence(name, &w).unwrap() < 1e-9, "{name}: restored after probing");
        }
        assert!(!probe.covers("blk.0.ffn_gate_inp.weight"), "the router stays as it is");
        let packed = probe.model.packed_experts.as_ref().expect("routed experts stay packed");
        assert_eq!(packed.swapped(), 0, "no decoded copy is kept once a probe is done");
    }

    /// Experts left packed and decoded per routed expert give the same
    /// predictions, bit for bit, as experts decoded at load.
    #[test]
    fn packed_experts_read_the_same_as_decoded_ones() {
        use crate::calib::tests::write_tiny_calib_gguf_moe;
        let path = std::env::temp_dir().join(format!("allpaca_packed_experts-{}.gguf", std::process::id()));
        write_tiny_calib_gguf_moe(&path, 64, 128, 0.3, 2, false, 4);
        let file = Arc::new(GgufFile::open(&path).unwrap());
        let decoded = TinyModel::from_gguf_host(&file).unwrap();
        let packed = TinyModel::from_gguf_host_packed_experts(file.clone()).unwrap();
        assert!(packed.packed_experts.is_some());
        assert!(packed.layers[0].experts.iter().all(|e| e.w_up.is_empty()), "nothing decoded at load");
        assert!(!decoded.layers[0].experts[0].w_up.is_empty());
        let texts = vec![vec![1u32, 3, 4, 5, 3, 4, 5], vec![1, 5, 5, 3]];
        let (a, _) = log_probs(&decoded, &texts, |_| None).unwrap();
        let (b, _) = log_probs(&packed, &texts, |_| None).unwrap();
        drop((decoded, packed, file));
        let _ = std::fs::remove_file(&path);
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "packed experts read differently");
    }

    /// Per-head q/k norms (`attn_q_norm`, `attn_k_norm`, as Qwen3 and Ling
    /// have) are applied on the CPU: with them, scaling the query and key
    /// projections changes nothing.
    #[test]
    fn the_cpu_forward_applies_the_per_head_q_and_k_norms() {
        use crate::calib::tests::write_tiny_calib_gguf;
        use allpaca_gguf::{write_gguf, GgufWriteTensor};
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let src = dir.join(format!("allpaca_qknorm_src-{pid}.gguf"));
        write_tiny_calib_gguf(&src, 64, 128);
        let file = GgufFile::open(&src).unwrap();
        // The fixture has 4 heads of 16.
        let with_norms = |path: &Path, scale: f32| {
            let mut owned: Vec<(String, Vec<u64>, Vec<f32>)> = file
                .tensors
                .iter()
                .map(|t| {
                    let mut w = file.tensor_f32(&t.name).unwrap();
                    if t.name.ends_with("attn_q.weight") || t.name.ends_with("attn_k.weight") {
                        w.iter_mut().for_each(|v| *v *= scale);
                    }
                    (t.name.clone(), t.dims.clone(), w)
                })
                .collect();
            let gains: Vec<f32> = (0..16).map(|i| 0.5 + i as f32 / 16.0).collect();
            owned.push(("blk.0.attn_q_norm.weight".into(), vec![16], gains.clone()));
            owned.push(("blk.0.attn_k_norm.weight".into(), vec![16], gains.iter().rev().copied().collect()));
            let views: Vec<GgufWriteTensor<'_>> =
                owned.iter().map(|(name, dims, data)| GgufWriteTensor { name: name.clone(), dims: dims.clone(), data }).collect();
            write_gguf(path, &file.metadata, &views).unwrap();
        };
        let (plain, scaled) = (dir.join(format!("allpaca_qknorm_1-{pid}.gguf")), dir.join(format!("allpaca_qknorm_10-{pid}.gguf")));
        with_norms(&plain, 1.0);
        with_norms(&scaled, 10.0);
        let read = |path: &Path| {
            let (model, _, _) = load_host(path).unwrap();
            log_probs(&model, &[vec![1u32, 3, 4, 5, 3, 4]], |_| None).unwrap().0
        };
        let (a, b) = (read(&plain), read(&scaled));
        for path in [src, plain, scaled] {
            let _ = std::fs::remove_file(path);
        }
        let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "scaling q and k moved the predictions by {worst}");
    }

    /// A copy of GGUF `src` at `dst` with each tensor passed through `edit`
    /// and `extra` tensors added.
    pub(crate) fn rewrite_gguf(src: &Path, dst: &Path, edit: impl Fn(&str, &mut Vec<f32>), extra: Vec<(String, Vec<u64>, Vec<f32>)>) {
        use allpaca_gguf::{write_gguf, GgufWriteTensor};
        let file = GgufFile::open(src).unwrap();
        let mut owned: Vec<(String, Vec<u64>, Vec<f32>)> = file
            .tensors
            .iter()
            .map(|t| {
                let mut w = file.tensor_f32(&t.name).unwrap();
                edit(&t.name, &mut w);
                (t.name.clone(), t.dims.clone(), w)
            })
            .collect();
        owned.extend(extra);
        let views: Vec<GgufWriteTensor<'_>> =
            owned.iter().map(|(name, dims, data)| GgufWriteTensor { name: name.clone(), dims: dims.clone(), data }).collect();
        write_gguf(dst, &file.metadata, &views).unwrap();
    }

    /// The shared expert's sigmoid gate (`ffn_gate_inp_shexp`, Qwen3-Next
    /// and Qwen3.5 MoE) scales its output on the CPU: a gate of zeros
    /// (sigmoid 0.5) reads as no gate with the shared expert's output halved.
    #[test]
    fn the_cpu_forward_gates_the_shared_expert() {
        use crate::calib::tests::write_tiny_calib_gguf_moe;
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let at = |tag: &str| dir.join(format!("allpaca_shexp_gate_{tag}-{pid}.gguf"));
        let (src, gated, halved) = (at("src"), at("gated"), at("halved"));
        write_tiny_calib_gguf_moe(&src, 64, 128, 0.3, 1, false, 4);
        let gate = vec![("blk.0.ffn_gate_inp_shexp.weight".to_string(), vec![64], vec![0.0f32; 64])];
        rewrite_gguf(&src, &gated, |_, _| {}, gate);
        rewrite_gguf(
            &src,
            &halved,
            |name, w| {
                if name == "blk.0.ffn_down_shexp.weight" {
                    w.iter_mut().for_each(|v| *v *= 0.5);
                }
            },
            Vec::new(),
        );
        let read = |path: &Path| {
            let (model, _, _) = load_host(path).unwrap();
            log_probs(&model, &[vec![1u32, 3, 4, 5, 3, 4]], |_| None).unwrap().0
        };
        let (a, b, plain) = (read(&gated), read(&halved), read(&src));
        for path in [src, gated, halved] {
            let _ = std::fs::remove_file(path);
        }
        let diff = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        assert!(diff(&plain, &b) > 1e-4, "halving the shared expert must show");
        assert!(diff(&a, &b) < 1e-5, "the gate did not halve the shared expert: {}", diff(&a, &b));
    }

    /// Gated delta-net blocks with an MoE feed-forward (Qwen3.5 MoE) run
    /// their experts on the CPU, and the probe covers them.
    #[test]
    fn delta_net_blocks_run_their_experts_on_the_cpu() {
        use crate::calib::tests::write_tiny_qwen35moe_gguf;
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let (src, silent) = (dir.join(format!("allpaca_gdn_moe-{pid}.gguf")), dir.join(format!("allpaca_gdn_moe_0-{pid}.gguf")));
        write_tiny_qwen35moe_gguf(&src, 4);
        rewrite_gguf(
            &src,
            &silent,
            |name, w| {
                if name == "blk.0.ffn_down_exps.weight" {
                    w.iter_mut().for_each(|v| *v = 0.0);
                }
            },
            Vec::new(),
        );
        let read = |path: &Path| {
            let (model, _, _) = load_host(path).unwrap();
            log_probs(&model, &[vec![1u32, 3, 4, 5, 3, 4]], |_| None).unwrap().0
        };
        let (a, b) = (read(&src), read(&silent));
        let mut probe = Probe::open(&src, &["hello world again hello", "again hello world"], 16).unwrap();
        let file = GgufFile::open(&src).unwrap();
        for name in ["blk.0.ffn_up_exps.weight", "blk.1.ffn_down_exps.weight", "blk.0.ffn_gate_shexp.weight"] {
            assert!(probe.covers(name), "{name}");
            let w = file.tensor_f32(name).unwrap();
            assert!(probe.divergence(name, &w).unwrap() < 1e-9, "{name}: its own weights");
        }
        drop((probe, file));
        for path in [src, silent] {
            let _ = std::fs::remove_file(path);
        }
        let diff = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        assert!(diff > 1e-4, "silencing block 0's experts changed nothing: they were not run ({diff})");
    }

    /// The probe swaps the gated delta-net projections of Qwen3.5 blocks.
    #[test]
    fn the_probe_swaps_the_delta_net_projections() {
        use crate::calib::tests::write_tiny_qwen35moe_gguf;
        let src = std::env::temp_dir().join(format!("allpaca_gdn_probe-{}.gguf", std::process::id()));
        write_tiny_qwen35moe_gguf(&src, 4);
        let mut probe = Probe::open(&src, &["hello world again hello", "again hello world"], 16).unwrap();
        let file = GgufFile::open(&src).unwrap();
        for kind in ["attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out"] {
            let name = format!("blk.1.{kind}.weight");
            assert!(probe.covers(&name), "{name}");
            let w = file.tensor_f32(&name).unwrap();
            assert!(probe.divergence(&name, &w).unwrap() < 1e-9, "{name}: its own weights");
            let moved = probe.divergence(&name, &vec![0.0; w.len()]).unwrap();
            assert!(moved > 1e-6, "{name}: zeroed changed nothing ({moved})");
            assert!(probe.divergence(&name, &w).unwrap() < 1e-9, "{name}: not restored");
        }
        drop((probe, file));
        let _ = std::fs::remove_file(src);
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

    /// Opt-in check of the CPU forward (experts packed) on a real model:
    /// greedy continuation of a prompt, to compare with llama.cpp's.
    /// `ALLPACA_PROBE_MODEL=<gguf> ALLPACA_GREEDY_PROMPT=<text> cargo test --release -p allpaca-core --lib real_cpu_greedy -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_cpu_greedy() {
        let (Ok(model), Ok(prompt)) = (std::env::var("ALLPACA_PROBE_MODEL"), std::env::var("ALLPACA_GREEDY_PROMPT")) else {
            return;
        };
        let t = std::time::Instant::now();
        let (model, tok, _) = load_host(Path::new(&model)).unwrap();
        println!("loaded in {:.1} s", t.elapsed().as_secs_f64());
        let mut ids = tok.encode(&prompt).unwrap();
        let vocab = model.config.vocab_size;
        for _ in 0..16 {
            let t = std::time::Instant::now();
            let (lp, _) = log_probs(&model, std::slice::from_ref(&ids), |_| None).unwrap();
            let last = &lp[lp.len() - vocab..];
            let next = (0..vocab).max_by(|&a, &b| last[a].total_cmp(&last[b])).unwrap() as u32;
            ids.push(next);
            println!("{next:>7} {:?}  ({:.1} s)", tok.decode(&[next]).unwrap_or_default(), t.elapsed().as_secs_f64());
        }
    }

    /// Opt-in: perplexity of a real model on the CPU forward, scored as
    /// `llama-perplexity -c N` scores it (chunks of N tokens, the first
    /// replaced by BOS, the second half of each scored), to compare.
    /// `ALLPACA_PROBE_MODEL=<gguf> ALLPACA_PROBE_TEXT=<txt> ALLPACA_PPL_CTX=128 cargo test ... real_cpu_perplexity -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_cpu_perplexity() {
        let (Ok(model), Ok(text)) = (std::env::var("ALLPACA_PROBE_MODEL"), std::env::var("ALLPACA_PROBE_TEXT")) else {
            return;
        };
        let n_ctx: usize = std::env::var("ALLPACA_PPL_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
        let (model, tok, _) = load_host(Path::new(&model)).unwrap();
        let text = std::fs::read_to_string(text).unwrap();
        let ids = tok.encode(&text).unwrap();
        let bos = tok.bos_id;
        let add_bos = bos.is_some_and(|b| ids.first() == Some(&b));
        let chunks: Vec<Vec<u32>> = ids
            .chunks_exact(n_ctx)
            .map(|c| {
                let mut c = c.to_vec();
                if let (true, Some(b)) = (add_bos, bos) {
                    c[0] = b;
                }
                c
            })
            .collect();
        let t = std::time::Instant::now();
        let (lp, _) = log_probs(&model, &chunks, |_| None).unwrap();
        let vocab = model.config.vocab_size;
        let (mut nll, mut count) = (0.0f64, 0usize);
        for (c, chunk) in chunks.iter().enumerate() {
            for pos in n_ctx / 2..n_ctx - 1 {
                let row = &lp[(c * n_ctx + pos) * vocab..][..vocab];
                nll -= f64::from(row[chunk[pos + 1] as usize]);
                count += 1;
            }
        }
        println!(
            "{} tokens, {} chunks of {n_ctx}, add_bos {add_bos}: PPL = {:.4}  ({:.0} s)",
            ids.len(),
            chunks.len(),
            (nll / count as f64).exp(),
            t.elapsed().as_secs_f64()
        );
    }

    /// Opt-in: a packed expert of a real model decodes to its slice of the
    /// whole decoded stack.
    #[test]
    #[ignore]
    fn real_packed_expert_matches_the_decoded_stack() {
        let Ok(model) = std::env::var("ALLPACA_PROBE_MODEL") else { return };
        let file = Arc::new(GgufFile::open(Path::new(&model)).unwrap());
        let cfg = ModelConfig::from_gguf(&file).unwrap();
        let packed = crate::host_experts::PackedExperts::from_gguf(file.clone(), &cfg).unwrap();
        let local = (0..cfg.n_layers).find(|&l| packed.has_layer(l)).unwrap();
        for (part, name) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].iter().enumerate() {
            let all = file.tensor_f32(&format!("blk.{}.{name}.weight", cfg.gguf_blk(local))).unwrap();
            let n = all.len() / cfg.n_experts;
            for e in [0, 5, cfg.n_experts - 1] {
                let ffn = packed.expert(local, e).unwrap();
                let got = [&ffn.w_gate, &ffn.w_up, &ffn.w_down][part];
                assert_eq!(got.as_slice(), &all[e * n..(e + 1) * n], "{name} expert {e}");
            }
        }
        println!("layer {local}: packed experts match");
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
