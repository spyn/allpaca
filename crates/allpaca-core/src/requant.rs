//! Requantise a GGUF to a mixed-precision file under an average-bits budget.
//!
//! Each weight matrix can go to one of a few types. GSQ ([`crate::gsq`])
//! produces the 2-bit option; the budgeted choice of one type per matrix
//! is RCO's ([`crate::rco`]) knapsack, which meets the average-bits budget
//! at the least total error. Everything else is copied unchanged.
//!
//! The error is each matrix's own error relative to its energy, weighted by
//! its size, with matrices counted as independent. Given measured inputs
//! ([`Options::stats`]) it is the error of the matrix's output on them;
//! without, inputs count as uncorrelated and it is the plain squared error
//! of the weights.
//!
//! With a probe ([`Options::probe`]) the choice uses instead how far each
//! option of each matrix, alone, moves the model's predictions on
//! calibration text ([`crate::sensitivity`]): one number in the same unit
//! for every matrix. It still adds the matrices' effects up as if they were
//! independent. Matrices the probe cannot run are estimated from their own
//! error.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use allpaca_gguf::{GgmlType, GgufFile, GgufStreamTensor, GgufStreamWriter, GgufValue};
use allpaca_types::{AllpacaError, FatalError};

use crate::calib::InputStats;
use crate::gsq::{output_error, quantise_2bit, Gram, GsqConfig};
use crate::rco::{knapsack, Problem};
use crate::sensitivity::Probe;

/// A type a matrix may be requantised to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// GSQ onto ggml `Q2_0`: 2.25 bits per weight.
    Q2_0,
    /// The K-quants, encoded as llama.cpp encodes them
    /// ([`allpaca_cpu::encode_q4_k`] and the like).
    Q3K,
    Q4_0,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}

impl Target {
    /// Every type, by size.
    pub const ALL: [Target; 7] =
        [Target::Q2_0, Target::Q3K, Target::Q4_0, Target::Q4K, Target::Q5K, Target::Q6K, Target::Q8_0];

    pub fn bits(self) -> f64 {
        match self {
            Target::Q2_0 => 2.25,
            Target::Q3K => 3.4375,
            Target::Q4_0 | Target::Q4K => 4.5,
            Target::Q5K => 5.5,
            Target::Q6K => 6.5625,
            Target::Q8_0 => 8.5,
        }
    }

    /// The type by the name llama.cpp gives it (`q4_k`, `Q2_0`).
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.name().eq_ignore_ascii_case(name))
    }

    pub fn name(self) -> &'static str {
        match self {
            Target::Q2_0 => "Q2_0",
            Target::Q3K => "Q3_K",
            Target::Q4_0 => "Q4_0",
            Target::Q4K => "Q4_K",
            Target::Q5K => "Q5_K",
            Target::Q6K => "Q6_K",
            Target::Q8_0 => "Q8_0",
        }
    }

    pub(crate) fn ggml(self) -> GgmlType {
        match self {
            Target::Q2_0 => GgmlType::Q2_0,
            Target::Q3K => GgmlType::Q3_K,
            Target::Q4_0 => GgmlType::Q4_0,
            Target::Q4K => GgmlType::Q4_K,
            Target::Q5K => GgmlType::Q5_K,
            Target::Q6K => GgmlType::Q6_K,
            Target::Q8_0 => GgmlType::Q8_0,
        }
    }

    /// Weights and bytes of one block.
    fn block(self) -> (usize, usize) {
        match self {
            Target::Q2_0 => (64, 18),
            Target::Q3K => (256, allpaca_cpu::Q3_K_BLOCK_BYTES),
            Target::Q4_0 => (32, 18),
            Target::Q4K => (256, allpaca_cpu::Q4_K_BLOCK_BYTES),
            Target::Q5K => (256, allpaca_cpu::Q5_K_BLOCK_BYTES),
            Target::Q6K => (256, allpaca_cpu::Q6_K_BLOCK_BYTES),
            Target::Q8_0 => (32, 34),
        }
    }

    /// Whether rows of `cols` weights are whole blocks of this type.
    pub fn fits(self, cols: usize) -> bool {
        cols % self.block().0 == 0
    }

    pub(crate) fn quantise(
        self,
        w: &[f32],
        rows: usize,
        cols: usize,
        gram: &Gram<'_>,
        gsq: &GsqConfig,
    ) -> Result<Vec<u8>, AllpacaError> {
        Ok(match self {
            Target::Q2_0 => quantise_2bit(w, rows, cols, gram, gsq)?.to_q2_0(),
            Target::Q3K => allpaca_cpu::encode_q3_k(w),
            Target::Q4_0 => allpaca_gguf::quantize_q4_0(w),
            Target::Q4K => allpaca_cpu::encode_q4_k(w),
            Target::Q5K => allpaca_cpu::encode_q5_k(w),
            Target::Q6K => allpaca_cpu::encode_q6_k(w),
            Target::Q8_0 => allpaca_gguf::quantize_q8_0(w),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    /// Average bits per weight over the requantised matrices.
    pub avg_bits: f64,
    /// The types a matrix may take ([`Target::ALL`] by default). A matrix
    /// is offered those whose blocks its rows are whole numbers of.
    pub targets: Vec<Target>,
    /// Matrices with fewer weights are copied as they are.
    pub min_weights: usize,
    pub gsq: GsqConfig,
    /// Steps of the budget the allocation is solved in.
    pub resolution: usize,
    /// Measured matrix inputs ([`crate::calib::capture`]). A matrix they
    /// cover is quantised and scored by its output error on those inputs;
    /// the others by their plain squared error.
    pub stats: Option<InputStats>,
    /// Share of a measured Gram's mean diagonal added to its diagonal.
    pub damping: f32,
    /// Choose the types by measured sensitivity ([`crate::sensitivity`])
    /// instead of each matrix's own error. A matrix the probe cannot swap
    /// counts its own error times the probed matrices' median divergence
    /// per unit of error.
    pub probe: Option<ProbeTexts>,
    /// A table of probe measurements: written after a probe, and when it
    /// exists, read instead of probing (`probe` is then not needed). With
    /// `probe` too, the probe measures only the matrices the table lacks.
    pub probe_table: Option<PathBuf>,
    /// Once the types are chosen, quantise the blocks' matrices in order,
    /// each for the original outputs on the inputs the ones before it
    /// leave it ([`crate::propagate`]). Needs every weight in F32 in RAM.
    pub propagate: Option<crate::propagate::Settings>,
    /// Called after each matrix is measured: matrices done, of how many, its name.
    pub progress: Option<fn(usize, usize, &str)>,
}

/// Calibration texts for the sensitivity probe.
#[derive(Debug, Clone)]
pub struct ProbeTexts {
    pub texts: Vec<String>,
    /// Tokens read of each text.
    pub max_tokens: usize,
    /// Probe each matrix at its fewest bits only, and take the others as
    /// that divergence scaled by the matrix's own error: one read of the
    /// texts per matrix instead of one per type.
    pub scaled: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            avg_bits: 3.0,
            targets: Target::ALL.to_vec(),
            min_weights: 4096,
            gsq: GsqConfig::default(),
            resolution: 4000,
            stats: None,
            damping: 0.01,
            probe: None,
            probe_table: None,
            propagate: None,
            progress: None,
        }
    }
}

/// What was done to one requantised matrix.
#[derive(Debug, Clone)]
pub struct Choice {
    pub name: String,
    pub weights: usize,
    pub target: Target,
    /// Error of each of [`Report::targets`] relative to the matrix's
    /// energy: of its output on the measured inputs when `calibrated`, of
    /// its weights otherwise. `NaN` for a type its rows do not fit.
    pub relative_error: Vec<f64>,
    pub calibrated: bool,
    /// With a probe: divergence of the model's predictions (nats per
    /// token) with this matrix alone at each of [`Report::targets`].
    pub divergence: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct Report {
    /// The types that were offered ([`Options::targets`]).
    pub targets: Vec<Target>,
    pub choices: Vec<Choice>,
    /// Tensors copied unchanged.
    pub copied: usize,
    /// Requantised matrices that used measured inputs.
    pub calibrated: usize,
    /// Matrices quantised in order with propagated calibration.
    pub propagated: usize,
}

impl Report {
    /// Average bits per weight actually reached over the requantised matrices.
    pub fn avg_bits(&self) -> f64 {
        let total: f64 = self.choices.iter().map(|c| c.weights as f64).sum();
        self.choices.iter().map(|c| c.weights as f64 * c.target.bits()).sum::<f64>() / total.max(1.0)
    }
}

/// Rows and row length of a matrix whose rows are whole blocks of at
/// least one of `targets` (ggml lists the row length first), or `None`
/// for anything else.
fn matrix_shape(dims: &[u64], min_weights: usize, targets: &[Target]) -> Option<(usize, usize)> {
    let cols = *dims.first()? as usize;
    let n: usize = dims.iter().product::<u64>() as usize;
    (dims.len() >= 2 && cols > 0 && targets.iter().any(|t| t.fits(cols)) && n >= min_weights).then(|| (n / cols, cols))
}

/// One line per matrix and type it fits: name, type, divergence.
fn write_probe_table<'a>(
    path: &Path,
    names: impl Iterator<Item = &'a str>,
    targets: &[Target],
    divergence: &[Vec<f64>],
) -> Result<(), AllpacaError> {
    let mut text = String::new();
    for (name, row) in names.zip(divergence) {
        for (t, d) in targets.iter().zip(row).filter(|(_, d)| !d.is_nan()) {
            text.push_str(&format!("{name}\t{}\t{d:e}\n", t.name()));
        }
    }
    std::fs::write(path, text).map_err(|e| AllpacaError::Fatal(FatalError::Io(format!("{}: {e}", path.display()))))
}

/// The divergences of a probe table, by matrix and type name.
fn read_probe_table(path: &Path) -> Result<HashMap<String, HashMap<String, f64>>, AllpacaError> {
    let bad = |why: String| AllpacaError::Fatal(FatalError::InvalidConfig(format!("{}: {why}", path.display())));
    let text = std::fs::read_to_string(path).map_err(|e| bad(e.to_string()))?;
    let mut table: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut f = line.split('\t');
        let (Some(name), Some(target), Some(d), None) = (f.next(), f.next(), f.next(), f.next()) else {
            return Err(bad(format!("not name, type, divergence: {line}")));
        };
        let d: f64 = d.trim().parse().map_err(|_| bad(format!("not a number: {line}")))?;
        table.entry(name.to_string()).or_default().insert(target.to_uppercase(), d);
    }
    Ok(table)
}

/// Requantise `input` into `output`.
pub fn requantise(input: &Path, output: &Path, opts: &Options) -> Result<Report, AllpacaError> {
    Ok(requantise_each(input, &[(output, opts.avg_bits)], opts)?.remove(0))
}

/// Requantise `input` once per `(output, average bits)`. Every matrix's
/// options are measured once for all of them, which is nearly all of the
/// work. `opts.avg_bits` is not used.
pub fn requantise_each(
    input: &Path,
    outputs: &[(&Path, f64)],
    opts: &Options,
) -> Result<Vec<Report>, AllpacaError> {
    let file = GgufFile::open(input)?;
    let targets = &opts.targets;
    if targets.is_empty() || targets.iter().enumerate().any(|(i, t)| targets[..i].contains(t)) {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "requantise: the types to choose from must be one or more, each once".into(),
        )));
    }
    // A saved table stands in for the probe: given probe texts as well,
    // the probe measures only what the table lacks.
    let saved = match &opts.probe_table {
        Some(path) if path.exists() => Some(read_probe_table(path)?),
        _ => None,
    };
    let mut probe = match &opts.probe {
        Some(p) => {
            let texts: Vec<&str> = p.texts.iter().map(String::as_str).collect();
            Some(Probe::open(input, &texts, p.max_tokens)?)
        }
        None => None,
    };
    let candidates: Vec<(usize, usize, usize)> = file
        .tensors
        .iter()
        .enumerate()
        .filter(|(_, t)| t.name.ends_with(".weight") && !matches!(t.ggml_type, GgmlType::I32 | GgmlType::Other))
        .filter_map(|(i, t)| matrix_shape(&t.dims, opts.min_weights, targets).map(|(rows, cols)| (i, rows, cols)))
        .collect();
    if candidates.is_empty() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "requantise: the file has no weight matrix whose rows are whole blocks of the types asked for".into(),
        )));
    }
    // The measured, damped Gram of a matrix's input, when the run covers it.
    let measured = |i: usize, cols: usize| -> Option<Vec<f32>> {
        let gram = opts.stats.as_ref()?.gram(&file.tensors[i].name)?;
        (gram.cols == cols).then(|| gram.damped(opts.damping))
    };
    // Error of every option for every matrix. The 2-bit result is kept: it
    // is small, and the search that makes it is the slow part.
    let mut errors = Vec::with_capacity(candidates.len());
    let mut two_bit = Vec::with_capacity(candidates.len());
    let mut calibrated = Vec::with_capacity(candidates.len());
    let mut divergence = Vec::with_capacity(candidates.len());
    // Whether a matrix's divergence is measured (or read from the table);
    // the others' is estimated from their own error once all are seen.
    let mut measured_by_probe = Vec::with_capacity(candidates.len());
    for &(i, rows, cols) in &candidates {
        let name = &file.tensors[i].name;
        let from_table = saved.as_ref().is_some_and(|s| s.contains_key(name));
        let covered = from_table || probe.as_ref().is_some_and(|p| p.covers(name));
        measured_by_probe.push(covered);
        let w = file.tensor_f32(name)?;
        let h = measured(i, cols);
        let gram = h.as_deref().map_or(Gram::Identity, Gram::Full);
        let energy = output_error(&w, &vec![0.0; w.len()], cols, &gram);
        // A type the rows do not fit is not an option for this matrix.
        let mut row = vec![f64::NAN; targets.len()];
        let mut moved = vec![f64::NAN; targets.len()];
        let mut decoded = vec![0.0f32; w.len()];
        let mut kept = Vec::new();
        let scaled = opts.probe.as_ref().is_some_and(|p| p.scaled) && probe.is_some() && !from_table;
        // The type a scaled probe measures: the fewest bits the rows fit.
        let anchor = targets
            .iter()
            .enumerate()
            .filter(|(_, t)| t.fits(cols))
            .min_by(|a, b| a.1.bits().total_cmp(&b.1.bits()))
            .map(|(at, _)| at);
        for (at, &target) in targets.iter().enumerate() {
            if !target.fits(cols) {
                continue;
            }
            let bytes = target.quantise(&w, rows, cols, &gram, &opts.gsq)?;
            allpaca_gguf::dequant_packed(target.ggml(), &bytes, w.len(), &mut decoded)?;
            row[at] = output_error(&w, &decoded, cols, &gram) / energy.max(1e-30);
            if let Some(saved) = saved.as_ref().filter(|_| from_table) {
                moved[at] = saved[name.as_str()].get(target.name()).copied().ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                        "requantise: the probe table has no {} for {name}",
                        target.name()
                    )))
                })?;
            } else if let Some(probe) = probe.as_mut().filter(|_| covered && (!scaled || Some(at) == anchor)) {
                moved[at] = probe.divergence(name, &decoded)?;
            }
            if target == Target::Q2_0 {
                kept = bytes;
            }
        }
        if let (true, true, Some(anchor)) = (covered, scaled, anchor) {
            let per_error = if row[anchor] > 0.0 { moved[anchor] / row[anchor] } else { 0.0 };
            for at in (0..targets.len()).filter(|&at| at != anchor && !row[at].is_nan()) {
                moved[at] = per_error * row[at];
            }
        }
        two_bit.push(kept);
        errors.push(row);
        calibrated.push(h.is_some());
        divergence.push(moved);
        if let Some(progress) = opts.progress {
            progress(errors.len(), candidates.len(), name);
        }
    }
    // The probe holds the whole model in F32: not needed past here.
    let probed = probe.take().is_some();
    if probed {
        if let Some(path) = &opts.probe_table {
            let measured: Vec<usize> = (0..candidates.len()).filter(|&c| measured_by_probe[c]).collect();
            let names = measured.iter().map(|&c| file.tensors[candidates[c].0].name.as_str());
            let rows: Vec<Vec<f64>> = measured.iter().map(|&c| divergence[c].clone()).collect();
            write_probe_table(path, names, targets, &rows)?;
        }
    }
    let probed = probed || saved.is_some();
    if probed {
        // A matrix the probe cannot run (a multi-token-prediction layer, the
        // fused delta-net projections) moves the predictions by its own
        // error times the typical divergence per unit of error.
        let mut per_error: Vec<f64> = (0..candidates.len())
            .filter(|&c| measured_by_probe[c])
            .flat_map(|c| errors[c].iter().zip(&divergence[c]).map(|(e, d)| d / e).collect::<Vec<_>>())
            .filter(|r| r.is_finite())
            .collect();
        per_error.sort_by(f64::total_cmp);
        let typical = per_error.get(per_error.len() / 2).copied().unwrap_or(1.0);
        for c in (0..candidates.len()).filter(|&c| !measured_by_probe[c]) {
            divergence[c] = errors[c].iter().map(|e| e * typical).collect();
        }
    }
    let weights: Vec<f64> = candidates.iter().map(|&(_, rows, cols)| (rows * cols) as f64).collect();
    // The loss of a matrix's option: how far it moved the model's predictions
    // when probed, otherwise its error weighted by its size.
    let mean_weight = weights.iter().sum::<f64>() / candidates.len() as f64;
    let loss: Vec<f64> = if probed {
        divergence.iter().flatten().copied().collect()
    } else {
        errors
            .iter()
            .zip(&weights)
            .flat_map(|(row, w)| row.iter().map(move |e| e * w / mean_weight))
            .collect()
    };
    // What is not an option can never be worth taking.
    let gain: Vec<f64> = loss.iter().map(|l| if l.is_nan() { f64::NEG_INFINITY } else { -l }).collect();
    let mut reports = Vec::with_capacity(outputs.len());
    for &(output, avg_bits) in outputs {
        let problem = Problem {
            costs: targets.iter().map(|t| t.bits()).collect(),
            weights: weights.clone(),
            budget: avg_bits,
        };
        // The loss is one term per matrix and option, so the knapsack optimum is exact.
        let assignment = knapsack(&problem, &gain, opts.resolution)?;

        let chosen = |i: usize| candidates.iter().position(|c| c.0 == i).map(|at| targets[assignment[at]]);
        let declared: Vec<GgufStreamTensor> = file
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let n = t.n_elements() as usize;
                let (ggml_type_id, nbytes) = match chosen(i) {
                    Some(target) => (target.ggml() as u32, n / target.block().0 * target.block().1),
                    None => (t.ggml_type_id, t.nbytes()),
                };
                GgufStreamTensor {
                    name: t.name.clone(),
                    // ggml order (innermost first), as the parser holds them.
                    dims: t.dims.clone(),
                    ggml_type_id,
                    nbytes,
                }
            })
            .collect();
        let mut metadata = file.metadata.clone();
        metadata.insert(
            "general.description".into(),
            GgufValue::String(format!(
                "requantised by Allpaca (GSQ 2-bit, K-quants, RCO allocation) to {:.2} bits per weight on average",
                avg_bits
            )),
        );
        for split in ["split.no", "split.count", "split.tensors.count"] {
            metadata.remove(split);
        }
        // In order, on the inputs the quantised model has: these replace
        // what each matrix alone was quantised to.
        let in_order = match &opts.propagate {
            Some(settings) => {
                let pick = |name: &str| {
                    let i = file.tensors.iter().position(|t| t.name == name)?;
                    let at = candidates.iter().position(|c| c.0 == i)?;
                    Some((targets[assignment[at]], candidates[at].1, candidates[at].2))
                };
                crate::propagate::quantise_in_order(input, settings, &pick, &opts.gsq, opts.damping, opts.progress)?
            }
            None => Default::default(),
        };
        let mut writer = GgufStreamWriter::create(output, &metadata, &declared)?;
        for (i, t) in file.tensors.iter().enumerate() {
            if let Some(bytes) = in_order.get(&t.name) {
                writer.write_tensor(bytes)?;
                continue;
            }
            match chosen(i) {
                Some(Target::Q2_0) => {
                    let at = candidates.iter().position(|c| c.0 == i).expect("chosen");
                    writer.write_tensor(&two_bit[at])?;
                }
                Some(target) => {
                    let (_, rows, cols) = candidates[candidates.iter().position(|c| c.0 == i).expect("chosen")];
                    let w = file.tensor_f32(&t.name)?;
                    writer.write_tensor(&target.quantise(&w, rows, cols, &Gram::Identity, &opts.gsq)?)?;
                }
                None => writer.write_tensor(file.tensor_bytes(&t.name)?)?,
            }
        }
        writer.finish()?;
        reports.push(Report {
            choices: candidates
                .iter()
                .zip(&assignment)
                .zip(&errors)
                .zip(&calibrated)
                .zip(&divergence)
                .map(|((((&(i, rows, cols), &o), row), &calibrated), moved)| Choice {
                    name: file.tensors[i].name.clone(),
                    weights: rows * cols,
                    target: targets[o],
                    relative_error: row.clone(),
                    calibrated,
                    divergence: probed.then(|| moved.clone()),
                })
                .collect(),
            targets: targets.clone(),
            copied: file.tensors.len() - candidates.len(),
            calibrated: calibrated.iter().filter(|c| **c).count(),
            propagated: in_order.len(),
        });
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_is_the_cheapest_for_unequal_matrices() {
        // Two large matrices and six small ones with the same error per
        // weight at each width: the first bits (2 -> 4) buy far more than
        // the next (4 -> 8), so nothing should get 8 bits while anything has 2.
        let weights = [32.0, 32.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0];
        let problem = Problem {
            costs: CLASSIC.iter().map(|t| t.bits()).collect(),
            weights: weights.to_vec(),
            budget: 4.0,
        };
        let per_weight = [0.127, 0.0093, 0.0];
        let gain: Vec<f64> = weights.iter().flat_map(|w| per_weight.iter().map(move |e| -e * w)).collect();
        let got = knapsack(&problem, &gain, 4000).unwrap();
        assert!((problem.cost_of(&got) - 4.0).abs() < 0.03, "{}", problem.cost_of(&got));
        assert!(!(got.contains(&0) && got.contains(&2)), "2-bit and 8-bit matrices together: {got:?}");
        assert!(got[0] == 1 && got[1] == 1, "the large matrices keep 4 bits: {got:?}");
    }
    use allpaca_gguf::{write_gguf, GgufWriteTensor};
    use std::collections::HashMap;

    /// The options before the K-quants: two, four and eight bits.
    const CLASSIC: [Target; 3] = [Target::Q2_0, Target::Q4_0, Target::Q8_0];

    fn noise(n: usize, seed: u32, amp: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    }

    #[test]
    fn one_pass_writes_a_file_per_budget() {
        let dir = std::env::temp_dir();
        let at = |tag: &str| dir.join(format!("allpaca_requant_each_{tag}-{}.gguf", std::process::id()));
        let (src, low, high, alone) = (at("in"), at("low"), at("high"), at("alone"));
        let data: Vec<Vec<f32>> = (0..4).map(|i| noise(64 * 64, 10 + i, 1.0 + i as f32)).collect();
        let names = ["blk.0.a.weight", "blk.0.b.weight", "blk.0.c.weight", "blk.0.d.weight"];
        let views: Vec<GgufWriteTensor<'_>> = names
            .iter()
            .zip(&data)
            .map(|(name, data)| GgufWriteTensor {
                name: (*name).into(),
                dims: vec![64, 64],
                data,
            })
            .collect();
        let mut meta = HashMap::new();
        meta.insert("general.architecture".to_string(), GgufValue::String("llama".into()));
        write_gguf(&src, &meta, &views).unwrap();

        let opts = Options {
            gsq: GsqConfig {
                steps: 20,
                ..GsqConfig::default()
            },
            ..Options::default()
        };
        // Two matrices at 2.25 bits and two at 4.5; then all four at 4.5.
        let reports = requantise_each(&src, &[(&low, 3.375), (&high, 4.5)], &opts).unwrap();
        assert_eq!(reports.len(), 2);
        assert!((reports[0].avg_bits() - 3.375).abs() < 1e-9, "{}", reports[0].avg_bits());
        assert!((reports[1].avg_bits() - 4.5).abs() < 1e-9, "{}", reports[1].avg_bits());
        let types = |path: &Path| -> Vec<GgmlType> {
            let file = GgufFile::open(path).unwrap();
            names.iter().map(|n| file.tensor(n).unwrap().ggml_type).collect()
        };
        assert_eq!(types(&low).iter().filter(|t| **t == GgmlType::Q2_0).count(), 2, "{:?}", types(&low));
        assert_eq!(types(&high), vec![GgmlType::Q4_0; 4]);
        // Each file is the one a run of its own would have written.
        requantise(&src, &alone, &Options { avg_bits: 3.375, ..opts }).unwrap();
        assert_eq!(std::fs::read(&low).unwrap(), std::fs::read(&alone).unwrap());
        for path in [src, low, high, alone] {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn requantise_meets_the_budget_and_spends_bits_where_they_help() {
        let dir = std::env::temp_dir();
        let (src, dst) = (
            dir.join(format!("allpaca_requant_in-{}.gguf", std::process::id())),
            dir.join(format!("allpaca_requant_out-{}.gguf", std::process::id())),
        );
        // Four equal matrices. "coarse" holds only three levels, which two
        // bits represent exactly; the others are dense noise.
        let coarse: Vec<f32> = (0..64 * 64).map(|i| [-0.5f32, 0.0, 0.5][(i * 7 + i / 5) % 3]).collect();
        let tensors: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
            ("blk.0.coarse.weight", vec![64, 64], coarse),
            ("blk.0.a.weight", vec![64, 64], noise(64 * 64, 1, 1.0)),
            ("blk.0.b.weight", vec![64, 64], noise(64 * 64, 2, 1.0)),
            ("blk.0.c.weight", vec![64, 64], noise(64 * 64, 3, 1.0)),
            ("blk.0.norm.weight", vec![64], vec![1.0; 64]),
            ("blk.0.odd.weight", vec![64, 60], noise(64 * 60, 4, 1.0)),
        ];
        let views: Vec<GgufWriteTensor<'_>> = tensors
            .iter()
            .map(|(name, dims, data)| GgufWriteTensor {
                name: (*name).into(),
                dims: dims.clone(),
                data,
            })
            .collect();
        let mut meta = HashMap::new();
        meta.insert("general.architecture".to_string(), GgufValue::String("llama".into()));
        meta.insert("general.name".to_string(), GgufValue::String("requant-test".into()));
        write_gguf(&src, &meta, &views).unwrap();

        let opts = Options {
            // One matrix at 2.25 bits and three at 4.5 average 3.9375.
            avg_bits: 3.9375,
            gsq: GsqConfig {
                steps: 60,
                ..GsqConfig::default()
            },
            ..Options::default()
        };
        let report = requantise(&src, &dst, &opts).unwrap();
        assert_eq!(report.copied, 2, "the norm and the matrix with 60-weight rows are copied");
        assert!((report.avg_bits() - 3.9375).abs() < 1e-9, "{}", report.avg_bits());
        let target = |name: &str| report.choices.iter().find(|c| c.name == name).unwrap().target;
        assert_eq!(target("blk.0.coarse.weight"), Target::Q2_0, "{:?}", report.choices);
        for name in ["blk.0.a.weight", "blk.0.b.weight", "blk.0.c.weight"] {
            assert_eq!(target(name), Target::Q4_0);
        }

        let out = GgufFile::open(&dst).unwrap();
        assert_eq!(out.meta_str("general.name"), Some("requant-test"));
        assert_eq!(out.tensor("blk.0.coarse.weight").unwrap().ggml_type, GgmlType::Q2_0);
        assert_eq!(out.tensor("blk.0.a.weight").unwrap().ggml_type, GgmlType::Q4_0);
        assert_eq!(out.tensor("blk.0.a.weight").unwrap().dims, vec![64, 64]);
        // Copied tensors are bit-identical; requantised ones stay close.
        assert_eq!(out.tensor_f32("blk.0.odd.weight").unwrap(), tensors[5].2);
        assert_eq!(out.tensor_f32("blk.0.norm.weight").unwrap(), tensors[4].2);
        let coarse_out = out.tensor_f32("blk.0.coarse.weight").unwrap();
        let worst = coarse_out.iter().zip(&tensors[0].2).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-3, "three-level weights survive two bits: {worst}");
        let a_out = out.tensor_f32("blk.0.a.weight").unwrap();
        let err: f32 = a_out.iter().zip(&tensors[1].2).map(|(a, b)| (a - b).powi(2)).sum();
        let energy: f32 = tensors[1].2.iter().map(|v| v * v).sum();
        assert!(err / energy < 0.02, "{}", err / energy);
        let _ = std::fs::remove_file(src);
        let _ = std::fs::remove_file(dst);
    }

    fn write_matrices(path: &Path, tensors: &[(&str, Vec<u64>, Vec<f32>)]) {
        let views: Vec<GgufWriteTensor<'_>> = tensors
            .iter()
            .map(|(name, dims, data)| GgufWriteTensor {
                name: (*name).into(),
                dims: dims.clone(),
                data,
            })
            .collect();
        let mut meta = HashMap::new();
        meta.insert("general.architecture".to_string(), GgufValue::String("llama".into()));
        write_gguf(path, &meta, &views).unwrap();
    }

    /// K-quants are options like the others: here three equal matrices may
    /// take 3 or 5 bits and the budget pays for two at 5. The one whose
    /// weights three bits hold exactly gets the 3.
    #[test]
    fn k_quant_options_are_chosen_under_the_budget() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let (src, dst) = (dir.join(format!("allpaca_requant_k_in-{pid}.gguf")), dir.join(format!("allpaca_requant_k_out-{pid}.gguf")));
        let coarse: Vec<f32> = (0..256 * 256).map(|i| ((i * 5 + i / 16) % 8) as f32 * 0.1 - 0.4).collect();
        let tensors = vec![
            ("blk.0.coarse.weight", vec![256, 256], coarse),
            ("blk.0.a.weight", vec![256, 256], noise(256 * 256, 1, 1.0)),
            ("blk.0.b.weight", vec![256, 256], noise(256 * 256, 2, 1.0)),
        ];
        write_matrices(&src, &tensors);
        let opts = Options {
            avg_bits: (3.4375 + 2.0 * 5.5) / 3.0,
            targets: vec![Target::Q3K, Target::Q5K],
            ..Options::default()
        };
        let report = requantise(&src, &dst, &opts).unwrap();
        assert_eq!(report.targets, [Target::Q3K, Target::Q5K]);
        assert!((report.avg_bits() - opts.avg_bits).abs() < 1e-9, "{}", report.avg_bits());
        let out = GgufFile::open(&dst).unwrap();
        for (name, want, ggml) in [
            ("blk.0.coarse.weight", Target::Q3K, GgmlType::Q3_K),
            ("blk.0.a.weight", Target::Q5K, GgmlType::Q5_K),
            ("blk.0.b.weight", Target::Q5K, GgmlType::Q5_K),
        ] {
            let choice = report.choices.iter().find(|c| c.name == name).unwrap();
            assert_eq!(choice.target, want, "{name}: {:?}", choice.relative_error);
            assert_eq!(out.tensor(name).unwrap().ggml_type, ggml);
            assert_eq!(out.tensor(name).unwrap().dims, vec![256, 256]);
        }
        let worst = |name: &str, at: usize| {
            let decoded = out.tensor_f32(name).unwrap();
            decoded.iter().zip(&tensors[at].2).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
        };
        assert!(worst("blk.0.coarse.weight", 0) < 2e-3, "eight-level weights survive three bits: {}", worst("blk.0.coarse.weight", 0));
        let a = out.tensor_f32("blk.0.a.weight").unwrap();
        let err: f32 = a.iter().zip(&tensors[1].2).map(|(a, b)| (a - b).powi(2)).sum();
        let energy: f32 = tensors[1].2.iter().map(|v| v * v).sum();
        assert!(err / energy < 0.003, "five bits: {}", err / energy);
        for path in [src, dst] {
            let _ = std::fs::remove_file(path);
        }
    }

    /// A K-quant block is 256 weights of one row: a matrix with shorter
    /// rows is not offered them, and of two options of the same size the
    /// one with the lower error is taken.
    #[test]
    fn k_quants_need_rows_of_whole_256_weight_blocks() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let (src, dst) =
            (dir.join(format!("allpaca_requant_k256_in-{pid}.gguf")), dir.join(format!("allpaca_requant_k256_out-{pid}.gguf")));
        let tensors = vec![
            ("blk.0.wide.weight", vec![256, 64], noise(256 * 64, 5, 1.0)),
            ("blk.0.narrow.weight", vec![64, 256], noise(256 * 64, 6, 1.0)),
        ];
        write_matrices(&src, &tensors);
        let report = requantise(&src, &dst, &Options { avg_bits: 4.5, ..Options::default() }).unwrap();
        assert_eq!(report.targets, Target::ALL);
        let at = |t: Target| Target::ALL.iter().position(|o| *o == t).unwrap();
        let choice = |name: &str| report.choices.iter().find(|c| c.name == name).unwrap();
        let (wide, narrow) = (choice("blk.0.wide.weight"), choice("blk.0.narrow.weight"));
        assert!(wide.relative_error.iter().all(|e| e.is_finite()), "{:?}", wide.relative_error);
        for t in [Target::Q3K, Target::Q4K, Target::Q5K, Target::Q6K] {
            assert!(narrow.relative_error[at(t)].is_nan(), "{t:?} is not for rows of 64");
        }
        assert_eq!(narrow.target, Target::Q4_0);
        let better = if wide.relative_error[at(Target::Q4K)] < wide.relative_error[at(Target::Q4_0)] { Target::Q4K } else { Target::Q4_0 };
        assert_eq!(wide.target, better, "{:?}", wide.relative_error);
        // More bits, less error, through every option of the wide matrix.
        let by_bits: Vec<f64> =
            [Target::Q2_0, Target::Q3K, Target::Q4K, Target::Q5K, Target::Q6K, Target::Q8_0].iter().map(|t| wide.relative_error[at(*t)]).collect();
        assert!(by_bits.windows(2).all(|w| w[1] < w[0]), "{by_bits:?}");
        for path in [src, dst] {
            let _ = std::fs::remove_file(path);
        }
    }

    /// With measured inputs every covered matrix is quantised for its
    /// output on them: the error there is lower than what quantising
    /// the weights alone gives.
    #[test]
    fn measured_inputs_lower_the_output_error() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let (src, plain, tuned) = (
            dir.join(format!("allpaca_requant_cal_in-{pid}.gguf")),
            dir.join(format!("allpaca_requant_cal_plain-{pid}.gguf")),
            dir.join(format!("allpaca_requant_cal_tuned-{pid}.gguf")),
        );
        crate::calib::tests::write_tiny_calib_gguf(&src, 64, 128);
        let stats = crate::calib::capture(&src, &["hello world again hello world", "again again hello"]).unwrap();
        // 2.25 bits: every matrix goes to Q2_0, so only the quantiser differs.
        let base = Options {
            avg_bits: 2.25,
            gsq: GsqConfig { steps: 80, ..GsqConfig::default() },
            ..Options::default()
        };
        let plain_report = requantise(&src, &plain, &base).unwrap();
        let tuned_report = requantise(&src, &tuned, &Options { stats: Some(stats.clone()), ..base.clone() }).unwrap();
        assert_eq!(plain_report.calibrated, 0);
        assert_eq!(tuned_report.calibrated, 7, "q, k, v, output, gate, up, down");
        assert_eq!(tuned_report.choices.len(), 7);

        let (original, plain_out, tuned_out) =
            (GgufFile::open(&src).unwrap(), GgufFile::open(&plain).unwrap(), GgufFile::open(&tuned).unwrap());
        let (mut plain_total, mut tuned_total) = (0.0f64, 0.0f64);
        for choice in &tuned_report.choices {
            assert_eq!(choice.target, Target::Q2_0);
            let gram = stats.gram(&choice.name).unwrap();
            let h = gram.damped(base.damping);
            let w = original.tensor_f32(&choice.name).unwrap();
            let on_inputs = |file: &GgufFile| {
                output_error(&w, &file.tensor_f32(&choice.name).unwrap(), gram.cols, &Gram::Full(&h))
            };
            let (p, t) = (on_inputs(&plain_out), on_inputs(&tuned_out));
            assert!(t <= p * 1.0001, "{}: {t} on the measured inputs, {p} without them", choice.name);
            plain_total += p;
            tuned_total += t;
        }
        assert!(tuned_total < 0.9 * plain_total, "{tuned_total} vs {plain_total}");
        for path in [src, plain, tuned] {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Quantised in order with propagated calibration, a two-bit model's
    /// predictions on the text stay closer to the original's than with
    /// every matrix quantised alone. Six blocks: with three, each matrix's
    /// own rounding dominates and there is little for the refit to win.
    #[test]
    fn propagated_calibration_keeps_the_predictions_closer() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let at = |tag: &str| dir.join(format!("allpaca_requant_prop_{tag}-{pid}.gguf"));
        let (src, alone, ordered) = (at("in"), at("alone"), at("ordered"));
        crate::calib::tests::write_tiny_calib_gguf_shape(&src, 64, 128, 0.3, 6, false);
        let texts = ["hello world again hello world again again hello", "again again hello world hello"];
        let stats = crate::calib::capture(&src, &texts).unwrap();
        let base = Options {
            avg_bits: 2.25,
            targets: vec![Target::Q2_0],
            gsq: GsqConfig { steps: 60, ..GsqConfig::default() },
            stats: Some(stats),
            ..Options::default()
        };
        requantise(&src, &alone, &base).unwrap();
        let settings = crate::propagate::Settings {
            texts: texts.iter().map(|t| t.to_string()).collect(),
            max_tokens: 32,
            ridge: 0.01,
        };
        let report = requantise(&src, &ordered, &Options { propagate: Some(settings), ..base.clone() }).unwrap();
        assert_eq!(report.propagated, 42, "seven matrices in each of six blocks");
        let kl = |path: &Path| crate::sensitivity::divergence_between(&src, path, &texts, 32).unwrap();
        let (apart, together) = (kl(&alone), kl(&ordered));
        assert!(apart > 0.0 && together < 0.95 * apart, "{together} in order, {apart} alone");
        // The same matrices took the same type either way.
        let (a, b) = (GgufFile::open(&alone).unwrap(), GgufFile::open(&ordered).unwrap());
        for t in &a.tensors {
            assert_eq!(b.tensor(&t.name).unwrap().ggml_type, t.ggml_type, "{}", t.name);
        }
        for path in [src, alone, ordered] {
            let _ = std::fs::remove_file(path);
        }
    }

    /// With a probe the bits follow what the output is sensitive to. Here
    /// the attention output projection is nearly zero, so the four
    /// attention matrices barely reach the output: they take the two
    /// bits and the feed-forward matrices keep four or more. By their
    /// own error the matrices are all alike.
    #[test]
    fn probe_spends_the_bits_on_what_moves_the_output() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let (src, dst) =
            (dir.join(format!("allpaca_requant_probe_in-{pid}.gguf")), dir.join(format!("allpaca_requant_probe_out-{pid}.gguf")));
        crate::calib::tests::write_tiny_calib_gguf_with(&src, 64, 128, 1e-4);
        let opts = Options {
            // Attention (4 * 4096 weights) at 2.25 bits, feed-forward (3 * 8192) at 4.5.
            avg_bits: 3.6,
            gsq: GsqConfig { steps: 40, ..GsqConfig::default() },
            probe: Some(ProbeTexts {
                texts: vec!["hello world again hello world".into(), "again again hello world".into()],
                max_tokens: 32,
                scaled: false,
            }),
            targets: CLASSIC.to_vec(),
            ..Options::default()
        };
        let report = requantise(&src, &dst, &opts).unwrap();
        assert_eq!(report.choices.len(), 7);
        assert!((report.avg_bits() - 3.6).abs() < 1e-9, "{}", report.avg_bits());
        for c in &report.choices {
            let d = c.divergence.clone().unwrap_or_else(|| panic!("{} was not probed", c.name));
            assert!(d.iter().all(|v| v.is_finite() && *v >= 0.0), "{}: {d:?}", c.name);
            if c.name.contains("attn_") {
                assert_eq!(c.target, Target::Q2_0, "{}: {d:?}", c.name);
            } else {
                assert_ne!(c.target, Target::Q2_0, "{}: {d:?}", c.name);
                assert!(d[0] > d[2], "{}: two bits move the output more than eight: {d:?}", c.name);
            }
        }
        let out = GgufFile::open(&dst).unwrap();
        assert_eq!(out.tensor("blk.0.attn_q.weight").unwrap().ggml_type, GgmlType::Q2_0);
        assert_ne!(out.tensor("blk.0.ffn_down.weight").unwrap().ggml_type, GgmlType::Q2_0);
        // The embedding is below the size limit here: copied as it was.
        assert_eq!(out.tensor("token_embd.weight").unwrap().ggml_type, GgmlType::F32);
        for path in [src, dst] {
            let _ = std::fs::remove_file(path);
        }
    }

    /// A scaled probe measures each matrix at its fewest bits only and
    /// scales that by the matrix's own error for the other types: the
    /// same choices as the full probe here, from a third of the reads.
    #[test]
    fn a_scaled_probe_measures_one_type_and_scales_the_rest() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let at = |tag: &str| dir.join(format!("allpaca_requant_scaled_{tag}-{pid}.gguf"));
        let (src, full, scaled) = (at("in"), at("full"), at("scaled"));
        crate::calib::tests::write_tiny_calib_gguf_with(&src, 64, 128, 1e-4);
        let texts = vec!["hello world again hello world".to_string(), "again again hello world".to_string()];
        let base = Options {
            avg_bits: 3.6,
            gsq: GsqConfig { steps: 40, ..GsqConfig::default() },
            targets: CLASSIC.to_vec(),
            probe: Some(ProbeTexts { texts: texts.clone(), max_tokens: 32, scaled: false }),
            ..Options::default()
        };
        let a = requantise(&src, &full, &base).unwrap();
        let b = requantise(
            &src,
            &scaled,
            &Options { probe: Some(ProbeTexts { texts, max_tokens: 32, scaled: true }), ..base.clone() },
        )
        .unwrap();
        for (a, b) in a.choices.iter().zip(&b.choices) {
            let (da, db) = (a.divergence.clone().unwrap(), b.divergence.clone().unwrap());
            // Q2_0 is the fewest bits: measured the same either way.
            assert_eq!(da[0], db[0], "{}", a.name);
            for t in 1..db.len() {
                let want = db[0] * b.relative_error[t] / b.relative_error[0];
                assert!((db[t] - want).abs() <= 1e-12 * want.abs().max(1.0), "{}: {db:?}", b.name);
            }
            assert_eq!(a.target, b.target, "{}: full {da:?}, scaled {db:?}", a.name);
        }
        for path in [src, full, scaled] {
            let _ = std::fs::remove_file(path);
        }
    }

    /// An MoE model: the stacked expert tensors and the shared expert are
    /// probed and given types like the other matrices, and the file reads.
    #[test]
    fn expert_tensors_are_probed_and_requantised() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let at = |tag: &str| dir.join(format!("allpaca_requant_moe_{tag}-{pid}.gguf"));
        let (src, dst) = (at("in"), at("out"));
        crate::calib::tests::write_tiny_calib_gguf_moe(&src, 64, 128, 0.3, 2, false, 4);
        let texts = vec!["hello world again hello world".to_string(), "again again hello world".to_string()];
        let opts = Options {
            avg_bits: 4.0,
            gsq: GsqConfig { steps: 20, ..GsqConfig::default() },
            targets: CLASSIC.to_vec(),
            probe: Some(ProbeTexts { texts: texts.clone(), max_tokens: 32, scaled: true }),
            ..Options::default()
        };
        let report = requantise(&src, &dst, &opts).unwrap();
        let names: Vec<&str> = report.choices.iter().map(|c| c.name.as_str()).collect();
        for name in ["blk.0.ffn_gate_exps.weight", "blk.1.ffn_down_exps.weight", "blk.0.ffn_up_shexp.weight"] {
            let c = report.choices.iter().find(|c| c.name == name).unwrap_or_else(|| panic!("{name} not in {names:?}"));
            let d = c.divergence.clone().unwrap();
            assert!(d.iter().all(|v| v.is_finite() && *v >= 0.0), "{name}: {d:?}");
        }
        let experts = report.choices.iter().find(|c| c.name == "blk.0.ffn_gate_exps.weight").unwrap();
        assert_eq!(experts.weights, 4 * 128 * 64);
        assert!((report.avg_bits() - 4.0).abs() < 0.05, "{}", report.avg_bits());
        let out = GgufFile::open(&dst).unwrap();
        let t = out.tensor("blk.0.ffn_gate_exps.weight").unwrap();
        assert_eq!(t.ggml_type, experts.target.ggml());
        // Every tensor keeps its shape, in the file's own order.
        let input = GgufFile::open(&src).unwrap();
        for t in &input.tensors {
            assert_eq!(out.tensor(&t.name).unwrap().dims, t.dims, "{}", t.name);
        }
        let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
        let kl = crate::sensitivity::divergence_between(&src, &dst, &texts, 32).unwrap();
        assert!(kl.is_finite() && kl > 0.0 && kl < 0.5, "{kl}");
        for path in [src, dst] {
            let _ = std::fs::remove_file(path);
        }
    }

    /// A matrix the probe cannot run (here a block past the model's last,
    /// as a multi-token-prediction layer is) is still requantised, with its
    /// divergence estimated from its own error, not copied as it was.
    #[test]
    fn matrices_the_probe_cannot_run_are_requantised_too() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let at = |tag: &str| dir.join(format!("allpaca_requant_unprobed_{tag}-{pid}.gguf"));
        let (base, src, dst) = (at("base"), at("in"), at("out"));
        crate::calib::tests::write_tiny_calib_gguf_moe(&base, 64, 128, 0.3, 1, false, 4);
        let extra = vec![("blk.1.ffn_up_exps.weight".to_string(), vec![64, 128, 4], noise(64 * 128 * 4, 7, 0.3))];
        crate::sensitivity::tests::rewrite_gguf(&base, &src, |_, _| {}, extra);
        let texts = vec!["hello world again hello world".to_string(), "again again hello world".to_string()];
        let opts = Options {
            avg_bits: 4.0,
            gsq: GsqConfig { steps: 20, ..GsqConfig::default() },
            targets: CLASSIC.to_vec(),
            probe: Some(ProbeTexts { texts, max_tokens: 32, scaled: true }),
            ..Options::default()
        };
        let report = requantise(&src, &dst, &opts).unwrap();
        let name = "blk.1.ffn_up_exps.weight";
        let c = report.choices.iter().find(|c| c.name == name).unwrap_or_else(|| panic!("{name} was copied"));
        let d = c.divergence.clone().unwrap();
        assert!(d.iter().all(|v| v.is_finite() && *v >= 0.0), "{d:?}");
        assert!(d[0] > d[1] && d[1] > d[2], "fewer bits, more divergence: {d:?}");
        let out = GgufFile::open(&dst).unwrap();
        assert_eq!(out.tensor(name).unwrap().ggml_type, c.target.ggml());
        for path in [base, src, dst] {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Given probe texts as well, a table supplies what it holds and the
    /// probe measures only what it lacks, then the table holds both.
    #[test]
    fn a_probe_fills_in_what_its_table_lacks() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let at = |tag: &str| dir.join(format!("allpaca_requant_fill_{tag}-{pid}"));
        let (src, out, table) = (at("in.gguf"), at("out.gguf"), at("table.tsv"));
        let _ = std::fs::remove_file(&table);
        crate::calib::tests::write_tiny_calib_gguf_with(&src, 64, 128, 0.3);
        let opts = Options {
            avg_bits: 4.0,
            gsq: GsqConfig { steps: 20, ..GsqConfig::default() },
            targets: CLASSIC.to_vec(),
            probe: Some(ProbeTexts { texts: vec!["hello world again hello world".into()], max_tokens: 32, scaled: false }),
            probe_table: Some(table.clone()),
            ..Options::default()
        };
        let first = requantise(&src, &out, &opts).unwrap();
        let dropped = "blk.0.ffn_up.weight";
        let full = std::fs::read_to_string(&table).unwrap();
        let rest: String = full.lines().filter(|l| !l.starts_with(dropped)).map(|l| format!("{l}\n")).collect();
        assert!(rest.len() < full.len(), "{dropped} was in the table");
        // A row the probe would not measure the same way: kept as the table has it.
        let kept = "blk.0.ffn_down.weight";
        let rest = rest.replace(&format!("{kept}\tQ2_0\t"), &format!("{kept}\tQ2_0\t9.5e0\n#"));
        let rest: String = rest.lines().filter(|l| !l.starts_with('#')).map(|l| format!("{l}\n")).collect();
        std::fs::write(&table, rest).unwrap();
        let second = requantise(&src, &out, &opts).unwrap();
        let find = |r: &Report, name: &str| r.choices.iter().find(|c| c.name == name).unwrap().divergence.clone().unwrap();
        assert_eq!(find(&second, dropped), find(&first, dropped), "measured again");
        assert_eq!(find(&second, kept)[0], 9.5, "taken from the table");
        let refilled = std::fs::read_to_string(&table).unwrap();
        assert!(refilled.lines().any(|l| l.starts_with(dropped)), "the table holds the new rows");
        assert!(refilled.contains(&format!("{kept}\tQ2_0\t9.5e0")), "and keeps its own");
        for path in [src, out, table] {
            let _ = std::fs::remove_file(path);
        }
    }

    /// A probe's measurements are saved to a table, and a run given the
    /// table takes them from it instead of probing: same types, same file.
    #[test]
    fn a_saved_probe_table_stands_in_for_the_probe() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let at = |tag: &str| dir.join(format!("allpaca_requant_table_{tag}-{pid}"));
        let (src, probed, reused, table) = (at("in.gguf"), at("probed.gguf"), at("reused.gguf"), at("table.tsv"));
        let _ = std::fs::remove_file(&table);
        crate::calib::tests::write_tiny_calib_gguf_with(&src, 64, 128, 1e-4);
        let base = Options {
            avg_bits: 3.6,
            gsq: GsqConfig { steps: 40, ..GsqConfig::default() },
            targets: CLASSIC.to_vec(),
            probe_table: Some(table.clone()),
            ..Options::default()
        };
        let first = requantise(
            &src,
            &probed,
            &Options {
                probe: Some(ProbeTexts { texts: vec!["hello world again hello world".into()], max_tokens: 32, scaled: false }),
                ..base.clone()
            },
        )
        .unwrap();
        assert!(table.exists(), "the probe wrote its table");
        let second = requantise(&src, &reused, &base).unwrap();
        assert_eq!(second.choices.len(), first.choices.len());
        for (a, b) in first.choices.iter().zip(&second.choices) {
            assert_eq!((&a.name, a.target), (&b.name, b.target));
            assert_eq!(a.divergence, b.divergence, "{}", a.name);
        }
        assert_eq!(std::fs::read(&probed).unwrap(), std::fs::read(&reused).unwrap());
        for path in [src, probed, reused, table] {
            let _ = std::fs::remove_file(path);
        }
    }
}
