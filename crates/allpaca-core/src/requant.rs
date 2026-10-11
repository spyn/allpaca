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
//! independent.

use std::path::Path;

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
    Q4_0,
    Q8_0,
}

impl Target {
    pub const ALL: [Target; 3] = [Target::Q2_0, Target::Q4_0, Target::Q8_0];

    pub fn bits(self) -> f64 {
        match self {
            Target::Q2_0 => 2.25,
            Target::Q4_0 => 4.5,
            Target::Q8_0 => 8.5,
        }
    }

    fn ggml(self) -> GgmlType {
        match self {
            Target::Q2_0 => GgmlType::Q2_0,
            Target::Q4_0 => GgmlType::Q4_0,
            Target::Q8_0 => GgmlType::Q8_0,
        }
    }

    fn quantise(
        self,
        w: &[f32],
        rows: usize,
        cols: usize,
        gram: &Gram<'_>,
        gsq: &GsqConfig,
    ) -> Result<Vec<u8>, AllpacaError> {
        Ok(match self {
            Target::Q2_0 => quantise_2bit(w, rows, cols, gram, gsq)?.to_q2_0(),
            Target::Q4_0 => allpaca_gguf::quantize_q4_0(w),
            Target::Q8_0 => allpaca_gguf::quantize_q8_0(w),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    /// Average bits per weight over the requantised matrices.
    pub avg_bits: f64,
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
    /// instead of each matrix's own error: only matrices the probe can
    /// swap are requantised, the rest are copied.
    pub probe: Option<ProbeTexts>,
    /// Called after each matrix is measured: matrices done, of how many, its name.
    pub progress: Option<fn(usize, usize, &str)>,
}

/// Calibration texts for the sensitivity probe.
#[derive(Debug, Clone)]
pub struct ProbeTexts {
    pub texts: Vec<String>,
    /// Tokens read of each text.
    pub max_tokens: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            avg_bits: 3.0,
            min_weights: 4096,
            gsq: GsqConfig::default(),
            resolution: 4000,
            stats: None,
            damping: 0.01,
            probe: None,
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
    /// Error of each of [`Target::ALL`] relative to the matrix's energy:
    /// of its output on the measured inputs when `calibrated`, of its
    /// weights otherwise.
    pub relative_error: [f64; 3],
    pub calibrated: bool,
    /// With a probe: divergence of the model's predictions (nats per
    /// token) with this matrix alone at each of [`Target::ALL`].
    pub divergence: Option<[f64; 3]>,
}

#[derive(Debug, Clone)]
pub struct Report {
    pub choices: Vec<Choice>,
    /// Tensors copied unchanged.
    pub copied: usize,
    /// Requantised matrices that used measured inputs.
    pub calibrated: usize,
}

impl Report {
    /// Average bits per weight actually reached over the requantised matrices.
    pub fn avg_bits(&self) -> f64 {
        let total: f64 = self.choices.iter().map(|c| c.weights as f64).sum();
        self.choices.iter().map(|c| c.weights as f64 * c.target.bits()).sum::<f64>() / total.max(1.0)
    }
}

/// Rows and row length of a matrix whose rows are whole 64-weight blocks
/// (ggml lists the row length first), or `None` for anything else.
fn matrix_shape(dims: &[u64], min_weights: usize) -> Option<(usize, usize)> {
    let cols = *dims.first()? as usize;
    let n: usize = dims.iter().product::<u64>() as usize;
    (dims.len() >= 2 && cols % 64 == 0 && n >= min_weights).then(|| (n / cols, cols))
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
        .filter(|(_, t)| probe.as_ref().is_none_or(|p| p.covers(&t.name)))
        .filter_map(|(i, t)| matrix_shape(&t.dims, opts.min_weights).map(|(rows, cols)| (i, rows, cols)))
        .collect();
    if candidates.is_empty() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "requantise: the file has no weight matrix with rows of whole 64-weight blocks".into(),
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
    for &(i, rows, cols) in &candidates {
        let name = &file.tensors[i].name;
        let w = file.tensor_f32(name)?;
        let h = measured(i, cols);
        let gram = h.as_deref().map_or(Gram::Identity, Gram::Full);
        let energy = output_error(&w, &vec![0.0; w.len()], cols, &gram);
        let mut row = [0.0f64; 3];
        let mut moved = [0.0f64; 3];
        let mut decoded = vec![0.0f32; w.len()];
        for (at, target) in Target::ALL.into_iter().enumerate() {
            let bytes = target.quantise(&w, rows, cols, &gram, &opts.gsq)?;
            allpaca_gguf::dequant_packed(target.ggml(), &bytes, w.len(), &mut decoded)?;
            row[at] = output_error(&w, &decoded, cols, &gram) / energy.max(1e-30);
            if let Some(probe) = probe.as_mut() {
                moved[at] = probe.divergence(name, &decoded)?;
            }
            if target == Target::Q2_0 {
                two_bit.push(bytes);
            }
        }
        errors.push(row);
        calibrated.push(h.is_some());
        divergence.push(moved);
        if let Some(progress) = opts.progress {
            progress(errors.len(), candidates.len(), name);
        }
    }
    // The probe holds the whole model in F32: not needed past here.
    let probed = probe.take().is_some();
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
    let gain: Vec<f64> = loss.iter().map(|l| -l).collect();
    let mut reports = Vec::with_capacity(outputs.len());
    for &(output, avg_bits) in outputs {
        let problem = Problem {
            costs: Target::ALL.iter().map(|t| t.bits()).collect(),
            weights: weights.clone(),
            budget: avg_bits,
        };
        // The loss is one term per matrix and option, so the knapsack optimum is exact.
        let assignment = knapsack(&problem, &gain, opts.resolution)?;

        let chosen = |i: usize| candidates.iter().position(|c| c.0 == i).map(|at| Target::ALL[assignment[at]]);
        let declared: Vec<GgufStreamTensor> = file
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let n = t.n_elements() as usize;
                let (ggml_type_id, nbytes) = match chosen(i) {
                    Some(Target::Q2_0) => (GgmlType::Q2_0 as u32, n / 64 * 18),
                    Some(Target::Q4_0) => (GgmlType::Q4_0 as u32, n / 32 * 18),
                    Some(Target::Q8_0) => (GgmlType::Q8_0 as u32, n / 32 * 34),
                    None => (t.ggml_type_id, t.nbytes()),
                };
                GgufStreamTensor {
                    name: t.name.clone(),
                    // The parser holds ggml order (innermost first); the writers take the reverse.
                    dims: t.dims.iter().rev().copied().collect(),
                    ggml_type_id,
                    nbytes,
                }
            })
            .collect();
        let mut metadata = file.metadata.clone();
        metadata.insert(
            "general.description".into(),
            GgufValue::String(format!(
                "requantised by Allpaca (GSQ 2-bit + RCO allocation) to {:.2} bits per weight on average",
                avg_bits
            )),
        );
        for split in ["split.no", "split.count", "split.tensors.count"] {
            metadata.remove(split);
        }
        let mut writer = GgufStreamWriter::create(output, &metadata, &declared)?;
        for (i, t) in file.tensors.iter().enumerate() {
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
                    target: Target::ALL[o],
                    relative_error: *row,
                    calibrated,
                    divergence: probed.then_some(*moved),
                })
                .collect(),
            copied: file.tensors.len() - candidates.len(),
            calibrated: calibrated.iter().filter(|c| **c).count(),
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
            costs: Target::ALL.iter().map(|t| t.bits()).collect(),
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
            }),
            ..Options::default()
        };
        let report = requantise(&src, &dst, &opts).unwrap();
        assert_eq!(report.choices.len(), 7);
        assert!((report.avg_bits() - 3.6).abs() < 1e-9, "{}", report.avg_bits());
        for c in &report.choices {
            let d = c.divergence.unwrap_or_else(|| panic!("{} was not probed", c.name));
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
}
