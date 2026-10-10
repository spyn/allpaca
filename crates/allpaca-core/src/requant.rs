//! Requantise a GGUF to a mixed-precision file under an average-bits budget.
//!
//! Each weight matrix can go to one of a few types. GSQ ([`crate::gsq`])
//! produces the 2-bit option; the budgeted choice of one type per matrix
//! is RCO's ([`crate::rco`]) knapsack, which meets the average-bits budget
//! at the least total error. Everything else is copied unchanged.
//!
//! The error is each matrix's own squared error relative to its energy,
//! weighted by its size: there is no calibration data here, so inputs count
//! as uncorrelated and matrices as independent.

use std::path::Path;

use allpaca_gguf::{GgmlType, GgufFile, GgufStreamTensor, GgufStreamWriter, GgufValue};
use allpaca_types::{AllpacaError, FatalError};

use crate::gsq::{output_error, quantise_2bit, Gram, GsqConfig};
use crate::rco::{knapsack, Problem};

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

    fn quantise(self, w: &[f32], rows: usize, cols: usize, gsq: &GsqConfig) -> Result<Vec<u8>, AllpacaError> {
        Ok(match self {
            Target::Q2_0 => quantise_2bit(w, rows, cols, &Gram::Identity, gsq)?.to_q2_0(),
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
}

impl Default for Options {
    fn default() -> Self {
        Self {
            avg_bits: 3.0,
            min_weights: 4096,
            gsq: GsqConfig::default(),
            resolution: 4000,
        }
    }
}

/// What was done to one requantised matrix.
#[derive(Debug, Clone)]
pub struct Choice {
    pub name: String,
    pub weights: usize,
    pub target: Target,
    /// Squared error of each of [`Target::ALL`] relative to the matrix's energy.
    pub relative_error: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct Report {
    pub choices: Vec<Choice>,
    /// Tensors copied unchanged.
    pub copied: usize,
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

fn relative_error(w: &[f32], bytes: &[u8], ty: GgmlType, cols: usize) -> Result<f64, AllpacaError> {
    let mut decoded = vec![0.0f32; w.len()];
    allpaca_gguf::dequant_packed(ty, bytes, w.len(), &mut decoded)?;
    let energy: f64 = w.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    Ok(output_error(w, &decoded, cols, &Gram::Identity) / energy.max(1e-30))
}

/// Requantise `input` into `output`.
pub fn requantise(input: &Path, output: &Path, opts: &Options) -> Result<Report, AllpacaError> {
    let file = GgufFile::open(input)?;
    let candidates: Vec<(usize, usize, usize)> = file
        .tensors
        .iter()
        .enumerate()
        .filter(|(_, t)| t.name.ends_with(".weight") && !matches!(t.ggml_type, GgmlType::I32 | GgmlType::Other))
        .filter_map(|(i, t)| matrix_shape(&t.dims, opts.min_weights).map(|(rows, cols)| (i, rows, cols)))
        .collect();
    if candidates.is_empty() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "requantise: the file has no weight matrix with rows of whole 64-weight blocks".into(),
        )));
    }
    // Error of every option for every matrix.
    let mut errors = Vec::with_capacity(candidates.len());
    for &(i, rows, cols) in &candidates {
        let w = file.tensor_f32(&file.tensors[i].name)?;
        let mut row = [0.0f64; 3];
        for (slot, target) in row.iter_mut().zip(Target::ALL) {
            *slot = relative_error(&w, &target.quantise(&w, rows, cols, &opts.gsq)?, target.ggml(), cols)?;
        }
        errors.push(row);
    }
    let problem = Problem {
        costs: Target::ALL.iter().map(|t| t.bits()).collect(),
        weights: candidates.iter().map(|&(_, rows, cols)| (rows * cols) as f64).collect(),
        budget: opts.avg_bits,
    };
    // The loss of a matrix's option is its error weighted by its size.
    let mean_weight = problem.weights.iter().sum::<f64>() / candidates.len() as f64;
    let loss: Vec<f64> = errors
        .iter()
        .zip(&problem.weights)
        .flat_map(|(row, w)| row.iter().map(move |e| e * w / mean_weight))
        .collect();
    // The loss is one term per matrix and option, so the knapsack optimum is exact.
    let gain: Vec<f64> = loss.iter().map(|l| -l).collect();
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
            opts.avg_bits
        )),
    );
    for split in ["split.no", "split.count", "split.tensors.count"] {
        metadata.remove(split);
    }
    let mut writer = GgufStreamWriter::create(output, &metadata, &declared)?;
    for (i, t) in file.tensors.iter().enumerate() {
        match chosen(i) {
            Some(target) => {
                let (_, rows, cols) = candidates[candidates.iter().position(|c| c.0 == i).expect("chosen")];
                let w = file.tensor_f32(&t.name)?;
                writer.write_tensor(&target.quantise(&w, rows, cols, &opts.gsq)?)?;
            }
            None => writer.write_tensor(file.tensor_bytes(&t.name)?)?,
        }
    }
    writer.finish()?;
    Ok(Report {
        choices: candidates
            .iter()
            .zip(&assignment)
            .zip(&errors)
            .map(|((&(i, rows, cols), &o), row)| Choice {
                name: file.tensors[i].name.clone(),
                weights: rows * cols,
                target: Target::ALL[o],
                relative_error: *row,
            })
            .collect(),
        copied: file.tensors.len() - candidates.len(),
    })
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
}
