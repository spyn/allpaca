//! Recover-LoRA runtime overlay (unmerged A/B deltas).

use std::collections::HashMap;
use std::path::Path;

use allpaca_safetensors::SafetensorsFile;
use allpaca_types::{AllpacaError, FatalError};

/// One LoRA pair attached to a named linear.
#[derive(Debug, Clone)]
pub struct LoraPair {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    /// `a` is `[r, in]`, `b` is `[out, r]` (row-major NT gemm style used elsewhere).
    pub rank: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub scale: f32,
}

impl LoraPair {
    /// `y += scale * ((x @ A^T) @ B^T)` with A `[r,in]`, B `[out,r]`.
    pub fn apply(&self, x: &[f32], y: &mut [f32]) {
        if self.rank == 0 || x.len() < self.in_features || y.len() < self.out_features {
            return;
        }
        let mut mid = vec![0.0f32; self.rank];
        for r in 0..self.rank {
            let row = &self.a[r * self.in_features..(r + 1) * self.in_features];
            let mut s = 0.0f32;
            for i in 0..self.in_features {
                s += x[i] * row[i];
            }
            mid[r] = s;
        }
        for o in 0..self.out_features {
            let row = &self.b[o * self.rank..(o + 1) * self.rank];
            let mut s = 0.0f32;
            for r in 0..self.rank {
                s += mid[r] * row[r];
            }
            y[o] += self.scale * s;
        }
    }
}

/// Recover-LoRA bank loaded from `lora_edge0_*.safetensors`.
#[derive(Debug, Clone, Default)]
pub struct RecoverLora {
    pub pairs: HashMap<String, LoraPair>,
    pub rank: usize,
    pub alpha: f32,
}

impl RecoverLora {
    pub fn scale(&self) -> f32 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f32
        }
    }

    /// Load adapter file. Keys ending in `.lora_A.weight` / `.lora_B.weight` (or `.lora_A` / `.lora_B`).
    pub fn load(path: &Path, rank: usize, alpha: f32) -> Result<Self, AllpacaError> {
        if !path.is_file() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "Edge0 LoRA missing: {}",
                path.display()
            ))));
        }
        let file = SafetensorsFile::open(path)?;
        let scale = if rank == 0 {
            0.0
        } else {
            alpha / rank as f32
        };
        let mut bases: HashMap<String, (Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<usize>>, Option<Vec<usize>>)> =
            HashMap::new();
        for (name, info) in &file.tensors {
            let (base, which) = if let Some(b) = name.strip_suffix(".lora_A.weight") {
                (b.to_string(), 'A')
            } else if let Some(b) = name.strip_suffix(".lora_B.weight") {
                (b.to_string(), 'B')
            } else if let Some(b) = name.strip_suffix(".lora_A") {
                (b.to_string(), 'A')
            } else if let Some(b) = name.strip_suffix(".lora_B") {
                (b.to_string(), 'B')
            } else {
                continue;
            };
            let data = file.tensor_f32(name)?;
            let entry = bases.entry(base).or_default();
            match which {
                'A' => {
                    entry.0 = Some(data);
                    entry.2 = Some(info.shape.clone());
                }
                'B' => {
                    entry.1 = Some(data);
                    entry.3 = Some(info.shape.clone());
                }
                _ => {}
            }
        }
        let mut pairs = HashMap::new();
        for (base, (a, b, a_shape, b_shape)) in bases {
            let (Some(a), Some(b), Some(a_shape), Some(b_shape)) = (a, b, a_shape, b_shape) else {
                continue;
            };
            // A: [r, in] or [in, r]; B: [out, r] or [r, out]. Prefer [r,in] / [out,r].
            let (rank_a, in_f) = if a_shape.len() == 2 {
                if a_shape[0] == rank || a_shape[0] < a_shape[1] {
                    (a_shape[0], a_shape[1])
                } else {
                    (a_shape[1], a_shape[0])
                }
            } else {
                continue;
            };
            let (out_f, rank_b) = if b_shape.len() == 2 {
                if b_shape[1] == rank_a || b_shape[1] < b_shape[0] {
                    (b_shape[0], b_shape[1])
                } else {
                    (b_shape[1], b_shape[0])
                }
            } else {
                continue;
            };
            if rank_a != rank_b {
                continue;
            }
            pairs.insert(
                base,
                LoraPair {
                    a,
                    b,
                    rank: rank_a,
                    in_features: in_f,
                    out_features: out_f,
                    scale,
                },
            );
        }
        Ok(Self {
            pairs,
            rank,
            alpha,
        })
    }

    pub fn apply_named(&self, name: &str, x: &[f32], y: &mut [f32]) {
        if let Some(p) = self.pairs.get(name) {
            p.apply(x, y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use allpaca_safetensors::{write_safetensors, SafetensorsDtype, SafetensorsWriteTensor};

    #[test]
    fn lora_apply_adds_scaled_delta() {
        let pair = LoraPair {
            a: vec![1.0, 0.0], // r=1, in=2
            b: vec![2.0],      // out=1, r=1
            rank: 1,
            in_features: 2,
            out_features: 1,
            scale: 0.5, // alpha/r with alpha=0.5 r=1
        };
        let x = [1.0f32, 0.0];
        let mut y = [0.0f32];
        pair.apply(&x, &mut y);
        // mid = 1; y += 0.5 * 1 * 2 = 1.0
        assert!((y[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn load_empty_file_errors() {
        let path = std::env::temp_dir().join("allpaca_missing_lora.safetensors");
        let _ = std::fs::remove_file(&path);
        assert!(RecoverLora::load(&path, 16, 32.0).is_err());
    }

    #[test]
    fn load_roundtrip_pair() {
        let path = std::env::temp_dir().join("allpaca_lora_test.safetensors");
        let a = [1.0f32, 0.0];
        let b = [2.0f32];
        write_safetensors(
            &path,
            &[
                SafetensorsWriteTensor {
                    name: "layers.0.self_attn.q_proj.lora_A.weight".into(),
                    dims: vec![1, 2],
                    dtype: SafetensorsDtype::F32,
                    data: &a,
                },
                SafetensorsWriteTensor {
                    name: "layers.0.self_attn.q_proj.lora_B.weight".into(),
                    dims: vec![1, 1],
                    dtype: SafetensorsDtype::F32,
                    data: &b,
                },
            ],
        )
        .unwrap();
        let lora = RecoverLora::load(&path, 1, 0.5).unwrap();
        assert_eq!(lora.pairs.len(), 1);
        let mut y = [0.0f32];
        lora.apply_named("layers.0.self_attn.q_proj", &[1.0, 0.0], &mut y);
        assert!((y[0] - 1.0).abs() < 1e-5);
        let _ = std::fs::remove_file(path);
    }
}
