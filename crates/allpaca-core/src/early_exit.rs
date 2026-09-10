//! Model-aware early exit (OPPORTUNITY §9 / SPECS).
//!
//! Only enable when the checkpoint declares exit heads + calibration. Prefer
//! self-speculative accept (confidence from logits) over a destructive mid-graph
//! abort when correctness is strict.

/// Manifest / runtime early-exit capability (SPECS `EarlyExitSpec`).
#[derive(Debug, Clone, PartialEq)]
pub struct EarlyExitSpec {
    pub supported: bool,
    pub exit_layers: Vec<usize>,
    pub min_confidence: f32,
    pub min_calibration_margin: f32,
    /// Checkpoint calibration scheme id (e.g. `layerskip-v1`); None = legacy / unset.
    pub calibration_version: Option<String>,
    /// When true, runtime prefers verify-before-accept (self-speculate) over bare abort.
    pub requires_verification: bool,
}

impl Default for EarlyExitSpec {
    fn default() -> Self {
        Self {
            supported: false,
            exit_layers: Vec::new(),
            min_confidence: 0.9,
            min_calibration_margin: 0.05,
            calibration_version: None,
            requires_verification: false,
        }
    }
}

/// True when the manifest carried a calibration scheme id (Phase 6 calibration data).
pub fn has_calibration_data(spec: &EarlyExitSpec) -> bool {
    spec.supported
        && spec
            .calibration_version
            .as_ref()
            .is_some_and(|v| !v.is_empty())
}

/// Load early-exit capability from GGUF metadata (`allpaca.early_exit.*`).
///
/// When `supported` is true and `exit_layers` is non-empty, the control plane
/// may self-speculate (shared lm_head) or use dedicated `blk.{i}.exit_head.weight`.
/// Optional `calibration_version` / `requires_verification` are part of calibration data.
pub fn spec_from_gguf(
    file: &allpaca_gguf::GgufFile,
    min_confidence: f32,
    min_calibration_margin: f32,
) -> EarlyExitSpec {
    spec_from_metadata(&file.metadata, min_confidence, min_calibration_margin)
}

/// Same as [`spec_from_gguf`] but from a metadata map (unit-test friendly).
pub fn spec_from_metadata(
    metadata: &std::collections::HashMap<String, allpaca_gguf::GgufValue>,
    min_confidence: f32,
    min_calibration_margin: f32,
) -> EarlyExitSpec {
    let supported = metadata
        .get("allpaca.early_exit.supported")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let exit_layers = metadata
        .get("allpaca.early_exit.exit_layers")
        .and_then(|v| v.as_u32_array())
        .unwrap_or_default()
        .into_iter()
        .map(|v| v as usize)
        .collect::<Vec<_>>();
    if !supported || exit_layers.is_empty() {
        return EarlyExitSpec::default();
    }
    let calibration_version = metadata
        .get("allpaca.early_exit.calibration_version")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty());
    let requires_verification = metadata
        .get("allpaca.early_exit.requires_verification")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    EarlyExitSpec {
        supported: true,
        exit_layers,
        min_confidence,
        min_calibration_margin,
        calibration_version,
        requires_verification,
    }
}

/// Per-layer exit head prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExitPrediction {
    pub confidence: f32,
    pub margin: f32,
}

/// Accept an early exit only when the model declares support and margins hold.
pub fn accept_exit(p: &ExitPrediction, s: &EarlyExitSpec) -> bool {
    s.supported && p.confidence >= s.min_confidence && p.margin >= s.min_calibration_margin
}

/// True when config asks for early exit but the loaded model cannot provide it.
pub fn gate_unsupported(enabled: bool, spec: &EarlyExitSpec) -> bool {
    enabled && !spec.supported
}

/// Softmax top-1 confidence and top-1−top-2 margin from raw logits.
pub fn prediction_from_logits(logits: &[f32]) -> ExitPrediction {
    if logits.is_empty() {
        return ExitPrediction {
            confidence: 0.0,
            margin: 0.0,
        };
    }
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for &v in logits {
        if v > best {
            second = best;
            best = v;
        } else if v > second {
            second = v;
        }
    }
    // Numerically stable 2-class softmax between top-1 and top-2.
    let m = if second.is_finite() {
        best.max(second)
    } else {
        best
    };
    let e1 = (best - m).exp();
    let e2 = if second.is_finite() {
        (second - m).exp()
    } else {
        0.0
    };
    let z = e1 + e2;
    let confidence = if z > 0.0 { e1 / z } else { 0.0 };
    let margin = if second.is_finite() {
        confidence - e2 / z
    } else {
        confidence
    };
    ExitPrediction {
        confidence,
        margin: margin.max(0.0),
    }
}

/// Whether `layer_idx` is an allowed early-exit point for this spec.
pub fn is_exit_layer(spec: &EarlyExitSpec, layer_idx: usize) -> bool {
    spec.supported && spec.exit_layers.iter().any(|&l| l == layer_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_requires_support_and_margins() {
        let mut s = EarlyExitSpec {
            supported: true,
            exit_layers: vec![16, 24],
            min_confidence: 0.9,
            min_calibration_margin: 0.05,
            ..EarlyExitSpec::default()
        };
        let ok = ExitPrediction {
            confidence: 0.95,
            margin: 0.1,
        };
        assert!(accept_exit(&ok, &s));
        s.supported = false;
        assert!(!accept_exit(&ok, &s));
        s.supported = true;
        assert!(!accept_exit(
            &ExitPrediction {
                confidence: 0.5,
                margin: 0.1
            },
            &s
        ));
    }

    #[test]
    fn gate_flags_enabled_without_manifest() {
        assert!(gate_unsupported(true, &EarlyExitSpec::default()));
        assert!(!gate_unsupported(false, &EarlyExitSpec::default()));
        assert!(!gate_unsupported(
            true,
            &EarlyExitSpec {
                supported: true,
                exit_layers: vec![8],
                ..EarlyExitSpec::default()
            }
        ));
    }

    #[test]
    fn prediction_from_logits_prefers_clear_winner() {
        let sharp = prediction_from_logits(&[1.0, 10.0, 0.5]);
        let flat = prediction_from_logits(&[1.0, 1.1, 1.05]);
        assert!(sharp.confidence > flat.confidence);
        assert!(sharp.margin > flat.margin);
        assert!(is_exit_layer(
            &EarlyExitSpec {
                supported: true,
                exit_layers: vec![1],
                ..EarlyExitSpec::default()
            },
            1
        ));
    }

    #[test]
    fn spec_from_metadata_loads_calibration_data() {
        use allpaca_gguf::GgufValue;
        use std::collections::HashMap;

        let mut meta = HashMap::new();
        meta.insert("allpaca.early_exit.supported".into(), GgufValue::Bool(true));
        meta.insert(
            "allpaca.early_exit.exit_layers".into(),
            GgufValue::Array(vec![GgufValue::Uint32(16), GgufValue::Uint32(24)]),
        );
        meta.insert(
            "allpaca.early_exit.calibration_version".into(),
            GgufValue::String("layerskip-v1".into()),
        );
        meta.insert(
            "allpaca.early_exit.requires_verification".into(),
            GgufValue::Bool(true),
        );
        let s = spec_from_metadata(&meta, 0.9, 0.05);
        assert!(s.supported);
        assert_eq!(s.exit_layers, vec![16, 24]);
        assert_eq!(s.calibration_version.as_deref(), Some("layerskip-v1"));
        assert!(s.requires_verification);
        assert!(has_calibration_data(&s));
    }

    #[test]
    fn unsupported_manifest_has_no_calibration_data() {
        assert!(!has_calibration_data(&EarlyExitSpec::default()));
    }
}
