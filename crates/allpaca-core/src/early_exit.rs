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
}

impl Default for EarlyExitSpec {
    fn default() -> Self {
        Self {
            supported: false,
            exit_layers: Vec::new(),
            min_confidence: 0.9,
            min_calibration_margin: 0.05,
        }
    }
}

/// Per-layer exit head prediction (future wire-up).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExitPrediction {
    pub confidence: f32,
    pub margin: f32,
}

/// Accept an early exit only when the model declares support and margins hold.
pub fn accept_exit(p: &ExitPrediction, s: &EarlyExitSpec) -> bool {
    s.supported
        && p.confidence >= s.min_confidence
        && p.margin >= s.min_calibration_margin
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
}
