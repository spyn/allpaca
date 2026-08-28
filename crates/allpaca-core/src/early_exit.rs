//! Model-aware early exit (OPPORTUNITY §9 / SPECS).
//!
//! Only enable when the checkpoint declares exit heads + calibration. Mid-graph
//! abort is not implemented yet — this module gates policy and records telemetry.

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
}
