use serde::{Deserialize, Serialize};

/// Detected hardware profile. Modes are shorthand for [`FeatureFlags`], not monoliths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineMode {
    /// Fast target GPU + slower draft GPU (e.g. RX 9070 + 9060 XT).
    Asymmetric,
    /// Peer GPUs of similar class (e.g. dual MI50).
    Symmetric,
    /// Constrained VRAM / CPU-only; lean on DRAM tiering and n-gram draft.
    MobileTiered,
}

/// Independent, composable behaviors. Pipeline modes only set defaults.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct FeatureFlags {
    pub asymmetric_drafting: bool,
    pub dram_tiering: bool,
    pub ngram_speculation: bool,
    pub draft_model_speculation: bool,
    pub json_constraints: bool,
    pub adaptive_draft_depth: bool,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self::from_mode(PipelineMode::MobileTiered)
    }
}

impl FeatureFlags {
    pub fn from_mode(mode: PipelineMode) -> Self {
        match mode {
            PipelineMode::Asymmetric => Self {
                asymmetric_drafting: true,
                dram_tiering: true,
                ngram_speculation: true,
                draft_model_speculation: true,
                json_constraints: true,
                adaptive_draft_depth: true,
            },
            PipelineMode::Symmetric => Self {
                asymmetric_drafting: false,
                dram_tiering: true,
                ngram_speculation: true,
                draft_model_speculation: true,
                json_constraints: true,
                adaptive_draft_depth: true,
            },
            PipelineMode::MobileTiered => Self {
                asymmetric_drafting: false,
                dram_tiering: true,
                ngram_speculation: true,
                draft_model_speculation: false,
                json_constraints: true,
                adaptive_draft_depth: true,
            },
        }
    }

    /// Config/CLI bits overlay the mode defaults. `None` keeps the mode value.
    pub fn overlay(&mut self, overlay: &FeatureFlagOverlay) {
        if let Some(v) = overlay.asymmetric_drafting {
            self.asymmetric_drafting = v;
        }
        if let Some(v) = overlay.dram_tiering {
            self.dram_tiering = v;
        }
        if let Some(v) = overlay.ngram_speculation {
            self.ngram_speculation = v;
        }
        if let Some(v) = overlay.draft_model_speculation {
            self.draft_model_speculation = v;
        }
        if let Some(v) = overlay.json_constraints {
            self.json_constraints = v;
        }
        if let Some(v) = overlay.adaptive_draft_depth {
            self.adaptive_draft_depth = v;
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureFlagOverlay {
    pub asymmetric_drafting: Option<bool>,
    pub dram_tiering: Option<bool>,
    pub ngram_speculation: Option<bool>,
    pub draft_model_speculation: Option<bool>,
    pub json_constraints: Option<bool>,
    pub adaptive_draft_depth: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_are_shorthand_not_monoliths() {
        let mut flags = FeatureFlags::from_mode(PipelineMode::Asymmetric);
        assert!(flags.asymmetric_drafting);
        flags.overlay(&FeatureFlagOverlay {
            asymmetric_drafting: Some(false),
            ..FeatureFlagOverlay::default()
        });
        assert!(!flags.asymmetric_drafting);
        assert!(flags.dram_tiering);
    }
}
