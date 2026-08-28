use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::device::DeviceSelector;
use crate::error::{AllpacaError, FatalError};
use crate::pipeline::{FeatureFlagOverlay, PipelineMode};

/// Compiled defaults < TOML file < CLI flags.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub engine: EngineSection,
    pub speculative: SpeculativeSection,
    pub prefill: PrefillSection,
    pub kv: KvSection,
    pub weights: WeightsSection,
    pub anchors: AnchorsSection,
    pub early_exit: EarlyExitSection,
    pub sampling: SamplingSection,
    pub structured_output: StructuredOutputSection,
    pub model: ModelSection,
    pub http: HttpSection,
    pub features: FeatureFlagOverlay,
}

impl Default for Config {
    fn default() -> Self {
        Self::compiled_defaults()
    }
}

impl Config {
    pub fn compiled_defaults() -> Self {
        Self {
            engine: EngineSection::default(),
            speculative: SpeculativeSection::default(),
            prefill: PrefillSection::default(),
            kv: KvSection::default(),
            weights: WeightsSection::default(),
            anchors: AnchorsSection::default(),
            early_exit: EarlyExitSection::default(),
            sampling: SamplingSection::default(),
            structured_output: StructuredOutputSection::default(),
            model: ModelSection::default(),
            http: HttpSection::default(),
            features: FeatureFlagOverlay::default(),
        }
    }

    pub fn from_toml_str(src: &str) -> Result<Self, AllpacaError> {
        let parsed: Config = toml::from_str(src).map_err(|e| {
            AllpacaError::Fatal(FatalError::InvalidConfig(format!("TOML parse: {e}")))
        })?;
        parsed.validate()?;
        Ok(parsed)
    }

    pub fn from_toml_file(path: &Path) -> Result<Self, AllpacaError> {
        let src = std::fs::read_to_string(path).map_err(|e| {
            AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "read {}: {e}",
                path.display()
            )))
        })?;
        Self::from_toml_str(&src)
    }

    pub fn merge_cli(&mut self, cli: &CliOverrides) {
        if let Some(path) = &cli.model {
            self.model.path = Some(path.clone());
        }
        if let Some(path) = &cli.draft_model {
            self.model.draft_path = Some(path.clone());
        }
        if let Some(dev) = &cli.target_device {
            self.engine.target_device = *dev;
        }
        if let Some(dev) = &cli.draft_device {
            self.engine.draft_device = *dev;
        }
        if let Some(threads) = cli.cpu_threads {
            self.engine.cpu_threads = threads;
        }
        if let Some(mode) = cli.pipeline_mode {
            self.engine.pipeline_mode = Some(mode);
        }
        if let Some(seed) = cli.seed {
            self.engine.determinism_seed = Some(seed);
        }
        if cli.cpu_only {
            self.engine.target_device = DeviceSelector::Cpu;
            self.engine.draft_device = DeviceSelector::Cpu;
            self.engine.cpu_only = true;
        }
        if let Some(enabled) = cli.speculative {
            self.speculative.enabled = enabled;
        }
        if let Some(host) = &cli.host {
            self.http.host = host.clone();
        }
        if let Some(port) = cli.port {
            self.http.port = port;
        }
        if cli.synthetic {
            self.model.synthetic = true;
        }
        if let Some(json) = cli.json_mode {
            self.structured_output.json_mode = json;
        }
    }

    pub fn validate(&self) -> Result<(), AllpacaError> {
        if self.kv.page_tokens == 0 {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "kv.page_tokens must be > 0".into(),
            )));
        }
        if !(0.0..=1.0).contains(&self.kv.hot_fraction) {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "kv.hot_fraction must be in [0, 1]".into(),
            )));
        }
        if self.speculative.max_depth < self.speculative.min_depth {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "speculative.max_depth < min_depth".into(),
            )));
        }
        if self.engine.max_active_requests == 0 {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "engine.max_active_requests must be > 0".into(),
            )));
        }
        if self.prefill.chunk_tokens == 0 {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "prefill.chunk_tokens must be > 0".into(),
            )));
        }
        if self.weights.ring_slots > 2 {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "weights.ring_slots must be 0..=2".into(),
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PrefillSection {
    /// Prompt tokens processed per Engine lock hold. A second Request can
    /// admit or decode between chunks when `engine.max_active_requests >= 2`.
    pub chunk_tokens: usize,
}

impl Default for PrefillSection {
    fn default() -> Self {
        Self { chunk_tokens: 512 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineSection {
    pub scheduler: String,
    pub target_device: DeviceSelector,
    pub draft_device: DeviceSelector,
    pub cpu_threads: usize,
    pub cpu_only: bool,
    /// `None` means auto-detect from the device inventory.
    pub pipeline_mode: Option<PipelineMode>,
    pub determinism_seed: Option<u64>,
    /// Concurrent Requests the Engine will admit. Default 1 keeps
    /// `RequestBusy` when a second `begin_request` arrives.
    pub max_active_requests: usize,
    /// How `Engine::poll` chooses among runnable Requests.
    pub poll_schedule: PollSchedule,
    /// When `poll_schedule = prefill_budget`, max consecutive prefill picks before a
    /// decode Request must be served (0 = unlimited). Ignored for other schedules.
    pub max_consecutive_prefill_picks: usize,
    /// `"auto"` (HIP when the `hip` feature sees devices), `"hip"`, or `"cpu"`.
    pub gpu_backend: String,
}

/// Continuous-batching poll selection policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PollSchedule {
    /// Longest-waiting Request wins (prefill and decode treated equally).
    #[default]
    Fair,
    /// Prefer decode/emitted Requests over in-flight prefill when both are runnable.
    DecodePriority,
    /// Fair among decode; allow at most `max_consecutive_prefill_picks` prefills in a row
    /// when a decode Request is also runnable (0 = unlimited, same as Fair).
    PrefillBudget,
}

impl Default for EngineSection {
    fn default() -> Self {
        Self {
            scheduler: "adaptive".into(),
            target_device: DeviceSelector::Auto,
            draft_device: DeviceSelector::Auto,
            cpu_threads: 4,
            cpu_only: false,
            pipeline_mode: None,
            determinism_seed: None,
            max_active_requests: 1,
            poll_schedule: PollSchedule::Fair,
            max_consecutive_prefill_picks: 0,
            gpu_backend: "auto".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DraftProviderKind {
    Ngram,
    DraftModel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TreeSection {
    pub enabled: bool,
    pub max_nodes: usize,
}

impl Default for TreeSection {
    fn default() -> Self {
        Self {
            enabled: false,
            max_nodes: 16,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpeculativeSection {
    pub enabled: bool,
    pub providers: Vec<DraftProviderKind>,
    pub max_depth: usize,
    pub min_depth: usize,
    pub adaptive: bool,
    pub ngram_n: usize,
    pub tree: TreeSection,
}

impl Default for SpeculativeSection {
    fn default() -> Self {
        Self {
            enabled: true,
            providers: vec![DraftProviderKind::Ngram, DraftProviderKind::DraftModel],
            max_depth: 8,
            min_depth: 2,
            adaptive: true,
            ngram_n: 3,
            tree: TreeSection::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KvSection {
    pub page_tokens: u32,
    pub hot_fraction: f32,
    /// Simulated VRAM budget on the CPU backend (bytes of f32 KV storage).
    pub vram_bytes: usize,
    pub dram_bytes: usize,
    /// Cold KV swap file budget. 0 = disabled (VRAM ↔ DRAM only).
    pub nvme_bytes: usize,
    /// Swap file path. None = a process-local temp file removed on drop.
    pub nvme_path: Option<PathBuf>,
}

impl Default for KvSection {
    fn default() -> Self {
        Self {
            page_tokens: 128,
            hot_fraction: 0.75,
            vram_bytes: 4 * 1024 * 1024,
            dram_bytes: 32 * 1024 * 1024,
            nvme_bytes: 0,
            nvme_path: None,
        }
    }
}

/// How residual MoE expert misses are served (FreeToken `q*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissPolicy {
    Qstar,
    Fill,
    Cpu,
}

impl Default for MissPolicy {
    fn default() -> Self {
        Self::Qstar
    }
}

/// Dense pin-prefix + ring, and (later) MoE expert LRU. `trunk_bytes == 0` is full resident.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WeightsSection {
    /// Budget for pinned FFN prefix plus ring slots. 0 = unlimited (current HIP contract).
    pub trunk_bytes: usize,
    /// GPU/hot expert-LRU budget. 0 = no expert slots (host pool only).
    pub cache_bytes: usize,
    /// Requested ring slots (1 or 2). 0 means 1. Second slot only if it fits `trunk_bytes`.
    pub ring_slots: usize,
    pub miss_policy: MissPolicy,
    /// DRAM budget for the expert host pool. 0 = unlimited (experts stay in RAM).
    /// When the pool exceeds this, experts pack to an aligned file and page in via Direct I/O.
    pub host_bytes: usize,
}

impl Default for WeightsSection {
    fn default() -> Self {
        Self {
            trunk_bytes: 0,
            cache_bytes: 0,
            ring_slots: 2,
            miss_policy: MissPolicy::Qstar,
            host_bytes: 0,
        }
    }
}

/// Recurrent-state checkpoints at special-token boundaries. Independent of KV.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnchorsSection {
    /// LRU slots for recurrent snapshots. 0 = disabled.
    pub slots: usize,
    /// Token ids treated as thinking / tool / turn boundaries.
    pub token_ids: Vec<u32>,
}

impl Default for AnchorsSection {
    fn default() -> Self {
        Self {
            slots: 0,
            token_ids: Vec::new(),
        }
    }
}

/// Model-aware early exit (SPECS). Off until a checkpoint declares exit heads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EarlyExitSection {
    /// Request early-exit policy. Gated when the model lacks a valid manifest.
    pub enabled: bool,
    pub min_confidence: f32,
    pub min_calibration_margin: f32,
}

impl Default for EarlyExitSection {
    fn default() -> Self {
        Self {
            enabled: false,
            min_confidence: 0.9,
            min_calibration_margin: 0.05,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingSection {
    pub mode: String,
    pub pinned_ring_slots: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplingSection {
    fn default() -> Self {
        Self {
            mode: "async_host".into(),
            pinned_ring_slots: 8,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct StructuredOutputSection {
    pub enabled: bool,
    pub json_mode: bool,
    /// Prefetch the next grammar mask on a background thread (`async` in TOML).
    #[serde(default, rename = "async")]
    pub async_masks: bool,
}

impl Default for StructuredOutputSection {
    fn default() -> Self {
        Self {
            enabled: true,
            json_mode: false,
            async_masks: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelSection {
    pub path: Option<PathBuf>,
    pub draft_path: Option<PathBuf>,
    /// HuggingFace-style `config.json` sidecar (SafeTensors / ONNX).
    pub config_path: Option<PathBuf>,
    /// Tokenizer sidecar JSON (vocab or HuggingFace `tokenizer.json`).
    pub tokenizer_path: Option<PathBuf>,
    pub synthetic: bool,
    pub id: String,
}

impl Default for ModelSection {
    fn default() -> Self {
        Self {
            path: None,
            draft_path: None,
            config_path: None,
            tokenizer_path: None,
            synthetic: false,
            id: "allpaca".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HttpSection {
    pub host: String,
    pub port: u16,
}

impl Default for HttpSection {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8080,
        }
    }
}

/// CLI flags that overlay a file (or compiled defaults).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CliOverrides {
    pub model: Option<PathBuf>,
    pub draft_model: Option<PathBuf>,
    pub target_device: Option<DeviceSelector>,
    pub draft_device: Option<DeviceSelector>,
    pub cpu_threads: Option<usize>,
    pub pipeline_mode: Option<PipelineMode>,
    pub seed: Option<u64>,
    pub cpu_only: bool,
    pub speculative: Option<bool>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub synthetic: bool,
    pub json_mode: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_defaults_then_toml_then_cli() {
        let defaults = Config::compiled_defaults();
        assert_eq!(defaults.http.port, 8080);
        assert_eq!(defaults.engine.cpu_threads, 4);
        assert!(defaults.speculative.enabled);
        assert!(!defaults.engine.cpu_only);

        let toml = r#"
            [engine]
            cpu_threads = 6
            [http]
            port = 9000
        "#;
        let mut cfg = Config::from_toml_str(toml).unwrap();
        assert_eq!(cfg.engine.cpu_threads, 6, "TOML overrides compiled default");
        assert_eq!(cfg.http.port, 9000, "TOML overrides compiled default");
        assert!(
            cfg.speculative.enabled,
            "unspecified TOML field keeps compiled default"
        );
        assert!(!cfg.engine.cpu_only);

        cfg.merge_cli(&CliOverrides {
            port: Some(8081),
            cpu_only: true,
            speculative: Some(false),
            ..CliOverrides::default()
        });
        assert_eq!(cfg.http.port, 8081, "CLI overrides TOML");
        assert!(!cfg.speculative.enabled, "CLI overrides compiled default");
        assert_eq!(cfg.engine.target_device, DeviceSelector::Cpu);
        assert!(cfg.engine.cpu_only);
        assert_eq!(
            cfg.engine.cpu_threads, 6,
            "CLI must not clobber unspecified TOML values"
        );
    }

    #[test]
    fn compiled_defaults_validate() {
        Config::compiled_defaults().validate().unwrap();
    }

    #[test]
    fn weights_trunk_bytes_defaults_to_unlimited_and_is_configurable() {
        let defaults = Config::compiled_defaults();
        assert_eq!(defaults.weights.trunk_bytes, 0);
        assert_eq!(defaults.weights.ring_slots, 2);
        assert_eq!(defaults.weights.miss_policy, MissPolicy::Qstar);
        assert_eq!(defaults.weights.host_bytes, 0);

        let cfg = Config::from_toml_str(
            r#"
            [weights]
            trunk_bytes = 4096
            ring_slots = 1
            miss_policy = "fill"
            host_bytes = 128
            "#,
        )
        .unwrap();
        assert_eq!(cfg.weights.trunk_bytes, 4096);
        assert_eq!(cfg.weights.ring_slots, 1);
        assert_eq!(cfg.weights.miss_policy, MissPolicy::Fill);
        assert_eq!(cfg.weights.host_bytes, 128);
        assert_eq!(
            cfg.kv.page_tokens, 128,
            "weights is composable, not a Pipeline Mode"
        );

        let mut bad = Config::compiled_defaults();
        bad.weights.ring_slots = 3;
        let err = bad.validate().unwrap_err();
        assert!(matches!(err, AllpacaError::Fatal(FatalError::InvalidConfig(_))));
    }

    #[test]
    fn prefill_chunk_tokens_defaults_to_512_and_is_configurable() {
        let defaults = Config::compiled_defaults();
        assert_eq!(
            defaults.prefill.chunk_tokens, 512,
            "compiled default matches PRD prefill.chunk_tokens"
        );

        let cfg = Config::from_toml_str(
            r#"
            [prefill]
            chunk_tokens = 64
            "#,
        )
        .unwrap();
        assert_eq!(cfg.prefill.chunk_tokens, 64);
        assert_eq!(
            cfg.engine.max_active_requests, 1,
            "prefill is composable, not a Pipeline Mode"
        );

        let mut bad = Config::compiled_defaults();
        bad.prefill.chunk_tokens = 0;
        let err = bad.validate().unwrap_err();
        assert_eq!(err.class(), crate::ErrorClass::Fatal);
        assert!(matches!(err, AllpacaError::Fatal(FatalError::InvalidConfig(_))));
    }

    #[test]
    fn max_active_requests_defaults_to_one_and_is_configurable() {
        let defaults = Config::compiled_defaults();
        assert_eq!(
            defaults.engine.max_active_requests, 1,
            "compiled default keeps single-Request Engine behavior"
        );

        let cfg = Config::from_toml_str(
            r#"
            [engine]
            max_active_requests = 4
            "#,
        )
        .unwrap();
        assert_eq!(cfg.engine.max_active_requests, 4);
        assert_eq!(
            cfg.engine.cpu_threads, 4,
            "max_active_requests is composable, not a Pipeline Mode"
        );

        let mut bad = Config::compiled_defaults();
        bad.engine.max_active_requests = 0;
        let err = bad.validate().unwrap_err();
        assert_eq!(err.class(), crate::ErrorClass::Fatal);
        assert!(matches!(err, AllpacaError::Fatal(FatalError::InvalidConfig(_))));
    }

    #[test]
    fn tree_speculation_is_off_by_default_and_enabled_via_toml() {
        let defaults = Config::compiled_defaults();
        assert!(
            !defaults.speculative.tree.enabled,
            "Token Forest Feature Flag defaults off"
        );
        assert!(
            defaults.speculative.tree.max_nodes > 0,
            "compiled max_nodes must be a conservative positive budget"
        );

        let cfg = Config::from_toml_str(
            r#"
            [speculative.tree]
            enabled = true
            max_nodes = 16
            "#,
        )
        .unwrap();
        assert!(cfg.speculative.tree.enabled);
        assert_eq!(cfg.speculative.tree.max_nodes, 16);
        assert!(cfg.speculative.enabled, "tree is composable, not a Pipeline Mode");
    }

    #[test]
    fn model_section_accepts_safetensors_sidecar_paths() {
        let cfg = Config::from_toml_str(
            r#"
            [model]
            path = "weights.safetensors"
            config_path = "config.json"
            tokenizer_path = "tokenizer.json"
            "#,
        )
        .unwrap();
        assert_eq!(
            cfg.model.path.as_deref(),
            Some(std::path::Path::new("weights.safetensors"))
        );
        assert_eq!(
            cfg.model.config_path.as_deref(),
            Some(std::path::Path::new("config.json"))
        );
        assert_eq!(
            cfg.model.tokenizer_path.as_deref(),
            Some(std::path::Path::new("tokenizer.json"))
        );
    }

    #[test]
    fn cli_model_flag_points_at_gguf_path() {
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            model: Some(PathBuf::from("path/to/model.gguf")),
            draft_model: Some(PathBuf::from("path/to/draft.gguf")),
            ..CliOverrides::default()
        });
        assert_eq!(
            cfg.model.path.as_deref(),
            Some(std::path::Path::new("path/to/model.gguf"))
        );
        assert_eq!(
            cfg.model.draft_path.as_deref(),
            Some(std::path::Path::new("path/to/draft.gguf"))
        );
    }
}
