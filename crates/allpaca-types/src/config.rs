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
    pub kv: KvSection,
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
            kv: KvSection::default(),
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
        Ok(())
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
}

impl Default for KvSection {
    fn default() -> Self {
        Self {
            page_tokens: 128,
            hot_fraction: 0.75,
            vram_bytes: 4 * 1024 * 1024,
            dram_bytes: 32 * 1024 * 1024,
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
}

impl Default for StructuredOutputSection {
    fn default() -> Self {
        Self {
            enabled: true,
            json_mode: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelSection {
    pub path: Option<PathBuf>,
    pub draft_path: Option<PathBuf>,
    pub synthetic: bool,
    pub id: String,
}

impl Default for ModelSection {
    fn default() -> Self {
        Self {
            path: None,
            draft_path: None,
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
}
