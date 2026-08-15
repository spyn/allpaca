use std::path::PathBuf;

use allpaca_types::{CliOverrides, DeviceSelector, PipelineMode};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "allpaca-server", about = "Allpaca OpenAI-compatible inference server")]
pub struct Cli {
    /// TOML config file. CLI flags override file values; file overrides compiled defaults.
    #[arg(long)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub model: Option<PathBuf>,
    #[arg(long)]
    pub draft_model: Option<PathBuf>,
    #[arg(long)]
    pub host: Option<String>,
    #[arg(long)]
    pub port: Option<u16>,
    #[arg(long)]
    pub target_device: Option<String>,
    #[arg(long)]
    pub draft_device: Option<String>,
    #[arg(long)]
    pub cpu_only: bool,
    /// Load a built-in tiny CPU model (no GGUF required).
    #[arg(long)]
    pub synthetic: bool,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub speculative: Option<bool>,
    #[arg(long)]
    pub json_mode: Option<bool>,
    #[arg(long)]
    pub pipeline_mode: Option<String>,
    #[arg(long)]
    pub cpu_threads: Option<usize>,
}

impl Cli {
    pub fn parse_args() -> Self {
        Self::parse()
    }

    pub fn to_overrides(&self) -> CliOverrides {
        CliOverrides {
            model: self.model.clone(),
            draft_model: self.draft_model.clone(),
            target_device: self
                .target_device
                .as_deref()
                .and_then(|s| DeviceSelector::parse(s).ok()),
            draft_device: self
                .draft_device
                .as_deref()
                .and_then(|s| DeviceSelector::parse(s).ok()),
            cpu_threads: self.cpu_threads,
            pipeline_mode: self.pipeline_mode.as_deref().and_then(parse_mode),
            seed: self.seed,
            cpu_only: self.cpu_only,
            speculative: self.speculative,
            host: self.host.clone(),
            port: self.port,
            synthetic: self.synthetic,
            json_mode: self.json_mode,
        }
    }
}

fn parse_mode(raw: &str) -> Option<PipelineMode> {
    match raw.to_ascii_lowercase().as_str() {
        "asymmetric" => Some(PipelineMode::Asymmetric),
        "symmetric" => Some(PipelineMode::Symmetric),
        "mobile_tiered" | "mobile" => Some(PipelineMode::MobileTiered),
        _ => None,
    }
}
