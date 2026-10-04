use std::collections::BTreeMap;
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
    pub multimodal: MultimodalSection,
    pub features: FeatureFlagOverlay,
    #[serde(default)]
    pub rpc: RpcSection,
    #[serde(default)]
    pub telemetry: TelemetrySection,
    /// Per-model MoE pin-prefer keep-sets, keyed by unresolved model spec.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tunes: BTreeMap<String, TuneEntry>,
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
            multimodal: MultimodalSection::default(),
            features: FeatureFlagOverlay::default(),
            rpc: RpcSection::default(),
            telemetry: TelemetrySection::default(),
            tunes: BTreeMap::new(),
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
        if let Some(providers) = &cli.speculative_providers {
            if !providers.is_empty() {
                self.speculative.providers = providers.clone();
            }
        }
        if let Some(host) = &cli.host {
            self.http.host = host.clone();
        }
        if let Some(port) = cli.port {
            self.http.port = port;
        }
        if let Some(password) = &cli.ui_password {
            if !password.is_empty() {
                self.http.password = Some(password.clone());
            }
        }
        if cli.synthetic {
            self.model.synthetic = true;
        }
        if cli.edge0_no_lora {
            self.model.edge0_no_lora = true;
        }
        if cli.edge0_no_prerouter {
            self.model.edge0_no_prerouter = true;
        }
        if let Some(json) = cli.json_mode {
            self.structured_output.json_mode = json;
        }
        if let Some(endpoints) = &cli.rpc_endpoints {
            self.rpc.endpoints = endpoints.clone();
        }
        if let Some(token) = &cli.rpc_token {
            if !token.is_empty() {
                self.rpc.token = Some(token.clone());
            }
        }
        if let Some(split) = &cli.tensor_split {
            let mut weights = Vec::new();
            for part in split {
                if let Ok(w) = part.parse::<f64>() {
                    if w > 0.0 {
                        weights.push(w);
                    }
                }
            }
            if !weights.is_empty() {
                self.rpc.tensor_split = Some(weights);
            }
        }
    }

    /// Slash-normalized lookup key for [`Self::tunes`].
    pub fn normalize_tune_key(spec: &str) -> String {
        spec.trim().replace('\\', "/")
    }

    /// Matching `[tunes]` entry for `spec` (path, Hub id, or filename).
    pub fn tune_for_spec(&self, spec: &str) -> Option<&TuneEntry> {
        lookup_tune(&self.tunes, spec)
    }

    /// Matching `[tunes]` entry for the configured `model.path`.
    pub fn tune_for_model(&self) -> Option<&TuneEntry> {
        let path = self.model.path.as_ref()?;
        self.tune_for_spec(&path.to_string_lossy())
    }

    /// Overlay a matching tune onto `weights.cache_bytes`. No-op when unset.
    pub fn apply_tune_overlay(&mut self) {
        let Some(entry) = self.tune_for_model().cloned() else {
            return;
        };
        if entry.cache_bytes > 0 {
            self.weights.cache_bytes = entry.cache_bytes;
        }
    }

    /// Insert or replace the tune for `spec`.
    pub fn upsert_tune(&mut self, spec: &str, entry: TuneEntry) {
        self.tunes.insert(Self::normalize_tune_key(spec), entry);
    }

    /// Write this config as TOML (used by `--tune` save when creating a new file).
    pub fn to_toml_string(&self) -> Result<String, AllpacaError> {
        toml::to_string_pretty(self).map_err(|e| {
            AllpacaError::Fatal(FatalError::InvalidConfig(format!("TOML serialize: {e}")))
        })
    }

    /// Patch `path` with `entry` under `[tunes."<spec>"]`, creating the file if needed.
    pub fn write_tune_file(
        path: &Path,
        spec: &str,
        entry: &TuneEntry,
        model_path: Option<&Path>,
    ) -> Result<(), AllpacaError> {
        let key = Self::normalize_tune_key(spec);
        let mut root: toml::Value = if path.exists() {
            let src = std::fs::read_to_string(path).map_err(|e| {
                AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                    "read {}: {e}",
                    path.display()
                )))
            })?;
            toml::from_str(&src).map_err(|e| {
                AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                    "TOML parse {}: {e}",
                    path.display()
                )))
            })?
        } else {
            toml::Value::Table(toml::map::Map::new())
        };
        let table = root.as_table_mut().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "tune config root must be a table".into(),
            ))
        })?;
        if let Some(model_path) = model_path {
            let model = table
                .entry("model".to_string())
                .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
            if let Some(m) = model.as_table_mut() {
                m.insert(
                    "path".into(),
                    toml::Value::String(model_path.to_string_lossy().replace('\\', "/")),
                );
            }
        }
        let tunes = table
            .entry("tunes".to_string())
            .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
        let tunes = tunes.as_table_mut().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "[tunes] must be a table".into(),
            ))
        })?;
        let value = toml::Value::try_from(entry).map_err(|e| {
            AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "serialize tune entry: {e}"
            )))
        })?;
        tunes.insert(key, value);
        let serialized = toml::to_string_pretty(&root).map_err(|e| {
            AllpacaError::Fatal(FatalError::InvalidConfig(format!("TOML serialize: {e}")))
        })?;
        std::fs::write(path, serialized).map_err(|e| {
            AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "write {}: {e}",
                path.display()
            )))
        })?;
        Ok(())
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
        if self.http.max_tokens == 0 {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "http.max_tokens must be > 0".into(),
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
    /// With `max_active_requests > 1`, decode the next token of every
    /// Request in plain decode (no speculation, grammar or vision) in one
    /// batched forward on HIP instead of one forward per Request.
    pub batch_decode: bool,
    /// `"auto"` (HIP, else CUDA, else Vulkan), `"hip"`, `"cuda"`, `"vulkan"`, or `"cpu"`.
    pub gpu_backend: String,
    /// Layer budget per local GPU in MiB, target GPU first, used instead of
    /// the budget measured from free VRAM at load. The measured value moves
    /// with desktop VRAM use, which can shift a layer between GPUs (or its
    /// experts onto RPC) from one load to the next; pin it for reproducible
    /// placement. Empty (default) or a `0` entry keeps the measured budget.
    #[serde(default)]
    pub local_vram_budget_mib: Vec<u64>,
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
            batch_decode: true,
            local_vram_budget_mib: Vec::new(),
            gpu_backend: "auto".into(),
        }
    }
}

/// Remote kernel-ABI workers (llama.cpp `--rpc` analogue).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RpcSection {
    /// `host:port` list. Empty = local-only.
    pub endpoints: Vec<String>,
    /// Relative layer weights in device order (local target, then RPC). `None` = MemInfo.
    pub tensor_split: Option<Vec<f64>>,
    /// KV sequence cap for remote layer caches (tokens). `0` (default) sizes
    /// it to the context the engine's own KV budget (`kv.vram_bytes`) holds
    /// for this model, so remote layers are not the shorter limit.
    pub max_seq: usize,
    /// Prompt rows sent to a remote layer span per round trip during prefill
    /// (the worker keeps a scratch of this many rows per GPU). `0` sends one
    /// token per round trip.
    pub prefill_rows: usize,
    /// Shared HELLO secret (`--rpc-token` / `ALLPACA_RPC_TOKEN`). Empty = loopback workers only.
    #[serde(default)]
    pub token: Option<String>,
    /// Where a layer's routed experts go with expert-parallel MoE, by layer
    /// index: `"local"` (beside the layer body on its GPU) or
    /// `"rpc:<host:port>/<gpu>"`. Layers not listed are placed automatically.
    /// The WebUI saves these per model.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub expert_homes: BTreeMap<String, String>,
}

impl Default for RpcSection {
    fn default() -> Self {
        Self {
            endpoints: Vec::new(),
            tensor_split: None,
            max_seq: 0,
            prefill_rows: 128,
            token: None,
            expert_homes: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DraftProviderKind {
    Ngram,
    DraftModel,
    /// Trailing NextN/MTP blocks from the target GGUF (self-speculation).
    Mtp,
    /// DeepSeek-V4-Flash DSpark sidecar (`dspark-*.gguf` / DFlash + Markov head).
    Dspark,
    /// DFlash2 block-diffusion draft (`model.draft_path`, arch `dflash` with a selector).
    Dflash2,
}

impl DraftProviderKind {
    pub fn parse_list(raw: &str) -> Result<Vec<Self>, String> {
        let mut out = Vec::new();
        for part in raw.split(|c| c == ',' || c == ';' || c == '|') {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            out.push(Self::parse_one(p)?);
        }
        if out.is_empty() {
            return Err("empty speculative providers list".into());
        }
        Ok(out)
    }

    fn parse_one(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "ngram" | "n-gram" => Ok(Self::Ngram),
            "draft" | "draft_model" | "draft-model" => Ok(Self::DraftModel),
            "mtp" | "nextn" | "next-n" => Ok(Self::Mtp),
            "dspark" | "dflash" | "draft-dspark" | "draft_dspark" => Ok(Self::Dspark),
            "dflash2" | "dflash-2" | "draft-dflash2" => Ok(Self::Dflash2),
            other => Err(format!(
                "unknown draft provider '{other}' (expected ngram, draft_model, mtp, dspark, dflash2)"
            )),
        }
    }
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
    /// Speculate only while at most this many Requests are active (0 = no
    /// limit). Beyond it, batched plain decode outruns batched verify: on
    /// Qwen3-Coder-Next with remote experts, 4 concurrent Requests ran 31-32
    /// tok/s without speculation and ~28 with n-gram, while 3 ran ~28 with
    /// n-gram and ~25 without.
    pub max_concurrent_requests: usize,
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
            max_concurrent_requests: 3,
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

/// Pin-prefer keep-set saved by `allpaca-server --tune` for one model spec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TuneEntry {
    /// Named bundle applied in the TUI (`late_layers`, catalog profile, …).
    pub profile: String,
    /// Selected group ids (`early`, `group_0`, catalog topic names).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<String>,
    /// Context length the budget was computed at (tokens).
    pub ctx_tokens: u32,
    /// GPU/hot expert cache overlay; 0 leaves `weights.cache_bytes` unchanged.
    pub cache_bytes: usize,
    /// Compact keep-set: per-layer expert indices to pin. Empty = expand `groups` at load.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub keep_experts: Vec<Vec<u32>>,
}

fn lookup_tune<'a>(tunes: &'a BTreeMap<String, TuneEntry>, spec: &str) -> Option<&'a TuneEntry> {
    let key = Config::normalize_tune_key(spec);
    if let Some(entry) = tunes.get(&key) {
        return Some(entry);
    }
    let path = Path::new(&key);
    if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
        if let Some(entry) = tunes.get(name) {
            return Some(entry);
        }
        for (k, entry) in tunes {
            if Path::new(k).file_name().and_then(|s| s.to_str()) == Some(name) {
                return Some(entry);
            }
        }
    }
    None
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
    /// Hub revision when `path` is `org/name` (default `main`; also `org/name@rev`).
    pub hf_revision: Option<String>,
    pub synthetic: bool,
    pub id: String,
    /// Edge0: skip Recover-LoRA install.
    #[serde(default)]
    pub edge0_no_lora: bool,
    /// Edge0: skip prerouter install (reactive expert fill only).
    #[serde(default)]
    pub edge0_no_prerouter: bool,
    /// Folder the WebUI scans (recursively) for models to switch to. Unset:
    /// the nearest `models` folder above `path`, else `./models`.
    #[serde(default)]
    pub models_dir: Option<PathBuf>,
}

impl Default for ModelSection {
    fn default() -> Self {
        Self {
            path: None,
            draft_path: None,
            config_path: None,
            tokenizer_path: None,
            hf_revision: None,
            synthetic: false,
            id: "allpaca".into(),
            edge0_no_lora: false,
            edge0_no_prerouter: false,
            models_dir: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HttpSection {
    pub host: String,
    pub port: u16,
    /// Shared operator password for the WebUI cookie and `Authorization: Bearer`.
    /// Empty/unset: the server binary generates one at boot and prints it to stderr
    /// (not tracing / `/v1/logs`).
    #[serde(default)]
    pub password: Option<String>,
    /// Cap for request `max_tokens`. WebUI slider max is 4096.
    #[serde(default = "default_http_max_tokens")]
    pub max_tokens: usize,
}

fn default_http_max_tokens() -> usize {
    4096
}

impl Default for HttpSection {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8080,
            password: None,
            max_tokens: default_http_max_tokens(),
        }
    }
}

/// Optional OpenTelemetry-style export (JSON OTLP body; no SDK).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TelemetrySection {
    /// When set, the server periodically POSTs OTLP/JSON metrics here (appends `/v1/metrics`
    /// if the URL has no path). Pull via `GET /v1/metrics/otlp` always remains available.
    pub otlp_endpoint: Option<String>,
}

impl Default for TelemetrySection {
    fn default() -> Self {
        Self {
            otlp_endpoint: None,
        }
    }
}

/// Multimodal encoder placement + optional vision weight path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct MultimodalSection {
    /// Device for the vision encoder (`hip:N`, `cuda:N`, `cpu`, or unset/`auto`).
    pub vision_device: Option<DeviceSelector>,
    /// SafeTensors with `patch_proj.weight` / optional `patch_proj.bias`, or a
    /// llama.cpp `mmproj` `.gguf` (Qwen3-VL `qwen3vl_merger` projector).
    /// When unset and `model.synthetic`, Engine installs deterministic patch weights.
    pub vision_path: Option<PathBuf>,
    /// Square resize before patching (must be multiple of `vision_patch`).
    pub vision_image_size: u32,
    /// Patch edge in pixels.
    pub vision_patch: u32,
    /// Optional ViT-lite transformer blocks after patch embed (0 = patch-only).
    #[serde(default)]
    pub vision_n_blocks: u32,
    /// mmproj encoders: most LLM tokens per image (0 = encoder default, 1024).
    #[serde(default)]
    pub vision_max_tokens: u32,
}

impl Default for MultimodalSection {
    fn default() -> Self {
        Self {
            vision_device: None,
            vision_path: None,
            vision_image_size: 64,
            vision_patch: 16,
            vision_n_blocks: 0,
            vision_max_tokens: 0,
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
    /// Comma-separated draft providers, e.g. `mtp`, `ngram,draft_model`.
    pub speculative_providers: Option<Vec<DraftProviderKind>>,
    pub host: Option<String>,
    pub port: Option<u16>,
    /// Shared WebUI / API password (`--ui-password`).
    pub ui_password: Option<String>,
    pub synthetic: bool,
    pub json_mode: Option<bool>,
    /// Comma-separated `host:port` RPC workers.
    pub rpc_endpoints: Option<Vec<String>>,
    /// Shared RPC HELLO token (`--rpc-token`).
    pub rpc_token: Option<String>,
    /// Comma-separated relative layer weights (same order as local target + RPC).
    pub tensor_split: Option<Vec<String>>,
    pub edge0_no_lora: bool,
    pub edge0_no_prerouter: bool,
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
            password = "operator"
        "#;
        let mut cfg = Config::from_toml_str(toml).unwrap();
        assert_eq!(cfg.engine.cpu_threads, 6, "TOML overrides compiled default");
        assert_eq!(cfg.http.port, 9000, "TOML overrides compiled default");
        assert_eq!(cfg.http.password.as_deref(), Some("operator"));
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
    fn example_config_file_parses() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../allpaca.example.toml");
        let cfg =
            Config::from_toml_file(&path).expect(path.to_str().unwrap_or("allpaca.example.toml"));
        assert_eq!(cfg.http.host, "127.0.0.1");
        assert_eq!(cfg.http.port, 8080);
        assert_eq!(cfg.http.password.as_deref(), Some("changeme"));
        assert_eq!(cfg.model.id, "allpaca");
        assert_eq!(cfg.engine.cpu_threads, 4);
        assert!(cfg.speculative.enabled);
        assert_eq!(cfg.speculative.providers.len(), 2);
        assert!(!cfg.speculative.tree.enabled);
        assert_eq!(cfg.prefill.chunk_tokens, 512);
        assert_eq!(cfg.weights.ring_slots, 2);
        assert_eq!(cfg.sampling.top_k, 40);
        assert!(cfg.rpc.endpoints.is_empty());
        assert!(cfg.telemetry.otlp_endpoint.is_none());
    }

    #[test]
    fn speculative_providers_parse_mtp_list() {
        let list = DraftProviderKind::parse_list("mtp").unwrap();
        assert_eq!(list, vec![DraftProviderKind::Mtp]);
        let multi = DraftProviderKind::parse_list("ngram,draft_model").unwrap();
        assert_eq!(
            multi,
            vec![DraftProviderKind::Ngram, DraftProviderKind::DraftModel]
        );
        assert!(DraftProviderKind::parse_list("nope").is_err());
        let ds = DraftProviderKind::parse_list("dspark").unwrap();
        assert_eq!(ds, vec![DraftProviderKind::Dspark]);
        // "dflash" stays the DSpark alias; DFlash2 is its own provider.
        let d2 = DraftProviderKind::parse_list("dflash2,dflash-2").unwrap();
        assert_eq!(d2, vec![DraftProviderKind::Dflash2, DraftProviderKind::Dflash2]);
        let aliases = DraftProviderKind::parse_list("dflash,draft-dspark").unwrap();
        assert_eq!(
            aliases,
            vec![DraftProviderKind::Dspark, DraftProviderKind::Dspark]
        );
    }

    #[test]
    fn compiled_defaults_validate() {
        Config::compiled_defaults().validate().unwrap();
    }

    #[test]
    fn multimodal_vision_device_parses_from_toml() {
        let cfg = Config::from_toml_str(
            r#"
            [multimodal]
            vision_device = "hip:1"
            vision_n_blocks = 1
            "#,
        )
        .unwrap();
        assert_eq!(cfg.multimodal.vision_device, Some(DeviceSelector::Hip(1)));
        assert_eq!(cfg.multimodal.vision_n_blocks, 1);
        assert!(Config::compiled_defaults()
            .multimodal
            .vision_device
            .is_none());
        assert_eq!(Config::compiled_defaults().multimodal.vision_n_blocks, 0);
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
        assert!(matches!(
            err,
            AllpacaError::Fatal(FatalError::InvalidConfig(_))
        ));
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
        assert!(matches!(
            err,
            AllpacaError::Fatal(FatalError::InvalidConfig(_))
        ));
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
        assert!(matches!(
            err,
            AllpacaError::Fatal(FatalError::InvalidConfig(_))
        ));
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
        assert!(
            cfg.speculative.enabled,
            "tree is composable, not a Pipeline Mode"
        );
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
    fn model_section_accepts_hf_repo_and_revision() {
        let cfg = Config::from_toml_str(
            r#"
            [model]
            path = "Qwen/Qwen2.5-0.5B-Instruct"
            hf_revision = "main"
            "#,
        )
        .unwrap();
        assert_eq!(
            cfg.model.path.as_deref(),
            Some(std::path::Path::new("Qwen/Qwen2.5-0.5B-Instruct"))
        );
        assert_eq!(cfg.model.hf_revision.as_deref(), Some("main"));
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

    #[test]
    fn rpc_toml_and_cli() {
        let toml = r#"
            [rpc]
            endpoints = ["127.0.0.1:50052"]
            tensor_split = [2.0, 1.0]
            max_seq = 128
        "#;
        let mut cfg = Config::from_toml_str(toml).unwrap();
        assert_eq!(cfg.rpc.endpoints, vec!["127.0.0.1:50052"]);
        assert_eq!(cfg.rpc.tensor_split.as_deref(), Some(&[2.0, 1.0][..]));
        assert_eq!(cfg.rpc.max_seq, 128);
        cfg.merge_cli(&CliOverrides {
            rpc_endpoints: Some(vec!["10.0.0.2:9".into()]),
            tensor_split: Some(vec!["3".into(), "1".into()]),
            ..CliOverrides::default()
        });
        assert_eq!(cfg.rpc.endpoints, vec!["10.0.0.2:9"]);
        assert_eq!(cfg.rpc.tensor_split.as_deref(), Some(&[3.0, 1.0][..]));
    }

    #[test]
    fn tune_entry_round_trip_and_overlay() {
        let toml = r#"
            [model]
            path = "models/DeepSeek-V4-Flash.gguf"
            [tunes."models/DeepSeek-V4-Flash.gguf"]
            profile = "late_layers"
            groups = ["late"]
            ctx_tokens = 32768
            cache_bytes = 8589934592
            keep_experts = [[0, 1], [2]]
        "#;
        let mut cfg = Config::from_toml_str(toml).unwrap();
        let entry = cfg.tune_for_model().expect("tune for model.path");
        assert_eq!(entry.profile, "late_layers");
        assert_eq!(entry.groups, vec!["late"]);
        assert_eq!(entry.ctx_tokens, 32768);
        assert_eq!(entry.cache_bytes, 8589934592);
        assert_eq!(entry.keep_experts, vec![vec![0, 1], vec![2]]);
        assert_eq!(cfg.weights.cache_bytes, 0, "overlay is explicit");
        cfg.apply_tune_overlay();
        assert_eq!(cfg.weights.cache_bytes, 8589934592);

        let by_name = cfg
            .tune_for_spec("DeepSeek-V4-Flash.gguf")
            .expect("filename lookup");
        assert_eq!(by_name.profile, "late_layers");

        cfg.upsert_tune(
            r"models\other.gguf",
            TuneEntry {
                profile: "balanced".into(),
                ctx_tokens: 65536,
                cache_bytes: 1024,
                ..TuneEntry::default()
            },
        );
        assert!(cfg.tunes.contains_key("models/other.gguf"));
        let dumped = cfg.to_toml_string().unwrap();
        let reparsed = Config::from_toml_str(&dumped).unwrap();
        assert_eq!(
            reparsed.tune_for_spec("models/other.gguf").unwrap().cache_bytes,
            1024
        );
    }

    #[test]
    fn write_tune_file_patches_existing_toml() {
        let dir = std::env::temp_dir().join(format!(
            "allpaca_tune_write_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("allpaca.toml");
        std::fs::write(&path, "[http]\nport = 9090\n").unwrap();
        let entry = TuneEntry {
            profile: "balanced".into(),
            groups: vec!["early".into()],
            ctx_tokens: 32768,
            cache_bytes: 4096,
            keep_experts: vec![vec![0]],
        };
        Config::write_tune_file(
            &path,
            "models/foo.gguf",
            &entry,
            Some(Path::new("models/foo.gguf")),
        )
        .unwrap();
        let cfg = Config::from_toml_file(&path).unwrap();
        assert_eq!(cfg.http.port, 9090);
        assert_eq!(
            cfg.model.path.as_deref(),
            Some(Path::new("models/foo.gguf"))
        );
        let saved = cfg.tune_for_spec("models/foo.gguf").unwrap();
        assert_eq!(saved.cache_bytes, 4096);
        assert_eq!(saved.groups, vec!["early"]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
