//! MoE tune: checkpoint probe, group expansion, keep-set ranking, budget math.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use allpaca_gguf::{is_dspark_filename, GgufFile};
use allpaca_safetensors::{
    discover_checkpoint, load_hf_config, resolve_sidecar, CheckpointKind, DiscoveredCheckpoint,
};
use allpaca_types::{Config, TuneEntry};
use serde::Deserialize;

use crate::hf::HubOptions;
use crate::model::ModelConfig;
use crate::weight_set::ExpertId;

/// How selected topic histograms are combined into a per-layer keep-set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RankRule {
    /// Sum of per-topic normalized histograms; keep the top `n`.
    Sum,
    /// Hand each slot to the currently least-covered selected topic (Flash Spark default).
    #[default]
    MaxMin,
}

/// Architecture facts needed to expand groups without loading tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeShape {
    pub n_layers: usize,
    pub n_experts: usize,
    pub n_expert_groups: usize,
    pub leading_dense_blocks: usize,
    pub n_kv_heads: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub dim: usize,
}

impl MoeShape {
    pub fn from_model_config(cfg: &ModelConfig) -> Self {
        Self {
            n_layers: cfg.n_layers,
            n_experts: cfg.n_experts,
            n_expert_groups: cfg.n_expert_groups.max(1),
            leading_dense_blocks: cfg.leading_dense_blocks,
            n_kv_heads: cfg.n_kv_heads.max(1),
            n_heads: cfg.n_heads.max(1),
            head_dim: cfg.head_dim().max(1),
            dim: cfg.dim.max(1),
        }
    }

    pub fn is_moe(&self) -> bool {
        self.n_experts > 0
    }

    pub fn moe_layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.leading_dense_blocks..self.n_layers
    }
}

/// Metadata-only probe of a checkpoint (no weight materialize).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeProbe {
    pub spec: String,
    pub path: PathBuf,
    pub name: String,
    pub shape: MoeShape,
    pub expert_bytes: u64,
    pub dense_bytes: u64,
    pub file_bytes: u64,
}

/// Live host + GPU memory the TUI header reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceSnapshot {
    pub ram_total_bytes: u64,
    pub ram_used_bytes: u64,
    pub gpu_name: String,
    pub gpu_memory_bytes: u64,
}

impl ResourceSnapshot {
    pub fn ram_free_bytes(&self) -> u64 {
        self.ram_total_bytes.saturating_sub(self.ram_used_bytes)
    }
}

/// Budget line the TUI and load path share so the screen cannot lie.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuneBudget {
    pub expert_pin_bytes: u64,
    pub kv_bytes: u64,
    pub dense_bytes: u64,
    pub resident_bytes: u64,
    pub free_after: i64,
    pub fits: bool,
    pub pin_count: usize,
    pub total_experts: usize,
}

/// Named selectable row (router group, layer band, or catalog topic).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuneGroup {
    pub id: String,
    pub title: String,
    pub kind: GroupKind,
    pub expert_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupKind {
    RouterGroup,
    LayerBand,
    Catalog,
}

/// Named bundle the TUI can apply with Enter.
#[derive(Debug, Clone, PartialEq)]
pub struct TuneProfile {
    pub id: String,
    pub title: String,
    pub description: String,
    pub groups: Vec<String>,
    pub ctx_tokens: u32,
    /// Fraction of GPU memory to reserve for the expert cache (0..=1).
    pub cache_fraction: f32,
}

/// Optional sidecar histograms (Allpaca or Flash Spark `coverage.json`).
#[derive(Debug, Clone, Default)]
pub struct KeepsetCatalog {
    pub n_layers: usize,
    pub n_experts: usize,
    pub topics: BTreeMap<String, Vec<Vec<f64>>>,
    pub profiles: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Deserialize, Default)]
struct CatalogFile {
    #[serde(default)]
    n_layers: usize,
    #[serde(default)]
    n_experts: usize,
    #[serde(default)]
    layers: usize,
    #[serde(default)]
    topics: BTreeMap<String, Vec<Vec<f64>>>,
    #[serde(default)]
    profiles: BTreeMap<String, Vec<String>>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

/// A discovered local or hub-cache checkpoint (not yet probed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredModel {
    pub spec: String,
    pub path: PathBuf,
    pub label: String,
    pub bytes: u64,
}

pub const CTX_STEPS: [u32; 4] = [32_768, 65_536, 131_072, 262_144];

/// Normalize a model spec for `[tunes]` keys.
pub fn normalize_spec(spec: &str) -> String {
    Config::normalize_tune_key(spec)
}

pub fn ids_from_keep_experts(keep: &[Vec<u32>]) -> Vec<ExpertId> {
    let mut out = Vec::new();
    for (layer, experts) in keep.iter().enumerate() {
        for &expert in experts {
            out.push(ExpertId {
                layer: layer as u32,
                expert,
            });
        }
    }
    out
}

pub fn keep_experts_from_ids(ids: &[ExpertId], n_layers: usize) -> Vec<Vec<u32>> {
    let mut layers = vec![Vec::new(); n_layers];
    for id in ids {
        let li = id.layer as usize;
        if li < layers.len() && !layers[li].contains(&id.expert) {
            layers[li].push(id.expert);
        }
    }
    for layer in &mut layers {
        layer.sort_unstable();
    }
    layers
}

/// Expand selected group ids into `(layer, expert)` pins.
pub fn expand_groups(groups: &[String], shape: &MoeShape) -> Vec<ExpertId> {
    if !shape.is_moe() || groups.is_empty() {
        return Vec::new();
    }
    let mut set = HashSet::new();
    for g in groups {
        match g.as_str() {
            "early" | "mid" | "late" => {
                for layer in band_layers(g, shape) {
                    for expert in 0..shape.n_experts {
                        set.insert(ExpertId {
                            layer: layer as u32,
                            expert: expert as u32,
                        });
                    }
                }
            }
            other if other.starts_with("group_") => {
                if let Some(idx) = other
                    .strip_prefix("group_")
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    for layer in shape.moe_layers() {
                        for expert in router_group_experts(idx, shape) {
                            set.insert(ExpertId {
                                layer: layer as u32,
                                expert: expert as u32,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let mut out: Vec<ExpertId> = set.into_iter().collect();
    out.sort_by_key(|id| (id.layer, id.expert));
    out
}

fn band_layers(band: &str, shape: &MoeShape) -> Vec<usize> {
    let layers: Vec<usize> = shape.moe_layers().collect();
    if layers.is_empty() {
        return Vec::new();
    }
    let n = layers.len();
    let third = (n + 2) / 3;
    let (start, end) = match band {
        "early" => (0, third.min(n)),
        "mid" => (third.min(n), (third * 2).min(n)),
        "late" => ((third * 2).min(n), n),
        _ => (0, 0),
    };
    layers[start..end].to_vec()
}

fn router_group_experts(group: usize, shape: &MoeShape) -> Vec<usize> {
    let n_groups = shape.n_expert_groups.max(1);
    if group >= n_groups || shape.n_experts == 0 {
        return Vec::new();
    }
    let size = shape.n_experts / n_groups;
    let rem = shape.n_experts % n_groups;
    let start = group * size + group.min(rem);
    let len = size + usize::from(group < rem);
    (start..start.saturating_add(len).min(shape.n_experts)).collect()
}

/// Groups the TUI can toggle for this probe (catalog topics, router groups, layer bands).
pub fn available_groups(probe: &MoeProbe, catalog: Option<&KeepsetCatalog>) -> Vec<TuneGroup> {
    let mut out = Vec::new();
    if !probe.shape.is_moe() {
        return out;
    }
    if let Some(cat) = catalog {
        for name in cat.topics.keys() {
            let ids = catalog_topic_ids(cat, name);
            out.push(TuneGroup {
                id: name.clone(),
                title: name.clone(),
                kind: GroupKind::Catalog,
                expert_count: ids.len(),
            });
        }
    }
    if probe.shape.n_expert_groups > 1 {
        for g in 0..probe.shape.n_expert_groups {
            let n = router_group_experts(g, &probe.shape).len() * probe.shape.moe_layers().count();
            out.push(TuneGroup {
                id: format!("group_{g}"),
                title: format!("Router group {g}"),
                kind: GroupKind::RouterGroup,
                expert_count: n,
            });
        }
    }
    for (id, title) in [
        ("early", "Early layers"),
        ("mid", "Mid layers"),
        ("late", "Late layers"),
    ] {
        let n = expand_groups(&[id.to_string()], &probe.shape).len();
        out.push(TuneGroup {
            id: id.to_string(),
            title: title.to_string(),
            kind: GroupKind::LayerBand,
            expert_count: n,
        });
    }
    out
}

/// Built-in + catalog profiles for the TUI.
pub fn available_profiles(probe: &MoeProbe, catalog: Option<&KeepsetCatalog>) -> Vec<TuneProfile> {
    let mut out = vec![
        TuneProfile {
            id: "max_resident".into(),
            title: "Max resident".into(),
            description: "Fill GPU expert cache; short context".into(),
            groups: Vec::new(),
            ctx_tokens: 32_768,
            cache_fraction: 0.70,
        },
        TuneProfile {
            id: "balanced".into(),
            title: "Balanced".into(),
            description: "Split expert cache and KV".into(),
            groups: Vec::new(),
            ctx_tokens: 65_536,
            cache_fraction: 0.40,
        },
        TuneProfile {
            id: "long_context".into(),
            title: "Long context".into(),
            description: "Keep KV for 256k; smaller expert pin".into(),
            groups: Vec::new(),
            ctx_tokens: 262_144,
            cache_fraction: 0.20,
        },
        TuneProfile {
            id: "early_layers".into(),
            title: "Early layers".into(),
            description: "Pin early MoE layers".into(),
            groups: vec!["early".into()],
            ctx_tokens: 32_768,
            cache_fraction: 0.45,
        },
        TuneProfile {
            id: "late_layers".into(),
            title: "Late layers".into(),
            description: "Pin late MoE layers".into(),
            groups: vec!["late".into()],
            ctx_tokens: 32_768,
            cache_fraction: 0.45,
        },
    ];
    if let Some(cat) = catalog {
        for (name, groups) in &cat.profiles {
            out.push(TuneProfile {
                id: name.clone(),
                title: name.clone(),
                description: groups.join(", "),
                groups: groups.clone(),
                ctx_tokens: 32_768,
                cache_fraction: 0.40,
            });
        }
    }
    if probe.shape.n_expert_groups > 1 {
        for g in 0..probe.shape.n_expert_groups {
            out.push(TuneProfile {
                id: format!("group_{g}"),
                title: format!("Router group {g}"),
                description: format!("Pin DeepSeek-style expert group {g}"),
                groups: vec![format!("group_{g}")],
                ctx_tokens: 32_768,
                cache_fraction: 0.40,
            });
        }
    }
    out
}

/// Keep-set from catalog topics using `rule` and `keep_n` experts per layer.
pub fn rank_keep_set(
    catalog: &KeepsetCatalog,
    topics: &[String],
    keep_n: usize,
    rule: RankRule,
) -> Vec<ExpertId> {
    if topics.is_empty() || keep_n == 0 {
        return Vec::new();
    }
    let n_layers = catalog.n_layers;
    let n_experts = catalog.n_experts;
    let mut out = Vec::new();
    for layer in 0..n_layers {
        let mut hists = Vec::new();
        for topic in topics {
            if let Some(layers) = catalog.topics.get(topic) {
                if let Some(row) = layers.get(layer) {
                    hists.push(normalize(row, n_experts));
                }
            }
        }
        if hists.is_empty() {
            continue;
        }
        let kept = match rule {
            RankRule::Sum => rank_sum(&hists, keep_n),
            RankRule::MaxMin => rank_maxmin(&hists, keep_n),
        };
        for expert in kept {
            out.push(ExpertId {
                layer: layer as u32,
                expert: expert as u32,
            });
        }
    }
    out
}

fn catalog_topic_ids(catalog: &KeepsetCatalog, topic: &str) -> Vec<ExpertId> {
    rank_keep_set(
        catalog,
        &[topic.to_string()],
        catalog.n_experts.max(1),
        RankRule::Sum,
    )
}

fn normalize(row: &[f64], n_experts: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_experts];
    for (i, v) in row.iter().take(n_experts).enumerate() {
        out[i] = if v.is_finite() && *v > 0.0 { *v } else { 0.0 };
    }
    let sum: f64 = out.iter().sum();
    if sum > 0.0 {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

fn rank_sum(hists: &[Vec<f64>], keep_n: usize) -> Vec<usize> {
    let n = hists.first().map(Vec::len).unwrap_or(0);
    let mut score = vec![0.0; n];
    for hist in hists {
        for (i, v) in hist.iter().enumerate().take(n) {
            score[i] += *v;
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|a, b| {
        score[*b]
            .partial_cmp(&score[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });
    idx.truncate(keep_n.min(n));
    idx.sort_unstable();
    idx
}

fn rank_maxmin(hists: &[Vec<f64>], keep_n: usize) -> Vec<usize> {
    let n = hists.first().map(Vec::len).unwrap_or(0);
    if n == 0 || keep_n == 0 {
        return Vec::new();
    }
    let mut kept = HashSet::new();
    let mut coverage = vec![0.0; hists.len()];
    while kept.len() < keep_n.min(n) {
        let topic = coverage
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut best = None;
        let mut best_mass = f64::NEG_INFINITY;
        for e in 0..n {
            if kept.contains(&e) {
                continue;
            }
            let mass = hists[topic].get(e).copied().unwrap_or(0.0);
            if mass > best_mass {
                best_mass = mass;
                best = Some(e);
            }
        }
        let Some(expert) = best else {
            break;
        };
        kept.insert(expert);
        for (t, hist) in hists.iter().enumerate() {
            coverage[t] += hist.get(expert).copied().unwrap_or(0.0);
        }
    }
    let mut idx: Vec<usize> = kept.into_iter().collect();
    idx.sort_unstable();
    idx
}

/// KV bytes for `ctx_tokens` at 2-byte K/V (tune-screen estimate).
pub fn estimate_kv_bytes(shape: &MoeShape, ctx_tokens: u32) -> u64 {
    let ctx = ctx_tokens.max(1) as u64;
    let layers = shape.n_layers.max(1) as u64;
    let kv_heads = shape.n_kv_heads.max(1) as u64;
    let head_dim = shape.head_dim.max(1) as u64;
    layers
        .saturating_mul(kv_heads)
        .saturating_mul(head_dim)
        .saturating_mul(2)
        .saturating_mul(ctx)
        .saturating_mul(2)
}

pub fn estimate_budget(
    probe: &MoeProbe,
    pin_ids: &[ExpertId],
    ctx_tokens: u32,
    cache_bytes: u64,
    resources: &ResourceSnapshot,
) -> TuneBudget {
    let pin_count = pin_ids.len();
    let expert_pin = (pin_count as u64).saturating_mul(probe.expert_bytes);
    let cache = cache_bytes.max(expert_pin);
    let kv = estimate_kv_bytes(&probe.shape, ctx_tokens);
    let dense = probe.dense_bytes;
    let resident = cache.saturating_add(kv).saturating_add(dense);
    let pool = if resources.gpu_memory_bytes > 0 {
        resources.gpu_memory_bytes
    } else {
        resources.ram_free_bytes()
    };
    let free_after = pool as i64 - resident as i64;
    TuneBudget {
        expert_pin_bytes: expert_pin,
        kv_bytes: kv,
        dense_bytes: dense,
        resident_bytes: resident,
        free_after,
        fits: free_after >= 0,
        pin_count,
        total_experts: probe.shape.n_layers.saturating_mul(probe.shape.n_experts),
    }
}

/// Cache bytes implied by a profile fraction of GPU (or RAM) plus pin floor.
pub fn cache_bytes_for(
    probe: &MoeProbe,
    pin_ids: &[ExpertId],
    fraction: f32,
    resources: &ResourceSnapshot,
) -> usize {
    let pool = if resources.gpu_memory_bytes > 0 {
        resources.gpu_memory_bytes
    } else {
        resources.ram_free_bytes()
    };
    let frac = if fraction.is_finite() {
        fraction.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let from_frac = (pool as f64 * f64::from(frac)) as u64;
    let pin = (pin_ids.len() as u64).saturating_mul(probe.expert_bytes);
    from_frac.max(pin) as usize
}

/// Expand a saved tune into pin ids (explicit keep-set, catalog, or group names).
pub fn keep_ids_for_entry(
    entry: &TuneEntry,
    shape: &MoeShape,
    catalog: Option<&KeepsetCatalog>,
) -> Vec<ExpertId> {
    if !entry.keep_experts.is_empty() {
        return ids_from_keep_experts(&entry.keep_experts);
    }
    if let Some(cat) = catalog {
        let topics: Vec<String> = entry
            .groups
            .iter()
            .filter(|g| cat.topics.contains_key(*g))
            .cloned()
            .collect();
        if !topics.is_empty() {
            let keep_n = ((shape.n_experts as f32) * 0.4).ceil() as usize;
            return rank_keep_set(cat, &topics, keep_n.max(1), RankRule::MaxMin);
        }
    }
    expand_groups(&entry.groups, shape)
}

pub fn load_catalog_beside(path: &Path) -> Option<KeepsetCatalog> {
    let dir = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent().unwrap_or(path).to_path_buf()
    };
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let candidates = [
        dir.join("allpaca-keepset.json"),
        dir.join(format!("{stem}.allpaca-keepset.json")),
        dir.join("coverage.json"),
    ];
    for cand in candidates {
        if cand.is_file() {
            if let Ok(cat) = load_catalog(&cand) {
                return Some(cat);
            }
        }
    }
    None
}

pub fn load_catalog(path: &Path) -> Result<KeepsetCatalog, String> {
    let src = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    parse_catalog(&src)
}

pub fn parse_catalog(src: &str) -> Result<KeepsetCatalog, String> {
    let file: CatalogFile = serde_json::from_str(src).map_err(|e| format!("catalog json: {e}"))?;
    let mut topics = file.topics;
    for (k, v) in file.extra {
        if let Some(name) = k.strip_prefix("counts_") {
            if let Ok(hist) = serde_json::from_value::<Vec<Vec<f64>>>(v) {
                topics.entry(name.to_string()).or_insert(hist);
            }
        }
    }
    let n_layers = file
        .n_layers
        .max(file.layers)
        .max(topics.values().map(Vec::len).max().unwrap_or(0));
    let n_experts = file.n_experts.max(
        topics
            .values()
            .filter_map(|layers| layers.first().map(Vec::len))
            .max()
            .unwrap_or(0),
    );
    Ok(KeepsetCatalog {
        n_layers,
        n_experts,
        topics,
        profiles: file.profiles,
    })
}

/// Local `./models`, config `model.path`, and the HuggingFace hub cache.
pub fn discover_models(config: &Config) -> Vec<DiscoveredModel> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    if let Some(path) = &config.model.path {
        push_discovered(&mut out, &mut seen, path, &path.to_string_lossy());
    }
    push_dir(&mut out, &mut seen, Path::new("models"));
    let cache = HubOptions::from_env().cache_dir;
    if cache.is_dir() {
        scan_hf_cache(&mut out, &mut seen, &cache);
    }
    out.sort_by(|a, b| a.label.cmp(&b.label));
    out
}

fn push_dir(out: &mut Vec<DiscoveredModel>, seen: &mut HashSet<String>, dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for ent in entries.flatten() {
        let path = ent.path();
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if path.is_dir() {
            if looks_like_checkpoint_dir(&path) {
                push_discovered(out, seen, &path, &path.to_string_lossy());
            } else {
                push_dir(out, seen, &path);
            }
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") && !is_dspark_filename(name) {
            push_discovered(out, seen, &path, &path.to_string_lossy());
        }
    }
}

fn scan_hf_cache(out: &mut Vec<DiscoveredModel>, seen: &mut HashSet<String>, cache: &Path) {
    let Ok(entries) = fs::read_dir(cache) else {
        return;
    };
    for ent in entries.flatten() {
        let path = ent.path();
        if !path.is_dir() {
            continue;
        }
        let snaps = path.join("snapshots");
        if snaps.is_dir() {
            if let Ok(revs) = fs::read_dir(&snaps) {
                for rev in revs.flatten() {
                    let snap = rev.path();
                    if snap.is_dir() && looks_like_checkpoint_dir(&snap) {
                        let spec = hub_spec_from_cache_dir(&path)
                            .unwrap_or_else(|| snap.to_string_lossy().into_owned());
                        push_discovered(out, seen, &snap, &spec);
                    }
                }
            }
        }
        push_dir(out, seen, &path);
    }
}

fn hub_spec_from_cache_dir(dir: &Path) -> Option<String> {
    let name = dir.file_name()?.to_str()?;
    let rest = name.strip_prefix("models--")?;
    let parts: Vec<&str> = rest.split("--").collect();
    if parts.len() >= 2 {
        Some(format!("{}/{}", parts[0], parts[1..].join("--")))
    } else {
        None
    }
}

fn looks_like_checkpoint_dir(dir: &Path) -> bool {
    discover_checkpoint(dir, None).is_ok()
        || dir.join("config.json").is_file()
        || dir.join("model.safetensors").is_file()
}

fn push_discovered(
    out: &mut Vec<DiscoveredModel>,
    seen: &mut HashSet<String>,
    path: &Path,
    spec: &str,
) {
    let spec = normalize_spec(spec);
    if !seen.insert(spec.clone()) {
        return;
    }
    let bytes = dir_or_file_bytes(path);
    let label = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(spec.as_str())
        .to_string();
    out.push(DiscoveredModel {
        spec,
        path: path.to_path_buf(),
        label,
        bytes,
    });
}

fn dir_or_file_bytes(path: &Path) -> u64 {
    if path.is_file() {
        return fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    let mut total = 0u64;
    if let Ok(entries) = fs::read_dir(path) {
        for ent in entries.flatten() {
            let p = ent.path();
            if p.is_file() {
                total = total.saturating_add(fs::metadata(&p).map(|m| m.len()).unwrap_or(0));
            }
        }
    }
    total
}

/// Open GGUF / SafeTensors metadata and estimate expert/dense sizes.
pub fn probe_checkpoint(spec: &str, path: &Path) -> Result<MoeProbe, String> {
    let file_bytes = dir_or_file_bytes(path);
    let discovered = discover_checkpoint(path, None).ok();
    if let Some(DiscoveredCheckpoint {
        kind: CheckpointKind::Safetensors,
        weights: st,
    }) = discovered.as_ref()
    {
        return probe_safetensors(spec, path, st, file_bytes);
    }
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        || discovered
            .as_ref()
            .is_some_and(|d| d.kind == CheckpointKind::Gguf)
        || path.is_dir()
    {
        if let Ok(probe) = probe_gguf(spec, path, file_bytes) {
            return Ok(probe);
        }
    }
    if path.join("config.json").is_file() || path.is_file() {
        if let Ok(probe) = probe_safetensors(spec, path, path, file_bytes) {
            return Ok(probe);
        }
    }
    Err(format!("could not probe {}", path.display()))
}

fn probe_gguf(spec: &str, path: &Path, file_bytes: u64) -> Result<MoeProbe, String> {
    let file = GgufFile::open(path).map_err(|e| e.to_string())?;
    let cfg = ModelConfig::from_gguf(&file).map_err(|e| e.to_string())?;
    let shape = MoeShape::from_model_config(&cfg);
    let expert_bytes = estimate_gguf_expert_bytes(&file, &cfg);
    let dense_bytes = file_bytes.saturating_sub(
        expert_bytes.saturating_mul(shape.n_experts as u64) * shape.moe_layers().count() as u64,
    );
    Ok(MoeProbe {
        spec: normalize_spec(spec),
        path: path.to_path_buf(),
        name: path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string(),
        shape,
        expert_bytes,
        dense_bytes,
        file_bytes,
    })
}

fn estimate_gguf_expert_bytes(file: &GgufFile, cfg: &ModelConfig) -> u64 {
    let n = cfg.n_experts.max(1);
    for i in cfg.leading_dense_blocks..cfg.n_layers {
        let blk = cfg.gguf_blk(i);
        let mut sum = 0u64;
        for suffix in [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ] {
            if let Some(info) = file.tensor(&format!("blk.{blk}.{suffix}")) {
                sum = sum.saturating_add(info.nbytes() as u64 / n as u64);
            }
        }
        if sum > 0 {
            return sum;
        }
    }
    let elems = (cfg.dim.max(1) * cfg.ff_dim.max(1) * 3) as u64;
    elems.saturating_mul(2)
}

fn probe_safetensors(
    spec: &str,
    root: &Path,
    weights: &Path,
    file_bytes: u64,
) -> Result<MoeProbe, String> {
    let cfg_path = resolve_sidecar(None, weights, "config.json");
    let cfg_path = if cfg_path.is_file() {
        cfg_path
    } else {
        resolve_sidecar(None, root, "config.json")
    };
    if !cfg_path.is_file() {
        return Err(format!("no config.json beside {}", root.display()));
    }
    let hf = load_hf_config(&cfg_path).map_err(|e| e.to_string())?;
    let text = hf.effective_text();
    let n_experts = hf
        .n_routed_experts
        .or(hf.num_local_experts)
        .or(hf.num_experts)
        .or(text.num_experts)
        .unwrap_or(0);
    let n_layers = text.num_hidden_layers.max(1);
    let n_heads = text.num_attention_heads.max(1);
    let n_kv = text.num_key_value_heads.unwrap_or(n_heads).max(1);
    let dim = text.hidden_size.max(1);
    let head_dim = text.head_dim.unwrap_or(dim / n_heads).max(1);
    let n_group = text.n_group.or(hf.n_group).unwrap_or(1).max(1);
    let ff = text
        .moe_intermediate_size
        .or(text.intermediate_size)
        .unwrap_or(dim * 4);
    let shape = MoeShape {
        n_layers,
        n_experts,
        n_expert_groups: n_group,
        leading_dense_blocks: 0,
        n_kv_heads: n_kv,
        n_heads,
        head_dim,
        dim,
    };
    let expert_bytes = if n_experts > 0 {
        (dim * ff * 3 * 2) as u64
    } else {
        0
    };
    let dense_bytes =
        file_bytes.saturating_sub(expert_bytes.saturating_mul(n_experts as u64) * n_layers as u64);
    Ok(MoeProbe {
        spec: normalize_spec(spec),
        path: root.to_path_buf(),
        name: root
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string(),
        shape,
        expert_bytes,
        dense_bytes,
        file_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_shape() -> MoeShape {
        MoeShape {
            n_layers: 3,
            n_experts: 8,
            n_expert_groups: 2,
            leading_dense_blocks: 0,
            n_kv_heads: 2,
            n_heads: 4,
            head_dim: 8,
            dim: 32,
        }
    }

    #[test]
    fn router_groups_and_layer_bands_expand() {
        let shape = tiny_shape();
        let g0 = expand_groups(&["group_0".into()], &shape);
        assert_eq!(g0.len(), 3 * 4, "2 groups of 4 experts × 3 layers");
        assert!(g0.iter().all(|id| id.expert < 4));
        let late = expand_groups(&["late".into()], &shape);
        assert!(late.iter().all(|id| id.layer == 2));
        assert_eq!(late.len(), 8);
        let early = expand_groups(&["early".into()], &shape);
        assert!(early.iter().all(|id| id.layer == 0));
    }

    #[test]
    fn maxmin_keep_set_matches_hand_computed_layer() {
        let catalog = KeepsetCatalog {
            n_layers: 1,
            n_experts: 4,
            topics: BTreeMap::from([
                ("t1".into(), vec![vec![8.0, 7.0, 1.0, 1.0]]),
                ("t2".into(), vec![vec![1.0, 1.0, 1.0, 10.0]]),
            ]),
            profiles: BTreeMap::new(),
        };
        let ids = rank_keep_set(&catalog, &["t1".into(), "t2".into()], 3, RankRule::MaxMin);
        let experts: Vec<u32> = ids.iter().map(|id| id.expert).collect();
        assert_eq!(experts, vec![0, 1, 3]);
    }

    #[test]
    fn catalog_json_counts_prefix_and_profiles() {
        let src = r#"{
            "n_experts": 2,
            "counts_python": [[3.0, 1.0]],
            "profiles": { "backend": ["python"] }
        }"#;
        let cat = parse_catalog(src).unwrap();
        assert!(cat.topics.contains_key("python"));
        assert_eq!(cat.profiles["backend"], vec!["python"]);
        assert_eq!(cat.n_layers, 1);
    }

    #[test]
    fn budget_fits_and_over() {
        let probe = MoeProbe {
            spec: "m".into(),
            path: PathBuf::from("m"),
            name: "m".into(),
            shape: tiny_shape(),
            expert_bytes: 100,
            dense_bytes: 50,
            file_bytes: 1000,
        };
        let ids = expand_groups(&["late".into()], &probe.shape);
        let tight = ResourceSnapshot {
            ram_total_bytes: 200,
            ram_used_bytes: 0,
            gpu_name: "cpu".into(),
            gpu_memory_bytes: 200,
        };
        let over = estimate_budget(&probe, &ids, 32_768, 0, &tight);
        assert!(!over.fits);
        let roomy = ResourceSnapshot {
            ram_total_bytes: 1 << 40,
            ram_used_bytes: 0,
            gpu_name: "gpu".into(),
            gpu_memory_bytes: 1 << 40,
        };
        let ok = estimate_budget(&probe, &ids, 32, 0, &roomy);
        assert!(ok.fits);
        assert_eq!(ok.pin_count, 8);
    }

    #[test]
    fn keep_experts_round_trip() {
        let ids = vec![
            ExpertId {
                layer: 0,
                expert: 1,
            },
            ExpertId {
                layer: 2,
                expert: 0,
            },
        ];
        let compact = keep_experts_from_ids(&ids, 3);
        assert_eq!(compact, vec![vec![1], vec![], vec![0]]);
        assert_eq!(ids_from_keep_experts(&compact), ids);
    }
}
