//! Per-model operator settings, kept as JSON next to the model
//! (`<model path>.allpaca.json`) and applied whenever that model loads:
//! context size (as the KV VRAM budget), disabled experts, expert names.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use allpaca_core::Engine;
use allpaca_types::Config;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelSettings {
    pub version: u32,
    /// Context length the KV budget was sized for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_tokens: Option<u64>,
    /// `kv.vram_bytes` for `context_tokens` on this model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_vram_bytes: Option<u64>,
    /// Layer index -> disabled routed expert indices.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub disabled_experts: BTreeMap<u32, Vec<u32>>,
    /// `"layer:expert"` -> operator-chosen name.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub expert_names: BTreeMap<String, String>,
    /// Layer index -> where its routed experts go: `"local"` or
    /// `"rpc:<host:port>/<gpu>"`. Applied when the model loads.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub expert_homes: BTreeMap<u32, String>,
}

pub const SETTINGS_VERSION: u32 = 1;

impl ModelSettings {
    /// Disabled sets as the engine takes them (dense, one entry per layer).
    pub fn disabled_dense(&self, n_layers: usize) -> Result<Vec<Vec<u32>>, String> {
        let mut out = vec![Vec::new(); n_layers];
        for (&layer, ids) in &self.disabled_experts {
            let l = layer as usize;
            if l >= n_layers {
                return Err(format!("disabled_experts layer {l} >= model layers {n_layers}"));
            }
            let mut ids = ids.clone();
            ids.sort_unstable();
            ids.dedup();
            out[l] = ids;
        }
        Ok(out)
    }

    /// Apply the context size to a config about to build an engine.
    pub fn apply_to_config(&self, config: &mut Config) {
        if let Some(bytes) = self.kv_vram_bytes.filter(|b| *b > 0) {
            config.kv.vram_bytes = bytes as usize;
        }
        // Pins belong to the model: a config built for another model must
        // not keep them.
        config.rpc.expert_homes = self.expert_homes_config();
    }

    /// The pins as `rpc.expert_homes` carries them.
    pub fn expert_homes_config(&self) -> BTreeMap<String, String> {
        self.expert_homes
            .iter()
            .map(|(layer, home)| (layer.to_string(), home.clone()))
            .collect()
    }

    /// Apply the disabled experts to a loaded engine. Logs and skips when
    /// they do not fit the model (a different model at the same path).
    pub fn apply_to_engine(&self, engine: &Engine) {
        if self.disabled_experts.is_empty() {
            return;
        }
        let Some(ctl) = engine.expert_controls() else {
            return;
        };
        let applied = self
            .disabled_dense(ctl.shape().n_layers)
            .and_then(|d| ctl.set_disabled(&d).map_err(|e| e.to_string()));
        match applied {
            Ok(()) => tracing::info!(
                layers = self.disabled_experts.len(),
                experts = self.disabled_experts.values().map(Vec::len).sum::<usize>(),
                "model settings: disabled experts applied"
            ),
            Err(e) => tracing::warn!(error = %e, "model settings: disabled experts not applied"),
        }
    }
}

/// The settings file for the served model. In memory only when the model has
/// no path (synthetic).
pub struct SettingsStore {
    /// The served model's settings file and its contents; switches with
    /// the model.
    inner: Mutex<(Option<PathBuf>, ModelSettings)>,
}

impl SettingsStore {
    pub fn path_for(config: &Config) -> Option<PathBuf> {
        if config.model.synthetic {
            return None;
        }
        let model = config.model.path.as_ref()?;
        let name = model.file_name()?.to_string_lossy().into_owned();
        Some(model.with_file_name(format!("{name}.allpaca.json")))
    }

    /// Open the settings for `config`'s model. A missing file is empty
    /// settings; an unreadable one is logged and left untouched until saved.
    pub fn open(config: &Config) -> Self {
        Self {
            inner: Mutex::new(Self::read(config)),
        }
    }

    fn read(config: &Config) -> (Option<PathBuf>, ModelSettings) {
        let path = Self::path_for(config);
        let current = path.as_deref().map(load).unwrap_or_default();
        if let Some(p) = &path {
            if p.exists() {
                tracing::info!(path = %p.display(), "model settings loaded");
            }
        }
        (path, current)
    }

    /// Point at `config`'s model (after a model switch).
    pub fn switch(&self, config: &Config) {
        let next = Self::read(config);
        if let Ok(mut g) = self.inner.lock() {
            *g = next;
        }
    }

    pub fn path(&self) -> Option<PathBuf> {
        self.inner.lock().ok().and_then(|g| g.0.clone())
    }

    pub fn get(&self) -> ModelSettings {
        self.inner.lock().map(|g| g.1.clone()).unwrap_or_default()
    }

    /// Replace and persist. Written to a temp file and renamed, so a crash
    /// never leaves a half-written file.
    pub fn set(&self, mut settings: ModelSettings) -> Result<(), String> {
        settings.version = SETTINGS_VERSION;
        let mut g = self.inner.lock().map_err(|_| "settings lock".to_string())?;
        if let Some(path) = &g.0 {
            let json = serde_json::to_string_pretty(&settings).map_err(|e| e.to_string())?;
            let tmp = path.with_extension("json.tmp");
            std::fs::write(&tmp, json).map_err(|e| format!("write {}: {e}", tmp.display()))?;
            std::fs::rename(&tmp, path).map_err(|e| format!("rename to {}: {e}", path.display()))?;
        }
        g.1 = settings;
        Ok(())
    }
}

fn load(path: &Path) -> ModelSettings {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
            tracing::warn!(path = %path.display(), error = %e, "model settings unreadable; using defaults");
            ModelSettings::default()
        }),
        Err(_) => ModelSettings::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_pins_reach_the_config_and_do_not_outlive_their_model() {
        let mut config = Config::compiled_defaults();
        let mut pinned = ModelSettings::default();
        pinned.expert_homes.insert(3, "local".into());
        pinned.expert_homes.insert(12, "rpc:192.168.69.2:50052/1".into());
        pinned.apply_to_config(&mut config);
        assert_eq!(config.rpc.expert_homes.get("3").map(String::as_str), Some("local"));
        assert_eq!(
            config.rpc.expert_homes.get("12").map(String::as_str),
            Some("rpc:192.168.69.2:50052/1")
        );
        // The pins survive the settings file.
        let json = serde_json::to_string(&pinned).unwrap();
        assert_eq!(serde_json::from_str::<ModelSettings>(&json).unwrap(), pinned);
        // Another model's (empty) settings clear them.
        ModelSettings::default().apply_to_config(&mut config);
        assert!(config.rpc.expert_homes.is_empty());
    }

    #[test]
    fn settings_path_sits_next_to_the_model() {
        let mut cfg = Config::compiled_defaults();
        cfg.model.path = Some(PathBuf::from("models/Qwen.Coder/UD-Q5_K_XL"));
        assert_eq!(
            SettingsStore::path_for(&cfg),
            Some(PathBuf::from("models/Qwen.Coder/UD-Q5_K_XL.allpaca.json"))
        );
        cfg.model.synthetic = true;
        assert_eq!(SettingsStore::path_for(&cfg), None);
    }

    #[test]
    fn round_trips_through_the_file() {
        let dir = std::env::temp_dir().join(format!("allpaca-settings-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut cfg = Config::compiled_defaults();
        cfg.model.path = Some(dir.join("m.gguf"));
        let store = SettingsStore::open(&cfg);
        assert_eq!(store.get(), ModelSettings::default());
        let mut s = ModelSettings {
            context_tokens: Some(65_536),
            kv_vram_bytes: Some(1 << 30),
            ..ModelSettings::default()
        };
        s.disabled_experts.insert(3, vec![7, 2]);
        s.expert_names.insert("3:7".into(), "rust lifetimes".into());
        store.set(s.clone()).unwrap();
        let again = SettingsStore::open(&cfg).get();
        assert_eq!(again.version, SETTINGS_VERSION);
        assert_eq!(again.disabled_experts, s.disabled_experts);
        assert_eq!(again.expert_names, s.expert_names);
        assert_eq!(again.disabled_dense(4).unwrap(), vec![vec![], vec![], vec![], vec![2, 7]]);
        assert!(again.disabled_dense(3).is_err());
        let mut c2 = Config::compiled_defaults();
        again.apply_to_config(&mut c2);
        assert_eq!(c2.kv.vram_bytes, 1 << 30);
        let _ = std::fs::remove_dir_all(dir);
    }
}
