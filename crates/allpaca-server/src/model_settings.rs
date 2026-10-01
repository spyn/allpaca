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
    path: Option<PathBuf>,
    current: Mutex<ModelSettings>,
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
        let path = Self::path_for(config);
        let current = path.as_deref().map(load).unwrap_or_default();
        if let Some(p) = &path {
            if p.exists() {
                tracing::info!(path = %p.display(), "model settings loaded");
            }
        }
        Self {
            path,
            current: Mutex::new(current),
        }
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn get(&self) -> ModelSettings {
        self.current.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Replace and persist. Written to a temp file and renamed, so a crash
    /// never leaves a half-written file.
    pub fn set(&self, mut settings: ModelSettings) -> Result<(), String> {
        settings.version = SETTINGS_VERSION;
        if let Some(path) = &self.path {
            let json = serde_json::to_string_pretty(&settings).map_err(|e| e.to_string())?;
            let tmp = path.with_extension("json.tmp");
            std::fs::write(&tmp, json).map_err(|e| format!("write {}: {e}", tmp.display()))?;
            std::fs::rename(&tmp, path).map_err(|e| format!("rename to {}: {e}", path.display()))?;
        }
        if let Ok(mut g) = self.current.lock() {
            *g = settings;
        }
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
