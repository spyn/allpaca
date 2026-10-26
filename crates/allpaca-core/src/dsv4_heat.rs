use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use allpaca_gguf::GgufFile;

use crate::model::ModelConfig;

pub(crate) fn sidecar_path(file: &GgufFile) -> Option<PathBuf> {
    let first = file.shard_paths()?.into_iter().next()?;
    let name = first.file_name()?.to_string_lossy();
    Some(first.with_file_name(format!("{name}.allpaca.heat")))
}

const HEAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ModelFingerprint {
    layers: usize,
    experts: usize,
    expert_slot_bytes: usize,
    shards: Vec<u64>,
}

impl ModelFingerprint {
    pub(crate) fn new(
        layers: usize,
        experts: usize,
        expert_slot_bytes: usize,
        shards: Vec<u64>,
    ) -> Self {
        Self {
            layers,
            experts,
            expert_slot_bytes,
            shards,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct HeatEntry {
    layer: usize,
    expert: usize,
    heat: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct TierHeat {
    pub(crate) routes: u64,
    entries: Vec<HeatEntry>,
}

impl TierHeat {
    pub(crate) fn from_map(map: &HashMap<(usize, usize), f32>, routes: u64) -> Self {
        let mut entries: Vec<_> = map
            .iter()
            .filter(|(_, heat)| heat.is_finite() && **heat > 0.0)
            .map(|(&(layer, expert), &heat)| HeatEntry {
                layer,
                expert,
                heat,
            })
            .collect();
        entries.sort_by_key(|entry| (entry.layer, entry.expert));
        Self { routes, entries }
    }

    pub(crate) fn to_map(&self) -> HashMap<(usize, usize), f32> {
        self.entries
            .iter()
            .map(|entry| ((entry.layer, entry.expert), entry.heat))
            .collect()
    }

    pub(crate) fn ranked(&self, limit: usize) -> Vec<(usize, usize)> {
        let mut entries: Vec<_> = self.entries.iter().collect();
        entries.sort_by(|a, b| {
            b.heat
                .total_cmp(&a.heat)
                .then_with(|| (a.layer, a.expert).cmp(&(b.layer, b.expert)))
        });
        entries
            .into_iter()
            .take(limit)
            .map(|entry| (entry.layer, entry.expert))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ExpertHeat {
    version: u32,
    model: ModelFingerprint,
    pub(crate) gpu: TierHeat,
    pub(crate) ram: TierHeat,
}

impl ExpertHeat {
    pub(crate) fn new(model: ModelFingerprint) -> Self {
        Self {
            version: HEAT_VERSION,
            model,
            gpu: TierHeat::default(),
            ram: TierHeat::default(),
        }
    }

    pub(crate) fn for_model(
        file: &GgufFile,
        cfg: &ModelConfig,
        expert_slot_bytes: usize,
    ) -> Option<Self> {
        let shards: Option<Vec<u64>> = file
            .shard_paths()?
            .iter()
            .map(|path| std::fs::metadata(path).ok().map(|meta| meta.len()))
            .collect();
        Some(Self::new(ModelFingerprint::new(
            cfg.n_layers,
            cfg.n_experts,
            expert_slot_bytes,
            shards?,
        )))
    }

    pub(crate) fn load_for_model(
        path: &Path,
        file: &GgufFile,
        cfg: &ModelConfig,
        expert_slot_bytes: usize,
    ) -> Option<Self> {
        let base = Self::for_model(file, cfg, expert_slot_bytes)?;
        Self::load(path, &base.model)
    }

    pub(crate) fn load(path: &Path, model: &ModelFingerprint) -> Option<Self> {
        if std::env::var("ALLPACA_DSV4_HEAT").is_ok_and(|v| v == "0") {
            return None;
        }
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return None,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "DeepSeek-V4 expert heat unreadable; cold start");
                return None;
            }
        };
        let heat: Self = match serde_json::from_str(&text) {
            Ok(heat) => heat,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "DeepSeek-V4 expert heat invalid; cold start");
                return None;
            }
        };
        let valid = heat.version == HEAT_VERSION
            && &heat.model == model
            && heat
                .gpu
                .entries
                .iter()
                .chain(&heat.ram.entries)
                .all(|entry| {
                    entry.layer < heat.model.layers
                        && entry.expert < heat.model.experts
                        && entry.heat.is_finite()
                        && entry.heat > 0.0
                });
        if !valid {
            tracing::warn!(path = %path.display(), "DeepSeek-V4 expert heat incompatible; cold start");
            return None;
        }
        Some(heat)
    }

    pub(crate) fn save(&self, path: &Path) -> io::Result<()> {
        if std::env::var("ALLPACA_DSV4_HEAT").is_ok_and(|v| v == "0") {
            return Ok(());
        }
        let json = serde_json::to_vec_pretty(self).map_err(io::Error::other)?;
        let tmp = path.with_extension(format!("heat.{}.tmp", std::process::id()));
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::ffi::OsString;
    use std::sync::Mutex;

    static HEAT_ENV_LOCK: Mutex<()> = Mutex::new(());

    struct HeatEnvDisabled {
        previous: Option<OsString>,
    }

    impl HeatEnvDisabled {
        fn set() -> Self {
            let previous = std::env::var_os("ALLPACA_DSV4_HEAT");
            // SAFETY: tests using this variable hold HEAT_ENV_LOCK for the guard's lifetime.
            unsafe { std::env::set_var("ALLPACA_DSV4_HEAT", "0") };
            Self { previous }
        }
    }

    impl Drop for HeatEnvDisabled {
        fn drop(&mut self) {
            // SAFETY: the same lock is still held by this test.
            unsafe {
                match self.previous.take() {
                    Some(value) => std::env::set_var("ALLPACA_DSV4_HEAT", value),
                    None => std::env::remove_var("ALLPACA_DSV4_HEAT"),
                }
            }
        }
    }

    fn fingerprint() -> ModelFingerprint {
        ModelFingerprint::new(43, 256, 13_400_000, vec![100, 200])
    }

    #[test]
    fn sidecar_round_trip_restores_both_tiers_and_ranks_globally() {
        let _lock = HEAT_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("allpaca-dsv4-heat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf.allpaca.heat");
        let fp = fingerprint();
        let mut saved = ExpertHeat::new(fp.clone());
        saved.gpu = TierHeat::from_map(
            &HashMap::from([((0, 1), 5.0), ((2, 0), 9.0), ((0, 0), 9.0), ((1, 3), 1.0)]),
            2_048,
        );
        saved.ram = TierHeat::from_map(&HashMap::from([((7, 2), 4.0), ((3, 9), 8.0)]), 4_096);

        saved.save(&path).unwrap();
        let loaded = ExpertHeat::load(&path, &fp).expect("valid sidecar");

        assert_eq!(loaded, saved);
        assert_eq!(loaded.gpu.ranked(3), vec![(0, 0), (2, 0), (0, 1)]);
        assert_eq!(loaded.ram.ranked(1), vec![(3, 9)]);
        assert_eq!(loaded.gpu.routes, 2_048);
        assert_eq!(loaded.ram.routes, 4_096);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn disabled_heat_neither_reads_nor_writes_and_bad_profiles_are_cold() {
        let _lock = HEAT_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir =
            std::env::temp_dir().join(format!("allpaca-dsv4-heat-opt-out-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf.allpaca.heat");
        let fp = fingerprint();
        let mut original = ExpertHeat::new(fp.clone());
        original.gpu = TierHeat::from_map(&HashMap::from([((0, 1), 5.0)]), 10);
        original.save(&path).unwrap();

        {
            let _disabled = HeatEnvDisabled::set();
            assert!(ExpertHeat::load(&path, &fp).is_none());
            let mut changed = original.clone();
            changed.gpu.routes = 999;
            changed.save(&path).unwrap();
        }
        assert_eq!(ExpertHeat::load(&path, &fp), Some(original));

        let incompatible = ModelFingerprint::new(1, 1, 1, vec![1]);
        assert!(ExpertHeat::load(&path, &incompatible).is_none());
        std::fs::write(&path, "not json").unwrap();
        assert!(ExpertHeat::load(&path, &fp).is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
