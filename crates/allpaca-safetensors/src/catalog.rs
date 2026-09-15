use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use allpaca_types::{AllpacaError, FatalError};
use serde::Deserialize;

use crate::names::{aliases, WeightKey};
use crate::parser::{SafetensorsFile, TensorInfo};

#[derive(Debug, Deserialize)]
struct WeightIndex {
    #[serde(default)]
    weight_map: HashMap<String, String>,
}

/// One or more SafeTensors files (HuggingFace `model.safetensors.index.json` shards).
pub struct SafetensorsCatalog {
    files: Vec<SafetensorsFile>,
    /// tensor name → file index
    location: HashMap<String, usize>,
}

impl SafetensorsCatalog {
    pub fn open(path: &Path) -> Result<Self, AllpacaError> {
        if path.is_dir() {
            let weights = discover_weight_path(path)?;
            return Self::open(&weights);
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if name.ends_with(".index.json") {
            return Self::open_index(path);
        }
        let file = SafetensorsFile::open(path)?;
        Self::from_files(vec![file])
    }

    pub fn from_file(file: SafetensorsFile) -> Self {
        let mut location = HashMap::new();
        for name in file.tensors.keys() {
            location.insert(name.clone(), 0);
        }
        Self {
            files: vec![file],
            location,
        }
    }

    fn from_files(files: Vec<SafetensorsFile>) -> Result<Self, AllpacaError> {
        let mut location = HashMap::new();
        for (idx, file) in files.iter().enumerate() {
            for name in file.tensors.keys() {
                if location.insert(name.clone(), idx).is_some() {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "duplicate SafeTensors tensor {name}"
                    ))));
                }
            }
        }
        Ok(Self { files, location })
    }

    fn open_index(index_path: &Path) -> Result<Self, AllpacaError> {
        let src = std::fs::read_to_string(index_path).map_err(|e| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "SafeTensors index {}: {e}",
                index_path.display()
            )))
        })?;
        let index: WeightIndex = serde_json::from_str(&src).map_err(|e| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "SafeTensors index {}: {e}",
                index_path.display()
            )))
        })?;
        if index.weight_map.is_empty() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "SafeTensors index {} has empty weight_map",
                index_path.display()
            ))));
        }
        let dir = index_path.parent().unwrap_or_else(|| Path::new("."));
        let mut unique: Vec<String> = index
            .weight_map
            .values()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        unique.sort();
        let mut files = Vec::with_capacity(unique.len());
        for rel in unique {
            files.push(SafetensorsFile::open(&dir.join(rel))?);
        }
        Self::from_files(files)
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.location.contains_key(name)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        let idx = *self.location.get(name)?;
        self.files.get(idx)?.tensor(name)
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, AllpacaError> {
        let idx = self.location.get(name).copied().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!("missing tensor {name}")))
        })?;
        self.files[idx].tensor_f32(name)
    }

    /// First present alias, or Fatal listing every candidate.
    pub fn tensor_f32_first(&self, names: &[String]) -> Result<Vec<f32>, AllpacaError> {
        for n in names {
            if self.has_tensor(n) {
                return self.tensor_f32(n);
            }
        }
        Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "missing tensor (tried {})",
            names.join(", ")
        ))))
    }

    pub fn tensor_f32_key(&self, key: WeightKey) -> Result<Vec<f32>, AllpacaError> {
        self.tensor_f32_first(&aliases(key))
    }

    pub fn tensor_f32_key_optional(&self, key: WeightKey) -> Result<Vec<f32>, AllpacaError> {
        let names = aliases(key);
        for n in &names {
            if self.has_tensor(n) {
                return self.tensor_f32(n);
            }
        }
        Ok(Vec::new())
    }
}

/// HuggingFace snapshot directory → weight file (`index.json` or `.safetensors`).
pub fn discover_weight_path(dir: &Path) -> Result<PathBuf, AllpacaError> {
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        return Ok(index);
    }
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(single);
    }
    let mut shards = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "SafeTensors dir {}: {e}",
            dir.display()
        )))
    })?;
    for ent in entries {
        let ent = ent.map_err(|e| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "SafeTensors dir {}: {e}",
                dir.display()
            )))
        })?;
        let path = ent.path();
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !name.ends_with(".safetensors") {
            continue;
        }
        if name.contains("vision") || name.contains("mmproj") || name.contains("audio") {
            continue;
        }
        shards.push(path);
    }
    shards.sort();
    match shards.len() {
        0 => Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "no SafeTensors weights in {}",
            dir.display()
        )))),
        1 => Ok(shards.remove(0)),
        _ => Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "multiple SafeTensors files in {} without model.safetensors.index.json",
            dir.display()
        )))),
    }
}

/// GGUF in a snapshot directory (Hub GGUF repos).
pub fn discover_gguf_path(dir: &Path, preferred: Option<&str>) -> Result<PathBuf, AllpacaError> {
    if let Some(name) = preferred {
        let p = dir.join(name);
        if p.is_file() {
            return Ok(p);
        }
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "GGUF {name} not in {}",
            dir.display()
        ))));
    }
    let mut found = collect_gguf_files(dir, 1);
    found.retain(|p| {
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let l = name.to_ascii_lowercase();
        !l.contains("dspark") && !l.contains("dflash")
    });
    found.sort_by_key(|p| rank_gguf_name(p));
    found.into_iter().next().ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "no GGUF weights in {}",
            dir.display()
        )))
    })
}

fn collect_gguf_files(dir: &Path, nested: u32) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    for ent in entries.flatten() {
        let path = ent.path();
        if path.is_dir() && nested > 0 {
            found.extend(collect_gguf_files(&path, nested - 1));
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            found.push(path);
        }
    }
    found
}

fn rank_gguf_name(path: &Path) -> (u8, u8, String) {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let split = if name.contains("00001-of") {
        0
    } else if name.contains("-of-") {
        1
    } else {
        2
    };
    let rank = if name.contains("ud-q8") || name.contains("q8_k") {
        0
    } else if name.contains("q8_0") {
        1
    } else if name.contains("q4_k_m") {
        2
    } else if name.contains("q4_k") {
        3
    } else if name.contains("q5_k") {
        4
    } else if name.contains("q6_k") {
        5
    } else if name.contains("f16") || name.contains("fp16") {
        6
    } else {
        7
    };
    (split, rank, name)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointKind {
    Safetensors,
    Gguf,
    Onnx,
}

#[derive(Debug, Clone)]
pub struct DiscoveredCheckpoint {
    pub kind: CheckpointKind,
    pub weights: PathBuf,
}

/// File or HuggingFace snapshot directory.
pub fn discover_checkpoint(
    path: &Path,
    preferred_filename: Option<&str>,
) -> Result<DiscoveredCheckpoint, AllpacaError> {
    if path.is_file() {
        let kind = match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => CheckpointKind::Safetensors,
            Some("onnx") => CheckpointKind::Onnx,
            Some("json") if path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.ends_with(".index.json")) =>
            {
                CheckpointKind::Safetensors
            }
            _ => CheckpointKind::Gguf,
        };
        return Ok(DiscoveredCheckpoint {
            kind,
            weights: path.to_path_buf(),
        });
    }
    if !path.is_dir() {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "model path not found: {}",
            path.display()
        ))));
    }
    if let Some(name) = preferred_filename {
        let explicit = path.join(name);
        if explicit.is_file() {
            return discover_checkpoint(&explicit, None);
        }
    }
    if path.join("model.safetensors.index.json").is_file() || path.join("model.safetensors").is_file()
    {
        return Ok(DiscoveredCheckpoint {
            kind: CheckpointKind::Safetensors,
            weights: discover_weight_path(path)?,
        });
    }
    if let Ok(weights) = discover_weight_path(path) {
        return Ok(DiscoveredCheckpoint {
            kind: CheckpointKind::Safetensors,
            weights,
        });
    }
    let onnx = path.join("model.onnx");
    if onnx.is_file() {
        return Ok(DiscoveredCheckpoint {
            kind: CheckpointKind::Onnx,
            weights: onnx,
        });
    }
    Ok(DiscoveredCheckpoint {
        kind: CheckpointKind::Gguf,
        weights: discover_gguf_path(path, preferred_filename)?,
    })
}
