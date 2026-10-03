//! Finds the models the WebUI can switch to: a recursive scan of the models
//! folder for GGUF files, with what their metadata says about them.
//!
//! Split GGUFs (`name-00001-of-00003.gguf`) count once, under their first
//! shard, with the shards' total size. Files are classified by
//! `general.type` (`model`, `mmproj`, `imatrix`) and by name (`mtp-*` MTP
//! draft heads, `dspark-*` sidecars); a model lists the vision projectors
//! and MTP drafts in its own folder as companions.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use allpaca_gguf::{GgufFile, GgufValue};
use allpaca_types::Config;
use serde::Serialize;

use crate::model_settings::SettingsStore;

const MAX_DEPTH: usize = 8;
const MAX_DESCRIPTION: usize = 400;

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Companions {
    /// Vision projectors (`general.type = mmproj`) in the same folder.
    pub vision: Vec<String>,
    /// MTP draft heads (`mtp-*.gguf`) in the same folder.
    pub mtp: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SavedSettings {
    pub context_tokens: Option<u64>,
    pub disabled_experts: usize,
    pub named_experts: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ModelEntry {
    /// Absolute path (the first shard for split models).
    pub path: String,
    /// Path relative to the models folder, `/`-separated.
    pub rel_path: String,
    /// `model`, `mmproj`, `mtp`, `imatrix`, `dspark` or `other`.
    pub kind: String,
    pub size_bytes: u64,
    pub shards: u32,
    pub name: Option<String>,
    pub basename: Option<String>,
    pub finetune: Option<String>,
    pub architecture: Option<String>,
    pub size_label: Option<String>,
    /// Quantization from the file name (`UD-Q5_K_XL`), else from
    /// `general.file_type`.
    pub quant: Option<String>,
    pub context_length: Option<u64>,
    pub block_count: Option<u64>,
    pub embedding_length: Option<u64>,
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
    /// MTP / NextN layers inside the model file itself.
    pub nextn_layers: Option<u64>,
    pub license: Option<String>,
    pub quantized_by: Option<String>,
    pub description: Option<String>,
    pub companions: Companions,
    /// A saved `<model>.allpaca.json` for this model.
    pub settings: Option<SavedSettings>,
    /// Why the header could not be read, if it could not.
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScanResult {
    pub models_dir: Option<String>,
    pub entries: Vec<ModelEntry>,
}

/// The folder to scan for `config`.
pub fn models_dir(config: &Config) -> Option<PathBuf> {
    if let Some(dir) = &config.model.models_dir {
        return Some(dir.clone());
    }
    if let Some(path) = &config.model.path {
        for anc in path.ancestors().skip(1) {
            if anc
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.eq_ignore_ascii_case("models"))
            {
                return Some(anc.to_path_buf());
            }
        }
    }
    let local = PathBuf::from("models");
    local.is_dir().then_some(local)
}

/// `name-00002-of-00003.gguf` -> (`name`, 2, 3).
fn split_parts(file_name: &str) -> Option<(&str, u32, u32)> {
    let stem = file_name.strip_suffix(".gguf")?;
    let (head, total) = stem.rsplit_once("-of-")?;
    let (prefix, index) = head.rsplit_once('-')?;
    if index.len() != 5 || total.len() != 5 {
        return None;
    }
    Some((prefix, index.parse().ok()?, total.parse().ok()?))
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > MAX_DEPTH {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut items: Vec<_> = rd.flatten().collect();
    items.sort_by_key(|e| e.file_name());
    for ent in items {
        let name = ent.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        let Ok(ft) = ent.file_type() else { continue };
        if ft.is_symlink() {
            continue;
        }
        let path = ent.path();
        if ft.is_dir() {
            walk(&path, depth + 1, out);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            out.push(path);
        }
    }
}

/// Quant label from a file name: the last `-`/`.`-separated part that looks
/// like one (`Q5_K_XL`, `UD-Q5_K_XL`, `IQ4_XS`, `F16`, `Q4_0_ROCMFP4_…`).
fn quant_from_name(stem: &str) -> Option<String> {
    let parts: Vec<&str> = stem.split(['-', '.']).collect();
    let mut found = None;
    for (i, p) in parts.iter().enumerate() {
        let up = p.to_ascii_uppercase();
        let b = up.as_bytes();
        let looks = (b.len() >= 2 && b[0] == b'Q' && b[1].is_ascii_digit())
            || (b.len() >= 3 && b[0] == b'I' && b[1] == b'Q' && b[2].is_ascii_digit())
            || (b.len() >= 3 && b[0] == b'T' && b[1] == b'Q' && b[2].is_ascii_digit())
            || matches!(up.as_str(), "F16" | "F32" | "BF16" | "MXFP4")
            || up.contains("ROCMFP4");
        if looks {
            let ud = i > 0 && parts[i - 1].eq_ignore_ascii_case("UD");
            found = Some(if ud { format!("UD-{up}") } else { up });
        }
    }
    found
}

/// llama.cpp `general.file_type` names.
fn file_type_name(ft: u32) -> Option<&'static str> {
    Some(match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        // ROCmFP4 presets (charlie12345/ROCmFPX): the name is the recipe, the
        // tensors are ggml type 100 (dual-scale) and/or 101 (FAST).
        100 => "ROCmFP4",
        101 => "ROCmFP4-LEAN",
        102 => "ROCmFP4-COHERENT",
        103 => "ROCmFP4-FAST",
        104 => "ROCmFP4-FAST-COHERENT",
        105 => "ROCmFP4-STRIX",
        106 => "ROCmFP4-STRIX-LEAN",
        _ => return None,
    })
}

fn md_str(md: &HashMap<String, GgufValue>, key: &str) -> Option<String> {
    md.get(key)
        .and_then(GgufValue::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn md_u64(md: &HashMap<String, GgufValue>, key: &str) -> Option<u64> {
    md.get(key).and_then(GgufValue::as_u64)
}

/// Header facts for one model file (cached; settings are looked up fresh).
fn read_entry(path: &Path, rel: &str, size_bytes: u64, shards: u32) -> ModelEntry {
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let stem = split_parts(file_name)
        .map(|(p, _, _)| p.to_string())
        .unwrap_or_else(|| file_name.trim_end_matches(".gguf").to_string());
    let mut e = ModelEntry {
        path: path.display().to_string(),
        rel_path: rel.to_string(),
        size_bytes,
        shards,
        quant: quant_from_name(&stem),
        ..ModelEntry::default()
    };
    let lower = file_name.to_ascii_lowercase();
    let gguf = match GgufFile::open(path) {
        Ok(g) => g,
        Err(err) => {
            e.kind = "other".into();
            e.error = Some(err.to_string());
            return e;
        }
    };
    let md = &gguf.metadata;
    let gtype = md_str(md, "general.type");
    e.kind = if gtype.as_deref() == Some("mmproj") || md_str(md, "general.architecture").as_deref() == Some("clip") {
        "mmproj"
    } else if gtype.as_deref() == Some("imatrix") || md.keys().any(|k| k.starts_with("imatrix.")) {
        "imatrix"
    } else if lower.starts_with("mtp-") {
        "mtp"
    } else if lower.starts_with("dspark-") {
        "dspark"
    } else {
        "model"
    }
    .to_string();
    e.name = md_str(md, "general.name");
    e.basename = md_str(md, "general.basename");
    e.finetune = md_str(md, "general.finetune");
    e.architecture = md_str(md, "general.architecture");
    e.size_label = md_str(md, "general.size_label");
    e.license = md_str(md, "general.license");
    e.quantized_by = md_str(md, "general.quantized_by");
    e.description = md_str(md, "general.description").map(|d| {
        if d.chars().count() > MAX_DESCRIPTION {
            format!("{}…", d.chars().take(MAX_DESCRIPTION).collect::<String>())
        } else {
            d
        }
    });
    // The header's preset wins for ROCmFP4 files: their names are not
    // reliable (a "STRIX" folder holding a COHERENT file).
    let file_type = md.get("general.file_type").and_then(GgufValue::as_u32);
    if e.quant.is_none() || file_type.is_some_and(|ft| (100..=106).contains(&ft)) {
        if let Some(name) = file_type.and_then(file_type_name) {
            e.quant = Some(name.to_string());
        }
    }
    if let Some(arch) = e.architecture.clone() {
        let k = |s: &str| format!("{arch}.{s}");
        e.context_length = md_u64(md, &k("context_length"));
        e.block_count = md_u64(md, &k("block_count"));
        e.embedding_length = md_u64(md, &k("embedding_length"));
        e.expert_count = md_u64(md, &k("expert_count")).filter(|n| *n > 0);
        e.expert_used_count = md_u64(md, &k("expert_used_count")).filter(|n| *n > 0);
        e.nextn_layers = md_u64(md, &k("nextn_predict_layers")).filter(|n| *n > 0);
    }
    e
}

type Cache = HashMap<PathBuf, (u64, Option<SystemTime>, ModelEntry)>;
static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

/// Scan `dir`. Header reads are cached by file size and mtime.
pub fn scan(dir: &Path) -> Vec<ModelEntry> {
    let mut files = Vec::new();
    walk(dir, 0, &mut files);
    // Group split shards under their first shard.
    let mut groups: BTreeMap<PathBuf, (PathBuf, u64, u32)> = BTreeMap::new();
    for f in &files {
        let size = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
        let name = f.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let key = match split_parts(name) {
            Some((prefix, _, total)) => f.with_file_name(format!("{prefix}-00001-of-{total:05}.gguf")),
            None => f.clone(),
        };
        let g = groups.entry(key.clone()).or_insert((key, 0, 0));
        g.1 += size;
        g.2 += 1;
    }
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let cache = cache.get_or_insert_with(HashMap::new);
    let mut out = Vec::new();
    for (first, (_, size, shards)) in groups {
        if !first.exists() {
            continue; // stray later shards without their first
        }
        let mtime = std::fs::metadata(&first).and_then(|m| m.modified()).ok();
        let rel = first
            .strip_prefix(dir)
            .unwrap_or(&first)
            .to_string_lossy()
            .replace('\\', "/");
        let entry = match cache.get(&first) {
            Some((s, m, e)) if *s == size && *m == mtime => e.clone(),
            _ => {
                let e = read_entry(&first, &rel, size, shards);
                cache.insert(first.clone(), (size, mtime, e.clone()));
                e
            }
        };
        out.push(entry);
    }
    // Companions: projectors and MTP drafts in the model's own folder.
    let parent = |e: &ModelEntry| Path::new(&e.path).parent().map(Path::to_path_buf);
    let sidecars: Vec<(Option<PathBuf>, String, String)> = out
        .iter()
        .filter(|e| e.kind == "mmproj" || e.kind == "mtp")
        .map(|e| (parent(e), e.kind.clone(), e.path.clone()))
        .collect();
    for e in out.iter_mut().filter(|e| e.kind == "model") {
        let dir = parent(e);
        for (d, kind, path) in &sidecars {
            if *d == dir {
                match kind.as_str() {
                    "mmproj" => e.companions.vision.push(path.clone()),
                    _ => e.companions.mtp.push(path.clone()),
                }
            }
        }
        let mut cfg = Config::compiled_defaults();
        cfg.model.path = Some(PathBuf::from(&e.path));
        e.settings = SettingsStore::path_for(&cfg)
            .filter(|p| p.exists())
            .map(|_| {
                let s = SettingsStore::open(&cfg).get();
                SavedSettings {
                    context_tokens: s.context_tokens,
                    disabled_experts: s.disabled_experts.values().map(Vec::len).sum(),
                    named_experts: s.expert_names.len(),
                }
            });
    }
    out.sort_by(|a, b| {
        let rank = |k: &str| if k == "model" { 0 } else { 1 };
        rank(&a.kind).cmp(&rank(&b.kind)).then_with(|| a.rel_path.to_lowercase().cmp(&b.rel_path.to_lowercase()))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_names_and_quant_labels() {
        assert_eq!(
            split_parts("Qwen3-Coder-Next-UD-Q5_K_XL-00002-of-00003.gguf"),
            Some(("Qwen3-Coder-Next-UD-Q5_K_XL", 2, 3))
        );
        assert_eq!(split_parts("model.gguf"), None);
        assert_eq!(quant_from_name("Qwen3-Coder-Next-UD-Q5_K_XL").as_deref(), Some("UD-Q5_K_XL"));
        assert_eq!(quant_from_name("qwen2.5-coder-0.5b-instruct-q8_0").as_deref(), Some("Q8_0"));
        assert_eq!(quant_from_name("Llama-3.2-3B-Instruct-f16").as_deref(), Some("F16"));
        assert_eq!(
            quant_from_name("Qwen3.8-27B-Q4_0_ROCMFP4_COHERENT").as_deref(),
            Some("Q4_0_ROCMFP4_COHERENT")
        );
        assert_eq!(quant_from_name("Qwen3-Next"), None);
        assert_eq!(file_type_name(17), Some("Q5_K_M"));
        assert_eq!(file_type_name(103), Some("ROCmFP4-FAST"));
    }

    fn write_gguf(path: &Path, meta: Vec<(&str, GgufValue)>) {
        use allpaca_gguf::{write_gguf, GgufWriteTensor};
        let meta: std::collections::HashMap<String, GgufValue> =
            meta.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        let data = vec![0.0f32; 4];
        let t = [GgufWriteTensor {
            name: "x.weight".into(),
            dims: vec![4],
            data: &data,
        }];
        write_gguf(path, &meta, &t).unwrap();
    }

    #[test]
    fn scan_groups_shards_classifies_and_reads_metadata() {
        let root = std::env::temp_dir().join(format!("allpaca-scan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let q = root.join("Qwen.Coder").join("UD-Q5_K_XL");
        let v = root.join("Vision27B");
        std::fs::create_dir_all(&q).unwrap();
        std::fs::create_dir_all(&v).unwrap();
        std::fs::create_dir_all(root.join(".hidden")).unwrap();
        use GgufValue::{String as S, Uint32 as U};
        write_gguf(
            &q.join("Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00002.gguf"),
            vec![
                ("general.architecture", S("qwen3next".into())),
                ("general.type", S("model".into())),
                ("general.name", S("Qwen3-Coder-Next".into())),
                ("general.size_label", S("512x2.5B".into())),
                ("general.license", S("apache-2.0".into())),
                ("qwen3next.context_length", U(262_144)),
                ("qwen3next.block_count", U(48)),
                ("qwen3next.expert_count", U(512)),
                ("qwen3next.expert_used_count", U(10)),
                ("split.count", U(2)),
            ],
        );
        write_gguf(
            &q.join("Qwen3-Coder-Next-UD-Q5_K_XL-00002-of-00002.gguf"),
            vec![("split.count", U(2)), ("split.no", U(1))],
        );
        write_gguf(
            &v.join("Model-27B-Q4_K_M.gguf"),
            vec![
                ("general.architecture", S("qwen35".into())),
                ("general.name", S("Model 27B".into())),
                ("general.file_type", U(102)),
                ("qwen35.nextn_predict_layers", U(1)),
            ],
        );
        write_gguf(
            &v.join("mmproj-F16.gguf"),
            vec![("general.architecture", S("clip".into())), ("general.type", S("mmproj".into()))],
        );
        write_gguf(&v.join("mtp-Model-27B-Q4_0.gguf"), vec![("general.architecture", S("qwen35".into()))]);
        write_gguf(&v.join("imatrix.gguf"), vec![("general.type", S("imatrix".into()))]);
        write_gguf(&root.join(".hidden").join("skip.gguf"), vec![]);
        std::fs::write(v.join("broken.gguf"), b"not a gguf").unwrap();

        let entries = scan(&root);
        let by = |rel: &str| entries.iter().find(|e| e.rel_path == rel).unwrap_or_else(|| panic!("{rel} missing: {entries:#?}"));
        assert!(entries.iter().all(|e| !e.rel_path.contains(".hidden")));
        assert!(entries.iter().all(|e| !e.rel_path.contains("00002-of")));
        let qwen = by("Qwen.Coder/UD-Q5_K_XL/Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00002.gguf");
        assert_eq!(qwen.kind, "model");
        assert_eq!(qwen.shards, 2);
        assert!(qwen.size_bytes > 0);
        assert_eq!(qwen.name.as_deref(), Some("Qwen3-Coder-Next"));
        assert_eq!(qwen.quant.as_deref(), Some("UD-Q5_K_XL"));
        assert_eq!((qwen.context_length, qwen.expert_count, qwen.expert_used_count), (Some(262_144), Some(512), Some(10)));
        let m27 = by("Vision27B/Model-27B-Q4_K_M.gguf");
        assert_eq!(m27.kind, "model");
        // The file name says Q4_K_M; the header's ROCmFP4 preset wins.
        assert_eq!(m27.quant.as_deref(), Some("ROCmFP4-COHERENT"));
        assert_eq!(m27.nextn_layers, Some(1));
        assert_eq!(m27.companions.vision.len(), 1);
        assert_eq!(m27.companions.mtp.len(), 1);
        assert_eq!(by("Vision27B/mmproj-F16.gguf").kind, "mmproj");
        assert_eq!(by("Vision27B/mtp-Model-27B-Q4_0.gguf").kind, "mtp");
        assert_eq!(by("Vision27B/imatrix.gguf").kind, "imatrix");
        let broken = by("Vision27B/broken.gguf");
        assert_eq!(broken.kind, "other");
        assert!(broken.error.is_some());
        // Models first.
        assert_eq!(entries[0].kind, "model");
        let _ = std::fs::remove_dir_all(root);
    }
}
