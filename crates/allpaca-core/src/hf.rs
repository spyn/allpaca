//! HuggingFace Hub download and `org/name` model specs.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use allpaca_gguf::{is_dspark_filename, parse_split_filename};
use allpaca_safetensors::{discover_checkpoint, CheckpointKind};
use allpaca_types::{AllpacaError, FatalError};
use serde::Deserialize;

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
const USER_AGENT: &str = "allpaca/0.1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfRef {
    pub repo: String,
    pub revision: String,
    pub filename: Option<String>,
}

#[derive(Debug, Clone)]
pub struct HubOptions {
    pub cache_dir: PathBuf,
    pub token: Option<String>,
    pub revision: Option<String>,
    pub offline: bool,
    pub endpoint: String,
}

impl HubOptions {
    pub fn from_env() -> Self {
        Self {
            cache_dir: default_cache_dir(),
            token: hub_token_from_env(),
            revision: None,
            offline: env_truthy("HF_HUB_OFFLINE"),
            endpoint: std::env::var("HF_ENDPOINT")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| DEFAULT_ENDPOINT.into()),
        }
    }

    pub fn with_revision(mut self, revision: Option<String>) -> Self {
        if revision.is_some() {
            self.revision = revision;
        }
        self
    }
}

/// Local file/dir, or a Hub snapshot directory after download.
pub fn resolve_model_spec(
    spec: &str,
    opts: &HubOptions,
) -> Result<(PathBuf, Option<String>), AllpacaError> {
    let trimmed = spec.trim();
    let as_path = PathBuf::from(trimmed);
    if as_path.exists() {
        return Ok((as_path, None));
    }
    let Some(mut href) = parse_hf_ref(trimmed, opts.revision.as_deref()) else {
        return Ok((as_path, None));
    };
    if let Some(rev) = &opts.revision {
        if href.revision == "main" {
            href.revision = rev.clone();
        }
    }
    let dir = ensure_hub_snapshot(&href, opts)?;
    Ok((dir, href.filename))
}

pub fn parse_hf_ref(spec: &str, default_revision: Option<&str>) -> Option<HfRef> {
    let raw = spec.trim();
    if raw.is_empty() {
        return None;
    }
    if Path::new(raw).exists() {
        return None;
    }
    let norm = normalize_spec(raw);
    if looks_like_filesystem(&norm) {
        return None;
    }
    let (rest, filename_colon) = split_filename(&norm);
    let (rest, rev_at) = split_revision(rest);
    let mut parts: Vec<&str> = rest.split('/').filter(|p| !p.is_empty()).collect();
    if parts.first().copied() == Some("models") && parts.len() > 3 {
        parts.remove(0);
    }
    let mut filename = filename_colon.map(str::to_string);
    let mut revision = rev_at.map(str::to_string);
    if parts.len() >= 4 && matches!(parts[2], "tree" | "resolve" | "blob") {
        revision = Some(parts[3].to_string());
        if parts.len() > 4 {
            filename = Some(parts[4..].join("/"));
        }
        parts.truncate(2);
    }
    if parts.len() != 2 {
        return None;
    }
    if !valid_component(parts[0]) || !valid_component(parts[1]) {
        return None;
    }
    if filename.is_none() && second_looks_like_weight_file(parts[1]) {
        return None;
    }
    Some(HfRef {
        repo: format!("{}/{}", parts[0], parts[1]),
        revision: revision
            .or_else(|| default_revision.map(str::to_string))
            .unwrap_or_else(|| "main".into()),
        filename,
    })
}

fn normalize_spec(raw: &str) -> String {
    let s = raw
        .strip_prefix("hf://")
        .unwrap_or(raw)
        .trim_start_matches("https://huggingface.co/")
        .trim_start_matches("http://huggingface.co/")
        .trim_start_matches("https://hf.co/")
        .trim_start_matches("http://hf.co/")
        .trim_start_matches('/');
    s.to_string()
}

fn looks_like_filesystem(s: &str) -> bool {
    s.starts_with('/')
        || s.starts_with('.')
        || s.contains('\\')
        || (s.len() >= 2 && s.as_bytes()[0].is_ascii_alphabetic() && s.as_bytes()[1] == b':')
}

fn valid_component(s: &str) -> bool {
    !s.is_empty() && s != "." && s != ".." && !s.contains('\\')
}

fn second_looks_like_weight_file(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.ends_with(".gguf") || lower.ends_with(".safetensors") || lower.ends_with(".onnx")
}

fn split_filename(s: &str) -> (&str, Option<&str>) {
    if looks_like_filesystem(s) {
        return (s, None);
    }
    match s.rsplit_once(':') {
        Some((repo, file)) if repo.contains('/') && !file.is_empty() => (repo, Some(file)),
        _ => (s, None),
    }
}

fn split_revision(s: &str) -> (&str, Option<&str>) {
    match s.rsplit_once('@') {
        Some((repo, rev)) if repo.contains('/') && !rev.is_empty() && !rev.contains('/') => {
            (repo, Some(rev))
        }
        _ => (s, None),
    }
}

fn default_cache_dir() -> PathBuf {
    if let Ok(v) = std::env::var("ALLPACA_HF_CACHE").or_else(|_| std::env::var("HF_HUB_CACHE")) {
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    if let Ok(v) = std::env::var("HF_HOME") {
        if !v.is_empty() {
            return PathBuf::from(v).join("hub");
        }
    }
    if let Ok(v) = std::env::var("XDG_CACHE_HOME") {
        if !v.is_empty() {
            return PathBuf::from(v).join("huggingface").join("hub");
        }
    }
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"));
    match home {
        Ok(h) => PathBuf::from(h)
            .join(".cache")
            .join("huggingface")
            .join("hub"),
        Err(_) => PathBuf::from(".allpaca-hf-cache"),
    }
}

fn hub_token_from_env() -> Option<String> {
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ALLPACA_HF_TOKEN"] {
        if let Ok(v) = std::env::var(key) {
            let t = v.trim().to_string();
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()?;
    let path = PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("token");
    let src = fs::read_to_string(path).ok()?;
    let t = src.trim().to_string();
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
}

fn env_truthy(key: &str) -> bool {
    matches!(
        std::env::var(key).ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
    )
}

fn repo_cache_root(cache: &Path, repo: &str) -> PathBuf {
    cache.join(format!("models--{}", repo.replace('/', "--")))
}

fn snapshot_complete(dir: &Path, preferred: Option<&str>) -> bool {
    discover_checkpoint(dir, preferred)
        .map(|d| match d.kind {
            CheckpointKind::Safetensors => dir.join("config.json").is_file(),
            CheckpointKind::Gguf | CheckpointKind::Onnx => d.weights.is_file(),
        })
        .unwrap_or(false)
}

fn find_cached_snapshot(opts: &HubOptions, href: &HfRef) -> Option<PathBuf> {
    let root = repo_cache_root(&opts.cache_dir, &href.repo);
    let refs = root.join("refs").join(&href.revision);
    if let Ok(sha) = fs::read_to_string(&refs) {
        let snap = root.join("snapshots").join(sha.trim());
        if snapshot_complete(&snap, href.filename.as_deref()) {
            return Some(snap);
        }
    }
    let named = root.join("snapshots").join(&href.revision);
    if snapshot_complete(&named, href.filename.as_deref()) {
        return Some(named);
    }
    let snaps = root.join("snapshots");
    let rd = fs::read_dir(&snaps).ok()?;
    let mut dirs: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.into_iter()
        .rev()
        .find(|p| snapshot_complete(p, href.filename.as_deref()))
}

fn ensure_hub_snapshot(href: &HfRef, opts: &HubOptions) -> Result<PathBuf, AllpacaError> {
    if let Some(dir) = find_cached_snapshot(opts, href) {
        tracing::info!(repo = %href.repo, path = %dir.display(), "using HuggingFace cache");
        return Ok(dir);
    }
    if opts.offline {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "HuggingFace {}@{} not in cache (HF_HUB_OFFLINE)",
            href.repo, href.revision
        ))));
    }
    pull_hub_snapshot(href, opts)
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    sha: Option<String>,
    #[serde(default)]
    siblings: Vec<Sibling>,
}

#[derive(Debug, Deserialize)]
struct Sibling {
    rfilename: String,
}

#[derive(Debug, Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: Option<String>,
    path: Option<String>,
}

fn pull_hub_snapshot(href: &HfRef, opts: &HubOptions) -> Result<PathBuf, AllpacaError> {
    let client = http_client()?;
    let info = fetch_model_info(&client, opts, href)?;
    let sha = info.sha.clone().unwrap_or_else(|| href.revision.clone());
    let mut names: Vec<String> = info.siblings.iter().map(|s| s.rfilename.clone()).collect();
    if names.is_empty() {
        names = fetch_tree_files(&client, opts, href)?;
    }
    let mut wanted = select_repo_files(&names, href.filename.as_deref())?;
    let root = repo_cache_root(&opts.cache_dir, &href.repo);
    let snap = root.join("snapshots").join(&sha);
    fs::create_dir_all(&snap).map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "HuggingFace cache {}: {e}",
            snap.display()
        )))
    })?;
    for rel in &wanted {
        download_repo_file(&client, opts, href, &sha, rel, &snap)?;
    }
    if wanted.iter().any(|f| f.ends_with(".index.json")) {
        let index_path = snap.join("model.safetensors.index.json");
        if index_path.is_file() {
            for shard in index_shard_names(&index_path)? {
                if !wanted.iter().any(|w| w == &shard) {
                    download_repo_file(&client, opts, href, &sha, &shard, &snap)?;
                    wanted.push(shard);
                }
            }
        }
    }
    let refs_dir = root.join("refs");
    let _ = fs::create_dir_all(&refs_dir);
    let _ = fs::write(refs_dir.join(&href.revision), format!("{sha}\n"));
    if !snapshot_complete(&snap, href.filename.as_deref()) {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "HuggingFace {} did not yield a loadable checkpoint",
            href.repo
        ))));
    }
    tracing::info!(
        repo = %href.repo,
        sha = %sha,
        files = wanted.len(),
        "downloaded HuggingFace snapshot"
    );
    Ok(snap)
}

fn fetch_model_info(
    client: &reqwest::blocking::Client,
    opts: &HubOptions,
    href: &HfRef,
) -> Result<ModelInfo, AllpacaError> {
    let url = format!(
        "{}/api/models/{}/revision/{}",
        opts.endpoint.trim_end_matches('/'),
        href.repo,
        href.revision
    );
    let mut req = client.get(&url);
    if let Some(t) = &opts.token {
        req = req.bearer_auth(t);
    }
    let resp = req.send().map_err(|e| hub_http_err(&href.repo, &e))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(hub_status_err(
            &href.repo,
            status.as_u16(),
            opts.token.is_some(),
        ));
    }
    resp.json().map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "HuggingFace {} API JSON: {e}",
            href.repo
        )))
    })
}

fn fetch_tree_files(
    client: &reqwest::blocking::Client,
    opts: &HubOptions,
    href: &HfRef,
) -> Result<Vec<String>, AllpacaError> {
    let url = format!(
        "{}/api/models/{}/tree/{}?recursive=1",
        opts.endpoint.trim_end_matches('/'),
        href.repo,
        href.revision
    );
    let mut req = client.get(&url);
    if let Some(t) = &opts.token {
        req = req.bearer_auth(t);
    }
    let resp = req.send().map_err(|e| hub_http_err(&href.repo, &e))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(hub_status_err(
            &href.repo,
            status.as_u16(),
            opts.token.is_some(),
        ));
    }
    let tree: Vec<TreeEntry> = resp.json().map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "HuggingFace {} tree JSON: {e}",
            href.repo
        )))
    })?;
    Ok(tree
        .into_iter()
        .filter(|e| e.kind.as_deref() == Some("file"))
        .filter_map(|e| e.path)
        .collect())
}

pub fn select_repo_files(
    siblings: &[String],
    preferred: Option<&str>,
) -> Result<Vec<String>, AllpacaError> {
    let mut out = Vec::new();
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
    ] {
        if siblings.iter().any(|s| s == name) {
            out.push(name.to_string());
        }
    }
    if let Some(pref) = preferred {
        if let Some(picked) = siblings
            .iter()
            .find(|s| *s == pref || s.ends_with(pref))
            .cloned()
        {
            if picked.to_ascii_lowercase().ends_with(".gguf") {
                for shard in expand_gguf_split_siblings(siblings, &picked) {
                    push_unique(&mut out, shard);
                }
                return Ok(out);
            }
            push_unique(&mut out, picked);
            if pref.ends_with(".index.json") {
                // shards filled after the index is on disk
            }
            return Ok(out);
        }
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "HuggingFace repo has no file {pref}"
        ))));
    }
    if siblings.iter().any(|s| s == "model.safetensors.index.json") {
        push_unique(&mut out, "model.safetensors.index.json".into());
        return Ok(out);
    }
    if siblings.iter().any(|s| s == "model.safetensors") {
        push_unique(&mut out, "model.safetensors".into());
        return Ok(out);
    }
    let st: Vec<String> = siblings
        .iter()
        .filter(|s| {
            let l = s.to_ascii_lowercase();
            l.ends_with(".safetensors")
                && !l.contains("vision")
                && !l.contains("mmproj")
                && !l.contains("audio")
        })
        .cloned()
        .collect();
    if st.len() == 1 {
        if let Some(one) = st.into_iter().next() {
            push_unique(&mut out, one);
        }
        return Ok(out);
    }
    if st.len() > 1 {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(
            "HuggingFace repo has multiple SafeTensors files; pass org/name:file.safetensors"
                .into(),
        )));
    }
    let mut ggufs: Vec<String> = siblings
        .iter()
        .filter(|s| {
            let base = std::path::Path::new(s.as_str())
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(s.as_str());
            s.to_ascii_lowercase().ends_with(".gguf") && !is_dspark_filename(base)
        })
        .cloned()
        .collect();
    ggufs.sort_by_key(|n| rank_gguf(n));
    if let Some(g) = ggufs.first().cloned() {
        for shard in expand_gguf_split_siblings(siblings, &g) {
            push_unique(&mut out, shard);
        }
        return Ok(out);
    }
    if siblings.iter().any(|s| s == "model.onnx") {
        push_unique(&mut out, "model.onnx".into());
        return Ok(out);
    }
    Err(AllpacaError::Fatal(FatalError::ModelLoad(
        "HuggingFace repo has no SafeTensors, GGUF, or ONNX weights".into(),
    )))
}

fn rank_gguf(name: &str) -> (u8, u8, String) {
    let l = name.to_ascii_lowercase();
    let base = std::path::Path::new(name)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(name);
    if is_dspark_filename(base) {
        return (9, 9, l);
    }
    let split = if l.contains("00001-of") {
        0
    } else if parse_split_filename(base).is_some() {
        1
    } else {
        2
    };
    let quant = if l.contains("ud-q8") || l.contains("q8_k") {
        0
    } else if l.contains("q8_0") {
        1
    } else if l.contains("q4_k_m") {
        2
    } else if l.contains("q4_k") {
        3
    } else if l.contains("q5_k") {
        4
    } else if l.contains("f16") || l.contains("fp16") {
        5
    } else {
        6
    };
    (split, quant, l)
}

fn expand_gguf_split_siblings(siblings: &[String], picked: &str) -> Vec<String> {
    let base = std::path::Path::new(picked)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(picked);
    let Some((stem, _, total)) = parse_split_filename(base) else {
        return vec![picked.to_string()];
    };
    let prefix = picked.strip_suffix(base).unwrap_or("").to_string();
    let mut out = Vec::new();
    for i in 1..=total {
        let padded = format!("{stem}-{i:05}-of-{total:05}.gguf");
        let loose = format!("{stem}-{i}-of-{total}.gguf");
        if let Some(hit) = siblings.iter().find(|s| {
            let b = std::path::Path::new(s.as_str())
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(s.as_str());
            b == padded
                || b == loose
                || *s == &format!("{prefix}{padded}")
                || *s == &format!("{prefix}{loose}")
        }) {
            out.push(hit.clone());
        }
    }
    if out.len() == total as usize {
        out
    } else {
        vec![picked.to_string()]
    }
}

fn index_shard_names(index_path: &Path) -> Result<Vec<String>, AllpacaError> {
    let src = fs::read_to_string(index_path).map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "SafeTensors index {}: {e}",
            index_path.display()
        )))
    })?;
    let v: serde_json::Value = serde_json::from_str(&src).map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "SafeTensors index {}: {e}",
            index_path.display()
        )))
    })?;
    let map = v
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "SafeTensors index {} missing weight_map",
                index_path.display()
            )))
        })?;
    let mut names: Vec<String> = map
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    Ok(names)
}

fn download_repo_file(
    client: &reqwest::blocking::Client,
    opts: &HubOptions,
    href: &HfRef,
    sha: &str,
    rel: &str,
    snap: &Path,
) -> Result<(), AllpacaError> {
    let dest = snap.join(rel);
    if dest.is_file() {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "HuggingFace cache {}: {e}",
                parent.display()
            )))
        })?;
    }
    let url = format!(
        "{}/{}/resolve/{}/{}",
        opts.endpoint.trim_end_matches('/'),
        href.repo,
        sha,
        encode_path(rel)
    );
    tracing::info!(file = rel, "downloading HuggingFace file");
    let mut req = client.get(&url);
    if let Some(t) = &opts.token {
        req = req.bearer_auth(t);
    }
    let mut resp = req.send().map_err(|e| hub_http_err(&href.repo, &e))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(hub_status_err(
            &href.repo,
            status.as_u16(),
            opts.token.is_some(),
        ));
    }
    let tmp = dest.with_extension("part");
    let mut file = File::create(&tmp).map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "HuggingFace write {}: {e}",
            tmp.display()
        )))
    })?;
    resp.copy_to(&mut file).map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "HuggingFace download {rel}: {e}"
        )))
    })?;
    file.flush().map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "HuggingFace flush {rel}: {e}"
        )))
    })?;
    fs::rename(&tmp, &dest).map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "HuggingFace rename {}: {e}",
            dest.display()
        )))
    })?;
    Ok(())
}

fn encode_path(s: &str) -> String {
    s.split('/')
        .map(|p| {
            let mut out = String::new();
            for b in p.bytes() {
                match b {
                    b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                        out.push(b as char);
                    }
                    _ => out.push_str(&format!("%{b:02X}")),
                }
            }
            out
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn http_client() -> Result<reqwest::blocking::Client, AllpacaError> {
    reqwest::blocking::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(None)
        .connect_timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| AllpacaError::Fatal(FatalError::ModelLoad(format!("HTTP client: {e}"))))
}

fn hub_http_err(repo: &str, err: &reqwest::Error) -> AllpacaError {
    AllpacaError::Fatal(FatalError::ModelLoad(format!("HuggingFace {repo}: {err}")))
}

fn hub_status_err(repo: &str, status: u16, has_token: bool) -> AllpacaError {
    let hint = if status == 401 || status == 403 {
        if has_token {
            " (token rejected or gated model)"
        } else {
            " (set HF_TOKEN for gated models)"
        }
    } else {
        ""
    };
    AllpacaError::Fatal(FatalError::ModelLoad(format!(
        "HuggingFace {repo} HTTP {status}{hint}"
    )))
}

fn push_unique(out: &mut Vec<String>, name: String) {
    if !out.iter().any(|s| s == &name) {
        out.push(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_org_name_and_url_and_revision() {
        let a = parse_hf_ref("Qwen/Qwen2.5-0.5B-Instruct", None).unwrap();
        assert_eq!(a.repo, "Qwen/Qwen2.5-0.5B-Instruct");
        assert_eq!(a.revision, "main");
        let b = parse_hf_ref("hf://Qwen/Qwen2.5-0.5B-Instruct@v1.0", None).unwrap();
        assert_eq!(b.revision, "v1.0");
        let c = parse_hf_ref(
            "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF:foo.gguf",
            None,
        )
        .unwrap();
        assert_eq!(c.repo, "bartowski/Llama-3.2-1B-Instruct-GGUF");
        assert_eq!(c.filename.as_deref(), Some("foo.gguf"));
        assert!(parse_hf_ref("models/foo.gguf", None).is_none());
        assert!(parse_hf_ref("/abs/path/model.gguf", None).is_none());
    }

    #[test]
    fn select_prefers_index_then_single_then_gguf() {
        let sib = [
            "config.json".into(),
            "tokenizer.json".into(),
            "model.safetensors.index.json".into(),
            "README.md".into(),
        ];
        let files = select_repo_files(&sib, None).unwrap();
        assert!(files.contains(&"config.json".into()));
        assert!(files.contains(&"model.safetensors.index.json".into()));
        let gguf = [
            "a-f16.gguf".into(),
            "a-Q8_0.gguf".into(),
            "config.json".into(),
        ];
        let files = select_repo_files(&gguf, None).unwrap();
        assert!(files.iter().any(|f| f.contains("Q8_0")));
        let picked = select_repo_files(&gguf, Some("a-f16.gguf")).unwrap();
        assert!(picked.contains(&"a-f16.gguf".into()));
        let split = [
            "UD-Q8_K_XL/model-00001-of-00002.gguf".into(),
            "UD-Q8_K_XL/model-00002-of-00002.gguf".into(),
            "dspark-model-Q8_0.gguf".into(),
            "config.json".into(),
        ];
        let files = select_repo_files(&split, None).unwrap();
        assert!(files.iter().any(|f| f.contains("00001-of")));
        assert!(files.iter().any(|f| f.contains("00002-of")));
        assert!(!files.iter().any(|f| f.contains("dspark")));
    }

    #[test]
    fn cache_hit_offline_uses_snapshot() {
        let root = std::env::temp_dir().join("allpaca_hf_cache_hit");
        let _ = fs::remove_dir_all(&root);
        let snap = repo_cache_root(&root, "acme/tiny")
            .join("snapshots")
            .join("deadbeef");
        fs::create_dir_all(&snap).unwrap();
        fs::write(
            snap.join("config.json"),
            r#"{"vocab_size":4,"hidden_size":8,"num_hidden_layers":1}"#,
        )
        .unwrap();
        allpaca_safetensors::write_safetensors(
            &snap.join("model.safetensors"),
            &[allpaca_safetensors::SafetensorsWriteTensor {
                name: "w".into(),
                dims: vec![1],
                dtype: allpaca_safetensors::SafetensorsDtype::F32,
                data: &[1.0],
            }],
        )
        .unwrap();
        let opts = HubOptions {
            cache_dir: root.clone(),
            token: None,
            revision: None,
            offline: true,
            endpoint: "http://127.0.0.1:1".into(),
        };
        let href = HfRef {
            repo: "acme/tiny".into(),
            revision: "main".into(),
            filename: None,
        };
        let dir = ensure_hub_snapshot(&href, &opts).expect("offline cache hit");
        assert_eq!(dir, snap);
        let _ = fs::remove_dir_all(root);
    }
}
