//! CUDA GPU ISA helpers shared by `build.rs` (via `#[path]`) and unit tests.
//!
//! Accepts env-style tokens (`sm_86`, `8.6`, `86`) and emits nvcc `-gencode` flags.
//! Default fat binary targets RTX 3060 (Ampere / sm_86) and Blackwell GeForce
//! laptop (sm_120).

/// Default fat binary: Ampere consumer (RTX 3060 12GB / sm_86) + Blackwell
/// GeForce laptop (RTX 50-series mobile / sm_120).
pub fn default_cuda_arches() -> Vec<String> {
    ["sm_86", "sm_120"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

pub fn parse_cuda_arches(raw: &str) -> Vec<String> {
    let mut out = Vec::new();
    for part in raw.split(|c: char| c == ',' || c == ';' || c.is_whitespace()) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let Some(sm) = normalize_cuda_arch(part) else {
            continue;
        };
        if !out.iter().any(|a| a == &sm) {
            out.push(sm);
        }
    }
    out
}

/// Accept `sm_86`, `compute_86`, `86`, `8.6`, `12.0`.
pub fn normalize_cuda_arch(raw: &str) -> Option<String> {
    let s = raw.trim().trim_start_matches('+').to_ascii_lowercase();
    if s.is_empty() || s == "native" || s == "auto" {
        return None;
    }
    if let Some(rest) = s.strip_prefix("sm_") {
        if rest.chars().all(|c| c.is_ascii_digit()) && !rest.is_empty() {
            return Some(format!("sm_{rest}"));
        }
        return None;
    }
    if let Some(rest) = s.strip_prefix("compute_") {
        if rest.chars().all(|c| c.is_ascii_digit()) && !rest.is_empty() {
            return Some(format!("sm_{rest}"));
        }
        return None;
    }
    if let Some((major, minor)) = s.split_once('.') {
        if major.chars().all(|c| c.is_ascii_digit())
            && minor.chars().all(|c| c.is_ascii_digit())
            && !major.is_empty()
            && !minor.is_empty()
        {
            return Some(format!("sm_{major}{minor}"));
        }
        return None;
    }
    if s.chars().all(|c| c.is_ascii_digit()) {
        return Some(format!("sm_{s}"));
    }
    None
}

pub fn gencode_flags(arches: &[String]) -> Vec<String> {
    let mut flags = Vec::with_capacity(arches.len() * 2);
    for sm in arches {
        let Some(num) = sm.strip_prefix("sm_") else {
            continue;
        };
        // SASS for the real GPU + matching PTX for minor forward-compat.
        flags.push(format!("-gencode=arch=compute_{num},code=sm_{num}"));
        flags.push(format!("-gencode=arch=compute_{num},code=compute_{num}"));
    }
    flags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_accepts_sm_compute_dotted_and_bare() {
        assert_eq!(normalize_cuda_arch("sm_86").as_deref(), Some("sm_86"));
        assert_eq!(normalize_cuda_arch("SM_120").as_deref(), Some("sm_120"));
        assert_eq!(normalize_cuda_arch("compute_86").as_deref(), Some("sm_86"));
        assert_eq!(normalize_cuda_arch("8.6").as_deref(), Some("sm_86"));
        assert_eq!(normalize_cuda_arch("12.0").as_deref(), Some("sm_120"));
        assert_eq!(normalize_cuda_arch("86").as_deref(), Some("sm_86"));
        assert_eq!(normalize_cuda_arch("+sm_86").as_deref(), Some("sm_86"));
    }

    #[test]
    fn normalize_rejects_native_auto_and_garbage() {
        assert_eq!(normalize_cuda_arch("native"), None);
        assert_eq!(normalize_cuda_arch("auto"), None);
        assert_eq!(normalize_cuda_arch(""), None);
        assert_eq!(normalize_cuda_arch("sm_"), None);
        assert_eq!(normalize_cuda_arch("blackwell"), None);
    }

    #[test]
    fn parse_dedupes_and_splits_common_list_forms() {
        assert_eq!(
            parse_cuda_arches("sm_86,sm_120;8.6 89"),
            vec![
                "sm_86".to_string(),
                "sm_120".to_string(),
                "sm_89".to_string()
            ]
        );
        assert_eq!(
            parse_cuda_arches("sm_86 +sm_120 native auto"),
            vec!["sm_86".to_string(), "sm_120".to_string()]
        );
    }

    #[test]
    fn default_fat_binary_is_3060_and_blackwell_laptop() {
        assert_eq!(
            default_cuda_arches(),
            vec!["sm_86".to_string(), "sm_120".to_string()]
        );
    }

    #[test]
    fn gencode_emits_sass_and_ptx_per_arch() {
        let flags = gencode_flags(&default_cuda_arches());
        assert_eq!(
            flags,
            vec![
                "-gencode=arch=compute_86,code=sm_86".to_string(),
                "-gencode=arch=compute_86,code=compute_86".to_string(),
                "-gencode=arch=compute_120,code=sm_120".to_string(),
                "-gencode=arch=compute_120,code=compute_120".to_string(),
            ]
        );
    }
}
