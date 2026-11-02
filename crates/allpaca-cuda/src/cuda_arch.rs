//! CUDA GPU ISA helpers shared by `build.rs` (via `#[path]`) and unit tests.
//!
//! Accepts env-style tokens (`sm_86`, `8.6`, `86`, `sm_37`) and emits nvcc
//! `-gencode` flags.
//!
//! Dual toolchain:
//! - Modern CUDA 12+: default fat binary Ampere (sm_86 / RTX 3060), Ada (sm_89),
//!   Blackwell GeForce laptop (sm_120). Kepler is filtered out.
//! - CUDA ≤11.x (driver 470, Kepler): include `sm_35` (Quadro K6000) and `sm_37`
//!   (Tesla K80) when the toolkit still accepts them
//!   (`scripts/fetch-cuda-toolkit.sh` → cuda-11.4). A `sm_37` binary does not
//!   run on `sm_35`, so neither stands in for the other.

#![allow(dead_code)] // lib links this for tests; build.rs uses the same file via #[path]

/// Default fat binary for modern toolkits: Ampere consumer (RTX 3060 / sm_86),
/// Ada laptop (sm_89), and Blackwell GeForce laptop (sm_120).
pub fn default_cuda_arches() -> Vec<String> {
    ["sm_86", "sm_89", "sm_120"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

/// Default arches for a toolkit major version. CUDA 12+ drops Kepler; ≤11 keeps
/// `sm_35` and `sm_37` ahead of the modern fat list so Kepler boxes build a
/// usable binary.
pub fn default_cuda_arches_for_toolkit(major: Option<u32>) -> Vec<String> {
    let mut out = Vec::new();
    if toolkit_supports_kepler(major) {
        out.push("sm_35".into());
        out.push("sm_37".into());
    }
    for sm in default_cuda_arches() {
        if !out.iter().any(|a| a == &sm) {
            out.push(sm);
        }
    }
    // Drop arches the toolkit cannot emit (e.g. sm_120 on CUDA 11).
    filter_arches_for_toolkit(out, major)
}

/// CUDA 12 removed sm_35 and sm_37; CUDA 11 cannot target Blackwell sm_120.
pub fn toolkit_supports_kepler(major: Option<u32>) -> bool {
    match major {
        Some(m) => m < 12,
        None => true, // unknown / 11.x extract: keep Kepler
    }
}

pub fn filter_arches_for_toolkit(arches: Vec<String>, major: Option<u32>) -> Vec<String> {
    arches
        .into_iter()
        .filter(|sm| arch_ok_for_toolkit(sm, major))
        .collect()
}

fn arch_ok_for_toolkit(sm: &str, major: Option<u32>) -> bool {
    let Some(num) = sm.strip_prefix("sm_") else {
        return false;
    };
    let Ok(n) = num.parse::<u32>() else {
        return false;
    };
    match major {
        // CUDA 12+ dropped Kepler (sm_35, sm_37) and older.
        Some(m) if m >= 12 && n < 50 => false,
        // CUDA 11.x tops out around sm_86; sm_89/sm_120 need newer toolkits.
        Some(m) if m < 12 && n >= 89 => false,
        _ => true,
    }
}

/// Map a product name (nvidia-smi) to an sm_* when compute_cap is missing
/// (driver 470 on Tesla K80).
pub fn sm_arch_from_product_name(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    if lower.contains("k80") || lower.contains("tesla k80") {
        return Some("sm_37");
    }
    None
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

/// Accept `sm_86`, `compute_86`, `86`, `8.6`, `12.0`, `3.7`.
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

/// Format a live device compute capability as `sm_86` / `sm_120` / `sm_37`.
pub fn sm_arch_from_capability(major: i32, minor: i32) -> Option<String> {
    // CUDA major.minor uses a single minor digit (8.6→sm_86, 12.0→sm_120).
    if major <= 0 || major > 99 || minor < 0 || minor > 9 {
        return None;
    }
    Some(format!("sm_{major}{minor}"))
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
        assert_eq!(normalize_cuda_arch("3.7").as_deref(), Some("sm_37"));
        assert_eq!(normalize_cuda_arch("sm_37").as_deref(), Some("sm_37"));
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
    fn default_fat_binary_is_3060_ada_and_blackwell_laptop() {
        assert_eq!(
            default_cuda_arches(),
            vec![
                "sm_86".to_string(),
                "sm_89".to_string(),
                "sm_120".to_string()
            ]
        );
    }

    #[test]
    fn toolkit_11_includes_kepler_drops_blackwell() {
        let arches = default_cuda_arches_for_toolkit(Some(11));
        assert!(arches.iter().any(|a| a == "sm_35"));
        assert!(arches.iter().any(|a| a == "sm_37"));
        assert!(arches.iter().any(|a| a == "sm_86"));
        assert!(!arches.iter().any(|a| a == "sm_120"));
        assert!(!arches.iter().any(|a| a == "sm_89"));
    }

    #[test]
    fn toolkit_12_drops_kepler_keeps_blackwell() {
        let arches = default_cuda_arches_for_toolkit(Some(12));
        assert!(!arches.iter().any(|a| a == "sm_35" || a == "sm_37"));
        assert_eq!(
            arches,
            vec![
                "sm_86".to_string(),
                "sm_89".to_string(),
                "sm_120".to_string()
            ]
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
                "-gencode=arch=compute_89,code=sm_89".to_string(),
                "-gencode=arch=compute_89,code=compute_89".to_string(),
                "-gencode=arch=compute_120,code=sm_120".to_string(),
                "-gencode=arch=compute_120,code=compute_120".to_string(),
            ]
        );
    }

    #[test]
    fn sm_arch_from_capability_covers_kepler_ampere_ada_blackwell() {
        assert_eq!(sm_arch_from_capability(3, 7).as_deref(), Some("sm_37"));
        assert_eq!(sm_arch_from_capability(8, 6).as_deref(), Some("sm_86"));
        assert_eq!(sm_arch_from_capability(8, 9).as_deref(), Some("sm_89"));
        assert_eq!(sm_arch_from_capability(12, 0).as_deref(), Some("sm_120"));
        assert_eq!(sm_arch_from_capability(0, 0), None);
        assert_eq!(sm_arch_from_product_name("Tesla K80"), Some("sm_37"));
    }
}
