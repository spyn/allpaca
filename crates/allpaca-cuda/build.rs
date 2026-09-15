//! Probe the CUDA toolkit and, when present, compile `cuda/` kernels and link them.
//!
//! Without a toolkit, `--features cuda` still compiles: Rust uses host stubs and
//! `device_count() == 0`. Set `CUDA_PATH` or rely on PATH `nvcc`, `/usr/local/cuda`,
//! or a user-local extract from `scripts/fetch-cuda-toolkit.sh`
//! (`~/.cache/allpaca/cuda-sdk/cuda-11.4`).
//!
//! GPU ISAs: `CUDA_ARCHES` / `ALLPACA_CUDA_ARCHES` / `TORCH_CUDA_ARCH_LIST`-style
//! lists (e.g. `37` or `sm_37`), else live NVIDIA GPUs (Tesla K80 → `sm_37`), else
//! a fat default that includes Kepler (`sm_37`) for driver-470 boxes. CUDA 12+
//! toolkits drop Kepler; those arches are filtered out automatically.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../cuda/src/kernels.cu");
    println!("cargo:rerun-if-changed=../../cuda/include/allpaca_cuda_kernels.h");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ARCHES");
    println!("cargo:rerun-if-env-changed=ALLPACA_CUDA_ARCHES");
    println!("cargo:rerun-if-env-changed=TORCH_CUDA_ARCH_LIST");
    println!("cargo:rerun-if-env-changed=ALLPACA_CUDA_SDK");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rustc-check-cfg=cfg(allpaca_cuda_runtime)");

    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let Some(cuda_root) = find_cuda_root() else {
        println!(
            "cargo:warning=CUDA toolkit not found; cuda feature is stub-only. Run scripts/fetch-cuda-toolkit.sh (K80/driver 470 needs 11.4) or set CUDA_PATH."
        );
        return;
    };

    let arches = cuda_arches(&cuda_root);
    println!(
        "cargo:warning=CUDA toolkit {} arches={}",
        cuda_root.display(),
        arches.join(",")
    );

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    if let Err(e) = build_kernels(&cuda_root, &out, &arches) {
        println!("cargo:warning=CUDA kernel build failed ({e}); cuda feature is stub-only.");
        return;
    }

    println!("cargo:rustc-link-search=native={}", out.display());
    let lib_dir = cuda_root.join("lib").join("x64");
    if lib_dir.is_dir() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    } else {
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_root.join("lib64").display()
        );
        // Some user-local extracts keep libs under lib/ not lib64/.
        let lib = cuda_root.join("lib");
        if lib.is_dir() {
            println!("cargo:rustc-link-search=native={}", lib.display());
        }
    }
    println!("cargo:rustc-link-lib=static=allpaca_cuda_kernels");
    println!("cargo:rustc-link-lib=cudart");
    if cfg!(windows) {
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
    println!("cargo:rustc-cfg=allpaca_cuda_runtime");
}

fn find_cuda_root() -> Option<PathBuf> {
    for key in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(p) = env::var(key) {
            let p = PathBuf::from(p.trim_end_matches(['\\', '/']));
            if cuda_looks_valid(&p) {
                return Some(p);
            }
        }
    }
    if let Some(from_nvcc) = cuda_root_from_nvcc() {
        return Some(from_nvcc);
    }
    for p in cached_cuda_roots() {
        if cuda_looks_valid(&p) {
            return Some(p);
        }
    }
    let base = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
    if let Ok(entries) = std::fs::read_dir(&base) {
        let mut vers: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        vers.sort();
        vers.reverse();
        for p in vers {
            if cuda_looks_valid(&p) {
                return Some(p);
            }
        }
    }
    for p in ["/usr/local/cuda", "/usr/local/cuda-11.4", "/usr/local/cuda-11.8", "/opt/cuda"] {
        let p = PathBuf::from(p);
        if cuda_looks_valid(&p) {
            return Some(p);
        }
    }
    None
}

fn cached_cuda_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(p) = env::var("ALLPACA_CUDA_SDK") {
        roots.push(PathBuf::from(p.trim_end_matches(['\\', '/'])));
    }
    if let Ok(home) = env::var("HOME") {
        let base = PathBuf::from(home).join(".cache/allpaca/cuda-sdk");
        // Prefer Kepler-compatible 11.4 on driver-470 boxes.
        for name in ["cuda-11.4", "cuda-11.8", "cuda-11.7", "cuda"] {
            roots.push(base.join(name));
        }
        if let Ok(entries) = std::fs::read_dir(&base) {
            let mut vers: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
            vers.sort();
            vers.reverse();
            roots.extend(vers);
        }
    }
    roots
}

fn cuda_root_from_nvcc() -> Option<PathBuf> {
    let out = Command::new("nvcc").arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let which = Command::new("where")
        .arg("nvcc")
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout);
            s.lines().next().map(|l| PathBuf::from(l.trim()))
        })
        .or_else(|| {
            Command::new("which")
                .arg("nvcc")
                .output()
                .ok()
                .and_then(|o| {
                    let s = String::from_utf8_lossy(&o.stdout);
                    s.lines().next().map(|l| PathBuf::from(l.trim()))
                })
        })?;
    let bin = which.parent()?;
    let root = bin.parent()?.to_path_buf();
    cuda_looks_valid(&root).then_some(root)
}

fn cuda_looks_valid(root: &Path) -> bool {
    root.join("include/cuda_runtime.h").is_file()
        && (root.join("bin/nvcc.exe").is_file() || root.join("bin/nvcc").is_file())
}

fn nvcc_bin(cuda_root: &Path) -> PathBuf {
    if cfg!(windows) {
        cuda_root.join("bin/nvcc.exe")
    } else {
        cuda_root.join("bin/nvcc")
    }
}

fn manifest_cuda_dir() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../cuda")
}

/// Toolkit major.minor from `nvcc --version` (e.g. 11.4). `None` if unreadable.
fn nvcc_version(cuda_root: &Path) -> Option<(u32, u32)> {
    let out = Command::new(nvcc_bin(cuda_root))
        .arg("--version")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // "Cuda compilation tools, release 11.4, V11.4.152"
    for line in text.lines() {
        if let Some(rest) = line.split("release ").nth(1) {
            let ver = rest.split(',').next()?.trim();
            let mut parts = ver.split('.');
            let major: u32 = parts.next()?.parse().ok()?;
            let minor: u32 = parts.next().unwrap_or("0").parse().ok()?;
            return Some((major, minor));
        }
    }
    None
}

fn toolkit_supports_kepler(cuda_root: &Path) -> bool {
    match nvcc_version(cuda_root) {
        Some((major, _)) => major < 12,
        // Assume older/unknown toolkits still accept sm_37; nvcc will error if not.
        None => true,
    }
}

/// Default fat binary: Kepler (K80) when the toolkit still supports it, plus
/// common Turing/Ampere/Ada targets used on newer CUDA boxes.
fn default_cuda_arches(cuda_root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if toolkit_supports_kepler(cuda_root) {
        out.push("sm_37".into());
    }
    for a in ["sm_75", "sm_86", "sm_89"] {
        out.push(a.into());
    }
    out
}

fn parse_cuda_arches(raw: &str) -> Vec<String> {
    let mut out = Vec::new();
    for part in raw.split(|c: char| c == ',' || c == ';' || c.is_whitespace()) {
        let part = part.trim();
        if part.is_empty() || part == "+" {
            continue;
        }
        // Accept 3.7 / 37 / sm_37 / compute_37.
        let normalized = normalize_arch_token(part);
        if let Some(sm) = normalized {
            if !out.iter().any(|a| a == &sm) {
                out.push(sm);
            }
        }
    }
    out
}

fn normalize_arch_token(raw: &str) -> Option<String> {
    let s = raw.trim().trim_start_matches("sm_").trim_start_matches("compute_");
    if s.is_empty() {
        return None;
    }
    // "3.7" → "37"
    let digits: String = if s.contains('.') {
        s.chars().filter(|c| c.is_ascii_digit()).collect()
    } else {
        s.chars().filter(|c| c.is_ascii_digit()).collect()
    };
    if digits.is_empty() {
        return None;
    }
    Some(format!("sm_{digits}"))
}

/// Map well-known product names to sm_XX when `nvidia-smi` lacks compute_cap
/// (driver 470 on Tesla K80).
fn arch_from_gpu_name(name: &str) -> Option<&'static str> {
    let n = name.to_ascii_lowercase();
    // Kepler
    if n.contains("k80") {
        return Some("sm_37");
    }
    if n.contains("k40") || n.contains("k20") || n.contains("k10") || n.contains("gtx titan") {
        return Some("sm_35");
    }
    // Maxwell
    if n.contains("m40") || n.contains("m60") || n.contains("gtx 9") {
        return Some("sm_52");
    }
    // Pascal
    if n.contains("p100") {
        return Some("sm_60");
    }
    if n.contains("p40") || n.contains("p4") || n.contains("gtx 10") || n.contains("titan x") {
        return Some("sm_61");
    }
    // Volta / Turing / Ampere / Ada (common)
    if n.contains("v100") {
        return Some("sm_70");
    }
    if n.contains("t4") || n.contains("rtx 20") || n.contains("quadro rtx") {
        return Some("sm_75");
    }
    if n.contains("a100") || n.contains("a30") || n.contains("a40") {
        return Some("sm_80");
    }
    if n.contains("rtx 30") || n.contains("rtx a") || n.contains("a10") || n.contains("a16") {
        return Some("sm_86");
    }
    if n.contains("rtx 40") || n.contains("l40") || n.contains("l4") {
        return Some("sm_89");
    }
    None
}

fn detect_cuda_arches() -> Vec<String> {
    // Newer drivers: nvidia-smi --query-gpu=compute_cap
    if let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            let mut arches = Vec::new();
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() || line.eq_ignore_ascii_case("[n/a]") {
                    continue;
                }
                if let Some(sm) = normalize_arch_token(line) {
                    if !arches.contains(&sm) {
                        arches.push(sm);
                    }
                }
            }
            if !arches.is_empty() {
                return arches;
            }
        }
    }

    // Driver 470 / K80: compute_cap query is missing; fall back to product name.
    if let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
    {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            let mut arches = Vec::new();
            for line in text.lines() {
                if let Some(sm) = arch_from_gpu_name(line.trim()) {
                    let sm = sm.to_string();
                    if !arches.contains(&sm) {
                        arches.push(sm);
                    }
                }
            }
            if !arches.is_empty() {
                return arches;
            }
        }
    }
    Vec::new()
}

fn cuda_arches(cuda_root: &Path) -> Vec<String> {
    for key in ["CUDA_ARCHES", "ALLPACA_CUDA_ARCHES", "TORCH_CUDA_ARCH_LIST"] {
        if let Ok(raw) = env::var(key) {
            let parsed = parse_cuda_arches(&raw);
            if !parsed.is_empty() {
                return filter_arches_for_toolkit(cuda_root, parsed);
            }
        }
    }
    let detected = detect_cuda_arches();
    if !detected.is_empty() {
        return filter_arches_for_toolkit(cuda_root, detected);
    }
    default_cuda_arches(cuda_root)
}

fn filter_arches_for_toolkit(cuda_root: &Path, arches: Vec<String>) -> Vec<String> {
    if toolkit_supports_kepler(cuda_root) {
        return arches;
    }
    // CUDA 12+: drop sm_3x / sm_5x / sm_6x (and warn).
    let filtered: Vec<String> = arches
        .into_iter()
        .filter(|a| {
            let keep = !a.starts_with("sm_3") && !a.starts_with("sm_5") && !a.starts_with("sm_6");
            if !keep {
                println!(
                    "cargo:warning=dropping {a}: CUDA 12+ toolkits no longer support Kepler/Maxwell/Pascal"
                );
            }
            keep
        })
        .collect();
    if filtered.is_empty() {
        default_cuda_arches(cuda_root)
    } else {
        filtered
    }
}

fn gencode_flags(arches: &[String]) -> Vec<String> {
    let mut flags = Vec::new();
    for a in arches {
        let digits = a.trim_start_matches("sm_");
        // SASS for the real chip + PTX virtual arch for forward compat within the
        // same major family where NVIDIA still supports it.
        flags.push(format!("-gencode=arch=compute_{digits},code=sm_{digits}"));
    }
    flags
}

fn host_compiler(cuda_root: &Path) -> Option<PathBuf> {
    if let Ok(cxx) = env::var("CXX") {
        let p = PathBuf::from(&cxx);
        if p.is_file() || Command::new(&cxx).arg("--version").status().is_ok() {
            return Some(p);
        }
    }
    // CUDA 11.4 officially tops out at gcc 10; Ubuntu 22.04 defaults to gcc 11.
    if let Some((major, _)) = nvcc_version(cuda_root) {
        if major <= 11 {
            for cand in ["g++-10", "/usr/bin/g++-10"] {
                let p = PathBuf::from(cand);
                if Command::new(&p).arg("--version").status().map(|s| s.success()).unwrap_or(false)
                {
                    return Some(p);
                }
            }
        }
    }
    None
}

fn build_kernels(cuda_root: &Path, out: &Path, arches: &[String]) -> Result<(), String> {
    let nvcc = nvcc_bin(cuda_root);
    let src = manifest_cuda_dir().join("src/kernels.cu");
    let include = manifest_cuda_dir().join("include");
    let obj = out.join(if cfg!(windows) {
        "kernels.obj"
    } else {
        "kernels.o"
    });
    let gencodes = gencode_flags(arches);

    let output = if cfg!(windows) {
        let vcvars = find_vcvars64()
            .ok_or_else(|| "vcvars64.bat not found (need MSVC for nvcc)".to_string())?;
        let bat = out.join("build_cuda_kernels.bat");
        let gencode_joined = gencodes.join(" ");
        let script = format!(
            "@echo off\r\ncall \"{vc}\"\r\n\"{nvcc}\" -c \"{src}\" -o \"{obj}\" -O2 -std=c++17 {gencodes} -I\"{inc}\" -I\"{cuda_inc}\"\r\n",
            vc = vcvars.display(),
            nvcc = nvcc.display(),
            src = src.display(),
            obj = obj.display(),
            gencodes = gencode_joined,
            inc = include.display(),
            cuda_inc = cuda_root.join("include").display(),
        );
        std::fs::write(&bat, script).map_err(|e| format!("write bat: {e}"))?;
        Command::new("cmd")
            .arg("/C")
            .arg(&bat)
            .env("CUDA_PATH", cuda_root)
            .output()
            .map_err(|e| format!("spawn nvcc via vcvars: {e}"))?
    } else {
        let mut cmd = Command::new(&nvcc);
        cmd.arg("-c")
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .arg("-O2")
            .arg("-std=c++17");
        for g in &gencodes {
            cmd.arg(g);
        }
        cmd.arg(format!("-I{}", include.display()))
            .arg(format!("-I{}", cuda_root.join("include").display()))
            .env("CUDA_PATH", cuda_root);
        if let Some(cxx) = host_compiler(cuda_root) {
            cmd.arg(format!("-ccbin={}", cxx.display()));
        } else if let Some((major, _)) = nvcc_version(cuda_root) {
            if major <= 11 {
                // Last resort on gcc 11+ hosts without g++-10.
                cmd.arg("-allow-unsupported-compiler");
            }
        }
        cmd.output().map_err(|e| format!("spawn nvcc: {e}"))?
    };
    if !output.status.success() {
        return Err(format!(
            "nvcc failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    archive_objects(out, &obj)
}

fn find_vcvars64() -> Option<PathBuf> {
    let pf86 = env::var_os("ProgramFiles(x86)")?;
    let vswhere = PathBuf::from(pf86).join(r"Microsoft Visual Studio\Installer\vswhere.exe");
    if vswhere.is_file() {
        let out = Command::new(&vswhere)
            .args(["-latest", "-property", "installationPath"])
            .output()
            .ok()?;
        if out.status.success() {
            let root = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let bat = PathBuf::from(root).join(r"VC\Auxiliary\Build\vcvars64.bat");
            if bat.is_file() {
                return Some(bat);
            }
        }
    }
    for year in ["2022", "2019"] {
        for ed in ["Community", "Professional", "Enterprise", "BuildTools"] {
            let bat = PathBuf::from(r"C:\Program Files\Microsoft Visual Studio")
                .join(year)
                .join(ed)
                .join(r"VC\Auxiliary\Build\vcvars64.bat");
            if bat.is_file() {
                return Some(bat);
            }
        }
    }
    None
}

fn archive_objects(out: &Path, obj: &Path) -> Result<(), String> {
    let lib = out.join(if cfg!(windows) {
        "allpaca_cuda_kernels.lib"
    } else {
        "liballpaca_cuda_kernels.a"
    });
    if cfg!(windows) {
        if let Ok(st) = Command::new("lib")
            .arg(format!("/OUT:{}", lib.display()))
            .arg(obj)
            .status()
        {
            if st.success() {
                return Ok(());
            }
        }
        // Retry via vcvars if PATH lacks lib.exe (cargo build env).
        if let Some(vcvars) = find_vcvars64() {
            let bat = out.join("archive_cuda.bat");
            let script = format!(
                "@echo off\r\ncall \"{vc}\"\r\nlib /OUT:\"{lib}\" \"{obj}\"\r\n",
                vc = vcvars.display(),
                lib = lib.display(),
                obj = obj.display(),
            );
            std::fs::write(&bat, script).map_err(|e| format!("write archive bat: {e}"))?;
            let st = Command::new("cmd")
                .arg("/C")
                .arg(&bat)
                .status()
                .map_err(|e| format!("archive via vcvars: {e}"))?;
            if st.success() && lib.is_file() {
                return Ok(());
            }
        }
        return Err("failed to archive CUDA kernel objects with lib.exe".into());
    }
    let st = Command::new("ar")
        .arg("rcs")
        .arg(&lib)
        .arg(obj)
        .status()
        .map_err(|e| format!("ar: {e}"))?;
    if st.success() {
        Ok(())
    } else {
        Err("failed to archive CUDA kernel objects".into())
    }
}
