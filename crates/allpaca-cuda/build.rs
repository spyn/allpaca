//! Probe the CUDA toolkit and, when present, compile `cuda/` kernels and link them.
//!
//! Without a toolkit, `--features cuda` still compiles: Rust uses host stubs and
//! `device_count() == 0`. A toolkit without an NVIDIA driver also yields 0 devices
//! at runtime (this machine's case). Set `CUDA_PATH` or rely on PATH `nvcc`.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../cuda/src/kernels.cu");
    println!("cargo:rerun-if-changed=../../cuda/include/allpaca_cuda_kernels.h");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rustc-check-cfg=cfg(allpaca_cuda_runtime)");

    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let Some(cuda_root) = find_cuda_root() else {
        println!(
            "cargo:warning=CUDA toolkit not found; cuda feature is stub-only. Install CUDA and set CUDA_PATH."
        );
        return;
    };

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    if let Err(e) = build_kernels(&cuda_root, &out) {
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
            cuda_root.join("lib").join("x64").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_root.join("lib64").display()
        );
    }
    println!("cargo:rustc-link-lib=static=allpaca_cuda_kernels");
    println!("cargo:rustc-link-lib=cudart");
    if cfg!(windows) {
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
    println!("cargo:rustc-cfg=allpaca_cuda_runtime");
}

fn find_cuda_root() -> Option<PathBuf> {
    if let Ok(p) = env::var("CUDA_PATH") {
        let p = PathBuf::from(p.trim_end_matches(['\\', '/']));
        if cuda_looks_valid(&p) {
            return Some(p);
        }
    }
    if let Some(from_nvcc) = cuda_root_from_nvcc() {
        return Some(from_nvcc);
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
    for p in ["/usr/local/cuda", "/opt/cuda"] {
        let p = PathBuf::from(p);
        if cuda_looks_valid(&p) {
            return Some(p);
        }
    }
    None
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

fn build_kernels(cuda_root: &Path, out: &Path) -> Result<(), String> {
    let nvcc = nvcc_bin(cuda_root);
    let src = manifest_cuda_dir().join("src/kernels.cu");
    let include = manifest_cuda_dir().join("include");
    let obj = out.join(if cfg!(windows) {
        "kernels.obj"
    } else {
        "kernels.o"
    });

    let output = if cfg!(windows) {
        let vcvars = find_vcvars64()
            .ok_or_else(|| "vcvars64.bat not found (need MSVC for nvcc)".to_string())?;
        let bat = out.join("build_cuda_kernels.bat");
        let script = format!(
            "@echo off\r\ncall \"{vc}\"\r\n\"{nvcc}\" -c \"{src}\" -o \"{obj}\" -O2 -std=c++17 -I\"{inc}\" -I\"{cuda_inc}\"\r\n",
            vc = vcvars.display(),
            nvcc = nvcc.display(),
            src = src.display(),
            obj = obj.display(),
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
        Command::new(&nvcc)
            .arg("-c")
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .arg("-O2")
            .arg("-std=c++17")
            .arg(format!("-I{}", include.display()))
            .arg(format!("-I{}", cuda_root.join("include").display()))
            .env("CUDA_PATH", cuda_root)
            .output()
            .map_err(|e| format!("spawn nvcc: {e}"))?
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
