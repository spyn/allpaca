//! Probe HIP/ROCm and, when present, compile `hip/` kernels and link them.
//!
//! Without an SDK, `--features hip` still compiles: Rust uses host stubs and
//! GPU tests skip (`device_count() == 0`). Set `HIP_PATH` or `ROCM_PATH`.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../hip/src/kernels.cpp");
    println!("cargo:rerun-if-changed=../../hip/include/allpaca_kernels.h");
    println!("cargo:rerun-if-changed=../../hip/CMakeLists.txt");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rustc-check-cfg=cfg(allpaca_hip_runtime)");

    if env::var("CARGO_FEATURE_HIP").is_err() {
        return;
    }

    let Some(hip_root) = find_hip_root() else {
        println!(
            "cargo:warning=HIP SDK not found; hip feature is stub-only. Install ROCm and set HIP_PATH or ROCM_PATH (Windows default: C:\\Program Files\\AMD\\ROCm\\*)."
        );
        return;
    };

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    if let Err(e) = build_kernels(&hip_root, &out) {
        println!("cargo:warning=HIP kernel build failed ({e}); hip feature is stub-only.");
        return;
    }

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-search=native={}", hip_root.join("lib").display());
    println!("cargo:rustc-link-lib=static=allpaca_kernels");
    println!("cargo:rustc-link-lib=amdhip64");
    if cfg!(windows) {
        println!("cargo:rustc-link-lib=dylib=amdhip64");
    }
    println!("cargo:rustc-cfg=allpaca_hip_runtime");
}

fn find_hip_root() -> Option<PathBuf> {
    for key in ["HIP_PATH", "ROCM_PATH"] {
        if let Ok(p) = env::var(key) {
            let p = PathBuf::from(p.trim_end_matches(['\\', '/']));
            if hip_looks_valid(&p) {
                return Some(p);
            }
        }
    }
    let extras = [
        r"C:\Program Files\AMD\ROCm\7.1",
        r"C:\Program Files\AMD\ROCm\7.0",
        r"C:\Program Files\AMD\ROCm\6.4",
        r"C:\Program Files\AMD\ROCm\6.3",
        r"C:\Program Files\AMD\ROCm\6.2",
        r"C:\Program Files\AMD\ROCm\5.7",
        "/opt/rocm",
        "/opt/rocm-7.1",
        "/opt/rocm-6.3",
    ];
    for p in extras {
        let p = PathBuf::from(p);
        if hip_looks_valid(&p) {
            return Some(p);
        }
    }
    if let Ok(entries) = std::fs::read_dir(r"C:\Program Files\AMD\ROCm") {
        let mut vers: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        vers.sort();
        vers.reverse();
        for p in vers {
            if hip_looks_valid(&p) {
                return Some(p);
            }
        }
    }
    None
}

fn hip_looks_valid(root: &Path) -> bool {
    root.join("include/hip/hip_runtime.h").is_file()
        && (root.join("bin/hipcc.exe").is_file() || root.join("bin/hipcc").is_file())
}

fn build_kernels(hip_root: &Path, out: &Path) -> Result<(), String> {
    match try_hipcc(hip_root, out) {
        Ok(()) => return Ok(()),
        Err(e) => println!("cargo:warning=hipcc path failed: {e}"),
    }
    try_cmake(hip_root, out)
}

fn hipcc_bin(hip_root: &Path) -> PathBuf {
    if cfg!(windows) {
        hip_root.join("bin/hipcc.exe")
    } else {
        hip_root.join("bin/hipcc")
    }
}

fn try_hipcc(hip_root: &Path, out: &Path) -> Result<(), String> {
    let hipcc = hipcc_bin(hip_root);
    let src = manifest_hip_dir().join("src/kernels.cpp");
    let obj = out.join(if cfg!(windows) { "kernels.obj" } else { "kernels.o" });
    let include = manifest_hip_dir().join("include");
    let output = if cfg!(windows) {
        let vcvars = find_vcvars64().ok_or_else(|| "vcvars64.bat not found (need MSVC for hipcc)".to_string())?;
        let bat = out.join("build_kernels.bat");
        let script = format!(
            "@echo off\r\ncall \"{vc}\"\r\n\"{hipcc}\" -c \"{src}\" -o \"{obj}\" -O2 -std=c++17 -I\"{inc}\" --offload-arch=gfx1201 --offload-arch=gfx1200 --offload-arch=gfx1100\r\n",
            vc = vcvars.display(),
            hipcc = hipcc.display(),
            src = src.display(),
            obj = obj.display(),
            inc = include.display(),
        );
        std::fs::write(&bat, script).map_err(|e| format!("write bat: {e}"))?;
        Command::new("cmd")
            .arg("/C")
            .arg(&bat)
            .env("HIP_PATH", hip_root)
            .output()
            .map_err(|e| format!("spawn hipcc via vcvars: {e}"))?
    } else {
        Command::new(&hipcc)
            .arg("-c")
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .arg("-O2")
            .arg("-std=c++17")
            .arg(format!("-I{}", include.display()))
            .arg("--offload-arch=gfx1201")
            .arg("--offload-arch=gfx1200")
            .arg("--offload-arch=gfx1100")
            .env("HIP_PATH", hip_root)
            .output()
            .map_err(|e| format!("spawn hipcc: {e}"))?
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!("hipcc failed: {stdout}{stderr}"));
    }
    archive_objects(hip_root, out, &obj)
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

fn archive_objects(hip_root: &Path, out: &Path, obj: &Path) -> Result<(), String> {
    let lib = out.join(if cfg!(windows) {
        "allpaca_kernels.lib"
    } else {
        "liballpaca_kernels.a"
    });
    let llvm_ar = if cfg!(windows) {
        hip_root.join("bin/llvm-ar.exe")
    } else {
        hip_root.join("bin/llvm-ar")
    };
    if llvm_ar.is_file() {
        let st = Command::new(&llvm_ar)
            .arg("rcs")
            .arg(&lib)
            .arg(obj)
            .status()
            .map_err(|e| format!("llvm-ar: {e}"))?;
        if st.success() {
            return Ok(());
        }
    }
    if cfg!(windows) {
        let st = Command::new("lib")
            .arg(format!("/OUT:{}", lib.display()))
            .arg(obj)
            .status()
            .map_err(|e| format!("lib.exe: {e}"))?;
        if st.success() {
            return Ok(());
        }
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
        Err("failed to archive kernel objects".into())
    }
}

fn try_cmake(hip_root: &Path, out: &Path) -> Result<(), String> {
    let build = out.join("cmake-hip");
    let _ = std::fs::create_dir_all(&build);
    let mut conf = Command::new("cmake");
    conf.arg("-S")
        .arg(manifest_hip_dir())
        .arg("-B")
        .arg(&build)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!("-DCMAKE_PREFIX_PATH={}", hip_root.display()))
        .env("HIP_PATH", hip_root)
        .env("ROCM_PATH", hip_root);
    if cfg!(windows) {
        conf.arg("-G").arg("Ninja");
    }
    let st = conf.status().map_err(|e| format!("cmake configure: {e}"))?;
    if !st.success() {
        return Err(format!("cmake configure failed: {st}"));
    }
    let st = Command::new("cmake")
        .arg("--build")
        .arg(&build)
        .arg("--config")
        .arg("Release")
        .status()
        .map_err(|e| format!("cmake build: {e}"))?;
    if !st.success() {
        return Err(format!("cmake build failed: {st}"));
    }
    copy_built_lib(&build, out)
}

fn copy_built_lib(build: &Path, out: &Path) -> Result<(), String> {
    let names = [
        "allpaca_kernels.lib",
        "liballpaca_kernels.a",
        "Release/allpaca_kernels.lib",
        "Debug/allpaca_kernels.lib",
    ];
    for n in names {
        let src = build.join(n);
        if src.is_file() {
            let dest = out.join(src.file_name().unwrap());
            std::fs::copy(&src, dest).map_err(|e| e.to_string())?;
            return Ok(());
        }
    }
    Err("cmake produced no allpaca_kernels library".into())
}

fn manifest_hip_dir() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../hip")
}
