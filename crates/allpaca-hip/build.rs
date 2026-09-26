//! Probe HIP/ROCm and, when present, compile `hip/` kernels and link them.
//!
//! Without an SDK, `--features hip` still compiles: Rust uses host stubs and
//! GPU tests skip (`device_count() == 0`). Set `HIP_PATH` or `ROCM_PATH`.
//!
//! Offload ISAs: `AMDGPU_TARGETS` / `HIP_AMDGPU_TARGETS` / `ALLPACA_OFFLOAD_ARCH`,
//! else live `rocminfo` GPUs, else a fat list that includes gfx906 (MI50 / Radeon
//! VII) and gfx900 (home Vega) plus the RDNA3/4 targets.

use std::env;
#[cfg(windows)]
use std::fs::File;
#[cfg(windows)]
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(windows)]
const DIRECT_STORAGE_VERSION: &str = "1.3.0";

fn main() {
    println!("cargo:rerun-if-changed=../../hip/src/kernels.cpp");
    println!("cargo:rerun-if-changed=../../hip/include/allpaca_kernels.h");
    println!("cargo:rerun-if-changed=../../hip/CMakeLists.txt");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=AMDGPU_TARGETS");
    println!("cargo:rerun-if-env-changed=HIP_AMDGPU_TARGETS");
    println!("cargo:rerun-if-env-changed=ALLPACA_OFFLOAD_ARCH");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(allpaca_hip_runtime)");
    println!("cargo:rustc-check-cfg=cfg(allpaca_rocblas)");
    println!("cargo:rustc-check-cfg=cfg(allpaca_hipblaslt)");
    println!("cargo:rustc-check-cfg=cfg(allpaca_direct_storage)");

    if env::var("CARGO_FEATURE_HIP").is_err() {
        return;
    }

    let Some(hip_root) = find_hip_root() else {
        println!(
            "cargo:warning=HIP SDK not found; hip feature is stub-only. Need hipcc + hip_runtime.h (runtime-only /opt/rocm is not enough). Install hipcc/hip-dev matching the runtime, or run scripts/fetch-rocm-hip-sdk.sh, and set HIP_PATH or ROCM_PATH."
        );
        return;
    };

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let direct_storage = windows_direct_storage(&out);
    let mut hipblaslt = hipblaslt_present(&hip_root);
    if hipblaslt && cfg!(windows) {
        if let Err(e) = generate_hipblaslt_import_lib(&hip_root, &out) {
            println!(
                "cargo:warning=hipBLASLt MSVC import lib failed ({e}); compiling without hipBLASLt"
            );
            hipblaslt = false;
        }
    }
    let arches = offload_arches();
    println!(
        "cargo:warning=HIP SDK {} offload={}",
        hip_root.display(),
        arches.join(",")
    );
    if let Err(e) = build_kernels(
        &hip_root,
        &out,
        hipblaslt,
        direct_storage.as_deref(),
        &arches,
    ) {
        println!("cargo:warning=HIP kernel build failed ({e}); hip feature is stub-only.");
        return;
    }

    println!("cargo:rustc-link-search=native={}", out.display());
    for libdir in hip_lib_dirs(&hip_root) {
        println!("cargo:rustc-link-search=native={}", libdir.display());
        if cfg!(unix) {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libdir.display());
        }
    }
    println!("cargo:rustc-link-lib=static=allpaca_kernels");
    println!("cargo:rustc-link-lib=amdhip64");
    if cfg!(windows) {
        println!("cargo:rustc-link-lib=dylib=amdhip64");
    }
    if cfg!(unix) {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    if rocblas_lib_exists(&hip_root) {
        println!("cargo:rustc-link-lib=rocblas");
        println!("cargo:rustc-cfg=allpaca_rocblas");
    }
    if hipblaslt {
        println!("cargo:rustc-link-lib=hipblaslt");
        println!("cargo:rustc-cfg=allpaca_hipblaslt");
    }
    if cfg!(windows) {
        println!("cargo:rustc-link-lib=dxgi");
        // PDH reads adapter-wide dedicated VRAM use (other processes included).
        println!("cargo:rustc-link-lib=pdh");
    }
    if let Some(sdk) = direct_storage {
        println!(
            "cargo:rustc-link-search=native={}",
            sdk.join("native/lib/x64").display()
        );
        println!("cargo:rustc-link-lib=dstorage");
        println!("cargo:rustc-link-lib=d3d12");
        println!("cargo:rustc-link-lib=dxgi");
        println!("cargo:rustc-cfg=allpaca_direct_storage");
    }
    println!("cargo:rustc-cfg=allpaca_hip_runtime");
}

#[cfg(not(windows))]
fn windows_direct_storage(_out: &Path) -> Option<PathBuf> {
    None
}

#[cfg(windows)]
fn windows_direct_storage(out: &Path) -> Option<PathBuf> {
    match restore_direct_storage_sdk(out) {
        Ok(sdk) => Some(sdk),
        Err(e) => {
            println!(
                "cargo:warning=Microsoft.Direct3D.DirectStorage {DIRECT_STORAGE_VERSION} unavailable ({e}); Windows NVMe-to-VRAM DMA disabled"
            );
            None
        }
    }
}

#[cfg(windows)]
fn restore_direct_storage_sdk(out: &Path) -> Result<PathBuf, String> {
    println!("cargo:rerun-if-env-changed=DOTNET_ROOT");

    let restore = out.join("directstorage-restore");
    std::fs::create_dir_all(&restore).map_err(|e| format!("create restore dir: {e}"))?;
    let project = restore.join("DirectStorage.proj");
    let project_xml = format!(
        r#"<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><TargetFramework>net8.0</TargetFramework></PropertyGroup>
  <ItemGroup><PackageReference Include="Microsoft.Direct3D.DirectStorage" Version="{DIRECT_STORAGE_VERSION}" /></ItemGroup>
</Project>
"#
    );
    std::fs::write(&project, project_xml).map_err(|e| format!("write restore project: {e}"))?;
    let packages = restore.join("packages");
    let output = Command::new("dotnet")
        .arg("restore")
        .arg(&project)
        .arg("--nologo")
        .arg("--verbosity")
        .arg("quiet")
        .arg(format!("-p:RestorePackagesPath={}", packages.display()))
        .output()
        .map_err(|e| format!("start dotnet restore: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "dotnet restore failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let package = packages
        .join("microsoft.direct3d.directstorage")
        .join(DIRECT_STORAGE_VERSION);
    let archive = package.join(format!(
        "microsoft.direct3d.directstorage.{DIRECT_STORAGE_VERSION}.nupkg"
    ));
    let sdk = out.join(format!("directstorage-{DIRECT_STORAGE_VERSION}"));
    let required = [
        "native/include/dstorage.h",
        "native/include/dstorageerr.h",
        "native/lib/x64/dstorage.lib",
        "native/bin/x64/dstorage.dll",
        "native/bin/x64/dstoragecore.dll",
    ];
    let file = File::open(&archive).map_err(|e| format!("open {}: {e}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| format!("open SDK archive: {e}"))?;
    for name in required {
        let destination = sdk.join(name);
        if destination.is_file() {
            continue;
        }
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create {}: {e}", parent.display()))?;
        }
        let mut source = zip
            .by_name(name)
            .map_err(|e| format!("SDK is missing {name}: {e}"))?;
        let mut bytes = Vec::with_capacity(source.size() as usize);
        source
            .read_to_end(&mut bytes)
            .map_err(|e| format!("read {name}: {e}"))?;
        File::create(&destination)
            .and_then(|mut f| f.write_all(&bytes))
            .map_err(|e| format!("extract {}: {e}", destination.display()))?;
    }

    let mut profile = out.to_path_buf();
    for _ in 0..3 {
        profile.pop();
    }
    for directory in [profile.clone(), profile.join("deps")] {
        std::fs::create_dir_all(&directory)
            .map_err(|e| format!("create {}: {e}", directory.display()))?;
        for dll in ["dstorage.dll", "dstoragecore.dll"] {
            std::fs::copy(sdk.join("native/bin/x64").join(dll), directory.join(dll))
                .map_err(|e| format!("copy {dll} to {}: {e}", directory.display()))?;
        }
    }
    Ok(sdk)
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
    for p in extra_hip_roots() {
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

fn rocblas_lib_exists(hip_root: &Path) -> bool {
    hip_root.join("lib/rocblas.lib").is_file()
        || hip_root.join("lib/librocblas.so").is_file()
        || hip_root.join("lib/librocblas.a").is_file()
}

fn hipblaslt_present(hip_root: &Path) -> bool {
    let hdr = hip_root.join("include/hipblaslt/hipblaslt.h");
    if !hdr.is_file() {
        return false;
    }
    hip_root.join("bin/libhipblaslt.dll").is_file()
        || hip_root.join("bin/hipblaslt.dll").is_file()
        || hip_root.join("lib/libhipblaslt.so").is_file()
        || hip_root.join("lib/hipblaslt.lib").is_file()
}

fn find_msvc_tool(name: &str) -> Option<PathBuf> {
    let pf86 = env::var_os("ProgramFiles(x86)")?;
    let vswhere = PathBuf::from(pf86).join(r"Microsoft Visual Studio\Installer\vswhere.exe");
    if !vswhere.is_file() {
        return None;
    }
    let out = Command::new(&vswhere)
        .args(["-latest", "-find", &format!(r"**\Hostx64\x64\{name}")])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let path = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();
    let p = PathBuf::from(path);
    p.is_file().then_some(p)
}

/// MinGW `libhipblaslt.dll.a` is not an MSVC import lib. Build one from the DLL.
fn generate_hipblaslt_import_lib(hip_root: &Path, out: &Path) -> Result<(), String> {
    if hip_root.join("lib/hipblaslt.lib").is_file() {
        return Ok(());
    }
    let lib_out = out.join("hipblaslt.lib");
    if lib_out.is_file() {
        return Ok(());
    }
    let dll = if hip_root.join("bin/libhipblaslt.dll").is_file() {
        hip_root.join("bin/libhipblaslt.dll")
    } else {
        hip_root.join("bin/hipblaslt.dll")
    };
    if !dll.is_file() {
        return Err("hipBLASLt DLL not found".into());
    }
    println!("cargo:rerun-if-changed={}", dll.display());
    let dumpbin =
        find_msvc_tool("dumpbin.exe").ok_or_else(|| "dumpbin.exe not found".to_string())?;
    let libexe = find_msvc_tool("lib.exe").ok_or_else(|| "lib.exe not found".to_string())?;
    let output = Command::new(&dumpbin)
        .arg("/EXPORTS")
        .arg(&dll)
        .output()
        .map_err(|e| format!("dumpbin: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "dumpbin /EXPORTS failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut names: Vec<String> = Vec::new();
    for line in stdout.lines() {
        let t = line.trim();
        if t.contains('?') || t.contains('@') {
            continue;
        }
        let Some(name) = t.split_whitespace().last() else {
            continue;
        };
        if name.starts_with("hipblasLt") && name.chars().all(|c| c.is_ascii_alphanumeric()) {
            names.push(name.to_string());
        }
    }
    names.sort();
    names.dedup();
    if names.is_empty() {
        return Err("no undecorated hipblasLt exports".into());
    }
    let def_path = out.join("hipblaslt.def");
    let mut def = String::from("LIBRARY libhipblaslt.dll\r\nEXPORTS\r\n");
    for n in &names {
        def.push_str(n);
        def.push_str("\r\n");
    }
    std::fs::write(&def_path, def).map_err(|e| format!("write def: {e}"))?;
    let st = Command::new(&libexe)
        .arg(format!("/DEF:{}", def_path.display()))
        .arg(format!("/OUT:{}", lib_out.display()))
        .arg("/MACHINE:X64")
        .status()
        .map_err(|e| format!("lib.exe: {e}"))?;
    if !st.success() || !lib_out.is_file() {
        return Err("lib.exe failed to write hipblaslt.lib".into());
    }
    Ok(())
}

fn extra_hip_roots() -> Vec<PathBuf> {
    let mut extras = vec![
        PathBuf::from(r"C:\Program Files\AMD\ROCm\7.1"),
        PathBuf::from(r"C:\Program Files\AMD\ROCm\7.0"),
        PathBuf::from(r"C:\Program Files\AMD\ROCm\6.4"),
        PathBuf::from(r"C:\Program Files\AMD\ROCm\6.3"),
        PathBuf::from(r"C:\Program Files\AMD\ROCm\6.2"),
        PathBuf::from(r"C:\Program Files\AMD\ROCm\5.7"),
        PathBuf::from("/opt/rocm"),
        PathBuf::from("/opt/rocm-7.1"),
        PathBuf::from("/opt/rocm-6.3"),
        PathBuf::from("/opt/rocm-6.2.4"),
        PathBuf::from("/opt/rocm-6.2"),
    ];
    if let Ok(home) = env::var("HOME") {
        let cache = PathBuf::from(&home).join(".cache/allpaca/rocm-sdk/opt");
        if let Ok(entries) = std::fs::read_dir(&cache) {
            let mut vers: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
            vers.sort();
            vers.reverse();
            extras.extend(vers);
        }
        extras.push(PathBuf::from(home).join(".cache/allpaca/rocm-sdk/opt/rocm-6.2.4"));
    }
    extras
}

fn hip_lib_dirs(hip_root: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![hip_root.join("lib")];
    let system = PathBuf::from("/opt/rocm/lib");
    if system.is_dir() && !dirs.contains(&system) {
        dirs.push(system);
    }
    dirs
}

fn hip_looks_valid(root: &Path) -> bool {
    root.join("include/hip/hip_runtime.h").is_file() && hipcc_bin(root).is_file()
}

/// Default fat binary when no live GPU is visible: Vega/GCN5 (MI50, Radeon VII,
/// home Vega) plus the RDNA3/4 targets the kernels were first brought up on.
fn default_offload_arches() -> Vec<String> {
    ["gfx906", "gfx900", "gfx1100", "gfx1200", "gfx1201"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn parse_offload_arches(raw: &str) -> Vec<String> {
    let mut out = Vec::new();
    for part in raw.split(|c: char| c == ',' || c == ';' || c.is_whitespace()) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if !out.iter().any(|a| a == part) {
            out.push(part.to_string());
        }
    }
    out
}

fn gfx_arch_token(raw: &str) -> Option<String> {
    let idx = raw.find("gfx")?;
    let rest = &raw[idx..];
    let token: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == ':')
        .collect();
    let base = token.split(':').next().unwrap_or("");
    if base.starts_with("gfx") && base.len() > 3 {
        Some(base.to_string())
    } else {
        None
    }
}

fn detect_offload_arches() -> Vec<String> {
    let candidates = [
        PathBuf::from("rocminfo"),
        PathBuf::from("/usr/bin/rocminfo"),
        PathBuf::from("/opt/rocm/bin/rocminfo"),
    ];
    for bin in candidates {
        let output = Command::new(&bin).output();
        let Ok(output) = output else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut arches = Vec::new();
        for line in stdout.lines() {
            let Some(arch) = gfx_arch_token(line) else {
                continue;
            };
            if !arches.contains(&arch) {
                arches.push(arch);
            }
        }
        if !arches.is_empty() {
            return arches;
        }
    }
    Vec::new()
}

fn offload_arches() -> Vec<String> {
    for key in [
        "AMDGPU_TARGETS",
        "HIP_AMDGPU_TARGETS",
        "ALLPACA_OFFLOAD_ARCH",
    ] {
        if let Ok(raw) = env::var(key) {
            let parsed = parse_offload_arches(&raw);
            if !parsed.is_empty() {
                return parsed;
            }
        }
    }
    let detected = detect_offload_arches();
    if !detected.is_empty() {
        return detected;
    }
    default_offload_arches()
}

fn build_kernels(
    hip_root: &Path,
    out: &Path,
    hipblaslt: bool,
    direct_storage: Option<&Path>,
    arches: &[String],
) -> Result<(), String> {
    match try_hipcc(hip_root, out, hipblaslt, direct_storage, arches) {
        Ok(()) => return Ok(()),
        Err(e) => println!("cargo:warning=hipcc path failed: {e}"),
    }
    try_cmake(hip_root, out, arches)
}

fn hipcc_bin(hip_root: &Path) -> PathBuf {
    if cfg!(windows) {
        hip_root.join("bin/hipcc.exe")
    } else {
        hip_root.join("bin/hipcc")
    }
}

fn apply_hip_env(cmd: &mut Command, hip_root: &Path) {
    cmd.env("HIP_PATH", hip_root);
    cmd.env("ROCM_PATH", hip_root);
    let clang = hip_root.join("lib/llvm/bin");
    if clang.join("clang++").is_file() || clang.join("clang++.exe").is_file() {
        cmd.env("HIP_CLANG_PATH", clang);
    }
}

fn try_hipcc(
    hip_root: &Path,
    out: &Path,
    hipblaslt: bool,
    direct_storage: Option<&Path>,
    arches: &[String],
) -> Result<(), String> {
    let hipcc = hipcc_bin(hip_root);
    let src = manifest_hip_dir().join("src/kernels.cpp");
    let obj = out.join(if cfg!(windows) {
        "kernels.obj"
    } else {
        "kernels.o"
    });
    let include = manifest_hip_dir().join("include");
    let arch_flags: Vec<String> = arches
        .iter()
        .map(|a| format!("--offload-arch={a}"))
        .collect();
    let mut extra = String::new();
    if rocblas_lib_exists(hip_root) {
        extra.push_str(" -DALLPACA_HAS_ROCBLAS");
    }
    if hipblaslt {
        extra.push_str(" -DALLPACA_HAS_HIPBLASLT");
    }
    if let Some(sdk) = direct_storage {
        extra.push_str(" -DALLPACA_HAS_DIRECTSTORAGE -I\"");
        extra.push_str(&sdk.join("native/include").display().to_string());
        extra.push('"');
    }
    let output = if cfg!(windows) {
        let vcvars = find_vcvars64()
            .ok_or_else(|| "vcvars64.bat not found (need MSVC for hipcc)".to_string())?;
        let bat = out.join("build_kernels.bat");
        let script = format!(
            "@echo off\r\ncall \"{vc}\"\r\n\"{hipcc}\" -c \"{src}\" -o \"{obj}\" -O2 -std=c++17 -I\"{inc}\"{extra} {arches}\r\n",
            vc = vcvars.display(),
            hipcc = hipcc.display(),
            src = src.display(),
            obj = obj.display(),
            inc = include.display(),
            extra = extra,
            arches = arch_flags.join(" "),
        );
        std::fs::write(&bat, script).map_err(|e| format!("write bat: {e}"))?;
        let mut cmd = Command::new("cmd");
        cmd.arg("/C").arg(&bat);
        apply_hip_env(&mut cmd, hip_root);
        cmd.output()
            .map_err(|e| format!("spawn hipcc via vcvars: {e}"))?
    } else {
        let mut cmd = Command::new(&hipcc);
        cmd.arg("-c")
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .arg("-O2")
            .arg("-std=c++17")
            .arg("-fPIC")
            .arg(format!("-I{}", include.display()));
        for flag in &arch_flags {
            cmd.arg(flag);
        }
        apply_hip_env(&mut cmd, hip_root);
        if rocblas_lib_exists(hip_root) {
            cmd.arg("-DALLPACA_HAS_ROCBLAS");
        }
        if hipblaslt {
            cmd.arg("-DALLPACA_HAS_HIPBLASLT");
        }
        cmd.output().map_err(|e| format!("spawn hipcc: {e}"))?
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
    let llvm_ar_name = if cfg!(windows) {
        "llvm-ar.exe"
    } else {
        "llvm-ar"
    };
    let llvm_ar_candidates = [
        hip_root.join("bin").join(llvm_ar_name),
        hip_root.join("lib/llvm/bin").join(llvm_ar_name),
        hip_root.join("llvm/bin").join(llvm_ar_name),
    ];
    for llvm_ar in llvm_ar_candidates {
        if !llvm_ar.is_file() {
            continue;
        }
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

fn try_cmake(hip_root: &Path, out: &Path, arches: &[String]) -> Result<(), String> {
    let build = out.join("cmake-hip");
    let _ = std::fs::create_dir_all(&build);
    let mut conf = Command::new("cmake");
    conf.arg("-S")
        .arg(manifest_hip_dir())
        .arg("-B")
        .arg(&build)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!("-DCMAKE_PREFIX_PATH={}", hip_root.display()))
        .arg(format!("-DAMDGPU_TARGETS={}", arches.join(";")));
    apply_hip_env(&mut conf, hip_root);
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
