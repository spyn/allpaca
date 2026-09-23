//! Probe rust-gpu and compile `allpaca-shaders` to SPIR-V when `--features vulkan`.
//!
//! Without a successful nested build, the crate still compiles: Rust uses host
//! stubs and `device_count() == 0`. Set `ALLPACA_SKIP_VULKAN_SHADERS=1` to skip.

use std::env;
use std::path::PathBuf;

const ENTRIES: &[&str] = &[
    "gemm_f32",
    "gemm_f16",
    "gemm_q8_0",
    "residual_add",
    "rmsnorm",
    "silu_mul",
    "embed_gather",
    "rope",
    "attention",
];

fn main() {
    println!("cargo:rerun-if-changed=../allpaca-shaders/src/lib.rs");
    println!("cargo:rerun-if-changed=../allpaca-shaders/Cargo.toml");
    println!("cargo:rerun-if-changed=glsl/gemm_f16.comp");
    println!("cargo:rerun-if-env-changed=ALLPACA_SKIP_VULKAN_SHADERS");
    println!("cargo:rerun-if-env-changed=GLSLANG_VALIDATOR");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rustc-check-cfg=cfg(allpaca_vulkan_runtime)");
    println!("cargo:rustc-check-cfg=cfg(allpaca_vulkan_glsl_f16)");

    if env::var("CARGO_FEATURE_VULKAN").is_err() {
        return;
    }
    if env::var("ALLPACA_SKIP_VULKAN_SHADERS").is_ok() {
        println!("cargo:warning=ALLPACA_SKIP_VULKAN_SHADERS set; vulkan feature is stub-only.");
        return;
    }

    match compile_shaders() {
        Ok(()) => {
            println!("cargo:rustc-cfg=allpaca_vulkan_runtime");
        }
        Err(e) => {
            println!(
                "cargo:warning=rust-gpu shader build failed ({e}); vulkan feature is stub-only."
            );
        }
    }
}

fn env_name(entry: &str) -> String {
    format!("ALLPACA_SPV_{}", entry.to_ascii_uppercase())
}

fn compile_shaders() -> Result<(), String> {
    use cargo_gpu_install::install::Install;
    use cargo_gpu_install::spirv_builder::SpirvMetadata;

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").map_err(|e| e.to_string())?);
    let shader_crate = manifest.join("../allpaca-shaders");
    if !shader_crate.join("Cargo.toml").is_file() {
        return Err("allpaca-shaders Cargo.toml missing".into());
    }

    let backend = Install::from_shader_crate(shader_crate.clone())
        .within_build_script()
        .run()
        .map_err(|e| format!("cargo-gpu-install: {e}"))?;

    let mut builder = backend.to_spirv_builder(shader_crate, "spirv-unknown-vulkan1.3");
    builder.build_script.defaults = true;
    builder.build_script.env_shader_spv_path = Some(false);
    builder.spirv_metadata = SpirvMetadata::NameVariables;
    builder.multimodule = true;
    let result = builder.build().map_err(|e| format!("spirv-builder: {e}"))?;
    let map = result.module.unwrap_multi();
    let out = PathBuf::from(env::var("OUT_DIR").map_err(|e| e.to_string())?);
    for entry in ENTRIES {
        let src = map.get(*entry).or_else(|| {
            map.iter()
                .find(|(k, _)| k.rsplit("::").next() == Some(*entry))
                .map(|(_, p)| p)
        });
        let Some(src) = src else {
            return Err(format!(
                "missing SPIR-V for {entry}; rust-gpu produced {:?}",
                map.keys().collect::<Vec<_>>()
            ));
        };
        let dest = out.join(format!("{entry}.spv"));
        std::fs::copy(src, &dest).map_err(|e| format!("copy {entry}.spv: {e}"))?;
        let dest = dest.display().to_string().replace('\\', "/");
        println!("cargo:rustc-env={}={dest}", env_name(entry));
    }
    try_compile_glsl_f16(&out);
    Ok(())
}

fn try_compile_glsl_f16(out: &std::path::Path) {
    let manifest = match env::var("CARGO_MANIFEST_DIR") {
        Ok(p) => PathBuf::from(p),
        Err(_) => return,
    };
    let src = manifest.join("glsl/gemm_f16.comp");
    if !src.is_file() {
        return;
    }
    let dest = out.join("gemm_f16_glsl.spv");
    let glslang = find_glslang();
    let output = std::process::Command::new(&glslang)
        .arg("-V")
        .arg("-S")
        .arg("comp")
        .arg("-e")
        .arg("main")
        .arg("-o")
        .arg(&dest)
        .arg(&src)
        .output();
    match output {
        Ok(o) if o.status.success() && dest.is_file() => {
            let dest = dest.display().to_string().replace('\\', "/");
            println!("cargo:rustc-cfg=allpaca_vulkan_glsl_f16");
            println!("cargo:rustc-env=ALLPACA_SPV_GEMM_F16_GLSL={dest}");
        }
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            println!(
                "cargo:warning=glslangValidator failed ({}); using rust-gpu F16 unpack: {err}",
                o.status
            );
        }
        Err(e) => {
            println!(
                "cargo:warning=glslangValidator not found ({}: {e}); using rust-gpu F16 unpack",
                glslang.display()
            );
        }
    }
}

fn find_glslang() -> PathBuf {
    if let Ok(p) = env::var("GLSLANG_VALIDATOR") {
        return PathBuf::from(p);
    }
    if let Some(p) = vulkan_sdk_glslang() {
        return p;
    }
    PathBuf::from("glslangValidator")
}

fn vulkan_sdk_glslang() -> Option<PathBuf> {
    let mut sdks = Vec::new();
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        sdks.push(PathBuf::from(sdk));
    }
    if let Ok(entries) = std::fs::read_dir(r"C:\VulkanSDK") {
        let mut vers: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        vers.sort_by_key(|e| e.file_name());
        for e in vers.into_iter().rev() {
            sdks.push(e.path());
        }
    }
    for sdk in sdks {
        let win = sdk.join("Bin/glslangValidator.exe");
        if win.is_file() {
            return Some(win);
        }
        let unix = sdk.join("bin/glslangValidator");
        if unix.is_file() {
            return Some(unix);
        }
    }
    None
}
