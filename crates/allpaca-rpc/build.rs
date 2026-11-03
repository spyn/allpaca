//! Fingerprints the sources a worker is built from, so a coordinator can
//! tell a worker built from other sources (`BUILD_ID`). Line endings do not
//! count: a checkout and a copy of a Windows working tree give the same id.

use std::path::{Path, PathBuf};

/// Everything a worker runs, relative to the workspace root.
const SOURCES: &[&str] = &[
    "crates/allpaca-rpc/src",
    "crates/allpaca-hip/src",
    "crates/allpaca-cpu/src",
    "crates/allpaca-cuda/src",
    "crates/allpaca-core/src/qwen4_gpu.rs",
    "crates/allpaca-core/src/qwen4_remote.rs",
    "crates/allpaca-core/src/qwen3next_remote.rs",
    "crates/allpaca-core/src/model.rs",
    "crates/allpaca-core/src/deltanet.rs",
    "hip/src",
    "hip/include",
    "cuda/src",
    "cuda/include",
];

fn collect(path: &Path, files: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            collect(&entry.path(), files);
        }
    } else if path.is_file() {
        files.push(path.to_path_buf());
    }
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let mut files = Vec::new();
    for source in SOURCES {
        println!("cargo:rerun-if-changed={}", root.join(source).display());
        collect(&root.join(source), &mut files);
    }
    let mut named: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|f| (f.strip_prefix(&root).unwrap_or(&f).to_string_lossy().replace('\\', "/"), f))
        .collect();
    named.sort();
    // FNV-1a over each file's name and its bytes without carriage returns.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |byte: u8| hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    for (name, path) in &named {
        name.bytes().for_each(&mut eat);
        eat(0);
        std::fs::read(path).unwrap_or_default().into_iter().filter(|b| *b != b'\r').for_each(&mut eat);
        eat(0);
    }
    println!("cargo:rustc-env=ALLPACA_RPC_BUILD_ID={hash:016x}");
    println!("cargo:rerun-if-changed=build.rs");
}
