//! One-time warnings for kernel paths that otherwise stay silent, and the
//! per-device map of weight type to kernel family.

use std::sync::atomic::{AtomicU32, Ordering};
#[cfg(test)]
use std::sync::{Mutex, MutexGuard};

use allpaca_types::AllpacaError;

/// A path that used to fail or slow down without a log line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    Mxfp4SwigluSplit,
    Dot4Scalar,
    #[cfg(test)]
    AttnNoScratch,
}

/// Which attention kernel a decode or prefill launch takes.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnDecode {
    /// One query and one chunk: split GQA writes the output directly.
    SplitOneChunk,
    /// One query, several chunks, scratch allocated: split GQA plus a merge.
    ScratchMerge,
    /// One query, several chunks, scratch missing: the Q-tiled kernel.
    NoScratchQtile,
    /// More than one query: the Q-tiled kernel is the normal path.
    QtilePrefill,
}

/// One weight type and the GEMM family that multiplies it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelFamily {
    pub ggml_type: u16,
    pub weight: &'static str,
    pub family: &'static str,
}

const FAMILIES: &[KernelFamily] = &[
    KernelFamily {
        ggml_type: 0,
        weight: "f32",
        family: "gemm-f32",
    },
    KernelFamily {
        ggml_type: 1,
        weight: "f16",
        family: "gemm-f16",
    },
    KernelFamily {
        ggml_type: 8,
        weight: "q8_0",
        family: "gemm-q8_0",
    },
    KernelFamily {
        ggml_type: 11,
        weight: "q3_k",
        family: "gemm-q3_k",
    },
    KernelFamily {
        ggml_type: 12,
        weight: "q4_k",
        family: "gemm-q4_k",
    },
    KernelFamily {
        ggml_type: 13,
        weight: "q5_k",
        family: "gemm-q5_k",
    },
    KernelFamily {
        ggml_type: 14,
        weight: "q6_k",
        family: "gemm-q6_k",
    },
    KernelFamily {
        ggml_type: 30,
        weight: "bf16",
        family: "gemm-bf16",
    },
    KernelFamily {
        ggml_type: 34,
        weight: "tq1_0",
        family: "gemm-tq1_0",
    },
    KernelFamily {
        ggml_type: 35,
        weight: "tq2_0",
        family: "gemm-tq2_0",
    },
    KernelFamily {
        ggml_type: 39,
        weight: "mxfp4",
        family: "gemm-mxfp4",
    },
    KernelFamily {
        ggml_type: 42,
        weight: "q2_0",
        family: "gemm-q2_0",
    },
    KernelFamily {
        ggml_type: 100,
        weight: "rocmfp4",
        family: "rocmfp4-i8",
    },
    KernelFamily {
        ggml_type: 101,
        weight: "rocmfp4-fast",
        family: "rocmfp4-fast-i8",
    },
    KernelFamily {
        ggml_type: 200,
        weight: "p48fp4",
        family: "p48fp4",
    },
];

#[cfg(test)]
static WARN_GATE: Mutex<()> = Mutex::new(());
static MXFP4_WARNED: AtomicU32 = AtomicU32::new(0);
static DOT4_WARNED: AtomicU32 = AtomicU32::new(0);
#[cfg(test)]
static ATTN_WARNED: AtomicU32 = AtomicU32::new(0);
static FAMILIES_LOGGED: AtomicU32 = AtomicU32::new(0);

fn slot(kind: Fallback) -> &'static AtomicU32 {
    match kind {
        Fallback::Mxfp4SwigluSplit => &MXFP4_WARNED,
        Fallback::Dot4Scalar => &DOT4_WARNED,
        #[cfg(test)]
        Fallback::AttnNoScratch => &ATTN_WARNED,
    }
}

/// Record `kind` and emit one warning. Later calls for the same kind are quiet.
pub fn note(kind: Fallback) -> bool {
    let printed = slot(kind);
    if printed
        .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return false;
    }
    match kind {
        Fallback::Mxfp4SwigluSplit => {
            tracing::warn!(
                fallback = "mxfp4-swiglu-split",
                "fused MXFP4 SwiGLU fell back to split GEMMs"
            );
        }
        Fallback::Dot4Scalar => {
            tracing::warn!(
                fallback = "dot4-scalar",
                "dot4 i8 using the scalar fallback"
            );
        }
        #[cfg(test)]
        Fallback::AttnNoScratch => {
            tracing::warn!(
                fallback = "attn-no-scratch",
                "attention decode fell back to the Q-tiled kernel because scratch was unavailable"
            );
        }
    }
    true
}

/// How many times `kind` has warned in this process (0 or 1).
#[cfg(test)]
pub fn warn_count(kind: Fallback) -> u32 {
    slot(kind).load(Ordering::Relaxed)
}

#[cfg(test)]
pub fn warned(kind: Fallback) -> bool {
    warn_count(kind) > 0
}

/// `true` when the fused MXFP4 SwiGLU ran. `Ok(false)` is the split-GEMM fallback.
///
/// A positive SwiGLU limit is only applied by the fused kernel, so a fused
/// failure in that mode stays an error instead of silently dropping the clamp.
pub fn fused_mxfp4_swiglu_held(
    fused: Result<(), AllpacaError>,
    limit: f32,
) -> Result<bool, AllpacaError> {
    match fused {
        Ok(()) => Ok(true),
        Err(err) if limit > 0.0 => Err(err),
        Err(_) => {
            note(Fallback::Mxfp4SwigluSplit);
            Ok(false)
        }
    }
}

fn arch_base(arch: &str) -> &str {
    arch.split(':').next().unwrap_or(arch)
}

/// `sdot4` on CDNA, `sudot4` on RDNA3/RDNA3.5/RDNA4, otherwise the scalar product.
pub fn dot4_family(arch: &str) -> &'static str {
    match arch_base(arch) {
        "gfx906" | "gfx908" | "gfx90a" | "gfx942" => "sdot4",
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1151" | "gfx1200" | "gfx1201" => "sudot4",
        _ => "scalar",
    }
}

/// Warn once when this device's dot4 ISA is the scalar product.
pub fn warn_dot4_if_scalar(arch: &str) {
    if dot4_family(arch) == "scalar" {
        note(Fallback::Dot4Scalar);
    }
}

/// Host copy of the device scalar branch in `allpaca_dot4_i8`.
#[cfg(test)]
pub fn dot4_i8_scalar(a: i32, b: i32, c: i32) -> i32 {
    note(Fallback::Dot4Scalar);
    let x = a.to_le_bytes();
    let y = b.to_le_bytes();
    let prod = |i: usize| i32::from(x[i] as i8) * i32::from(y[i] as i8);
    c + prod(0) + prod(1) + prod(2) + prod(3)
}

/// Same choice as `launch_attention_tiled`: scratch is only required for a
/// single query split across more than one chunk.
#[cfg(test)]
pub fn attn_decode_choice(n_q: i32, n_chunks: i32, scratch_ok: bool) -> AttnDecode {
    if n_q == 1 && n_chunks <= 1 {
        return AttnDecode::SplitOneChunk;
    }
    if n_q == 1 && scratch_ok {
        return AttnDecode::ScratchMerge;
    }
    if n_q == 1 {
        note(Fallback::AttnNoScratch);
        return AttnDecode::NoScratchQtile;
    }
    AttnDecode::QtilePrefill
}

/// gfx1200 and gfx1201 have dedicated GEMMs for ggml types 100, 101 and 200.
#[cfg(test)]
pub fn gfx12_dispatches_custom_fp4(arch: &str) -> bool {
    let base = arch_base(arch);
    if base != "gfx1200" && base != "gfx1201" {
        return false;
    }
    [100u16, 101, 200].iter().all(|ty| {
        FAMILIES
            .iter()
            .any(|row| row.ggml_type == *ty && !row.family.contains("dequant"))
    })
}

/// Lines a startup log would emit for `device` on `arch`.
#[cfg(test)]
pub fn kernel_family_lines(device: i32, arch: &str) -> Vec<String> {
    let mut lines = Vec::with_capacity(FAMILIES.len() + 1);
    lines.push(format!(
        "device {device} arch {arch} dot4 {}",
        dot4_family(arch)
    ));
    for row in FAMILIES {
        lines.push(format!(
            "device {device} arch {arch} ggml {} {} -> {}",
            row.ggml_type, row.weight, row.family
        ));
    }
    lines
}

/// Log each weight type's kernel family once per device. A scalar dot4 warns too.
pub fn log_kernel_families(device: i32, arch: &str) {
    if (0..31).contains(&device) {
        let bit = 1u32 << device;
        if FAMILIES_LOGGED.fetch_or(bit, Ordering::Relaxed) & bit != 0 {
            return;
        }
    }
    warn_dot4_if_scalar(arch);
    let dot4 = dot4_family(arch);
    tracing::info!(device, arch, dot4, "kernel family");
    for row in FAMILIES {
        tracing::info!(
            device,
            arch,
            ggml_type = row.ggml_type,
            weight = row.weight,
            family = row.family,
            "kernel family"
        );
    }
}

#[cfg(test)]
fn lock_warnings() -> MutexGuard<'static, ()> {
    WARN_GATE.lock().unwrap_or_else(|err| err.into_inner())
}

#[cfg(test)]
fn reset_warnings() {
    MXFP4_WARNED.store(0, Ordering::Relaxed);
    DOT4_WARNED.store(0, Ordering::Relaxed);
    ATTN_WARNED.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use allpaca_types::DegradedError;

    use super::*;

    fn rejected() -> AllpacaError {
        AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "fused mxfp4 swiglu rejected the shape".into(),
        ))
    }

    #[test]
    fn fused_mxfp4_swiglu_failure_warns_once_when_the_limit_is_off() {
        let _gate = lock_warnings();
        reset_warnings();
        let first = fused_mxfp4_swiglu_held(Err(rejected()), 0.0).unwrap();
        assert!(!first, "a rejected fused kernel must take the split path");
        assert!(warned(Fallback::Mxfp4SwigluSplit));
        let second = fused_mxfp4_swiglu_held(Err(rejected()), 0.0).unwrap();
        assert!(!second);
        assert_eq!(warn_count(Fallback::Mxfp4SwigluSplit), 1);
    }

    #[test]
    fn a_fused_mxfp4_failure_with_the_limit_on_stays_an_error() {
        let _gate = lock_warnings();
        reset_warnings();
        assert!(fused_mxfp4_swiglu_held(Err(rejected()), 1.0).is_err());
        assert_eq!(warn_count(Fallback::Mxfp4SwigluSplit), 0);
        assert!(fused_mxfp4_swiglu_held(Ok(()), 0.0).unwrap());
        assert_eq!(warn_count(Fallback::Mxfp4SwigluSplit), 0);
    }

    #[test]
    fn scalar_dot4_warns_once_and_matches_the_byte_product() {
        let _gate = lock_warnings();
        reset_warnings();
        assert_eq!(dot4_family("gfx1201"), "sudot4");
        assert_eq!(dot4_family("gfx1200:sramecc-:xnack-"), "sudot4");
        assert_eq!(dot4_family("gfx906:sramecc+:xnack-"), "sdot4");
        assert_eq!(dot4_family("gfx1151"), "sudot4");
        warn_dot4_if_scalar("gfx1201");
        assert!(!warned(Fallback::Dot4Scalar));
        warn_dot4_if_scalar("gfx900");
        assert!(warned(Fallback::Dot4Scalar));
        warn_dot4_if_scalar("gfx900");
        assert_eq!(warn_count(Fallback::Dot4Scalar), 1);
        let a = i32::from_le_bytes([1, 2, 3, 4]);
        let b = i32::from_le_bytes([5, 6, 7, 8]);
        assert_eq!(dot4_i8_scalar(a, b, 9), 9 + 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8);
        assert_eq!(warn_count(Fallback::Dot4Scalar), 1);
    }

    #[test]
    fn attention_without_scratch_warns_once() {
        let _gate = lock_warnings();
        reset_warnings();
        assert_eq!(attn_decode_choice(1, 1, false), AttnDecode::SplitOneChunk);
        assert_eq!(attn_decode_choice(1, 4, true), AttnDecode::ScratchMerge);
        assert!(!warned(Fallback::AttnNoScratch));
        assert_eq!(attn_decode_choice(1, 4, false), AttnDecode::NoScratchQtile);
        assert_eq!(attn_decode_choice(1, 4, false), AttnDecode::NoScratchQtile);
        assert_eq!(warn_count(Fallback::AttnNoScratch), 1);
        assert_eq!(attn_decode_choice(8, 4, false), AttnDecode::QtilePrefill);
        assert_eq!(warn_count(Fallback::AttnNoScratch), 1);
    }

    #[test]
    fn gfx12_dispatch_covers_types_100_101_and_200() {
        for arch in ["gfx1200", "gfx1201:sramecc-:xnack-"] {
            assert!(gfx12_dispatches_custom_fp4(arch), "{arch}");
            let lines = kernel_family_lines(0, arch);
            for ty in ["ggml 100 ", "ggml 101 ", "ggml 200 "] {
                assert!(
                    lines
                        .iter()
                        .any(|line| line.contains(ty) && !line.contains("dequant")),
                    "{arch} missing {ty} in {lines:?}"
                );
            }
            assert!(lines.iter().any(|line| line.contains("rocmfp4-i8")));
            assert!(lines.iter().any(|line| line.contains("rocmfp4-fast-i8")));
            assert!(lines.iter().any(|line| line.contains("-> p48fp4")));
            assert!(lines.iter().any(|line| line.contains("sudot4")));
        }
        assert!(!gfx12_dispatches_custom_fp4("gfx906"));
        assert!(!gfx12_dispatches_custom_fp4("gfx900"));
    }
}
