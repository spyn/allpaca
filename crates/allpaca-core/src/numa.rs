//! NUMA-aware expert placement (GEM vertical).
//!
//! - Probe host NUMA nodes
//! - Assign each expert a preferred node
//! - Bind the current thread to that node's CPUs, then first-touch allocate
//!   so host expert pages land on the preferred node
//! - Re-bind before CPU `q*` expert MLP so reads stay local

use std::sync::atomic::{AtomicU64, Ordering};

use crate::weight_set::{ExpertId, LayerFfn};

/// Host NUMA map used to co-locate expert DRAM with CPU expert threads.
#[derive(Debug)]
pub struct NumaTopology {
    /// Logical NUMA nodes (at least 1).
    pub n_nodes: usize,
    /// True when probe used OS topology APIs (vs synthetic fallback).
    pub probed: bool,
    /// Successful thread binds (affinity applied).
    binds_ok: AtomicU64,
    /// Bind skipped (single node, unsupported OS, or affinity API failed).
    binds_skip: AtomicU64,
    /// Expert LayerFfn clones allocated under a node bind (first-touch pin).
    pins: AtomicU64,
}

impl Clone for NumaTopology {
    fn clone(&self) -> Self {
        Self {
            n_nodes: self.n_nodes,
            probed: self.probed,
            binds_ok: AtomicU64::new(self.binds_ok.load(Ordering::Relaxed)),
            binds_skip: AtomicU64::new(self.binds_skip.load(Ordering::Relaxed)),
            pins: AtomicU64::new(self.pins.load(Ordering::Relaxed)),
        }
    }
}

impl PartialEq for NumaTopology {
    fn eq(&self, other: &Self) -> bool {
        self.n_nodes == other.n_nodes && self.probed == other.probed
    }
}

impl Eq for NumaTopology {}

impl NumaTopology {
    /// Probe the host. Single-node machines still return `n_nodes = 1`.
    pub fn probe() -> Self {
        match probe_node_count() {
            Some(n) => Self::new(n.max(1), true),
            None => Self::new(1, false),
        }
    }

    /// Test / spike helper: force a topology without touching the OS.
    pub fn with_nodes(n_nodes: usize) -> Self {
        Self::new(n_nodes.max(1), false)
    }

    fn new(n_nodes: usize, probed: bool) -> Self {
        Self {
            n_nodes,
            probed,
            binds_ok: AtomicU64::new(0),
            binds_skip: AtomicU64::new(0),
            pins: AtomicU64::new(0),
        }
    }

    /// Interleave experts across nodes: `(layer*1024 + expert) % n_nodes`.
    pub fn preferred_node(&self, id: ExpertId) -> usize {
        let key = (id.layer as usize)
            .wrapping_mul(1024)
            .wrapping_add(id.expert as usize);
        key % self.n_nodes
    }

    pub fn binds_ok(&self) -> u64 {
        self.binds_ok.load(Ordering::Relaxed)
    }

    pub fn binds_skip(&self) -> u64 {
        self.binds_skip.load(Ordering::Relaxed)
    }

    pub fn pins(&self) -> u64 {
        self.pins.load(Ordering::Relaxed)
    }

    /// Run `f` with the calling thread bound to `node` when the OS allows it.
    pub fn run_on_node<R>(&self, node: usize, f: impl FnOnce() -> R) -> R {
        let node = node % self.n_nodes;
        let _guard = self.bind_current_thread(node);
        f()
    }

    /// First-touch clone of expert weights onto `node` (bind → allocate → copy).
    pub fn clone_ffn_on_node(&self, node: usize, src: &LayerFfn) -> LayerFfn {
        let node = node % self.n_nodes;
        let out = self.run_on_node(node, || LayerFfn {
            ffn_norm: src.ffn_norm.clone(),
            w_gate: src.w_gate.clone(),
            w_up: src.w_up.clone(),
            w_down: src.w_down.clone(),
        });
        self.pins.fetch_add(1, Ordering::Relaxed);
        out
    }

    fn bind_current_thread(&self, node: usize) -> AffinityGuard {
        if self.n_nodes <= 1 {
            self.binds_skip.fetch_add(1, Ordering::Relaxed);
            return AffinityGuard::noop();
        }
        match try_bind_current_thread(node) {
            Some(prev) => {
                self.binds_ok.fetch_add(1, Ordering::Relaxed);
                AffinityGuard::restoring(prev)
            }
            None => {
                self.binds_skip.fetch_add(1, Ordering::Relaxed);
                AffinityGuard::noop()
            }
        }
    }
}

/// Restores prior affinity on drop.
struct AffinityGuard {
    prev: Option<ThreadAffinity>,
}

impl AffinityGuard {
    fn noop() -> Self {
        Self { prev: None }
    }

    fn restoring(prev: ThreadAffinity) -> Self {
        Self { prev: Some(prev) }
    }
}

impl Drop for AffinityGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.prev.take() {
            restore_thread_affinity(prev);
        }
    }
}

#[derive(Clone, Copy)]
struct ThreadAffinity {
    #[cfg(windows)]
    mask: usize,
    #[cfg(all(unix, not(target_os = "macos")))]
    mask: u64,
    #[cfg(any(target_os = "macos", not(any(windows, unix))))]
    _unused: (),
}

#[cfg(windows)]
fn probe_node_count() -> Option<usize> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetNumaHighestNodeNumber(highest_node_number: *mut u32) -> i32;
    }
    let mut highest: u32 = 0;
    let ok = unsafe { GetNumaHighestNodeNumber(&mut highest) };
    if ok == 0 {
        return None;
    }
    Some(highest as usize + 1)
}

#[cfg(windows)]
fn try_bind_current_thread(node: usize) -> Option<ThreadAffinity> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn GetNumaNodeProcessorMask(node: u8, processor_mask: *mut u64) -> i32;
        fn SetThreadAffinityMask(thread: *mut std::ffi::c_void, mask: usize) -> usize;
    }
    if node > u8::MAX as usize {
        return None;
    }
    let mut mask: u64 = 0;
    let ok = unsafe { GetNumaNodeProcessorMask(node as u8, &mut mask) };
    if ok == 0 || mask == 0 {
        return None;
    }
    let thread = unsafe { GetCurrentThread() };
    let prev = unsafe { SetThreadAffinityMask(thread, mask as usize) };
    if prev == 0 {
        return None;
    }
    Some(ThreadAffinity { mask: prev })
}

#[cfg(windows)]
fn restore_thread_affinity(prev: ThreadAffinity) {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn SetThreadAffinityMask(thread: *mut std::ffi::c_void, mask: usize) -> usize;
    }
    let thread = unsafe { GetCurrentThread() };
    let _ = unsafe { SetThreadAffinityMask(thread, prev.mask) };
}

#[cfg(all(unix, not(target_os = "macos")))]
fn probe_node_count() -> Option<usize> {
    let mut n = 0usize;
    let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") else {
        return None;
    };
    for e in entries.flatten() {
        let name = e.file_name();
        let s = name.to_string_lossy();
        if s.starts_with("node") && s[4..].chars().all(|c| c.is_ascii_digit()) {
            n += 1;
        }
    }
    if n == 0 {
        None
    } else {
        Some(n)
    }
}

/// Soft bind via `taskset`-style mask built from sysfs (no libc dep).
#[cfg(all(unix, not(target_os = "macos")))]
fn try_bind_current_thread(node: usize) -> Option<ThreadAffinity> {
    let path = format!("/sys/devices/system/node/node{node}/cpulist");
    let text = std::fs::read_to_string(path).ok()?;
    let mask = cpulist_to_mask(&text)?;
    if mask == 0 {
        return None;
    }
    // sched_setaffinity(0, ...) — declare locally to avoid a libc crate.
    #[repr(C)]
    struct CpuSet {
        bits: [u64; 16],
    }
    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut CpuSet) -> i32;
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const CpuSet) -> i32;
    }
    let mut prev = CpuSet { bits: [0; 16] };
    if unsafe { sched_getaffinity(0, std::mem::size_of::<CpuSet>(), &mut prev) } != 0 {
        return None;
    }
    let mut set = CpuSet { bits: [0; 16] };
    set.bits[0] = mask;
    if unsafe { sched_setaffinity(0, std::mem::size_of::<CpuSet>(), &set) } != 0 {
        return None;
    }
    Some(ThreadAffinity { mask: prev.bits[0] })
}

#[cfg(all(unix, not(target_os = "macos")))]
fn restore_thread_affinity(prev: ThreadAffinity) {
    #[repr(C)]
    struct CpuSet {
        bits: [u64; 16],
    }
    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const CpuSet) -> i32;
    }
    let mut set = CpuSet { bits: [0; 16] };
    set.bits[0] = prev.mask;
    let _ = unsafe { sched_setaffinity(0, std::mem::size_of::<CpuSet>(), &set) };
}

#[cfg(all(unix, not(target_os = "macos")))]
fn cpulist_to_mask(text: &str) -> Option<u64> {
    let mut mask = 0u64;
    for part in text.trim().split(',') {
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let start: u32 = a.parse().ok()?;
            let end: u32 = b.parse().ok()?;
            for cpu in start..=end {
                if cpu < 64 {
                    mask |= 1u64 << cpu;
                }
            }
        } else {
            let cpu: u32 = part.parse().ok()?;
            if cpu < 64 {
                mask |= 1u64 << cpu;
            }
        }
    }
    Some(mask)
}

#[cfg(target_os = "macos")]
fn probe_node_count() -> Option<usize> {
    Some(1)
}

#[cfg(target_os = "macos")]
fn try_bind_current_thread(_node: usize) -> Option<ThreadAffinity> {
    None
}

#[cfg(target_os = "macos")]
fn restore_thread_affinity(_prev: ThreadAffinity) {}

#[cfg(not(any(windows, unix)))]
fn probe_node_count() -> Option<usize> {
    None
}

#[cfg(not(any(windows, unix)))]
fn try_bind_current_thread(_node: usize) -> Option<ThreadAffinity> {
    None
}

#[cfg(not(any(windows, unix)))]
fn restore_thread_affinity(_prev: ThreadAffinity) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reports_at_least_one_node() {
        let topo = NumaTopology::probe();
        assert!(topo.n_nodes >= 1, "topology={topo:?}");
    }

    #[test]
    fn preferred_node_interleaves_across_two_nodes() {
        let topo = NumaTopology::with_nodes(2);
        let mut seen = [false; 2];
        for e in 0..8u32 {
            let node = topo.preferred_node(ExpertId {
                layer: 0,
                expert: e,
            });
            assert!(node < 2);
            seen[node] = true;
        }
        assert!(seen[0] && seen[1], "experts must spread across both nodes");
    }

    #[test]
    fn preferred_node_stable_for_same_expert() {
        let topo = NumaTopology::with_nodes(4);
        let id = ExpertId {
            layer: 3,
            expert: 7,
        };
        assert_eq!(topo.preferred_node(id), topo.preferred_node(id));
    }

    #[test]
    fn run_on_node_executes_and_counts_bind_attempt() {
        let topo = NumaTopology::with_nodes(2);
        let v = topo.run_on_node(1, || 42);
        assert_eq!(v, 42);
        assert!(
            topo.binds_ok() + topo.binds_skip() >= 1,
            "bind attempt must be recorded"
        );
    }

    #[test]
    fn clone_ffn_on_node_preserves_weights_and_pins() {
        let topo = NumaTopology::with_nodes(2);
        let src = LayerFfn {
            w_gate: vec![1.0, 2.0],
            w_up: vec![3.0],
            w_down: vec![4.0, 5.0],
            ..LayerFfn::default()
        };
        let before = topo.pins();
        let got = topo.clone_ffn_on_node(0, &src);
        assert_eq!(got.w_gate, src.w_gate);
        assert_eq!(got.w_up, src.w_up);
        assert_eq!(got.w_down, src.w_down);
        assert_eq!(topo.pins(), before + 1);
    }
}
