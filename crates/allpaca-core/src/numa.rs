//! NUMA topology spike (GEM): probe nodes + preferred expert placement.
//!
//! Full path later: pin expert pages to the preferred node and bind the CPU
//! expert worker to that node's CPUs before `q*` host runs. This module only
//! exposes probe + assignment so HostExpertPool can record intent.

use crate::weight_set::ExpertId;

/// Host NUMA map used to co-locate expert DRAM with CPU expert threads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaTopology {
    /// Logical NUMA nodes (at least 1).
    pub n_nodes: usize,
    /// True when probe used OS topology APIs (vs synthetic fallback).
    pub probed: bool,
}

impl NumaTopology {
    /// Probe the host. Single-node machines still return `n_nodes = 1`.
    pub fn probe() -> Self {
        match probe_node_count() {
            Some(n) => Self {
                n_nodes: n.max(1),
                probed: true,
            },
            None => Self {
                n_nodes: 1,
                probed: false,
            },
        }
    }

    /// Test / spike helper: force a topology without touching the OS.
    pub fn with_nodes(n_nodes: usize) -> Self {
        Self {
            n_nodes: n_nodes.max(1),
            probed: false,
        }
    }

    /// Interleave experts across nodes: `(layer*1024 + expert) % n_nodes`.
    pub fn preferred_node(&self, id: ExpertId) -> usize {
        let key = (id.layer as usize)
            .wrapping_mul(1024)
            .wrapping_add(id.expert as usize);
        key % self.n_nodes
    }
}

#[cfg(windows)]
fn probe_node_count() -> Option<usize> {
    // GetNumaHighestNodeNumber: highest node index present (0-based).
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

#[cfg(target_os = "macos")]
fn probe_node_count() -> Option<usize> {
    // UMA: treat as one node for placement math.
    Some(1)
}

#[cfg(not(any(windows, unix)))]
fn probe_node_count() -> Option<usize> {
    None
}

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
}
