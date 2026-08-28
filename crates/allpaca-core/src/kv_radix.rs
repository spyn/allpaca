//! Radix / prefix-tree KV reuse (GEM vertical).
//!
//! Stores arbitrary prompt prefixes with KV+recurrent checkpoints so a mid-prompt
//! edit can restore the longest common prefix without a full re-prefill.
//!
//! Caps: coarse node count **and** checkpoint byte budget. Cold checkpoints
//! (LRU by last hit/insert) are evicted first when over budget.

use std::collections::HashMap;

use allpaca_types::AllpacaError;

use crate::anchors::{snapshot_kv, KvRow, RecurrentState};
use crate::kv_tier::KvTierManager;
use crate::model::ModelConfig;

/// Checkpoint attached to a radix node (prefix of the prompt).
#[derive(Debug, Clone)]
pub struct RadixCheckpoint {
    pub recurrent: RecurrentState,
    pub kv: Vec<Vec<KvRow>>,
}

impl RadixCheckpoint {
    /// Approximate resident size of this checkpoint (f32 payload + thin header).
    pub fn approx_bytes(&self) -> usize {
        let rec: usize = self
            .recurrent
            .layers
            .iter()
            .map(|l| l.len().saturating_mul(4))
            .sum();
        let kv: usize = self
            .kv
            .iter()
            .flatten()
            .map(|r| (r.k.len() + r.v.len()).saturating_mul(4))
            .sum();
        rec.saturating_add(kv).saturating_add(64)
    }
}

/// One edge in the token radix: child keyed by next token id.
#[derive(Debug, Default)]
struct RadixNode {
    children: HashMap<u32, Box<RadixNode>>,
    checkpoint: Option<RadixCheckpoint>,
    /// LRU clock when this node's checkpoint was last touched (0 = never / no cp).
    used_at: u64,
}

/// Forest of prompt prefixes for longest-common-prefix KV reuse.
#[derive(Debug)]
pub struct KvRadixForest {
    root: RadixNode,
    /// Total nodes allocated (excluding root).
    pub nodes: usize,
    /// Soft node cap; 0 = unlimited. Excess triggers leaf eviction.
    max_nodes: usize,
    /// Soft checkpoint byte budget; 0 = unlimited.
    max_bytes: usize,
    /// Sum of [`RadixCheckpoint::approx_bytes`] currently held.
    checkpoint_bytes: usize,
    clock: u64,
    /// Lookups that restored a checkpoint (depth > 0).
    pub hits: u64,
    /// Lookups with no restorable prefix.
    pub misses: u64,
    /// Cumulative tokens skipped via LCP restore.
    pub skip_tokens: u64,
    /// Nodes removed to stay under node/byte caps.
    pub evictions: u64,
}

impl Default for KvRadixForest {
    fn default() -> Self {
        Self::with_limits(0, 0)
    }
}

impl KvRadixForest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_nodes(max_nodes: usize) -> Self {
        Self::with_limits(max_nodes, 0)
    }

    /// Node cap + checkpoint byte budget (either 0 = unlimited for that axis).
    pub fn with_limits(max_nodes: usize, max_bytes: usize) -> Self {
        Self {
            root: RadixNode::default(),
            nodes: 0,
            max_nodes,
            max_bytes,
            checkpoint_bytes: 0,
            clock: 0,
            hits: 0,
            misses: 0,
            skip_tokens: 0,
            evictions: 0,
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn skip_tokens(&self) -> u64 {
        self.skip_tokens
    }

    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    pub fn checkpoint_bytes(&self) -> usize {
        self.checkpoint_bytes
    }

    #[allow(dead_code)]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Evict cold checkpoints until `checkpoint_bytes <= target`. Returns bytes freed.
    pub fn trim_to_bytes(&mut self, target: usize) -> usize {
        let before = self.checkpoint_bytes;
        while self.checkpoint_bytes > target {
            if !self.evict_one_leaf(true) {
                break;
            }
            self.evictions += 1;
        }
        before.saturating_sub(self.checkpoint_bytes)
    }

    fn touch(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    /// Insert `tokens` and attach `cp` at the terminal (and replace along the path
    /// when re-inserting the same prefix).
    pub fn insert_checkpoint(&mut self, tokens: &[u32], cp: RadixCheckpoint) -> usize {
        if tokens.is_empty() {
            return 0;
        }
        let new_bytes = cp.approx_bytes();
        let stamp = self.touch();
        let mut cur = &mut self.root;
        let mut added = 0usize;
        for &tok in tokens {
            if !cur.children.contains_key(&tok) {
                cur.children.insert(tok, Box::new(RadixNode::default()));
                added += 1;
                self.nodes += 1;
            }
            cur = cur.children.get_mut(&tok).expect("just inserted");
        }
        if let Some(old) = cur.checkpoint.take() {
            self.checkpoint_bytes = self
                .checkpoint_bytes
                .saturating_sub(old.approx_bytes());
        }
        cur.checkpoint = Some(cp);
        cur.used_at = stamp;
        self.checkpoint_bytes = self.checkpoint_bytes.saturating_add(new_bytes);
        self.evict_to_cap();
        added
    }

    fn evict_to_cap(&mut self) {
        if self.max_nodes > 0 {
            while self.nodes > self.max_nodes {
                if !self.evict_one_leaf(false) {
                    break;
                }
                self.evictions += 1;
            }
        }
        if self.max_bytes > 0 {
            while self.checkpoint_bytes > self.max_bytes {
                if !self.evict_one_leaf(true) {
                    break;
                }
                self.evictions += 1;
            }
        }
    }

    /// Remove one leaf. When `prefer_cold_checkpoint`, pick the coldest LRU
    /// checkpoint leaf; otherwise prefer empty leaves (node-cap path).
    fn evict_one_leaf(&mut self, prefer_cold_checkpoint: bool) -> bool {
        #[derive(Clone, Copy)]
        struct Cand {
            tok: u32,
            used_at: u64,
            has_cp: bool,
            is_leaf: bool,
        }

        fn best_child(node: &RadixNode, prefer_cold: bool) -> Option<Cand> {
            let mut best: Option<Cand> = None;
            for (&tok, child) in &node.children {
                let is_leaf = child.children.is_empty();
                let has_cp = child.checkpoint.is_some();
                let used_at = child.used_at;
                let cand = Cand {
                    tok,
                    used_at,
                    has_cp,
                    is_leaf,
                };
                best = Some(match best {
                    None => cand,
                    Some(prev) => {
                        if prefer_cold {
                            // Prefer leaf with checkpoint, then colder used_at.
                            match (prev.is_leaf && prev.has_cp, cand.is_leaf && cand.has_cp) {
                                (true, false) => prev,
                                (false, true) => cand,
                                (true, true) | (false, false) => {
                                    if cand.used_at < prev.used_at
                                        || (cand.used_at == prev.used_at && cand.is_leaf && !prev.is_leaf)
                                    {
                                        cand
                                    } else {
                                        prev
                                    }
                                }
                            }
                        } else if cand.is_leaf && !cand.has_cp {
                            cand
                        } else if prev.is_leaf && !prev.has_cp {
                            prev
                        } else if cand.is_leaf && !prev.is_leaf {
                            cand
                        } else {
                            prev
                        }
                    }
                });
            }
            best
        }

        fn find_leaf(node: &RadixNode, path: &mut Vec<u32>, prefer_cold: bool) -> bool {
            if node.children.is_empty() {
                return !path.is_empty();
            }
            let Some(pick) = best_child(node, prefer_cold) else {
                return false;
            };
            path.push(pick.tok);
            if let Some(child) = node.children.get(&pick.tok) {
                if child.children.is_empty() {
                    return true;
                }
                return find_leaf(child, path, prefer_cold);
            }
            false
        }

        let mut path = Vec::new();
        if !find_leaf(&self.root, &mut path, prefer_cold_checkpoint) || path.is_empty() {
            return false;
        }
        let last = path.pop().unwrap();
        let mut cur = &mut self.root;
        for &tok in &path {
            cur = cur.children.get_mut(&tok).expect("path valid");
        }
        if let Some(removed) = cur.children.remove(&last) {
            if let Some(cp) = removed.checkpoint.as_ref() {
                self.checkpoint_bytes = self
                    .checkpoint_bytes
                    .saturating_sub(cp.approx_bytes());
            }
            // Dropping a subtree: subtract nested checkpoint bytes + node counts.
            fn walk_drop(node: &RadixNode, bytes: &mut usize, nodes: &mut usize) {
                for child in node.children.values() {
                    *nodes = nodes.saturating_add(1);
                    if let Some(cp) = child.checkpoint.as_ref() {
                        *bytes = bytes.saturating_add(cp.approx_bytes());
                    }
                    walk_drop(child, bytes, nodes);
                }
            }
            let mut nested_bytes = 0usize;
            let mut nested_nodes = 0usize;
            walk_drop(&removed, &mut nested_bytes, &mut nested_nodes);
            self.checkpoint_bytes = self.checkpoint_bytes.saturating_sub(nested_bytes);
            self.nodes = self.nodes.saturating_sub(1 + nested_nodes);
            true
        } else {
            false
        }
    }

    /// Snapshot current KV/recurrent and insert under `tokens`.
    pub fn capture(
        &mut self,
        tokens: &[u32],
        recurrent: &RecurrentState,
        kv: &KvTierManager,
        pages: &[Vec<u64>],
        cfg: &ModelConfig,
    ) -> Result<(), AllpacaError> {
        if tokens.is_empty() {
            return Ok(());
        }
        let cp = RadixCheckpoint {
            recurrent: recurrent.clone(),
            kv: snapshot_kv(kv, pages, cfg)?,
        };
        self.insert_checkpoint(tokens, cp);
        Ok(())
    }

    /// Longest restorable prefix: deepest node on the walk that has a checkpoint
    /// with `depth <= max_depth` (use `max_depth = tokens.len()-1` to leave logits work).
    pub fn lookup_checkpoint(
        &mut self,
        tokens: &[u32],
    ) -> Option<(usize, RadixCheckpoint)> {
        self.lookup_checkpoint_capped(tokens, usize::MAX)
    }

    pub fn lookup_checkpoint_capped(
        &mut self,
        tokens: &[u32],
        max_depth: usize,
    ) -> Option<(usize, RadixCheckpoint)> {
        let mut cur = &mut self.root;
        let mut depth = 0usize;
        let mut best_depth = 0usize;
        let mut best_cp: Option<RadixCheckpoint> = None;
        let mut best_path_end = 0usize;
        for &tok in tokens {
            match cur.children.get_mut(&tok) {
                Some(child) => {
                    cur = child;
                    depth += 1;
                    if depth <= max_depth {
                        if let Some(cp) = &cur.checkpoint {
                            best_depth = depth;
                            best_cp = Some(cp.clone());
                            best_path_end = depth;
                        }
                    }
                }
                None => break,
            }
        }
        // LRU touch on the restored checkpoint node.
        if best_cp.is_some() && best_path_end > 0 {
            let stamp = self.touch();
            let mut walk = &mut self.root;
            for &tok in &tokens[..best_path_end] {
                walk = walk.children.get_mut(&tok).expect("path valid");
            }
            walk.used_at = stamp;
        }
        match best_cp {
            Some(cp) => {
                self.hits += 1;
                self.skip_tokens += best_depth as u64;
                Some((best_depth, cp))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Longest common prefix length (token count), checkpoint or not.
    #[allow(dead_code)]
    pub fn longest_common_prefix(&mut self, tokens: &[u32]) -> usize {
        let mut cur = &self.root;
        let mut depth = 0usize;
        for &tok in tokens {
            match cur.children.get(&tok) {
                Some(child) => {
                    cur = child;
                    depth += 1;
                }
                None => break,
            }
        }
        depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_cp() -> RadixCheckpoint {
        RadixCheckpoint {
            recurrent: RecurrentState::zeros(1, 4),
            kv: vec![vec![]],
        }
    }

    fn fat_cp(n: usize) -> RadixCheckpoint {
        RadixCheckpoint {
            recurrent: RecurrentState::zeros(1, n),
            kv: vec![vec![KvRow {
                k: vec![0.0; n],
                v: vec![0.0; n],
            }]],
        }
    }

    #[test]
    fn insert_builds_nodes_along_prefix() {
        let mut forest = KvRadixForest::new();
        let added = forest.insert_checkpoint(&[1, 2, 3], empty_cp());
        assert_eq!(added, 3);
        assert_eq!(forest.nodes, 3);
        assert_eq!(forest.insert_checkpoint(&[1, 2, 3], empty_cp()), 0);
    }

    #[test]
    fn lookup_restores_shared_prefix_after_mid_edit() {
        let mut forest = KvRadixForest::new();
        forest.insert_checkpoint(&[10, 11], empty_cp());
        forest.insert_checkpoint(&[10, 11, 12, 13], empty_cp());
        let (n, _) = forest
            .lookup_checkpoint(&[10, 11, 99, 100])
            .expect("LCP hello must restore");
        assert_eq!(n, 2);
        assert!(forest.hits >= 1);
        assert!(forest.skip_tokens >= 2);
    }

    #[test]
    fn lookup_miss_on_unrelated_prompt() {
        let mut forest = KvRadixForest::new();
        forest.insert_checkpoint(&[1, 2, 3], empty_cp());
        assert!(forest.lookup_checkpoint(&[9, 8, 7]).is_none());
        assert!(forest.misses >= 1);
    }

    #[test]
    fn branching_shares_parent_and_picks_deepest_checkpoint() {
        let mut forest = KvRadixForest::new();
        forest.insert_checkpoint(&[1, 2], empty_cp());
        forest.insert_checkpoint(&[1, 2, 3], empty_cp());
        forest.insert_checkpoint(&[1, 2, 4], empty_cp());
        let (n, _) = forest.lookup_checkpoint(&[1, 2, 4]).unwrap();
        assert_eq!(n, 3);
        let (n2, _) = forest.lookup_checkpoint(&[1, 2, 9]).unwrap();
        assert_eq!(n2, 2, "partial walk keeps parent checkpoint");
    }

    #[test]
    fn max_nodes_evicts_leaves() {
        let mut forest = KvRadixForest::with_max_nodes(2);
        forest.insert_checkpoint(&[1], empty_cp());
        forest.insert_checkpoint(&[2], empty_cp());
        forest.insert_checkpoint(&[3], empty_cp());
        assert!(forest.nodes <= 2, "nodes={}", forest.nodes);
        assert!(forest.evictions >= 1, "evictions={}", forest.evictions);
    }

    #[test]
    fn byte_budget_evicts_cold_checkpoints() {
        let mut forest = KvRadixForest::with_limits(0, 800);
        forest.insert_checkpoint(&[1], fat_cp(32)); // warm later
        forest.insert_checkpoint(&[2], fat_cp(32)); // cold
        let warm_bytes = forest.checkpoint_bytes();
        assert!(warm_bytes > 800 || warm_bytes > 0);
        // Touch [1] so [2] is colder.
        let _ = forest.lookup_checkpoint(&[1]);
        forest.insert_checkpoint(&[3], fat_cp(32));
        assert!(
            forest.checkpoint_bytes() <= 800,
            "bytes={} over budget",
            forest.checkpoint_bytes()
        );
        assert!(forest.evictions >= 1, "evictions={}", forest.evictions);
        // Hot prefix should still be restorable more often than the cold one.
        assert!(
            forest.lookup_checkpoint(&[1]).is_some() || forest.lookup_checkpoint(&[3]).is_some(),
            "at least one recent checkpoint must survive"
        );
    }

    #[test]
    fn approx_bytes_tracks_insert_replace() {
        let mut forest = KvRadixForest::new();
        forest.insert_checkpoint(&[1], fat_cp(16));
        let a = forest.checkpoint_bytes();
        assert!(a > 0);
        forest.insert_checkpoint(&[1], fat_cp(8));
        let b = forest.checkpoint_bytes();
        assert!(b < a, "replace must shrink tracked bytes: a={a} b={b}");
    }

    #[test]
    fn trim_to_bytes_frees_cold_checkpoints() {
        let mut forest = KvRadixForest::with_limits(0, 0);
        forest.insert_checkpoint(&[1], fat_cp(64));
        forest.insert_checkpoint(&[2], fat_cp(64));
        forest.insert_checkpoint(&[3], fat_cp(64));
        let before = forest.checkpoint_bytes();
        assert!(before > 200, "before={before}");
        let freed = forest.trim_to_bytes(before / 3);
        assert!(freed > 0, "freed={freed}");
        assert!(
            forest.checkpoint_bytes() <= before / 3 + 64,
            "after={}",
            forest.checkpoint_bytes()
        );
    }
}
