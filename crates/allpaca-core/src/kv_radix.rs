//! Radix / prefix-tree KV reuse (GEM vertical).
//!
//! Stores arbitrary prompt prefixes with KV+recurrent checkpoints so a mid-prompt
//! edit can restore the longest common prefix without a full re-prefill.

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

/// One edge in the token radix: child keyed by next token id.
#[derive(Debug, Default)]
struct RadixNode {
    children: HashMap<u32, Box<RadixNode>>,
    checkpoint: Option<RadixCheckpoint>,
}

/// Forest of prompt prefixes for longest-common-prefix KV reuse.
#[derive(Debug, Default)]
pub struct KvRadixForest {
    root: RadixNode,
    /// Total nodes allocated (excluding root).
    pub nodes: usize,
    /// Lookups that restored a checkpoint (depth > 0).
    pub hits: u64,
    /// Lookups with no restorable prefix.
    pub misses: u64,
    /// Cumulative tokens skipped via LCP restore.
    pub skip_tokens: u64,
}

impl KvRadixForest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn skip_tokens(&self) -> u64 {
        self.skip_tokens
    }

    /// Insert `tokens` and attach `cp` at the terminal (and replace along the path
    /// when re-inserting the same prefix).
    pub fn insert_checkpoint(&mut self, tokens: &[u32], cp: RadixCheckpoint) -> usize {
        if tokens.is_empty() {
            return 0;
        }
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
        cur.checkpoint = Some(cp);
        added
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
        let mut cur = &self.root;
        let mut depth = 0usize;
        let mut best: Option<(usize, RadixCheckpoint)> = None;
        for &tok in tokens {
            match cur.children.get(&tok) {
                Some(child) => {
                    cur = child;
                    depth += 1;
                    if depth <= max_depth {
                        if let Some(cp) = &cur.checkpoint {
                            best = Some((depth, cp.clone()));
                        }
                    }
                }
                None => break,
            }
        }
        match best {
            Some((n, cp)) => {
                self.hits += 1;
                self.skip_tokens += n as u64;
                Some((n, cp))
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
}

