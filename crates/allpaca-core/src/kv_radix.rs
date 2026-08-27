//! Radix / prefix-tree KV reuse spike (GEM).
//!
//! Semantic anchors only hit exact special-token boundaries. This forest stores
//! arbitrary token prefixes so an edit mid-prompt can still recover the longest
//! common prefix (LCP) without a full re-prefill.
//!
//! Spike scope: token-radix LCP lookup + insert. Next vertical wires checkpoints
//! (page ids / DenseKv snapshots) at selected depths into [`crate::kv_tier`].

use std::collections::HashMap;

/// One edge in the token radix: child keyed by next token id.
#[derive(Debug, Default)]
struct RadixNode {
    children: HashMap<u32, Box<RadixNode>>,
    /// True when some stored prefix ends at this node (checkpoint candidate).
    terminal: bool,
}

/// Forest of prompt prefixes for longest-common-prefix KV reuse.
#[derive(Debug, Default)]
pub struct KvRadixForest {
    root: RadixNode,
    /// Total nodes allocated (excluding root).
    pub nodes: usize,
    /// Successful LCP lookups that returned depth > 0.
    pub hits: u64,
    /// Lookups with no shared prefix.
    pub misses: u64,
}

impl KvRadixForest {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `tokens` as a reusable prefix. Returns nodes newly created.
    pub fn insert(&mut self, tokens: &[u32]) -> usize {
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
        cur.terminal = true;
        added
    }

    /// Longest common prefix length (token count) shared with any inserted path.
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
        if depth > 0 {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
        depth
    }

    /// Tokens skipped vs full re-prefill when LCP is `lcp` of a prompt of `prompt_len`.
    pub fn skip_tokens(lcp: usize, prompt_len: usize) -> usize {
        lcp.min(prompt_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_builds_nodes_along_prefix() {
        let mut forest = KvRadixForest::new();
        let added = forest.insert(&[1, 2, 3]);
        assert_eq!(added, 3);
        assert_eq!(forest.nodes, 3);
        assert_eq!(forest.insert(&[1, 2, 3]), 0, "re-insert is free");
        assert_eq!(forest.nodes, 3);
    }

    #[test]
    fn lcp_recovers_shared_prefix_after_mid_edit() {
        let mut forest = KvRadixForest::new();
        // "hello world" as token ids
        forest.insert(&[10, 11, 12, 13]);
        // edit tail: "hello there"
        let lcp = forest.longest_common_prefix(&[10, 11, 99, 100]);
        assert_eq!(lcp, 2, "must keep 'hello' prefix across edit");
        assert_eq!(KvRadixForest::skip_tokens(lcp, 4), 2);
        assert!(forest.hits >= 1);
    }

    #[test]
    fn lcp_miss_on_unrelated_prompt() {
        let mut forest = KvRadixForest::new();
        forest.insert(&[1, 2, 3]);
        assert_eq!(forest.longest_common_prefix(&[9, 8, 7]), 0);
        assert!(forest.misses >= 1);
    }

    #[test]
    fn branching_shares_parent_nodes() {
        let mut forest = KvRadixForest::new();
        forest.insert(&[1, 2, 3]);
        let before = forest.nodes;
        forest.insert(&[1, 2, 4]);
        assert_eq!(forest.nodes, before + 1, "only the divergent token adds a node");
        assert_eq!(forest.longest_common_prefix(&[1, 2, 4]), 3);
        assert_eq!(forest.longest_common_prefix(&[1, 2, 3]), 3);
    }
}
