use allpaca_types::AllpacaError;

use crate::draft::NGramProvider;
use crate::model::{DenseKv, TinyModel};

const NONE: u32 = u32::MAX;

/// Candidate node in a Token Forest (SPECS / OPPORTUNITY_DESIGN).
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub token_id: i32,
    pub parent: i32,
    pub depth: u16,
    pub logprob: f32,
    pub flags: u32,
}

/// Breadth- and depth-limited draft tree built on the CPU control plane.
#[derive(Debug, Clone, Default)]
pub struct TokenForest {
    pub nodes: Vec<TreeNode>,
    pub first_child: Vec<u32>,
    pub next_sibling: Vec<u32>,
}

impl TokenForest {
    fn with_capacity(n: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(n),
            first_child: Vec::with_capacity(n),
            next_sibling: Vec::with_capacity(n),
        }
    }

    fn add_node(&mut self, token_id: u32, parent: i32, depth: u16, logprob: f32) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(TreeNode {
            token_id: token_id as i32,
            parent,
            depth,
            logprob,
            flags: 0,
        });
        self.first_child.push(NONE);
        self.next_sibling.push(NONE);
        if parent >= 0 {
            let p = parent as usize;
            if self.first_child[p] == NONE {
                self.first_child[p] = idx as u32;
            } else {
                let mut sib = self.first_child[p];
                while self.next_sibling[sib as usize] != NONE {
                    sib = self.next_sibling[sib as usize];
                }
                self.next_sibling[sib as usize] = idx as u32;
            }
        }
        idx
    }

    pub fn children(&self, parent: i32) -> impl Iterator<Item = usize> + '_ {
        let start = if parent < 0 {
            let mut i = 0usize;
            while i < self.nodes.len() && self.nodes[i].parent >= 0 {
                i += 1;
            }
            if i < self.nodes.len() {
                i as u32
            } else {
                NONE
            }
        } else if (parent as usize) < self.first_child.len() {
            self.first_child[parent as usize]
        } else {
            NONE
        };
        ChildIter {
            forest: self,
            next: start,
            roots_only: parent < 0,
        }
    }

    pub fn path_tokens(&self, mut node: usize) -> Vec<u32> {
        let mut rev = Vec::new();
        loop {
            rev.push(self.nodes[node].token_id as u32);
            let parent = self.nodes[node].parent;
            if parent < 0 {
                break;
            }
            node = parent as usize;
        }
        rev.reverse();
        rev
    }

    pub fn is_valid_path(&self, tokens: &[u32]) -> bool {
        if tokens.is_empty() {
            return true;
        }
        let mut parent = -1i32;
        for tok in tokens {
            let mut found = None;
            for child in self.children(parent) {
                if self.nodes[child].token_id as u32 == *tok {
                    found = Some(child as i32);
                    break;
                }
            }
            match found {
                Some(idx) => parent = idx,
                None => return false,
            }
        }
        true
    }

    /// Ancestry attention mask over forest nodes: mask[i*n + j] if j is ancestor of i (or i).
    pub fn ancestry_mask(&self) -> Vec<bool> {
        let n = self.nodes.len();
        let mut mask = vec![false; n * n];
        for i in 0..n {
            let mut j = i as i32;
            while j >= 0 {
                mask[i * n + j as usize] = true;
                j = self.nodes[j as usize].parent;
            }
        }
        mask
    }
}

struct ChildIter<'a> {
    forest: &'a TokenForest,
    next: u32,
    roots_only: bool,
}

impl Iterator for ChildIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == NONE || (self.next as usize) >= self.forest.nodes.len() {
            return None;
        }
        let cur = self.next as usize;
        if self.roots_only {
            self.next = (cur as u32).saturating_add(1);
            while (self.next as usize) < self.forest.nodes.len()
                && self.forest.nodes[self.next as usize].parent >= 0
            {
                self.next += 1;
            }
            if (self.next as usize) >= self.forest.nodes.len() {
                self.next = NONE;
            }
        } else {
            self.next = self.forest.next_sibling[cur];
        }
        Some(cur)
    }
}

/// CPU Token Forest construction from NGram and/or Draft Model providers.
pub fn build_token_forest(
    ngram: &NGramProvider,
    draft: Option<&TinyModel>,
    ctx: &[u32],
    max_nodes: usize,
    max_depth: usize,
) -> Result<TokenForest, AllpacaError> {
    let max_nodes = max_nodes.max(1);
    let max_depth = max_depth.max(1);
    let branch = adaptive_branch_budget(max_nodes, max_depth);
    let mut forest = TokenForest::with_capacity(max_nodes);
    let mut frontier: Vec<(i32, Vec<u32>)> = vec![(-1, ctx.to_vec())];

    while forest.nodes.len() < max_nodes && !frontier.is_empty() {
        let (parent, path) = frontier.remove(0);
        let depth = if parent < 0 {
            1u16
        } else {
            forest.nodes[parent as usize].depth + 1
        };
        if depth as usize > max_depth {
            continue;
        }
        let cands = propose_branches(ngram, draft, &path, branch)?;
        for (tok, logprob) in cands {
            if forest.nodes.len() >= max_nodes {
                break;
            }
            if parent >= 0 {
                let mut sib = forest.first_child[parent as usize];
                let mut dup = false;
                while sib != NONE {
                    if forest.nodes[sib as usize].token_id as u32 == tok {
                        dup = true;
                        break;
                    }
                    sib = forest.next_sibling[sib as usize];
                }
                if dup {
                    continue;
                }
            } else if forest
                .nodes
                .iter()
                .any(|n| n.parent < 0 && n.token_id as u32 == tok)
            {
                continue;
            }
            let idx = forest.add_node(tok, parent, depth, logprob);
            if (depth as usize) < max_depth {
                let mut child_path = path.clone();
                child_path.push(tok);
                frontier.push((idx as i32, child_path));
            }
        }
    }
    Ok(forest)
}

fn adaptive_branch_budget(max_nodes: usize, max_depth: usize) -> usize {
    if max_depth == 0 {
        return 1;
    }
    let per_level = (max_nodes as f64).powf(1.0 / max_depth as f64).ceil() as usize;
    per_level.clamp(2, 4)
}

fn propose_branches(
    ngram: &NGramProvider,
    draft: Option<&TinyModel>,
    ctx: &[u32],
    k: usize,
) -> Result<Vec<(u32, f32)>, AllpacaError> {
    let mut out = Vec::new();
    if let Some(model) = draft {
        out.extend(draft_top_k(model, ctx, k)?);
    }
    for (tok, lp) in ngram.top_next(ctx, k) {
        if !out.iter().any(|(t, _)| *t == tok) {
            out.push((tok, lp));
        }
    }
    out.truncate(k);
    Ok(out)
}

fn draft_top_k(
    model: &TinyModel,
    ctx: &[u32],
    k: usize,
) -> Result<Vec<(u32, f32)>, AllpacaError> {
    if ctx.is_empty() || k == 0 {
        return Ok(Vec::new());
    }
    let mut cache = DenseKv::default();
    let vocab = model.config.vocab_size;
    let mut logits = vec![0.0f32; vocab];
    for (pos, tok) in ctx.iter().enumerate() {
        model.forward_dense(*tok, pos, &mut cache, &mut logits)?;
    }
    Ok(top_k_logprobs(&logits, k))
}

fn top_k_logprobs(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, v)| (i as u32, *v))
        .collect();
    idx.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    idx.truncate(k.max(1));
    if idx.is_empty() {
        return idx;
    }
    let max = idx[0].1;
    let sum: f32 = idx.iter().map(|(_, v)| (v - max).exp()).sum();
    let log_z = sum.ln();
    idx.into_iter()
        .map(|(t, v)| (t, v - max - log_z))
        .collect()
}
