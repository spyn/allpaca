use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub json_mode: bool,
    /// JSON Schema text compiled by llguidance. Takes precedence over `json_mode`.
    #[serde(default)]
    pub json_schema: Option<String>,
    /// Lark grammar compiled by llguidance.
    #[serde(default)]
    pub grammar: Option<String>,
    pub seed: Option<u64>,
    pub stop_token_ids: Vec<u32>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 32,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            json_mode: false,
            json_schema: None,
            grammar: None,
            seed: None,
            stop_token_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationOutput {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenEvent {
    pub token_id: u32,
    pub text: String,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StepResult {
    Token(TokenEvent),
    Finished(FinishReason),
    /// Prompt prefill advanced by `prefill.chunk_tokens`; decode has not started.
    Prefill {
        tokens_done: usize,
        tokens_total: usize,
    },
}

/// One continuous-batching tick: which Request advanced and what it produced.
#[derive(Debug, Clone, PartialEq)]
pub struct PollEvent {
    pub request_id: u64,
    pub result: StepResult,
}

/// AVX2-friendly allowed-token bitset (64-token words).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenMask {
    pub words: Vec<u64>,
}

impl TokenMask {
    pub fn allow_all(vocab: usize) -> Self {
        let n_words = vocab.div_ceil(64);
        let mut words = vec![u64::MAX; n_words];
        if vocab % 64 != 0 {
            let leftover = vocab % 64;
            words[n_words - 1] = (1u64 << leftover) - 1;
        }
        Self { words }
    }

    pub fn empty(vocab: usize) -> Self {
        Self {
            words: vec![0; vocab.div_ceil(64)],
        }
    }

    pub fn allowed(&self, token: usize) -> bool {
        let w = token >> 6;
        let b = token & 63;
        w < self.words.len() && ((self.words[w] >> b) & 1) == 1
    }

    pub fn set(&mut self, token: usize, on: bool) {
        let w = token >> 6;
        let b = token & 63;
        if w >= self.words.len() {
            return;
        }
        if on {
            self.words[w] |= 1u64 << b;
        } else {
            self.words[w] &= !(1u64 << b);
        }
    }

    pub fn any(&self) -> bool {
        self.words.iter().any(|w| *w != 0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintState {
    pub state_id: u64,
    pub mask: TokenMask,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_mask_roundtrip() {
        let mut mask = TokenMask::empty(70);
        assert!(!mask.allowed(3));
        mask.set(3, true);
        mask.set(69, true);
        assert!(mask.allowed(3));
        assert!(mask.allowed(69));
        assert!(!mask.allowed(4));
        assert!(mask.any());
    }
}
