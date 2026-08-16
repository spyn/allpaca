use ahash::AHashMap;
use allpaca_gguf::GgufTokenizer;
use allpaca_types::{AllpacaError, FatalError};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::Tokenizer;

pub struct EngineTokenizer {
    inner: Tokenizer,
    pub tokens: Vec<String>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub add_bos: bool,
}

impl EngineTokenizer {
    pub fn from_gguf(tok: GgufTokenizer) -> Self {
        Self {
            inner: tok.inner,
            tokens: tok.tokens,
            bos_id: tok.bos_id,
            eos_id: tok.eos_id,
            add_bos: tok.add_bos,
        }
    }

    pub fn synthetic(vocab_size: usize) -> Result<Self, AllpacaError> {
        let tokens = synthetic_vocab(vocab_size);
        let mut vocab = AHashMap::new();
        for (i, tok) in tokens.iter().enumerate() {
            vocab.insert(tok.clone(), i as u32);
        }
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .map_err(|e| AllpacaError::Fatal(FatalError::Tokenizer(e.to_string())))?;
        let mut inner = Tokenizer::new(model);
        inner.with_pre_tokenizer(Some(Whitespace));
        Ok(Self {
            inner,
            tokens,
            bos_id: Some(1),
            eos_id: Some(2),
            add_bos: true,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, AllpacaError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| AllpacaError::Fatal(FatalError::Tokenizer(e.to_string())))?;
        let mut ids = encoding.get_ids().to_vec();
        if self.add_bos {
            if let Some(bos) = self.bos_id {
                if ids.first().copied() != Some(bos) {
                    ids.insert(0, bos);
                }
            }
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, AllpacaError> {
        self.inner
            .decode(ids, true)
            .map_err(|e| AllpacaError::Fatal(FatalError::Tokenizer(e.to_string())))
    }

    pub fn decode_one(&self, id: u32) -> String {
        self.tokens
            .get(id as usize)
            .cloned()
            .unwrap_or_else(|| format!("<{id}>"))
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

pub fn synthetic_vocab(vocab_size: usize) -> Vec<String> {
    let mut tokens = vec![
        "<unk>".into(),
        "<s>".into(),
        "</s>".into(),
        "hello".into(),
        "world".into(),
        "{".into(),
        "}".into(),
        "\"".into(),
        ":".into(),
        ",".into(),
        "[".into(),
        "]".into(),
        "true".into(),
        "false".into(),
        "null".into(),
        "a".into(),
        "b".into(),
        "c".into(),
        "ok".into(),
    ];
    for d in 0..=9 {
        tokens.push(d.to_string());
    }
    let mut i = 0;
    while tokens.len() < vocab_size {
        tokens.push(format!("tok{i}"));
        i += 1;
    }
    tokens.truncate(vocab_size);
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_encodes_hello() {
        let tok = EngineTokenizer::synthetic(64).unwrap();
        let ids = tok.encode("hello").unwrap();
        assert!(ids.contains(&3));
        assert_eq!(tok.decode_one(3), "hello");
    }
}
