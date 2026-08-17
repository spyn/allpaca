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

    pub fn from_vocab(
        tokens: Vec<String>,
        bos_id: Option<u32>,
        eos_id: Option<u32>,
    ) -> Result<Self, AllpacaError> {
        if tokens.is_empty() {
            return Err(AllpacaError::Fatal(FatalError::Tokenizer(
                "tokenizer vocab is empty".into(),
            )));
        }
        let mut vocab = AHashMap::new();
        for (i, tok) in tokens.iter().enumerate() {
            vocab.insert(tok.clone(), i as u32);
        }
        let unk = tokens
            .iter()
            .find(|t| *t == "<unk>" || *t == "[UNK]")
            .cloned()
            .unwrap_or_else(|| tokens[0].clone());
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token(unk)
            .build()
            .map_err(|e| AllpacaError::Fatal(FatalError::Tokenizer(e.to_string())))?;
        let mut inner = Tokenizer::new(model);
        inner.with_pre_tokenizer(Some(Whitespace));
        Ok(Self {
            inner,
            tokens,
            bos_id,
            eos_id,
            add_bos: bos_id.is_some(),
        })
    }

    pub fn synthetic(vocab_size: usize) -> Result<Self, AllpacaError> {
        Self::from_vocab(synthetic_vocab(vocab_size), Some(1), Some(2))
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
        let mut out = String::new();
        for &id in ids {
            let piece = self.decode_one(id);
            if is_special_piece(&piece) {
                continue;
            }
            out.push_str(&piece);
        }
        Ok(out)
    }

    pub fn decode_one(&self, id: u32) -> String {
        self.tokens
            .get(id as usize)
            .map(|piece| normalize_piece(piece))
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

/// GPT-2 / Llama-3 BPE space marker (U+0120 LATIN CAPITAL LETTER G WITH DOT ABOVE).
const GPT2_SPACE: char = '\u{0120}';
/// SentencePiece / Llama-2 underline space marker (U+2581).
const SPM_SPACE: char = '\u{2581}';

fn normalize_piece(piece: &str) -> String {
    if let Some(byte) = parse_hex_byte_token(piece) {
        return byte_to_string(byte);
    }
    piece.replace(GPT2_SPACE, " ").replace(SPM_SPACE, " ")
}

fn parse_hex_byte_token(piece: &str) -> Option<u8> {
    let rest = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if rest.len() != 2 {
        return None;
    }
    u8::from_str_radix(rest, 16).ok()
}

fn byte_to_string(byte: u8) -> String {
    if byte.is_ascii() {
        (byte as char).to_string()
    } else {
        String::from_utf8_lossy(&[byte]).into_owned()
    }
}

fn is_special_piece(piece: &str) -> bool {
    piece.starts_with('<') && piece.ends_with('>')
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

    #[test]
    fn decode_one_maps_gpt2_and_sentencepiece_space_markers() {
        let tok = EngineTokenizer::from_vocab(
            vec![
                "<unk>".into(),
                "hello".into(),
                "Ġworld".into(),
                "▁foo".into(),
                "<0x0A>".into(),
            ],
            None,
            None,
        )
        .unwrap();
        assert_eq!(tok.decode_one(1), "hello");
        assert_eq!(tok.decode_one(2), " world");
        assert_eq!(tok.decode_one(3), " foo");
        assert_eq!(tok.decode_one(4), "\n");
        assert_eq!(tok.decode(&[0, 1, 2, 3]).unwrap(), "hello world foo");
    }
}
