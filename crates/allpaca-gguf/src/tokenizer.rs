use ahash::AHashMap;
use allpaca_types::{AllpacaError, FatalError};
use tokenizers::models::bpe::BPE;
use tokenizers::models::unigram::Unigram;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::Tokenizer;

use crate::parser::{GgufFile, GgufValue};

pub struct GgufTokenizer {
    pub inner: Tokenizer,
    pub tokens: Vec<String>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub unk_id: Option<u32>,
    pub add_bos: bool,
    pub model_kind: String,
}

/// Build a HuggingFace `tokenizers` instance from GGUF vocab metadata.
pub fn extract_tokenizer(file: &GgufFile) -> Result<GgufTokenizer, AllpacaError> {
    let tokens = file
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufValue::as_string_array)
        .ok_or_else(|| {
            AllpacaError::Fatal(FatalError::Tokenizer(
                "GGUF missing tokenizer.ggml.tokens".into(),
            ))
        })?;
    if tokens.is_empty() {
        return Err(AllpacaError::Fatal(FatalError::Tokenizer(
            "GGUF tokenizer vocab is empty".into(),
        )));
    }

    let kind = file
        .meta_str("tokenizer.ggml.model")
        .unwrap_or("wordlevel")
        .to_string();
    let scores = file
        .metadata
        .get("tokenizer.ggml.scores")
        .and_then(GgufValue::as_f32_array);
    let merges = file
        .metadata
        .get("tokenizer.ggml.merges")
        .and_then(GgufValue::as_string_array);
    let unk_id = file.meta_u32("tokenizer.ggml.unknown_token_id");
    let bos_id = file.meta_u32("tokenizer.ggml.bos_token_id");
    let eos_id = file.meta_u32("tokenizer.ggml.eos_token_id");
    let add_bos = file
        .metadata
        .get("tokenizer.ggml.add_bos_token")
        .and_then(|v| match v {
            GgufValue::Bool(b) => Some(*b),
            _ => None,
        })
        .unwrap_or(true);

    let inner = match kind.as_str() {
        "gpt2" => build_bpe(&tokens, merges.as_deref(), unk_id)?,
        "llama" => build_unigram_or_wordlevel(&tokens, scores.as_deref(), unk_id)?,
        _ => build_wordlevel(&tokens, unk_id)?,
    };

    Ok(GgufTokenizer {
        inner,
        tokens,
        bos_id,
        eos_id,
        unk_id,
        add_bos,
        model_kind: kind,
    })
}

fn unk_token(tokens: &[String], unk_id: Option<u32>) -> String {
    unk_id
        .and_then(|id| tokens.get(id as usize).cloned())
        .unwrap_or_else(|| tokens[0].clone())
}

fn build_wordlevel(tokens: &[String], unk_id: Option<u32>) -> Result<Tokenizer, AllpacaError> {
    let mut vocab = AHashMap::new();
    for (i, tok) in tokens.iter().enumerate() {
        vocab.insert(tok.clone(), i as u32);
    }
    let unk = unk_token(tokens, unk_id);
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token(unk)
        .build()
        .map_err(|e| AllpacaError::Fatal(FatalError::Tokenizer(e.to_string())))?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    Ok(tokenizer)
}

fn build_unigram_or_wordlevel(
    tokens: &[String],
    scores: Option<&[f32]>,
    unk_id: Option<u32>,
) -> Result<Tokenizer, AllpacaError> {
    if let Some(scores) = scores {
        if scores.len() == tokens.len() {
            let vocab: Vec<(String, f64)> = tokens
                .iter()
                .zip(scores.iter())
                .map(|(t, s)| (t.clone(), f64::from(*s)))
                .collect();
            let unk = unk_id.map(|id| id as usize);
            match Unigram::from(vocab, unk, false) {
                Ok(model) => return Ok(Tokenizer::new(model)),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Unigram tokenizer build failed; falling back to WordLevel"
                    );
                }
            }
        }
    }
    build_wordlevel(tokens, unk_id)
}

fn build_bpe(
    tokens: &[String],
    merges: Option<&[String]>,
    unk_id: Option<u32>,
) -> Result<Tokenizer, AllpacaError> {
    let Some(merges) = merges.filter(|m| !m.is_empty()) else {
        return build_wordlevel(tokens, unk_id);
    };
    let vocab: AHashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    let merge_pairs: Vec<(String, String)> = merges
        .iter()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            Some((parts.next()?.to_string(), parts.next()?.to_string()))
        })
        .collect();
    let unk = unk_token(tokens, unk_id);
    match BPE::builder()
        .vocab_and_merges(vocab, merge_pairs)
        .unk_token(unk)
        .build()
    {
        Ok(model) => Ok(Tokenizer::new(model)),
        Err(e) => {
            tracing::warn!(error = %e, "BPE tokenizer build failed; falling back to WordLevel");
            build_wordlevel(tokens, unk_id)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{GgufFile, GgufValue};
    use crate::writer::write_gguf;
    use std::collections::HashMap;

    #[test]
    fn extracts_wordlevel_vocab() {
        let path = std::env::temp_dir().join("allpaca_tok.gguf");
        let mut meta = HashMap::new();
        meta.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("wordlevel".into()),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("<unk>".into()),
                GgufValue::String("hello".into()),
                GgufValue::String("world".into()),
            ]),
        );
        meta.insert(
            "tokenizer.ggml.unknown_token_id".into(),
            GgufValue::Uint32(0),
        );
        write_gguf(&path, &meta, &[]).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let tok = extract_tokenizer(&file).unwrap();
        let ids = tok.inner.encode("hello world", false).unwrap();
        assert_eq!(ids.get_ids(), &[1, 2]);
        let _ = std::fs::remove_file(path);
    }
}
