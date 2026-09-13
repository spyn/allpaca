use ahash::AHashMap;
use allpaca_types::{AllpacaError, FatalError};
use tokenizers::models::bpe::BPE;
use tokenizers::models::unigram::Unigram;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::{AddedToken, Tokenizer};

use crate::parser::{GgufFile, GgufValue};

/// ggml `tokenizer.ggml.token_type` values (llama.cpp / llama.h).
const TOKEN_TYPE_UNKNOWN: i32 = 2;
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

pub struct GgufTokenizer {
    pub inner: Tokenizer,
    pub tokens: Vec<String>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub eot_id: Option<u32>,
    pub eom_id: Option<u32>,
    pub unk_id: Option<u32>,
    pub sep_id: Option<u32>,
    pub pad_id: Option<u32>,
    pub mask_id: Option<u32>,
    pub fim_pre_id: Option<u32>,
    pub fim_suf_id: Option<u32>,
    pub fim_mid_id: Option<u32>,
    pub fim_pad_id: Option<u32>,
    pub fim_rep_id: Option<u32>,
    pub fim_sep_id: Option<u32>,
    pub add_bos: bool,
    pub model_kind: String,
}

impl GgufTokenizer {
    /// End-of-generation / stop ids (llama special_eog-style): eos, eot, eom.
    pub fn stop_token_ids(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for id in [self.eos_id, self.eot_id, self.eom_id] {
            if let Some(id) = id {
                if !out.contains(&id) {
                    out.push(id);
                }
            }
        }
        out
    }
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
    let token_types = file
        .metadata
        .get("tokenizer.ggml.token_type")
        .and_then(GgufValue::as_i32_array);
    let n = tokens.len() as u32;
    let unk_id = meta_token_id(file, "tokenizer.ggml.unknown_token_id", n);
    let bos_id = meta_token_id(file, "tokenizer.ggml.bos_token_id", n);
    let eos_id = meta_token_id(file, "tokenizer.ggml.eos_token_id", n);
    let eot_id = meta_token_id(file, "tokenizer.ggml.eot_token_id", n);
    let eom_id = meta_token_id(file, "tokenizer.ggml.eom_token_id", n);
    let sep_id = meta_token_id(file, "tokenizer.ggml.seperator_token_id", n)
        .or_else(|| meta_token_id(file, "tokenizer.ggml.separator_token_id", n));
    let pad_id = meta_token_id(file, "tokenizer.ggml.padding_token_id", n);
    let mask_id = meta_token_id(file, "tokenizer.ggml.mask_token_id", n);
    let fim_pre_id = meta_token_id(file, "tokenizer.ggml.fim_pre_token_id", n)
        .or_else(|| meta_token_id(file, "tokenizer.ggml.prefix_token_id", n));
    let fim_suf_id = meta_token_id(file, "tokenizer.ggml.fim_suf_token_id", n)
        .or_else(|| meta_token_id(file, "tokenizer.ggml.suffix_token_id", n));
    let fim_mid_id = meta_token_id(file, "tokenizer.ggml.fim_mid_token_id", n)
        .or_else(|| meta_token_id(file, "tokenizer.ggml.middle_token_id", n));
    let fim_pad_id = meta_token_id(file, "tokenizer.ggml.fim_pad_token_id", n);
    let fim_rep_id = meta_token_id(file, "tokenizer.ggml.fim_rep_token_id", n)
        .or_else(|| meta_token_id(file, "tokenizer.ggml.fim_repo_token_id", n));
    let fim_sep_id = meta_token_id(file, "tokenizer.ggml.fim_sep_token_id", n);
    // llama.cpp: BPE/gpt2 add_bos defaults false; SPM `llama` defaults true. GGUF key overrides.
    let add_bos = file
        .metadata
        .get("tokenizer.ggml.add_bos_token")
        .and_then(|v| match v {
            GgufValue::Bool(b) => Some(*b),
            _ => None,
        })
        .unwrap_or(kind.as_str() == "llama");

    let mut inner = match kind.as_str() {
        "gpt2" => build_bpe(&tokens, merges.as_deref(), unk_id)?,
        "llama" => build_unigram_or_wordlevel(&tokens, scores.as_deref(), unk_id)?,
        _ => build_wordlevel(&tokens, unk_id)?,
    };
    // llama.cpp partitions CONTROL / USER_DEFINED / UNKNOWN before BPE.
    register_special_tokens(&mut inner, &tokens, token_types.as_deref());

    Ok(GgufTokenizer {
        inner,
        tokens,
        bos_id,
        eos_id,
        eot_id,
        eom_id,
        unk_id,
        sep_id,
        pad_id,
        mask_id,
        fim_pre_id,
        fim_suf_id,
        fim_mid_id,
        fim_pad_id,
        fim_rep_id,
        fim_sep_id,
        add_bos,
        model_kind: kind,
    })
}

fn meta_token_id(file: &GgufFile, key: &str, vocab: u32) -> Option<u32> {
    let id = file.meta_u32(key)?;
    if id < vocab {
        Some(id)
    } else {
        None
    }
}

fn is_special_token_type(ty: i32) -> bool {
    matches!(
        ty,
        TOKEN_TYPE_UNKNOWN | TOKEN_TYPE_CONTROL | TOKEN_TYPE_USER_DEFINED
    )
}

/// Heuristic when `tokenizer.ggml.token_type` is absent (incomplete GGUF).
fn is_controlish_piece_heuristic(piece: &str) -> bool {
    if piece.len() < 3 {
        return false;
    }
    (piece.starts_with("<|") && piece.ends_with("|>"))
        || piece == "<think>"
        || piece == "</think>"
        || piece == "<s>"
        || piece == "</s>"
}

fn register_special_tokens(tokenizer: &mut Tokenizer, tokens: &[String], types: Option<&[i32]>) {
    let use_types = types.filter(|t| t.len() == tokens.len());
    let mut idxs: Vec<usize> = (0..tokens.len())
        .filter(|&i| {
            if let Some(tys) = use_types {
                is_special_token_type(tys[i])
            } else {
                is_controlish_piece_heuristic(&tokens[i])
            }
        })
        .collect();
    // Longest first — matches llama.cpp special-token partition order.
    idxs.sort_by(|&a, &b| tokens[b].len().cmp(&tokens[a].len()).then(a.cmp(&b)));
    let added: Vec<AddedToken> = idxs
        .into_iter()
        .map(|i| {
            AddedToken::from(tokens[i].as_str(), true)
                .normalized(false)
                .special(true)
        })
        .collect();
    if !added.is_empty() {
        let _ = tokenizer.add_special_tokens(&added);
    }
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
    let vocab: AHashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    let merge_pairs: Vec<(String, String)> = merges
        .unwrap_or(&[])
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
        Ok(model) => Ok(byte_level_bpe(model)),
        Err(e) => {
            tracing::warn!(error = %e, "BPE tokenizer build failed; falling back to WordLevel");
            build_wordlevel(tokens, unk_id)
        }
    }
}

fn byte_level_bpe(model: BPE) -> Tokenizer {
    let mut tokenizer = Tokenizer::new(model);
    // Llama-3 / GPT-2: ByteLevel maps space to Ġ then BPE. add_prefix_space
    // is false so "Hello world" is Hello + Ġworld, not ĠHello.
    let byte_level = ByteLevel::new(false, true, true);
    tokenizer.with_pre_tokenizer(Some(byte_level.clone()));
    tokenizer.with_decoder(Some(byte_level));
    tokenizer
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

    fn gpt2_hello_world_meta() -> HashMap<String, GgufValue> {
        let g = "\u{0120}";
        let mut meta = HashMap::new();
        meta.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("gpt2".into()),
        );
        let tokens: Vec<String> = vec![
            "<unk>".into(),
            "<s>".into(),
            "</s>".into(),
            "H".into(),
            "e".into(),
            "l".into(),
            "o".into(),
            "w".into(),
            "r".into(),
            "d".into(),
            g.into(),
            "He".into(),
            "Hel".into(),
            "Hell".into(),
            "Hello".into(),
            format!("{g}w"),
            format!("{g}wo"),
            format!("{g}wor"),
            format!("{g}worl"),
            format!("{g}world"),
        ];
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(tokens.into_iter().map(GgufValue::String).collect()),
        );
        let merges: Vec<String> = vec![
            "H e".into(),
            "He l".into(),
            "Hel l".into(),
            "Hell o".into(),
            format!("{g} w"),
            format!("{g}w o"),
            format!("{g}wo r"),
            format!("{g}wor l"),
            format!("{g}worl d"),
        ];
        meta.insert(
            "tokenizer.ggml.merges".into(),
            GgufValue::Array(merges.into_iter().map(GgufValue::String).collect()),
        );
        meta.insert(
            "tokenizer.ggml.unknown_token_id".into(),
            GgufValue::Uint32(0),
        );
        meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
        meta.insert(
            "tokenizer.ggml.add_bos_token".into(),
            GgufValue::Bool(false),
        );
        meta
    }

    #[test]
    fn gpt2_bpe_encodes_hello_world_as_bytelevel_pieces_not_unk() {
        let path = std::env::temp_dir().join("allpaca_gpt2_tok.gguf");
        write_gguf(&path, &gpt2_hello_world_meta(), &[]).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let tok = extract_tokenizer(&file).unwrap();
        assert_eq!(tok.model_kind, "gpt2");
        let encoded = tok.inner.encode("Hello world", false).unwrap();
        let ids = encoded.get_ids();
        assert_eq!(
            ids,
            &[14, 19],
            "ByteLevel BPE must emit Hello + Ġworld, not unk/whitespace; tokens={:?} ids={ids:?}",
            encoded.get_tokens()
        );
        assert_ne!(ids, &[14, 0], "must not be WordLevel split of raw 'world'");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn gpt2_bpe_keeps_chatml_specials_as_single_tokens() {
        let path = std::env::temp_dir().join("allpaca_gpt2_chatml.gguf");
        let mut meta = gpt2_hello_world_meta();
        let tokens = meta
            .get_mut("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                GgufValue::Array(a) => Some(a),
                _ => None,
            })
            .unwrap();
        tokens.push(GgufValue::String("<|im_start|>".into()));
        tokens.push(GgufValue::String("<|im_end|>".into()));
        tokens.push(GgufValue::String("<think>".into()));
        write_gguf(&path, &meta, &[]).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let tok = extract_tokenizer(&file).unwrap();
        let encoded = tok
            .inner
            .encode("<|im_start|>user\nHello<|im_end|>\n<think>\n", false)
            .unwrap();
        let pieces = encoded.get_tokens();
        assert!(
            pieces.iter().any(|p| p == "<|im_start|>"),
            "expected single <|im_start|>, got {pieces:?}"
        );
        assert!(
            pieces.iter().any(|p| p == "<|im_end|>"),
            "expected single <|im_end|>, got {pieces:?}"
        );
        assert!(
            pieces.iter().any(|p| p == "<think>"),
            "expected single <think>, got {pieces:?}"
        );
        assert!(
            !pieces.iter().any(|p| p == "im"),
            "must not split chat marker into 'im'; got {pieces:?}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn token_type_registers_unknown_and_named_eot_for_stops() {
        let path = std::env::temp_dir().join("allpaca_gpt2_token_type.gguf");
        let mut meta = gpt2_hello_world_meta();
        let tokens = meta
            .get_mut("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                GgufValue::Array(a) => Some(a),
                _ => None,
            })
            .unwrap();
        // Odd control piece that heuristics would miss (no <|…|>).
        tokens.push(GgufValue::String("[CTRL]".into()));
        let ctrl_id = (tokens.len() - 1) as u32;
        tokens.push(GgufValue::String("<|eot_id|>".into()));
        let eot_id = (tokens.len() - 1) as u32;
        let n = tokens.len();
        let mut types = vec![1i32; n]; // NORMAL
        types[0] = TOKEN_TYPE_UNKNOWN;
        types[ctrl_id as usize] = TOKEN_TYPE_CONTROL;
        types[eot_id as usize] = TOKEN_TYPE_CONTROL;
        meta.insert(
            "tokenizer.ggml.token_type".into(),
            GgufValue::Array(types.into_iter().map(GgufValue::Int32).collect()),
        );
        meta.insert(
            "tokenizer.ggml.eot_token_id".into(),
            GgufValue::Uint32(eot_id),
        );
        meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));
        write_gguf(&path, &meta, &[]).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let tok = extract_tokenizer(&file).unwrap();
        assert_eq!(tok.eot_id, Some(eot_id));
        assert_eq!(tok.eos_id, Some(2));
        assert_eq!(tok.stop_token_ids(), vec![2, eot_id]);
        let encoded = tok.inner.encode("hi[CTRL]<|eot_id|>", false).unwrap();
        let pieces = encoded.get_tokens();
        assert!(
            pieces.iter().any(|p| p == "[CTRL]"),
            "token_type CONTROL must register [CTRL]; got {pieces:?}"
        );
        assert!(
            pieces.iter().any(|p| p == "<|eot_id|>"),
            "expected <|eot_id|>; got {pieces:?}"
        );
        // With token_type present, heuristic-only pieces like bare <s> that are
        // NORMAL must not be forced special — still OK if they encode via BPE.
        let _ = std::fs::remove_file(path);
    }
}
