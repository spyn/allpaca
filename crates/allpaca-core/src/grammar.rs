use std::sync::Arc;

use allpaca_types::{AllpacaError, FatalError, TokenMask};
use llguidance::toktrie::{ApproximateTokEnv, TokRxInfo, TokTrie};

use crate::tokenizer::EngineTokenizer;

pub(crate) fn tok_env_from_tokenizer(tok: &EngineTokenizer) -> llguidance::toktrie::TokEnv {
    let eos = tok.eos_id.unwrap_or(0);
    let mut info = TokRxInfo::new(tok.tokens.len() as u32, eos);
    info.tok_bos = tok.bos_id;
    info.tok_unk = tok
        .tokens
        .iter()
        .position(|t| t == "<unk>" || t == "[UNK]")
        .map(|i| i as u32);
    let words: Vec<Vec<u8>> = (0..tok.tokens.len())
        .map(|id| trie_token_bytes(tok, id as u32))
        .collect();
    Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)))
}

fn trie_token_bytes(tok: &EngineTokenizer, id: u32) -> Vec<u8> {
    let raw = tok
        .tokens
        .get(id as usize)
        .map(String::as_str)
        .unwrap_or("");
    let decoded = tok.decode_one(id);
    let special = Some(id) == tok.bos_id
        || Some(id) == tok.eos_id
        || raw == "<unk>"
        || raw == "[UNK]"
        || (decoded.starts_with('<') && decoded.ends_with('>') && decoded.len() >= 2);
    if special {
        let mut bytes = vec![TokTrie::SPECIAL_TOKEN_MARKER];
        bytes.extend_from_slice(raw.as_bytes());
        if bytes.len() == 1 {
            bytes.push(b'?');
        }
        bytes
    } else if decoded.is_empty() {
        vec![0]
    } else {
        decoded.into_bytes()
    }
}

pub(crate) fn vob_to_mask(vob: &llguidance::toktrie::SimpleVob, vocab: usize) -> TokenMask {
    let mut mask = TokenMask::empty(vocab);
    vob.iter_set_entries(|id| {
        if id < vocab {
            mask.set(id, true);
        }
    });
    mask
}

pub(crate) fn factory_from_tokenizer(
    tok: &EngineTokenizer,
) -> Result<llguidance::ParserFactory, AllpacaError> {
    let env = tok_env_from_tokenizer(tok);
    let mut factory = llguidance::ParserFactory::new_simple(&env).map_err(|e| {
        AllpacaError::Fatal(FatalError::InvalidConfig(format!("grammar tokenizer: {e}")))
    })?;
    factory.quiet();
    Ok(factory)
}

pub(crate) fn matcher_from_factory(
    factory: &llguidance::ParserFactory,
    json_schema: Option<&str>,
    lark: Option<&str>,
) -> Result<llguidance::Matcher, AllpacaError> {
    let grammar = compile_grammar(json_schema, lark)?;
    let parser = factory.create_parser(grammar).map_err(|e| {
        AllpacaError::Fatal(FatalError::InvalidConfig(format!("grammar compile: {e}")))
    })?;
    let matcher = llguidance::Matcher::new(Ok(parser));
    if let Some(err) = matcher.get_error() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "grammar matcher: {err}"
        ))));
    }
    Ok(matcher)
}

fn compile_grammar(
    json_schema: Option<&str>,
    lark: Option<&str>,
) -> Result<llguidance::api::TopLevelGrammar, AllpacaError> {
    if let Some(schema) = json_schema {
        let val: serde_json::Value = serde_json::from_str(schema).map_err(|e| {
            AllpacaError::Fatal(FatalError::InvalidConfig(format!("json_schema: {e}")))
        })?;
        Ok(llguidance::api::TopLevelGrammar::from_json_schema(val))
    } else if let Some(lark) = lark {
        Ok(llguidance::api::TopLevelGrammar::from_lark(
            lark.to_string(),
        ))
    } else {
        Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "no grammar specified".into(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_schema() -> &'static str {
        r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}"#
    }

    #[test]
    fn schema_mask_after_complete_object_allows_eos() {
        let tok = EngineTokenizer::synthetic(64).unwrap();
        let factory = factory_from_tokenizer(&tok).unwrap();
        let mut matcher = matcher_from_factory(&factory, Some(ok_schema()), None).unwrap();
        let pieces = ["{", "\"", "ok", "\"", ":", "true", "}"];
        for piece in pieces {
            let id = tok.tokens.iter().position(|t| t == piece).unwrap() as u32;
            let vob = matcher.compute_mask().expect("mask before consume");
            let mask = vob_to_mask(&vob, tok.vocab_size());
            assert!(
                mask.allowed(id as usize),
                "schema must allow {piece:?} (id={id})"
            );
            matcher
                .consume_token(id)
                .unwrap_or_else(|e| panic!("consume {piece:?}: {e}"));
        }
        let vob = matcher
            .compute_mask_or_eos()
            .expect("mask after complete object");
        let mask = vob_to_mask(&vob, tok.vocab_size());
        let eos = tok.eos_id.expect("synthetic eos") as usize;
        assert!(
            mask.allowed(eos) || matcher.is_stopped() || matcher.is_accepting().unwrap_or(false),
            "complete schema object must allow EOS or be accepting; set={} stopped={} accepting={:?} err={:?}",
            mask.any(),
            matcher.is_stopped(),
            matcher.is_accepting().ok(),
            matcher.get_error()
        );
        assert!(
            mask.allowed(eos),
            "complete schema object must keep EOS in the mask so generate can stop"
        );
    }

    #[test]
    fn schema_mask_at_start_allows_object_not_letter() {
        let tok = EngineTokenizer::synthetic(64).unwrap();
        let factory = factory_from_tokenizer(&tok).unwrap();
        let mut matcher = matcher_from_factory(&factory, Some(ok_schema()), None).unwrap();
        let vob = matcher.compute_mask().unwrap();
        let mask = vob_to_mask(&vob, tok.vocab_size());
        let open = tok.tokens.iter().position(|t| t == "{").unwrap();
        let letter = tok.tokens.iter().position(|t| t == "a").unwrap();
        assert!(mask.allowed(open), "schema must allow '{{' as first token");
        assert!(
            !mask.allowed(letter),
            "schema must reject a bare letter before the object starts"
        );
    }
}
