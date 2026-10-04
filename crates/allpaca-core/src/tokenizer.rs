use ahash::AHashMap;
use allpaca_gguf::GgufTokenizer;
use allpaca_safetensors::SidecarTokenizer;
use allpaca_types::{AllpacaError, FatalError};
use serde_json::Value;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::Tokenizer;

pub struct EngineTokenizer {
    inner: Tokenizer,
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
    /// HuggingFace `tokenizer.json` decoder (ByteLevel / BPE), not piece-concat.
    hf_native_decode: bool,
    /// GGUF `gpt2` vocab: pieces are GPT-2 byte-level strings (`Ġ` space,
    /// `Ċ` newline, UTF-8 split across pieces), not text.
    byte_level: bool,
}

impl EngineTokenizer {
    pub fn from_gguf(tok: GgufTokenizer) -> Self {
        Self {
            inner: tok.inner,
            tokens: tok.tokens,
            bos_id: tok.bos_id,
            eos_id: tok.eos_id,
            eot_id: tok.eot_id,
            eom_id: tok.eom_id,
            unk_id: tok.unk_id,
            sep_id: tok.sep_id,
            pad_id: tok.pad_id,
            mask_id: tok.mask_id,
            fim_pre_id: tok.fim_pre_id,
            fim_suf_id: tok.fim_suf_id,
            fim_mid_id: tok.fim_mid_id,
            fim_pad_id: tok.fim_pad_id,
            fim_rep_id: tok.fim_rep_id,
            fim_sep_id: tok.fim_sep_id,
            add_bos: tok.add_bos,
            hf_native_decode: false,
            byte_level: tok.model_kind == "gpt2",
        }
    }

    /// End-of-generation ids: GGUF eos/eot/eom plus ChatML/Llama pieces in vocab.
    ///
    /// `<|im_end|>` is not always `tokenizer.ggml.eot_token_id`. Without it in
    /// the stop set, ChatML models keep emitting the next turn (role-echo).
    pub fn stop_token_ids(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for id in [self.eos_id, self.eot_id, self.eom_id] {
            if let Some(id) = id {
                push_unique(&mut out, id);
            }
        }
        for piece in CHAT_EOG_PIECES {
            if let Some(id) = self.token_id(piece) {
                push_unique(&mut out, id);
            }
        }
        out
    }

    pub fn token_id(&self, piece: &str) -> Option<u32> {
        self.tokens
            .iter()
            .position(|t| t == piece)
            .map(|i| i as u32)
    }

    /// Vocab has ChatML turn delimiters as single pieces.
    pub fn uses_chatml(&self) -> bool {
        self.token_id("<|im_start|>").is_some() && self.token_id("<|im_end|>").is_some()
    }

    /// Vocab has Llama 3 header and end-of-turn specials.
    pub fn uses_llama3_chat(&self) -> bool {
        self.token_id("<|start_header_id|>").is_some()
            && self.token_id("<|end_header_id|>").is_some()
            && self.token_id("<|eot_id|>").is_some()
    }

    /// Vocab has Ling's (`bailingmoe2`) role specials.
    pub fn uses_ling_chat(&self) -> bool {
        self.token_id("<role>").is_some()
            && self.token_id("</role>").is_some()
            && self.token_id("<|role_end|>").is_some()
    }

    /// ChatML, Llama 3 or Ling turns when the vocab has their specials;
    /// otherwise `role: text` (tiny/synthetic).
    ///
    /// Does **not** open `<think>` — that is Qwen-specific and role-echoes on
    /// Nemotron Lightning.
    pub fn format_chat(&self, messages: &[(&str, &str)], add_generation_prompt: bool) -> String {
        if self.uses_chatml() {
            format_chatml(messages, add_generation_prompt)
        } else if self.uses_ling_chat() {
            format_ling_chat(messages, add_generation_prompt)
        } else if self.uses_llama3_chat() {
            format_llama3_chat(messages, add_generation_prompt)
        } else {
            format_chat_colon(messages, add_generation_prompt)
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
            eot_id: None,
            eom_id: None,
            unk_id: None,
            sep_id: None,
            pad_id: None,
            mask_id: None,
            fim_pre_id: None,
            fim_suf_id: None,
            fim_mid_id: None,
            fim_pad_id: None,
            fim_rep_id: None,
            fim_sep_id: None,
            add_bos: bos_id.is_some(),
            hf_native_decode: false,
            byte_level: false,
        })
    }

    pub fn synthetic(vocab_size: usize) -> Result<Self, AllpacaError> {
        Self::from_vocab(synthetic_vocab(vocab_size), Some(1), Some(2))
    }

    /// Token ids of `text` alone, never with a BOS token in front.
    pub fn encode_plain(&self, text: &str) -> Result<Vec<u32>, AllpacaError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| AllpacaError::Fatal(FatalError::Tokenizer(e.to_string())))?;
        Ok(encoding.get_ids().to_vec())
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

    pub fn from_sidecar(side: SidecarTokenizer) -> Result<Self, AllpacaError> {
        match side {
            SidecarTokenizer::Vocab(v) => Self::from_vocab(v.tokens, v.bos_id, v.eos_id),
            SidecarTokenizer::HuggingFace {
                tokenizer_json,
                tokenizer_config,
                generation_config,
            } => Self::from_hf_files(
                &tokenizer_json,
                tokenizer_config.as_deref(),
                generation_config.as_deref(),
            ),
        }
    }

    pub fn from_hf_files(
        tokenizer_json: &std::path::Path,
        tokenizer_config: Option<&std::path::Path>,
        generation_config: Option<&std::path::Path>,
    ) -> Result<Self, AllpacaError> {
        let inner = Tokenizer::from_file(tokenizer_json).map_err(|e| {
            AllpacaError::Fatal(FatalError::Tokenizer(format!(
                "HuggingFace tokenizer.json {}: {e}",
                tokenizer_json.display()
            )))
        })?;
        let n = inner.get_vocab_size(true);
        let mut tokens = Vec::with_capacity(n);
        for i in 0..n {
            tokens.push(
                inner
                    .id_to_token(i as u32)
                    .unwrap_or_else(|| format!("<{i}>")),
            );
        }
        if tokens.is_empty() {
            return Err(AllpacaError::Fatal(FatalError::Tokenizer(
                "HuggingFace tokenizer.json vocab is empty".into(),
            )));
        }
        let tok_cfg = read_json_object(tokenizer_config)?;
        let gen_cfg = read_json_object(generation_config)?;
        let mut bos_id = gen_cfg
            .as_ref()
            .and_then(|v| json_u32_first(v.get("bos_token_id")));
        let mut eos_id = gen_cfg
            .as_ref()
            .and_then(|v| json_u32_first(v.get("eos_token_id")));
        let eot_id = gen_cfg
            .as_ref()
            .and_then(|v| json_u32_second(v.get("eos_token_id")));
        if bos_id.is_none() {
            bos_id = tok_cfg
                .as_ref()
                .and_then(|v| json_token_string(v.get("bos_token")))
                .and_then(|s| tokens.iter().position(|t| t == &s).map(|i| i as u32));
        }
        if eos_id.is_none() {
            eos_id = tok_cfg
                .as_ref()
                .and_then(|v| json_token_string(v.get("eos_token")))
                .and_then(|s| tokens.iter().position(|t| t == &s).map(|i| i as u32));
        }
        let add_bos = tok_cfg
            .as_ref()
            .and_then(|v| v.get("add_bos_token"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Ok(Self {
            inner,
            tokens,
            bos_id,
            eos_id,
            eot_id,
            eom_id: None,
            unk_id: None,
            sep_id: None,
            pad_id: None,
            mask_id: None,
            fim_pre_id: None,
            fim_suf_id: None,
            fim_mid_id: None,
            fim_pad_id: None,
            fim_rep_id: None,
            fim_sep_id: None,
            add_bos,
            hf_native_decode: true,
            byte_level: false,
        })
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, AllpacaError> {
        if self.hf_native_decode {
            return self
                .inner
                .decode(ids, true)
                .map_err(|e| AllpacaError::Fatal(FatalError::Tokenizer(e.to_string())));
        }
        if self.byte_level {
            // Collect bytes first: one UTF-8 char can span several pieces.
            let mut bytes = Vec::new();
            for &id in ids {
                match self.tokens.get(id as usize) {
                    Some(piece) if is_special_piece(piece) => {}
                    Some(piece) => push_gpt2_piece_bytes(piece, &mut bytes),
                    None => bytes.extend_from_slice(format!("<{id}>").as_bytes()),
                }
            }
            return Ok(String::from_utf8_lossy(&bytes).into_owned());
        }
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
        let Some(piece) = self.tokens.get(id as usize) else {
            return format!("<{id}>");
        };
        if self.byte_level && !is_special_piece(piece) {
            let mut bytes = Vec::new();
            push_gpt2_piece_bytes(piece, &mut bytes);
            return String::from_utf8_lossy(&bytes).into_owned();
        }
        normalize_piece(piece)
    }

    /// Raw bytes of one token, for streaming: a UTF-8 character can span
    /// several tokens (byte-level BPE pieces, SPM `<0xNN>` byte tokens), so
    /// text must be cut only on complete characters. See [`Utf8Stream`].
    pub fn decode_one_bytes(&self, id: u32) -> Vec<u8> {
        let Some(piece) = self.tokens.get(id as usize) else {
            return format!("<{id}>").into_bytes();
        };
        if is_special_piece(piece) {
            return normalize_piece(piece).into_bytes();
        }
        if self.byte_level {
            let mut bytes = Vec::new();
            push_gpt2_piece_bytes(piece, &mut bytes);
            return bytes;
        }
        if let Some(byte) = parse_hex_byte_token(piece) {
            return vec![byte];
        }
        normalize_piece(piece).into_bytes()
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

/// Turns a stream of token bytes into text without splitting a character:
/// an incomplete UTF-8 sequence at the end is held until the next token
/// completes it. Bytes that can never be valid UTF-8 become U+FFFD.
#[derive(Debug, Default, Clone)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    /// Append `bytes`, return the text that is now complete.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).expect("valid prefix"));
                    match e.error_len() {
                        // Incomplete trailing sequence: wait for more bytes.
                        None => {
                            self.pending.drain(..valid);
                            return out;
                        }
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }

    /// Whatever is still held, lossily (end of the reply).
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

/// llama.cpp-style extra EOG pieces (not always in GGUF eot/eos keys).
const CHAT_EOG_PIECES: &[&str] = &[
    "<|im_end|>",
    "<|eot_id|>",
    "<|role_end|>",
    "<|eom_id|>",
    "<|end_of_turn|>",
    "<|endoftext|>",
    "<|end_of_text|>",
];

fn push_unique(out: &mut Vec<u32>, id: u32) {
    if !out.contains(&id) {
        out.push(id);
    }
}

/// ChatML turns. Does not open `<think>`.
pub fn format_chatml(messages: &[(&str, &str)], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(content);
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

/// Llama 3 turns. `<|begin_of_text|>` is the tokenizer's BOS, not part of
/// the text.
pub fn format_llama3_chat(messages: &[(&str, &str)], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        out.push_str("<|start_header_id|>");
        out.push_str(role);
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(content.trim());
        out.push_str("<|eot_id|>");
    }
    if add_generation_prompt {
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    out
}

/// Ling 2.0 turns, as the GGUF's template renders them without tools: a
/// system turn always opens the prompt and ends with the thinking switch.
pub fn format_ling_chat(messages: &[(&str, &str)], add_generation_prompt: bool) -> String {
    let mut out = String::from("<role>SYSTEM</role>");
    let lead_system = messages.first().is_some_and(|(role, _)| *role == "system");
    if let Some((_, content)) = messages.first().filter(|_| lead_system) {
        out.push_str(content);
        out.push('\n');
    }
    out.push_str("detailed thinking off<|role_end|>");
    for (role, content) in messages.iter().skip(usize::from(lead_system)) {
        let tag = match *role {
            "user" => "HUMAN",
            "assistant" => "ASSISTANT",
            _ => "SYSTEM",
        };
        out.push_str("<role>");
        out.push_str(tag);
        out.push_str("</role>");
        out.push_str(content);
        out.push_str("<|role_end|>");
    }
    if add_generation_prompt {
        out.push_str("<role>ASSISTANT</role>");
    }
    out
}

/// Colon fallback for synthetic / non-ChatML vocabs.
pub fn format_chat_colon(messages: &[(&str, &str)], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        out.push_str(role);
        out.push_str(": ");
        out.push_str(content);
        out.push('\n');
    }
    if add_generation_prompt {
        out.push_str("assistant:");
    }
    out
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

/// Invert GPT-2 `bytes_to_unicode`: printable Latin-1 bytes map to their own
/// code point, the other 68 bytes (0..=32, 127..=160, 173) to U+0100.. in
/// order. Chars outside that alphabet (added tokens) pass through as UTF-8.
fn gpt2_shifted_bytes() -> &'static [u8; 68] {
    static TABLE: std::sync::OnceLock<[u8; 68]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0u8; 68];
        let shifted = (0u8..=255).filter(|b| matches!(b, 0..=32 | 127..=160 | 173));
        for (slot, byte) in table.iter_mut().zip(shifted) {
            *slot = byte;
        }
        table
    })
}

fn push_gpt2_piece_bytes(piece: &str, out: &mut Vec<u8>) {
    for ch in piece.chars() {
        let c = ch as u32;
        let printable =
            (33..=126).contains(&c) || (161..=172).contains(&c) || (174..=255).contains(&c);
        if printable {
            out.push(c as u8);
        } else if (256..256 + 68).contains(&c) {
            out.push(gpt2_shifted_bytes()[(c - 256) as usize]);
        } else {
            let mut buf = [0u8; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
}

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

fn read_json_object(path: Option<&std::path::Path>) -> Result<Option<Value>, AllpacaError> {
    let Some(path) = path else {
        return Ok(None);
    };
    if !path.is_file() {
        return Ok(None);
    }
    let src = std::fs::read_to_string(path).map_err(|e| {
        AllpacaError::Fatal(FatalError::Tokenizer(format!(
            "tokenizer sidecar {}: {e}",
            path.display()
        )))
    })?;
    let value: Value = serde_json::from_str(&src).map_err(|e| {
        AllpacaError::Fatal(FatalError::Tokenizer(format!(
            "tokenizer sidecar {}: {e}",
            path.display()
        )))
    })?;
    Ok(Some(value))
}

fn json_token_string(value: Option<&Value>) -> Option<String> {
    let value = value?;
    if let Some(s) = value.as_str() {
        return Some(s.to_string());
    }
    value
        .get("content")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn json_u32(value: &Value) -> Option<u32> {
    value.as_u64().and_then(|n| u32::try_from(n).ok())
}

fn json_u32_first(value: Option<&Value>) -> Option<u32> {
    let value = value?;
    if let Some(n) = json_u32(value) {
        return Some(n);
    }
    value.as_array()?.iter().find_map(json_u32)
}

fn json_u32_second(value: Option<&Value>) -> Option<u32> {
    value?.as_array()?.get(1).and_then(json_u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama3_chat_turns() {
        let got = format_llama3_chat(&[("system", "Be brief."), ("user", "Hi ")], true);
        let want = concat!(
            "<|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|>",
            "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        );
        assert_eq!(got, want);
    }

    /// An emoji split across tokens comes out whole, never as U+FFFD.
    #[test]
    fn utf8_stream_joins_characters_split_across_tokens() {
        let wave = "\u{1F44B}".as_bytes(); // 4 bytes
        let mut st = Utf8Stream::default();
        assert_eq!(st.push(b"Hi there! "), "Hi there! ");
        assert_eq!(st.push(&wave[..2]), "");
        assert_eq!(st.push(&wave[2..]), "\u{1F44B}");
        assert_eq!(st.push(&[0xFF, b'a']), "\u{FFFD}a");
        assert_eq!(st.push(&wave[..1]), "");
        assert_eq!(st.finish(), "\u{FFFD}");
        assert_eq!(st.finish(), "");
    }

    #[test]
    fn synthetic_encodes_hello() {
        let tok = EngineTokenizer::synthetic(64).unwrap();
        let ids = tok.encode("hello").unwrap();
        assert!(ids.contains(&3));
        assert_eq!(tok.decode_one(3), "hello");
    }

    #[test]
    fn gpt2_byte_level_pieces_decode_to_utf8() {
        let mut bytes = Vec::new();
        // "Ċ" newline, "Ġ" space, "Ã©" is 'é' split into two byte chars.
        push_gpt2_piece_bytes("\u{010A}", &mut bytes);
        push_gpt2_piece_bytes("\u{0120}caf", &mut bytes);
        push_gpt2_piece_bytes("\u{00C3}", &mut bytes);
        push_gpt2_piece_bytes("\u{00A9}\u{0109}!", &mut bytes);
        assert_eq!(String::from_utf8(bytes).unwrap(), "\n caf\u{e9}\t!");
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

    #[test]
    fn encode_hello_world_on_gpt2_merges_is_bytelevel_bpe_not_unk() {
        use allpaca_gguf::{write_gguf, GgufFile, GgufValue};
        use std::collections::HashMap;

        let path = std::env::temp_dir().join("allpaca_engine_gpt2_tok.gguf");
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
        write_gguf(&path, &meta, &[]).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let gguf_tok = allpaca_gguf::extract_tokenizer(&file).unwrap();
        let tok = EngineTokenizer::from_gguf(gguf_tok);
        let ids = tok.encode("Hello world").unwrap();
        assert_eq!(
            ids,
            vec![14, 19],
            "EngineTokenizer gpt2 encode must be Hello + Ġworld BPE ids, got {ids:?}"
        );
        assert_ne!(ids, vec![14, 0], "not a whitespace split of raw world");
        assert_eq!(tok.decode(&ids).unwrap(), "Hello world");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn from_gguf_propagates_eot_into_stop_token_ids() {
        use allpaca_gguf::{write_gguf, GgufFile, GgufValue};
        use std::collections::HashMap;

        let path = std::env::temp_dir().join("allpaca_engine_eot_tok.gguf");
        let mut meta = HashMap::new();
        meta.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("wordlevel".into()),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("<unk>".into()),
                GgufValue::String("</s>".into()),
                GgufValue::String("<|eot_id|>".into()),
                GgufValue::String("hi".into()),
            ]),
        );
        meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(1));
        meta.insert("tokenizer.ggml.eot_token_id".into(), GgufValue::Uint32(2));
        meta.insert(
            "tokenizer.ggml.add_bos_token".into(),
            GgufValue::Bool(false),
        );
        write_gguf(&path, &meta, &[]).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let tok = EngineTokenizer::from_gguf(allpaca_gguf::extract_tokenizer(&file).unwrap());
        assert_eq!(tok.stop_token_ids(), vec![1, 2]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn chatml_im_end_is_stop_without_eot_metadata() {
        let tokens = vec![
            "<unk>".into(),
            "</s>".into(),
            "<|im_start|>".into(),
            "<|im_end|>".into(),
            "hi".into(),
        ];
        let tok = EngineTokenizer::from_vocab(tokens, None, Some(1)).expect("vocab");
        assert!(tok.uses_chatml());
        assert_eq!(tok.token_id("<|im_end|>"), Some(3));
        assert_eq!(tok.stop_token_ids(), vec![1, 3]);
        assert_eq!(
            tok.format_chat(&[("user", "hello")], true),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(
            !tok.format_chat(&[("user", "hello")], true)
                .contains("<think>"),
            "must not open Qwen think on ChatML"
        );
    }

    #[test]
    fn from_hf_tokenizer_json_wordlevel_encodes_hello() {
        let dir = std::env::temp_dir().join("allpaca_hf_tok_json");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{
                "version": "1.0",
                "added_tokens": [
                    {"id": 0, "special": true, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
                    {"id": 1, "special": true, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
                    {"id": 2, "special": true, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
                ],
                "pre_tokenizer": {"type": "Whitespace"},
                "model": {
                    "type": "WordLevel",
                    "unk_token": "<unk>",
                    "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3}
                }
            }"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"bos_token": "<s>", "eos_token": "</s>", "add_bos_token": false}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"bos_token_id": 1, "eos_token_id": [2, 2]}"#,
        )
        .unwrap();
        let tok = EngineTokenizer::from_hf_files(
            &dir.join("tokenizer.json"),
            Some(&dir.join("tokenizer_config.json")),
            Some(&dir.join("generation_config.json")),
        )
        .expect("HF tokenizer.json");
        assert_eq!(tok.bos_id, Some(1));
        assert_eq!(tok.eos_id, Some(2));
        assert!(!tok.add_bos);
        let ids = tok.encode("hello").expect("encode");
        assert_eq!(ids, vec![3]);
        assert_eq!(tok.decode(&ids).expect("decode"), "hello");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn synthetic_vocab_keeps_colon_chat() {
        let tok = EngineTokenizer::synthetic(32).expect("synthetic");
        assert!(!tok.uses_chatml());
        assert_eq!(
            tok.format_chat(&[("user", "hello")], true),
            "user: hello\nassistant:"
        );
    }

    fn gated_qwen_gguf_path() -> Option<std::path::PathBuf> {
        for key in ["ALLPACA_ROCMFP4_GGUF", "ALLPACA_QWEN_GGUF"] {
            if let Ok(raw) = std::env::var(key) {
                let path = std::path::PathBuf::from(raw);
                if path.is_file() {
                    return Some(path);
                }
            }
        }
        let qwen = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF/Qwen3.8-27B-Q4_0_ROCMFP4_COHERENT.gguf");
        qwen.is_file().then_some(qwen)
    }

    /// llama.cpp `llama_tokenize(..., add_special=true, parse_special=true)` on the
    /// hello+`<think>` ChatML string (vocab-only dump from ROCmFPX llama.dll).
    const QWEN_CHATML_HELLO_THINK: &str =
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n";
    const LLAMA_CPP_QWEN_CHATML_HELLO_THINK_IDS: &[u32] = &[
        248045, 846, 198, 14556, 248046, 198, 248045, 74455, 198, 248068, 198,
    ];

    #[test]
    fn encode_qwen_chatml_hello_think_matches_llamacpp_ids() {
        let Some(path) = gated_qwen_gguf_path() else {
            eprintln!("skip ChatML encode parity: no Qwen GGUF");
            return;
        };
        let file = allpaca_gguf::GgufFile::open(&path).expect("open Qwen GGUF");
        let tok = EngineTokenizer::from_gguf(allpaca_gguf::extract_tokenizer(&file).unwrap());
        let ids = tok.encode(QWEN_CHATML_HELLO_THINK).expect("encode ChatML");
        let pieces: Vec<String> = ids.iter().map(|&id| tok.decode_one(id)).collect();
        assert_eq!(
            ids,
            LLAMA_CPP_QWEN_CHATML_HELLO_THINK_IDS,
            "EngineTokenizer encode must match llama.cpp parse_special ids; allpaca n={} pieces={pieces:?}",
            ids.len()
        );
    }

    const QWEN_OVERNIGHT_ESSAY: &str =
        "<|im_start|>user\nWrite a detailed 400-word essay about why hello world programs matter for learning to code. Do not stop early.<|im_end|>\n<|im_start|>assistant\n<think>\n";

    #[test]
    fn encode_qwen_overnight_essay_does_not_prepend_bos() {
        let Some(path) = gated_qwen_gguf_path() else {
            eprintln!("skip essay BOS check: no Qwen GGUF");
            return;
        };
        let file = allpaca_gguf::GgufFile::open(&path).expect("open Qwen GGUF");
        let tok = EngineTokenizer::from_gguf(allpaca_gguf::extract_tokenizer(&file).unwrap());
        assert!(
            !tok.add_bos,
            "Qwen GGUF omits add_bos_token; BPE default is false"
        );
        let ids = tok.encode(QWEN_OVERNIGHT_ESSAY).expect("encode essay");
        assert_ne!(
            ids.first().copied(),
            Some(248044),
            "must not prepend BOS 248044; n={} ids={ids:?}",
            ids.len()
        );
        assert_eq!(
            ids.first().copied(),
            Some(248045),
            "essay prompt must start with <|im_start|>; ids={ids:?}"
        );
    }

    #[test]
    fn decode_qwen_keeps_content_words_in_complete_sentence() {
        let Some(path) = gated_qwen_gguf_path() else {
            eprintln!("skip decode roundtrip: no Qwen GGUF");
            return;
        };
        let file = allpaca_gguf::GgufFile::open(&path).expect("open Qwen GGUF");
        let tok = EngineTokenizer::from_gguf(allpaca_gguf::extract_tokenizer(&file).unwrap());
        let sentence =
            "The user wants a 400-word essay about why hello world programs matter for learning to code.";
        let ids = tok.encode(sentence).expect("encode sentence");
        let text = tok.decode(&ids).expect("decode sentence");
        let pieces: Vec<String> = ids.iter().map(|&id| tok.decode_one(id)).collect();
        for word in ["user", "400-word", "hello", "programs", "learning"] {
            assert!(
                text.contains(word),
                "decode must keep {word:?}; pieces={pieces:?} text={text:?}"
            );
        }
    }
}
