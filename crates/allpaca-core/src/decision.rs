//! Decision models: a typed question about a `state` is answered in one
//! forward pass, no token is generated. Request and response follow
//! llama.cpp's `POST /v1/systemone` (ggml-org/llama.cpp#29818), and so do the
//! prompts and the readout of each model type (`<arch>.decision.type`).
//!
//! Supported types, both on the `qwen35` architecture:
//! - `openjev`: every option gets a letter; the scores are the logits of the
//!   letters after the prompt.
//! - `kev`: every option ends with `<|box_end|>`; a small head projects the
//!   hidden state of the last token and of each end token, and the score of
//!   an option is the dot product of the two.

use allpaca_gguf::{GgufFile, GgufValue};
use allpaca_types::{AllpacaError, FatalError};
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::fmt;

use crate::tokenizer::EngineTokenizer;

/// A JSON value whose objects keep the order of their keys: the options of a
/// question are shown to the model, and answered, in the order they were sent.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(serde_json::Number),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    /// Truthiness as in the prompt templates: null, false, 0 and empty
    /// strings, arrays and objects are skipped.
    fn is_set(&self) -> bool {
        match self {
            Json::Null => false,
            Json::Bool(b) => *b,
            Json::Num(n) => n.as_f64().is_some_and(|v| v != 0.0),
            Json::Str(s) => !s.is_empty(),
            Json::Arr(a) => !a.is_empty(),
            Json::Obj(o) => !o.is_empty(),
        }
    }

    fn float(v: f64) -> Json {
        serde_json::Number::from_f64(v).map_or(Json::Null, Json::Num)
    }

    /// A string as it is, anything else as JSON text in the template's style
    /// (`{"a": 1, "b": [1, 2]}`).
    fn text_or_json(&self) -> String {
        match self {
            Json::Str(s) => s.clone(),
            other => {
                let mut out = String::new();
                other.dump_spaced(&mut out);
                out
            }
        }
    }

    fn dump_spaced(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => out.push_str(&number_short(n)),
            Json::Str(s) => {
                out.push('"');
                for c in s.chars() {
                    match c {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\u{8}' => out.push_str("\\b"),
                        '\u{c}' => out.push_str("\\f"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                        c => out.push(c),
                    }
                }
                out.push('"');
            }
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    item.dump_spaced(out);
                }
                out.push(']');
            }
            Json::Obj(entries) => {
                out.push('{');
                for (i, (key, item)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    Json::Str(key.clone()).dump_spaced(out);
                    out.push_str(": ");
                    item.dump_spaced(out);
                }
                out.push('}');
            }
        }
    }
}

/// A number as the reference's template engine writes it: integers as they
/// are, the rest with six significant digits (C++ stream default).
fn number_short(n: &serde_json::Number) -> String {
    if n.is_i64() || n.is_u64() {
        return n.to_string();
    }
    let v = n.as_f64().unwrap_or(0.0);
    if v == 0.0 {
        return "0".into();
    }
    let sci = format!("{v:.5e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let trim = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if !(-5..6).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        return format!("{}e{sign}{:02}", trim(mantissa), exp.abs());
    }
    trim(&format!("{v:.*}", (5 - exp).max(0) as usize))
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Json;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_bool<E>(self, v: bool) -> Result<Json, E> {
                Ok(Json::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Json, E> {
                Ok(Json::Num(v.into()))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Json, E> {
                Ok(Json::Num(v.into()))
            }
            fn visit_f64<E>(self, v: f64) -> Result<Json, E> {
                Ok(Json::float(v))
            }
            fn visit_str<E>(self, v: &str) -> Result<Json, E> {
                Ok(Json::Str(v.to_string()))
            }
            fn visit_string<E>(self, v: String) -> Result<Json, E> {
                Ok(Json::Str(v))
            }
            fn visit_unit<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_none<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_some<D: Deserializer<'de>>(self, de: D) -> Result<Json, D::Error> {
                Json::deserialize(de)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Json::Arr(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry::<String, Json>()? {
                    entries.push(entry);
                }
                Ok(Json::Obj(entries))
            }
        }
        de.deserialize_any(V)
    }
}

impl Serialize for Json {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        match self {
            Json::Null => ser.serialize_unit(),
            Json::Bool(b) => ser.serialize_bool(*b),
            Json::Num(n) => n.serialize(ser),
            Json::Str(s) => ser.serialize_str(s),
            Json::Arr(items) => {
                let mut seq = ser.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Json::Obj(entries) => {
                let mut map = ser.serialize_map(Some(entries.len()))?;
                for (key, item) in entries {
                    map.serialize_entry(key, item)?;
                }
                map.end()
            }
        }
    }
}

/// Why a decision request was not answered.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionError {
    /// The request is malformed (HTTP 400).
    Invalid(String),
    /// The loaded model cannot serve it (HTTP 501).
    Unsupported(String),
}

impl fmt::Display for DecisionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DecisionError::Invalid(m) | DecisionError::Unsupported(m) => f.write_str(m),
        }
    }
}

fn invalid(msg: impl Into<String>) -> DecisionError {
    DecisionError::Invalid(msg.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionType {
    Choice,
    Score,
    Noul,
}

impl QuestionType {
    fn name(self) -> &'static str {
        match self {
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
            QuestionType::Noul => "noul",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionOption {
    pub key: String,
    /// `Json::Null` when not given.
    pub description: Json,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub id: String,
    pub kind: QuestionType,
    pub instructions: Json,
    /// In the order of the model's scores.
    pub options: Vec<DecisionOption>,
}

/// A parsed `/v1/systemone` request.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRequest {
    pub state: Json,
    pub questions: Vec<Question>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionKind {
    OpenJev,
    Kev,
    /// A type Allpaca has no readout for (`laya`, `lev`, ...).
    Other,
}

/// What the engine reads from the forward pass of a decision prompt.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionRead {
    /// The logits of these tokens after the last prompt token.
    Labels(Vec<u32>),
    /// The hidden state of every `marker` token and of the last token.
    Pointer { marker: u32 },
}

/// The pointer head of a `kev` model: a token's output is `[q | k]`.
#[derive(Debug, Clone)]
struct PointerHead {
    /// `n_out` rows of `dim`.
    weight: Vec<f32>,
    bias: Vec<f32>,
    n_out: usize,
    dim: usize,
}

/// The decision side of a loaded model, from its `<arch>.decision.*` metadata.
#[derive(Debug, Clone)]
pub struct DecisionModel {
    pub kind: DecisionKind,
    pub type_name: String,
    /// `"<type>"` or `"<type>.<option count bucket>"` -> softmax temperature.
    temperatures: BTreeMap<String, f32>,
    max_options: usize,
    /// `noul` options are `[true, false]` instead of `[false, true]`.
    noul_true_first: bool,
    /// `openjev`: the token of each option letter.
    labels: Vec<u32>,
    /// `kev`: the token that ends an option.
    marker: Option<u32>,
    head: Option<PointerHead>,
}

const OPENJEV_LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

fn load_err(msg: String) -> AllpacaError {
    AllpacaError::Fatal(FatalError::ModelLoad(msg))
}

impl DecisionModel {
    /// `None` for a model without `<arch>.decision.type`.
    pub fn from_gguf(
        file: &GgufFile,
        tokenizer: &EngineTokenizer,
    ) -> Result<Option<Self>, AllpacaError> {
        let arch = file.meta_str("general.architecture").unwrap_or("");
        let prefix = format!("{arch}.decision.");
        let Some(type_name) = file.meta_str(&format!("{prefix}type")) else {
            return Ok(None);
        };
        let mut temperatures = BTreeMap::new();
        let temp_prefix = format!("{prefix}temperature.");
        for (key, value) in &file.metadata {
            let Some(name) = key.strip_prefix(&temp_prefix) else {
                continue;
            };
            let temp = match value {
                GgufValue::Float32(v) => *v,
                GgufValue::Float64(v) => *v as f32,
                _ => continue,
            };
            if temp <= 0.0 {
                return Err(load_err(format!("invalid decision temperature: {key} = {temp}")));
            }
            temperatures.insert(name.to_string(), temp);
        }
        let mut model = DecisionModel {
            kind: DecisionKind::Other,
            type_name: type_name.to_string(),
            temperatures,
            max_options: 0,
            noul_true_first: false,
            labels: Vec::new(),
            marker: None,
            head: None,
        };
        match type_name {
            "openjev" => {
                for letter in OPENJEV_LETTERS.chars() {
                    let ids = tokenizer.encode_plain(&letter.to_string())?;
                    let [id] = ids[..] else {
                        return Err(load_err(format!(
                            "decision label '{letter}' is not a single token"
                        )));
                    };
                    model.labels.push(id);
                }
                model.kind = DecisionKind::OpenJev;
                model.max_options = model.labels.len();
                model.noul_true_first = true;
            }
            "kev" => {
                let marker = tokenizer
                    .token_id("<|box_end|>")
                    .ok_or_else(|| load_err("decision model has no <|box_end|> token".into()))?;
                let info = file
                    .tensor("cls.output.weight")
                    .ok_or_else(|| load_err("decision model has no cls.output.weight".into()))?;
                let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
                let [dim, n_out] = dims[..] else {
                    return Err(load_err(format!("cls.output.weight has shape {dims:?}")));
                };
                let weight = file.tensor_f32("cls.output.weight")?;
                let bias = if file.tensor("cls.output.bias").is_some() {
                    file.tensor_f32("cls.output.bias")?
                } else {
                    vec![0.0; n_out]
                };
                if weight.len() != dim * n_out || bias.len() != n_out || n_out % 2 != 0 {
                    return Err(load_err(format!(
                        "decision head shape mismatch: weight {} bias {} for {dim} x {n_out}",
                        weight.len(),
                        bias.len()
                    )));
                }
                model.kind = DecisionKind::Kev;
                model.max_options = 255;
                model.marker = Some(marker);
                model.head = Some(PointerHead { weight, bias, n_out, dim });
            }
            other => {
                tracing::warn!(
                    decision_type = other,
                    "decision model type is not supported; /v1/systemone will answer 501"
                );
            }
        }
        if model.kind != DecisionKind::Other {
            tracing::info!(decision_type = type_name, "decision model loaded");
        }
        Ok(Some(model))
    }

    /// A `kev`-style model whose head reads `q` from the first hidden value
    /// and `k` from the second.
    #[cfg(test)]
    pub(crate) fn test_pointer(marker: u32, dim: usize) -> Self {
        let mut weight = vec![0.0; 2 * dim];
        weight[0] = 1.0;
        weight[dim + 1] = 1.0;
        DecisionModel {
            kind: DecisionKind::Kev,
            type_name: "kev".into(),
            temperatures: BTreeMap::new(),
            max_options: 255,
            noul_true_first: false,
            labels: Vec::new(),
            marker: Some(marker),
            head: Some(PointerHead {
                weight,
                bias: vec![0.0; 2],
                n_out: 2,
                dim,
            }),
        }
    }

    /// Validate a request body. Options come out in the order the model
    /// scores them.
    pub fn parse(&self, body: &str) -> Result<DecisionRequest, DecisionError> {
        if self.kind == DecisionKind::Other {
            return Err(DecisionError::Unsupported(format!(
                "decision model type \"{}\" is not supported",
                self.type_name
            )));
        }
        let body: Json =
            serde_json::from_str(body).map_err(|e| invalid(format!("invalid JSON: {e}")))?;
        if body.get("images").is_some_and(|v| v.is_set()) {
            return Err(DecisionError::Unsupported(
                "image input is not supported for decision models".into(),
            ));
        }
        let state = match body.get("state") {
            Some(state) if !state.is_null() => state.clone(),
            _ => return Err(invalid("\"state\" must be provided")),
        };
        let questions = match body.get("questions") {
            Some(Json::Obj(entries)) if !entries.is_empty() => entries,
            _ => return Err(invalid("\"questions\" must be a non-empty object")),
        };
        let mut out = Vec::with_capacity(questions.len());
        for (id, q) in questions {
            let err = |msg: &str| invalid(format!("questions.{id}: {msg}"));
            if !matches!(q, Json::Obj(_)) {
                return Err(err("must be an object"));
            }
            let instructions = match q.get("instructions") {
                Some(v) if !v.is_null() => v.clone(),
                _ => return Err(err("\"instructions\" must be provided")),
            };
            let null = Json::Null;
            let criteria = q.get("criteria").unwrap_or(&null);
            let type_name = match q.get("type") {
                Some(Json::Str(s)) => s.as_str(),
                _ => "",
            };
            let (kind, options) = match type_name {
                "choice" => {
                    let Json::Obj(entries) = criteria else {
                        return Err(err("\"criteria\" must be a non-empty object"));
                    };
                    if entries.is_empty() {
                        return Err(err("\"criteria\" must be a non-empty object"));
                    }
                    let options = entries
                        .iter()
                        .map(|(key, description)| DecisionOption {
                            key: key.clone(),
                            description: description.clone(),
                        })
                        .collect();
                    (QuestionType::Choice, options)
                }
                "score" => {
                    let levels = match criteria {
                        Json::Arr(levels) if (2..=10).contains(&levels.len()) => levels,
                        _ => return Err(err("\"criteria\" must be an array of 2 to 10 levels")),
                    };
                    let options = levels
                        .iter()
                        .enumerate()
                        .map(|(i, description)| DecisionOption {
                            key: i.to_string(),
                            description: description.clone(),
                        })
                        .collect();
                    (QuestionType::Score, options)
                }
                "noul" => {
                    if !matches!(criteria, Json::Null | Json::Obj(_)) {
                        return Err(err("\"criteria\" must be an object"));
                    }
                    let mut options: Vec<DecisionOption> = ["false", "true"]
                        .iter()
                        .map(|key| DecisionOption {
                            key: key.to_string(),
                            description: criteria.get(key).cloned().unwrap_or(Json::Null),
                        })
                        .collect();
                    if self.noul_true_first {
                        options.swap(0, 1);
                    }
                    (QuestionType::Noul, options)
                }
                _ => return Err(err("\"type\" must be one of: choice, score, noul")),
            };
            if options.len() > self.max_options {
                return Err(err(&format!(
                    "too many options ({}), this model supports at most {}",
                    options.len(),
                    self.max_options
                )));
            }
            out.push(Question {
                id: id.clone(),
                kind,
                instructions,
                options,
            });
        }
        Ok(DecisionRequest {
            state,
            questions: out,
        })
    }

    /// The prompt of one question, as the model's `systemone` template
    /// renders it.
    pub fn render(&self, state: &Json, question: &Question) -> String {
        match self.kind {
            DecisionKind::Kev => render_kev(state, question),
            _ => render_openjev(state, question),
        }
    }

    /// What to read from the forward pass of `question`'s prompt.
    pub fn read(&self, question: &Question) -> DecisionRead {
        match (self.kind, self.marker) {
            (DecisionKind::Kev, Some(marker)) => DecisionRead::Pointer { marker },
            _ => DecisionRead::Labels(self.labels[..question.options.len()].to_vec()),
        }
    }

    /// `kev` scores from `rows`: one final hidden state per option end
    /// token, then the one of the last token.
    pub(crate) fn pointer_scores(&self, rows: &[f32]) -> Result<Vec<f32>, AllpacaError> {
        let head = self.head.as_ref().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig("decision model has no pointer head".into()))
        })?;
        if rows.is_empty() || rows.len() % head.dim != 0 {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "decision hidden states: {} values for width {}",
                rows.len(),
                head.dim
            ))));
        }
        let half = head.n_out / 2;
        let project = |row: &[f32], from: usize| -> Vec<f32> {
            (from..from + half)
                .map(|o| {
                    let w = &head.weight[o * head.dim..(o + 1) * head.dim];
                    head.bias[o] + w.iter().zip(row).map(|(a, b)| a * b).sum::<f32>()
                })
                .collect()
        };
        let n = rows.len() / head.dim;
        let q = project(&rows[(n - 1) * head.dim..], 0);
        let scale = (half as f32).sqrt();
        Ok((0..n - 1)
            .map(|i| {
                let k = project(&rows[i * head.dim..(i + 1) * head.dim], half);
                q.iter().zip(&k).map(|(a, b)| a * b).sum::<f32>() / scale
            })
            .collect())
    }

    fn temperature(&self, question: &Question) -> f32 {
        let n = question.options.len();
        let bucket = match n {
            0..=2 => "2",
            3..=5 => "3_5",
            6..=10 => "6_10",
            _ => "11",
        };
        let name = question.kind.name();
        self.temperatures
            .get(&format!("{name}.{bucket}"))
            .or_else(|| self.temperatures.get(name))
            .copied()
            .unwrap_or(1.0)
    }

    /// The answer object of one question from the model's raw scores.
    pub fn answer(&self, question: &Question, scores: &[f32]) -> Result<Json, AllpacaError> {
        let n = question.options.len();
        if scores.len() != n || scores.iter().any(|s| s.is_nan()) {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "decision result does not match the options".into(),
            )));
        }
        let temperature = f64::from(self.temperature(question));
        let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f64> = scores
            .iter()
            .map(|s| (f64::from(s - top) / temperature).exp())
            .collect();
        let sum: f64 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);

        let mut answer = vec![("type".to_string(), Json::Str(question.kind.name().into()))];
        if question.kind == QuestionType::Noul {
            let p_true = question
                .options
                .iter()
                .position(|o| o.key == "true")
                .map_or(0.0, |i| probs[i]);
            answer.push(("noul".into(), Json::float(p_true)));
            return Ok(Json::Obj(answer));
        }
        let probabilities = Json::Obj(
            question
                .options
                .iter()
                .zip(&probs)
                .map(|(o, p)| (o.key.clone(), Json::float(*p)))
                .collect(),
        );
        let best = probs
            .iter()
            .enumerate()
            .fold(0, |best, (i, p)| if *p > probs[best] { i } else { best });
        if question.kind == QuestionType::Choice {
            answer.push(("choice".into(), Json::Str(question.options[best].key.clone())));
            answer.push(("probabilities".into(), probabilities));
            answer.push(("confidence".into(), Json::float(confidence_choice(&probs))));
        } else {
            let expected: f64 = probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
            let legend = Json::Obj(
                question
                    .options
                    .iter()
                    .map(|o| (o.key.clone(), o.description.clone()))
                    .collect(),
            );
            answer.push(("score".into(), Json::float(expected)));
            answer.push(("legend".into(), legend));
            answer.push(("probabilities".into(), probabilities));
            answer.push(("confidence".into(), Json::float(confidence_score(&probs, best))));
        }
        Ok(Json::Obj(answer))
    }
}

/// 0 when every option is equally likely, 1 when one option has it all.
fn confidence_choice(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let uniform = 1.0 / probs.len() as f64;
    let top = probs.iter().copied().fold(0.0, f64::max);
    ((top - uniform) / (1.0 - uniform)).max(0.0)
}

/// Mean distance to the most likely level, against the one of a uniform
/// distribution around its centre.
fn confidence_score(probs: &[f64], mode: usize) -> f64 {
    let n = probs.len();
    if n < 2 {
        return 1.0;
    }
    let mut dist = 0.0;
    let mut dist_uniform = 0.0;
    for (i, p) in probs.iter().enumerate() {
        dist += p * (i as f64 - mode as f64).abs();
        dist_uniform += (i as f64 - (n - 1) as f64 / 2.0).abs() / n as f64;
    }
    (1.0 - dist / dist_uniform).max(0.0)
}

fn render_openjev(state: &Json, question: &Question) -> String {
    let mut out = String::from("<|im_start|>user\nState:\n");
    out.push_str(&state.text_or_json());
    out.push_str("\n\nQuestion: ");
    out.push_str(&question.instructions.text_or_json());
    if question.kind == QuestionType::Score {
        out.push_str(" Rate along the ordered levels below (lowest first).");
    }
    out.push_str("\nOptions:\n");
    for (option, letter) in question.options.iter().zip(OPENJEV_LETTERS.chars()) {
        out.push('[');
        out.push(letter);
        out.push_str("] ");
        let described = option.description.is_set();
        if question.kind != QuestionType::Noul {
            out.push_str(&option.key);
            out.push_str(": ");
            if described {
                out.push_str(&option.description.text_or_json());
            }
        } else {
            let (name, default) = if option.key == "true" {
                ("yes: ", "The statement is true.")
            } else {
                ("no: ", "The statement is false.")
            };
            out.push_str(name);
            if described {
                out.push_str(&option.description.text_or_json());
            } else {
                out.push_str(default);
            }
        }
        out.push('\n');
    }
    out.push_str(
        "\nAnswer with the letter of the best option only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    );
    out
}

fn render_kev(state: &Json, question: &Question) -> String {
    let mut out = String::from("<|fim_prefix|>");
    out.push_str(&kev_text(state));
    out.push_str("<|fim_middle|>");
    out.push_str(&kev_text(&question.instructions));
    for option in &question.options {
        out.push_str("<|box_start|>");
        let description = if option.description.is_null() {
            String::new()
        } else {
            kev_text(&option.description)
        };
        if question.kind == QuestionType::Score {
            out.push_str(&description);
        } else {
            let key = kev_text(&Json::Str(option.key.clone()));
            out.push_str(match (question.kind, key.as_str()) {
                (QuestionType::Noul, "true") => "yes",
                (QuestionType::Noul, _) => "no",
                _ => &key,
            });
            if !description.is_empty() {
                out.push_str(": ");
                out.push_str(&description);
            }
        }
        out.push_str("<|box_end|>");
    }
    out.push_str("<|fim_suffix|>");
    out
}

/// `kev` takes text only: a JSON value is flattened, object keys become
/// labels and array items a dashed list.
fn kev_render(value: &Json, indent: usize) -> String {
    let pad = " ".repeat(2 * indent);
    match value {
        Json::Null => String::new(),
        Json::Str(s) => s.clone(),
        Json::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Json::Num(n) => n.to_string(),
        Json::Arr(items) => {
            let mut out = String::new();
            for item in items {
                let text = kev_render(item, indent + 1);
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&pad);
                out.push_str("- ");
                out.push_str(text.trim_start_matches([' ', '\t', '\n', '\r']));
            }
            out
        }
        Json::Obj(entries) => {
            let mut out = String::new();
            for (key, item) in entries {
                let nested = matches!(item, Json::Arr(_) | Json::Obj(_));
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&pad);
                out.push_str(key);
                out.push_str(if nested { ":\n" } else { ": " });
                out.push_str(&kev_render(item, if nested { indent + 1 } else { 0 }));
            }
            out
        }
    }
}

/// Flattened text in which `<|name|>` cannot be read as a special token:
/// its bars become broken bars.
fn kev_text(value: &Json) -> String {
    let text = kev_render(value, 0);
    let mut out = String::with_capacity(text.len());
    let mut rest = text.as_str();
    while let Some(at) = rest.find("<|") {
        let after = &rest[at + 2..];
        let name_len = after
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(after.len());
        if name_len > 0 && after[name_len..].starts_with("|>") {
            out.push_str(&rest[..at]);
            out.push_str("<\u{a6}");
            out.push_str(&after[..name_len]);
            out.push_str("\u{a6}>");
            rest = &after[name_len + 2..];
        } else {
            out.push_str(&rest[..at + 2]);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(kind: DecisionKind) -> DecisionModel {
        DecisionModel {
            kind,
            type_name: "test".into(),
            temperatures: BTreeMap::from([("noul".to_string(), 2.0)]),
            max_options: 52,
            noul_true_first: kind == DecisionKind::OpenJev,
            labels: (100..152).collect(),
            marker: (kind == DecisionKind::Kev).then_some(7),
            head: None,
        }
    }

    const BODY: &str = r#"{
        "state": {"message": "charged twice", "order": {"id": 4471, "items": ["case", "charger"]}, "vip": true},
        "questions": {
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"shipping": null, "billing": "money"}},
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", "today"]}
        }
    }"#;

    #[test]
    fn request_keeps_the_order_of_questions_and_options() {
        let req = model(DecisionKind::OpenJev).parse(BODY).unwrap();
        let ids: Vec<&str> = req.questions.iter().map(|q| q.id.as_str()).collect();
        assert_eq!(ids, ["team", "angry", "urgency"]);
        let keys = |i: usize| -> Vec<&str> {
            req.questions[i].options.iter().map(|o| o.key.as_str()).collect()
        };
        assert_eq!(keys(0), ["shipping", "billing"]);
        assert_eq!(keys(1), ["true", "false"]);
        assert_eq!(keys(2), ["0", "1"]);
        // kev scores `false` first.
        let req = model(DecisionKind::Kev).parse(BODY).unwrap();
        assert_eq!(req.questions[1].options[0].key, "false");
    }

    #[test]
    fn bad_requests_are_rejected_with_the_question_named() {
        let m = model(DecisionKind::OpenJev);
        let err = |body: &str| m.parse(body).unwrap_err();
        assert_eq!(
            err(r#"{"questions": {"a": {"type": "noul", "instructions": "x"}}}"#),
            invalid("\"state\" must be provided")
        );
        assert_eq!(
            err(r#"{"state": "s", "questions": {}}"#),
            invalid("\"questions\" must be a non-empty object")
        );
        assert_eq!(
            err(r#"{"state": "s", "questions": {"a": {"type": "score", "instructions": "x", "criteria": ["one"]}}}"#),
            invalid("questions.a: \"criteria\" must be an array of 2 to 10 levels")
        );
        assert_eq!(
            err(r#"{"state": "s", "questions": {"a": {"type": "rank", "instructions": "x"}}}"#),
            invalid("questions.a: \"type\" must be one of: choice, score, noul")
        );
        assert!(matches!(
            err(r#"{"state": "s", "images": ["data:image/png;base64,AA"], "questions": {"a": {"type": "noul", "instructions": "x"}}}"#),
            DecisionError::Unsupported(_)
        ));
        assert!(matches!(
            model(DecisionKind::Other).parse(BODY).unwrap_err(),
            DecisionError::Unsupported(_)
        ));
    }

    #[test]
    fn openjev_prompt_letters_the_options() {
        let m = model(DecisionKind::OpenJev);
        let req = m.parse(BODY).unwrap();
        assert_eq!(
            m.render(&req.state, &req.questions[0]),
            "<|im_start|>user\nState:\n{\"message\": \"charged twice\", \"order\": {\"id\": 4471, \"items\": [\"case\", \"charger\"]}, \"vip\": true}\n\n\
             Question: Which team?\nOptions:\n[A] shipping: \n[B] billing: money\n\n\
             Answer with the letter of the best option only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let noul = m.render(&Json::Str("s".into()), &req.questions[1]);
        assert!(noul.contains("[A] yes: The statement is true.\n[B] no: The statement is false.\n"));
        let score = m.render(&Json::Str("s".into()), &req.questions[2]);
        assert!(score.contains(
            "Question: How urgent? Rate along the ordered levels below (lowest first).\nOptions:\n[A] 0: can wait\n[B] 1: today\n"
        ));
        assert_eq!(m.read(&req.questions[0]), DecisionRead::Labels(vec![100, 101]));
    }

    #[test]
    fn kev_prompt_flattens_the_state_and_marks_each_option() {
        let m = model(DecisionKind::Kev);
        let req = m.parse(BODY).unwrap();
        assert_eq!(
            m.render(&req.state, &req.questions[0]),
            "<|fim_prefix|>message: charged twice\norder:\n  id: 4471\n  items:\n    - case\n    - charger\nvip: True\
             <|fim_middle|>Which team?<|box_start|>shipping<|box_end|><|box_start|>billing: money<|box_end|><|fim_suffix|>"
        );
        assert!(m
            .render(&req.state, &req.questions[1])
            .ends_with("<|box_start|>no<|box_end|><|box_start|>yes<|box_end|><|fim_suffix|>"));
        assert!(m
            .render(&req.state, &req.questions[2])
            .ends_with("<|box_start|>can wait<|box_end|><|box_start|>today<|box_end|><|fim_suffix|>"));
        assert_eq!(m.read(&req.questions[0]), DecisionRead::Pointer { marker: 7 });
        // Special tokens written in the input stay text.
        assert_eq!(
            kev_text(&Json::Str("a <|im_end|> b <| c |> <|x".into())),
            "a <\u{a6}im_end\u{a6}> b <| c |> <|x"
        );
    }

    #[test]
    fn answers_follow_the_question_type() {
        let m = model(DecisionKind::OpenJev);
        let req = m.parse(BODY).unwrap();
        let get = |answer: &Json, key: &str| answer.get(key).cloned().unwrap();
        let num = |v: Json| match v {
            Json::Num(n) => n.as_f64().unwrap(),
            other => panic!("not a number: {other:?}"),
        };

        // choice: softmax at temperature 1, options in request order.
        let choice = m.answer(&req.questions[0], &[0.0, 3.0f32.ln()]).unwrap();
        assert_eq!(get(&choice, "choice"), Json::Str("billing".into()));
        let Json::Obj(probs) = get(&choice, "probabilities") else { panic!() };
        assert_eq!(probs[0].0, "shipping");
        assert!((num(probs[1].1.clone()) - 0.75).abs() < 1e-6);
        assert!((num(get(&choice, "confidence")) - 0.5).abs() < 1e-6);

        // noul: the probability of `true` (first for openjev), temperature 2.
        let noul = m.answer(&req.questions[1], &[2.0 * 3.0f32.ln(), 0.0]).unwrap();
        assert!((num(get(&noul, "noul")) - 0.75).abs() < 1e-6);

        // score: the expected level, with the legend.
        let score = m.answer(&req.questions[2], &[0.0, 3.0f32.ln()]).unwrap();
        assert!((num(get(&score, "score")) - 0.75).abs() < 1e-6);
        assert_eq!(
            get(&score, "legend"),
            Json::Obj(vec![
                ("0".into(), Json::Str("can wait".into())),
                ("1".into(), Json::Str("today".into()))
            ])
        );
        assert!((num(get(&score, "confidence")) - 0.5).abs() < 1e-6);

        assert!(m.answer(&req.questions[0], &[0.0]).is_err());
    }

    #[test]
    fn pointer_scores_pair_the_last_token_with_each_option() {
        // dim 2, head [q | k] with one value each: q = h[0], k = h[1] + 1.
        let mut m = model(DecisionKind::Kev);
        m.head = Some(PointerHead {
            weight: vec![1.0, 0.0, 0.0, 1.0],
            bias: vec![0.0, 1.0],
            n_out: 2,
            dim: 2,
        });
        // Two option rows, then the last token's row (q = 3).
        let rows = [9.0, 1.0, 9.0, -2.0, 3.0, 9.0];
        assert_eq!(m.pointer_scores(&rows).unwrap(), [6.0, -3.0]);
        assert!(m.pointer_scores(&rows[..5]).is_err());
    }

    #[test]
    fn numbers_print_like_the_reference_template_engine() {
        let n = |text: &str| number_short(&serde_json::from_str(text).unwrap());
        assert_eq!(n("4471"), "4471");
        assert_eq!(n("1234.56"), "1234.56");
        assert_eq!(n("0.1"), "0.1");
        assert_eq!(n("2.0"), "2");
        assert_eq!(n("1234567.5"), "1.23457e+06");
        assert_eq!(n("0.000001234"), "1.234e-06");
    }
}
