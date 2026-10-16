//! Benchmark a running server through its chat endpoint: time to the first
//! token, decode rate and a hash of the text, per prompt and round.
//!
//! Rates are wall-clock between the first and the last streamed piece. The
//! hash is what tells two builds or configurations apart: at temperature 0
//! the same model must give the same text.

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The prompts used when none are given: code, explanation, prose.
pub const PROMPTS: [&str; 3] = [
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "Explain how a B-tree differs from an LSM tree and when each is the better choice for a database.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
];

#[derive(Debug, Clone)]
pub struct Target {
    /// Server base URL, e.g. `http://127.0.0.1:8091`.
    pub url: String,
    pub key: String,
    pub max_tokens: usize,
}

/// One request's result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub tag: String,
    pub round: usize,
    pub prompt: usize,
    /// Seconds from the request to the first streamed piece.
    pub first_token_s: f64,
    /// Streamed pieces (a piece is a token unless the server merges them).
    pub pieces: usize,
    /// Pieces per second between the first and the last one.
    pub rate: f64,
    /// First ten hex digits of the text's SHA-256.
    pub sha: String,
    /// Seconds since the Unix epoch when the request finished.
    pub at: u64,
    /// Requests running at the same time, this one included.
    pub concurrency: usize,
}

/// The text a streamed chat chunk adds (reasoning and content), if any.
pub fn piece_of(line: &str) -> Option<String> {
    let data = line.trim().strip_prefix("data:")?.trim();
    let chunk: serde_json::Value = serde_json::from_str(data).ok()?;
    let delta = chunk.get("choices")?.get(0)?.get("delta")?;
    let part = |key: &str| delta.get(key).and_then(|v| v.as_str()).unwrap_or("");
    let piece = format!("{}{}", part("reasoning_content"), part("content"));
    (!piece.is_empty()).then_some(piece)
}

/// The error a stream line reports, if it is one: `{"error": "..."}` or
/// `{"error": {"message": "..."}}`.
pub fn error_of(line: &str) -> Option<String> {
    let data = line.trim().strip_prefix("data:")?.trim();
    let chunk: serde_json::Value = serde_json::from_str(data).ok()?;
    let error = chunk.get("error")?;
    let message = error.get("message").unwrap_or(error);
    Some(message.as_str().map_or_else(|| message.to_string(), str::to_string))
}

pub fn sha10(text: &str) -> String {
    Sha256::digest(text.as_bytes()).iter().take(5).map(|b| format!("{b:02x}")).collect()
}

/// Pieces per second from the first to the last of `pieces` pieces.
pub fn rate(pieces: usize, first_to_last: Duration) -> f64 {
    if pieces < 2 || first_to_last.is_zero() {
        return 0.0;
    }
    (pieces - 1) as f64 / first_to_last.as_secs_f64()
}

/// One streamed chat completion of `prompt` at temperature 0.
pub fn chat(target: &Target, prompt: &str) -> Result<(f64, usize, f64, String), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(7200))
        .build()
        .map_err(|e| e.to_string())?;
    let body = serde_json::json!({
        "model": "bench",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": target.max_tokens,
        "temperature": 0.0,
        "stream": true,
    });
    let start = Instant::now();
    let response = client
        .post(format!("{}/v1/chat/completions", target.url.trim_end_matches('/')))
        .bearer_auth(&target.key)
        .json(&body)
        .send()
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("server answered {}", response.status()));
    }
    let (mut first, mut last, mut pieces, mut text) = (None, None, 0usize, String::new());
    for line in BufReader::new(response).lines() {
        let line = line.map_err(|e| e.to_string())?;
        if let Some(error) = error_of(&line) {
            return Err(format!("the server reported: {error}"));
        }
        if let Some(piece) = piece_of(&line) {
            let now = Instant::now();
            first.get_or_insert(now);
            last = Some(now);
            pieces += 1;
            text.push_str(&piece);
        }
    }
    let (Some(first), Some(last)) = (first, last) else {
        return Err("the server streamed no text".into());
    };
    Ok(((first - start).as_secs_f64(), pieces, rate(pieces, last - first), sha10(&text)))
}

fn now_unix() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// `rounds` rounds of `prompts` against `target`, numbered from
/// `first_round`, `concurrency` requests at a time (the prompts of a round
/// are spread over them), after one short warm-up request. Each sample is
/// handed to `seen` as it arrives.
#[allow(clippy::too_many_arguments)]
pub fn run(
    target: &Target,
    tag: &str,
    prompts: &[String],
    first_round: usize,
    rounds: usize,
    concurrency: usize,
    warmup_tokens: usize,
    seen: &mut dyn FnMut(&Sample),
) -> Result<Vec<Sample>, String> {
    if warmup_tokens > 0 {
        let warm = Target { max_tokens: warmup_tokens, ..target.clone() };
        chat(&warm, prompts.first().map_or(PROMPTS[0], String::as_str))?;
    }
    let concurrency = concurrency.max(1);
    let mut samples = Vec::new();
    for round in first_round..first_round + rounds {
        for group in (0..prompts.len()).collect::<Vec<_>>().chunks(concurrency) {
            let results: Vec<(usize, Result<(f64, usize, f64, String), String>)> = std::thread::scope(|scope| {
                let handles: Vec<_> =
                    group.iter().map(|&p| (p, scope.spawn(move || chat(target, &prompts[p])))).collect();
                handles
                    .into_iter()
                    .map(|(p, h)| (p, h.join().unwrap_or_else(|_| Err("request thread panicked".into()))))
                    .collect()
            });
            for (prompt, result) in results {
                let (first_token_s, pieces, rate, sha) = result?;
                let sample = Sample {
                    tag: tag.to_string(),
                    round,
                    prompt,
                    first_token_s,
                    pieces,
                    rate,
                    sha,
                    at: now_unix(),
                    concurrency: group.len(),
                };
                seen(&sample);
                samples.push(sample);
            }
        }
    }
    Ok(samples)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    match values.len() {
        0 => 0.0,
        n if n % 2 == 1 => values[n / 2],
        n => (values[n / 2 - 1] + values[n / 2]) / 2.0,
    }
}

/// A tag's samples summed up.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub tag: String,
    pub samples: usize,
    pub rate_min: f64,
    pub rate_median: f64,
    pub rate_max: f64,
    pub first_token_median_s: f64,
    /// Requests that ran at the same time (the most of any sample).
    pub concurrency: usize,
    /// The hash of each prompt's text, or `None` for a prompt whose text
    /// was not the same every round.
    pub shas: Vec<Option<String>>,
}

/// One summary per tag, in order of first appearance.
pub fn summarize(samples: &[Sample]) -> Vec<Summary> {
    let mut tags: Vec<&str> = Vec::new();
    for s in samples {
        if !tags.contains(&s.tag.as_str()) {
            tags.push(&s.tag);
        }
    }
    tags.iter()
        .map(|tag| {
            let of_tag: Vec<&Sample> = samples.iter().filter(|s| s.tag == *tag).collect();
            let rates: Vec<f64> = of_tag.iter().map(|s| s.rate).collect();
            let n_prompts = of_tag.iter().map(|s| s.prompt + 1).max().unwrap_or(0);
            let shas = (0..n_prompts)
                .map(|p| {
                    let mut of_prompt = of_tag.iter().filter(|s| s.prompt == p).map(|s| &s.sha);
                    let first = of_prompt.next()?;
                    of_prompt.all(|sha| sha == first).then(|| first.clone())
                })
                .collect();
            Summary {
                tag: tag.to_string(),
                samples: of_tag.len(),
                rate_min: rates.iter().copied().fold(f64::INFINITY, f64::min),
                rate_median: median(rates.clone()),
                rate_max: rates.iter().copied().fold(0.0, f64::max),
                first_token_median_s: median(of_tag.iter().map(|s| s.first_token_s).collect()),
                concurrency: of_tag.iter().map(|s| s.concurrency).max().unwrap_or(1),
                shas,
            }
        })
        .collect()
}

/// The summaries as a table: rates per request, the total over the
/// requests running at once (median times how many), that total against
/// the first tag's, and whether a tag's texts equal the first tag's.
pub fn table(summaries: &[Summary]) -> String {
    let mut out = format!(
        "{:<28} {:>4} {:>8} {:>8} {:>8} {:>7} {:>8} {:>8} {:>9}  {}\n",
        "tag", "n", "min", "median", "max", "at once", "total", "vs first", "1st tok s", "text"
    );
    let total = |s: &Summary| s.rate_median * s.concurrency as f64;
    let base = summaries.first();
    for s in summaries {
        let against = base.filter(|b| total(b) > 0.0).map_or(String::new(), |b| {
            format!("{:+.1}%", (total(s) / total(b) - 1.0) * 100.0)
        });
        let text = match (s.shas.iter().all(Option::is_some), base) {
            (false, _) => "differs between rounds".to_string(),
            (true, Some(b)) if b.tag != s.tag && b.shas == s.shas => "same as first".to_string(),
            (true, Some(b)) if b.tag != s.tag => "differs from first".to_string(),
            _ => s.shas.iter().flatten().cloned().collect::<Vec<_>>().join(" "),
        };
        out += &format!(
            "{:<28} {:>4} {:>8.2} {:>8.2} {:>8.2} {:>7} {:>8.2} {:>8} {:>9.2}  {}\n",
            s.tag,
            s.samples,
            s.rate_min,
            s.rate_median,
            s.rate_max,
            s.concurrency,
            total(s),
            against,
            s.first_token_median_s,
            text
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(tag: &str, round: usize, prompt: usize, rate: f64, sha: &str) -> Sample {
        Sample {
            tag: tag.into(),
            round,
            prompt,
            first_token_s: 0.5 + prompt as f64 / 10.0,
            pieces: 200,
            rate,
            sha: sha.into(),
            at: 0,
            concurrency: 1,
        }
    }

    #[test]
    fn a_stream_line_gives_its_reasoning_and_content_text() {
        let line = r#"data: {"choices":[{"delta":{"reasoning_content":"think ","content":"hi"}}]}"#;
        assert_eq!(piece_of(line).as_deref(), Some("think hi"));
        assert_eq!(piece_of(r#"data: {"choices":[{"delta":{"content":"x"}}]}"#).as_deref(), Some("x"));
        // Role-only chunks, the end marker, comments and blank lines add nothing.
        assert_eq!(piece_of(r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#), None);
        assert_eq!(piece_of("data: [DONE]"), None);
        assert_eq!(piece_of(": keep-alive"), None);
        assert_eq!(piece_of(""), None);
        assert_eq!(piece_of(r#"data: {"choices":[]}"#), None);
        // An error line is reported as one, in either shape.
        assert_eq!(error_of(r#"data: {"error":"another request is already active"}"#).as_deref(), Some("another request is already active"));
        assert_eq!(error_of(r#"data: {"error":{"message":"busy","type":"x"}}"#).as_deref(), Some("busy"));
        assert_eq!(error_of(line), None);
    }

    #[test]
    fn the_rate_counts_the_gaps_between_pieces() {
        assert_eq!(rate(201, Duration::from_secs(10)), 20.0);
        assert_eq!(rate(1, Duration::from_secs(10)), 0.0);
        assert_eq!(rate(5, Duration::ZERO), 0.0);
        assert_eq!(sha10("abc"), "ba7816bf8f");
    }

    #[test]
    fn summaries_compare_rates_and_texts_with_the_first_tag() {
        let samples = vec![
            sample("base", 1, 0, 16.0, "aa"),
            sample("base", 1, 1, 18.0, "bb"),
            sample("new", 1, 0, 20.0, "aa"),
            sample("new", 1, 1, 22.0, "bb"),
            sample("base", 2, 0, 17.0, "aa"),
            sample("base", 2, 1, 19.0, "bb"),
            sample("new", 2, 0, 21.0, "aa"),
            sample("new", 2, 1, 23.0, "bb"),
            sample("drift", 1, 0, 17.5, "aa"),
            sample("drift", 2, 0, 17.5, "zz"),
            sample("other", 1, 0, 35.0, "cc"),
            Sample { concurrency: 2, ..sample("pair", 1, 0, 14.0, "aa") },
            Sample { concurrency: 2, ..sample("pair", 1, 1, 14.0, "bb") },
        ];
        let sums = summarize(&samples);
        assert_eq!(sums.iter().map(|s| s.tag.as_str()).collect::<Vec<_>>(), ["base", "new", "drift", "other", "pair"]);
        assert_eq!((sums[0].samples, sums[0].rate_min, sums[0].rate_median, sums[0].rate_max), (4, 16.0, 17.5, 19.0));
        assert_eq!(sums[1].rate_median, 21.5);
        assert_eq!(sums[0].shas, [Some("aa".to_string()), Some("bb".to_string())]);
        assert_eq!(sums[2].shas, [None], "a prompt whose text changed between rounds");
        let table = table(&sums);
        let row = |tag: &str| table.lines().find(|l| l.starts_with(tag)).unwrap().to_string();
        assert!(row("new").contains("+22.9%") && row("new").ends_with("same as first"), "{table}");
        assert!(row("drift").ends_with("differs between rounds"), "{table}");
        assert!(row("other").contains("+100.0%") && row("other").ends_with("differs from first"), "{table}");
        assert!(row("base").ends_with("aa bb"), "{table}");
        // Two at once at 14 each is 28 in total: +60% on the base's 17.5.
        assert!(row("pair").contains("28.00") && row("pair").contains("+60.0%"), "{table}");
        assert!(row("pair").ends_with("same as first"), "{table}");
    }
}
