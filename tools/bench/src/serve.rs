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

use crate::roofline;
use crate::snap::{self, RooflineSnap};

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

/// One GPU (or RPC card) that held weights for a start, for the roofline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceBand {
    /// Inventory key (`hip:0`, `cuda:1`, …). Empty in older jsonl.
    #[serde(default)]
    pub device: String,
    pub name: String,
    /// Peak memory bandwidth in GB/s (published table or inventory).
    pub peak_gbs: f64,
    /// Weight bytes on this device for the bytes/token split. 0 = unknown.
    #[serde(default)]
    pub weight_bytes: u64,
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
    /// Weight bytes read per decode token (dense = resident; MoE = active).
    #[serde(default)]
    pub weight_bytes: u64,
    /// KV bytes read per decode token (or device pool upper bound).
    #[serde(default)]
    pub kv_bytes: u64,
    /// True when `kv_bytes` is the whole device KV pool, not a per-sequence estimate.
    #[serde(default)]
    pub kv_is_pool: bool,
    /// True for an MoE model whose active weight bytes are unknown.
    #[serde(default)]
    pub weights_unknown: bool,
    /// Devices that held weights for this start. Empty = no roofline line.
    #[serde(default)]
    pub devices: Vec<DeviceBand>,
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

/// Copy roofline fields from a server snap onto a sample.
pub fn apply_snap(sample: &mut Sample, snap: &RooflineSnap) {
    sample.weight_bytes = snap.weight_bytes;
    sample.kv_bytes = snap.kv_bytes;
    sample.kv_is_pool = snap.kv_is_pool;
    sample.weights_unknown = snap.weights_unknown;
    sample.devices = snap.devices.clone();
}

/// `rounds` rounds of `prompts` against `target`, numbered from
/// `first_round`, `concurrency` requests at a time (the prompts of a round
/// are spread over them), after one short warm-up request. Each sample is
/// handed to `seen` as it arrives. After warm-up, `/v1/status` (and
/// settings) fill bytes/device fields for the roofline report.
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
    let snap = match snap::fetch(target, warmup_tokens as u64) {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("allpaca-bench: roofline snap skipped: {e}");
            None
        }
    };
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
                let mut sample = Sample {
                    tag: tag.to_string(),
                    round,
                    prompt,
                    first_token_s,
                    pieces,
                    rate,
                    sha,
                    at: now_unix(),
                    concurrency: group.len(),
                    weight_bytes: 0,
                    kv_bytes: 0,
                    kv_is_pool: false,
                    weights_unknown: false,
                    devices: Vec::new(),
                };
                if let Some(snap) = &snap {
                    apply_snap(&mut sample, snap);
                }
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

/// One `(tag, round)` group for the roofline section (an `ab` fresh start,
/// or one round of `run` against a long-lived server).
#[derive(Debug, Clone, PartialEq)]
pub struct FreshStart {
    pub tag: String,
    pub round: usize,
    pub rate_median: f64,
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    pub kv_is_pool: bool,
    pub weights_unknown: bool,
    pub devices: Vec<DeviceBand>,
}

/// One row per `(tag, round)` that recorded weight bytes, in first-seen order.
pub fn fresh_starts(samples: &[Sample]) -> Vec<FreshStart> {
    let mut keys: Vec<(String, usize)> = Vec::new();
    for s in samples {
        if s.weight_bytes == 0 && s.kv_bytes == 0 && !s.weights_unknown {
            continue;
        }
        let key = (s.tag.clone(), s.round);
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys.into_iter()
        .map(|(tag, round)| {
            let of: Vec<&Sample> = samples.iter().filter(|s| s.tag == tag && s.round == round).collect();
            let rates: Vec<f64> = of.iter().map(|s| s.rate).collect();
            let first = of[0];
            FreshStart {
                tag,
                round,
                rate_median: median(rates),
                weight_bytes: first.weight_bytes,
                kv_bytes: first.kv_bytes,
                kv_is_pool: first.kv_is_pool,
                weights_unknown: first.weights_unknown,
                devices: first.devices.clone(),
            }
        })
        .collect()
}

/// Ceiling tok/s from per-device bytes and peaks: 1 / Σ(Bᵢ / peakᵢ).
pub fn split_ceiling_tok_s(parts: &[(u64, f64)]) -> f64 {
    let mut secs = 0.0;
    for &(bytes, peak) in parts {
        if peak <= 0.0 || bytes == 0 {
            continue;
        }
        secs += bytes as f64 / (peak * 1e9);
    }
    if secs <= 0.0 {
        0.0
    } else {
        1.0 / secs
    }
}

fn format_gb(bytes: f64) -> String {
    format!("{:.2} GB", bytes / 1e9)
}

/// Spread `kv` across devices in proportion to their weight bytes.
fn distribute_kv(weight_parts: &[(u64, f64)], kv: u64) -> Vec<(u64, f64)> {
    let total_w: u64 = weight_parts.iter().map(|(w, _)| *w).sum();
    if total_w == 0 || kv == 0 {
        return weight_parts.to_vec();
    }
    let mut left = kv;
    let n = weight_parts.len();
    weight_parts
        .iter()
        .enumerate()
        .map(|(i, &(w, peak))| {
            let share = if i + 1 == n {
                left
            } else {
                let s = (kv as u128 * w as u128 / total_w as u128) as u64;
                left = left.saturating_sub(s);
                s
            };
            (w.saturating_add(share), peak)
        })
        .collect()
}

/// Roofline lines per `(tag, round)`, or empty when no sample recorded bytes.
pub fn roofline_table(starts: &[FreshStart]) -> String {
    if starts.is_empty() {
        return String::new();
    }
    let mut out = String::from("roofline by round (bytes/token, effective GB/s, % of peak)\n");
    out += &format!(
        "{:<20} {:>5} {:>8} {:>12} {:>10} {:>8} {:>8}  {}\n",
        "tag", "round", "tok/s", "bytes/tok", "GB/s", "% peak", "peak", "devices"
    );
    let mut any_pool_kv = false;
    let mut any_unknown = false;
    for s in starts {
        if s.weights_unknown {
            any_unknown = true;
            out += &format!(
                "{:<20} {:>5} {:>8.2} {:>12} {:>10} {:>8} {:>8}  {}\n",
                s.tag, s.round, s.rate_median, "n/a", "n/a", "n/a", "", ""
            );
            continue;
        }
        any_pool_kv |= s.kv_is_pool;
        let bpt = roofline::bytes_per_token(s.weight_bytes, s.kv_bytes);
        let gbs = roofline::effective_gbs(bpt, s.rate_median);
        let (pct, peak_label, device_note) = if s.devices.len() > 1
            && s.devices.iter().any(|d| d.weight_bytes > 0)
        {
            let weight_parts: Vec<(u64, f64)> = s
                .devices
                .iter()
                .map(|d| (d.weight_bytes, d.peak_gbs))
                .collect();
            let parts = distribute_kv(&weight_parts, s.kv_bytes);
            let ceiling = split_ceiling_tok_s(&parts);
            let pct = if ceiling > 0.0 { s.rate_median / ceiling * 100.0 } else { 0.0 };
            let note = s
                .devices
                .iter()
                .map(|d| {
                    let key = if d.device.is_empty() {
                        d.name.as_str()
                    } else {
                        d.device.as_str()
                    };
                    format!("{key}@{:.0}", d.peak_gbs)
                })
                .collect::<Vec<_>>()
                .join("+");
            (pct, "split".to_string(), note)
        } else {
            let peak = s
                .devices
                .first()
                .map(|d| roofline::peak_for(&d.name, d.peak_gbs))
                .unwrap_or(0.0);
            let pct = roofline::pct_of_peak(gbs, peak);
            let name = s.devices.first().map(|d| d.name.as_str()).unwrap_or("?");
            (pct, format!("{peak:.0}"), name.to_string())
        };
        let bytes_col = if s.kv_is_pool {
            format!("≤{}/tok", format_gb(bpt))
        } else {
            format!("{}/tok", format_gb(bpt))
        };
        out += &format!(
            "{:<20} {:>5} {:>8.2} {:>12} {:>10.1} {:>7.1}% {:>8}  {}\n",
            s.tag,
            s.round,
            s.rate_median,
            bytes_col,
            gbs,
            pct,
            peak_label,
            device_note
        );
    }
    if any_pool_kv {
        out += "note: ≤ marks KV from the device pool (upper bound; not per-sequence)\n";
    }
    if any_unknown {
        out += "note: n/a marks an MoE model with its expert size unknown to the server\n";
    }
    out
}

/// Rate summary plus roofline fresh-start lines when bytes were recorded.
pub fn report_text(samples: &[Sample]) -> String {
    let mut out = table(&summarize(samples));
    let roof = roofline_table(&fresh_starts(samples));
    if !roof.is_empty() {
        out.push('\n');
        out.push_str(&roof);
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
            weight_bytes: 0,
            kv_bytes: 0,
            kv_is_pool: false,
            weights_unknown: false,
            devices: Vec::new(),
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

    fn with_roofline(mut s: Sample, weight: u64, kv: u64, device: &str, peak: f64) -> Sample {
        s.weight_bytes = weight;
        s.kv_bytes = kv;
        s.devices = vec![DeviceBand {
            device: format!("hip:0"),
            name: device.into(),
            peak_gbs: peak,
            weight_bytes: weight,
        }];
        s
    }

    #[test]
    fn report_lists_each_round_with_bytes_gbs_and_pct_of_peak() {
        // Two ab rounds on the 9070: same bytes, different rates — listed apart.
        let samples = vec![
            with_roofline(sample("ornith", 1, 0, 36.0, "aa"), 9_500_000_000, 0, "RX 9070", 640.0),
            with_roofline(sample("ornith", 1, 1, 37.4, "bb"), 9_500_000_000, 0, "RX 9070", 640.0),
            with_roofline(sample("ornith", 2, 0, 35.0, "aa"), 9_500_000_000, 0, "RX 9070", 640.0),
            with_roofline(sample("ornith", 2, 1, 36.0, "bb"), 9_500_000_000, 0, "RX 9070", 640.0),
        ];
        let text = report_text(&samples);
        assert!(text.contains("roofline by round"), "{text}");
        let starts: Vec<&str> = text
            .lines()
            .filter(|l| l.trim_start().starts_with("ornith") && l.contains("GB/tok"))
            .collect();
        assert_eq!(starts.len(), 2, "{text}");
        assert!(starts[0].contains("9.50 GB/tok") && starts[0].contains("640") && starts[0].contains('%'), "{text}");
        assert!(starts[1].contains("9.50 GB/tok"), "{text}");
        // Round 1 median 36.7 → ~54.5% of 640; round 2 median 35.5 → ~52.7%.
        assert!(starts[0].contains("54.") || starts[0].contains("55."), "round1 pct: {}", starts[0]);
        assert!(starts[1].contains("52.") || starts[1].contains("53."), "round2 pct: {}", starts[1]);
    }

    #[test]
    fn moe_with_unknown_weights_prints_na_not_a_dense_figure() {
        let mut s = with_roofline(sample("o35", 1, 0, 39.3, "aa"), 0, 0, "RX 9070", 640.0);
        s.weights_unknown = true;
        let text = report_text(&[s]);
        let line = text.lines().find(|l| l.starts_with("o35") && l.contains("n/a")).unwrap_or_else(|| panic!("{text}"));
        assert!(!line.contains('%'), "{line}");
        assert!(text.contains("expert size unknown"), "{text}");
    }

    #[test]
    fn old_jsonl_without_bytes_still_prints_the_rate_table() {
        let samples = vec![sample("base", 1, 0, 16.0, "aa")];
        let text = report_text(&samples);
        assert!(text.contains("median"), "{text}");
        assert!(!text.contains("roofline by round"), "no roofline without bytes: {text}");
    }

    #[test]
    fn layer_split_reports_share_of_split_ceiling() {
        // 10 GB on 640 + 6 GB on 320 → ceiling ≈ 1/(10/640e9 + 6/320e9) ≈ 29.1 tok/s.
        // Measured 17.9 → ~61% of ceiling.
        let mut s = sample("q38", 1, 0, 17.9, "aa");
        s.weight_bytes = 16_000_000_000;
        s.kv_bytes = 0;
        s.devices = vec![
            DeviceBand {
                device: "hip:0".into(),
                name: "RX 9070".into(),
                peak_gbs: 640.0,
                weight_bytes: 10_000_000_000,
            },
            DeviceBand {
                device: "hip:1".into(),
                name: "RX 9060 XT".into(),
                peak_gbs: 320.0,
                weight_bytes: 6_000_000_000,
            },
        ];
        let text = report_text(&[s]);
        let line = text.lines().find(|l| l.contains("q38") && l.contains("GB/tok")).unwrap();
        assert!(line.contains("split"), "{line}");
        assert!(line.contains("61.") || line.contains("62.") || line.contains("60."), "{line}");
    }
}
