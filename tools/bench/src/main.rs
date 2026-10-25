//! `allpaca-bench`: measure a server's decode speed and compare builds or
//! configurations.
//!
//!   run     prompts against a running server
//!   ab      start a server per config in turn, round by round, and compare
//!   report  the table of a results file
//!   synthetic  the built-in tiny CPU model (no server)
//!
//! `ab` alternates the configs within each round on purpose: a machine's
//! speed drifts by several percent over minutes, so "all of A, then all of
//! B" measures the drift as well.

mod roofline;
mod serve;
mod snap;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use allpaca_core::Engine;
use allpaca_types::{CliOverrides, Config, GenerationParams};
use clap::{Parser, Subcommand};

use serve::{Sample, Target};

#[derive(Parser)]
#[command(name = "allpaca-bench", about = "Benchmark an Allpaca server")]
struct Args {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(clap::Args, Clone)]
struct Load {
    /// Tokens generated per prompt.
    #[arg(long, default_value_t = 200)]
    tokens: usize,
    /// Times the prompt set is run.
    #[arg(long, default_value_t = 2)]
    rounds: usize,
    /// Requests sent at the same time.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
    /// Tokens of the warm-up request before the first round (0 = none).
    #[arg(long, default_value_t = 40)]
    warmup: usize,
    /// A file with one prompt per line, instead of the three built-in ones.
    #[arg(long)]
    prompts: Option<PathBuf>,
    /// Append every sample to this file, one JSON object per line.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the prompts against a running server.
    Run {
        /// Server base URL.
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        /// The server's password (`Authorization: Bearer`). Env: `ALLPACA_BENCH_KEY`.
        #[arg(long)]
        key: Option<String>,
        /// Name of this run in the results.
        #[arg(long, default_value = "run")]
        tag: String,
        #[command(flatten)]
        load: Load,
    },
    /// Start a server for each config in turn, every round, and compare them.
    Ab {
        /// A server config (TOML); give it two or more times. Its file name is its tag.
        #[arg(long = "config", required = true)]
        configs: Vec<PathBuf>,
        /// The server binary.
        #[arg(long, default_value = "target/release/allpaca-server")]
        server: PathBuf,
        /// Seconds to wait for a server to load its model.
        #[arg(long, default_value_t = 900)]
        load_timeout: u64,
        #[command(flatten)]
        load: Load,
    },
    /// Print the comparison table of a results file written with `--out`.
    Report {
        file: PathBuf,
        /// Only these tags, in this order (default: all, in order of appearance).
        #[arg(long)]
        tag: Vec<String>,
    },
    /// The built-in tiny CPU model, without a server.
    Synthetic {
        #[arg(long, default_value_t = 32)]
        tokens: usize,
        #[arg(long, default_value = "hello world")]
        prompt: String,
    },
}

fn prompts(load: &Load) -> Result<Vec<String>, String> {
    match &load.prompts {
        None => Ok(serve::PROMPTS.iter().map(|p| p.to_string()).collect()),
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let lines: Vec<String> = text.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect();
            if lines.is_empty() {
                return Err(format!("{} has no prompts", path.display()));
            }
            Ok(lines)
        }
    }
}

fn record(out: Option<&Path>, sample: &Sample) {
    println!(
        "{} round {} prompt {}: first token {:.2}s, {} pieces, {:.2}/s, text {}",
        sample.tag, sample.round, sample.prompt, sample.first_token_s, sample.pieces, sample.rate, sample.sha
    );
    let Some(path) = out else {
        return;
    };
    let line = serde_json::to_string(sample).unwrap_or_default();
    let written = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| writeln!(f, "{line}"));
    if let Err(e) = written {
        eprintln!("could not write {}: {e}", path.display());
    }
}

fn run(target: &Target, tag: &str, load: &Load, first_round: usize, rounds: usize) -> Result<Vec<Sample>, String> {
    serve::run(
        target,
        tag,
        &prompts(load)?,
        first_round,
        rounds,
        load.concurrency,
        load.warmup,
        &mut |s| record(load.out.as_deref(), s),
    )
}

/// A started server; stopped when dropped.
struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        // Child::kill ends only this process. taskkill /T also ends any process
        // it started, in case one of them holds the port. Neither waits for the
        // port to close; `start` checks that before the next server starts.
        #[cfg(windows)]
        {
            let _ = Command::new("taskkill")
                .args(["/F", "/T", "/PID", &self.0.id().to_string()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        let _ = self.0.wait();
    }
}

/// Whether something still serves HTTP at `target`. Any response counts, a 401
/// or 503 too: an old server with another password is still running. Only a
/// failed connection means the port is free.
fn port_answers(client: &reqwest::blocking::Client, target: &Target) -> bool {
    client
        .get(format!("{}/v1/models", target.url))
        .bearer_auth(&target.key)
        .send()
        .is_ok()
}

/// Start `server` with `config` and wait until it answers.
fn start(server: &Path, config: &Path, timeout: Duration) -> Result<(Server, Target), String> {
    let cfg = Config::from_toml_file(config).map_err(|e| format!("{}: {e}", config.display()))?;
    let key = cfg
        .http
        .password
        .clone()
        .filter(|p| !p.is_empty())
        .ok_or_else(|| format!("{}: set [http] password, the bench needs it to log in", config.display()))?;
    let host = if cfg.http.host == "0.0.0.0" { "127.0.0.1" } else { cfg.http.host.as_str() };
    let target = Target {
        url: format!("http://{host}:{}", cfg.http.port),
        key,
        max_tokens: 0,
    };
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())?;
    // A previous server can still be answering on this port. Starting now would
    // time that process and report a fresh start.
    let down_by = Instant::now() + Duration::from_secs(20);
    while port_answers(&client, &target) {
        if Instant::now() > down_by {
            return Err(format!(
                "{}: port {} is still serving, so this would not be a fresh start",
                config.display(),
                cfg.http.port
            ));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    // A null stdout crashes the HIP server on Windows (exit 0xffffffff) before
    // it listens. Keep the log beside the config so a failed start is readable.
    let log_path = config.with_extension("server.log");
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("{}: {e}", log_path.display()))?;
    let err = log.try_clone().map_err(|e| format!("{}: {e}", log_path.display()))?;
    let child = Command::new(server)
        .arg("--config")
        .arg(config)
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(err))
        .spawn()
        .map_err(|e| format!("{}: {e}", server.display()))?;
    let mut server = Server(child);
    let started = Instant::now();
    loop {
        if let Ok(Some(status)) = server.0.try_wait() {
            return Err(format!("the server for {} exited ({status}) before it was ready", config.display()));
        }
        let ready = client
            .get(format!("{}/v1/models", target.url))
            .bearer_auth(&target.key)
            .send()
            .is_ok_and(|r| r.status().is_success());
        if ready {
            return Ok((server, target));
        }
        if started.elapsed() > timeout {
            return Err(format!("the server for {} was not ready after {}s", config.display(), timeout.as_secs()));
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn tag_of(config: &Path) -> String {
    config.file_stem().map_or_else(|| "config".to_string(), |s| s.to_string_lossy().into_owned())
}

fn ab(configs: &[PathBuf], server: &Path, load_timeout: u64, load: &Load) -> Result<(), String> {
    let mut samples = Vec::new();
    for round in 1..=load.rounds {
        for config in configs {
            let tag = tag_of(config);
            let loading = Instant::now();
            let (running, target) = start(server, config, Duration::from_secs(load_timeout))?;
            println!("{tag}: server ready after {:.0}s", loading.elapsed().as_secs_f64());
            let target = Target { max_tokens: load.tokens, ..target };
            // One round per start.
            samples.extend(run(&target, &tag, load, round, 1)?);
            drop(running);
        }
    }
    println!("\n{}", serve::report_text(&samples));
    Ok(())
}

fn report(file: &Path, tags: &[String]) -> Result<(), String> {
    let text = std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
    let mut samples: Vec<Sample> = text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    if !tags.is_empty() {
        samples.retain(|s| tags.contains(&s.tag));
        samples.sort_by_key(|s| tags.iter().position(|t| *t == s.tag));
    }
    if samples.is_empty() {
        return Err(format!("{} has no samples", file.display()));
    }
    print!("{}", serve::report_text(&samples));
    Ok(())
}

fn synthetic(tokens: usize, prompt: &str) {
    let mut cfg = Config::compiled_defaults();
    cfg.merge_cli(&CliOverrides {
        synthetic: true,
        cpu_only: true,
        seed: Some(1),
        ..CliOverrides::default()
    });
    let engine = Engine::new(cfg).expect("engine");
    let params = GenerationParams {
        max_tokens: tokens,
        temperature: 0.0,
        seed: Some(1),
        ..GenerationParams::default()
    };
    let start = Instant::now();
    let out = engine.generate(prompt, params).expect("generate");
    let elapsed = start.elapsed();
    let tps = out.completion_tokens as f64 / elapsed.as_secs_f64().max(1e-9);
    println!(
        "tokens={} prompt_tokens={} elapsed_ms={:.2} tok/s={:.2} finish={:?}",
        out.completion_tokens,
        out.prompt_tokens,
        elapsed.as_secs_f64() * 1000.0,
        tps,
        out.finish_reason
    );
}

fn main() {
    let result = match Args::parse().command {
        Cmd::Run { url, key, tag, load } => {
            let key = key.or_else(|| std::env::var("ALLPACA_BENCH_KEY").ok()).unwrap_or_default();
            let target = Target { url, key, max_tokens: load.tokens };
            run(&target, &tag, &load, 1, load.rounds)
                .map(|samples| println!("\n{}", serve::report_text(&samples)))
        }
        Cmd::Ab { configs, server, load_timeout, load } => ab(&configs, &server, load_timeout, &load),
        Cmd::Report { file, tag } => report(&file, &tag),
        Cmd::Synthetic { tokens, prompt } => {
            synthetic(tokens, &prompt);
            Ok(())
        }
    };
    if let Err(e) = result {
        eprintln!("allpaca-bench: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// A server on a free local port that answers every request with `status`.
    fn answering(status: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let _ = write!(stream, "HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            }
        });
        url
    }

    fn client() -> reqwest::blocking::Client {
        reqwest::blocking::Client::builder().timeout(Duration::from_secs(5)).build().unwrap()
    }

    fn target(url: String) -> Target {
        Target { url, key: "wrong".into(), max_tokens: 0 }
    }

    #[test]
    fn an_old_server_refusing_the_key_still_counts_as_serving() {
        assert!(port_answers(&client(), &target(answering("401 Unauthorized"))));
        assert!(port_answers(&client(), &target(answering("503 Service Unavailable"))));
        assert!(port_answers(&client(), &target(answering("200 OK"))));
    }

    #[test]
    fn a_closed_port_is_gone() {
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        assert!(!port_answers(&client(), &target(format!("http://127.0.0.1:{port}"))));
    }
}
