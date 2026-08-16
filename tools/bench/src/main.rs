use std::time::Instant;

use allpaca_core::Engine;
use allpaca_types::{CliOverrides, Config, GenerationParams};
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 32)]
    tokens: usize,
    #[arg(long, default_value = "hello world")]
    prompt: String,
}

fn main() {
    let args = Args::parse();
    let mut cfg = Config::compiled_defaults();
    cfg.merge_cli(&CliOverrides {
        synthetic: true,
        cpu_only: true,
        seed: Some(1),
        ..CliOverrides::default()
    });
    let engine = Engine::new(cfg).expect("engine");
    let params = GenerationParams {
        max_tokens: args.tokens,
        temperature: 0.0,
        seed: Some(1),
        ..GenerationParams::default()
    };
    let start = Instant::now();
    let out = engine.generate(&args.prompt, params).expect("generate");
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
