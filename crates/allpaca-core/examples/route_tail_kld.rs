//! What the route-tail skip does to a qwen4exp model's predictions: KL
//! divergence (nats per token) and top-token agreement against the exact run,
//! both on the CPU host forward.
//!
//!     cargo run --release -p allpaca-core --example route_tail_kld -- <model.gguf> <text.txt> [--tokens N] [--rank R]...
//!
//! Paragraphs of the text (split at blank lines) are read at most N tokens
//! each (default 48); one result per `--rank` (default 7).
use std::path::Path;

use allpaca_core::sensitivity::route_tail_divergence;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut tokens = 48usize;
    let mut ranks: Vec<u32> = Vec::new();
    while let Some(at) = args.iter().position(|a| a == "--tokens" || a == "--rank") {
        let v = args.get(at + 1).and_then(|v| v.parse::<u32>().ok()).expect("a number after the flag");
        if args[at] == "--tokens" {
            tokens = v as usize;
        } else {
            ranks.push(v);
        }
        args.drain(at..at + 2);
    }
    if ranks.is_empty() {
        ranks.push(7);
    }
    if args.len() < 2 {
        eprintln!("usage: route_tail_kld <model.gguf> <text.txt> [--tokens N] [--rank R]...");
        std::process::exit(2);
    }
    let text = std::fs::read_to_string(&args[1]).expect("text file").replace("\r\n", "\n");
    let texts: Vec<&str> = text.split("\n\n").map(str::trim).filter(|t| !t.is_empty()).collect();
    for rank in ranks {
        let start = std::time::Instant::now();
        match route_tail_divergence(Path::new(&args[0]), &texts, tokens, rank) {
            Ok((kl, agree)) => println!(
                "rank {rank}: {kl:.5} nats/token, top-1 agreement {:.1}%  ({:.0} s)",
                agree * 100.0,
                start.elapsed().as_secs_f32()
            ),
            Err(e) => {
                eprintln!("rank {rank}: {e}");
                std::process::exit(1);
            }
        }
    }
}
