//! How far a requantised GGUF's predictions are from the original's: mean
//! KL divergence (nats per token) over a text, both models on the CPU.
//!
//!     cargo run --release -p allpaca-core --example quant_kld -- <reference.gguf> <text.txt> <other.gguf>... [--tokens N]
//!
//! Paragraphs of the text (split at blank lines) are read separately, at
//! most N tokens of each (default 256).

use std::path::Path;

use allpaca_core::sensitivity::divergence_between;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut tokens = 256usize;
    if let Some(at) = args.iter().position(|a| a == "--tokens") {
        tokens = args.get(at + 1).and_then(|v| v.parse().ok()).expect("--tokens N");
        args.drain(at..at + 2);
    }
    if args.len() < 3 {
        eprintln!("usage: quant_kld <reference.gguf> <text.txt> <other.gguf>... [--tokens N]");
        std::process::exit(2);
    }
    let text = std::fs::read_to_string(&args[1]).expect("text file");
    let texts: Vec<&str> = text.split("\n\n").map(str::trim).filter(|t| !t.is_empty()).collect();
    for other in &args[2..] {
        let start = std::time::Instant::now();
        match divergence_between(Path::new(&args[0]), Path::new(other), &texts, tokens) {
            Ok(d) => println!("{d:.5} nats/token  {other}  ({:.0} s)", start.elapsed().as_secs_f32()),
            Err(e) => {
                eprintln!("{other}: {e}");
                std::process::exit(1);
            }
        }
    }
}
