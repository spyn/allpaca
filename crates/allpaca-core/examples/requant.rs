//! Requantise a GGUF to mixed precision under an average-bits budget:
//! GSQ makes the 2-bit option, RCO chooses a type per weight matrix.
//!
//!     cargo run --release -p allpaca-core --example requant -- <in.gguf> <out.gguf> [avg bits] [GSQ steps]

use std::path::Path;

use allpaca_core::gsq::GsqConfig;
use allpaca_core::requant::{requantise, Options};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: requant <in.gguf> <out.gguf> [avg bits, default 3.0] [GSQ steps, default 300]");
        std::process::exit(2);
    }
    let mut opts = Options::default();
    if let Some(bits) = args.get(2) {
        opts.avg_bits = bits.parse().expect("avg bits");
    }
    if let Some(steps) = args.get(3) {
        opts.gsq = GsqConfig {
            steps: steps.parse().expect("GSQ steps"),
            ..opts.gsq
        };
    }
    match requantise(Path::new(&args[0]), Path::new(&args[1]), &opts) {
        Ok(report) => {
            for c in &report.choices {
                println!(
                    "{:<44} {:>10} weights -> {:?}  (relative error Q2_0 {:.4}  Q4_0 {:.4}  Q8_0 {:.6})",
                    c.name, c.weights, c.target, c.relative_error[0], c.relative_error[1], c.relative_error[2]
                );
            }
            println!(
                "{} matrices requantised to {:.3} bits per weight on average, {} tensors copied",
                report.choices.len(),
                report.avg_bits(),
                report.copied
            );
        }
        Err(e) => {
            eprintln!("requant failed: {e}");
            std::process::exit(1);
        }
    }
}
