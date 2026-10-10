//! Requantise a GGUF to mixed precision under an average-bits budget:
//! GSQ makes the 2-bit option, RCO chooses a type per weight matrix.
//!
//!     cargo run --release -p allpaca-core --example requant -- <in.gguf> <out.gguf> [avg bits] [GSQ steps] [calibration.txt]
//!
//! With a calibration text the model reads it first on the CPU (every weight
//! decoded to F32 in RAM) and each matrix is quantised for its output on the
//! inputs it got. Paragraphs (split at blank lines) are read separately.
//!
//! `--probe [N]` (needs the calibration text) also chooses the types by
//! measured sensitivity: each matrix is swapped for each of its options in
//! turn and the text reread (N tokens per paragraph, default 64), which
//! costs three reads of the text per matrix.

use std::path::Path;

use allpaca_core::gsq::GsqConfig;
use allpaca_core::requant::{requantise, Options, ProbeTexts};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut probe_tokens = None;
    if let Some(at) = args.iter().position(|a| a == "--probe") {
        let given = args.get(at + 1).and_then(|v| v.parse::<usize>().ok());
        probe_tokens = Some(given.unwrap_or(64));
        args.drain(at..at + 1 + usize::from(given.is_some()));
    }
    if args.len() < 2 {
        eprintln!(
            "usage: requant <in.gguf> <out.gguf> [avg bits, default 3.0] [GSQ steps, default 300] [calibration.txt] [--probe [tokens]]"
        );
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
    if let Some(path) = args.get(4) {
        let text = std::fs::read_to_string(path).expect("calibration text");
        let texts: Vec<&str> = text.split("\n\n").map(str::trim).filter(|t| !t.is_empty()).collect();
        if let Some(max_tokens) = probe_tokens {
            opts.probe = Some(ProbeTexts { texts: texts.iter().map(|t| t.to_string()).collect(), max_tokens });
        }
        let start = std::time::Instant::now();
        match allpaca_core::calib::capture(Path::new(&args[0]), &texts) {
            Ok(stats) => {
                println!(
                    "calibration: {} texts, {} input sites measured in {:.0} s",
                    texts.len(),
                    stats.len(),
                    start.elapsed().as_secs_f32()
                );
                opts.stats = Some(stats);
            }
            Err(e) => {
                eprintln!("calibration failed: {e}");
                std::process::exit(1);
            }
        }
    }
    if probe_tokens.is_some() && opts.probe.is_none() {
        eprintln!("--probe needs a calibration text");
        std::process::exit(2);
    }
    match requantise(Path::new(&args[0]), Path::new(&args[1]), &opts) {
        Ok(report) => {
            for c in &report.choices {
                println!(
                    "{:<44} {:>10} weights -> {:?}  ({} error Q2_0 {:.4}  Q4_0 {:.4}  Q8_0 {:.6})",
                    c.name,
                    c.weights,
                    c.target,
                    if c.calibrated { "output" } else { "weight" },
                    c.relative_error[0],
                    c.relative_error[1],
                    c.relative_error[2]
                );
                if let Some(d) = c.divergence {
                    println!("{:<44} divergence Q2_0 {:.5}  Q4_0 {:.5}  Q8_0 {:.6}", "", d[0], d[1], d[2]);
                }
            }
            println!(
                "{} matrices requantised to {:.3} bits per weight on average ({} on measured inputs), {} tensors copied",
                report.choices.len(),
                report.avg_bits(),
                report.calibrated,
                report.copied
            );
        }
        Err(e) => {
            eprintln!("requant failed: {e}");
            std::process::exit(1);
        }
    }
}
