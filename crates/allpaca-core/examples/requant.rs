//! Requantise a GGUF to mixed precision under an average-bits budget:
//! GSQ makes the 2-bit option, RCO chooses a type per weight matrix.
//!
//!     cargo run --release -p allpaca-core --example requant -- <in.gguf> <out.gguf> [avg bits] [GSQ steps] [calibration.txt]
//!
//! Several budgets (`3.0,4.0`) are written from one pass over the model, to
//! `<out>-3.00.gguf` and `<out>-4.00.gguf`.
//!
//! With a calibration text the model reads it first on the CPU (every weight
//! decoded to F32 in RAM) and each matrix is quantised for its output on the
//! inputs it got. Paragraphs (split at blank lines) are read separately.
//!
//! `--probe [N]` (needs the calibration text) also chooses the types by
//! measured sensitivity: each matrix is swapped for each of its options in
//! turn and the text reread (N tokens per paragraph, default 64), which
//! costs three reads of the text per matrix.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use allpaca_core::gsq::GsqConfig;
use allpaca_core::requant::{requantise_each, Options, ProbeTexts};

static STARTED: OnceLock<Instant> = OnceLock::new();

fn progress(done: usize, total: usize, name: &str) {
    let secs = STARTED.get_or_init(Instant::now).elapsed().as_secs_f64();
    eprintln!("[{done}/{total}] {name}  {:.0} s, about {:.0} min left", secs, secs / done as f64 * (total - done) as f64 / 60.0);
}

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
    let mut opts = Options {
        progress: Some(progress),
        ..Options::default()
    };
    let budgets: Vec<f64> = match args.get(2) {
        Some(bits) => bits.split(',').map(|b| b.parse().expect("avg bits")).collect(),
        None => vec![opts.avg_bits],
    };
    let out = Path::new(&args[1]);
    let outputs: Vec<(PathBuf, f64)> = budgets
        .iter()
        .map(|&bits| match budgets.len() {
            1 => (out.to_path_buf(), bits),
            _ => {
                let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("requant");
                (out.with_file_name(format!("{stem}-{bits:.2}.gguf")), bits)
            }
        })
        .collect();
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
    STARTED.get_or_init(Instant::now);
    let each: Vec<(&Path, f64)> = outputs.iter().map(|(path, bits)| (path.as_path(), *bits)).collect();
    let reports = match requantise_each(Path::new(&args[0]), &each, &opts) {
        Ok(reports) => reports,
        Err(e) => {
            eprintln!("requant failed: {e}");
            std::process::exit(1);
        }
    };
    for (report, (path, _)) in reports.iter().zip(&outputs) {
        println!("== {}", path.display());
        {
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
    }
}
