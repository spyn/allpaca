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
//! costs a read of the text per matrix and type. With `--scaled` it reads
//! the text once per matrix, at its fewest bits, and scales that by the
//! matrix's own error for the other types.
//!
//! `--propagate [ridge]` (needs the calibration text) quantises the blocks
//! in order once the types are chosen, each matrix for the original outputs
//! on the inputs the quantised matrices before it produce. `ridge` (default
//! 1.0) holds a matrix to its own weights: lower fits the text more closely.
//!
//! `--probe-table <file>` saves the probe's measurements to the file, and
//! when the file exists, reads them from it instead of probing.
//!
//! `--types q2_0,q3_k,q4_k,q5_k,q6_k,q8_0` limits the types a matrix may
//! take (default: all of those and `q4_0`).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use allpaca_core::gsq::GsqConfig;
use allpaca_core::requant::{requantise_each, Options, ProbeTexts, Target};

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
    let mut ridge = None;
    if let Some(at) = args.iter().position(|a| a == "--propagate") {
        let given = args.get(at + 1).and_then(|v| v.parse::<f32>().ok());
        ridge = Some(given.unwrap_or(1.0));
        args.drain(at..at + 1 + usize::from(given.is_some()));
    }
    let scaled = args.iter().any(|a| a == "--scaled");
    args.retain(|a| a != "--scaled");
    let mut probe_table = None;
    if let Some(at) = args.iter().position(|a| a == "--probe-table") {
        match args.get(at + 1) {
            Some(path) => probe_table = Some(PathBuf::from(path)),
            None => {
                eprintln!("--probe-table takes a file");
                std::process::exit(2);
            }
        }
        args.drain(at..at + 2);
    }
    let mut types = None;
    if let Some(at) = args.iter().position(|a| a == "--types") {
        let list = args.get(at + 1).cloned().unwrap_or_default();
        let named: Option<Vec<Target>> = list.split(',').map(|n| Target::named(n.trim())).collect();
        match named.filter(|t| !t.is_empty()) {
            Some(named) => types = Some(named),
            None => {
                eprintln!("--types takes a list such as q2_0,q4_k,q6_k (got {list:?})");
                std::process::exit(2);
            }
        }
        args.drain(at..(at + 2).min(args.len()));
    }
    if args.len() < 2 {
        eprintln!(
            "usage: requant <in.gguf> <out.gguf> [avg bits, default 3.0] [GSQ steps, default 300] [calibration.txt] [--probe [tokens]] [--scaled] [--probe-table file] [--propagate [ridge]] [--types q2_0,q4_k,...]"
        );
        std::process::exit(2);
    }
    let mut opts = Options {
        progress: Some(progress),
        ..Options::default()
    };
    opts.probe_table = probe_table;
    if let Some(types) = types {
        opts.targets = types;
    }
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
        let text = std::fs::read_to_string(path).expect("calibration text").replace("\r\n", "\n");
        let texts: Vec<&str> = text.split("\n\n").map(str::trim).filter(|t| !t.is_empty()).collect();
        if let Some(max_tokens) = probe_tokens {
            opts.probe = Some(ProbeTexts { texts: texts.iter().map(|t| t.to_string()).collect(), max_tokens, scaled });
        }
        if let Some(ridge) = ridge {
            opts.propagate = Some(allpaca_core::propagate::Settings {
                texts: texts.iter().map(|t| t.to_string()).collect(),
                max_tokens: 4096,
                ridge,
            });
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
    if (probe_tokens.is_some() && opts.probe.is_none()) || (ridge.is_some() && opts.propagate.is_none()) {
        eprintln!("--probe and --propagate need a calibration text");
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
            // A type a matrix's rows do not fit has no number.
            let table = |values: &[f64]| -> String {
                report
                    .targets
                    .iter()
                    .zip(values)
                    .filter(|(_, v)| !v.is_nan())
                    .map(|(t, v)| format!("{} {v:.5}", t.name()))
                    .collect::<Vec<_>>()
                    .join("  ")
            };
            for c in &report.choices {
                println!(
                    "{:<44} {:>10} weights -> {}  ({} error {})",
                    c.name,
                    c.weights,
                    c.target.name(),
                    if c.calibrated { "output" } else { "weight" },
                    table(&c.relative_error)
                );
                if let Some(d) = &c.divergence {
                    println!("{:<44} divergence {}", "", table(d));
                }
            }
            println!(
                "{} matrices requantised to {:.3} bits per weight on average ({} on measured inputs, {} in order), {} tensors copied",
                report.choices.len(),
                report.avg_bits(),
                report.calibrated,
                report.propagated,
                report.copied
            );
        }
    }
}
