//! What latent sparse attention keeps of a model's real attention (board
//! P20): for dumped queries and keys of each full-attention layer, the share
//! of the full attention mass the selected keys hold and, when values were
//! dumped, the relative error of the attention output.
//!
//!     ALLPACA_LSA_DUMP=<dir> ALLPACA_LSA_DUMP_V=1 allpaca-server ...   (run one long request)
//!     cargo run --release -p allpaca-core --example lsa_probe -- <dir> \
//!         --map dlsa=<maps.safetensors> [--map pca=<maps.safetensors>] \
//!         [--n-rot 64] [--k 512 --k 1024 ...] [--bins 8192,32768,131072] [--json out.jsonl]
//!
//! A query is counted in the bin of the first `--bins` edge at or above its
//! position (the keys it sees), and only once it reads fewer keys than all
//! (position + 1 > sink + k + window). Rows per bin, layer, map and budget:
//! queries, mean / 10th percentile / minimum kept mass (averaged over the
//! heads), the worst head's mean kept mass, and mean / 90th percentile
//! relative output error. Two baselines share the budget: `oracle` reads the
//! k keys with the most head-averaged mass, `recent` the k keys before the
//! window.
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use allpaca_core::lsa::{LsaGeometry, LsaIndex, LsaMap, LsaShape};

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// Run `f` over `0..n` on every core, results in order.
fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).min(n.max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<(usize, T)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= n {
                            break local;
                        }
                        local.push((i, f(i)));
                    }
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("worker")).collect()
    });
    out.sort_by_key(|(i, _)| *i);
    out.into_iter().map(|(_, t)| t).collect()
}

/// One query's result for one selector and budget.
#[derive(Clone, Copy, Default)]
struct Kept {
    /// Kept mass averaged over the heads.
    mass: f32,
    /// The head that keeps least.
    worst_head: f32,
    /// Relative L2 error of the attention output, NaN without values.
    out_err: f32,
    /// Keys read.
    keys: u32,
}

struct Layer<'a> {
    geo: LsaGeometry,
    k: &'a [f32],
    v: Option<&'a [f32]>,
}

impl Layer<'_> {
    /// Per-head attention probabilities of `q` over keys `0..=pos`,
    /// `n_heads * (pos + 1)` floats.
    fn probs(&self, q: &[f32], pos: usize) -> Vec<f32> {
        let g = &self.geo;
        let kv_dim = g.n_kv_heads * g.head_dim;
        let group = g.n_heads / g.n_kv_heads;
        let scale = 1.0 / (g.head_dim as f32).sqrt();
        let n = pos + 1;
        let mut p = vec![0.0f32; g.n_heads * n];
        for h in 0..g.n_heads {
            let qh = &q[h * g.head_dim..(h + 1) * g.head_dim];
            let row = &mut p[h * n..(h + 1) * n];
            let k_off = (h / group) * g.head_dim;
            let mut max = f32::NEG_INFINITY;
            for (t, s) in row.iter_mut().enumerate() {
                *s = dot(qh, &self.k[t * kv_dim + k_off..t * kv_dim + k_off + g.head_dim]) * scale;
                max = max.max(*s);
            }
            let mut sum = 0.0f64;
            for s in row.iter_mut() {
                *s = (*s - max).exp();
                sum += *s as f64;
            }
            let inv = (1.0 / sum) as f32;
            row.iter_mut().for_each(|s| *s *= inv);
        }
        p
    }

    /// The attention output over `keys` (all of them when `None`), each
    /// head's probabilities renormalised over the keys it reads.
    fn output(&self, p: &[f32], n: usize, keys: Option<&[u32]>) -> Vec<f32> {
        let g = &self.geo;
        let v = self.v.expect("values");
        let kv_dim = g.n_kv_heads * g.head_dim;
        let group = g.n_heads / g.n_kv_heads;
        let mut out = vec![0.0f32; g.n_heads * g.head_dim];
        for h in 0..g.n_heads {
            let row = &p[h * n..(h + 1) * n];
            let v_off = (h / group) * g.head_dim;
            let o = &mut out[h * g.head_dim..(h + 1) * g.head_dim];
            let mut add = |t: usize, w: f32| {
                for (o, x) in o.iter_mut().zip(&v[t * kv_dim + v_off..t * kv_dim + v_off + g.head_dim]) {
                    *o += w * x;
                }
            };
            match keys {
                None => (0..n).for_each(|t| add(t, row[t])),
                Some(keys) => {
                    let mass: f32 = keys.iter().map(|&t| row[t as usize]).sum();
                    let inv = 1.0 / mass.max(1e-30);
                    keys.iter().for_each(|&t| add(t as usize, row[t as usize] * inv));
                }
            }
        }
        out
    }

    fn kept(&self, p: &[f32], n: usize, keys: &[u32], full_out: Option<&[f32]>) -> Kept {
        let g = &self.geo;
        let per_head: Vec<f32> = (0..g.n_heads)
            .map(|h| keys.iter().map(|&t| p[h * n + t as usize]).sum::<f32>())
            .collect();
        let out_err = match full_out {
            None => f32::NAN,
            Some(full) => {
                let sparse = self.output(p, n, Some(keys));
                let num: f32 = full.iter().zip(&sparse).map(|(a, b)| (a - b) * (a - b)).sum();
                let den: f32 = full.iter().map(|a| a * a).sum();
                (num / den.max(1e-30)).sqrt()
            }
        };
        Kept {
            mass: per_head.iter().sum::<f32>() / g.n_heads as f32,
            worst_head: per_head.iter().copied().fold(f32::INFINITY, f32::min),
            out_err,
            keys: keys.len() as u32,
        }
    }
}

/// Sink, the `k` given long-range keys and the window of a query at `pos`.
fn with_sink_and_window(shape: &LsaShape, pos: usize, mut long: Vec<u32>) -> Vec<u32> {
    let window_start = (pos + 1).saturating_sub(shape.window);
    long.sort_unstable();
    let mut keys: Vec<u32> = (0..shape.sink.min(window_start) as u32).collect();
    keys.extend(long);
    keys.extend(window_start as u32..=pos as u32);
    keys
}

fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    sorted[((sorted.len() - 1) as f32 * q).round() as usize]
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut maps: Vec<(String, PathBuf)> = Vec::new();
    let mut budgets: Vec<usize> = Vec::new();
    let mut bins: Vec<usize> = vec![8192, 32768, 131072, usize::MAX];
    let mut n_rot = 64usize;
    let mut json: Option<PathBuf> = None;
    let mut max_queries = usize::MAX;
    while let Some(at) = args.iter().position(|a| a.starts_with("--")) {
        let flag = args[at].clone();
        let value = args.get(at + 1).cloned().unwrap_or_else(|| panic!("a value after {flag}"));
        match flag.as_str() {
            "--map" => {
                let (name, path) = value.split_once('=').expect("--map name=path");
                maps.push((name.to_string(), PathBuf::from(path)));
            }
            "--k" => budgets.push(value.parse().expect("--k N")),
            "--bins" => {
                bins = value.split(',').map(|b| b.parse().expect("--bins a,b,c")).collect();
                bins.push(usize::MAX);
            }
            "--n-rot" => n_rot = value.parse().expect("--n-rot N"),
            "--json" => json = Some(PathBuf::from(value)),
            "--max-queries" => max_queries = value.parse().expect("--max-queries N"),
            other => panic!("unknown flag {other}"),
        }
        args.drain(at..at + 2);
    }
    if args.len() != 1 || maps.is_empty() {
        eprintln!("usage: lsa_probe <dump dir> --map name=maps.safetensors [--map ...] [--k N]... [--bins a,b] [--n-rot N] [--json out.jsonl] [--max-queries N]");
        std::process::exit(2);
    }
    if budgets.is_empty() {
        budgets = vec![512, 1024, 2048, 4096];
    }
    let dir = PathBuf::from(&args[0]);
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).expect("meta.json")).expect("meta.json");
    let field = |name: &str| meta[name].as_u64().unwrap_or_else(|| panic!("meta.json: {name}")) as usize;
    let geo = LsaGeometry {
        n_heads: field("n_heads"),
        n_kv_heads: field("n_kv_heads"),
        head_dim: field("head_dim"),
        n_rot,
    };
    geo.check().expect("geometry");
    let kv_dim = geo.n_kv_heads * geo.head_dim;
    let q_dim = geo.n_heads * geo.head_dim;

    let mut layers: Vec<usize> = std::fs::read_dir(&dir)
        .expect("dump dir")
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            name.strip_prefix('L')?.strip_suffix(".k.f32")?.parse().ok()
        })
        .collect();
    layers.sort_unstable();
    let mut json_out = json.map(|p| std::fs::File::create(p).expect("json file"));
    // (bin, selector, k) -> every query's result, over all layers.
    let mut overall: BTreeMap<(usize, String, usize), Vec<Kept>> = BTreeMap::new();
    println!("| layer | keys <= | selector | k | queries | keys read | mass mean | mass p10 | mass min | worst head | out err mean | out err p90 |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for &layer_idx in &layers {
        let started = std::time::Instant::now();
        let k = read_f32(&dir.join(format!("L{layer_idx}.k.f32")));
        let v_path = dir.join(format!("L{layer_idx}.v.f32"));
        let v = v_path.exists().then(|| read_f32(&v_path));
        let q = read_f32(&dir.join(format!("L{layer_idx}.q.f32")));
        let qpos = read_u32(&dir.join(format!("L{layer_idx}.qpos.u32")));
        let tokens = k.len() / kv_dim;
        assert_eq!(q.len(), qpos.len() * q_dim, "layer {layer_idx}: query rows");
        let layer = Layer { geo, k: &k, v: v.as_deref() };

        // One index per map, built over every key.
        let indexes: Vec<(String, LsaMap, LsaIndex)> = maps
            .iter()
            .map(|(name, path)| {
                let map = LsaMap::load(path, layer_idx, geo.key_dim()).expect("map");
                let index_keys = par_map(tokens.div_ceil(1024), |chunk| {
                    let mut input = Vec::new();
                    let mut out = vec![0.0f32; map.index_dim];
                    let mut rows = Vec::new();
                    for t in chunk * 1024..((chunk + 1) * 1024).min(tokens) {
                        geo.key_input(&k[t * kv_dim..(t + 1) * kv_dim], &mut input);
                        map.index_key(&input, &mut out);
                        rows.extend_from_slice(&out);
                    }
                    rows
                })
                .concat();
                let mut index = LsaIndex::new(LsaShape::paper(4096), map.index_dim).expect("index");
                index.push(&index_keys).expect("push");
                (name.clone(), map, index)
            })
            .collect();

        let min_budget = *budgets.iter().min().expect("budgets");
        let wanted: Vec<usize> = (0..qpos.len())
            .filter(|&i| (qpos[i] as usize) < tokens && qpos[i] as usize + 1 > LsaShape::paper(min_budget).max_keys())
            .take(max_queries)
            .collect();
        // Per query: (selector, k) -> result.
        let results: Vec<Vec<(String, usize, Kept)>> = par_map(wanted.len(), |w| {
            let i = wanted[w];
            let pos = qpos[i] as usize;
            let n = pos + 1;
            let q_row = &q[i * q_dim..(i + 1) * q_dim];
            let p = layer.probs(q_row, pos);
            let full_out = layer.v.is_some().then(|| layer.output(&p, n, None));
            let mut avg = vec![0.0f32; n];
            for row in p.chunks(n) {
                avg.iter_mut().zip(row).for_each(|(a, r)| *a += r);
            }
            let mut out = Vec::new();
            let mut input = Vec::new();
            for &budget in &budgets {
                let shape = LsaShape::paper(budget);
                if n <= shape.max_keys() {
                    continue;
                }
                let window_start = n - shape.window;
                let mut ranked: Vec<u32> = (shape.sink as u32..window_start as u32).collect();
                ranked.sort_unstable_by(|a, b| avg[*b as usize].total_cmp(&avg[*a as usize]).then(a.cmp(b)));
                ranked.truncate(budget);
                let oracle = with_sink_and_window(&shape, pos, ranked);
                out.push(("oracle".to_string(), budget, layer.kept(&p, n, &oracle, full_out.as_deref())));
                let recent = with_sink_and_window(
                    &shape,
                    pos,
                    (window_start.saturating_sub(budget).max(shape.sink) as u32..window_start as u32).collect(),
                );
                out.push(("recent".to_string(), budget, layer.kept(&p, n, &recent, full_out.as_deref())));
                for (name, map, index) in &indexes {
                    let mut index_query = vec![0.0f32; map.index_dim];
                    geo.query_input(q_row, &mut input);
                    map.index_query(&input, &mut index_query);
                    let keys = index
                        .select_budget(pos, &index_query, shape.top_k, shape.top_p)
                        .expect("select")
                        .expect("sparse");
                    out.push((name.clone(), budget, layer.kept(&p, n, &keys, full_out.as_deref())));
                }
            }
            out
        });

        let mut rows: BTreeMap<(usize, String, usize), Vec<Kept>> = BTreeMap::new();
        for (w, per_query) in results.iter().enumerate() {
            let pos = qpos[wanted[w]] as usize;
            let bin = *bins.iter().find(|b| pos < **b).expect("a bin");
            for (name, budget, kept) in per_query {
                rows.entry((bin, name.clone(), *budget)).or_default().push(*kept);
                overall.entry((bin, name.clone(), *budget)).or_default().push(*kept);
            }
        }
        for ((bin, name, budget), kept) in &rows {
            print_row(&format!("{layer_idx}"), *bin, name, *budget, kept, json_out.as_mut());
        }
        eprintln!(
            "layer {layer_idx}: {tokens} keys, {} of {} queries sparse, {:.0} s",
            wanted.len(),
            qpos.len(),
            started.elapsed().as_secs_f32()
        );
    }
    for ((bin, name, budget), kept) in &overall {
        print_row("all", *bin, name, *budget, kept, json_out.as_mut());
    }
}

fn print_row(layer: &str, bin: usize, name: &str, budget: usize, kept: &[Kept], json: Option<&mut std::fs::File>) {
    let mean = |f: &dyn Fn(&Kept) -> f32| kept.iter().map(f).sum::<f32>() / kept.len() as f32;
    let mut mass: Vec<f32> = kept.iter().map(|k| k.mass).collect();
    mass.sort_by(f32::total_cmp);
    let mut err: Vec<f32> = kept.iter().map(|k| k.out_err).collect();
    err.sort_by(f32::total_cmp);
    let bin_label = if bin == usize::MAX { "more".to_string() } else { bin.to_string() };
    let (mass_mean, worst, err_mean, keys) = (
        mean(&|k| k.mass),
        mean(&|k| k.worst_head),
        mean(&|k| k.out_err),
        mean(&|k| k.keys as f32),
    );
    println!(
        "| {layer} | {bin_label} | {name} | {budget} | {} | {keys:.0} | {mass_mean:.4} | {:.4} | {:.4} | {worst:.4} | {err_mean:.4} | {:.4} |",
        kept.len(),
        percentile(&mass, 0.1),
        mass[0],
        percentile(&err, 0.9),
    );
    if let Some(file) = json {
        let line = serde_json::json!({
            "layer": layer, "keys_upto": bin_label, "selector": name, "k": budget, "queries": kept.len(),
            "keys_read": keys, "mass_mean": mass_mean, "mass_p10": percentile(&mass, 0.1), "mass_min": mass[0],
            "worst_head_mean": worst, "out_err_mean": err_mean, "out_err_p90": percentile(&err, 0.9),
        });
        writeln!(file, "{line}").expect("json line");
    }
}
