//! Gated device-timed BF16 NT GEMM vs the Triton gfx906 sidecar.
//!
//! Off unless `ALLPACA_SIDECAR_GEMM=1`. Writes JSON to stdout (`SIDECAR_JSON:`)
//! and to `ALLPACA_SIDECAR_OUT` when that path is set.
//!
//! Layout matches `gemm_nt_bf16`: `C[m,n] = A[m,k]_f32 * dequant(B[n,k]_bf16)^T`.
//! Shapes are DSV4-Flash shexp (ff=2048, dim=4096). Not a generate() path.

use std::time::Instant;

use allpaca_hip::{
    device_count, device_free, device_malloc, gemm_nt_bf16_device, memcpy, set_device,
    stream_synchronize, MEMCPY_H2D,
};

const WARMUP: u32 = 5;
const ITERS: u32 = 20;

/// DSV4-Flash shexp: gate/up N=2048 K=4096; down N=4096 K=2048.
const SHAPES: [(usize, usize, usize); 5] = [
    (1, 2048, 4096),
    (32, 2048, 4096),
    (40, 2048, 4096),
    (1, 4096, 2048),
    (32, 4096, 2048),
];

struct DevBuf {
    ptr: *mut u8,
    bytes: usize,
}

impl DevBuf {
    fn alloc(bytes: usize) -> Self {
        let ptr = device_malloc(bytes, 0).unwrap_or_else(|e| panic!("hipMalloc({bytes}): {e}"));
        Self { ptr, bytes }
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        device_free(self.ptr, self.bytes);
    }
}

struct ShapeRow {
    m: usize,
    n: usize,
    k: usize,
    us_mean: f64,
    gflops: f64,
}

fn fill_host(m: usize, n: usize, k: usize, seed: usize) -> (Vec<f32>, Vec<u16>) {
    let mut a = vec![0.0f32; m * k];
    let mut b_f = vec![0.0f32; n * k];
    for (i, v) in a.iter_mut().enumerate() {
        *v = ((i * 17 + seed * 3) % 50) as f32 * 0.02 - 0.5;
    }
    for (i, v) in b_f.iter_mut().enumerate() {
        *v = ((i * 13 + seed * 7) % 50) as f32 * 0.02 - 0.5;
    }
    let b: Vec<u16> = b_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
    (a, b)
}

fn bench_shape(m: usize, n: usize, k: usize, seed: usize) -> ShapeRow {
    set_device(0).expect("set_device(0)");
    let (a, b) = fill_host(m, n, k, seed);
    let a_bytes = a.len() * 4;
    let b_bytes = b.len() * 2;
    let c_bytes = m * n * 4;
    let da = DevBuf::alloc(a_bytes);
    let db = DevBuf::alloc(b_bytes);
    let dc = DevBuf::alloc(c_bytes);
    memcpy(da.ptr, a.as_ptr().cast(), a_bytes, MEMCPY_H2D).expect("H2D A");
    memcpy(db.ptr, b.as_ptr().cast(), b_bytes, MEMCPY_H2D).expect("H2D B");
    let launch = || {
        gemm_nt_bf16_device(
            da.ptr.cast(),
            db.ptr.cast(),
            dc.ptr.cast(),
            m as i32,
            n as i32,
            k as i32,
        )
        .expect("gemm_nt_bf16_device");
    };
    for _ in 0..WARMUP {
        launch();
    }
    stream_synchronize().expect("warmup sync");
    let t0 = Instant::now();
    for _ in 0..ITERS {
        launch();
    }
    stream_synchronize().expect("timed sync");
    let us_mean = t0.elapsed().as_secs_f64() * 1e6 / f64::from(ITERS);
    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let gflops = flops / (us_mean * 1e-6) / 1e9;
    ShapeRow {
        m,
        n,
        k,
        us_mean,
        gflops,
    }
}

fn json_payload(rows: &[ShapeRow]) -> String {
    let mut shapes = String::new();
    for (i, r) in rows.iter().enumerate() {
        if i > 0 {
            shapes.push(',');
        }
        shapes.push_str(&format!(
            "{{\"m\":{},\"n\":{},\"k\":{},\"us_mean\":{:.3},\"gflops\":{:.2}}}",
            r.m, r.n, r.k, r.us_mean, r.gflops
        ));
    }
    format!(
        "{{\"backend\":\"hip\",\"kernel\":\"gemm_nt_bf16\",\"warmup\":{WARMUP},\"iters\":{ITERS},\"shapes\":[{shapes}]}}"
    )
}

#[test]
fn sidecar_bf16_nt_gemm_device_timed_json() {
    if std::env::var("ALLPACA_SIDECAR_GEMM").ok().as_deref() != Some("1") {
        return;
    }
    if device_count() == 0 {
        eprintln!("sidecar_gemm: no HIP device");
        return;
    }
    let mut rows = Vec::with_capacity(SHAPES.len());
    for (seed, (m, n, k)) in SHAPES.iter().copied().enumerate() {
        let row = bench_shape(m, n, k, seed);
        eprintln!(
            "hip gemm_nt_bf16 {m}x{n}x{k}  {:.1} us  {:.1} GFLOP/s",
            row.us_mean, row.gflops
        );
        rows.push(row);
    }
    let payload = json_payload(&rows);
    println!("SIDECAR_JSON:{payload}");
    if let Ok(path) = std::env::var("ALLPACA_SIDECAR_OUT") {
        if let Some(dir) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("mkdir {dir:?}: {e}"));
        }
        std::fs::write(&path, format!("{payload}\n"))
            .unwrap_or_else(|e| panic!("write {path}: {e}"));
    }
}
