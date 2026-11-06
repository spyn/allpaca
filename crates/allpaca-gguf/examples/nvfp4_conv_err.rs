//! Per-tensor weight error of NVFP4 and of its ROCmFP4 conversion against the
//! original (BF16) weights, one tensor at a time to keep memory small.
//!
//!     cargo run --release -p allpaca-gguf --example nvfp4_conv_err -- \
//!         <orig.gguf> <nvfp4.gguf> <converted.gguf> [layer-step]
//!
//! Columns: relative RMS error (rms(a-ref)/rms(ref)) of the NVFP4 file, of the
//! converted ROCmFP4 file, of the device re-pack used by the native kernel,
//! and the NVFP4 -> converted error (how far the conversion moved the weights).
use std::path::Path;

fn rel(a: &[f32], r: &[f32]) -> f64 {
    let (mut e, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(r) {
        let d = f64::from(*x) - f64::from(*y);
        e += d * d;
        n += f64::from(*y) * f64::from(*y);
    }
    (e / n.max(1e-300)).sqrt()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let step: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let orig = allpaca_gguf::GgufFile::open(Path::new(&a[0])).unwrap();
    let nv = allpaca_gguf::GgufFile::open(Path::new(&a[1])).unwrap();
    let cv = allpaca_gguf::GgufFile::open(Path::new(&a[2])).unwrap();
    println!("{:28} {:>9} {:>9} {:>9} {:>9}", "tensor", "nvfp4", "converted", "dev-pack", "nv->conv");
    let mut sums = [0f64; 4];
    let mut count = 0f64;
    for t in &nv.tensors {
        if t.ggml_type != allpaca_gguf::GgmlType::Nvfp4 {
            continue;
        }
        let layer: usize = t.name.split('.').nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        if layer % step != 0 {
            continue;
        }
        let r = orig.tensor_f32(&t.name).unwrap();
        let n = nv.tensor_f32(&t.name).unwrap();
        let c = cv.tensor_f32(&t.name).unwrap();
        let ts = nv.nvfp4_tensor_scale(&t.name).unwrap().unwrap_or(1.0);
        let dev = allpaca_cpu::repack_nvfp4_dev(nv.tensor_bytes(&t.name).unwrap(), n.len(), ts).unwrap();
        let mut d = vec![0f32; n.len()];
        allpaca_cpu::dequant_nvfp4_dev(&dev, n.len(), &mut d);
        let row = [rel(&n, &r), rel(&c, &r), rel(&d, &r), rel(&c, &n)];
        println!("{:28} {:9.4} {:9.4} {:9.4} {:9.4}", t.name, row[0], row[1], row[2], row[3]);
        for (s, v) in sums.iter_mut().zip(row) {
            *s += v;
        }
        count += 1.0;
    }
    println!(
        "{:28} {:9.4} {:9.4} {:9.4} {:9.4}  ({count} tensors)",
        "mean",
        sums[0] / count,
        sums[1] / count,
        sums[2] / count,
        sums[3] / count
    );
}
