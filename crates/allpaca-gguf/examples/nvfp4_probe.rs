//! Print NVFP4 tensors, their `.scale` values and a few dequantised weights.
use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("gguf path");
    let f = allpaca_gguf::GgufFile::open(Path::new(&path)).unwrap();
    let (mut n40, mut min_s, mut max_s) = (0, f32::MAX, f32::MIN);
    for t in &f.tensors {
        if t.ggml_type != allpaca_gguf::GgmlType::Nvfp4 {
            continue;
        }
        n40 += 1;
        let s = f.nvfp4_tensor_scale(&t.name).unwrap();
        if let Some(s) = s {
            min_s = min_s.min(s);
            max_s = max_s.max(s);
        }
        if n40 <= 3 {
            let v = f.tensor_f32(&t.name).unwrap();
            let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
            println!("{} dims={:?} scale={s:?} rms={rms:e} first={:?}", t.name, t.dims, &v[..4]);
        }
    }
    println!("nvfp4 tensors={n40} scale range {min_s:e}..{max_s:e}");
    println!("scale tensors:");
    for t in f.tensors.iter().filter(|t| t.name.ends_with("scale")).take(6) {
        println!("  {} {:?}", t.name, t.dims);
    }
}
