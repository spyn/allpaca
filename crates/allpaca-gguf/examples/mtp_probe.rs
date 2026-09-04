use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("gguf path");
    let f = allpaca_gguf::GgufFile::open(Path::new(&path)).unwrap();
    let nextn = f.meta_u32("qwen35.nextn_predict_layers").unwrap_or(0);
    let blocks = f.meta_u32("qwen35.block_count").unwrap_or(0);
    println!("block_count={blocks} nextn={nextn}");
    let start = blocks.saturating_sub(nextn);
    for t in &f.tensors {
        if let Some(rest) = t.name.strip_prefix("blk.") {
            let n: u32 = rest
                .split('.')
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if n >= start {
                println!("{} dims={:?} ty={}", t.name, t.dims, t.ggml_type as i32);
            }
        } else if t.name.contains("nextn") || t.name.contains("mtp") {
            println!("META_T {} dims={:?}", t.name, t.dims);
        }
    }
}
