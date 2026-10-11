//! Print the token ids Allpaca assigns to a text file, as `[1, 2, 3]`, for
//! comparison with `llama-tokenize --ids`.
//!
//! `cargo run -p allpaca-gguf --example tokenize -- <model.gguf> <text file>`

use allpaca_gguf::{extract_tokenizer, GgufFile};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: tokenize <model.gguf> <text file>");
        std::process::exit(2);
    }
    let file = GgufFile::open(Path::new(&args[1])).expect("open gguf");
    let tokenizer = extract_tokenizer(&file).expect("tokenizer");
    let text = std::fs::read_to_string(&args[2]).expect("read text");
    let ids = tokenizer.inner.encode(text, false).expect("encode");
    println!("{:?}", ids.get_ids());
}
