//! Unbuffered read throughput of one file at several thread counts and block
//! sizes: what the drive can deliver, to compare with what the DSV4 expert
//! tier gets (board task P18). Windows only (FILE_FLAG_NO_BUFFERING).
//!
//!     cargo run --release -p allpaca-core --example ssd_ceiling -- <file> [seconds-per-config]
#[cfg(windows)]
fn main() {
    use std::io::{Read, Seek, SeekFrom};
    use std::os::windows::fs::OpenOptionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    const NO_BUFFERING: u32 = 0x2000_0000;
    let path = std::env::args().nth(1).expect("file");
    let secs: u64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let len = std::fs::metadata(&path).expect("metadata").len();
    println!("{path}: {:.1} GB; {secs} s per row; unbuffered random aligned reads", len as f64 / 1e9);
    println!("{:>8} {:>10} {:>10} {:>10}", "threads", "block", "MB/s", "reads/s");
    for &block in &[512 * 1024usize, 4 * 1024 * 1024] {
        for &threads in &[1usize, 2, 4, 6, 8, 16, 32] {
            let bytes = AtomicU64::new(0);
            let reads = AtomicU64::new(0);
            let stop = Instant::now() + Duration::from_secs(secs);
            std::thread::scope(|s| {
                for t in 0..threads {
                    let (bytes, reads, path) = (&bytes, &reads, &path);
                    s.spawn(move || {
                        let mut f = std::fs::OpenOptions::new()
                            .read(true)
                            .custom_flags(NO_BUFFERING)
                            .open(path)
                            .expect("open unbuffered");
                        // 4 KiB aligned buffer.
                        let layout = std::alloc::Layout::from_size_align(block, 4096).unwrap();
                        let ptr = unsafe { std::alloc::alloc(layout) };
                        let buf = unsafe { std::slice::from_raw_parts_mut(ptr, block) };
                        let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ (t as u64 + 1) * 0xD1B5_4A32_D192_ED03;
                        while Instant::now() < stop {
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            let off = (x % (len / block as u64 - 1)) * block as u64;
                            f.seek(SeekFrom::Start(off)).unwrap();
                            let n = f.read(buf).unwrap();
                            bytes.fetch_add(n as u64, Ordering::Relaxed);
                            reads.fetch_add(1, Ordering::Relaxed);
                        }
                        unsafe { std::alloc::dealloc(ptr, layout) };
                    });
                }
            });
            let b = bytes.load(Ordering::Relaxed) as f64;
            println!(
                "{threads:>8} {:>9}K {:>10.0} {:>10.0}",
                block / 1024,
                b / secs as f64 / 1e6,
                reads.load(Ordering::Relaxed) as f64 / secs as f64
            );
        }
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("ssd_ceiling is Windows only");
}
