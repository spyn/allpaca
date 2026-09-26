//! Decode GEMV achieved-bandwidth probe at Qwen3.8-27B layer shapes.
//!
//! Opt-in (it only prints numbers):
//! `cargo test -p allpaca-hip --features hip --release --test gemv_bandwidth -- --ignored --nocapture`
//!
//! Weight bytes are filler (bandwidth, not numerics). Tensors below ~64 MiB
//! can sit in RDNA Infinity Cache across iterations and read above VRAM
//! bandwidth; a GPU that also drives a desktop reports noisy numbers.
#[cfg(feature = "hip")]
#[test]
#[ignore]
fn gemv_bandwidth_at_qwen_shapes() {
    use std::time::Instant;
    if allpaca_hip::device_count() == 0 {
        return;
    }
    // (name, block bytes as staged on device, elements per block)
    let kinds: [(&str, usize, usize); 4] =
        [("q4_k", 144, 256), ("q5_k", 176, 256), ("q6_k", 210, 256), ("q8_0", 36, 32)];
    // FFN up/gate, FFN down, fused GDN qkv.
    let shapes = [(17408usize, 5120usize), (5120, 17408), (10240, 5120)];
    for dev in 0..allpaca_hip::device_count() as i32 {
        allpaca_hip::set_device(dev).unwrap();
        for (name, block_bytes, block_elems) in kinds {
            for (n, k) in shapes {
                for m in [1usize, 4] {
                    let b_bytes = n * (k / block_elems) * block_bytes;
                    let b = allpaca_hip::device_malloc(b_bytes, dev).unwrap();
                    let filler: Vec<u8> = (0..b_bytes).map(|i| (i * 131 % 251) as u8 & 0x3f).collect();
                    allpaca_hip::memcpy(b, filler.as_ptr(), b_bytes, allpaca_hip::MEMCPY_H2D).unwrap();
                    let a = allpaca_hip::device_malloc(m * k * 4, dev).unwrap() as *const f32;
                    let c = allpaca_hip::device_malloc(m * n * 4, dev).unwrap() as *mut f32;
                    let (mi, ni, ki) = (m as i32, n as i32, k as i32);
                    let run = || match name {
                        "q4_k" => allpaca_hip::gemm_nt_q4_k_device(a, b, c, mi, ni, ki),
                        "q5_k" => allpaca_hip::gemm_nt_q5_k_device(a, b, c, mi, ni, ki),
                        "q6_k" => allpaca_hip::gemm_nt_q6_k_device(a, b, c, mi, ni, ki),
                        _ => allpaca_hip::gemm_nt_q8_0_device(a, b, c, mi, ni, ki),
                    };
                    for _ in 0..5 {
                        run().unwrap();
                    }
                    allpaca_hip::stream_synchronize().unwrap();
                    let iters = 50;
                    let t = Instant::now();
                    for _ in 0..iters {
                        run().unwrap();
                    }
                    allpaca_hip::stream_synchronize().unwrap();
                    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
                    eprintln!(
                        "hip{dev} {name} m={m} n={n} k={k}: {us:8.1} us {:7.1} GB/s",
                        b_bytes as f64 / us / 1e3
                    );
                    allpaca_hip::device_free(b, b_bytes);
                    allpaca_hip::device_free(a as *mut u8, m * k * 4);
                    allpaca_hip::device_free(c as *mut u8, m * n * 4);
                }
            }
        }
    }
}
