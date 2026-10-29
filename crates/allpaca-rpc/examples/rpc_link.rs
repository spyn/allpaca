//! Round trips and bulk throughput to RPC workers, one line per figure.
//!
//!   cargo run --release -p allpaca-rpc --example rpc_link -- [--calls N] [--bulk-mib M] HOST:PORT...
//!
//! `ALLPACA_RPC_TOKEN` is sent in HELLO when set.

use allpaca_rpc::{link::measure_link, RpcSession};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut calls = 200usize;
    let mut bulk_mib = 64u64;
    let mut endpoints = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--calls" => calls = args.next().ok_or("--calls needs a value")?.parse()?,
            "--bulk-mib" => bulk_mib = args.next().ok_or("--bulk-mib needs a value")?.parse()?,
            _ => endpoints.push(arg),
        }
    }
    if endpoints.is_empty() || calls == 0 {
        return Err("usage: rpc_link [--calls N] [--bulk-mib M] HOST:PORT...".into());
    }
    let token = std::env::var("ALLPACA_RPC_TOKEN").unwrap_or_default();
    for endpoint in &endpoints {
        let mut session = RpcSession::connect_addr_with_token(endpoint, &token)?;
        let report = measure_link(&mut session, calls, bulk_mib << 20)?;
        for r in &report.round_trips {
            println!(
                "{endpoint}  {:<12} median {:>8.1} us  p90 {:>8.1} us  ({} calls)",
                r.label, r.median_us, r.p90_us, r.calls
            );
        }
        println!(
            "{endpoint}  bulk {bulk_mib} MiB  h2d {:.1} MB/s  d2h {:.1} MB/s",
            report.h2d_bytes_per_s / 1e6,
            report.d2h_bytes_per_s / 1e6
        );
    }
    Ok(())
}
