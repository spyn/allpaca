//! Link figures for one worker: request/reply round trips at a few payload
//! sizes, and bulk upload/download throughput. Used to size a new node
//! (board S1) before any model runs over it.

use crate::{RpcError, RpcSession};

/// Round trips of one kind of call.
#[derive(Debug, Clone, PartialEq)]
pub struct RoundTrip {
    /// What was sent: `"empty"`, `"h2d 64 B"`, `"d2h 8192 B"`, ...
    pub label: String,
    pub bytes: u64,
    pub calls: usize,
    pub median_us: f64,
    pub p90_us: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinkReport {
    pub round_trips: Vec<RoundTrip>,
    /// Bytes per second, `bulk_bytes` in one call each way.
    pub h2d_bytes_per_s: f64,
    pub d2h_bytes_per_s: f64,
}

/// Measure `calls` round trips of each kind, then one bulk upload and one
/// bulk download of `bulk_bytes`. Allocates one worker buffer and frees it.
pub fn measure_link(
    session: &mut RpcSession,
    calls: usize,
    bulk_bytes: u64,
) -> Result<LinkReport, RpcError> {
    const SMALL: [u64; 2] = [64, 8192];
    let buf = session.alloc(bulk_bytes.max(SMALL[1]))?;
    let result = (|| {
        let mut round_trips = vec![time_calls("empty", 0, calls, || {
            session.device_count().map(drop)
        })?];
        for bytes in SMALL {
            let data = vec![0x5au8; bytes as usize];
            round_trips.push(time_calls(&format!("h2d {bytes} B"), bytes, calls, || {
                session.h2d(buf, 0, &data)
            })?);
            round_trips.push(time_calls(&format!("d2h {bytes} B"), bytes, calls, || {
                session.d2h(buf, 0, bytes).map(drop)
            })?);
        }
        let bulk = vec![0xa5u8; bulk_bytes as usize];
        let t = std::time::Instant::now();
        session.h2d(buf, 0, &bulk)?;
        let h2d_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let back = session.d2h(buf, 0, bulk_bytes)?;
        let d2h_s = t.elapsed().as_secs_f64();
        if back != bulk {
            return Err(RpcError::Protocol(
                "bulk download differs from the upload".into(),
            ));
        }
        Ok(LinkReport {
            round_trips,
            h2d_bytes_per_s: bulk_bytes as f64 / h2d_s,
            d2h_bytes_per_s: bulk_bytes as f64 / d2h_s,
        })
    })();
    let freed = session.free(buf);
    let report = result?;
    freed?;
    Ok(report)
}

fn time_calls(
    label: &str,
    bytes: u64,
    calls: usize,
    mut call: impl FnMut() -> Result<(), RpcError>,
) -> Result<RoundTrip, RpcError> {
    let mut us = Vec::with_capacity(calls);
    for _ in 0..calls {
        let t = std::time::Instant::now();
        call()?;
        us.push(t.elapsed().as_secs_f64() * 1e6);
    }
    us.sort_by(f64::total_cmp);
    // Nearest rank: the smallest sample with at least q of the calls at or below it.
    let rank = |q: f64| us[((q * calls as f64).ceil() as usize).clamp(1, calls) - 1];
    Ok(RoundTrip {
        label: label.to_string(),
        bytes,
        calls,
        median_us: rank(0.5),
        p90_us: rank(0.9),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{serve_connection, CpuWorker, WorkerBackend};
    use std::net::TcpListener;

    #[test]
    fn a_live_worker_reports_every_call_kind_and_both_bulk_directions() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
        });
        let mut session = RpcSession::connect(addr).unwrap();

        let report = measure_link(&mut session, 8, 1 << 20).unwrap();

        let labels: Vec<&str> = report
            .round_trips
            .iter()
            .map(|r| r.label.as_str())
            .collect();
        assert_eq!(
            labels,
            ["empty", "h2d 64 B", "d2h 64 B", "h2d 8192 B", "d2h 8192 B"]
        );
        for r in &report.round_trips {
            assert_eq!(r.calls, 8, "{}", r.label);
            assert!(r.median_us > 0.0 && r.p90_us >= r.median_us, "{r:?}");
        }
        assert!(report.h2d_bytes_per_s > 0.0 && report.d2h_bytes_per_s > 0.0);
        drop(session);
        worker.join().unwrap();
    }
}
