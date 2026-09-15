//! OTLP HTTP JSON metrics push (no OpenTelemetry SDK).
//!
//! Posts the same body as `GET /v1/metrics/otlp` via reqwest.

use std::sync::Arc;
use std::time::Duration;

use allpaca_core::Engine;

/// Resolve the OTLP HTTP metrics URL.
///
/// OTLP/HTTP conventionally uses `POST …/v1/metrics`. When `endpoint` has no
/// path (or only `/`), we append `/v1/metrics`. If a non-root path is already
/// present, it is used as-is (caller may point at a collector proxy path).
pub fn otlp_metrics_url(endpoint: &str) -> String {
    let trimmed = endpoint.trim_end_matches('/');
    match url_path(trimmed) {
        None | Some("/") | Some("") => format!("{trimmed}/v1/metrics"),
        Some(_) => trimmed.to_string(),
    }
}

fn url_path(endpoint: &str) -> Option<&str> {
    let after_scheme = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))?;
    after_scheme.find('/').map(|i| &after_scheme[i..])
}

/// HTTP POST `body` as `application/json` to the OTLP metrics URL derived from `endpoint`.
pub fn push_otlp_metrics(endpoint: &str, body: &str) -> Result<(), String> {
    let url = otlp_metrics_url(endpoint);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent("allpaca/0.1")
        .build()
        .map_err(|e| format!("otlp push client: {e}"))?;
    let resp = client
        .post(&url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body.to_string())
        .send()
        .map_err(|e| format!("otlp push send: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("otlp push HTTP {}", resp.status()));
    }
    Ok(())
}

/// When `endpoint` is `Some`, spawn a periodic push of Engine OTLP JSON.
/// Returns `None` when endpoint is unset (no push task).
pub fn maybe_spawn_otlp_push(
    engine: Arc<Engine>,
    endpoint: Option<String>,
    interval: Duration,
) -> Option<tokio::task::JoinHandle<()>> {
    let endpoint = endpoint?;
    Some(tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            let engine = Arc::clone(&engine);
            let endpoint = endpoint.clone();
            let result = tokio::task::spawn_blocking(move || {
                let snap = engine.stats();
                let body = engine.metrics().export_otlp_json(&snap);
                push_otlp_metrics(&endpoint, &body)
            })
            .await;
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!(error = %e, "otlp metrics push failed"),
                Err(e) => tracing::warn!(error = %e, "otlp metrics push join failed"),
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use allpaca_types::{CliOverrides, Config};
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{mpsc, Mutex};

    fn accept_one_post(listener: TcpListener) -> String {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut buf = vec![0u8; 65536];
        let n = stream.read(&mut buf).expect("read");
        let req = String::from_utf8_lossy(&buf[..n]).to_string();
        let _ =
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        req
    }

    #[test]
    fn push_otlp_metrics_posts_json_with_metric_names() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            tx.send(accept_one_post(listener)).unwrap();
        });

        let body = r#"{"resourceMetrics":[{"scopeMetrics":[{"metrics":[{"name":"allpaca_requests_total"}]}]}]}"#;
        push_otlp_metrics(&format!("http://{addr}"), body).expect("push");

        let req = rx.recv_timeout(Duration::from_secs(5)).expect("request");
        assert!(req.starts_with("POST /v1/metrics "), "path: {req}");
        assert!(
            req.contains("Content-Type: application/json")
                || req.contains("content-type: application/json")
        );
        assert!(req.contains("resourceMetrics"));
        assert!(req.contains("allpaca_requests_total"));
    }

    #[test]
    fn otlp_metrics_url_keeps_explicit_path() {
        assert_eq!(
            otlp_metrics_url("http://collector:4318/custom/metrics"),
            "http://collector:4318/custom/metrics"
        );
        assert_eq!(
            otlp_metrics_url("http://collector:4318/"),
            "http://collector:4318/v1/metrics"
        );
    }

    #[tokio::test]
    async fn configured_endpoint_pushes_at_least_once() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let received = Arc::new(Mutex::new(None::<String>));
        let received_c = Arc::clone(&received);
        std::thread::spawn(move || {
            let req = accept_one_post(listener);
            *received_c.lock().unwrap() = Some(req);
        });

        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            synthetic: true,
            cpu_only: true,
            seed: Some(1),
            ..CliOverrides::default()
        });
        let engine = Arc::new(Engine::new(cfg).unwrap());
        let handle = maybe_spawn_otlp_push(
            engine,
            Some(format!("http://{addr}")),
            Duration::from_millis(50),
        )
        .expect("spawn when endpoint set");

        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if received.lock().unwrap().is_some() {
                break;
            }
            if tokio::time::Instant::now() > deadline {
                panic!("timed out waiting for OTLP push POST");
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        handle.abort();

        let req = received.lock().unwrap().clone().expect("body");
        assert!(req.contains("resourceMetrics"));
        assert!(req.contains("allpaca_requests_total"));
    }

    #[tokio::test]
    async fn no_push_task_when_endpoint_none() {
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            synthetic: true,
            cpu_only: true,
            seed: Some(1),
            ..CliOverrides::default()
        });
        let engine = Arc::new(Engine::new(cfg).unwrap());
        assert!(maybe_spawn_otlp_push(engine, None, Duration::from_millis(50)).is_none());
    }
}
