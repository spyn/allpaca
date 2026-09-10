//! Minimal OTLP-ish HTTP JSON metrics export (no OpenTelemetry SDK).
//!
//! Builds a `resourceMetrics` body compatible with collectors that accept
//! OTLP/JSON, using the same `allpaca_*` names as Prometheus `/metrics`.

use allpaca_types::StatsSnapshot;
use serde_json::{json, Value};

/// Build an OTLP HTTP JSON metrics body from a [`StatsSnapshot`].
pub fn export_otlp_json(s: &StatsSnapshot) -> String {
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let metrics = otlp_metric_list(s, now_ns);
    let body = json!({
        "resourceMetrics": [{
            "resource": {
                "attributes": [
                    { "key": "service.name", "value": { "stringValue": "allpaca" } },
                    { "key": "allpaca.backend", "value": { "stringValue": s.backend } },
                    { "key": "allpaca.model_id", "value": { "stringValue": s.model_id } },
                ]
            },
            "scopeMetrics": [{
                "scope": { "name": "allpaca", "version": env!("CARGO_PKG_VERSION") },
                "metrics": metrics,
            }]
        }]
    });
    body.to_string()
}

fn sum_metric(name: &str, as_int: u64, time_unix_nano: u64, monotonic: bool) -> Value {
    json!({
        "name": name,
        "sum": {
            "dataPoints": [{
                "asInt": as_int.to_string(),
                "timeUnixNano": time_unix_nano.to_string(),
            }],
            "aggregationTemporality": 2,
            "isMonotonic": monotonic,
        }
    })
}

fn gauge_metric(name: &str, as_double: f64, time_unix_nano: u64) -> Value {
    json!({
        "name": name,
        "gauge": {
            "dataPoints": [{
                "asDouble": as_double,
                "timeUnixNano": time_unix_nano.to_string(),
            }]
        }
    })
}

fn otlp_metric_list(s: &StatsSnapshot, t: u64) -> Vec<Value> {
    vec![
        sum_metric("allpaca_requests_total", s.requests_total, t, true),
        gauge_metric("allpaca_active_requests", s.active_requests as f64, t),
        sum_metric("allpaca_prompt_tokens_total", s.prompt_tokens, t, true),
        sum_metric("allpaca_tokens_generated_total", s.tokens_generated, t, true),
        sum_metric("allpaca_draft_proposed_total", s.draft_proposed, t, true),
        sum_metric("allpaca_draft_accepted_total", s.draft_accepted, t, true),
        gauge_metric("allpaca_acceptance_rate", s.acceptance_rate, t),
        sum_metric("allpaca_poll_ticks_total", s.poll_ticks, t, true),
        sum_metric("allpaca_batch_gemm_plans_total", s.batch_gemm_plans, t, true),
        sum_metric("allpaca_batch_gemm_executes_total", s.batch_gemm_executes, t, true),
        sum_metric("allpaca_batch_gemm_hip_kernels_total", s.batch_gemm_hip_kernels, t, true),
        sum_metric("allpaca_micro_batch_plans_total", s.micro_batch_plans, t, true),
        gauge_metric("allpaca_vram_pages", s.vram_pages as f64, t),
        gauge_metric("allpaca_dram_pages", s.dram_pages as f64, t),
        gauge_metric("allpaca_ttft_ms_last", s.ttft_ms_last as f64, t),
        gauge_metric("allpaca_itl_ms_ema", s.itl_ms_ema, t),
        gauge_metric("allpaca_gpu_util", s.gpu_util, t),
        gauge_metric(
            "allpaca_vision_weight_bytes_on_device",
            s.vision_weight_bytes_on_device as f64,
            t,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use allpaca_types::StatsSnapshot;

    fn empty_snapshot() -> StatsSnapshot {
        // Plain string avoids `json!` recursion-limit blowups as StatsSnapshot grows.
        serde_json::from_str(
            r#"{
            "pipeline_mode": "cpu",
            "transfer_path": "host",
            "p2p_capable": false,
            "p2p_enabled": false,
            "p2p_status": "p2p_unavailable",
            "p2p_gbps": 0.0,
            "host_staging_gbps": 0.0,
            "p2p_us": 0.0,
            "host_staging_us": 0.0,
            "target_device": -1,
            "draft_device": -1,
            "requests_total": 3,
            "active_requests": 1,
            "prompt_tokens": 10,
            "prefill_chunks": 0,
            "tokens_generated": 5,
            "draft_proposed": 0,
            "draft_accepted": 0,
            "acceptance_rate": 0.0,
            "page_migrations": 0,
            "vram_pages": 0,
            "dram_pages": 0,
            "ttft_ms_last": 12,
            "itl_ms_ema": 1.5,
            "degraded_events": 0,
            "tree_nodes": 0,
            "tree_accepted_path_len": 0,
            "tree_path_valid": false,
            "tree_gpu_verifies": 0,
            "linear_packed_verifies": 0,
            "accepted_tokens_per_target_eval": 0.0,
            "model_id": "test",
            "backend": "cpu",
            "gpu_forwards": 0,
            "target_gpu_forwards": 0,
            "draft_gpu_forwards": 0,
            "host_staging_bytes": 0,
            "draft_tokens_copied": 0,
            "weight_bytes_migrated": 0,
            "device_memory_bytes": 0,
            "weight_bytes_on_device": 0,
            "weight_h2d_bytes": 0,
            "kv_bytes_on_device": 0,
            "draft_kv_bytes_on_device": 0,
            "elementwise_gpu_ops": 0,
            "residual_gpu_ops": 0,
            "activation_h2d_bytes": 0,
            "activation_d2h_bytes": 0,
            "q8_gemm_us": 0.0,
            "attention_us": 0.0,
            "lm_head_us": 0.0,
            "other_gpu_us": 0.0,
            "host_us": 0.0
        }"#,
        )
        .expect("minimal StatsSnapshot")
    }

    #[test]
    fn otlp_json_contains_allpaca_metric_names() {
        let body = export_otlp_json(&empty_snapshot());
        assert!(body.contains("resourceMetrics"));
        for name in [
            "allpaca_requests_total",
            "allpaca_tokens_generated_total",
            "allpaca_poll_ticks_total",
            "allpaca_batch_gemm_plans_total",
            "allpaca_batch_gemm_executes_total",
            "allpaca_batch_gemm_hip_kernels_total",
            "allpaca_gpu_util",
        ] {
            assert!(body.contains(name), "missing {name} in {body}");
        }
        let v: Value = serde_json::from_str(&body).expect("valid json");
        let metrics = &v["resourceMetrics"][0]["scopeMetrics"][0]["metrics"];
        assert!(metrics.as_array().map(|a| a.len()).unwrap_or(0) >= 5);
    }
}
