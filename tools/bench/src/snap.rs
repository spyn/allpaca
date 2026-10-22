//! Pull weight/KV bytes and device peaks from a running server for the
//! roofline section of `report`.

use serde::Deserialize;

use crate::roofline;
use crate::serve::{DeviceBand, Target};

/// Bytes and devices to stamp onto every sample of one fresh start.
#[derive(Debug, Clone, PartialEq)]
pub struct RooflineSnap {
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    pub devices: Vec<DeviceBand>,
}

#[derive(Debug, Deserialize)]
struct StatusBody {
    #[serde(default)]
    stats: Option<StatsBits>,
    #[serde(default)]
    devices: Vec<DeviceBits>,
}

#[derive(Debug, Deserialize)]
struct StatsBits {
    #[serde(default)]
    weight_bytes_on_device: u64,
    #[serde(default)]
    kv_bytes_on_device: u64,
}

#[derive(Debug, Deserialize)]
struct DeviceBits {
    #[serde(default)]
    name: String,
    #[serde(default)]
    bandwidth_gbps: f64,
    #[serde(default)]
    kind: String,
}

#[derive(Debug, Deserialize)]
struct SettingsBody {
    #[serde(default)]
    experts: Option<ExpertBits>,
    #[serde(default)]
    placement: Vec<PlacementBits>,
}

#[derive(Debug, Deserialize)]
struct ExpertBits {
    #[serde(default)]
    n_experts: u64,
    #[serde(default)]
    n_active: u64,
}

#[derive(Debug, Deserialize)]
struct PlacementBits {
    #[serde(default)]
    name: String,
    #[serde(default)]
    weight_bytes: Option<u64>,
}

/// Build a snap from `/v1/status` and optional `/v1/model/settings` JSON.
///
/// Dense models use resident weight bytes. MoE models scale resident weights
/// by `n_active / n_experts` so the roofline uses active bytes (bench-only
/// estimate; shared tensors are scaled down too).
pub fn from_json(status_json: &str, settings_json: Option<&str>) -> Result<RooflineSnap, String> {
    let status: StatusBody = serde_json::from_str(status_json).map_err(|e| e.to_string())?;
    let stats = status.stats.unwrap_or(StatsBits {
        weight_bytes_on_device: 0,
        kv_bytes_on_device: 0,
    });
    let settings: Option<SettingsBody> = settings_json
        .map(|s| serde_json::from_str(s).map_err(|e| e.to_string()))
        .transpose()?;

    let mut weight = stats.weight_bytes_on_device;
    if let Some(ex) = settings.as_ref().and_then(|s| s.experts.as_ref()) {
        if ex.n_experts > 0 && ex.n_active > 0 && ex.n_active < ex.n_experts {
            weight = ((weight as u128) * (ex.n_active as u128) / (ex.n_experts as u128)) as u64;
        }
    }

    let mut devices: Vec<DeviceBand> = status
        .devices
        .iter()
        .filter(|d| is_accelerator(d))
        .map(|d| DeviceBand {
            name: d.name.clone(),
            peak_gbs: roofline::peak_for(&d.name, d.bandwidth_gbps),
            weight_bytes: 0,
        })
        .collect();

    // Prefer placement weight bytes when present (layer split).
    if let Some(settings) = &settings {
        for p in &settings.placement {
            if let Some(wb) = p.weight_bytes {
                if let Some(dev) = devices.iter_mut().find(|d| names_match(&d.name, &p.name)) {
                    let scaled = scale_moe(wb, settings.experts.as_ref());
                    dev.weight_bytes = scaled;
                } else if !p.name.is_empty() {
                    devices.push(DeviceBand {
                        name: p.name.clone(),
                        peak_gbs: roofline::peak_for(&p.name, 0.0),
                        weight_bytes: scale_moe(wb, settings.experts.as_ref()),
                    });
                }
            }
        }
        let placed: u64 = devices.iter().map(|d| d.weight_bytes).sum();
        if placed > 0 {
            weight = placed;
        }
    }

    // Single-GPU: put all weight bytes on the only device row.
    if devices.len() == 1 && devices[0].weight_bytes == 0 {
        devices[0].weight_bytes = weight;
    }

    Ok(RooflineSnap {
        weight_bytes: weight,
        kv_bytes: stats.kv_bytes_on_device,
        devices,
    })
}

fn scale_moe(bytes: u64, experts: Option<&ExpertBits>) -> u64 {
    match experts {
        Some(ex) if ex.n_experts > 0 && ex.n_active > 0 && ex.n_active < ex.n_experts => {
            ((bytes as u128) * (ex.n_active as u128) / (ex.n_experts as u128)) as u64
        }
        _ => bytes,
    }
}

fn names_match(a: &str, b: &str) -> bool {
    let a = a.to_ascii_lowercase();
    let b = b.to_ascii_lowercase();
    a == b || a.contains(&b) || b.contains(&a)
}

/// GET `/v1/status` and `/v1/model/settings` from a running server.
pub fn fetch(target: &Target) -> Result<RooflineSnap, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| e.to_string())?;
    let base = target.url.trim_end_matches('/');
    let status = client
        .get(format!("{base}/v1/status"))
        .bearer_auth(&target.key)
        .send()
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.text())
        .map_err(|e| format!("GET /v1/status: {e}"))?;
    let settings = client
        .get(format!("{base}/v1/model/settings"))
        .bearer_auth(&target.key)
        .send()
        .ok()
        .and_then(|r| r.error_for_status().ok())
        .and_then(|r| r.text().ok());
    from_json(&status, settings.as_deref())
}

fn is_accelerator(d: &DeviceBits) -> bool {
    let k = d.kind.to_ascii_lowercase();
    if k.contains("cpu") || k.contains("nvme") {
        return false;
    }
    if k.contains("hip") || k.contains("cuda") || k.contains("vulkan") || k.contains("gpu") {
        return true;
    }
    let n = d.name.to_ascii_lowercase();
    n.contains("radeon")
        || n.contains("geforce")
        || n.contains("mi50")
        || n.contains("tesla")
        || n.contains("rtx")
        || n.contains("instinct")
        || n.starts_with("rpc[")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_status_gives_resident_weights_and_9070_peak() {
        let status = r#"{
            "stats": {"weight_bytes_on_device": 9500000000, "kv_bytes_on_device": 100000000},
            "devices": [
                {"id": 0, "kind": "HipGpu", "name": "AMD Radeon RX 9070", "bandwidth_gbps": 600.0, "memory_bytes": 1, "peers": []},
                {"id": -1, "kind": "Cpu", "name": "cpu", "bandwidth_gbps": 50.0, "memory_bytes": 0, "peers": []}
            ]
        }"#;
        let snap = from_json(status, None).unwrap();
        assert_eq!(snap.weight_bytes, 9_500_000_000);
        assert_eq!(snap.kv_bytes, 100_000_000);
        assert_eq!(snap.devices.len(), 1);
        assert_eq!(snap.devices[0].peak_gbs, 640.0);
        assert_eq!(snap.devices[0].weight_bytes, 9_500_000_000);
    }

    #[test]
    fn moe_scales_resident_weights_by_active_over_experts() {
        let status = r#"{
            "stats": {"weight_bytes_on_device": 16000000000, "kv_bytes_on_device": 0},
            "devices": [{"id": 0, "kind": "HipGpu", "name": "AMD Radeon RX 9070", "bandwidth_gbps": 640.0, "memory_bytes": 1, "peers": []}]
        }"#;
        // 128 experts, top-8 → 1/16 of resident ≈ 1 GB active.
        let settings = r#"{
            "experts": {"n_layers": 48, "n_experts": 128, "n_active": 8, "recording": false, "disabled_total": 0},
            "placement": []
        }"#;
        let snap = from_json(status, Some(settings)).unwrap();
        assert_eq!(snap.weight_bytes, 1_000_000_000);
    }

    #[test]
    fn layer_split_keeps_per_device_weight_bytes() {
        let status = r#"{
            "stats": {"weight_bytes_on_device": 16000000000, "kv_bytes_on_device": 0},
            "devices": [
                {"id": 0, "kind": "HipGpu", "name": "AMD Radeon RX 9070", "bandwidth_gbps": 640.0, "memory_bytes": 1, "peers": []},
                {"id": 1, "kind": "HipGpu", "name": "AMD Radeon RX 9060 XT", "bandwidth_gbps": 320.0, "memory_bytes": 1, "peers": []}
            ]
        }"#;
        let settings = r#"{
            "experts": null,
            "placement": [
                {"device": "hip:0", "name": "AMD Radeon RX 9070", "weight_bytes": 10000000000, "n_layers": 20, "layers": [], "expert_layers": 0, "roles": []},
                {"device": "hip:1", "name": "AMD Radeon RX 9060 XT", "weight_bytes": 6000000000, "n_layers": 20, "layers": [], "expert_layers": 0, "roles": []}
            ]
        }"#;
        let snap = from_json(status, Some(settings)).unwrap();
        assert_eq!(snap.weight_bytes, 16_000_000_000);
        assert_eq!(snap.devices[0].weight_bytes, 10_000_000_000);
        assert_eq!(snap.devices[1].weight_bytes, 6_000_000_000);
        assert_eq!(snap.devices[0].peak_gbs, 640.0);
        assert_eq!(snap.devices[1].peak_gbs, 320.0);
    }
}
