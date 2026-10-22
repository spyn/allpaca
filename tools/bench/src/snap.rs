//! Pull weight/KV bytes and device peaks from a running server for the
//! roofline section of `report`.

use serde::Deserialize;

use crate::roofline;
use crate::serve::{DeviceBand, Target};

/// Bytes and devices to stamp onto every sample of one start/round.
#[derive(Debug, Clone, PartialEq)]
pub struct RooflineSnap {
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    /// True when `kv_bytes` is the whole device KV pool (upper bound), not
    /// an estimate for the sequence being decoded.
    pub kv_is_pool: bool,
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
    id: i32,
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
    #[serde(default)]
    kv: Option<KvBits>,
}

#[derive(Debug, Deserialize)]
struct ExpertBits {
    #[serde(default)]
    n_experts: u64,
    #[serde(default)]
    n_active: u64,
    #[serde(default)]
    n_layers: u64,
    #[serde(default)]
    expert_slot_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct PlacementBits {
    #[serde(default)]
    device: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    weight_bytes: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct KvBits {
    #[serde(default)]
    full_bytes_per_token: u64,
    #[serde(default)]
    swa_bytes_per_token: u64,
    #[serde(default)]
    swa_window: u64,
}

impl KvBits {
    /// KV bytes for a sequence of `tokens` (same formula as `KvEstimate::bytes_for`).
    fn bytes_for(&self, tokens: u64) -> u64 {
        let swa = if self.swa_window > 0 {
            tokens.min(self.swa_window)
        } else {
            tokens
        };
        tokens
            .saturating_mul(self.full_bytes_per_token)
            .saturating_add(swa.saturating_mul(self.swa_bytes_per_token))
    }
}

/// Build a snap from `/v1/status` and optional `/v1/model/settings` JSON.
///
/// Dense models use resident weight bytes. MoE models use
/// `shared + n_active × slot × n_layers` when `expert_slot_bytes` is known,
/// so attention / embeddings / LM head are not scaled by the expert fraction.
///
/// `context_hint` is the tokens already on the warm sequence (warm-up length);
/// when settings expose a KV estimate, that yields KV bytes for the decode
/// rather than the whole device pool.
pub fn from_json(
    status_json: &str,
    settings_json: Option<&str>,
    context_hint: u64,
) -> Result<RooflineSnap, String> {
    let status: StatusBody = serde_json::from_str(status_json).map_err(|e| e.to_string())?;
    let stats = status.stats.unwrap_or(StatsBits {
        weight_bytes_on_device: 0,
        kv_bytes_on_device: 0,
    });
    let settings: Option<SettingsBody> = settings_json
        .map(|s| serde_json::from_str(s).map_err(|e| e.to_string()))
        .transpose()?;

    let mut weight = moe_or_dense_weight(stats.weight_bytes_on_device, settings.as_ref().and_then(|s| s.experts.as_ref()));

    let (kv_bytes, kv_is_pool) = match settings.as_ref().and_then(|s| s.kv.as_ref()) {
        Some(kv) if kv.full_bytes_per_token > 0 || kv.swa_bytes_per_token > 0 => {
            let tokens = context_hint.max(1);
            (kv.bytes_for(tokens), false)
        }
        _ => (stats.kv_bytes_on_device, stats.kv_bytes_on_device > 0),
    };

    let mut devices: Vec<DeviceBand> = status
        .devices
        .iter()
        .filter(|d| is_accelerator(d))
        .map(|d| DeviceBand {
            device: device_key(d),
            name: d.name.clone(),
            peak_gbs: roofline::peak_for(&d.name, d.bandwidth_gbps),
            weight_bytes: 0,
        })
        .collect();

    // Prefer placement weight bytes when present (layer split). Match on the
    // placement `device` key (`hip:0`), not the card name — identical MI50s
    // / K80s share a name.
    if let Some(settings) = &settings {
        for p in &settings.placement {
            let Some(wb) = p.weight_bytes else { continue };
            let scaled = moe_or_dense_weight(wb, settings.experts.as_ref());
            if let Some(dev) = devices.iter_mut().find(|d| d.device == p.device) {
                dev.weight_bytes = scaled;
            } else if !p.device.is_empty() || !p.name.is_empty() {
                devices.push(DeviceBand {
                    device: if p.device.is_empty() {
                        p.name.clone()
                    } else {
                        p.device.clone()
                    },
                    name: if p.name.is_empty() {
                        p.device.clone()
                    } else {
                        p.name.clone()
                    },
                    peak_gbs: roofline::peak_for(&p.name, 0.0),
                    weight_bytes: scaled,
                });
            }
        }
        let placed: u64 = devices.iter().map(|d| d.weight_bytes).sum();
        if placed > 0 {
            weight = placed;
        }
    }

    if devices.len() == 1 && devices[0].weight_bytes == 0 {
        devices[0].weight_bytes = weight;
    }

    Ok(RooflineSnap {
        weight_bytes: weight,
        kv_bytes,
        kv_is_pool,
        devices,
    })
}

/// Active weights for MoE when slot size is known; else resident (dense).
fn moe_or_dense_weight(resident: u64, experts: Option<&ExpertBits>) -> u64 {
    let Some(ex) = experts else {
        return resident;
    };
    if ex.n_experts == 0 || ex.n_active == 0 || ex.n_active >= ex.n_experts || ex.expert_slot_bytes == 0 || ex.n_layers == 0
    {
        return resident;
    }
    let pool = ex
        .expert_slot_bytes
        .saturating_mul(ex.n_experts)
        .saturating_mul(ex.n_layers);
    let shared = resident.saturating_sub(pool);
    roofline::moe_active_weight_bytes(shared, ex.expert_slot_bytes, ex.n_active, ex.n_layers)
}

fn device_key(d: &DeviceBits) -> String {
    let k = d.kind.to_ascii_lowercase();
    if k.contains("hip") {
        format!("hip:{}", d.id)
    } else if k.contains("cuda") {
        format!("cuda:{}", d.id)
    } else if k.contains("vulkan") {
        format!("vulkan:{}", d.id)
    } else if d.name.starts_with("rpc[") {
        d.name.clone()
    } else {
        format!("{}:{}", d.kind, d.id)
    }
}

/// GET `/v1/status` and `/v1/model/settings` from a running server.
pub fn fetch(target: &Target, context_hint: u64) -> Result<RooflineSnap, String> {
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
    from_json(&status, settings.as_deref(), context_hint)
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
        let snap = from_json(status, None, 40).unwrap();
        assert_eq!(snap.weight_bytes, 9_500_000_000);
        assert_eq!(snap.kv_bytes, 100_000_000);
        assert!(snap.kv_is_pool);
        assert_eq!(snap.devices.len(), 1);
        assert_eq!(snap.devices[0].device, "hip:0");
        assert_eq!(snap.devices[0].peak_gbs, 640.0);
        assert_eq!(snap.devices[0].weight_bytes, 9_500_000_000);
    }

    #[test]
    fn moe_keeps_shared_weights_and_scales_only_experts() {
        // 3 GB shared + 34 GB expert pool (256 × slot × layers), top-8.
        // Active ≈ 3 + 8/256×34 ≈ 4.06 GB — not 37/32 ≈ 1.2 GB.
        let status = r#"{
            "stats": {"weight_bytes_on_device": 37000000000, "kv_bytes_on_device": 0},
            "devices": [{"id": 0, "kind": "HipGpu", "name": "AMD Radeon RX 9070", "bandwidth_gbps": 640.0, "memory_bytes": 1, "peers": []}]
        }"#;
        let settings = r#"{
            "experts": {
                "n_layers": 40,
                "n_experts": 256,
                "n_active": 8,
                "expert_slot_bytes": 3320312,
                "recording": false,
                "disabled_total": 0
            },
            "placement": []
        }"#;
        // 3320312 * 256 * 40 ≈ 34.0e9
        let snap = from_json(status, Some(settings), 40).unwrap();
        let shared = 37_000_000_000u64.saturating_sub(3_320_312u64 * 256 * 40);
        let want = roofline::moe_active_weight_bytes(shared, 3_320_312, 8, 40);
        assert_eq!(snap.weight_bytes, want);
        assert!(snap.weight_bytes > 3_000_000_000, "shared must remain: {}", snap.weight_bytes);
        assert!(snap.weight_bytes < 6_000_000_000, "must not keep full pool: {}", snap.weight_bytes);
    }

    #[test]
    fn identical_card_names_keep_separate_placement_rows() {
        let status = r#"{
            "stats": {"weight_bytes_on_device": 16000000000, "kv_bytes_on_device": 0},
            "devices": [
                {"id": 0, "kind": "HipGpu", "name": "AMD Instinct MI50", "bandwidth_gbps": 1000.0, "memory_bytes": 1, "peers": []},
                {"id": 1, "kind": "HipGpu", "name": "AMD Instinct MI50", "bandwidth_gbps": 1000.0, "memory_bytes": 1, "peers": []}
            ]
        }"#;
        let settings = r#"{
            "experts": null,
            "placement": [
                {"device": "hip:0", "name": "AMD Instinct MI50", "weight_bytes": 10000000000, "n_layers": 20, "layers": [], "expert_layers": 0, "roles": []},
                {"device": "hip:1", "name": "AMD Instinct MI50", "weight_bytes": 6000000000, "n_layers": 20, "layers": [], "expert_layers": 0, "roles": []}
            ]
        }"#;
        let snap = from_json(status, Some(settings), 40).unwrap();
        assert_eq!(snap.devices.len(), 2);
        assert_eq!(snap.devices[0].weight_bytes, 10_000_000_000);
        assert_eq!(snap.devices[1].weight_bytes, 6_000_000_000);
        assert_eq!(snap.devices[0].peak_gbs, 1024.0);
        assert_eq!(snap.weight_bytes, 16_000_000_000);
    }

    #[test]
    fn kv_estimate_from_settings_beats_device_pool() {
        let status = r#"{
            "stats": {"weight_bytes_on_device": 1000, "kv_bytes_on_device": 999999999},
            "devices": [{"id": 0, "kind": "HipGpu", "name": "AMD Radeon RX 9070", "bandwidth_gbps": 640.0, "memory_bytes": 1, "peers": []}]
        }"#;
        let settings = r#"{
            "kv": {
                "full_bytes_per_token": 1000,
                "swa_bytes_per_token": 0,
                "swa_window": 0,
                "context_tokens": 8192,
                "budget_bytes": 1,
                "attention_layers": 1,
                "page_tokens": 1
            },
            "placement": []
        }"#;
        let snap = from_json(status, Some(settings), 40).unwrap();
        assert_eq!(snap.kv_bytes, 40_000);
        assert!(!snap.kv_is_pool);
    }
}
