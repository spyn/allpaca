//! Bytes/token, effective bandwidth, and % of device peak for bench reports.

/// Published peaks used when a device name is recognised. Otherwise the
/// inventory's measured `bandwidth_gbps` is used.
pub fn peak_gbs(device_name: &str) -> Option<f64> {
    let n = device_name.to_ascii_lowercase();
    if n.contains("9070") && !n.contains("9060") {
        // Non-XT RX 9070: 56 CUs / 640 GB/s (not the 64-CU XT / R9700).
        Some(640.0)
    } else if n.contains("9060") {
        Some(320.0)
    } else if n.contains("8060s") || n.contains("gfx1151") {
        // Strix Halo: LPDDR5X-8000 on 256 bits, shared with the CPU (calculated).
        Some(256.0)
    } else if n.contains("mi50") || n.contains("gfx906") {
        Some(1024.0)
    } else if n.contains("3060") {
        Some(360.0)
    } else if n.contains("3080 ti") {
        Some(912.0)
    } else if n.contains("3070 ti") {
        Some(608.0)
    } else if n.contains("k80") || n.contains("tesla k80") {
        Some(240.0)
    } else {
        None
    }
}

/// Peak for a device: known table first, else the inventory bandwidth.
pub fn peak_for(device_name: &str, inventory_gbs: f64) -> f64 {
    peak_gbs(device_name).unwrap_or(inventory_gbs)
}

/// Weights plus KV read once per decode token.
pub fn bytes_per_token(weight_bytes: u64, kv_bytes: u64) -> f64 {
    weight_bytes.saturating_add(kv_bytes) as f64
}

/// MoE: shared (attention / routers / dense) plus the top-k experts each
/// MoE layer touches, not the full resident expert pool.
pub fn moe_active_weight_bytes(
    shared_bytes: u64,
    expert_slot_bytes: u64,
    n_active: u64,
    n_moe_layers: u64,
) -> u64 {
    shared_bytes.saturating_add(
        expert_slot_bytes
            .saturating_mul(n_active)
            .saturating_mul(n_moe_layers),
    )
}

/// Effective memory bandwidth: bytes/token × tok/s, in GB/s.
pub fn effective_gbs(bytes_per_token: f64, tok_per_s: f64) -> f64 {
    if !(bytes_per_token.is_finite() && tok_per_s.is_finite()) || bytes_per_token <= 0.0 || tok_per_s <= 0.0 {
        return 0.0;
    }
    bytes_per_token * tok_per_s / 1e9
}

/// Share of peak bandwidth (0–100+).
pub fn pct_of_peak(effective_gbs: f64, peak_gbs: f64) -> f64 {
    if !(effective_gbs.is_finite() && peak_gbs.is_finite()) || peak_gbs <= 0.0 {
        return 0.0;
    }
    effective_gbs / peak_gbs * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_devices_have_published_peaks() {
        assert_eq!(peak_gbs("AMD Radeon RX 9070"), Some(640.0));
        assert_eq!(peak_gbs("Radeon RX 9070 XT"), Some(640.0)); // still 640 table; XT is a different card we do not own
        assert_eq!(peak_gbs("AMD Radeon RX 9060 XT"), Some(320.0));
        assert_eq!(peak_gbs("AMD Radeon 8060S Graphics"), Some(256.0));
        assert_eq!(peak_gbs("AMD Instinct MI50"), Some(1024.0));
        assert_eq!(peak_gbs("NVIDIA GeForce RTX 3060"), Some(360.0));
        assert_eq!(peak_gbs("Tesla K80"), Some(240.0));
        assert_eq!(peak_gbs("NVIDIA GeForce RTX 3070 Ti"), Some(608.0));
        assert_eq!(peak_gbs("NVIDIA GeForce RTX 3080 Ti"), Some(912.0));
        assert_eq!(peak_gbs("Mysterious GPU"), None);
        assert_eq!(peak_for("Mysterious GPU", 123.0), 123.0);
    }

    #[test]
    fn dense_bytes_and_bandwidth_match_the_roofline_shape() {
        // Ornith-9B Q8 ~9.5 GB/token at 36.7 tok/s on a 640 GB/s 9070 → ~55%.
        let bpt = bytes_per_token(9_500_000_000, 0);
        assert!((bpt - 9.5e9).abs() < 1.0);
        let gbs = effective_gbs(bpt, 36.7);
        let pct = pct_of_peak(gbs, 640.0);
        assert!((gbs - 348.65).abs() < 0.1, "gbs={gbs}");
        assert!((pct - 54.48).abs() < 0.1, "pct={pct}");
    }

    #[test]
    fn moe_counts_only_the_active_experts_per_layer() {
        // 1 GB shared + 4 layers × 2 active × 100 MB expert → 1.8 GB weights.
        let w = moe_active_weight_bytes(1_000_000_000, 100_000_000, 2, 4);
        assert_eq!(w, 1_800_000_000);
        assert_eq!(bytes_per_token(w, 200_000_000), 2.0e9);
    }
}
