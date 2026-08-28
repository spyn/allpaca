//! §12 cache-aware GEMV tile shape selection for RDNA Infinity Cache.
//!
//! The runtime cannot pin Infinity Cache residency. Instead pick column-tile
//! width (32 / 64 / 128) so the A-tile + concurrent B columns stay in a
//! working set the L2/Infinity Cache can absorb.

/// Working-set estimate for one A super-block tile reused across `cols` columns.
/// A tile is 256 f32; each Q5_K/Q6_K column touches one 176/210-byte block per K-chunk.
pub fn gemv_tile_working_bytes(cols: i32, block_bytes: usize) -> usize {
    let cols = cols.max(1) as usize;
    let a_tile = 256usize.saturating_mul(4);
    let b_chunk = cols.saturating_mul(block_bytes);
    a_tile.saturating_add(b_chunk)
}

/// Pick GEMV column tile for MoE expert decode on AMD RDNA.
///
/// Heuristic (OPPORTUNITY §12): prefer the widest tile whose working set fits in
/// ~1/8 of the reported Infinity/L2 cache budget so many concurrent GEMVs leave
/// headroom. Falls back to 64 (current default) when cache size is unknown.
pub fn select_gemv_tile_cols(n: i32, k: i32, infinity_cache_bytes: usize, block_bytes: usize) -> i32 {
    let _ = (n, k);
    let budget = if infinity_cache_bytes == 0 {
        // Unknown device: keep the shipped default.
        return 64;
    } else {
        infinity_cache_bytes / 8
    };
    let candidates = [128i32, 64, 32];
    for &cols in &candidates {
        if gemv_tile_working_bytes(cols, block_bytes) <= budget {
            return cols;
        }
    }
    32
}

/// Infer a usable Infinity Cache budget from L2 cache size (HIP proxy).
/// RDNA3/4 Infinity Cache is larger than L2; scale L2 up when present.
pub fn infinity_cache_from_l2(l2_bytes: usize) -> usize {
    if l2_bytes == 0 {
        return 0;
    }
    // Typical RDNA4: 32–64 MiB Infinity Cache with a few MiB L2.
    l2_bytes.saturating_mul(8).clamp(16 << 20, 64 << 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_cache_keeps_default_64() {
        assert_eq!(select_gemv_tile_cols(256, 4096, 0, 176), 64);
    }

    #[test]
    fn large_cache_prefers_wide_tile() {
        assert_eq!(
            select_gemv_tile_cols(256, 4096, 64 << 20, 176),
            128,
            "64 MiB Infinity Cache should allow COLS=128"
        );
    }

    #[test]
    fn tiny_cache_narrows_tile() {
        // Budget = 64KiB/8 = 8KiB. COLS=128 needs ~23KiB; 64 needs ~12KiB; 32 needs ~6KiB.
        assert_eq!(select_gemv_tile_cols(64, 256, 64 << 10, 176), 32);
    }

    #[test]
    fn infinity_cache_from_l2_scales_and_clamps() {
        assert_eq!(infinity_cache_from_l2(0), 0);
        assert_eq!(infinity_cache_from_l2(4 << 20), 32 << 20);
        assert_eq!(infinity_cache_from_l2(16 << 20), 64 << 20);
    }
}
