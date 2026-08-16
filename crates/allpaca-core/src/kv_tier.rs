use std::collections::HashMap;

use allpaca_cpu::copy_async;
use allpaca_types::{
    AllpacaError, DeviceBuffer, DeviceBufferKind, FatalError, KvPage, KvResidency, PageId,
    TransientError,
};

use crate::telemetry::Metrics;

struct BufferPool {
    data: Vec<f32>,
    bump: usize,
    live: usize,
    capacity: usize,
    free: Vec<(usize, usize)>,
}

impl BufferPool {
    fn new(capacity_bytes: usize) -> Self {
        let capacity = (capacity_bytes / 4).max(1);
        Self {
            data: vec![0.0; capacity],
            bump: 0,
            live: 0,
            capacity,
            free: Vec::new(),
        }
    }

    fn alloc(&mut self, len: usize) -> Result<usize, AllpacaError> {
        if let Some(idx) = self.free.iter().position(|(_, l)| *l >= len) {
            let (off, l) = self.free.remove(idx);
            if l > len {
                self.free.push((off + len, l - len));
            }
            self.live += len;
            return Ok(off);
        }
        if self.bump + len > self.capacity {
            return Err(AllpacaError::Transient(TransientError::Backpressure(
                "KV pool exhausted".into(),
            )));
        }
        let off = self.bump;
        self.bump += len;
        self.live += len;
        Ok(off)
    }

    fn free(&mut self, off: usize, len: usize) {
        self.free.push((off, len));
        self.live = self.live.saturating_sub(len);
    }

    fn slice(&self, off: usize, len: usize) -> &[f32] {
        &self.data[off..off + len]
    }

    fn slice_mut(&mut self, off: usize, len: usize) -> &mut [f32] {
        &mut self.data[off..off + len]
    }
}

/// Two-tier KV manager: VRAM (hot) <-> Pinned DRAM (warm). No NVMe in MVP.
pub struct KvTierManager {
    pages: HashMap<PageId, KvPage>,
    vram: BufferPool,
    dram: BufferPool,
    next_id: u64,
    clock: u64,
    hot_fraction: f32,
    vram_capacity: usize,
}

impl KvTierManager {
    pub fn new(vram_bytes: usize, dram_bytes: usize, hot_fraction: f32) -> Self {
        Self {
            pages: HashMap::new(),
            vram: BufferPool::new(vram_bytes),
            dram: BufferPool::new(dram_bytes),
            next_id: 1,
            clock: 1,
            hot_fraction,
            vram_capacity: (vram_bytes / 4).max(1),
        }
    }

    pub fn page_floats(page_tokens: u32, n_kv_heads: usize, head_dim: usize) -> usize {
        page_tokens as usize * n_kv_heads * head_dim * 2
    }

    pub fn alloc_page(
        &mut self,
        layer: u32,
        page_index: u32,
        page_tokens: u32,
        n_kv_heads: usize,
        head_dim: usize,
        metrics: &Metrics,
    ) -> Result<PageId, AllpacaError> {
        let len = Self::page_floats(page_tokens, n_kv_heads, head_dim);
        self.evict_if_needed(len, metrics)?;
        let (tier, offset, kind) = match self.vram.alloc(len) {
            Ok(off) => (KvResidency::Vram, off, DeviceBufferKind::CpuVram),
            Err(_) => {
                let off = self.dram.alloc(len).map_err(|_| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(
                        "VRAM and pinned DRAM KV pools are full".into(),
                    ))
                })?;
                (KvResidency::PinnedDram, off, DeviceBufferKind::CpuPinnedDram)
            }
        };
        let id = self.next_id;
        self.next_id += 1;
        self.clock += 1;
        self.pages.insert(
            id,
            KvPage {
                id,
                layer,
                page_index,
                filled: 0,
                capacity_tokens: page_tokens,
                tier,
                refcount: 1,
                last_access: self.clock,
                predicted_reuse: 1.0,
                location: DeviceBuffer {
                    kind,
                    device_id: -1,
                    offset,
                    len,
                },
            },
        );
        self.refresh_metrics(metrics);
        Ok(id)
    }

    pub fn get(&self, id: PageId) -> Option<&KvPage> {
        self.pages.get(&id)
    }

    pub fn get_mut(&mut self, id: PageId) -> Option<&mut KvPage> {
        self.pages.get_mut(&id)
    }

    pub fn touch(&mut self, id: PageId, reuse: f32) {
        self.clock += 1;
        if let Some(page) = self.pages.get_mut(&id) {
            page.last_access = self.clock;
            page.predicted_reuse = reuse;
        }
    }

    pub fn retain(&mut self, id: PageId) {
        if let Some(page) = self.pages.get_mut(&id) {
            page.refcount = page.refcount.saturating_add(1);
        }
    }

    pub fn release(&mut self, id: PageId, metrics: &Metrics) {
        let Some(page) = self.pages.get_mut(&id) else {
            return;
        };
        page.refcount = page.refcount.saturating_sub(1);
        if page.refcount > 0 {
            return;
        }
        let loc = page.location.clone();
        match page.tier {
            KvResidency::Vram => self.vram.free(loc.offset, loc.len),
            KvResidency::PinnedDram => self.dram.free(loc.offset, loc.len),
        }
        self.pages.remove(&id);
        self.refresh_metrics(metrics);
    }

    pub fn write_slot(
        &mut self,
        id: PageId,
        slot: u32,
        n_kv_heads: usize,
        head_dim: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), AllpacaError> {
        let page = self.pages.get(&id).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::OutOfMemory(format!("unknown KV page {id}")))
        })?;
        let kv_width = n_kv_heads * head_dim;
        let k_off = page.location.offset + slot as usize * kv_width;
        let v_off = page.location.offset + page.capacity_tokens as usize * kv_width + slot as usize * kv_width;
        match page.tier {
            KvResidency::Vram => {
                self.vram.slice_mut(k_off, kv_width).copy_from_slice(k);
                self.vram.slice_mut(v_off, kv_width).copy_from_slice(v);
            }
            KvResidency::PinnedDram => {
                self.dram.slice_mut(k_off, kv_width).copy_from_slice(k);
                self.dram.slice_mut(v_off, kv_width).copy_from_slice(v);
            }
        }
        if let Some(page) = self.pages.get_mut(&id) {
            page.filled = page.filled.max(slot + 1);
        }
        Ok(())
    }

    pub fn read_kv(
        &self,
        id: PageId,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), AllpacaError> {
        let page = self.pages.get(&id).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::OutOfMemory(format!("unknown KV page {id}")))
        })?;
        let kv_width = n_kv_heads * head_dim;
        let filled = page.filled as usize;
        let cap = page.capacity_tokens as usize;
        let k_off = page.location.offset;
        let v_off = page.location.offset + cap * kv_width;
        let k = match page.tier {
            KvResidency::Vram => self.vram.slice(k_off, filled * kv_width).to_vec(),
            KvResidency::PinnedDram => self.dram.slice(k_off, filled * kv_width).to_vec(),
        };
        let v = match page.tier {
            KvResidency::Vram => self.vram.slice(v_off, filled * kv_width).to_vec(),
            KvResidency::PinnedDram => self.dram.slice(v_off, filled * kv_width).to_vec(),
        };
        Ok((k, v))
    }

    pub fn migrate_to_dram(&mut self, id: PageId, metrics: &Metrics) -> Result<(), AllpacaError> {
        let page = self.pages.get(&id).cloned().ok_or_else(|| {
            AllpacaError::Transient(TransientError::PageMigration(format!("missing {id}")))
        })?;
        if page.tier == KvResidency::PinnedDram {
            return Ok(());
        }
        let len = page.location.len;
        let dst = self.dram.alloc(len)?;
        copy_f32(
            self.dram.slice_mut(dst, len),
            self.vram.slice(page.location.offset, len),
        )?;
        self.vram.free(page.location.offset, len);
        if let Some(page) = self.pages.get_mut(&id) {
            page.tier = KvResidency::PinnedDram;
            page.location.kind = DeviceBufferKind::CpuPinnedDram;
            page.location.offset = dst;
        }
        metrics
            .page_migrations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.refresh_metrics(metrics);
        Ok(())
    }

    pub fn migrate_to_vram(&mut self, id: PageId, metrics: &Metrics) -> Result<(), AllpacaError> {
        let page = self.pages.get(&id).cloned().ok_or_else(|| {
            AllpacaError::Transient(TransientError::PageMigration(format!("missing {id}")))
        })?;
        if page.tier == KvResidency::Vram {
            return Ok(());
        }
        let len = page.location.len;
        self.evict_if_needed(len, metrics)?;
        let dst = self.vram.alloc(len)?;
        copy_f32(
            self.vram.slice_mut(dst, len),
            self.dram.slice(page.location.offset, len),
        )?;
        self.dram.free(page.location.offset, len);
        if let Some(page) = self.pages.get_mut(&id) {
            page.tier = KvResidency::Vram;
            page.location.kind = DeviceBufferKind::CpuVram;
            page.location.offset = dst;
        }
        metrics
            .page_migrations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.refresh_metrics(metrics);
        Ok(())
    }

    fn evict_if_needed(&mut self, needed: usize, metrics: &Metrics) -> Result<(), AllpacaError> {
        let hot_cap = ((self.vram_capacity as f32) * self.hot_fraction).max(1.0) as usize;
        while self.vram.live + needed > hot_cap {
            let victim = self.lowest_vram_page();
            let Some(id) = victim else {
                break;
            };
            self.migrate_to_dram(id, metrics)?;
        }
        Ok(())
    }

    fn lowest_vram_page(&self) -> Option<PageId> {
        self.pages
            .values()
            .filter(|p| p.tier == KvResidency::Vram)
            .min_by(|a, b| {
                a.priority(self.clock, 1.0, 0.1)
                    .partial_cmp(&b.priority(self.clock, 1.0, 0.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.id)
    }

    fn refresh_metrics(&self, metrics: &Metrics) {
        let vram = self
            .pages
            .values()
            .filter(|p| p.tier == KvResidency::Vram)
            .count() as u64;
        let dram = self
            .pages
            .values()
            .filter(|p| p.tier == KvResidency::PinnedDram)
            .count() as u64;
        metrics
            .vram_pages
            .store(vram, std::sync::atomic::Ordering::Relaxed);
        metrics
            .dram_pages
            .store(dram, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn counts(&self) -> (u64, u64) {
        let vram = self
            .pages
            .values()
            .filter(|p| p.tier == KvResidency::Vram)
            .count() as u64;
        let dram = self
            .pages
            .values()
            .filter(|p| p.tier == KvResidency::PinnedDram)
            .count() as u64;
        (vram, dram)
    }
}

fn copy_f32(dst: &mut [f32], src: &[f32]) -> Result<(), AllpacaError> {
    let mut src_b = vec![0u8; src.len() * 4];
    let mut dst_b = vec![0u8; dst.len() * 4];
    for (i, v) in src.iter().enumerate() {
        src_b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    copy_async(&mut dst_b, &src_b)?;
    for i in 0..dst.len() {
        let mut b = [0u8; 4];
        b.copy_from_slice(&dst_b[i * 4..i * 4 + 4]);
        dst[i] = f32::from_le_bytes(b);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::Metrics;

    #[test]
    fn spills_to_dram_when_vram_hot_set_full() {
        let metrics = Metrics::default();
        let mut kv = KvTierManager::new(256, 4096, 0.4);
        let mut ids = Vec::new();
        for i in 0..8 {
            ids.push(kv.alloc_page(0, i, 2, 1, 4, &metrics).unwrap());
        }
        let (vram, dram) = kv.counts();
        assert!(dram > 0, "expected DRAM spill, vram={vram} dram={dram}");
        assert!(metrics.page_migrations.load(std::sync::atomic::Ordering::Relaxed) > 0);
    }
}
