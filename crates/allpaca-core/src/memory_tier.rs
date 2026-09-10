//! Future memory-tier backends (OPPORTUNITY §12 CXL/PIM interface).
//!
//! Hardware CXL/PIM remains Out; simulators exercise the trait with delayed
//! tickets so the scheduler can evolve without embedding device assumptions.

use std::collections::HashMap;

use allpaca_types::{AllpacaError, DegradedError, PageId};

/// Opaque handle returned by [`MemoryTierBackend::prefetch`].
pub type ReadTicket = u64;

/// Pluggable page fetch backend (VRAM / pinned DRAM / NVMe / future CXL/PIM).
pub trait MemoryTierBackend: Send + Sync {
    fn prefetch(
        &mut self,
        page: PageId,
        dst: &mut [u8],
    ) -> Result<ReadTicket, AllpacaError>;
    fn ready(&self, ticket: ReadTicket) -> bool;
    fn release(&mut self, page: PageId);
}

/// In-process pinned DRAM: prefetch completes immediately and tracks tickets.
#[derive(Debug, Default)]
pub struct PinnedDramBackend {
    next_ticket: ReadTicket,
    /// ticket → page
    outstanding: HashMap<ReadTicket, PageId>,
    /// page → last ticket (for release)
    pages: HashMap<PageId, ReadTicket>,
}

impl PinnedDramBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn outstanding_len(&self) -> usize {
        self.outstanding.len()
    }
}

impl MemoryTierBackend for PinnedDramBackend {
    fn prefetch(
        &mut self,
        page: PageId,
        dst: &mut [u8],
    ) -> Result<ReadTicket, AllpacaError> {
        // Conceptual fill: zero the destination; real DRAM copy is elsewhere.
        dst.fill(0);
        self.next_ticket = self.next_ticket.saturating_add(1).max(1);
        let ticket = self.next_ticket;
        self.outstanding.insert(ticket, page);
        self.pages.insert(page, ticket);
        Ok(ticket)
    }

    fn ready(&self, ticket: ReadTicket) -> bool {
        self.outstanding.contains_key(&ticket)
    }

    fn release(&mut self, page: PageId) {
        if let Some(ticket) = self.pages.remove(&page) {
            self.outstanding.remove(&ticket);
        }
    }
}

#[derive(Debug)]
struct SimTicket {
    #[allow(dead_code)]
    page: PageId,
    remaining: u32,
}

/// Simulated CXL.mem path: host HashMap store + delayed ticket readiness.
#[derive(Debug)]
pub struct CxlMemBackend {
    unimplemented: bool,
    store: HashMap<PageId, Vec<u8>>,
    next_ticket: ReadTicket,
    outstanding: HashMap<ReadTicket, SimTicket>,
    pages: HashMap<PageId, ReadTicket>,
    latency_ticks: u32,
}

impl Default for CxlMemBackend {
    fn default() -> Self {
        Self::simulated(1)
    }
}

impl CxlMemBackend {
    /// Soft simulator with `latency_ticks` delay before [`MemoryTierBackend::ready`].
    pub fn simulated(latency_ticks: u32) -> Self {
        Self {
            unimplemented: false,
            store: HashMap::new(),
            next_ticket: 0,
            outstanding: HashMap::new(),
            pages: HashMap::new(),
            latency_ticks,
        }
    }

    /// Legacy degraded stub (feature disabled).
    pub fn unimplemented() -> Self {
        Self {
            unimplemented: true,
            store: HashMap::new(),
            next_ticket: 0,
            outstanding: HashMap::new(),
            pages: HashMap::new(),
            latency_ticks: 0,
        }
    }

    pub fn put(&mut self, page: PageId, data: Vec<u8>) {
        self.store.insert(page, data);
    }

    /// Advance simulated CXL latency by one tick for all outstanding tickets.
    pub fn tick(&mut self) {
        for t in self.outstanding.values_mut() {
            t.remaining = t.remaining.saturating_sub(1);
        }
    }
}

impl MemoryTierBackend for CxlMemBackend {
    fn prefetch(
        &mut self,
        page: PageId,
        dst: &mut [u8],
    ) -> Result<ReadTicket, AllpacaError> {
        if self.unimplemented {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CxlMemBackend not implemented".into(),
            )));
        }
        if let Some(src) = self.store.get(&page) {
            let n = dst.len().min(src.len());
            dst[..n].copy_from_slice(&src[..n]);
            if n < dst.len() {
                dst[n..].fill(0);
            }
        } else {
            dst.fill(0);
        }
        self.next_ticket = self.next_ticket.saturating_add(1).max(1);
        let ticket = self.next_ticket;
        self.outstanding.insert(
            ticket,
            SimTicket {
                page,
                remaining: self.latency_ticks,
            },
        );
        self.pages.insert(page, ticket);
        Ok(ticket)
    }

    fn ready(&self, ticket: ReadTicket) -> bool {
        self.outstanding
            .get(&ticket)
            .is_some_and(|t| t.remaining == 0)
    }

    fn release(&mut self, page: PageId) {
        if let Some(ticket) = self.pages.remove(&page) {
            self.outstanding.remove(&ticket);
        }
    }
}

/// Simulated PIM path: copy + in-memory XOR checksum, then delayed ready.
#[derive(Debug)]
pub struct PimBackend {
    unimplemented: bool,
    store: HashMap<PageId, Vec<u8>>,
    next_ticket: ReadTicket,
    outstanding: HashMap<ReadTicket, SimTicket>,
    pages: HashMap<PageId, ReadTicket>,
    latency_ticks: u32,
}

impl Default for PimBackend {
    fn default() -> Self {
        Self::simulated(1)
    }
}

impl PimBackend {
    pub fn simulated(latency_ticks: u32) -> Self {
        Self {
            unimplemented: false,
            store: HashMap::new(),
            next_ticket: 0,
            outstanding: HashMap::new(),
            pages: HashMap::new(),
            latency_ticks,
        }
    }

    pub fn unimplemented() -> Self {
        Self {
            unimplemented: true,
            store: HashMap::new(),
            next_ticket: 0,
            outstanding: HashMap::new(),
            pages: HashMap::new(),
            latency_ticks: 0,
        }
    }

    pub fn put(&mut self, page: PageId, data: Vec<u8>) {
        self.store.insert(page, data);
    }

    pub fn tick(&mut self) {
        for t in self.outstanding.values_mut() {
            t.remaining = t.remaining.saturating_sub(1);
        }
    }
}

/// Fold XOR of all bytes into a u32 and write it to the first 4 bytes of `dst`.
pub fn pim_xor_checksum(dst: &mut [u8]) {
    let mut cs = 0u32;
    for &b in dst.iter() {
        cs ^= u32::from(b);
    }
    if dst.len() >= 4 {
        dst[..4].copy_from_slice(&cs.to_le_bytes());
    }
}

impl MemoryTierBackend for PimBackend {
    fn prefetch(
        &mut self,
        page: PageId,
        dst: &mut [u8],
    ) -> Result<ReadTicket, AllpacaError> {
        if self.unimplemented {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "PimBackend not implemented".into(),
            )));
        }
        if let Some(src) = self.store.get(&page) {
            let n = dst.len().min(src.len());
            dst[..n].copy_from_slice(&src[..n]);
            if n < dst.len() {
                dst[n..].fill(0);
            }
        } else {
            dst.fill(0);
        }
        pim_xor_checksum(dst);
        self.next_ticket = self.next_ticket.saturating_add(1).max(1);
        let ticket = self.next_ticket;
        self.outstanding.insert(
            ticket,
            SimTicket {
                page,
                remaining: self.latency_ticks,
            },
        );
        self.pages.insert(page, ticket);
        Ok(ticket)
    }

    fn ready(&self, ticket: ReadTicket) -> bool {
        self.outstanding
            .get(&ticket)
            .is_some_and(|t| t.remaining == 0)
    }

    fn release(&mut self, page: PageId) {
        if let Some(ticket) = self.pages.remove(&page) {
            self.outstanding.remove(&ticket);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_dram_ticket_lifecycle() {
        let mut backend = PinnedDramBackend::new();
        let mut buf = [1u8; 16];
        let t = backend.prefetch(42, &mut buf).unwrap();
        assert!(backend.ready(t));
        assert_eq!(buf, [0u8; 16]);
        assert_eq!(backend.outstanding_len(), 1);
        backend.release(42);
        assert!(!backend.ready(t));
        assert_eq!(backend.outstanding_len(), 0);
    }

    #[test]
    fn cxl_and_pim_unimplemented_are_degraded() {
        let mut cxl = CxlMemBackend::unimplemented();
        let mut buf = [0u8; 4];
        let err = cxl.prefetch(1, &mut buf).unwrap_err();
        assert!(matches!(
            err,
            AllpacaError::Degraded(DegradedError::FeatureDisabled(_))
        ));
        let mut pim = PimBackend::unimplemented();
        let err = pim.prefetch(1, &mut buf).unwrap_err();
        assert!(matches!(
            err,
            AllpacaError::Degraded(DegradedError::FeatureDisabled(_))
        ));
    }

    #[test]
    fn cxl_sim_delayed_ready_after_ticks() {
        let mut cxl = CxlMemBackend::simulated(2);
        cxl.put(7, vec![9, 8, 7, 6]);
        let mut buf = [0u8; 4];
        let t = cxl.prefetch(7, &mut buf).unwrap();
        assert_eq!(buf, [9, 8, 7, 6]);
        assert!(!cxl.ready(t));
        cxl.tick();
        assert!(!cxl.ready(t));
        cxl.tick();
        assert!(cxl.ready(t));
        cxl.release(7);
        assert!(!cxl.ready(t));
    }

    #[test]
    fn pim_sim_checksum_and_delayed_ready() {
        let mut pim = PimBackend::simulated(1);
        let payload = vec![1u8, 2, 3, 4, 5];
        pim.put(3, payload.clone());
        let mut buf = [0u8; 5];
        let t = pim.prefetch(3, &mut buf).unwrap();
        let mut expected = payload;
        expected.resize(5, 0);
        pim_xor_checksum(&mut expected);
        assert_eq!(&buf[..], expected.as_slice());
        assert!(!pim.ready(t));
        pim.tick();
        assert!(pim.ready(t));
    }
}
