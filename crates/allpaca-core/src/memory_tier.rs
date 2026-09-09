//! Future memory-tier backends (OPPORTUNITY §12 CXL/PIM interface).
//!
//! Hardware CXL/PIM remains Out; this trait lets the scheduler evolve without
//! embedding device assumptions into model code.

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

/// Future CXL.mem path — not implemented on this host.
#[derive(Debug, Default)]
pub struct CxlMemBackend;

impl MemoryTierBackend for CxlMemBackend {
    fn prefetch(
        &mut self,
        _page: PageId,
        _dst: &mut [u8],
    ) -> Result<ReadTicket, AllpacaError> {
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CxlMemBackend not implemented".into(),
        )))
    }

    fn ready(&self, _ticket: ReadTicket) -> bool {
        false
    }

    fn release(&mut self, _page: PageId) {}
}

/// Future PIM path — not implemented on this host.
#[derive(Debug, Default)]
pub struct PimBackend;

impl MemoryTierBackend for PimBackend {
    fn prefetch(
        &mut self,
        _page: PageId,
        _dst: &mut [u8],
    ) -> Result<ReadTicket, AllpacaError> {
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "PimBackend not implemented".into(),
        )))
    }

    fn ready(&self, _ticket: ReadTicket) -> bool {
        false
    }

    fn release(&mut self, _page: PageId) {}
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
    fn cxl_and_pim_are_degraded() {
        let mut cxl = CxlMemBackend;
        let mut buf = [0u8; 4];
        let err = cxl.prefetch(1, &mut buf).unwrap_err();
        assert!(matches!(
            err,
            AllpacaError::Degraded(DegradedError::FeatureDisabled(_))
        ));
        let mut pim = PimBackend;
        let err = pim.prefetch(1, &mut buf).unwrap_err();
        assert!(matches!(
            err,
            AllpacaError::Degraded(DegradedError::FeatureDisabled(_))
        ));
    }
}
