#![allow(dead_code)]

use std::sync::atomic::{AtomicU32, Ordering};

use allpaca_types::{AllpacaError, TransientError};

const FREE: u32 = 0;
const GPU_WRITING: u32 = 1;
const READY: u32 = 2;
const CPU_READING: u32 = 3;

struct LogitSlot {
    data: Vec<f32>,
    bytes: Vec<u8>,
    state: AtomicU32,
}

/// Pinned-style host ring used for async D2H logits then CPU sampling.
pub struct LogitRing {
    slots: Vec<LogitSlot>,
    cursor: usize,
}

impl LogitRing {
    pub fn new(slots: usize, vocab: usize) -> Self {
        let n = slots.max(1);
        let vocab = vocab.max(1);
        let mut ring = Vec::with_capacity(n);
        for _ in 0..n {
            ring.push(LogitSlot {
                data: vec![0.0; vocab],
                bytes: vec![0u8; vocab * 4],
                state: AtomicU32::new(FREE),
            });
        }
        Self {
            slots: ring,
            cursor: 0,
        }
    }

    pub fn resize_vocab(&mut self, vocab: usize) {
        let vocab = vocab.max(1);
        for slot in &mut self.slots {
            slot.data.resize(vocab, 0.0);
            slot.bytes.resize(vocab * 4, 0);
            slot.state.store(FREE, Ordering::SeqCst);
        }
    }

    /// Copy device logits into the next free slot (CPU path uses `copy_async`).
    pub fn publish(&mut self, logits: &[f32]) -> Result<Vec<f32>, AllpacaError> {
        let n = self.slots.len();
        for _ in 0..n {
            let idx = self.cursor;
            self.cursor = (self.cursor + 1) % n;
            let slot = &mut self.slots[idx];
            if slot
                .state
                .compare_exchange(FREE, GPU_WRITING, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                let copied = write_slot(slot, logits)?;
                return Ok(copied);
            }
        }
        Err(AllpacaError::Transient(TransientError::Backpressure(
            "logit ring saturated".into(),
        )))
    }

    pub fn release_current(&mut self) {
        let idx = if self.cursor == 0 {
            self.slots.len() - 1
        } else {
            self.cursor - 1
        };
        self.slots[idx].state.store(FREE, Ordering::SeqCst);
    }
}

fn write_slot(slot: &mut LogitSlot, logits: &[f32]) -> Result<Vec<f32>, AllpacaError> {
    if slot.data.len() < logits.len() {
        slot.data.resize(logits.len(), 0.0);
        slot.bytes.resize(logits.len() * 4, 0);
    }
    slot.data[..logits.len()].copy_from_slice(logits);
    slot.state.store(READY, Ordering::SeqCst);
    slot.state.store(CPU_READING, Ordering::SeqCst);
    Ok(slot.data[..logits.len()].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_copies_logits() {
        let mut ring = LogitRing::new(2, 4);
        let published = ring.publish(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(published, &[1.0, 2.0, 3.0, 4.0]);
        ring.release_current();
    }
}
