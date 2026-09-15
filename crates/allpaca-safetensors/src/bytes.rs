use memmap2::Mmap;

/// Backing store for a SafeTensors file. Mapped files stay on disk; tests can
/// pass an owned buffer.
pub(crate) enum FileBytes {
    Owned(Vec<u8>),
    Mapped(Mmap),
}

impl FileBytes {
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v.as_slice(),
            Self::Mapped(m) => m.as_ref(),
        }
    }
}
