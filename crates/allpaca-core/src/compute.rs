use allpaca_hip::HipWorkspace;
use allpaca_rpc::RpcSession;

/// Compute backend for a layer run. Local HIP stays pointer-based; RPC is a
/// remote session. CPU fallback is the existing host path (no variant).
pub enum ComputeDev<'a> {
    Hip(&'a HipWorkspace),
    Rpc(&'a mut RpcSession),
}

impl<'a> ComputeDev<'a> {
    pub fn is_rpc(&self) -> bool {
        matches!(self, Self::Rpc(_))
    }
}
