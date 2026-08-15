//! CPU fallback for every kernel the control plane may call.

mod kernels;

pub use kernels::{
    attention, copy_async, gemm, matmul, matmul_nt, rmsnorm, rope, silu, softmax, CpuBackend,
};
