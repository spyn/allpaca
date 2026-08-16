//! CPU fallback for every kernel the control plane may call.

mod kernels;

pub use kernels::{
    attention, attention_with_mask, copy_async, dequant_q4_0, dequant_q8_0, f16_to_f32, f32_to_f16,
    gemm, matmul, matmul_nt, quantize_q4_0, quantize_q8_0, rmsnorm, rope, silu, softmax, CpuBackend,
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
};
