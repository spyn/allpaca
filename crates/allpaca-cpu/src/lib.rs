//! CPU fallback for every kernel the control plane may call.

mod kernels;

pub use kernels::{
    attention, attention_with_mask, copy_async, dequant_q2_k, dequant_q3_k, dequant_q4_0,
    dequant_q4_1, dequant_q4_k, dequant_q5_0, dequant_q5_1, dequant_q5_k, dequant_q6_k,
    dequant_q8_0, dequant_q8_1, dequant_rocmfp4, f16_to_f32, f32_to_f16, gemm, matmul, matmul_nt,
    quantize_q4_0, quantize_q4_k, quantize_q5_k, quantize_q6_k, quantize_q8_0, quantize_rocmfp4,
    rmsnorm, rocmfp4_scale, rocmfp4_scale_is_valid, rope, rope_imrope_neox, rope_imrope_neox_heads,
    silu, softmax, CpuBackend, Q2_K_BLOCK_BYTES, Q2_K_BLOCK_ELEMS, Q3_K_BLOCK_BYTES,
    Q3_K_BLOCK_ELEMS, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS, Q4_1_BLOCK_BYTES, Q4_1_BLOCK_ELEMS,
    Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMS, Q5_1_BLOCK_BYTES,
    Q5_1_BLOCK_ELEMS, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS,
    Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS, Q8_1_BLOCK_BYTES, Q8_1_BLOCK_ELEMS, ROCMFP4_BLOCK_BYTES,
    ROCMFP4_BLOCK_ELEMS, ROCMFP4_CODEBOOK,
};
