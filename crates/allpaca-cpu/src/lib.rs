//! CPU fallback for every kernel the control plane may call.

mod kernels;
mod lowbit;
mod mxfp4_expert;
mod p48fp4;

pub use lowbit::{
    dequant_q1_0, dequant_q2_0, quantize_q1_0, quantize_q2_0, Q1_0_BLOCK_BYTES, Q1_0_BLOCK_ELEMS,
    Q2_0_BLOCK_BYTES, Q2_0_BLOCK_ELEMS,
};
pub use mxfp4_expert::{mxfp4_swiglu_experts, Mxfp4Expert};
pub use p48fp4::{
    dequant_p48fp4, e4m3_to_f32, p48fp4_interleave, quantize_p48fp4, P48FP4_BLOCK_BYTES,
    P48FP4_BLOCK_ELEMS,
};
pub use kernels::{
    det_expf, det_silu_mul,
    attention, attention_softcap, attention_with_mask, attention_with_mask_softcap, bf16_to_f32,
    copy_async, dequant_iq4_nl, dequant_iq4_xs, dequant_mxfp4, dequant_q2_k, dequant_q3_k,
    dequant_q4_0, dequant_q4_1, dequant_q4_k, dequant_q5_0, dequant_q5_1, dequant_q5_k,
    dequant_q6_k, dequant_q8_0, dequant_q8_1, dequant_rocmfp4, dequant_rocmfp4_fast, dequant_tq1_0,
    dequant_tq2_0,
    e8m0_to_fp32, e8m0_to_fp32_half, f16_to_f32, f32_to_bf16, f32_to_f16, gelu, gemm, matmul,
    matmul_nt, quantize_mxfp4, quantize_q4_0, quantize_q4_k, quantize_q5_k, quantize_q6_k,
    quantize_q8_0, quantize_rocmfp4, quantize_rocmfp4_fast, quantize_tq2_0, rmsnorm, rocmfp4_scale, rocmfp4_scale_is_valid,
    rope, rope_imrope_neox, rope_imrope_neox_at, rope_imrope_neox_heads, rope_imrope_neox_heads_at, rope_offset, rope_offset_back, silu, softmax,
    CpuBackend, IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES, IQ4_XS_BLOCK_ELEMS,
    KVALUES_IQ4NL, KVALUES_MXFP4, MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS, Q2_K_BLOCK_BYTES,
    Q2_K_BLOCK_ELEMS, Q3_K_BLOCK_BYTES, Q3_K_BLOCK_ELEMS, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS,
    Q4_1_BLOCK_BYTES, Q4_1_BLOCK_ELEMS, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q5_0_BLOCK_BYTES,
    Q5_0_BLOCK_ELEMS, Q5_1_BLOCK_BYTES, Q5_1_BLOCK_ELEMS, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMS,
    Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS, Q8_1_BLOCK_BYTES,
    Q8_1_BLOCK_ELEMS, ROCMFP4_BLOCK_BYTES, ROCMFP4_FAST_BLOCK_BYTES, ROCMFP4_BLOCK_ELEMS, ROCMFP4_CODEBOOK,
    TQ1_0_BLOCK_BYTES, TQ1_0_BLOCK_ELEMS, TQ2_0_BLOCK_BYTES, TQ2_0_BLOCK_ELEMS,
};
