#include "../include/allpaca_kernels.h"

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

extern "C" int allpaca_gemm(const float *a, const float *b, float *c, int32_t m, int32_t n,
                            int32_t k, allpaca_stream_t stream) {
    (void)a;
    (void)b;
    (void)c;
    (void)m;
    (void)n;
    (void)k;
    (void)stream;
    /* TODO: HIP GEMM (e.g. rocBLAS / custom kernel). Control plane falls back to allpaca-cpu. */
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_attention(const float *q, const float *k, const float *v, float *out,
                                 int32_t n_q, int32_t n_kv, int32_t n_heads, int32_t n_kv_heads,
                                 int32_t head_dim, int32_t causal, allpaca_stream_t stream) {
    (void)q;
    (void)k;
    (void)v;
    (void)out;
    (void)n_q;
    (void)n_kv;
    (void)n_heads;
    (void)n_kv_heads;
    (void)head_dim;
    (void)causal;
    (void)stream;
    /* TODO: HIP attention kernel. */
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_copy_async(void *dst, const void *src, size_t bytes,
                                  allpaca_stream_t stream) {
    (void)dst;
    (void)src;
    (void)bytes;
    (void)stream;
#ifdef __HIP_PLATFORM_AMD__
    hipStream_t s = static_cast<hipStream_t>(stream);
    hipError_t err = hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, s);
    return err == hipSuccess ? ALLPACA_OK : ALLPACA_ERR_DEVICE;
#else
    return ALLPACA_ERR_NOT_IMPLEMENTED;
#endif
}
