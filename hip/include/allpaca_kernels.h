#ifndef ALLPACA_KERNELS_H
#define ALLPACA_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ALLPACA_OK 0
#define ALLPACA_ERR_NOT_IMPLEMENTED 1
#define ALLPACA_ERR_INVALID 2
#define ALLPACA_ERR_DEVICE 3

typedef void *allpaca_stream_t;

/* Flat Kernel ABI. No scheduling or memory policy lives here. */
int allpaca_gemm(const float *a, const float *b, float *c, int32_t m, int32_t n, int32_t k,
                 allpaca_stream_t stream);

int allpaca_attention(const float *q, const float *k, const float *v, float *out, int32_t n_q,
                      int32_t n_kv, int32_t n_heads, int32_t n_kv_heads, int32_t head_dim,
                      int32_t causal, allpaca_stream_t stream);

int allpaca_copy_async(void *dst, const void *src, size_t bytes, allpaca_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif
