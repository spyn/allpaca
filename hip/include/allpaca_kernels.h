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

#define ALLPACA_MEMCPY_H2D 0
#define ALLPACA_MEMCPY_D2H 1
#define ALLPACA_MEMCPY_D2D 2
#define ALLPACA_MEMCPY_DEFAULT 3

typedef void *allpaca_stream_t;

/* Flat Kernel ABI. No scheduling or memory policy lives here. */

int allpaca_hip_device_count(int32_t *count);

int allpaca_hip_device_props(int32_t id, char *name, int32_t name_len, uint64_t *memory_bytes,
                             double *bandwidth_gbps);

int allpaca_hip_can_p2p(int32_t src, int32_t dst, int32_t *can);

int allpaca_hip_malloc(void **ptr, size_t bytes, int32_t device);

int allpaca_hip_free(void *ptr);

int allpaca_hip_host_malloc(void **ptr, size_t bytes);

int allpaca_hip_host_free(void *ptr);

int allpaca_hip_memcpy(void *dst, const void *src, size_t bytes, int32_t kind,
                       allpaca_stream_t stream);

int allpaca_hip_memcpy_peer(void *dst, int32_t dst_dev, const void *src, int32_t src_dev,
                            size_t bytes, allpaca_stream_t stream);

int allpaca_hip_benchmark(int32_t src, int32_t dst, size_t bytes, double *p2p_gbps,
                          double *host_gbps);

int allpaca_hip_mem_used(int32_t device, uint64_t *used_bytes);

int allpaca_gemm(const float *a, const float *b, float *c, int32_t m, int32_t n, int32_t k,
                 allpaca_stream_t stream);

/* C[m,n] = A[m,k] * B[n,k]^T  (row-major). Device pointers. */
int allpaca_gemm_nt(const float *a, const float *b, float *c, int32_t m, int32_t n, int32_t k,
                    allpaca_stream_t stream);

int allpaca_attention(const float *q, const float *k, const float *v, float *out, int32_t n_q,
                      int32_t n_kv, int32_t n_heads, int32_t n_kv_heads, int32_t head_dim,
                      int32_t causal, allpaca_stream_t stream);

int allpaca_copy_async(void *dst, const void *src, size_t bytes, allpaca_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif
