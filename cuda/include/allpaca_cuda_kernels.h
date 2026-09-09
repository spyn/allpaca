#ifndef ALLPACA_CUDA_KERNELS_H
#define ALLPACA_CUDA_KERNELS_H

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

typedef void *allpaca_cuda_stream_t;

/* Flat Kernel ABI for the CUDA backend. Prefixed so HIP + CUDA can coexist
 * in one process without duplicate `allpaca_gemm*` symbols (ADR 0001). */

int allpaca_cuda_device_count(int32_t *count);

int allpaca_cuda_device_props(int32_t id, char *name, int32_t name_len, uint64_t *memory_bytes,
                              double *bandwidth_gbps);

int allpaca_cuda_malloc(void **ptr, size_t bytes, int32_t device);

int allpaca_cuda_free(void *ptr);

int allpaca_cuda_memcpy(void *dst, const void *src, size_t bytes, int32_t kind,
                        allpaca_cuda_stream_t stream);

int allpaca_cuda_set_device(int32_t device);

int allpaca_cuda_stream_synchronize(allpaca_cuda_stream_t stream);

/* C[m,n] = A[m,k] * B[n,k]^T (row-major). Device pointers. */
int allpaca_cuda_gemm_nt(const float *a, const float *b, float *c, int32_t m, int32_t n, int32_t k,
                         allpaca_cuda_stream_t stream);

int allpaca_cuda_residual_add(const float *x, const float *y, float *out, int32_t n,
                              allpaca_cuda_stream_t stream);

int allpaca_cuda_rmsnorm(const float *x, const float *w, float *out, int32_t n, float eps,
                         allpaca_cuda_stream_t stream);

int allpaca_cuda_silu_mul(const float *gate, const float *up, float *out, int32_t n,
                          allpaca_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* ALLPACA_CUDA_KERNELS_H */
