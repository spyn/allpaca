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
#define ALLPACA_MEMCPY_D2H_ASYNC 4

typedef void *allpaca_cuda_stream_t;

/* Flat Kernel ABI for the CUDA backend. Prefixed so HIP + CUDA can coexist
 * in one process without duplicate `allpaca_gemm*` symbols (ADR 0001). */

int allpaca_cuda_device_count(int32_t *count);

int allpaca_cuda_device_props(int32_t id, char *name, int32_t name_len, uint64_t *memory_bytes,
                              double *bandwidth_gbps);

/* Compute capability as `sm_86` / `sm_120` plus L2 cache bytes (HIP device_arch). */
int allpaca_cuda_device_arch(int32_t id, char *arch, int32_t arch_len, int32_t *l2_bytes);

int allpaca_cuda_malloc(void **ptr, size_t bytes, int32_t device);

int allpaca_cuda_free(void *ptr);

/* Stream-ordered free (cudaFreeAsync). Null stream falls back to cudaFree. */
int allpaca_cuda_free_async(void *ptr, allpaca_cuda_stream_t stream);

int allpaca_cuda_host_malloc(void **ptr, size_t bytes);

int allpaca_cuda_host_free(void *ptr);

/* Pin existing host pages for DMA (HIP host_register). flags: cudaHostRegister* bits. */
int allpaca_cuda_host_register(void *ptr, size_t bytes, unsigned int flags);

int allpaca_cuda_host_unregister(void *ptr);

int allpaca_cuda_memcpy(void *dst, const void *src, size_t bytes, int32_t kind,
                        allpaca_cuda_stream_t stream);

/* Batch of `count` 1D copies, same `kind`, one stream (per-copy cudaMemcpyAsync). */
int allpaca_cuda_memcpy_batch(void **dsts, void **srcs, size_t *sizes, size_t count, int32_t kind,
                              allpaca_cuda_stream_t stream);

int allpaca_cuda_set_device(int32_t device);

int allpaca_cuda_can_p2p(int32_t src, int32_t dst, int32_t *can);

int allpaca_cuda_enable_peer_access(int32_t src, int32_t dst);

int allpaca_cuda_memcpy_peer(void *dst, int32_t dst_dev, const void *src, int32_t src_dev,
                             size_t bytes, allpaca_cuda_stream_t stream);

/* Driver-measured free/total VRAM (includes other processes). */
int allpaca_cuda_mem_info(int32_t device, uint64_t *free_bytes, uint64_t *total_bytes);

int allpaca_cuda_mem_used(int32_t device, uint64_t *used_bytes);

int allpaca_cuda_stream_synchronize(allpaca_cuda_stream_t stream);

/* Non-blocking compute/copy stream (mirrors allpaca_hip_stream_create). */
int allpaca_cuda_stream_create(allpaca_cuda_stream_t *stream);
int allpaca_cuda_stream_destroy(allpaca_cuda_stream_t stream);

int allpaca_cuda_stream_wait_event(allpaca_cuda_stream_t stream, void *event);

int allpaca_cuda_event_create(void **event);
int allpaca_cuda_event_create_timing(void **event);
int allpaca_cuda_event_destroy(void *event);
int allpaca_cuda_event_record(void *event, allpaca_cuda_stream_t stream);
int allpaca_cuda_event_elapsed_us(void *start, void *stop, double *us);

/* C[m,n] = A[m,k] * B[n,k]^T (row-major). Device pointers. */
int allpaca_cuda_gemm_nt(const float *a, const float *b, float *c, int32_t m, int32_t n, int32_t k,
                         allpaca_cuda_stream_t stream);

int allpaca_cuda_residual_add(const float *x, const float *y, float *out, int32_t n,
                              allpaca_cuda_stream_t stream);

int allpaca_cuda_rmsnorm(const float *x, const float *w, float *out, int32_t n, float eps,
                         allpaca_cuda_stream_t stream);

int allpaca_cuda_silu_mul(const float *gate, const float *up, float *out, int32_t n,
                          allpaca_cuda_stream_t stream);

/* Gather one token row from F32 embed table[vocab, dim] -> out[dim]. */
int allpaca_cuda_embed_gather(const float *table, int32_t token, float *out, int32_t dim,
                              allpaca_cuda_stream_t stream);

/* In-place interleaved RoPE on x[n_heads, head_dim]. Matches HIP allpaca_rope. */
int allpaca_cuda_rope(float *x, int32_t n_heads, int32_t head_dim, int32_t pos, float theta,
                      allpaca_cuda_stream_t stream);

/* Online-softmax attention (decode/prefill). q:[n_q,n_heads,d] k/v:[n_kv,n_kv_heads,d].
 * Matches HIP allpaca_attention argument order. */
int allpaca_cuda_attention(const float *q, const float *k, const float *v, float *out, int32_t n_q,
                           int32_t n_kv, int32_t n_heads, int32_t n_kv_heads, int32_t head_dim,
                           int32_t causal, allpaca_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* ALLPACA_CUDA_KERNELS_H */
