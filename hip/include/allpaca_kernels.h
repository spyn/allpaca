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
#define ALLPACA_MEMCPY_D2H_ASYNC 4

typedef void *allpaca_stream_t;

/* Flat Kernel ABI. No scheduling or memory policy lives here. */

int allpaca_hip_device_count(int32_t *count);

int allpaca_hip_device_props(int32_t id, char *name, int32_t name_len, uint64_t *memory_bytes,
                             double *bandwidth_gbps);

int allpaca_hip_can_p2p(int32_t src, int32_t dst, int32_t *can);

int allpaca_hip_enable_peer_access(int32_t src, int32_t dst);

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

int allpaca_hip_set_device(int32_t device);

int allpaca_hip_stream_synchronize(allpaca_stream_t stream);

int allpaca_hip_event_create(void **event);
int allpaca_hip_event_destroy(void *event);
int allpaca_hip_event_record(void *event, allpaca_stream_t stream);
int allpaca_hip_event_elapsed_us(void *start, void *stop, double *us);

int allpaca_gemm(const float *a, const float *b, float *c, int32_t m, int32_t n, int32_t k,
                 allpaca_stream_t stream);

/* C[m,n] = A[m,k] * B[n,k]^T  (row-major). Device pointers.
 * Decode (M<=8) uses cache-aware A-tile GEMV (§12). */
int allpaca_gemm_nt(const float *a, const float *b, float *c, int32_t m, int32_t n, int32_t k,
                    allpaca_stream_t stream);

/* Untiled warp GEMV baseline for §12 before/after tracers. */
int allpaca_gemm_nt_untiled(const float *a, const float *b, float *c, int32_t m, int32_t n,
                            int32_t k, allpaca_stream_t stream);

/* C[m,n] = A[m,k] * dequant(B_q8_0[n,k])^T. Decode default is float A × Q8.
 * B is 36-byte staged Q8_0, not raw GGUF. M>8 uses a tiled kernel. */
int allpaca_gemm_nt_q8_0(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                         int32_t k, allpaca_stream_t stream);

/* Integer A×Q8 GEMV (quantize A tiles, int32 MAC). Not the decode default. */
int allpaca_gemm_nt_q8_0_intdot(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                int32_t k, allpaca_stream_t stream);

/* C[m,n] = A[m,k] * dequant(B_q4_k[n,k])^T. B is native GGUF Q4_K super-blocks.
 * Default is cache-aware A-tile GEMV (§12). */
int allpaca_gemm_nt_q4_k(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                         int32_t k, allpaca_stream_t stream);

/* Untiled one-thread-per-column Q4_K GEMV (before §12 tile). Tracer / A-B only. */
int allpaca_gemm_nt_q4_k_untiled(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                 int32_t k, allpaca_stream_t stream);

/* C[m,n] = A[m,k] * dequant(B_q6_k[n,k])^T. Native 210-byte Q6_K super-blocks (GEM mixed-quant). */
int allpaca_gemm_nt_q6_k(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                         int32_t k, allpaca_stream_t stream);

/* C[m,n] = A[m,k] * dequant(B_q5_k[n,k])^T. Native 176-byte Q5_K super-blocks (GEM mixed-quant). */
int allpaca_gemm_nt_q5_k(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                         int32_t k, allpaca_stream_t stream);

/* C[m,n] = A[m,k] * dequant(B_f16[n,k])^T. B is packed IEEE F16, row-major [n,k].
 * Decode (M<=8) uses cache-aware A-tile GEMV (§12); M>8 uses a tiled prefill kernel. */
int allpaca_gemm_nt_f16(const float *a, const uint16_t *b, float *c, int32_t m, int32_t n,
                        int32_t k, allpaca_stream_t stream);

/* Untiled F16 GEMV baseline for §12 before/after tracers. */
int allpaca_gemm_nt_f16_untiled(const float *a, const uint16_t *b, float *c, int32_t m, int32_t n,
                                int32_t k, allpaca_stream_t stream);

int allpaca_attention(const float *q, const float *k, const float *v, float *out, int32_t n_q,
                      int32_t n_kv, int32_t n_heads, int32_t n_kv_heads, int32_t head_dim,
                      int32_t causal, allpaca_stream_t stream);

/* Same layout as allpaca_attention. mask is [n_q, n_kv] uint8 (nonzero = attend). */
int allpaca_attention_masked(const float *q, const float *k, const float *v, float *out,
                             int32_t n_q, int32_t n_kv, int32_t n_heads, int32_t n_kv_heads,
                             int32_t head_dim, const uint8_t *mask, allpaca_stream_t stream);

int allpaca_copy_async(void *dst, const void *src, size_t bytes, allpaca_stream_t stream);

int allpaca_rmsnorm(const float *x, const float *weight, float *out, int32_t n, float eps,
                    allpaca_stream_t stream);

/* out[rows, n] = RMSNorm(x[rows, n], weight[n]). One launch; one block per row. */
int allpaca_rmsnorm_rows(const float *x, const float *weight, float *out, int32_t rows, int32_t n,
                         float eps, allpaca_stream_t stream);

int allpaca_rope(float *x, int32_t n_heads, int32_t head_dim, int32_t pos, float theta,
                 allpaca_stream_t stream);

int allpaca_silu_mul(float *gate, const float *up, int32_t n, allpaca_stream_t stream);

int allpaca_embed_gather(const float *table, int32_t token, float *out, int32_t dim,
                         allpaca_stream_t stream);

/* out[n_tok, dim] = table[tokens[i]]. Decode stays on allpaca_embed_gather. */
int allpaca_embed_gather_rows(const float *table, const int32_t *tokens, float *out, int32_t n_tok,
                              int32_t dim, allpaca_stream_t stream);

int allpaca_embed_gather_f16(const uint16_t *table, int32_t token, float *out, int32_t dim,
                             allpaca_stream_t stream);

int allpaca_embed_gather_f16_rows(const uint16_t *table, const int32_t *tokens, float *out,
                                  int32_t n_tok, int32_t dim, allpaca_stream_t stream);

int allpaca_residual_add(float *x, const float *residual, int32_t n, allpaca_stream_t stream);

int allpaca_scale_f32(float *x, int32_t n, float scale, allpaca_stream_t stream);

/* x += residual, then RMSNorm(x) -> out. One launch for the decode epilogue. */
int allpaca_residual_rmsnorm(float *x, const float *residual, const float *weight, float *out,
                             int32_t n, float eps, allpaca_stream_t stream);

/* RoPE on Q and K in one launch. Decode uses n_tok=1 via allpaca_rope_qk. */
int allpaca_rope_qk(float *q, float *k, int32_t n_q_heads, int32_t n_k_heads, int32_t head_dim,
                    int32_t pos, float theta, allpaca_stream_t stream);

/* Prefill: Q/K are [n_tok, heads*head_dim]; position is start_pos + token. One launch. */
int allpaca_rope_qk_rows(float *q, float *k, int32_t n_tok, int32_t n_q_heads, int32_t n_k_heads,
                         int32_t head_dim, int32_t start_pos, float theta, allpaca_stream_t stream);

int allpaca_dequant_q8_0(const uint8_t *blocks, float *out, int32_t n, allpaca_stream_t stream);

int allpaca_embed_gather_q8_0(const uint8_t *table, int32_t token, float *out, int32_t dim,
                              allpaca_stream_t stream);

int allpaca_embed_gather_q8_0_rows(const uint8_t *table, const int32_t *tokens, float *out,
                                   int32_t n_tok, int32_t dim, allpaca_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif
