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

int allpaca_hip_device_arch(int32_t id, char *arch, int32_t arch_len, int32_t *l2_bytes);

int allpaca_hip_can_p2p(int32_t src, int32_t dst, int32_t *can);

int allpaca_hip_enable_peer_access(int32_t src, int32_t dst);

int allpaca_hip_malloc(void **ptr, size_t bytes, int32_t device);

int allpaca_hip_memset(void *ptr, int value, size_t bytes, allpaca_stream_t stream);

int allpaca_hip_free(void *ptr);

/* Stream-ordered free (hipFreeAsync). Null stream falls back to hipFree. */
int allpaca_hip_free_async(void *ptr, allpaca_stream_t stream);

int allpaca_hip_host_malloc(void **ptr, size_t bytes);

int allpaca_hip_host_free(void *ptr);

int allpaca_hip_host_register(void *ptr, size_t bytes, unsigned int flags);

int allpaca_hip_host_unregister(void *ptr);

/* Windows DirectStorage 1.3 buffer backed by a shareable D3D12 resource and
 * imported into HIP external memory. `handle` owns both API objects. */
int allpaca_dstorage_alloc(void **handle, void **hip_ptr, size_t bytes, int32_t device);

int allpaca_dstorage_read(void *handle, const uint16_t *path, uint64_t file_offset,
                          uint64_t buffer_offset, size_t bytes);

int allpaca_dstorage_free(void *handle);

int allpaca_hip_memcpy(void *dst, const void *src, size_t bytes, int32_t kind,
                       allpaca_stream_t stream);

/* Batch of `count` 1D copies, all of the same `kind`, on one stream.
 * Uses hipMemcpyBatchAsync when the loaded runtime exports it (ROCm 7.1+),
 * otherwise falls back to per-copy hipMemcpyAsync on the same stream. The
 * observable ordering is identical either way. */
int allpaca_hip_memcpy_batch(void **dsts, void **srcs, size_t *sizes, size_t count, int32_t kind,
                             allpaca_stream_t stream);

int allpaca_hip_memset(void *dst, int32_t value, size_t bytes, allpaca_stream_t stream);

/* Non-zero when the runtime exports hipMemcpyBatchAsync (telemetry / tests). */
int allpaca_hip_has_memcpy_batch(void);

int allpaca_hip_memcpy_peer(void *dst, int32_t dst_dev, const void *src, int32_t src_dev,
                            size_t bytes, allpaca_stream_t stream);

/* Read-only stride touch so WDDM faults peer-GPU weights in before decode. */
int allpaca_hip_touch_resident(void *ptr, size_t bytes, int32_t device);

int allpaca_hip_benchmark(int32_t src, int32_t dst, size_t bytes, double *p2p_gbps,
                          double *host_gbps);

int allpaca_hip_mem_used(int32_t device, uint64_t *used_bytes);

/* Driver-measured VRAM for `device`. Unlike the engine's own allocation
 * counter this includes other processes (compositor, browser), which is what
 * the scheduler needs before it grows the expert pool. */
int allpaca_hip_mem_info(int32_t device, uint64_t *free_bytes, uint64_t *total_bytes);

/* Windows: dedicated VRAM on `device` not resident for any process (PDH
 * "GPU Adapter Memory" Dedicated Usage vs the adapter's DedicatedVideoMemory).
 * hipMemGetInfo there does not see the desktop's share, and WDDM demotes
 * allocations past what is really left to shared system memory. Other
 * platforms: ALLPACA_ERR_NOT_IMPLEMENTED. */
int allpaca_hip_local_headroom(int32_t device, uint64_t *bytes);

int allpaca_hip_set_device(int32_t device);

int allpaca_hip_stream_synchronize(allpaca_stream_t stream);

/* Non-blocking side stream (expert H2D) so copies overlap the null compute
 * stream instead of serializing behind it. */
int allpaca_hip_stream_create(allpaca_stream_t *stream);
int allpaca_hip_stream_destroy(allpaca_stream_t stream);

/* Make `stream` wait for `event` without blocking the host (cross-stream join). */
int allpaca_hip_stream_wait_event(allpaca_stream_t stream, void *event);

int allpaca_hip_event_create(void **event);
int allpaca_hip_event_destroy(void *event);
int allpaca_hip_event_record(void *event, allpaca_stream_t stream);
int allpaca_hip_event_synchronize(void *event);
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

/* C[m,n] = A[m,k] * dequant(B_tq2_0[n,k])^T. B is native GGUF TQ2_0 blocks (66 B). */
int allpaca_gemm_nt_tq2_0(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                          int32_t k, allpaca_stream_t stream);

/* C[m,n] = A[m,k] * dequant(B_tq1_0[n,k])^T. B is native GGUF TQ1_0 blocks (54 B). */
int allpaca_gemm_nt_tq1_0(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                          int32_t k, allpaca_stream_t stream);

/* C[m,n] = A[m,k] * dequant(B_rocmfp4[n,k])^T. B is native 18-byte ROCmFP4
 * blocks (ggml_type_id=100): qs[16] then two unsigned-E4M3 half-block scales.
 * Default is the cache-aware A-tile GEMV (§12). */
int allpaca_gemm_nt_rocmfp4(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                            int32_t k, allpaca_stream_t stream);

/* Untiled one-thread-per-column ROCmFP4 GEMV (before §12 tile). Tracer / A-B only. */
int allpaca_gemm_nt_rocmfp4_untiled(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                    int32_t k, allpaca_stream_t stream);

/* C[m,n] = A[m,k] * dequant(B_q6_k[n,k])^T. Native 210-byte Q6_K super-blocks (GEM mixed-quant).
 * Default tile cols from §12 picker; use _cols / _untiled for tracers. */
int allpaca_gemm_nt_q6_k(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                         int32_t k, allpaca_stream_t stream);
int allpaca_gemm_nt_q6_k_cols(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                              int32_t k, int32_t cols, allpaca_stream_t stream);
int allpaca_gemm_nt_q6_k_untiled(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                 int32_t k, allpaca_stream_t stream);

/* C[m,n] = A[m,k] * dequant(B_q3_k[n,k])^T. Native 110-byte Q3_K super-blocks, k % 256 == 0. */
int allpaca_gemm_nt_q3_k(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                         int32_t k, allpaca_stream_t stream);

/* C[m,n] = A[m,k] * dequant(B_q5_k[n,k])^T. Native 176-byte Q5_K super-blocks (GEM mixed-quant). */
int allpaca_gemm_nt_q5_k(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                         int32_t k, allpaca_stream_t stream);
int allpaca_gemm_nt_q5_k_cols(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                              int32_t k, int32_t cols, allpaca_stream_t stream);
int allpaca_gemm_nt_q5_k_untiled(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                 int32_t k, allpaca_stream_t stream);

/* C[m,n] = A[m,k] * dequant(B_f16[n,k])^T. B is packed IEEE F16, row-major [n,k].
 * Decode (M<=8) uses cache-aware A-tile GEMV (§12); M>8 uses a tiled prefill kernel. */
int allpaca_gemm_nt_f16(const float *a, const uint16_t *b, float *c, int32_t m, int32_t n,
                        int32_t k, allpaca_stream_t stream);

/* Untiled F16 GEMV baseline for §12 before/after tracers. */
int allpaca_gemm_nt_f16_untiled(const float *a, const uint16_t *b, float *c, int32_t m, int32_t n,
                                int32_t k, allpaca_stream_t stream);

/* C[m,n] = A[m,k] * dequant(B_bf16[n,k])^T. B is packed bfloat16, row-major [n,k].
 * gfx906 has no BF16 ALU: load u16, expand to F32, FMA. Decode M<=8 is A-tile GEMV. */
int allpaca_gemm_nt_bf16(const float *a, const uint16_t *b, float *c, int32_t m, int32_t n,
                         int32_t k, allpaca_stream_t stream);

int allpaca_gemm_nt_bf16_untiled(const float *a, const uint16_t *b, float *c, int32_t m, int32_t n,
                                 int32_t k, allpaca_stream_t stream);

/* C[m,n] = A[m,k] * dequant(B_mxfp4[n,k])^T. ggml MXFP4: 17-byte blocks / 32 elems. */
int allpaca_gemm_nt_mxfp4(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                          int32_t k, allpaca_stream_t stream);

int allpaca_gemm_nt_mxfp4_untiled(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                  int32_t k, allpaca_stream_t stream);

/* C[m,n] = silu(A * dequant(Bg)^T) * (A * dequant(Bu)^T). Same MXFP4 layout. */
int allpaca_mxfp4_swiglu_gate_up(const float *a, const uint8_t *bg, const uint8_t *bu, float *c,
                                 int32_t m, int32_t n, int32_t k, allpaca_stream_t stream);

int allpaca_attention(const float *q, const float *k, const float *v, float *out, int32_t n_q,
                      int32_t n_kv, int32_t n_heads, int32_t n_kv_heads, int32_t head_dim,
                      int32_t causal, allpaca_stream_t stream);

/* Online-softmax attention with a virtual sink logit per head (`sinks[n_heads]`; null = none). */
int allpaca_attention_sinks(const float *q, const float *k, const float *v, float *out, int32_t n_q,
                            int32_t n_kv, int32_t n_heads, int32_t n_kv_heads, int32_t head_dim,
                            int32_t causal, const float *sinks, allpaca_stream_t stream);

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

/* out[rows, n] = LayerNorm(x[rows, n], weight[n], bias[n]). Mean/var per row; not RMSNorm. */
int allpaca_layernorm_rows(const float *x, const float *weight, const float *bias, float *out,
                           int32_t rows, int32_t n, float eps, allpaca_stream_t stream);

int allpaca_rope(float *x, int32_t n_heads, int32_t head_dim, int32_t pos, float theta,
                 allpaca_stream_t stream);

int allpaca_silu_mul(float *gate, const float *up, int32_t n, allpaca_stream_t stream);

/* Spark2_5 FFN: gate[i] = gelu(gate[i]) * up[i] (erf GELU). */
int allpaca_gelu_mul(float *gate, const float *up, int32_t n, allpaca_stream_t stream);

/* x[i] = relu(x[i])^2 in place (Nemotron Lightning / DeepSeek gate-less MoE). */
int allpaca_relu_sqr(float *x, int32_t n, allpaca_stream_t stream);

/* Qwen gated-attn: q_full[n_heads][2*head_dim] -> q, gate each [n_heads][head_dim]. */
int allpaca_deinterleave_q_gate(const float *q_full, float *q, float *gate, int32_t n_heads,
                                int32_t head_dim, allpaca_stream_t stream);

/* x[i] *= sigmoid(gate[i]) in place. */
int allpaca_sigmoid_mul(float *x, const float *gate, int32_t n, allpaca_stream_t stream);

/* Spark2_5 head-wise gate: attn[tok,h,:] *= sigmoid(scores[tok,h]). */
int allpaca_scale_heads_sigmoid(float *attn, const float *scores, int32_t n_tok, int32_t n_heads,
                                int32_t head_dim, allpaca_stream_t stream);

/* Per-head RMSNorm: x is [n_heads, head_dim], weight is [head_dim]. */
int allpaca_rmsnorm_heads(float *x, const float *weight, int32_t n_heads, int32_t head_dim,
                          float eps, allpaca_stream_t stream);

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

/* acc[idx[i], :] += src[i, :] * w[i]. Concurrent rows (and dual streams) use
 * atomicAdd so two experts can land on the same token. */
int allpaca_scatter_add_scale_rows(const float *src, const int32_t *idx, const float *w,
                                   float *acc, int32_t m, int32_t dim, int32_t n_tok,
                                   allpaca_stream_t stream);

/* Packed HC residual layout `[n_tok][hc][dim]`. `pre`/`post` are `[n_tok][hc]`;
 * `comb` is `[n_tok][src][dst]` with `comb[t, src, dst] = comb[t*hc*hc + dst + src*hc]`.
 * `hc` is 1..=8. */
int allpaca_hc_mix_pre(const float *packed, const float *pre, float *curs, int32_t n_tok,
                       int32_t hc, int32_t dim, allpaca_stream_t stream);
int allpaca_hc_post(const float *x, const float *residual, const float *post, const float *comb,
                    float *out, int32_t n_tok, int32_t hc, int32_t dim, allpaca_stream_t stream);

int allpaca_scale_f32(float *x, int32_t n, float scale, allpaca_stream_t stream);

/* x += residual, then RMSNorm(x) -> out. One launch for the decode epilogue. */
int allpaca_residual_rmsnorm(float *x, const float *residual, const float *weight, float *out,
                             int32_t n, float eps, allpaca_stream_t stream);

/* Qwen3.5 Gated DeltaNet one-token step. q/k L2-normalized in place; q scaled
 * by 1/sqrt(head_k); state is [n_v][head_v][head_k] (V-first). */
int allpaca_gated_delta_step(float *q, float *k, const float *v, const float *decay,
                             const float *beta, float *state, float *out, int32_t n_v_heads,
                             int32_t head_v, int32_t head_k, float eps, allpaca_stream_t stream);

/* Depthwise causal conv1d + SiLU for one GDN token. kernel layout [c][k]. */
int allpaca_gdn_conv1d_step(const float *x, const float *kernel, float *conv_state, float *out,
                            int32_t qkv_dim, int32_t conv_kernel, allpaca_stream_t stream);

/* Tile K-heads to V-heads: dst[vh] = src[vh % n_k] (ggml repeat). */
int allpaca_expand_k_heads(const float *src, float *dst, int32_t n_k, int32_t n_v, int32_t head_k,
                           allpaca_stream_t stream);

/* decay = exp(softplus(alpha + dt_bias) * a); beta = sigmoid(beta) in-place. */
int allpaca_gdn_decay_beta(const float *alpha, const float *dt_bias, const float *a, float *decay,
                           float *beta, int32_t n, allpaca_stream_t stream);

/* Nemotron-H Mamba-2 one-token causal depthwise conv over x/B/C. */
int allpaca_mamba2_conv_step(const float *xbc, const float *kernel, const float *bias,
                             float *conv_state, float *out, int32_t conv_dim,
                             int32_t conv_kernel, allpaca_stream_t stream);

/* Nemotron-H Mamba-2 selective SSD + D skip + z gate + grouped RMSNorm.
 * state is [n_heads][head_dim][d_state], xbc is x|B|C. */
int allpaca_mamba2_ssd_step(const float *z, const float *xbc, const float *dt,
                            const float *a, const float *d, const float *dt_bias,
                            const float *norm, float *state, float *out,
                            int32_t d_inner, int32_t d_state, int32_t n_heads,
                            int32_t n_groups, float eps, allpaca_stream_t stream);

/* RoPE on Q and K in one launch. Decode uses n_tok=1 via allpaca_rope_qk. */
int allpaca_rope_qk(float *q, float *k, int32_t n_q_heads, int32_t n_k_heads, int32_t head_dim,
                    int32_t pos, float theta, allpaca_stream_t stream);

/* Prefill: Q/K are [n_tok, heads*head_dim]; position is start_pos + token. One launch. */
int allpaca_rope_qk_rows(float *q, float *k, int32_t n_tok, int32_t n_q_heads, int32_t n_k_heads,
                         int32_t head_dim, int32_t start_pos, float theta, allpaca_stream_t stream);

/* DeepSeek-V4 partial RoPE on x[n_tok, n_heads, head_dim]: rotate [offset, offset+n_rot).
 * n_rot==0 means full-head RoPE. back!=0 is ggml rope_ext_back (negated sine). */
int allpaca_rope_offset_rows(float *x, int32_t n_tok, int32_t n_heads, int32_t head_dim,
                             int32_t n_rot, int32_t offset, int32_t start_pos, float theta,
                             int32_t back, allpaca_stream_t stream);

/* Qwen3.5 IMRoPE / partial NeoX: rotate first n_rot dims; sections[4] select (t,h,w,e). */
int allpaca_rope_imrope_qk(float *q, float *k, int32_t n_q_heads, int32_t n_k_heads,
                           int32_t head_dim, int32_t n_rot, int32_t pos, float theta,
                           const int32_t sections[4], allpaca_stream_t stream);

int allpaca_dequant_q8_0(const uint8_t *blocks, float *out, int32_t n, allpaca_stream_t stream);

int allpaca_embed_gather_q8_0(const uint8_t *table, int32_t token, float *out, int32_t dim,
                              allpaca_stream_t stream);

int allpaca_embed_gather_q8_0_rows(const uint8_t *table, const int32_t *tokens, float *out,
                                   int32_t n_tok, int32_t dim, allpaca_stream_t stream);

/* Native 18-byte ROCmFP4 blocks; row stride = ceil(dim/32)*18. */
int allpaca_embed_gather_rocmfp4(const uint8_t *table, int32_t token, float *out, int32_t dim,
                                 allpaca_stream_t stream);

int allpaca_embed_gather_rocmfp4_rows(const uint8_t *table, const int32_t *tokens, float *out,
                                      int32_t n_tok, int32_t dim, allpaca_stream_t stream);

#ifdef __cplusplus
}
#endif

#endif
