#include "../include/allpaca_kernels.h"
#include <stdint.h>

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define ALLPACA_HAS_HIP 1
#else
#define ALLPACA_HAS_HIP 0
#endif

#if ALLPACA_HAS_HIP

__device__ float allpaca_f16_bits_to_f32(uint16_t h);
__device__ float allpaca_f16le_to_f32(const uint8_t *p);

static hipStream_t as_stream(allpaca_stream_t stream) {
    return static_cast<hipStream_t>(stream);
}

static int hip_status(hipError_t err) {
    return err == hipSuccess ? ALLPACA_OK : ALLPACA_ERR_DEVICE;
}

static void clear_last_error() {
    (void)hipGetLastError();
}

/* C[m,n] = A[m,k] * B[k,n]. Shared-memory tiles, coalesced K/N loads. */
__global__ void gemm_tiled_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TX = 16;
    constexpr int TY = 16;
    constexpr int TM = BM / TY;
    constexpr int TN = BN / TX;

    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    const int row0 = static_cast<int>(blockIdx.y) * BM;
    const int col0 = static_cast<int>(blockIdx.x) * BN;
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tid = ty * TX + tx;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < k; k0 += BK) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int idx = tid + t * (TX * TY);
            const int r = idx / BK;
            const int p = idx % BK;
            const int gr = row0 + r;
            const int gp = k0 + p;
            As[r][p] = (gr < m && gp < k) ? a[static_cast<size_t>(gr) * k + gp] : 0.0f;
        }
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int idx = tid + t * (TX * TY);
            const int p = idx / BN;
            const int c = idx % BN;
            const int gp = k0 + p;
            const int gc = col0 + c;
            Bs[p][c] = (gp < k && gc < n) ? b[static_cast<size_t>(gp) * n + gc] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int p = 0; p < BK; ++p) {
            float av[TM];
            float bv[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                av[i] = As[ty * TM + i][p];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                bv[j] = Bs[p][tx * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += av[i] * bv[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = row0 + ty * TM + i;
        if (gr >= m) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = col0 + tx * TN + j;
            if (gc < n) {
                c[static_cast<size_t>(gr) * n + gc] = acc[i][j];
            }
        }
    }
}

/* Decode GEMV: C[m,n] = A[m,k] * B[n,k]^T. Four columns per warp (lm_head-scale n). */
__global__ void gemv_nt_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
    constexpr int WARPS = 8;
    constexpr int COLS = 4;
    constexpr int SUB = 8;
    const int row = static_cast<int>(blockIdx.y);
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int sub = lane / SUB;
    const int sublane = lane % SUB;
    const int col = (static_cast<int>(blockIdx.x) * WARPS + warp) * COLS + sub;
    if (row >= m) {
        return;
    }
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    float acc = 0.0f;
    if (col < n) {
        const float *brow = b + static_cast<size_t>(col) * static_cast<size_t>(k);
        const int k4 = k & ~3;
        const bool vec = k4 > 0 && (reinterpret_cast<uintptr_t>(arow) % 16u) == 0 &&
                         (reinterpret_cast<uintptr_t>(brow) % 16u) == 0;
        if (vec) {
            for (int p = sublane * 4; p < k4; p += SUB * 4) {
                const float4 av = *reinterpret_cast<const float4 *>(arow + p);
                const float4 bv = *reinterpret_cast<const float4 *>(brow + p);
                acc += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w;
            }
            for (int p = k4 + sublane; p < k; p += SUB) {
                acc += arow[p] * brow[p];
            }
        } else {
            for (int p = sublane; p < k; p += SUB) {
                acc += arow[p] * brow[p];
            }
        }
    }
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, 8);
    }
    if (col < n && sublane == 0) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + col] = acc;
    }
}

/* Staged Q8_0: 36-byte blocks, f32 scale at 0, qs[32] at 4 (4-byte aligned). */
constexpr int Q8_STAGED_STRIDE = 36;
constexpr int Q8_STAGED_QS = 4;

/* Decode Q8 GEMV: 8 cols/warp, 16 warps, shared A tiles (no per-column A reload). */
__global__ void gemv_nt_q8_0_kernel(const float *a, const uint8_t *b, float *c, int m, int n,
                                    int k) {
    constexpr int WARPS = 16;
    constexpr int COLS = 8;
    constexpr int SUB = 4;
    constexpr int TILE = 4;
    __shared__ float As[TILE * 32];

    const int row = static_cast<int>(blockIdx.y);
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int sub = lane / SUB;
    const int sublane = lane % SUB;
    const int col = (static_cast<int>(blockIdx.x) * WARPS + warp) * COLS + sub;
    if (row >= m) {
        return;
    }
    const int nblocks = (k + 31) / 32;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    const size_t row_stride = static_cast<size_t>(nblocks) * static_cast<size_t>(Q8_STAGED_STRIDE);
    const uint8_t *brow = (col < n) ? b + static_cast<size_t>(col) * row_stride : nullptr;
    float acc = 0.0f;
    const int tid = static_cast<int>(threadIdx.x);

    for (int blk0 = 0; blk0 < nblocks; blk0 += TILE) {
        const int ntile = (blk0 + TILE <= nblocks) ? TILE : (nblocks - blk0);
        const int a0 = blk0 * 32;
        const int nfloat = ntile * 32;
        for (int i = tid; i < nfloat; i += static_cast<int>(blockDim.x)) {
            const int gi = a0 + i;
            As[i] = (gi < k) ? arow[gi] : 0.0f;
        }
        __syncthreads();

        if (brow) {
            for (int t = 0; t < ntile; ++t) {
                const uint8_t *bp = brow + static_cast<size_t>(blk0 + t) * Q8_STAGED_STRIDE;
                const float d = *reinterpret_cast<const float *>(bp);
                const int qbase = sublane * 8;
                const uint8_t *qp = bp + Q8_STAGED_QS + qbase;
                const uint32_t qlo = *reinterpret_cast<const uint32_t *>(qp);
                const uint32_t qhi = *reinterpret_cast<const uint32_t *>(qp + 4);
                const float *as = As + t * 32 + qbase;
                float dot = as[0] * static_cast<float>(static_cast<int8_t>(qlo & 0xff));
                dot += as[1] * static_cast<float>(static_cast<int8_t>((qlo >> 8) & 0xff));
                dot += as[2] * static_cast<float>(static_cast<int8_t>((qlo >> 16) & 0xff));
                dot += as[3] * static_cast<float>(static_cast<int8_t>((qlo >> 24) & 0xff));
                dot += as[4] * static_cast<float>(static_cast<int8_t>(qhi & 0xff));
                dot += as[5] * static_cast<float>(static_cast<int8_t>((qhi >> 8) & 0xff));
                dot += as[6] * static_cast<float>(static_cast<int8_t>((qhi >> 16) & 0xff));
                dot += as[7] * static_cast<float>(static_cast<int8_t>((qhi >> 24) & 0xff));
                acc += d * dot;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int off = 2; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, 4);
    }
    if (col < n && sublane == 0) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + col] = acc;
    }
}

/* C[m,n] = A[m,k] * B[n,k]^T. Shared-memory tiles, coalesced row loads. */
__global__ void gemm_nt_tiled_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TX = 16;
    constexpr int TY = 16;
    constexpr int TM = BM / TY;
    constexpr int TN = BN / TX;

    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BN][BK + 1];

    const int row0 = static_cast<int>(blockIdx.y) * BM;
    const int col0 = static_cast<int>(blockIdx.x) * BN;
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tid = ty * TX + tx;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < k; k0 += BK) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int idx = tid + t * (TX * TY);
            const int r = idx / BK;
            const int p = idx % BK;
            const int gr = row0 + r;
            const int gp = k0 + p;
            As[r][p] = (gr < m && gp < k) ? a[static_cast<size_t>(gr) * k + gp] : 0.0f;
        }
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int idx = tid + t * (TX * TY);
            const int col = idx / BK;
            const int p = idx % BK;
            const int gc = col0 + col;
            const int gp = k0 + p;
            Bs[col][p] = (gc < n && gp < k) ? b[static_cast<size_t>(gc) * k + gp] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int p = 0; p < BK; ++p) {
            float av[TM];
            float bv[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                av[i] = As[ty * TM + i][p];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                bv[j] = Bs[tx * TN + j][p];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += av[i] * bv[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = row0 + ty * TM + i;
        if (gr >= m) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = col0 + tx * TN + j;
            if (gc < n) {
                c[static_cast<size_t>(gr) * n + gc] = acc[i][j];
            }
        }
    }
}

__device__ float warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down(v, off, 32);
    }
    return __shfl(v, 0, 32);
}

/* Decode/prefill: one warp per (query, head). Online softmax, coalesced KV. */
__global__ void attention_flash_kernel(const float *q, const float *k, const float *v, float *out,
                                       int n_q, int n_kv, int n_heads, int n_kv_heads, int head_dim,
                                       int causal, const uint8_t *mask) {
    const int qi = static_cast<int>(blockIdx.x);
    const int h = static_cast<int>(blockIdx.y);
    if (qi >= n_q || h >= n_heads) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const int reps = n_heads / n_kv_heads;
    const int kv_h = h / reps;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int t_max = causal ? qi + (n_kv > n_q ? n_kv - n_q : 0) : n_kv - 1;
    const float *qh = q + (static_cast<size_t>(qi) * n_heads + h) * head_dim;
    float *oh = out + (static_cast<size_t>(qi) * n_heads + h) * head_dim;
    const size_t kv_stride = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);

    constexpr int MAX_DPT = 16;
    float qreg[MAX_DPT];
    float acc[MAX_DPT];
#pragma unroll
    for (int i = 0; i < MAX_DPT; ++i) {
        qreg[i] = 0.0f;
        acc[i] = 0.0f;
    }

    const bool vec4 = (head_dim % 4) == 0 && head_dim > 0 &&
                      (reinterpret_cast<uintptr_t>(qh) % 16u) == 0;
    const int nvec = vec4 ? head_dim / 4 : head_dim;
    if (vec4) {
        int nd = 0;
        for (int v = lane; v < nvec && nd + 3 < MAX_DPT; v += 32) {
            const float4 qv = *reinterpret_cast<const float4 *>(qh + v * 4);
            qreg[nd] = qv.x;
            qreg[nd + 1] = qv.y;
            qreg[nd + 2] = qv.z;
            qreg[nd + 3] = qv.w;
            nd += 4;
        }
    } else {
        int nd = 0;
        for (int d = lane; d < head_dim && nd < MAX_DPT; d += 32) {
            qreg[nd++] = qh[d];
        }
    }

    float m = -1.0e30f;
    float l = 0.0f;

    for (int t = 0; t < n_kv; ++t) {
        const bool valid = mask ? mask[static_cast<size_t>(qi) * static_cast<size_t>(n_kv) +
                                       static_cast<size_t>(t)] != 0
                                : t <= t_max;
        const float *kh = k + static_cast<size_t>(t) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
        const float *vh = v + static_cast<size_t>(t) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
        float partial = 0.0f;
        if (vec4) {
            int i = 0;
            for (int vec = lane; vec < nvec && i + 3 < MAX_DPT; vec += 32) {
                float4 kv;
                if (valid && (reinterpret_cast<uintptr_t>(kh + vec * 4) % 16u) == 0) {
                    kv = *reinterpret_cast<const float4 *>(kh + vec * 4);
                } else {
                    kv.x = (valid && vec * 4 < head_dim) ? kh[vec * 4] : 0.0f;
                    kv.y = (valid && vec * 4 + 1 < head_dim) ? kh[vec * 4 + 1] : 0.0f;
                    kv.z = (valid && vec * 4 + 2 < head_dim) ? kh[vec * 4 + 2] : 0.0f;
                    kv.w = (valid && vec * 4 + 3 < head_dim) ? kh[vec * 4 + 3] : 0.0f;
                }
                partial += qreg[i] * kv.x + qreg[i + 1] * kv.y + qreg[i + 2] * kv.z +
                           qreg[i + 3] * kv.w;
                i += 4;
            }
        } else {
            int i = 0;
            for (int d = lane; d < head_dim && i < MAX_DPT; d += 32) {
                partial += qreg[i++] * (valid ? kh[d] : 0.0f);
            }
        }
        float s = warp_sum32(partial) * scale;
        if (!valid) {
            s = -1.0e30f;
        }
        const float m_new = fmaxf(m, s);
        const float alpha = expf(m - m_new);
        const float p = expf(s - m_new);
        l = l * alpha + p;
        if (vec4) {
            int i = 0;
            for (int vec = lane; vec < nvec && i + 3 < MAX_DPT; vec += 32) {
                float4 vv;
                if (valid && (reinterpret_cast<uintptr_t>(vh + vec * 4) % 16u) == 0) {
                    vv = *reinterpret_cast<const float4 *>(vh + vec * 4);
                } else {
                    vv.x = (valid && vec * 4 < head_dim) ? vh[vec * 4] : 0.0f;
                    vv.y = (valid && vec * 4 + 1 < head_dim) ? vh[vec * 4 + 1] : 0.0f;
                    vv.z = (valid && vec * 4 + 2 < head_dim) ? vh[vec * 4 + 2] : 0.0f;
                    vv.w = (valid && vec * 4 + 3 < head_dim) ? vh[vec * 4 + 3] : 0.0f;
                }
                acc[i] = acc[i] * alpha + p * vv.x;
                acc[i + 1] = acc[i + 1] * alpha + p * vv.y;
                acc[i + 2] = acc[i + 2] * alpha + p * vv.z;
                acc[i + 3] = acc[i + 3] * alpha + p * vv.w;
                i += 4;
            }
        } else {
            int i = 0;
            for (int d = lane; d < head_dim && i < MAX_DPT; d += 32) {
                acc[i] = acc[i] * alpha + p * (valid ? vh[d] : 0.0f);
                i++;
            }
        }
        m = m_new;
    }

    const float inv = (l == 0.0f) ? 0.0f : 1.0f / l;
    if (vec4) {
        int i = 0;
        for (int vec = lane; vec < nvec && i + 3 < MAX_DPT; vec += 32) {
            if (vec * 4 + 3 < head_dim && (reinterpret_cast<uintptr_t>(oh + vec * 4) % 16u) == 0) {
                float4 o;
                o.x = acc[i] * inv;
                o.y = acc[i + 1] * inv;
                o.z = acc[i + 2] * inv;
                o.w = acc[i + 3] * inv;
                *reinterpret_cast<float4 *>(oh + vec * 4) = o;
            } else {
                if (vec * 4 < head_dim) {
                    oh[vec * 4] = acc[i] * inv;
                }
                if (vec * 4 + 1 < head_dim) {
                    oh[vec * 4 + 1] = acc[i + 1] * inv;
                }
                if (vec * 4 + 2 < head_dim) {
                    oh[vec * 4 + 2] = acc[i + 2] * inv;
                }
                if (vec * 4 + 3 < head_dim) {
                    oh[vec * 4 + 3] = acc[i + 3] * inv;
                }
            }
            i += 4;
        }
    } else {
        int i = 0;
        for (int d = lane; d < head_dim && i < MAX_DPT; d += 32) {
            oh[d] = acc[i++] * inv;
        }
    }
}

static int launch_gemm(const float *a, const float *b, float *c, int32_t m, int32_t n, int32_t k,
                       hipStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    constexpr int BM = 64;
    constexpr int BN = 64;
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((n + BN - 1) / BN), static_cast<unsigned>((m + BM - 1) / BM));
    hipLaunchKernelGGL(gemm_tiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    return hip_status(hipGetLastError());
}

static int launch_gemm_nt(const float *a, const float *b, float *c, int32_t m, int32_t n, int32_t k,
                          hipStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    /* Decode is always M=1; small-M prefill stays on the coalesced GEMV. */
    if (m <= 8) {
        constexpr int warps = 8;
        constexpr int cols = 4;
        constexpr int cols_per_block = warps * cols;
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((n + cols_per_block - 1) / cols_per_block),
                  static_cast<unsigned>(m));
        hipLaunchKernelGGL(gemv_nt_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    } else {
        constexpr int BM = 64;
        constexpr int BN = 64;
        dim3 block(16, 16);
        dim3 grid(static_cast<unsigned>((n + BN - 1) / BN), static_cast<unsigned>((m + BM - 1) / BM));
        hipLaunchKernelGGL(gemm_nt_tiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    }
    return hip_status(hipGetLastError());
}

static int launch_gemm_nt_q8_0(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                               int32_t k, hipStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    constexpr int warps = 16;
    constexpr int cols = 8;
    constexpr int cols_per_block = warps * cols;
    dim3 block(512);
    dim3 grid(static_cast<unsigned>((n + cols_per_block - 1) / cols_per_block),
              static_cast<unsigned>(m));
    hipLaunchKernelGGL(gemv_nt_q8_0_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_hip_device_count(int32_t *count) {
    if (!count) {
        return ALLPACA_ERR_INVALID;
    }
    int n = 0;
    hipError_t err = hipGetDeviceCount(&n);
    if (err != hipSuccess) {
        *count = 0;
        return ALLPACA_ERR_DEVICE;
    }
    *count = n;
    return ALLPACA_OK;
}

extern "C" int allpaca_hip_device_props(int32_t id, char *name, int32_t name_len,
                                        uint64_t *memory_bytes, double *bandwidth_gbps) {
    if (!name || name_len <= 0 || !memory_bytes || !bandwidth_gbps) {
        return ALLPACA_ERR_INVALID;
    }
    hipDeviceProp_t prop{};
    hipError_t err = hipGetDeviceProperties(&prop, id);
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    int i = 0;
    for (; i < name_len - 1 && prop.name[i]; ++i) {
        name[i] = prop.name[i];
    }
    name[i] = '\0';
    *memory_bytes = static_cast<uint64_t>(prop.totalGlobalMem);
    /* Relative bandwidth from clock * bus width. GDDR transfers on both edges. */
    const double clock_ghz = static_cast<double>(prop.memoryClockRate) / 1.0e6;
    const double bytes_per_cycle = static_cast<double>(prop.memoryBusWidth) / 8.0 * 2.0;
    *bandwidth_gbps = clock_ghz * bytes_per_cycle;
    return ALLPACA_OK;
}

extern "C" int allpaca_hip_can_p2p(int32_t src, int32_t dst, int32_t *can) {
    if (!can) {
        return ALLPACA_ERR_INVALID;
    }
    int yes = 0;
    hipError_t err = hipDeviceCanAccessPeer(&yes, src, dst);
    if (err != hipSuccess) {
        *can = 0;
        return ALLPACA_ERR_DEVICE;
    }
    *can = yes;
    return ALLPACA_OK;
}

extern "C" int allpaca_hip_enable_peer_access(int32_t src, int32_t dst) {
    hipError_t err = hipSetDevice(src);
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    err = hipDeviceEnablePeerAccess(dst, 0);
    if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) {
        return ALLPACA_ERR_DEVICE;
    }
    return ALLPACA_OK;
}

extern "C" int allpaca_hip_malloc(void **ptr, size_t bytes, int32_t device) {
    if (!ptr || bytes == 0) {
        return ALLPACA_ERR_INVALID;
    }
    hipError_t err = hipSetDevice(device);
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    return hip_status(hipMalloc(ptr, bytes));
}

extern "C" int allpaca_hip_free(void *ptr) {
    if (!ptr) {
        return ALLPACA_OK;
    }
    hipError_t err = hipStreamSynchronize(nullptr);
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    return hip_status(hipFree(ptr));
}

extern "C" int allpaca_hip_host_malloc(void **ptr, size_t bytes) {
    if (!ptr || bytes == 0) {
        return ALLPACA_ERR_INVALID;
    }
    return hip_status(hipHostMalloc(ptr, bytes, hipHostMallocDefault));
}

extern "C" int allpaca_hip_host_free(void *ptr) {
    if (!ptr) {
        return ALLPACA_OK;
    }
    return hip_status(hipHostFree(ptr));
}

extern "C" int allpaca_hip_memcpy(void *dst, const void *src, size_t bytes, int32_t kind,
                                 allpaca_stream_t stream) {
    if (!dst || !src) {
        return ALLPACA_ERR_INVALID;
    }
    hipMemcpyKind mk = hipMemcpyDefault;
    switch (kind) {
    case ALLPACA_MEMCPY_H2D:
        mk = hipMemcpyHostToDevice;
        break;
    case ALLPACA_MEMCPY_D2H:
        mk = hipMemcpyDeviceToHost;
        break;
    case ALLPACA_MEMCPY_D2D:
        mk = hipMemcpyDeviceToDevice;
        break;
    default:
        mk = hipMemcpyDefault;
        break;
    }
    hipError_t err = hipMemcpyAsync(dst, src, bytes, mk, as_stream(stream));
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    /* Host-visible D2H is the wait point; H2D/D2D stay queued on the stream. */
    if (kind == ALLPACA_MEMCPY_D2H) {
        return hip_status(hipStreamSynchronize(as_stream(stream)));
    }
    return ALLPACA_OK;
}

extern "C" int allpaca_hip_memcpy_peer(void *dst, int32_t dst_dev, const void *src, int32_t src_dev,
                                      size_t bytes, allpaca_stream_t stream) {
    if (!dst || !src) {
        return ALLPACA_ERR_INVALID;
    }
    hipError_t err =
        hipMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, as_stream(stream));
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    return hip_status(hipStreamSynchronize(as_stream(stream)));
}

extern "C" int allpaca_hip_benchmark(int32_t src, int32_t dst, size_t bytes, double *p2p_gbps,
                                    double *host_gbps) {
    if (!p2p_gbps || !host_gbps || bytes == 0) {
        return ALLPACA_ERR_INVALID;
    }
    *p2p_gbps = 0.0;
    *host_gbps = 0.0;

    void *d_src = nullptr;
    void *d_dst = nullptr;
    void *h_pin = nullptr;
    int rc = allpaca_hip_malloc(&d_src, bytes, src);
    if (rc != ALLPACA_OK) {
        return rc;
    }
    rc = allpaca_hip_malloc(&d_dst, bytes, dst);
    if (rc != ALLPACA_OK) {
        (void)hipFree(d_src);
        return rc;
    }
    if (hipHostMalloc(&h_pin, bytes, hipHostMallocDefault) != hipSuccess) {
        (void)hipFree(d_src);
        (void)hipFree(d_dst);
        return ALLPACA_ERR_DEVICE;
    }

    hipEvent_t start{};
    hipEvent_t stop{};
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);

    /* Enable is attempted even when the raw query is false; policy lives in Rust. */
    (void)allpaca_hip_enable_peer_access(src, dst);
    (void)allpaca_hip_enable_peer_access(dst, src);
    (void)hipSetDevice(src);
    (void)hipEventRecord(start, nullptr);
    hipError_t peer_err = hipMemcpyPeer(d_dst, dst, d_src, src, bytes);
    (void)hipEventRecord(stop, nullptr);
    (void)hipEventSynchronize(stop);
    if (peer_err == hipSuccess) {
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, start, stop);
        if (ms > 0.0f) {
            *p2p_gbps = (static_cast<double>(bytes) * 8.0 / 1.0e9) / (static_cast<double>(ms) / 1.0e3);
        }
    }

    (void)hipSetDevice(src);
    (void)hipEventRecord(start, nullptr);
    (void)hipMemcpy(h_pin, d_src, bytes, hipMemcpyDeviceToHost);
    (void)hipSetDevice(dst);
    (void)hipMemcpy(d_dst, h_pin, bytes, hipMemcpyHostToDevice);
    (void)hipEventRecord(stop, nullptr);
    (void)hipEventSynchronize(stop);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, start, stop);
    if (ms > 0.0f) {
        *host_gbps = (static_cast<double>(bytes) * 8.0 / 1.0e9) / (static_cast<double>(ms) / 1.0e3);
    }

    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    (void)hipHostFree(h_pin);
    (void)hipFree(d_src);
    (void)hipFree(d_dst);
    return ALLPACA_OK;
}

extern "C" int allpaca_hip_set_device(int32_t device) {
    return hip_status(hipSetDevice(device));
}

extern "C" int allpaca_hip_stream_synchronize(allpaca_stream_t stream) {
    return hip_status(hipStreamSynchronize(as_stream(stream)));
}

extern "C" int allpaca_hip_event_create(void **event) {
    if (!event) {
        return ALLPACA_ERR_INVALID;
    }
    hipEvent_t ev = nullptr;
    hipError_t err = hipEventCreateWithFlags(&ev, hipEventDefault);
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    *event = ev;
    return ALLPACA_OK;
}

extern "C" int allpaca_hip_event_destroy(void *event) {
    if (!event) {
        return ALLPACA_OK;
    }
    return hip_status(hipEventDestroy(static_cast<hipEvent_t>(event)));
}

extern "C" int allpaca_hip_event_record(void *event, allpaca_stream_t stream) {
    if (!event) {
        return ALLPACA_ERR_INVALID;
    }
    return hip_status(hipEventRecord(static_cast<hipEvent_t>(event), as_stream(stream)));
}

extern "C" int allpaca_hip_event_elapsed_us(void *start, void *stop, double *us) {
    if (!start || !stop || !us) {
        return ALLPACA_ERR_INVALID;
    }
    float ms = 0.0f;
    hipError_t err = hipEventElapsedTime(&ms, static_cast<hipEvent_t>(start),
                                         static_cast<hipEvent_t>(stop));
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    *us = static_cast<double>(ms) * 1000.0;
    return ALLPACA_OK;
}

extern "C" int allpaca_hip_mem_used(int32_t device, uint64_t *used_bytes) {
    if (!used_bytes) {
        return ALLPACA_ERR_INVALID;
    }
    hipError_t err = hipSetDevice(device);
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    size_t free_b = 0;
    size_t total_b = 0;
    err = hipMemGetInfo(&free_b, &total_b);
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    *used_bytes = static_cast<uint64_t>(total_b - free_b);
    return ALLPACA_OK;
}

extern "C" int allpaca_gemm(const float *a, const float *b, float *c, int32_t m, int32_t n,
                            int32_t k, allpaca_stream_t stream) {
    return launch_gemm(a, b, c, m, n, k, as_stream(stream));
}

extern "C" int allpaca_gemm_nt(const float *a, const float *b, float *c, int32_t m, int32_t n,
                               int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt(a, b, c, m, n, k, as_stream(stream));
}

extern "C" int allpaca_gemm_nt_q8_0(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                    int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_q8_0(a, b, c, m, n, k, as_stream(stream));
}

extern "C" int allpaca_attention(const float *q, const float *k, const float *v, float *out,
                                 int32_t n_q, int32_t n_kv, int32_t n_heads, int32_t n_kv_heads,
                                 int32_t head_dim, int32_t causal, allpaca_stream_t stream) {
    if (!q || !k || !v || !out || n_q <= 0 || n_kv <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_dim <= 0 || (n_heads % n_kv_heads) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    dim3 block(32);
    dim3 grid(static_cast<unsigned>(n_q), static_cast<unsigned>(n_heads));
    hipLaunchKernelGGL(attention_flash_kernel, grid, block, 0, as_stream(stream), q, k, v, out, n_q,
                       n_kv, n_heads, n_kv_heads, head_dim, causal, static_cast<const uint8_t *>(nullptr));
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_attention_masked(const float *q, const float *k, const float *v, float *out,
                                        int32_t n_q, int32_t n_kv, int32_t n_heads,
                                        int32_t n_kv_heads, int32_t head_dim, const uint8_t *mask,
                                        allpaca_stream_t stream) {
    if (!q || !k || !v || !out || !mask || n_q <= 0 || n_kv <= 0 || n_heads <= 0 ||
        n_kv_heads <= 0 || head_dim <= 0 || (n_heads % n_kv_heads) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    dim3 block(32);
    dim3 grid(static_cast<unsigned>(n_q), static_cast<unsigned>(n_heads));
    hipLaunchKernelGGL(attention_flash_kernel, grid, block, 0, as_stream(stream), q, k, v, out, n_q,
                       n_kv, n_heads, n_kv_heads, head_dim, 0, mask);
    return hip_status(hipGetLastError());
}

__global__ void rmsnorm_kernel(const float *x, const float *weight, float *out, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) {
        ss += x[i] * x[i];
    }
    const float inv = rsqrtf(ss / static_cast<float>(n) + eps);
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        out[idx] = x[idx] * inv * weight[idx];
    }
}

__global__ void rope_kernel(float *x, int n_heads, int head_dim, int pos, float theta) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int half = head_dim / 2;
    const int total = n_heads * half;
    if (idx >= total || head_dim % 2 != 0) {
        return;
    }
    const int h = idx / half;
    const int i = idx % half;
    const float freq = powf(theta, -static_cast<float>(i) / static_cast<float>(half));
    const float angle = static_cast<float>(pos) * freq;
    const float s = sinf(angle);
    const float c = cosf(angle);
    float *hvec = x + h * head_dim;
    const float a = hvec[2 * i];
    const float b = hvec[2 * i + 1];
    hvec[2 * i] = a * c - b * s;
    hvec[2 * i + 1] = a * s + b * c;
}

__global__ void silu_mul_kernel(float *gate, const float *up, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    const float g = gate[idx];
    gate[idx] = (g / (1.0f + expf(-g))) * up[idx];
}

extern "C" int allpaca_rmsnorm(const float *x, const float *weight, float *out, int32_t n, float eps,
                               allpaca_stream_t stream) {
    if (!x || !weight || !out || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 64;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(rmsnorm_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), x, weight, out, n,
                       eps);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_rope(float *x, int32_t n_heads, int32_t head_dim, int32_t pos, float theta,
                            allpaca_stream_t stream) {
    if (!x || n_heads <= 0 || head_dim <= 0 || (head_dim % 2) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int total = n_heads * (head_dim / 2);
    const int threads = 64;
    const int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(rope_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), x, n_heads,
                       head_dim, pos, theta);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_silu_mul(float *gate, const float *up, int32_t n, allpaca_stream_t stream) {
    if (!gate || !up || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 64;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(silu_mul_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), gate, up, n);
    return hip_status(hipGetLastError());
}

__global__ void embed_gather_kernel(const float *table, int token, float *out, int dim) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= dim) {
        return;
    }
    out[idx] = table[token * dim + idx];
}

__global__ void residual_add_kernel(float *x, const float *residual, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    x[idx] += residual[idx];
}

extern "C" int allpaca_embed_gather(const float *table, int32_t token, float *out, int32_t dim,
                                    allpaca_stream_t stream) {
    if (!table || !out || dim <= 0 || token < 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 64;
    const int blocks = (dim + threads - 1) / threads;
    hipLaunchKernelGGL(embed_gather_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), table, token, out,
                       dim);
    return hip_status(hipGetLastError());
}

__device__ float allpaca_f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u) << 16);
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            float f = (static_cast<float>(mant) / 1024.0f) * (1.0f / 16384.0f);
            return (h & 0x8000u) ? -f : f;
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = bits;
    return conv.f;
}

__device__ float allpaca_f16le_to_f32(const uint8_t *p) {
    const uint16_t h = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    return allpaca_f16_bits_to_f32(h);
}

__global__ void dequant_q8_0_kernel(const uint8_t *blocks, float *out, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const int b = i / 32;
    const int j = i % 32;
    const uint8_t *blk = blocks + static_cast<size_t>(b) * Q8_STAGED_STRIDE;
    const float d = *reinterpret_cast<const float *>(blk);
    out[i] = d * static_cast<float>(static_cast<int8_t>(blk[Q8_STAGED_QS + j]));
}

__global__ void embed_gather_q8_0_kernel(const uint8_t *table, int token, float *out, int dim) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= dim) {
        return;
    }
    const int blocks_per_row = (dim + 31) / 32;
    const int b = i / 32;
    const int j = i % 32;
    const uint8_t *blk =
        table + (static_cast<size_t>(token) * static_cast<size_t>(blocks_per_row) +
                 static_cast<size_t>(b)) *
                    Q8_STAGED_STRIDE;
    const float d = *reinterpret_cast<const float *>(blk);
    out[i] = d * static_cast<float>(static_cast<int8_t>(blk[Q8_STAGED_QS + j]));
}

extern "C" int allpaca_dequant_q8_0(const uint8_t *qblocks, float *out, int32_t n,
                                    allpaca_stream_t stream) {
    if (!qblocks || !out || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    /* Workspace reuse must not overwrite an in-flight consumer of `out`. */
    if (hipStreamSynchronize(as_stream(stream)) != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    clear_last_error();
    const int threads = 64;
    const int nblocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(dequant_q8_0_kernel, dim3(static_cast<unsigned>(nblocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), qblocks, out, n);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_embed_gather_q8_0(const uint8_t *table, int32_t token, float *out, int32_t dim,
                                         allpaca_stream_t stream) {
    if (!table || !out || dim <= 0 || token < 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 64;
    const int blocks = (dim + threads - 1) / threads;
    hipLaunchKernelGGL(embed_gather_q8_0_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), table, token, out,
                       dim);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_residual_add(float *x, const float *residual, int32_t n,
                                    allpaca_stream_t stream) {
    if (!x || !residual || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 64;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(residual_add_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), x, residual, n);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_copy_async(void *dst, const void *src, size_t bytes,
                                  allpaca_stream_t stream) {
    if (!dst || !src) {
        return ALLPACA_ERR_INVALID;
    }
    hipError_t err = hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, as_stream(stream));
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    return hip_status(hipStreamSynchronize(as_stream(stream)));
}

#else /* !ALLPACA_HAS_HIP */

extern "C" int allpaca_hip_device_count(int32_t *count) {
    if (count) {
        *count = 0;
    }
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_device_props(int32_t, char *, int32_t, uint64_t *, double *) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_can_p2p(int32_t, int32_t, int32_t *can) {
    if (can) {
        *can = 0;
    }
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_enable_peer_access(int32_t, int32_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_malloc(void **, size_t, int32_t) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_free(void *) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_host_malloc(void **, size_t) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_host_free(void *) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_memcpy(void *, const void *, size_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_memcpy_peer(void *, int32_t, const void *, int32_t, size_t,
                                      allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_benchmark(int32_t, int32_t, size_t, double *, double *) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_mem_used(int32_t, uint64_t *) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_set_device(int32_t) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_stream_synchronize(allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_event_create(void **) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_event_destroy(void *) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_event_record(void *, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_event_elapsed_us(void *, void *, double *) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm(const float *, const float *, float *, int32_t, int32_t, int32_t,
                            allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt(const float *, const float *, float *, int32_t, int32_t, int32_t,
                               allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_q8_0(const float *, const uint8_t *, float *, int32_t, int32_t,
                                    int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_attention(const float *, const float *, const float *, float *, int32_t,
                                 int32_t, int32_t, int32_t, int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_attention_masked(const float *, const float *, const float *, float *,
                                        int32_t, int32_t, int32_t, int32_t, int32_t, const uint8_t *,
                                        allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_copy_async(void *, const void *, size_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_rmsnorm(const float *, const float *, float *, int32_t, float,
                               allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_rope(float *, int32_t, int32_t, int32_t, float, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_silu_mul(float *, const float *, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_embed_gather(const float *, int32_t, float *, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_residual_add(float *, const float *, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_dequant_q8_0(const uint8_t *, float *, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_embed_gather_q8_0(const uint8_t *, int32_t, float *, int32_t,
                                         allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

#endif
