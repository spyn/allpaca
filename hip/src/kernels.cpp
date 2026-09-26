#include "../include/allpaca_kernels.h"
#include <stdint.h>

#if defined(_WIN32) && defined(ALLPACA_HAS_DIRECTSTORAGE)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <dstorage.h>
#include <dxgi1_6.h>
#include <limits>
#include <mutex>
#include <new>
#include <wrl/client.h>
#endif

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#define ALLPACA_HAS_HIP 1
#else
#define ALLPACA_HAS_HIP 0
#endif

#if ALLPACA_HAS_HIP

#include <mutex>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_6.h>
#include <pdh.h>
#include <cwchar>
#endif

#ifdef ALLPACA_HAS_ROCBLAS
#include <rocblas/rocblas.h>
#endif
#ifdef ALLPACA_HAS_HIPBLASLT
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hipblaslt/hipblaslt.h>
#endif

__device__ float allpaca_f16_bits_to_f32(uint16_t h);
__device__ float allpaca_bf16_bits_to_f32(uint16_t h);
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

/* Decode GEMV: C[m,n] = A[m,k] * B[n,k]^T. Baseline: each column re-reads A from global. */
__global__ void gemv_nt_untiled_kernel(const float *a, const float *b, float *c, int m, int n,
                                       int k) {
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

/* Cache-aware F32 decode GEMV (lm_head): 256-float A tile in LDS, reused by 64 columns. */
__global__ void gemv_nt_tiled_kernel(const float *a, const float *b, float *c, int m, int n,
                                     int k) {
    constexpr int COLS = 64;
    constexpr int TILE = 256;
    __shared__ float As[TILE];
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(blockIdx.x) * COLS + tid;
    if (row >= m) {
        return;
    }
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    const float *brow =
        (tid < COLS && col < n) ? b + static_cast<size_t>(col) * static_cast<size_t>(k) : nullptr;
    float acc = 0.f;
    for (int k0 = 0; k0 < k; k0 += TILE) {
        const int chunk = (k0 + TILE <= k) ? TILE : (k - k0);
        if (tid < TILE) {
            As[tid] = (k0 + tid < k) ? arow[k0 + tid] : 0.f;
        }
        __syncthreads();
        if (brow) {
            const float *bchunk = brow + k0;
            for (int p = 0; p < chunk; ++p) {
                acc += As[p] * bchunk[p];
            }
        }
        __syncthreads();
    }
    if (brow) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
    }
}

/* Decode F16 GEMV baseline: eight columns per warp, A from global each column. */
__global__ void gemv_nt_f16_untiled_kernel(const float *a, const uint16_t *b, float *c, int m,
                                           int n, int k) {
    constexpr int WARPS = 16;
    constexpr int COLS = 8;
    constexpr int SUB = 4;
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
        const uint16_t *brow = b + static_cast<size_t>(col) * static_cast<size_t>(k);
        const int k8 = k & ~7;
        const bool vec = k8 > 0 && (reinterpret_cast<uintptr_t>(arow) % 16u) == 0 &&
                         (reinterpret_cast<uintptr_t>(brow) % 16u) == 0;
        if (vec) {
            for (int p = sublane * 8; p < k8; p += SUB * 8) {
                const float4 av0 = *reinterpret_cast<const float4 *>(arow + p);
                const float4 av1 = *reinterpret_cast<const float4 *>(arow + p + 4);
                const uint4 hv = *reinterpret_cast<const uint4 *>(brow + p);
                const uint16_t h0 = static_cast<uint16_t>(hv.x);
                const uint16_t h1 = static_cast<uint16_t>(hv.x >> 16);
                const uint16_t h2 = static_cast<uint16_t>(hv.y);
                const uint16_t h3 = static_cast<uint16_t>(hv.y >> 16);
                const uint16_t h4 = static_cast<uint16_t>(hv.z);
                const uint16_t h5 = static_cast<uint16_t>(hv.z >> 16);
                const uint16_t h6 = static_cast<uint16_t>(hv.w);
                const uint16_t h7 = static_cast<uint16_t>(hv.w >> 16);
                acc += av0.x * allpaca_f16_bits_to_f32(h0) + av0.y * allpaca_f16_bits_to_f32(h1) +
                       av0.z * allpaca_f16_bits_to_f32(h2) + av0.w * allpaca_f16_bits_to_f32(h3) +
                       av1.x * allpaca_f16_bits_to_f32(h4) + av1.y * allpaca_f16_bits_to_f32(h5) +
                       av1.z * allpaca_f16_bits_to_f32(h6) + av1.w * allpaca_f16_bits_to_f32(h7);
            }
            for (int p = k8 + sublane; p < k; p += SUB) {
                acc += arow[p] * allpaca_f16_bits_to_f32(brow[p]);
            }
        } else {
            for (int p = sublane; p < k; p += SUB) {
                acc += arow[p] * allpaca_f16_bits_to_f32(brow[p]);
            }
        }
    }
#pragma unroll
    for (int off = 2; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, 4);
    }
    if (col < n && sublane == 0) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + col] = acc;
    }
}

/* Cache-aware F16 decode GEMV: 256-float A tile in LDS, reused by 64 columns. */
__global__ void gemv_nt_f16_tiled_kernel(const float *a, const uint16_t *b, float *c, int m, int n,
                                         int k) {
    constexpr int COLS = 64;
    constexpr int TILE = 256;
    __shared__ float As[TILE];
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(blockIdx.x) * COLS + tid;
    if (row >= m) {
        return;
    }
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    const uint16_t *brow =
        (tid < COLS && col < n) ? b + static_cast<size_t>(col) * static_cast<size_t>(k) : nullptr;
    float acc = 0.f;
    for (int k0 = 0; k0 < k; k0 += TILE) {
        const int chunk = (k0 + TILE <= k) ? TILE : (k - k0);
        if (tid < TILE) {
            As[tid] = (k0 + tid < k) ? arow[k0 + tid] : 0.f;
        }
        __syncthreads();
        if (brow) {
            const uint16_t *bchunk = brow + k0;
            for (int p = 0; p < chunk; ++p) {
                acc += As[p] * allpaca_f16_bits_to_f32(bchunk[p]);
            }
        }
        __syncthreads();
    }
    if (brow) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
    }
}

/* gfx906 BF16 decode GEMV: expand u16→F32 in registers (no BF16 FMA). */
__global__ void gemv_nt_bf16_untiled_kernel(const float *a, const uint16_t *b, float *c, int m,
                                            int n, int k) {
    constexpr int WARPS = 16;
    constexpr int COLS = 8;
    constexpr int SUB = 4;
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
        const uint16_t *brow = b + static_cast<size_t>(col) * static_cast<size_t>(k);
        for (int p = sublane; p < k; p += SUB) {
            acc += arow[p] * allpaca_bf16_bits_to_f32(brow[p]);
        }
    }
#pragma unroll
    for (int off = 2; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, 4);
    }
    if (col < n && sublane == 0) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + col] = acc;
    }
}

/* gfx906 Triton sidecar won M=1 with BLOCK 1x64x32 / 4 warps. HIP GEMV used to
 * give each of 64 columns the whole K loop on one thread (TILE=256). Split K
 * across 2 threads/column, BK=32, vec4 A + uint2 B — same math, no Triton in
 * generate(). */
__global__ void gemv_nt_bf16_tiled_kernel(const float *a, const uint16_t *b, float *c, int m, int n,
                                          int k) {
    constexpr int COLS = 64;
    constexpr int KSPLIT = 2;
    constexpr int TILE = 32;
    constexpr int PER = TILE / KSPLIT;
    __shared__ float As[TILE];
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int col_local = tid / KSPLIT;
    const int kt = tid % KSPLIT;
    const int col = static_cast<int>(blockIdx.x) * COLS + col_local;
    if (row >= m) {
        return;
    }
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    const uint16_t *brow =
        (col < n) ? b + static_cast<size_t>(col) * static_cast<size_t>(k) : nullptr;
    float acc = 0.f;
    for (int k0 = 0; k0 < k; k0 += TILE) {
        const int chunk = (k0 + TILE <= k) ? TILE : (k - k0);
        if (tid < TILE / 4) {
            const int p = tid * 4;
            const int gp = k0 + p;
            const int nvalid = (gp < k) ? (k - gp) : 0;
            const size_t off = static_cast<size_t>(gp);
            if (nvalid >= 4 && (off & 3u) == 0) {
                const float4 v = *reinterpret_cast<const float4 *>(arow + off);
                As[p] = v.x;
                As[p + 1] = v.y;
                As[p + 2] = v.z;
                As[p + 3] = v.w;
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    As[p + i] = (gp + i < k) ? arow[off + static_cast<size_t>(i)] : 0.f;
                }
            }
        }
        __syncthreads();
        if (brow) {
            const int p0 = kt * PER;
            if (p0 < chunk) {
                const int take = chunk - p0 < PER ? chunk - p0 : PER;
                const uint16_t *bp = brow + k0 + p0;
                const float *ap = As + p0;
                int i = 0;
                for (; i + 4 <= take; i += 4) {
                    const size_t boff = static_cast<size_t>(k0 + p0 + i);
                    uint16_t bits[4];
                    if ((boff & 3u) == 0) {
                        const uint2 v = *reinterpret_cast<const uint2 *>(bp + i);
                        bits[0] = static_cast<uint16_t>(v.x);
                        bits[1] = static_cast<uint16_t>(v.x >> 16);
                        bits[2] = static_cast<uint16_t>(v.y);
                        bits[3] = static_cast<uint16_t>(v.y >> 16);
                    } else {
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            bits[t] = bp[i + t];
                        }
                    }
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        acc += ap[i + t] * allpaca_bf16_bits_to_f32(bits[t]);
                    }
                }
                for (; i < take; ++i) {
                    acc += ap[i] * allpaca_bf16_bits_to_f32(bp[i]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int off = KSPLIT / 2; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, KSPLIT);
    }
    if (brow && kt == 0) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
    }
}

/* gfx906 Triton sidecar (nlzy/triton-gfx906) beat 64x64x16 on DSV4 shexp:
 * BM=64 wastes half the FMAs at M=32/40; BK=32 + vec4 K-loads closed most of
 * the 1.8x gap without taking Triton into generate(). */
template <int BM, int BN, int BK>
__global__ void gemm_nt_bf16_tiled_kernel(const float *a, const uint16_t *b, float *c, int m, int n,
                                          int k) {
    constexpr int TX = 16;
    constexpr int TY = 16;
    constexpr int TM = BM / TY;
    constexpr int TN = BN / TX;
    constexpr int NTHR = TX * TY;
    constexpr int A_VEC_PER_THREAD = (BM * BK) / (4 * NTHR);
    constexpr int B_VEC_PER_THREAD = (BN * BK) / (4 * NTHR);
    static_assert(BM % TY == 0 && BN % TX == 0, "tile not divisible by 16x16");
    static_assert(BK % 4 == 0, "BK must be float4-wide");
    static_assert((BM * BK) % (4 * NTHR) == 0 && (BN * BK) % (4 * NTHR) == 0, "vec mapping");
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
        for (int t = 0; t < A_VEC_PER_THREAD; ++t) {
            const int elem = (tid + t * NTHR) * 4;
            const int r = elem / BK;
            const int p = elem % BK;
            const int gr = row0 + r;
            const int gp = k0 + p;
            const int nvalid = (gr < m && gp < k) ? (k - gp) : 0;
            const size_t off = static_cast<size_t>(gr) * static_cast<size_t>(k) +
                               static_cast<size_t>(gp);
            if (nvalid >= 4 && (off & 3u) == 0) {
                const float4 v = *reinterpret_cast<const float4 *>(a + off);
                As[r][p] = v.x;
                As[r][p + 1] = v.y;
                As[r][p + 2] = v.z;
                As[r][p + 3] = v.w;
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    As[r][p + i] = (gr < m && gp + i < k) ? a[off + static_cast<size_t>(i)] : 0.0f;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < B_VEC_PER_THREAD; ++t) {
            const int elem = (tid + t * NTHR) * 4;
            const int col = elem / BK;
            const int p = elem % BK;
            const int gc = col0 + col;
            const int gp = k0 + p;
            const int nvalid = (gc < n && gp < k) ? (k - gp) : 0;
            const size_t off = static_cast<size_t>(gc) * static_cast<size_t>(k) +
                               static_cast<size_t>(gp);
            uint16_t bits[4] = {0, 0, 0, 0};
            if (nvalid >= 4 && (off & 3u) == 0) {
                const uint2 v = *reinterpret_cast<const uint2 *>(b + off);
                bits[0] = static_cast<uint16_t>(v.x);
                bits[1] = static_cast<uint16_t>(v.x >> 16);
                bits[2] = static_cast<uint16_t>(v.y);
                bits[3] = static_cast<uint16_t>(v.y >> 16);
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (gc < n && gp + i < k) {
                        bits[i] = b[off + static_cast<size_t>(i)];
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                Bs[col][p + i] = allpaca_bf16_bits_to_f32(bits[i]);
            }
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

/* Staged Q8_0: 36-byte blocks, f32 scale at 0, qs[32] at 4 (4-byte aligned). */
constexpr int Q8_STAGED_STRIDE = 36;
constexpr int Q8_STAGED_QS = 4;

#ifndef ALLPACA_Q8_GEMV_INTDOT
/* Decode default stays float A × Q8: int-dot was not faster (weight-load bound). */
#define ALLPACA_Q8_GEMV_INTDOT 0
#endif

__device__ inline int32_t i8dot4(uint32_t a, uint32_t b, int32_t c) {
#if defined(__HIP_DEVICE_COMPILE__)
    union {
        uint32_t u;
        char4 c4;
    } ua, ub;
    ua.u = a;
    ub.u = b;
    return amd_mixed_dot(ua.c4, ub.c4, c, false);
#else
    int32_t acc = c;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc += static_cast<int32_t>(static_cast<int8_t>((a >> (8 * i)) & 0xff)) *
               static_cast<int32_t>(static_cast<int8_t>((b >> (8 * i)) & 0xff));
    }
    return acc;
#endif
}

/* One lane's 8-weight Q8_0 dot, with explicitly rounded ops so the 1-row and
 * multi-row kernels compute every row bit-identically (free FMA contraction
 * fused them differently, and a speculative verify row then drifted from
 * decode). */
__device__ inline float q8_dot8(const float *as, uint32_t qlo, uint32_t qhi) {
    const auto q = [](uint32_t w, int i) {
        return static_cast<float>(static_cast<int8_t>((w >> (8 * i)) & 0xff));
    };
    float dot = __fmul_rn(as[0], q(qlo, 0));
#pragma unroll
    for (int i = 1; i < 4; ++i) {
        dot = __fmaf_rn(as[i], q(qlo, i), dot);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        dot = __fmaf_rn(as[4 + i], q(qhi, i), dot);
    }
    return dot;
}

/* Float A × Q8 weight (fallback if integer MAC is slower). */
__global__ void gemv_nt_q8_0_f32_kernel(const float *a, const uint8_t *b, float *c, int m, int n,
                                        int k) {
    constexpr int WARPS = 16;
    constexpr int COLS = 8;
    constexpr int SUB = 4;
    constexpr int TILE = 8;
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
                acc = __fmaf_rn(d, q8_dot8(As + t * 32 + qbase, qlo, qhi), acc);
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

/* Q8 GEMV for 2..8 A rows (speculative verify spans, short chunks).
 * Same 36-byte staged layout and lane map as `gemv_nt_q8_0_f32_kernel`, but a
 * block owns MR rows: each weight block is loaded and decoded once and feeds
 * MR accumulators, instead of a grid row per A row re-reading all of B. */
template <int MR>
__global__ void gemv_nt_q8_0_f32_rows_kernel(const float *a, const uint8_t *b, float *c, int m,
                                             int n, int k) {
    constexpr int WARPS = 16;
    constexpr int COLS = 8;
    constexpr int SUB = 4;
    constexpr int TILE = 8;
    __shared__ float As[MR][TILE * 32];

    const int row0 = static_cast<int>(blockIdx.y) * MR;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int sub = lane / SUB;
    const int sublane = lane % SUB;
    const int col = (static_cast<int>(blockIdx.x) * WARPS + warp) * COLS + sub;
    const int nblocks = (k + 31) / 32;
    const size_t row_stride = static_cast<size_t>(nblocks) * static_cast<size_t>(Q8_STAGED_STRIDE);
    const uint8_t *brow = (col < n) ? b + static_cast<size_t>(col) * row_stride : nullptr;
    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        acc[r] = 0.0f;
    }
    const int tid = static_cast<int>(threadIdx.x);

    for (int blk0 = 0; blk0 < nblocks; blk0 += TILE) {
        const int ntile = (blk0 + TILE <= nblocks) ? TILE : (nblocks - blk0);
        const int a0 = blk0 * 32;
        const int nfloat = ntile * 32;
        for (int i = tid; i < MR * nfloat; i += static_cast<int>(blockDim.x)) {
            const int r = i / nfloat;
            const int j = i % nfloat;
            const int gi = a0 + j;
            As[r][j] = (row0 + r < m && gi < k)
                           ? a[static_cast<size_t>(row0 + r) * static_cast<size_t>(k) + gi]
                           : 0.0f;
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
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    acc[r] = __fmaf_rn(d, q8_dot8(&As[r][t * 32 + qbase], qlo, qhi), acc[r]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MR; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 2; off > 0; off >>= 1) {
            v += __shfl_down(v, off, 4);
        }
        if (col < n && sublane == 0 && row0 + r < m) {
            c[static_cast<size_t>(row0 + r) * static_cast<size_t>(n) + col] = v;
        }
    }
}

/* Decode Q8 GEMV: quantize A tiles to int8, int32 MAC with Q8 weights (ggml vec_dot_q8 idea). */
__global__ void gemv_nt_q8_0_intdot_kernel(const float *a, const uint8_t *b, float *c, int m, int n,
                                           int k) {
    constexpr int WARPS = 16;
    constexpr int COLS = 8;
    constexpr int SUB = 4;
    constexpr int TILE = 8;
    __shared__ float As[TILE * 32];
    __shared__ int8_t Aq[TILE * 32];
    __shared__ float Ad[TILE];

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

        if (warp < ntile) {
            const float v = As[warp * 32 + lane];
            float amax = fabsf(v);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                amax = fmaxf(amax, __shfl_xor(amax, off, 32));
            }
            const float d = (amax == 0.0f) ? 0.0f : amax / 127.0f;
            const float inv = (d == 0.0f) ? 0.0f : 1.0f / d;
            int q = static_cast<int>(roundf(v * inv));
            q = q > 127 ? 127 : (q < -127 ? -127 : q);
            Aq[warp * 32 + lane] = static_cast<int8_t>(q);
            if (lane == 0) {
                Ad[warp] = d;
            }
        }
        __syncthreads();

        if (brow) {
            for (int t = 0; t < ntile; ++t) {
                const uint8_t *bp = brow + static_cast<size_t>(blk0 + t) * Q8_STAGED_STRIDE;
                const float dw = *reinterpret_cast<const float *>(bp);
                const int qbase = sublane * 8;
                const uint8_t *qp = bp + Q8_STAGED_QS + qbase;
                const uint32_t qlo = *reinterpret_cast<const uint32_t *>(qp);
                const uint32_t qhi = *reinterpret_cast<const uint32_t *>(qp + 4);
                const uint32_t alo = *reinterpret_cast<const uint32_t *>(Aq + t * 32 + qbase);
                const uint32_t ahi = *reinterpret_cast<const uint32_t *>(Aq + t * 32 + qbase + 4);
                const int32_t sumi = i8dot4(alo, qlo, i8dot4(ahi, qhi, 0));
                acc += Ad[t] * dw * static_cast<float>(sumi);
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

/* Prefill Q8 NT GEMM: dequant one B tile into shared so columns are reused across A rows.
 * Decode (M<=8) stays on gemv_nt_q8_0_f32_kernel. */
__global__ void gemm_nt_q8_0_tiled_kernel(const float *a, const uint8_t *b, float *c, int m, int n,
                                         int k) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
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
    const int nblocks = (k + 31) / 32;
    const size_t brow_stride = static_cast<size_t>(nblocks) * static_cast<size_t>(Q8_STAGED_STRIDE);

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < k; k0 += BK) {
        const int blk = k0 / BK;
        for (int t = tid; t < BM * BK; t += TX * TY) {
            const int r = t / BK;
            const int p = t % BK;
            const int gr = row0 + r;
            const int gp = k0 + p;
            As[r][p] = (gr < m && gp < k) ? a[static_cast<size_t>(gr) * static_cast<size_t>(k) + gp]
                                         : 0.0f;
        }
        for (int t = tid; t < BN * BK; t += TX * TY) {
            const int col = t / BK;
            const int p = t % BK;
            const int gc = col0 + col;
            const int gp = k0 + p;
            float v = 0.0f;
            if (gc < n && gp < k) {
                const uint8_t *bp = b + static_cast<size_t>(gc) * brow_stride +
                                    static_cast<size_t>(blk) * static_cast<size_t>(Q8_STAGED_STRIDE);
                const float d = *reinterpret_cast<const float *>(bp);
                v = d * static_cast<float>(static_cast<int8_t>(bp[Q8_STAGED_QS + p]));
            }
            Bs[col][p] = v;
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
                c[static_cast<size_t>(gr) * static_cast<size_t>(n) + gc] = acc[i][j];
            }
        }
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

/* Prefill F16 NT GEMM: convert B tiles to shared F32 so F16 weights reuse across A rows.
 * Decode (M<=8) stays on gemv_nt_f16_kernel. */
__global__ void gemm_nt_f16_tiled_kernel(const float *a, const uint16_t *b, float *c, int m, int n,
                                         int k) {
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
            Bs[col][p] = (gc < n && gp < k)
                             ? allpaca_f16_bits_to_f32(b[static_cast<size_t>(gc) * k + gp])
                             : 0.0f;
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
                                       int causal, const uint8_t *mask, const float *sinks) {
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
    if (sinks) {
        m = sinks[h];
        l = 1.0f;
    }

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

#ifdef ALLPACA_HAS_ROCBLAS
/* Row-major C[m,n] = A[m,k] * B[n,k]^T via column-major rocBLAS (swap A/B, N/M). */
static int launch_gemm_nt_rocblas(const float *a, const float *b, float *c, int32_t m, int32_t n,
                                  int32_t k, hipStream_t stream) {
    static std::mutex mu;
    static rocblas_handle handle = nullptr;
    static bool tried = false;
    std::lock_guard<std::mutex> lock(mu);
    if (!tried) {
        tried = true;
        if (rocblas_create_handle(&handle) != rocblas_status_success) {
            handle = nullptr;
        }
    }
    if (!handle) {
        return ALLPACA_ERR_NOT_IMPLEMENTED;
    }
    if (rocblas_set_stream(handle, stream) != rocblas_status_success) {
        return ALLPACA_ERR_DEVICE;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    rocblas_status st = rocblas_sgemm(handle, rocblas_operation_transpose, rocblas_operation_none, n,
                                      m, k, &alpha, b, k, a, k, &beta, c, n);
    return st == rocblas_status_success ? ALLPACA_OK : ALLPACA_ERR_DEVICE;
}
#endif

#ifdef ALLPACA_HAS_HIPBLASLT
static void ensure_hipblaslt_tensile_path() {
    const char *cur = std::getenv("HIPBLASLT_TENSILE_LIBPATH");
    if (cur && cur[0]) {
        return;
    }
    const char *hip = std::getenv("HIP_PATH");
    if (!hip || !hip[0]) {
        hip = std::getenv("ROCM_PATH");
    }
    if (!hip || !hip[0]) {
        return;
    }
    size_t n = std::strlen(hip);
    while (n > 0 && (hip[n - 1] == '\\' || hip[n - 1] == '/')) {
        n--;
    }
    char path[1024];
#ifdef _WIN32
    const char *fmt = "%.*s\\bin\\hipblaslt\\library";
#else
    const char *fmt = "%.*s/bin/hipblaslt/library";
#endif
    if (std::snprintf(path, sizeof(path), fmt, static_cast<int>(n), hip) < 0) {
        return;
    }
#ifdef _WIN32
    _putenv_s("HIPBLASLT_TENSILE_LIBPATH", path);
#else
    setenv("HIPBLASLT_TENSILE_LIBPATH", path, 0);
#endif
}

struct HipblasLtDescs {
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    ~HipblasLtDescs() {
        if (a) {
            hipblasLtMatrixLayoutDestroy(a);
        }
        if (b) {
            hipblasLtMatrixLayoutDestroy(b);
        }
        if (c) {
            hipblasLtMatrixLayoutDestroy(c);
        }
        if (desc) {
            hipblasLtMatmulDescDestroy(desc);
        }
        if (pref) {
            hipblasLtMatmulPreferenceDestroy(pref);
        }
    }
};

/* Column-major swap matching rocBLAS: C_col[n,m] = B[n,k]^T * A[m,k].
 * Row-major ORDER_ROW found algos on gfx1201 but mismatched CPU (e.g. 1x8x4). */
static int lt_make_col_rocblas(HipblasLtDescs *d, int32_t m, int32_t n, int32_t k) {
    if (hipblasLtMatrixLayoutCreate(&d->a, HIP_R_32F, static_cast<uint64_t>(k),
                                    static_cast<uint64_t>(n), k) != HIPBLAS_STATUS_SUCCESS) {
        return ALLPACA_ERR_DEVICE;
    }
    if (hipblasLtMatrixLayoutCreate(&d->b, HIP_R_32F, static_cast<uint64_t>(k),
                                    static_cast<uint64_t>(m), k) != HIPBLAS_STATUS_SUCCESS) {
        return ALLPACA_ERR_DEVICE;
    }
    if (hipblasLtMatrixLayoutCreate(&d->c, HIP_R_32F, static_cast<uint64_t>(n),
                                    static_cast<uint64_t>(m), n) != HIPBLAS_STATUS_SUCCESS) {
        return ALLPACA_ERR_DEVICE;
    }
    if (hipblasLtMatmulDescCreate(&d->desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
        HIPBLAS_STATUS_SUCCESS) {
        return ALLPACA_ERR_DEVICE;
    }
    hipblasOperation_t transA = HIPBLAS_OP_T;
    hipblasOperation_t transB = HIPBLAS_OP_N;
    if (hipblasLtMatmulDescSetAttribute(d->desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA,
                                        sizeof(transA)) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatmulDescSetAttribute(d->desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB,
                                        sizeof(transB)) != HIPBLAS_STATUS_SUCCESS) {
        return ALLPACA_ERR_DEVICE;
    }
    return ALLPACA_OK;
}

struct HipblasLtDev {
    hipblasLtHandle_t handle = nullptr;
    bool tried = false;
    void *workspace = nullptr;
    size_t workspace_bytes = 0;
    int32_t cache_m = 0;
    int32_t cache_n = 0;
    int32_t cache_k = 0;
    int cache_status = 0;
    hipblasLtMatmulAlgo_t algo{};
    size_t algo_ws = 0;
};

static int lt_grow_workspace(HipblasLtDev *st, size_t need, hipStream_t stream) {
    if (need == 0) {
        return ALLPACA_OK;
    }
    if (need <= st->workspace_bytes && st->workspace) {
        return ALLPACA_OK;
    }
    if (st->workspace) {
        (void)hipStreamSynchronize(stream);
        (void)hipFree(st->workspace);
        st->workspace = nullptr;
        st->workspace_bytes = 0;
    }
    size_t bytes = (need + 15u) & ~size_t{15};
    if (hipMalloc(&st->workspace, bytes) != hipSuccess) {
        st->workspace = nullptr;
        return ALLPACA_ERR_DEVICE;
    }
    st->workspace_bytes = bytes;
    return ALLPACA_OK;
}

static int lt_run(HipblasLtDev *st, const float *a, const float *b, float *c, int32_t m, int32_t n,
                  int32_t k, hipStream_t stream, int *out_algos) {
    HipblasLtDescs d;
    int rc = lt_make_col_rocblas(&d, m, n, k);
    if (rc != ALLPACA_OK) {
        return rc;
    }
    if (hipblasLtMatmulPreferenceCreate(&d.pref) != HIPBLAS_STATUS_SUCCESS) {
        return ALLPACA_ERR_DEVICE;
    }
    const uint64_t max_ws = 64ull * 1024ull * 1024ull;
    if (hipblasLtMatmulPreferenceSetAttribute(d.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_ws, sizeof(max_ws)) != HIPBLAS_STATUS_SUCCESS) {
        return ALLPACA_ERR_DEVICE;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    hipblasLtMatmulAlgo_t algo{};
    size_t ws = 0;
    bool have_algo = st->cache_status == 1 && st->cache_m == m && st->cache_n == n && st->cache_k == k;
    if (have_algo) {
        algo = st->algo;
        ws = st->algo_ws;
        if (out_algos) {
            *out_algos = 1;
        }
    } else {
        hipblasLtMatmulHeuristicResult_t heur[8];
        int returned = 0;
        hipblasStatus_t hs = hipblasLtMatmulAlgoGetHeuristic(st->handle, d.desc, d.a, d.b, d.c, d.c,
                                                             d.pref, 8, heur, &returned);
        if (out_algos) {
            *out_algos = returned;
        }
        if (hs != HIPBLAS_STATUS_SUCCESS || returned <= 0) {
            return ALLPACA_ERR_NOT_IMPLEMENTED;
        }
        bool picked = false;
        for (int i = 0; i < returned; i++) {
            if (heur[i].state != HIPBLAS_STATUS_SUCCESS) {
                continue;
            }
            algo = heur[i].algo;
            ws = heur[i].workspaceSize;
            picked = true;
            break;
        }
        if (!picked) {
            return ALLPACA_ERR_NOT_IMPLEMENTED;
        }
    }
    if (lt_grow_workspace(st, ws, stream) != ALLPACA_OK) {
        return ALLPACA_ERR_DEVICE;
    }
    hipblasStatus_t ms =
        hipblasLtMatmul(st->handle, d.desc, &alpha, b, d.a, a, d.b, &beta, c, d.c, c, d.c, &algo,
                        st->workspace, st->workspace_bytes, stream);
    if (ms != HIPBLAS_STATUS_SUCCESS) {
        return ALLPACA_ERR_DEVICE;
    }
    st->cache_m = m;
    st->cache_n = n;
    st->cache_k = k;
    st->cache_status = 1;
    st->algo = algo;
    st->algo_ws = ws;
    return ALLPACA_OK;
}

static int launch_gemm_nt_hipblaslt(const float *a, const float *b, float *c, int32_t m, int32_t n,
                                    int32_t k, hipStream_t stream) {
    static std::mutex mu;
    static HipblasLtDev devs[16];
    static bool tensile_set = false;
    static bool logged_ok = false;
    static bool logged_miss = false;
    std::lock_guard<std::mutex> lock(mu);
    if (!tensile_set) {
        tensile_set = true;
        ensure_hipblaslt_tensile_path();
    }
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess || device < 0 || device >= 16) {
        return ALLPACA_ERR_DEVICE;
    }
    HipblasLtDev &st = devs[device];
    if (!st.tried) {
        st.tried = true;
        if (hipblasLtCreate(&st.handle) != HIPBLAS_STATUS_SUCCESS) {
            st.handle = nullptr;
        }
    }
    if (!st.handle) {
        return ALLPACA_ERR_NOT_IMPLEMENTED;
    }
    if (st.cache_status == 2 && st.cache_m == m && st.cache_n == n && st.cache_k == k) {
        return ALLPACA_ERR_NOT_IMPLEMENTED;
    }
    int algos = 0;
    int rc = lt_run(&st, a, b, c, m, n, k, stream, &algos);
    if (rc == ALLPACA_OK) {
        if (!logged_ok) {
            logged_ok = true;
            std::fprintf(stderr, "allpaca: hipBLASLt F32 GEMM ok algos=%d m=%d n=%d k=%d ws=%zu\n",
                         algos, m, n, k, st.algo_ws);
        }
        return ALLPACA_OK;
    }
    st.cache_m = m;
    st.cache_n = n;
    st.cache_k = k;
    st.cache_status = 2;
    if (!logged_miss) {
        logged_miss = true;
        std::fprintf(stderr, "allpaca: hipBLASLt F32 GEMM miss m=%d n=%d k=%d algos=%d\n", m, n, k,
                     algos);
    }
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}
#endif

static int launch_gemm_nt(const float *a, const float *b, float *c, int32_t m, int32_t n, int32_t k,
                          hipStream_t stream, int tiled) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
#ifdef ALLPACA_HAS_HIPBLASLT
    /* M=1 lm_head: hipBLASLt was ~378 ms vs ~0.5 ms custom GEMV. Keep Lt for wide/prefill only. */
    if (m > 8) {
        int rc = launch_gemm_nt_hipblaslt(a, b, c, m, n, k, stream);
        if (rc == ALLPACA_OK) {
            return ALLPACA_OK;
        }
    }
#endif
#ifdef ALLPACA_HAS_ROCBLAS
    /* Decode is M=1; rocBLAS lost on skinny GEMV, so only wide/prefill GEMM uses it. */
    if (m > 8) {
        int rc = launch_gemm_nt_rocblas(a, b, c, m, n, k, stream);
        if (rc == ALLPACA_OK) {
            return ALLPACA_OK;
        }
    }
#endif
    /* Decode is always M=1; small-M prefill stays on the coalesced GEMV. */
    if (m <= 8) {
        if (tiled) {
            constexpr int COLS = 64;
            dim3 block(256);
            dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS), static_cast<unsigned>(m));
            hipLaunchKernelGGL(gemv_nt_tiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
        } else {
            constexpr int warps = 8;
            constexpr int cols = 4;
            constexpr int cols_per_block = warps * cols;
            dim3 block(256);
            dim3 grid(static_cast<unsigned>((n + cols_per_block - 1) / cols_per_block),
                      static_cast<unsigned>(m));
            hipLaunchKernelGGL(gemv_nt_untiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
        }
    } else {
        constexpr int BM = 64;
        constexpr int BN = 64;
        dim3 block(16, 16);
        dim3 grid(static_cast<unsigned>((n + BN - 1) / BN), static_cast<unsigned>((m + BM - 1) / BM));
        hipLaunchKernelGGL(gemm_nt_tiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    }
    return hip_status(hipGetLastError());
}

static int launch_gemm_nt_f16(const float *a, const uint16_t *b, float *c, int32_t m, int32_t n,
                              int32_t k, hipStream_t stream, int tiled) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    /* Decode M<=8 stays on GEMV. Prefill M>8 reuses F16 B tiles across A rows. */
    if (m > 8) {
        constexpr int BM = 64;
        constexpr int BN = 64;
        dim3 block(16, 16);
        dim3 grid(static_cast<unsigned>((n + BN - 1) / BN),
                  static_cast<unsigned>((m + BM - 1) / BM));
        hipLaunchKernelGGL(gemm_nt_f16_tiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
        return hip_status(hipGetLastError());
    }
    if (tiled) {
        constexpr int COLS = 64;
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS), static_cast<unsigned>(m));
        hipLaunchKernelGGL(gemv_nt_f16_tiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    } else {
        constexpr int warps = 16;
        constexpr int cols = 8;
        constexpr int cols_per_block = warps * cols;
        dim3 block(512);
        dim3 grid(static_cast<unsigned>((n + cols_per_block - 1) / cols_per_block),
                  static_cast<unsigned>(m));
        hipLaunchKernelGGL(gemv_nt_f16_untiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    }
    return hip_status(hipGetLastError());
}

static int launch_gemm_nt_bf16(const float *a, const uint16_t *b, float *c, int32_t m, int32_t n,
                               int32_t k, hipStream_t stream, int tiled) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    if (m > 8) {
        /* M<=40: 32x64x32 (DSV4 shexp prefill 32/40). Wider M: 64x64x32. */
        dim3 block(16, 16);
        if (m <= 40) {
            constexpr int BM = 32;
            constexpr int BN = 64;
            dim3 grid(static_cast<unsigned>((n + BN - 1) / BN),
                      static_cast<unsigned>((m + BM - 1) / BM));
            hipLaunchKernelGGL((gemm_nt_bf16_tiled_kernel<32, 64, 32>), grid, block, 0, stream, a, b,
                               c, m, n, k);
        } else {
            constexpr int BM = 64;
            constexpr int BN = 64;
            dim3 grid(static_cast<unsigned>((n + BN - 1) / BN),
                      static_cast<unsigned>((m + BM - 1) / BM));
            hipLaunchKernelGGL((gemm_nt_bf16_tiled_kernel<64, 64, 32>), grid, block, 0, stream, a, b,
                               c, m, n, k);
        }
        return hip_status(hipGetLastError());
    }
    if (tiled) {
        constexpr int COLS = 64;
        dim3 block(128);
        dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS), static_cast<unsigned>(m));
        hipLaunchKernelGGL(gemv_nt_bf16_tiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    } else {
        constexpr int warps = 16;
        constexpr int cols = 8;
        constexpr int cols_per_block = warps * cols;
        dim3 block(512);
        dim3 grid(static_cast<unsigned>((n + cols_per_block - 1) / cols_per_block),
                  static_cast<unsigned>(m));
        hipLaunchKernelGGL(gemv_nt_bf16_untiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    }
    return hip_status(hipGetLastError());
}

static int launch_gemm_nt_q8_0(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                               int32_t k, hipStream_t stream, int intdot) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    /* M=1 decode stays on the 36-byte GEMV. Prefill M>8 reuses Q8 B tiles across A rows. */
    // Up to 16 rows stay on the GEMV kernels: a verify span (max_depth 8 plus
    // a deferred correction) must match decode bit for bit, and the tiled
    // prefill GEMM sums in another order.
    if (!intdot && m > 16) {
        constexpr int BM = 64;
        constexpr int BN = 64;
        dim3 block(16, 16);
        dim3 grid(static_cast<unsigned>((n + BN - 1) / BN),
                  static_cast<unsigned>((m + BM - 1) / BM));
        hipLaunchKernelGGL(gemm_nt_q8_0_tiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
        return hip_status(hipGetLastError());
    }
    constexpr int warps = 16;
    constexpr int cols = 8;
    constexpr int cols_per_block = warps * cols;
    dim3 block(512);
    const unsigned gx = static_cast<unsigned>((n + cols_per_block - 1) / cols_per_block);
    if (!intdot && m > 1) {
        // 2..16 rows share each weight block load (verify spans).
        if (m == 2) {
            hipLaunchKernelGGL(gemv_nt_q8_0_f32_rows_kernel<2>, dim3(gx, 1), block, 0, stream, a,
                               b, c, m, n, k);
        } else if (m <= 4) {
            hipLaunchKernelGGL(gemv_nt_q8_0_f32_rows_kernel<4>, dim3(gx, 1), block, 0, stream, a,
                               b, c, m, n, k);
        } else {
            hipLaunchKernelGGL(gemv_nt_q8_0_f32_rows_kernel<8>,
                               dim3(gx, static_cast<unsigned>((m + 7) / 8)), block, 0, stream, a,
                               b, c, m, n, k);
        }
        return hip_status(hipGetLastError());
    }
    dim3 grid(gx, static_cast<unsigned>(m));
    if (intdot) {
        hipLaunchKernelGGL(gemv_nt_q8_0_intdot_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    } else {
        hipLaunchKernelGGL(gemv_nt_q8_0_f32_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    }
    return hip_status(hipGetLastError());
}

__device__ void q4k_scale_min(int j, const uint8_t *scales, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0x0F) | static_cast<uint8_t>((scales[j - 4] >> 6) << 4);
        *m = (scales[j + 4] >> 4) | static_cast<uint8_t>((scales[j] >> 6) << 4);
    }
}

/* Warp-per-column K-quant GEMV (Q4_K / Q5_K / Q6_K).
 *
 * The per-thread-column kernels below stream a whole weight row per thread:
 * adjacent lanes read addresses a row (~4 KiB) apart, so no load coalesces
 * and a 5120x17408 Q6_K GEMV ran at ~15% of VRAM bandwidth on RDNA4. Here a
 * 32-lane group owns one output column and every lane dequantizes 8 of each
 * 256-weight super-block, laid out so lane-adjacent bytes (and A floats) are
 * contiguous. Up to MR rows of A reuse each weight load, so small verify
 * batches and prefill chunks re-read B ceil(m/MR) times instead of m times.
 * Lane groups are 32 wide on wave32 and wave64 alike (warp_sum32 width 32). */
constexpr int KQ_LANES = 32;
constexpr int KQ_COLS_PER_BLOCK = 8;

__device__ inline void kq_lane_q4_k(const uint8_t *blk, int l, float *w, int *off) {
    const float d = allpaca_f16le_to_f32(blk);
    const float dmin = allpaca_f16le_to_f32(blk + 2);
    const uint8_t *scales = blk + 4;
    const uint8_t *qs = blk + 16;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        uint8_t sc1 = 0, m1 = 0, sc2 = 0, m2 = 0;
        q4k_scale_min(2 * g, scales, &sc1, &m1);
        q4k_scale_min(2 * g + 1, scales, &sc2, &m2);
        const uint8_t q = qs[g * 32 + l];
        w[2 * g] = d * static_cast<float>(sc1) * static_cast<float>(q & 0x0F) -
                   dmin * static_cast<float>(m1);
        w[2 * g + 1] = d * static_cast<float>(sc2) * static_cast<float>(q >> 4) -
                       dmin * static_cast<float>(m2);
        off[2 * g] = g * 64 + l;
        off[2 * g + 1] = g * 64 + 32 + l;
    }
}

__device__ inline void kq_lane_q5_k(const uint8_t *blk, int l, float *w, int *off) {
    const float d = allpaca_f16le_to_f32(blk);
    const float dmin = allpaca_f16le_to_f32(blk + 2);
    const uint8_t *scales = blk + 4;
    const uint8_t qh = blk[16 + l];
    const uint8_t *qs = blk + 48;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        uint8_t sc1 = 0, m1 = 0, sc2 = 0, m2 = 0;
        q4k_scale_min(2 * g, scales, &sc1, &m1);
        q4k_scale_min(2 * g + 1, scales, &sc2, &m2);
        const uint8_t q = qs[g * 32 + l];
        const float lo = static_cast<float>(q & 0x0F) + (((qh >> (2 * g)) & 1) ? 16.f : 0.f);
        const float hi = static_cast<float>(q >> 4) + (((qh >> (2 * g + 1)) & 1) ? 16.f : 0.f);
        w[2 * g] = d * static_cast<float>(sc1) * lo - dmin * static_cast<float>(m1);
        w[2 * g + 1] = d * static_cast<float>(sc2) * hi - dmin * static_cast<float>(m2);
        off[2 * g] = g * 64 + l;
        off[2 * g + 1] = g * 64 + 32 + l;
    }
}

__device__ inline void kq_lane_q6_k(const uint8_t *blk, int l, float *w, int *off) {
    const float d = allpaca_f16le_to_f32(blk + 208);
    const int8_t *sc = reinterpret_cast<const int8_t *>(blk + 192);
    const int is = l / 16;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint8_t ql0 = blk[h * 64 + l];
        const uint8_t ql1 = blk[h * 64 + 32 + l];
        const uint8_t qh = blk[128 + h * 32 + l];
        const int8_t *s = sc + h * 8 + is;
        const int q1 = static_cast<int>((ql0 & 0x0F) | ((qh & 3) << 4)) - 32;
        const int q2 = static_cast<int>((ql1 & 0x0F) | (((qh >> 2) & 3) << 4)) - 32;
        const int q3 = static_cast<int>((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32;
        const int q4 = static_cast<int>((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32;
        w[4 * h + 0] = d * static_cast<float>(s[0]) * static_cast<float>(q1);
        w[4 * h + 1] = d * static_cast<float>(s[2]) * static_cast<float>(q2);
        w[4 * h + 2] = d * static_cast<float>(s[4]) * static_cast<float>(q3);
        w[4 * h + 3] = d * static_cast<float>(s[6]) * static_cast<float>(q4);
        off[4 * h + 0] = h * 128 + l;
        off[4 * h + 1] = h * 128 + 32 + l;
        off[4 * h + 2] = h * 128 + 64 + l;
        off[4 * h + 3] = h * 128 + 96 + l;
    }
}

/* v2 lane maps: fewer, wider loads and 2-4 scales per lane instead of 8.
 *
 * Q4_K / Q5_K: 8 lanes share one 64-weight group g = l/8. Lane l loads the
 * 4 qs bytes at j0 = 4*(l%8) as one u32: low nibbles are weights g*64+j0..+3
 * (sub-block 2g), high nibbles g*64+32+j0..+3 (sub-block 2g+1). Each half is
 * d*sum(a*q) - m*sum(a), with A read as float4. Needs 4-byte aligned blocks
 * and 16-byte aligned A rows (the launcher checks both). */
/* Q4_K/Q5_K 6-bit scale/min pair for sub-blocks 2g and 2g+1, branch-free.
 * `q4k_scale_min` branches on j < 4; in the v2 map j differs across 8-lane
 * groups, so a branch diverges the warp. Bytes: s[0..3] low scales,
 * s[4..7] low mins, s[8..11] packed nibbles + the 2 high bits in s[0..7]. */
__device__ inline void kq_scale_min_pair(const uint8_t *scales, int g, uint32_t &sc1,
                                         uint32_t &m1, uint32_t &sc2, uint32_t &m2) {
    const uint32_t w0 = *reinterpret_cast<const uint32_t *>(scales);
    const uint32_t w1 = *reinterpret_cast<const uint32_t *>(scales + 4);
    const uint32_t w2 = *reinterpret_cast<const uint32_t *>(scales + 8);
    const int j1 = 2 * g;
    const int j2 = 2 * g + 1;
    auto byte = [](uint32_t w, int i) { return (w >> (8 * (i & 3))) & 0xFFu; };
    // j < 4: sc = s[j] & 63, m = s[j+4] & 63.
    // j >= 4: sc = (s[j+4] & 15) | ((s[j-4] >> 6) << 4),
    //         m = (s[j+4] >> 4) | ((s[j] >> 6) << 4).
    const bool hi = g >= 2;
    const uint32_t a1 = byte(w0, j1), b1 = byte(w1, j1), c1 = byte(w2, j1);
    const uint32_t a2 = byte(w0, j2), b2 = byte(w1, j2), c2 = byte(w2, j2);
    sc1 = hi ? ((c1 & 15) | ((a1 >> 6) << 4)) : (a1 & 63);
    m1 = hi ? ((c1 >> 4) | ((b1 >> 6) << 4)) : (b1 & 63);
    sc2 = hi ? ((c2 & 15) | ((a2 >> 6) << 4)) : (a2 & 63);
    m2 = hi ? ((c2 >> 4) | ((b2 >> 6) << 4)) : (b2 & 63);
}

struct Kq45Lane {
    float dl, ml, dh, mh;
    float lo[4], hi[4];
    int off;
};

template <bool Q5>
__device__ inline void kq_lane_q45_v2(const uint8_t *blk, int l, Kq45Lane &w) {
    const int g = l >> 3;
    const int j0 = (l & 7) * 4;
    const float d = allpaca_f16le_to_f32(blk);
    const float dmin = allpaca_f16le_to_f32(blk + 2);
    uint32_t sc1 = 0, m1 = 0, sc2 = 0, m2 = 0;
    kq_scale_min_pair(blk + 4, g, sc1, m1, sc2, m2);
    const uint8_t *qs = blk + (Q5 ? 48 : 16);
    const uint32_t q = *reinterpret_cast<const uint32_t *>(qs + g * 32 + j0);
    uint32_t hb_lo = 0, hb_hi = 0;
    if constexpr (Q5) {
        const uint32_t qh = *reinterpret_cast<const uint32_t *>(blk + 16 + j0);
        hb_lo = (qh >> (2 * g)) & 0x01010101u;
        hb_hi = (qh >> (2 * g + 1)) & 0x01010101u;
    }
    const uint32_t lo = (q & 0x0F0F0F0Fu) | (hb_lo << 4);
    const uint32_t hi = ((q >> 4) & 0x0F0F0F0Fu) | (hb_hi << 4);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        w.lo[i] = static_cast<float>((lo >> (8 * i)) & 0xFF);
        w.hi[i] = static_cast<float>((hi >> (8 * i)) & 0xFF);
    }
    w.dl = d * static_cast<float>(sc1);
    w.ml = dmin * static_cast<float>(m1);
    w.dh = d * static_cast<float>(sc2);
    w.mh = dmin * static_cast<float>(m2);
    w.off = g * 64 + j0;
}

__device__ inline float kq_apply_q45_v2(const Kq45Lane &w, const float *ab) {
    const float4 alo = *reinterpret_cast<const float4 *>(ab + w.off);
    const float4 ahi = *reinterpret_cast<const float4 *>(ab + w.off + 32);
    const float dot_lo = alo.x * w.lo[0] + alo.y * w.lo[1] + alo.z * w.lo[2] + alo.w * w.lo[3];
    const float dot_hi = ahi.x * w.hi[0] + ahi.y * w.hi[1] + ahi.z * w.hi[2] + ahi.w * w.hi[3];
    const float sum_lo = alo.x + alo.y + alo.z + alo.w;
    const float sum_hi = ahi.x + ahi.y + ahi.z + ahi.w;
    return w.dl * dot_lo - w.ml * sum_lo + w.dh * dot_hi - w.mh * sum_hi;
}

/* Q6_K (210-byte blocks, only 2-byte aligned): lane l owns the ql byte pair
 * p = 2l of each 64-byte half h. p < 32 holds weights h*128+p (low nibble,
 * qh bits 0-1) and h*128+64+p (high, bits 4-5); p >= 32 holds h*128+p (low,
 * bits 2-3) and h*128+64+p (high, bits 6-7), with p' = p-32 indexing qh.
 * Both weights of a pair share one scale. A is read as float2. */
struct Kq6Lane {
    float s_lo[2], s_hi[2];
    float q[2][4];
    int p;
};

__device__ inline void kq_lane_q6_k_v2(const uint8_t *blk, int l, Kq6Lane &w) {
    const float d = allpaca_f16le_to_f32(blk + 208);
    const int8_t *sc = reinterpret_cast<const int8_t *>(blk + 192);
    const int p = 2 * l;
    const bool second = p >= 32;
    const int pp = second ? p - 32 : p;
    const int sh_lo = second ? 2 : 0;
    const int sh_hi = second ? 6 : 4;
    const int is = pp / 16 + (second ? 2 : 0);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t ql = *reinterpret_cast<const uint16_t *>(blk + h * 64 + p);
        const uint32_t qh = *reinterpret_cast<const uint16_t *>(blk + 128 + h * 32 + pp);
        w.q[h][0] = static_cast<float>(
            static_cast<int>((ql & 0x0F) | (((qh >> sh_lo) & 3) << 4)) - 32);
        w.q[h][1] = static_cast<float>(
            static_cast<int>(((ql >> 8) & 0x0F) | (((qh >> (8 + sh_lo)) & 3) << 4)) - 32);
        w.q[h][2] = static_cast<float>(
            static_cast<int>(((ql >> 4) & 0x0F) | (((qh >> sh_hi) & 3) << 4)) - 32);
        w.q[h][3] = static_cast<float>(
            static_cast<int>(((ql >> 12) & 0x0F) | (((qh >> (8 + sh_hi)) & 3) << 4)) - 32);
        w.s_lo[h] = d * static_cast<float>(sc[h * 8 + is]);
        w.s_hi[h] = d * static_cast<float>(sc[h * 8 + is + 4]);
    }
    w.p = p;
}

__device__ inline float kq_apply_q6_k_v2(const Kq6Lane &w, const float *ab) {
    float acc = 0.f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const float2 alo = *reinterpret_cast<const float2 *>(ab + h * 128 + w.p);
        const float2 ahi = *reinterpret_cast<const float2 *>(ab + h * 128 + 64 + w.p);
        acc += w.s_lo[h] * (alo.x * w.q[h][0] + alo.y * w.q[h][1]) +
               w.s_hi[h] * (ahi.x * w.q[h][2] + ahi.y * w.q[h][3]);
    }
    return acc;
}

template <int QT>
struct KqBlock;
template <>
struct KqBlock<3> {
    static constexpr int bytes = 110;
};
template <>
struct KqBlock<4> {
    static constexpr int bytes = 144;
};
template <>
struct KqBlock<5> {
    static constexpr int bytes = 176;
};
template <>
struct KqBlock<6> {
    static constexpr int bytes = 210;
};

template <int QT, int MR>
__global__ void gemv_nt_kquant_warp_kernel(const float *a, const uint8_t *b, float *c, int m,
                                           int n, int k) {
    constexpr int BB = KqBlock<QT>::bytes;
    const int lane = static_cast<int>(threadIdx.x) % KQ_LANES;
    const int col = static_cast<int>(blockIdx.x) * KQ_COLS_PER_BLOCK +
                    static_cast<int>(threadIdx.x) / KQ_LANES;
    const int row0 = static_cast<int>(blockIdx.y) * MR;
    if (col >= n) {
        return;
    }
    const int nblocks = k / 256;
    const uint8_t *brow = b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * BB;
    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        acc[r] = 0.f;
    }
#pragma unroll 2
    for (int ib = 0; ib < nblocks; ++ib) {
        const uint8_t *blk = brow + static_cast<size_t>(ib) * BB;
        const float *ab = a + static_cast<size_t>(ib) * 256;
        float w[8];
        int off[8];
        if constexpr (QT == 4) {
            kq_lane_q4_k(blk, lane, w, off);
        } else if constexpr (QT == 5) {
            kq_lane_q5_k(blk, lane, w, off);
        } else {
            kq_lane_q6_k(blk, lane, w, off);
        }
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            if (row0 + r < m) {
                const float *ar = ab + static_cast<size_t>(row0 + r) * static_cast<size_t>(k);
                float s = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    s += ar[off[j]] * w[j];
                }
                acc[r] += s;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        const float s = warp_sum32(acc[r]);
        if (lane == 0 && row0 + r < m) {
            c[static_cast<size_t>(row0 + r) * static_cast<size_t>(n) + static_cast<size_t>(col)] =
                s;
        }
    }
}

/* v2 lane maps (see `kq_lane_q45_v2` / `kq_lane_q6_k_v2`) for 1..MR A rows
 * read straight from global; L1/L2 serve the reuse across a block's column
 * warps. Staging A in LDS was tried: a barrier per super-block cost single-row
 * decode ~30%, and multi-row stays arithmetic-bound either way. */
template <int QT, int MR>
__global__ void gemv_nt_kquant_v2_kernel(const float *a, const uint8_t *b, float *c, int m, int n,
                                         int k) {
    constexpr int BB = KqBlock<QT>::bytes;
    const int lane = static_cast<int>(threadIdx.x) % KQ_LANES;
    const int col = static_cast<int>(blockIdx.x) * KQ_COLS_PER_BLOCK +
                    static_cast<int>(threadIdx.x) / KQ_LANES;
    const int row0 = static_cast<int>(blockIdx.y) * MR;
    if (col >= n) {
        return;
    }
    const int nblocks = k / 256;
    const uint8_t *brow = b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * BB;
    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        acc[r] = 0.f;
    }
#pragma unroll 2
    for (int ib = 0; ib < nblocks; ++ib) {
        const uint8_t *blk = brow + static_cast<size_t>(ib) * BB;
        const float *ab = a + static_cast<size_t>(ib) * 256;
        if constexpr (QT == 6) {
            Kq6Lane w;
            kq_lane_q6_k_v2(blk, lane, w);
#pragma unroll
            for (int r = 0; r < MR; ++r) {
                if (row0 + r < m) {
                    acc[r] += kq_apply_q6_k_v2(
                        w, ab + static_cast<size_t>(row0 + r) * static_cast<size_t>(k));
                }
            }
        } else {
            Kq45Lane w;
            kq_lane_q45_v2<QT == 5>(blk, lane, w);
#pragma unroll
            for (int r = 0; r < MR; ++r) {
                if (row0 + r < m) {
                    acc[r] += kq_apply_q45_v2(
                        w, ab + static_cast<size_t>(row0 + r) * static_cast<size_t>(k));
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        const float s = warp_sum32(acc[r]);
        if (lane == 0 && row0 + r < m) {
            c[static_cast<size_t>(row0 + r) * static_cast<size_t>(n) + static_cast<size_t>(col)] =
                s;
        }
    }
}

/* K-quant GEMV on int8 activations (llama.cpp mmvq): A is quantized once
 * per GEMV to int8 with one f32 scale per 32 values, and every row is a
 * signed dot4 against the unpacked weight bytes. With each lane owning 32
 * weights, a row costs a few dot4s and one scale FMA per lane, so 1..4 rows
 * run at weight bandwidth. Every m uses this path so a verify row reproduces
 * the decode (m == 1) result bit for bit. */
__global__ void quantize_rows_q8_kernel(const float *a, int8_t *aq, float *ad, int m, int k) {
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int blk = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / 32) + warp;
    const int row = static_cast<int>(blockIdx.y);
    const int kb = k / 32;
    if (row >= m || blk >= kb) {
        return;
    }
    const size_t i = static_cast<size_t>(row) * static_cast<size_t>(k) + blk * 32 + lane;
    const float v = a[i];
    float amax = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, off, 32));
    }
    const float d = amax == 0.f ? 0.f : amax / 127.f;
    const float inv = d == 0.f ? 0.f : 1.f / d;
    int q = static_cast<int>(roundf(v * inv));
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    aq[i] = static_cast<int8_t>(q);
    if (lane == 0) {
        ad[static_cast<size_t>(row) * kb + blk] = d;
    }
}

__device__ inline uint32_t kq_u32_from_u16(const uint8_t *p) {
    const uint16_t *h = reinterpret_cast<const uint16_t *>(p);
    return static_cast<uint32_t>(h[0]) | (static_cast<uint32_t>(h[1]) << 16);
}

/* Lane maps: LPC lanes per column, W = 128 / LPC contiguous weights per lane
 * in each of two places. Q4_K/Q5_K: sub-blocks 2g (low nibbles) and 2g+1
 * (high) of group g = sl / (LPC / 4), from j0 = W * (sl % (LPC / 4)). Q6_K:
 * and Q3_K: weights g*32 + j0 .. +W-1 of each 128-half, one scale (W <= 16),
 * values offset by -32 (Q6_K) or -4 (Q3_K). Integer
 * dots accumulate across a lane's W weights; the scales apply once per row. */
template <int NC>
__device__ inline void kq_load_u32s(const void *p, uint32_t (&v)[NC]) {
    if constexpr (NC == 4) {
        const uint4 x = *reinterpret_cast<const uint4 *>(p);
        v[0] = x.x, v[1] = x.y, v[2] = x.z, v[3] = x.w;
    } else if constexpr (NC == 2) {
        const uint2 x = *reinterpret_cast<const uint2 *>(p);
        v[0] = x.x, v[1] = x.y;
    } else {
        v[0] = *reinterpret_cast<const uint32_t *>(p);
    }
}

template <int QT, int MR, int LPC>
__global__ void gemv_nt_kquant_i8_kernel(const int8_t *aq, const float *ad, const uint8_t *b,
                                         float *c, int m, int n, int k) {
    constexpr int BB = KqBlock<QT>::bytes;
    constexpr int W = 128 / LPC;
    constexpr int NC = W / 4;
    constexpr int LPG = LPC / 4;
    constexpr uint32_t ONES = 0x01010101u;
    const int sl = static_cast<int>(threadIdx.x) % LPC;
    const int col = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / LPC) +
                    static_cast<int>(threadIdx.x) / LPC;
    const int row0 = static_cast<int>(blockIdx.y) * MR;
    const bool live = col < n;
    const int nblocks = k / 256;
    const int kb = k / 32;
    const uint8_t *brow =
        b + static_cast<size_t>(live ? col : 0) * static_cast<size_t>(nblocks) * BB;
    const int g = sl / LPG;
    const int j0 = (sl % LPG) * W;
    const int8_t *arow[MR];
    const float *drow[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        const size_t ar = static_cast<size_t>(row0 + r < m ? row0 + r : row0);
        arow[r] = aq + ar * static_cast<size_t>(k);
        drow[r] = ad + ar * static_cast<size_t>(kb);
    }
    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        acc[r] = 0.f;
    }
    if (live) {
        for (int ib = 0; ib < nblocks; ++ib) {
            const uint8_t *blk = brow + static_cast<size_t>(ib) * BB;
            if constexpr (QT == 3) {
                // hmask[32] qs[64] scales[12] d: weight = d * (sc - 32) * (q2 | h << 2 - 4).
                const float d = allpaca_f16le_to_f32(blk + 108);
                const uint8_t *sc = blk + 96;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t q[NC];
#pragma unroll
                    for (int t = 0; t < NC; ++t) {
                        const uint32_t qs = kq_u32_from_u16(blk + 32 + h * 32 + j0 + 4 * t);
                        const uint32_t hm = kq_u32_from_u16(blk + j0 + 4 * t);
                        q[t] = ((qs >> (2 * g)) & 0x03030303u) | (((hm >> (4 * h + g)) & ONES) << 2);
                    }
                    const int is = h * 8 + 2 * g + j0 / 16;
                    const int s6 = ((sc[is & 7] >> (4 * (is >> 3))) & 0xF) |
                                   (((sc[8 + (is & 3)] >> (2 * (is >> 2))) & 3) << 4);
                    const float s = d * static_cast<float>(s6 - 32);
                    const int bi = ib * 8 + h * 4 + g;
#pragma unroll
                    for (int r = 0; r < MR; ++r) {
                        uint32_t av[NC];
                        kq_load_u32s<NC>(arow[r] + bi * 32 + j0, av);
                        int32_t dot = 0, sum = 0;
#pragma unroll
                        for (int t = 0; t < NC; ++t) {
                            dot = i8dot4(av[t], q[t], dot);
                            sum = i8dot4(av[t], ONES, sum);
                        }
                        acc[r] += s * drow[r][bi] * static_cast<float>(dot - 4 * sum);
                    }
                }
            } else if constexpr (QT == 6) {
                const float d = allpaca_f16le_to_f32(blk + 208);
                const int8_t *sc = reinterpret_cast<const int8_t *>(blk + 192);
                const int lo_off = (g & 1) * 32 + j0;
                const int nib = (g >> 1) * 4;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t q[NC];
#pragma unroll
                    for (int t = 0; t < NC; ++t) {
                        const uint32_t ql = kq_u32_from_u16(blk + h * 64 + lo_off + 4 * t);
                        const uint32_t qh = kq_u32_from_u16(blk + 128 + h * 32 + j0 + 4 * t);
                        q[t] = ((ql >> nib) & 0x0F0F0F0Fu) | (((qh >> (2 * g)) & 0x03030303u) << 4);
                    }
                    const float s = d * static_cast<float>(sc[h * 8 + j0 / 16 + 2 * g]);
                    const int bi = ib * 8 + h * 4 + g;
#pragma unroll
                    for (int r = 0; r < MR; ++r) {
                        uint32_t av[NC];
                        kq_load_u32s<NC>(arow[r] + bi * 32 + j0, av);
                        int32_t dot = 0, sum = 0;
#pragma unroll
                        for (int t = 0; t < NC; ++t) {
                            dot = i8dot4(av[t], q[t], dot);
                            sum = i8dot4(av[t], ONES, sum);
                        }
                        acc[r] += s * drow[r][bi] * static_cast<float>(dot - 32 * sum);
                    }
                }
            } else {
                const float d = allpaca_f16le_to_f32(blk);
                const float dmin = allpaca_f16le_to_f32(blk + 2);
                uint32_t sc1 = 0, m1 = 0, sc2 = 0, m2 = 0;
                kq_scale_min_pair(blk + 4, g, sc1, m1, sc2, m2);
                // Weight blocks are only 4-byte aligned.
                const uint32_t *qp = reinterpret_cast<const uint32_t *>(
                    blk + (QT == 5 ? 48 : 16) + g * 32 + j0);
                uint32_t qv[NC];
#pragma unroll
                for (int t = 0; t < NC; ++t) {
                    qv[t] = qp[t];
                }
                uint32_t lo[NC], hi[NC];
#pragma unroll
                for (int t = 0; t < NC; ++t) {
                    uint32_t hb_lo = 0, hb_hi = 0;
                    if constexpr (QT == 5) {
                        const uint32_t qh =
                            *reinterpret_cast<const uint32_t *>(blk + 16 + j0 + 4 * t);
                        hb_lo = (qh >> (2 * g)) & ONES;
                        hb_hi = (qh >> (2 * g + 1)) & ONES;
                    }
                    lo[t] = (qv[t] & 0x0F0F0F0Fu) | (hb_lo << 4);
                    hi[t] = ((qv[t] >> 4) & 0x0F0F0F0Fu) | (hb_hi << 4);
                }
                const float dl = d * static_cast<float>(sc1);
                const float ml = dmin * static_cast<float>(m1);
                const float dh = d * static_cast<float>(sc2);
                const float mh = dmin * static_cast<float>(m2);
                const int bl = ib * 8 + 2 * g;
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    uint32_t alo[NC], ahi[NC];
                    kq_load_u32s<NC>(arow[r] + bl * 32 + j0, alo);
                    kq_load_u32s<NC>(arow[r] + bl * 32 + 32 + j0, ahi);
                    int32_t il = 0, sl_ = 0, ih = 0, sh = 0;
#pragma unroll
                    for (int t = 0; t < NC; ++t) {
                        il = i8dot4(alo[t], lo[t], il);
                        sl_ = i8dot4(alo[t], ONES, sl_);
                        ih = i8dot4(ahi[t], hi[t], ih);
                        sh = i8dot4(ahi[t], ONES, sh);
                    }
                    acc[r] += drow[r][bl] * (dl * static_cast<float>(il) -
                                             ml * static_cast<float>(sl_)) +
                              drow[r][bl + 1] * (dh * static_cast<float>(ih) -
                                                 mh * static_cast<float>(sh));
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = LPC / 2; off > 0; off >>= 1) {
            v += __shfl_down(v, off, LPC);
        }
        if (live && sl == 0 && row0 + r < m) {
            c[static_cast<size_t>(row0 + r) * static_cast<size_t>(n) + static_cast<size_t>(col)] =
                v;
        }
    }
}

/* Per-device int8 activation scratch, grow-only. The lock is held from
 * quantize through GEMV enqueue, so on one stream (the legacy null stream is
 * shared by every host thread) no other GEMV's quantize can land between
 * them. A switch to another stream first waits, on the GPU, for all work
 * already queued on the previous one. */
constexpr int KQ_ACT_DEVICES = 16;
struct KqActScratch {
    std::mutex mu;
    void *p = nullptr;
    size_t cap = 0;
    hipStream_t stream = nullptr;
    hipEvent_t handoff = nullptr;
};
static KqActScratch g_kq_act[KQ_ACT_DEVICES];

/* Locks the device's scratch for `stream`; null when unavailable. */
static KqActScratch *kq_act_acquire(hipStream_t stream, size_t need,
                                    std::unique_lock<std::mutex> &lock) {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= KQ_ACT_DEVICES) {
        return nullptr;
    }
    KqActScratch &s = g_kq_act[dev];
    lock = std::unique_lock<std::mutex>(s.mu);
    if (s.p && s.stream != stream) {
        if (!s.handoff &&
            hipEventCreateWithFlags(&s.handoff, hipEventDisableTiming) != hipSuccess) {
            s.handoff = nullptr;
            return nullptr;
        }
        if (hipEventRecord(s.handoff, s.stream) != hipSuccess ||
            hipStreamWaitEvent(stream, s.handoff, 0) != hipSuccess) {
            return nullptr;
        }
    }
    if (s.cap < need) {
        if (s.p) {
            (void)hipStreamSynchronize(stream);
            (void)hipFree(s.p);
            s.p = nullptr;
            s.cap = 0;
        }
        const size_t cap = need > (size_t(1) << 20) ? need : (size_t(1) << 20);
        if (hipMalloc(&s.p, cap) != hipSuccess) {
            s.p = nullptr;
            return nullptr;
        }
        s.cap = cap;
    }
    s.stream = stream;
    return &s;
}

/* Returns ALLPACA_ERR_INVALID when the int8 path cannot run (caller falls back). */
template <int QT>
static int launch_kquant_i8(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                            int32_t k, hipStream_t stream) {
    const size_t aq_bytes = (static_cast<size_t>(m) * k + 255) & ~size_t(255);
    const size_t need = aq_bytes + static_cast<size_t>(m) * (k / 32) * sizeof(float);
    std::unique_lock<std::mutex> lock;
    KqActScratch *scratch = kq_act_acquire(stream, need, lock);
    if (!scratch) {
        return ALLPACA_ERR_INVALID;
    }
    uint8_t *base = static_cast<uint8_t *>(scratch->p);
    int8_t *aq = reinterpret_cast<int8_t *>(base);
    float *ad = reinterpret_cast<float *>(base + aq_bytes);
    const int kb = k / 32;
    hipLaunchKernelGGL(quantize_rows_q8_kernel,
                       dim3(static_cast<unsigned>((kb + 7) / 8), static_cast<unsigned>(m)),
                       dim3(256), 0, stream, a, aq, ad, m, k);
    // 8 lanes x 32 weights per column: at 32 lanes x 8 weights the per-row
    // A loads and scale math dominated (Q5_K m=4 cost 1.7x m=1; now ~1.05x).
    constexpr int LPC = 8;
    dim3 block(256);
    const int cols = 256 / LPC;
    const unsigned gx = static_cast<unsigned>((n + cols - 1) / cols);
    const int mr = m == 1 ? 1 : (m == 2 ? 2 : (m <= 4 ? 4 : 8));
    const dim3 grid(gx, static_cast<unsigned>((m + mr - 1) / mr));
#define KQ_I8_LAUNCH(MRV, LPCV)                                                                    \
    hipLaunchKernelGGL((gemv_nt_kquant_i8_kernel<QT, MRV, LPCV>), grid, block, 0, stream, aq, ad, \
                       b, c, m, n, k)
    if (mr == 1) {
        KQ_I8_LAUNCH(1, LPC);
    } else if (mr == 2) {
        KQ_I8_LAUNCH(2, LPC);
    } else if (mr == 4) {
        KQ_I8_LAUNCH(4, LPC);
    } else {
        KQ_I8_LAUNCH(8, LPC);
    }
#undef KQ_I8_LAUNCH
    return hip_status(hipGetLastError());
}

template <int QT, bool V2>
static int launch_kquant_rows(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                              int32_t k, hipStream_t stream) {
    dim3 block(KQ_LANES * KQ_COLS_PER_BLOCK);
    const unsigned gx = static_cast<unsigned>((n + KQ_COLS_PER_BLOCK - 1) / KQ_COLS_PER_BLOCK);
    const int mr = m == 1 ? 1 : (m == 2 ? 2 : (m <= 4 ? 4 : 8));
    const dim3 grid(gx, static_cast<unsigned>((m + mr - 1) / mr));
#define KQ_LAUNCH(MRV)                                                                              \
    if (V2) {                                                                                       \
        hipLaunchKernelGGL((gemv_nt_kquant_v2_kernel<QT, MRV>), grid, block, 0, stream, a, b, c,  \
                           m, n, k);                                                               \
    } else {                                                                                        \
        hipLaunchKernelGGL((gemv_nt_kquant_warp_kernel<QT, MRV>), grid, block, 0, stream, a, b,   \
                           c, m, n, k);                                                            \
    }
    if (mr == 1) {
        KQ_LAUNCH(1)
    } else if (mr == 2) {
        KQ_LAUNCH(2)
    } else if (mr == 4) {
        KQ_LAUNCH(4)
    } else {
        KQ_LAUNCH(8)
    }
#undef KQ_LAUNCH
    return hip_status(hipGetLastError());
}

/* Returns ALLPACA_ERR_INVALID when k is not whole super-blocks (caller falls back). */
template <int QT>
static int launch_gemm_nt_kquant_warp(const float *a, const uint8_t *b, float *c, int32_t m,
                                      int32_t n, int32_t k, hipStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || k % 256 != 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    // v2 reads B as u32 (Q4_K/Q5_K) or u16 (Q6_K) and A as float4/float2.
    const uintptr_t ba = reinterpret_cast<uintptr_t>(b);
    const bool v2 = (reinterpret_cast<uintptr_t>(a) % 16 == 0) &&
                    (QT == 6 ? ba % 2 == 0 : ba % 4 == 0);
    if (QT == 6 ? ba % 2 == 0 : ba % 4 == 0) {
        const int rc = launch_kquant_i8<QT>(a, b, c, m, n, k, stream);
        if (rc != ALLPACA_ERR_INVALID) {
            return rc;
        }
    }
    if (v2) {
        return launch_kquant_rows<QT, true>(a, b, c, m, n, k, stream);
    }
    return launch_kquant_rows<QT, false>(a, b, c, m, n, k, stream);
}

/* Q3_K has only the int8 kernel (its 110-byte blocks are 2-byte aligned). */
static int launch_gemm_nt_q3_k(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                               int32_t k, hipStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || k % 256 != 0 || !a || !b || !c ||
        reinterpret_cast<uintptr_t>(b) % 2 != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    return launch_kquant_i8<3>(a, b, c, m, n, k, stream);
}

extern "C" int allpaca_gemm_nt_q3_k(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                    int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_q3_k(a, b, c, m, n, k, as_stream(stream));
}

/* One thread per output column. Native 144-byte Q4_K super-blocks along K.
 * Baseline: each column re-reads A from global (no LDS tile). */
__global__ void gemv_nt_q4_k_untiled_kernel(const float *a, const uint8_t *b, float *c, int m,
                                            int n, int k) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    if (col >= n || row >= m) {
        return;
    }
    const int nblocks = (k + 255) / 256;
    const uint8_t *row_blocks = b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 144;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    float acc = 0.f;
    for (int ib = 0; ib < nblocks; ++ib) {
        const uint8_t *blk = row_blocks + static_cast<size_t>(ib) * 144;
        const float d = allpaca_f16le_to_f32(blk);
        const float dmin = allpaca_f16le_to_f32(blk + 2);
        const uint8_t *scales = blk + 4;
        const uint8_t *qs = blk + 16;
        const int k0 = ib * 256;
        int q_off = 0;
        int is = 0;
        for (int g = 0; g < 4; ++g) {
            uint8_t sc1 = 0;
            uint8_t m1 = 0;
            uint8_t sc2 = 0;
            uint8_t m2 = 0;
            q4k_scale_min(is, scales, &sc1, &m1);
            q4k_scale_min(is + 1, scales, &sc2, &m2);
            const float d1 = d * static_cast<float>(sc1);
            const float min1 = dmin * static_cast<float>(m1);
            const float d2 = d * static_cast<float>(sc2);
            const float min2 = dmin * static_cast<float>(m2);
            for (int l = 0; l < 32; ++l) {
                const int i0 = k0 + g * 64 + l;
                const int i1 = i0 + 32;
                const uint8_t packed = qs[q_off + l];
                if (i0 < k) {
                    acc += arow[i0] * (d1 * static_cast<float>(packed & 0x0F) - min1);
                }
                if (i1 < k) {
                    acc += arow[i1] * (d2 * static_cast<float>(packed >> 4) - min2);
                }
            }
            q_off += 32;
            is += 2;
        }
    }
    c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
}

/* Cache-aware decode tile (§12): one 256-float A super-block in LDS, reused by COLS columns. */
__global__ void gemv_nt_q4_k_tiled_kernel(const float *a, const uint8_t *b, float *c, int m, int n,
                                          int k) {
    constexpr int COLS = 64;
    __shared__ float As[256];
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(blockIdx.x) * COLS + tid;
    if (row >= m) {
        return;
    }
    const int nblocks = (k + 255) / 256;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    const uint8_t *brow =
        (tid < COLS && col < n)
            ? b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 144
            : nullptr;
    float acc = 0.f;
    for (int ib = 0; ib < nblocks; ++ib) {
        const int k0 = ib * 256;
        As[tid] = (k0 + tid < k) ? arow[k0 + tid] : 0.f;
        __syncthreads();
        if (brow) {
            const uint8_t *blk = brow + static_cast<size_t>(ib) * 144;
            const float d = allpaca_f16le_to_f32(blk);
            const float dmin = allpaca_f16le_to_f32(blk + 2);
            const uint8_t *scales = blk + 4;
            const uint8_t *qs = blk + 16;
            int q_off = 0;
            int is = 0;
            for (int g = 0; g < 4; ++g) {
                uint8_t sc1 = 0;
                uint8_t m1 = 0;
                uint8_t sc2 = 0;
                uint8_t m2 = 0;
                q4k_scale_min(is, scales, &sc1, &m1);
                q4k_scale_min(is + 1, scales, &sc2, &m2);
                const float d1 = d * static_cast<float>(sc1);
                const float min1 = dmin * static_cast<float>(m1);
                const float d2 = d * static_cast<float>(sc2);
                const float min2 = dmin * static_cast<float>(m2);
                for (int l = 0; l < 32; ++l) {
                    const int i0 = g * 64 + l;
                    const int i1 = i0 + 32;
                    const uint8_t packed = qs[q_off + l];
                    acc += As[i0] * (d1 * static_cast<float>(packed & 0x0F) - min1);
                    acc += As[i1] * (d2 * static_cast<float>(packed >> 4) - min2);
                }
                q_off += 32;
                is += 2;
            }
        }
        __syncthreads();
    }
    if (brow) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
    }
}

static int launch_gemm_nt_q4_k(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                               int32_t k, hipStream_t stream, int tiled) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    if (tiled && k % 256 == 0) {
        return launch_gemm_nt_kquant_warp<4>(a, b, c, m, n, k, stream);
    }
    if (tiled) {
        constexpr int COLS = 64;
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS), static_cast<unsigned>(m));
        hipLaunchKernelGGL(gemv_nt_q4_k_tiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    } else {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((n + 255) / 256), static_cast<unsigned>(m));
        hipLaunchKernelGGL(gemv_nt_q4_k_untiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    }
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_gemm_nt_q4_k(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                    int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_q4_k(a, b, c, m, n, k, as_stream(stream), 1);
}

/* Untiled baseline for §12 before/after tracers. */
extern "C" int allpaca_gemm_nt_q4_k_untiled(const float *a, const uint8_t *b, float *c, int32_t m,
                                            int32_t n, int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_q4_k(a, b, c, m, n, k, as_stream(stream), 0);
}

/* GGUF TQ2_0: 256 elems / 66 bytes (qs[64] + f16 d). value = d * (trit - 1). */
__global__ void gemv_nt_tq2_0_tiled_kernel(const float *a, const uint8_t *b, float *c, int m,
                                           int n, int k) {
    constexpr int COLS = 64;
    __shared__ float As[256];
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(blockIdx.x) * COLS + tid;
    if (row >= m) {
        return;
    }
    const int nblocks = (k + 255) / 256;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    const uint8_t *brow =
        (tid < COLS && col < n)
            ? b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 66
            : nullptr;
    float acc = 0.f;
    for (int ib = 0; ib < nblocks; ++ib) {
        const int k0 = ib * 256;
        As[tid] = (k0 + tid < k) ? arow[k0 + tid] : 0.f;
        __syncthreads();
        if (brow) {
            const uint8_t *blk = brow + static_cast<size_t>(ib) * 66;
            const float d = allpaca_f16le_to_f32(blk + 64);
            const uint8_t *qs = blk;
            for (int group = 0; group < 2; ++group) {
                for (int shift = 0; shift < 4; ++shift) {
                    for (int m_i = 0; m_i < 32; ++m_i) {
                        const int i = group * 128 + shift * 32 + m_i;
                        const uint8_t trit =
                            static_cast<uint8_t>((qs[group * 32 + m_i] >> (shift * 2)) & 3);
                        acc += As[i] * (d * (static_cast<float>(trit) - 1.f));
                    }
                }
            }
        }
        __syncthreads();
    }
    if (brow) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
    }
}

extern "C" int allpaca_gemm_nt_tq2_0(const float *a, const uint8_t *b, float *c, int32_t m,
                                     int32_t n, int32_t k, allpaca_stream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    constexpr int COLS = 64;
    dim3 block(256);
    dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS), static_cast<unsigned>(m));
    hipLaunchKernelGGL(gemv_nt_tq2_0_tiled_kernel, grid, block, 0, as_stream(stream), a, b, c, m,
                       n, k);
    return hip_status(hipGetLastError());
}

/* llama.cpp fixed-point base-3 trit decode for TQ1_0. */
__device__ inline float tq1_0_decode_trit(uint8_t byte, int digit) {
    static const uint8_t kPow3[5] = {1, 3, 9, 27, 81};
    const uint8_t q = static_cast<uint8_t>(byte * kPow3[digit]);
    return static_cast<float>(static_cast<int16_t>((static_cast<uint16_t>(q) * 3u) >> 8) - 1);
}

/* GGUF TQ1_0: 256 elems / 54 bytes (qs[48] + qh[4] + f16 d at 52). */
__global__ void gemv_nt_tq1_0_tiled_kernel(const float *a, const uint8_t *b, float *c, int m,
                                           int n, int k) {
    constexpr int COLS = 64;
    __shared__ float As[256];
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(blockIdx.x) * COLS + tid;
    if (row >= m) {
        return;
    }
    const int nblocks = (k + 255) / 256;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    const uint8_t *brow =
        (tid < COLS && col < n)
            ? b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 54
            : nullptr;
    float acc = 0.f;
    for (int ib = 0; ib < nblocks; ++ib) {
        const int k0 = ib * 256;
        As[tid] = (k0 + tid < k) ? arow[k0 + tid] : 0.f;
        __syncthreads();
        if (brow) {
            const uint8_t *blk = brow + static_cast<size_t>(ib) * 54;
            const float d = allpaca_f16le_to_f32(blk + 52);
            int dst = 0;
            for (int digit = 0; digit < 5; ++digit) {
                for (int m_i = 0; m_i < 32; ++m_i) {
                    acc += As[dst++] * (d * tq1_0_decode_trit(blk[m_i], digit));
                }
            }
            for (int digit = 0; digit < 5; ++digit) {
                for (int m_i = 0; m_i < 16; ++m_i) {
                    acc += As[dst++] * (d * tq1_0_decode_trit(blk[32 + m_i], digit));
                }
            }
            for (int digit = 0; digit < 4; ++digit) {
                for (int m_i = 0; m_i < 4; ++m_i) {
                    acc += As[dst++] * (d * tq1_0_decode_trit(blk[48 + m_i], digit));
                }
            }
        }
        __syncthreads();
    }
    if (brow) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
    }
}

extern "C" int allpaca_gemm_nt_tq1_0(const float *a, const uint8_t *b, float *c, int32_t m,
                                     int32_t n, int32_t k, allpaca_stream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    constexpr int COLS = 64;
    dim3 block(256);
    dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS), static_cast<unsigned>(m));
    hipLaunchKernelGGL(gemv_nt_tq1_0_tiled_kernel, grid, block, 0, as_stream(stream), a, b, c, m,
                       n, k);
    return hip_status(hipGetLastError());
}

/* ROCmFP4 (ggml_type_id=100) signed codebook at half-scale: E2M1-derived with
 * the top magnitude retuned 12 -> 10. Index is the raw 4-bit code. */
__constant__ const int8_t k_rocmfp4_codebook[16] = {
    0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10,
};

/* Unsigned finite E4M3 scale byte -> half-scale multiplier. 0x7f / 0xff are
 * NaN in this encoding and decode to 0, matching the CPU reference. */
__device__ inline float rocmfp4_scale(uint8_t e) {
    if (e == 0 || e == 0x7f || e == 0xff) {
        return 0.f;
    }
    const uint32_t exp = static_cast<uint32_t>((e >> 3) & 0x0F);
    const uint32_t man = static_cast<uint32_t>(e & 0x07);
    if (exp == 0) {
        return static_cast<float>(man) * (1.f / 1024.f);
    }
    const uint32_t bits = ((exp + 119u) << 23) | (man << 20);
    float out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

/* One thread per output column. Native 18-byte ROCmFP4 blocks along K.
 * Baseline: each column re-reads A from global (no LDS tile). */
__global__ void gemv_nt_rocmfp4_untiled_kernel(const float *a, const uint8_t *b, float *c, int m,
                                               int n, int k) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    if (col >= n || row >= m) {
        return;
    }
    const int nblocks = (k + 31) / 32;
    const uint8_t *row_blocks = b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 18;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    float acc = 0.f;
    for (int ib = 0; ib < nblocks; ++ib) {
        const uint8_t *blk = row_blocks + static_cast<size_t>(ib) * 18;
        const float d0 = rocmfp4_scale(blk[16]);
        const float d1 = rocmfp4_scale(blk[17]);
        const int k0 = ib * 32;
        for (int j = 0; j < 16; ++j) {
            const uint8_t packed = blk[j];
            const int i0 = k0 + j;
            const int i1 = i0 + 16;
            if (i0 < k) {
                acc += arow[i0] * static_cast<float>(k_rocmfp4_codebook[packed & 0x0F]) * d0;
            }
            if (i1 < k) {
                acc += arow[i1] * static_cast<float>(k_rocmfp4_codebook[packed >> 4]) * d1;
            }
        }
    }
    c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
}

// Decode GEMV: 32 lanes split K for each column. One thread walking all of K
// made the local ROCmFP4 projections the largest remaining per-token cost.
__global__ void gemv_nt_rocmfp4_ksplit_kernel(const float *a, const uint8_t *b, float *c, int m,
                                              int n, int k) {
    constexpr int LANES = 32;
    constexpr int GROUPS = 8;
    const int lane = static_cast<int>(threadIdx.x) % LANES;
    const int group = static_cast<int>(threadIdx.x) / LANES;
    const int col = static_cast<int>(blockIdx.x) * GROUPS + group;
    const int row = static_cast<int>(blockIdx.y);
    if (row >= m) {
        return;
    }
    const int nblocks = (k + 31) / 32;
    float acc = 0.f;
    if (col < n) {
        const uint8_t *row_blocks =
            b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 18;
        const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
        for (int ib = lane; ib < nblocks; ib += LANES) {
            const uint8_t *blk = row_blocks + static_cast<size_t>(ib) * 18;
            const float d0 = rocmfp4_scale(blk[16]);
            const float d1 = rocmfp4_scale(blk[17]);
            const int k0 = ib * 32;
            for (int j = 0; j < 16; ++j) {
                const uint8_t packed = blk[j];
                const int i0 = k0 + j;
                const int i1 = i0 + 16;
                if (i0 < k) {
                    acc += arow[i0] * static_cast<float>(k_rocmfp4_codebook[packed & 0x0F]) * d0;
                }
                if (i1 < k) {
                    acc += arow[i1] * static_cast<float>(k_rocmfp4_codebook[packed >> 4]) * d1;
                }
            }
        }
    }
    __shared__ float smem[GROUPS * LANES];
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = LANES >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            smem[group * LANES + lane] += smem[group * LANES + lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0 && col < n) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] =
            smem[group * LANES];
    }
}

/* Cache-aware decode tile (§12): a 256-float A slab in LDS, reused by COLS
 * columns. Each slab covers 8 ROCmFP4 blocks so the LDS shape matches the
 * Q4_K/Q6_K kernels and one __syncthreads pair amortizes over 8 blocks. */
__global__ void gemv_nt_rocmfp4_tiled_kernel(const float *a, const uint8_t *b, float *c, int m,
                                             int n, int k) {
    constexpr int COLS = 64;
    constexpr int TILE = 256;
    constexpr int BLOCKS_PER_TILE = TILE / 32;
    __shared__ float As[TILE];
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(blockIdx.x) * COLS + tid;
    if (row >= m) {
        return;
    }
    const int nblocks = (k + 31) / 32;
    const int ntiles = (k + TILE - 1) / TILE;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    const uint8_t *brow =
        (tid < COLS && col < n)
            ? b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 18
            : nullptr;
    float acc = 0.f;
    for (int it = 0; it < ntiles; ++it) {
        const int k0 = it * TILE;
        As[tid] = (k0 + tid < k) ? arow[k0 + tid] : 0.f;
        __syncthreads();
        if (brow) {
            for (int t = 0; t < BLOCKS_PER_TILE; ++t) {
                const int ib = it * BLOCKS_PER_TILE + t;
                if (ib >= nblocks) {
                    break;
                }
                const uint8_t *blk = brow + static_cast<size_t>(ib) * 18;
                const float d0 = rocmfp4_scale(blk[16]);
                const float d1 = rocmfp4_scale(blk[17]);
                const int base = t * 32;
                for (int j = 0; j < 16; ++j) {
                    const uint8_t packed = blk[j];
                    acc += As[base + j] * static_cast<float>(k_rocmfp4_codebook[packed & 0x0F]) * d0;
                    acc += As[base + 16 + j] * static_cast<float>(k_rocmfp4_codebook[packed >> 4]) *
                           d1;
                }
            }
        }
        __syncthreads();
    }
    if (brow) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
    }
}

static int launch_gemm_nt_rocmfp4(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                  int32_t k, hipStream_t stream, int tiled) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    if (tiled) {
        /* ksplit is the proven decode kernel (row = blockIdx.y). The COLS=64
         * tiled path was built for large-M prefill and is not trusted for
         * Qwen gated/IMRoPE short prompts (M~11 produced garbage tokens). */
        constexpr int GROUPS = 8;
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((n + GROUPS - 1) / GROUPS), static_cast<unsigned>(m));
        hipLaunchKernelGGL(gemv_nt_rocmfp4_ksplit_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    } else {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((n + 255) / 256), static_cast<unsigned>(m));
        hipLaunchKernelGGL(gemv_nt_rocmfp4_untiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    }
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_gemm_nt_rocmfp4(const float *a, const uint8_t *b, float *c, int32_t m,
                                       int32_t n, int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_rocmfp4(a, b, c, m, n, k, as_stream(stream), 1);
}

/* Untiled baseline for §12 before/after tracers. */
extern "C" int allpaca_gemm_nt_rocmfp4_untiled(const float *a, const uint8_t *b, float *c, int32_t m,
                                               int32_t n, int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_rocmfp4(a, b, c, m, n, k, as_stream(stream), 0);
}

/* ggml MXFP4 (ggml_type_id=39): 17-byte blocks, one E8M0 scale + 16 nibbles / 32 elems. */
__constant__ const int8_t k_mxfp4_codebook[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

__device__ inline float mxfp4_e8m0_half(uint8_t e) {
    uint32_t bits;
    if (e == 0) {
        bits = 0x00400000u;
    } else {
        bits = static_cast<uint32_t>(e) << 23;
    }
    float full;
    __builtin_memcpy(&full, &bits, sizeof(full));
    return full * 0.5f;
}

__global__ void gemv_nt_mxfp4_untiled_kernel(const float *a, const uint8_t *b, float *c, int m,
                                             int n, int k) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    if (col >= n || row >= m) {
        return;
    }
    const int nblocks = (k + 31) / 32;
    const uint8_t *row_blocks = b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 17;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    float acc = 0.f;
    for (int ib = 0; ib < nblocks; ++ib) {
        const uint8_t *blk = row_blocks + static_cast<size_t>(ib) * 17;
        const float d = mxfp4_e8m0_half(blk[0]);
        const int k0 = ib * 32;
        for (int j = 0; j < 16; ++j) {
            const uint8_t packed = blk[1 + j];
            const int i0 = k0 + j;
            const int i1 = i0 + 16;
            if (i0 < k) {
                acc += arow[i0] * static_cast<float>(k_mxfp4_codebook[packed & 0x0F]) * d;
            }
            if (i1 < k) {
                acc += arow[i1] * static_cast<float>(k_mxfp4_codebook[packed >> 4]) * d;
            }
        }
    }
    c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
}

/* ChatML keeps COLS=64 so the grid stays n/64 blocks per row. Widening COLS
 * to 256 underfilled gfx906. M=1 uses this GEMV; M>1 uses the B-reuse GEMM.
 * Same K-split as BF16 GEMV: 2 threads/column, whole MXFP4 blocks per thread. */
__global__ void gemv_nt_mxfp4_tiled_kernel(const float *a, const uint8_t *b, float *c, int m, int n,
                                           int k) {
    constexpr int COLS = 64;
    constexpr int KSPLIT = 2;
    constexpr int TILE = 256;
    constexpr int BLOCKS_PER_TILE = TILE / 32;
    constexpr int BLOCKS_PER_THREAD = BLOCKS_PER_TILE / KSPLIT;
    __shared__ float As[TILE];
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int col_local = tid / KSPLIT;
    const int kt = tid % KSPLIT;
    const int col = static_cast<int>(blockIdx.x) * COLS + col_local;
    if (row >= m) {
        return;
    }
    const int nblocks = (k + 31) / 32;
    const int ntiles = (k + TILE - 1) / TILE;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    const uint8_t *brow =
        (col < n) ? b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 17 : nullptr;
    float acc = 0.f;
    for (int it = 0; it < ntiles; ++it) {
        const int k0 = it * TILE;
        if (tid < TILE / 4) {
            const int p = tid * 4;
            const int gp = k0 + p;
            const int nvalid = (gp < k) ? (k - gp) : 0;
            const size_t off = static_cast<size_t>(gp);
            if (nvalid >= 4 && (off & 3u) == 0) {
                const float4 v = *reinterpret_cast<const float4 *>(arow + off);
                As[p] = v.x;
                As[p + 1] = v.y;
                As[p + 2] = v.z;
                As[p + 3] = v.w;
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    As[p + i] = (gp + i < k) ? arow[off + static_cast<size_t>(i)] : 0.f;
                }
            }
        }
        __syncthreads();
        if (brow) {
#pragma unroll
            for (int t = 0; t < BLOCKS_PER_THREAD; ++t) {
                const int ib = it * BLOCKS_PER_TILE + kt * BLOCKS_PER_THREAD + t;
                if (ib >= nblocks) {
                    break;
                }
                const uint8_t *blk = brow + static_cast<size_t>(ib) * 17;
                const float d = mxfp4_e8m0_half(blk[0]);
                const int base = (kt * BLOCKS_PER_THREAD + t) * 32;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const uint8_t packed = blk[1 + j];
                    acc += As[base + j] * static_cast<float>(k_mxfp4_codebook[packed & 0x0F]) * d;
                    acc += As[base + 16 + j] * static_cast<float>(k_mxfp4_codebook[packed >> 4]) * d;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int off = KSPLIT / 2; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, KSPLIT);
    }
    if (brow && kt == 0) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
    }
}

/* COLS=64 / TILE_K=256: one float4 A load per thread per row. */
template <int TILE_M>
__device__ inline void mxfp4_load_a_tile(float *As, const float *a, int row0, int m, int k, int k0,
                                         int tid) {
    constexpr int TILE_K = 256;
#pragma unroll 8
    for (int r = 0; r < TILE_M; ++r) {
        const int row = row0 + r;
        const int p = tid * 4;
        const int gp = k0 + p;
        float *dst = As + r * TILE_K + p;
        const int nvalid = (row < m && gp < k) ? (k - gp) : 0;
        const size_t off =
            static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(gp);
        if (nvalid >= 4 && (off & 3u) == 0) {
            const float4 v = *reinterpret_cast<const float4 *>(a + off);
            dst[0] = v.x;
            dst[1] = v.y;
            dst[2] = v.z;
            dst[3] = v.w;
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                dst[i] = (row < m && gp + i < k) ? a[off + static_cast<size_t>(i)] : 0.f;
            }
        }
    }
}

/* Reuse each MXFP4 B column across TILE_M A rows. ChatML ~20–30 uses TILE_M=16
 * (TILE_M=32 was 32KiB LDS / acc[32] and lost on gfx906). COLS stays 64. */
template <int TILE_M>
__global__ void gemm_nt_mxfp4_tiled_m_kernel(const float *a, const uint8_t *b, float *c, int m,
                                             int n, int k) {
    constexpr int COLS = 64;
    constexpr int TILE_K = 256;
    constexpr int BLOCKS_PER_TILE = TILE_K / 32;
    static_assert(TILE_M >= 1 && (TILE_M * TILE_K) % (4 * COLS) == 0, "mxfp4 M-tile mapping");
    __shared__ float As[TILE_M * TILE_K];
    const int row0 = static_cast<int>(blockIdx.y) * TILE_M;
    const int tid = static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(blockIdx.x) * COLS + tid;
    if (row0 >= m) {
        return;
    }
    const int nblocks = (k + 31) / 32;
    const int ntiles = (k + TILE_K - 1) / TILE_K;
    const uint8_t *brow =
        (col < n) ? b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 17 : nullptr;
    float acc[TILE_M];
#pragma unroll 8
    for (int r = 0; r < TILE_M; ++r) {
        acc[r] = 0.f;
    }
    for (int it = 0; it < ntiles; ++it) {
        const int k0 = it * TILE_K;
        mxfp4_load_a_tile<TILE_M>(As, a, row0, m, k, k0, tid);
        __syncthreads();
        if (brow) {
            for (int t = 0; t < BLOCKS_PER_TILE; ++t) {
                const int ib = it * BLOCKS_PER_TILE + t;
                if (ib >= nblocks) {
                    break;
                }
                const uint8_t *blk = brow + static_cast<size_t>(ib) * 17;
                const float d = mxfp4_e8m0_half(blk[0]);
                const int base = t * 32;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const uint8_t packed = blk[1 + j];
                    const float c0 = static_cast<float>(k_mxfp4_codebook[packed & 0x0F]) * d;
                    const float c1 = static_cast<float>(k_mxfp4_codebook[packed >> 4]) * d;
#pragma unroll 8
                    for (int r = 0; r < TILE_M; ++r) {
                        const float *ar = As + r * TILE_K + base;
                        acc[r] += ar[j] * c0;
                        acc[r] += ar[16 + j] * c1;
                    }
                }
            }
        }
        __syncthreads();
    }
    if (brow) {
#pragma unroll 8
        for (int r = 0; r < TILE_M; ++r) {
            const int row = row0 + r;
            if (row < m) {
                c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] =
                    acc[r];
            }
        }
    }
}

static int launch_gemm_nt_mxfp4(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                int32_t k, hipStream_t stream, int tiled) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    if (tiled) {
        constexpr int COLS = 64;
        if (m > 1) {
            dim3 block(64);
            if (m > 8) {
                constexpr int TILE_M = 16;
                dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS),
                          static_cast<unsigned>((m + TILE_M - 1) / TILE_M));
                hipLaunchKernelGGL((gemm_nt_mxfp4_tiled_m_kernel<16>), grid, block, 0, stream, a, b,
                                   c, m, n, k);
            } else {
                constexpr int TILE_M = 8;
                dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS),
                          static_cast<unsigned>((m + TILE_M - 1) / TILE_M));
                hipLaunchKernelGGL((gemm_nt_mxfp4_tiled_m_kernel<8>), grid, block, 0, stream, a, b,
                                   c, m, n, k);
            }
        } else {
            dim3 block(128);
            dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS), static_cast<unsigned>(m));
            hipLaunchKernelGGL(gemv_nt_mxfp4_tiled_kernel, grid, block, 0, stream, a, b, c, m, n,
                               k);
        }
    } else {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((n + 255) / 256), static_cast<unsigned>(m));
        hipLaunchKernelGGL(gemv_nt_mxfp4_untiled_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    }
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_gemm_nt_mxfp4(const float *a, const uint8_t *b, float *c, int32_t m,
                                     int32_t n, int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_mxfp4(a, b, c, m, n, k, as_stream(stream), 1);
}

extern "C" int allpaca_gemm_nt_mxfp4_untiled(const float *a, const uint8_t *b, float *c, int32_t m,
                                             int32_t n, int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_mxfp4(a, b, c, m, n, k, as_stream(stream), 0);
}

__device__ inline float mxfp4_silu(float x) { return x / (1.0f + expf(-x)); }

/* One A-tile load, two MXFP4 B rows, write silu(gate)*up. */
__global__ void mxfp4_swiglu_gate_up_tiled_kernel(const float *a, const uint8_t *bg,
                                                  const uint8_t *bu, float *c, int m, int n,
                                                  int k) {
    constexpr int COLS = 64;
    constexpr int TILE = 256;
    constexpr int BLOCKS_PER_TILE = TILE / 32;
    __shared__ float As[TILE];
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(blockIdx.x) * COLS + tid;
    if (row >= m) {
        return;
    }
    const int nblocks = (k + 31) / 32;
    const int ntiles = (k + TILE - 1) / TILE;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    const bool live = tid < COLS && col < n;
    const uint8_t *grow =
        live ? bg + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 17 : nullptr;
    const uint8_t *urow =
        live ? bu + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 17 : nullptr;
    float acc_g = 0.f;
    float acc_u = 0.f;
    for (int it = 0; it < ntiles; ++it) {
        const int k0 = it * TILE;
        As[tid] = (k0 + tid < k) ? arow[k0 + tid] : 0.f;
        __syncthreads();
        if (grow) {
#pragma unroll
            for (int t = 0; t < BLOCKS_PER_TILE; ++t) {
                const int ib = it * BLOCKS_PER_TILE + t;
                if (ib >= nblocks) {
                    break;
                }
                const uint8_t *blk_g = grow + static_cast<size_t>(ib) * 17;
                const uint8_t *blk_u = urow + static_cast<size_t>(ib) * 17;
                const float dg = mxfp4_e8m0_half(blk_g[0]);
                const float du = mxfp4_e8m0_half(blk_u[0]);
                const int base = t * 32;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const uint8_t pg = blk_g[1 + j];
                    const uint8_t pu = blk_u[1 + j];
                    const float a0 = As[base + j];
                    const float a1 = As[base + 16 + j];
                    acc_g += a0 * static_cast<float>(k_mxfp4_codebook[pg & 0x0F]) * dg;
                    acc_g += a1 * static_cast<float>(k_mxfp4_codebook[pg >> 4]) * dg;
                    acc_u += a0 * static_cast<float>(k_mxfp4_codebook[pu & 0x0F]) * du;
                    acc_u += a1 * static_cast<float>(k_mxfp4_codebook[pu >> 4]) * du;
                }
            }
        }
        __syncthreads();
    }
    if (grow) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] =
            mxfp4_silu(acc_g) * acc_u;
    }
}

/* Same B-reuse as tiled-M GEMM: dequant gate/up once per column across TILE_M rows. */
template <int TILE_M>
__global__ void mxfp4_swiglu_gate_up_tiled_m_kernel(const float *a, const uint8_t *bg,
                                                    const uint8_t *bu, float *c, int m, int n,
                                                    int k) {
    constexpr int COLS = 64;
    constexpr int TILE_K = 256;
    constexpr int BLOCKS_PER_TILE = TILE_K / 32;
    static_assert(TILE_M >= 1 && (TILE_M * TILE_K) % (4 * COLS) == 0, "mxfp4 fused M-tile mapping");
    __shared__ float As[TILE_M * TILE_K];
    const int row0 = static_cast<int>(blockIdx.y) * TILE_M;
    const int tid = static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(blockIdx.x) * COLS + tid;
    if (row0 >= m) {
        return;
    }
    const int nblocks = (k + 31) / 32;
    const int ntiles = (k + TILE_K - 1) / TILE_K;
    const bool live = col < n;
    const uint8_t *grow =
        live ? bg + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 17 : nullptr;
    const uint8_t *urow =
        live ? bu + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 17 : nullptr;
    float acc_g[TILE_M];
    float acc_u[TILE_M];
#pragma unroll 8
    for (int r = 0; r < TILE_M; ++r) {
        acc_g[r] = 0.f;
        acc_u[r] = 0.f;
    }
    for (int it = 0; it < ntiles; ++it) {
        const int k0 = it * TILE_K;
        mxfp4_load_a_tile<TILE_M>(As, a, row0, m, k, k0, tid);
        __syncthreads();
        if (grow) {
            for (int t = 0; t < BLOCKS_PER_TILE; ++t) {
                const int ib = it * BLOCKS_PER_TILE + t;
                if (ib >= nblocks) {
                    break;
                }
                const uint8_t *blk_g = grow + static_cast<size_t>(ib) * 17;
                const uint8_t *blk_u = urow + static_cast<size_t>(ib) * 17;
                const float dg = mxfp4_e8m0_half(blk_g[0]);
                const float du = mxfp4_e8m0_half(blk_u[0]);
                const int base = t * 32;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const uint8_t pg = blk_g[1 + j];
                    const uint8_t pu = blk_u[1 + j];
                    const float g0 = static_cast<float>(k_mxfp4_codebook[pg & 0x0F]) * dg;
                    const float g1 = static_cast<float>(k_mxfp4_codebook[pg >> 4]) * dg;
                    const float u0 = static_cast<float>(k_mxfp4_codebook[pu & 0x0F]) * du;
                    const float u1 = static_cast<float>(k_mxfp4_codebook[pu >> 4]) * du;
#pragma unroll 8
                    for (int r = 0; r < TILE_M; ++r) {
                        const float *ar = As + r * TILE_K + base;
                        acc_g[r] += ar[j] * g0;
                        acc_g[r] += ar[16 + j] * g1;
                        acc_u[r] += ar[j] * u0;
                        acc_u[r] += ar[16 + j] * u1;
                    }
                }
            }
        }
        __syncthreads();
    }
    if (grow) {
#pragma unroll 8
        for (int r = 0; r < TILE_M; ++r) {
            const int row = row0 + r;
            if (row < m) {
                c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] =
                    mxfp4_silu(acc_g[r]) * acc_u[r];
            }
        }
    }
}

extern "C" int allpaca_mxfp4_swiglu_gate_up(const float *a, const uint8_t *bg, const uint8_t *bu,
                                            float *c, int32_t m, int32_t n, int32_t k,
                                            allpaca_stream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !bg || !bu || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    constexpr int COLS = 64;
    if (m > 1) {
        dim3 block(64);
        if (m > 8) {
            constexpr int TILE_M = 16;
            dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS),
                      static_cast<unsigned>((m + TILE_M - 1) / TILE_M));
            hipLaunchKernelGGL((mxfp4_swiglu_gate_up_tiled_m_kernel<16>), grid, block, 0,
                               as_stream(stream), a, bg, bu, c, m, n, k);
        } else {
            constexpr int TILE_M = 8;
            dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS),
                      static_cast<unsigned>((m + TILE_M - 1) / TILE_M));
            hipLaunchKernelGGL((mxfp4_swiglu_gate_up_tiled_m_kernel<8>), grid, block, 0,
                               as_stream(stream), a, bg, bu, c, m, n, k);
        }
    } else {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS), static_cast<unsigned>(m));
        hipLaunchKernelGGL(mxfp4_swiglu_gate_up_tiled_kernel, grid, block, 0, as_stream(stream), a,
                           bg, bu, c, m, n, k);
    }
    return hip_status(hipGetLastError());
}

/* Fused Q6_K GEMV (GEM mixed-precision): 210-byte super-blocks, A tile in LDS.
 * COLS is the §12 cache-aware column tile (32/64/128). */
template <int COLS>
__global__ void gemv_nt_q6_k_tiled_kernel(const float *a, const uint8_t *b, float *c, int m, int n,
                                          int k) {
    constexpr int TILE = 256;
    __shared__ float As[TILE];
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(blockIdx.x) * COLS + tid;
    if (row >= m) {
        return;
    }
    const int nblocks = (k + 255) / 256;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    const uint8_t *brow =
        (tid < COLS && col < n)
            ? b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 210
            : nullptr;
    float acc = 0.f;
    for (int ib = 0; ib < nblocks; ++ib) {
        const int k0 = ib * 256;
        As[tid] = (k0 + tid < k) ? arow[k0 + tid] : 0.f;
        __syncthreads();
        if (brow) {
            const uint8_t *blk = brow + static_cast<size_t>(ib) * 210;
            const uint8_t *ql = blk;
            const uint8_t *qh = blk + 128;
            const uint8_t *scales = blk + 192;
            const float d = allpaca_f16le_to_f32(blk + 208);
            for (int half = 0; half < 2; ++half) {
                const uint8_t *ql_h = ql + half * 64;
                const uint8_t *qh_h = qh + half * 32;
                const uint8_t *sc_h = scales + half * 8;
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int q1 =
                        static_cast<int>(ql_h[l] & 0x0f | ((qh_h[l] >> 0) & 3) << 4) - 32;
                    const int q2 =
                        static_cast<int>(ql_h[l + 32] & 0x0f | ((qh_h[l] >> 2) & 3) << 4) - 32;
                    const int q3 =
                        static_cast<int>((ql_h[l] >> 4) | ((qh_h[l] >> 4) & 3) << 4) - 32;
                    const int q4 =
                        static_cast<int>((ql_h[l + 32] >> 4) | ((qh_h[l] >> 6) & 3) << 4) - 32;
                    const int i0 = half * 128 + l;
                    const int i1 = half * 128 + 32 + l;
                    const int i2 = half * 128 + 64 + l;
                    const int i3 = half * 128 + 96 + l;
                    acc += As[i0] * (d * static_cast<float>(static_cast<int8_t>(sc_h[is])) *
                                     static_cast<float>(q1));
                    acc += As[i1] * (d * static_cast<float>(static_cast<int8_t>(sc_h[is + 2])) *
                                     static_cast<float>(q2));
                    acc += As[i2] * (d * static_cast<float>(static_cast<int8_t>(sc_h[is + 4])) *
                                     static_cast<float>(q3));
                    acc += As[i3] * (d * static_cast<float>(static_cast<int8_t>(sc_h[is + 6])) *
                                     static_cast<float>(q4));
                }
            }
        }
        __syncthreads();
    }
    if (brow) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
    }
}

__global__ void gemv_nt_q6_k_untiled_kernel(const float *a, const uint8_t *b, float *c, int m,
                                            int n, int k) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    if (col >= n || row >= m) {
        return;
    }
    const int nblocks = (k + 255) / 256;
    const uint8_t *brow = b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 210;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    float acc = 0.f;
    for (int ib = 0; ib < nblocks; ++ib) {
        const uint8_t *blk = brow + static_cast<size_t>(ib) * 210;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;
        const uint8_t *scales = blk + 192;
        const float d = allpaca_f16le_to_f32(blk + 208);
        const int k0 = ib * 256;
        for (int half = 0; half < 2; ++half) {
            const uint8_t *ql_h = ql + half * 64;
            const uint8_t *qh_h = qh + half * 32;
            const uint8_t *sc_h = scales + half * 8;
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 =
                    static_cast<int>(ql_h[l] & 0x0f | ((qh_h[l] >> 0) & 3) << 4) - 32;
                const int q2 =
                    static_cast<int>(ql_h[l + 32] & 0x0f | ((qh_h[l] >> 2) & 3) << 4) - 32;
                const int q3 =
                    static_cast<int>((ql_h[l] >> 4) | ((qh_h[l] >> 4) & 3) << 4) - 32;
                const int q4 =
                    static_cast<int>((ql_h[l + 32] >> 4) | ((qh_h[l] >> 6) & 3) << 4) - 32;
                const int i0 = k0 + half * 128 + l;
                const int i1 = k0 + half * 128 + 32 + l;
                const int i2 = k0 + half * 128 + 64 + l;
                const int i3 = k0 + half * 128 + 96 + l;
                if (i0 < k) {
                    acc += arow[i0] * (d * static_cast<float>(static_cast<int8_t>(sc_h[is])) *
                                       static_cast<float>(q1));
                }
                if (i1 < k) {
                    acc += arow[i1] * (d * static_cast<float>(static_cast<int8_t>(sc_h[is + 2])) *
                                       static_cast<float>(q2));
                }
                if (i2 < k) {
                    acc += arow[i2] * (d * static_cast<float>(static_cast<int8_t>(sc_h[is + 4])) *
                                       static_cast<float>(q3));
                }
                if (i3 < k) {
                    acc += arow[i3] * (d * static_cast<float>(static_cast<int8_t>(sc_h[is + 6])) *
                                       static_cast<float>(q4));
                }
            }
        }
    }
    c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
}

static int launch_gemm_nt_q6_k_cols(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                    int32_t k, int32_t cols, hipStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    if (k % 256 == 0) {
        // `cols` tiled the per-thread kernel; the warp kernel has no column tile.
        return launch_gemm_nt_kquant_warp<6>(a, b, c, m, n, k, stream);
    }
    clear_last_error();
    int use = cols;
    if (use != 32 && use != 64 && use != 128) {
        use = 64;
    }
    dim3 block(256);
    dim3 grid(static_cast<unsigned>((n + use - 1) / use), static_cast<unsigned>(m));
    if (use == 32) {
        hipLaunchKernelGGL(gemv_nt_q6_k_tiled_kernel<32>, grid, block, 0, stream, a, b, c, m, n, k);
    } else if (use == 128) {
        hipLaunchKernelGGL(gemv_nt_q6_k_tiled_kernel<128>, grid, block, 0, stream, a, b, c, m, n, k);
    } else {
        hipLaunchKernelGGL(gemv_nt_q6_k_tiled_kernel<64>, grid, block, 0, stream, a, b, c, m, n, k);
    }
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_gemm_nt_q6_k(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                    int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_q6_k_cols(a, b, c, m, n, k, 64, as_stream(stream));
}

extern "C" int allpaca_gemm_nt_q6_k_cols(const float *a, const uint8_t *b, float *c, int32_t m,
                                         int32_t n, int32_t k, int32_t cols,
                                         allpaca_stream_t stream) {
    return launch_gemm_nt_q6_k_cols(a, b, c, m, n, k, cols, as_stream(stream));
}

extern "C" int allpaca_gemm_nt_q6_k_untiled(const float *a, const uint8_t *b, float *c, int32_t m,
                                            int32_t n, int32_t k, allpaca_stream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    dim3 block(256);
    dim3 grid(static_cast<unsigned>((n + 255) / 256), static_cast<unsigned>(m));
    hipLaunchKernelGGL(gemv_nt_q6_k_untiled_kernel, grid, block, 0, as_stream(stream), a, b, c, m,
                       n, k);
    return hip_status(hipGetLastError());
}

/* Fused Q5_K GEMV: 176-byte super-blocks (qh + qs), A tile in LDS. */
template <int COLS>
__global__ void gemv_nt_q5_k_tiled_kernel(const float *a, const uint8_t *b, float *c, int m, int n,
                                          int k) {
    __shared__ float As[256];
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int col = static_cast<int>(blockIdx.x) * COLS + tid;
    if (row >= m) {
        return;
    }
    const int nblocks = (k + 255) / 256;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    const uint8_t *brow =
        (tid < COLS && col < n)
            ? b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 176
            : nullptr;
    float acc = 0.f;
    for (int ib = 0; ib < nblocks; ++ib) {
        const int k0 = ib * 256;
        As[tid] = (k0 + tid < k) ? arow[k0 + tid] : 0.f;
        __syncthreads();
        if (brow) {
            const uint8_t *blk = brow + static_cast<size_t>(ib) * 176;
            const float d = allpaca_f16le_to_f32(blk);
            const float dmin = allpaca_f16le_to_f32(blk + 2);
            const uint8_t *scales = blk + 4;
            const uint8_t *qh = blk + 16;
            const uint8_t *qs = blk + 48;
            int q_off = 0;
            int is = 0;
            uint8_t u1 = 1;
            uint8_t u2 = 2;
            for (int g = 0; g < 4; ++g) {
                uint8_t sc1 = 0;
                uint8_t m1 = 0;
                uint8_t sc2 = 0;
                uint8_t m2 = 0;
                q4k_scale_min(is, scales, &sc1, &m1);
                q4k_scale_min(is + 1, scales, &sc2, &m2);
                const float d1 = d * static_cast<float>(sc1);
                const float min1 = dmin * static_cast<float>(m1);
                const float d2 = d * static_cast<float>(sc2);
                const float min2 = dmin * static_cast<float>(m2);
                for (int l = 0; l < 32; ++l) {
                    const int i0 = g * 64 + l;
                    const int i1 = i0 + 32;
                    const uint8_t packed = qs[q_off + l];
                    const float qlo =
                        static_cast<float>(packed & 0x0F) + ((qh[l] & u1) ? 16.f : 0.f);
                    const float qhi =
                        static_cast<float>(packed >> 4) + ((qh[l] & u2) ? 16.f : 0.f);
                    acc += As[i0] * (d1 * qlo - min1);
                    acc += As[i1] * (d2 * qhi - min2);
                }
                q_off += 32;
                is += 2;
                u1 = static_cast<uint8_t>(u1 << 2);
                u2 = static_cast<uint8_t>(u2 << 2);
            }
        }
        __syncthreads();
    }
    if (brow) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
    }
}

__global__ void gemv_nt_q5_k_untiled_kernel(const float *a, const uint8_t *b, float *c, int m,
                                            int n, int k) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    if (col >= n || row >= m) {
        return;
    }
    const int nblocks = (k + 255) / 256;
    const uint8_t *brow = b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 176;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    float acc = 0.f;
    for (int ib = 0; ib < nblocks; ++ib) {
        const uint8_t *blk = brow + static_cast<size_t>(ib) * 176;
        const float d = allpaca_f16le_to_f32(blk);
        const float dmin = allpaca_f16le_to_f32(blk + 2);
        const uint8_t *scales = blk + 4;
        const uint8_t *qh = blk + 16;
        const uint8_t *qs = blk + 48;
        const int k0 = ib * 256;
        int q_off = 0;
        int is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (int g = 0; g < 4; ++g) {
            uint8_t sc1 = 0;
            uint8_t m1 = 0;
            uint8_t sc2 = 0;
            uint8_t m2 = 0;
            q4k_scale_min(is, scales, &sc1, &m1);
            q4k_scale_min(is + 1, scales, &sc2, &m2);
            const float d1 = d * static_cast<float>(sc1);
            const float min1 = dmin * static_cast<float>(m1);
            const float d2 = d * static_cast<float>(sc2);
            const float min2 = dmin * static_cast<float>(m2);
            for (int l = 0; l < 32; ++l) {
                const int i0 = k0 + g * 64 + l;
                const int i1 = i0 + 32;
                const uint8_t packed = qs[q_off + l];
                const float qlo =
                    static_cast<float>(packed & 0x0F) + ((qh[l] & u1) ? 16.f : 0.f);
                const float qhi =
                    static_cast<float>(packed >> 4) + ((qh[l] & u2) ? 16.f : 0.f);
                if (i0 < k) {
                    acc += arow[i0] * (d1 * qlo - min1);
                }
                if (i1 < k) {
                    acc += arow[i1] * (d2 * qhi - min2);
                }
            }
            q_off += 32;
            is += 2;
            u1 = static_cast<uint8_t>(u1 << 2);
            u2 = static_cast<uint8_t>(u2 << 2);
        }
    }
    c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
}

static int launch_gemm_nt_q5_k_cols(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                    int32_t k, int32_t cols, hipStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    if (k % 256 == 0) {
        // `cols` tiled the per-thread kernel; the warp kernel has no column tile.
        return launch_gemm_nt_kquant_warp<5>(a, b, c, m, n, k, stream);
    }
    clear_last_error();
    int use = cols;
    if (use != 32 && use != 64 && use != 128) {
        use = 64;
    }
    dim3 block(256);
    dim3 grid(static_cast<unsigned>((n + use - 1) / use), static_cast<unsigned>(m));
    if (use == 32) {
        hipLaunchKernelGGL(gemv_nt_q5_k_tiled_kernel<32>, grid, block, 0, stream, a, b, c, m, n, k);
    } else if (use == 128) {
        hipLaunchKernelGGL(gemv_nt_q5_k_tiled_kernel<128>, grid, block, 0, stream, a, b, c, m, n, k);
    } else {
        hipLaunchKernelGGL(gemv_nt_q5_k_tiled_kernel<64>, grid, block, 0, stream, a, b, c, m, n, k);
    }
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_gemm_nt_q5_k(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                    int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_q5_k_cols(a, b, c, m, n, k, 64, as_stream(stream));
}

extern "C" int allpaca_gemm_nt_q5_k_cols(const float *a, const uint8_t *b, float *c, int32_t m,
                                         int32_t n, int32_t k, int32_t cols,
                                         allpaca_stream_t stream) {
    return launch_gemm_nt_q5_k_cols(a, b, c, m, n, k, cols, as_stream(stream));
}

extern "C" int allpaca_gemm_nt_q5_k_untiled(const float *a, const uint8_t *b, float *c, int32_t m,
                                            int32_t n, int32_t k, allpaca_stream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    dim3 block(256);
    dim3 grid(static_cast<unsigned>((n + 255) / 256), static_cast<unsigned>(m));
    hipLaunchKernelGGL(gemv_nt_q5_k_untiled_kernel, grid, block, 0, as_stream(stream), a, b, c, m,
                       n, k);
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
    char composed[256];
    const char *label = prop.name;
    if (prop.gcnArchName[0] != '\0'
        && (prop.name[0] == '\0' || std::strcmp(prop.name, "AMD Radeon Graphics") == 0)) {
        std::snprintf(composed, sizeof(composed), "AMD %s", prop.gcnArchName);
        label = composed;
    }
    int i = 0;
    for (; i < name_len - 1 && label[i]; ++i) {
        name[i] = label[i];
    }
    name[i] = '\0';
    *memory_bytes = static_cast<uint64_t>(prop.totalGlobalMem);
    /* Relative bandwidth from clock * bus width. GDDR transfers on both edges. */
    const double clock_ghz = static_cast<double>(prop.memoryClockRate) / 1.0e6;
    const double bytes_per_cycle = static_cast<double>(prop.memoryBusWidth) / 8.0 * 2.0;
    *bandwidth_gbps = clock_ghz * bytes_per_cycle;
    return ALLPACA_OK;
}

extern "C" int allpaca_hip_device_arch(int32_t id, char *arch, int32_t arch_len,
                                       int32_t *l2_bytes) {
    if (!arch || arch_len <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    hipDeviceProp_t prop{};
    hipError_t err = hipGetDeviceProperties(&prop, id);
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    int i = 0;
    for (; i < arch_len - 1 && prop.gcnArchName[i]; ++i) {
        arch[i] = prop.gcnArchName[i];
    }
    arch[i] = '\0';
    if (l2_bytes) {
        *l2_bytes = prop.l2CacheSize;
    }
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

#if defined(_WIN32)
static std::mutex g_win_vram_mu;
static IDXGIAdapter3 *g_win_adapter[8];
static uint64_t g_win_reserved[8];

static IDXGIAdapter3 *dxgi_adapter_for_hip(int32_t device) {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
        return nullptr;
    }
    IDXGIFactory1 *factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void **>(&factory))) ||
        factory == nullptr) {
        return nullptr;
    }
    IDXGIAdapter3 *found = nullptr;
    for (UINT i = 0; found == nullptr; ++i) {
        IDXGIAdapter1 *ad = nullptr;
        HRESULT hr = factory->EnumAdapters1(i, &ad);
        if (hr == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (FAILED(hr) || ad == nullptr) {
            continue;
        }
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(ad->GetDesc1(&desc))) {
            char utf8[256]{};
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, utf8, sizeof(utf8), nullptr,
                                nullptr);
            // LUID identifies the adapter even with two identical boards;
            // the name match is the fallback when HIP leaves it zeroed.
            char luid[8]{};
            std::memcpy(luid, &desc.AdapterLuid.LowPart, 4);
            std::memcpy(luid + 4, &desc.AdapterLuid.HighPart, 4);
            const bool has_luid = std::memcmp(prop.luid, "\0\0\0\0\0\0\0\0", 8) != 0;
            const bool same_luid = has_luid && std::memcmp(prop.luid, luid, 8) == 0;
            const bool same = utf8[0] != '\0' && prop.name[0] != '\0' &&
                              (std::strstr(utf8, prop.name) != nullptr ||
                               std::strstr(prop.name, utf8) != nullptr);
            const bool rx9060 = std::strstr(prop.name, "9060") && std::strstr(utf8, "9060");
            const bool rx9070 = std::strstr(prop.name, "9070") && std::strstr(utf8, "9070");
            if (same_luid || (!has_luid && (same || rx9060 || rx9070))) {
                (void)ad->QueryInterface(__uuidof(IDXGIAdapter3),
                                         reinterpret_cast<void **>(&found));
            }
        }
        ad->Release();
    }
    factory->Release();
    return found;
}

static void win_reserve_hip_device(int32_t device, size_t bytes) {
    if (device < 0 || device >= 8) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_win_vram_mu);
    g_win_reserved[device] += bytes;
    if (g_win_adapter[device] == nullptr) {
        g_win_adapter[device] = dxgi_adapter_for_hip(device);
    }
    if (g_win_adapter[device] != nullptr) {
        (void)g_win_adapter[device]->SetVideoMemoryReservation(
            0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, g_win_reserved[device]);
    }
}

static PDH_HQUERY g_win_pdh_query[8];
static PDH_HCOUNTER g_win_pdh_dedicated[8];

/* Dedicated VRAM on `device` not resident for any process, as Task Manager
 * shows it. hipMemGetInfo and the per-process DXGI budget on Windows both
 * report the board as nearly free while the desktop (dwm, browsers) holds
 * several GiB of it; allocations past what is really left are demoted by WDDM
 * to shared system memory and then read over PCIe on every kernel. */
static bool win_local_headroom(int32_t device, uint64_t *headroom) {
    if (device < 0 || device >= 8) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_win_vram_mu);
    if (g_win_adapter[device] == nullptr) {
        g_win_adapter[device] = dxgi_adapter_for_hip(device);
    }
    if (g_win_adapter[device] == nullptr) {
        return false;
    }
    DXGI_ADAPTER_DESC1 desc{};
    if (FAILED(g_win_adapter[device]->GetDesc1(&desc)) || desc.DedicatedVideoMemory == 0) {
        return false;
    }
    if (g_win_pdh_query[device] == nullptr) {
        wchar_t path[160];
        std::swprintf(path, sizeof(path) / sizeof(path[0]),
                      L"\\GPU Adapter Memory(luid_0x%08X_0x%08X_phys_0)\\Dedicated Usage",
                      static_cast<unsigned>(desc.AdapterLuid.HighPart),
                      static_cast<unsigned>(desc.AdapterLuid.LowPart));
        PDH_HQUERY q = nullptr;
        PDH_HCOUNTER c = nullptr;
        if (PdhOpenQueryW(nullptr, 0, &q) != ERROR_SUCCESS) {
            return false;
        }
        if (PdhAddEnglishCounterW(q, path, 0, &c) != ERROR_SUCCESS) {
            PdhCloseQuery(q);
            return false;
        }
        g_win_pdh_query[device] = q;
        g_win_pdh_dedicated[device] = c;
    }
    PDH_FMT_COUNTERVALUE v{};
    if (PdhCollectQueryData(g_win_pdh_query[device]) != ERROR_SUCCESS ||
        PdhGetFormattedCounterValue(g_win_pdh_dedicated[device], PDH_FMT_LARGE, nullptr, &v) !=
            ERROR_SUCCESS ||
        v.largeValue < 0) {
        return false;
    }
    const uint64_t used = static_cast<uint64_t>(v.largeValue);
    const uint64_t total = static_cast<uint64_t>(desc.DedicatedVideoMemory);
    *headroom = total > used ? total - used : 0;
    return true;
}
#endif

static void retain_default_pool(int32_t device) {
    hipMemPool_t pool = nullptr;
    if (hipDeviceGetDefaultMemPool(&pool, device) != hipSuccess || pool == nullptr) {
        return;
    }
    uint64_t thresh = ~uint64_t{0};
    (void)hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &thresh);
}

extern "C" int allpaca_hip_malloc(void **ptr, size_t bytes, int32_t device) {
    if (!ptr || bytes == 0) {
        return ALLPACA_ERR_INVALID;
    }
    hipError_t err = hipSetDevice(device);
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    retain_default_pool(device);
    *ptr = nullptr;
    err = hipExtMallocWithFlags(ptr, bytes, hipDeviceMallocContiguous);
    if (err != hipSuccess || *ptr == nullptr) {
        err = hipMalloc(ptr, bytes);
    }
    if (err != hipSuccess || *ptr == nullptr) {
        return hip_status(err != hipSuccess ? err : hipErrorOutOfMemory);
    }
#if defined(_WIN32)
    win_reserve_hip_device(device, bytes);
#endif
    return ALLPACA_OK;
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

extern "C" int allpaca_hip_free_async(void *ptr, allpaca_stream_t stream) {
    if (!ptr) {
        return ALLPACA_OK;
    }
    hipStream_t s = as_stream(stream);
    if (s) {
        return hip_status(hipFreeAsync(ptr, s));
    }
    return allpaca_hip_free(ptr);
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

extern "C" int allpaca_hip_host_register(void *ptr, size_t bytes, unsigned int flags) {
    if (!ptr || bytes == 0) {
        return ALLPACA_ERR_INVALID;
    }
    return hip_status(hipHostRegister(ptr, bytes, flags));
}

extern "C" int allpaca_hip_host_unregister(void *ptr) {
    if (!ptr) {
        return ALLPACA_OK;
    }
    return hip_status(hipHostUnregister(ptr));
}

#if defined(_WIN32) && defined(ALLPACA_HAS_DIRECTSTORAGE)
using Microsoft::WRL::ComPtr;

struct allpaca_dstorage_buffer {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Fence> fence;
    ComPtr<IDStorageFactory> factory;
    ComPtr<IDStorageQueue> queue;
    ComPtr<IDStorageStatusArray> status;
    hipExternalMemory_t external_memory = nullptr;
    hipExternalSemaphore_t external_semaphore = nullptr;
    void *hip_ptr = nullptr;
    size_t bytes = 0;
    uint64_t fence_value = 0;
    std::mutex lock;
};

static void destroy_dstorage_buffer(allpaca_dstorage_buffer *buffer) {
    if (!buffer) {
        return;
    }
    (void)hipStreamSynchronize(nullptr);
    if (buffer->hip_ptr) {
        (void)hipFree(buffer->hip_ptr);
    }
    if (buffer->external_semaphore) {
        (void)hipDestroyExternalSemaphore(buffer->external_semaphore);
    }
    if (buffer->external_memory) {
        (void)hipDestroyExternalMemory(buffer->external_memory);
    }
    delete buffer;
}

extern "C" int allpaca_dstorage_alloc(void **handle, void **hip_ptr, size_t bytes,
                                       int32_t device_id) {
    if (!handle || !hip_ptr || bytes == 0) {
        return ALLPACA_ERR_INVALID;
    }
    *handle = nullptr;
    *hip_ptr = nullptr;
    if (hipSetDevice(device_id) != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }

    ComPtr<IDXGIFactory6> dxgi;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&dxgi)))) {
        return ALLPACA_ERR_DEVICE;
    }
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        if (dxgi->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        DXGI_ADAPTER_DESC1 adapter_desc{};
        if (FAILED(adapter->GetDesc1(&adapter_desc)) ||
            (adapter_desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
            continue;
        }

        ComPtr<ID3D12Device> d3d_device;
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                     IID_PPV_ARGS(&d3d_device)))) {
            continue;
        }
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap.CreationNodeMask = 1;
        heap.VisibleNodeMask = 1;
        D3D12_RESOURCE_DESC resource_desc{};
        resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resource_desc.Width = static_cast<UINT64>(bytes);
        resource_desc.Height = 1;
        resource_desc.DepthOrArraySize = 1;
        resource_desc.MipLevels = 1;
        resource_desc.SampleDesc.Count = 1;
        resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        resource_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ComPtr<ID3D12Resource> resource;
        if (FAILED(d3d_device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_SHARED, &resource_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                IID_PPV_ARGS(&resource)))) {
            continue;
        }

        HANDLE resource_handle = nullptr;
        if (FAILED(d3d_device->CreateSharedHandle(resource.Get(), nullptr, GENERIC_ALL, nullptr,
                                                  &resource_handle))) {
            continue;
        }
        hipExternalMemoryHandleDesc memory_desc{};
        memory_desc.type = hipExternalMemoryHandleTypeD3D12Resource;
        memory_desc.handle.win32.handle = resource_handle;
        memory_desc.size = static_cast<unsigned long long>(bytes);
        hipExternalMemory_t external_memory = nullptr;
        hipError_t hip_result = hipImportExternalMemory(&external_memory, &memory_desc);
        CloseHandle(resource_handle);
        if (hip_result != hipSuccess) {
            (void)hipGetLastError();
            continue;
        }
        hipExternalMemoryBufferDesc buffer_desc{};
        buffer_desc.size = static_cast<unsigned long long>(bytes);
        void *mapped = nullptr;
        hip_result = hipExternalMemoryGetMappedBuffer(&mapped, external_memory, &buffer_desc);
        if (hip_result != hipSuccess || !mapped) {
            (void)hipDestroyExternalMemory(external_memory);
            (void)hipGetLastError();
            continue;
        }

        ComPtr<ID3D12Fence> fence;
        if (FAILED(d3d_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence)))) {
            (void)hipFree(mapped);
            (void)hipDestroyExternalMemory(external_memory);
            continue;
        }
        HANDLE fence_handle = nullptr;
        if (FAILED(d3d_device->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, nullptr,
                                                  &fence_handle))) {
            (void)hipFree(mapped);
            (void)hipDestroyExternalMemory(external_memory);
            continue;
        }
        hipExternalSemaphoreHandleDesc semaphore_desc{};
        semaphore_desc.type = hipExternalSemaphoreHandleTypeD3D12Fence;
        semaphore_desc.handle.win32.handle = fence_handle;
        hipExternalSemaphore_t external_semaphore = nullptr;
        hip_result = hipImportExternalSemaphore(&external_semaphore, &semaphore_desc);
        CloseHandle(fence_handle);
        if (hip_result != hipSuccess) {
            (void)hipFree(mapped);
            (void)hipDestroyExternalMemory(external_memory);
            (void)hipGetLastError();
            continue;
        }

        ComPtr<IDStorageFactory> factory;
        if (FAILED(DStorageGetFactory(IID_PPV_ARGS(&factory)))) {
            (void)hipDestroyExternalSemaphore(external_semaphore);
            (void)hipFree(mapped);
            (void)hipDestroyExternalMemory(external_memory);
            continue;
        }
        DSTORAGE_QUEUE_DESC queue_desc{};
        queue_desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        queue_desc.Capacity = DSTORAGE_MIN_QUEUE_CAPACITY;
        queue_desc.Priority = DSTORAGE_PRIORITY_NORMAL;
        queue_desc.Name = "allpaca NVMe-to-VRAM";
        queue_desc.Device = d3d_device.Get();
        ComPtr<IDStorageQueue> queue;
        if (FAILED(factory->CreateQueue(&queue_desc, IID_PPV_ARGS(&queue)))) {
            (void)hipDestroyExternalSemaphore(external_semaphore);
            (void)hipFree(mapped);
            (void)hipDestroyExternalMemory(external_memory);
            continue;
        }
        ComPtr<IDStorageStatusArray> status;
        if (FAILED(factory->CreateStatusArray(1, "allpaca DirectStorage completion",
                                              IID_PPV_ARGS(&status)))) {
            (void)hipDestroyExternalSemaphore(external_semaphore);
            (void)hipFree(mapped);
            (void)hipDestroyExternalMemory(external_memory);
            continue;
        }

        auto *result = new (std::nothrow) allpaca_dstorage_buffer();
        if (!result) {
            (void)hipDestroyExternalSemaphore(external_semaphore);
            (void)hipFree(mapped);
            (void)hipDestroyExternalMemory(external_memory);
            return ALLPACA_ERR_DEVICE;
        }
        result->device = d3d_device;
        result->resource = resource;
        result->fence = fence;
        result->factory = factory;
        result->queue = queue;
        result->status = status;
        result->external_memory = external_memory;
        result->external_semaphore = external_semaphore;
        result->hip_ptr = mapped;
        result->bytes = bytes;
        *handle = result;
        *hip_ptr = mapped;
        return ALLPACA_OK;
    }
    return ALLPACA_ERR_DEVICE;
}

extern "C" int allpaca_dstorage_read(void *handle, const uint16_t *path, uint64_t file_offset,
                                      uint64_t buffer_offset, size_t bytes) {
    auto *buffer = static_cast<allpaca_dstorage_buffer *>(handle);
    if (!buffer || !path || bytes == 0 || bytes > std::numeric_limits<UINT32>::max() ||
        buffer_offset > buffer->bytes || bytes > buffer->bytes - buffer_offset) {
        return ALLPACA_ERR_INVALID;
    }
    std::lock_guard<std::mutex> guard(buffer->lock);
    ComPtr<IDStorageFile> file;
    if (FAILED(buffer->factory->OpenFile(reinterpret_cast<const WCHAR *>(path),
                                         IID_PPV_ARGS(&file)))) {
        return ALLPACA_ERR_DEVICE;
    }

    DSTORAGE_REQUEST request{};
    request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
    request.Source.File.Source = file.Get();
    request.Source.File.Offset = file_offset;
    request.Source.File.Size = static_cast<UINT32>(bytes);
    request.Destination.Buffer.Resource = buffer->resource.Get();
    request.Destination.Buffer.Offset = buffer_offset;
    request.Destination.Buffer.Size = static_cast<UINT32>(bytes);
    request.UncompressedSize = static_cast<UINT32>(bytes);
    request.Name = "allpaca expert fill";

    const uint64_t fence_value = ++buffer->fence_value;
    buffer->queue->EnqueueRequest(&request);
    buffer->queue->EnqueueSignal(buffer->fence.Get(), fence_value);
    buffer->queue->EnqueueStatus(buffer->status.Get(), 0);
    buffer->queue->Submit();
    const ULONGLONG deadline = GetTickCount64() + 30000;
    while (!buffer->status->IsComplete(0)) {
        if (GetTickCount64() >= deadline) {
            return ALLPACA_ERR_DEVICE;
        }
        Sleep(0);
    }
    if (FAILED(buffer->status->GetHResult(0))) {
        return ALLPACA_ERR_DEVICE;
    }

    hipExternalSemaphoreWaitParams wait{};
    wait.params.fence.value = fence_value;
    if (hipWaitExternalSemaphoresAsync(&buffer->external_semaphore, &wait, 1, nullptr) !=
        hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    return ALLPACA_OK;
}

extern "C" int allpaca_dstorage_free(void *handle) {
    destroy_dstorage_buffer(static_cast<allpaca_dstorage_buffer *>(handle));
    return ALLPACA_OK;
}
#else
extern "C" int allpaca_dstorage_alloc(void **, void **, size_t, int32_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}
extern "C" int allpaca_dstorage_read(void *, const uint16_t *, uint64_t, uint64_t, size_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}
extern "C" int allpaca_dstorage_free(void *) { return ALLPACA_ERR_NOT_IMPLEMENTED; }
#endif

static hipMemcpyKind to_hip_memcpy_kind(int32_t kind) {
    switch (kind) {
    case ALLPACA_MEMCPY_H2D:
        return hipMemcpyHostToDevice;
    case ALLPACA_MEMCPY_D2H:
    case ALLPACA_MEMCPY_D2H_ASYNC:
        return hipMemcpyDeviceToHost;
    case ALLPACA_MEMCPY_D2D:
        return hipMemcpyDeviceToDevice;
    default:
        return hipMemcpyDefault;
    }
}

/* hipMemcpyBatchAsync shipped in ROCm 7.1, but some SDK headers (Windows 7.1)
 * do not declare it even though the runtime exports it. Resolve it from the
 * loaded runtime rather than declaring the prototype, so a missing symbol
 * degrades to per-copy async instead of breaking the build. `attrs`,
 * `attrsIdxs` and `numAttrs` are documented as unsupported by HIP, so the
 * attribute pointer is always null and its layout never matters here. */
typedef hipError_t (*allpaca_memcpy_batch_fn)(void **dsts, void **srcs, size_t *sizes, size_t count,
                                              void *attrs, size_t *attrsIdxs, size_t numAttrs,
                                              size_t *failIdx, hipStream_t stream);

static allpaca_memcpy_batch_fn memcpy_batch_fn() {
    static allpaca_memcpy_batch_fn fn = []() -> allpaca_memcpy_batch_fn {
        void *sym = nullptr;
        hipDriverProcAddressQueryResult status{};
        if (hipGetProcAddress("hipMemcpyBatchAsync", &sym, HIP_VERSION, 0, &status) == hipSuccess &&
            sym) {
            return reinterpret_cast<allpaca_memcpy_batch_fn>(sym);
        }
        return nullptr;
    }();
    return fn;
}

extern "C" int allpaca_hip_has_memcpy_batch(void) { return memcpy_batch_fn() ? 1 : 0; }

extern "C" int allpaca_hip_memcpy_batch(void **dsts, void **srcs, size_t *sizes, size_t count,
                                        int32_t kind, allpaca_stream_t stream) {
    if (!dsts || !srcs || !sizes) {
        return ALLPACA_ERR_INVALID;
    }
    if (count == 0) {
        return ALLPACA_OK;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!dsts[i] || !srcs[i]) {
            return ALLPACA_ERR_INVALID;
        }
    }
    clear_last_error();
    hipStream_t s = as_stream(stream);
    allpaca_memcpy_batch_fn batch = memcpy_batch_fn();
    if (batch) {
        size_t fail_idx = 0;
        hipError_t err = batch(dsts, srcs, sizes, count, nullptr, nullptr, 0, &fail_idx, s);
        if (err != hipSuccess) {
            return ALLPACA_ERR_DEVICE;
        }
    } else {
        const hipMemcpyKind mk = to_hip_memcpy_kind(kind);
        for (size_t i = 0; i < count; ++i) {
            hipError_t err = hipMemcpyAsync(dsts[i], srcs[i], sizes[i], mk, s);
            if (err != hipSuccess) {
                return ALLPACA_ERR_DEVICE;
            }
        }
    }
    /* Host-visible D2H is the wait point; async D2H / H2D / D2D stay queued. */
    if (kind == ALLPACA_MEMCPY_D2H) {
        return hip_status(hipStreamSynchronize(s));
    }
    return ALLPACA_OK;
}

extern "C" int allpaca_hip_memcpy(void *dst, const void *src, size_t bytes, int32_t kind,
                                 allpaca_stream_t stream) {
    if (!dst || !src) {
        return ALLPACA_ERR_INVALID;
    }
    const hipMemcpyKind mk = to_hip_memcpy_kind(kind);
    hipError_t err = hipMemcpyAsync(dst, src, bytes, mk, as_stream(stream));
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    /* Host-visible D2H is the wait point; async D2H / H2D / D2D stay queued. */
    if (kind == ALLPACA_MEMCPY_D2H) {
        return hip_status(hipStreamSynchronize(as_stream(stream)));
    }
    return ALLPACA_OK;
}

extern "C" int allpaca_hip_memset(void *dst, int32_t value, size_t bytes, allpaca_stream_t stream) {
    if (!dst && bytes) {
        return ALLPACA_ERR_INVALID;
    }
    if (bytes == 0) {
        return ALLPACA_OK;
    }
    hipStream_t s = as_stream(stream);
    const int fill = value & 0xff;
    if (s) {
        return hip_status(hipMemsetAsync(dst, fill, bytes, s));
    }
    return hip_status(hipMemset(dst, fill, bytes));
}

__device__ unsigned int g_touch_sink;

__global__ void touch_resident_kernel(const uint8_t *p, size_t bytes) {
    constexpr size_t kStride = 4096;
    const size_t n = (bytes + kStride - 1) / kStride;
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned int acc = 0;
    for (; i < n; i += step) {
        const size_t off = i * kStride;
        if (off + sizeof(unsigned int) <= bytes) {
            acc += *reinterpret_cast<const volatile unsigned int *>(p + off);
        }
    }
    if (threadIdx.x == 0) {
        atomicAdd(&g_touch_sink, acc);
    }
}

extern "C" int allpaca_hip_touch_resident(void *ptr, size_t bytes, int32_t device) {
    if (!ptr || bytes == 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipError_t err = hipSetDevice(device);
    if (err != hipSuccess) {
        return hip_status(err);
    }
    constexpr int kThreads = 256;
    const int blocks = 512;
    hipLaunchKernelGGL(touch_resident_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(kThreads)), 0, nullptr,
                       static_cast<const uint8_t *>(ptr), bytes);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_hip_memcpy_peer(void *dst, int32_t dst_dev, const void *src, int32_t src_dev,
                                      size_t bytes, allpaca_stream_t stream) {
    if (!dst || !src) {
        return ALLPACA_ERR_INVALID;
    }
    int prev = 0;
    if (hipGetDevice(&prev) != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    /* Drain src compute. hipMemcpyPeerAsync on the dst stream does not wait
       for kernels on the src device; ping-pong layer splits copied a stale x. */
    if (hipSetDevice(src_dev) != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    hipError_t err = hipStreamSynchronize(nullptr);
    if (err != hipSuccess) {
        (void)hipSetDevice(prev);
        return ALLPACA_ERR_DEVICE;
    }
    hipStream_t s = as_stream(stream);
    err = hipMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, s);
    if (err != hipSuccess) {
        (void)hipSetDevice(prev);
        return ALLPACA_ERR_DEVICE;
    }
    err = hipStreamSynchronize(s);
    (void)hipSetDevice(prev);
    return hip_status(err);
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

extern "C" int allpaca_hip_stream_create(allpaca_stream_t *stream) {
    if (!stream) {
        return ALLPACA_ERR_INVALID;
    }
    hipStream_t s = nullptr;
    /* Non-blocking so this stream never implicitly serializes with the null
     * compute stream; the join is an explicit event wait. */
    hipError_t err = hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    *stream = s;
    return ALLPACA_OK;
}

extern "C" int allpaca_hip_stream_destroy(allpaca_stream_t stream) {
    if (!stream) {
        return ALLPACA_OK;
    }
    return hip_status(hipStreamDestroy(static_cast<hipStream_t>(stream)));
}

extern "C" int allpaca_hip_stream_wait_event(allpaca_stream_t stream, void *event) {
    if (!event) {
        return ALLPACA_ERR_INVALID;
    }
    return hip_status(
        hipStreamWaitEvent(as_stream(stream), static_cast<hipEvent_t>(event), 0));
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

extern "C" int allpaca_hip_event_synchronize(void *event) {
    if (!event) {
        return ALLPACA_ERR_INVALID;
    }
    return hip_status(hipEventSynchronize(static_cast<hipEvent_t>(event)));
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

extern "C" int allpaca_hip_mem_info(int32_t device, uint64_t *free_bytes, uint64_t *total_bytes) {
    if (!free_bytes || !total_bytes) {
        return ALLPACA_ERR_INVALID;
    }
    // hipMemGetInfo reports whichever device is current, so a different one has
    // to be bound first and then restored: the caller is the telemetry sampler
    // running on a decode thread that goes straight back to launching kernels,
    // and leaving it bound to the draft device would silently redirect target
    // work. hipSetDevice takes a driver-wide lock, so the common case -- the
    // device is already current -- skips the rebinding entirely rather than
    // contending with every other thread's launches.
    int prev = 0;
    hipError_t err = hipGetDevice(&prev);
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    const bool rebind = prev != device;
    if (rebind) {
        err = hipSetDevice(device);
        if (err != hipSuccess) {
            return ALLPACA_ERR_DEVICE;
        }
    }
    size_t free_b = 0;
    size_t total_b = 0;
    err = hipMemGetInfo(&free_b, &total_b);
    if (rebind && hipSetDevice(prev) != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    *free_bytes = static_cast<uint64_t>(free_b);
    *total_bytes = static_cast<uint64_t>(total_b);
    return ALLPACA_OK;
}

extern "C" int allpaca_hip_local_headroom(int32_t device, uint64_t *bytes) {
    if (!bytes) {
        return ALLPACA_ERR_INVALID;
    }
#if defined(_WIN32)
    return win_local_headroom(device, bytes) ? ALLPACA_OK : ALLPACA_ERR_NOT_IMPLEMENTED;
#else
    (void)device;
    return ALLPACA_ERR_NOT_IMPLEMENTED;
#endif
}

extern "C" int allpaca_gemm(const float *a, const float *b, float *c, int32_t m, int32_t n,
                            int32_t k, allpaca_stream_t stream) {
    return launch_gemm(a, b, c, m, n, k, as_stream(stream));
}

extern "C" int allpaca_gemm_nt(const float *a, const float *b, float *c, int32_t m, int32_t n,
                               int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt(a, b, c, m, n, k, as_stream(stream), 1);
}

extern "C" int allpaca_gemm_nt_untiled(const float *a, const float *b, float *c, int32_t m,
                                       int32_t n, int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt(a, b, c, m, n, k, as_stream(stream), 0);
}

extern "C" int allpaca_gemm_nt_q8_0(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                    int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_q8_0(a, b, c, m, n, k, as_stream(stream), ALLPACA_Q8_GEMV_INTDOT);
}

extern "C" int allpaca_gemm_nt_f16(const float *a, const uint16_t *b, float *c, int32_t m, int32_t n,
                                  int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_f16(a, b, c, m, n, k, as_stream(stream), 1);
}

extern "C" int allpaca_gemm_nt_f16_untiled(const float *a, const uint16_t *b, float *c, int32_t m,
                                           int32_t n, int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_f16(a, b, c, m, n, k, as_stream(stream), 0);
}

extern "C" int allpaca_gemm_nt_bf16(const float *a, const uint16_t *b, float *c, int32_t m, int32_t n,
                                   int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_bf16(a, b, c, m, n, k, as_stream(stream), 1);
}

extern "C" int allpaca_gemm_nt_bf16_untiled(const float *a, const uint16_t *b, float *c, int32_t m,
                                            int32_t n, int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_bf16(a, b, c, m, n, k, as_stream(stream), 0);
}

extern "C" int allpaca_gemm_nt_q8_0_intdot(const float *a, const uint8_t *b, float *c, int32_t m,
                                           int32_t n, int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_q8_0(a, b, c, m, n, k, as_stream(stream), 1);
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
                       n_kv, n_heads, n_kv_heads, head_dim, causal, static_cast<const uint8_t *>(nullptr),
                       static_cast<const float *>(nullptr));
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_attention_sinks(const float *q, const float *k, const float *v, float *out,
                                      int32_t n_q, int32_t n_kv, int32_t n_heads, int32_t n_kv_heads,
                                      int32_t head_dim, int32_t causal, const float *sinks,
                                      allpaca_stream_t stream) {
    if (!q || !k || !v || !out || n_q <= 0 || n_kv <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_dim <= 0 || (n_heads % n_kv_heads) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    dim3 block(32);
    dim3 grid(static_cast<unsigned>(n_q), static_cast<unsigned>(n_heads));
    hipLaunchKernelGGL(attention_flash_kernel, grid, block, 0, as_stream(stream), q, k, v, out, n_q,
                       n_kv, n_heads, n_kv_heads, head_dim, causal, static_cast<const uint8_t *>(nullptr),
                       sinks);
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
                       n_kv, n_heads, n_kv_heads, head_dim, 0, mask, static_cast<const float *>(nullptr));
    return hip_status(hipGetLastError());
}

__device__ float block_reduce_sum(float v, float *red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
        if (static_cast<int>(threadIdx.x) < s) {
            red[threadIdx.x] += red[threadIdx.x + s];
        }
        __syncthreads();
    }
    return red[0];
}

/* RMSNorm: one block per row (decode launches 1). */
__global__ void rmsnorm_kernel(const float *x, const float *weight, float *out, int n, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    x += static_cast<size_t>(row) * static_cast<size_t>(n);
    out += static_cast<size_t>(row) * static_cast<size_t>(n);
    __shared__ float red[256];
    float local = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        const float v = x[i];
        local += v * v;
    }
    const float ss = block_reduce_sum(local, red);
    const float inv = rsqrtf(ss / static_cast<float>(n) + eps);
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        out[i] = x[i] * inv * weight[i];
    }
}

/* Mean/var LayerNorm (ViT / CLIP). Not RMSNorm. One block per row. */
__global__ void layernorm_kernel(const float *x, const float *weight, const float *bias, float *out,
                                 int n, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    x += static_cast<size_t>(row) * static_cast<size_t>(n);
    out += static_cast<size_t>(row) * static_cast<size_t>(n);
    __shared__ float red[256];
    float local = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        local += x[i];
    }
    const float mean = block_reduce_sum(local, red) / static_cast<float>(n);
    local = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        const float d = x[i] - mean;
        local += d * d;
    }
    const float ss = block_reduce_sum(local, red);
    const float inv = rsqrtf(ss / static_cast<float>(n) + eps);
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        out[i] = (x[i] - mean) * inv * weight[i] + bias[i];
    }
}

__global__ void residual_rmsnorm_kernel(float *x, const float *residual, const float *weight,
                                        float *out, int n, float eps) {
    __shared__ float red[256];
    float local = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        const float v = x[i] + residual[i];
        x[i] = v;
        local += v * v;
    }
    const float ss = block_reduce_sum(local, red);
    const float inv = rsqrtf(ss / static_cast<float>(n) + eps);
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        out[i] = x[i] * inv * weight[i];
    }
}

/* One block per V-head. Warp shuffle, then one smem reduce across warps. */
__device__ float gdn_block_sum(float v, float *smem) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down(v, offset);
    }
    if (lane == 0) {
        smem[warp] = v;
    }
    __syncthreads();
    float sum = 0.f;
    if (tid == 0) {
        const int nwarps = (static_cast<int>(blockDim.x) + 31) >> 5;
        for (int w = 0; w < nwarps; ++w) {
            sum += smem[w];
        }
        smem[0] = sum;
    }
    __syncthreads();
    sum = smem[0];
    __syncthreads();
    return sum;
}

// One block per V-head, one thread per K-lane. The scalar kernel used a single
// thread for the whole head_v x head_k state and dominated decode.
__global__ void gated_delta_step_parallel_kernel(float *q, float *k, const float *v,
                                                 const float *decay, const float *beta, float *state,
                                                 float *out, int n_v_heads, int head_v, int head_k,
                                                 float eps) {
    const int h = static_cast<int>(blockIdx.x);
    const int j = static_cast<int>(threadIdx.x);
    if (h >= n_v_heads) {
        return;
    }
    __shared__ float smem[256];
    float *q_h = q + h * head_k;
    float *k_h = k + h * head_k;
    const float *v_h = v + h * head_v;
    float *state_h = state + static_cast<int>(static_cast<long long>(h) * head_v * head_k);
    float *out_h = out + h * head_v;

    const float qv0 = (j < head_k) ? q_h[j] : 0.f;
    const float kv0 = (j < head_k) ? k_h[j] : 0.f;
    const float ss_q = gdn_block_sum(qv0 * qv0, smem);
    const float ss_k = gdn_block_sum(kv0 * kv0, smem);
    const float inv_q = rsqrtf(ss_q + eps);
    const float inv_k = rsqrtf(ss_k + eps);
    const float q_scale = rsqrtf(static_cast<float>(head_k));
    const float qv = qv0 * inv_q * q_scale;
    const float kv = kv0 * inv_k;
    if (j < head_k) {
        q_h[j] = qv;
        k_h[j] = kv;
    }
    __syncthreads();

    const float g = decay[h];
    const float b = beta[h];
    for (int i = 0; i < head_v; ++i) {
        float *row = state_h + i * head_k;
        const float rj0 = (j < head_k) ? row[j] * g : 0.f;
        const float acc = gdn_block_sum(rj0 * kv, smem);
        const float delta = b * (v_h[i] - acc);
        const float rj = rj0 + delta * kv;
        if (j < head_k) {
            row[j] = rj;
        }
        const float y = gdn_block_sum(rj * qv, smem);
        if (j == 0) {
            out_h[i] = y;
        }
    }
}

/* In-place per-head L2 norm of q (times 1/sqrt(head_k)) and k: one warp per head. */
__global__ void gdn_qk_l2norm_kernel(float *q, float *k, int n_heads, int head_k, float eps) {
    const int h = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (h >= n_heads) {
        return;
    }
    float *q_h = q + static_cast<size_t>(h) * head_k;
    float *k_h = k + static_cast<size_t>(h) * head_k;
    float ss_q = 0.f, ss_k = 0.f;
    for (int j = lane; j < head_k; j += 32) {
        ss_q += q_h[j] * q_h[j];
        ss_k += k_h[j] * k_h[j];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss_q += __shfl_xor(ss_q, off, 32);
        ss_k += __shfl_xor(ss_k, off, 32);
    }
    const float sq = rsqrtf(ss_q + eps) * rsqrtf(static_cast<float>(head_k));
    const float sk = rsqrtf(ss_k + eps);
    for (int j = lane; j < head_k; j += 32) {
        q_h[j] *= sq;
        k_h[j] *= sk;
    }
}

/* Delta-rule state update on already-normalized q/k. One warp per state row
 * (each lane holds NJ = head_k / 32 contiguous columns), reductions are warp
 * shuffles, and a head's rows spread over several blocks. The block-per-head
 * kernel walked all head_v rows with two block-wide reductions each (~768
 * barriers per call on n_v_heads blocks) and was pure latency. */
constexpr int GDN_ROWS_PER_WARP = 4;
constexpr int GDN_WARPS = 4;
template <int NJ>
__global__ void gated_delta_step_warp_kernel(const float *q, const float *k, const float *v,
                                             const float *decay, const float *beta, float *state,
                                             float *out, int n_v_heads, int head_v, int head_k) {
    const int h = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    if (h >= n_v_heads) {
        return;
    }
    const float *q_h = q + static_cast<size_t>(h) * head_k + lane * NJ;
    const float *k_h = k + static_cast<size_t>(h) * head_k + lane * NJ;
    float qv[NJ], kv[NJ];
#pragma unroll
    for (int t = 0; t < NJ; ++t) {
        qv[t] = q_h[t];
        kv[t] = k_h[t];
    }
    const float g = decay[h];
    const float b = beta[h];
    const int row0 = (static_cast<int>(blockIdx.y) * GDN_WARPS + warp) * GDN_ROWS_PER_WARP;
#pragma unroll
    for (int r = 0; r < GDN_ROWS_PER_WARP; ++r) {
        const int i = row0 + r;
        if (i >= head_v) {
            break;
        }
        float *row = state + (static_cast<size_t>(h) * head_v + i) * head_k + lane * NJ;
        float s[NJ];
        float acc = 0.f;
#pragma unroll
        for (int t = 0; t < NJ; ++t) {
            s[t] = row[t] * g;
            acc += s[t] * kv[t];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_xor(acc, off, 32);
        }
        const float delta = b * (v[static_cast<size_t>(h) * head_v + i] - acc);
        float y = 0.f;
#pragma unroll
        for (int t = 0; t < NJ; ++t) {
            s[t] += delta * kv[t];
            row[t] = s[t];
            y += s[t] * qv[t];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            y += __shfl_xor(y, off, 32);
        }
        if (lane == 0) {
            out[static_cast<size_t>(h) * head_v + i] = y;
        }
    }
}

__global__ void gated_delta_step_kernel(float *q, float *k, const float *v, const float *decay,
                                        const float *beta, float *state, float *out, int n_v_heads,
                                        int head_v, int head_k, float eps) {
    const int h = static_cast<int>(blockIdx.x);
    if (h >= n_v_heads || threadIdx.x != 0) {
        return;
    }
    float *q_h = q + h * head_k;
    float *k_h = k + h * head_k;
    const float *v_h = v + h * head_v;
    float *state_h = state + h * head_v * head_k;
    float *out_h = out + h * head_v;

    float ss_q = 0.0f;
    float ss_k = 0.0f;
    for (int j = 0; j < head_k; ++j) {
        ss_q += q_h[j] * q_h[j];
        ss_k += k_h[j] * k_h[j];
    }
    const float inv_q = rsqrtf(ss_q + eps);
    const float inv_k = rsqrtf(ss_k + eps);
    const float q_scale = rsqrtf(static_cast<float>(head_k));
    for (int j = 0; j < head_k; ++j) {
        q_h[j] *= inv_q * q_scale;
        k_h[j] *= inv_k;
    }

    const float g = decay[h];
    const float b = beta[h];
    for (int i = 0; i < head_v; ++i) {
        float *row = state_h + i * head_k;
        float acc = 0.0f;
        for (int j = 0; j < head_k; ++j) {
            row[j] *= g;
            acc += row[j] * k_h[j];
        }
        const float delta = b * (v_h[i] - acc);
        float y = 0.0f;
        for (int j = 0; j < head_k; ++j) {
            row[j] += delta * k_h[j];
            y += row[j] * q_h[j];
        }
        out_h[i] = y;
    }
}

__device__ void rope_pair(float *hvec, int i, int half, int pos, float theta) {
    const float freq = powf(theta, -static_cast<float>(i) / static_cast<float>(half));
    const float angle = static_cast<float>(pos) * freq;
    const float s = sinf(angle);
    const float c = cosf(angle);
    const float a = hvec[2 * i];
    const float b = hvec[2 * i + 1];
    hvec[2 * i] = a * c - b * s;
    hvec[2 * i + 1] = a * s + b * c;
}

__device__ void rope_back_pair(float *hvec, int i, int half, int pos, float theta) {
    const float freq = powf(theta, -static_cast<float>(i) / static_cast<float>(half));
    const float angle = static_cast<float>(pos) * freq;
    const float s = sinf(angle);
    const float c = cosf(angle);
    const float a = hvec[2 * i];
    const float b = hvec[2 * i + 1];
    hvec[2 * i] = a * c + b * s;
    hvec[2 * i + 1] = -a * s + b * c;
}

__global__ void rope_offset_rows_kernel(float *x, int n_tok, int n_heads, int head_dim, int n_rot,
                                        int offset, int start_pos, float theta, int back) {
    const int tok = static_cast<int>(blockIdx.y);
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int half = n_rot / 2;
    const int total = n_heads * half;
    if (tok >= n_tok || idx >= total || n_rot <= 0 || (n_rot % 2) != 0) {
        return;
    }
    const int h = idx / half;
    const int i = idx % half;
    float *hvec = x + (static_cast<size_t>(tok) * static_cast<size_t>(n_heads) +
                       static_cast<size_t>(h)) *
                          static_cast<size_t>(head_dim) +
                  static_cast<size_t>(offset);
    if (back) {
        rope_back_pair(hvec, i, half, start_pos + tok, theta);
    } else {
        rope_pair(hvec, i, half, start_pos + tok, theta);
    }
}

__global__ void rope_kernel(float *x, int n_heads, int head_dim, int pos, float theta) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int half = head_dim / 2;
    const int total = n_heads * half;
    if (idx >= total || head_dim % 2 != 0) {
        return;
    }
    rope_pair(x + (idx / half) * head_dim, idx % half, half, pos, theta);
}

__global__ void rope_qk_kernel(float *q, float *k, int n_q_heads, int n_k_heads, int head_dim,
                               int start_pos, float theta) {
    const int tok = static_cast<int>(blockIdx.y);
    const int pos = start_pos + tok;
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int half = head_dim / 2;
    const int q_pairs = n_q_heads * half;
    const int total = q_pairs + n_k_heads * half;
    if (idx >= total || head_dim % 2 != 0) {
        return;
    }
    q += static_cast<size_t>(tok) * static_cast<size_t>(n_q_heads) * static_cast<size_t>(head_dim);
    k += static_cast<size_t>(tok) * static_cast<size_t>(n_k_heads) * static_cast<size_t>(head_dim);
    if (idx < q_pairs) {
        rope_pair(q + (idx / half) * head_dim, idx % half, half, pos, theta);
    } else {
        const int j = idx - q_pairs;
        rope_pair(k + (j / half) * head_dim, j % half, half, pos, theta);
    }
}

__global__ void silu_mul_kernel(float *gate, const float *up, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    const float g = gate[idx];
    gate[idx] = (g / (1.0f + expf(-g))) * up[idx];
}

__global__ void gelu_mul_kernel(float *gate, const float *up, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    const float g = gate[idx];
    /* Exact GELU via erff — matches allpaca-cpu::gelu. */
    const float gelu = 0.5f * g * (1.0f + erff(g * 0.7071067811865476f));
    gate[idx] = gelu * up[idx];
}

__global__ void relu_sqr_kernel(float *x, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    const float v = x[idx];
    x[idx] = v > 0.0f ? v * v : 0.0f;
}

__global__ void scale_heads_sigmoid_kernel(float *attn, const float *scores, int n_tok,
                                           int n_heads, int head_dim) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int n = n_tok * n_heads * head_dim;
    if (idx >= n) {
        return;
    }
    const int head_elems = n_heads * head_dim;
    const int tok = idx / head_elems;
    const int within = idx % head_elems;
    const int h = within / head_dim;
    const float s = scores[tok * n_heads + h];
    attn[idx] *= 1.0f / (1.0f + expf(-s));
}

__global__ void deinterleave_q_gate_kernel(const float *q_full, float *q, float *gate, int n_heads,
                                           int head_dim) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int n = n_heads * head_dim;
    if (idx >= n) {
        return;
    }
    const int h = idx / head_dim;
    const int d = idx % head_dim;
    const int src = h * 2 * head_dim + d;
    q[idx] = q_full[src];
    gate[idx] = q_full[src + head_dim];
}

__global__ void sigmoid_mul_kernel(float *x, const float *gate, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    const float g = gate[idx];
    x[idx] *= 1.0f / (1.0f + expf(-g));
}

__global__ void rmsnorm_heads_kernel(float *x, const float *weight, int head_dim, float eps) {
    // One block per head; x row at blockIdx.x.
    float *row = x + static_cast<size_t>(blockIdx.x) * static_cast<size_t>(head_dim);
    __shared__ float shared[256];
    float sum = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < head_dim; i += static_cast<int>(blockDim.x)) {
        const float v = row[i];
        sum += v * v;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv = rsqrtf(shared[0] / static_cast<float>(head_dim) + eps);
    for (int i = static_cast<int>(threadIdx.x); i < head_dim; i += static_cast<int>(blockDim.x)) {
        row[i] = row[i] * inv * weight[i];
    }
}

extern "C" int allpaca_rmsnorm_rows(const float *x, const float *weight, float *out, int32_t rows,
                                    int32_t n, float eps, allpaca_stream_t stream) {
    if (!x || !weight || !out || rows <= 0 || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(rmsnorm_kernel, dim3(static_cast<unsigned>(rows)), dim3(256), 0,
                       as_stream(stream), x, weight, out, n, eps);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_rmsnorm(const float *x, const float *weight, float *out, int32_t n, float eps,
                               allpaca_stream_t stream) {
    return allpaca_rmsnorm_rows(x, weight, out, 1, n, eps, stream);
}

extern "C" int allpaca_layernorm_rows(const float *x, const float *weight, const float *bias,
                                      float *out, int32_t rows, int32_t n, float eps,
                                      allpaca_stream_t stream) {
    if (!x || !weight || !bias || !out || rows <= 0 || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(layernorm_kernel, dim3(static_cast<unsigned>(rows)), dim3(256), 0,
                       as_stream(stream), x, weight, bias, out, n, eps);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_residual_rmsnorm(float *x, const float *residual, const float *weight,
                                        float *out, int32_t n, float eps, allpaca_stream_t stream) {
    if (!x || !residual || !weight || !out || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(residual_rmsnorm_kernel, dim3(1), dim3(256), 0, as_stream(stream), x, residual,
                       weight, out, n, eps);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_gated_delta_step(float *q, float *k, const float *v, const float *decay,
                                        const float *beta, float *state, float *out,
                                        int32_t n_v_heads, int32_t head_v, int32_t head_k, float eps,
                                        allpaca_stream_t stream) {
    if (!q || !k || !v || !decay || !beta || !state || !out || n_v_heads <= 0 || head_v <= 0 ||
        head_k <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int nj = head_k % 32 == 0 ? head_k / 32 : 0;
    if (nj == 1 || nj == 2 || nj == 4 || nj == 8) {
        hipStream_t s = as_stream(stream);
        hipLaunchKernelGGL(gdn_qk_l2norm_kernel, dim3(static_cast<unsigned>(n_v_heads)), dim3(32), 0,
                           s, q, k, n_v_heads, head_k, eps);
        constexpr int rows_per_block = GDN_WARPS * GDN_ROWS_PER_WARP;
        const dim3 grid(static_cast<unsigned>(n_v_heads),
                        static_cast<unsigned>((head_v + rows_per_block - 1) / rows_per_block));
        const dim3 block(32 * GDN_WARPS);
#define GDN_WARP_LAUNCH(NJV)                                                                       \
    hipLaunchKernelGGL(gated_delta_step_warp_kernel<NJV>, grid, block, 0, s, q, k, v, decay, beta, \
                       state, out, n_v_heads, head_v, head_k)
        if (nj == 1) {
            GDN_WARP_LAUNCH(1);
        } else if (nj == 2) {
            GDN_WARP_LAUNCH(2);
        } else if (nj == 4) {
            GDN_WARP_LAUNCH(4);
        } else {
            GDN_WARP_LAUNCH(8);
        }
#undef GDN_WARP_LAUNCH
        return hip_status(hipGetLastError());
    }
    const bool parallel = head_k > 0 && head_k <= 256 && (head_k & (head_k - 1)) == 0;
    if (parallel) {
        hipLaunchKernelGGL(gated_delta_step_parallel_kernel, dim3(static_cast<unsigned>(n_v_heads)),
                           dim3(static_cast<unsigned>(head_k)), 0, as_stream(stream), q, k, v, decay,
                           beta, state, out, n_v_heads, head_v, head_k, eps);
    } else {
        hipLaunchKernelGGL(gated_delta_step_kernel, dim3(static_cast<unsigned>(n_v_heads)), dim3(1),
                           0, as_stream(stream), q, k, v, decay, beta, state, out, n_v_heads, head_v,
                           head_k, eps);
    }
    return hip_status(hipGetLastError());
}

__device__ inline float gdn_silu(float x) { return x / (1.0f + expf(-x)); }

__device__ inline float gdn_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

__global__ void gdn_conv1d_compute_kernel(const float *x, const float *kernel,
                                          const float *conv_state, float *out, int qkv_dim,
                                          int conv_kernel) {
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= qkv_dim) {
        return;
    }
    const int hist = conv_kernel - 1;
    float acc = 0.0f;
    for (int i = 0; i < hist; ++i) {
        acc += kernel[i + c * conv_kernel] * conv_state[i * qkv_dim + c];
    }
    acc += kernel[(conv_kernel - 1) + c * conv_kernel] * x[c];
    out[c] = gdn_silu(acc);
}

__global__ void gdn_conv1d_shift_kernel(const float *x, float *conv_state, int qkv_dim,
                                        int conv_kernel) {
    const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= qkv_dim) {
        return;
    }
    const int hist = conv_kernel - 1;
    if (hist <= 0) {
        return;
    }
    for (int i = 0; i < hist - 1; ++i) {
        conv_state[i * qkv_dim + c] = conv_state[(i + 1) * qkv_dim + c];
    }
    conv_state[(hist - 1) * qkv_dim + c] = x[c];
}

__global__ void expand_k_heads_kernel(const float *src, float *dst, int n_k, int n_v, int head_k) {
    const int vh = static_cast<int>(blockIdx.x);
    const int j = static_cast<int>(threadIdx.x);
    if (vh >= n_v || j >= head_k) {
        return;
    }
    const int kh = vh % n_k;
    dst[vh * head_k + j] = src[kh * head_k + j];
}

__global__ void gdn_decay_beta_kernel(const float *alpha, const float *dt_bias, const float *a,
                                      float *decay, float *beta, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    decay[i] = expf(gdn_softplus(alpha[i] + dt_bias[i]) * a[i]);
    beta[i] = 1.0f / (1.0f + expf(-beta[i]));
}

extern "C" int allpaca_gdn_conv1d_step(const float *x, const float *kernel, float *conv_state,
                                       float *out, int32_t qkv_dim, int32_t conv_kernel,
                                       allpaca_stream_t stream) {
    if (!x || !kernel || !conv_state || !out || qkv_dim <= 0 || conv_kernel <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (qkv_dim + threads - 1) / threads;
    hipLaunchKernelGGL(gdn_conv1d_compute_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), x, kernel,
                       conv_state, out, qkv_dim, conv_kernel);
    int rc = hip_status(hipGetLastError());
    if (rc != ALLPACA_OK) {
        return rc;
    }
    hipLaunchKernelGGL(gdn_conv1d_shift_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), x, conv_state,
                       qkv_dim, conv_kernel);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_expand_k_heads(const float *src, float *dst, int32_t n_k, int32_t n_v,
                                      int32_t head_k, allpaca_stream_t stream) {
    if (!src || !dst || n_k <= 0 || n_v <= 0 || head_k <= 0 || (n_v % n_k) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(expand_k_heads_kernel, dim3(static_cast<unsigned>(n_v)),
                       dim3(static_cast<unsigned>(head_k)), 0, as_stream(stream), src, dst, n_k, n_v,
                       head_k);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_gdn_decay_beta(const float *alpha, const float *dt_bias, const float *a,
                                      float *decay, float *beta, int32_t n, allpaca_stream_t stream) {
    if (!alpha || !dt_bias || !a || !decay || !beta || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(gdn_decay_beta_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), alpha, dt_bias, a,
                       decay, beta, n);
    return hip_status(hipGetLastError());
}

__global__ void mamba2_ssd_kernel(const float *z, const float *xbc, const float *dt,
                                  const float *a, const float *d, const float *dt_bias,
                                  const float *norm, float *state, float *out, int d_inner,
                                  int d_state, int n_heads, int n_groups, float eps) {
    const int group = static_cast<int>(blockIdx.x);
    const int group_width = d_inner / n_groups;
    const int head_dim = d_inner / n_heads;
    const int heads_per_group = n_heads / n_groups;
    const int group_start = group * group_width;
    const int b_off = d_inner + group * d_state;
    const int c_off = d_inner + n_groups * d_state + group * d_state;
    __shared__ float sum_sq;
    if (threadIdx.x == 0) {
        sum_sq = 0.0f;
    }
    __syncthreads();
    for (int local = static_cast<int>(threadIdx.x); local < group_width;
         local += static_cast<int>(blockDim.x)) {
        const int channel = group_start + local;
        const int head = group * heads_per_group + local / head_dim;
        const float delta = gdn_softplus(dt[head] + dt_bias[head]);
        const float decay = expf(delta * a[head]);
        float acc = 0.0f;
        float *row = state + channel * d_state;
        for (int n = 0; n < d_state; ++n) {
            const float value = decay * row[n] + delta * xbc[b_off + n] * xbc[channel];
            row[n] = value;
            acc += xbc[c_off + n] * value;
        }
        const float gated = (acc + d[head] * xbc[channel]) * gdn_silu(z[channel]);
        out[channel] = gated;
        atomicAdd(&sum_sq, gated * gated);
    }
    __syncthreads();
    const float inv = rsqrtf(sum_sq / static_cast<float>(group_width) + eps);
    for (int local = static_cast<int>(threadIdx.x); local < group_width;
         local += static_cast<int>(blockDim.x)) {
        const int channel = group_start + local;
        out[channel] *= inv * norm[channel];
    }
}

__global__ void mamba2_conv_compute_kernel(const float *x, const float *kernel,
                                           const float *bias, const float *conv_state,
                                           float *out, int conv_dim, int conv_kernel) {
    const int channel = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= conv_dim) {
        return;
    }
    const int hist = conv_kernel - 1;
    float acc = bias ? bias[channel] : 0.0f;
    for (int k = 0; k < hist; ++k) {
        acc += kernel[k + channel * conv_kernel] * conv_state[k * conv_dim + channel];
    }
    acc += kernel[hist + channel * conv_kernel] * x[channel];
    out[channel] = gdn_silu(acc);
}

extern "C" int allpaca_mamba2_conv_step(const float *xbc, const float *kernel,
                                        const float *bias, float *conv_state, float *out,
                                        int32_t conv_dim, int32_t conv_kernel,
                                        allpaca_stream_t stream) {
    if (!xbc || !kernel || !conv_state || !out || conv_dim <= 0 || conv_kernel <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (conv_dim + threads - 1) / threads;
    hipLaunchKernelGGL(mamba2_conv_compute_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), xbc, kernel,
                       bias, conv_state, out, conv_dim, conv_kernel);
    int rc = hip_status(hipGetLastError());
    if (rc != ALLPACA_OK) {
        return rc;
    }
    hipLaunchKernelGGL(gdn_conv1d_shift_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), xbc, conv_state,
                       conv_dim, conv_kernel);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_mamba2_ssd_step(const float *z, const float *xbc, const float *dt,
                                       const float *a, const float *d, const float *dt_bias,
                                       const float *norm, float *state, float *out,
                                       int32_t d_inner, int32_t d_state, int32_t n_heads,
                                       int32_t n_groups, float eps, allpaca_stream_t stream) {
    if (!z || !xbc || !dt || !a || !d || !dt_bias || !norm || !state || !out ||
        d_inner <= 0 || d_state <= 0 || n_heads <= 0 || n_groups <= 0 ||
        d_inner % n_heads != 0 || n_heads % n_groups != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(mamba2_ssd_kernel, dim3(static_cast<unsigned>(n_groups)), dim3(256), 0,
                       as_stream(stream), z, xbc, dt, a, d, dt_bias, norm, state, out, d_inner,
                       d_state, n_heads, n_groups, eps);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_rope_qk_rows(float *q, float *k, int32_t n_tok, int32_t n_q_heads,
                                    int32_t n_k_heads, int32_t head_dim, int32_t start_pos,
                                    float theta, allpaca_stream_t stream) {
    if (!q || !k || n_tok <= 0 || n_q_heads <= 0 || n_k_heads <= 0 || head_dim <= 0 ||
        (head_dim % 2) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int total = (n_q_heads + n_k_heads) * (head_dim / 2);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(rope_qk_kernel, dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(n_tok)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), q, k, n_q_heads,
                       n_k_heads, head_dim, start_pos, theta);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_rope_qk(float *q, float *k, int32_t n_q_heads, int32_t n_k_heads,
                               int32_t head_dim, int32_t pos, float theta, allpaca_stream_t stream) {
    return allpaca_rope_qk_rows(q, k, 1, n_q_heads, n_k_heads, head_dim, pos, theta, stream);
}

extern "C" int allpaca_rope_offset_rows(float *x, int32_t n_tok, int32_t n_heads, int32_t head_dim,
                                        int32_t n_rot, int32_t offset, int32_t start_pos,
                                        float theta, int32_t back, allpaca_stream_t stream) {
    if (!x || n_tok <= 0 || n_heads <= 0 || head_dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    int rot = n_rot;
    int off = offset;
    if (rot == 0) {
        rot = head_dim;
        off = 0;
    }
    if (rot < 0 || off < 0 || (rot % 2) != 0 || off + rot > head_dim) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int total = n_heads * (rot / 2);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(rope_offset_rows_kernel,
                       dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(n_tok)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), x, n_tok, n_heads,
                       head_dim, rot, off, start_pos, theta, back);
    return hip_status(hipGetLastError());
}

/* NeoX pair on first n_rot dims; optional IMRoPE sector angles (matches CPU rope_imrope_neox). */
__device__ void rope_imrope_pair(float *hvec, int i, int half, int n_rot, int pos, float theta,
                                int s0, int s1, int s2, int s3) {
    const int sect_dims = s0 + s1 + s2 + s3;
    const bool use_imrope = sect_dims > 0 && s0 > 0 && s1 > 0;
    const float theta_scale = powf(theta, -2.0f / static_cast<float>(n_rot));
    float theta_t = static_cast<float>(pos);
    float theta_h = static_cast<float>(pos);
    float theta_w = static_cast<float>(pos);
    float theta_e = 0.0f;
    float angle = 0.0f;
    for (int j = 0; j <= i; ++j) {
        if (use_imrope) {
            const int sector = j % sect_dims;
            if (sector % 3 == 1 && sector < 3 * s1) {
                angle = theta_h;
            } else if (sector % 3 == 2 && sector < 3 * s2) {
                angle = theta_w;
            } else if (sector % 3 == 0 && sector < 3 * s0) {
                angle = theta_t;
            } else {
                angle = theta_e;
            }
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        } else {
            angle = theta_t;
            theta_t *= theta_scale;
        }
    }
    const float s = sinf(angle);
    const float c = cosf(angle);
    const float a = hvec[i];
    const float b = hvec[i + half];
    hvec[i] = a * c - b * s;
    hvec[i + half] = a * s + b * c;
}

__global__ void rope_imrope_qk_kernel(float *q, float *k, int n_q_heads, int n_k_heads, int head_dim,
                                      int n_rot, int pos, float theta, int s0, int s1, int s2,
                                      int s3) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int half = n_rot / 2;
    const int q_pairs = n_q_heads * half;
    const int total = q_pairs + n_k_heads * half;
    if (idx >= total || n_rot <= 0 || n_rot > head_dim || (n_rot & 1) != 0) {
        return;
    }
    if (idx < q_pairs) {
        const int h = idx / half;
        const int i = idx % half;
        rope_imrope_pair(q + h * head_dim, i, half, n_rot, pos, theta, s0, s1, s2, s3);
    } else {
        const int j = idx - q_pairs;
        const int h = j / half;
        const int i = j % half;
        rope_imrope_pair(k + h * head_dim, i, half, n_rot, pos, theta, s0, s1, s2, s3);
    }
}

extern "C" int allpaca_rope_imrope_qk(float *q, float *k, int32_t n_q_heads, int32_t n_k_heads,
                                      int32_t head_dim, int32_t n_rot, int32_t pos, float theta,
                                      const int32_t sections[4], allpaca_stream_t stream) {
    if (!q || !k || !sections || n_q_heads <= 0 || n_k_heads <= 0 || head_dim <= 0 || n_rot <= 0 ||
        n_rot > head_dim || (n_rot & 1) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int half = n_rot / 2;
    const int total = (n_q_heads + n_k_heads) * half;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(rope_imrope_qk_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), q, k, n_q_heads,
                       n_k_heads, head_dim, n_rot, pos, theta, sections[0], sections[1], sections[2],
                       sections[3]);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_rope(float *x, int32_t n_heads, int32_t head_dim, int32_t pos, float theta,
                            allpaca_stream_t stream) {
    if (!x || n_heads <= 0 || head_dim <= 0 || (head_dim % 2) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int total = n_heads * (head_dim / 2);
    const int threads = 256;
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
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(silu_mul_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), gate, up, n);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_gelu_mul(float *gate, const float *up, int32_t n, allpaca_stream_t stream) {
    if (!gate || !up || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(gelu_mul_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), gate, up, n);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_relu_sqr(float *x, int32_t n, allpaca_stream_t stream) {
    if (!x || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(relu_sqr_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), x, n);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_deinterleave_q_gate(const float *q_full, float *q, float *gate,
                                           int32_t n_heads, int32_t head_dim,
                                           allpaca_stream_t stream) {
    if (!q_full || !q || !gate || n_heads <= 0 || head_dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int n = n_heads * head_dim;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(deinterleave_q_gate_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), q_full, q, gate,
                       n_heads, head_dim);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_sigmoid_mul(float *x, const float *gate, int32_t n,
                                   allpaca_stream_t stream) {
    if (!x || !gate || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(sigmoid_mul_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), x, gate, n);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_scale_heads_sigmoid(float *attn, const float *scores, int32_t n_tok,
                                           int32_t n_heads, int32_t head_dim,
                                           allpaca_stream_t stream) {
    if (!attn || !scores || n_tok <= 0 || n_heads <= 0 || head_dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int n = n_tok * n_heads * head_dim;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(scale_heads_sigmoid_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), attn, scores,
                       n_tok, n_heads, head_dim);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_rmsnorm_heads(float *x, const float *weight, int32_t n_heads,
                                     int32_t head_dim, float eps, allpaca_stream_t stream) {
    if (!x || !weight || n_heads <= 0 || head_dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(rmsnorm_heads_kernel, dim3(static_cast<unsigned>(n_heads)), dim3(256), 0,
                       as_stream(stream), x, weight, head_dim, eps);
    return hip_status(hipGetLastError());
}

__global__ void embed_gather_kernel(const float *table, int token, float *out, int dim) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= dim) {
        return;
    }
    out[idx] = table[token * dim + idx];
}

__global__ void embed_gather_rows_kernel(const float *table, const int32_t *tokens, float *out,
                                         int n_tok, int dim) {
    const int row = static_cast<int>(blockIdx.y);
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n_tok || idx >= dim) {
        return;
    }
    const int token = tokens[row];
    out[static_cast<size_t>(row) * static_cast<size_t>(dim) + idx] =
        table[static_cast<size_t>(token) * static_cast<size_t>(dim) + idx];
}

__global__ void residual_add_kernel(float *x, const float *residual, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    x[idx] += residual[idx];
}

__global__ void scatter_add_scale_rows_kernel(const float *src, const int32_t *idx, const float *w,
                                              float *acc, int m, int dim, int n_tok) {
    const int row = static_cast<int>(blockIdx.y);
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= m || j >= dim) {
        return;
    }
    const int t = idx[row];
    if (t < 0 || t >= n_tok) {
        return;
    }
    atomicAdd(&acc[static_cast<size_t>(t) * static_cast<size_t>(dim) + static_cast<size_t>(j)],
              src[static_cast<size_t>(row) * static_cast<size_t>(dim) + static_cast<size_t>(j)] *
                  w[row]);
}

/* curs[t, i] = sum_s packed[t, s, i] * pre[t, s] */
__global__ void hc_mix_pre_kernel(const float *packed, const float *pre, float *curs, int n_tok,
                                  int hc, int dim) {
    const int t = static_cast<int>(blockIdx.y);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= n_tok || i >= dim) {
        return;
    }
    float acc = 0.0f;
    const float *pre_t = pre + static_cast<size_t>(t) * static_cast<size_t>(hc);
    for (int s = 0; s < hc; ++s) {
        const size_t off = (static_cast<size_t>(t) * static_cast<size_t>(hc) + static_cast<size_t>(s)) *
                               static_cast<size_t>(dim) +
                           static_cast<size_t>(i);
        acc += packed[off] * pre_t[s];
    }
    curs[static_cast<size_t>(t) * static_cast<size_t>(dim) + static_cast<size_t>(i)] = acc;
}

/* out[t, dst, i] = x[t, i] * post[t, dst] + sum_src residual[t, src, i] * comb[t, src, dst] */
__global__ void hc_post_kernel(const float *x, const float *residual, const float *post,
                               const float *comb, float *out, int n_tok, int hc, int dim) {
    const int t = static_cast<int>(blockIdx.y);
    const int dst = static_cast<int>(blockIdx.z);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= n_tok || dst >= hc || i >= dim) {
        return;
    }
    const size_t td = static_cast<size_t>(t) * static_cast<size_t>(dim) + static_cast<size_t>(i);
    float acc = x[td] * post[static_cast<size_t>(t) * static_cast<size_t>(hc) + static_cast<size_t>(dst)];
    const float *comb_t = comb + static_cast<size_t>(t) * static_cast<size_t>(hc) * static_cast<size_t>(hc);
    for (int src = 0; src < hc; ++src) {
        const size_t roff =
            (static_cast<size_t>(t) * static_cast<size_t>(hc) + static_cast<size_t>(src)) *
                static_cast<size_t>(dim) +
            static_cast<size_t>(i);
        acc += residual[roff] * comb_t[static_cast<size_t>(dst) + static_cast<size_t>(src) * static_cast<size_t>(hc)];
    }
    const size_t ooff = (static_cast<size_t>(t) * static_cast<size_t>(hc) + static_cast<size_t>(dst)) *
                            static_cast<size_t>(dim) +
                        static_cast<size_t>(i);
    out[ooff] = acc;
}

__global__ void scale_f32_kernel(float *x, float scale, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    x[idx] *= scale;
}

extern "C" int allpaca_embed_gather(const float *table, int32_t token, float *out, int32_t dim,
                                    allpaca_stream_t stream) {
    if (!table || !out || dim <= 0 || token < 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    hipLaunchKernelGGL(embed_gather_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), table, token, out,
                       dim);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_embed_gather_rows(const float *table, const int32_t *tokens, float *out,
                                         int32_t n_tok, int32_t dim, allpaca_stream_t stream) {
    if (!table || !tokens || !out || n_tok <= 0 || dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    hipLaunchKernelGGL(embed_gather_rows_kernel,
                       dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(n_tok)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), table, tokens, out,
                       n_tok, dim);
    return hip_status(hipGetLastError());
}

__device__ float allpaca_bf16_bits_to_f32(uint16_t h) {
    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = static_cast<uint32_t>(h) << 16;
    return conv.f;
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

__global__ void embed_gather_f16_kernel(const uint16_t *table, int token, float *out, int dim) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= dim) {
        return;
    }
    out[idx] = allpaca_f16_bits_to_f32(table[static_cast<size_t>(token) * static_cast<size_t>(dim) +
                                            static_cast<size_t>(idx)]);
}

__global__ void embed_gather_f16_rows_kernel(const uint16_t *table, const int32_t *tokens, float *out,
                                             int n_tok, int dim) {
    const int row = static_cast<int>(blockIdx.y);
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n_tok || idx >= dim) {
        return;
    }
    const int token = tokens[row];
    out[static_cast<size_t>(row) * static_cast<size_t>(dim) + idx] = allpaca_f16_bits_to_f32(
        table[static_cast<size_t>(token) * static_cast<size_t>(dim) + static_cast<size_t>(idx)]);
}

extern "C" int allpaca_embed_gather_f16(const uint16_t *table, int32_t token, float *out, int32_t dim,
                                        allpaca_stream_t stream) {
    if (!table || !out || dim <= 0 || token < 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    hipLaunchKernelGGL(embed_gather_f16_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), table, token, out,
                       dim);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_embed_gather_f16_rows(const uint16_t *table, const int32_t *tokens, float *out,
                                             int32_t n_tok, int32_t dim, allpaca_stream_t stream) {
    if (!table || !tokens || !out || n_tok <= 0 || dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    hipLaunchKernelGGL(embed_gather_f16_rows_kernel,
                       dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(n_tok)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), table, tokens, out,
                       n_tok, dim);
    return hip_status(hipGetLastError());
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

__global__ void embed_gather_q8_0_rows_kernel(const uint8_t *table, const int32_t *tokens, float *out,
                                              int n_tok, int dim) {
    const int row = static_cast<int>(blockIdx.y);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n_tok || i >= dim) {
        return;
    }
    const int token = tokens[row];
    const int blocks_per_row = (dim + 31) / 32;
    const int b = i / 32;
    const int j = i % 32;
    const uint8_t *blk =
        table + (static_cast<size_t>(token) * static_cast<size_t>(blocks_per_row) +
                 static_cast<size_t>(b)) *
                    Q8_STAGED_STRIDE;
    const float d = *reinterpret_cast<const float *>(blk);
    out[static_cast<size_t>(row) * static_cast<size_t>(dim) + i] =
        d * static_cast<float>(static_cast<int8_t>(blk[Q8_STAGED_QS + j]));
}

extern "C" int allpaca_dequant_q8_0(const uint8_t *qblocks, float *out, int32_t n,
                                    allpaca_stream_t stream) {
    if (!qblocks || !out || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    /* Same-stream order keeps workspace reuse after in-flight readers. */
    clear_last_error();
    const int threads = 256;
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
    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    hipLaunchKernelGGL(embed_gather_q8_0_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), table, token, out,
                       dim);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_embed_gather_q8_0_rows(const uint8_t *table, const int32_t *tokens, float *out,
                                              int32_t n_tok, int32_t dim, allpaca_stream_t stream) {
    if (!table || !tokens || !out || n_tok <= 0 || dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    hipLaunchKernelGGL(embed_gather_q8_0_rows_kernel,
                       dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(n_tok)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), table, tokens, out,
                       n_tok, dim);
    return hip_status(hipGetLastError());
}

__global__ void embed_gather_rocmfp4_kernel(const uint8_t *table, int token, float *out, int dim) {
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
                    18u;
    if (j < 16) {
        out[i] = static_cast<float>(k_rocmfp4_codebook[blk[j] & 0x0F]) * rocmfp4_scale(blk[16]);
    } else {
        out[i] =
            static_cast<float>(k_rocmfp4_codebook[blk[j - 16] >> 4]) * rocmfp4_scale(blk[17]);
    }
}

__global__ void embed_gather_rocmfp4_rows_kernel(const uint8_t *table, const int32_t *tokens,
                                                 float *out, int n_tok, int dim) {
    const int row = static_cast<int>(blockIdx.y);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n_tok || i >= dim) {
        return;
    }
    const int token = tokens[row];
    const int blocks_per_row = (dim + 31) / 32;
    const int b = i / 32;
    const int j = i % 32;
    const uint8_t *blk =
        table + (static_cast<size_t>(token) * static_cast<size_t>(blocks_per_row) +
                 static_cast<size_t>(b)) *
                    18u;
    float v;
    if (j < 16) {
        v = static_cast<float>(k_rocmfp4_codebook[blk[j] & 0x0F]) * rocmfp4_scale(blk[16]);
    } else {
        v = static_cast<float>(k_rocmfp4_codebook[blk[j - 16] >> 4]) * rocmfp4_scale(blk[17]);
    }
    out[static_cast<size_t>(row) * static_cast<size_t>(dim) + i] = v;
}

extern "C" int allpaca_embed_gather_rocmfp4(const uint8_t *table, int32_t token, float *out,
                                            int32_t dim, allpaca_stream_t stream) {
    if (!table || !out || dim <= 0 || token < 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    hipLaunchKernelGGL(embed_gather_rocmfp4_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), table, token, out,
                       dim);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_embed_gather_rocmfp4_rows(const uint8_t *table, const int32_t *tokens,
                                                 float *out, int32_t n_tok, int32_t dim,
                                                 allpaca_stream_t stream) {
    if (!table || !tokens || !out || n_tok <= 0 || dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    hipLaunchKernelGGL(embed_gather_rocmfp4_rows_kernel,
                       dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(n_tok)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), table, tokens, out,
                       n_tok, dim);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_residual_add(float *x, const float *residual, int32_t n,
                                    allpaca_stream_t stream) {
    if (!x || !residual || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(residual_add_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), x, residual, n);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_scatter_add_scale_rows(const float *src, const int32_t *idx, const float *w,
                                              float *acc, int32_t m, int32_t dim, int32_t n_tok,
                                              allpaca_stream_t stream) {
    if (!src || !idx || !w || !acc || m <= 0 || dim <= 0 || n_tok <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    hipLaunchKernelGGL(scatter_add_scale_rows_kernel,
                       dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(m)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), src, idx, w, acc,
                       m, dim, n_tok);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_hc_mix_pre(const float *packed, const float *pre, float *curs, int32_t n_tok,
                                  int32_t hc, int32_t dim, allpaca_stream_t stream) {
    if (!packed || !pre || !curs || n_tok <= 0 || hc <= 0 || hc > 8 || dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int xblocks = (dim + threads - 1) / threads;
    hipLaunchKernelGGL(hc_mix_pre_kernel,
                       dim3(static_cast<unsigned>(xblocks), static_cast<unsigned>(n_tok)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), packed, pre, curs,
                       n_tok, hc, dim);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_hc_post(const float *x, const float *residual, const float *post,
                               const float *comb, float *out, int32_t n_tok, int32_t hc, int32_t dim,
                               allpaca_stream_t stream) {
    if (!x || !residual || !post || !comb || !out || n_tok <= 0 || hc <= 0 || hc > 8 || dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    if (residual == out) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int xblocks = (dim + threads - 1) / threads;
    hipLaunchKernelGGL(hc_post_kernel,
                       dim3(static_cast<unsigned>(xblocks), static_cast<unsigned>(n_tok),
                            static_cast<unsigned>(hc)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), x, residual, post,
                       comb, out, n_tok, hc, dim);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_scale_f32(float *x, int32_t n, float scale, allpaca_stream_t stream) {
    if (!x || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(scale_f32_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), x, scale, n);
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

extern "C" int allpaca_hip_device_arch(int32_t, char *, int32_t, int32_t *) {
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

extern "C" int allpaca_hip_memset(void *, int, size_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_free(void *) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_free_async(void *, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_host_malloc(void **, size_t) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_host_free(void *) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_host_register(void *, size_t, unsigned int) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_host_unregister(void *) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_memcpy(void *, const void *, size_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_memcpy_batch(void **, void **, size_t *, size_t, int32_t,
                                        allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_memset(void *, int32_t, size_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_has_memcpy_batch(void) { return 0; }

extern "C" int allpaca_hip_memcpy_peer(void *, int32_t, const void *, int32_t, size_t,
                                      allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_touch_resident(void *, size_t, int32_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_benchmark(int32_t, int32_t, size_t, double *, double *) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_mem_used(int32_t, uint64_t *) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_mem_info(int32_t, uint64_t *, uint64_t *) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_local_headroom(int32_t, uint64_t *) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_set_device(int32_t) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_stream_synchronize(allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_stream_create(allpaca_stream_t *) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_stream_destroy(allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_stream_wait_event(allpaca_stream_t, void *) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_event_create(void **) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_event_destroy(void *) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_event_record(void *, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_event_synchronize(void *) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

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

extern "C" int allpaca_gemm_nt_untiled(const float *, const float *, float *, int32_t, int32_t,
                                       int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_q8_0(const float *, const uint8_t *, float *, int32_t, int32_t,
                                    int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_q8_0_intdot(const float *, const uint8_t *, float *, int32_t, int32_t,
                                           int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_q4_k(const float *, const uint8_t *, float *, int32_t, int32_t,
                                    int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_q4_k_untiled(const float *, const uint8_t *, float *, int32_t,
                                            int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_tq2_0(const float *, const uint8_t *, float *, int32_t, int32_t,
                                     int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_tq1_0(const float *, const uint8_t *, float *, int32_t, int32_t,
                                     int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_rocmfp4(const float *, const uint8_t *, float *, int32_t, int32_t,
                                       int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_rocmfp4_untiled(const float *, const uint8_t *, float *, int32_t,
                                               int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_mxfp4(const float *, const uint8_t *, float *, int32_t, int32_t,
                                     int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_mxfp4_untiled(const float *, const uint8_t *, float *, int32_t,
                                             int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_mxfp4_swiglu_gate_up(const float *, const uint8_t *, const uint8_t *,
                                            float *, int32_t, int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_q6_k(const float *, const uint8_t *, float *, int32_t, int32_t,
                                    int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_q6_k_cols(const float *, const uint8_t *, float *, int32_t, int32_t,
                                         int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_q6_k_untiled(const float *, const uint8_t *, float *, int32_t,
                                            int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_q5_k(const float *, const uint8_t *, float *, int32_t, int32_t,
                                    int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_q3_k(const float *, const uint8_t *, float *, int32_t, int32_t,
                                    int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_q5_k_cols(const float *, const uint8_t *, float *, int32_t, int32_t,
                                         int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_q5_k_untiled(const float *, const uint8_t *, float *, int32_t,
                                            int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_f16(const float *, const uint16_t *, float *, int32_t, int32_t,
                                  int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_f16_untiled(const float *, const uint16_t *, float *, int32_t,
                                           int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_bf16(const float *, const uint16_t *, float *, int32_t, int32_t,
                                   int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_bf16_untiled(const float *, const uint16_t *, float *, int32_t,
                                            int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_attention(const float *, const float *, const float *, float *, int32_t,
                                 int32_t, int32_t, int32_t, int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_attention_sinks(const float *, const float *, const float *, float *,
                                      int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                                      const float *, allpaca_stream_t) {
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

extern "C" int allpaca_rmsnorm_rows(const float *, const float *, float *, int32_t, int32_t, float,
                                    allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_layernorm_rows(const float *, const float *, const float *, float *, int32_t,
                                      int32_t, float, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_rope(float *, int32_t, int32_t, int32_t, float, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_silu_mul(float *, const float *, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gelu_mul(float *, const float *, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_relu_sqr(float *, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_deinterleave_q_gate(const float *, float *, float *, int32_t, int32_t,
                                           allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_sigmoid_mul(float *, const float *, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_scale_heads_sigmoid(float *, const float *, int32_t, int32_t, int32_t,
                                           allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_rmsnorm_heads(float *, const float *, int32_t, int32_t, float,
                                     allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_embed_gather(const float *, int32_t, float *, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_embed_gather_rows(const float *, const int32_t *, float *, int32_t, int32_t,
                                         allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_embed_gather_f16(const uint16_t *, int32_t, float *, int32_t,
                                        allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_embed_gather_f16_rows(const uint16_t *, const int32_t *, float *, int32_t,
                                             int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_residual_add(float *, const float *, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_scatter_add_scale_rows(const float *, const int32_t *, const float *, float *,
                                              int32_t, int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hc_mix_pre(const float *, const float *, float *, int32_t, int32_t, int32_t,
                                  allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hc_post(const float *, const float *, const float *, const float *, float *,
                               int32_t, int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_scale_f32(float *, int32_t, float, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_residual_rmsnorm(float *, const float *, const float *, float *, int32_t,
                                        float, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gated_delta_step(float *, float *, const float *, const float *,
                                        const float *, float *, float *, int32_t, int32_t, int32_t,
                                        float, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gdn_conv1d_step(const float *, const float *, float *, float *, int32_t,
                                       int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_expand_k_heads(const float *, float *, int32_t, int32_t, int32_t,
                                      allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gdn_decay_beta(const float *, const float *, const float *, float *, float *,
                                      int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_mamba2_conv_step(const float *, const float *, const float *, float *,
                                         float *, int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_mamba2_ssd_step(const float *, const float *, const float *, const float *,
                                        const float *, const float *, const float *, float *,
                                        float *, int32_t, int32_t, int32_t, int32_t, float,
                                        allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_rope_qk(float *, float *, int32_t, int32_t, int32_t, int32_t, float,
                               allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_rope_qk_rows(float *, float *, int32_t, int32_t, int32_t, int32_t, int32_t,
                                    float, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_rope_offset_rows(float *, int32_t, int32_t, int32_t, int32_t, int32_t,
                                        int32_t, float, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_rope_imrope_qk(float *, float *, int32_t, int32_t, int32_t, int32_t, int32_t,
                                      float, const int32_t *, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_dequant_q8_0(const uint8_t *, float *, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_embed_gather_q8_0(const uint8_t *, int32_t, float *, int32_t,
                                         allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_embed_gather_q8_0_rows(const uint8_t *, const int32_t *, float *, int32_t,
                                              int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_embed_gather_rocmfp4(const uint8_t *, int32_t, float *, int32_t,
                                            allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_embed_gather_rocmfp4_rows(const uint8_t *, const int32_t *, float *, int32_t,
                                                 int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

#endif
