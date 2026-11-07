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

/* Register slots per attention-row lane in the fixed-size kernel (the flash
 * kernel, and the DPT = ATTN_MAX_DPT template instantiation). */
constexpr int ATTN_MAX_DPT = 16;

/* Attention-kernel template bucket for a head_dim: ceil(head_dim/32) rounded
 * up to a compiled instantiation. 0 is the per-row flash kernel (head_dim
 * outside 1..=32 * ATTN_MAX_DPT). Pure arithmetic, defined outside the HIP
 * gate so stub builds export the same value; it both picks the kernel and
 * sizes the chunk-partial scratch of a call. */
static constexpr int attn_dpt_bucket(int head_dim) {
    if (head_dim <= 0 || head_dim > 32 * ATTN_MAX_DPT) {
        return 0;
    }
    const int dpt = (head_dim + 31) / 32;
    return dpt <= 4 ? 4 : dpt <= 8 ? 8 : 16;
}

extern "C" int32_t allpaca_attn_dpt(int32_t head_dim) { return attn_dpt_bucket(head_dim); }

#if ALLPACA_HAS_HIP

#include <atomic>
#include <mutex>

/* Every kernel launch in this file goes through hipLaunchKernelGGL; count them
 * (process-wide, relaxed) so `/stats` can report launches per token. */
static std::atomic<uint64_t> g_kernel_launches{0};
/* Launches and device writes enqueued by the calling thread (see `KqActCopy`). */
static thread_local uint64_t t_device_writes = 0;
#undef hipLaunchKernelGGL
#define hipLaunchKernelGGL(kernelName, ...)                                  \
    do {                                                                     \
        hipLaunchKernelGGLInternal((kernelName), __VA_ARGS__);               \
        g_kernel_launches.fetch_add(1, std::memory_order_relaxed);           \
        ++t_device_writes;                                                   \
    } while (0)

/* Device writes that are not our kernel launches: copies, memsets, BLAS calls
 * and DirectStorage, plus frees and host registrations (they change what an
 * address holds). With the launch count this is an epoch that moves on every
 * enqueue that may change device memory, so a cached int8 activation is
 * reused only while it cannot be stale (see `KqActCopy`). The HIP runtime
 * calls that write are redirected to counted forms below the includes, so a
 * new one cannot skip the count; other writers call `note_device_write`.
 * Every count comes after its enqueue: a write another thread enqueues
 * behind a quantize then always moves the epoch after the quantize read it. */
static std::atomic<uint64_t> g_dev_writes{0};
static thread_local uint64_t t_act_quant_reuses = 0;
static inline void note_device_write() {
    g_dev_writes.fetch_add(1, std::memory_order_relaxed);
    ++t_device_writes;
}
static inline uint64_t device_write_epoch() {
    return g_kernel_launches.load(std::memory_order_relaxed) +
           g_dev_writes.load(std::memory_order_relaxed);
}

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
#include <filesystem>
#include <system_error>
#endif

/* Counted forms of every HIP runtime call that may change device memory or
 * what an address holds (see `note_device_write`). A copy to the host writes
 * no device memory and is not counted. */
template <typename Call> static inline auto counted_write(Call call) {
    auto r = call();
    note_device_write();
    return r;
}
static inline hipError_t counted_hip_memcpy(void *dst, const void *src, size_t bytes,
                                            hipMemcpyKind kind) {
    const hipError_t err = ::hipMemcpy(dst, src, bytes, kind);
    if (kind != hipMemcpyDeviceToHost) {
        note_device_write();
    }
    return err;
}
/* Debug (P21): ALLPACA_SYNC_COPY is a bit mask of copy kinds that wait for their
 * stream right after the call (1 host->device, 2 device->host, 4 device->device,
 * 8 any other); add 16 to limit it to the null stream, 32 to side streams, 64 to copies of 1 MiB or more, 128 to smaller ones.
 * 0 (unset) leaves every copy asynchronous. */
static inline unsigned debug_sync_copy_mask() {
    static const unsigned mask = []() -> unsigned {
        const char *v = std::getenv("ALLPACA_SYNC_COPY");
        return v ? static_cast<unsigned>(std::strtoul(v, nullptr, 10)) : 0u;
    }();
    return mask;
}
static inline hipError_t counted_hip_memcpy_async(void *dst, const void *src, size_t bytes,
                                                  hipMemcpyKind kind, hipStream_t stream) {
    hipError_t err = ::hipMemcpyAsync(dst, src, bytes, kind, stream);
    if (kind != hipMemcpyDeviceToHost) {
        note_device_write();
    }
    if (err == hipSuccess) {
        const unsigned m = debug_sync_copy_mask();
        if (m) {
            const unsigned bit = kind == hipMemcpyHostToDevice     ? 1u
                                 : kind == hipMemcpyDeviceToHost   ? 2u
                                 : kind == hipMemcpyDeviceToDevice ? 4u
                                                                    : 8u;
            const bool stream_ok = ((m & 16u) == 0 || stream == nullptr) &&
                                   ((m & 32u) == 0 || stream != nullptr);
            const bool size_ok = ((m & 64u) == 0 || bytes >= (1u << 20)) &&
                                 ((m & 128u) == 0 || bytes < (1u << 20));
            if ((m & bit) && stream_ok && size_ok) {
                err = ::hipStreamSynchronize(stream);
            }
        }
    }
    return err;
}
#define hipMemcpy(...) counted_hip_memcpy(__VA_ARGS__)
#define hipMemcpyAsync(...) counted_hip_memcpy_async(__VA_ARGS__)
#define ALLPACA_COUNTED_WRITE(name, ...) counted_write([&] { return ::name(__VA_ARGS__); })
#define hipMemcpyPeer(...) ALLPACA_COUNTED_WRITE(hipMemcpyPeer, __VA_ARGS__)
#define hipMemcpyPeerAsync(...) ALLPACA_COUNTED_WRITE(hipMemcpyPeerAsync, __VA_ARGS__)
#define hipMemcpy2D(...) ALLPACA_COUNTED_WRITE(hipMemcpy2D, __VA_ARGS__)
#define hipMemcpy2DAsync(...) ALLPACA_COUNTED_WRITE(hipMemcpy2DAsync, __VA_ARGS__)
#define hipMemset(...) ALLPACA_COUNTED_WRITE(hipMemset, __VA_ARGS__)
#define hipMemsetAsync(...) ALLPACA_COUNTED_WRITE(hipMemsetAsync, __VA_ARGS__)
#define hipMemsetD32Async(...) ALLPACA_COUNTED_WRITE(hipMemsetD32Async, __VA_ARGS__)
/* Frees and host (un)registrations can also hand an address to another kind
 * of memory, so they move this count too (see `KqActCopy`). */
static std::atomic<uint64_t> g_address_changes{0};
template <typename Call> static inline auto counted_remap(Call call) {
    auto r = call();
    g_address_changes.fetch_add(1, std::memory_order_relaxed);
    note_device_write();
    return r;
}
#define ALLPACA_COUNTED_REMAP(name, ...) counted_remap([&] { return ::name(__VA_ARGS__); })
#define hipFree(...) ALLPACA_COUNTED_REMAP(hipFree, __VA_ARGS__)
#define hipFreeAsync(...) ALLPACA_COUNTED_REMAP(hipFreeAsync, __VA_ARGS__)
#define hipHostFree(...) ALLPACA_COUNTED_REMAP(hipHostFree, __VA_ARGS__)
#define hipHostRegister(...) ALLPACA_COUNTED_REMAP(hipHostRegister, __VA_ARGS__)
#define hipHostUnregister(...) ALLPACA_COUNTED_REMAP(hipHostUnregister, __VA_ARGS__)
#ifdef ALLPACA_HAS_ROCBLAS
#define rocblas_sgemm(...) ALLPACA_COUNTED_WRITE(rocblas_sgemm, __VA_ARGS__)
#endif
#ifdef ALLPACA_HAS_HIPBLASLT
#define hipblasLtMatmul(...) ALLPACA_COUNTED_WRITE(hipblasLtMatmul, __VA_ARGS__)
#endif

/* IEEE half to float with the hardware conversion (exact, subnormals included). */
__device__ inline float allpaca_f16_bits_to_f32(uint16_t h) {
    return static_cast<float>(__builtin_bit_cast(_Float16, h));
}
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

/* F32 GEMV for few rows and at most a couple of thousand columns: one block
 * per output (column, row), threads stride K so loads coalesce, then fixed
 * 32-lane groups reduce on both wave32 and wave64 cards. The 64-column tiled
 * kernel put a whole 32-output GEMV (GDN alpha/beta) on one CU with per-thread
 * strided rows: ~30 us for 256 KB. */
__global__ void gemv_nt_colblock_kernel(const float *a, const float *b, float *c, int m, int n,
                                        int k) {
    __shared__ float part[32];
    const int col = static_cast<int>(blockIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (col >= n || row >= m) {
        return;
    }
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    const float *brow = b + static_cast<size_t>(col) * static_cast<size_t>(k);
    float acc = 0.f;
    for (int i = tid; i < k; i += static_cast<int>(blockDim.x)) {
        acc = fmaf(arow[i], brow[i], acc);
    }
    const int ws = 32;
    for (int off = ws / 2; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, ws);
    }
    const int lane = tid % ws;
    const int wid = tid / ws;
    if (lane == 0) {
        part[wid] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.f;
        const int nw = (static_cast<int>(blockDim.x) + ws - 1) / ws;
        for (int w = 0; w < nw; ++w) {
            sum += part[w];
        }
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = sum;
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
/* blockIdx.z == 1 computes the second matrix (b2 -> c2, n2 columns) of a pair
 * sharing the input `a` (a shared expert's gate and up, a GDN layer's qkv and
 * z): same per-output math, one launch instead of two. */
__global__ void gemv_nt_q8_0_f32_kernel(const float *a, const uint8_t *b0, float *c0, int m, int n0,
                                        int k, const uint8_t *b2, float *c2, int n2) {
    const uint8_t *b = blockIdx.z ? b2 : b0;
    float *c = blockIdx.z ? c2 : c0;
    const int n = blockIdx.z ? n2 : n0;
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

/* Power-of-two exponent that brings `amax` into [2^13, 2^14): scaled values
 * then fit f16 (max 65504) with room for the hi/lo split, and scaling by a
 * power of two is exact. Zero, inf and NaN rows are left unscaled; the shift
 * is clamped so the scale itself stays a normal float. */
__device__ __forceinline__ int wmma_f16_shift(float amax) {
    if (!(amax > 0.0f) || !isfinite(amax)) {
        return 0;
    }
    int e = 0;
    (void)frexpf(amax, &e); /* amax = f * 2^e, f in [0.5, 1) */
    const int sh = 14 - e;
    return sh < -100 ? -100 : (sh > 100 ? 100 : sh);
}

/* gfx12 f16 WMMA prefill tile (experimental.row_variant_kernels). One wave32
 * computes C[16,16] = A[16,k] * B_nt[16,k]^T. B is packed f16 row-major [n,k],
 * already the transposed operand WMMA wants. A is f32 and is NOT rounded to
 * f16: each row is scaled by a power of two so its largest |a| lies in
 * [2^13, 2^14), then split as a*s = hi + lo with hi = f16(a*s) and
 * lo = f16(a*s - hi). Two WMMAs (hi*B, lo*B) accumulate in f32; hi*b and
 * lo*b are exact in f32, so A keeps ~22 bits and no value overflows f16.
 * The row's result is unscaled exactly at the end.
 *
 * Lane layout (RDNA4, not the RDNA3 replicated layout): lane L holds A row L%16
 * and B column L%16 at K positions 8*(L/16) + 0..7. Accumulator element g is
 * C row (L/16)*8+g, column L%16. A row's result depends only on that row, but
 * its bits differ from the decode GEMV's.
 *
 * NOT USED BY ANY FORWARD: on Llama-3.2-3B F16 prefill it was ~4x slower than
 * the existing F16 path (one wave per tile, no LDS reuse of A or B), so it is
 * a tested building block. Any future caller must sit inside the prefill WMMA
 * scope (experimental.row_variant_kernels).
 * Non-gfx12 images leave the body empty; the host launch sends those devices
 * to the scalar f16 kernel. */
__global__ void __launch_bounds__(32) gemm_nt_f16_wmma_kernel(const float *a, const uint16_t *b, float *c,
                                                              int m, int n, int k) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using f16x8 = __attribute__((ext_vector_type(8))) _Float16;
    using f32x8 = __attribute__((ext_vector_type(8))) float;

    const int lane = static_cast<int>(threadIdx.x);
    const int row0 = static_cast<int>(blockIdx.y) * 16;
    const int col0 = static_cast<int>(blockIdx.x) * 16;
    const int a_row = row0 + (lane % 16);
    const int b_col = col0 + (lane % 16);
    const int k_group = 8 * (lane / 16);
    const float *ar = a + static_cast<size_t>(a_row < m ? a_row : 0) * k;

    /* Row scale: this lane sees half the row's K positions, lane L^16 the rest. */
    float amax = 0.0f;
    if (a_row < m) {
        for (int k0 = 0; k0 < k; k0 += 16) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int kp = k0 + k_group + e;
                if (kp < k) {
                    amax = fmaxf(amax, fabsf(ar[kp]));
                }
            }
        }
    }
    amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
    const int sh = wmma_f16_shift(amax);
    const float s = ldexpf(1.0f, sh);

    f32x8 acc;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        acc[g] = 0.0f;
    }
    for (int k0 = 0; k0 < k; k0 += 16) {
        f16x8 hi_frag;
        f16x8 lo_frag;
        f16x8 b_frag;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int kp = k0 + k_group + e;
            float x = 0.0f;
            if (a_row < m && kp < k) {
                x = ar[kp] * s;
            }
            const _Float16 hi = static_cast<_Float16>(x);
            hi_frag[e] = hi;
            lo_frag[e] = static_cast<_Float16>(x - static_cast<float>(hi));
            b_frag[e] = static_cast<_Float16>(0.0f);
            if (b_col < n && kp < k) {
                b_frag[e] = __builtin_bit_cast(_Float16, b[static_cast<size_t>(b_col) * k + kp]);
            }
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(hi_frag, b_frag, acc);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(lo_frag, b_frag, acc);
    }

    const int col_out = col0 + (lane % 16);
    const int row_base = row0 + (lane / 16) * 8;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        /* Row (L/16)*8+g's shift lives in lane (L/16)*8+g. */
        const int row_sh = __shfl(sh, (lane / 16) * 8 + g, 32);
        const int row_out = row_base + g;
        if (row_out < m && col_out < n) {
            c[static_cast<size_t>(row_out) * n + col_out] = ldexpf(acc[g], -row_sh);
        }
    }
#else
    (void)a;
    (void)b;
    (void)c;
    (void)m;
    (void)n;
    (void)k;
#endif
}

/* Scalar int8 NT GEMM: C[m,n] = sum_k A[m,k] * B[n,k], int32 accumulate, f32 store.
 * This is the reference the gfx12 WMMA kernel has to match. */
__global__ void gemm_nt_i8_kernel(const int8_t *a, const int8_t *b, float *c, int m, int n, int k) {
    const int col = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
    if (row >= m || col >= n) {
        return;
    }
    int acc = 0;
    const int8_t *ar = a + static_cast<size_t>(row) * k;
    const int8_t *br = b + static_cast<size_t>(col) * k;
    for (int p = 0; p < k; ++p) {
        acc += static_cast<int>(ar[p]) * static_cast<int>(br[p]);
    }
    c[static_cast<size_t>(row) * n + col] = static_cast<float>(acc);
}

__device__ __forceinline__ int allpaca_pack_i8x4(int8_t e0, int8_t e1, int8_t e2, int8_t e3) {
    return static_cast<int>(static_cast<uint8_t>(e0)) | (static_cast<int>(static_cast<uint8_t>(e1)) << 8) |
           (static_cast<int>(static_cast<uint8_t>(e2)) << 16) |
           (static_cast<int>(static_cast<uint8_t>(e3)) << 24);
}

/* gfx12 int8 WMMA. Same tile geometry as the f16 kernel: lane L owns A row L%16
 * and B column L%16, eight int8 at K 8*(L/16)+0..7, packed little-endian into
 * two 32-bit registers. Accumulator element g is C row (L/16)*8+g, column L%16.
 *
 * NOT USED YET: a building block for a future int8 (Q8_0) prefill GEMM. No
 * forward calls it; only its unit test does. Wiring it in needs the int8
 * activation quantize and per-block scales, under the same experimental
 * prefill scope as the f16 WMMA GEMM. */
__global__ void gemm_nt_i8_wmma_kernel(const int8_t *a, const int8_t *b, float *c, int m, int n,
                                       int k) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using i32x2 = __attribute__((ext_vector_type(2))) int;
    using i32x8 = __attribute__((ext_vector_type(8))) int;
    const int lane = static_cast<int>(threadIdx.x);
    const int row0 = static_cast<int>(blockIdx.y) * 16;
    const int col0 = static_cast<int>(blockIdx.x) * 16;
    const int a_row = row0 + (lane % 16);
    const int b_col = col0 + (lane % 16);
    const int k_group = 8 * (lane / 16);
    i32x8 acc;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        acc[g] = 0;
    }
    for (int k0 = 0; k0 < k; k0 += 16) {
        int8_t av[8], bv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int kp = k0 + k_group + e;
            av[e] = (a_row < m && kp < k) ? a[static_cast<size_t>(a_row) * k + kp] : 0;
            bv[e] = (b_col < n && kp < k) ? b[static_cast<size_t>(b_col) * k + kp] : 0;
        }
        i32x2 a_frag;
        i32x2 b_frag;
        a_frag[0] = allpaca_pack_i8x4(av[0], av[1], av[2], av[3]);
        a_frag[1] = allpaca_pack_i8x4(av[4], av[5], av[6], av[7]);
        b_frag[0] = allpaca_pack_i8x4(bv[0], bv[1], bv[2], bv[3]);
        b_frag[1] = allpaca_pack_i8x4(bv[4], bv[5], bv[6], bv[7]);
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_frag, true, b_frag, acc, false);
    }
    const int col_out = col0 + (lane % 16);
    const int row_base = row0 + (lane / 16) * 8;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        const int row_out = row_base + g;
        if (row_out < m && col_out < n) {
            c[static_cast<size_t>(row_out) * n + col_out] = static_cast<float>(acc[g]);
        }
    }
#else
    (void)a;
    (void)b;
    (void)c;
    (void)m;
    (void)n;
    (void)k;
#endif
}

/* Deterministic math: every rounding spelled out, so HIP, CUDA and the CPU
 * reference (`allpaca_cpu::det_*`) give the same bits. The vendors' expf, sinf,
 * cosf, powf and rsqrtf differ in the last bits, and the int8 activations of
 * the K-quant GEMV turn a last-bit difference into a step of 1/127 of a block's
 * largest value (3e-7 in, 3e-4 out, measured), so layers on an NVIDIA worker
 * changed the text against an AMD one. `/` and sqrtf are correctly rounded on
 * both.
 * exp(x) = 2^n * exp(r), n = rint(x / ln 2), |r| <= ln 2 / 2, a degree-7
 * Taylor polynomial in Horner form (error < 1e-8 relative), then 2^n applied
 * as two exact power-of-two products. */
__device__ inline float det_pow2i(int k) {
    return __int_as_float((k + 127) << 23);
}

__device__ inline float det_expf(float x) {
    x = fminf(fmaxf(x, -103.0f), 88.5f);
    const float n = rintf(__fmul_rn(x, 1.44269504088896341f));
    float r = fmaf(n, -0.693145751953125f, x);
    r = fmaf(n, -1.42860682030941723212e-6f, r);
    float p = 1.98412698412698413e-4f;
    p = fmaf(p, r, 1.38888888888888889e-3f);
    p = fmaf(p, r, 8.33333333333333333e-3f);
    p = fmaf(p, r, 4.16666666666666667e-2f);
    p = fmaf(p, r, 1.66666666666666667e-1f);
    p = fmaf(p, r, 0.5f);
    p = fmaf(p, r, 1.0f);
    p = fmaf(p, r, 1.0f);
    const int ni = static_cast<int>(n);
    const int a = ni / 2;
    return __fmul_rn(__fmul_rn(p, det_pow2i(a)), det_pow2i(ni - a));
}

/* ln(x), positive normal x: x = m * 2^e, m in [sqrt(1/2), sqrt(2)], then
 * 2 * atanh(s), s = (m - 1) / (m + 1). */
__device__ inline float det_logf(float x) {
    const int bits = __float_as_int(x);
    int e = ((bits >> 23) & 255) - 127;
    float m = __int_as_float((bits & 0x007fffff) | 0x3f800000);
    if (m > 1.41421356f) {
        m = __fmul_rn(m, 0.5f);
        e += 1;
    }
    const float f = __fadd_rn(m, -1.0f);
    const float s = __fdiv_rn(f, __fadd_rn(2.0f, f));
    const float z = __fmul_rn(s, s);
    float p = 2.2222222e-1f;
    p = fmaf(p, z, 2.8571428e-1f);
    p = fmaf(p, z, 0.4f);
    p = fmaf(p, z, 6.6666666e-1f);
    p = fmaf(p, z, 2.0f);
    return fmaf(static_cast<float>(e), 0.69314718f, __fmul_rn(s, p));
}

/* base^y, positive base: single precision throughout, about 1e-6 relative at
 * worst, which for RoPE is the slowest pairs (2e-6 rad at position 10k). */
__device__ inline float det_powf(float base, float y) {
    return det_expf(__fmul_rn(y, det_logf(base)));
}

/* sin and cos of x >= 0: reduction by pi / 2 in two fused steps, then Taylor
 * polynomials on |r| <= pi / 4. */
__device__ inline void det_sincosf(float x, float &s, float &c) {
    const float n = rintf(__fmul_rn(x, 6.3661977e-1f));
    float r = fmaf(n, -1.5707964f, x);
    r = fmaf(n, 4.371139e-8f, r);
    const float z = __fmul_rn(r, r);
    float p = 2.7557319e-6f;
    p = fmaf(p, z, -1.984127e-4f);
    p = fmaf(p, z, 8.333334e-3f);
    p = fmaf(p, z, -1.6666667e-1f);
    const float sin_r = fmaf(__fmul_rn(r, z), p, r);
    float q = -2.755732e-7f;
    q = fmaf(q, z, 2.4801588e-5f);
    q = fmaf(q, z, -1.3888889e-3f);
    q = fmaf(q, z, 4.1666668e-2f);
    q = fmaf(q, z, -0.5f);
    q = fmaf(q, z, 1.0f);
    const int k = static_cast<int>(n) & 3;
    s = k == 0 ? sin_r : (k == 1 ? q : (k == 2 ? -sin_r : -q));
    c = k == 0 ? q : (k == 1 ? -sin_r : (k == 2 ? -q : sin_r));
}

/* 1 / sqrt(x). The hardware reciprocal roots are not the same across vendors. */
__device__ inline float det_rsqrtf(float x) {
    return __fdiv_rn(1.0f, sqrtf(x));
}

/* silu(g) * u, bit-identical across backends. */
__device__ inline float det_silu_mul(float g, float u) {
    return __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, det_expf(-g))), u);
}

/* x * sigmoid(g). */
__device__ inline float det_sigmoid_mul(float x, float g) {
    return __fmul_rn(x, __fdiv_rn(1.0f, __fadd_rn(1.0f, det_expf(-g))));
}

/* Turns the pair (a, b) by `angle`, or back by it when `forward` is 0. */
__device__ inline void det_rotate(float &a, float &b, float angle, int forward) {
    float s, c;
    det_sincosf(angle, s, c);
    s = forward ? s : -s;
    const float x = fmaf(a, c, -__fmul_rn(b, s));
    const float y = fmaf(a, s, __fmul_rn(b, c));
    a = x;
    b = y;
}

__device__ float warp_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down(v, off, 32);
    }
    return __shfl(v, 0, 32);
}

/* Decode/prefill attention. One 32-lane warp owns one (query, head) row and
 * walks its keys in order with an online softmax, ATTN_KT keys a step
 * (attn_tile_step). `attn_row_*` and `attn_tile_step` hold the per-row
 * arithmetic so the split and Q-tiled kernels run the same instructions on
 * the same values: a row's result does not depend on which kernel, or how
 * many rows, computed it. */
/* Per-lane register slots for one (query, head) row at template DPT (the
 * head_dim/32 bucket): the float4 path holds 4 dims per slot, so round the
 * array up to a multiple of 4; the scalar path never touches past DPT. Fewer
 * live registers is the point of the templating, so the arrays must be
 * exactly this size, not ATTN_MAX_DPT. */
template <int DPT>
constexpr int attn_row_dims() {
    return ((DPT + 3) / 4) * 4;
}

template <int DPT>
struct AttnRow {
    float qreg[attn_row_dims<DPT>()];
    float acc[attn_row_dims<DPT>()];
    float m;
    float l;
};

template <int DPT>
__device__ __forceinline__ void attn_row_init(AttnRow<DPT> &r, const float *qh, int lane, int head_dim,
                                              int nvec, bool vec4, const float *sinks, int h) {
#pragma unroll
    for (int i = 0; i < attn_row_dims<DPT>(); ++i) {
        r.qreg[i] = 0.0f;
        r.acc[i] = 0.0f;
    }
    /* Fixed trip counts with a per-lane guard (not `v < nvec` bounds): the
     * arrays then stay in registers; with runtime bounds gfx906 kept them in
     * scratch memory. Same operations in the same order either way. */
    if (vec4) {
#pragma unroll
        for (int j = 0; j < attn_row_dims<DPT>() / 4; ++j) {
            const int v = lane + 32 * j;
            if (v < nvec) {
                const float4 qv = *reinterpret_cast<const float4 *>(qh + v * 4);
                r.qreg[4 * j] = qv.x;
                r.qreg[4 * j + 1] = qv.y;
                r.qreg[4 * j + 2] = qv.z;
                r.qreg[4 * j + 3] = qv.w;
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < DPT; ++j) {
            const int d = lane + 32 * j;
            if (d < head_dim) {
                r.qreg[j] = qh[d];
            }
        }
    }
    r.m = -1.0e30f;
    r.l = 0.0f;
    if (sinks) {
        r.m = sinks[h];
        r.l = 1.0f;
    }
}

/* One key: `kh`/`vh` are that key's head rows in global memory.
 * Contraction is off and every fused multiply-add is written out (here and in
 * attn_tile_step): left to the compiler, kernels fused differently once their
 * loops were unrolled, and their rows stopped matching. */
template <int DPT>
__device__ __forceinline__ void attn_row_step(AttnRow<DPT> &r, const float *kh, const float *vh,
                                              bool valid, int lane, int head_dim, int nvec,
                                              bool vec4, float scale) {
#pragma clang fp contract(off)
    float partial = 0.0f;
    if (vec4) {
#pragma unroll
        for (int j = 0; j < attn_row_dims<DPT>() / 4; ++j) {
            const int vec = lane + 32 * j;
            if (vec >= nvec) {
                continue;
            }
            const int i = 4 * j;
            float4 kv;
            if (valid && (reinterpret_cast<uintptr_t>(kh + vec * 4) % 16u) == 0) {
                kv = *reinterpret_cast<const float4 *>(kh + vec * 4);
            } else {
                kv.x = (valid && vec * 4 < head_dim) ? kh[vec * 4] : 0.0f;
                kv.y = (valid && vec * 4 + 1 < head_dim) ? kh[vec * 4 + 1] : 0.0f;
                kv.z = (valid && vec * 4 + 2 < head_dim) ? kh[vec * 4 + 2] : 0.0f;
                kv.w = (valid && vec * 4 + 3 < head_dim) ? kh[vec * 4 + 3] : 0.0f;
            }
            float d4 = r.qreg[i] * kv.x;
            d4 = fmaf(r.qreg[i + 1], kv.y, d4);
            d4 = fmaf(r.qreg[i + 2], kv.z, d4);
            d4 = fmaf(r.qreg[i + 3], kv.w, d4);
            partial = partial + d4;
        }
    } else {
#pragma unroll
        for (int j = 0; j < DPT; ++j) {
            const int d = lane + 32 * j;
            if (d < head_dim) {
                partial = fmaf(r.qreg[j], valid ? kh[d] : 0.0f, partial);
            }
        }
    }
    float s = warp_sum32(partial) * scale;
    if (!valid) {
        s = -1.0e30f;
    }
    const float m_new = fmaxf(r.m, s);
    const float alpha = det_expf(r.m - m_new);
    const float p = det_expf(s - m_new);
    r.l = fmaf(r.l, alpha, p);
    if (vec4) {
#pragma unroll
        for (int j = 0; j < attn_row_dims<DPT>() / 4; ++j) {
            const int vec = lane + 32 * j;
            if (vec >= nvec) {
                continue;
            }
            const int i = 4 * j;
            float4 vv;
            if (valid && (reinterpret_cast<uintptr_t>(vh + vec * 4) % 16u) == 0) {
                vv = *reinterpret_cast<const float4 *>(vh + vec * 4);
            } else {
                vv.x = (valid && vec * 4 < head_dim) ? vh[vec * 4] : 0.0f;
                vv.y = (valid && vec * 4 + 1 < head_dim) ? vh[vec * 4 + 1] : 0.0f;
                vv.z = (valid && vec * 4 + 2 < head_dim) ? vh[vec * 4 + 2] : 0.0f;
                vv.w = (valid && vec * 4 + 3 < head_dim) ? vh[vec * 4 + 3] : 0.0f;
            }
            r.acc[i] = fmaf(r.acc[i], alpha, p * vv.x);
            r.acc[i + 1] = fmaf(r.acc[i + 1], alpha, p * vv.y);
            r.acc[i + 2] = fmaf(r.acc[i + 2], alpha, p * vv.z);
            r.acc[i + 3] = fmaf(r.acc[i + 3], alpha, p * vv.w);
        }
    } else {
#pragma unroll
        for (int j = 0; j < DPT; ++j) {
            const int d = lane + 32 * j;
            if (d < head_dim) {
                r.acc[j] = fmaf(r.acc[j], alpha, p * (valid ? vh[d] : 0.0f));
            }
        }
    }
    r.m = m_new;
}

/* Keys per attn_tile_step. */
constexpr int ATTN_KT = 16;

/* One stage of the transposing reduction in attn_tile_step: lanes `m` apart
 * swap halves of their `2*H` key partials and add what they keep to what they
 * receive. */
template <int H>
__device__ __forceinline__ void attn_keys_stage(float (&p)[ATTN_KT], int lane, int m) {
#pragma clang fp contract(off)
#pragma unroll
    for (int i = 0; i < H; ++i) {
        const bool up = (lane & m) != 0;
        const float send = up ? p[i] : p[i + H];
        const float keep = up ? p[i + H] : p[i];
        p[i] = keep + __shfl_xor(send, m, 32);
    }
}

/* ATTN_KT keys at once, rows `kt`/`vt` of head_dim floats in LDS. Each lane
 * takes partial dot products over its dims for every key; a transposing
 * reduction leaves key lane/2's score on lanes 2j and 2j+1; one max, one
 * pair of expf and one sum serve all the keys, and the values accumulate in
 * key order. `valid` is for key lane/2. Key/value rows of keys whose weight
 * is exactly zero are not read into the accumulators. */
template <int DPT>
__device__ __forceinline__ void attn_tile_step(AttnRow<DPT> &r, const float *kt, const float *vt, bool valid,
                                               int lane, int head_dim, int nvec, bool vec4, float scale) {
#pragma clang fp contract(off)
    float part[ATTN_KT];
#pragma unroll
    for (int j = 0; j < ATTN_KT; ++j) {
        const float *kh = kt + static_cast<size_t>(j) * head_dim;
        float partial = 0.0f;
        if (vec4) {
#pragma unroll
            for (int g = 0; g < attn_row_dims<DPT>() / 4; ++g) {
                const int vec = lane + 32 * g;
                if (vec < nvec) {
                    const float4 kv = *reinterpret_cast<const float4 *>(kh + vec * 4);
                    float d4 = r.qreg[4 * g] * kv.x;
                    d4 = fmaf(r.qreg[4 * g + 1], kv.y, d4);
                    d4 = fmaf(r.qreg[4 * g + 2], kv.z, d4);
                    d4 = fmaf(r.qreg[4 * g + 3], kv.w, d4);
                    partial = partial + d4;
                }
            }
        } else {
#pragma unroll
            for (int g = 0; g < DPT; ++g) {
                const int d = lane + 32 * g;
                if (d < head_dim) {
                    partial = fmaf(r.qreg[g], kh[d], partial);
                }
            }
        }
        part[j] = partial;
    }
    attn_keys_stage<8>(part, lane, 16);
    attn_keys_stage<4>(part, lane, 8);
    attn_keys_stage<2>(part, lane, 4);
    attn_keys_stage<1>(part, lane, 2);
    float s = (part[0] + __shfl_xor(part[0], 1, 32)) * scale;
    if (!valid) {
        s = -1.0e30f;
    }
    float top = s;
#pragma unroll
    for (int m = 2; m < 32; m <<= 1) {
        top = fmaxf(top, __shfl_xor(top, m, 32));
    }
    const float m_new = fmaxf(r.m, top);
    const float alpha = det_expf(r.m - m_new);
    const float p = valid ? det_expf(s - m_new) : 0.0f;
    float sum = p;
#pragma unroll
    for (int m = 2; m < 32; m <<= 1) {
        sum = sum + __shfl_xor(sum, m, 32);
    }
    r.l = fmaf(r.l, alpha, sum);
#pragma unroll
    for (int i = 0; i < attn_row_dims<DPT>(); ++i) {
        r.acc[i] = r.acc[i] * alpha;
    }
#pragma unroll
    for (int j = 0; j < ATTN_KT; ++j) {
        const float pj = __shfl(p, 2 * j, 32);
        if (pj == 0.0f) {
            continue;
        }
        const float *vh = vt + static_cast<size_t>(j) * head_dim;
        if (vec4) {
#pragma unroll
            for (int g = 0; g < attn_row_dims<DPT>() / 4; ++g) {
                const int vec = lane + 32 * g;
                if (vec < nvec) {
                    const float4 vv = *reinterpret_cast<const float4 *>(vh + vec * 4);
                    r.acc[4 * g] = fmaf(pj, vv.x, r.acc[4 * g]);
                    r.acc[4 * g + 1] = fmaf(pj, vv.y, r.acc[4 * g + 1]);
                    r.acc[4 * g + 2] = fmaf(pj, vv.z, r.acc[4 * g + 2]);
                    r.acc[4 * g + 3] = fmaf(pj, vv.w, r.acc[4 * g + 3]);
                }
            }
        } else {
#pragma unroll
            for (int g = 0; g < DPT; ++g) {
                const int d = lane + 32 * g;
                if (d < head_dim) {
                    r.acc[g] = fmaf(pj, vh[d], r.acc[g]);
                }
            }
        }
    }
    r.m = m_new;
}

template <int DPT>
__device__ __forceinline__ void attn_row_store(const AttnRow<DPT> &r, float *oh, int lane, int head_dim,
                                               int nvec, bool vec4) {
    const float inv = (r.l == 0.0f) ? 0.0f : 1.0f / r.l;
    if (vec4) {
#pragma unroll
        for (int j = 0; j < attn_row_dims<DPT>() / 4; ++j) {
            const int vec = lane + 32 * j;
            if (vec >= nvec) {
                continue;
            }
            const int i = 4 * j;
            if (vec * 4 + 3 < head_dim && (reinterpret_cast<uintptr_t>(oh + vec * 4) % 16u) == 0) {
                float4 o;
                o.x = r.acc[i] * inv;
                o.y = r.acc[i + 1] * inv;
                o.z = r.acc[i + 2] * inv;
                o.w = r.acc[i + 3] * inv;
                *reinterpret_cast<float4 *>(oh + vec * 4) = o;
            } else {
                if (vec * 4 < head_dim) {
                    oh[vec * 4] = r.acc[i] * inv;
                }
                if (vec * 4 + 1 < head_dim) {
                    oh[vec * 4 + 1] = r.acc[i + 1] * inv;
                }
                if (vec * 4 + 2 < head_dim) {
                    oh[vec * 4 + 2] = r.acc[i + 2] * inv;
                }
                if (vec * 4 + 3 < head_dim) {
                    oh[vec * 4 + 3] = r.acc[i + 3] * inv;
                }
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < DPT; ++j) {
            const int d = lane + 32 * j;
            if (d < head_dim) {
                oh[d] = r.acc[j] * inv;
            }
        }
    }
}

/* Last key row `qi` may see: causal rows are bottom-right aligned. */
__device__ __forceinline__ int attn_row_tmax(int qi, int n_q, int n_kv, int causal) {
    return causal ? qi + (n_kv > n_q ? n_kv - n_q : 0) : n_kv - 1;
}

/* A row with more than ATTN_CHUNK keys: the keys split into chunks of
 * ATTN_CHUNK, each an online softmax from an empty state (sinks in chunk 0
 * only), folded by attn_fold in one fixed tree (see AttnFoldStack). Every
 * kernel builds that tree, so a row's bits do not depend on which kernel
 * computed it, and a row with one chunk is exactly the single-pass result.
 * The chunking depends only on the keys a row sees, never on the device.
 * Decode runs the chunks of a long row in parallel (attention_split_gqa_kernel,
 * then attention_merge_pair_kernel for the tree's lowest levels and
 * attention_merge_kernel for the rest).
 * Inside a chunk the keys go ATTN_KT at a time (attn_tile_step). 128 against
 * 256 took a 2051-key MI50 decode row from 217 to 141 us. */
constexpr int ATTN_CHUNK = 128;
static_assert(ATTN_CHUNK % ATTN_KT == 0, "chunks are whole key tiles");
/* The fold tree: at stride s = 1, 2, 4, ... the partial at left + s folds into
 * the one at left, for every left that is a multiple of 2s with left + s <
 * n_chunks. The Q-tiled kernel meets the chunks one at a time and builds the
 * same tree with a carry stack: push a chunk, fold while the top two spans are
 * the same size, and at the end fold what is left right to left (left to right
 * gives another tree whenever n_chunks has three or more set bits, e.g. 7).
 * The stack proper (blocks past the register levels, see AttnFoldRegs) holds
 * at most floor(log2(n_chunks >> reg)) + 2 partials: 16 entries cover
 * 2^16 chunks (8M keys) with at least one register level. Past that the
 * kernel traps rather than fold in another order. */
constexpr int ATTN_FOLD_STACK = 16;
/* Floats per (row, head, chunk) partial: every lane's accumulators, m, l.
 * Sized by the head_dim bucket so the scratch holds only what the launched
 * kernels write; the host launcher uses the same value for one call. */
template <int DPT>
constexpr int attn_part() {
    return 32 * attn_row_dims<DPT>() + 2;
}

template <int DPT>
__device__ __forceinline__ void attn_row_reset(AttnRow<DPT> &r) {
#pragma unroll
    for (int i = 0; i < attn_row_dims<DPT>(); ++i) {
        r.acc[i] = 0.0f;
    }
    r.m = -1.0e30f;
    r.l = 0.0f;
}

/* into (+)= (m_b, l_b, acc_b), rounding spelled out. DPT is named at the
 * call sites: it sits inside the array bound, a non-deduced context. */
template <int DPT>
__device__ __forceinline__ void attn_fold(float &m, float &l, float *acc, float m_b, float l_b,
                                          const float *acc_b) {
#pragma clang fp contract(off)
    const float mm = fmaxf(m, m_b);
    const float a = det_expf(m - mm);
    const float b = det_expf(m_b - mm);
    l = fmaf(l, a, __fmul_rn(l_b, b));
#pragma unroll
    for (int i = 0; i < attn_row_dims<DPT>(); ++i) {
        acc[i] = fmaf(acc[i], a, __fmul_rn(acc_b[i], b));
    }
    m = mm;
}

/* The carry stack of the fold tree (see ATTN_FOLD_STACK), one per warp lane.
 * Its lowest attn_fold_reg<DPT>() levels (blocks of 1, 2, 4, ... chunks) sit
 * in registers, indexed only by unrolled loops; bigger blocks go to the stack
 * proper, which lives in scratch memory. With every block in scratch, the
 * Q-tiled prefill of a 4096-key row took 7.5% longer on gfx1151; 4 levels
 * brought it to about 1%. Wider rows keep fewer levels so the kernel's
 * occupancy stays as it was (gfx1201: 16, 16 and 12 waves for DPT 4, 8, 16).
 * The tree does not depend on the count. */
constexpr int ATTN_FOLD_REG_MAX = 4;
template <int DPT>
__host__ __device__ constexpr int attn_fold_reg() {
    return DPT <= 4 ? 4 : DPT <= 8 ? 2 : 1;
}

/* The register levels and the stack proper are separate objects: one
 * object with both went wholly to scratch, register levels included. */
template <int DPT>
struct AttnFoldRegs {
    float racc[ATTN_FOLD_REG_MAX][attn_row_dims<DPT>()];
    float rm[ATTN_FOLD_REG_MAX];
    float rl[ATTN_FOLD_REG_MAX];
    bool has[ATTN_FOLD_REG_MAX];
};

template <int DPT>
struct AttnFoldStack {
    float acc[ATTN_FOLD_STACK][attn_row_dims<DPT>()];
    float m[ATTN_FOLD_STACK];
    float l[ATTN_FOLD_STACK];
    int span[ATTN_FOLD_STACK];
    int sp;
};

template <int DPT>
__device__ __forceinline__ void attn_tree_init(AttnFoldRegs<DPT> &rg, AttnFoldStack<DPT> &st) {
#pragma unroll
    for (int i = 0; i < attn_fold_reg<DPT>(); ++i) {
        rg.has[i] = false;
    }
    st.sp = 0;
}

/* True when the stack holds no chunk. */
template <int DPT>
__device__ __forceinline__ bool attn_tree_empty(const AttnFoldRegs<DPT> &rg, const AttnFoldStack<DPT> &st) {
    bool any = st.sp > 0;
#pragma unroll
    for (int i = 0; i < attn_fold_reg<DPT>(); ++i) {
        any = any || rg.has[i];
    }
    return !any;
}

/* Pushes one finished chunk: it carries up the register levels like a binary
 * counter (each fold is older-left, newer-right), and a carry out of the top
 * one goes to the stack proper, folding while its top two spans match. */
template <int DPT>
__device__ __forceinline__ void attn_tree_push(AttnFoldRegs<DPT> &rg, AttnFoldStack<DPT> &st, const AttnRow<DPT> &r) {
    constexpr int dims = attn_row_dims<DPT>();
    float cacc[dims];
#pragma unroll
    for (int j = 0; j < dims; ++j) {
        cacc[j] = r.acc[j];
    }
    float cm = r.m;
    float cl = r.l;
    bool done = false;
#pragma unroll
    for (int i = 0; i < attn_fold_reg<DPT>(); ++i) {
        if (!done) {
            if (!rg.has[i]) {
#pragma unroll
                for (int j = 0; j < dims; ++j) {
                    rg.racc[i][j] = cacc[j];
                }
                rg.rm[i] = cm;
                rg.rl[i] = cl;
                rg.has[i] = true;
                done = true;
            } else {
                attn_fold<DPT>(rg.rm[i], rg.rl[i], rg.racc[i], cm, cl, cacc);
#pragma unroll
                for (int j = 0; j < dims; ++j) {
                    cacc[j] = rg.racc[i][j];
                }
                cm = rg.rm[i];
                cl = rg.rl[i];
                rg.has[i] = false;
            }
        }
    }
    if (done) {
        return;
    }
    if (st.sp >= ATTN_FOLD_STACK) {
        __builtin_trap();
    }
    for (int j = 0; j < dims; ++j) {
        st.acc[st.sp][j] = cacc[j];
    }
    st.m[st.sp] = cm;
    st.l[st.sp] = cl;
    st.span[st.sp] = 1 << attn_fold_reg<DPT>();
    ++st.sp;
    while (st.sp >= 2 && st.span[st.sp - 1] == st.span[st.sp - 2]) {
        attn_fold<DPT>(st.m[st.sp - 2], st.l[st.sp - 2], st.acc[st.sp - 2], st.m[st.sp - 1], st.l[st.sp - 1],
                       st.acc[st.sp - 1]);
        st.span[st.sp - 2] += st.span[st.sp - 1];
        --st.sp;
    }
}

/* Folds what the carry left, right to left, into `r`: the register levels
 * from the lowest (the rightmost blocks), then the stack proper from its top.
 * The stack must not be empty. */
template <int DPT>
__device__ __forceinline__ void attn_tree_finish(AttnFoldRegs<DPT> &rg, AttnFoldStack<DPT> &st, AttnRow<DPT> &r) {
    constexpr int dims = attn_row_dims<DPT>();
    bool any = false;
#pragma unroll
    for (int i = 0; i < attn_fold_reg<DPT>(); ++i) {
        if (rg.has[i]) {
            if (any) {
                attn_fold<DPT>(rg.rm[i], rg.rl[i], rg.racc[i], r.m, r.l, r.acc);
            }
#pragma unroll
            for (int j = 0; j < dims; ++j) {
                r.acc[j] = rg.racc[i][j];
            }
            r.m = rg.rm[i];
            r.l = rg.rl[i];
            any = true;
        }
    }
    for (int i = st.sp - 1; i >= 0; --i) {
        if (any) {
            attn_fold<DPT>(st.m[i], st.l[i], st.acc[i], r.m, r.l, r.acc);
        }
        for (int j = 0; j < dims; ++j) {
            r.acc[j] = st.acc[i][j];
        }
        r.m = st.m[i];
        r.l = st.l[i];
        any = true;
    }
}

/* One warp per (query, head), keys from global memory one at a time
 * (attn_row_step): head dims too wide for the tiled kernels' LDS. Untemplated:
 * any head_dim, with the full ATTN_MAX_DPT slot count. */
__global__ void __launch_bounds__(32) attention_flash_kernel(const float *q, const float *k, const float *v, float *out,
                                       int n_q, int n_kv, int n_heads, int n_kv_heads, int head_dim,
                                       int causal, const uint8_t *mask, const float *sinks) {
    const int qi = static_cast<int>(blockIdx.x);
    const int h = static_cast<int>(blockIdx.y);
    if (qi >= n_q || h >= n_heads) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const int kv_h = h / (n_heads / n_kv_heads);
    const float scale = det_rsqrtf(static_cast<float>(head_dim));
    const int t_max = attn_row_tmax(qi, n_q, n_kv, causal);
    const float *qh = q + (static_cast<size_t>(qi) * n_heads + h) * head_dim;
    float *oh = out + (static_cast<size_t>(qi) * n_heads + h) * head_dim;
    const size_t kv_stride = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
    const bool vec4 = (head_dim % 4) == 0 && head_dim > 0 &&
                      (reinterpret_cast<uintptr_t>(qh) % 16u) == 0;
    const int nvec = vec4 ? head_dim / 4 : head_dim;
    AttnRow<ATTN_MAX_DPT> r;
    attn_row_init(r, qh, lane, head_dim, nvec, vec4, sinks, h);
    /* Causal rows stop at their last visible key: a masked step adds exactly
     * nothing (p = 0, alpha = 1). */
    const int t_end = mask ? n_kv : min(n_kv, t_max + 1);
    for (int t = 0; t < t_end; ++t) {
        const bool valid = mask ? mask[static_cast<size_t>(qi) * static_cast<size_t>(n_kv) +
                                       static_cast<size_t>(t)] != 0
                                : t <= t_max;
        const float *kh = k + static_cast<size_t>(t) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
        const float *vh = v + static_cast<size_t>(t) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
        attn_row_step(r, kh, vh, valid, lane, head_dim, nvec, vec4, scale);
    }
    attn_row_store(r, oh, lane, head_dim, nvec, vec4);
}

/* Copies keys [t0, t0 + tile) of key/value head `kv_h` into LDS, zeros from
 * `t_lim` on; float4 at a time when the rows allow it (a 2051-key MI50 decode
 * row took 189 us with float copies, 141 with float4). */
__device__ __forceinline__ void attn_stage_tile(float *k_lds, float *v_lds, const float *k, const float *v,
                                                int t0, int t_lim, int tile, int head_dim, size_t kv_stride,
                                                int kv_h) {
    const size_t base = static_cast<size_t>(kv_h) * head_dim;
    if ((head_dim % 4) == 0 && (kv_stride % 4) == 0 && (reinterpret_cast<uintptr_t>(k) % 16u) == 0 &&
        (reinterpret_cast<uintptr_t>(v) % 16u) == 0) {
        const int row4 = head_dim / 4;
        const int n4 = tile * row4;
        for (int e = static_cast<int>(threadIdx.x); e < n4; e += static_cast<int>(blockDim.x)) {
            const int t = t0 + e / row4;
            const size_t src = static_cast<size_t>(t) * kv_stride + base + static_cast<size_t>(e % row4) * 4;
            const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            reinterpret_cast<float4 *>(k_lds)[e] = t < t_lim ? *reinterpret_cast<const float4 *>(k + src) : z;
            reinterpret_cast<float4 *>(v_lds)[e] = t < t_lim ? *reinterpret_cast<const float4 *>(v + src) : z;
        }
        return;
    }
    const int n = tile * head_dim;
    for (int e = static_cast<int>(threadIdx.x); e < n; e += static_cast<int>(blockDim.x)) {
        const int t = t0 + e / head_dim;
        const size_t src = static_cast<size_t>(t) * kv_stride + base + static_cast<size_t>(e % head_dim);
        k_lds[e] = t < t_lim ? k[src] : 0.0f;
        v_lds[e] = t < t_lim ? v[src] : 0.0f;
    }
}

/* ATTN_QTILE consecutive queries of one head per block, one warp each. The
 * block stages `tile` keys of K and V in LDS and every warp reads them from
 * there: each key is fetched from memory once per block instead of once per
 * query. Per row, the key tiles, their order and the chunk folds are those of
 * attention_split_gqa_kernel. `tile` is a multiple of ATTN_KT. */
constexpr int ATTN_QTILE = 8;

template <int DPT, bool VEC4>
__global__ void __launch_bounds__(32 * ATTN_QTILE) attention_qtile_kernel(const float *q, const float *k, const float *v, float *out,
                                       int n_q, int n_kv, int n_heads, int n_kv_heads, int head_dim,
                                       int causal, const uint8_t *mask, const float *sinks,
                                       int tile) {
    extern __shared__ float attn_lds[];
    float *k_lds = attn_lds;
    float *v_lds = attn_lds + static_cast<size_t>(tile) * head_dim;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int q0 = static_cast<int>(blockIdx.x) * ATTN_QTILE;
    const int qi = q0 + warp;
    const int h = static_cast<int>(blockIdx.y);
    const bool active = qi < n_q;
    const int kv_h = h / (n_heads / n_kv_heads);
    const float scale = det_rsqrtf(static_cast<float>(head_dim));
    const size_t kv_stride = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
    const int qrow = active ? qi : q0;
    const float *qh = q + (static_cast<size_t>(qrow) * n_heads + h) * head_dim;
    float *oh = out + (static_cast<size_t>(qrow) * n_heads + h) * head_dim;
    /* The caller checked head_dim % 4 and the alignment of q: every row
     * offset is a multiple of head_dim floats. */
    constexpr bool vec4 = VEC4;
    const int nvec = vec4 ? head_dim / 4 : head_dim;
    const int t_max = attn_row_tmax(qi, n_q, n_kv, causal);
    const int t_end = active ? (mask ? n_kv : min(n_kv, t_max + 1)) : 0;
    /* Keys the block needs: up to the last row's end. */
    const int q_last = min(n_q, q0 + ATTN_QTILE) - 1;
    const int block_end = mask ? n_kv : min(n_kv, attn_row_tmax(q_last, n_q, n_kv, causal) + 1);
    AttnRow<DPT> r;
    attn_row_init(r, qh, lane, head_dim, nvec, vec4, sinks, h);
    /* Finished chunks before the current one, as the fold tree's carry stack. */
    AttnFoldRegs<DPT> rg;
    AttnFoldStack<DPT> st;
    attn_tree_init(rg, st);
    int next_fold = ATTN_CHUNK;
    for (int t0 = 0; t0 < block_end; t0 += tile) {
        __syncthreads();
        attn_stage_tile(k_lds, v_lds, k, v, t0, n_kv, tile, head_dim, kv_stride, kv_h);
        __syncthreads();
        const int stop = min(t_end, t0 + tile);
        for (int s = t0; s < stop; s += ATTN_KT) {
            if (s == next_fold) {
                next_fold += ATTN_CHUNK;
                attn_tree_push<DPT>(rg, st, r);
                attn_row_reset(r);
            }
            const int t = s + lane / 2;
            const bool valid = t < t_end && (!mask || mask[static_cast<size_t>(qi) * static_cast<size_t>(n_kv) +
                                                            static_cast<size_t>(t)] != 0);
            const size_t at = static_cast<size_t>(s - t0) * head_dim;
            attn_tile_step(r, k_lds + at, v_lds + at, valid, lane, head_dim, nvec, vec4, scale);
        }
    }
    if (active) {
        if (!attn_tree_empty(rg, st)) {
            attn_tree_push<DPT>(rg, st, r);
            attn_tree_finish<DPT>(rg, st, r);
        }
        attn_row_store(r, oh, lane, head_dim, nvec, vec4);
    }
}

/* Query heads per attention_split_gqa_kernel block (a warp each). */
constexpr int ATTN_GQA_HEADS = 8;

/* One query, one chunk of keys per block: the block takes up to
 * ATTN_GQA_HEADS query heads of one key/value head (blockDim.x / 32 of them),
 * stages `tile` keys of K and V in LDS at a time, and every warp steps its row
 * from there, so each key is read once per block instead of once per query
 * head. With several chunks the chunk's online softmax goes to `part` (see
 * ATTN_PART) for attention_merge_kernel; with one, the row goes to `out`,
 * as the merge would write it. `tile` is a multiple of ATTN_KT. */
template <int DPT, bool VEC4>
__global__ void __launch_bounds__(32 * ATTN_GQA_HEADS) attention_split_gqa_kernel(
    const float *q, const float *k, const float *v, float *part, float *out, int n_kv, int n_heads,
    int n_kv_heads, int head_dim, int causal, const uint8_t *mask, const float *sinks, int n_chunks, int tile) {
    extern __shared__ float attn_lds[];
    float *k_lds = attn_lds;
    float *v_lds = attn_lds + static_cast<size_t>(tile) * head_dim;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int per_block = static_cast<int>(blockDim.x) / 32;
    const int per_kv = n_heads / n_kv_heads;
    const int groups = (per_kv + per_block - 1) / per_block;
    const int kv_h = static_cast<int>(blockIdx.x) / groups;
    const int h0 = kv_h * per_kv + (static_cast<int>(blockIdx.x) % groups) * per_block;
    const int h_end = min(kv_h * per_kv + per_kv, h0 + per_block);
    const int h = h0 + warp;
    const bool active = h < h_end;
    const int c = static_cast<int>(blockIdx.y);
    const float scale = det_rsqrtf(static_cast<float>(head_dim));
    const int t_max = attn_row_tmax(0, 1, n_kv, causal);
    const float *qh = q + static_cast<size_t>(active ? h : h0) * head_dim;
    const size_t kv_stride = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
    /* The caller checked head_dim % 4 and the alignment of q: every row
     * offset is a multiple of head_dim floats. */
    constexpr bool vec4 = VEC4;
    const int nvec = vec4 ? head_dim / 4 : head_dim;
    AttnRow<DPT> r;
    attn_row_init(r, qh, lane, head_dim, nvec, vec4, c == 0 ? sinks : nullptr, active ? h : h0);
    const int t_end = mask ? n_kv : min(n_kv, t_max + 1);
    const int t1 = min(t_end, (c + 1) * ATTN_CHUNK);
    for (int t0 = c * ATTN_CHUNK; t0 < t1; t0 += tile) {
        __syncthreads();
        attn_stage_tile(k_lds, v_lds, k, v, t0, t1, tile, head_dim, kv_stride, kv_h);
        __syncthreads();
        const int stop = min(t1, t0 + tile);
        for (int s = t0; s < stop; s += ATTN_KT) {
            const int t = s + lane / 2;
            const bool valid = t < t1 && (!mask || mask[t] != 0);
            const size_t at = static_cast<size_t>(s - t0) * head_dim;
            attn_tile_step(r, k_lds + at, v_lds + at, valid, lane, head_dim, nvec, vec4, scale);
        }
    }
    if (!active) {
        return;
    }
    if (n_chunks == 1) {
        attn_row_store(r, out + static_cast<size_t>(h) * head_dim, lane, head_dim, nvec, vec4);
        return;
    }
    float *dst = part + (static_cast<size_t>(h) * n_chunks + c) * attn_part<DPT>();
#pragma unroll
    for (int i = 0; i < attn_row_dims<DPT>(); ++i) {
        dst[lane * attn_row_dims<DPT>() + i] = r.acc[i];
    }
    if (lane == 0) {
        dst[32 * attn_row_dims<DPT>()] = r.m;
        dst[32 * attn_row_dims<DPT>() + 1] = r.l;
    }
}

/* Partials attention_merge_kernel folds itself, in one warp. Below that the
 * fold tree's lowest levels run as attention_merge_pair_kernel launches, one
 * per level: a launch costs more than a few folds, so short rows (up to 16
 * chunks, 2048 keys) take none. */
constexpr int ATTN_MERGE_WARP = 16;

/* The pair levels' last stride, doubled: the smallest power of two S with
 * ceil(n_chunks / S) <= ATTN_MERGE_WARP. After the levels with stride < S,
 * the partials at 0, S, 2S, ... hold the tree's aligned blocks, and the
 * tree's upper levels are the same tree over those. */
__host__ __device__ __forceinline__ int attn_merge_spacing(int n_chunks) {
    int spacing = 1;
    while ((n_chunks + spacing - 1) / spacing > ATTN_MERGE_WARP) {
        spacing *= 2;
    }
    return spacing;
}

/* One tree level of a decode row's partials (see ATTN_FOLD_STACK): block x is
 * the pair (2 * stride * x, + stride), block y the head; folds in place. */
template <int DPT>
__global__ void __launch_bounds__(32) attention_merge_pair_kernel(float *part, int n_chunks, int stride) {
    const int left = static_cast<int>(blockIdx.x) * 2 * stride;
    const int right = left + stride;
    const int h = static_cast<int>(blockIdx.y);
    if (right >= n_chunks) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    constexpr int dims = attn_row_dims<DPT>();
    float *base = part + static_cast<size_t>(h) * n_chunks * attn_part<DPT>();
    float *dst = base + static_cast<size_t>(left) * attn_part<DPT>();
    const float *src = base + static_cast<size_t>(right) * attn_part<DPT>();
    float acc[dims];
    float acc_b[dims];
#pragma unroll
    for (int i = 0; i < dims; ++i) {
        acc[i] = dst[lane * dims + i];
        acc_b[i] = src[lane * dims + i];
    }
    float m = dst[32 * dims];
    float l = dst[32 * dims + 1];
    attn_fold<DPT>(m, l, acc, src[32 * dims], src[32 * dims + 1], acc_b);
#pragma unroll
    for (int i = 0; i < dims; ++i) {
        dst[lane * dims + i] = acc[i];
    }
    if (lane == 0) {
        dst[32 * dims] = m;
        dst[32 * dims + 1] = l;
    }
}

/* The fold tree's levels from attn_merge_spacing up, in place in `part`, then
 * writes the row. The levels below have already run
 * (attention_merge_pair_kernel). One warp owns the row's partials; each lane
 * reads back only what it wrote, except m and l, which lane 0 writes. */
template <int DPT>
__global__ void __launch_bounds__(32) attention_merge_kernel(const float *q, float *part, float *out, int n_heads,
                                       int head_dim, int n_chunks) {
    const int qi = static_cast<int>(blockIdx.x);
    const int h = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const float *qh = q + (static_cast<size_t>(qi) * n_heads + h) * head_dim;
    float *oh = out + (static_cast<size_t>(qi) * n_heads + h) * head_dim;
    const bool vec4 = (head_dim % 4) == 0 && head_dim > 0 &&
                      (reinterpret_cast<uintptr_t>(qh) % 16u) == 0;
    const int nvec = vec4 ? head_dim / 4 : head_dim;
    float *src = part + (static_cast<size_t>(qi) * n_heads + h) * n_chunks * attn_part<DPT>();
    constexpr int dims = attn_row_dims<DPT>();
    for (int stride = attn_merge_spacing(n_chunks); stride < n_chunks; stride *= 2) {
        for (int left = 0; left + stride < n_chunks; left += 2 * stride) {
            float *dst = src + static_cast<size_t>(left) * attn_part<DPT>();
            const float *pc = dst + static_cast<size_t>(stride) * attn_part<DPT>();
            float acc[dims];
            float acc_b[dims];
#pragma unroll
            for (int i = 0; i < dims; ++i) {
                acc[i] = dst[lane * dims + i];
                acc_b[i] = pc[lane * dims + i];
            }
            float m = dst[32 * dims];
            float l = dst[32 * dims + 1];
            attn_fold<DPT>(m, l, acc, pc[32 * dims], pc[32 * dims + 1], acc_b);
#pragma unroll
            for (int i = 0; i < dims; ++i) {
                dst[lane * dims + i] = acc[i];
            }
            __threadfence_block();
            if (lane == 0) {
                dst[32 * dims] = m;
                dst[32 * dims + 1] = l;
            }
            __threadfence_block();
        }
    }
    AttnRow<DPT> r;
#pragma unroll
    for (int i = 0; i < dims; ++i) {
        r.acc[i] = src[lane * dims + i];
    }
    r.m = src[32 * dims];
    r.l = src[32 * dims + 1];
    attn_row_store(r, oh, lane, head_dim, nvec, vec4);
}

/* Per-device scratch for chunk partials, grow-only. The lock is held from the
 * split launch through the merge launch; a switch to another stream first
 * synchronizes the device (the old stream may be gone). */
constexpr int ATTN_PART_DEVICES = 16;
struct AttnPartScratch {
    std::mutex mu;
    float *p = nullptr;
    size_t cap = 0;
    hipStream_t stream = nullptr;
};
static AttnPartScratch g_attn_part[ATTN_PART_DEVICES];

/* gfx12 WMMA flash prefill (experimental.row_variant_kernels). One wave owns 16
 * query rows of one head. QK^T and PV are v_wmma_f32_16x16x16_f16 tiles with
 * f32 accumulation; the online softmax is the attn_row_step recurrence (max,
 * exp, rescale) per 16-key tile. Q, K, P and V are rounded to f16, so a row's
 * bits differ from the decode kernels (relative error ~1e-3): the engine runs
 * this kernel only in prefill with the option on. Each Q row and each K row
 * (one token, over head_dim) is scaled by a power of two into [2^13, 2^14)
 * before rounding and the scale is taken out exactly in f32; each V row (one
 * token) is scaled down only, with the inverse folded into P. Every scale
 * depends on one row or token alone, so a row's bits do not depend on keys
 * past its causal limit.
 *
 * Q fragments stay in registers for the whole key loop. The loop stops at the
 * causal limit of the block's last row. K and V are only read. head_dim must
 * be a multiple of 16 and at most 128 (the host checks). Non-gfx12 images
 * leave the body empty; other devices do not launch this kernel. */
__global__ void __launch_bounds__(32) attention_wmma_prefill_kernel(const float *q, const float *k,
                                                                    const float *v, float *out, int n_q,
                                                                    int n_kv, int n_heads, int n_kv_heads,
                                                                    int head_dim, int causal) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    using f16x8 = __attribute__((ext_vector_type(8))) _Float16;
    using f32x8 = __attribute__((ext_vector_type(8))) float;
    const int lane = static_cast<int>(threadIdx.x);
    const int row0 = static_cast<int>(blockIdx.x) * 16;
    const int h = static_cast<int>(blockIdx.y);
    if (h >= n_heads) {
        return;
    }
    const int kv_h = h / (n_heads / n_kv_heads);
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const size_t q_stride = static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
    const size_t kv_stride = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
    /* head_dim <= 128: at most eight 16-wide tiles of the head. */
    constexpr int kMaxDTiles = 8;
    const int n_dtiles = head_dim / 16;
    const int half = 8 * (lane / 16); /* this lane's K offset inside a 16-panel */

    /* Q: this lane holds row row0 + lane%16, dims dt*16 + half + 0..7. */
    const int q_row = row0 + (lane % 16);
    const float *qr = q + (static_cast<size_t>(q_row < n_q ? q_row : 0) * n_heads + h) * head_dim;
    float qmax = 0.0f;
#pragma unroll
    for (int dt = 0; dt < kMaxDTiles; ++dt) {
        if (dt < n_dtiles && q_row < n_q) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                qmax = fmaxf(qmax, fabsf(qr[dt * 16 + half + e]));
            }
        }
    }
    qmax = fmaxf(qmax, __shfl_xor(qmax, 16, 32));
    const int q_sh = wmma_f16_shift(qmax);
    const float qs = ldexpf(1.0f, q_sh);
    f16x8 q_frag[kMaxDTiles];
#pragma unroll
    for (int dt = 0; dt < kMaxDTiles; ++dt) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            q_frag[dt][e] = static_cast<_Float16>(0.0f);
            if (dt < n_dtiles && q_row < n_q) {
                q_frag[dt][e] = static_cast<_Float16>(qr[dt * 16 + half + e] * qs);
            }
        }
    }
    /* Accumulator element g is row (L/16)*8+g; its Q shift lives in that lane. */
    float row_unscale[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        row_unscale[g] = ldexpf(scale, -__shfl(q_sh, (lane / 16) * 8 + g, 32));
    }

    float m_run[8];
    float l_run[8];
    f32x8 acc[kMaxDTiles];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        m_run[g] = -1.0e30f;
        l_run[g] = 0.0f;
    }
#pragma unroll
    for (int dt = 0; dt < kMaxDTiles; ++dt) {
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            acc[dt][g] = 0.0f;
        }
    }

    /* Keys past the last row's causal limit are masked for every row here. */
    const int last_row = (row0 + 15 < n_q ? row0 + 15 : n_q - 1);
    const int t_lim = attn_row_tmax(last_row, n_q, n_kv, causal) + 1;
    const int t_end = t_lim < n_kv ? t_lim : n_kv;

    for (int k0 = 0; k0 < t_end; k0 += 16) {
        /* K: this lane's B column is token k0 + lane%16 (dims as for Q). */
        const int k_tok = k0 + (lane % 16);
        const float *kr = k + static_cast<size_t>(k_tok < n_kv ? k_tok : 0) * kv_stride +
                          static_cast<size_t>(kv_h) * head_dim;
        float kmax = 0.0f;
        if (k_tok < n_kv) {
            for (int dt = 0; dt < n_dtiles; ++dt) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    kmax = fmaxf(kmax, fabsf(kr[dt * 16 + half + e]));
                }
            }
        }
        kmax = fmaxf(kmax, __shfl_xor(kmax, 16, 32));
        const int k_sh = wmma_f16_shift(kmax);
        const float ks = ldexpf(1.0f, k_sh);
        f32x8 scores;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            scores[g] = 0.0f;
        }
#pragma unroll
        for (int dt = 0; dt < kMaxDTiles; ++dt) {
            if (dt < n_dtiles) {
                f16x8 b_frag;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    b_frag[e] = static_cast<_Float16>(0.0f);
                    if (k_tok < n_kv) {
                        b_frag[e] = static_cast<_Float16>(kr[dt * 16 + half + e] * ks);
                    }
                }
                scores = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(q_frag[dt], b_frag, scores);
            }
        }
        /* Score element g: row (L/16)*8+g, token k_tok (this lane's column). */
        const float tok_unscale = ldexpf(1.0f, -k_sh);

        float p_lane[8];
        float alpha[8];
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int qrow = row0 + (lane / 16) * 8 + g;
            const int t_max = qrow < n_q ? attn_row_tmax(qrow, n_q, n_kv, causal) : -1;
            const bool valid = qrow < n_q && k_tok < n_kv && k_tok <= t_max;
            const float s = valid ? (scores[g] * row_unscale[g]) * tok_unscale : -1.0e30f;
            float vmax = s;
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) {
                vmax = fmaxf(vmax, __shfl_xor(vmax, off, 32));
            }
            const float m_new = fmaxf(m_run[g], vmax);
            alpha[g] = __expf(m_run[g] - m_new);
            const float p = valid ? __expf(s - m_new) : 0.0f;
            float l_tile = p;
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) {
                l_tile += __shfl_xor(l_tile, off, 32);
            }
            l_run[g] = alpha[g] * l_run[g] + l_tile;
            m_run[g] = m_new;
            p_lane[g] = p;
        }

        /* V scale, per token so a row never depends on keys past its causal
         * limit (a scale shared over a 16-key tile did): lane L takes token
         * k0 + L%16's largest |v| over head_dim (lane L^16 the other half) and
         * scales that token's V down by a power of two when it would not fit
         * f16 (never up, at most 2^15). The inverse goes into P, which stays
         * <= 2^15. V past 2^29 still overflows f16. */
        float v_tmax = 0.0f;
        if (k_tok < n_kv) {
            const float *vr = v + static_cast<size_t>(k_tok) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
            for (int dt = 0; dt < n_dtiles; ++dt) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    v_tmax = fmaxf(v_tmax, fabsf(vr[dt * 16 + half + e]));
                }
            }
        }
        v_tmax = fmaxf(v_tmax, __shfl_xor(v_tmax, 16, 32));
        int v_sh = wmma_f16_shift(v_tmax);
        v_sh = v_sh > 0 ? 0 : (v_sh < -15 ? -15 : v_sh);
        /* This lane's P and V operands cover tokens k0 + half + e; token
         * k0 + j's shift lives in lane j. */
        float tok_vs[8];
        float tok_pinv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int sh = __shfl(v_sh, half + e, 32);
            tok_vs[e] = ldexpf(1.0f, sh);
            tok_pinv[e] = ldexpf(1.0f, -sh);
        }

        /* P as the A operand: lane L needs row L%16 at tokens half + 0..7. */
        f16x8 p_frag;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            p_frag[e] = static_cast<_Float16>(0.0f);
        }
#pragma unroll
        for (int gsrc = 0; gsrc < 8; ++gsrc) {
            const float vreg = p_lane[gsrc];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int r = lane % 16;
                const int src = (r / 8) * 16 + half + e;
                const float got = __shfl(vreg, src, 32);
                if ((r % 8) == gsrc) {
                    p_frag[e] = static_cast<_Float16>(got * tok_pinv[e]);
                }
            }
        }
#pragma unroll
        for (int dt = 0; dt < kMaxDTiles; ++dt) {
            if (dt >= n_dtiles) {
                continue;
            }
            /* V: this lane's B column is dim dt*16 + lane%16, tokens k0 + half + 0..7
             * (lane L^16 holds the other 8 tokens of the column), each scaled
             * by its own token's power of two. */
            const int d = dt * 16 + (lane % 16);
            f16x8 b_frag;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int token = k0 + half + e;
                const float vv = token < n_kv
                                     ? v[static_cast<size_t>(token) * kv_stride + static_cast<size_t>(kv_h) * head_dim + d]
                                     : 0.0f;
                b_frag[e] = static_cast<_Float16>(vv * tok_vs[e]);
            }
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                acc[dt][g] *= alpha[g];
            }
            acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(p_frag, b_frag, acc[dt]);
        }
    }
#pragma unroll
    for (int dt = 0; dt < kMaxDTiles; ++dt) {
        if (dt >= n_dtiles) {
            continue;
        }
        const int d0 = dt * 16;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int qrow = row0 + (lane / 16) * 8 + g;
            const int d = d0 + (lane % 16);
            if (qrow < n_q && l_run[g] > 0.0f) {
                out[static_cast<size_t>(qrow) * q_stride + static_cast<size_t>(h) * head_dim + d] =
                    acc[dt][g] / l_run[g];
            }
        }
    }
#else
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
#endif
}

static void warn_attn_no_scratch_once() {
    static std::atomic<int> printed{0};
    if (printed.exchange(1, std::memory_order_relaxed) == 0) {
        std::fprintf(stderr,
                     "allpaca: attention decode fell back to the Q-tiled kernel because scratch was unavailable\n");
    }
}

/* One query, a head_dim in the DPT bucket: the tiled kernels, sized on DPT.
 * n_q == 1: attention_split_gqa_kernel, its chunks in parallel plus a merge
 * when the row is longer than one chunk. Several rows: the Q-tiled kernel.
 * Both give a row the same bits. */
template <int DPT>
static void launch_attention_tiled(const float *q, const float *k, const float *v, float *out, int n_q,
                                   int n_kv, int n_heads, int n_kv_heads, int head_dim, int causal,
                                   const uint8_t *mask, const float *sinks, hipStream_t stream) {
    /* About 2048 floats of K (and of V) per tile, at least one key tile: 16
     * keys at head_dim 128 or 256. On an MI50 (3199-token prompt, 14 layers)
     * 4096 lost occupancy and 1024 paid more barriers; for a 2051-key decode
     * row at head_dim 256, 32 keys were slower than 16. */
    const int tile = max(ATTN_KT, 2048 / head_dim / ATTN_KT * ATTN_KT);
    const size_t lds = 2 * static_cast<size_t>(tile) * head_dim * sizeof(float);
    const int n_chunks = n_kv > ATTN_CHUNK ? (n_kv + ATTN_CHUNK - 1) / ATTN_CHUNK : 1;
    /* Heads that share a key/value head read each key once. */
    const int per_kv = n_kv_heads > 0 ? n_heads / n_kv_heads : 1;
    const int per_block = min(per_kv, ATTN_GQA_HEADS);
    const int groups = (per_kv + per_block - 1) / per_block;
    /* float4 rows; the same for every row of the call (see the kernels). */
    const bool vec4 = (head_dim % 4) == 0 && (reinterpret_cast<uintptr_t>(q) % 16u) == 0;
    const dim3 split_grid(static_cast<unsigned>(n_kv_heads * groups), static_cast<unsigned>(n_chunks));
    if (n_q == 1 && n_chunks == 1) {
        hipLaunchKernelGGL((vec4 ? attention_split_gqa_kernel<DPT, true> : attention_split_gqa_kernel<DPT, false>),
                           split_grid, dim3(32 * per_block), lds, stream, q, k, v,
                           static_cast<float *>(nullptr), out, n_kv, n_heads, n_kv_heads, head_dim, causal, mask,
                           sinks, n_chunks, tile);
        return;
    }
    if (n_q == 1) {
        int dev = 0;
        if (hipGetDevice(&dev) == hipSuccess && dev >= 0 && dev < ATTN_PART_DEVICES) {
            AttnPartScratch &s = g_attn_part[dev];
            std::lock_guard<std::mutex> lock(s.mu);
            bool ok = true;
            if (s.p && s.stream != stream && hipDeviceSynchronize() != hipSuccess) {
                ok = false;
            }
            const size_t need = static_cast<size_t>(n_heads) * n_chunks * attn_part<DPT>();
            if (ok && s.cap < need) {
                if (s.p) {
                    (void)hipDeviceSynchronize();
                    (void)hipFree(s.p);
                    s.p = nullptr;
                    s.cap = 0;
                }
                const size_t cap = need > (size_t(1) << 18) ? need : (size_t(1) << 18);
                if (hipMalloc(&s.p, cap * sizeof(float)) == hipSuccess) {
                    s.cap = cap;
                } else {
                    s.p = nullptr;
                    ok = false;
                }
            }
            if (ok) {
                s.stream = stream;
                hipLaunchKernelGGL((vec4 ? attention_split_gqa_kernel<DPT, true> : attention_split_gqa_kernel<DPT, false>),
                                   split_grid, dim3(32 * per_block), lds, stream, q, k,
                                   v, s.p, static_cast<float *>(nullptr), n_kv, n_heads, n_kv_heads, head_dim, causal,
                                   mask, sinks, n_chunks, tile);
                /* The fold tree's lowest levels (see attn_merge_spacing). */
                const int spacing = attn_merge_spacing(n_chunks);
                for (int stride = 1; stride < spacing; stride *= 2) {
                    const int pairs = (n_chunks - stride + 2 * stride - 1) / (2 * stride);
                    hipLaunchKernelGGL(attention_merge_pair_kernel<DPT>,
                                       dim3(static_cast<unsigned>(pairs), static_cast<unsigned>(n_heads)), dim3(32),
                                       0, stream, s.p, n_chunks, stride);
                }
                hipLaunchKernelGGL(attention_merge_kernel<DPT>, dim3(1, static_cast<unsigned>(n_heads)),
                                   dim3(32), 0, stream, q, s.p, out, n_heads, head_dim, n_chunks);
                return;
            }
        }
        /* No scratch: the Q-tiled kernel computes the same chunked row. */
        warn_attn_no_scratch_once();
    }
    dim3 block(32 * ATTN_QTILE);
    dim3 grid(static_cast<unsigned>((n_q + ATTN_QTILE - 1) / ATTN_QTILE),
              static_cast<unsigned>(n_heads));
    hipLaunchKernelGGL((vec4 ? attention_qtile_kernel<DPT, true> : attention_qtile_kernel<DPT, false>),
                       grid, block, lds, stream, q, k, v, out, n_q, n_kv,
                       n_heads, n_kv_heads, head_dim, causal, mask, sinks, tile);
}

/* One query: attention_split_gqa_kernel, its chunks in parallel plus a merge
 * when the row is longer than one chunk. Several rows: the Q-tiled kernel.
 * Both give a row the same bits. Head dims past 32 * ATTN_MAX_DPT (or
 * non-positive) take the per-row kernel, whichever the row count. */
static void launch_attention(const float *q, const float *k, const float *v, float *out, int n_q,
                             int n_kv, int n_heads, int n_kv_heads, int head_dim, int causal,
                             const uint8_t *mask, const float *sinks, hipStream_t stream) {
    switch (attn_dpt_bucket(head_dim)) {
    case 4:
        launch_attention_tiled<4>(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, mask,
                                  sinks, stream);
        return;
    case 8:
        launch_attention_tiled<8>(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, mask,
                                  sinks, stream);
        return;
    case 16:
        launch_attention_tiled<16>(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, mask,
                                   sinks, stream);
        return;
    default:
        break;
    }
    dim3 block(32);
    dim3 grid(static_cast<unsigned>(n_q), static_cast<unsigned>(n_heads));
    hipLaunchKernelGGL(attention_flash_kernel, grid, block, 0, stream, q, k, v, out, n_q, n_kv,
                       n_heads, n_kv_heads, head_dim, causal, mask, sinks);
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
    /* A rocBLAS handle belongs to the device that was current when it was
     * made (it picks that device's kernel library), so each device has its
     * own. A device where a GEMM failed is not asked again. */
    struct Dev {
        rocblas_handle handle = nullptr;
        bool tried = false;
    };
    static std::mutex mu;
    static Dev devs[16];
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess || device < 0 || device >= 16) {
        return ALLPACA_ERR_DEVICE;
    }
    std::lock_guard<std::mutex> lock(mu);
    Dev &d = devs[device];
    if (!d.tried) {
        d.tried = true;
        if (rocblas_create_handle(&d.handle) != rocblas_status_success) {
            d.handle = nullptr;
        }
    }
    if (!d.handle) {
        return ALLPACA_ERR_NOT_IMPLEMENTED;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    rocblas_status st = rocblas_set_stream(d.handle, stream);
    if (st == rocblas_status_success) {
        st = rocblas_sgemm(d.handle, rocblas_operation_transpose, rocblas_operation_none, n, m, k,
                           &alpha, b, k, a, k, &beta, c, n);
    }
    if (st != rocblas_status_success) {
        std::fprintf(stderr, "allpaca: rocBLAS sgemm failed on device %d (status %d); using the built-in GEMM there\n",
                     device, static_cast<int>(st));
        rocblas_destroy_handle(d.handle);
        d.handle = nullptr;
        return ALLPACA_ERR_DEVICE;
    }
    return ALLPACA_OK;
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
    /* ROCm 10 keeps one folder per GPU family under this one and finds it by
     * itself; pointing it here makes every lookup fail. Only the flat layout
     * (library files directly in the folder) wants the hint. */
    bool flat = false;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().filename().string().rfind("TensileLibrary", 0) == 0) {
            flat = true;
            break;
        }
    }
    if (!flat) {
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
    /* What the library answered for a shape: the algorithm lookup costs
     * ~30 ms, and a layer's attention GEMMs alternate between four shapes. */
    struct Shape {
        int32_t m = 0;
        int32_t n = 0;
        int32_t k = 0;
        int status = 0; /* 1 = algo usable, 2 = no algorithm */
        hipblasLtMatmulAlgo_t algo{};
        size_t algo_ws = 0;
    };
    static constexpr int SHAPES = 64;
    Shape shapes[SHAPES];
    int next_shape = 0;
    size_t algo_ws = 0;

    Shape *find(int32_t m, int32_t n, int32_t k) {
        for (Shape &s : shapes) {
            if (s.status != 0 && s.m == m && s.n == n && s.k == k) {
                return &s;
            }
        }
        return nullptr;
    }

    void remember(int32_t m, int32_t n, int32_t k, int status, const hipblasLtMatmulAlgo_t &algo, size_t ws) {
        Shape *s = find(m, n, k);
        if (!s) {
            s = &shapes[next_shape];
            next_shape = (next_shape + 1) % SHAPES;
        }
        s->m = m;
        s->n = n;
        s->k = k;
        s->status = status;
        s->algo = algo;
        s->algo_ws = ws;
    }
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
    const HipblasLtDev::Shape *known = st->find(m, n, k);
    if (known && known->status == 1) {
        algo = known->algo;
        ws = known->algo_ws;
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
    st->remember(m, n, k, 1, algo, ws);
    st->algo_ws = ws;
    return ALLPACA_OK;
}

/* Off unless ALLPACA_HIPBLASLT=1. On two RDNA4 cards (SDK 7.1, Windows) it
 * matched the fallback below at M = 187 and 512 on one card and offered no
 * algorithm on the other, while each algorithm lookup cost ~30 ms and the
 * first call ~2.7 s: a 23-row prefill of 21 layers took 2 s instead of 20 ms. */
static bool hipblaslt_wanted() {
    static const bool on = [] {
        const char *v = std::getenv("ALLPACA_HIPBLASLT");
        return v && v[0] == '1';
    }();
    return on;
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
    if (const HipblasLtDev::Shape *known = st.find(m, n, k); known && known->status == 2) {
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
    st.remember(m, n, k, 2, hipblasLtMatmulAlgo_t{}, 0);
    if (!logged_miss) {
        logged_miss = true;
        std::fprintf(stderr, "allpaca: hipBLASLt F32 GEMM miss m=%d n=%d k=%d algos=%d\n", m, n, k,
                     algos);
    }
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}
#endif

/* Row-exact mode (batched decode over several Requests): each row of an
 * M-row GEMM must equal that row computed alone. The small-M GEMV kernels
 * compute rows independently; the tiled / BLAS GEMMs used above them sum in
 * another order. With the mode on, wider M runs as row chunks on the GEMV
 * kernels. Host-side and per calling thread, like the launches it steers. */
static thread_local int g_exact_rows = 0;

extern "C" int32_t allpaca_hip_set_exact_rows(int32_t on) {
    const int prev = g_exact_rows;
    g_exact_rows = on != 0;
    return prev;
}

template <typename F>
static int exact_row_chunks(int32_t m, int32_t chunk, F &&launch) {
    for (int32_t r0 = 0; r0 < m; r0 += chunk) {
        const int rc = launch(r0, m - r0 < chunk ? m - r0 : chunk);
        if (rc != ALLPACA_OK) {
            return rc;
        }
    }
    return ALLPACA_OK;
}

static int launch_gemm_nt(const float *a, const float *b, float *c, int32_t m, int32_t n, int32_t k,
                          hipStream_t stream, int tiled) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    if (g_exact_rows && m > 8) {
        return exact_row_chunks(m, 8, [&](int32_t r0, int32_t rows) {
            return launch_gemm_nt(a + static_cast<size_t>(r0) * k, b,
                                  c + static_cast<size_t>(r0) * n, rows, n, k, stream, tiled);
        });
    }
    clear_last_error();
#ifdef ALLPACA_HAS_HIPBLASLT
    /* M=1 lm_head: hipBLASLt was ~378 ms vs ~0.5 ms custom GEMV. Keep Lt for wide/prefill only. */
    if (m > 8 && hipblaslt_wanted()) {
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
    /* A failed library attempt above must not fail the kernel that replaces it. */
    clear_last_error();
    /* Decode is always M=1; small-M prefill stays on the coalesced GEMV. */
    if (m <= 8 && tiled && n <= 2048) {
        hipLaunchKernelGGL(gemv_nt_colblock_kernel, dim3(static_cast<unsigned>(n), static_cast<unsigned>(m)),
                           dim3(256), 0, stream, a, b, c, m, n, k);
        return hip_status(hipGetLastError());
    }
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
    if (g_exact_rows && m > 8) {
        return exact_row_chunks(m, 8, [&](int32_t r0, int32_t rows) {
            return launch_gemm_nt_f16(a + static_cast<size_t>(r0) * k, b,
                                      c + static_cast<size_t>(r0) * n, rows, n, k, stream, tiled);
        });
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

/* gfx1200 / gfx1201 (RDNA4, the WMMA kernels' target), cached per device:
 * hipGetDeviceProperties is far too slow to call on every launch.
 * 0 = not asked yet, 1 = no, 2 = yes. Devices past the table are asked
 * every time (none exist on our machines). */
constexpr int GFX12_CACHE_DEVICES = 16;
static std::atomic<int> g_gfx12_arch[GFX12_CACHE_DEVICES];

static int device_arch_is_gfx12(int device) {
    if (device < 0) {
        return 0;
    }
    if (device < GFX12_CACHE_DEVICES) {
        const int cached = g_gfx12_arch[device].load(std::memory_order_relaxed);
        if (cached != 0) {
            return cached == 2;
        }
    }
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
        return 0;
    }
    const int yes = std::strncmp(prop.gcnArchName, "gfx1200", 7) == 0 ||
                    std::strncmp(prop.gcnArchName, "gfx1201", 7) == 0;
    if (device < GFX12_CACHE_DEVICES) {
        g_gfx12_arch[device].store(yes ? 2 : 1, std::memory_order_relaxed);
    }
    return yes;
}

extern "C" int32_t allpaca_hip_device_is_gfx12(int32_t device) { return device_arch_is_gfx12(device); }

/* Prefill WMMA scope (experimental.row_variant_kernels). WMMA attention
 * gives a row other bits than the decode kernels, so the engine
 * opens this scope only around a prompt's prefill chunk, never around decode,
 * verify or batched forwards, and only when the option is on. Exact-row mode
 * wins over it. Per calling thread, like g_exact_rows. */
static thread_local int g_prefill_wmma = 0;

/* Returns the previous value so a scope can restore it. */
extern "C" int32_t allpaca_hip_set_prefill_wmma(int32_t on) {
    const int prev = g_prefill_wmma;
    g_prefill_wmma = on != 0;
    return prev;
}

extern "C" int32_t allpaca_hip_prefill_wmma_active(void) { return g_prefill_wmma && !g_exact_rows; }

/* WMMA kernel launches: process-wide and by the calling thread (tests). */
static std::atomic<uint64_t> g_wmma_launches{0};
static thread_local uint64_t t_wmma_launches = 0;
static inline void note_wmma_launch() {
    g_wmma_launches.fetch_add(1, std::memory_order_relaxed);
    ++t_wmma_launches;
}

extern "C" uint64_t allpaca_hip_wmma_launches(void) { return g_wmma_launches.load(std::memory_order_relaxed); }
extern "C" uint64_t allpaca_hip_thread_wmma_launches(void) { return t_wmma_launches; }

/* gfx12 launches the WMMA tile. Every other arch keeps the existing f16 kernel. */
static int launch_gemm_nt_f16_wmma(const float *a, const uint16_t *b, float *c, int32_t m,
                                   int32_t n, int32_t k, hipStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    /* Exact-row mode and every non-gfx12 device stay on the GEMV/tiled kernel. */
    if (g_exact_rows || !device_arch_is_gfx12(dev)) {
        return launch_gemm_nt_f16(a, b, c, m, n, k, stream, 1);
    }
    clear_last_error();
    dim3 block(32);
    dim3 grid(static_cast<unsigned>((n + 15) / 16), static_cast<unsigned>((m + 15) / 16));
    hipLaunchKernelGGL(gemm_nt_f16_wmma_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    note_wmma_launch();
    return hip_status(hipGetLastError());
}

static int launch_gemm_nt_i8(const int8_t *a, const int8_t *b, float *c, int32_t m, int32_t n,
                             int32_t k, hipStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((n + 15) / 16), static_cast<unsigned>((m + 15) / 16));
    hipLaunchKernelGGL(gemm_nt_i8_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    return hip_status(hipGetLastError());
}

static int launch_gemm_nt_i8_wmma(const int8_t *a, const int8_t *b, float *c, int32_t m, int32_t n,
                                  int32_t k, hipStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    if (!device_arch_is_gfx12(dev)) {
        return launch_gemm_nt_i8(a, b, c, m, n, k, stream);
    }
    clear_last_error();
    dim3 block(32);
    dim3 grid(static_cast<unsigned>((n + 15) / 16), static_cast<unsigned>((m + 15) / 16));
    hipLaunchKernelGGL(gemm_nt_i8_wmma_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    note_wmma_launch();
    return hip_status(hipGetLastError());
}

/* Memory-bound BF16 GEMV: one 32-lane group per output column, each lane
 * reading 8 consecutive weights (16 bytes) per step, so a group streams 512
 * contiguous bytes of the row at a time. `k` is a multiple of 8 and `a`, `b`
 * are 16-byte aligned. */
__global__ void gemv_nt_bf16_wide_kernel(const float *a, const uint16_t *b, float *c, int m, int n,
                                         int k) {
    constexpr int GROUPS = 8;
    const int row = static_cast<int>(blockIdx.y);
    const int grp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int col = static_cast<int>(blockIdx.x) * GROUPS + grp;
    if (row >= m) {
        return;
    }
    float acc = 0.f;
    if (col < n) {
        const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
        const uint16_t *brow = b + static_cast<size_t>(col) * static_cast<size_t>(k);
        for (int p = lane * 8; p < k; p += 256) {
            const uint4 w = *reinterpret_cast<const uint4 *>(brow + p);
            const float4 x0 = *reinterpret_cast<const float4 *>(arow + p);
            const float4 x1 = *reinterpret_cast<const float4 *>(arow + p + 4);
            acc += x0.x * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.x)) +
                   x0.y * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.x >> 16)) +
                   x0.z * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.y)) +
                   x0.w * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.y >> 16)) +
                   x1.x * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.z)) +
                   x1.y * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.z >> 16)) +
                   x1.z * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.w)) +
                   x1.w * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.w >> 16));
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor(acc, off, 32);
    }
    if (col < n && lane == 0) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + col] = acc;
    }
}

/* gemv_nt_bf16_wide_kernel for 2..8 rows a block (grid y = blocks of 8 rows): every row shares
 * each weight load and sums as it would alone. On the MI50 at 16 rows: 10240x320 157 -> 113 us,
 * 512x2560 89 -> 66 us; one row stays on the wide kernel (this one is slower there). */
__global__ void gemv_nt_bf16_wide_rows_kernel(const float *a, const uint16_t *b, float *c, int m,
                                              int n, int k) {
    constexpr int GROUPS = 8;
    constexpr int MAX_M = 8;
    const int row0 = static_cast<int>(blockIdx.y) * MAX_M;
    const int rows = m - row0 < MAX_M ? m - row0 : MAX_M;
    const int grp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int col = static_cast<int>(blockIdx.x) * GROUPS + grp;
    float acc[MAX_M];
#pragma unroll
    for (int r = 0; r < MAX_M; ++r) {
        acc[r] = 0.f;
    }
    if (col < n) {
        const uint16_t *brow = b + static_cast<size_t>(col) * static_cast<size_t>(k);
        for (int p = lane * 8; p < k; p += 256) {
            const uint4 w = *reinterpret_cast<const uint4 *>(brow + p);
#pragma unroll
            for (int r = 0; r < MAX_M; ++r) {
                if (r < rows) {
                    const float *arow = a + static_cast<size_t>(row0 + r) * static_cast<size_t>(k);
                    const float4 x0 = *reinterpret_cast<const float4 *>(arow + p);
                    const float4 x1 = *reinterpret_cast<const float4 *>(arow + p + 4);
                    acc[r] += x0.x * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.x)) +
                              x0.y * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.x >> 16)) +
                              x0.z * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.y)) +
                              x0.w * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.y >> 16)) +
                              x1.x * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.z)) +
                              x1.y * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.z >> 16)) +
                              x1.z * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.w)) +
                              x1.w * allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.w >> 16));
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MAX_M; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 32);
        }
        if (col < n && lane == 0 && r < rows) {
            c[static_cast<size_t>(row0 + r) * static_cast<size_t>(n) + col] = v;
        }
    }
}

/* The same products with a whole block of 256 per output (grid n), every row (m <= 8) in the
 * block so the weights are read once: for few, long rows, where a 32-lane group per output leaves
 * most of the card idle. With grid (n, m) a block takes row blockIdx.y alone. Each row's sums run
 * as they would alone either way. */
__global__ void gemv_nt_bf16_splitk_kernel(const float *a, const uint16_t *b, float *c, int m, int n,
                                           int k) {
    constexpr int MAX_M = 8;
    __shared__ float part[MAX_M][8];
    if (gridDim.y > 1) {
        a += static_cast<size_t>(blockIdx.y) * static_cast<size_t>(k);
        c += static_cast<size_t>(blockIdx.y) * static_cast<size_t>(n);
        m = 1;
    }
    const int col = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const uint16_t *brow = b + static_cast<size_t>(col) * static_cast<size_t>(k);
    float acc[MAX_M];
#pragma unroll
    for (int r = 0; r < MAX_M; ++r) {
        acc[r] = 0.f;
    }
    for (int p = tid * 8; p < k; p += 2048) {
        const uint4 w = *reinterpret_cast<const uint4 *>(brow + p);
        const float w0 = allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.x));
        const float w1 = allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.x >> 16));
        const float w2 = allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.y));
        const float w3 = allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.y >> 16));
        const float w4 = allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.z));
        const float w5 = allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.z >> 16));
        const float w6 = allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.w));
        const float w7 = allpaca_bf16_bits_to_f32(static_cast<uint16_t>(w.w >> 16));
#pragma unroll
        for (int r = 0; r < MAX_M; ++r) {
            if (r < m) {
                const float *arow = a + static_cast<size_t>(r) * static_cast<size_t>(k);
                const float4 x0 = *reinterpret_cast<const float4 *>(arow + p);
                const float4 x1 = *reinterpret_cast<const float4 *>(arow + p + 4);
                acc[r] += x0.x * w0 + x0.y * w1 + x0.z * w2 + x0.w * w3 + x1.x * w4 + x1.y * w5 +
                          x1.z * w6 + x1.w * w7;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MAX_M; ++r) {
        if (r < m) {
            float v = acc[r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_xor(v, off, 32);
            }
            if (tid % 32 == 0) {
                part[r][tid / 32] = v;
            }
        }
    }
    __syncthreads();
    if (tid < m) {
        const float *q = part[tid];
        c[static_cast<size_t>(tid) * static_cast<size_t>(n) + col] =
            ((q[0] + q[1]) + (q[2] + q[3])) + ((q[4] + q[5]) + (q[6] + q[7]));
    }
}

/* Whether a BF16 GEMV of this shape takes the split-k kernel (by shape only, so every m agrees). */
static bool bf16_gemv_splitk(int32_t n, int32_t k) {
    return (n <= 64 && k >= 2048) || (n <= 1024 && k >= 8192);
}

/* ALLPACA_BF16_GEMV_WIDE=0 keeps the older shared-tile kernels. */
static bool bf16_gemv_wide() {
    static const bool on = [] {
        const char *v = std::getenv("ALLPACA_BF16_GEMV_WIDE");
        return !(v && v[0] == '0');
    }();
    return on;
}

static int launch_gemm_nt_bf16(const float *a, const uint16_t *b, float *c, int32_t m, int32_t n,
                               int32_t k, hipStream_t stream, int tiled) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    if (g_exact_rows && m > 8) {
        return exact_row_chunks(m, 8, [&](int32_t r0, int32_t rows) {
            return launch_gemm_nt_bf16(a + static_cast<size_t>(r0) * k, b,
                                       c + static_cast<size_t>(r0) * n, rows, n, k, stream, tiled);
        });
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
    if (bf16_gemv_wide() && k % 8 == 0 && (reinterpret_cast<uintptr_t>(a) & 15u) == 0 &&
        (reinterpret_cast<uintptr_t>(b) & 15u) == 0 && bf16_gemv_splitk(n, k)) {
        /* Few outputs: a block per row as well, or the card idles. */
        const unsigned rows_grid = n <= 64 ? static_cast<unsigned>(m) : 1u;
        hipLaunchKernelGGL(gemv_nt_bf16_splitk_kernel, dim3(static_cast<unsigned>(n), rows_grid),
                           dim3(256), 0, stream, a, b, c, m, n, k);
    } else if (bf16_gemv_wide() && k % 8 == 0 && (reinterpret_cast<uintptr_t>(a) & 15u) == 0 &&
        (reinterpret_cast<uintptr_t>(b) & 15u) == 0) {
        constexpr int GROUPS = 8;
        dim3 block(GROUPS * 32);
        const unsigned gx = static_cast<unsigned>((n + GROUPS - 1) / GROUPS);
        if (m > 1) {
            hipLaunchKernelGGL(gemv_nt_bf16_wide_rows_kernel, dim3(gx, static_cast<unsigned>((m + 7) / 8)),
                               block, 0, stream, a, b, c, m, n, k);
        } else {
            hipLaunchKernelGGL(gemv_nt_bf16_wide_kernel, dim3(gx, 1u), block, 0, stream, a, b, c, m,
                               n, k);
        }
    } else if (tiled) {
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

static int launch_q8_0_i8(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                          int32_t k, hipStream_t stream);

/* Q8_0 GEMVs (m <= 16) over rows of at least Q8_I8_MIN_K values run on int8
 * activations unless ALLPACA_Q8_F32_GEMV=1. The choice depends on the
 * weight's shape only, so a weight takes the same kernel at every m and
 * verify rows match decode. Measured on xeres: 5120-wide rows (Qwen3.8-27B)
 * gain at every m; 2048- and 4096-wide rows (Qwen3-Coder-Next) lose more to the extra
 * quantize launch and the lost f32 pair launch than they gain. */
constexpr int32_t Q8_I8_MIN_K = 5120;

static bool q8_0_i8_enabled() {
    static const bool on = [] {
        const char *v = std::getenv("ALLPACA_Q8_F32_GEMV");
        return !(v && v[0] == '1');
    }();
    return on;
}

static bool kq_wave64();

/* Activation precision must not depend on the card: narrow-row integer gains
 * on the MI50 otherwise change a worker's output from the gfx12 path. */
static bool q8_0_uses_i8(int32_t n, int32_t k) {
    (void)n;
    return q8_0_i8_enabled() && k >= Q8_I8_MIN_K;
}

extern "C" int allpaca_q8_0_gemv_rounds_activations(int32_t m, int32_t n, int32_t k) {
    return m >= 1 && m <= 16 && k % 32 == 0 && q8_0_uses_i8(n, k) ? 1 : 0;
}

static int launch_gemm_nt_q8_0(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                               int32_t k, hipStream_t stream, int intdot) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    if (g_exact_rows && m > 16) {
        return exact_row_chunks(m, 16, [&](int32_t r0, int32_t rows) {
            return launch_gemm_nt_q8_0(a + static_cast<size_t>(r0) * k, b,
                                       c + static_cast<size_t>(r0) * n, rows, n, k, stream, intdot);
        });
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
    // Decode and verify rows (m <= 16) share the int8 kernel, so a verify row
    // still reproduces decode bit for bit; the f32 kernels below are the
    // ALLPACA_Q8_F32_GEMV fallback.
    if (!intdot && q8_0_uses_i8(n, k)) {
        const int rc = launch_q8_0_i8(a, b, c, m, n, k, stream);
        if (rc != ALLPACA_ERR_INVALID) {
            return rc;
        }
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
        hipLaunchKernelGGL(gemv_nt_q8_0_f32_kernel, grid, block, 0, stream, a, b, c, m, n, k, b, c, n);
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
/* Staged Q8_0: eight 36-byte blocks (f32 scale, 32 int8) per 256 values. */
template <>
struct KqBlock<8> {
    static constexpr int bytes = 288;
};
/* MoESQ paired-4:8 FP4 (tensor type 200): eight 12-byte blocks per 256 values. */
template <>
struct KqBlock<48> {
    static constexpr int bytes = 96;
};

/* Twice the E2M1 value of a paired-4:8 code (bit 3 is the sign). */
__constant__ const int8_t k_p48_levels2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

__device__ inline float kq_p48_half(uint32_t h) {
    const uint32_t e = (h >> 10) & 0x1fu;
    const uint32_t man = h & 0x3ffu;
    float v;
    if (e == 0) {
        v = static_cast<float>(man) * 5.9604644775390625e-8f;
    } else {
        const uint32_t bits = ((e + 112u) << 23) | (man << 13);
        __builtin_memcpy(&v, &bits, 4);
    }
    return (h & 0x8000u) ? -v : v;
}

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
/* Which order the rows' activation loads take in the int8 GEMV kernels
 * (K-quant and ROCmFP4). Same arithmetic either way; only the generated code
 * differs. Measured on Qwen3.8 27B prefill (679 tokens):
 *   - clang 24 (ROCm 10.1) waits for each row's loads in turn when they sit
 *     inside the row loop: ROCmFP4 5.63 s, Q5_K mix 7.46 s. With every row's
 *     loads issued first: ROCmFP4 4.95 s, Q5_K mix 6.67 s.
 *   - clang 21 (SDK 7.1) already overlaps them: ROCmFP4 4.96 s, Q5_K mix
 *     6.45 s; loads-first costs it about 2.5% on ROCmFP4 (5.10 s). */
#ifndef ALLPACA_I8_LOADS_FIRST
#if defined(__clang_major__) && __clang_major__ >= 22
#define ALLPACA_I8_LOADS_FIRST 1
#else
#define ALLPACA_I8_LOADS_FIRST 0
#endif
#endif

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

/* Packed-byte subtract without saturation (llama.cpp #27962, #29478).
 * Q3_K codes are 0..7, so subtracting 4 stays in int8 and
 * `dot4(q - 4, a)` equals `dot4(q, a) - 4 * sum(a)`. */
__device__ inline uint32_t allpaca_vsub4(uint32_t a, uint32_t b) {
    const uint32_t a_large = a | 0x80808080u;
    const uint32_t b_small = b & 0x7f7f7f7fu;
    const uint32_t low7 = a_large - b_small;
    const uint32_t flip = (a ^ ~b) & 0x80808080u;
    return low7 ^ flip;
}

/* One block of the int8 K-quant GEMV: columns from `bx`, rows `row0 .. +MR`. */
template <int QT, int MR, int LPC, bool LF = ALLPACA_I8_LOADS_FIRST != 0>
__device__ inline void kquant_i8_block(const int8_t *aq, const float *ad, const uint8_t *b,
                                       float *c, int m, int n, int k, int bx, int row0) {
    constexpr int BB = KqBlock<QT>::bytes;
    constexpr int W = 128 / LPC;
    constexpr int NC = W / 4;
    constexpr int LPG = LPC / 4;
    constexpr uint32_t ONES = 0x01010101u;
    const int sl = static_cast<int>(threadIdx.x) % LPC;
    const int col = bx * (static_cast<int>(blockDim.x) / LPC) +
                    static_cast<int>(threadIdx.x) / LPC;
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
            if constexpr (QT == 8) {
                // Lane sl takes every LPC-th 32-value block, summed as
                // `gemv_nt_q8_0_i8_kernel` does, so both give the same bits.
                for (int t = sl; t < 8; t += LPC) {
                    const uint8_t *q8 = blk + t * 36;
                    const float ws = *reinterpret_cast<const float *>(q8);
                    const uint32_t *qp = reinterpret_cast<const uint32_t *>(q8 + 4);
                    uint32_t q[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        q[j] = qp[j];
                    }
                    const int bi = ib * 8 + t;
#pragma unroll
                    for (int r = 0; r < MR; ++r) {
                        const uint32_t *ap = reinterpret_cast<const uint32_t *>(arow[r] + bi * 32);
                        int32_t dot = 0;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            dot = i8dot4(ap[j], q[j], dot);
                        }
                        acc[r] = fmaf(ws * drow[r][bi], static_cast<float>(dot), acc[r]);
                    }
                }
            } else if constexpr (QT == 48) {
                // Half of every 8 weights are pruned in pairs: each chunk of 8
                // is one dot4 of its four kept weights with the activations
                // at their places. Lane sl takes every LPC-th 32-value block.
                for (int t = sl; t < 8; t += LPC) {
                    uint32_t w[3];
                    __builtin_memcpy(w, blk + t * 12, 12);
                    const float ws = 0.5f * kq_p48_half(w[2] >> 16);
                    uint32_t q[4];
                    int first[4], second[4];
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const uint32_t nib = (w[2] >> (4 * c)) & 0xfu;
                        const uint32_t pairs = w[c >> 1] >> (16 * (c & 1));
                        first[c] = nib ? __builtin_ctz(nib) : 0;
                        second[c] = nib ? 31 - __builtin_clz(nib) : 0;
                        q[c] = nib == 0u
                                   ? 0u
                                   : (static_cast<uint32_t>(static_cast<uint8_t>(k_p48_levels2[pairs & 0xfu])) |
                                      (static_cast<uint32_t>(static_cast<uint8_t>(k_p48_levels2[(pairs >> 4) & 0xfu])) << 8) |
                                      (static_cast<uint32_t>(static_cast<uint8_t>(k_p48_levels2[(pairs >> 8) & 0xfu])) << 16) |
                                      (static_cast<uint32_t>(static_cast<uint8_t>(k_p48_levels2[(pairs >> 12) & 0xfu])) << 24));
                    }
                    const int bi = ib * 8 + t;
#pragma unroll
                    for (int r = 0; r < MR; ++r) {
                        const int8_t *ap = arow[r] + bi * 32;
                        int32_t dot = 0;
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            uint16_t lo, hi;
                            __builtin_memcpy(&lo, ap + c * 8 + 2 * first[c], 2);
                            __builtin_memcpy(&hi, ap + c * 8 + 2 * second[c], 2);
                            dot = i8dot4(static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16), q[c], dot);
                        }
                        acc[r] = fmaf(ws * drow[r][bi], static_cast<float>(dot), acc[r]);
                    }
                }
            } else if constexpr (QT == 3) {
                // hmask[32] qs[64] scales[12] d: weight = d * (sc - 32) * (q2 | h << 2 - 4).
                const float d = allpaca_f16le_to_f32(blk + 108);
                const uint8_t *sc = blk + 96;
                // `q - 4` packed saves a dot per word: 0.45-0.64x the time at 8 rows on
                // RDNA4 and 0.87x on the MI50, but on the MI50 2 rows of 5120^2 ran 12%
                // slower (65.3 -> 73.2 us), so gfx9 keeps `dot(q) - 4 * dot(1)` up to 2
                // rows. Both give the same int32.
#if defined(__GFX9__)
                constexpr bool SUB4 = MR > 2;
#else
                constexpr bool SUB4 = true;
#endif
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    uint32_t q[NC];
#pragma unroll
                    for (int t = 0; t < NC; ++t) {
                        const uint32_t qs = kq_u32_from_u16(blk + 32 + h * 32 + j0 + 4 * t);
                        const uint32_t hm = kq_u32_from_u16(blk + j0 + 4 * t);
                        const uint32_t v =
                            ((qs >> (2 * g)) & 0x03030303u) | (((hm >> (4 * h + g)) & ONES) << 2);
                        q[t] = SUB4 ? allpaca_vsub4(v, 0x04040404u) : v;
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
                            if constexpr (!SUB4) {
                                sum = i8dot4(av[t], ONES, sum);
                            }
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
                    if constexpr (LF) {
                        uint32_t av[MR][NC];
                        float ds[MR];
#pragma unroll
                        for (int r = 0; r < MR; ++r) {
                            kq_load_u32s<NC>(arow[r] + bi * 32 + j0, av[r]);
                            ds[r] = drow[r][bi];
                        }
#pragma unroll
                        for (int r = 0; r < MR; ++r) {
                            int32_t dot = 0, sum = 0;
#pragma unroll
                            for (int t = 0; t < NC; ++t) {
                                dot = i8dot4(av[r][t], q[t], dot);
                                sum = i8dot4(av[r][t], ONES, sum);
                            }
                            acc[r] += s * ds[r] * static_cast<float>(dot - 32 * sum);
                        }
                    } else {
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
                if constexpr (LF) {
                    uint32_t alo[MR][NC], ahi[MR][NC];
                    float dsl[MR], dsh[MR];
#pragma unroll
                    for (int r = 0; r < MR; ++r) {
                        kq_load_u32s<NC>(arow[r] + bl * 32 + j0, alo[r]);
                        kq_load_u32s<NC>(arow[r] + bl * 32 + 32 + j0, ahi[r]);
                        dsl[r] = drow[r][bl];
                        dsh[r] = drow[r][bl + 1];
                    }
#pragma unroll
                    for (int r = 0; r < MR; ++r) {
                        int32_t il = 0, sl_ = 0, ih = 0, sh = 0;
#pragma unroll
                        for (int t = 0; t < NC; ++t) {
                            il = i8dot4(alo[r][t], lo[t], il);
                            sl_ = i8dot4(alo[r][t], ONES, sl_);
                            ih = i8dot4(ahi[r][t], hi[t], ih);
                            sh = i8dot4(ahi[r][t], ONES, sh);
                        }
                        acc[r] += dsl[r] * (dl * static_cast<float>(il) -
                                            ml * static_cast<float>(sl_)) +
                                  dsh[r] * (dh * static_cast<float>(ih) -
                                            mh * static_cast<float>(sh));
                    }
                } else {
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

template <int QT, int MR, int LPC, bool LF = ALLPACA_I8_LOADS_FIRST != 0>
__global__ void gemv_nt_kquant_i8_kernel(const int8_t *aq, const float *ad, const uint8_t *b,
                                         float *c, int m, int n, int k) {
    kquant_i8_block<QT, MR, LPC, LF>(aq, ad, b, c, m, n, k, static_cast<int>(blockIdx.x),
                                     static_cast<int>(blockIdx.y) * MR);
}

/* Q3_K..Q6_K int8 GEMV for 9..16 rows (a prefill pass). `kquant_i8_block`
 * reads every row's activations from global memory per lane, so 16 rows ran
 * on load latency (Q3_K 6144x2560 at ~0.7 TMAC/s). Here a block's 256
 * threads copy the 16 rows of one super-block (4 KB + scales) to LDS, the
 * next one while computing this one, and every lane keeps its 16 sums in
 * registers. Each lane sums the same terms in the same order as
 * `kquant_i8_block` and the columns reduce over the same lanes, so a row
 * gets the bits it gets alone. */
constexpr int KQ_TILE_ROWS = 16;

template <int QT, int LPC>
__global__ __launch_bounds__(256) void gemv_nt_kquant_i8_tile_kernel(
    const int8_t *aq, const float *ad, const uint8_t *b, float *c, int m, int n, int k) {
    constexpr int R = KQ_TILE_ROWS;
    constexpr int BB = KqBlock<QT>::bytes;
    constexpr int W = 128 / LPC;
    constexpr int NC = W / 4;
    constexpr int LPG = LPC / 4;
    constexpr uint32_t ONES = 0x01010101u;
    __shared__ uint4 sa[2][R * 16];
    __shared__ float sd[2][R * 8];
    const int tid = static_cast<int>(threadIdx.x);
    const int sl = tid % LPC;
    const int col = static_cast<int>(blockIdx.x) * (256 / LPC) + tid / LPC;
    const bool live = col < n;
    const int row0 = static_cast<int>(blockIdx.y) * R;
    const int rows = m - row0 < R ? m - row0 : R;
    const int nblocks = k / 256;
    const int kb = k / 32;
    const uint8_t *brow =
        b + static_cast<size_t>(live ? col : 0) * static_cast<size_t>(nblocks) * BB;
    const int g = sl / LPG;
    const int j0 = (sl % LPG) * W;
    // Thread tid copies 16 bytes of row tid / 16 and, below 128, one scale.
    const int sr = tid >> 4;
    const int8_t *asrc = aq + static_cast<size_t>(row0 + (sr < rows ? sr : 0)) * k + (tid & 15) * 16;
    const int dr = tid >> 3;
    const float *dsrc = ad + static_cast<size_t>(row0 + (dr < rows ? dr : 0)) * kb + (tid & 7);
    uint4 na = sr < rows ? *reinterpret_cast<const uint4 *>(asrc) : make_uint4(0, 0, 0, 0);
    float nd = tid < R * 8 && dr < rows ? dsrc[0] : 0.f;
    sa[0][tid] = na;
    if (tid < R * 8) {
        sd[0][tid] = nd;
    }
    __syncthreads();
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r] = 0.f;
    }
    for (int ib = 0; ib < nblocks; ++ib) {
        const int buf = ib & 1;
        const bool more = ib + 1 < nblocks;
        if (more) {
            na = sr < rows ? *reinterpret_cast<const uint4 *>(asrc + (ib + 1) * 256)
                           : make_uint4(0, 0, 0, 0);
            nd = tid < R * 8 && dr < rows ? dsrc[(ib + 1) * 8] : 0.f;
        }
        const int8_t *sab = reinterpret_cast<const int8_t *>(sa[buf]);
        const float *sdb = sd[buf];
        if (live) {
            const uint8_t *blk = brow + static_cast<size_t>(ib) * BB;
            if constexpr (QT == 3) {
                const float d = allpaca_f16le_to_f32(blk + 108);
                const uint8_t *sc = blk + 96;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    // q - 4 as signed bytes: one dot gives `dot - 4 * sum`.
                    uint32_t q[NC];
#pragma unroll
                    for (int t = 0; t < NC; ++t) {
                        const uint32_t qs = kq_u32_from_u16(blk + 32 + h * 32 + j0 + 4 * t);
                        const uint32_t hm = kq_u32_from_u16(blk + j0 + 4 * t);
                        const uint32_t v =
                            ((qs >> (2 * g)) & 0x03030303u) | (((hm >> (4 * h + g)) & ONES) << 2);
                        q[t] = allpaca_vsub4(v, 0x04040404u);
                    }
                    const int is = h * 8 + 2 * g + j0 / 16;
                    const int s6 = ((sc[is & 7] >> (4 * (is >> 3))) & 0xF) |
                                   (((sc[8 + (is & 3)] >> (2 * (is >> 2))) & 3) << 4);
                    const float s = d * static_cast<float>(s6 - 32);
                    const int lb = h * 4 + g;
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        uint32_t av[NC];
                        kq_load_u32s<NC>(sab + r * 256 + lb * 32 + j0, av);
                        int32_t dot = 0;
#pragma unroll
                        for (int t = 0; t < NC; ++t) {
                            dot = i8dot4(av[t], q[t], dot);
                        }
                        acc[r] += s * sdb[r * 8 + lb] * static_cast<float>(dot);
                    }
                }
            } else if constexpr (QT == 6) {
                const float d = allpaca_f16le_to_f32(blk + 208);
                const int8_t *sc = reinterpret_cast<const int8_t *>(blk + 192);
                const int lo_off = (g & 1) * 32 + j0;
                const int nib = (g >> 1) * 4;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    // q - 32 as signed bytes: one dot gives `dot - 32 * sum`.
                    uint32_t q[NC];
#pragma unroll
                    for (int t = 0; t < NC; ++t) {
                        const uint32_t ql = kq_u32_from_u16(blk + h * 64 + lo_off + 4 * t);
                        const uint32_t qh = kq_u32_from_u16(blk + 128 + h * 32 + j0 + 4 * t);
                        const uint32_t v =
                            ((ql >> nib) & 0x0F0F0F0Fu) | (((qh >> (2 * g)) & 0x03030303u) << 4);
                        q[t] = ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
                    }
                    const float s = d * static_cast<float>(sc[h * 8 + j0 / 16 + 2 * g]);
                    const int lb = h * 4 + g;
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        uint32_t av[NC];
                        kq_load_u32s<NC>(sab + r * 256 + lb * 32 + j0, av);
                        int32_t dot = 0;
#pragma unroll
                        for (int t = 0; t < NC; ++t) {
                            dot = i8dot4(av[t], q[t], dot);
                        }
                        acc[r] += s * sdb[r * 8 + lb] * static_cast<float>(dot);
                    }
                }
            } else {
                const float d = allpaca_f16le_to_f32(blk);
                const float dmin = allpaca_f16le_to_f32(blk + 2);
                uint32_t sc1 = 0, m1 = 0, sc2 = 0, m2 = 0;
                kq_scale_min_pair(blk + 4, g, sc1, m1, sc2, m2);
                const uint32_t *qp = reinterpret_cast<const uint32_t *>(
                    blk + (QT == 5 ? 48 : 16) + g * 32 + j0);
                uint32_t lo[NC], hi[NC];
#pragma unroll
                for (int t = 0; t < NC; ++t) {
                    const uint32_t qv = qp[t];
                    uint32_t hb_lo = 0, hb_hi = 0;
                    if constexpr (QT == 5) {
                        const uint32_t qh =
                            *reinterpret_cast<const uint32_t *>(blk + 16 + j0 + 4 * t);
                        hb_lo = (qh >> (2 * g)) & ONES;
                        hb_hi = (qh >> (2 * g + 1)) & ONES;
                    }
                    lo[t] = (qv & 0x0F0F0F0Fu) | (hb_lo << 4);
                    hi[t] = ((qv >> 4) & 0x0F0F0F0Fu) | (hb_hi << 4);
                }
                const float dl = d * static_cast<float>(sc1);
                const float ml = dmin * static_cast<float>(m1);
                const float dh = d * static_cast<float>(sc2);
                const float mh = dmin * static_cast<float>(m2);
                const int lb = 2 * g;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    uint32_t alo[NC], ahi[NC];
                    kq_load_u32s<NC>(sab + r * 256 + lb * 32 + j0, alo);
                    kq_load_u32s<NC>(sab + r * 256 + lb * 32 + 32 + j0, ahi);
                    int32_t il = 0, sl_ = 0, ih = 0, sh = 0;
#pragma unroll
                    for (int t = 0; t < NC; ++t) {
                        il = i8dot4(alo[t], lo[t], il);
                        sl_ = i8dot4(alo[t], ONES, sl_);
                        ih = i8dot4(ahi[t], hi[t], ih);
                        sh = i8dot4(ahi[t], ONES, sh);
                    }
                    acc[r] += sdb[r * 8 + lb] * (dl * static_cast<float>(il) -
                                                 ml * static_cast<float>(sl_)) +
                              sdb[r * 8 + lb + 1] * (dh * static_cast<float>(ih) -
                                                     mh * static_cast<float>(sh));
                }
            }
        }
        if (more) {
            sa[buf ^ 1][tid] = na;
            if (tid < R * 8) {
                sd[buf ^ 1][tid] = nd;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = LPC / 2; off > 0; off >>= 1) {
            v += __shfl_down(v, off, LPC);
        }
        if (live && sl == 0 && r < rows) {
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
    /* The activation the int8 rows hold (see `KqActCopy`). */
    const float *src = nullptr;
    int src_m = 0;
    int src_k = 0;
    hipStream_t src_stream = nullptr;
    bool src_device = false;
    uint64_t src_addresses = ~uint64_t(0);
    uint64_t epoch = 0;
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
        s.src = nullptr;
        if (hipMalloc(&s.p, cap) != hipSuccess) {
            s.p = nullptr;
            return nullptr;
        }
        s.cap = cap;
    }
    s.stream = stream;
    return &s;
}

/* The int8 copy of an m x k activation `a` in the scratch's rows `aq` and
 * per-block scales `ad`, for one GEMV writing the m x n output `c`; hold it
 * inside the scratch lock until that GEMV is enqueued. The constructor
 * quantizes `a` unless the scratch already holds exactly it: same source,
 * shape and stream, a source in device memory (a host or managed buffer can
 * change without any enqueue), and no device write enqueued since the GEMV
 * that last read the copy. The destructor marks the copy current, unless a
 * write meanwhile could have changed `a`: one from another thread (the
 * process-wide count moved more than this thread's), or this GEMV's own
 * output overlapping `a`. GEMVs sharing one input (q/k/v, gate/up) then
 * quantize it once, with the same bits. ALLPACA_KQ_ACT_REUSE=0 quantizes
 * every time. */
class KqActCopy {
  public:
    KqActCopy(KqActScratch *s, const float *a, int8_t *aq, float *ad, int m, int k,
              hipStream_t stream, const float *c, int n)
        : s_(s), a_(a), a_end_(a + static_cast<size_t>(m) * k), c_(c),
          c_end_(c + static_cast<size_t>(m) * n), epoch_(device_write_epoch()),
          own_(t_device_writes) {
        if (reuse_enabled() && s->src == a && s->src_m == m && s->src_k == k &&
            s->src_stream == stream && s->src_device && s->epoch == epoch_) {
            ++t_act_quant_reuses;
            return;
        }
        const int kb = k / 32;
        hipLaunchKernelGGL(quantize_rows_q8_kernel,
                           dim3(static_cast<unsigned>((kb + 7) / 8), static_cast<unsigned>(m)),
                           dim3(256), 0, stream, a, aq, ad, m, k);
        if (!reuse_enabled() || hipPeekAtLastError() != hipSuccess) {
            s->src = nullptr;
            return;
        }
        // Looked up again once a free or (un)registration may have given the
        // address to another kind of memory.
        const uint64_t addresses = g_address_changes.load(std::memory_order_relaxed);
        if (s->src != a || s->src_addresses != addresses) {
            hipPointerAttribute_t attr{};
            s->src_device = hipPointerGetAttributes(&attr, a) == hipSuccess &&
                            attr.type == hipMemoryTypeDevice && !attr.isManaged;
            (void)hipGetLastError();
            s->src_addresses = addresses;
        }
        s->src = a;
        s->src_m = m;
        s->src_k = k;
        s->src_stream = stream;
        s->epoch = 0;
    }

    ~KqActCopy() {
        const uint64_t epoch = device_write_epoch();
        const bool only_ours = epoch - epoch_ == t_device_writes - own_;
        const auto at = [](const float *p) { return reinterpret_cast<uintptr_t>(p); };
        const bool overlaps = at(c_) < at(a_end_) && at(a_) < at(c_end_);
        if (s_->src == a_ && only_ours && !overlaps) {
            s_->epoch = epoch;
        } else {
            s_->src = nullptr;
        }
    }

    KqActCopy(const KqActCopy &) = delete;
    KqActCopy &operator=(const KqActCopy &) = delete;

  private:
    static bool reuse_enabled() {
        static const bool on = [] {
            const char *v = std::getenv("ALLPACA_KQ_ACT_REUSE");
            return !(v && v[0] == '0');
        }();
        return on;
    }

    KqActScratch *s_;
    const float *a_, *a_end_, *c_, *c_end_;
    uint64_t epoch_, own_;
};

extern "C" uint64_t allpaca_hip_act_quant_reuses(void) { return t_act_quant_reuses; }

/* Rows that share one weight block load in the int8 kernels. A block costs
 * the same however many of its rows are live, so the rows are spread evenly
 * over the fewest blocks of at most 8: m = 5 runs as one block of 5 and
 * m = 9 as two of 5, not as blocks of 8. */
static inline int kq_i8_rows_per_block(int m) {
    const int blocks = (m + 7) / 8;
    const int mr = (m + blocks - 1) / blocks;
    return mr == 3 ? 4 : mr;
}

/* Whether the bound device runs 64-lane wavefronts (gfx9, the MI50). */
static bool kq_wave64() {
    static std::atomic<int> seen[KQ_ACT_DEVICES] = {};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= KQ_ACT_DEVICES) {
        return false;
    }
    int known = seen[dev].load(std::memory_order_relaxed);
    if (known == 0) {
        int lanes = 32;
        if (hipDeviceGetAttribute(&lanes, hipDeviceAttributeWarpSize, dev) != hipSuccess) {
            lanes = 32;
        }
        known = lanes == 64 ? 2 : 1;
        seen[dev].store(known, std::memory_order_relaxed);
    }
    return known == 2;
}

/* Lanes per output column of the int8 K-quant GEMV: 8 on every card. Lanes set
 * the order a column sums in, so they decide its bits, and the MI50 must give
 * the bits every other worker gives. 32 lanes ran Q6_K 2560x6144 in 34 us
 * against 69 and 16 rows in 84 against 231, but moved Q6_K and narrow outputs
 * by the last bits, which the next int8 GEMV turns into 1% (S1). */
static int kq_i8_lanes(int qt, int n) {
    (void)qt;
    (void)n;
    return 8;
}

/* On wave32 cards (RDNA), verify batches (2..8 rows) and prefill blocks
 * issue every row's activation loads before any of the row's math
 * (`kquant_i8_block<..., LF = true>`). Same arithmetic in the same order, so
 * the same bits as the m = 1 decode kernel; only the schedule differs.
 * Without it clang 21 (HIP SDK 7.1) keeps Q4_K at 8 rows and Q5_K at 5..8
 * rows in ~60 VGPRs and waits for each row's loads in turn (board P6,
 * measured, 5120x5120: Q4_K m=8 68 -> 41 us on the RX 9070, 108 -> 66 on the
 * RX 9060 XT; m=1 is 29 / 53). Decode (m = 1) and wave64 cards (the MI50, not
 * measured) keep the compiler default. Override: 0 = that pick, 1 = off,
 * 2 = on; it only serves tests and benches. Without one,
 * ALLPACA_KQ_LOADS_FIRST=0 runs every row count off (an SDK 7.1 build's
 * schedule before P6) and =1 on, for A/B runs of a whole server. */
static std::atomic<int> g_kq_lf_override{0};

static int kq_lf_env() {
    static const int mode = [] {
        const char *v = std::getenv("ALLPACA_KQ_LOADS_FIRST");
        if (v && v[0] == '0') {
            return 1;
        }
        if (v && v[0] == '1') {
            return 2;
        }
        return 0;
    }();
    return mode;
}

extern "C" int allpaca_kquant_i8_set_loads_first(int32_t mode) {
    if (mode < 0 || mode > 2) {
        return ALLPACA_ERR_INVALID;
    }
    g_kq_lf_override.store(mode, std::memory_order_relaxed);
    return ALLPACA_OK;
}

template <int QT, int LPC, bool LF = ALLPACA_I8_LOADS_FIRST != 0>
static void kq_i8_dispatch(dim3 grid, dim3 block, hipStream_t stream, int mr, const int8_t *aq,
                           const float *ad, const uint8_t *b, float *c, int m, int n, int k) {
#define KQ_I8_LAUNCH(MRV)                                                                          \
    hipLaunchKernelGGL((gemv_nt_kquant_i8_kernel<QT, MRV, LPC, LF>), grid, block, 0, stream, aq,   \
                       ad, b, c, m, n, k)
    if (mr == 1) {
        KQ_I8_LAUNCH(1);
    } else if (mr == 2) {
        KQ_I8_LAUNCH(2);
    } else if (mr == 4) {
        KQ_I8_LAUNCH(4);
    } else if (mr == 5) {
        KQ_I8_LAUNCH(5);
    } else if (mr == 6) {
        KQ_I8_LAUNCH(6);
    } else if (mr == 7) {
        KQ_I8_LAUNCH(7);
    } else {
        KQ_I8_LAUNCH(8);
    }
#undef KQ_I8_LAUNCH
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
    KqActCopy act(scratch, a, aq, ad, m, k, stream, c, n);
    // 8 lanes x 32 weights per column: at 32 lanes x 8 weights the per-row
    // A loads and scale math dominated on RDNA (Q5_K m=4 cost 1.7x m=1; now
    // ~1.05x).
    const int lpc = kq_i8_lanes(QT, n);
    dim3 block(256);
    const int cols = 256 / lpc;
    const unsigned gx = static_cast<unsigned>((n + cols - 1) / cols);
    const int mr = kq_i8_rows_per_block(m);
    const dim3 grid(gx, static_cast<unsigned>((m + mr - 1) / mr));
    if constexpr (QT >= 3 && QT <= 6) {
        // MI50, 16 rows: Q3_K 6144x2560 232 -> 91 us, Q6_K 7680x2560
        // 231 -> 84, Q4_K/Q5_K 2560x6144 ~163 -> ~106. At 9 rows Q4_K lost
        // (92 -> 106 us), so Q4_K/Q5_K start at 12.
        if (kq_wave64() && m >= ((QT == 4 || QT == 5) ? 12 : 9)) {
            const dim3 tgrid(gx, static_cast<unsigned>((m + KQ_TILE_ROWS - 1) / KQ_TILE_ROWS));
            hipLaunchKernelGGL((gemv_nt_kquant_i8_tile_kernel<QT, 8>), tgrid, block, 0, stream, aq,
                               ad, b, c, m, n, k);
            return hip_status(hipGetLastError());
        }
        int lf = g_kq_lf_override.load(std::memory_order_relaxed);
        if (lf == 0) {
            lf = kq_lf_env();
        }
        if (lf == 0 && m >= 2 && !kq_wave64()) {
            lf = 2;
        }
        if (lf == 2) {
            kq_i8_dispatch<QT, 8, true>(grid, block, stream, mr, aq, ad, b, c, m, n, k);
            return hip_status(hipGetLastError());
        }
        if (lf == 1) {
            kq_i8_dispatch<QT, 8, false>(grid, block, stream, mr, aq, ad, b, c, m, n, k);
            return hip_status(hipGetLastError());
        }
    }
    kq_i8_dispatch<QT, 8>(grid, block, stream, mr, aq, ad, b, c, m, n, k);
    return hip_status(hipGetLastError());
}

/* Q8_0 GEMV on int8 activations (the K-quant mmvq scheme for 36-byte staged
 * Q8_0 blocks: f32 scale, 32 int8). LPC lanes per column take every LPC-th
 * 32-value block; MR rows share each weight block load. A row's sum order
 * does not depend on MR, so m = 1 decode and m <= 16 verify rows agree bit
 * for bit. */
template <int MR>
__global__ void gemv_nt_q8_0_i8_kernel(const int8_t *aq, const float *ad, const uint8_t *b,
                                       float *c, int m, int n, int k) {
    constexpr int LPC = 8;
    const int sl = static_cast<int>(threadIdx.x) % LPC;
    const int col = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / LPC) +
                    static_cast<int>(threadIdx.x) / LPC;
    const int row0 = static_cast<int>(blockIdx.y) * MR;
    const bool live = col < n;
    const int kb = k / 32;
    const uint8_t *brow = b + static_cast<size_t>(live ? col : 0) * static_cast<size_t>(kb) * 36;
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
        for (int bi = sl; bi < kb; bi += LPC) {
            const uint8_t *blk = brow + static_cast<size_t>(bi) * 36;
            const float ws = *reinterpret_cast<const float *>(blk);
            const uint32_t *qp = reinterpret_cast<const uint32_t *>(blk + 4);
            uint32_t q[8];
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                q[t] = qp[t];
            }
#pragma unroll
            for (int r = 0; r < MR; ++r) {
                const uint4 *ap = reinterpret_cast<const uint4 *>(arow[r] + bi * 32);
                const uint4 a0 = ap[0];
                const uint4 a1 = ap[1];
                int32_t dot = 0;
                dot = i8dot4(a0.x, q[0], dot);
                dot = i8dot4(a0.y, q[1], dot);
                dot = i8dot4(a0.z, q[2], dot);
                dot = i8dot4(a0.w, q[3], dot);
                dot = i8dot4(a1.x, q[4], dot);
                dot = i8dot4(a1.y, q[5], dot);
                dot = i8dot4(a1.z, q[6], dot);
                dot = i8dot4(a1.w, q[7], dot);
                acc[r] = fmaf(ws * drow[r][bi], static_cast<float>(dot), acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = LPC / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 32);
        }
        if (live && sl == 0 && row0 + r < m) {
            c[static_cast<size_t>(row0 + r) * static_cast<size_t>(n) + col] = v;
        }
    }
}

/* Returns ALLPACA_ERR_INVALID when the int8 path cannot run (caller falls back). */
static int launch_q8_0_i8(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                          int32_t k, hipStream_t stream) {
    if (m <= 0 || m > 16 || k % 32 != 0 || reinterpret_cast<uintptr_t>(b) % 4 != 0) {
        return ALLPACA_ERR_INVALID;
    }
    const size_t aq_bytes = (static_cast<size_t>(m) * k + 255) & ~size_t(255);
    const size_t need = aq_bytes + static_cast<size_t>(m) * (k / 32) * sizeof(float);
    std::unique_lock<std::mutex> lock;
    KqActScratch *scratch = kq_act_acquire(stream, need, lock);
    if (!scratch) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    uint8_t *base = static_cast<uint8_t *>(scratch->p);
    int8_t *aq = reinterpret_cast<int8_t *>(base);
    float *ad = reinterpret_cast<float *>(base + aq_bytes);
    KqActCopy act(scratch, a, aq, ad, m, k, stream, c, n);
    constexpr int LPC = 8;
    // Smaller blocks spread narrow weights over the MI50's CUs (640x2560
    // 7.5 -> 5.9 us); the sum order does not depend on the block size.
    const int threads = kq_wave64() ? 128 : 256;
    dim3 block(threads);
    const unsigned gx = static_cast<unsigned>((n + (threads / LPC) - 1) / (threads / LPC));
    const int mr = kq_i8_rows_per_block(m);
    const dim3 grid(gx, static_cast<unsigned>((m + mr - 1) / mr));
    if (mr == 1) {
        hipLaunchKernelGGL(gemv_nt_q8_0_i8_kernel<1>, grid, block, 0, stream, aq, ad, b, c, m, n, k);
    } else if (mr == 2) {
        hipLaunchKernelGGL(gemv_nt_q8_0_i8_kernel<2>, grid, block, 0, stream, aq, ad, b, c, m, n, k);
    } else if (mr == 4) {
        hipLaunchKernelGGL(gemv_nt_q8_0_i8_kernel<4>, grid, block, 0, stream, aq, ad, b, c, m, n, k);
    } else if (mr == 5) {
        hipLaunchKernelGGL(gemv_nt_q8_0_i8_kernel<5>, grid, block, 0, stream, aq, ad, b, c, m, n, k);
    } else if (mr == 6) {
        hipLaunchKernelGGL(gemv_nt_q8_0_i8_kernel<6>, grid, block, 0, stream, aq, ad, b, c, m, n, k);
    } else if (mr == 7) {
        hipLaunchKernelGGL(gemv_nt_q8_0_i8_kernel<7>, grid, block, 0, stream, aq, ad, b, c, m, n, k);
    } else {
        hipLaunchKernelGGL(gemv_nt_q8_0_i8_kernel<8>, grid, block, 0, stream, aq, ad, b, c, m, n, k);
    }
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

/* Lanes per column the int8 K-quant GEMV launches for type `qt` (3..6) and `n`
 * outputs on the bound device, so host models can sum in the same order. */
extern "C" int allpaca_kquant_i8_lanes(int32_t qt, int32_t n) {
    if (qt < 3 || qt > 6 || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    return kq_i8_lanes(qt, n);
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

/* ggml Q2_0: 64 weights / 18 bytes (f16 d, then 2 bits per weight, weight j in
 * bits 2*(j%4) of byte j/4). value = d * (code - 1); d may be negative. Each
 * output column runs on four lanes, lane l taking blocks l, l+4, ... */
/* Dot of one ggml Q2_0 block (18 bytes) with 64 activations, scale applied. With the block on
 * two bytes and the activations on sixteen, the codes load two at a time and the activations four
 * at a time; the sums run in the same order either way. The crumbs stay as
 * `float(q >> s & 3) - 1`, which compiles to v_cvt_f32_ubyte plus a subtract: a v_perm_b32 to
 * signed bytes (llama.cpp #29927) then needs a sign fix and a convert per crumb, and measured
 * 1.6-1.9x slower on gfx1200/gfx1201 and 1.2-1.6x on gfx906 (board P5, 2026-10-24). */
__device__ inline float q2_0_block_dot(const uint8_t *blk, const float *x) {
    float s = 0.f;
    if ((reinterpret_cast<uintptr_t>(blk) & 1u) == 0 && (reinterpret_cast<uintptr_t>(x) & 15u) == 0) {
        const uint16_t *codes = reinterpret_cast<const uint16_t *>(blk + 2);
        const float4 *xv = reinterpret_cast<const float4 *>(x);
        for (int w = 0; w < 8; ++w) {
            const uint32_t pair = codes[w];
            for (int half = 0; half < 2; ++half) {
                const uint32_t q = (pair >> (8 * half)) & 0xffu;
                const float4 v = xv[2 * w + half];
                s += (static_cast<float>(q & 3u) - 1.f) * v.x;
                s += (static_cast<float>((q >> 2) & 3u) - 1.f) * v.y;
                s += (static_cast<float>((q >> 4) & 3u) - 1.f) * v.z;
                s += (static_cast<float>(q >> 6) - 1.f) * v.w;
            }
        }
    } else {
        for (int j = 0; j < 16; ++j) {
            const uint32_t q = blk[2 + j];
            s += (static_cast<float>(q & 3u) - 1.f) * x[4 * j];
            s += (static_cast<float>((q >> 2) & 3u) - 1.f) * x[4 * j + 1];
            s += (static_cast<float>((q >> 4) & 3u) - 1.f) * x[4 * j + 2];
            s += (static_cast<float>(q >> 6) - 1.f) * x[4 * j + 3];
        }
    }
    return allpaca_f16le_to_f32(blk) * s;
}

__global__ void gemv_nt_q2_0_kernel(const float *a, const uint8_t *b, float *c, int m, int n,
                                    int k) {
    constexpr int LANES = 4;
    constexpr int COLS = 256 / LANES;
    __shared__ float part[256];
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid % LANES;
    const int col = static_cast<int>(blockIdx.x) * COLS + tid / LANES;
    const int nblocks = k / 64;
    float acc = 0.f;
    if (row < m && col < n) {
        const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
        const uint8_t *brow = b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 18;
        for (int ib = lane; ib < nblocks; ib += LANES) {
            acc += q2_0_block_dot(brow + static_cast<size_t>(ib) * 18, arow + static_cast<size_t>(ib) * 64);
        }
    }
    part[tid] = acc;
    __syncthreads();
    if (lane == 0 && row < m && col < n) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] =
            (part[tid] + part[tid + 1]) + (part[tid + 2] + part[tid + 3]);
    }
}

extern "C" int allpaca_gemm_nt_q2_0(const float *a, const uint8_t *b, float *c, int32_t m,
                                    int32_t n, int32_t k, allpaca_stream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || k % 64 != 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    dim3 block(256);
    dim3 grid(static_cast<unsigned>((n + 63) / 64), static_cast<unsigned>(m));
    hipLaunchKernelGGL(gemv_nt_q2_0_kernel, grid, block, 0, as_stream(stream), a, b, c, m, n, k);
    return hip_status(hipGetLastError());
}

/* Routed experts held as Q2_0 slots: one slot is gate[ff, dim], up[ff, dim],
 * down[dim, ff], each packed row-major. */
#define ALLPACA_Q2_MOE_MAX_PICKS 32
struct Q2MoePicks {
    int32_t slot[ALLPACA_Q2_MOE_MAX_PICKS];
    float w[ALLPACA_Q2_MOE_MAX_PICKS];
};

/* c + the dot of four signed bytes of a with four bytes of b (b's bytes are 0..127 or signed;
 * here they are the 2-bit codes 0..3). The scalar branch matches
 * `fallback::dot4_i8_scalar` and warns once; gfx900 is the build target that
 * takes it. */
#if !defined(__gfx906__) && !defined(__gfx908__) && !defined(__gfx90a__) && !defined(__gfx942__) && \
    !defined(__gfx1100__) && !defined(__gfx1101__) && !defined(__gfx1102__) &&                       \
    !defined(__gfx1151__) && !defined(__gfx1200__) && !defined(__gfx1201__)
__device__ int g_allpaca_dot4_scalar_warned = 0;
#endif
__device__ inline int allpaca_dot4_i8(int a, int b, int c) {
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    if (atomicCAS(&g_allpaca_dot4_scalar_warned, 0, 1) == 0) {
        printf("allpaca: dot4 i8 using the scalar fallback\n");
    }
    const int8_t *x = reinterpret_cast<const int8_t *>(&a);
    const int8_t *y = reinterpret_cast<const int8_t *>(&b);
    return c + x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
#endif
}

/* A row of activations rounded to 8 bits per 64-value block, as allpaca_cpu's Q2_0Input:
 * codes round(v * 127 / max|block|), the block's step max / 127 and the sum of its codes. The
 * codes are kept in dot order: within each 16 values, value 4i + k sits at 4k + i, so the codes
 * of one 2-bit field of four code bytes line up with one 32-bit word. */
struct Q2Act {
    const int8_t *codes;
    const float *step;
    const float *sum;
};

/* The Q2_0 expert kernels' scratch (q2_0_moe_scratch_floats on the Rust side): the hidden rows
 * h (rows * n_pick * ff), then steps and sums of the inputs and of h, then their codes. */
struct Q2MoeScratch {
    float *h;
    float *x_step, *x_sum, *h_step, *h_sum;
    int8_t *x_codes, *h_codes;
};

static Q2MoeScratch q2_moe_scratch(float *base, int rows, int n_pick, int dim, int ff) {
    const size_t nh = static_cast<size_t>(rows) * n_pick * ff;
    const size_t nx = static_cast<size_t>(rows) * dim;
    Q2MoeScratch s;
    s.h = base;
    s.x_step = base + nh;
    s.x_sum = s.x_step + nx / 64;
    s.h_step = s.x_sum + nx / 64;
    s.h_sum = s.h_step + nh / 64;
    s.x_codes = reinterpret_cast<int8_t *>(s.h_sum + nh / 64);
    s.h_codes = s.x_codes + nx;
    return s;
}

/* Round `blocks` 64-value blocks of x to Q2Act form: 32 lanes a block, values j and j + 32 on
 * lane j (the max and the integer sum do not depend on the order). */
__global__ void q2_0_round_act_kernel(const float *x, int8_t *codes, float *step, float *sum,
                                      int blocks) {
    const int b = static_cast<int>(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    const int j = static_cast<int>(threadIdx.x % 32);
    if (b >= blocks) {
        return;
    }
    const float *src = x + static_cast<size_t>(b) * 64;
    int8_t *dst = codes + static_cast<size_t>(b) * 64;
    const float v0 = src[j];
    const float v1 = src[j + 32];
    float mx = fmaxf(fabsf(v0), fabsf(v1));
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor(mx, off, 32));
    }
    const bool live = mx > 0.f && isfinite(mx);
    const float inv = live ? 127.f / mx : 0.f;
    const int c0 = live ? static_cast<int>(roundf(v0 * inv)) : 0;
    const int c1 = live ? static_cast<int>(roundf(v1 * inv)) : 0;
    dst[(j & ~15) + 4 * (j & 3) + ((j >> 2) & 3)] = static_cast<int8_t>(c0);
    dst[32 + (j & ~15) + 4 * (j & 3) + ((j >> 2) & 3)] = static_cast<int8_t>(c1);
    int total = c0 + c1;
    for (int off = 16; off > 0; off >>= 1) {
        total += __shfl_xor(total, off, 32);
    }
    if (j == 0) {
        step[b] = live ? mx / 127.f : 0.f;
        sum[b] = static_cast<float>(total);
    }
}

static void q2_0_round_act(const float *x, int8_t *codes, float *step, float *sum, size_t n,
                           hipStream_t stream) {
    const int blocks = static_cast<int>(n / 64);
    hipLaunchKernelGGL(q2_0_round_act_kernel, dim3(static_cast<unsigned>((blocks + 7) / 8)),
                       dim3(256), 0, stream, x, codes, step, sum, blocks);
}

/* Dot of one Q2_0 block (18 bytes, on two bytes) with block b of a rounded row: the block's
 * scale times the step times (sum of code * activation code - sum of activation codes), as
 * allpaca_cpu's gemv_q2_0_i8. */
__device__ inline float q2_0_block_dot_i8(const uint8_t *blk, const Q2Act &a, size_t b) {
    const uint16_t *codes = reinterpret_cast<const uint16_t *>(blk + 2);
    const int *xa = reinterpret_cast<const int *>(a.codes + b * 64);
    int s = 0;
    for (int m = 0; m < 4; ++m) {
        const uint32_t w = static_cast<uint32_t>(codes[2 * m]) |
                           (static_cast<uint32_t>(codes[2 * m + 1]) << 16);
        for (int k = 0; k < 4; ++k) {
            s = allpaca_dot4_i8(xa[4 * m + k], static_cast<int>((w >> (2 * k)) & 0x03030303u), s);
        }
    }
    return allpaca_f16le_to_f32(blk) * a.step[b] * (static_cast<float>(s) - a.sum[b]);
}

/* h[p, r] = silu(gate_p[r] . x) * (up_p[r] . x): grid (ff / 64, picks). `pg` and `pu` are
 * the block's 256-float shared scratch. */
__device__ inline void q2_0_moe_gate_up_body(const Q2Act &x, const uint8_t *pool,
                                             const int32_t *slots, float *h, int dim, int ff,
                                             size_t slot_bytes, float *pg, float *pu) {
    constexpr int LANES = 4;
    const int p = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid % LANES;
    const int r = static_cast<int>(blockIdx.x) * (256 / LANES) + tid / LANES;
    const int nblocks = dim / 64;
    const size_t row_bytes = static_cast<size_t>(nblocks) * 18;
    float g = 0.f;
    float u = 0.f;
    if (r < ff) {
        const uint8_t *gate = pool + static_cast<size_t>(slots[p]) * slot_bytes +
                              static_cast<size_t>(r) * row_bytes;
        const uint8_t *up = gate + static_cast<size_t>(ff) * row_bytes;
        for (int ib = lane; ib < nblocks; ib += LANES) {
            g += q2_0_block_dot_i8(gate + static_cast<size_t>(ib) * 18, x, ib);
            u += q2_0_block_dot_i8(up + static_cast<size_t>(ib) * 18, x, ib);
        }
    }
    pg[tid] = g;
    pu[tid] = u;
    __syncthreads();
    if (lane == 0 && r < ff) {
        const float gs = (pg[tid] + pg[tid + 1]) + (pg[tid + 2] + pg[tid + 3]);
        const float us = (pu[tid] + pu[tid + 1]) + (pu[tid + 2] + pu[tid + 3]);
        h[static_cast<size_t>(p) * static_cast<size_t>(ff) + static_cast<size_t>(r)] =
            det_silu_mul(gs, us);
    }
}

__global__ void q2_0_moe_gate_up_kernel(Q2Act x, const uint8_t *pool, Q2MoePicks picks,
                                        float *h, int dim, int ff, size_t slot_bytes) {
    __shared__ float pg[256];
    __shared__ float pu[256];
    q2_0_moe_gate_up_body(x, pool, picks.slot, h, dim, ff, slot_bytes, pg, pu);
}

/* Words of one row's device picks: 32 int32 slots, then their 32 float weights. */
#define ALLPACA_Q4X_PICK_WORDS (2 * ALLPACA_Q2_MOE_MAX_PICKS)

/* The same with the picks in device memory, row blockIdx.z of several: x by dim, picks by
 * ALLPACA_Q4X_PICK_WORDS, h by n_pick (gridDim.y) * ff. */
__global__ void q2_0_moe_gate_up_dev_kernel(Q2Act x, const uint8_t *pool,
                                            const int32_t *picks, float *h, int dim, int ff,
                                            size_t slot_bytes) {
    __shared__ float pg[256];
    __shared__ float pu[256];
    const size_t row = blockIdx.z;
    const size_t first = row * static_cast<size_t>(dim / 64);
    const Q2Act xr = {x.codes + first * 64, x.step + first, x.sum + first};
    q2_0_moe_gate_up_body(xr, pool,
                          picks + row * ALLPACA_Q4X_PICK_WORDS,
                          h + row * gridDim.y * static_cast<size_t>(ff), dim, ff, slot_bytes, pg, pu);
}

/* out[c] = sum_p w_p * (down_p[c] . h[p]): grid (dim / 64). `part` is the block's 256-float
 * shared scratch. */
__device__ inline void q2_0_moe_down_body(const Q2Act &h, const uint8_t *pool, const int32_t *slots,
                                          const float *w, int n_pick, float *out, int dim, int ff,
                                          size_t slot_bytes, float *part) {
    constexpr int LANES = 4;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid % LANES;
    const int col = static_cast<int>(blockIdx.x) * (256 / LANES) + tid / LANES;
    const int nblocks = ff / 64;
    const size_t down_off = 2 * static_cast<size_t>(ff) * static_cast<size_t>(dim / 64) * 18;
    float acc = 0.f;
    if (col < dim) {
        for (int p = 0; p < n_pick; ++p) {
            const uint8_t *down = pool + static_cast<size_t>(slots[p]) * slot_bytes + down_off +
                                  static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 18;
            const size_t first = static_cast<size_t>(p) * static_cast<size_t>(nblocks);
            float s = 0.f;
            for (int ib = lane; ib < nblocks; ib += LANES) {
                s += q2_0_block_dot_i8(down + static_cast<size_t>(ib) * 18, h, first + ib);
            }
            acc += w[p] * s;
        }
    }
    part[tid] = acc;
    __syncthreads();
    if (lane == 0 && col < dim) {
        out[col] = (part[tid] + part[tid + 1]) + (part[tid + 2] + part[tid + 3]);
    }
}

__global__ void q2_0_moe_down_kernel(Q2Act h, const uint8_t *pool, Q2MoePicks picks,
                                     int n_pick, float *out, int dim, int ff, size_t slot_bytes) {
    __shared__ float part[256];
    q2_0_moe_down_body(h, pool, picks.slot, picks.w, n_pick, out, dim, ff, slot_bytes, part);
}

/* Row blockIdx.z of several: h by n_pick * ff, picks by ALLPACA_Q4X_PICK_WORDS, out by dim. */
__global__ void q2_0_moe_down_dev_kernel(Q2Act h, const uint8_t *pool, const int32_t *picks,
                                         int n_pick, float *out, int dim, int ff,
                                         size_t slot_bytes) {
    __shared__ float part[256];
    const size_t row = blockIdx.z;
    const int32_t *slots = picks + row * ALLPACA_Q4X_PICK_WORDS;
    const float *w = reinterpret_cast<const float *>(slots + ALLPACA_Q2_MOE_MAX_PICKS);
    const size_t first = row * static_cast<size_t>(n_pick) * static_cast<size_t>(ff / 64);
    const Q2Act hr = {h.codes + first * 64, h.step + first, h.sum + first};
    q2_0_moe_down_body(hr, pool, slots,
                       w, n_pick, out + row * static_cast<size_t>(dim), dim, ff, slot_bytes, part);
}

/* The dot of a pair of Q2_0 blocks (36 bytes from a 4-aligned address) with blocks b and b + 1
 * of a rounded row, as two q2_0_block_dot_i8 summed. Words: 0 = d0 | codes0[0..1], 1..3 =
 * codes0[2..13], 4 = codes0[14..15] | d1, 5..8 = codes1. */
__device__ inline float q2_0_pair_dot_i8(const uint32_t *wd, const Q2Act &a, size_t b) {
    uint32_t w[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        w[i] = wd[i];
    }
    const uint32_t c0[4] = {(w[0] >> 16) | (w[1] << 16), (w[1] >> 16) | (w[2] << 16),
                            (w[2] >> 16) | (w[3] << 16), (w[3] >> 16) | (w[4] << 16)};
    const int4 *xa = reinterpret_cast<const int4 *>(a.codes + b * 64);
    int s0 = 0;
    int s1 = 0;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int4 a0 = xa[m];
        const int4 a1 = xa[4 + m];
        s0 = allpaca_dot4_i8(a0.x, static_cast<int>(c0[m] & 0x03030303u), s0);
        s0 = allpaca_dot4_i8(a0.y, static_cast<int>((c0[m] >> 2) & 0x03030303u), s0);
        s0 = allpaca_dot4_i8(a0.z, static_cast<int>((c0[m] >> 4) & 0x03030303u), s0);
        s0 = allpaca_dot4_i8(a0.w, static_cast<int>((c0[m] >> 6) & 0x03030303u), s0);
        s1 = allpaca_dot4_i8(a1.x, static_cast<int>(w[5 + m] & 0x03030303u), s1);
        s1 = allpaca_dot4_i8(a1.y, static_cast<int>((w[5 + m] >> 2) & 0x03030303u), s1);
        s1 = allpaca_dot4_i8(a1.z, static_cast<int>((w[5 + m] >> 4) & 0x03030303u), s1);
        s1 = allpaca_dot4_i8(a1.w, static_cast<int>((w[5 + m] >> 6) & 0x03030303u), s1);
    }
    const float d0 = allpaca_f16_bits_to_f32(static_cast<uint16_t>(w[0] & 0xffffu));
    const float d1 = allpaca_f16_bits_to_f32(static_cast<uint16_t>(w[4] >> 16));
    return d0 * a.step[b] * (static_cast<float>(s0) - a.sum[b]) +
           d1 * a.step[b + 1] * (static_cast<float>(s1) - a.sum[b + 1]);
}

/* Blocks b and b + 1 of a rounded row, loaded once for q2_0_pair_dot_pre against several
 * weight pairs. */
struct Q2PairAct {
    int4 a[8];
    float step0, step1, sum0, sum1;
};

__device__ inline Q2PairAct q2_0_pair_act(const Q2Act &a, size_t b) {
    Q2PairAct p;
    const int4 *xa = reinterpret_cast<const int4 *>(a.codes + b * 64);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        p.a[i] = xa[i];
    }
    p.step0 = a.step[b];
    p.step1 = a.step[b + 1];
    p.sum0 = a.sum[b];
    p.sum1 = a.sum[b + 1];
    return p;
}

/* q2_0_pair_dot_i8 on preloaded activations: the same operations in the same order. */
__device__ inline float q2_0_pair_dot_pre(const uint32_t *wd, const Q2PairAct &x) {
    uint32_t w[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        w[i] = wd[i];
    }
    const uint32_t c0[4] = {(w[0] >> 16) | (w[1] << 16), (w[1] >> 16) | (w[2] << 16),
                            (w[2] >> 16) | (w[3] << 16), (w[3] >> 16) | (w[4] << 16)};
    int s0 = 0;
    int s1 = 0;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int4 a0 = x.a[m];
        const int4 a1 = x.a[4 + m];
        s0 = allpaca_dot4_i8(a0.x, static_cast<int>(c0[m] & 0x03030303u), s0);
        s0 = allpaca_dot4_i8(a0.y, static_cast<int>((c0[m] >> 2) & 0x03030303u), s0);
        s0 = allpaca_dot4_i8(a0.z, static_cast<int>((c0[m] >> 4) & 0x03030303u), s0);
        s0 = allpaca_dot4_i8(a0.w, static_cast<int>((c0[m] >> 6) & 0x03030303u), s0);
        s1 = allpaca_dot4_i8(a1.x, static_cast<int>(w[5 + m] & 0x03030303u), s1);
        s1 = allpaca_dot4_i8(a1.y, static_cast<int>((w[5 + m] >> 2) & 0x03030303u), s1);
        s1 = allpaca_dot4_i8(a1.z, static_cast<int>((w[5 + m] >> 4) & 0x03030303u), s1);
        s1 = allpaca_dot4_i8(a1.w, static_cast<int>((w[5 + m] >> 6) & 0x03030303u), s1);
    }
    const float d0 = allpaca_f16_bits_to_f32(static_cast<uint16_t>(w[0] & 0xffffu));
    const float d1 = allpaca_f16_bits_to_f32(static_cast<uint16_t>(w[4] >> 16));
    return d0 * x.step0 * (static_cast<float>(s0) - x.sum0) +
           d1 * x.step1 * (static_cast<float>(s1) - x.sum1);
}

/* The sum over groups of L lanes (L <= 32); every lane of the group gets it. */
template <int L> __device__ inline float q2_lane_sum(float v) {
#pragma unroll
    for (int off = L / 2; off > 0; off >>= 1) {
        v += __shfl_xor(v, off, L);
    }
    return v;
}

/* q2_0_moe_gate_up_dev_kernel over block pairs: L lanes per hidden row, 256 / L rows a block.
 * Needs dim % 128 == 0, a 4-aligned pool and 16-aligned codes (q2_0_pairs_fit). */
template <int L>
__global__ void q2_0_moe_gate_up_pairs_kernel(Q2Act x, const uint8_t *pool, const int32_t *picks,
                                              float *h, int dim, int ff, size_t slot_bytes) {
    const size_t row = blockIdx.z;
    const int p = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid % L;
    const int r = static_cast<int>(blockIdx.x) * (256 / L) + tid / L;
    const size_t first = row * static_cast<size_t>(dim / 64);
    const Q2Act xr = {x.codes + first * 64, x.step + first, x.sum + first};
    const int32_t *slots = picks + row * ALLPACA_Q4X_PICK_WORDS;
    float g = 0.f;
    float u = 0.f;
    if (r < ff) {
        const size_t row_words = static_cast<size_t>(dim / 64) * 18 / 4;
        const uint32_t *gate =
            reinterpret_cast<const uint32_t *>(pool + static_cast<size_t>(slots[p]) * slot_bytes) +
            static_cast<size_t>(r) * row_words;
        const uint32_t *up = gate + static_cast<size_t>(ff) * row_words;
        for (int q = lane; q < dim / 128; q += L) {
            g += q2_0_pair_dot_i8(gate + 9 * q, xr, 2 * static_cast<size_t>(q));
            u += q2_0_pair_dot_i8(up + 9 * q, xr, 2 * static_cast<size_t>(q));
        }
    }
    g = q2_lane_sum<L>(g);
    u = q2_lane_sum<L>(u);
    if (lane == 0 && r < ff) {
        h[(row * gridDim.y + static_cast<size_t>(p)) * static_cast<size_t>(ff) + static_cast<size_t>(r)] =
            det_silu_mul(g, u);
    }
}

/* q2_0_moe_down_dev_kernel over block pairs: L lanes per output column share every (pick,
 * pair). Needs ff % 128 == 0 as well. */
template <int L>
__global__ void q2_0_moe_down_pairs_kernel(Q2Act h, const uint8_t *pool, const int32_t *picks,
                                           int n_pick, float *out, int dim, int ff,
                                           size_t slot_bytes) {
    const size_t row = blockIdx.z;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid % L;
    const int col = static_cast<int>(blockIdx.x) * (256 / L) + tid / L;
    const int32_t *slots = picks + row * ALLPACA_Q4X_PICK_WORDS;
    const float *w = reinterpret_cast<const float *>(slots + ALLPACA_Q2_MOE_MAX_PICKS);
    const int npairs = ff / 128;
    const size_t first = row * static_cast<size_t>(n_pick) * static_cast<size_t>(ff / 64);
    const Q2Act hr = {h.codes + first * 64, h.step + first, h.sum + first};
    const size_t down_off = 2 * static_cast<size_t>(ff) * static_cast<size_t>(dim / 64) * 18;
    const size_t col_words = static_cast<size_t>(ff / 64) * 18 / 4;
    float acc = 0.f;
    if (col < dim) {
        for (int j = lane; j < n_pick * npairs; j += L) {
            const int p = j / npairs;
            const int q = j % npairs;
            const uint32_t *down =
                reinterpret_cast<const uint32_t *>(pool + static_cast<size_t>(slots[p]) * slot_bytes +
                                                   down_off) +
                static_cast<size_t>(col) * col_words;
            acc += w[p] * q2_0_pair_dot_i8(down + 9 * q, hr,
                                           static_cast<size_t>(p) * (ff / 64) + 2 * static_cast<size_t>(q));
        }
    }
    acc = q2_lane_sum<L>(acc);
    if (lane == 0 && col < dim) {
        out[row * static_cast<size_t>(dim) + static_cast<size_t>(col)] = acc;
    }
}

/* q2_0_moe_gate_up_pairs_kernel<L> with the block's weight rows and its row's activations
 * staged in LDS. There each lane reads its 36-byte block pair from a 36-byte stride and its
 * activations from a 128-byte one, so every load instruction touched dozens of cache lines
 * (16 rows 398 us, one row 35 us = 263 GB/s). Here the block copies its 256 / L gate rows and
 * up rows (contiguous) and the row's codes with consecutive words per thread, the codes
 * transposed so lane q's eight int4s sit at [m][q]. The lanes then do the same arithmetic.
 * LDS bytes: q2_0_gate_up_lds_bytes. */
static size_t q2_0_gate_up_lds_bytes(int L, int dim) {
    const size_t row_words = static_cast<size_t>(dim / 64) * 18 / 4;
    return 2 * static_cast<size_t>(256 / L) * row_words * 4 + static_cast<size_t>(dim) +
           2 * static_cast<size_t>(dim / 64) * 4;
}

template <int L>
__global__ __launch_bounds__(256) void q2_0_moe_gate_up_lds_kernel(Q2Act x, const uint8_t *pool,
                                                                   const int32_t *picks, float *h,
                                                                   int dim, int ff,
                                                                   size_t slot_bytes) {
    extern __shared__ uint32_t q2_lds[];
    constexpr int G = 256 / L;
    const size_t row = blockIdx.z;
    const int p = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid % L;
    const int gi = tid / L;
    const int r0 = static_cast<int>(blockIdx.x) * G;
    const int nr = ff - r0 < G ? ff - r0 : G;
    const int npairs = dim / 128;
    const int nb = dim / 64;
    const int row_words = nb * 18 / 4;
    uint32_t *sw = q2_lds;
    int4 *sx = reinterpret_cast<int4 *>(sw + 2 * G * row_words);
    float *sst = reinterpret_cast<float *>(sx + 8 * npairs);
    const size_t first = row * static_cast<size_t>(nb);
    const int32_t *slots = picks + row * ALLPACA_Q4X_PICK_WORDS;
    const uint32_t *gate =
        reinterpret_cast<const uint32_t *>(pool + static_cast<size_t>(slots[p]) * slot_bytes) +
        static_cast<size_t>(r0) * row_words;
    const uint32_t *up = gate + static_cast<size_t>(ff) * row_words;
    for (int i = tid; i < nr * row_words; i += 256) {
        sw[i] = gate[i];
        sw[G * row_words + i] = up[i];
    }
    const uint32_t *xc = reinterpret_cast<const uint32_t *>(x.codes + first * 64);
    for (int i = tid; i < dim / 4; i += 256) {
        reinterpret_cast<uint32_t *>(sx + ((i & 31) >> 2) * npairs + (i >> 5))[i & 3] = xc[i];
    }
    for (int i = tid; i < nb; i += 256) {
        sst[i] = x.step[first + i];
        sst[nb + i] = x.sum[first + i];
    }
    __syncthreads();
    float g = 0.f;
    float u = 0.f;
    if (gi < nr) {
        for (int q = lane; q < npairs; q += L) {
            Q2PairAct a;
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                a.a[m] = sx[m * npairs + q];
            }
            a.step0 = sst[2 * q];
            a.step1 = sst[2 * q + 1];
            a.sum0 = sst[nb + 2 * q];
            a.sum1 = sst[nb + 2 * q + 1];
            g += q2_0_pair_dot_pre(sw + gi * row_words + 9 * q, a);
            u += q2_0_pair_dot_pre(sw + (G + gi) * row_words + 9 * q, a);
        }
    }
    g = q2_lane_sum<L>(g);
    u = q2_lane_sum<L>(u);
    if (lane == 0 && gi < nr) {
        h[(row * gridDim.y + static_cast<size_t>(p)) * static_cast<size_t>(ff) +
          static_cast<size_t>(r0 + gi)] = det_silu_mul(g, u);
    }
}

/* Whether the pair kernels can run on these shapes and pointers. */
static bool q2_0_pairs_fit(const uint8_t *pool, const Q2MoeScratch &sc, int dim, int ff) {
    return dim % 128 == 0 && ff % 128 == 0 && reinterpret_cast<uintptr_t>(pool) % 4 == 0 &&
           reinterpret_cast<uintptr_t>(sc.x_codes) % 16 == 0 &&
           reinterpret_cast<uintptr_t>(sc.h_codes) % 16 == 0;
}

/* Softmax over n router scores, the k best by weight (ties to the lower expert) into
 * slots[0..k], their weights renormalised over the picks and times `scale` into w[0..k]. One
 * block of 256 threads a row; the sums run in expert order on one thread, as the host's do. The
 * picks come from a bitonic sort of (weight, expert) keys in LDS: k rounds of a 32-lane
 * arg-max cost ~23 of the kernel's 34 us. */
#define ALLPACA_Q4X_MAX_EXPERTS 1024
#define ALLPACA_Q4X_ROUTE_THREADS 256
__global__ __launch_bounds__(ALLPACA_Q4X_ROUTE_THREADS) void q4x_route_topk_kernel(
    const float *scores, int n, int k, float scale, int32_t *picks) {
    constexpr int T = ALLPACA_Q4X_ROUTE_THREADS;
    __shared__ __align__(16) float p[ALLPACA_Q4X_MAX_EXPERTS];
    __shared__ unsigned long long key[ALLPACA_Q4X_MAX_EXPERTS];
    __shared__ float red[T / 32];
    __shared__ float total;
    const int tid = static_cast<int>(threadIdx.x);
    /* Row blockIdx.x of several: scores by n, picks by ALLPACA_Q4X_PICK_WORDS. */
    scores += static_cast<size_t>(blockIdx.x) * static_cast<size_t>(n);
    int32_t *slots = picks + static_cast<size_t>(blockIdx.x) * ALLPACA_Q4X_PICK_WORDS;
    float *w = reinterpret_cast<float *>(slots + ALLPACA_Q2_MOE_MAX_PICKS);
    float mx = -INFINITY;
    for (int e = tid; e < n; e += T) {
        mx = fmaxf(mx, scores[e]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor(mx, off, 32));
    }
    if (tid % 32 == 0) {
        red[tid / 32] = mx;
    }
    __syncthreads();
    mx = red[0];
#pragma unroll
    for (int i = 1; i < T / 32; ++i) {
        mx = fmaxf(mx, red[i]);
    }
    for (int e = tid; e < n; e += T) {
        p[e] = expf(scores[e] - mx);
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.f;
        int e = 0;
        for (; e + 16 <= n; e += 16) {
            float4 v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[i] = reinterpret_cast<const float4 *>(p + e)[i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sum += v[i].x;
                sum += v[i].y;
                sum += v[i].z;
                sum += v[i].w;
            }
        }
        for (; e < n; ++e) {
            sum += p[e];
        }
        total = fmaxf(sum, 1e-9f);
    }
    __syncthreads();
    const float sum = total;
    int size = 2;
    while (size < n) {
        size <<= 1;
    }
    /* Larger weight first, then the lower expert; the weights are >= 0 and padding sorts last. */
    for (int e = tid; e < size; e += T) {
        unsigned long long kv = 0;
        if (e < n) {
            const float v = p[e] / sum;
            p[e] = v;
            kv = (static_cast<unsigned long long>(__float_as_uint(v) | 0x80000000u) << 32) |
                 static_cast<unsigned long long>(0xffffffffu - static_cast<uint32_t>(e));
        }
        key[e] = kv;
    }
    __syncthreads();
    for (int span = 2; span <= size; span <<= 1) {
        for (int stride = span / 2; stride > 0; stride >>= 1) {
            for (int t = tid; t < size / 2; t += T) {
                const int a = 2 * t - (t & (stride - 1));
                const int b = a + stride;
                const unsigned long long ka = key[a];
                const unsigned long long kb = key[b];
                if ((ka < kb) == ((a & span) == 0)) {
                    key[a] = kb;
                    key[b] = ka;
                }
            }
            __syncthreads();
        }
    }
    if (tid == 0) {
        float part = 0.f;
        for (int r = 0; r < k; ++r) {
            const int e = static_cast<int>(0xffffffffu - static_cast<uint32_t>(key[r]));
            slots[r] = e;
            part += p[e];
        }
        part = fmaxf(part, 1e-9f);
        for (int r = 0; r < k; ++r) {
            w[r] = p[slots[r]] / part * scale;
        }
    }
}

extern "C" int allpaca_q2_0_moe_ffn(const float *x, const uint8_t *pool, const int32_t *slots,
                                    const float *weights, int32_t n_pick, float *h, float *out,
                                    int32_t dim, int32_t ff, allpaca_stream_t stream) {
    if (n_pick <= 0 || n_pick > ALLPACA_Q2_MOE_MAX_PICKS || dim <= 0 || ff <= 0 || dim % 64 != 0 ||
        ff % 64 != 0 || !x || !pool || !slots || !weights || !h || !out) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    Q2MoePicks picks = {};
    for (int p = 0; p < n_pick; ++p) {
        if (slots[p] < 0) {
            return ALLPACA_ERR_INVALID;
        }
        picks.slot[p] = slots[p];
        picks.w[p] = weights[p];
    }
    const size_t slot_bytes = 3 * static_cast<size_t>(ff) * static_cast<size_t>(dim / 64) * 18;
    const Q2MoeScratch sc = q2_moe_scratch(h, 1, n_pick, dim, ff);
    q2_0_round_act(x, sc.x_codes, sc.x_step, sc.x_sum, static_cast<size_t>(dim), as_stream(stream));
    const Q2Act xa = {sc.x_codes, sc.x_step, sc.x_sum};
    hipLaunchKernelGGL(q2_0_moe_gate_up_kernel,
                       dim3(static_cast<unsigned>((ff + 63) / 64), static_cast<unsigned>(n_pick)),
                       dim3(256), 0, as_stream(stream), xa, pool, picks, sc.h, dim, ff, slot_bytes);
    q2_0_round_act(sc.h, sc.h_codes, sc.h_step, sc.h_sum, static_cast<size_t>(n_pick) * ff,
                   as_stream(stream));
    const Q2Act ha = {sc.h_codes, sc.h_step, sc.h_sum};
    hipLaunchKernelGGL(q2_0_moe_down_kernel, dim3(static_cast<unsigned>((dim + 63) / 64)), dim3(256),
                       0, as_stream(stream), ha, pool, picks, n_pick, out, dim, ff, slot_bytes);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_q4x_routed_ffn(const float *x, const float *scores, const uint8_t *pool,
                                      int32_t n_experts, int32_t n_pick, float scale, void *picks,
                                      float *h, float *out, int32_t dim, int32_t ff, int32_t rows,
                                      allpaca_stream_t stream) {
    if (n_pick <= 0 || n_pick > ALLPACA_Q2_MOE_MAX_PICKS || n_pick > n_experts ||
        n_experts > ALLPACA_Q4X_MAX_EXPERTS || dim <= 0 || ff <= 0 || dim % 64 != 0 ||
        ff % 64 != 0 || rows <= 0 || !x || !scores || !pool || !picks || !h || !out) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    int32_t *slots = static_cast<int32_t *>(picks);
    const size_t slot_bytes = 3 * static_cast<size_t>(ff) * static_cast<size_t>(dim / 64) * 18;
    const unsigned r = static_cast<unsigned>(rows);
    hipLaunchKernelGGL(q4x_route_topk_kernel, dim3(r), dim3(ALLPACA_Q4X_ROUTE_THREADS), 0, as_stream(stream), scores,
                       n_experts, n_pick, scale, slots);
    const Q2MoeScratch sc = q2_moe_scratch(h, rows, n_pick, dim, ff);
    q2_0_round_act(x, sc.x_codes, sc.x_step, sc.x_sum, static_cast<size_t>(rows) * dim,
                   as_stream(stream));
    const Q2Act xa = {sc.x_codes, sc.x_step, sc.x_sum};
    const bool pairs = q2_0_pairs_fit(pool, sc, dim, ff);
    const unsigned picks_n = static_cast<unsigned>(n_pick);
    /* The same lanes for any number of rows, so a row's sums do not depend on its company. */
    const size_t gu_lds = q2_0_gate_up_lds_bytes(32, dim);
    if (pairs && kq_wave64() && gu_lds <= 32768 &&
        reinterpret_cast<uintptr_t>(xa.codes) % 4 == 0) {
        // MI50: one row 34.8 -> 29.6 us, 16 rows 398 -> 363 us. The RX 9070 lost (routed
        // experts one row 42 -> 47.5 us, 16 rows 396 -> 430 us).
        hipLaunchKernelGGL(q2_0_moe_gate_up_lds_kernel<32>,
                           dim3(static_cast<unsigned>((ff + 7) / 8), picks_n, r), dim3(256),
                           gu_lds, as_stream(stream), xa, pool, slots, sc.h, dim, ff, slot_bytes);
    } else if (pairs) {
        hipLaunchKernelGGL(q2_0_moe_gate_up_pairs_kernel<32>,
                           dim3(static_cast<unsigned>((ff + 7) / 8), picks_n, r), dim3(256), 0,
                           as_stream(stream), xa, pool, slots, sc.h, dim, ff, slot_bytes);
    } else {
        hipLaunchKernelGGL(q2_0_moe_gate_up_dev_kernel,
                           dim3(static_cast<unsigned>((ff + 63) / 64), picks_n, r), dim3(256), 0,
                           as_stream(stream), xa, pool, slots, sc.h, dim, ff, slot_bytes);
    }
    q2_0_round_act(sc.h, sc.h_codes, sc.h_step, sc.h_sum,
                   static_cast<size_t>(rows) * n_pick * ff, as_stream(stream));
    const Q2Act ha = {sc.h_codes, sc.h_step, sc.h_sum};
    if (pairs) {
        hipLaunchKernelGGL(q2_0_moe_down_pairs_kernel<16>,
                           dim3(static_cast<unsigned>((dim + 15) / 16), 1, r), dim3(256), 0,
                           as_stream(stream), ha, pool, slots, n_pick, out, dim, ff, slot_bytes);
    } else {
        hipLaunchKernelGGL(q2_0_moe_down_dev_kernel,
                           dim3(static_cast<unsigned>((dim + 63) / 64), 1, r), dim3(256), 0,
                           as_stream(stream), ha, pool, slots, n_pick, out, dim, ff, slot_bytes);
    }
    return hip_status(hipGetLastError());
}

/* x[i] = silu(x[i] * s). */
__global__ void silu_scale_kernel(float *x, int n, float s) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        const float v = x[i] * s;
        x[i] = v / (1.f + expf(-v));
    }
}

extern "C" int allpaca_silu_scale(float *x, int32_t n, float s, allpaca_stream_t stream) {
    if (n <= 0 || !x) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(silu_scale_kernel, dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256), 0,
                       as_stream(stream), x, n, s);
    return hip_status(hipGetLastError());
}

/* qwen4exp hyper-connections over `rows` tokens of hc streams of dim floats:
 * x[(r * hc + s) * dim + d]. */

/* xn[r, s] = x[r, s] / rms(x[r, s]) * gains[s], one block per stream. */
__global__ void q4x_stream_norm_kernel(const float *x, const float *gains, float *xn, int hc,
                                       int dim, float eps) {
    __shared__ float part[256];
    const int tid = static_cast<int>(threadIdx.x);
    const size_t base = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(dim);
    const size_t gbase = static_cast<size_t>(blockIdx.x % static_cast<unsigned>(hc)) *
                         static_cast<size_t>(dim);
    float ss = 0.f;
    for (int d = tid; d < dim; d += 256) {
        const float v = x[base + d];
        ss += v * v;
    }
    part[tid] = ss;
    __syncthreads();
    for (int step = 128; step > 0; step >>= 1) {
        if (tid < step) {
            part[tid] += part[tid + step];
        }
        __syncthreads();
    }
    const float inv = rsqrtf(part[0] / static_cast<float>(dim) + eps);
    for (int d = tid; d < dim; d += 256) {
        xn[base + d] = x[base + d] * inv * gains[gbase + d];
    }
}

/* mixed[r, d] = mean over streams of xn[r, s, d] * sigmoid(gate[r, s, d]). */
__global__ void q4x_gated_mean_kernel(const float *xn, const float *gate, float *mixed, int rows,
                                      int hc, int dim) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= rows * dim) {
        return;
    }
    const size_t base = static_cast<size_t>(i / dim) * static_cast<size_t>(hc) *
                        static_cast<size_t>(dim) + static_cast<size_t>(i % dim);
    float acc = 0.f;
    for (int s = 0; s < hc; ++s) {
        const size_t at = base + static_cast<size_t>(s) * static_cast<size_t>(dim);
        acc += xn[at] / (1.f + expf(-gate[at]));
    }
    mixed[i] = acc / static_cast<float>(hc);
}

/* streams[r, s, d] += out[r, d] * 2 * sigmoid(logits[r, s] / hc). */
__global__ void q4x_combine_kernel(float *streams, const float *out, const float *logits, int rows,
                                   int hc, int dim) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= rows * hc * dim) {
        return;
    }
    const int stream = i / dim;
    const int row = stream / hc;
    const float w = 2.f / (1.f + expf(-logits[stream] / static_cast<float>(hc)));
    streams[i] += out[static_cast<size_t>(row) * static_cast<size_t>(dim) + i % dim] * w;
}

/* out[r, d] = routed[r, d] + sigmoid(gate[r]) * shared[r, d]; either part may be null. */
__global__ void q4x_ffn_sum_kernel(const float *routed, const float *shared, const float *gate,
                                   float *out, int rows, int dim) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= rows * dim) {
        return;
    }
    float v = routed ? routed[i] : 0.f;
    if (shared) {
        v += shared[i] / (1.f + expf(-gate[i / dim]));
    }
    out[i] = v;
}

/* Sum of v over a block of 256 threads; every thread gets it. */
__device__ inline float q4x_sum256(float v, float *part) {
    const int tid = static_cast<int>(threadIdx.x);
    __syncthreads();
    part[tid] = v;
    __syncthreads();
    for (int step = 128; step > 0; step >>= 1) {
        if (tid < step) {
            part[tid] += part[tid + step];
        }
        __syncthreads();
    }
    return part[0];
}

/* The n-gram layer of `rows` tokens in order, one block per stream s. A
 * token's stream is the query of a gate on the normed key; the open value is
 * normed again into the conv input x, and the stream gets the gated value
 * plus silu of a conv over x and the conv inputs (kern - 1 - k) * ngram tokens
 * back. history[hist, hc * dim] holds the last hist conv inputs, oldest first. */
__global__ void q4x_ple_apply_kernel(float *streams, const float *key, const float *value,
                                     const float *nk, const float *nq, const float *nc,
                                     const float *conv_w, float *history, int rows, int hc,
                                     int dim, int kern, int ngram, float eps) {
    __shared__ float part[256];
    const int tid = static_cast<int>(threadIdx.x);
    const size_t hc_dim = static_cast<size_t>(hc) * static_cast<size_t>(dim);
    const size_t g = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(dim);
    const int hist = (kern - 1) * ngram;
    for (int r = 0; r < rows; ++r) {
        float *st = streams + static_cast<size_t>(r) * hc_dim + g;
        const float *k = key + static_cast<size_t>(r) * hc_dim + g;
        const float *v = value + static_cast<size_t>(r) * static_cast<size_t>(dim);
        float sk = 0.f, sq = 0.f;
        for (int d = tid; d < dim; d += 256) {
            sk += k[d] * k[d];
            sq += st[d] * st[d];
        }
        const float ik = rsqrtf(q4x_sum256(sk, part) / static_cast<float>(dim) + eps);
        const float iq = rsqrtf(q4x_sum256(sq, part) / static_cast<float>(dim) + eps);
        float dot = 0.f;
        for (int d = tid; d < dim; d += 256) {
            dot += (k[d] * ik * nk[g + d]) * (st[d] * iq * nq[g + d]);
        }
        const float s = q4x_sum256(dot, part) / sqrtf(static_cast<float>(dim));
        // Signed square root before the sigmoid.
        const float signed_s = s == 0.f ? 0.f : copysignf(sqrtf(fmaxf(fabsf(s), 1e-6f)), s);
        const float open = 1.f / (1.f + expf(-signed_s));
        float sg = 0.f;
        for (int d = tid; d < dim; d += 256) {
            const float gv = v[d] * open;
            sg += gv * gv;
        }
        const float ig = rsqrtf(q4x_sum256(sg, part) / static_cast<float>(dim) + eps);
        for (int d = tid; d < dim; d += 256) {
            const size_t c = g + static_cast<size_t>(d);
            const float gv = v[d] * open;
            const float x = gv * ig * nc[c];
            const float *w = conv_w + c * static_cast<size_t>(kern);
            float acc = w[kern - 1] * x;
            for (int t = 0; t < kern - 1; ++t) {
                const int back = (kern - 1 - t) * ngram;
                acc += w[t] * history[static_cast<size_t>(hist - back) * hc_dim + c];
            }
            st[d] += gv + acc / (1.f + expf(-acc));
            for (int p = 0; p + 1 < hist; ++p) {
                history[static_cast<size_t>(p) * hc_dim + c] = history[static_cast<size_t>(p + 1) * hc_dim + c];
            }
            if (hist > 0) {
                history[static_cast<size_t>(hist - 1) * hc_dim + c] = x;
            }
        }
    }
}

extern "C" int allpaca_q4x_ple_apply(float *streams, const float *key, const float *value,
                                     const float *nk, const float *nq, const float *nc,
                                     const float *conv_w, float *history, int32_t rows, int32_t hc,
                                     int32_t dim, int32_t kern, int32_t ngram, float eps,
                                     allpaca_stream_t stream) {
    if (rows <= 0 || hc <= 0 || dim <= 0 || kern <= 0 || ngram <= 0 || !streams || !key || !value ||
        !nk || !nq || !nc || !conv_w || (kern > 1 && !history)) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(q4x_ple_apply_kernel, dim3(static_cast<unsigned>(hc)), dim3(256), 0,
                       as_stream(stream), streams, key, value, nk, nq, nc, conv_w, history, rows,
                       hc, dim, kern, ngram, eps);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_q4x_stream_norm(const float *x, const float *gains, float *xn, int32_t rows,
                                       int32_t hc, int32_t dim, float eps,
                                       allpaca_stream_t stream) {
    if (rows <= 0 || hc <= 0 || dim <= 0 || !x || !gains || !xn) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(q4x_stream_norm_kernel, dim3(static_cast<unsigned>(rows * hc)), dim3(256), 0,
                       as_stream(stream), x, gains, xn, hc, dim, eps);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_q4x_gated_mean(const float *xn, const float *gate, float *mixed,
                                      int32_t rows, int32_t hc, int32_t dim,
                                      allpaca_stream_t stream) {
    if (rows <= 0 || hc <= 0 || dim <= 0 || !xn || !gate || !mixed) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(q4x_gated_mean_kernel,
                       dim3(static_cast<unsigned>((rows * dim + 255) / 256)), dim3(256), 0,
                       as_stream(stream), xn, gate, mixed, rows, hc, dim);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_q4x_combine(float *streams, const float *out, const float *logits,
                                   int32_t rows, int32_t hc, int32_t dim,
                                   allpaca_stream_t stream) {
    if (rows <= 0 || hc <= 0 || dim <= 0 || !streams || !out || !logits) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(q4x_combine_kernel,
                       dim3(static_cast<unsigned>((rows * hc * dim + 255) / 256)), dim3(256), 0,
                       as_stream(stream), streams, out, logits, rows, hc, dim);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_q4x_ffn_sum(const float *routed, const float *shared, const float *gate,
                                   float *out, int32_t rows, int32_t dim,
                                   allpaca_stream_t stream) {
    if (rows <= 0 || dim <= 0 || !out || (shared && !gate) || (!routed && !shared)) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(q4x_ffn_sum_kernel, dim3(static_cast<unsigned>((rows * dim + 255) / 256)),
                       dim3(256), 0, as_stream(stream), routed, shared, gate, out, rows, dim);
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

/* One thread per output column. Native ROCmFP4 blocks along K: BB = 18 is
 * the dual-scale layout (ggml type 100: qs[16], e[2]), BB = 17 the FAST
 * single-scale layout (type 101: qs[16], e), where both halves read the one
 * scale byte. Baseline: each column re-reads A from global (no LDS tile). */
template <int BB>
__global__ void gemv_nt_rocmfp4_untiled_kernel(const float *a, const uint8_t *b, float *c, int m,
                                               int n, int k) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    if (col >= n || row >= m) {
        return;
    }
    const int nblocks = (k + 31) / 32;
    const uint8_t *row_blocks = b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * BB;
    const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    float acc = 0.f;
    for (int ib = 0; ib < nblocks; ++ib) {
        const uint8_t *blk = row_blocks + static_cast<size_t>(ib) * BB;
        const float d0 = rocmfp4_scale(blk[16]);
        const float d1 = rocmfp4_scale(blk[BB - 1]);
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
template <int BB>
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
            b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * BB;
        const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
        for (int ib = lane; ib < nblocks; ib += LANES) {
            const uint8_t *blk = row_blocks + static_cast<size_t>(ib) * BB;
            const float d0 = rocmfp4_scale(blk[16]);
            const float d1 = rocmfp4_scale(blk[BB - 1]);
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

/* Four 4-bit ROCmFP4 codes, one per byte, to their signed half-scale levels
 * (0,1,2,3,4,6,8,10, negated when bit 3 is set) as four int8 bytes. */
__device__ inline uint32_t rocmfp4_levels_i8x4(uint32_t code) {
    const uint32_t mag3 = code & 0x07070707u;
    // 0xFF in each byte whose magnitude index is 5..7 (levels 6, 8, 10).
    const uint32_t big = (((mag3 + 0x03030303u) >> 3) & 0x01010101u) * 0xFFu;
    const uint32_t mag = mag3 + ((((mag3 | 0x80808080u) - 0x04040404u) & 0x07070707u) & big);
    // Negate where the sign bit is set and the level is not zero (so the
    // per-byte +1 never carries into its neighbour).
    const uint32_t nonzero = (((mag + 0x7F7F7F7Fu) >> 7) & 0x01010101u) * 0xFFu;
    const uint32_t neg = (((code >> 3) & 0x01010101u) * 0xFFu) & nonzero;
    return (mag ^ neg) + (neg & 0x01010101u);
}

/* As rocmfp4_levels_i8x4 for the NVFP4 E2M1 codebook (0,1,2,3,4,6,8,12). */
__device__ inline uint32_t nvfp4_levels_i8x4(uint32_t code) {
    const uint32_t mag3 = code & 0x07070707u;
    const uint32_t big = (((mag3 + 0x03030303u) >> 3) & 0x01010101u) * 0xFFu;
    uint32_t mag = mag3 + ((((mag3 | 0x80808080u) - 0x04040404u) & 0x07070707u) & big);
    // Index 7 is 12, not 10: add 2 where all three magnitude bits are set.
    const uint32_t seven = mag3 & (mag3 >> 1) & (mag3 >> 2) & 0x01010101u;
    mag += seven << 1;
    const uint32_t nonzero = (((mag + 0x7F7F7F7Fu) >> 7) & 0x01010101u) * 0xFFu;
    const uint32_t neg = (((code >> 3) & 0x01010101u) * 0xFFu) & nonzero;
    return (mag ^ neg) + (neg & 0x01010101u);
}

/* f16 bits to f32 for the NVFP4 device block scales. */
__device__ inline float nvfp4_f16_scale(const uint8_t *p) {
    uint16_t h;
    __builtin_memcpy(&h, p, 2);
    return static_cast<float>(__builtin_bit_cast(_Float16, h));
}

/* ROCmFP4 GEMV on int8 activations (the K-quant int8 scheme, llama.cpp
 * mmvq): A is quantized once per GEMV to int8 with one f32 scale per 32
 * values, and each lane takes whole 32-weight blocks: two signed dot4 chains
 * (low nibbles = weights 0..15, high = 16..31) per row and one scale product.
 * MR rows share one pass over the weights, so 2..8 verify rows cost little
 * more than one. Every m uses this path, so a verify or batched row
 * reproduces the decode (m == 1) row bit for bit. BB = 18 dual-scale (type
 * 100), BB = 17 FAST single-scale (type 101). */
template <int BB, int MR>
__global__ void gemv_nt_rocmfp4_i8_kernel(const int8_t *aq, const float *ad, const uint8_t *b,
                                          float *c, int m, int n, int k) {
#pragma clang fp contract(off)
    constexpr int LPC = 8;
    const int sl = static_cast<int>(threadIdx.x) % LPC;
    const int col = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / LPC) +
                    static_cast<int>(threadIdx.x) / LPC;
    const int row0 = static_cast<int>(blockIdx.y) * MR;
    const bool live = col < n;
    const int kb = k / 32;
    const uint8_t *brow = b + static_cast<size_t>(live ? col : 0) * static_cast<size_t>(kb) * BB;
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
        for (int ib = sl; ib < kb; ib += LPC) {
            const uint8_t *blk = brow + static_cast<size_t>(ib) * BB;
            // Blocks are 17 or 18 bytes: unaligned loads.
            uint32_t x[4];
            __builtin_memcpy(x, blk, 16);
            // BB = 20: NVFP4 device block, f16 scales and the 12-top codebook.
            const float d0 = BB == 20 ? nvfp4_f16_scale(blk + 16) : rocmfp4_scale(blk[16]);
            const float d1 = BB == 20 ? nvfp4_f16_scale(blk + 18) : rocmfp4_scale(blk[BB - 1]);
            uint32_t wl[4], wh[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                if (BB == 20) {
                    wl[t] = nvfp4_levels_i8x4(x[t] & 0x0F0F0F0Fu);
                    wh[t] = nvfp4_levels_i8x4((x[t] >> 4) & 0x0F0F0F0Fu);
                } else {
                    wl[t] = rocmfp4_levels_i8x4(x[t] & 0x0F0F0F0Fu);
                    wh[t] = rocmfp4_levels_i8x4((x[t] >> 4) & 0x0F0F0F0Fu);
                }
            }
#if ALLPACA_I8_LOADS_FIRST
            uint32_t al[MR][4], ah[MR][4];
            float ds[MR];
#pragma unroll
            for (int r = 0; r < MR; ++r) {
                kq_load_u32s<4>(arow[r] + ib * 32, al[r]);
                kq_load_u32s<4>(arow[r] + ib * 32 + 16, ah[r]);
                ds[r] = drow[r][ib];
            }
#pragma unroll
            for (int r = 0; r < MR; ++r) {
                int32_t dl = 0, dh = 0;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    dl = i8dot4(al[r][t], wl[t], dl);
                    dh = i8dot4(ah[r][t], wh[t], dh);
                }
                acc[r] = acc[r] + ds[r] * (d0 * static_cast<float>(dl) +
                                           d1 * static_cast<float>(dh));
            }
#else
#pragma unroll
            for (int r = 0; r < MR; ++r) {
                uint32_t al[4], ah[4];
                kq_load_u32s<4>(arow[r] + ib * 32, al);
                kq_load_u32s<4>(arow[r] + ib * 32 + 16, ah);
                int32_t dl = 0, dh = 0;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    dl = i8dot4(al[t], wl[t], dl);
                    dh = i8dot4(ah[t], wh[t], dh);
                }
                acc[r] = acc[r] + drow[r][ib] * (d0 * static_cast<float>(dl) +
                                                 d1 * static_cast<float>(dh));
            }
#endif
        }
    }
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = LPC / 2; off > 0; off >>= 1) {
            v = v + __shfl_down(v, off, LPC);
        }
        if (live && sl == 0 && row0 + r < m) {
            c[static_cast<size_t>(row0 + r) * static_cast<size_t>(n) + static_cast<size_t>(col)] =
                v;
        }
    }
}

/* Returns ALLPACA_ERR_INVALID when the int8 path cannot run (caller falls back). */
template <int BB>
static int launch_rocmfp4_i8(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                             int32_t k, hipStream_t stream) {
    if (k % 32 != 0) {
        return ALLPACA_ERR_INVALID;
    }
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
    KqActCopy act(scratch, a, aq, ad, m, k, stream, c, n);
    constexpr int COLS = 256 / 8;
    const dim3 block(256);
    const int mr = m == 1 ? 1 : (m == 2 ? 2 : (m <= 4 ? 4 : 8));
    const dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS),
                    static_cast<unsigned>((m + mr - 1) / mr));
#define FP4_I8_LAUNCH(MRV)                                                                         \
    hipLaunchKernelGGL((gemv_nt_rocmfp4_i8_kernel<BB, MRV>), grid, block, 0, stream, aq, ad, b, c, \
                       m, n, k)
    if (mr == 1) {
        FP4_I8_LAUNCH(1);
    } else if (mr == 2) {
        FP4_I8_LAUNCH(2);
    } else if (mr == 4) {
        FP4_I8_LAUNCH(4);
    } else {
        FP4_I8_LAUNCH(8);
    }
#undef FP4_I8_LAUNCH
    return hip_status(hipGetLastError());
}

/* `tiled`: 1 = decode path (int8 activations, float ksplit when K is not
 * whole blocks), 2 = float ksplit kernel (A/B and reference), 0 = untiled. */
static int launch_gemm_nt_rocmfp4(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                  int32_t k, hipStream_t stream, int tiled, bool fast = false) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    /* ALLPACA_ROCMFP4_F32_GEMV=1 keeps decode on float activations, to tell
     * int8-activation rounding apart from a model-level difference. */
    static const bool f32_only = [] {
        const char *v = std::getenv("ALLPACA_ROCMFP4_F32_GEMV");
        return v && v[0] == '1';
    }();
    if (tiled == 1 && !f32_only) {
        const int rc = fast ? launch_rocmfp4_i8<17>(a, b, c, m, n, k, stream)
                            : launch_rocmfp4_i8<18>(a, b, c, m, n, k, stream);
        if (rc != ALLPACA_ERR_INVALID) {
            return rc;
        }
    }
    if (tiled) {
        /* ksplit is the proven decode kernel (row = blockIdx.y). The COLS=64
         * tiled path was built for large-M prefill and is not trusted for
         * Qwen gated/IMRoPE short prompts (M~11 produced garbage tokens). */
        constexpr int GROUPS = 8;
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((n + GROUPS - 1) / GROUPS), static_cast<unsigned>(m));
        if (fast) {
            hipLaunchKernelGGL(gemv_nt_rocmfp4_ksplit_kernel<17>, grid, block, 0, stream, a, b, c, m,
                               n, k);
        } else {
            hipLaunchKernelGGL(gemv_nt_rocmfp4_ksplit_kernel<18>, grid, block, 0, stream, a, b, c, m,
                               n, k);
        }
    } else {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((n + 255) / 256), static_cast<unsigned>(m));
        if (fast) {
            hipLaunchKernelGGL(gemv_nt_rocmfp4_untiled_kernel<17>, grid, block, 0, stream, a, b, c, m,
                               n, k);
        } else {
            hipLaunchKernelGGL(gemv_nt_rocmfp4_untiled_kernel<18>, grid, block, 0, stream, a, b, c, m,
                               n, k);
        }
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

/* ROCmFP4 FAST (ggml_type_id=101): 17-byte single-scale blocks. */
extern "C" int allpaca_gemm_nt_rocmfp4_fast(const float *a, const uint8_t *b, float *c, int32_t m,
                                            int32_t n, int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_rocmfp4(a, b, c, m, n, k, as_stream(stream), 1, true);
}

/* NVFP4 (ggml type 40) after the host re-pack: 20-byte blocks of 32 weights
 * (qs[16], two f16 scales with the tensor scale folded in). Int8 path only,
 * so K must be a multiple of 32. */
extern "C" int allpaca_gemm_nt_nvfp4(const float *a, const uint8_t *b, float *c, int32_t m,
                                     int32_t n, int32_t k, allpaca_stream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    return launch_rocmfp4_i8<20>(a, b, c, m, n, k, as_stream(stream));
}

/* Float-activation ROCmFP4 GEMV (32 lanes split K): the kernel the int8
 * path replaced. Kept for A/B and as the exact-float reference. `fast`
 * selects 17-byte type-101 blocks. */
extern "C" int allpaca_gemm_nt_rocmfp4_f32(const float *a, const uint8_t *b, float *c, int32_t m,
                                           int32_t n, int32_t k, int32_t fast,
                                           allpaca_stream_t stream) {
    return launch_gemm_nt_rocmfp4(a, b, c, m, n, k, as_stream(stream), 2, fast != 0);
}

/* MoESQ paired-4:8 FP4 (Allpaca tensor type 200): 12-byte blocks of 32 weights. Each 8
 * weights are 4 pairs of which 2 are kept, a pair being one byte of two E2M1 codes (lower
 * index in the low nibble): 8 value bytes (2 per chunk of 8), 2 mask bytes (a nibble per
 * chunk, bit i = pair i kept), then the block's scale as an IEEE half. */
__constant__ const float k_e2m1_levels[16] = {0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
                                              -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__device__ inline float p48fp4_half(uint32_t h) {
    const uint32_t e = (h >> 10) & 0x1fu;
    const uint32_t man = h & 0x3ffu;
    float v;
    if (e == 0) {
        v = static_cast<float>(man) * 5.9604644775390625e-8f; /* 2^-24 */
    } else {
        const uint32_t bits = ((e + 112u) << 23) | (man << 13);
        __builtin_memcpy(&v, &bits, 4);
    }
    return (h & 0x8000u) ? -v : v;
}

/* Sum over one block of weight * a, before the block scale. Reads 4 of every 8 activations. */
__device__ inline float p48fp4_block_dot(const uint8_t *blk, const float *x, float *scale) {
#pragma clang fp contract(off)
    uint32_t w[3];
    __builtin_memcpy(w, blk, 12);
    *scale = p48fp4_half(w[2] >> 16);
    float s = 0.f;
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
        const uint32_t nib = (w[2] >> (4 * chunk)) & 0xfu;
        if (nib == 0u) {
            continue;
        }
        const int first = __builtin_ctz(nib);
        const int second = 31 - __builtin_clz(nib);
        const uint32_t pairs = w[chunk >> 1] >> (16 * (chunk & 1));
        const float *xc = x + chunk * 8;
        s = s + k_e2m1_levels[pairs & 0xfu] * xc[2 * first];
        s = s + k_e2m1_levels[(pairs >> 4) & 0xfu] * xc[2 * first + 1];
        s = s + k_e2m1_levels[(pairs >> 8) & 0xfu] * xc[2 * second];
        s = s + k_e2m1_levels[(pairs >> 12) & 0xfu] * xc[2 * second + 1];
    }
    return s;
}

/* C[m,n] = A[m,k] * W^T. Eight lanes share a column and split its blocks; a row's result does
 * not depend on m, so decode and batched rows agree bit for bit. */
__global__ void gemv_nt_p48fp4_kernel(const float *a, const uint8_t *b, float *c, int m, int n,
                                      int k) {
#pragma clang fp contract(off)
    constexpr int LPC = 8;
    const int sl = static_cast<int>(threadIdx.x) % LPC;
    const int col = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) / LPC) +
                    static_cast<int>(threadIdx.x) / LPC;
    const int row = static_cast<int>(blockIdx.y);
    const bool live = col < n && row < m;
    const int kb = k / 32;
    float acc = 0.f;
    if (live) {
        const uint8_t *brow = b + static_cast<size_t>(col) * static_cast<size_t>(kb) * 12;
        const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
        for (int ib = sl; ib < kb; ib += LPC) {
            float scale;
            const float s = p48fp4_block_dot(brow + static_cast<size_t>(ib) * 12, arow + ib * 32,
                                             &scale);
            acc = acc + scale * s;
        }
    }
#pragma unroll
    for (int off = LPC / 2; off > 0; off >>= 1) {
        acc = acc + __shfl_down(acc, off, LPC);
    }
    if (live && sl == 0) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
    }
}

extern "C" int allpaca_gemm_nt_p48fp4(const float *a, const uint8_t *b, float *c, int32_t m,
                                      int32_t n, int32_t k, allpaca_stream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || (k % 32) != 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    constexpr int COLS = 256 / 8;
    hipLaunchKernelGGL(gemv_nt_p48fp4_kernel,
                       dim3(static_cast<unsigned>((n + COLS - 1) / COLS), static_cast<unsigned>(m)),
                       dim3(256), 0, as_stream(stream), a, b, c, m, n, k);
    return hip_status(hipGetLastError());
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

/* MXFP4 code -> value without a table load: magnitudes 0,1,2,3,4,6,8,12,
 * bit 3 is the sign. */
__device__ inline float mxfp4_code(uint32_t n) {
    const float mag =
        static_cast<float>((0x0C08060403020100ull >> ((n & 7u) * 8u)) & 0xFFull);
    return (n & 8u) ? -mag : mag;
}

/* Unscaled dot product of one 17-byte block with 32 inputs (16-byte aligned).
 * Byte j of the block holds elements j (low nibble) and j + 16 (high). */
__device__ inline float mxfp4_block_dot(const uint8_t *blk, const float *x) {
    uint64_t q0;
    uint64_t q1;
    __builtin_memcpy(&q0, blk + 1, 8);
    __builtin_memcpy(&q1, blk + 9, 8);
    float xs[32];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        const float4 v = *reinterpret_cast<const float4 *>(x + 4 * q);
        xs[4 * q] = v.x;
        xs[4 * q + 1] = v.y;
        xs[4 * q + 2] = v.z;
        xs[4 * q + 3] = v.w;
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t b0 = static_cast<uint32_t>(q0 >> (8 * j)) & 0xFFu;
        const uint32_t b1 = static_cast<uint32_t>(q1 >> (8 * j)) & 0xFFu;
        s += xs[j] * mxfp4_code(b0 & 15u) + xs[16 + j] * mxfp4_code(b0 >> 4);
        s += xs[8 + j] * mxfp4_code(b1 & 15u) + xs[24 + j] * mxfp4_code(b1 >> 4);
    }
    return s;
}

/* Memory-bound MXFP4 GEMV: one 32-lane group per output column, each lane
 * taking every 32nd block of the row, so a group streams the row's blocks in
 * order. `k` is a multiple of 32 and `a` is 16-byte aligned. */
__global__ void gemv_nt_mxfp4_wide_kernel(const float *a, const uint8_t *b, float *c, int m, int n,
                                          int k) {
    constexpr int GROUPS = 8;
    const int row = static_cast<int>(blockIdx.y);
    const int grp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int col = static_cast<int>(blockIdx.x) * GROUPS + grp;
    if (row >= m) {
        return;
    }
    const int nblocks = k / 32;
    float acc = 0.f;
    if (col < n) {
        const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
        const uint8_t *brow = b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 17;
        for (int ib = lane; ib < nblocks; ib += 32) {
            const uint8_t *blk = brow + static_cast<size_t>(ib) * 17;
            acc += mxfp4_e8m0_half(blk[0]) * mxfp4_block_dot(blk, arow + ib * 32);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor(acc, off, 32);
    }
    if (col < n && lane == 0) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + col] = acc;
    }
}

__device__ inline float mxfp4_swiglu(float g, float u, float limit);

/* The same layout for the fused gate/up step: writes silu(gate)*up. */
__global__ void mxfp4_swiglu_gate_up_wide_kernel(const float *a, const uint8_t *bg,
                                                 const uint8_t *bu, float *c, int m, int n,
                                                 int k, float limit) {
    constexpr int GROUPS = 8;
    const int row = static_cast<int>(blockIdx.y);
    const int grp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int col = static_cast<int>(blockIdx.x) * GROUPS + grp;
    if (row >= m) {
        return;
    }
    const int nblocks = k / 32;
    float acc_g = 0.f;
    float acc_u = 0.f;
    if (col < n) {
        const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
        const uint8_t *grow = bg + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 17;
        const uint8_t *urow = bu + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 17;
        for (int ib = lane; ib < nblocks; ib += 32) {
            const uint8_t *g = grow + static_cast<size_t>(ib) * 17;
            const uint8_t *u = urow + static_cast<size_t>(ib) * 17;
            const float *x = arow + ib * 32;
            acc_g += mxfp4_e8m0_half(g[0]) * mxfp4_block_dot(g, x);
            acc_u += mxfp4_e8m0_half(u[0]) * mxfp4_block_dot(u, x);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc_g += __shfl_xor(acc_g, off, 32);
        acc_u += __shfl_xor(acc_u, off, 32);
    }
    if (col < n && lane == 0) {
        const float x = acc_g;
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + col] =
            mxfp4_swiglu(x, acc_u, limit);
    }
}

/* ALLPACA_MXFP4_WIDE=0 keeps the older shared-tile kernels for single rows. */
static bool mxfp4_wide(const float *a, int32_t m, int32_t k) {
    static const bool on = [] {
        const char *v = std::getenv("ALLPACA_MXFP4_WIDE");
        return !(v && v[0] == '0');
    }();
    return on && m == 1 && k % 32 == 0 && (reinterpret_cast<uintptr_t>(a) & 15u) == 0;
}

static int launch_gemm_nt_mxfp4(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                int32_t k, hipStream_t stream, int tiled) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    if (tiled && mxfp4_wide(a, m, k)) {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((n + 7) / 8), 1);
        hipLaunchKernelGGL(gemv_nt_mxfp4_wide_kernel, grid, block, 0, stream, a, b, c, m, n, k);
        return hip_status(hipGetLastError());
    }
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

/* silu(gate) * up; with limit > 0 the gate is capped at limit first and up to +-limit
 * (DeepSeek-V4 swiglu_clamp_exp). */
__device__ inline float mxfp4_swiglu(float g, float u, float limit) {
    if (limit > 0.0f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    return mxfp4_silu(g) * u;
}

/* One A-tile load, two MXFP4 B rows, write silu(gate)*up. */
__global__ void mxfp4_swiglu_gate_up_tiled_kernel(const float *a, const uint8_t *bg,
                                                  const uint8_t *bu, float *c, int m, int n,
                                                  int k, float limit) {
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
            mxfp4_swiglu(acc_g, acc_u, limit);
    }
}

/* Same B-reuse as tiled-M GEMM: dequant gate/up once per column across TILE_M rows. */
template <int TILE_M>
__global__ void mxfp4_swiglu_gate_up_tiled_m_kernel(const float *a, const uint8_t *bg,
                                                    const uint8_t *bu, float *c, int m, int n,
                                                    int k, float limit) {
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
                    mxfp4_swiglu(acc_g[r], acc_u[r], limit);
            }
        }
    }
}

extern "C" int allpaca_mxfp4_swiglu_gate_up(const float *a, const uint8_t *bg, const uint8_t *bu,
                                            float *c, int32_t m, int32_t n, int32_t k,
                                            float limit, allpaca_stream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !bg || !bu || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    if (mxfp4_wide(a, m, k)) {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((n + 7) / 8), 1);
        hipLaunchKernelGGL(mxfp4_swiglu_gate_up_wide_kernel, grid, block, 0, as_stream(stream), a,
                           bg, bu, c, m, n, k, limit);
        return hip_status(hipGetLastError());
    }
    constexpr int COLS = 64;
    if (m > 1) {
        dim3 block(64);
        if (m > 8) {
            constexpr int TILE_M = 16;
            dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS),
                      static_cast<unsigned>((m + TILE_M - 1) / TILE_M));
            hipLaunchKernelGGL((mxfp4_swiglu_gate_up_tiled_m_kernel<16>), grid, block, 0,
                               as_stream(stream), a, bg, bu, c, m, n, k, limit);
        } else {
            constexpr int TILE_M = 8;
            dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS),
                      static_cast<unsigned>((m + TILE_M - 1) / TILE_M));
            hipLaunchKernelGGL((mxfp4_swiglu_gate_up_tiled_m_kernel<8>), grid, block, 0,
                               as_stream(stream), a, bg, bu, c, m, n, k, limit);
        }
    } else {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((n + COLS - 1) / COLS), static_cast<unsigned>(m));
        hipLaunchKernelGGL(mxfp4_swiglu_gate_up_tiled_kernel, grid, block, 0, as_stream(stream), a,
                           bg, bu, c, m, n, k, limit);
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

extern "C" uint64_t allpaca_hip_kernel_launches(void) {
    return g_kernel_launches.load(std::memory_order_relaxed);
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

/* hipDeviceProp_t::multiProcessorCount: CUs on CDNA/GCN, WGPs on RDNA. */
extern "C" int allpaca_hip_device_multiprocessors(int32_t id, int32_t *count) {
    if (!count) {
        return ALLPACA_ERR_INVALID;
    }
    hipDeviceProp_t prop{};
    hipError_t err = hipGetDeviceProperties(&prop, id);
    if (err != hipSuccess) {
        *count = 0;
        return ALLPACA_ERR_DEVICE;
    }
    *count = prop.multiProcessorCount;
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

/* Device buffers as physically contiguous allocations: the default on Windows only. On Linux
 * (ROCm 6.2, MI50) kernels read contiguous buffers at about half the speed of hipMalloc ones.
 * ALLPACA_HIP_CONTIGUOUS=0/1 overrides. */
static bool hip_malloc_contiguous() {
    static const bool on = [] {
        const char *v = std::getenv("ALLPACA_HIP_CONTIGUOUS");
        if (v && v[0]) {
            return v[0] != '0';
        }
#if defined(_WIN32)
        return true;
#else
        return false;
#endif
    }();
    return on;
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
    err = hip_malloc_contiguous() ? hipExtMallocWithFlags(ptr, bytes, hipDeviceMallocContiguous)
                                  : hipErrorUnknown;
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
    note_device_write();
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
 * attribute pointer is always null and its layout never matters here.
 *
 * Off unless ALLPACA_MEMCPY_BATCH=1. The SDK 7.1 Windows runtime does not
 * export it after all, so the per-copy path is what every measured run used.
 * The ROCm 10 runtime exports it, and there it reads the source when the
 * stream runs, not during the call as hipMemcpyAsync does for pageable
 * memory. Callers reuse their source once the call returns, and DeepSeek-V4
 * expert uploads came out corrupted (non-finite logits). */
typedef hipError_t (*allpaca_memcpy_batch_fn)(void **dsts, void **srcs, size_t *sizes, size_t count,
                                              void *attrs, size_t *attrsIdxs, size_t numAttrs,
                                              size_t *failIdx, hipStream_t stream);

static allpaca_memcpy_batch_fn memcpy_batch_fn() {
    static allpaca_memcpy_batch_fn fn = []() -> allpaca_memcpy_batch_fn {
        const char *want = std::getenv("ALLPACA_MEMCPY_BATCH");
        if (!want || want[0] != '1') {
            return nullptr;
        }
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
        if (to_hip_memcpy_kind(kind) != hipMemcpyDeviceToHost) {
            note_device_write();
        }
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

// Queue a fill on `stream`, including the null stream. `allpaca_hip_memset`
// uses the synchronous call for the null stream, which finishes the work
// before it returns and cannot stand in for queued source compute.
extern "C" int allpaca_hip_memset_async(void *dst, int32_t value, size_t bytes,
                                        allpaca_stream_t stream) {
    if (!dst && bytes) {
        return ALLPACA_ERR_INVALID;
    }
    if (bytes == 0) {
        return ALLPACA_OK;
    }
    return hip_status(hipMemsetAsync(dst, value & 0xff, bytes, as_stream(stream)));
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

/* SDK 7.1 `hipStreamWaitEvent` on the null stream drains that event on the
   CPU (measured ~13 ms against a full source backlog). A blocking stream's
   wait returns immediately, and the destination null stream joins every
   blocking stream on that device, so later null-stream work still happens
   after the copy. Events stay in a pool: destroying one that is still
   recorded can sync, which is the stall this hop exists to avoid. */
static constexpr int kPeerDevs = 16;
static constexpr int kPeerEvents = 32;

struct PeerSrcEvents {
    hipEvent_t events[kPeerEvents]{};
    int n = 0;
};

static std::mutex g_peer_mu;
/* Indexed by the device an event is recorded on: the source for `ready`,
   the destination for `copied` (peer_hold_source). */
static PeerSrcEvents g_peer_src[kPeerDevs]{};
/* Process lifetime. Destroying a join stream waits for the hop queued on it. */
static hipStream_t g_peer_join[kPeerDevs]{};

static hipEvent_t peer_take_event(PeerSrcEvents &pool) {
    for (int i = 0; i < pool.n;) {
        if (hipEventQuery(pool.events[i]) == hipSuccess) {
            hipEvent_t ready = pool.events[i];
            pool.events[i] = pool.events[pool.n - 1];
            pool.n--;
            return ready;
        }
        clear_last_error();
        ++i;
    }
    hipEvent_t ready = nullptr;
    if (hipEventCreateWithFlags(&ready, hipEventDisableTiming) != hipSuccess) {
        clear_last_error();
        return nullptr;
    }
    return ready;
}

static void peer_stash_event(PeerSrcEvents &pool, hipEvent_t ready) {
    if (!ready) {
        return;
    }
    if (pool.n < kPeerEvents) {
        pool.events[pool.n++] = ready;
        return;
    }
    /* More than 32 hops from one device are still queued. Waiting for the
       oldest is the overflow path; the steady hop never fills the pool. */
    (void)hipEventSynchronize(pool.events[0]);
    (void)hipEventDestroy(pool.events[0]);
    pool.events[0] = ready;
}

static hipStream_t peer_join_stream(int32_t dst_dev) {
    hipStream_t &join = g_peer_join[dst_dev];
    if (!join && hipStreamCreate(&join) != hipSuccess) {
        join = nullptr;
        clear_last_error();
    }
    return join;
}

/* The hop used to return only after the copy, so callers reuse `src` at once.
   The source device's join stream waits for the copy and its null stream
   joins that stream, so a later source write cannot reach the copy. Called
   with `dst_dev` current; leaves `src_dev` current. */
static hipError_t peer_hold_source(hipStream_t copy_stream, int32_t dst_dev, int32_t src_dev) {
    PeerSrcEvents &pool = g_peer_src[dst_dev];
    hipEvent_t copied = peer_take_event(pool);
    if (!copied) {
        return hipErrorOutOfMemory;
    }
    hipError_t err = hipEventRecord(copied, copy_stream);
    if (err == hipSuccess) {
        err = hipSetDevice(src_dev);
    }
    hipStream_t hold = err == hipSuccess ? peer_join_stream(src_dev) : nullptr;
    if (err == hipSuccess && !hold) {
        err = hipErrorOutOfMemory;
    }
    if (err == hipSuccess) {
        err = hipStreamWaitEvent(hold, copied, 0);
    }
    peer_stash_event(pool, copied);
    return err;
}

extern "C" int allpaca_hip_memcpy_peer(void *dst, int32_t dst_dev, const void *src, int32_t src_dev,
                                      size_t bytes, allpaca_stream_t stream) {
    if (!dst || !src) {
        return ALLPACA_ERR_INVALID;
    }
    if (src_dev < 0 || dst_dev < 0 || src_dev >= kPeerDevs || dst_dev >= kPeerDevs) {
        return ALLPACA_ERR_DEVICE;
    }
    int prev = 0;
    if (hipGetDevice(&prev) != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    /* The source null stream holds the compute that produced `src`.
       hipMemcpyPeerAsync does not wait for that stream, and the HIP SDK 7.1
       host-staged peer path does not drain it either (ROCm #10553), so a
       ping-pong split used to copy a stale activation. */
    std::lock_guard<std::mutex> lock(g_peer_mu);
    if (hipSetDevice(src_dev) != hipSuccess) {
        (void)hipSetDevice(prev);
        return ALLPACA_ERR_DEVICE;
    }
    PeerSrcEvents &pool = g_peer_src[src_dev];
    hipEvent_t ready = peer_take_event(pool);
    if (!ready) {
        (void)hipSetDevice(prev);
        return ALLPACA_ERR_DEVICE;
    }
    hipError_t err = hipEventRecord(ready, nullptr);
    if (err != hipSuccess) {
        peer_stash_event(pool, ready);
        (void)hipSetDevice(prev);
        return ALLPACA_ERR_DEVICE;
    }
    if (hipSetDevice(dst_dev) != hipSuccess) {
        peer_stash_event(pool, ready);
        (void)hipSetDevice(prev);
        return ALLPACA_ERR_DEVICE;
    }
    hipStream_t s = as_stream(stream);
    if (!s) {
        s = peer_join_stream(dst_dev);
    }
    if (!s) {
        peer_stash_event(pool, ready);
        (void)hipSetDevice(prev);
        return ALLPACA_ERR_DEVICE;
    }
    err = hipStreamWaitEvent(s, ready, 0);
    if (err == hipSuccess) {
        err = hipMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, s);
    }
    peer_stash_event(pool, ready);
    if (err == hipSuccess) {
        err = peer_hold_source(s, dst_dev, src_dev);
    }
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

/* One small rocBLAS GEMM on `device`. The first GEMM a process runs on a
 * device pays the library's start-up there, whatever its size (about 4 s on
 * an RX 9070 with ROCm 10.1, 0.2-0.4 s on an RX 9060 XT). Calling this at
 * load keeps that out of the first request. */
extern "C" int allpaca_hip_blas_warmup(int32_t device) {
#ifdef ALLPACA_HAS_ROCBLAS
    if (hipSetDevice(device) != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    constexpr int32_t N = 16;
    const size_t bytes = static_cast<size_t>(3) * N * N * sizeof(float);
    void *mem = nullptr;
    if (hipMalloc(&mem, bytes) != hipSuccess || !mem) {
        clear_last_error();
        return ALLPACA_ERR_DEVICE;
    }
    (void)hipMemset(mem, 0, bytes);
    float *buf = static_cast<float *>(mem);
    const int rc = launch_gemm_nt_rocblas(buf, buf + N * N, buf + 2 * N * N, N, N, N, nullptr);
    (void)hipDeviceSynchronize();
    (void)hipFree(mem);
    clear_last_error();
    return rc;
#else
    (void)device;
    return ALLPACA_ERR_NOT_IMPLEMENTED;
#endif
}

extern "C" int allpaca_hip_stream_synchronize(allpaca_stream_t stream) {
    return hip_status(hipStreamSynchronize(as_stream(stream)));
}

/* Debug only (P21): wait for every stream on the current device. */
extern "C" int allpaca_hip_device_synchronize(void) {
    return hip_status(hipDeviceSynchronize());
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

__global__ void gemv_nt_q4_0_f32_kernel(const float *a, const uint8_t *b, float *c, int m, int n,
                                        int k) {
    constexpr int WARPS = 8;
    const int row = static_cast<int>(blockIdx.y);
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int col = static_cast<int>(blockIdx.x) * WARPS + warp;
    const int nblocks = k / 32;
    float acc = 0.0f;
    if (col < n) {
        const float *arow = a + static_cast<size_t>(row) * static_cast<size_t>(k);
        const uint8_t *brow = b + static_cast<size_t>(col) * static_cast<size_t>(nblocks) * 18;
        for (int blk = lane; blk < nblocks; blk += 32) {
            const uint8_t *bp = brow + static_cast<size_t>(blk) * 18;
            const float d = allpaca_f16_bits_to_f32(static_cast<uint16_t>(bp[0]) |
                                                    (static_cast<uint16_t>(bp[1]) << 8));
            const int base = blk * 32;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const uint8_t packed = bp[2 + j];
                acc = __fmaf_rn(arow[base + j], d * static_cast<float>((packed & 15) - 8), acc);
                acc = __fmaf_rn(arow[base + j + 16], d * static_cast<float>((packed >> 4) - 8), acc);
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, 32);
    }
    if (col < n && lane == 0) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + col] = acc;
    }
}

static int launch_gemm_nt_q4_0(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                               int32_t k, hipStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || k % 32 != 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    dim3 grid(static_cast<unsigned>((n + 7) / 8), static_cast<unsigned>(m));
    hipLaunchKernelGGL(gemv_nt_q4_0_f32_kernel, grid, dim3(256), 0, stream, a, b, c, m, n, k);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_gemm_nt_q4_0(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n,
                                    int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_q4_0(a, b, c, m, n, k, as_stream(stream));
}

extern "C" int allpaca_gemm_nt_f16(const float *a, const uint16_t *b, float *c, int32_t m, int32_t n,
                                  int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_f16(a, b, c, m, n, k, as_stream(stream), 1);
}

extern "C" int allpaca_gemm_nt_f16_wmma(const float *a, const uint16_t *b, float *c, int32_t m,
                                       int32_t n, int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_f16_wmma(a, b, c, m, n, k, as_stream(stream));
}

extern "C" int allpaca_gemm_nt_i8(const int8_t *a, const int8_t *b, float *c, int32_t m, int32_t n,
                                  int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_i8(a, b, c, m, n, k, as_stream(stream));
}

extern "C" int allpaca_gemm_nt_i8_wmma(const int8_t *a, const int8_t *b, float *c, int32_t m,
                                      int32_t n, int32_t k, allpaca_stream_t stream) {
    return launch_gemm_nt_i8_wmma(a, b, c, m, n, k, as_stream(stream));
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
    launch_attention(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, static_cast<const uint8_t *>(nullptr),
                     static_cast<const float *>(nullptr), as_stream(stream));
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_attention_wmma_prefill(const float *q, const float *k, const float *v, float *out,
                                             int32_t n_q, int32_t n_kv, int32_t n_heads,
                                             int32_t n_kv_heads, int32_t head_dim, int32_t causal,
                                             allpaca_stream_t stream) {
    if (!q || !k || !v || !out || n_q <= 0 || n_kv <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_dim <= 0 || (n_heads % n_kv_heads) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    if (!device_arch_is_gfx12(dev) || n_q <= 1 || (head_dim % 16) != 0 || head_dim > 128) {
        return allpaca_attention(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, stream);
    }
    clear_last_error();
    dim3 block(32);
    dim3 grid(static_cast<unsigned>((n_q + 15) / 16), static_cast<unsigned>(n_heads));
    hipLaunchKernelGGL(attention_wmma_prefill_kernel, grid, block, 0, as_stream(stream), q, k, v, out, n_q,
                       n_kv, n_heads, n_kv_heads, head_dim, causal);
    note_wmma_launch();
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
    launch_attention(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, static_cast<const uint8_t *>(nullptr),
                     sinks, as_stream(stream));
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
    launch_attention(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, 0, mask,
                     static_cast<const float *>(nullptr), as_stream(stream));
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
        local = fmaf(v, v, local);
    }
    const float ss = block_reduce_sum(local, red);
    const float inv = det_rsqrtf(__fadd_rn(__fdiv_rn(ss, static_cast<float>(n)), eps));
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        out[i] = __fmul_rn(__fmul_rn(x[i], inv), weight[i]);
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
        local = fmaf(v, v, local);
    }
    const float ss = block_reduce_sum(local, red);
    const float inv = det_rsqrtf(__fadd_rn(__fdiv_rn(ss, static_cast<float>(n)), eps));
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        out[i] = __fmul_rn(__fmul_rn(x[i], inv), weight[i]);
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
    const float inv_q = det_rsqrtf(ss_q + eps);
    const float inv_k = det_rsqrtf(ss_k + eps);
    const float q_scale = det_rsqrtf(static_cast<float>(head_k));
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
        ss_q = fmaf(q_h[j], q_h[j], ss_q);
        ss_k = fmaf(k_h[j], k_h[j], ss_k);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss_q = __fadd_rn(ss_q, __shfl_xor(ss_q, off, 32));
        ss_k = __fadd_rn(ss_k, __shfl_xor(ss_k, off, 32));
    }
    const float sq = __fmul_rn(det_rsqrtf(ss_q + eps), det_rsqrtf(static_cast<float>(head_k)));
    const float sk = det_rsqrtf(ss_k + eps);
    for (int j = lane; j < head_k; j += 32) {
        q_h[j] = __fmul_rn(q_h[j], sq);
        k_h[j] = __fmul_rn(k_h[j], sk);
    }
}

/* Delta-rule state update on already-normalized q/k. One warp per state row
 * (each lane holds NJ = head_k / 32 contiguous columns), reductions are warp
 * shuffles, and a head's rows spread over several blocks. The block-per-head
 * kernel walked all head_v rows with two block-wide reductions each (~768
 * barriers per call on n_v_heads blocks) and was pure latency. */
constexpr int GDN_ROWS_PER_WARP = 4;
constexpr int GDN_WARPS = 4;
/* `n_k` > 0: q and k hold n_k heads and V head h reads K head
 * h / (n_v_heads / n_k) (interleave) or h % n_k (tile), instead of an expanded
 * per-V-head copy. */
template <int NJ>
__global__ void gated_delta_step_warp_kernel(const float *q, const float *k, const float *v,
                                             const float *decay, const float *beta, float *state,
                                             float *out, int n_v_heads, int head_v, int head_k,
                                             int n_k, int interleave) {
    const int h = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    if (h >= n_v_heads) {
        return;
    }
    const int kh = n_k > 0 ? (interleave ? h / (n_v_heads / n_k) : h % n_k) : h;
    const float *q_h = q + static_cast<size_t>(kh) * head_k + lane * NJ;
    const float *k_h = k + static_cast<size_t>(kh) * head_k + lane * NJ;
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
            s[t] = __fmul_rn(row[t], g);
            acc = fmaf(s[t], kv[t], acc);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc = __fadd_rn(acc, __shfl_xor(acc, off, 32));
        }
        const float delta = __fmul_rn(b, __fsub_rn(v[static_cast<size_t>(h) * head_v + i], acc));
        float y = 0.f;
#pragma unroll
        for (int t = 0; t < NJ; ++t) {
            s[t] = __fadd_rn(s[t], __fmul_rn(delta, kv[t]));
            row[t] = s[t];
            y = fmaf(s[t], qv[t], y);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            y = __fadd_rn(y, __shfl_xor(y, off, 32));
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
    const float inv_q = det_rsqrtf(ss_q + eps);
    const float inv_k = det_rsqrtf(ss_k + eps);
    const float q_scale = det_rsqrtf(static_cast<float>(head_k));
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
    const float freq = det_powf(theta, -static_cast<float>(i) / static_cast<float>(half));
    det_rotate(hvec[2 * i], hvec[2 * i + 1], __fmul_rn(static_cast<float>(pos), freq), 1);
}

__device__ void rope_back_pair(float *hvec, int i, int half, int pos, float theta) {
    const float freq = det_powf(theta, -static_cast<float>(i) / static_cast<float>(half));
    det_rotate(hvec[2 * i], hvec[2 * i + 1], __fmul_rn(static_cast<float>(pos), freq), 0);
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

/* rope_offset_rows with YaRN frequencies: pair i turns by theta^(-i/half) times a blend of 1
 * (pairs up to `low`) and `scale` (pairs from `high`). Magnitudes are not scaled. */
__global__ void rope_yarn_rows_kernel(float *x, int n_tok, int n_heads, int head_dim, int n_rot,
                                      int offset, int start_pos, float theta, float scale,
                                      float low, float high, int back) {
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
    const float y = (static_cast<float>(i) - low) / fmaxf(0.001f, high - low);
    const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
    const float freq = __fmul_rn(det_powf(theta, -static_cast<float>(i) / static_cast<float>(half)),
                                 fmaf(scale, 1.0f - ramp, ramp));
    det_rotate(hvec[2 * i], hvec[2 * i + 1], __fmul_rn(static_cast<float>(start_pos + tok), freq), !back);
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
    gate[idx] = det_silu_mul(gate[idx], up[idx]);
}

/* silu(min(gate, limit)) * clamp(up, -limit, limit): DeepSeek-V4's capped
 * SwiGLU for the BF16 shared expert, same element math as mxfp4_swiglu. */
__global__ void silu_mul_limit_kernel(float *gate, const float *up, int n, float limit) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    gate[idx] = mxfp4_swiglu(gate[idx], up[idx], limit);
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
    attn[idx] = det_sigmoid_mul(attn[idx], s);
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

__global__ void gather_rows_kernel(float *dst, const float *src, const int32_t *rows, int n,
                                   int row_len) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n * row_len) {
        return;
    }
    const size_t row = static_cast<size_t>(rows[idx / row_len]);
    dst[idx] = src[row * static_cast<size_t>(row_len) + static_cast<size_t>(idx % row_len)];
}

__global__ void sigmoid_mul_kernel(float *x, const float *gate, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    const float g = gate[idx];
    x[idx] = det_sigmoid_mul(x[idx], g);
}

__global__ void rmsnorm_heads_kernel(float *x, const float *weight, int head_dim, float eps) {
    // One block per head; x row at blockIdx.x.
    float *row = x + static_cast<size_t>(blockIdx.x) * static_cast<size_t>(head_dim);
    __shared__ float shared[256];
    float sum = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < head_dim; i += static_cast<int>(blockDim.x)) {
        const float v = row[i];
        sum = fmaf(v, v, sum);
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv = det_rsqrtf(__fadd_rn(__fdiv_rn(shared[0], static_cast<float>(head_dim)), eps));
    for (int i = static_cast<int>(threadIdx.x); i < head_dim; i += static_cast<int>(blockDim.x)) {
        row[i] = __fmul_rn(__fmul_rn(row[i], inv), weight[i]);
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
                       state, out, n_v_heads, head_v, head_k, 0, 0)
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

__device__ inline float gdn_silu(float x) {
    return __fdiv_rn(x, __fadd_rn(1.0f, det_expf(-x)));
}

__device__ inline float gdn_softplus(float x) {
    return (x > 20.0f) ? x : det_logf(__fadd_rn(1.0f, det_expf(x)));
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

/* V head vh reads K head vh % n_k (tile, Qwen3.5) or vh / (n_v / n_k)
 * (interleave, Qwen3-Next). */
__global__ void expand_k_heads_kernel(const float *src, float *dst, int n_k, int n_v, int head_k,
                                      int interleave) {
    const int vh = static_cast<int>(blockIdx.x);
    const int j = static_cast<int>(threadIdx.x);
    if (vh >= n_v || j >= head_k) {
        return;
    }
    const int kh = interleave ? vh / (n_v / n_k) : vh % n_k;
    dst[vh * head_k + j] = src[kh * head_k + j];
}

__global__ void gdn_decay_beta_kernel(const float *alpha, const float *dt_bias, const float *a,
                                      float *decay, float *beta, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    decay[i] = det_expf(__fmul_rn(gdn_softplus(__fadd_rn(alpha[i], dt_bias[i])), a[i]));
    beta[i] = __fdiv_rn(1.0f, __fadd_rn(1.0f, det_expf(-beta[i])));
}

/* One GDN decode step's element-wise front: thread c < qkv_dim runs the
 * depthwise conv + SiLU for channel c, then shifts that channel's history
 * (as gdn_conv1d_compute then gdn_conv1d_shift did); the next n_v threads
 * compute decay and sigmoid(beta) (as gdn_decay_beta did). */
__global__ void gdn_step_prep_kernel(const float *x, const float *kernel, float *conv_state,
                                     float *out, int qkv_dim, int conv_kernel, const float *alpha,
                                     const float *dt_bias, const float *a, float *decay,
                                     float *beta, int n_v) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= qkv_dim) {
        const int h = i - qkv_dim;
        if (h < n_v) {
            decay[h] = det_expf(__fmul_rn(gdn_softplus(__fadd_rn(alpha[h], dt_bias[h])), a[h]));
            beta[h] = __fdiv_rn(1.0f, __fadd_rn(1.0f, det_expf(-beta[h])));
        }
        return;
    }
    const int c = i;
    const int hist = conv_kernel - 1;
    float acc = 0.0f;
    for (int j = 0; j < hist; ++j) {
        acc += kernel[j + c * conv_kernel] * conv_state[j * qkv_dim + c];
    }
    acc += kernel[(conv_kernel - 1) + c * conv_kernel] * x[c];
    out[c] = gdn_silu(acc);
    if (hist <= 0) {
        return;
    }
    for (int j = 0; j < hist - 1; ++j) {
        conv_state[j * qkv_dim + c] = conv_state[(j + 1) * qkv_dim + c];
    }
    conv_state[(hist - 1) * qkv_dim + c] = x[c];
}

/* A whole GDN decode step in three launches (was seven): prep (conv, shift,
 * decay, beta), q/k L2 norm on the n_k K heads in place in `conv_out`, and the
 * delta step reading each V head's K head directly (no expanded copies).
 * Bit-identical to conv1d_step + 2x expand_k_heads + decay_beta +
 * gated_delta_step. ALLPACA_ERR_INVALID when the warp kernel does not cover
 * head_k (the caller falls back to the unfused sequence). Folding the L2 norm
 * into the delta step (2 launches) measured no gain and drifted ~1e-7. */
extern "C" int allpaca_gdn_step_fused(const float *qkv, const float *conv_w, float *conv_state,
                                      float *conv_out, const float *alpha, const float *dt_bias,
                                      const float *a, float *decay, float *beta, float *state,
                                      float *out, int32_t n_v, int32_t n_k, int32_t head_v,
                                      int32_t head_k, int32_t conv_kernel, float eps,
                                      int32_t interleave, allpaca_stream_t stream) {
    const int nj = head_k % 32 == 0 ? head_k / 32 : 0;
    if (!qkv || !conv_w || !conv_state || !conv_out || !alpha || !dt_bias || !a || !decay ||
        !beta || !state || !out || n_v <= 0 || n_k <= 0 || n_v % n_k != 0 || head_v <= 0 ||
        conv_kernel <= 0 || !(nj == 1 || nj == 2 || nj == 4 || nj == 8)) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipStream_t s = as_stream(stream);
    const int key_dim = n_k * head_k;
    const int qkv_dim = 2 * key_dim + n_v * head_v;
    const int threads = 256;
    const int blocks = (qkv_dim + n_v + threads - 1) / threads;
    hipLaunchKernelGGL(gdn_step_prep_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, s, qkv, conv_w, conv_state,
                       conv_out, qkv_dim, conv_kernel, alpha, dt_bias, a, decay, beta, n_v);
    float *q = conv_out;
    float *k = conv_out + key_dim;
    const float *v = conv_out + 2 * key_dim;
    hipLaunchKernelGGL(gdn_qk_l2norm_kernel, dim3(static_cast<unsigned>(n_k)), dim3(32), 0, s, q,
                       k, n_k, head_k, eps);
    constexpr int rows_per_block = GDN_WARPS * GDN_ROWS_PER_WARP;
    const dim3 grid(static_cast<unsigned>(n_v),
                    static_cast<unsigned>((head_v + rows_per_block - 1) / rows_per_block));
    const dim3 block(32 * GDN_WARPS);
    if (nj == 1) {
        hipLaunchKernelGGL(gated_delta_step_warp_kernel<1>, grid, block, 0, s, q, k, v, decay,
                           beta, state, out, n_v, head_v, head_k, n_k, interleave);
    } else if (nj == 2) {
        hipLaunchKernelGGL(gated_delta_step_warp_kernel<2>, grid, block, 0, s, q, k, v, decay,
                           beta, state, out, n_v, head_v, head_k, n_k, interleave);
    } else if (nj == 4) {
        hipLaunchKernelGGL(gated_delta_step_warp_kernel<4>, grid, block, 0, s, q, k, v, decay,
                           beta, state, out, n_v, head_v, head_k, n_k, interleave);
    } else {
        hipLaunchKernelGGL(gated_delta_step_warp_kernel<8>, grid, block, 0, s, q, k, v, decay,
                           beta, state, out, n_v, head_v, head_k, n_k, interleave);
    }
    return hip_status(hipGetLastError());
}

/* gdn_step_prep_kernel for `rows` tokens in order, in place: x[r] (qkv_dim)
 * becomes the token's conv output, alpha[r] (n_v) its decay and beta[r] its
 * sigmoid; each channel walks the tokens with its own history. */
__global__ void gdn_steps_prep_kernel(float *x, const float *kernel, float *conv_state,
                                      int qkv_dim, int conv_kernel, float *alpha,
                                      const float *dt_bias, const float *a, float *beta, int n_v,
                                      int rows) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= qkv_dim) {
        const int h = i - qkv_dim;
        if (h < n_v) {
            for (int r = 0; r < rows; ++r) {
                const size_t at = static_cast<size_t>(r) * n_v + h;
                alpha[at] = det_expf(__fmul_rn(gdn_softplus(__fadd_rn(alpha[at], dt_bias[h])), a[h]));
                beta[at] = __fdiv_rn(1.0f, __fadd_rn(1.0f, det_expf(-beta[at])));
            }
        }
        return;
    }
    const int c = i;
    const int hist = conv_kernel - 1;
    for (int r = 0; r < rows; ++r) {
        float *xr = x + static_cast<size_t>(r) * qkv_dim;
        const float in = xr[c];
        float acc = 0.0f;
        for (int j = 0; j < hist; ++j) {
            acc += kernel[j + c * conv_kernel] * conv_state[j * qkv_dim + c];
        }
        acc += kernel[(conv_kernel - 1) + c * conv_kernel] * in;
        xr[c] = gdn_silu(acc);
        if (hist > 0) {
            for (int j = 0; j < hist - 1; ++j) {
                conv_state[j * qkv_dim + c] = conv_state[(j + 1) * qkv_dim + c];
            }
            conv_state[(hist - 1) * qkv_dim + c] = in;
        }
    }
}

/* gdn_qk_l2norm_kernel on token blockIdx.y's q and k, `stride` floats apart. */
__global__ void gdn_qk_l2norm_rows_kernel(float *q, float *k, int n_heads, int head_k, float eps,
                                          int stride) {
    const size_t off = static_cast<size_t>(blockIdx.y) * static_cast<size_t>(stride);
    const int h = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (h >= n_heads) {
        return;
    }
    float *q_h = q + off + static_cast<size_t>(h) * head_k;
    float *k_h = k + off + static_cast<size_t>(h) * head_k;
    float ss_q = 0.f, ss_k = 0.f;
    for (int j = lane; j < head_k; j += 32) {
        ss_q = fmaf(q_h[j], q_h[j], ss_q);
        ss_k = fmaf(k_h[j], k_h[j], ss_k);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ss_q = __fadd_rn(ss_q, __shfl_xor(ss_q, o, 32));
        ss_k = __fadd_rn(ss_k, __shfl_xor(ss_k, o, 32));
    }
    const float sq = __fmul_rn(det_rsqrtf(ss_q + eps), det_rsqrtf(static_cast<float>(head_k)));
    const float sk = det_rsqrtf(ss_k + eps);
    for (int j = lane; j < head_k; j += 32) {
        q_h[j] = __fmul_rn(q_h[j], sq);
        k_h[j] = __fmul_rn(k_h[j], sk);
    }
}

/* gated_delta_step_warp_kernel for `rows` tokens in order: a warp keeps its
 * state rows in registers across the tokens, so the state is read and
 * written once a pass, not once a token. Token r's q, k and v start at
 * qkv + r * stride; its decay and beta at r * n_v_heads; its output at
 * out + r * n_v_heads * head_v. Each token's arithmetic is the one-token
 * kernel's, so a pass gives the bits its tokens give one at a time. */
template <int NJ>
__global__ void gated_delta_steps_warp_kernel(const float *qkv, int stride, int key_dim,
                                              const float *decay, const float *beta,
                                              float *state, float *out, int n_v_heads,
                                              int head_v, int head_k, int n_k, int interleave,
                                              int rows) {
    const int h = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    if (h >= n_v_heads) {
        return;
    }
    const int kh = n_k > 0 ? (interleave ? h / (n_v_heads / n_k) : h % n_k) : h;
    const int row0 = (static_cast<int>(blockIdx.y) * GDN_WARPS + warp) * GDN_ROWS_PER_WARP;
    float st[GDN_ROWS_PER_WARP][NJ];
#pragma unroll
    for (int r = 0; r < GDN_ROWS_PER_WARP; ++r) {
        const int i = row0 + r;
#pragma unroll
        for (int t = 0; t < NJ; ++t) {
            st[r][t] = i < head_v
                           ? state[(static_cast<size_t>(h) * head_v + i) * head_k + lane * NJ + t]
                           : 0.f;
        }
    }
    for (int tok = 0; tok < rows; ++tok) {
        const float *base = qkv + static_cast<size_t>(tok) * static_cast<size_t>(stride);
        const float *q_h = base + static_cast<size_t>(kh) * head_k + lane * NJ;
        const float *k_h = base + key_dim + static_cast<size_t>(kh) * head_k + lane * NJ;
        const float *v = base + 2 * static_cast<size_t>(key_dim);
        float qv[NJ], kv[NJ];
#pragma unroll
        for (int t = 0; t < NJ; ++t) {
            qv[t] = q_h[t];
            kv[t] = k_h[t];
        }
        const float g = decay[static_cast<size_t>(tok) * n_v_heads + h];
        const float b = beta[static_cast<size_t>(tok) * n_v_heads + h];
        float *out_t = out + static_cast<size_t>(tok) * n_v_heads * head_v;
#pragma unroll
        for (int r = 0; r < GDN_ROWS_PER_WARP; ++r) {
            const int i = row0 + r;
            if (i >= head_v) {
                break;
            }
            float s[NJ];
            float acc = 0.f;
#pragma unroll
            for (int t = 0; t < NJ; ++t) {
                s[t] = __fmul_rn(st[r][t], g);
                acc = fmaf(s[t], kv[t], acc);
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                acc = __fadd_rn(acc, __shfl_xor(acc, o, 32));
            }
            const float delta = __fmul_rn(b, __fsub_rn(v[static_cast<size_t>(h) * head_v + i], acc));
            float y = 0.f;
#pragma unroll
            for (int t = 0; t < NJ; ++t) {
                s[t] = __fadd_rn(s[t], __fmul_rn(delta, kv[t]));
                st[r][t] = s[t];
                y = fmaf(s[t], qv[t], y);
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                y = __fadd_rn(y, __shfl_xor(y, o, 32));
            }
            if (lane == 0) {
                out_t[static_cast<size_t>(h) * head_v + i] = y;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < GDN_ROWS_PER_WARP; ++r) {
        const int i = row0 + r;
        if (i < head_v) {
#pragma unroll
            for (int t = 0; t < NJ; ++t) {
                state[(static_cast<size_t>(h) * head_v + i) * head_k + lane * NJ + t] = st[r][t];
            }
        }
    }
}

/* allpaca_gdn_step_fused for `rows` tokens in three launches, in place:
 * qkv[r] (2 * n_k * head_k + n_v * head_v floats) becomes the token's
 * normed conv output, alpha[r] (n_v) its decay, beta[r] its sigmoid; out[r]
 * gets n_v * head_v. Bit-identical to `rows` one-token steps. */
extern "C" int allpaca_gdn_steps(float *qkv, const float *conv_w, float *conv_state, float *alpha,
                                 const float *dt_bias, const float *a, float *beta, float *state,
                                 float *out, int32_t n_v, int32_t n_k, int32_t head_v,
                                 int32_t head_k, int32_t conv_kernel, float eps,
                                 int32_t interleave, int32_t rows, allpaca_stream_t stream) {
    const int nj = head_k % 32 == 0 ? head_k / 32 : 0;
    if (!qkv || !conv_w || !conv_state || !alpha || !dt_bias || !a || !beta || !state || !out ||
        n_v <= 0 || n_k <= 0 || n_v % n_k != 0 || head_v <= 0 || conv_kernel <= 0 || rows <= 0 ||
        !(nj == 1 || nj == 2 || nj == 4 || nj == 8)) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipStream_t s = as_stream(stream);
    const int key_dim = n_k * head_k;
    const int qkv_dim = 2 * key_dim + n_v * head_v;
    const int threads = 256;
    const int blocks = (qkv_dim + n_v + threads - 1) / threads;
    hipLaunchKernelGGL(gdn_steps_prep_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, s, qkv, conv_w, conv_state,
                       qkv_dim, conv_kernel, alpha, dt_bias, a, beta, n_v, rows);
    hipLaunchKernelGGL(gdn_qk_l2norm_rows_kernel,
                       dim3(static_cast<unsigned>(n_k), static_cast<unsigned>(rows)), dim3(32), 0,
                       s, qkv, qkv + key_dim, n_k, head_k, eps, qkv_dim);
    constexpr int rows_per_block = GDN_WARPS * GDN_ROWS_PER_WARP;
    const dim3 grid(static_cast<unsigned>(n_v),
                    static_cast<unsigned>((head_v + rows_per_block - 1) / rows_per_block));
    const dim3 block(32 * GDN_WARPS);
#define GDN_STEPS(NJV)                                                                             \
    hipLaunchKernelGGL(gated_delta_steps_warp_kernel<NJV>, grid, block, 0, s, qkv, qkv_dim,      \
                       key_dim, alpha, beta, state, out, n_v, head_v, head_k, n_k, interleave,   \
                       rows)
    if (nj == 1) {
        GDN_STEPS(1);
    } else if (nj == 2) {
        GDN_STEPS(2);
    } else if (nj == 4) {
        GDN_STEPS(4);
    } else {
        GDN_STEPS(8);
    }
#undef GDN_STEPS
    return hip_status(hipGetLastError());
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
                                      int32_t head_k, int32_t interleave, allpaca_stream_t stream) {
    if (!src || !dst || n_k <= 0 || n_v <= 0 || head_k <= 0 || (n_v % n_k) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(expand_k_heads_kernel, dim3(static_cast<unsigned>(n_v)),
                       dim3(static_cast<unsigned>(head_k)), 0, as_stream(stream), src, dst, n_k, n_v,
                       head_k, interleave);
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

extern "C" int allpaca_rope_yarn_rows(float *x, int32_t n_tok, int32_t n_heads, int32_t head_dim,
                                      int32_t n_rot, int32_t offset, int32_t start_pos,
                                      float theta, float scale, float low, float high,
                                      int32_t back, allpaca_stream_t stream) {
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
    hipLaunchKernelGGL(rope_yarn_rows_kernel,
                       dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(n_tok)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), x, n_tok, n_heads,
                       head_dim, rot, off, start_pos, theta, scale, low, high, back);
    return hip_status(hipGetLastError());
}

/* NeoX pair on first n_rot dims; optional IMRoPE sector angles (matches CPU rope_imrope_neox). */
__device__ void rope_imrope_pair(float *hvec, int i, int half, int n_rot, int pos_t, int pos_h,
                                int pos_w, float theta, int s0, int s1, int s2, int s3) {
    const int sect_dims = s0 + s1 + s2 + s3;
    const bool use_imrope = sect_dims > 0 && s0 > 0 && s1 > 0;
    const float theta_scale = det_powf(theta, -2.0f / static_cast<float>(n_rot));
    float theta_t = static_cast<float>(pos_t);
    float theta_h = static_cast<float>(pos_h);
    float theta_w = static_cast<float>(pos_w);
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
    det_rotate(hvec[i], hvec[i + half], angle, 1);
}

__global__ void rope_imrope_qk_kernel(float *q, float *k, int n_q_heads, int n_k_heads, int head_dim,
                                      int n_rot, int pos_t, int pos_h, int pos_w, float theta,
                                      int s0, int s1, int s2, int s3) {
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
        rope_imrope_pair(q + h * head_dim, i, half, n_rot, pos_t, pos_h, pos_w, theta, s0, s1, s2,
                         s3);
    } else {
        const int j = idx - q_pairs;
        const int h = j / half;
        const int i = j % half;
        rope_imrope_pair(k + h * head_dim, i, half, n_rot, pos_t, pos_h, pos_w, theta, s0, s1, s2,
                         s3);
    }
}

extern "C" int allpaca_rope_imrope_qk(float *q, float *k, int32_t n_q_heads, int32_t n_k_heads,
                                      int32_t head_dim, int32_t n_rot, const int32_t pos[3],
                                      float theta, const int32_t sections[4],
                                      allpaca_stream_t stream) {
    if (!q || !k || !pos || !sections || n_q_heads <= 0 || n_k_heads <= 0 || head_dim <= 0 || n_rot <= 0 ||
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
                       n_k_heads, head_dim, n_rot, pos[0], pos[1], pos[2], theta, sections[0],
                       sections[1], sections[2], sections[3]);
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

extern "C" int allpaca_silu_mul_limit(float *gate, const float *up, int32_t n, float limit,
                                      allpaca_stream_t stream) {
    if (!gate || !up || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(silu_mul_limit_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), gate, up, n,
                       limit);
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

extern "C" int allpaca_gather_rows(float *dst, const float *src, const int32_t *rows, int32_t n,
                                   int32_t row_len, allpaca_stream_t stream) {
    if (!dst || !src || !rows || n <= 0 || row_len <= 0 ||
        static_cast<int64_t>(n) * row_len > INT32_MAX) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (n * row_len + threads - 1) / threads;
    hipLaunchKernelGGL(gather_rows_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), dst, src, rows, n,
                       row_len);
    return hip_status(hipGetLastError());
}

/* QSA (qwen4exp sparse attention) indexer on the card.
 * qsa_norm_rope: RMS norm with gain, then IMRoPE at `pos`, of one dim-wide
 * vector held in LDS by a block of 256 (same reduction tree as
 * rmsnorm_heads_kernel). */
__device__ void qsa_norm_rope(float *v, const float *gain, int dim, int n_rot, int pos, float theta,
                              int s0, int s1, int s2, int s3, float eps, float *red) {
    const int tid = static_cast<int>(threadIdx.x);
    float sum = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        sum = fmaf(v[i], v[i], sum);
    }
    red[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        __syncthreads();
    }
    const float inv = det_rsqrtf(__fadd_rn(__fdiv_rn(red[0], static_cast<float>(dim)), eps));
    for (int i = tid; i < dim; i += 256) {
        v[i] = __fmul_rn(__fmul_rn(v[i], inv), gain[i]);
    }
    __syncthreads();
    const int half = n_rot / 2;
    for (int i = tid; i < half; i += 256) {
        rope_imrope_pair(v, i, half, n_rot, pos, pos, pos, theta, s0, s1, s2, s3);
    }
    __syncthreads();
}

#define QSA_MAX_DIM 512

/* Pool p of `ratio` raw keys: their mean, normed and rotated to p * ratio. */
__global__ __launch_bounds__(256) void qsa_pool_kernel(const float *raw, const float *gain, float *pooled,
                                                       int first, int ratio, int dim, int n_rot, float theta,
                                                       int s0, int s1, int s2, int s3, float eps) {
    __shared__ float v[QSA_MAX_DIM];
    __shared__ float red[256];
    const int pool = first + static_cast<int>(blockIdx.x);
    const float *members = raw + static_cast<size_t>(pool) * ratio * dim;
    const float inv = 1.0f / static_cast<float>(ratio);
    for (int i = static_cast<int>(threadIdx.x); i < dim; i += 256) {
        float s = 0.0f;
        for (int m = 0; m < ratio; ++m) {
            s += members[static_cast<size_t>(m) * dim + i];
        }
        v[i] = s * inv;
    }
    __syncthreads();
    qsa_norm_rope(v, gain, dim, n_rot, pool * ratio, theta, s0, s1, s2, s3, eps, red);
    for (int i = static_cast<int>(threadIdx.x); i < dim; i += 256) {
        pooled[static_cast<size_t>(pool) * dim + i] = v[i];
    }
}

/* Indexer query head blockIdx.x (of row blockIdx.x / heads), in place. */
__global__ __launch_bounds__(256) void qsa_query_kernel(float *q, const float *gain, int heads, int dim, int n_rot,
                                                        int pos, float theta, int s0, int s1, int s2, int s3,
                                                        float eps) {
    __shared__ float v[QSA_MAX_DIM];
    __shared__ float red[256];
    float *head = q + static_cast<size_t>(blockIdx.x) * dim;
    for (int i = static_cast<int>(threadIdx.x); i < dim; i += 256) {
        v[i] = head[i];
    }
    __syncthreads();
    qsa_norm_rope(v, gain, dim, n_rot, pos + static_cast<int>(blockIdx.x) / heads, theta, s0, s1, s2, s3, eps, red);
    for (int i = static_cast<int>(threadIdx.x); i < dim; i += 256) {
        head[i] = v[i];
    }
}

#define QSA_SCORE_POOLS 64
#define QSA_SCORE_THREADS 256

/* LDS of a qsa_score_kernel block: the row's queries, its pools' keys in
 * rows padded by one float (so a thread per pool reads its own bank), and
 * the per-head sums. */
static size_t qsa_score_lds_bytes(int heads, int dim) {
    return (static_cast<size_t>(heads) * dim + static_cast<size_t>(QSA_SCORE_POOLS) * (dim + 1) +
            static_cast<size_t>(QSA_SCORE_POOLS) * heads) *
           sizeof(float);
}

/* A pool's score for row blockIdx.y: its key against every query head,
 * negative products dropped, the rest summed, each sum in sequence order
 * and without fused multiply-adds (like the host's). The block's keys are
 * staged in LDS first (coalesced reads instead of each thread walking its
 * own key through global memory); a thread per (pool, head) product, then
 * a thread per pool adds the heads in order. */
__global__ __launch_bounds__(QSA_SCORE_THREADS) void qsa_score_kernel(const float *q, const float *pooled,
                                                                      float *scores, int pos, int heads, int dim,
                                                                      int ratio, int top_pools, int score_stride,
                                                                      float scale) {
    extern __shared__ float qs[];
    const int r = static_cast<int>(blockIdx.y);
    const int visible = (pos + r + 1) / ratio;
    const int first = static_cast<int>(blockIdx.x) * QSA_SCORE_POOLS;
    if (visible <= top_pools || first >= visible) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    float *ks = qs + heads * dim;
    float *hs = ks + QSA_SCORE_POOLS * (dim + 1);
    const float *qr = q + static_cast<size_t>(r) * heads * dim;
    for (int i = tid; i < heads * dim; i += QSA_SCORE_THREADS) {
        qs[i] = qr[i];
    }
    const int n = min(QSA_SCORE_POOLS, visible - first);
    const float *src = pooled + static_cast<size_t>(first) * dim;
    for (int i = tid; i < n * dim; i += QSA_SCORE_THREADS) {
        ks[(i / dim) * (dim + 1) + i % dim] = src[i];
    }
    __syncthreads();
    {
#pragma clang fp contract(off)
        for (int j = tid; j < QSA_SCORE_POOLS * heads; j += QSA_SCORE_THREADS) {
            const int pool = j % QSA_SCORE_POOLS, h = j / QSA_SCORE_POOLS;
            if (pool < n) {
                const float *qh = qs + h * dim;
                const float *key = ks + pool * (dim + 1);
                float s = 0.0f;
                for (int i = 0; i < dim; ++i) {
                    s = s + qh[i] * key[i];
                }
                hs[h * QSA_SCORE_POOLS + pool] = s;
            }
        }
        __syncthreads();
        if (tid < n) {
            float total = 0.0f;
            for (int h = 0; h < heads; ++h) {
                total = total + fmaxf(hs[h * QSA_SCORE_POOLS + tid], 0.0f) * scale;
            }
            scores[static_cast<size_t>(r) * score_stride + first + tid] = total;
        }
    }
}

#define QSA_SELECT_THREADS 256
static_assert(QSA_SELECT_THREADS == 256, "a thread per histogram bin");

/* Row blockIdx.x's cells: the `top_pools` best pools (of equal scores the
 * earlier pool wins) in ascending order, `ratio` cells each, then the
 * incomplete tail. A radix select on the scores' bits (never negative, so
 * they order as unsigned) finds the last score kept, then a scan in pool
 * order places the kept pools. */
__global__ __launch_bounds__(QSA_SELECT_THREADS) void qsa_select_kernel(const float *scores, int32_t *cells, int pos,
                                                                        int ratio, int top_pools, int stride,
                                                                        int score_stride) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t scan[QSA_SELECT_THREADS];
    __shared__ uint32_t sel[2];
    const int r = static_cast<int>(blockIdx.x);
    const int rank = pos + r;
    const int visible = (rank + 1) / ratio;
    if (visible <= top_pools) {
        return;
    }
    const int tid = static_cast<int>(threadIdx.x);
    const uint32_t *keys = reinterpret_cast<const uint32_t *>(scores + static_cast<size_t>(r) * score_stride);
    auto key_of = [&](int p) {
        const uint32_t k = keys[p];
        return (k & 0x80000000u) ? 0u : k;
    };
    uint32_t prefix = 0, mask = 0, remaining = static_cast<uint32_t>(top_pools);
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[tid] = 0;
        __syncthreads();
        for (int p = tid; p < visible; p += QSA_SELECT_THREADS) {
            const uint32_t k = key_of(p);
            if ((k & mask) == prefix) {
                atomicAdd(&hist[(k >> shift) & 255u], 1u);
            }
        }
        __syncthreads();
        // The bin the `remaining`-th best score falls in: thread t holds the
        // count of bins 255 - t and up (an inclusive scan from the top).
        scan[tid] = hist[255 - tid];
        __syncthreads();
        for (int off = 1; off < 256; off <<= 1) {
            const uint32_t add = tid >= off ? scan[tid - off] : 0u;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        const uint32_t above = tid > 0 ? scan[tid - 1] : 0u;
        if (above < remaining && scan[tid] >= remaining) {
            sel[0] = static_cast<uint32_t>(255 - tid);
            sel[1] = remaining - above;
        }
        __syncthreads();
        prefix |= sel[0] << shift;
        mask |= 255u << shift;
        remaining = sel[1];
        __syncthreads();
    }
    // `prefix` is the score of the last pool kept; of that score the
    // earliest `remaining` pools are kept.
    const uint32_t threshold = prefix, ties = remaining;
    uint32_t kept = 0, tied = 0;
    for (int base = 0; base < visible; base += QSA_SELECT_THREADS) {
        const int p = base + tid;
        const uint32_t k = p < visible ? key_of(p) : 0u;
        const uint32_t is_tie = (p < visible && k == threshold) ? 1u : 0u;
        const uint32_t above = (p < visible && k > threshold) ? 1u : 0u;
        // Inclusive scan of ties (low half) and pools above (high half).
        scan[tid] = is_tie | (above << 16);
        __syncthreads();
        for (int off = 1; off < QSA_SELECT_THREADS; off <<= 1) {
            const uint32_t add = tid >= off ? scan[tid - off] : 0u;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        const uint32_t ties_before = tied + (scan[tid] & 0xffffu) - is_tie;
        const bool keep = above || (is_tie && ties_before < ties);
        if (keep) {
            const uint32_t at = kept + (scan[tid] >> 16) - above + min(ties_before, ties) - min(tied, ties);
            int32_t *out = cells + static_cast<size_t>(r) * stride + static_cast<size_t>(at) * ratio;
            for (int i = 0; i < ratio; ++i) {
                out[i] = p * ratio + i;
            }
        }
        const uint32_t chunk = scan[QSA_SELECT_THREADS - 1];
        const uint32_t chunk_ties = chunk & 0xffffu;
        kept += (chunk >> 16) + min(tied + chunk_ties, ties) - min(tied, ties);
        tied += chunk_ties;
        __syncthreads();
    }
    for (int c = visible * ratio + tid; c <= rank; c += QSA_SELECT_THREADS) {
        cells[static_cast<size_t>(r) * stride + top_pools * ratio + (c - visible * ratio)] = c;
    }
}

extern "C" int allpaca_qsa_pool(const float *raw, const float *gain, float *pooled, int32_t first, int32_t n,
                                int32_t ratio, int32_t dim, int32_t n_rot, float theta, const int32_t sections[4],
                                float eps, allpaca_stream_t stream) {
    if (!raw || !gain || !pooled || !sections || first < 0 || n <= 0 || ratio <= 0 || dim <= 0 ||
        dim > QSA_MAX_DIM || n_rot <= 0 || n_rot > dim || (n_rot & 1) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(qsa_pool_kernel, dim3(static_cast<unsigned>(n)), dim3(256), 0, as_stream(stream), raw, gain,
                       pooled, first, ratio, dim, n_rot, theta, sections[0], sections[1], sections[2], sections[3],
                       eps);
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_qsa_select(float *q, const float *gain, const float *pooled, float *scores, int32_t *cells,
                                  int32_t rows, int32_t pos, int32_t heads, int32_t dim, int32_t n_rot, float theta,
                                  const int32_t sections[4], float eps, int32_t ratio, int32_t top_pools,
                                  int32_t stride, int32_t score_stride, allpaca_stream_t stream) {
    const int64_t last = static_cast<int64_t>(pos) + rows;
    if (!q || !gain || !pooled || !scores || !cells || !sections || rows <= 0 || pos < 0 || heads <= 0 ||
        dim <= 0 || dim > QSA_MAX_DIM || n_rot <= 0 || n_rot > dim || (n_rot & 1) != 0 || ratio <= 0 ||
        top_pools <= 0 || top_pools >= 65536 || last > INT32_MAX || stride < top_pools * ratio + ratio - 1 ||
        score_stride < last / ratio || qsa_score_lds_bytes(heads, dim) > 65536) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const auto st = as_stream(stream);
    hipLaunchKernelGGL(qsa_query_kernel, dim3(static_cast<unsigned>(rows * heads)), dim3(256), 0, st, q, gain, heads,
                       dim, n_rot, pos, theta, sections[0], sections[1], sections[2], sections[3], eps);
    const int most = static_cast<int>(last / ratio);
    const float scale = 1.0f / sqrtf(static_cast<float>(dim));
    hipLaunchKernelGGL(qsa_score_kernel,
                       dim3(static_cast<unsigned>((most + QSA_SCORE_POOLS - 1) / QSA_SCORE_POOLS),
                            static_cast<unsigned>(rows)),
                       dim3(QSA_SCORE_THREADS), qsa_score_lds_bytes(heads, dim), st, q, pooled, scores, pos, heads, dim,
                       ratio, top_pools, score_stride, scale);
    hipLaunchKernelGGL(qsa_select_kernel, dim3(static_cast<unsigned>(rows)), dim3(QSA_SELECT_THREADS), 0, st, scores,
                       cells, pos, ratio, top_pools, stride, score_stride);
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

/* GDN output: out = silu(z) * rmsnorm_head(y) * weight in one launch, one
 * block of 256 per head. Same reduction tree as rmsnorm_heads_kernel and the
 * same element math as silu_mul_kernel, so it gives the bits the copy +
 * rmsnorm_heads + silu_mul + copy sequence gave. `out` may alias `z` or `y`. */
__global__ void gdn_norm_gate_kernel(const float *y, const float *z, const float *weight,
                                     float *out, int head_dim, float eps) {
    const size_t base = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(head_dim);
    const float *row = y + base;
    __shared__ float shared[256];
    float sum = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < head_dim; i += static_cast<int>(blockDim.x)) {
        const float v = row[i];
        sum = fmaf(v, v, sum);
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv = det_rsqrtf(__fadd_rn(__fdiv_rn(shared[0], static_cast<float>(head_dim)), eps));
    for (int i = static_cast<int>(threadIdx.x); i < head_dim; i += static_cast<int>(blockDim.x)) {
        const float normed = __fmul_rn(__fmul_rn(row[i], inv), weight[i]);
        out[base + i] = det_silu_mul(z[base + i], normed);
    }
}

extern "C" int allpaca_gdn_norm_gate(const float *y, const float *z, const float *weight,
                                     float *out, int32_t n_heads, int32_t head_dim, float eps,
                                     allpaca_stream_t stream) {
    if (!y || !z || !weight || !out || n_heads <= 0 || head_dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(gdn_norm_gate_kernel, dim3(static_cast<unsigned>(n_heads)), dim3(256), 0,
                       as_stream(stream), y, z, weight, out, head_dim, eps);
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

__global__ void round_f16_kernel(float *x, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        x[idx] = static_cast<float>(static_cast<_Float16>(x[idx]));
    }
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

/* Two Q8_0 GEMVs over one input row (m == 1) in one launch: c1 = a * b1^T
 * ([n1, k]) and c2 = a * b2^T ([n2, k]), each bit-identical to its own GEMV
 * launch. ALLPACA_ERR_INVALID where the single GEMV would take another kernel. */
extern "C" int allpaca_gemm_nt_q8_0_pair(const float *a, const uint8_t *b1, float *c1,
                                         const uint8_t *b2, float *c2, int32_t m, int32_t n1,
                                         int32_t n2, int32_t k, allpaca_stream_t stream) {
    if (!a || !b1 || !c1 || !b2 || !c2 || m != 1 || n1 <= 0 || n2 <= 0 || k <= 0 ||
        ALLPACA_Q8_GEMV_INTDOT || q8_0_uses_i8(n1, k) || q8_0_uses_i8(n2, k)) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    constexpr int cols_per_block = 16 * 8;
    const int n = n1 > n2 ? n1 : n2;
    const unsigned gx = static_cast<unsigned>((n + cols_per_block - 1) / cols_per_block);
    hipLaunchKernelGGL(gemv_nt_q8_0_f32_kernel, dim3(gx, 1, 2), dim3(512), 0, as_stream(stream), a,
                       b1, c1, m, n1, k, b2, c2, n2);
    return hip_status(hipGetLastError());
}

/* x[i] += y[i] * s, then y[i] = x[i]: scale_f32(y, s) + residual_add(x, y) +
 * copy(y <- x) in one launch. The three kernels rounded the product and the
 * sum separately; contraction is off here because __fmul_rn/__fadd_rn still
 * compiled to an FMA (1-ulp differences); see moe_combine_kernel. */
__global__ void scale_add_copy_kernel(float *x, float *y, float s, int n) {
#pragma clang fp contract(off)
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const float v = x[i] + y[i] * s;
    x[i] = v;
    y[i] = v;
}

extern "C" int allpaca_scale_add_copy(float *x, float *y, float s, int32_t n,
                                      allpaca_stream_t stream) {
    if (!x || !y || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(scale_add_copy_kernel, dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256),
                       0, as_stream(stream), x, y, s, n);
    return hip_status(hipGetLastError());
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

template <int BB>
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
                    static_cast<size_t>(BB);
    if (j < 16) {
        out[i] = static_cast<float>(k_rocmfp4_codebook[blk[j] & 0x0F]) * rocmfp4_scale(blk[16]);
    } else {
        out[i] =
            static_cast<float>(k_rocmfp4_codebook[blk[j - 16] >> 4]) * rocmfp4_scale(blk[BB - 1]);
    }
}

template <int BB>
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
                    static_cast<size_t>(BB);
    float v;
    if (j < 16) {
        v = static_cast<float>(k_rocmfp4_codebook[blk[j] & 0x0F]) * rocmfp4_scale(blk[16]);
    } else {
        v = static_cast<float>(k_rocmfp4_codebook[blk[j - 16] >> 4]) * rocmfp4_scale(blk[BB - 1]);
    }
    out[static_cast<size_t>(row) * static_cast<size_t>(dim) + i] = v;
}

static int launch_embed_gather_rocmfp4(const uint8_t *table, int32_t token, float *out,
                                       int32_t dim, hipStream_t stream, bool fast) {
    if (!table || !out || dim <= 0 || token < 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const dim3 grid(static_cast<unsigned>((dim + threads - 1) / threads));
    const dim3 block(static_cast<unsigned>(threads));
    if (fast) {
        hipLaunchKernelGGL(embed_gather_rocmfp4_kernel<17>, grid, block, 0, stream, table, token, out,
                           dim);
    } else {
        hipLaunchKernelGGL(embed_gather_rocmfp4_kernel<18>, grid, block, 0, stream, table, token, out,
                           dim);
    }
    return hip_status(hipGetLastError());
}

static int launch_embed_gather_rocmfp4_rows(const uint8_t *table, const int32_t *tokens, float *out,
                                            int32_t n_tok, int32_t dim, hipStream_t stream,
                                            bool fast) {
    if (!table || !tokens || !out || n_tok <= 0 || dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const dim3 grid(static_cast<unsigned>((dim + threads - 1) / threads),
                    static_cast<unsigned>(n_tok));
    const dim3 block(static_cast<unsigned>(threads));
    if (fast) {
        hipLaunchKernelGGL(embed_gather_rocmfp4_rows_kernel<17>, grid, block, 0, stream, table,
                           tokens, out, n_tok, dim);
    } else {
        hipLaunchKernelGGL(embed_gather_rocmfp4_rows_kernel<18>, grid, block, 0, stream, table,
                           tokens, out, n_tok, dim);
    }
    return hip_status(hipGetLastError());
}

extern "C" int allpaca_embed_gather_rocmfp4(const uint8_t *table, int32_t token, float *out,
                                            int32_t dim, allpaca_stream_t stream) {
    return launch_embed_gather_rocmfp4(table, token, out, dim, as_stream(stream), false);
}

extern "C" int allpaca_embed_gather_rocmfp4_rows(const uint8_t *table, const int32_t *tokens,
                                                 float *out, int32_t n_tok, int32_t dim,
                                                 allpaca_stream_t stream) {
    return launch_embed_gather_rocmfp4_rows(table, tokens, out, n_tok, dim, as_stream(stream),
                                            false);
}

/* ROCmFP4 FAST (ggml_type_id=101) embed rows: 17-byte blocks. */
extern "C" int allpaca_embed_gather_rocmfp4_fast(const uint8_t *table, int32_t token, float *out,
                                                 int32_t dim, allpaca_stream_t stream) {
    return launch_embed_gather_rocmfp4(table, token, out, dim, as_stream(stream), true);
}

extern "C" int allpaca_embed_gather_rocmfp4_fast_rows(const uint8_t *table, const int32_t *tokens,
                                                      float *out, int32_t n_tok, int32_t dim,
                                                      allpaca_stream_t stream) {
    return launch_embed_gather_rocmfp4_rows(table, tokens, out, n_tok, dim, as_stream(stream),
                                            true);
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

extern "C" int allpaca_round_f16(float *x, int32_t n, allpaca_stream_t stream) {
    if (!x || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(round_f16_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), x, n);
    return hip_status(hipGetLastError());
}

/* x[r][c] = act(x[r][c] + bias[c]); act 1 = tanh GELU (ggml_gelu). */
__global__ void bias_act_rows_kernel(float *x, const float *bias, int64_t total, int n, int act) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    float v = x[idx] + bias[idx % n];
    if (act == 1) {
        const float k0 = 0.7978845608028654f;
        v = 0.5f * v * (1.0f + tanhf(k0 * (v + 0.044715f * v * v * v)));
    }
    x[idx] = v;
}

extern "C" int allpaca_bias_act_rows(float *x, const float *bias, int32_t rows, int32_t n,
                                     int32_t act, allpaca_stream_t stream) {
    if (!x || !bias || rows <= 0 || n <= 0 || act < 0 || act > 1) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int64_t total = static_cast<int64_t>(rows) * n;
    const int threads = 256;
    const unsigned blocks = static_cast<unsigned>((total + threads - 1) / threads);
    hipLaunchKernelGGL(bias_act_rows_kernel, dim3(blocks), dim3(threads), 0, as_stream(stream), x,
                       bias, total, n, act);
    return hip_status(hipGetLastError());
}

/* ggml GGML_ROPE_TYPE_VISION, sections [d/4; 4]: pairs (i, i + d/2); the first
 * quarter of pairs turns with pos_row[t] and the second with pos_col[t], each
 * quarter restarting the frequency ladder. x is [n_tok][n_heads * head_dim]. */
__global__ void vision_rope_rows_kernel(float *x, const int32_t *pos_row, const int32_t *pos_col,
                                        int n_tok, int n_heads, int head_dim, float theta) {
    const int half = head_dim / 2;
    const int quarter = half / 2;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t per_tok = static_cast<int64_t>(n_heads) * half;
    if (idx >= per_tok * n_tok) {
        return;
    }
    const int t = static_cast<int>(idx / per_tok);
    const int rem = static_cast<int>(idx % per_tok);
    const int h = rem / half;
    const int i = rem % half;
    const bool row_part = i < quarter;
    const float pos = static_cast<float>(row_part ? pos_row[t] : pos_col[t]);
    const int k = row_part ? i : i - quarter;
    const float angle = pos * powf(theta, -2.0f * static_cast<float>(k) / static_cast<float>(half));
    const float s = sinf(angle);
    const float c = cosf(angle);
    float *hv = x + (static_cast<int64_t>(t) * n_heads + h) * head_dim;
    const float x0 = hv[i];
    const float x1 = hv[i + half];
    hv[i] = x0 * c - x1 * s;
    hv[i + half] = x0 * s + x1 * c;
}

extern "C" int allpaca_vision_rope_rows(float *x, const int32_t *pos_row, const int32_t *pos_col,
                                        int32_t n_tok, int32_t n_heads, int32_t head_dim,
                                        float theta, allpaca_stream_t stream) {
    if (!x || !pos_row || !pos_col || n_tok <= 0 || n_heads <= 0 || head_dim <= 0 ||
        head_dim % 4 != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int64_t total = static_cast<int64_t>(n_tok) * n_heads * (head_dim / 2);
    const int threads = 256;
    const unsigned blocks = static_cast<unsigned>((total + threads - 1) / threads);
    hipLaunchKernelGGL(vision_rope_rows_kernel, dim3(blocks), dim3(threads), 0, as_stream(stream), x,
                       pos_row, pos_col, n_tok, n_heads, head_dim, theta);
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

/* Busy-wait (not interrupt-driven) host synchronization for `device`. Must
 * run before the device's context exists. Windows wakes a blocked host
 * thread tens of us after the GPU finishes; decode syncs once per layer. */
/* Block reductions over any block size and wave width. `scratch` holds at
 * least blockDim / warpSize values; every thread gets the result. */
__device__ inline float moe_block_sum(float v, float *scratch) {
    const int ws = warpSize;
    for (int off = ws / 2; off > 0; off >>= 1) {
        v += __shfl_down(v, off, ws);
    }
    const int tid = static_cast<int>(threadIdx.x);
    __syncthreads();
    if (tid % ws == 0) {
        scratch[tid / ws] = v;
    }
    __syncthreads();
    float total = 0.f;
    const int nw = (static_cast<int>(blockDim.x) + ws - 1) / ws;
    for (int w = 0; w < nw; ++w) {
        total += scratch[w];
    }
    return total;
}

__device__ inline float moe_block_max(float v, float *scratch) {
    const int ws = warpSize;
    for (int off = ws / 2; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_down(v, off, ws));
    }
    const int tid = static_cast<int>(threadIdx.x);
    __syncthreads();
    if (tid % ws == 0) {
        scratch[tid / ws] = v;
    }
    __syncthreads();
    float m = -INFINITY;
    const int nw = (static_cast<int>(blockDim.x) + ws - 1) / ws;
    for (int w = 0; w < nw; ++w) {
        m = fmaxf(m, scratch[w]);
    }
    return m;
}

/* Softmax router on the GPU, one block per token row: softmax over n_exp
 * logits, top-k (ties to the lower index), renormalize over the k, as the
 * host `topk_from_logits`. Writes the row-sorted pair table the fused MoE
 * kernels read, and optionally the shared-expert gate sigmoid(x . g) per
 * row, so a layer with local experts needs no host round trip.
 * `disabled` (optional, n_exp bytes): those experts take no part in the
 * softmax and are never picked. `counts` (optional, n_exp): +1 per pick. */
__global__ void moe_route_softmax_topk_kernel(const float *logits, int n_exp, int k,
                                              const float *x, const float *shexp_gate, int dim,
                                              float *w_out, int32_t *row_out, int32_t *exp_out,
                                              int32_t *row_start, float *shexp_scale, int rows,
                                              const uint8_t *disabled, uint32_t *counts) {
    extern __shared__ float route_lds[];
    float *prob = route_lds;
    float *scratch = route_lds + n_exp;
    __shared__ float best_v[64];
    __shared__ int best_i[64];
    __shared__ float sel_w[64];
    __shared__ int sel_e[64];
    const int r = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int bd = static_cast<int>(blockDim.x);
    const float *lg = logits + static_cast<size_t>(r) * n_exp;
    float m = -INFINITY;
    for (int i = tid; i < n_exp; i += bd) {
        if (!(disabled && disabled[i])) {
            m = fmaxf(m, lg[i]);
        }
    }
    m = moe_block_max(m, scratch);
    float sum = 0.f;
    for (int i = tid; i < n_exp; i += bd) {
        const float e = (disabled && disabled[i]) ? 0.f : expf(lg[i] - m);
        prob[i] = e;
        sum += e;
    }
    sum = fmaxf(moe_block_sum(sum, scratch), 1e-9f);
    for (int i = tid; i < n_exp; i += bd) {
        // Probabilities are >= 0, so a disabled expert at -1 is never picked.
        prob[i] = (disabled && disabled[i]) ? -1.f : prob[i] / sum;
    }
    __syncthreads();
    const int ws = warpSize;
    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY;
        int bi = n_exp;
        for (int i = tid; i < n_exp; i += bd) {
            const float v = prob[i];
            if (v > bv || (v == bv && i < bi)) {
                bv = v;
                bi = i;
            }
        }
        for (int off = ws / 2; off > 0; off >>= 1) {
            const float ov = __shfl_down(bv, off, ws);
            const int oi = __shfl_down(bi, off, ws);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (tid % ws == 0) {
            best_v[tid / ws] = bv;
            best_i[tid / ws] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            float v = best_v[0];
            int idx = best_i[0];
            for (int w = 1; w < (bd + ws - 1) / ws; ++w) {
                if (best_v[w] > v || (best_v[w] == v && best_i[w] < idx)) {
                    v = best_v[w];
                    idx = best_i[w];
                }
            }
            sel_w[j] = v;
            sel_e[j] = idx;
            prob[idx] = -INFINITY;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float part = 0.f;
        for (int j = 0; j < k; ++j) {
            part += sel_w[j];
        }
        part = fmaxf(part, 1e-9f);
        for (int j = 0; j < k; ++j) {
            const size_t p = static_cast<size_t>(r) * k + j;
            w_out[p] = sel_w[j] / part;
            row_out[p] = r;
            exp_out[p] = sel_e[j];
            if (counts) {
                atomicAdd(&counts[sel_e[j]], 1u);
            }
        }
        row_start[r] = r * k;
        if (r == rows - 1) {
            row_start[rows] = rows * k;
        }
    }
    if (shexp_gate) {
        const float *xr = x + static_cast<size_t>(r) * dim;
        float dot = 0.f;
        for (int i = tid; i < dim; i += bd) {
            dot += xr[i] * shexp_gate[i];
        }
        dot = moe_block_sum(dot, scratch);
        if (tid == 0) {
            const float z = expf(dot >= 0.f ? -dot : dot);
            shexp_scale[r] = dot >= 0.f ? 1.f / (1.f + z) : z / (1.f + z);
        }
    }
}

extern "C" int allpaca_moe_route_softmax_topk(const float *logits, int32_t rows, int32_t n_exp,
                                              int32_t k, const float *x, const float *shexp_gate,
                                              int32_t dim, float *w_out, int32_t *row_out,
                                              int32_t *exp_out, int32_t *row_start,
                                              float *shexp_scale, const uint8_t *disabled,
                                              uint32_t *counts, allpaca_stream_t stream) {
    if (!logits || rows <= 0 || n_exp <= 0 || k <= 0 || k > 64 || k > n_exp || !w_out ||
        !row_out || !exp_out || !row_start || (shexp_gate && (!x || !shexp_scale || dim <= 0))) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const size_t lds = (static_cast<size_t>(n_exp) + 64) * sizeof(float);
    hipLaunchKernelGGL(moe_route_softmax_topk_kernel, dim3(static_cast<unsigned>(rows)), dim3(256),
                       lds, as_stream(stream), logits, n_exp, k, x, shexp_gate, dim, w_out, row_out,
                       exp_out, row_start, shexp_scale, rows, disabled, counts);
    return hip_status(hipGetLastError());
}

__global__ void scale_rows_kernel(float *x, const float *scale, int rows, int dim) {
    const int d = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = static_cast<int>(blockIdx.y);
    if (d < dim && r < rows) {
        x[static_cast<size_t>(r) * dim + d] *= scale[r];
    }
}

/* x[r][:] *= scale[r], scales in device memory. */
extern "C" int allpaca_scale_rows(float *x, const float *scale, int32_t rows, int32_t dim,
                                  allpaca_stream_t stream) {
    if (!x || !scale || rows <= 0 || dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(scale_rows_kernel,
                       dim3(static_cast<unsigned>((dim + 255) / 256), static_cast<unsigned>(rows)),
                       dim3(256), 0, as_stream(stream), x, scale, rows, dim);
    return hip_status(hipGetLastError());
}

/* Chunked top-k: block (c, r) keeps the k largest of x[r][c*chunk .. (c+1)*chunk),
 * largest first, ties to the lower index. out_v/out_i are [rows][n_chunks][k]; a
 * chunk shorter than k pads with -inf / -1. */
constexpr int TOPK_CHUNK_MAX = 2048;
constexpr int TOPK_THREADS = 256;

__global__ void topk_chunks_kernel(const float *x, int n, int chunk, int k, float *out_v,
                                   int32_t *out_i) {
    __shared__ float vals[TOPK_CHUNK_MAX];
    __shared__ float red_v[TOPK_THREADS];
    __shared__ int red_i[TOPK_THREADS];
    const int c = static_cast<int>(blockIdx.x);
    const int r = static_cast<int>(blockIdx.y);
    const int n_chunks = static_cast<int>(gridDim.x);
    const int base = c * chunk;
    const int len = min(chunk, n - base);
    const float *row = x + static_cast<size_t>(r) * n;
    for (int i = static_cast<int>(threadIdx.x); i < chunk; i += TOPK_THREADS) {
        vals[i] = i < len ? row[base + i] : -INFINITY;
    }
    __syncthreads();
    const size_t out0 = (static_cast<size_t>(r) * n_chunks + c) * k;
    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int i = static_cast<int>(threadIdx.x); i < len; i += TOPK_THREADS) {
            const float v = vals[i];
            if (v > bv || (v == bv && i < bi)) {
                bv = v;
                bi = i;
            }
        }
        red_v[threadIdx.x] = bv;
        red_i[threadIdx.x] = bi;
        __syncthreads();
        for (int sstep = TOPK_THREADS / 2; sstep > 0; sstep >>= 1) {
            if (static_cast<int>(threadIdx.x) < sstep) {
                const float ov = red_v[threadIdx.x + sstep];
                const int oi = red_i[threadIdx.x + sstep];
                if (ov > red_v[threadIdx.x] || (ov == red_v[threadIdx.x] && oi < red_i[threadIdx.x])) {
                    red_v[threadIdx.x] = ov;
                    red_i[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const int win = red_i[0];
            const bool ok = win < len;
            out_v[out0 + j] = ok ? red_v[0] : -INFINITY;
            out_i[out0 + j] = ok ? base + win : -1;
            if (ok) {
                vals[win] = -INFINITY;
            }
        }
        __syncthreads();
    }
}

extern "C" int allpaca_topk_chunks(const float *x, int32_t rows, int32_t n, int32_t chunk,
                                   int32_t k, float *out_v, int32_t *out_i,
                                   allpaca_stream_t stream) {
    if (!x || !out_v || !out_i || rows <= 0 || n <= 0 || chunk <= 0 || chunk > TOPK_CHUNK_MAX ||
        k <= 0 || k > chunk) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int n_chunks = (n + chunk - 1) / chunk;
    hipLaunchKernelGGL(topk_chunks_kernel,
                       dim3(static_cast<unsigned>(n_chunks), static_cast<unsigned>(rows)),
                       dim3(TOPK_THREADS), 0, as_stream(stream), x, n, chunk, k, out_v, out_i);
    return hip_status(hipGetLastError());
}

/* NeoX RoPE over rows: x[n_tok][n_heads][head_dim], row t at position start_pos + t,
 * pairs (i, i + head_dim/2), freq theta^(-2i/head_dim). */
__global__ void rope_neox_rows_kernel(float *x, int n_tok, int n_heads, int head_dim, int start_pos,
                                      float theta) {
    const int tok = static_cast<int>(blockIdx.y);
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int half = head_dim / 2;
    if (tok >= n_tok || idx >= n_heads * half) {
        return;
    }
    const int h = idx / half;
    const int i = idx % half;
    float *hvec = x + (static_cast<size_t>(tok) * n_heads + h) * static_cast<size_t>(head_dim);
    const float freq = det_powf(theta, -static_cast<float>(i) / static_cast<float>(half));
    det_rotate(hvec[i], hvec[i + half], __fmul_rn(static_cast<float>(start_pos + tok), freq), 1);
}

extern "C" int allpaca_rope_neox_rows(float *x, int32_t n_tok, int32_t n_heads, int32_t head_dim,
                                      int32_t start_pos, float theta, allpaca_stream_t stream) {
    if (!x || n_tok <= 0 || n_heads <= 0 || head_dim <= 0 || (head_dim & 1) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int total = n_heads * (head_dim / 2);
    hipLaunchKernelGGL(rope_neox_rows_kernel,
                       dim3(static_cast<unsigned>((total + 255) / 256), static_cast<unsigned>(n_tok)),
                       dim3(256), 0, as_stream(stream), x, n_tok, n_heads, head_dim, start_pos,
                       theta);
    return hip_status(hipGetLastError());
}

/* NeoX RoPE as rope_neox_rows_kernel with llama.cpp's frequency factors:
 * pair i turns by pos * theta^(-2i/head_dim) / ff[i] (gemma4's proportional
 * rope; ff null = all ones). */
__global__ void rope_neox_ff_rows_kernel(float *x, int n_tok, int n_heads, int head_dim, int start_pos,
                                         float theta, const float *ff) {
    const int tok = static_cast<int>(blockIdx.y);
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int half = head_dim / 2;
    if (tok >= n_tok || idx >= n_heads * half) {
        return;
    }
    const int h = idx / half;
    const int i = idx % half;
    float *hvec = x + (static_cast<size_t>(tok) * n_heads + h) * static_cast<size_t>(head_dim);
    float freq = det_powf(theta, -static_cast<float>(i) / static_cast<float>(half));
    if (ff) {
        freq = __fdiv_rn(freq, ff[i]);
    }
    det_rotate(hvec[i], hvec[i + half], __fmul_rn(static_cast<float>(start_pos + tok), freq), 1);
}

extern "C" int allpaca_rope_neox_ff_rows(float *x, int32_t n_tok, int32_t n_heads, int32_t head_dim,
                                         int32_t start_pos, float theta, const float *ff,
                                         allpaca_stream_t stream) {
    if (!x || n_tok <= 0 || n_heads <= 0 || head_dim <= 0 || (head_dim & 1) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int total = n_heads * (head_dim / 2);
    hipLaunchKernelGGL(rope_neox_ff_rows_kernel,
                       dim3(static_cast<unsigned>((total + 255) / 256), static_cast<unsigned>(n_tok)),
                       dim3(256), 0, as_stream(stream), x, n_tok, n_heads, head_dim, start_pos, theta, ff);
    return hip_status(hipGetLastError());
}

/* x = cap * tanh(x / cap) (final logit softcap). */
__global__ void softcap_kernel(float *x, int n, float cap) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        x[idx] = cap * tanhf(x[idx] / cap);
    }
}

extern "C" int allpaca_softcap(float *x, int32_t n, float cap, allpaca_stream_t stream) {
    if (!x || n <= 0 || !(cap > 0.0f)) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(softcap_kernel, dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256), 0,
                       as_stream(stream), x, n, cap);
    return hip_status(hipGetLastError());
}

/* gate = gelu_tanh(gate) * up (ggml_gelu, PyTorch's gelu "tanh"). */
__global__ void gelu_tanh_mul_kernel(float *gate, const float *up, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    const float g = gate[idx];
    const float gelu = 0.5f * g * (1.0f + tanhf(0.7978845608f * (g + 0.044715f * g * g * g)));
    gate[idx] = gelu * up[idx];
}

extern "C" int allpaca_gelu_tanh_mul(float *gate, const float *up, int32_t n, allpaca_stream_t stream) {
    if (!gate || !up || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(gelu_tanh_mul_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), gate, up, n);
    return hip_status(hipGetLastError());
}

/* DFlash2 two-tap dynamic convolution along each block of `block` rows:
 * out[t][c] = sum_tap (dyn[t][g + groups*(tap + kernel*side)] + base[c + dim*(tap + kernel*side)])
 *             * h[t - tap][c], taps that would cross the block start skipped. */
__global__ void dflash_conv_kernel(const float *h, const float *dyn, const float *base, float *out,
                                   int rows, int block, int dim, int kernel, int group, int side) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= rows * dim) {
        return;
    }
    const int t = idx / dim;
    const int c = idx - t * dim;
    const int groups = dim / group;
    const int proj = 2 * kernel * groups;
    const int in_block = t % block;
    float acc = 0.0f;
    for (int tap = 0; tap < kernel && tap <= in_block; ++tap) {
        const int k = tap + kernel * side;
        const float w = dyn[static_cast<size_t>(t) * proj + groups * k + c / group] +
                        base[static_cast<size_t>(dim) * k + c];
        acc += w * h[static_cast<size_t>(t - tap) * dim + c];
    }
    out[idx] = acc;
}

extern "C" int allpaca_dflash_conv(const float *h, const float *dyn, const float *base, float *out,
                                   int32_t rows, int32_t block, int32_t dim, int32_t kernel,
                                   int32_t group, int32_t side, allpaca_stream_t stream) {
    if (!h || !dyn || !base || !out || h == out || rows <= 0 || block <= 0 || dim <= 0 ||
        kernel <= 0 || group <= 0 || dim % group != 0 || side < 0 || side > 1) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    const int n = rows * dim;
    hipLaunchKernelGGL(dflash_conv_kernel, dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256),
                       0, as_stream(stream), h, dyn, base, out, rows, block, dim, kernel, group,
                       side);
    return hip_status(hipGetLastError());
}

__global__ void copy_f32_kernel(float *dst, const float *src, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        dst[i] = src[i];
    }
}

/* Copy by kernel instead of DMA. Either side may be pinned host memory
 * (hipHostMalloc is device-mapped): on Windows a small hipMemcpy costs
 * ~80 us of DMA setup, a launch and sync a fraction of that. */
extern "C" int allpaca_copy_f32_kernel(float *dst, const float *src, int32_t n,
                                       allpaca_stream_t stream) {
    if (!dst || !src || n < 0) {
        return ALLPACA_ERR_INVALID;
    }
    if (n == 0) {
        return ALLPACA_OK;
    }
    clear_last_error();
    hipLaunchKernelGGL(copy_f32_kernel, dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256), 0,
                       as_stream(stream), dst, src, n);
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

/* Quantize `k` floats (whole 32-value blocks) into LDS exactly as
 * `quantize_rows_q8_kernel` does, so the fused MoE path matches the GEMV path
 * bit for bit. Every thread of the block takes part; ends with a barrier. */
__device__ inline void moe_quantize_lds(const float *v, int k, int8_t *aq, float *ad) {
    // One 32-lane group per 32-value block, as `quantize_rows_q8_kernel`: a
    // lane per value, amax by shuffle. (A thread per block serialized ~64
    // global loads per GEMV block.)
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int groups = static_cast<int>(blockDim.x) / 32;
    const int kb = k / 32;
    for (int blk = static_cast<int>(threadIdx.x) / 32; blk < kb; blk += groups) {
        const float x = v[blk * 32 + lane];
        float amax = fabsf(x);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, off, 32));
        }
        const float d = amax == 0.f ? 0.f : amax / 127.f;
        const float inv = d == 0.f ? 0.f : 1.f / d;
        int q = static_cast<int>(roundf(x * inv));
        q = q > 127 ? 127 : (q < -127 ? -127 : q);
        aq[blk * 32 + lane] = static_cast<int8_t>(q);
        if (lane == 0) {
            ad[blk] = d;
        }
    }
    __syncthreads();
}

/* Test hook: det_silu_mul over n values. */
__global__ void det_silu_mul_kernel(const float *g, const float *u, float *out, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        out[i] = det_silu_mul(g[i], u[i]);
    }
}

extern "C" int allpaca_det_silu_mul(const float *g, const float *u, float *out, int32_t n,
                                    allpaca_stream_t stream) {
    if (!g || !u || !out || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipLaunchKernelGGL(det_silu_mul_kernel, dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256),
                       0, as_stream(stream), g, u, out, n);
    return hip_status(hipGetLastError());
}

/* SiLU(gate) * up, then quantize, into LDS (the down GEMV's activation). */
__device__ inline void moe_silu_quantize_lds(const float *g, const float *u, int k, float *tmp,
                                             int8_t *aq, float *ad) {
    for (int i = static_cast<int>(threadIdx.x); i < k; i += static_cast<int>(blockDim.x)) {
        tmp[i] = det_silu_mul(g[i], u[i]);
    }
    __syncthreads();
    moe_quantize_lds(tmp, k, aq, ad);
}

/* Pair groups: group g covers pairs group_start[g] .. group_start[g+1], all
 * routed to expert[g] (at most MR of them). A null group_start means one pair
 * per group. Each block reads its expert's weights once for all the group's
 * rows; per-row arithmetic is the same for any MR. */
__device__ inline void moe_group(const int32_t *group_start, int g, int &p0, int &mr) {
    p0 = group_start ? group_start[g] : g;
    mr = group_start ? group_start[g + 1] - p0 : 1;
}

/* Gate (z = 0) and up (z = 1) for every group in one launch. Block (x, g)
 * gathers the group's token rows, quantized, into LDS and runs the int8
 * K-quant GEMV against expert[g] of the stacked tensor for all of them. */
template <int QG, int QU, int LPC, int MR>
__global__ void moe_gate_up_kernel(const float *x, const int8_t *xq, const float *xd,
                                   const int32_t *pair_row, const int32_t *group_start,
                                   const int32_t *expert, const uint8_t *gate,
                                   unsigned long long gate_stride, const uint8_t *up,
                                   unsigned long long up_stride, float *g, float *u, int ff,
                                   int dim) {
    extern __shared__ unsigned char moe_lds[];
    const int gi = static_cast<int>(blockIdx.y);
    int p0, mr;
    moe_group(group_start, gi, p0, mr);
    const int kb = dim / 32;
    const int8_t *aq;
    const float *ad;
    if (MR == 1 && xq) {
        // One pre-quantized row: read it in place.
        const size_t row = static_cast<size_t>(pair_row[p0]);
        aq = xq + row * dim;
        ad = xd + row * kb;
    } else {
        int8_t *lq = reinterpret_cast<int8_t *>(moe_lds);
        float *ld = reinterpret_cast<float *>(moe_lds + ((static_cast<size_t>(MR) * dim + 15) & ~size_t(15)));
        for (int r = 0; r < mr; ++r) {
            const size_t row = static_cast<size_t>(pair_row[p0 + r]);
            if (xq) {
                for (int i = static_cast<int>(threadIdx.x); i < dim; i += static_cast<int>(blockDim.x)) {
                    lq[r * dim + i] = xq[row * dim + i];
                }
                for (int i = static_cast<int>(threadIdx.x); i < kb; i += static_cast<int>(blockDim.x)) {
                    ld[r * kb + i] = xd[row * kb + i];
                }
            } else {
                moe_quantize_lds(x + row * dim, dim, lq + r * dim, ld + r * kb);
            }
        }
        __syncthreads();
        aq = lq;
        ad = ld;
    }
    const size_t e = static_cast<size_t>(expert[gi]);
    if (blockIdx.z == 0) {
        kquant_i8_block<QG, MR, LPC>(aq, ad, gate + e * gate_stride,
                                     g + static_cast<size_t>(p0) * ff, mr, ff, dim,
                                     static_cast<int>(blockIdx.x), 0);
    } else {
        kquant_i8_block<QU, MR, LPC>(aq, ad, up + e * up_stride,
                                     u + static_cast<size_t>(p0) * ff, mr, ff, dim,
                                     static_cast<int>(blockIdx.x), 0);
    }
}

/* SiLU(gate) * up per pair, int8-quantized once into `hq` / `hd` for the
 * down GEMV (each down block used to redo it for its pair's row). */
__global__ void moe_silu_quant_kernel(const float *g, const float *u, int8_t *hq, float *hd,
                                      int ff) {
    extern __shared__ unsigned char moe_lds[];
    float *tmp = reinterpret_cast<float *>(moe_lds);
    const size_t p = blockIdx.x;
    moe_silu_quantize_lds(g + p * ff, u + p * ff, ff, tmp, hq + p * ff, hd + p * (ff / 32));
}

/* Down for every group: the group's quantized hidden rows (adjacent, since a
 * group's pairs are) times expert[g]. */
template <int QD, int LPC, int MR>
__global__ void moe_down_kernel(const int8_t *hq, const float *hd, const int32_t *group_start,
                                const int32_t *expert, const uint8_t *down,
                                unsigned long long down_stride, float *y, int dim, int ff) {
    const int gi = static_cast<int>(blockIdx.y);
    int p0, mr;
    moe_group(group_start, gi, p0, mr);
    kquant_i8_block<QD, MR, LPC>(hq + static_cast<size_t>(p0) * ff,
                                 hd + static_cast<size_t>(p0) * (ff / 32),
                                 down + static_cast<size_t>(expert[gi]) * down_stride,
                                 y + static_cast<size_t>(p0) * dim, mr, dim, ff,
                                 static_cast<int>(blockIdx.x), 0);
}

/* out[r] = sum of w[p] * y[p] over row r's pairs `row_start[r] .. row_start[r+1]`,
 * rounded like `scale_f32` then `residual_add` per expert (no FMA), so the
 * fused path matches the per-expert path bit for bit. Contraction is off
 * explicitly with plain operators: __fmul_rn/__fadd_rn (and this pragma around
 * them) still compiled to v_fmac_f32. */
__global__ void moe_combine_kernel(const float *y, const float *w, const int32_t *row_start,
                                   const int32_t *row_pairs, float *out, int rows, int dim) {
#pragma clang fp contract(off)
    const int d = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = static_cast<int>(blockIdx.y);
    if (d >= dim || r >= rows) {
        return;
    }
    float acc = 0.f;
    for (int i = row_start[r]; i < row_start[r + 1]; ++i) {
        const int p = row_pairs ? row_pairs[i] : i;
        acc = acc + y[static_cast<size_t>(p) * dim + d] * w[p];
    }
    out[static_cast<size_t>(r) * dim + d] = acc;
}


constexpr int MOE_LPC = 8; // lanes per column; 16 cut 1-row calls 10% but slowed grouped rows 40%

template <int QG, int QU, int MR>
static void moe_gate_up_launch(dim3 grid, size_t lds, hipStream_t st, const float *x,
                               const int8_t *xq, const float *xd, const int32_t *pair_row,
                               const int32_t *group_start, const int32_t *expert,
                               const uint8_t *gate, unsigned long long gs, const uint8_t *up,
                               unsigned long long us, float *g, float *u, int ff, int dim) {
    hipLaunchKernelGGL((moe_gate_up_kernel<QG, QU, MOE_LPC, MR>), grid, dim3(256), lds, st, x, xq,
                       xd, pair_row, group_start, expert, gate, gs, up, us, g, u, ff, dim);
}

template <int MR>
static bool moe_gate_up_types(int qg, int qu, dim3 grid, size_t lds, hipStream_t st,
                              const float *x, const int8_t *xq, const float *xd,
                              const int32_t *pair_row, const int32_t *group_start,
                              const int32_t *expert, const uint8_t *gate, unsigned long long gs,
                              const uint8_t *up, unsigned long long us, float *g, float *u, int ff,
                              int dim) {
#define MOE_GU(A, B)                                                                               \
    if (qg == A && qu == B) {                                                                      \
        moe_gate_up_launch<A, B, MR>(grid, lds, st, x, xq, xd, pair_row, group_start, expert,     \
                                     gate, gs, up, us, g, u, ff, dim);                             \
        return true;                                                                               \
    }
    MOE_GU(4, 4) MOE_GU(4, 5) MOE_GU(4, 6)
    MOE_GU(5, 4) MOE_GU(5, 5) MOE_GU(5, 6)
    MOE_GU(6, 4) MOE_GU(6, 5) MOE_GU(6, 6)
    MOE_GU(8, 8)
    MOE_GU(48, 48)
#undef MOE_GU
    return false;
}

template <int MR>
static bool moe_down_types(int qd, dim3 grid, hipStream_t st, const int8_t *hq,
                           const float *hd, const int32_t *group_start, const int32_t *expert,
                           const uint8_t *down, unsigned long long ds, float *y, int dim, int ff) {
#define MOE_D(Q)                                                                                   \
    if (qd == Q) {                                                                                 \
        hipLaunchKernelGGL((moe_down_kernel<Q, MOE_LPC, MR>), grid, dim3(256), 0, st, hq, hd,     \
                           group_start, expert, down, ds, y, dim, ff);                            \
        return true;                                                                               \
    }
    MOE_D(4) MOE_D(5) MOE_D(6) MOE_D(8) MOE_D(48)
#undef MOE_D
    return false;
}

static size_t moe_align(size_t v) { return (v + 255) & ~size_t(255); }

/* Scratch after the caller's packed inputs: gate, up, per-pair expert output. */
extern "C" uint64_t allpaca_moe_ffn_scratch_bytes(int32_t rows, int32_t dim, int32_t ff,
                                                  int32_t pairs) {
    (void)rows;
    const size_t p = static_cast<size_t>(pairs);
    return 2 * moe_align(p * ff * 4) + moe_align(p * dim * 4) + moe_align(p * ff) +
           moe_align(p * (ff / 32) * 4);
}

/* Expert-parallel MoE FFN over stacked experts (`qt` 4/5/6 = Q4_K/Q5_K/Q6_K,
 * 8 = Q8_0 in 36-byte staged blocks, 48 = MoESQ paired-4:8 FP4; gate and up are
 * both K-quant, both Q8_0 or both paired-4:8):
 * out[r] = sum over row r's pairs of w * down_e(silu(gate_e x_r) * up_e x_r).
 * Pairs are sorted by row; `row_start` has rows + 1 entries. Three launches,
 * whatever the expert count: an MI50 pays ~10 us per dependent launch, and at
 * one token the whole MoE reads only ~23 MB. */
extern "C" int allpaca_moe_ffn_kquant(const float *x, const int8_t *xq, const float *xd,
                                      int32_t rows, int32_t dim, int32_t ff, int32_t pairs,
                                      const int32_t *pair_row, const int32_t *expert,
                                      const float *pair_w, const int32_t *row_start,
                                      int32_t groups, const int32_t *group_start,
                                      const int32_t *row_pairs, int32_t max_group,
                                      const uint8_t *gate, int32_t gate_qt, uint64_t gate_stride,
                                      const uint8_t *up, int32_t up_qt, uint64_t up_stride,
                                      const uint8_t *down, int32_t down_qt, uint64_t down_stride,
                                      void *scratch, float *out, allpaca_stream_t stream) {
    if ((!x && !(xq && xd)) || !out || !scratch || rows <= 0 || pairs <= 0 || groups <= 0 ||
        dim % 256 != 0 || ff % 256 != 0 || !pair_row || !expert || !pair_w || !row_start ||
        (group_start == nullptr && groups != pairs) || max_group < 1 || max_group > 8) {
        return ALLPACA_ERR_INVALID;
    }
    auto kq = [](int32_t qt) { return qt == 4 || qt == 5 || qt == 6 || qt == 8 || qt == 48; };
    if (!kq(gate_qt) || !kq(up_qt) || !kq(down_qt)) {
        return ALLPACA_ERR_INVALID;
    }
    clear_last_error();
    hipStream_t st = as_stream(stream);
    uint8_t *base = static_cast<uint8_t *>(scratch);
    const size_t p = static_cast<size_t>(pairs);
    float *g = reinterpret_cast<float *>(base);
    base += moe_align(p * ff * 4);
    float *u = reinterpret_cast<float *>(base);
    base += moe_align(p * ff * 4);
    float *y = reinterpret_cast<float *>(base);
    base += moe_align(p * dim * 4);
    int8_t *hq = reinterpret_cast<int8_t *>(base);
    base += moe_align(p * ff);
    float *hd = reinterpret_cast<float *>(base);
    constexpr int COLS = 256 / MOE_LPC;
    const int mrc = max_group <= 1 ? 1 : (max_group <= 2 ? 2 : (max_group <= 4 ? 4 : 8));
    const size_t lds_up = ((static_cast<size_t>(mrc) * dim + 15) & ~size_t(15)) +
                          static_cast<size_t>(mrc) * (dim / 32) * 4;
    const dim3 grid_up(static_cast<unsigned>((ff + COLS - 1) / COLS), static_cast<unsigned>(groups), 2);
    const dim3 grid_down(static_cast<unsigned>((dim + COLS - 1) / COLS), static_cast<unsigned>(groups));
    bool ok = false;
#define MOE_MR(M)                                                                                  \
    ok = moe_gate_up_types<M>(gate_qt, up_qt, grid_up, lds_up, st, x, xq, xd, pair_row,            \
                              group_start, expert, gate, gate_stride, up, up_stride, g, u, ff,     \
                              dim);                                                                \
    if (ok) {                                                                                      \
        hipLaunchKernelGGL(moe_silu_quant_kernel, dim3(static_cast<unsigned>(pairs)), dim3(256),   \
                           static_cast<size_t>(ff) * 4, st, g, u, hq, hd, ff);                     \
    }                                                                                              \
    ok = ok && moe_down_types<M>(down_qt, grid_down, st, hq, hd, group_start, expert, down,        \
                                 down_stride, y, dim, ff)
    switch (mrc) {
    case 1: MOE_MR(1); break;
    case 2: MOE_MR(2); break;
    case 4: MOE_MR(4); break;
    default: MOE_MR(8); break;
    }
#undef MOE_MR
    if (!ok) {
        return ALLPACA_ERR_INVALID;
    }
    hipLaunchKernelGGL(moe_combine_kernel,
                       dim3(static_cast<unsigned>((dim + 255) / 256), static_cast<unsigned>(rows)),
                       dim3(256), 0, st, y, pair_w, row_start, row_pairs, out, rows, dim);
    return hip_status(hipGetLastError());
}

#else /* !ALLPACA_HAS_HIP */

extern "C" uint64_t allpaca_moe_ffn_scratch_bytes(int32_t, int32_t, int32_t, int32_t) {
    return 0;
}

extern "C" int allpaca_det_silu_mul(const float *, const float *, float *, int32_t,
                                    allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int32_t allpaca_hip_set_exact_rows(int32_t) { return 0; }

extern "C" int allpaca_moe_ffn_kquant(const float *, const int8_t *, const float *, int32_t, int32_t, int32_t, int32_t,
                                      const int32_t *, const int32_t *, const float *,
                                      const int32_t *, int32_t, const int32_t *, const int32_t *,
                                      int32_t, const uint8_t *, int32_t, uint64_t,
                                      const uint8_t *, int32_t, uint64_t, const uint8_t *, int32_t,
                                      uint64_t, void *, float *, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" uint64_t allpaca_hip_kernel_launches(void) { return 0; }
extern "C" int32_t allpaca_hip_device_is_gfx12(int32_t) { return 0; }
extern "C" int32_t allpaca_hip_set_prefill_wmma(int32_t) { return 0; }
extern "C" int32_t allpaca_hip_prefill_wmma_active(void) { return 0; }
extern "C" uint64_t allpaca_hip_wmma_launches(void) { return 0; }
extern "C" uint64_t allpaca_hip_thread_wmma_launches(void) { return 0; }

extern "C" uint64_t allpaca_hip_act_quant_reuses(void) { return 0; }

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

extern "C" int allpaca_hip_device_multiprocessors(int32_t, int32_t *count) {
    if (count) {
        *count = 0;
    }
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

extern "C" int allpaca_hip_memset_async(void *, int32_t, size_t, allpaca_stream_t) {
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

extern "C" int allpaca_hip_blas_warmup(int32_t) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_hip_stream_synchronize(allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_hip_device_synchronize(void) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

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

extern "C" int allpaca_q8_0_gemv_rounds_activations(int32_t, int32_t, int32_t) {
    return 0;
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

extern "C" int allpaca_gemm_nt_q4_0(const float *, const uint8_t *, float *, int32_t, int32_t,
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

extern "C" int allpaca_gemm_nt_rocmfp4_fast(const float *, const uint8_t *, float *, int32_t,
                                            int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_nvfp4(const float *, const uint8_t *, float *, int32_t, int32_t,
                                     int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_p48fp4(const float *, const uint8_t *, float *, int32_t, int32_t,
                                      int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_rocmfp4_f32(const float *, const uint8_t *, float *, int32_t,
                                           int32_t, int32_t, int32_t, allpaca_stream_t) {
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
                                            float *, int32_t, int32_t, int32_t, float,
                                            allpaca_stream_t) {
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

extern "C" int allpaca_kquant_i8_lanes(int32_t, int32_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_kquant_i8_set_loads_first(int32_t) { return ALLPACA_ERR_NOT_IMPLEMENTED; }

extern "C" int allpaca_gemm_nt_q2_0(const float *, const uint8_t *, float *, int32_t, int32_t,
                                    int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_q2_0_moe_ffn(const float *, const uint8_t *, const int32_t *, const float *,
                                    int32_t, float *, float *, int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_q4x_routed_ffn(const float *, const float *, const uint8_t *, int32_t,
                                      int32_t, float, void *, float *, float *, int32_t, int32_t,
                                      int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_silu_scale(float *, int32_t, float, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_q4x_stream_norm(const float *, const float *, float *, int32_t, int32_t,
                                       int32_t, float, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_q4x_ple_apply(float *, const float *, const float *, const float *,
                                     const float *, const float *, const float *, float *, int32_t,
                                     int32_t, int32_t, int32_t, int32_t, float, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_q4x_gated_mean(const float *, const float *, float *, int32_t, int32_t,
                                      int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_q4x_combine(float *, const float *, const float *, int32_t, int32_t,
                                   int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_q4x_ffn_sum(const float *, const float *, const float *, float *, int32_t,
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

extern "C" int allpaca_gemm_nt_f16_wmma(const float *, const uint16_t *, float *, int32_t, int32_t,
                                       int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_i8(const int8_t *, const int8_t *, float *, int32_t, int32_t, int32_t,
                                  allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_i8_wmma(const int8_t *, const int8_t *, float *, int32_t, int32_t,
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

extern "C" int allpaca_attention_wmma_prefill(const float *, const float *, const float *, float *,
                                             int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                                             allpaca_stream_t) {
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

extern "C" int allpaca_moe_route_softmax_topk(const float *, int32_t, int32_t, int32_t,
                                              const float *, const float *, int32_t, float *,
                                              int32_t *, int32_t *, int32_t *, float *,
                                              const uint8_t *, uint32_t *, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_scale_rows(float *, const float *, int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_topk_chunks(const float *, int32_t, int32_t, int32_t, int32_t, float *,
                                   int32_t *, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_rope_neox_rows(float *, int32_t, int32_t, int32_t, int32_t, float,
                                      allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_rope_neox_ff_rows(float *, int32_t, int32_t, int32_t, int32_t, float,
                                         const float *, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gelu_tanh_mul(float *, const float *, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_softcap(float *, int32_t, float, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_dflash_conv(const float *, const float *, const float *, float *, int32_t,
                                   int32_t, int32_t, int32_t, int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_copy_f32_kernel(float *, const float *, int32_t, allpaca_stream_t) {
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

extern "C" int allpaca_gather_rows(float *, const float *, const int32_t *, int32_t, int32_t,
                                   allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_qsa_pool(const float *, const float *, float *, int32_t, int32_t, int32_t, int32_t, int32_t,
                                float, const int32_t *, float, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_qsa_select(float *, const float *, const float *, float *, int32_t *, int32_t, int32_t,
                                  int32_t, int32_t, int32_t, float, const int32_t *, float, int32_t, int32_t,
                                  int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_scale_heads_sigmoid(float *, const float *, int32_t, int32_t, int32_t,
                                           allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt_q8_0_pair(const float *, const uint8_t *, float *,
                                         const uint8_t *, float *, int32_t, int32_t, int32_t,
                                         int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_scale_add_copy(float *, float *, float, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gdn_norm_gate(const float *, const float *, const float *, float *,
                                     int32_t, int32_t, float, allpaca_stream_t) {
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

extern "C" int allpaca_round_f16(float *, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_bias_act_rows(float *, const float *, int32_t, int32_t, int32_t,
                                     allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_vision_rope_rows(float *, const int32_t *, const int32_t *, int32_t, int32_t,
                                        int32_t, float, allpaca_stream_t) {
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

extern "C" int allpaca_gdn_step_fused(const float *, const float *, float *, float *,
                                      const float *, const float *, const float *, float *, float *,
                                      float *, float *, int32_t, int32_t, int32_t, int32_t, int32_t,
                                      float, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gdn_steps(float *, const float *, float *, float *, const float *,
                                 const float *, float *, float *, float *, int32_t, int32_t,
                                 int32_t, int32_t, int32_t, float, int32_t, int32_t,
                                 allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gdn_conv1d_step(const float *, const float *, float *, float *, int32_t,
                                       int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_expand_k_heads(const float *, float *, int32_t, int32_t, int32_t, int32_t,
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

extern "C" int allpaca_rope_yarn_rows(float *, int32_t, int32_t, int32_t, int32_t, int32_t,
                                      int32_t, float, float, float, float, int32_t,
                                      allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_rope_imrope_qk(float *, float *, int32_t, int32_t, int32_t, int32_t,
                                      const int32_t *, float, const int32_t *, allpaca_stream_t) {
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

extern "C" int allpaca_embed_gather_rocmfp4_fast(const uint8_t *, int32_t, float *, int32_t,
                                                 allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_embed_gather_rocmfp4_fast_rows(const uint8_t *, const int32_t *, float *,
                                                      int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

#endif
