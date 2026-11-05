/* Allpaca CUDA kernel ABI. Compiles with nvcc even when no NVIDIA GPU is present;
 * device APIs then report count=0 / ERR_DEVICE. */

#include "allpaca_cuda_kernels.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <mutex>

#define ALLPACA_HAS_CUDA 1

static int cuda_status(cudaError_t err) {
    if (err == cudaSuccess) {
        return ALLPACA_OK;
    }
    /* API return already encodes the failure; clear the sticky mirror so a
     * later cudaGetLastError() after an unrelated launch is not poisoned
     * (e.g. cudaHostRegister / cudaFreeAsync unsupported on Kepler). */
    (void)cudaGetLastError();
    return ALLPACA_ERR_DEVICE;
}

static cudaStream_t as_stream(allpaca_cuda_stream_t stream) {
    return static_cast<cudaStream_t>(stream);
}

extern "C" int allpaca_cuda_device_count(int32_t *count) {
    if (!count) {
        return ALLPACA_ERR_INVALID;
    }
    *count = 0;
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) {
        /* No driver / no device: treat as zero GPUs, not a hard build failure. */
        return ALLPACA_OK;
    }
    *count = n > 0 ? n : 0;
    return ALLPACA_OK;
}

extern "C" int allpaca_cuda_device_props(int32_t id, char *name, int32_t name_len,
                                        uint64_t *memory_bytes, double *bandwidth_gbps) {
    if (!name || name_len <= 0 || !memory_bytes || !bandwidth_gbps) {
        return ALLPACA_ERR_INVALID;
    }
    cudaDeviceProp prop;
    if (cuda_status(cudaGetDeviceProperties(&prop, id)) != ALLPACA_OK) {
        return ALLPACA_ERR_DEVICE;
    }
    std::snprintf(name, static_cast<size_t>(name_len), "%s", prop.name);
    *memory_bytes = static_cast<uint64_t>(prop.totalGlobalMem);
    /* CUDA 13 dropped memoryClockRate from cudaDeviceProp; query attributes.
     * Same theoretical peak as HIP: clock (GHz) * bus bytes * 2 (DDR). */
    int mem_clock_khz = 0;
    int bus_width_bits = 0;
    const cudaError_t clk =
        cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, id);
    const cudaError_t bus =
        cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, id);
    *bandwidth_gbps = 0.0;
    if (clk == cudaSuccess && bus == cudaSuccess && mem_clock_khz > 0 && bus_width_bits > 0) {
        const double clock_ghz = static_cast<double>(mem_clock_khz) / 1.0e6;
        const double bytes_per_cycle = static_cast<double>(bus_width_bits) / 8.0 * 2.0;
        *bandwidth_gbps = clock_ghz * bytes_per_cycle;
    }
    return ALLPACA_OK;
}

extern "C" int allpaca_cuda_device_arch(int32_t id, char *arch, int32_t arch_len,
                                       int32_t *l2_bytes) {
    if (!arch || arch_len <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    cudaDeviceProp prop;
    if (cuda_status(cudaGetDeviceProperties(&prop, id)) != ALLPACA_OK) {
        return ALLPACA_ERR_DEVICE;
    }
    /* major.minor → sm_86 / sm_120 (single-digit minor). */
    if (prop.major <= 0 || prop.minor < 0 || prop.minor > 9) {
        return ALLPACA_ERR_DEVICE;
    }
    std::snprintf(arch, static_cast<size_t>(arch_len), "sm_%d%d", prop.major, prop.minor);
    if (l2_bytes) {
        *l2_bytes = prop.l2CacheSize;
    }
    return ALLPACA_OK;
}

extern "C" int allpaca_cuda_malloc(void **ptr, size_t bytes, int32_t device) {
    if (!ptr || bytes == 0) {
        return ALLPACA_ERR_INVALID;
    }
    if (cuda_status(cudaSetDevice(device)) != ALLPACA_OK) {
        return ALLPACA_ERR_DEVICE;
    }
    return cuda_status(cudaMalloc(ptr, bytes));
}

extern "C" int allpaca_cuda_free(void *ptr) {
    if (!ptr) {
        return ALLPACA_OK;
    }
    return cuda_status(cudaFree(ptr));
}

extern "C" int allpaca_cuda_free_async(void *ptr, allpaca_cuda_stream_t stream) {
    if (!ptr) {
        return ALLPACA_OK;
    }
    cudaStream_t s = as_stream(stream);
    if (s) {
        cudaError_t err = cudaFreeAsync(ptr, s);
        if (err == cudaSuccess) {
            return ALLPACA_OK;
        }
        /* Kepler / older drivers: stream-ordered alloc unsupported. */
        (void)cudaGetLastError();
    }
    return cuda_status(cudaFree(ptr));
}

extern "C" int allpaca_cuda_host_malloc(void **ptr, size_t bytes) {
    if (!ptr || bytes == 0) {
        return ALLPACA_ERR_INVALID;
    }
    return cuda_status(cudaHostAlloc(ptr, bytes, cudaHostAllocDefault));
}

extern "C" int allpaca_cuda_host_free(void *ptr) {
    if (!ptr) {
        return ALLPACA_OK;
    }
    return cuda_status(cudaFreeHost(ptr));
}

extern "C" int allpaca_cuda_host_register(void *ptr, size_t bytes, unsigned int flags) {
    if (!ptr || bytes == 0) {
        return ALLPACA_ERR_INVALID;
    }
    return cuda_status(cudaHostRegister(ptr, bytes, flags));
}

extern "C" int allpaca_cuda_host_unregister(void *ptr) {
    if (!ptr) {
        return ALLPACA_OK;
    }
    return cuda_status(cudaHostUnregister(ptr));
}

extern "C" int allpaca_cuda_memcpy(void *dst, const void *src, size_t bytes, int32_t kind,
                                   allpaca_cuda_stream_t stream) {
    if ((!dst && bytes) || (!src && bytes)) {
        return ALLPACA_ERR_INVALID;
    }
    cudaMemcpyKind ck = cudaMemcpyDefault;
    bool force_async = false;
    switch (kind) {
    case ALLPACA_MEMCPY_H2D:
        ck = cudaMemcpyHostToDevice;
        break;
    case ALLPACA_MEMCPY_D2H:
        ck = cudaMemcpyDeviceToHost;
        break;
    case ALLPACA_MEMCPY_D2D:
        ck = cudaMemcpyDeviceToDevice;
        break;
    case ALLPACA_MEMCPY_D2H_ASYNC:
        ck = cudaMemcpyDeviceToHost;
        force_async = true;
        break;
    default:
        ck = cudaMemcpyDefault;
        break;
    }
    cudaStream_t s = as_stream(stream);
    if (s || force_async) {
        return cuda_status(cudaMemcpyAsync(dst, src, bytes, ck, s));
    }
    return cuda_status(cudaMemcpy(dst, src, bytes, ck));
}

extern "C" int allpaca_cuda_memcpy_batch(void **dsts, void **srcs, size_t *sizes, size_t count,
                                         int32_t kind, allpaca_cuda_stream_t stream) {
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
    cudaMemcpyKind ck = cudaMemcpyDefault;
    switch (kind) {
    case ALLPACA_MEMCPY_H2D:
        ck = cudaMemcpyHostToDevice;
        break;
    case ALLPACA_MEMCPY_D2H:
    case ALLPACA_MEMCPY_D2H_ASYNC:
        ck = cudaMemcpyDeviceToHost;
        break;
    case ALLPACA_MEMCPY_D2D:
        ck = cudaMemcpyDeviceToDevice;
        break;
    default:
        ck = cudaMemcpyDefault;
        break;
    }
    cudaStream_t s = as_stream(stream);
    for (size_t i = 0; i < count; ++i) {
        cudaError_t err = cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], ck, s);
        if (err != cudaSuccess) {
            return cuda_status(err);
        }
    }
    if (kind == ALLPACA_MEMCPY_D2H) {
        if (s) {
            return cuda_status(cudaStreamSynchronize(s));
        }
        return cuda_status(cudaDeviceSynchronize());
    }
    return ALLPACA_OK;
}

extern "C" int allpaca_cuda_memset(void *dst, int32_t value, size_t bytes,
                                   allpaca_cuda_stream_t stream) {
    if (!dst && bytes) {
        return ALLPACA_ERR_INVALID;
    }
    if (bytes == 0) {
        return ALLPACA_OK;
    }
    cudaStream_t s = as_stream(stream);
    const int fill = value & 0xff;
    if (s) {
        return cuda_status(cudaMemsetAsync(dst, fill, bytes, s));
    }
    return cuda_status(cudaMemset(dst, fill, bytes));
}

extern "C" int allpaca_cuda_set_device(int32_t device) {
    return cuda_status(cudaSetDevice(device));
}

extern "C" int allpaca_cuda_can_p2p(int32_t src, int32_t dst, int32_t *can) {
    if (!can) {
        return ALLPACA_ERR_INVALID;
    }
    int yes = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&yes, src, dst);
    if (err != cudaSuccess) {
        *can = 0;
        return cuda_status(err);
    }
    *can = yes;
    return ALLPACA_OK;
}

extern "C" int allpaca_cuda_enable_peer_access(int32_t src, int32_t dst) {
    cudaError_t err = cudaSetDevice(src);
    if (err != cudaSuccess) {
        return cuda_status(err);
    }
    err = cudaDeviceEnablePeerAccess(dst, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        /* Success, but the runtime still records it as the last error. */
        (void)cudaGetLastError();
        return ALLPACA_OK;
    }
    return cuda_status(err);
}

extern "C" int allpaca_cuda_memcpy_peer(void *dst, int32_t dst_dev, const void *src, int32_t src_dev,
                                        size_t bytes, allpaca_cuda_stream_t stream) {
    if ((!dst && bytes) || (!src && bytes)) {
        return ALLPACA_ERR_INVALID;
    }
    cudaStream_t s = as_stream(stream);
    cudaError_t err = cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, s);
    if (err != cudaSuccess) {
        return cuda_status(err);
    }
    if (s) {
        return cuda_status(cudaStreamSynchronize(s));
    }
    return cuda_status(cudaDeviceSynchronize());
}

extern "C" int allpaca_cuda_mem_info(int32_t device, uint64_t *free_bytes, uint64_t *total_bytes) {
    if (!free_bytes || !total_bytes) {
        return ALLPACA_ERR_INVALID;
    }
    int prev = 0;
    cudaError_t err = cudaGetDevice(&prev);
    if (err != cudaSuccess) {
        return cuda_status(err);
    }
    const bool rebind = prev != device;
    if (rebind) {
        err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            return cuda_status(err);
        }
    }
    size_t free_b = 0;
    size_t total_b = 0;
    err = cudaMemGetInfo(&free_b, &total_b);
    if (rebind) {
        (void)cudaSetDevice(prev);
    }
    if (err != cudaSuccess) {
        return cuda_status(err);
    }
    *free_bytes = static_cast<uint64_t>(free_b);
    *total_bytes = static_cast<uint64_t>(total_b);
    return ALLPACA_OK;
}

extern "C" int allpaca_cuda_mem_used(int32_t device, uint64_t *used_bytes) {
    if (!used_bytes) {
        return ALLPACA_ERR_INVALID;
    }
    uint64_t free_b = 0;
    uint64_t total_b = 0;
    int rc = allpaca_cuda_mem_info(device, &free_b, &total_b);
    if (rc != ALLPACA_OK) {
        return rc;
    }
    *used_bytes = total_b - free_b;
    return ALLPACA_OK;
}

extern "C" int allpaca_cuda_stream_synchronize(allpaca_cuda_stream_t stream) {
    cudaStream_t s = as_stream(stream);
    if (s) {
        return cuda_status(cudaStreamSynchronize(s));
    }
    return cuda_status(cudaDeviceSynchronize());
}

extern "C" int allpaca_cuda_stream_create(allpaca_cuda_stream_t *stream) {
    if (!stream) {
        return ALLPACA_ERR_INVALID;
    }
    cudaStream_t s = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        *stream = nullptr;
        return cuda_status(err);
    }
    *stream = s;
    return ALLPACA_OK;
}

extern "C" int allpaca_cuda_stream_destroy(allpaca_cuda_stream_t stream) {
    if (!stream) {
        return ALLPACA_OK;
    }
    return cuda_status(cudaStreamDestroy(as_stream(stream)));
}

extern "C" int allpaca_cuda_event_create(void **event) {
    if (!event) {
        return ALLPACA_ERR_INVALID;
    }
    cudaEvent_t ev = nullptr;
    cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        *event = nullptr;
        return cuda_status(err);
    }
    *event = ev;
    return ALLPACA_OK;
}

extern "C" int allpaca_cuda_event_create_timing(void **event) {
    if (!event) {
        return ALLPACA_ERR_INVALID;
    }
    cudaEvent_t ev = nullptr;
    cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDefault);
    if (err != cudaSuccess) {
        *event = nullptr;
        return cuda_status(err);
    }
    *event = ev;
    return ALLPACA_OK;
}

extern "C" int allpaca_cuda_event_destroy(void *event) {
    if (!event) {
        return ALLPACA_OK;
    }
    return cuda_status(cudaEventDestroy(static_cast<cudaEvent_t>(event)));
}

extern "C" int allpaca_cuda_event_record(void *event, allpaca_cuda_stream_t stream) {
    if (!event) {
        return ALLPACA_ERR_INVALID;
    }
    return cuda_status(cudaEventRecord(static_cast<cudaEvent_t>(event), as_stream(stream)));
}

extern "C" int allpaca_cuda_event_elapsed_us(void *start, void *stop, double *us) {
    if (!start || !stop || !us) {
        return ALLPACA_ERR_INVALID;
    }
    float ms = 0.0f;
    cudaError_t err = cudaEventElapsedTime(
        &ms, static_cast<cudaEvent_t>(start), static_cast<cudaEvent_t>(stop));
    if (err != cudaSuccess) {
        return cuda_status(err);
    }
    *us = static_cast<double>(ms) * 1000.0;
    return ALLPACA_OK;
}

extern "C" int allpaca_cuda_stream_wait_event(allpaca_cuda_stream_t stream, void *event) {
    if (!event) {
        return ALLPACA_ERR_INVALID;
    }
    return cuda_status(
        cudaStreamWaitEvent(as_stream(stream), static_cast<cudaEvent_t>(event), 0));
}

__global__ void gemm_nt_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.f;
    const float *a_row = a + static_cast<size_t>(row) * static_cast<size_t>(k);
    const float *b_row = b + static_cast<size_t>(col) * static_cast<size_t>(k);
    for (int t = 0; t < k; ++t) {
        acc += a_row[t] * b_row[t];
    }
    c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc;
}

extern "C" int allpaca_cuda_gemm_nt(const float *a, const float *b, float *c, int32_t m, int32_t n,
                                    int32_t k, allpaca_cuda_stream_t stream) {
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    gemm_nt_kernel<<<grid, block, 0, as_stream(stream)>>>(a, b, c, m, n, k);
    return cuda_status(cudaGetLastError());
}

/* Staged Q8_0: f32 scale at 0, qs at 4, 36-byte stride (HIP Q8_STAGED_*). */
constexpr int Q8_STAGED_STRIDE = 36;
constexpr int Q8_STAGED_QS = 4;

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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
        acc += __shfl_down(acc, off, 4);
#else
        acc += __shfl_down_sync(0xffffffffu, acc, off, 4);
#endif
    }
    if (col < n && sublane == 0) {
        c[static_cast<size_t>(row) * static_cast<size_t>(n) + col] = acc;
    }
}

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

extern "C" int allpaca_cuda_gemm_nt_q8_0(const float *a, const uint8_t *b, float *c, int32_t m,
                                         int32_t n, int32_t k, allpaca_cuda_stream_t stream) {
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    if (m > 8) {
        constexpr int BM = 64;
        constexpr int BN = 64;
        dim3 block(16, 16);
        dim3 grid(static_cast<unsigned>((n + BN - 1) / BN),
                  static_cast<unsigned>((m + BM - 1) / BM));
        gemm_nt_q8_0_tiled_kernel<<<grid, block, 0, as_stream(stream)>>>(a, b, c, m, n, k);
    } else {
        constexpr int warps = 16;
        constexpr int cols = 8;
        constexpr int cols_per_block = warps * cols;
        dim3 block(512);
        dim3 grid(static_cast<unsigned>((n + cols_per_block - 1) / cols_per_block),
                  static_cast<unsigned>(m));
        gemv_nt_q8_0_f32_kernel<<<grid, block, 0, as_stream(stream)>>>(a, b, c, m, n, k);
    }
    return cuda_status(cudaGetLastError());
}

__global__ void residual_add_kernel(const float *x, const float *y, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] + y[i];
    }
}

extern "C" int allpaca_cuda_residual_add(const float *x, const float *y, float *out, int32_t n,
                                         allpaca_cuda_stream_t stream) {
    if (!x || !y || !out || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    int block = 256;
    int grid = (n + block - 1) / block;
    residual_add_kernel<<<grid, block, 0, as_stream(stream)>>>(x, y, out, n);
    return cuda_status(cudaGetLastError());
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

__global__ void silu_mul_kernel(const float *gate, const float *up, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = det_silu_mul(gate[i], up[i]);
    }
}

extern "C" int allpaca_cuda_silu_mul(const float *gate, const float *up, float *out, int32_t n,
                                     allpaca_cuda_stream_t stream) {
    if (!gate || !up || !out || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    int block = 256;
    int grid = (n + block - 1) / block;
    silu_mul_kernel<<<grid, block, 0, as_stream(stream)>>>(gate, up, out, n);
    return cuda_status(cudaGetLastError());
}

__global__ void gelu_mul_kernel(const float *gate, const float *up, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        /* Exact GELU via erff — matches HIP allpaca_gelu_mul / approx CPU gelu. */
        float gelu = 0.5f * g * (1.0f + erff(g * 0.7071067811865476f));
        out[i] = gelu * up[i];
    }
}

extern "C" int allpaca_cuda_gelu_mul(const float *gate, const float *up, float *out, int32_t n,
                                     allpaca_cuda_stream_t stream) {
    if (!gate || !up || !out || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    int block = 256;
    int grid = (n + block - 1) / block;
    gelu_mul_kernel<<<grid, block, 0, as_stream(stream)>>>(gate, up, out, n);
    return cuda_status(cudaGetLastError());
}

__global__ void relu_sqr_kernel(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v > 0.f ? v * v : 0.f;
    }
}

extern "C" int allpaca_cuda_relu_sqr(float *x, int32_t n, allpaca_cuda_stream_t stream) {
    if (!x || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_sqr_kernel<<<grid, block, 0, as_stream(stream)>>>(x, n);
    return cuda_status(cudaGetLastError());
}

__global__ void silu_kernel(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = v / (1.f + expf(-v));
    }
}

extern "C" int allpaca_cuda_silu(float *x, int32_t n, allpaca_cuda_stream_t stream) {
    if (!x || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    int block = 256;
    int grid = (n + block - 1) / block;
    silu_kernel<<<grid, block, 0, as_stream(stream)>>>(x, n);
    return cuda_status(cudaGetLastError());
}

__global__ void scale_f32_kernel(float *x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= scale;
    }
}

extern "C" int allpaca_cuda_scale_f32(float *x, int32_t n, float scale,
                                      allpaca_cuda_stream_t stream) {
    if (!x || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    int block = 256;
    int grid = (n + block - 1) / block;
    scale_f32_kernel<<<grid, block, 0, as_stream(stream)>>>(x, scale, n);
    return cuda_status(cudaGetLastError());
}

__global__ void expand_k_heads_kernel(const float *src, float *dst, int n_k, int n_v, int head_k) {
    int vh = static_cast<int>(blockIdx.x);
    int j = static_cast<int>(threadIdx.x);
    if (vh >= n_v || j >= head_k) {
        return;
    }
    int kh = vh % n_k;
    dst[vh * head_k + j] = src[kh * head_k + j];
}

extern "C" int allpaca_cuda_expand_k_heads(const float *src, float *dst, int32_t n_k, int32_t n_v,
                                           int32_t head_k, allpaca_cuda_stream_t stream) {
    if (!src || !dst || n_k <= 0 || n_v <= 0 || head_k <= 0 || (n_v % n_k) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    expand_k_heads_kernel<<<static_cast<unsigned>(n_v), static_cast<unsigned>(head_k), 0,
                            as_stream(stream)>>>(src, dst, n_k, n_v, head_k);
    return cuda_status(cudaGetLastError());
}

__device__ inline float gdn_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

__global__ void gdn_decay_beta_kernel(const float *alpha, const float *dt_bias, const float *a,
                                      float *decay, float *beta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    decay[i] = expf(gdn_softplus(alpha[i] + dt_bias[i]) * a[i]);
    beta[i] = 1.0f / (1.0f + expf(-beta[i]));
}

extern "C" int allpaca_cuda_gdn_decay_beta(const float *alpha, const float *dt_bias, const float *a,
                                           float *decay, float *beta, int32_t n,
                                           allpaca_cuda_stream_t stream) {
    if (!alpha || !dt_bias || !a || !decay || !beta || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    int block = 256;
    int grid = (n + block - 1) / block;
    gdn_decay_beta_kernel<<<grid, block, 0, as_stream(stream)>>>(alpha, dt_bias, a, decay, beta, n);
    return cuda_status(cudaGetLastError());
}

__device__ inline float gdn_silu(float x) { return x / (1.0f + expf(-x)); }

__global__ void gdn_conv1d_compute_kernel(const float *x, const float *kernel,
                                          const float *conv_state, float *out, int qkv_dim,
                                          int conv_kernel) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= qkv_dim) {
        return;
    }
    int hist = conv_kernel - 1;
    float acc = 0.0f;
    for (int i = 0; i < hist; ++i) {
        acc += kernel[i + c * conv_kernel] * conv_state[i * qkv_dim + c];
    }
    acc += kernel[(conv_kernel - 1) + c * conv_kernel] * x[c];
    out[c] = gdn_silu(acc);
}

__global__ void gdn_conv1d_shift_kernel(const float *x, float *conv_state, int qkv_dim,
                                        int conv_kernel) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= qkv_dim) {
        return;
    }
    int hist = conv_kernel - 1;
    if (hist <= 0) {
        return;
    }
    for (int i = 0; i < hist - 1; ++i) {
        conv_state[i * qkv_dim + c] = conv_state[(i + 1) * qkv_dim + c];
    }
    conv_state[(hist - 1) * qkv_dim + c] = x[c];
}

extern "C" int allpaca_cuda_gdn_conv1d_step(const float *x, const float *kernel, float *conv_state,
                                            float *out, int32_t qkv_dim, int32_t conv_kernel,
                                            allpaca_cuda_stream_t stream) {
    if (!x || !kernel || !conv_state || !out || qkv_dim <= 0 || conv_kernel <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    int threads = 256;
    int blocks = (qkv_dim + threads - 1) / threads;
    gdn_conv1d_compute_kernel<<<blocks, threads, 0, as_stream(stream)>>>(
        x, kernel, conv_state, out, qkv_dim, conv_kernel);
    int rc = cuda_status(cudaGetLastError());
    if (rc != ALLPACA_OK) {
        return rc;
    }
    gdn_conv1d_shift_kernel<<<blocks, threads, 0, as_stream(stream)>>>(x, conv_state, qkv_dim,
                                                                       conv_kernel);
    return cuda_status(cudaGetLastError());
}

/* One block per V-head. Thread 0 runs the AR delta step (matches host deltanet). */
__global__ void gated_delta_step_kernel(float *q, float *k, const float *v, const float *decay,
                                        const float *beta, float *state, float *out, int n_v_heads,
                                        int head_v, int head_k, float eps) {
    int h = static_cast<int>(blockIdx.x);
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
    float inv_q = rsqrtf(ss_q + eps);
    float inv_k = rsqrtf(ss_k + eps);
    float q_scale = rsqrtf(static_cast<float>(head_k));
    for (int j = 0; j < head_k; ++j) {
        q_h[j] *= inv_q * q_scale;
        k_h[j] *= inv_k;
    }

    float g = decay[h];
    float b = beta[h];
    for (int i = 0; i < head_v; ++i) {
        float *row = state_h + i * head_k;
        float acc = 0.0f;
        for (int j = 0; j < head_k; ++j) {
            row[j] *= g;
            acc += row[j] * k_h[j];
        }
        float delta = b * (v_h[i] - acc);
        float y = 0.0f;
        for (int j = 0; j < head_k; ++j) {
            row[j] += delta * k_h[j];
            y += row[j] * q_h[j];
        }
        out_h[i] = y;
    }
}

extern "C" int allpaca_cuda_gated_delta_step(float *q, float *k, const float *v,
                                             const float *decay, const float *beta, float *state,
                                             float *out, int32_t n_v_heads, int32_t head_v,
                                             int32_t head_k, float eps,
                                             allpaca_cuda_stream_t stream) {
    if (!q || !k || !v || !decay || !beta || !state || !out || n_v_heads <= 0 || head_v <= 0 ||
        head_k <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    gated_delta_step_kernel<<<static_cast<unsigned>(n_v_heads), 1, 0, as_stream(stream)>>>(
        q, k, v, decay, beta, state, out, n_v_heads, head_v, head_k, eps);
    return cuda_status(cudaGetLastError());
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

__global__ void mamba2_conv_compute_kernel(const float *x, const float *kernel, const float *bias,
                                           const float *conv_state, float *out, int conv_dim,
                                           int conv_kernel) {
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

extern "C" int allpaca_cuda_mamba2_conv_step(const float *xbc, const float *kernel,
                                             const float *bias, float *conv_state, float *out,
                                             int32_t conv_dim, int32_t conv_kernel,
                                             allpaca_cuda_stream_t stream) {
    if (!xbc || !kernel || !conv_state || !out || conv_dim <= 0 || conv_kernel <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    int threads = 256;
    int blocks = (conv_dim + threads - 1) / threads;
    mamba2_conv_compute_kernel<<<blocks, threads, 0, as_stream(stream)>>>(
        xbc, kernel, bias, conv_state, out, conv_dim, conv_kernel);
    int rc = cuda_status(cudaGetLastError());
    if (rc != ALLPACA_OK) {
        return rc;
    }
    gdn_conv1d_shift_kernel<<<blocks, threads, 0, as_stream(stream)>>>(xbc, conv_state, conv_dim,
                                                                       conv_kernel);
    return cuda_status(cudaGetLastError());
}

extern "C" int allpaca_cuda_mamba2_ssd_step(const float *z, const float *xbc, const float *dt,
                                            const float *a, const float *d, const float *dt_bias,
                                            const float *norm, float *state, float *out,
                                            int32_t d_inner, int32_t d_state, int32_t n_heads,
                                            int32_t n_groups, float eps,
                                            allpaca_cuda_stream_t stream) {
    if (!z || !xbc || !dt || !a || !d || !dt_bias || !norm || !state || !out || d_inner <= 0 ||
        d_state <= 0 || n_heads <= 0 || n_groups <= 0 || d_inner % n_heads != 0 ||
        n_heads % n_groups != 0) {
        return ALLPACA_ERR_INVALID;
    }
    mamba2_ssd_kernel<<<static_cast<unsigned>(n_groups), 256, 0, as_stream(stream)>>>(
        z, xbc, dt, a, d, dt_bias, norm, state, out, d_inner, d_state, n_heads, n_groups, eps);
    return cuda_status(cudaGetLastError());
}

__global__ void scale_heads_sigmoid_kernel(float *attn, const float *scores, int n_tok, int n_heads,
                                           int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = n_tok * n_heads * head_dim;
    if (idx >= n) {
        return;
    }
    int head_elems = n_heads * head_dim;
    int tok = idx / head_elems;
    int within = idx % head_elems;
    int h = within / head_dim;
    float s = scores[tok * n_heads + h];
    attn[idx] = det_sigmoid_mul(attn[idx], s);
}

extern "C" int allpaca_cuda_scale_heads_sigmoid(float *attn, const float *scores, int32_t n_tok,
                                                int32_t n_heads, int32_t head_dim,
                                                allpaca_cuda_stream_t stream) {
    if (!attn || !scores || n_tok <= 0 || n_heads <= 0 || head_dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    int n = n_tok * n_heads * head_dim;
    int block = 256;
    int grid = (n + block - 1) / block;
    scale_heads_sigmoid_kernel<<<grid, block, 0, as_stream(stream)>>>(attn, scores, n_tok, n_heads,
                                                                      head_dim);
    return cuda_status(cudaGetLastError());
}

/* One block per row. Decode uses rows=1. n is reduced in shared memory (256 threads). */
__global__ void rmsnorm_kernel(const float *x, const float *w, float *out, int n, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    x += static_cast<size_t>(row) * static_cast<size_t>(n);
    out += static_cast<size_t>(row) * static_cast<size_t>(n);
    __shared__ float buf[256];
    int tid = threadIdx.x;
    float acc = 0.f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = x[i];
        acc = fmaf(v, v, acc);
    }
    buf[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            buf[tid] += buf[tid + stride];
        }
        __syncthreads();
    }
    const float inv = det_rsqrtf(__fadd_rn(__fdiv_rn(buf[0], static_cast<float>(n)), eps));
    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = __fmul_rn(__fmul_rn(x[i], inv), w[i]);
    }
}

extern "C" int allpaca_cuda_rmsnorm_rows(const float *x, const float *w, float *out, int32_t rows,
                                        int32_t n, float eps, allpaca_cuda_stream_t stream) {
    if (!x || !w || !out || rows <= 0 || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    rmsnorm_kernel<<<static_cast<unsigned>(rows), 256, 0, as_stream(stream)>>>(x, w, out, n, eps);
    return cuda_status(cudaGetLastError());
}

extern "C" int allpaca_cuda_rmsnorm(const float *x, const float *w, float *out, int32_t n, float eps,
                                    allpaca_cuda_stream_t stream) {
    return allpaca_cuda_rmsnorm_rows(x, w, out, 1, n, eps, stream);
}

extern "C" int allpaca_cuda_rmsnorm_heads(float *x, const float *weight, int32_t n_heads,
                                          int32_t head_dim, float eps,
                                          allpaca_cuda_stream_t stream) {
    /* In-place: rmsnorm_kernel finishes the reduction before any store, so x==out is safe. */
    return allpaca_cuda_rmsnorm_rows(x, weight, x, n_heads, head_dim, eps, stream);
}

__global__ void embed_gather_kernel(const float *table, int token, float *out, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) {
        return;
    }
    out[idx] = table[static_cast<size_t>(token) * static_cast<size_t>(dim) + static_cast<size_t>(idx)];
}

extern "C" int allpaca_cuda_embed_gather(const float *table, int32_t token, float *out, int32_t dim,
                                         allpaca_cuda_stream_t stream) {
    if (!table || !out || dim <= 0 || token < 0) {
        return ALLPACA_ERR_INVALID;
    }
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    embed_gather_kernel<<<blocks, threads, 0, as_stream(stream)>>>(table, token, out, dim);
    return cuda_status(cudaGetLastError());
}

__device__ void rope_pair(float *hvec, int i, int half, int pos, float theta) {
    const float freq = det_powf(theta, -static_cast<float>(i) / static_cast<float>(half));
    det_rotate(hvec[2 * i], hvec[2 * i + 1], __fmul_rn(static_cast<float>(pos), freq), 1);
}

__global__ void rope_kernel(float *x, int n_heads, int head_dim, int pos, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total = n_heads * half;
    if (idx >= total || (head_dim % 2) != 0) {
        return;
    }
    rope_pair(x + (idx / half) * head_dim, idx % half, half, pos, theta);
}

extern "C" int allpaca_cuda_rope(float *x, int32_t n_heads, int32_t head_dim, int32_t pos,
                                 float theta, allpaca_cuda_stream_t stream) {
    if (!x || n_heads <= 0 || head_dim <= 0 || (head_dim % 2) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    int total = n_heads * (head_dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rope_kernel<<<blocks, threads, 0, as_stream(stream)>>>(x, n_heads, head_dim, pos, theta);
    return cuda_status(cudaGetLastError());
}

__global__ void rope_qk_kernel(float *q, float *k, int n_q_heads, int n_k_heads, int head_dim,
                               int q_stride, int k_stride, int start_pos, float theta) {
    const int tok = static_cast<int>(blockIdx.y);
    const int pos = start_pos + tok;
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int half = head_dim / 2;
    const int q_pairs = n_q_heads * half;
    const int total = q_pairs + n_k_heads * half;
    if (idx >= total || head_dim % 2 != 0) {
        return;
    }
    q += static_cast<size_t>(tok) * static_cast<size_t>(q_stride);
    k += static_cast<size_t>(tok) * static_cast<size_t>(k_stride);
    if (idx < q_pairs) {
        rope_pair(q + (idx / half) * head_dim, idx % half, half, pos, theta);
    } else {
        const int j = idx - q_pairs;
        rope_pair(k + (j / half) * head_dim, j % half, half, pos, theta);
    }
}

extern "C" int allpaca_cuda_rope_qk_rows(float *q, float *k, int32_t n_tok, int32_t n_q_heads,
                                         int32_t n_k_heads, int32_t head_dim, int32_t q_stride,
                                         int32_t k_stride, int32_t start_pos, float theta,
                                         allpaca_cuda_stream_t stream) {
    if (!q || !k || n_tok <= 0 || n_q_heads <= 0 || n_k_heads <= 0 || head_dim <= 0 ||
        q_stride <= 0 || k_stride <= 0 || (head_dim % 2) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    const int total = (n_q_heads + n_k_heads) * (head_dim / 2);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    rope_qk_kernel<<<dim3(blocks, n_tok), threads, 0, as_stream(stream)>>>(
        q, k, n_q_heads, n_k_heads, head_dim, q_stride, k_stride, start_pos, theta);
    return cuda_status(cudaGetLastError());
}

extern "C" int allpaca_cuda_rope_qk(float *q, float *k, int32_t n_q_heads, int32_t n_k_heads,
                                    int32_t head_dim, int32_t q_stride, int32_t k_stride, int32_t pos,
                                    float theta, allpaca_cuda_stream_t stream) {
    return allpaca_cuda_rope_qk_rows(q, k, 1, n_q_heads, n_k_heads, head_dim, q_stride, k_stride,
                                     pos, theta, stream);
}

/* NeoX pair on first n_rot dims; optional IMRoPE sector angles (CPU rope_imrope_neox). */
__device__ void rope_imrope_pair(float *hvec, int i, int half, int n_rot, int pos, float theta,
                                int s0, int s1, int s2, int s3) {
    const int sect_dims = s0 + s1 + s2 + s3;
    const bool use_imrope = sect_dims > 0 && s0 > 0 && s1 > 0;
    const float theta_scale = det_powf(theta, -2.0f / static_cast<float>(n_rot));
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
    det_rotate(hvec[i], hvec[i + half], angle, 1);
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

extern "C" int allpaca_cuda_rope_imrope_qk(float *q, float *k, int32_t n_q_heads, int32_t n_k_heads,
                                           int32_t head_dim, int32_t n_rot, int32_t pos, float theta,
                                           const int32_t sections[4], allpaca_cuda_stream_t stream) {
    if (!q || !k || !sections || n_q_heads <= 0 || n_k_heads <= 0 || head_dim <= 0 || n_rot <= 0 ||
        n_rot > head_dim || (n_rot & 1) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    const int half = n_rot / 2;
    const int total = (n_q_heads + n_k_heads) * half;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    rope_imrope_qk_kernel<<<blocks, threads, 0, as_stream(stream)>>>(
        q, k, n_q_heads, n_k_heads, head_dim, n_rot, pos, theta, sections[0], sections[1],
        sections[2], sections[3]);
    return cuda_status(cudaGetLastError());
}

/* Decode/prefill attention, a port of the HIP kernels in hip/src/kernels.cpp:
 * one 32-lane warp owns one (query, head) row and walks its keys with an
 * online softmax, ATTN_KT keys a step (attn_tile_step). The tiles, chunks,
 * reductions and rounding are HIP's, so a row gets the same bits on both
 * backends, whichever kernel computed it and however many rows it ran with.
 * Change one side and the other together; `dense_digest` in allpaca-rpc
 * compares them. nvcc has no `fp contract(off)`, so every operation here that
 * could be fused is written out. */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define ATTN_SHFL_XOR(v, m) __shfl_xor((v), (m), 32)
#define ATTN_SHFL(v, l) __shfl((v), (l), 32)
#else
#define ATTN_SHFL_XOR(v, m) __shfl_xor_sync(0xffffffffu, (v), (m), 32)
#define ATTN_SHFL(v, l) __shfl_sync(0xffffffffu, (v), (l), 32)
#endif

constexpr int ATTN_MAX_DPT = 16;

/* head_dim / 32, rounded up to the buckets the kernels are built for. Past
 * 32 * ATTN_MAX_DPT only the first 512 dims would be read: no supported model
 * has such a head. */
static constexpr int attn_dpt_bucket(int head_dim) {
    const int dpt = (head_dim + 31) / 32;
    return dpt <= 4 ? 4 : dpt <= 8 ? 8 : 16;
}

template <int DPT>
__host__ __device__ constexpr int attn_row_dims() {
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
        r.qreg[i] = 0.f;
        r.acc[i] = 0.f;
    }
    if (vec4) {
#pragma unroll
        for (int j = 0; j < attn_row_dims<DPT>() / 4; ++j) {
            const int v = lane + 32 * j;
            if (v < nvec) {
                r.qreg[4 * j] = qh[v * 4];
                r.qreg[4 * j + 1] = qh[v * 4 + 1];
                r.qreg[4 * j + 2] = qh[v * 4 + 2];
                r.qreg[4 * j + 3] = qh[v * 4 + 3];
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
    r.l = 0.f;
    if (sinks) {
        r.m = sinks[h];
        r.l = 1.f;
    }
}

/* Keys per attn_tile_step. */
constexpr int ATTN_KT = 16;

/* One stage of the transposing reduction in attn_tile_step: lanes `m` apart
 * swap halves of their `2*H` key partials and add what they keep to what they
 * receive. */
template <int H>
__device__ __forceinline__ void attn_keys_stage(float (&p)[ATTN_KT], int lane, int m) {
#pragma unroll
    for (int i = 0; i < H; ++i) {
        const bool up = (lane & m) != 0;
        const float send = up ? p[i] : p[i + H];
        const float keep = up ? p[i + H] : p[i];
        p[i] = __fadd_rn(keep, ATTN_SHFL_XOR(send, m));
    }
}

/* ATTN_KT keys at once, each key's head row `ks` floats after the last; keys
 * from `n_read` on are not read (they are past the row's end, `valid` is false
 * for them, and their score is dropped). A transposing reduction leaves key
 * lane/2's score on lanes 2j and 2j+1; one max, one pair of exps and one sum
 * serve all the keys, and the values accumulate in key order. */
template <int DPT>
__device__ __forceinline__ void attn_tile_step(AttnRow<DPT> &r, const float *kt, const float *vt, size_t ks,
                                               int n_read, bool valid, int lane, int head_dim, int nvec,
                                               bool vec4, float scale, float softcap) {
    float part[ATTN_KT];
#pragma unroll
    for (int j = 0; j < ATTN_KT; ++j) {
        const bool read = j < n_read;
        const float *kh = kt + static_cast<size_t>(j) * ks;
        float partial = 0.f;
        if (vec4) {
#pragma unroll
            for (int g = 0; g < attn_row_dims<DPT>() / 4; ++g) {
                const int vec = lane + 32 * g;
                if (vec < nvec) {
                    const float *kv = kh + vec * 4;
                    float d4 = __fmul_rn(r.qreg[4 * g], read ? kv[0] : 0.f);
                    d4 = fmaf(r.qreg[4 * g + 1], read ? kv[1] : 0.f, d4);
                    d4 = fmaf(r.qreg[4 * g + 2], read ? kv[2] : 0.f, d4);
                    d4 = fmaf(r.qreg[4 * g + 3], read ? kv[3] : 0.f, d4);
                    partial = __fadd_rn(partial, d4);
                }
            }
        } else {
#pragma unroll
            for (int g = 0; g < DPT; ++g) {
                const int d = lane + 32 * g;
                if (d < head_dim) {
                    partial = fmaf(r.qreg[g], read ? kh[d] : 0.f, partial);
                }
            }
        }
        part[j] = partial;
    }
    attn_keys_stage<8>(part, lane, 16);
    attn_keys_stage<4>(part, lane, 8);
    attn_keys_stage<2>(part, lane, 4);
    attn_keys_stage<1>(part, lane, 2);
    float s = __fmul_rn(__fadd_rn(part[0], ATTN_SHFL_XOR(part[0], 1)), scale);
    if (softcap > 0.f) {
        s = __fmul_rn(tanhf(s / softcap), softcap);
    }
    if (!valid) {
        s = -1.0e30f;
    }
    float top = s;
#pragma unroll
    for (int m = 2; m < 32; m <<= 1) {
        top = fmaxf(top, ATTN_SHFL_XOR(top, m));
    }
    const float m_new = fmaxf(r.m, top);
    const float alpha = det_expf(r.m - m_new);
    const float p = valid ? det_expf(s - m_new) : 0.f;
    float sum = p;
#pragma unroll
    for (int m = 2; m < 32; m <<= 1) {
        sum = __fadd_rn(sum, ATTN_SHFL_XOR(sum, m));
    }
    r.l = fmaf(r.l, alpha, sum);
#pragma unroll
    for (int i = 0; i < attn_row_dims<DPT>(); ++i) {
        r.acc[i] = __fmul_rn(r.acc[i], alpha);
    }
#pragma unroll
    for (int j = 0; j < ATTN_KT; ++j) {
        const float pj = ATTN_SHFL(p, 2 * j);
        if (pj == 0.f) {
            continue;
        }
        const float *vh = vt + static_cast<size_t>(j) * ks;
        if (vec4) {
#pragma unroll
            for (int g = 0; g < attn_row_dims<DPT>() / 4; ++g) {
                const int vec = lane + 32 * g;
                if (vec < nvec) {
                    const float *vv = vh + vec * 4;
                    r.acc[4 * g] = fmaf(pj, vv[0], r.acc[4 * g]);
                    r.acc[4 * g + 1] = fmaf(pj, vv[1], r.acc[4 * g + 1]);
                    r.acc[4 * g + 2] = fmaf(pj, vv[2], r.acc[4 * g + 2]);
                    r.acc[4 * g + 3] = fmaf(pj, vv[3], r.acc[4 * g + 3]);
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
    const float inv = (r.l == 0.f) ? 0.f : __fdiv_rn(1.f, r.l);
    if (vec4) {
#pragma unroll
        for (int j = 0; j < attn_row_dims<DPT>() / 4; ++j) {
            const int vec = lane + 32 * j;
            if (vec < nvec) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    if (vec * 4 + e < head_dim) {
                        oh[vec * 4 + e] = __fmul_rn(r.acc[4 * j + e], inv);
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < DPT; ++j) {
            const int d = lane + 32 * j;
            if (d < head_dim) {
                oh[d] = __fmul_rn(r.acc[j], inv);
            }
        }
    }
}

/* Last key row `qi` may see: causal rows are bottom-right aligned. */
__device__ __forceinline__ int attn_row_tmax(int qi, int n_q, int n_kv, int causal) {
    return causal ? qi + (n_kv > n_q ? n_kv - n_q : 0) : n_kv - 1;
}

/* A row with more than ATTN_CHUNK keys: the keys split into chunks, each an
 * online softmax from an empty state (sinks in chunk 0 only), folded by
 * attn_fold in one fixed tree, the same as HIP's (see ATTN_FOLD_STACK). A row
 * with one chunk is the single-pass result. */
constexpr int ATTN_CHUNK = 128;
static_assert(ATTN_CHUNK % ATTN_KT == 0, "chunks are whole key tiles");
/* The fold tree: at stride s = 1, 2, 4, ... the partial at left + s folds into
 * the one at left, for every left that is a multiple of 2s with left + s <
 * n_chunks. The Q-tiled kernel builds the same tree with a carry stack: push a
 * chunk, fold while the top two spans are the same size, then fold what is
 * left right to left. The stack proper (blocks past the register levels)
 * holds at most floor(log2(n_chunks >> reg)) + 2 partials: 16 entries cover
 * 2^16 chunks (8M keys); past that the kernel traps. */
constexpr int ATTN_FOLD_STACK = 16;

/* Floats per (row, head, chunk) partial: every lane's accumulators, m, l. */
template <int DPT>
__host__ __device__ constexpr int attn_part() {
    return 32 * attn_row_dims<DPT>() + 2;
}

template <int DPT>
__device__ __forceinline__ void attn_row_reset(AttnRow<DPT> &r) {
#pragma unroll
    for (int i = 0; i < attn_row_dims<DPT>(); ++i) {
        r.acc[i] = 0.f;
    }
    r.m = -1.0e30f;
    r.l = 0.f;
}

template <int DPT>
__device__ __forceinline__ void attn_fold(float &m, float &l, float *acc, float m_b, float l_b,
                                          const float *acc_b) {
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
        __trap();
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

/* Copies keys [t0, t0 + tile) of key/value head `kv_h` into shared memory,
 * zeros from `t_lim` on. */
__device__ __forceinline__ void attn_stage_tile(float *k_lds, float *v_lds, const float *k, const float *v,
                                                int t0, int t_lim, int tile, int head_dim, size_t kv_stride,
                                                int kv_h) {
    const size_t base = static_cast<size_t>(kv_h) * head_dim;
    const int n = tile * head_dim;
    for (int e = static_cast<int>(threadIdx.x); e < n; e += static_cast<int>(blockDim.x)) {
        const int t = t0 + e / head_dim;
        const size_t src = static_cast<size_t>(t) * kv_stride + base + static_cast<size_t>(e % head_dim);
        k_lds[e] = t < t_lim ? k[src] : 0.f;
        v_lds[e] = t < t_lim ? v[src] : 0.f;
    }
}

/* ATTN_QTILE consecutive queries of one head per block, one warp each. With
 * `tile` > 0 the block stages `tile` keys of K and V in shared memory and every
 * warp reads them from there; with 0 (the tile would not fit, NVIDIA has 48 KiB
 * to AMD's 64) the warps read global memory. Same arithmetic either way. */
constexpr int ATTN_QTILE = 8;

template <int DPT, bool VEC4>
__global__ void __launch_bounds__(32 * ATTN_QTILE)
    attention_qtile_kernel(const float *q, const float *k, const float *v, float *out, int n_q,
                           int n_kv, int n_heads, int n_kv_heads, int head_dim, int causal,
                           const uint8_t *mask, const float *sinks, float softcap, int tile) {
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
    constexpr bool vec4 = VEC4;
    const int nvec = vec4 ? head_dim / 4 : head_dim;
    const int t_max = attn_row_tmax(qi, n_q, n_kv, causal);
    const int t_end = active ? (mask ? n_kv : min(n_kv, t_max + 1)) : 0;
    const int q_last = min(n_q, q0 + ATTN_QTILE) - 1;
    const int block_end = mask ? n_kv : min(n_kv, attn_row_tmax(q_last, n_q, n_kv, causal) + 1);
    const int step = tile > 0 ? tile : ATTN_KT;
    AttnRow<DPT> r;
    attn_row_init(r, qh, lane, head_dim, nvec, vec4, sinks, h);
    /* Finished chunks before the current one, as the fold tree's carry stack. */
    AttnFoldRegs<DPT> rg;
    AttnFoldStack<DPT> st;
    attn_tree_init(rg, st);
    int next_fold = ATTN_CHUNK;
    for (int t0 = 0; t0 < block_end; t0 += step) {
        if (tile > 0) {
            __syncthreads();
            attn_stage_tile(k_lds, v_lds, k, v, t0, n_kv, tile, head_dim, kv_stride, kv_h);
            __syncthreads();
        }
        const int stop = min(t_end, t0 + step);
        for (int s = t0; s < stop; s += ATTN_KT) {
            if (s == next_fold) {
                next_fold += ATTN_CHUNK;
                attn_tree_push<DPT>(rg, st, r);
                attn_row_reset(r);
            }
            const int t = s + lane / 2;
            const bool valid = t < t_end && (!mask || mask[static_cast<size_t>(qi) * static_cast<size_t>(n_kv) +
                                                            static_cast<size_t>(t)] != 0);
            const size_t base = static_cast<size_t>(s) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
            const size_t at = static_cast<size_t>(s - t0) * head_dim;
            attn_tile_step(r, tile > 0 ? k_lds + at : k + base, tile > 0 ? v_lds + at : v + base,
                           tile > 0 ? static_cast<size_t>(head_dim) : kv_stride,
                           tile > 0 ? ATTN_KT : n_kv - s, valid, lane, head_dim, nvec, vec4, scale, softcap);
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

/* One query, one chunk of keys per block: up to ATTN_GQA_HEADS query heads of
 * one key/value head, each key staged once per block (`tile` as in the Q-tiled
 * kernel). With several chunks the chunk's softmax goes to `part` for
 * attention_merge_kernel; with one, the row goes to `out`. */
template <int DPT, bool VEC4>
__global__ void __launch_bounds__(32 * ATTN_GQA_HEADS)
    attention_split_gqa_kernel(const float *q, const float *k, const float *v, float *part, float *out,
                               int n_kv, int n_heads, int n_kv_heads, int head_dim, int causal,
                               const uint8_t *mask, const float *sinks, float softcap, int n_chunks,
                               int tile) {
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
    constexpr bool vec4 = VEC4;
    const int nvec = vec4 ? head_dim / 4 : head_dim;
    const int step = tile > 0 ? tile : ATTN_KT;
    AttnRow<DPT> r;
    attn_row_init(r, qh, lane, head_dim, nvec, vec4, c == 0 ? sinks : nullptr, active ? h : h0);
    const int t_end = mask ? n_kv : min(n_kv, t_max + 1);
    const int t1 = min(t_end, (c + 1) * ATTN_CHUNK);
    for (int t0 = c * ATTN_CHUNK; t0 < t1; t0 += step) {
        if (tile > 0) {
            __syncthreads();
            attn_stage_tile(k_lds, v_lds, k, v, t0, t1, tile, head_dim, kv_stride, kv_h);
            __syncthreads();
        }
        const int stop = min(t1, t0 + step);
        for (int s = t0; s < stop; s += ATTN_KT) {
            const int t = s + lane / 2;
            const bool valid = t < t1 && (!mask || mask[t] != 0);
            const size_t base = static_cast<size_t>(s) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
            const size_t at = static_cast<size_t>(s - t0) * head_dim;
            attn_tile_step(r, tile > 0 ? k_lds + at : k + base, tile > 0 ? v_lds + at : v + base,
                           tile > 0 ? static_cast<size_t>(head_dim) : kv_stride,
                           tile > 0 ? ATTN_KT : t1 - s, valid, lane, head_dim, nvec, vec4, scale, softcap);
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
__global__ void __launch_bounds__(32)
    attention_merge_kernel(const float *q, float *part, float *out, int n_heads, int head_dim,
                           int n_chunks) {
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
            __syncwarp();
            if (lane == 0) {
                dst[32 * dims] = m;
                dst[32 * dims + 1] = l;
            }
            __syncwarp();
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

constexpr int ATTN_PART_DEVICES = 16;
struct AttnPartScratch {
    std::mutex mu;
    float *p = nullptr;
    size_t cap = 0;
    cudaStream_t stream = nullptr;
};
static AttnPartScratch g_attn_part[ATTN_PART_DEVICES];

/* NVIDIA's shared memory per block. */
constexpr size_t ATTN_MAX_LDS = 48 * 1024;

template <int DPT, bool VEC4>
static void launch_attention_tiled(const float *q, const float *k, const float *v, float *out, int n_q,
                                   int n_kv, int n_heads, int n_kv_heads, int head_dim, int causal,
                                   const uint8_t *mask, const float *sinks, float softcap,
                                   cudaStream_t stream) {
    /* About 2048 floats of K (and of V) per tile, at least one key tile, as on
     * HIP; a tile that doesn't fit is read from global memory. */
    int tile = max(ATTN_KT, 2048 / head_dim / ATTN_KT * ATTN_KT);
    size_t lds = 2 * static_cast<size_t>(tile) * head_dim * sizeof(float);
    if (lds > ATTN_MAX_LDS) {
        tile = 0;
        lds = 0;
    }
    const int n_chunks = n_kv > ATTN_CHUNK ? (n_kv + ATTN_CHUNK - 1) / ATTN_CHUNK : 1;
    const int per_kv = n_kv_heads > 0 ? n_heads / n_kv_heads : 1;
    const int per_block = min(per_kv, ATTN_GQA_HEADS);
    const int groups = (per_kv + per_block - 1) / per_block;
    const dim3 split_grid(static_cast<unsigned>(n_kv_heads * groups), static_cast<unsigned>(n_chunks));
    const dim3 split_block(static_cast<unsigned>(32 * per_block));
    if (n_q == 1 && n_chunks == 1) {
        attention_split_gqa_kernel<DPT, VEC4><<<split_grid, split_block, lds, stream>>>(
            q, k, v, static_cast<float *>(nullptr), out, n_kv, n_heads, n_kv_heads, head_dim, causal, mask,
            sinks, softcap, n_chunks, tile);
        return;
    }
    if (n_q == 1) {
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < ATTN_PART_DEVICES) {
            AttnPartScratch &s = g_attn_part[dev];
            std::lock_guard<std::mutex> lock(s.mu);
            bool ok = !(s.p && s.stream != stream && cudaDeviceSynchronize() != cudaSuccess);
            const size_t need = static_cast<size_t>(n_heads) * n_chunks * attn_part<DPT>();
            if (ok && s.cap < need) {
                if (s.p) {
                    (void)cudaDeviceSynchronize();
                    (void)cudaFree(s.p);
                    s.p = nullptr;
                    s.cap = 0;
                }
                const size_t cap = need > (size_t(1) << 18) ? need : (size_t(1) << 18);
                if (cudaMalloc(&s.p, cap * sizeof(float)) == cudaSuccess) {
                    s.cap = cap;
                } else {
                    s.p = nullptr;
                    (void)cudaGetLastError();
                    ok = false;
                }
            }
            if (ok) {
                s.stream = stream;
                attention_split_gqa_kernel<DPT, VEC4><<<split_grid, split_block, lds, stream>>>(
                    q, k, v, s.p, static_cast<float *>(nullptr), n_kv, n_heads, n_kv_heads, head_dim, causal,
                    mask, sinks, softcap, n_chunks, tile);
                /* The fold tree's lowest levels (see attn_merge_spacing). */
                const int spacing = attn_merge_spacing(n_chunks);
                for (int stride = 1; stride < spacing; stride *= 2) {
                    const int pairs = (n_chunks - stride + 2 * stride - 1) / (2 * stride);
                    attention_merge_pair_kernel<DPT>
                        <<<dim3(static_cast<unsigned>(pairs), static_cast<unsigned>(n_heads)), 32, 0, stream>>>(
                            s.p, n_chunks, stride);
                }
                attention_merge_kernel<DPT><<<dim3(1, static_cast<unsigned>(n_heads)), 32, 0, stream>>>(
                    q, s.p, out, n_heads, head_dim, n_chunks);
                return;
            }
        }
    }
    const dim3 block(32 * ATTN_QTILE);
    const dim3 grid(static_cast<unsigned>((n_q + ATTN_QTILE - 1) / ATTN_QTILE),
                    static_cast<unsigned>(n_heads));
    attention_qtile_kernel<DPT, VEC4><<<grid, block, lds, stream>>>(q, k, v, out, n_q, n_kv, n_heads,
                                                                    n_kv_heads, head_dim, causal, mask,
                                                                    sinks, softcap, tile);
}

template <int DPT>
static void launch_attention_bucket(const float *q, const float *k, const float *v, float *out, int n_q,
                                    int n_kv, int n_heads, int n_kv_heads, int head_dim, int causal,
                                    const uint8_t *mask, const float *sinks, float softcap,
                                    cudaStream_t stream) {
    /* float4 rows on HIP; the arithmetic order follows it, so this must be the
     * same test. The same for every row of the call. */
    if ((head_dim % 4) == 0 && (reinterpret_cast<uintptr_t>(q) % 16u) == 0) {
        launch_attention_tiled<DPT, true>(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
                                          mask, sinks, softcap, stream);
    } else {
        launch_attention_tiled<DPT, false>(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
                                           mask, sinks, softcap, stream);
    }
}

/* One query: the split kernel, its chunks in parallel plus a merge when the row
 * is longer than one chunk. Several rows: the Q-tiled kernel. A row gets the
 * same bits from either. */
static void launch_attention(const float *q, const float *k, const float *v, float *out, int n_q,
                             int n_kv, int n_heads, int n_kv_heads, int head_dim, int causal,
                             const uint8_t *mask, const float *sinks, float softcap,
                             cudaStream_t stream) {
    switch (attn_dpt_bucket(head_dim)) {
    case 4:
        launch_attention_bucket<4>(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, mask,
                                   sinks, softcap, stream);
        return;
    case 8:
        launch_attention_bucket<8>(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, mask,
                                   sinks, softcap, stream);
        return;
    default:
        launch_attention_bucket<16>(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, mask,
                                    sinks, softcap, stream);
        return;
    }
}

extern "C" int allpaca_cuda_attention_sinks(const float *q, const float *k, const float *v,
                                            float *out, int32_t n_q, int32_t n_kv, int32_t n_heads,
                                            int32_t n_kv_heads, int32_t head_dim, int32_t causal,
                                            const float *sinks, allpaca_cuda_stream_t stream) {
    if (!q || !k || !v || !out || n_q <= 0 || n_kv <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_dim <= 0 || (n_heads % n_kv_heads) != 0 || head_dim > 512) {
        return ALLPACA_ERR_INVALID;
    }
    launch_attention(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, nullptr,
                     sinks, 0.f, as_stream(stream));
    return cuda_status(cudaGetLastError());
}

extern "C" int allpaca_cuda_attention_masked(const float *q, const float *k, const float *v,
                                             float *out, int32_t n_q, int32_t n_kv, int32_t n_heads,
                                             int32_t n_kv_heads, int32_t head_dim,
                                             const uint8_t *mask, allpaca_cuda_stream_t stream) {
    if (!q || !k || !v || !out || !mask || n_q <= 0 || n_kv <= 0 || n_heads <= 0 ||
        n_kv_heads <= 0 || head_dim <= 0 || (n_heads % n_kv_heads) != 0 || head_dim > 512) {
        return ALLPACA_ERR_INVALID;
    }
    launch_attention(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, 0, mask,
                     nullptr, 0.f, as_stream(stream));
    return cuda_status(cudaGetLastError());
}

extern "C" int allpaca_cuda_attention(const float *q, const float *k, const float *v, float *out,
                                      int32_t n_q, int32_t n_kv, int32_t n_heads, int32_t n_kv_heads,
                                      int32_t head_dim, int32_t causal,
                                      allpaca_cuda_stream_t stream) {
    return allpaca_cuda_attention_sinks(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim,
                                        causal, nullptr, stream);
}

extern "C" int allpaca_cuda_attention_softcap(const float *q, const float *k, const float *v,
                                              float *out, int32_t n_q, int32_t n_kv, int32_t n_heads,
                                              int32_t n_kv_heads, int32_t head_dim, int32_t causal,
                                              float softcap, allpaca_cuda_stream_t stream) {
    if (!q || !k || !v || !out || n_q <= 0 || n_kv <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_dim <= 0 || (n_heads % n_kv_heads) != 0 || head_dim > 512) {
        return ALLPACA_ERR_INVALID;
    }
    if (softcap <= 0.f) {
        return allpaca_cuda_attention(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
                                      stream);
    }
    launch_attention(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, nullptr,
                     nullptr, softcap, as_stream(stream));
    return cuda_status(cudaGetLastError());
}

extern "C" int allpaca_cuda_attention_masked_softcap(const float *q, const float *k, const float *v,
                                                     float *out, int32_t n_q, int32_t n_kv,
                                                     int32_t n_heads, int32_t n_kv_heads,
                                                     int32_t head_dim, const uint8_t *mask,
                                                     float softcap, allpaca_cuda_stream_t stream) {
    if (!q || !k || !v || !out || !mask || n_q <= 0 || n_kv <= 0 || n_heads <= 0 ||
        n_kv_heads <= 0 || head_dim <= 0 || (n_heads % n_kv_heads) != 0 || head_dim > 512) {
        return ALLPACA_ERR_INVALID;
    }
    if (softcap <= 0.f) {
        return allpaca_cuda_attention_masked(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim,
                                             mask, stream);
    }
    launch_attention(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, 0, mask,
                     nullptr, softcap, as_stream(stream));
    return cuda_status(cudaGetLastError());
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
    x[idx] = det_sigmoid_mul(x[idx], g);
}

extern "C" int allpaca_cuda_deinterleave_q_gate(const float *q_full, float *q, float *gate,
                                                int32_t n_heads, int32_t head_dim,
                                                allpaca_cuda_stream_t stream) {
    if (!q_full || !q || !gate || n_heads <= 0 || head_dim <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    const int n = n_heads * head_dim;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    deinterleave_q_gate_kernel<<<static_cast<unsigned>(blocks), static_cast<unsigned>(threads), 0,
                                 as_stream(stream)>>>(q_full, q, gate, n_heads, head_dim);
    return cuda_status(cudaGetLastError());
}

extern "C" int allpaca_cuda_sigmoid_mul(float *x, const float *gate, int32_t n,
                                        allpaca_cuda_stream_t stream) {
    if (!x || !gate || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    sigmoid_mul_kernel<<<static_cast<unsigned>(blocks), static_cast<unsigned>(threads), 0,
                         as_stream(stream)>>>(x, gate, n);
    return cuda_status(cudaGetLastError());
}

/* ---------------------------------------------------------------------------
 * Int8-activation K-quant GEMV and the fused MoE FFN (Q4_K / Q5_K / Q6_K),
 * ported from hip/src/kernels.cpp (`kquant_i8_block`, `allpaca_moe_ffn_kquant`).
 * Activations are quantized per 32 values to int8 with one f32 scale, and
 * every row is a signed dot4 against the unpacked weight bytes. The same
 * block routine serves every m, so a verify row reproduces the decode row.
 * ------------------------------------------------------------------------- */

#include <mutex>

__device__ inline int32_t kq_dot4(uint32_t a, uint32_t b, int32_t c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(static_cast<int>(a), static_cast<int>(b), c);
#else
    const int8_t *x = reinterpret_cast<const int8_t *>(&a);
    const int8_t *y = reinterpret_cast<const int8_t *>(&b);
    return c + x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
#endif
}

__device__ inline float kq_f16le(const uint8_t *p) {
    const uint16_t h = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u) << 16);
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            const float f = (static_cast<float>(mant) / 1024.0f) * (1.0f / 16384.0f);
            return (h & 0x8000u) ? -f : f;
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return __uint_as_float(bits);
}

template <int QT>
struct KqBytes;
template <>
struct KqBytes<4> {
    static constexpr int bytes = 144;
};
template <>
struct KqBytes<5> {
    static constexpr int bytes = 176;
};
template <>
struct KqBytes<6> {
    static constexpr int bytes = 210;
};

__global__ void kq_quantize_rows_kernel(const float *a, int8_t *aq, float *ad, int m, int k) {
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
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 32));
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

__device__ inline void kq_scale_min_pair(const uint8_t *scales, int g, uint32_t &sc1, uint32_t &m1,
                                         uint32_t &sc2, uint32_t &m2) {
    const uint32_t w0 = *reinterpret_cast<const uint32_t *>(scales);
    const uint32_t w1 = *reinterpret_cast<const uint32_t *>(scales + 4);
    const uint32_t w2 = *reinterpret_cast<const uint32_t *>(scales + 8);
    const int j1 = 2 * g;
    const int j2 = 2 * g + 1;
    auto byte = [](uint32_t w, int i) { return (w >> (8 * (i & 3))) & 0xFFu; };
    const bool hi = g >= 2;
    const uint32_t a1 = byte(w0, j1), b1 = byte(w1, j1), c1 = byte(w2, j1);
    const uint32_t a2 = byte(w0, j2), b2 = byte(w1, j2), c2 = byte(w2, j2);
    sc1 = hi ? ((c1 & 15) | ((a1 >> 6) << 4)) : (a1 & 63);
    m1 = hi ? ((c1 >> 4) | ((b1 >> 6) << 4)) : (b1 & 63);
    sc2 = hi ? ((c2 & 15) | ((a2 >> 6) << 4)) : (a2 & 63);
    m2 = hi ? ((c2 >> 4) | ((b2 >> 6) << 4)) : (b2 & 63);
}

/* One block of the int8 K-quant GEMV: columns from `bx`, rows `row0 .. +MR`.
 * LPC lanes per column, W = 128 / LPC weights per lane in each of two places
 * (see the HIP kernel for the lane maps). */
template <int QT, int MR, int LPC>
__device__ inline void kq_i8_block(const int8_t *aq, const float *ad, const uint8_t *b, float *c,
                                   int m, int n, int k, int bx, int row0) {
    constexpr int BB = KqBytes<QT>::bytes;
    constexpr int W = 128 / LPC;
    constexpr int NC = W / 4;
    constexpr int LPG = LPC / 4;
    constexpr uint32_t ONES = 0x01010101u;
    const int sl = static_cast<int>(threadIdx.x) % LPC;
    const int col = bx * (static_cast<int>(blockDim.x) / LPC) + static_cast<int>(threadIdx.x) / LPC;
    const bool live = col < n;
    const int nblocks = k / 256;
    const int kb = k / 32;
    const uint8_t *brow = b + static_cast<size_t>(live ? col : 0) * static_cast<size_t>(nblocks) * BB;
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
            if constexpr (QT == 6) {
                const float d = kq_f16le(blk + 208);
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
                            dot = kq_dot4(av[t], q[t], dot);
                            sum = kq_dot4(av[t], ONES, sum);
                        }
                        acc[r] += s * drow[r][bi] * static_cast<float>(dot - 32 * sum);
                    }
                }
            } else {
                const float d = kq_f16le(blk);
                const float dmin = kq_f16le(blk + 2);
                uint32_t sc1 = 0, m1 = 0, sc2 = 0, m2 = 0;
                kq_scale_min_pair(blk + 4, g, sc1, m1, sc2, m2);
                const uint32_t *qp =
                    reinterpret_cast<const uint32_t *>(blk + (QT == 5 ? 48 : 16) + g * 32 + j0);
                uint32_t lo[NC], hi[NC];
#pragma unroll
                for (int t = 0; t < NC; ++t) {
                    const uint32_t qv = qp[t];
                    uint32_t hb_lo = 0, hb_hi = 0;
                    if constexpr (QT == 5) {
                        const uint32_t qh = *reinterpret_cast<const uint32_t *>(blk + 16 + j0 + 4 * t);
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
                const int bl = ib * 8 + 2 * g;
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    uint32_t alo[NC], ahi[NC];
                    kq_load_u32s<NC>(arow[r] + bl * 32 + j0, alo);
                    kq_load_u32s<NC>(arow[r] + bl * 32 + 32 + j0, ahi);
                    int32_t il = 0, sl_ = 0, ih = 0, sh = 0;
#pragma unroll
                    for (int t = 0; t < NC; ++t) {
                        il = kq_dot4(alo[t], lo[t], il);
                        sl_ = kq_dot4(alo[t], ONES, sl_);
                        ih = kq_dot4(ahi[t], hi[t], ih);
                        sh = kq_dot4(ahi[t], ONES, sh);
                    }
                    acc[r] += drow[r][bl] * (dl * static_cast<float>(il) - ml * static_cast<float>(sl_)) +
                              drow[r][bl + 1] * (dh * static_cast<float>(ih) - mh * static_cast<float>(sh));
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = LPC / 2; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off, LPC);
        }
        if (live && sl == 0 && row0 + r < m) {
            c[static_cast<size_t>(row0 + r) * static_cast<size_t>(n) + static_cast<size_t>(col)] = v;
        }
    }
}

template <int QT, int MR, int LPC>
__global__ void kq_i8_gemv_kernel(const int8_t *aq, const float *ad, const uint8_t *b, float *c,
                                  int m, int n, int k) {
    kq_i8_block<QT, MR, LPC>(aq, ad, b, c, m, n, k, static_cast<int>(blockIdx.x),
                             static_cast<int>(blockIdx.y) * MR);
}

/* Per-device int8 activation scratch, grow-only. The lock is held from
 * quantize through GEMV enqueue. On a switch to another stream the device is
 * synchronized first: the previous stream may already be destroyed (an RPC
 * client's workspace goes away when it disconnects), so it cannot take an
 * event record. Switches are rare (a new client or engine). */
constexpr int KQ_DEVICES = 16;
struct KqScratch {
    std::mutex mu;
    void *p = nullptr;
    size_t cap = 0;
    cudaStream_t stream = nullptr;
};
static KqScratch g_kq[KQ_DEVICES];

static KqScratch *kq_acquire(cudaStream_t stream, size_t need, std::unique_lock<std::mutex> &lock) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= KQ_DEVICES) {
        return nullptr;
    }
    KqScratch &s = g_kq[dev];
    lock = std::unique_lock<std::mutex>(s.mu);
    if (s.p && s.stream != stream && cudaDeviceSynchronize() != cudaSuccess) {
        return nullptr;
    }
    if (s.cap < need) {
        if (s.p) {
            (void)cudaStreamSynchronize(stream);
            (void)cudaFree(s.p);
            s.p = nullptr;
            s.cap = 0;
        }
        const size_t cap = need > (size_t(1) << 20) ? need : (size_t(1) << 20);
        if (cudaMalloc(&s.p, cap) != cudaSuccess) {
            s.p = nullptr;
            return nullptr;
        }
        s.cap = cap;
    }
    s.stream = stream;
    return &s;
}

template <int QT>
static int kq_gemm_launch(const float *a, const uint8_t *b, float *c, int32_t m, int32_t n, int32_t k,
                          cudaStream_t stream) {
    const uintptr_t ba = reinterpret_cast<uintptr_t>(b);
    if (m <= 0 || n <= 0 || k <= 0 || k % 256 != 0 || !a || !b || !c ||
        (QT == 6 ? ba % 2 != 0 : ba % 4 != 0)) {
        return ALLPACA_ERR_INVALID;
    }
    (void)cudaGetLastError();
    const size_t aq_bytes = (static_cast<size_t>(m) * k + 255) & ~size_t(255);
    const size_t need = aq_bytes + static_cast<size_t>(m) * (k / 32) * sizeof(float);
    std::unique_lock<std::mutex> lock;
    KqScratch *scratch = kq_acquire(stream, need, lock);
    if (!scratch) {
        return ALLPACA_ERR_DEVICE;
    }
    uint8_t *base = static_cast<uint8_t *>(scratch->p);
    int8_t *aq = reinterpret_cast<int8_t *>(base);
    float *ad = reinterpret_cast<float *>(base + aq_bytes);
    const int kb = k / 32;
    kq_quantize_rows_kernel<<<dim3(static_cast<unsigned>((kb + 7) / 8), static_cast<unsigned>(m)),
                              dim3(256), 0, stream>>>(a, aq, ad, m, k);
    constexpr int LPC = 8;
    const int cols = 256 / LPC;
    const unsigned gx = static_cast<unsigned>((n + cols - 1) / cols);
    const int mr = m == 1 ? 1 : (m == 2 ? 2 : (m <= 4 ? 4 : 8));
    const dim3 grid(gx, static_cast<unsigned>((m + mr - 1) / mr));
    if (mr == 1) {
        kq_i8_gemv_kernel<QT, 1, LPC><<<grid, 256, 0, stream>>>(aq, ad, b, c, m, n, k);
    } else if (mr == 2) {
        kq_i8_gemv_kernel<QT, 2, LPC><<<grid, 256, 0, stream>>>(aq, ad, b, c, m, n, k);
    } else if (mr == 4) {
        kq_i8_gemv_kernel<QT, 4, LPC><<<grid, 256, 0, stream>>>(aq, ad, b, c, m, n, k);
    } else {
        kq_i8_gemv_kernel<QT, 8, LPC><<<grid, 256, 0, stream>>>(aq, ad, b, c, m, n, k);
    }
    return cuda_status(cudaGetLastError());
}

extern "C" int allpaca_cuda_gemm_nt_kquant(int32_t qt, const float *a, const uint8_t *b, float *c,
                                           int32_t m, int32_t n, int32_t k,
                                           allpaca_cuda_stream_t stream) {
    switch (qt) {
    case 4: return kq_gemm_launch<4>(a, b, c, m, n, k, as_stream(stream));
    case 5: return kq_gemm_launch<5>(a, b, c, m, n, k, as_stream(stream));
    case 6: return kq_gemm_launch<6>(a, b, c, m, n, k, as_stream(stream));
    default: return ALLPACA_ERR_INVALID;
    }
}

/* ---------------------------- fused MoE FFN ------------------------------ */

__device__ inline void moe_quantize_lds(const float *v, int k, int8_t *aq, float *ad) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int groups = static_cast<int>(blockDim.x) / 32;
    const int kb = k / 32;
    for (int blk = static_cast<int>(threadIdx.x) / 32; blk < kb; blk += groups) {
        const float x = v[blk * 32 + lane];
        float amax = fabsf(x);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 32));
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

__device__ inline void moe_group(const int32_t *group_start, int g, int &p0, int &mr) {
    p0 = group_start ? group_start[g] : g;
    mr = group_start ? group_start[g + 1] - p0 : 1;
}

template <int QG, int QU, int LPC, int MR>
__global__ void moe_gate_up_kernel(const float *x, const int8_t *xq, const float *xd,
                                   const int32_t *pair_row, const int32_t *group_start,
                                   const int32_t *expert, const uint8_t *gate,
                                   unsigned long long gate_stride, const uint8_t *up,
                                   unsigned long long up_stride, float *g, float *u, int ff, int dim) {
    extern __shared__ unsigned char moe_lds[];
    const int gi = static_cast<int>(blockIdx.y);
    int p0, mr;
    moe_group(group_start, gi, p0, mr);
    const int kb = dim / 32;
    const int8_t *aq;
    const float *ad;
    if (MR == 1 && xq) {
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
        kq_i8_block<QG, MR, LPC>(aq, ad, gate + e * gate_stride, g + static_cast<size_t>(p0) * ff, mr,
                                 ff, dim, static_cast<int>(blockIdx.x), 0);
    } else {
        kq_i8_block<QU, MR, LPC>(aq, ad, up + e * up_stride, u + static_cast<size_t>(p0) * ff, mr, ff,
                                 dim, static_cast<int>(blockIdx.x), 0);
    }
}

__global__ void det_silu_mul_kernel(const float *g, const float *u, float *out, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        out[i] = det_silu_mul(g[i], u[i]);
    }
}

/* Test hook: det_silu_mul over n values. */
extern "C" int allpaca_cuda_det_silu_mul(const float *g, const float *u, float *out, int32_t n,
                                         allpaca_cuda_stream_t stream) {
    if (!g || !u || !out || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    (void)cudaGetLastError();
    det_silu_mul_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, as_stream(stream)>>>(g, u, out, n);
    return cuda_status(cudaGetLastError());
}

__global__ void moe_silu_quant_kernel(const float *g, const float *u, int8_t *hq, float *hd, int ff) {
    extern __shared__ unsigned char moe_lds[];
    float *tmp = reinterpret_cast<float *>(moe_lds);
    const size_t p = blockIdx.x;
    for (int i = static_cast<int>(threadIdx.x); i < ff; i += static_cast<int>(blockDim.x)) {
        tmp[i] = det_silu_mul(g[p * ff + i], u[p * ff + i]);
    }
    __syncthreads();
    moe_quantize_lds(tmp, ff, hq + p * ff, hd + p * (ff / 32));
}

template <int QD, int LPC, int MR>
__global__ void moe_down_kernel(const int8_t *hq, const float *hd, const int32_t *group_start,
                                const int32_t *expert, const uint8_t *down,
                                unsigned long long down_stride, float *y, int dim, int ff) {
    const int gi = static_cast<int>(blockIdx.y);
    int p0, mr;
    moe_group(group_start, gi, p0, mr);
    kq_i8_block<QD, MR, LPC>(hq + static_cast<size_t>(p0) * ff, hd + static_cast<size_t>(p0) * (ff / 32),
                             down + static_cast<size_t>(expert[gi]) * down_stride,
                             y + static_cast<size_t>(p0) * dim, mr, dim, ff,
                             static_cast<int>(blockIdx.x), 0);
}

/* out[r] = sum over row r's pairs (routing order) of w[p] * y[p], with
 * separate rounding for the product and the sum (no FMA), as on HIP. */
__global__ void moe_combine_kernel(const float *y, const float *w, const int32_t *row_start,
                                   const int32_t *row_pairs, float *out, int rows, int dim) {
    const int d = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int r = static_cast<int>(blockIdx.y);
    if (d >= dim || r >= rows) {
        return;
    }
    float acc = 0.f;
    for (int i = row_start[r]; i < row_start[r + 1]; ++i) {
        const int p = row_pairs ? row_pairs[i] : i;
        acc = __fadd_rn(acc, __fmul_rn(y[static_cast<size_t>(p) * dim + d], w[p]));
    }
    out[static_cast<size_t>(r) * dim + d] = acc;
}

constexpr int MOE_LPC = 8;

template <int MR>
static bool moe_gate_up_types(int qg, int qu, dim3 grid, size_t lds, cudaStream_t st, const float *x,
                              const int8_t *xq, const float *xd, const int32_t *pair_row,
                              const int32_t *group_start, const int32_t *expert, const uint8_t *gate,
                              unsigned long long gs, const uint8_t *up, unsigned long long us, float *g,
                              float *u, int ff, int dim) {
#define MOE_GU(A, B)                                                                               \
    if (qg == A && qu == B) {                                                                      \
        moe_gate_up_kernel<A, B, MOE_LPC, MR><<<grid, 256, lds, st>>>(                             \
            x, xq, xd, pair_row, group_start, expert, gate, gs, up, us, g, u, ff, dim);           \
        return true;                                                                               \
    }
    MOE_GU(4, 4) MOE_GU(4, 5) MOE_GU(4, 6)
    MOE_GU(5, 4) MOE_GU(5, 5) MOE_GU(5, 6)
    MOE_GU(6, 4) MOE_GU(6, 5) MOE_GU(6, 6)
#undef MOE_GU
    return false;
}

template <int MR>
static bool moe_down_types(int qd, dim3 grid, cudaStream_t st, const int8_t *hq, const float *hd,
                           const int32_t *group_start, const int32_t *expert, const uint8_t *down,
                           unsigned long long ds, float *y, int dim, int ff) {
#define MOE_D(Q)                                                                                   \
    if (qd == Q) {                                                                                 \
        moe_down_kernel<Q, MOE_LPC, MR><<<grid, 256, 0, st>>>(hq, hd, group_start, expert, down,  \
                                                              ds, y, dim, ff);                     \
        return true;                                                                               \
    }
    MOE_D(4) MOE_D(5) MOE_D(6)
#undef MOE_D
    return false;
}

static size_t moe_align(size_t v) { return (v + 255) & ~size_t(255); }

extern "C" uint64_t allpaca_cuda_moe_ffn_scratch_bytes(int32_t rows, int32_t dim, int32_t ff,
                                                       int32_t pairs) {
    (void)rows;
    const size_t p = static_cast<size_t>(pairs);
    return 2 * moe_align(p * ff * 4) + moe_align(p * dim * 4) + moe_align(p * ff) +
           moe_align(p * (ff / 32) * 4);
}

/* Same contract as HIP `allpaca_moe_ffn_kquant`. */
extern "C" int allpaca_cuda_moe_ffn_kquant(
    const float *x, const int8_t *xq, const float *xd, int32_t rows, int32_t dim, int32_t ff,
    int32_t pairs, const int32_t *pair_row, const int32_t *expert, const float *pair_w,
    const int32_t *row_start, int32_t groups, const int32_t *group_start, const int32_t *row_pairs,
    int32_t max_group, const uint8_t *gate, int32_t gate_qt, uint64_t gate_stride, const uint8_t *up,
    int32_t up_qt, uint64_t up_stride, const uint8_t *down, int32_t down_qt, uint64_t down_stride,
    void *scratch, float *out, allpaca_cuda_stream_t stream) {
    if ((!x && !(xq && xd)) || !out || !scratch || rows <= 0 || pairs <= 0 || groups <= 0 ||
        dim % 256 != 0 || ff % 256 != 0 || !pair_row || !expert || !pair_w || !row_start ||
        (group_start == nullptr && groups != pairs) || max_group < 1 || max_group > 8) {
        return ALLPACA_ERR_INVALID;
    }
    auto kq = [](int32_t qt) { return qt == 4 || qt == 5 || qt == 6; };
    if (!kq(gate_qt) || !kq(up_qt) || !kq(down_qt)) {
        return ALLPACA_ERR_INVALID;
    }
    (void)cudaGetLastError();
    cudaStream_t st = as_stream(stream);
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
    const size_t lds_silu = static_cast<size_t>(ff) * 4;
    /* Default dynamic shared memory limit: dim <= ~5k at 8 rows, ff <= 12k. */
    if (lds_up > 48 * 1024 || lds_silu > 48 * 1024) {
        return ALLPACA_ERR_INVALID;
    }
    const dim3 grid_up(static_cast<unsigned>((ff + COLS - 1) / COLS), static_cast<unsigned>(groups), 2);
    const dim3 grid_down(static_cast<unsigned>((dim + COLS - 1) / COLS), static_cast<unsigned>(groups));
    bool ok = false;
#define MOE_MR(M)                                                                                  \
    ok = moe_gate_up_types<M>(gate_qt, up_qt, grid_up, lds_up, st, x, xq, xd, pair_row,            \
                              group_start, expert, gate, gate_stride, up, up_stride, g, u, ff,     \
                              dim);                                                                \
    if (ok) {                                                                                      \
        moe_silu_quant_kernel<<<static_cast<unsigned>(pairs), 256, lds_silu, st>>>(g, u, hq, hd,  \
                                                                                   ff);            \
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
    moe_combine_kernel<<<dim3(static_cast<unsigned>((dim + 255) / 256), static_cast<unsigned>(rows)), 256,
                         0, st>>>(y, pair_w, row_start, row_pairs, out, rows, dim);
    return cuda_status(cudaGetLastError());
}
