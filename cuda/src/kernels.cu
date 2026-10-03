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
            return ALLPACA_ERR_DEVICE;
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
        return ALLPACA_ERR_DEVICE;
    }
    *can = yes;
    return ALLPACA_OK;
}

extern "C" int allpaca_cuda_enable_peer_access(int32_t src, int32_t dst) {
    cudaError_t err = cudaSetDevice(src);
    if (err != cudaSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    err = cudaDeviceEnablePeerAccess(dst, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
        return ALLPACA_ERR_DEVICE;
    }
    return ALLPACA_OK;
}

extern "C" int allpaca_cuda_memcpy_peer(void *dst, int32_t dst_dev, const void *src, int32_t src_dev,
                                        size_t bytes, allpaca_cuda_stream_t stream) {
    if ((!dst && bytes) || (!src && bytes)) {
        return ALLPACA_ERR_INVALID;
    }
    cudaStream_t s = as_stream(stream);
    cudaError_t err = cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, s);
    if (err != cudaSuccess) {
        return ALLPACA_ERR_DEVICE;
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
        return ALLPACA_ERR_DEVICE;
    }
    const bool rebind = prev != device;
    if (rebind) {
        err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            return ALLPACA_ERR_DEVICE;
        }
    }
    size_t free_b = 0;
    size_t total_b = 0;
    err = cudaMemGetInfo(&free_b, &total_b);
    if (rebind) {
        (void)cudaSetDevice(prev);
    }
    if (err != cudaSuccess) {
        return ALLPACA_ERR_DEVICE;
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
        return ALLPACA_ERR_DEVICE;
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
        return ALLPACA_ERR_DEVICE;
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
        return ALLPACA_ERR_DEVICE;
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
        return ALLPACA_ERR_DEVICE;
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

__global__ void silu_mul_kernel(const float *gate, const float *up, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float s = g / (1.f + expf(-g));
        out[i] = s * up[i];
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
    attn[idx] *= 1.0f / (1.0f + expf(-s));
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
        acc += v * v;
    }
    buf[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            buf[tid] += buf[tid + stride];
        }
        __syncthreads();
    }
    float inv = rsqrtf(buf[0] / static_cast<float>(n) + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = x[i] * inv * w[i];
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
    float freq = powf(theta, -static_cast<float>(i) / static_cast<float>(half));
    float angle = static_cast<float>(pos) * freq;
    float s = sinf(angle);
    float c = cosf(angle);
    float a = hvec[2 * i];
    float b = hvec[2 * i + 1];
    hvec[2 * i] = a * c - b * s;
    hvec[2 * i + 1] = a * s + b * c;
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

__device__ float warp_sum32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        /* XOR butterfly leaves the full sum on every lane (needed: all 32
         * threads share one attention score). Down-shuffle only fills lane 0.
         * Kepler/Maxwell/Pascal lack *_sync; Tesla K80 is sm_37. */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
        v += __shfl_xor(v, offset);
#else
        v += __shfl_xor_sync(0xffffffffu, v, offset);
#endif
    }
    return v;
}

/* Warp-per-row online softmax. Supports head_dim <= 512 (16 dims/lane × 32).
 * Optional sinks[h] is a virtual logit with no V (HIP attention_flash_kernel).
 * softcap>0 applies Gemma logit softcap after scale. `attn_row_*` hold the
 * per-row arithmetic so the per-row and Q-tiled kernels give the same bits
 * for a row whichever computed it. */
constexpr int ATTN_MAX_DPT = 16;

struct AttnRow {
    float qreg[ATTN_MAX_DPT];
    float acc[ATTN_MAX_DPT];
    float m;
    float l;
};

/* Fixed trip counts with a per-lane guard keep the arrays in registers. */
__device__ __forceinline__ void attn_row_init(AttnRow &r, const float *qh, int lane, int head_dim,
                                              const float *sinks, int h) {
#pragma unroll
    for (int j = 0; j < ATTN_MAX_DPT; ++j) {
        const int d = lane + 32 * j;
        r.qreg[j] = d < head_dim ? qh[d] : 0.f;
        r.acc[j] = 0.f;
    }
    r.m = -1.0e30f;
    r.l = 0.f;
    if (sinks) {
        r.m = sinks[h];
        r.l = 1.f;
    }
}

/* Rounding is spelled out (__fmaf_rn / __fmul_rn): left to contraction, nvcc
 * fused `acc * alpha + p * v` differently in the two kernels and their rows
 * differed in the last bit. */
__device__ __forceinline__ void attn_row_step(AttnRow &r, const float *kh, const float *vh,
                                              bool valid, int lane, int head_dim, float scale,
                                              float softcap) {
    float partial = 0.f;
#pragma unroll
    for (int j = 0; j < ATTN_MAX_DPT; ++j) {
        const int d = lane + 32 * j;
        if (d < head_dim) {
            partial = __fmaf_rn(r.qreg[j], valid ? kh[d] : 0.f, partial);
        }
    }
    float s = __fmul_rn(warp_sum32(partial), scale);
    if (softcap > 0.f) {
        s = __fmul_rn(tanhf(s / softcap), softcap);
    }
    if (!valid) {
        s = -1.0e30f;
    }
    const float m_new = fmaxf(r.m, s);
    const float alpha = expf(r.m - m_new);
    const float p = expf(s - m_new);
    r.l = __fmaf_rn(r.l, alpha, p);
#pragma unroll
    for (int j = 0; j < ATTN_MAX_DPT; ++j) {
        const int d = lane + 32 * j;
        if (d < head_dim) {
            r.acc[j] = __fmaf_rn(r.acc[j], alpha, __fmul_rn(p, valid ? vh[d] : 0.f));
        }
    }
    r.m = m_new;
}

__device__ __forceinline__ void attn_row_store(const AttnRow &r, float *oh, int lane, int head_dim) {
    const float inv = (r.l == 0.f) ? 0.f : 1.f / r.l;
#pragma unroll
    for (int j = 0; j < ATTN_MAX_DPT; ++j) {
        const int d = lane + 32 * j;
        if (d < head_dim) {
            oh[d] = r.acc[j] * inv;
        }
    }
}

__device__ __forceinline__ int attn_row_tmax(int qi, int n_q, int n_kv, int causal) {
    return causal ? qi + (n_kv > n_q ? n_kv - n_q : 0) : n_kv - 1;
}

/* Rows longer than ATTN_CHUNK keys fold 256-key chunks in ascending order
 * (see the HIP kernels); every kernel follows it, one chunk is single-pass. */
constexpr int ATTN_CHUNK = 256;
constexpr int ATTN_PART = 32 * ATTN_MAX_DPT + 2;

__device__ __forceinline__ void attn_row_reset(AttnRow &r) {
#pragma unroll
    for (int i = 0; i < ATTN_MAX_DPT; ++i) {
        r.acc[i] = 0.f;
    }
    r.m = -1.0e30f;
    r.l = 0.f;
}

__device__ __forceinline__ void attn_fold(float &m, float &l, float (&acc)[ATTN_MAX_DPT], float m_b,
                                          float l_b, const float (&acc_b)[ATTN_MAX_DPT]) {
    const float mm = fmaxf(m, m_b);
    const float a = expf(m - mm);
    const float b = expf(m_b - mm);
    l = __fmaf_rn(l, a, __fmul_rn(l_b, b));
#pragma unroll
    for (int i = 0; i < ATTN_MAX_DPT; ++i) {
        acc[i] = __fmaf_rn(acc[i], a, __fmul_rn(acc_b[i], b));
    }
    m = mm;
}

__global__ void __launch_bounds__(32)
    attention_online_kernel(const float *q, const float *k, const float *v, float *out, int n_q,
                            int n_kv, int n_heads, int n_kv_heads, int head_dim, int causal,
                            const uint8_t *mask, const float *sinks, float softcap) {
    const int qi = static_cast<int>(blockIdx.x);
    const int h = static_cast<int>(blockIdx.y);
    if (qi >= n_q || h >= n_heads) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const int kv_h = h / (n_heads / n_kv_heads);
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int t_max = attn_row_tmax(qi, n_q, n_kv, causal);
    const float *qh = q + (static_cast<size_t>(qi) * n_heads + h) * head_dim;
    float *oh = out + (static_cast<size_t>(qi) * n_heads + h) * head_dim;
    const size_t kv_stride = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
    AttnRow r;
    attn_row_init(r, qh, lane, head_dim, sinks, h);
    const int t_end = mask ? n_kv : min(n_kv, t_max + 1);
    for (int t = 0; t < t_end; ++t) {
        const bool valid = mask ? mask[static_cast<size_t>(qi) * static_cast<size_t>(n_kv) +
                                       static_cast<size_t>(t)] != 0
                                : t <= t_max;
        const float *kh = k + static_cast<size_t>(t) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
        const float *vh = v + static_cast<size_t>(t) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
        attn_row_step(r, kh, vh, valid, lane, head_dim, scale, softcap);
    }
    attn_row_store(r, oh, lane, head_dim);
}

constexpr int ATTN_QTILE = 8;

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
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const size_t kv_stride = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
    const int qrow = active ? qi : q0;
    const float *qh = q + (static_cast<size_t>(qrow) * n_heads + h) * head_dim;
    float *oh = out + (static_cast<size_t>(qrow) * n_heads + h) * head_dim;
    const int t_max = attn_row_tmax(qi, n_q, n_kv, causal);
    const int t_end = active ? (mask ? n_kv : min(n_kv, t_max + 1)) : 0;
    const int q_last = min(n_q, q0 + ATTN_QTILE) - 1;
    const int block_end = mask ? n_kv : min(n_kv, attn_row_tmax(q_last, n_q, n_kv, causal) + 1);
    AttnRow r;
    attn_row_init(r, qh, lane, head_dim, sinks, h);
    float m_acc[ATTN_MAX_DPT];
    float m_m = -1.0e30f, m_l = 0.f;
    bool folded = false;
    int next_fold = ATTN_CHUNK;
    const int n_tile = tile * head_dim;
    for (int t0 = 0; t0 < block_end; t0 += tile) {
        __syncthreads();
        for (int e = static_cast<int>(threadIdx.x); e < n_tile; e += static_cast<int>(blockDim.x)) {
            const int t = t0 + e / head_dim;
            const int d = e % head_dim;
            const size_t src = static_cast<size_t>(t) * kv_stride +
                               static_cast<size_t>(kv_h) * head_dim + static_cast<size_t>(d);
            k_lds[e] = t < n_kv ? k[src] : 0.f;
            v_lds[e] = t < n_kv ? v[src] : 0.f;
        }
        __syncthreads();
        const int stop = min(t_end, t0 + tile);
        for (int t = t0; t < stop; ++t) {
            if (t == next_fold) {
                next_fold += ATTN_CHUNK;
                if (folded) {
                    attn_fold(m_m, m_l, m_acc, r.m, r.l, r.acc);
                } else {
#pragma unroll
                    for (int i = 0; i < ATTN_MAX_DPT; ++i) {
                        m_acc[i] = r.acc[i];
                    }
                    m_m = r.m;
                    m_l = r.l;
                    folded = true;
                }
                attn_row_reset(r);
            }
            const bool valid = mask ? mask[static_cast<size_t>(qi) * static_cast<size_t>(n_kv) +
                                           static_cast<size_t>(t)] != 0
                                    : t <= t_max;
            attn_row_step(r, k_lds + static_cast<size_t>(t - t0) * head_dim,
                          v_lds + static_cast<size_t>(t - t0) * head_dim, valid, lane, head_dim,
                          scale, softcap);
        }
    }
    if (active) {
        if (folded) {
            attn_fold(m_m, m_l, m_acc, r.m, r.l, r.acc);
#pragma unroll
            for (int i = 0; i < ATTN_MAX_DPT; ++i) {
                r.acc[i] = m_acc[i];
            }
            r.m = m_m;
            r.l = m_l;
        }
        attn_row_store(r, oh, lane, head_dim);
    }
}

__global__ void __launch_bounds__(32)
    attention_split_kernel(const float *q, const float *k, const float *v, float *part, int n_q,
                           int n_kv, int n_heads, int n_kv_heads, int head_dim, int causal,
                           const uint8_t *mask, const float *sinks, float softcap, int n_chunks) {
    const int qi = static_cast<int>(blockIdx.x);
    const int h = static_cast<int>(blockIdx.y);
    const int c = static_cast<int>(blockIdx.z);
    const int lane = static_cast<int>(threadIdx.x);
    const int kv_h = h / (n_heads / n_kv_heads);
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int t_max = attn_row_tmax(qi, n_q, n_kv, causal);
    const float *qh = q + (static_cast<size_t>(qi) * n_heads + h) * head_dim;
    const size_t kv_stride = static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
    AttnRow r;
    attn_row_init(r, qh, lane, head_dim, c == 0 ? sinks : nullptr, h);
    const int t_end = mask ? n_kv : min(n_kv, t_max + 1);
    const int t1 = min(t_end, (c + 1) * ATTN_CHUNK);
    for (int t = c * ATTN_CHUNK; t < t1; ++t) {
        const bool valid = mask ? mask[static_cast<size_t>(qi) * static_cast<size_t>(n_kv) +
                                       static_cast<size_t>(t)] != 0
                                : t <= t_max;
        const float *kh = k + static_cast<size_t>(t) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
        const float *vh = v + static_cast<size_t>(t) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
        attn_row_step(r, kh, vh, valid, lane, head_dim, scale, softcap);
    }
    float *dst = part + ((static_cast<size_t>(qi) * n_heads + h) * n_chunks + c) * ATTN_PART;
#pragma unroll
    for (int i = 0; i < ATTN_MAX_DPT; ++i) {
        dst[lane * ATTN_MAX_DPT + i] = r.acc[i];
    }
    if (lane == 0) {
        dst[32 * ATTN_MAX_DPT] = r.m;
        dst[32 * ATTN_MAX_DPT + 1] = r.l;
    }
}

__global__ void __launch_bounds__(32)
    attention_merge_kernel(const float *part, float *out, int n_heads, int head_dim, int n_chunks) {
    const int qi = static_cast<int>(blockIdx.x);
    const int h = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    float *oh = out + (static_cast<size_t>(qi) * n_heads + h) * head_dim;
    const float *src = part + (static_cast<size_t>(qi) * n_heads + h) * n_chunks * ATTN_PART;
    AttnRow r;
#pragma unroll
    for (int i = 0; i < ATTN_MAX_DPT; ++i) {
        r.acc[i] = src[lane * ATTN_MAX_DPT + i];
    }
    r.m = src[32 * ATTN_MAX_DPT];
    r.l = src[32 * ATTN_MAX_DPT + 1];
    for (int c = 1; c < n_chunks; ++c) {
        const float *pc = src + static_cast<size_t>(c) * ATTN_PART;
        float acc_b[ATTN_MAX_DPT];
#pragma unroll
        for (int i = 0; i < ATTN_MAX_DPT; ++i) {
            acc_b[i] = pc[lane * ATTN_MAX_DPT + i];
        }
        attn_fold(r.m, r.l, r.acc, pc[32 * ATTN_MAX_DPT], pc[32 * ATTN_MAX_DPT + 1], acc_b);
    }
    attn_row_store(r, oh, lane, head_dim);
}

constexpr int ATTN_PART_DEVICES = 16;
struct AttnPartScratch {
    std::mutex mu;
    float *p = nullptr;
    size_t cap = 0;
    cudaStream_t stream = nullptr;
};
static AttnPartScratch g_attn_part[ATTN_PART_DEVICES];

/* One query: per-row kernel, or chunks in parallel plus a merge for a row
 * longer than one chunk. Several rows: Q-tiled. A row gets the same bits. */
static void launch_attention(const float *q, const float *k, const float *v, float *out, int n_q,
                             int n_kv, int n_heads, int n_kv_heads, int head_dim, int causal,
                             const uint8_t *mask, const float *sinks, float softcap,
                             cudaStream_t stream) {
    const int tile = head_dim > 0 ? 2048 / head_dim : 0;
    const int n_chunks = (n_kv + ATTN_CHUNK - 1) / ATTN_CHUNK;
    if (n_q == 1 && n_chunks > 1) {
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < ATTN_PART_DEVICES) {
            AttnPartScratch &s = g_attn_part[dev];
            std::lock_guard<std::mutex> lock(s.mu);
            bool ok = !(s.p && s.stream != stream && cudaDeviceSynchronize() != cudaSuccess);
            const size_t need = static_cast<size_t>(n_heads) * n_chunks * ATTN_PART;
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
                attention_split_kernel<<<dim3(1, static_cast<unsigned>(n_heads),
                                              static_cast<unsigned>(n_chunks)),
                                         32, 0, stream>>>(q, k, v, s.p, n_q, n_kv, n_heads, n_kv_heads,
                                                          head_dim, causal, mask, sinks, softcap,
                                                          n_chunks);
                attention_merge_kernel<<<dim3(1, static_cast<unsigned>(n_heads)), 32, 0, stream>>>(
                    s.p, out, n_heads, head_dim, n_chunks);
                return;
            }
        }
    }
    if ((n_q >= 2 || n_chunks > 1) && tile >= 1) {
        dim3 block(32 * ATTN_QTILE);
        dim3 grid(static_cast<unsigned>((n_q + ATTN_QTILE - 1) / ATTN_QTILE),
                  static_cast<unsigned>(n_heads));
        const size_t lds = 2 * static_cast<size_t>(tile) * head_dim * sizeof(float);
        attention_qtile_kernel<<<grid, block, lds, stream>>>(q, k, v, out, n_q, n_kv, n_heads,
                                                             n_kv_heads, head_dim, causal, mask,
                                                             sinks, softcap, tile);
        return;
    }
    dim3 block(32);
    dim3 grid(static_cast<unsigned>(n_q), static_cast<unsigned>(n_heads));
    attention_online_kernel<<<grid, block, 0, stream>>>(q, k, v, out, n_q, n_kv, n_heads,
                                                        n_kv_heads, head_dim, causal, mask, sinks,
                                                        softcap);
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
    x[idx] *= 1.0f / (1.0f + expf(-g));
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

/* exp(x) with every rounding spelled out, so HIP, CUDA and the CPU
 * reference (`allpaca_cpu::det_expf`) give the same bits: the vendors' expf
 * differ in the last bit for some inputs, and the MoE SiLU used it, so expert
 * layers on an NVIDIA worker changed the text against an AMD one.
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

/* silu(g) * u for the MoE expert FFN, bit-identical across backends. */
__device__ inline float det_silu_mul(float g, float u) {
    return __fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, det_expf(-g))), u);
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
