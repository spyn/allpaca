/* Allpaca CUDA kernel ABI. Compiles with nvcc even when no NVIDIA GPU is present;
 * device APIs then report count=0 / ERR_DEVICE. */

#include "allpaca_cuda_kernels.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

#define ALLPACA_HAS_CUDA 1

static int cuda_status(cudaError_t err) {
    if (err == cudaSuccess) {
        return ALLPACA_OK;
    }
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
        return cuda_status(cudaFreeAsync(ptr, s));
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
        acc += __shfl_down_sync(0xffffffffu, acc, off);
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
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    /* shfl_down leaves the full sum on lane 0; broadcast so every lane uses it. */
    return __shfl_sync(0xffffffffu, v, 0);
}

/* Warp-per-head online softmax. Supports head_dim <= 512 (16 dims/lane × 32).
 * Optional sinks[h] is a virtual logit with no V (HIP attention_flash_kernel).
 * softcap>0 applies Gemma logit softcap after scale. */
__global__ void attention_online_kernel(const float *q, const float *k, const float *v, float *out,
                                        int n_q, int n_kv, int n_heads, int n_kv_heads, int head_dim,
                                        int causal, const uint8_t *mask, const float *sinks,
                                        float softcap) {
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
    int nd = 0;
    for (int d = lane; d < head_dim && nd < MAX_DPT; d += 32) {
        qreg[nd] = qh[d];
        acc[nd] = 0.f;
        nd++;
    }

    float m = -1.0e30f;
    float l = 0.f;
    if (sinks) {
        m = sinks[h];
        l = 1.f;
    }

    for (int t = 0; t < n_kv; ++t) {
        const bool valid = mask ? mask[static_cast<size_t>(qi) * static_cast<size_t>(n_kv) +
                                       static_cast<size_t>(t)] != 0
                                : t <= t_max;
        const float *kh = k + static_cast<size_t>(t) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
        const float *vh = v + static_cast<size_t>(t) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
        float partial = 0.f;
        int i = 0;
        for (int d = lane; d < head_dim && i < MAX_DPT; d += 32) {
            partial += qreg[i++] * (valid ? kh[d] : 0.f);
        }
        float s = warp_sum32(partial) * scale;
        if (softcap > 0.f) {
            s = tanhf(s / softcap) * softcap;
        }
        if (!valid) {
            s = -1.0e30f;
        }
        const float m_new = fmaxf(m, s);
        const float alpha = expf(m - m_new);
        const float p = expf(s - m_new);
        l = l * alpha + p;
        i = 0;
        for (int d = lane; d < head_dim && i < MAX_DPT; d += 32) {
            acc[i] = acc[i] * alpha + p * (valid ? vh[d] : 0.f);
            i++;
        }
        m = m_new;
    }

    const float inv = (l == 0.f) ? 0.f : 1.f / l;
    int i = 0;
    for (int d = lane; d < head_dim && i < MAX_DPT; d += 32) {
        oh[d] = acc[i++] * inv;
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
    dim3 block(32);
    dim3 grid(static_cast<unsigned>(n_q), static_cast<unsigned>(n_heads));
    attention_online_kernel<<<grid, block, 0, as_stream(stream)>>>(
        q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, nullptr, sinks, 0.f);
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
    dim3 block(32);
    dim3 grid(static_cast<unsigned>(n_q), static_cast<unsigned>(n_heads));
    attention_online_kernel<<<grid, block, 0, as_stream(stream)>>>(
        q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, 0, mask, nullptr, 0.f);
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
    dim3 block(32);
    dim3 grid(static_cast<unsigned>(n_q), static_cast<unsigned>(n_heads));
    attention_online_kernel<<<grid, block, 0, as_stream(stream)>>>(
        q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, nullptr, nullptr, softcap);
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
    dim3 block(32);
    dim3 grid(static_cast<unsigned>(n_q), static_cast<unsigned>(n_heads));
    attention_online_kernel<<<grid, block, 0, as_stream(stream)>>>(
        q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, 0, mask, nullptr, softcap);
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
