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

__device__ float warp_sum32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    /* shfl_down leaves the full sum on lane 0; broadcast so every lane uses it. */
    return __shfl_sync(0xffffffffu, v, 0);
}

/* Warp-per-head online softmax. Supports head_dim <= 512 (16 dims/lane × 32).
 * Optional sinks[h] is a virtual logit with no V (HIP attention_flash_kernel). */
__global__ void attention_online_kernel(const float *q, const float *k, const float *v, float *out,
                                        int n_q, int n_kv, int n_heads, int n_kv_heads, int head_dim,
                                        int causal, const float *sinks) {
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
        const bool valid = t <= t_max;
        const float *kh = k + static_cast<size_t>(t) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
        const float *vh = v + static_cast<size_t>(t) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
        float partial = 0.f;
        int i = 0;
        for (int d = lane; d < head_dim && i < MAX_DPT; d += 32) {
            partial += qreg[i++] * (valid ? kh[d] : 0.f);
        }
        float s = warp_sum32(partial) * scale;
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
    attention_online_kernel<<<grid, block, 0, as_stream(stream)>>>(q, k, v, out, n_q, n_kv, n_heads,
                                                                   n_kv_heads, head_dim, causal,
                                                                   sinks);
    return cuda_status(cudaGetLastError());
}

extern "C" int allpaca_cuda_attention(const float *q, const float *k, const float *v, float *out,
                                      int32_t n_q, int32_t n_kv, int32_t n_heads, int32_t n_kv_heads,
                                      int32_t head_dim, int32_t causal,
                                      allpaca_cuda_stream_t stream) {
    return allpaca_cuda_attention_sinks(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim,
                                        causal, nullptr, stream);
}
