#include "../include/allpaca_kernels.h"

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define ALLPACA_HAS_HIP 1
#else
#define ALLPACA_HAS_HIP 0
#endif

#if ALLPACA_HAS_HIP

static hipStream_t as_stream(allpaca_stream_t stream) {
    return static_cast<hipStream_t>(stream);
}

static int hip_status(hipError_t err) {
    return err == hipSuccess ? ALLPACA_OK : ALLPACA_ERR_DEVICE;
}

__global__ void gemm_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) {
        acc += a[row * k + p] * b[p * n + col];
    }
    c[row * n + col] = acc;
}

__global__ void gemm_nt_kernel(const float *a, const float *b, float *c, int m, int n, int k) {
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) {
        acc += a[row * k + p] * b[col * k + p];
    }
    c[row * n + col] = acc;
}

__global__ void attention_kernel(const float *q, const float *k, const float *v, float *out,
                                 int n_q, int n_kv, int n_heads, int n_kv_heads, int head_dim,
                                 int causal) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = n_q * n_heads;
    if (idx >= total) {
        return;
    }
    const int qi = idx / n_heads;
    const int h = idx % n_heads;
    const int reps = n_heads / n_kv_heads;
    const int kv_h = h / reps;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float *qh = q + (qi * n_heads + h) * head_dim;

    float max_s = -1.0e30f;
    for (int t = 0; t < n_kv; ++t) {
        if (causal && t > qi + (n_kv - n_q > 0 ? n_kv - n_q : 0)) {
            continue;
        }
        const float *kh = k + (t * n_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += qh[d] * kh[d];
        }
        const float s = dot * scale;
        if (s > max_s) {
            max_s = s;
        }
    }

    float sum = 0.0f;
    for (int t = 0; t < n_kv; ++t) {
        if (causal && t > qi + (n_kv - n_q > 0 ? n_kv - n_q : 0)) {
            continue;
        }
        const float *kh = k + (t * n_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += qh[d] * kh[d];
        }
        sum += expf(dot * scale - max_s);
    }
    if (sum == 0.0f) {
        sum = 1.0f;
    }

    float *oh = out + (qi * n_heads + h) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        oh[d] = 0.0f;
    }
    for (int t = 0; t < n_kv; ++t) {
        if (causal && t > qi + (n_kv - n_q > 0 ? n_kv - n_q : 0)) {
            continue;
        }
        const float *kh = k + (t * n_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += qh[d] * kh[d];
        }
        const float w = expf(dot * scale - max_s) / sum;
        const float *vh = v + (t * n_kv_heads + kv_h) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            oh[d] += w * vh[d];
        }
    }
}

static int launch_gemm(const float *a, const float *b, float *c, int32_t m, int32_t n, int32_t k,
                       hipStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((n + 15) / 16), static_cast<unsigned>((m + 15) / 16));
    hipLaunchKernelGGL(gemm_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    return hip_status(hipStreamSynchronize(stream));
}

static int launch_gemm_nt(const float *a, const float *b, float *c, int32_t m, int32_t n, int32_t k,
                          hipStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c) {
        return ALLPACA_ERR_INVALID;
    }
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((n + 15) / 16), static_cast<unsigned>((m + 15) / 16));
    hipLaunchKernelGGL(gemm_nt_kernel, grid, block, 0, stream, a, b, c, m, n, k);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    return hip_status(hipStreamSynchronize(stream));
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
    return hip_status(hipStreamSynchronize(as_stream(stream)));
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

extern "C" int allpaca_attention(const float *q, const float *k, const float *v, float *out,
                                 int32_t n_q, int32_t n_kv, int32_t n_heads, int32_t n_kv_heads,
                                 int32_t head_dim, int32_t causal, allpaca_stream_t stream) {
    if (!q || !k || !v || !out || n_q <= 0 || n_kv <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_dim <= 0 || (n_heads % n_kv_heads) != 0) {
        return ALLPACA_ERR_INVALID;
    }
    const int total = n_q * n_heads;
    const int threads = 64;
    const int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(attention_kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), 0, as_stream(stream), q, k, v, out, n_q,
                       n_kv, n_heads, n_kv_heads, head_dim, causal);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return ALLPACA_ERR_DEVICE;
    }
    return hip_status(hipStreamSynchronize(as_stream(stream)));
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

extern "C" int allpaca_gemm(const float *, const float *, float *, int32_t, int32_t, int32_t,
                            allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_gemm_nt(const float *, const float *, float *, int32_t, int32_t, int32_t,
                               allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_attention(const float *, const float *, const float *, float *, int32_t,
                                 int32_t, int32_t, int32_t, int32_t, int32_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

extern "C" int allpaca_copy_async(void *, const void *, size_t, allpaca_stream_t) {
    return ALLPACA_ERR_NOT_IMPLEMENTED;
}

#endif
