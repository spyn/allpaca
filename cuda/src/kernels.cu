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
#if CUDART_VERSION < 12000
    /* Theoretical peak: 2 * clock(kHz) * bus_width(bits) / 8 → GB/s.
     * CUDA 12+ removed memoryClockRate; leave 0 and let the scheduler
     * treat bandwidth as unknown until a real benchmark runs. */
    *bandwidth_gbps = 2.0 * (static_cast<double>(prop.memoryClockRate) / 1.0e6) *
                      (static_cast<double>(prop.memoryBusWidth) / 8.0);
#else
    *bandwidth_gbps = 0.0;
#endif
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

extern "C" int allpaca_cuda_memcpy(void *dst, const void *src, size_t bytes, int32_t kind,
                                   allpaca_cuda_stream_t stream) {
    if ((!dst && bytes) || (!src && bytes)) {
        return ALLPACA_ERR_INVALID;
    }
    cudaMemcpyKind ck = cudaMemcpyDefault;
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
    default:
        ck = cudaMemcpyDefault;
        break;
    }
    cudaStream_t s = as_stream(stream);
    if (s) {
        return cuda_status(cudaMemcpyAsync(dst, src, bytes, ck, s));
    }
    return cuda_status(cudaMemcpy(dst, src, bytes, ck));
}

extern "C" int allpaca_cuda_set_device(int32_t device) {
    return cuda_status(cudaSetDevice(device));
}

extern "C" int allpaca_cuda_stream_synchronize(allpaca_cuda_stream_t stream) {
    cudaStream_t s = as_stream(stream);
    if (s) {
        return cuda_status(cudaStreamSynchronize(s));
    }
    return cuda_status(cudaDeviceSynchronize());
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

/* Single-block RMSNorm for decode-sized rows (n fits one reduction). */
__global__ void rmsnorm_kernel(const float *x, const float *w, float *out, int n, float eps) {
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

extern "C" int allpaca_cuda_rmsnorm(const float *x, const float *w, float *out, int32_t n, float eps,
                                    allpaca_cuda_stream_t stream) {
    if (!x || !w || !out || n <= 0) {
        return ALLPACA_ERR_INVALID;
    }
    rmsnorm_kernel<<<1, 256, 0, as_stream(stream)>>>(x, w, out, n, eps);
    return cuda_status(cudaGetLastError());
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

/* Warp-per-head online softmax. Supports head_dim <= 512 (16 dims/lane × 32). */
__global__ void attention_online_kernel(const float *q, const float *k, const float *v, float *out,
                                        int n_q, int n_kv, int n_heads, int n_kv_heads, int head_dim,
                                        int causal) {
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

extern "C" int allpaca_cuda_attention(const float *q, const float *k, const float *v, float *out,
                                      int32_t n_q, int32_t n_kv, int32_t n_heads, int32_t n_kv_heads,
                                      int32_t head_dim, int32_t causal,
                                      allpaca_cuda_stream_t stream) {
    if (!q || !k || !v || !out || n_q <= 0 || n_kv <= 0 || n_heads <= 0 || n_kv_heads <= 0 ||
        head_dim <= 0 || (n_heads % n_kv_heads) != 0 || head_dim > 512) {
        return ALLPACA_ERR_INVALID;
    }
    dim3 block(32);
    dim3 grid(static_cast<unsigned>(n_q), static_cast<unsigned>(n_heads));
    attention_online_kernel<<<grid, block, 0, as_stream(stream)>>>(q, k, v, out, n_q, n_kv, n_heads,
                                                                   n_kv_heads, head_dim, causal);
    return cuda_status(cudaGetLastError());
}
