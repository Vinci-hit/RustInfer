#include "scalar.h"

// ===================== scalar_mul kernels =====================

__global__ void scalar_mul_f32_kernel(float* dst, const float* src, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    // float4 vectorized path
    int n4 = n / 4;
    for (int i = idx; i < n4; i += stride) {
        float4 v = reinterpret_cast<const float4*>(src)[i];
        v.x *= val; v.y *= val; v.z *= val; v.w *= val;
        reinterpret_cast<float4*>(dst)[i] = v;
    }
    // tail elements
    int base = n4 * 4;
    for (int i = base + idx; i < n; i += stride) {
        dst[i] = src[i] * val;
    }
}

void scalar_mul_f32_forward(float* dst, const float* src, float val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_mul_f32_kernel<<<blocks, threads, 0, stream>>>(dst, src, val, n);
}

__global__ void scalar_mul_bf16_kernel(__nv_bfloat16* dst, const __nv_bfloat16* src, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    __nv_bfloat162 val2 = __float2bfloat162_rn(val);
    int n2 = n / 2;
    for (int i = idx; i < n2; i += stride) {
        __nv_bfloat162 v = reinterpret_cast<const __nv_bfloat162*>(src)[i];
        reinterpret_cast<__nv_bfloat162*>(dst)[i] = __hmul2(v, val2);
    }
    if (n % 2 != 0 && idx == 0) {
        dst[n - 1] = __float2bfloat16(__bfloat162float(src[n - 1]) * val);
    }
}

void scalar_mul_bf16_forward(__nv_bfloat16* dst, const __nv_bfloat16* src, float val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_mul_bf16_kernel<<<blocks, threads, 0, stream>>>(dst, src, val, n);
}

__global__ void scalar_mul_f16_kernel(__half* dst, const __half* src, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    __half2 val2 = __float2half2_rn(val);
    int n2 = n / 2;
    for (int i = idx; i < n2; i += stride) {
        __half2 v = reinterpret_cast<const __half2*>(src)[i];
        reinterpret_cast<__half2*>(dst)[i] = __hmul2(v, val2);
    }
    if (n % 2 != 0 && idx == 0) {
        dst[n - 1] = __float2half(__half2float(src[n - 1]) * val);
    }
}

void scalar_mul_f16_forward(__half* dst, const __half* src, float val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_mul_f16_kernel<<<blocks, threads, 0, stream>>>(dst, src, val, n);
}

// ===================== scalar_add kernels =====================

__global__ void scalar_add_f32_kernel(float* dst, const float* src, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n4 = n / 4;
    for (int i = idx; i < n4; i += stride) {
        float4 v = reinterpret_cast<const float4*>(src)[i];
        v.x += val; v.y += val; v.z += val; v.w += val;
        reinterpret_cast<float4*>(dst)[i] = v;
    }
    int base = n4 * 4;
    for (int i = base + idx; i < n; i += stride) {
        dst[i] = src[i] + val;
    }
}

void scalar_add_f32_forward(float* dst, const float* src, float val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_add_f32_kernel<<<blocks, threads, 0, stream>>>(dst, src, val, n);
}

__global__ void scalar_add_bf16_kernel(__nv_bfloat16* dst, const __nv_bfloat16* src, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    __nv_bfloat162 val2 = __float2bfloat162_rn(val);
    int n2 = n / 2;
    for (int i = idx; i < n2; i += stride) {
        __nv_bfloat162 v = reinterpret_cast<const __nv_bfloat162*>(src)[i];
        reinterpret_cast<__nv_bfloat162*>(dst)[i] = __hadd2(v, val2);
    }
    if (n % 2 != 0 && idx == 0) {
        dst[n - 1] = __float2bfloat16(__bfloat162float(src[n - 1]) + val);
    }
}

void scalar_add_bf16_forward(__nv_bfloat16* dst, const __nv_bfloat16* src, float val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_add_bf16_kernel<<<blocks, threads, 0, stream>>>(dst, src, val, n);
}

__global__ void scalar_add_f16_kernel(__half* dst, const __half* src, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    __half2 val2 = __float2half2_rn(val);
    int n2 = n / 2;
    for (int i = idx; i < n2; i += stride) {
        __half2 v = reinterpret_cast<const __half2*>(src)[i];
        reinterpret_cast<__half2*>(dst)[i] = __hadd2(v, val2);
    }
    if (n % 2 != 0 && idx == 0) {
        dst[n - 1] = __float2half(__half2float(src[n - 1]) + val);
    }
}

void scalar_add_f16_forward(__half* dst, const __half* src, float val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_add_f16_kernel<<<blocks, threads, 0, stream>>>(dst, src, val, n);
}

// ===================== silu_inplace kernels =====================
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

__device__ __forceinline__ float silu_scalar(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void silu_inplace_f32_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    // float4 vectorized
    int n4 = n / 4;
    for (int i = idx; i < n4; i += stride) {
        float4 v = reinterpret_cast<float4*>(data)[i];
        v.x = silu_scalar(v.x);
        v.y = silu_scalar(v.y);
        v.z = silu_scalar(v.z);
        v.w = silu_scalar(v.w);
        reinterpret_cast<float4*>(data)[i] = v;
    }
    int base = n4 * 4;
    for (int i = base + idx; i < n; i += stride) {
        data[i] = silu_scalar(data[i]);
    }
}

void silu_inplace_f32_forward(float* data, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    silu_inplace_f32_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

__global__ void silu_inplace_bf16_kernel(__nv_bfloat16* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n2 = n / 2;
    for (int i = idx; i < n2; i += stride) {
        __nv_bfloat162 v = reinterpret_cast<__nv_bfloat162*>(data)[i];
        float lo = __bfloat162float(v.x);
        float hi = __bfloat162float(v.y);
        v.x = __float2bfloat16(silu_scalar(lo));
        v.y = __float2bfloat16(silu_scalar(hi));
        reinterpret_cast<__nv_bfloat162*>(data)[i] = v;
    }
    if (n % 2 != 0 && idx == 0) {
        float x = __bfloat162float(data[n - 1]);
        data[n - 1] = __float2bfloat16(silu_scalar(x));
    }
}

void silu_inplace_bf16_forward(__nv_bfloat16* data, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    silu_inplace_bf16_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

__global__ void silu_inplace_f16_kernel(__half* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n2 = n / 2;
    for (int i = idx; i < n2; i += stride) {
        __half2 v = reinterpret_cast<__half2*>(data)[i];
        float lo = __half2float(v.x);
        float hi = __half2float(v.y);
        v.x = __float2half(silu_scalar(lo));
        v.y = __float2half(silu_scalar(hi));
        reinterpret_cast<__half2*>(data)[i] = v;
    }
    if (n % 2 != 0 && idx == 0) {
        float x = __half2float(data[n - 1]);
        data[n - 1] = __float2half(silu_scalar(x));
    }
}

void silu_inplace_f16_forward(__half* data, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    silu_inplace_f16_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

// ===================== tanh_inplace kernels =====================

__global__ void tanh_inplace_f32_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n4 = n / 4;
    for (int i = idx; i < n4; i += stride) {
        float4 v = reinterpret_cast<float4*>(data)[i];
        v.x = tanhf(v.x); v.y = tanhf(v.y); v.z = tanhf(v.z); v.w = tanhf(v.w);
        reinterpret_cast<float4*>(data)[i] = v;
    }
    int base = n4 * 4;
    for (int i = base + idx; i < n; i += stride) {
        data[i] = tanhf(data[i]);
    }
}

void tanh_inplace_f32_forward(float* data, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    tanh_inplace_f32_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

__global__ void tanh_inplace_bf16_kernel(__nv_bfloat16* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n2 = n / 2;
    for (int i = idx; i < n2; i += stride) {
        __nv_bfloat162 v = reinterpret_cast<__nv_bfloat162*>(data)[i];
        v.x = __float2bfloat16(tanhf(__bfloat162float(v.x)));
        v.y = __float2bfloat16(tanhf(__bfloat162float(v.y)));
        reinterpret_cast<__nv_bfloat162*>(data)[i] = v;
    }
    if (n % 2 != 0 && idx == 0) {
        data[n - 1] = __float2bfloat16(tanhf(__bfloat162float(data[n - 1])));
    }
}

void tanh_inplace_bf16_forward(__nv_bfloat16* data, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    tanh_inplace_bf16_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

__global__ void tanh_inplace_f16_kernel(__half* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n2 = n / 2;
    for (int i = idx; i < n2; i += stride) {
        __half2 v = reinterpret_cast<__half2*>(data)[i];
        v.x = __float2half(tanhf(__half2float(v.x)));
        v.y = __float2half(tanhf(__half2float(v.y)));
        reinterpret_cast<__half2*>(data)[i] = v;
    }
    if (n % 2 != 0 && idx == 0) {
        data[n - 1] = __float2half(tanhf(__half2float(data[n - 1])));
    }
}

void tanh_inplace_f16_forward(__half* data, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    tanh_inplace_f16_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

// ===================== scalar_mul_inplace_from_dev kernels =====================
//
// 与 scalar_mul_*_kernel 同构，但从 device 读取标量系数，kernel 参数里不再
// 含随 step 变化的 host float —— 这是 CUDA Graph capture 里唯一能复用的
// 写法（graph 执行时 kernel 参数固定不变，内存里的值可以变）。

__global__ void scalar_mul_inplace_from_dev_f32_kernel(float* x, const float* d_val, int n) {
    float val = *d_val;  // 每线程都读，L1 cache 命中，代价极低
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n4 = n / 4;
    for (int i = idx; i < n4; i += stride) {
        float4 v = reinterpret_cast<float4*>(x)[i];
        v.x *= val; v.y *= val; v.z *= val; v.w *= val;
        reinterpret_cast<float4*>(x)[i] = v;
    }
    int base = n4 * 4;
    for (int i = base + idx; i < n; i += stride) {
        x[i] *= val;
    }
}

void scalar_mul_inplace_from_dev_f32_forward(float* x, const float* d_val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_mul_inplace_from_dev_f32_kernel<<<blocks, threads, 0, stream>>>(x, d_val, n);
}

__global__ void scalar_mul_inplace_from_dev_bf16_kernel(__nv_bfloat16* x, const float* d_val, int n) {
    float val = *d_val;
    __nv_bfloat162 val2 = __float2bfloat162_rn(val);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n2 = n / 2;
    for (int i = idx; i < n2; i += stride) {
        __nv_bfloat162 v = reinterpret_cast<__nv_bfloat162*>(x)[i];
        reinterpret_cast<__nv_bfloat162*>(x)[i] = __hmul2(v, val2);
    }
    if (n % 2 != 0 && idx == 0) {
        x[n - 1] = __float2bfloat16(__bfloat162float(x[n - 1]) * val);
    }
}

void scalar_mul_inplace_from_dev_bf16_forward(__nv_bfloat16* x, const float* d_val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_mul_inplace_from_dev_bf16_kernel<<<blocks, threads, 0, stream>>>(x, d_val, n);
}

__global__ void scalar_mul_inplace_from_dev_f16_kernel(__half* x, const float* d_val, int n) {
    float val = *d_val;
    __half2 val2 = __float2half2_rn(val);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n2 = n / 2;
    for (int i = idx; i < n2; i += stride) {
        __half2 v = reinterpret_cast<__half2*>(x)[i];
        reinterpret_cast<__half2*>(x)[i] = __hmul2(v, val2);
    }
    if (n % 2 != 0 && idx == 0) {
        x[n - 1] = __float2half(__half2float(x[n - 1]) * val);
    }
}

void scalar_mul_inplace_from_dev_f16_forward(__half* x, const float* d_val, int n, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    scalar_mul_inplace_from_dev_f16_kernel<<<blocks, threads, 0, stream>>>(x, d_val, n);
}

// ===================== sinusoid_embedding_from_dev kernels =====================
//
// PyTorch timestep_embedding 语义：
//   half = dim / 2
//   freqs[i] = exp(-ln(10000) * i / half),  i = 0..half
//   arg[i]   = t[0] * freqs[i]
//   emb      = [cos(arg[0..half]) | sin(arg[0..half])]    // [dim]
//
// dim 为 256 量级；单 block、每个线程算一个元素足够。
// 输出 dtype 由 kernel 决定（我们只需 bf16，其它两版本对称提供）。

__device__ __forceinline__ float te_freq(int i, int half) {
    // exp(-log(10000) * i / half) == pow(10000, -i / half)
    constexpr float kLog10000 = 9.21034037f;
    return __expf(-kLog10000 * (float)i / (float)half);
}

__global__ void sinusoid_embedding_from_dev_f32_kernel(float* out, const float* d_t, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    int half = dim >> 1;
    float t = *d_t;
    if (i < half) {
        float arg = t * te_freq(i, half);
        out[i] = __cosf(arg);
    } else {
        int j = i - half;
        float arg = t * te_freq(j, half);
        out[i] = __sinf(arg);
    }
}

void sinusoid_embedding_from_dev_f32_forward(float* out, const float* d_t, int dim, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    sinusoid_embedding_from_dev_f32_kernel<<<blocks, threads, 0, stream>>>(out, d_t, dim);
}

__global__ void sinusoid_embedding_from_dev_bf16_kernel(__nv_bfloat16* out, const float* d_t, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    int half = dim >> 1;
    float t = *d_t;
    float v;
    if (i < half) {
        float arg = t * te_freq(i, half);
        v = __cosf(arg);
    } else {
        int j = i - half;
        float arg = t * te_freq(j, half);
        v = __sinf(arg);
    }
    out[i] = __float2bfloat16(v);
}

void sinusoid_embedding_from_dev_bf16_forward(__nv_bfloat16* out, const float* d_t, int dim, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    sinusoid_embedding_from_dev_bf16_kernel<<<blocks, threads, 0, stream>>>(out, d_t, dim);
}

__global__ void sinusoid_embedding_from_dev_f16_kernel(__half* out, const float* d_t, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    int half = dim >> 1;
    float t = *d_t;
    float v;
    if (i < half) {
        float arg = t * te_freq(i, half);
        v = __cosf(arg);
    } else {
        int j = i - half;
        float arg = t * te_freq(j, half);
        v = __sinf(arg);
    }
    out[i] = __float2half(v);
}

void sinusoid_embedding_from_dev_f16_forward(__half* out, const float* d_t, int dim, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    sinusoid_embedding_from_dev_f16_kernel<<<blocks, threads, 0, stream>>>(out, d_t, dim);
}
