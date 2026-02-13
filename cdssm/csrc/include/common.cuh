// CDSSM CUDA Common Utilities
// Shared utilities for all CDSSM CUDA kernels
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <cassert>

namespace cdssm {

// Error Checking Macros

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            throw std::runtime_error(cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// BF16 <-> FP32 Conversion

__device__ __forceinline__ float bf16_to_fp32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 fp32_to_bf16(float x) {
    return __float2bfloat16(x);
}

// Vector conversions for 2-element loads
__device__ __forceinline__ void bf16x2_to_fp32x2(
    __nv_bfloat162 x, float& out0, float& out1
) {
    out0 = __low2float(x);
    out1 = __high2float(x);
}

__device__ __forceinline__ __nv_bfloat162 fp32x2_to_bf16x2(float x0, float x1) {
    return __floats2bfloat162_rn(x0, x1);
}

// Read-Only Cache Loads (__ldg)
// Use these for read-only data to leverage the texture/L1 cache path

// Load bf16 via read-only cache and convert to fp32
__device__ __forceinline__ float ldg_bf16_to_fp32(const __nv_bfloat16* ptr) {
    return __bfloat162float(__ldg(ptr));
}

// Load bf16x2 (pair) via read-only cache and convert to fp32x2
__device__ __forceinline__ void ldg_bf16x2_to_fp32x2(
    const __nv_bfloat162* ptr, float& out0, float& out1
) {
    __nv_bfloat162 val = __ldg(ptr);
    out0 = __low2float(val);
    out1 = __high2float(val);
}

// Load fp32 via read-only cache
__device__ __forceinline__ float ldg_fp32(const float* ptr) {
    return __ldg(ptr);
}

// FP16 <-> FP32 Conversion (for compatibility)

__device__ __forceinline__ float fp16_to_fp32(__half x) {
    return __half2float(x);
}

__device__ __forceinline__ __half fp32_to_fp16(float x) {
    return __float2half(x);
}

// Atomic Operations

// Native fp32 atomicAdd is supported on all modern GPUs
// For bf16, we use fp32 accumulation (standard practice)

// Atomic add for bf16. H100 (SM 9.0) has native atomicAdd for __nv_bfloat16.
__device__ __forceinline__ void atomicAdd_bf16(
    __nv_bfloat16* addr, float val
) {
    // Native bf16 atomicAdd on Hopper (SM 9.0+)
    atomicAdd(addr, __float2bfloat16(val));
}

// Activation Functions

// SiLU (Swish): x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// SiLU derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
__device__ __forceinline__ float silu_backward(float x, float grad_out) {
    float sig = 1.0f / (1.0f + expf(-x));
    return grad_out * sig * (1.0f + x * (1.0f - sig));
}

// Sigmoid
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Softplus: log(1 + exp(x)), numerically stable
__device__ __forceinline__ float softplus(float x) {
    // For large x, softplus(x) ≈ x to avoid overflow
    if (x > 20.0f) {
        return x;
    }
    // For small x, use log(1 + exp(x))
    return logf(1.0f + expf(x));
}

// Softplus derivative: sigmoid(x)
__device__ __forceinline__ float softplus_backward(float x, float grad_out) {
    return grad_out * sigmoid(x);
}

// Math Utilities

// Integer ceiling division
__host__ __device__ __forceinline__ int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

// Power of 2 check
__host__ __device__ __forceinline__ bool is_power_of_2(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

// Round up to next power of 2
__host__ __device__ __forceinline__ int next_power_of_2(int x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

// Thread/Block Index Utilities

// Flattened thread index within block
__device__ __forceinline__ int flat_thread_idx() {
    return threadIdx.x + threadIdx.y * blockDim.x +
           threadIdx.z * blockDim.x * blockDim.y;
}

// Flattened block index
__device__ __forceinline__ int flat_block_idx() {
    return blockIdx.x + blockIdx.y * gridDim.x +
           blockIdx.z * gridDim.x * gridDim.y;
}

// Total threads per block
__device__ __forceinline__ int threads_per_block() {
    return blockDim.x * blockDim.y * blockDim.z;
}

// Memory Access Helpers

// Compute 3D index offset: (batch, seq, dim)
__device__ __forceinline__ int idx_3d(
    int b, int s, int d,
    int stride_b, int stride_s, int stride_d
) {
    return b * stride_b + s * stride_s + d * stride_d;
}

// Compute 4D index offset: (batch, seq, dim, state)
__device__ __forceinline__ int idx_4d(
    int b, int s, int d, int st,
    int stride_b, int stride_s, int stride_d, int stride_st
) {
    return b * stride_b + s * stride_s + d * stride_d + st * stride_st;
}

// Tensor Shape/Stride Utilities

struct TensorInfo3D {
    int size0, size1, size2;           // Dimensions
    int stride0, stride1, stride2;      // Strides

    __host__ __device__ __forceinline__ int offset(int i0, int i1, int i2) const {
        return i0 * stride0 + i1 * stride1 + i2 * stride2;
    }
};

struct TensorInfo4D {
    int size0, size1, size2, size3;
    int stride0, stride1, stride2, stride3;

    __host__ __device__ __forceinline__ int offset(int i0, int i1, int i2, int i3) const {
        return i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3;
    }
};

// Launch Configuration Helpers

// Select block size based on dimension (matching Triton autotune patterns)
inline int select_block_size(int dim, int max_block = 256) {
    if (dim <= 32) return 32;
    if (dim <= 64) return 64;
    if (dim <= 128) return 128;
    return max_block;
}

// Get CUDA stream from PyTorch
inline cudaStream_t get_cuda_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

// Warp-Level Primitives

constexpr int WARP_SIZE = 32;

// Warp-level sum reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Broadcast value from lane 0 to all lanes
__device__ __forceinline__ float warp_broadcast(float val, int src_lane = 0) {
    return __shfl_sync(0xffffffff, val, src_lane);
}

// Constants
//
// Derivation references (see cdssm/config/defaults.py derive_epsilon_hierarchy):
//
// EPS_DEFAULT (1e-6): Cayley det safety → compute_eps (FP32 machine epsilon).
//   Conservative: det = 1 + (αdt)² + (ωdt)² ≥ 1, so compute_eps suffices.
//   Python config: config.eps_cayley_det.
//
// NORM_EPS_DEFAULT (1e-5): L2 norm safety → io_eps² (BF16 machine epsilon squared).
//   Constraint: rsqrt gradient O(eps^{-1/2}) must be representable in I/O dtype.
//   eps^{-1/2} < 1/io_eps → eps > io_eps² = 6.1e-5. CUDA uses 1e-5 (conservative).
//   Python config: config.eps_norm.
//
// softplus threshold (20.0f in softplus()): Mathematical identity.
//   log(1 + exp(20)) = 20 + 2.1e-9 ≈ 20. Error < FP32 machine epsilon.
//   Not tunable — determined by FP32 precision.

constexpr float EPS_DEFAULT = 1e-6f;
constexpr float NORM_EPS_DEFAULT = 1e-5f;

// Dtype-Aware Epsilon

// Machine epsilon approximations for different precisions
constexpr float MACHINE_EPS_FP32 = 1.19e-7f;
constexpr float MACHINE_EPS_FP16 = 9.77e-4f;
constexpr float MACHINE_EPS_BF16 = 7.81e-3f;

// Get adaptive eps based on whether input is bfloat16
// For bf16: 100 * machine_eps = 0.78 (rounded to safe value)
// For fp32: 100 * machine_eps = 1.19e-5 (use 1e-6 for tighter bound)
__device__ __forceinline__ float get_adaptive_eps(bool is_bf16) {
    return is_bf16 ? 0.78f : 1e-6f;
}

// Get adaptive eps with expected magnitude scaling
__device__ __forceinline__ float get_adaptive_eps_scaled(bool is_bf16, float expected_magnitude) {
    float base_eps = is_bf16 ? MACHINE_EPS_BF16 : MACHINE_EPS_FP32;
    return 100.0f * base_eps * expected_magnitude;
}

}  // namespace cdssm
