// CDSSM CUDA Common Utilities
// Shared utilities, constants, and primitives for all CDSSM CUDA kernels.
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <cassert>

namespace cdssm {


// Compile-Time Limits

//
// These constants define the maximum supported dimensions for register-resident
// state arrays. Runtime dimensions (N, D) are passed as kernel parameters and
// validated via TORCH_CHECK at launch time.
//
// MAX_STATE_DIM: Maximum N (state dimension). Must be even for complex pairs.
//   N/2 independent complex eigenvalue pairs per head. Typical: 8, 16, 32.
//
// MAX_HEAD_DIM: Maximum D (head dimension / key-query dimension).
//   Determines block size for scan kernels (one thread per D element).
//   Must be a multiple of WARP_SIZE.
//
// MAX_WARPS: Derived from MAX_HEAD_DIM / WARP_SIZE. Used for shared memory
//   allocation in cross-warp reductions.

constexpr int WARP_SIZE     = 32;
constexpr int MAX_STATE_DIM = 32;
constexpr int MAX_HEAD_DIM  = 128;
constexpr int MAX_WARPS     = MAX_HEAD_DIM / WARP_SIZE;  // 4


// Error Checking


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
// For x > 20, softplus(x) ≈ x (error < FP32 machine epsilon).
__device__ __forceinline__ float softplus(float x) {
    if (x > 20.0f) return x;
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


// Warp-Level Primitives


// Warp-level sum reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


// Stream Utility


// Get CUDA stream from PyTorch
inline cudaStream_t get_cuda_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}


// Epsilon Constants

//
// Derivation references (see cdssm/config/model.py derive_epsilon_hierarchy):
//
// EPS_DEFAULT (1e-6): Cayley determinant safety.
//   det = 1 + (αdt)² + (ωdt)² ≥ 1, so FP32 machine epsilon suffices.
//   Conservative value used for numerical stability.
//
// softplus threshold (20.0f in softplus()): Mathematical identity.
//   log(1 + exp(20)) = 20 + 2.1e-9 ≈ 20. Error < FP32 machine epsilon.

constexpr float EPS_DEFAULT = 1e-6f;

}  // namespace cdssm
