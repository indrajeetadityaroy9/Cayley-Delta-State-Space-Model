// KSSM Conv1d + SiLU Kernel - CUDA Implementation
//
// Fuses 4 kernel launches into 1:
// 1. Transpose (B, S, D) -> (B, D, S)
// 2. Conv1d (depthwise, kernel_size=4)
// 3. Transpose (B, D, S) -> (B, S, D)
// 4. SiLU activation
//
// Key optimizations:
// - No intermediate tensors (transpose is implicit in indexing)
// - Depthwise conv is a 4-tap FIR filter per channel
// - Causal padding handled in-kernel
// - SiLU computed in-place before output write
//
// Port of kssm/kernels/conv1d_silu.py

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/common.cuh"
#include "../include/reduction.cuh"

namespace kssm {

// ============================================================================
// Forward Kernel
// ============================================================================

template<int BLOCK_S, int BLOCK_D>
__global__ void conv1d_silu_fwd_kernel(
    // Inputs
    const __nv_bfloat16* __restrict__ x,       // (batch, seq, d_inner)
    const __nv_bfloat16* __restrict__ weight,  // (d_inner, kernel_size) flattened from (d_inner, 1, kernel_size)
    const __nv_bfloat16* __restrict__ bias,    // (d_inner,)
    // Output
    __nv_bfloat16* __restrict__ out,           // (batch, seq, d_inner)
    // Dimensions
    int batch,
    int seq_len,
    int d_inner,
    int kernel_size,
    // Strides for x and out (batch, seq, d_inner)
    int stride_xb, int stride_xs, int stride_xd,
    int stride_ob, int stride_os, int stride_od
) {
    // Grid: (batch, cdiv(seq_len, BLOCK_S), cdiv(d_inner, BLOCK_D))
    int pid_b = blockIdx.x;   // Batch index
    int pid_s = blockIdx.y;   // Sequence block index
    int pid_d = blockIdx.z;   // Feature block index

    int d_local = threadIdx.x;
    int d_offset = pid_d * BLOCK_D + d_local;

    if (d_offset >= d_inner) return;

    // Load bias (one per channel)
    float bias_val = bf16_to_fp32(bias[d_offset]);

    // Load weights (kernel_size=4 per channel)
    // Weight shape is (d_inner, kernel_size), stored contiguously
    float w0 = bf16_to_fp32(weight[d_offset * kernel_size + 0]);
    float w1 = bf16_to_fp32(weight[d_offset * kernel_size + 1]);
    float w2 = bf16_to_fp32(weight[d_offset * kernel_size + 2]);
    float w3 = bf16_to_fp32(weight[d_offset * kernel_size + 3]);

    // Process each sequence position in the block
    int base = pid_b * stride_xb + d_offset * stride_xd;

    #pragma unroll 4
    for (int s_idx = 0; s_idx < BLOCK_S; s_idx++) {
        int s = pid_s * BLOCK_S + s_idx;
        if (s >= seq_len) break;

        // Compute convolution with causal padding
        // For kernel_size=4: output[t] = w0*x[t-3] + w1*x[t-2] + w2*x[t-1] + w3*x[t]
        // where x[i] = 0 for i < 0

        // Load input values with boundary checks
        float x0 = (s >= 3) ? bf16_to_fp32(x[base + (s - 3) * stride_xs]) : 0.0f;
        float x1 = (s >= 2) ? bf16_to_fp32(x[base + (s - 2) * stride_xs]) : 0.0f;
        float x2 = (s >= 1) ? bf16_to_fp32(x[base + (s - 1) * stride_xs]) : 0.0f;
        float x3 = bf16_to_fp32(x[base + s * stride_xs]);

        // Convolution: y = w0*x[t-3] + w1*x[t-2] + w2*x[t-1] + w3*x[t] + bias
        float y = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3 + bias_val;

        // SiLU activation: silu(y) = y * sigmoid(y)
        float out_val = silu(y);

        // Store output
        int out_idx = pid_b * stride_ob + s * stride_os + d_offset * stride_od;
        out[out_idx] = fp32_to_bf16(out_val);
    }
}

// ============================================================================
// Backward Kernel
// ============================================================================

template<int BLOCK_S, int BLOCK_D>
__global__ void conv1d_silu_bwd_kernel(
    // Inputs
    const __nv_bfloat16* __restrict__ x,         // (batch, seq, d_inner)
    const __nv_bfloat16* __restrict__ weight,    // (d_inner, kernel_size)
    const __nv_bfloat16* __restrict__ bias,      // (d_inner,)
    const __nv_bfloat16* __restrict__ grad_out,  // (batch, seq, d_inner)
    // Outputs
    __nv_bfloat16* __restrict__ grad_x,  // (batch, seq, d_inner) - direct write (gather pattern)
    float* __restrict__ grad_weight,     // (d_inner, kernel_size) - atomically accumulated
    float* __restrict__ grad_bias,       // (d_inner,) - atomically accumulated
    // Dimensions
    int batch,
    int seq_len,
    int d_inner,
    int kernel_size,
    // Strides
    int stride_xb, int stride_xs, int stride_xd,
    int stride_ob, int stride_os, int stride_od
) {
    // Grid: (batch, cdiv(seq_len, BLOCK_S), cdiv(d_inner, BLOCK_D))
    int pid_b = blockIdx.x;
    int pid_s = blockIdx.y;
    int pid_d = blockIdx.z;

    int d_local = threadIdx.x;
    int d_offset = pid_d * BLOCK_D + d_local;

    if (d_offset >= d_inner) return;

    // Load weights and bias
    float w0 = bf16_to_fp32(weight[d_offset * kernel_size + 0]);
    float w1 = bf16_to_fp32(weight[d_offset * kernel_size + 1]);
    float w2 = bf16_to_fp32(weight[d_offset * kernel_size + 2]);
    float w3 = bf16_to_fp32(weight[d_offset * kernel_size + 3]);
    float bias_val = bf16_to_fp32(bias[d_offset]);

    int x_base = pid_b * stride_xb + d_offset * stride_xd;
    int go_base = pid_b * stride_ob + d_offset * stride_od;

    // Accumulators for weight gradients (reduce atomic contention)
    float grad_w0_acc = 0.0f;
    float grad_w1_acc = 0.0f;
    float grad_w2_acc = 0.0f;
    float grad_w3_acc = 0.0f;
    float grad_bias_acc = 0.0f;

    // Helper: compute grad_y (gradient through SiLU of conv output) at position t
    // conv_out[t] = w0*x[t-3] + w1*x[t-2] + w2*x[t-1] + w3*x[t] + bias
    // grad_y[t] = grad_out[t] * dsilu_dy(conv_out[t])
    #define LOAD_X_AT(t) ((t) >= 0 && (t) < seq_len ? bf16_to_fp32(x[x_base + (t) * stride_xs]) : 0.0f)
    #define LOAD_GO_AT(t) ((t) >= 0 && (t) < seq_len ? bf16_to_fp32(grad_out[go_base + (t) * stride_os]) : 0.0f)

    #define COMPUTE_GRAD_Y(t, result) { \
        float _x0 = LOAD_X_AT((t) - 3); \
        float _x1 = LOAD_X_AT((t) - 2); \
        float _x2 = LOAD_X_AT((t) - 1); \
        float _x3 = LOAD_X_AT(t); \
        float _y = w0 * _x0 + w1 * _x1 + w2 * _x2 + w3 * _x3 + bias_val; \
        float _sig = sigmoid(_y); \
        float _dsilu = _sig * (1.0f + _y * (1.0f - _sig)); \
        result = LOAD_GO_AT(t) * _dsilu; \
    }

    #pragma unroll 4
    for (int s_idx = 0; s_idx < BLOCK_S; s_idx++) {
        int s = pid_s * BLOCK_S + s_idx;
        if (s >= seq_len) break;

        // === Gather-based grad_x computation ===
        // grad_x[s] = sum over k where output position (s+k) used x[s] with weight w[3-k]
        // conv_out[s]   uses x[s] with w3  =>  grad_y[s]   * w3
        // conv_out[s+1] uses x[s] with w2  =>  grad_y[s+1] * w2
        // conv_out[s+2] uses x[s] with w1  =>  grad_y[s+2] * w1
        // conv_out[s+3] uses x[s] with w0  =>  grad_y[s+3] * w0
        float grad_x_val = 0.0f;

        // grad_y at position s (always valid since s < seq_len)
        float gy_s;
        COMPUTE_GRAD_Y(s, gy_s);
        grad_x_val += gy_s * w3;

        if (s + 1 < seq_len) {
            float gy_s1;
            COMPUTE_GRAD_Y(s + 1, gy_s1);
            grad_x_val += gy_s1 * w2;
        }
        if (s + 2 < seq_len) {
            float gy_s2;
            COMPUTE_GRAD_Y(s + 2, gy_s2);
            grad_x_val += gy_s2 * w1;
        }
        if (s + 3 < seq_len) {
            float gy_s3;
            COMPUTE_GRAD_Y(s + 3, gy_s3);
            grad_x_val += gy_s3 * w0;
        }

        // Direct write (no atomicAdd needed - each (batch, s, d_offset) is unique)
        int gx_idx = pid_b * stride_ob + s * stride_os + d_offset * stride_od;
        grad_x[gx_idx] = fp32_to_bf16(grad_x_val);

        // === Weight and bias gradients (same as before, using gy_s from position s) ===
        float x0 = LOAD_X_AT(s - 3);
        float x1 = LOAD_X_AT(s - 2);
        float x2 = LOAD_X_AT(s - 1);
        float x3 = LOAD_X_AT(s);

        grad_bias_acc += gy_s;
        grad_w0_acc += gy_s * x0;
        grad_w1_acc += gy_s * x1;
        grad_w2_acc += gy_s * x2;
        grad_w3_acc += gy_s * x3;
    }

    #undef LOAD_X_AT
    #undef LOAD_GO_AT
    #undef COMPUTE_GRAD_Y

    // Atomic add accumulated weight/bias gradients
    atomicAdd(&grad_bias[d_offset], grad_bias_acc);
    atomicAdd(&grad_weight[d_offset * kernel_size + 0], grad_w0_acc);
    atomicAdd(&grad_weight[d_offset * kernel_size + 1], grad_w1_acc);
    atomicAdd(&grad_weight[d_offset * kernel_size + 2], grad_w2_acc);
    atomicAdd(&grad_weight[d_offset * kernel_size + 3], grad_w3_acc);
}

// ============================================================================
// Kernel Launch Functions
// ============================================================================

torch::Tensor conv1d_silu_fwd_cuda(
    torch::Tensor x,       // (batch, seq, d_inner)
    torch::Tensor weight,  // (d_inner, 1, kernel_size)
    torch::Tensor bias     // (d_inner,)
) {
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA tensor");

    // Ensure contiguous
    x = x.contiguous();

    int batch = x.size(0);
    int seq_len = x.size(1);
    int d_inner = x.size(2);
    int kernel_size = weight.size(2);

    TORCH_CHECK(kernel_size == 4, "Only kernel_size=4 is supported");

    // Flatten weight for easier indexing: (d_inner, kernel_size)
    auto weight_flat = weight.view({d_inner, kernel_size}).contiguous();

    // Allocate output
    auto out = torch::empty({batch, seq_len, d_inner},
                            torch::TensorOptions()
                                .dtype(torch::kBFloat16)
                                .device(x.device()));

    // Select block sizes (matching Triton autotune configs)
    const int BLOCK_S = 64;
    const int BLOCK_D = 128;

    dim3 grid(batch, cdiv(seq_len, BLOCK_S), cdiv(d_inner, BLOCK_D));
    dim3 block(BLOCK_D);

    cudaStream_t stream = get_cuda_stream();

    conv1d_silu_fwd_kernel<BLOCK_S, BLOCK_D><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        batch, seq_len, d_inner, kernel_size,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2)
    );

    CUDA_CHECK_LAST();

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
conv1d_silu_bwd_cuda(
    torch::Tensor x,         // (batch, seq, d_inner)
    torch::Tensor weight,    // (d_inner, 1, kernel_size)
    torch::Tensor bias,      // (d_inner,)
    torch::Tensor grad_out   // (batch, seq, d_inner)
) {
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA tensor");

    // Ensure contiguous
    x = x.contiguous();
    grad_out = grad_out.contiguous();

    int batch = x.size(0);
    int seq_len = x.size(1);
    int d_inner = x.size(2);
    int kernel_size = weight.size(2);

    TORCH_CHECK(kernel_size == 4, "Only kernel_size=4 is supported");

    auto weight_flat = weight.view({d_inner, kernel_size}).contiguous();

    // grad_x uses gather pattern (direct write), so empty is sufficient
    auto grad_x = torch::empty({batch, seq_len, d_inner},
                               torch::TensorOptions()
                                   .dtype(torch::kBFloat16)
                                   .device(x.device()));
    auto grad_weight = torch::zeros({d_inner, kernel_size},
                                    torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(x.device()));
    auto grad_bias = torch::zeros({d_inner},
                                  torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(x.device()));

    // Select block sizes
    const int BLOCK_S = 64;
    const int BLOCK_D = 128;

    dim3 grid(batch, cdiv(seq_len, BLOCK_S), cdiv(d_inner, BLOCK_D));
    dim3 block(BLOCK_D);

    cudaStream_t stream = get_cuda_stream();

    conv1d_silu_bwd_kernel<BLOCK_S, BLOCK_D><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_x.data_ptr()),
        grad_weight.data_ptr<float>(),
        grad_bias.data_ptr<float>(),
        batch, seq_len, d_inner, kernel_size,
        x.stride(0), x.stride(1), x.stride(2),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2)
    );

    CUDA_CHECK_LAST();

    // Reshape grad_weight back to (d_inner, 1, kernel_size)
    grad_weight = grad_weight.view({d_inner, 1, kernel_size});

    // Convert weight/bias grads to input dtypes; grad_x is already bf16
    return std::make_tuple(
        grad_x,
        grad_weight.to(weight.dtype()),
        grad_bias.to(bias.dtype())
    );
}

}  // namespace kssm
