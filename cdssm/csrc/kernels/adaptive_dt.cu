// CDSSM Adaptive Timestep — CUDA Implementation
//
// Fuses the AdaptiveTimestep computation:
//   char_freq = alpha + |omega| + eps
//   dt_scale = softplus(log_dt_scale[h])
//   dt_raw = dt_scale / char_freq
//   dt_max = (2 - delta) / (alpha + eps)
//   w = sigmoid((omega_thresh - |omega|) / smoothness)
//   dt = w * min(dt_raw, dt_max) + (1-w) * dt_raw
//
// Grid: (B, cdiv(L, BLOCK_L), H)
// Block: BLOCK_L threads
//
// BF16 I/O, FP32 compute.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/common.cuh"

namespace cdssm {

constexpr int ADT_BLOCK_L = 256;

// Forward Kernel

__global__ void adaptive_dt_fwd_kernel(
    const __nv_bfloat16* __restrict__ alpha_in,       // (B, L, H)
    const __nv_bfloat16* __restrict__ omega_in,       // (B, L, H)
    const float* __restrict__ log_dt_scale,           // (H,) FP32 parameter
    __nv_bfloat16* __restrict__ dt_out,               // (B, L, H)
    float omega_thresh,
    float delta,
    float smoothness,
    float eps,
    int B, int L, int H
) {
    const int b = blockIdx.x;
    const int l_block = blockIdx.y;
    const int h = blockIdx.z;

    const int l = l_block * ADT_BLOCK_L + threadIdx.x;
    if (b >= B || l >= L || h >= H) return;

    const int idx = (b * L + l) * H + h;

    float a = bf16_to_fp32(__ldg(&alpha_in[idx]));
    float w = bf16_to_fp32(__ldg(&omega_in[idx]));

    // Learned scale (per-head, shared across B and L)
    float log_scale = __ldg(&log_dt_scale[h]);
    float dt_scale = softplus(log_scale);

    // Characteristic frequency
    float char_freq = a + fabsf(w) + eps;

    // Raw adaptive dt
    float dt_raw = dt_scale / char_freq;

    // Safety cap: dt_max = (2 - delta) / (alpha + eps)
    float dt_max = (2.0f - delta) / (a + eps);

    // Smooth blend: w → 1 when |omega| << omega_thresh
    float omega_abs = fabsf(w);
    float blend = sigmoid((omega_thresh - omega_abs) / smoothness);

    // Apply cap smoothly
    float dt_capped = fminf(dt_raw, dt_max);
    float dt_val = blend * dt_capped + (1.0f - blend) * dt_raw;

    dt_out[idx] = fp32_to_bf16(dt_val);
}

// Backward Kernel

__global__ void adaptive_dt_bwd_kernel(
    const __nv_bfloat16* __restrict__ grad_dt_in,     // (B, L, H)
    const __nv_bfloat16* __restrict__ alpha_in,       // (B, L, H)
    const __nv_bfloat16* __restrict__ omega_in,       // (B, L, H)
    const float* __restrict__ log_dt_scale,           // (H,)
    float* __restrict__ grad_alpha,                    // (B, L, H)
    float* __restrict__ grad_omega,                    // (B, L, H)
    float* __restrict__ grad_log_dt_scale,            // (H,) — atomically accumulated
    float omega_thresh,
    float delta,
    float smoothness,
    float eps,
    int B, int L, int H
) {
    const int b = blockIdx.x;
    const int l_block = blockIdx.y;
    const int h = blockIdx.z;

    const int l = l_block * ADT_BLOCK_L + threadIdx.x;
    if (b >= B || l >= L || h >= H) return;

    const int idx = (b * L + l) * H + h;

    float g_dt = bf16_to_fp32(__ldg(&grad_dt_in[idx]));
    float a = bf16_to_fp32(__ldg(&alpha_in[idx]));
    float w = bf16_to_fp32(__ldg(&omega_in[idx]));
    float log_scale = __ldg(&log_dt_scale[h]);

    // Recompute forward
    float dt_scale = softplus(log_scale);
    float char_freq = a + fabsf(w) + eps;
    float dt_raw = dt_scale / char_freq;
    float dt_max = (2.0f - delta) / (a + eps);
    float omega_abs = fabsf(w);
    float blend = sigmoid((omega_thresh - omega_abs) / smoothness);
    float dt_capped = fminf(dt_raw, dt_max);
    bool is_capped = dt_raw > dt_max;

    // dt = blend * dt_capped + (1-blend) * dt_raw
    //    = blend * min(dt_raw, dt_max) + (1-blend) * dt_raw

    // d(dt)/d(dt_raw) = blend * (dt_raw <= dt_max ? 1 : 0) + (1-blend)
    //                 = 1 - blend * (dt_raw > dt_max ? 1 : 0)
    float ddt_raw = g_dt * (1.0f - (is_capped ? blend : 0.0f));

    // d(dt)/d(dt_max) = blend * (dt_raw > dt_max ? 1 : 0)
    float ddt_max = g_dt * (is_capped ? blend : 0.0f);

    // d(dt)/d(blend) = dt_capped - dt_raw
    float dblend = g_dt * (dt_capped - dt_raw);

    // d(blend)/d(omega_abs) = -sigmoid'(...) / smoothness = -blend*(1-blend)/smoothness
    float dblend_domega_abs = -blend * (1.0f - blend) / smoothness;

    // d(dt_raw)/d(dt_scale) = 1/char_freq
    float ddt_scale = ddt_raw / char_freq;

    // d(dt_raw)/d(char_freq) = -dt_scale / char_freq^2
    float dchar_freq = ddt_raw * (-dt_scale / (char_freq * char_freq));

    // d(dt_max)/d(alpha) = -(2-delta) / (alpha+eps)^2
    float da_from_max = ddt_max * (-(2.0f - delta) / ((a + eps) * (a + eps)));

    // d(char_freq)/d(alpha) = 1
    // d(char_freq)/d(|omega|) = 1
    float da_from_freq = dchar_freq;
    float domega_abs = dchar_freq + dblend * dblend_domega_abs;

    // d(|omega|)/d(omega) = sign(omega)
    float sign_w = (w >= 0.0f) ? 1.0f : -1.0f;
    float g_omega_val = domega_abs * sign_w;
    float g_alpha_val = da_from_freq + da_from_max;

    // d(dt_scale)/d(log_dt_scale) = sigmoid(log_dt_scale)
    float dlog_scale = ddt_scale * sigmoid(log_scale);

    grad_alpha[idx] = g_alpha_val;
    grad_omega[idx] = g_omega_val;

    // Atomic accumulation for per-head parameter gradient
    atomicAdd(&grad_log_dt_scale[h], dlog_scale);
}


// Launch Functions

torch::Tensor adaptive_dt_fwd_cuda(
    torch::Tensor alpha,         // (B, L, H) BF16
    torch::Tensor omega,         // (B, L, H) BF16
    torch::Tensor log_dt_scale,  // (H,) FP32
    float omega_thresh,
    float delta,
    float smoothness,
    float eps
) {
    TORCH_CHECK(alpha.is_cuda(), "alpha must be CUDA");

    alpha = alpha.contiguous();
    omega = omega.contiguous();
    log_dt_scale = log_dt_scale.contiguous();

    const int B = alpha.size(0);
    const int L = alpha.size(1);
    const int H = alpha.size(2);

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(alpha.device());
    auto dt_out = torch::empty({B, L, H}, opts);

    dim3 grid(B, cdiv(L, ADT_BLOCK_L), H);
    dim3 block(ADT_BLOCK_L);
    cudaStream_t stream = get_cuda_stream();

    adaptive_dt_fwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(alpha.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(omega.data_ptr()),
        log_dt_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dt_out.data_ptr()),
        omega_thresh, delta, smoothness, eps,
        B, L, H
    );
    CUDA_CHECK_LAST();

    return dt_out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
adaptive_dt_bwd_cuda(
    torch::Tensor grad_dt,       // (B, L, H) BF16
    torch::Tensor alpha,         // (B, L, H) BF16
    torch::Tensor omega,         // (B, L, H) BF16
    torch::Tensor log_dt_scale,  // (H,) FP32
    float omega_thresh,
    float delta,
    float smoothness,
    float eps
) {
    TORCH_CHECK(alpha.is_cuda(), "alpha must be CUDA");

    grad_dt = grad_dt.contiguous();
    alpha = alpha.contiguous();
    omega = omega.contiguous();
    log_dt_scale = log_dt_scale.contiguous();

    const int B = alpha.size(0);
    const int L = alpha.size(1);
    const int H = alpha.size(2);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(alpha.device());
    auto grad_alpha = torch::zeros({B, L, H}, opts);
    auto grad_omega = torch::zeros({B, L, H}, opts);
    auto grad_log_dt_scale = torch::zeros({H}, opts);

    dim3 grid(B, cdiv(L, ADT_BLOCK_L), H);
    dim3 block(ADT_BLOCK_L);
    cudaStream_t stream = get_cuda_stream();

    adaptive_dt_bwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_dt.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(alpha.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(omega.data_ptr()),
        log_dt_scale.data_ptr<float>(),
        grad_alpha.data_ptr<float>(),
        grad_omega.data_ptr<float>(),
        grad_log_dt_scale.data_ptr<float>(),
        omega_thresh, delta, smoothness, eps,
        B, L, H
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(grad_alpha, grad_omega, grad_log_dt_scale);
}

}  // namespace cdssm
