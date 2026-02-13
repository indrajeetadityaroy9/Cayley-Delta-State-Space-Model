// CDSSM Fused K/Q L2 Normalization — CUDA
//
// Fuses F.normalize(K, dim=-1) and F.normalize(Q, dim=-1) into a single kernel.
//
// Each (b, l, h) position owns a D-dimensional K vector and D-dimensional Q vector.
// We normalize both in one kernel launch using warp-level reduction for the
// sum-of-squares computation.
//
// Grid: (B * L, H)
// Block: D threads (D = head_dim = 64, which is exactly 2 warps)
//
// BF16 I/O, FP32 compute.
// Forward: Normalize K and Q in-place (conceptually; writes to output tensors).
// Backward: d/dx(x/||x||) = (I - x_hat * x_hat^T) / ||x|| applied per-vector.
//
// We save K_norm and Q_norm (the normalized outputs) for backward recomputation.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/common.cuh"

namespace cdssm {

// Forward Kernel

__global__ void normalize_kq_fwd_kernel(
    const __nv_bfloat16* __restrict__ K_in,     // (B, L, H, D) BF16
    const __nv_bfloat16* __restrict__ Q_in,     // (B, L, H, D) BF16
    __nv_bfloat16* __restrict__ K_out,           // (B, L, H, D) BF16
    __nv_bfloat16* __restrict__ Q_out,           // (B, L, H, D) BF16
    int B, int L, int H, int D
) {
    // Grid: (B*L, H), Block: D threads
    const int bl = blockIdx.x;           // flattened (b, l)
    const int h = blockIdx.y;
    const int d = threadIdx.x;

    if (bl >= B * L || h >= H || d >= D) return;

    // Linear index for (bl, h, d) in (B*L, H, D) layout
    const int base_idx = (bl * H + h) * D;

    // Load K[bl, h, d] and Q[bl, h, d]
    float k_val = bf16_to_fp32(__ldg(&K_in[base_idx + d]));
    float q_val = bf16_to_fp32(__ldg(&Q_in[base_idx + d]));

    // Compute sum of squares for K using warp reduction
    // D=64 = 2 warps, so we need cross-warp reduction via shared memory
    float k_sq = k_val * k_val;
    float q_sq = q_val * q_val;

    // Shared memory for cross-warp reduction (2 warps * 2 values)
    __shared__ float smem[4];  // [k_warp0, k_warp1, q_warp0, q_warp1]

    // Warp-level reduction
    float k_warp_sum = warp_reduce_sum(k_sq);
    float q_warp_sum = warp_reduce_sum(q_sq);

    int warp_id = d / WARP_SIZE;  // 0 or 1
    int lane_id = d % WARP_SIZE;

    if (lane_id == 0) {
        smem[warp_id] = k_warp_sum;
        smem[2 + warp_id] = q_warp_sum;
    }
    __syncthreads();

    // Final reduction (both warps read the two partial sums)
    float k_sum_sq = smem[0] + smem[1];
    float q_sum_sq = smem[2] + smem[3];

    // L2 normalize
    float k_inv_norm = rsqrtf(k_sum_sq + EPS_DEFAULT);
    float q_inv_norm = rsqrtf(q_sum_sq + EPS_DEFAULT);

    K_out[base_idx + d] = fp32_to_bf16(k_val * k_inv_norm);
    Q_out[base_idx + d] = fp32_to_bf16(q_val * q_inv_norm);
}


// Backward Kernel
// d/dx(x/||x||) = (I/||x|| - x_hat * x_hat^T / ||x||) = (1/||x||) * (I - x_hat * x_hat^T)
// For a single component: grad_in[d] = (grad_out[d] - x_hat[d] * dot(grad_out, x_hat)) / ||x||
// But we can also express it as: grad_in[d] = (grad_out[d] - x_hat[d] * dot(grad_out, x_hat)) * inv_norm
// where inv_norm = 1/||x|| and x_hat = x * inv_norm

__global__ void normalize_kq_bwd_kernel(
    const __nv_bfloat16* __restrict__ grad_K_out, // (B, L, H, D)
    const __nv_bfloat16* __restrict__ grad_Q_out, // (B, L, H, D)
    const __nv_bfloat16* __restrict__ K_in,        // (B, L, H, D) original input
    const __nv_bfloat16* __restrict__ Q_in,        // (B, L, H, D) original input
    __nv_bfloat16* __restrict__ grad_K_in,         // (B, L, H, D)
    __nv_bfloat16* __restrict__ grad_Q_in,         // (B, L, H, D)
    int B, int L, int H, int D
) {
    const int bl = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;

    if (bl >= B * L || h >= H || d >= D) return;

    const int base_idx = (bl * H + h) * D;

    // Load inputs
    float k_val = bf16_to_fp32(__ldg(&K_in[base_idx + d]));
    float q_val = bf16_to_fp32(__ldg(&Q_in[base_idx + d]));
    float gk_out = bf16_to_fp32(__ldg(&grad_K_out[base_idx + d]));
    float gq_out = bf16_to_fp32(__ldg(&grad_Q_out[base_idx + d]));

    // Recompute norms
    float k_sq = k_val * k_val;
    float q_sq = q_val * q_val;

    __shared__ float smem[4];
    float k_warp_sum = warp_reduce_sum(k_sq);
    float q_warp_sum = warp_reduce_sum(q_sq);

    int warp_id = d / WARP_SIZE;
    int lane_id = d % WARP_SIZE;

    if (lane_id == 0) {
        smem[warp_id] = k_warp_sum;
        smem[2 + warp_id] = q_warp_sum;
    }
    __syncthreads();

    float k_sum_sq = smem[0] + smem[1];
    float q_sum_sq = smem[2] + smem[3];

    float k_inv_norm = rsqrtf(k_sum_sq + EPS_DEFAULT);
    float q_inv_norm = rsqrtf(q_sum_sq + EPS_DEFAULT);

    // Normalized vectors
    float k_hat = k_val * k_inv_norm;
    float q_hat = q_val * q_inv_norm;

    // dot(grad_out, x_hat) via reduction
    float k_dot = gk_out * k_hat;
    float q_dot = gq_out * q_hat;

    float k_dot_warp = warp_reduce_sum(k_dot);
    float q_dot_warp = warp_reduce_sum(q_dot);

    // Reuse smem for dot products
    __syncthreads();
    if (lane_id == 0) {
        smem[warp_id] = k_dot_warp;
        smem[2 + warp_id] = q_dot_warp;
    }
    __syncthreads();

    float k_dot_total = smem[0] + smem[1];
    float q_dot_total = smem[2] + smem[3];

    // grad_in = (grad_out - x_hat * dot(grad_out, x_hat)) * inv_norm
    float gk_in = (gk_out - k_hat * k_dot_total) * k_inv_norm;
    float gq_in = (gq_out - q_hat * q_dot_total) * q_inv_norm;

    grad_K_in[base_idx + d] = fp32_to_bf16(gk_in);
    grad_Q_in[base_idx + d] = fp32_to_bf16(gq_in);
}


// Launch Functions

std::tuple<torch::Tensor, torch::Tensor>
normalize_kq_fwd_cuda(
    torch::Tensor K,   // (B, L, H, D) BF16
    torch::Tensor Q    // (B, L, H, D) BF16
) {
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");

    K = K.contiguous();
    Q = Q.contiguous();

    const int B = K.size(0);
    const int L = K.size(1);
    const int H = K.size(2);
    const int D = K.size(3);

    TORCH_CHECK(D <= 1024, "head_dim must be <= 1024");

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(K.device());
    auto K_out = torch::empty_like(K);
    auto Q_out = torch::empty_like(Q);

    dim3 grid(B * L, H);
    dim3 block(D);
    cudaStream_t stream = get_cuda_stream();

    normalize_kq_fwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(K_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(Q_out.data_ptr()),
        B, L, H, D
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(K_out, Q_out);
}

std::tuple<torch::Tensor, torch::Tensor>
normalize_kq_bwd_cuda(
    torch::Tensor grad_K_out,   // (B, L, H, D) BF16
    torch::Tensor grad_Q_out,   // (B, L, H, D) BF16
    torch::Tensor K,            // (B, L, H, D) BF16 original input
    torch::Tensor Q             // (B, L, H, D) BF16 original input
) {
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");

    grad_K_out = grad_K_out.contiguous();
    grad_Q_out = grad_Q_out.contiguous();
    K = K.contiguous();
    Q = Q.contiguous();

    const int B = K.size(0);
    const int L = K.size(1);
    const int H = K.size(2);
    const int D = K.size(3);

    auto grad_K_in = torch::empty_like(K);
    auto grad_Q_in = torch::empty_like(Q);

    dim3 grid(B * L, H);
    dim3 block(D);
    cudaStream_t stream = get_cuda_stream();

    normalize_kq_bwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_K_out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(grad_Q_out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_K_in.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_Q_in.data_ptr()),
        B, L, H, D
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(grad_K_in, grad_Q_in);
}

}  // namespace cdssm
