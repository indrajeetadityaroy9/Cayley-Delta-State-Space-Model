// KSSM Inter-Chunk Sequential Scan — CUDA Implementation
//
// Propagates state across chunks via sequential 2x2 matrix recurrence:
//   state[k+1] = total_A[k] @ state[k] + final_local_h[k]
//
// Grid: (B, H) — one block per (batch, head)
// Block: D threads — each thread owns column d of state[2,D]
//
// BF16 I/O, FP32 compute.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/common.cuh"
#include "../include/reduction.cuh"

namespace kssm {

// ============================================================================
// Forward Kernel
// ============================================================================

__global__ void inter_chunk_scan_fwd_kernel(
    const __nv_bfloat16* __restrict__ total_A,        // (B, NC, H, 2, 2)
    const __nv_bfloat16* __restrict__ final_local_h,  // (B, NC, H, 2, D)
    __nv_bfloat16* __restrict__ chunk_states,          // (B, NC, H, 2, D)
    int B, int NC, int H, int D
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (b >= B || h >= H) return;

    const int d = threadIdx.x;
    if (d >= D) return;

    // Strides for total_A: (B, NC, H, 2, 2)
    const int tA_base = (b * NC * H + h) * 4;
    const int tA_step = H * 4;  // stride along NC

    // Strides for final_local_h and chunk_states: (B, NC, H, 2, D)
    const int flh_base = (b * NC * H + h) * 2 * D;
    const int flh_step = H * 2 * D;

    // Running state
    float s0 = 0.0f, s1 = 0.0f;

    for (int k = 0; k < NC; k++) {
        // Store current state BEFORE update
        int cs_off = flh_base + k * flh_step;
        chunk_states[cs_off + d]     = fp32_to_bf16(s0);
        chunk_states[cs_off + D + d] = fp32_to_bf16(s1);

        // Load total_A[b, k, h, :, :] (shared across threads)
        int tA_off = tA_base + k * tA_step;
        float a11 = bf16_to_fp32(__ldg(&total_A[tA_off + 0]));
        float a12 = bf16_to_fp32(__ldg(&total_A[tA_off + 1]));
        float a21 = bf16_to_fp32(__ldg(&total_A[tA_off + 2]));
        float a22 = bf16_to_fp32(__ldg(&total_A[tA_off + 3]));

        // Load final_local_h[b, k, h, :, d]
        int flh_off = flh_base + k * flh_step;
        float flh0 = bf16_to_fp32(__ldg(&final_local_h[flh_off + d]));
        float flh1 = bf16_to_fp32(__ldg(&final_local_h[flh_off + D + d]));

        // state = A @ state + flh
        float new_s0 = a11 * s0 + a12 * s1 + flh0;
        float new_s1 = a21 * s0 + a22 * s1 + flh1;
        s0 = new_s0;
        s1 = new_s1;
    }
}

// ============================================================================
// Backward Kernel
// ============================================================================
//
// Forward: state[k+1] = A[k] @ state[k] + flh[k], state[0] = 0
//          chunk_states[k] = state[k] (BEFORE update)
//
// Backward: reverse scan propagating grad_state
//   At chunk k:
//     d/d(flh[k]) = ds
//     d/d(A[k]) = ds outer state[k]  (reduce over D)
//     d/d(state[k]) = A[k]^T @ ds + grad_chunk_states[k]

__global__ void inter_chunk_scan_bwd_kernel(
    const __nv_bfloat16* __restrict__ grad_chunk_states,
    const __nv_bfloat16* __restrict__ total_A,
    const __nv_bfloat16* __restrict__ chunk_states,
    __nv_bfloat16* __restrict__ grad_total_A,
    __nv_bfloat16* __restrict__ grad_final_local_h,
    int B, int NC, int H, int D
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (b >= B || h >= H) return;

    const int d = threadIdx.x;
    if (d >= D) return;

    constexpr int MW = 4;
    __shared__ float smem[MW];

    const int tA_base = (b * NC * H + h) * 4;
    const int tA_step = H * 4;
    const int cs_base = (b * NC * H + h) * 2 * D;
    const int cs_step = H * 2 * D;

    int num_warps = (D + WARP_SIZE - 1) / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // ds = gradient w.r.t. state[NC] (= final state after all chunks)
    // Initialize to 0 since there's no downstream gradient beyond the last chunk
    float ds0 = 0.0f, ds1 = 0.0f;

    for (int k = NC - 1; k >= 0; k--) {
        int tA_off = tA_base + k * tA_step;
        int cs_off = cs_base + k * cs_step;

        // Load A[k]
        float a11 = bf16_to_fp32(__ldg(&total_A[tA_off + 0]));
        float a12 = bf16_to_fp32(__ldg(&total_A[tA_off + 1]));
        float a21 = bf16_to_fp32(__ldg(&total_A[tA_off + 2]));
        float a22 = bf16_to_fp32(__ldg(&total_A[tA_off + 3]));

        // Load state[k] = chunk_states[k] (state BEFORE this chunk's update)
        float sk0 = bf16_to_fp32(__ldg(&chunk_states[cs_off + d]));
        float sk1 = bf16_to_fp32(__ldg(&chunk_states[cs_off + D + d]));

        // ds = d/d(state[k+1]) — gradient from downstream

        // d/d(flh[k]) = ds
        grad_final_local_h[cs_off + d]     = fp32_to_bf16(ds0);
        grad_final_local_h[cs_off + D + d] = fp32_to_bf16(ds1);

        // d/d(A[k]): sum_d(ds[i,d] * state[k,j,d]) for each (i,j)
        // Reduce ds0*sk0, ds0*sk1, ds1*sk0, ds1*sk1 across D threads
        float vals[4] = {ds0 * sk0, ds0 * sk1, ds1 * sk0, ds1 * sk1};
        for (int i = 0; i < 4; i++) {
            float warp_sum = warp_reduce_sum(vals[i]);
            if (lane_id == 0) smem[warp_id] = warp_sum;
            __syncthreads();
            float total = 0.0f;
            for (int w = 0; w < num_warps; w++) total += smem[w];
            __syncthreads();
            vals[i] = total;
        }
        if (d == 0) {
            grad_total_A[tA_off + 0] = fp32_to_bf16(vals[0]);
            grad_total_A[tA_off + 1] = fp32_to_bf16(vals[1]);
            grad_total_A[tA_off + 2] = fp32_to_bf16(vals[2]);
            grad_total_A[tA_off + 3] = fp32_to_bf16(vals[3]);
        }

        // d/d(state[k]) = A[k]^T @ ds + grad_chunk_states[k]
        float gcs0 = bf16_to_fp32(__ldg(&grad_chunk_states[cs_off + d]));
        float gcs1 = bf16_to_fp32(__ldg(&grad_chunk_states[cs_off + D + d]));

        float new_ds0 = a11 * ds0 + a21 * ds1 + gcs0;
        float new_ds1 = a12 * ds0 + a22 * ds1 + gcs1;
        ds0 = new_ds0;
        ds1 = new_ds1;
    }
}


// ============================================================================
// Launch Functions
// ============================================================================

torch::Tensor inter_chunk_scan_fwd_cuda(
    torch::Tensor total_A,        // (B, NC, H, 2, 2) BF16
    torch::Tensor final_local_h   // (B, NC, H, 2, D) BF16
) {
    TORCH_CHECK(total_A.is_cuda(), "total_A must be CUDA");
    TORCH_CHECK(final_local_h.is_cuda(), "final_local_h must be CUDA");

    total_A = total_A.contiguous();
    final_local_h = final_local_h.contiguous();

    const int B  = total_A.size(0);
    const int NC = total_A.size(1);
    const int H  = total_A.size(2);
    const int D  = final_local_h.size(4);

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(total_A.device());
    auto chunk_states = torch::empty({B, NC, H, 2, D}, opts);

    dim3 grid(B, H);
    dim3 block(D);
    cudaStream_t stream = get_cuda_stream();

    inter_chunk_scan_fwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(total_A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(final_local_h.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(chunk_states.data_ptr()),
        B, NC, H, D
    );
    CUDA_CHECK_LAST();

    return chunk_states;
}

std::tuple<torch::Tensor, torch::Tensor>
inter_chunk_scan_bwd_cuda(
    torch::Tensor grad_chunk_states,  // (B, NC, H, 2, D)
    torch::Tensor total_A,            // (B, NC, H, 2, 2)
    torch::Tensor chunk_states        // (B, NC, H, 2, D)
) {
    TORCH_CHECK(total_A.is_cuda(), "total_A must be CUDA");

    grad_chunk_states = grad_chunk_states.contiguous();
    total_A = total_A.contiguous();
    chunk_states = chunk_states.contiguous();

    const int B  = total_A.size(0);
    const int NC = total_A.size(1);
    const int H  = total_A.size(2);
    const int D  = chunk_states.size(4);

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(total_A.device());
    auto grad_total_A      = torch::zeros({B, NC, H, 2, 2}, opts);
    auto grad_final_local_h = torch::zeros({B, NC, H, 2, D}, opts);

    dim3 grid(B, H);
    dim3 block(D);
    cudaStream_t stream = get_cuda_stream();

    inter_chunk_scan_bwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_chunk_states.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(total_A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(chunk_states.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_total_A.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_final_local_h.data_ptr()),
        B, NC, H, D
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(grad_total_A, grad_final_local_h);
}

}  // namespace kssm
