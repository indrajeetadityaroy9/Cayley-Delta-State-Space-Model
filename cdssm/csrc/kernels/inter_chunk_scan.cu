// CDSSM Inter-Chunk Sequential Scan — Complex Diagonal (v2)
//
// Propagates state across chunks via sequential complex diagonal recurrence:
//   state[k+1] = total_A[k] * state[k] + final_local_h[k]
// where * is element-wise complex multiplication (diagonal A).
//
// State is N-dimensional with re/im interleaved: h[2j] = re, h[2j+1] = im.
// A_bar stores conj(mu), so complex multiply gives correct rotation direction.
//
// Grid: (B, H) — one block per (batch, head)
// Block: D threads — each thread owns column d of state[N, D]
//
// BF16 I/O, FP32 compute.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/common.cuh"

namespace cdssm {

// Forward Kernel

__global__ void inter_chunk_scan_fwd_kernel(
    const __nv_bfloat16* __restrict__ total_A,        // (B, NC, H, N)
    const __nv_bfloat16* __restrict__ final_local_h,  // (B, NC, H, N, D)
    __nv_bfloat16* __restrict__ chunk_states,          // (B, NC, H, N, D)
    int B, int NC, int H, int D, int N
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (b >= B || h >= H) return;

    const int d = threadIdx.x;
    if (d >= D) return;

    const int N_half = N / 2;

    // Strides for total_A: (B, NC, H, N)
    const int tA_base = (b * NC * H + h) * N;
    const int tA_step = H * N;

    // Strides for final_local_h and chunk_states: (B, NC, H, N, D)
    const int flh_base = (b * NC * H + h) * N * D;
    const int flh_step = H * N * D;

    // Running state
    float s[MAX_STATE_DIM];
    for (int i = 0; i < N; i++) s[i] = 0.0f;

    for (int k = 0; k < NC; k++) {
        // Store current state BEFORE update
        int cs_off = flh_base + k * flh_step;
        for (int i = 0; i < N; i++) {
            chunk_states[cs_off + i * D + d] = fp32_to_bf16(s[i]);
        }

        // Load total_A[b, k, h, :] (N elements, shared across threads)
        int tA_off = tA_base + k * tA_step;
        float a_vals[MAX_STATE_DIM];
        for (int i = 0; i < N; i++) {
            a_vals[i] = bf16_to_fp32(__ldg(&total_A[tA_off + i]));
        }

        // Load final_local_h[b, k, h, :, d]
        int flh_off = flh_base + k * flh_step;
        float flh_vals[MAX_STATE_DIM];
        for (int i = 0; i < N; i++) {
            flh_vals[i] = bf16_to_fp32(__ldg(&final_local_h[flh_off + i * D + d]));
        }

        // state = A * state + flh (complex diagonal multiply)
        for (int j = 0; j < N_half; j++) {
            float re = a_vals[2 * j];
            float im = a_vals[2 * j + 1];
            float sr = s[2 * j];
            float si = s[2 * j + 1];

            // Complex multiply (stored conj(mu))
            s[2 * j]     = re * sr - im * si + flh_vals[2 * j];
            s[2 * j + 1] = re * si + im * sr + flh_vals[2 * j + 1];
        }
    }
}

// Backward Kernel

__global__ void inter_chunk_scan_bwd_kernel(
    const __nv_bfloat16* __restrict__ grad_chunk_states,
    const __nv_bfloat16* __restrict__ total_A,
    const __nv_bfloat16* __restrict__ chunk_states,
    __nv_bfloat16* __restrict__ grad_total_A,
    __nv_bfloat16* __restrict__ grad_final_local_h,
    int B, int NC, int H, int D, int N
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (b >= B || h >= H) return;

    const int d = threadIdx.x;
    if (d >= D) return;

    const int N_half = N / 2;

    __shared__ float smem[MAX_WARPS * MAX_STATE_DIM];

    const int tA_base = (b * NC * H + h) * N;
    const int tA_step = H * N;
    const int cs_base = (b * NC * H + h) * N * D;
    const int cs_step = H * N * D;

    int num_warps = (D + WARP_SIZE - 1) / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // ds = gradient w.r.t. state after all chunks
    float ds[MAX_STATE_DIM];
    for (int i = 0; i < N; i++) ds[i] = 0.0f;

    for (int k = NC - 1; k >= 0; k--) {
        int tA_off = tA_base + k * tA_step;
        int cs_off = cs_base + k * cs_step;

        // Load A[k]
        float a_vals[MAX_STATE_DIM];
        for (int i = 0; i < N; i++) {
            a_vals[i] = bf16_to_fp32(__ldg(&total_A[tA_off + i]));
        }

        // Load state[k] = chunk_states[k]
        float sk[MAX_STATE_DIM];
        for (int i = 0; i < N; i++) {
            sk[i] = bf16_to_fp32(__ldg(&chunk_states[cs_off + i * D + d]));
        }

        // d/d(flh[k]) = ds
        for (int i = 0; i < N; i++) {
            grad_final_local_h[cs_off + i * D + d] = fp32_to_bf16(ds[i]);
        }

        // d/d(A[k]): For complex diagonal, grad_A[k,2j] and grad_A[k,2j+1]
        // Forward: s_new[2j]   = re*sr - im*si + flh_re
        //          s_new[2j+1] = re*si + im*sr + flh_im
        // d/d(re) = ds_re * sr + ds_im * si (sum over D)
        // d/d(im) = ds_re * (-si) + ds_im * sr (sum over D)
        float gA_local[MAX_STATE_DIM];
        for (int j = 0; j < N_half; j++) {
            float sr = sk[2 * j], si = sk[2 * j + 1];
            float dsr = ds[2 * j], dsi = ds[2 * j + 1];
            gA_local[2 * j]     = dsr * sr + dsi * si;      // d/d(re)
            gA_local[2 * j + 1] = -dsr * si + dsi * sr;     // d/d(im)
        }

        // Reduce gA_local across D threads (batched)
        for (int i = 0; i < N; i++) {
            gA_local[i] = warp_reduce_sum(gA_local[i]);
        }
        if (lane_id == 0) {
            for (int i = 0; i < N; i++) {
                smem[warp_id * MAX_STATE_DIM + i] = gA_local[i];
            }
        }
        __syncthreads();

        if (d == 0) {
            for (int i = 0; i < N; i++) {
                float total = 0.0f;
                for (int w = 0; w < num_warps; w++) {
                    total += smem[w * MAX_STATE_DIM + i];
                }
                grad_total_A[tA_off + i] = fp32_to_bf16(total);
            }
        }
        __syncthreads();

        // d/d(state[k]) = conj(A[k]) * ds + grad_chunk_states[k]
        // conj of stored conj(mu) = mu = (re, -im)
        float gcs[MAX_STATE_DIM];
        for (int i = 0; i < N; i++) {
            gcs[i] = bf16_to_fp32(__ldg(&grad_chunk_states[cs_off + i * D + d]));
        }

        for (int j = 0; j < N_half; j++) {
            float re = a_vals[2 * j];
            float im = a_vals[2 * j + 1];
            float dsr = ds[2 * j];
            float dsi = ds[2 * j + 1];

            // conj(stored) * ds = (re, -im) * (dsr, dsi)
            // = (re*dsr + im*dsi, re*dsi - im*dsr)
            ds[2 * j]     = re * dsr + im * dsi + gcs[2 * j];
            ds[2 * j + 1] = re * dsi - im * dsr + gcs[2 * j + 1];
        }
    }
}


// Launch Functions

torch::Tensor inter_chunk_scan_fwd_cuda(
    torch::Tensor total_A,        // (B, NC, H, N) BF16
    torch::Tensor final_local_h,  // (B, NC, H, N, D) BF16
    int state_dim
) {
    TORCH_CHECK(total_A.is_cuda(), "total_A must be CUDA");
    TORCH_CHECK(final_local_h.is_cuda(), "final_local_h must be CUDA");

    total_A = total_A.contiguous();
    final_local_h = final_local_h.contiguous();

    const int B  = total_A.size(0);
    const int NC = total_A.size(1);
    const int H  = total_A.size(2);
    const int N  = state_dim;
    const int D  = final_local_h.size(4);

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(total_A.device());
    auto chunk_states = torch::empty({B, NC, H, N, D}, opts);

    dim3 grid(B, H);
    dim3 block(D);
    cudaStream_t stream = get_cuda_stream();

    inter_chunk_scan_fwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(total_A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(final_local_h.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(chunk_states.data_ptr()),
        B, NC, H, D, N
    );
    CUDA_CHECK_LAST();

    return chunk_states;
}

std::tuple<torch::Tensor, torch::Tensor>
inter_chunk_scan_bwd_cuda(
    torch::Tensor grad_chunk_states,  // (B, NC, H, N, D)
    torch::Tensor total_A,            // (B, NC, H, N)
    torch::Tensor chunk_states,       // (B, NC, H, N, D)
    int state_dim
) {
    TORCH_CHECK(total_A.is_cuda(), "total_A must be CUDA");

    grad_chunk_states = grad_chunk_states.contiguous();
    total_A = total_A.contiguous();
    chunk_states = chunk_states.contiguous();

    const int B  = total_A.size(0);
    const int NC = total_A.size(1);
    const int H  = total_A.size(2);
    const int N  = state_dim;
    const int D  = chunk_states.size(4);

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(total_A.device());
    auto grad_total_A       = torch::zeros({B, NC, H, N}, opts);
    auto grad_final_local_h = torch::zeros({B, NC, H, N, D}, opts);

    dim3 grid(B, H);
    dim3 block(D);
    size_t smem_size = MAX_WARPS * MAX_STATE_DIM * sizeof(float);
    cudaStream_t stream = get_cuda_stream();

    inter_chunk_scan_bwd_kernel<<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_chunk_states.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(total_A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(chunk_states.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_total_A.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_final_local_h.data_ptr()),
        B, NC, H, D, N
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(grad_total_A, grad_final_local_h);
}

}  // namespace cdssm
