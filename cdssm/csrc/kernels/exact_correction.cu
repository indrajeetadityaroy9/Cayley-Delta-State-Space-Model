// CDSSM v2 Exact Correction — Chunk Initial State Propagation
//
// Fuses the "exact correction" step of the chunkwise parallel scan.
// Propagates chunk initial states through dynamics (erasure + rotation)
// WITHOUT injection (no V terms). Stores the intermediate states as
// corrections that are later added to the local scan results.
//
// Algorithm (per chunk, per head):
//   Initialize h from chunk_states[b, nc, h, :, d]
//   For each timestep t in 0..C-1:
//     corrections[b, nc, t, h, :, d] = h   (state BEFORE update)
//     kTh[s] = sum_d(h[s,d] * k[d])        (batched retrieval: 2 syncthreads)
//     h[s,d] -= beta * kTh[s] * k[d]       (selective erasure)
//     Complex diagonal rotation:
//       h_new[2j]   = re*h[2j]   - im*h[2j+1]
//       h_new[2j+1] = re*h[2j+1] + im*h[2j]
//
// Grid: (B*NC, H)  — one block per (chunk, head)
// Block: D threads  — each thread owns column d of the (N,D) state
//
// BF16 I/O, FP32 compute. Register-resident state.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/common.cuh"

namespace cdssm {


// Forward Kernel


__global__ void exact_correction_fwd_kernel(
    // Inputs (BF16, contiguous)
    const __nv_bfloat16* __restrict__ A_chunk,       // (B, NC, C, H, N)
    const __nv_bfloat16* __restrict__ K_chunk,       // (B, NC, C, H, D)
    const __nv_bfloat16* __restrict__ beta_chunk,    // (B, NC, C, H)
    const __nv_bfloat16* __restrict__ chunk_states,  // (B, NC, H, N, D)
    // Output (BF16)
    __nv_bfloat16* __restrict__ corrections,         // (B, NC, C, H, N, D)
    // Dimensions
    int B, int NC, int C, int H, int D, int N
) {
    // blockIdx.x = b * NC + nc, blockIdx.y = head
    const int bnc_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    if (bnc_idx >= B * NC || head_idx >= H) return;

    const int b  = bnc_idx / NC;
    const int nc = bnc_idx % NC;

    const int d = threadIdx.x;
    if (d >= D) return;

    __shared__ float smem[MAX_WARPS * MAX_STATE_DIM];

    const int warp_id   = d / WARP_SIZE;
    const int lane_id   = d % WARP_SIZE;
    const int num_warps = (D + WARP_SIZE - 1) / WARP_SIZE;

    // Stride computations for 5D contiguous tensors
    // A_chunk: (B, NC, C, H, N)
    const int A_base = ((b * NC + nc) * C * H + head_idx) * N;
    const int A_step = H * N;

    // K_chunk: (B, NC, C, H, D)
    const int K_base = ((b * NC + nc) * C * H + head_idx) * D;
    const int K_step = H * D;

    // beta_chunk: (B, NC, C, H)
    const int b_base = (b * NC + nc) * C * H + head_idx;
    const int b_step = H;

    // chunk_states: (B, NC, H, N, D)
    const int cs_base = ((b * NC + nc) * H + head_idx) * N * D;

    // corrections: (B, NC, C, H, N, D)
    const int corr_base = ((b * NC + nc) * C * H + head_idx) * N * D;
    const int corr_step = H * N * D;

    // Register-resident state: h[N], initialized from chunk_states
    float h[MAX_STATE_DIM];
    for (int s = 0; s < N; s++) {
        h[s] = bf16_to_fp32(__ldg(&chunk_states[cs_base + s * D + d]));
    }

    for (int t = 0; t < C; t++) {
        // Store corrections BEFORE erasure/rotation
        int corr_i = corr_base + t * corr_step;
        for (int s = 0; s < N; s++) {
            corrections[corr_i + s * D + d] = fp32_to_bf16(h[s]);
        }

        // Load A_bar[t]
        int Ai = A_base + t * A_step;
        float a[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            a[s] = bf16_to_fp32(__ldg(&A_chunk[Ai + s]));
        }

        // Load k[t,d] and beta[t]
        float k_d  = bf16_to_fp32(__ldg(&K_chunk[K_base + t * K_step + d]));
        float beta = bf16_to_fp32(__ldg(&beta_chunk[b_base + t * b_step]));

        // 1. Batched kTh reduction: kTh[s] = sum_d(h[s,d] * k[d])
        float kTh_local[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            kTh_local[s] = h[s] * k_d;
        }
        // Warp-level reduction
        for (int s = 0; s < N; s++) {
            kTh_local[s] = warp_reduce_sum(kTh_local[s]);
        }
        // Write warp results to shared memory
        if (lane_id == 0) {
            for (int s = 0; s < N; s++) {
                smem[warp_id * MAX_STATE_DIM + s] = kTh_local[s];
            }
        }
        __syncthreads();

        // Cross-warp reduction
        float kTh[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            kTh[s] = 0.0f;
            #pragma unroll
            for (int w = 0; w < MAX_WARPS; w++) {
                if (w < num_warps) kTh[s] += smem[w * MAX_STATE_DIM + s];
            }
        }
        __syncthreads();

        // 2. Selective erasure
        for (int s = 0; s < N; s++) {
            h[s] -= beta * kTh[s] * k_d;
        }

        // 3. Complex diagonal rotation (pairs j=0..N/2-1)
        //    A_bar stores conj(mu): re/im interleaved
        for (int j = 0; j < N / 2; j++) {
            float re = a[2 * j];
            float im = a[2 * j + 1];
            float h_re = h[2 * j];
            float h_im = h[2 * j + 1];
            h[2 * j]     = re * h_re - im * h_im;
            h[2 * j + 1] = re * h_im + im * h_re;
        }
    }
}


// Backward Kernel

//
// Reverse-mode sequential scan through the chunk.
//
// Forward step t:
//   h_before = corrections[t]              (state before update at step t)
//   kTh[s] = sum_d(h_before[s,d] * k[d])
//   h_erased[s,d] = h_before[s,d] - beta * kTh[s] * k[d]
//   h_after[2j,d]   = re*h_erased[2j,d]   - im*h_erased[2j+1,d]
//   h_after[2j+1,d] = re*h_erased[2j+1,d] + im*h_erased[2j,d]
//
// Backward: propagate dh (gradient w.r.t. state leaving step t) in reverse.
// At the end, remaining dh flows into grad_chunk_states.

__global__ void exact_correction_bwd_kernel(
    // Upstream gradients
    const __nv_bfloat16* __restrict__ grad_corrections,  // (B, NC, C, H, N, D)
    // Saved from forward
    const __nv_bfloat16* __restrict__ A_chunk,           // (B, NC, C, H, N)
    const __nv_bfloat16* __restrict__ K_chunk,           // (B, NC, C, H, D)
    const __nv_bfloat16* __restrict__ beta_chunk,        // (B, NC, C, H)
    const __nv_bfloat16* __restrict__ chunk_states,      // (B, NC, H, N, D)
    const __nv_bfloat16* __restrict__ corrections,       // (B, NC, C, H, N, D)
    // Output gradients
    __nv_bfloat16* __restrict__ grad_A,                  // (B, NC, C, H, N)
    __nv_bfloat16* __restrict__ grad_K,                  // (B, NC, C, H, D)
    float* __restrict__ grad_beta,                        // (B, NC, C, H) FP32
    __nv_bfloat16* __restrict__ grad_states,             // (B, NC, H, N, D)
    // Dimensions
    int B, int NC, int C, int H, int D, int N
) {
    const int bnc_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    if (bnc_idx >= B * NC || head_idx >= H) return;

    const int b  = bnc_idx / NC;
    const int nc = bnc_idx % NC;

    const int d = threadIdx.x;
    if (d >= D) return;

    __shared__ float smem[MAX_WARPS * MAX_STATE_DIM];

    const int warp_id   = d / WARP_SIZE;
    const int lane_id   = d % WARP_SIZE;
    const int num_warps = (D + WARP_SIZE - 1) / WARP_SIZE;

    // Strides (same layout as forward)
    const int A_base = ((b * NC + nc) * C * H + head_idx) * N;
    const int A_step = H * N;
    const int K_base = ((b * NC + nc) * C * H + head_idx) * D;
    const int K_step = H * D;
    const int b_base = (b * NC + nc) * C * H + head_idx;
    const int b_step = H;
    const int corr_base = ((b * NC + nc) * C * H + head_idx) * N * D;
    const int corr_step = H * N * D;
    const int cs_base   = ((b * NC + nc) * H + head_idx) * N * D;

    // State gradient accumulator
    float dh[MAX_STATE_DIM];
    #pragma unroll
    for (int s = 0; s < MAX_STATE_DIM; s++) dh[s] = 0.0f;

    for (int t = C - 1; t >= 0; t--) {
        // ---- Add upstream gradient from grad_corrections[t] ----
        int corr_i = corr_base + t * corr_step;
        for (int s = 0; s < N; s++) {
            dh[s] += bf16_to_fp32(__ldg(&grad_corrections[corr_i + s * D + d]));
        }

        // ---- Load forward values ----
        int Ai = A_base + t * A_step;
        float a[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            a[s] = bf16_to_fp32(__ldg(&A_chunk[Ai + s]));
        }

        float k_d  = bf16_to_fp32(__ldg(&K_chunk[K_base + t * K_step + d]));
        float beta = bf16_to_fp32(__ldg(&beta_chunk[b_base + t * b_step]));

        // ---- Load corrections[t] = h_before (state before erasure/rotation) ----
        float h_before[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            h_before[s] = bf16_to_fp32(__ldg(&corrections[corr_i + s * D + d]));
        }

        // ---- Recompute kTh = sum_d(h_before[s,d] * k[d]) ----
        float kTh_local[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) kTh_local[s] = h_before[s] * k_d;
        for (int s = 0; s < N; s++) kTh_local[s] = warp_reduce_sum(kTh_local[s]);
        if (lane_id == 0) {
            for (int s = 0; s < N; s++) smem[warp_id * MAX_STATE_DIM + s] = kTh_local[s];
        }
        __syncthreads();
        float kTh[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            kTh[s] = 0.0f;
            #pragma unroll
            for (int w = 0; w < MAX_WARPS; w++) {
                if (w < num_warps) kTh[s] += smem[w * MAX_STATE_DIM + s];
            }
        }
        __syncthreads();

        // ---- Recompute h_erased = h_before - beta * kTh * k ----
        float hm[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            hm[s] = h_before[s] - beta * kTh[s] * k_d;
        }

        
        // Backward through rotation: complex diagonal
        //   Forward: h_after[2j]   = re*hm[2j]   - im*hm[2j+1]
        //            h_after[2j+1] = re*hm[2j+1] + im*hm[2j]
        //   Backward: multiply by conj(conj(mu)) = mu = (re, -im)
        //     dhm[2j]   = re*dh[2j]   + im*dh[2j+1]
        //     dhm[2j+1] = re*dh[2j+1] - im*dh[2j]
        
        float dhm[MAX_STATE_DIM];
        for (int j = 0; j < N / 2; j++) {
            float re = a[2 * j];
            float im = a[2 * j + 1];
            float dh_re = dh[2 * j];
            float dh_im = dh[2 * j + 1];
            dhm[2 * j]     = re * dh_re + im * dh_im;
            dhm[2 * j + 1] = re * dh_im - im * dh_re;
        }

        
        // grad_A from state path (needs cross-thread reduction)
        //   d/d(stored_re) = dh_re * hm_re + dh_im * hm_im
        //   d/d(stored_im) = -dh_re * hm_im + dh_im * hm_re
        
        float gA_local[MAX_STATE_DIM];
        for (int j = 0; j < N / 2; j++) {
            float hm_re = hm[2 * j];
            float hm_im = hm[2 * j + 1];
            float dh_re = dh[2 * j];
            float dh_im = dh[2 * j + 1];
            gA_local[2 * j]     = dh_re * hm_re + dh_im * hm_im;
            gA_local[2 * j + 1] = -dh_re * hm_im + dh_im * hm_re;
        }
        // Batched reduction for grad_A
        for (int s = 0; s < N; s++) gA_local[s] = warp_reduce_sum(gA_local[s]);
        if (lane_id == 0) {
            for (int s = 0; s < N; s++) smem[warp_id * MAX_STATE_DIM + s] = gA_local[s];
        }
        __syncthreads();
        float gA[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            gA[s] = 0.0f;
            #pragma unroll
            for (int w = 0; w < MAX_WARPS; w++) {
                if (w < num_warps) gA[s] += smem[w * MAX_STATE_DIM + s];
            }
        }
        __syncthreads();

        // Store grad_A[t] (thread 0 writes)
        if (d == 0) {
            for (int s = 0; s < N; s++) {
                grad_A[Ai + s] = fp32_to_bf16(gA[s]);
            }
        }

        
        // Backward through erasure: h_erased = h_before - beta*(kTh outer k)
        

        // zeta[s] = sum_d(dhm[s,d] * k[d]) — batched reduction
        float zeta_local[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) zeta_local[s] = dhm[s] * k_d;
        for (int s = 0; s < N; s++) zeta_local[s] = warp_reduce_sum(zeta_local[s]);
        if (lane_id == 0) {
            for (int s = 0; s < N; s++) smem[warp_id * MAX_STATE_DIM + s] = zeta_local[s];
        }
        __syncthreads();
        float zeta[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            zeta[s] = 0.0f;
            #pragma unroll
            for (int w = 0; w < MAX_WARPS; w++) {
                if (w < num_warps) zeta[s] += smem[w * MAX_STATE_DIM + s];
            }
        }
        __syncthreads();

        // grad_k from erasure (per-thread):
        //   -beta * sum_s(kTh[s]*dhm[s,d] + h_before[s,d]*zeta[s])
        float gk = 0.0f;
        for (int s = 0; s < N; s++) {
            gk += kTh[s] * dhm[s] + h_before[s] * zeta[s];
        }
        gk *= -beta;

        // grad_beta from erasure: -sum_s(kTh[s]*zeta[s]) — needs cross-thread reduction
        float gbeta_local = 0.0f;
        for (int s = 0; s < N; s++) {
            gbeta_local -= kTh[s] * zeta[s];
        }
        float gbeta_warp = warp_reduce_sum(gbeta_local);
        if (lane_id == 0) smem[warp_id * MAX_STATE_DIM] = gbeta_warp;
        __syncthreads();
        float gbeta = 0.0f;
        #pragma unroll
        for (int w = 0; w < MAX_WARPS; w++) {
            if (w < num_warps) gbeta += smem[w * MAX_STATE_DIM];
        }
        __syncthreads();

        // Store grad_K, grad_beta
        grad_K[K_base + t * K_step + d] = fp32_to_bf16(gk);

        if (d == 0) {
            grad_beta[b_base + t * b_step] = gbeta;
        }

        // Propagate dh backward to h_before for next iteration
        //   dh_new[s,d] = dhm[s,d] - beta * zeta[s] * k[d]
        for (int s = 0; s < N; s++) {
            dh[s] = dhm[s] - beta * zeta[s] * k_d;
        }
    }

    // ---- Accumulate remaining dh into grad_chunk_states ----
    for (int s = 0; s < N; s++) {
        grad_states[cs_base + s * D + d] = fp32_to_bf16(dh[s]);
    }
}



// Launch Functions


torch::Tensor
exact_correction_fwd_cuda(
    torch::Tensor A_chunk,       // (B, NC, C, H, N)  BF16
    torch::Tensor K_chunk,       // (B, NC, C, H, D)  BF16
    torch::Tensor beta_chunk,    // (B, NC, C, H)     BF16
    torch::Tensor chunk_states,  // (B, NC, H, N, D)  BF16
    int state_dim                // N (runtime)
) {
    TORCH_CHECK(A_chunk.is_cuda(), "A_chunk must be CUDA");
    TORCH_CHECK(K_chunk.is_cuda(), "K_chunk must be CUDA");
    TORCH_CHECK(beta_chunk.is_cuda(), "beta_chunk must be CUDA");
    TORCH_CHECK(chunk_states.is_cuda(), "chunk_states must be CUDA");

    A_chunk      = A_chunk.contiguous();
    K_chunk      = K_chunk.contiguous();
    beta_chunk   = beta_chunk.contiguous();
    chunk_states = chunk_states.contiguous();

    const int B  = A_chunk.size(0);
    const int NC = A_chunk.size(1);
    const int C  = A_chunk.size(2);
    const int H  = A_chunk.size(3);
    const int D  = K_chunk.size(4);
    const int N  = state_dim;

    TORCH_CHECK(D <= MAX_HEAD_DIM, "head_dim must be <= ", MAX_HEAD_DIM);
    TORCH_CHECK(N <= MAX_STATE_DIM, "state_dim must be <= ", MAX_STATE_DIM);
    TORCH_CHECK(N % 2 == 0, "state_dim must be even (complex pairs)");

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(A_chunk.device());
    auto corrections = torch::empty({B, NC, C, H, N, D}, opts);

    dim3 grid(B * NC, H);
    dim3 block(D);
    size_t smem_bytes = MAX_WARPS * MAX_STATE_DIM * sizeof(float);
    cudaStream_t stream = get_cuda_stream();

    exact_correction_fwd_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A_chunk.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K_chunk.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(beta_chunk.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(chunk_states.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(corrections.data_ptr()),
        B, NC, C, H, D, N
    );
    CUDA_CHECK_LAST();

    return corrections;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
exact_correction_bwd_cuda(
    torch::Tensor grad_corrections,  // (B, NC, C, H, N, D) BF16
    torch::Tensor A_chunk,           // (B, NC, C, H, N)    BF16
    torch::Tensor K_chunk,           // (B, NC, C, H, D)    BF16
    torch::Tensor beta_chunk,        // (B, NC, C, H)       BF16
    torch::Tensor chunk_states,      // (B, NC, H, N, D)    BF16
    torch::Tensor corrections,       // (B, NC, C, H, N, D) BF16
    int state_dim                    // N (runtime)
) {
    TORCH_CHECK(A_chunk.is_cuda(), "A_chunk must be CUDA");

    grad_corrections = grad_corrections.contiguous();
    A_chunk          = A_chunk.contiguous();
    K_chunk          = K_chunk.contiguous();
    beta_chunk       = beta_chunk.contiguous();
    chunk_states     = chunk_states.contiguous();
    corrections      = corrections.contiguous();

    const int B  = A_chunk.size(0);
    const int NC = A_chunk.size(1);
    const int C  = A_chunk.size(2);
    const int H  = A_chunk.size(3);
    const int D  = K_chunk.size(4);
    const int N  = state_dim;

    TORCH_CHECK(N <= MAX_STATE_DIM, "state_dim must be <= ", MAX_STATE_DIM);
    TORCH_CHECK(N % 2 == 0, "state_dim must be even (complex pairs)");

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(A_chunk.device());
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(A_chunk.device());

    auto grad_A     = torch::zeros({B, NC, C, H, N}, opts_bf16);
    auto grad_K     = torch::zeros({B, NC, C, H, D}, opts_bf16);
    auto grad_beta  = torch::zeros({B, NC, C, H}, opts_fp32);
    auto grad_states = torch::zeros({B, NC, H, N, D}, opts_bf16);

    dim3 grid(B * NC, H);
    dim3 block(D);
    size_t smem_bytes = MAX_WARPS * MAX_STATE_DIM * sizeof(float);
    cudaStream_t stream = get_cuda_stream();

    exact_correction_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_corrections.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(A_chunk.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K_chunk.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(beta_chunk.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(chunk_states.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(corrections.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_A.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_K.data_ptr()),
        grad_beta.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(grad_states.data_ptr()),
        B, NC, C, H, D, N
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(grad_A, grad_K, grad_beta, grad_states);
}

}  // namespace cdssm
