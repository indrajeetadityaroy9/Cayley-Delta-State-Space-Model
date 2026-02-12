// CDSSM Intra-Chunk Delta-Rule Scan — CUDA Implementation
//
// Fuses the sequential delta-rule scan within each chunk into a single kernel.
// This is the critical performance bottleneck: the Python version issues 5+
// einsum/matmul calls per timestep × 64 timesteps per chunk.
//
// Algorithm (per chunk, per head):
//   State h ∈ R^(2×D) is a matrix memory.
//   For each timestep t in 0..C-1:
//     kTh = h @ k           (retrieval: cross-thread reduction over D)
//     h_mod = h - β*(kTh⊗k) (selective erasure)
//     h = A @ h_mod          (Cayley rotation-damping, 2×2 broadcast over D)
//     h += β*(v⊗k)          (injection)
//   Also tracks cumulative A_bar product (2×2) for inter-chunk correction.
//
// Grid: (B*NC, H)  — one block per (chunk, head)
// Block: D threads  — each thread owns column d of the (2,D) state
//
// BF16 I/O, FP32 compute. Register-resident state.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/common.cuh"
#include "../include/reduction.cuh"

namespace cdssm {

constexpr int MAX_HEAD_DIM = 128;
constexpr int MAX_WARPS = MAX_HEAD_DIM / WARP_SIZE;  // 4

// Helper: cross-block reduction for D threads (up to 4 warps)
// Reduces `val` across all D threads, returns the scalar sum to every thread.
// Uses shared memory `smem` which must have at least MAX_WARPS floats.
__device__ __forceinline__ float block_reduce_to_scalar(
    float val, float* smem, int D
) {
    float warp_sum = warp_reduce_sum(val);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = (D + WARP_SIZE - 1) / WARP_SIZE;

    if (lane_id == 0) {
        smem[warp_id] = warp_sum;
    }
    __syncthreads();

    float total = 0.0f;
    #pragma unroll
    for (int w = 0; w < MAX_WARPS; w++) {
        if (w < num_warps) total += smem[w];
    }
    return total;
}

// Forward Kernel

__global__ void intra_chunk_scan_fwd_kernel(
    // Inputs (BF16, contiguous)
    const __nv_bfloat16* __restrict__ A_flat,     // (BNC, C, H, 2, 2)
    const __nv_bfloat16* __restrict__ K_flat,     // (BNC, C, H, D)
    const __nv_bfloat16* __restrict__ V_flat,     // (BNC, C, H, 2)
    const __nv_bfloat16* __restrict__ beta_flat,  // (BNC, C, H)
    // Outputs (BF16)
    __nv_bfloat16* __restrict__ local_h,          // (BNC, C, H, 2, D)
    __nv_bfloat16* __restrict__ cum_A,            // (BNC, C, H, 2, 2)
    // Dimensions
    int BNC, int C, int H, int D
) {
    const int chunk_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    if (chunk_idx >= BNC || head_idx >= H) return;

    const int d = threadIdx.x;
    if (d >= D) return;

    // Shared memory for cross-warp reductions (2 values: kTh_0, kTh_1)
    __shared__ float smem[MAX_WARPS];

    // Stride computations for contiguous tensors
    // A_flat: (BNC, C, H, 2, 2) → innermost dims are [2,2] = 4 elements
    const int A_base = (chunk_idx * C * H + head_idx) * 4;
    const int A_step = H * 4;  // stride along C dimension

    // K_flat: (BNC, C, H, D)
    const int K_base = (chunk_idx * C * H + head_idx) * D;
    const int K_step = H * D;

    // V_flat: (BNC, C, H, 2)
    const int V_base = (chunk_idx * C * H + head_idx) * 2;
    const int V_step = H * 2;

    // beta_flat: (BNC, C, H)
    const int b_base = chunk_idx * C * H + head_idx;
    const int b_step = H;

    // local_h: (BNC, C, H, 2, D) → innermost [2,D] = 2*D elements
    const int lh_base = (chunk_idx * C * H + head_idx) * 2 * D;
    const int lh_step = H * 2 * D;

    // cum_A: (BNC, C, H, 2, 2) → innermost [2,2] = 4 elements
    const int cA_base = (chunk_idx * C * H + head_idx) * 4;
    const int cA_step = H * 4;

    // Register-resident state
    float h0 = 0.0f, h1 = 0.0f;

    // Cumulative A_bar product (identity)
    float cA00 = 1.0f, cA01 = 0.0f;
    float cA10 = 0.0f, cA11 = 1.0f;

    for (int t = 0; t < C; t++) {
        // Load A_bar[t] (shared: same for all threads)
        int Ai = A_base + t * A_step;
        float a11 = bf16_to_fp32(__ldg(&A_flat[Ai + 0]));
        float a12 = bf16_to_fp32(__ldg(&A_flat[Ai + 1]));
        float a21 = bf16_to_fp32(__ldg(&A_flat[Ai + 2]));
        float a22 = bf16_to_fp32(__ldg(&A_flat[Ai + 3]));

        // Load k[t,d] (per-thread), v[t], beta[t] (shared)
        float k_d  = bf16_to_fp32(__ldg(&K_flat[K_base + t * K_step + d]));
        int Vi = V_base + t * V_step;
        float v0   = bf16_to_fp32(__ldg(&V_flat[Vi + 0]));
        float v1   = bf16_to_fp32(__ldg(&V_flat[Vi + 1]));
        float beta = bf16_to_fp32(__ldg(&beta_flat[b_base + t * b_step]));

        // 1. Delta-rule retrieval: kTh_s = sum_d(h[s,d] * k[d])
        float kTh_0 = block_reduce_to_scalar(h0 * k_d, smem, D);
        __syncthreads();
        float kTh_1 = block_reduce_to_scalar(h1 * k_d, smem, D);
        __syncthreads();

        // 2. Selective erasure
        h0 -= beta * kTh_0 * k_d;
        h1 -= beta * kTh_1 * k_d;

        // 3. Apply A_bar rotation-damping (2×2 @ 2-vector, broadcast over D)
        float h0_new = a11 * h0 + a12 * h1;
        float h1_new = a21 * h0 + a22 * h1;
        h0 = h0_new;
        h1 = h1_new;

        // 4. Injection
        h0 += beta * v0 * k_d;
        h1 += beta * v1 * k_d;

        // Store local_h[chunk, t, head, s, d]
        int lhi = lh_base + t * lh_step;
        local_h[lhi + d]     = fp32_to_bf16(h0);  // s=0
        local_h[lhi + D + d] = fp32_to_bf16(h1);  // s=1

        // Update cumulative A_bar product: cumA = A[t] @ cumA
        float new00 = a11 * cA00 + a12 * cA10;
        float new01 = a11 * cA01 + a12 * cA11;
        float new10 = a21 * cA00 + a22 * cA10;
        float new11 = a21 * cA01 + a22 * cA11;
        cA00 = new00; cA01 = new01;
        cA10 = new10; cA11 = new11;

        // Store cum_A[chunk, t, head, :, :] — only thread 0 writes
        if (d == 0) {
            int cAi = cA_base + t * cA_step;
            cum_A[cAi + 0] = fp32_to_bf16(cA00);
            cum_A[cAi + 1] = fp32_to_bf16(cA01);
            cum_A[cAi + 2] = fp32_to_bf16(cA10);
            cum_A[cAi + 3] = fp32_to_bf16(cA11);
        }
    }
}

// Backward Kernel
//
// Reverse-mode sequential scan through the chunk.
//
// Forward step t:
//   h_prev = state entering step t (= local_h[t-1], or 0 for t=0)
//   kTh[s] = sum_d(h_prev[s,d] * k[d])
//   h_mod[s,d] = h_prev[s,d] - beta * kTh[s] * k[d]
//   h_evolved[s,d] = A[s,0]*h_mod[0,d] + A[s,1]*h_mod[1,d]
//   h_out[s,d] = h_evolved[s,d] + beta * v[s] * k[d]
//   local_h[t] = h_out
//
// Backward: propagate dh (gradient of loss w.r.t. state) in reverse.
//
// At step t, dh = dL/d(h_out[t]) = grad_local_h[t] + dh_from_step_{t+1}
//
// Derivatives w.r.t. inputs at step t:
//   d/dv[s]: sum_d(beta * k[d] * dh[s,d])
//   d/dk[d] from inject: beta * (v[0]*dh[0,d] + v[1]*dh[1,d])
//   d/dbeta from inject: sum_{s,d}(v[s]*k[d]*dh[s,d])
//
//   dh_mod[s,d] = A[0,s]*dh[0,d] + A[1,s]*dh[1,d]  (A^T @ dh)
//   d/dA[i,j] from rotation: sum_d(dh[i,d] * h_mod[j,d])
//
//   zeta[s] = sum_d(dh_mod[s,d] * k[d])
//   d/dk[d] from erase: -beta * (kTh[0]*dh_mod[0,d] + kTh[1]*dh_mod[1,d])
//                       -beta * (h_prev[0,d]*zeta[0] + h_prev[1,d]*zeta[1])
//   d/dbeta from erase: -sum_{s,d}(kTh[s]*k[d]*dh_mod[s,d])
//                      = -(kTh[0]*zeta[0] + kTh[1]*zeta[1])
//
//   dh_prev[s,d] = dh_mod[s,d] - beta * zeta[s] * k[d]

__global__ void intra_chunk_scan_bwd_kernel(
    // Upstream gradients
    const __nv_bfloat16* __restrict__ grad_local_h,  // (BNC, C, H, 2, D)
    const __nv_bfloat16* __restrict__ grad_cum_A,    // (BNC, C, H, 2, 2)
    // Saved from forward
    const __nv_bfloat16* __restrict__ A_flat,        // (BNC, C, H, 2, 2)
    const __nv_bfloat16* __restrict__ K_flat,        // (BNC, C, H, D)
    const __nv_bfloat16* __restrict__ V_flat,        // (BNC, C, H, 2)
    const __nv_bfloat16* __restrict__ beta_flat,     // (BNC, C, H)
    const __nv_bfloat16* __restrict__ local_h,       // (BNC, C, H, 2, D)
    const __nv_bfloat16* __restrict__ cum_A,         // (BNC, C, H, 2, 2)
    // Output gradients
    __nv_bfloat16* __restrict__ grad_A,              // (BNC, C, H, 2, 2)
    __nv_bfloat16* __restrict__ grad_K,              // (BNC, C, H, D)
    __nv_bfloat16* __restrict__ grad_V,              // (BNC, C, H, 2)
    float* __restrict__ grad_beta,                    // (BNC, C, H) FP32
    // Dimensions
    int BNC, int C, int H, int D
) {
    const int chunk_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    if (chunk_idx >= BNC || head_idx >= H) return;

    const int d = threadIdx.x;
    if (d >= D) return;

    __shared__ float smem[MAX_WARPS];

    // Strides (same layout as forward)
    const int A_base  = (chunk_idx * C * H + head_idx) * 4;
    const int A_step  = H * 4;
    const int K_base  = (chunk_idx * C * H + head_idx) * D;
    const int K_step  = H * D;
    const int V_base  = (chunk_idx * C * H + head_idx) * 2;
    const int V_step  = H * 2;
    const int b_base  = chunk_idx * C * H + head_idx;
    const int b_step  = H;
    const int lh_base = (chunk_idx * C * H + head_idx) * 2 * D;
    const int lh_step = H * 2 * D;
    const int cA_base = (chunk_idx * C * H + head_idx) * 4;
    const int cA_step = H * 4;

    // ---- State gradient accumulator ----
    float dh0 = 0.0f, dh1 = 0.0f;

    // ---- CumA gradient accumulator (2x2, register-only) ----
    // cumA[t] = A[t] @ cumA[t-1], cumA[-1] = I
    // Backward: dcumA_accum[t] = grad_cum_A[t] + A[t+1]^T @ dcumA_accum[t+1]
    // grad_A[t] from cumA path = dcumA_accum[t] @ cumA[t-1]^T
    float dcA00 = 0.0f, dcA01 = 0.0f;
    float dcA10 = 0.0f, dcA11 = 0.0f;

    for (int t = C - 1; t >= 0; t--) {
        // ---- Add upstream gradient from local_h[t] ----
        int lhi = lh_base + t * lh_step;
        dh0 += bf16_to_fp32(__ldg(&grad_local_h[lhi + d]));
        dh1 += bf16_to_fp32(__ldg(&grad_local_h[lhi + D + d]));

        // ---- Load forward values ----
        int Ai = A_base + t * A_step;
        float a11 = bf16_to_fp32(__ldg(&A_flat[Ai + 0]));
        float a12 = bf16_to_fp32(__ldg(&A_flat[Ai + 1]));
        float a21 = bf16_to_fp32(__ldg(&A_flat[Ai + 2]));
        float a22 = bf16_to_fp32(__ldg(&A_flat[Ai + 3]));

        float k_d  = bf16_to_fp32(__ldg(&K_flat[K_base + t * K_step + d]));
        int Vi = V_base + t * V_step;
        float v0   = bf16_to_fp32(__ldg(&V_flat[Vi + 0]));
        float v1   = bf16_to_fp32(__ldg(&V_flat[Vi + 1]));
        float beta = bf16_to_fp32(__ldg(&beta_flat[b_base + t * b_step]));

        // h_prev[s,d] = local_h[t-1, s, d], or 0 if t=0
        float hp0, hp1;
        if (t > 0) {
            int prev_i = lh_base + (t - 1) * lh_step;
            hp0 = bf16_to_fp32(__ldg(&local_h[prev_i + d]));
            hp1 = bf16_to_fp32(__ldg(&local_h[prev_i + D + d]));
        } else {
            hp0 = 0.0f;
            hp1 = 0.0f;
        }

        // ---- Recompute kTh = h_prev @ k ----
        float kTh_0 = block_reduce_to_scalar(hp0 * k_d, smem, D);
        __syncthreads();
        float kTh_1 = block_reduce_to_scalar(hp1 * k_d, smem, D);
        __syncthreads();

        // ---- Recompute h_mod = h_prev - beta * (kTh outer k) ----
        float hm0 = hp0 - beta * kTh_0 * k_d;
        float hm1 = hp1 - beta * kTh_1 * k_d;

        // 4. Backward through injection: h_out = h_evolved + beta*(v⊗k)
        // dh is the gradient w.r.t. h_out
        // grad_v: sum_d(beta * k_d * dh[s,d])
        float gv0 = block_reduce_to_scalar(beta * k_d * dh0, smem, D);
        __syncthreads();
        float gv1 = block_reduce_to_scalar(beta * k_d * dh1, smem, D);
        __syncthreads();

        // grad_k from injection (per-thread)
        float gk_inject = beta * (v0 * dh0 + v1 * dh1);

        // grad_beta from injection: sum_{s,d}(v[s]*k[d]*dh[s,d])
        float gbeta_inject = block_reduce_to_scalar(
            v0 * k_d * dh0 + v1 * k_d * dh1, smem, D);
        __syncthreads();

        // Store grad_V (thread 0 writes the 2 values)
        if (d == 0) {
            int gVi = V_base + t * V_step;
            grad_V[gVi + 0] = fp32_to_bf16(gv0);
            grad_V[gVi + 1] = fp32_to_bf16(gv1);
        }

        // 3. Backward through rotation: h_evolved = A @ h_mod
        // dh_mod = A^T @ dh  (dh is still the gradient at h_out = h_evolved here,
        //   since injection doesn't modify h_evolved's gradient contribution)
        float dhm0 = a11 * dh0 + a21 * dh1;  // A^T row 0
        float dhm1 = a12 * dh0 + a22 * dh1;  // A^T row 1

        // grad_A: dh[i,d] * h_mod[j,d] → sum over D
        float gA00 = block_reduce_to_scalar(dh0 * hm0, smem, D);
        __syncthreads();
        float gA01 = block_reduce_to_scalar(dh0 * hm1, smem, D);
        __syncthreads();
        float gA10 = block_reduce_to_scalar(dh1 * hm0, smem, D);
        __syncthreads();
        float gA11 = block_reduce_to_scalar(dh1 * hm1, smem, D);
        __syncthreads();

        // CumA gradient path (register-only, no cross-thread reduction)
        // Accumulate: dcA_accum[t] = grad_cum_A[t] + A[t+1]^T @ dcA_accum[t+1]
        // (At this point dcA00..11 holds dcA_accum from step t+1, or 0 for t=C-1)
        // Apply A[t+1]^T to dcA_accum was done at end of previous iteration.
        // Now add grad_cum_A[t]:
        int gcAi = cA_base + t * cA_step;
        dcA00 += bf16_to_fp32(__ldg(&grad_cum_A[gcAi + 0]));
        dcA01 += bf16_to_fp32(__ldg(&grad_cum_A[gcAi + 1]));
        dcA10 += bf16_to_fp32(__ldg(&grad_cum_A[gcAi + 2]));
        dcA11 += bf16_to_fp32(__ldg(&grad_cum_A[gcAi + 3]));

        // grad_A[t] from cumA path = dcA_accum[t] @ cumA[t-1]^T
        // cumA[t-1] is the 2x2 matrix at step t-1
        float prevCA00, prevCA01, prevCA10, prevCA11;
        if (t > 0) {
            int pcAi = cA_base + (t - 1) * cA_step;
            prevCA00 = bf16_to_fp32(__ldg(&cum_A[pcAi + 0]));
            prevCA01 = bf16_to_fp32(__ldg(&cum_A[pcAi + 1]));
            prevCA10 = bf16_to_fp32(__ldg(&cum_A[pcAi + 2]));
            prevCA11 = bf16_to_fp32(__ldg(&cum_A[pcAi + 3]));
        } else {
            // cumA[-1] = I
            prevCA00 = 1.0f; prevCA01 = 0.0f;
            prevCA10 = 0.0f; prevCA11 = 1.0f;
        }

        // dcA @ prevCA^T (2x2 @ 2x2^T)
        float gA_cA00 = dcA00 * prevCA00 + dcA01 * prevCA01;
        float gA_cA01 = dcA00 * prevCA10 + dcA01 * prevCA11;
        float gA_cA10 = dcA10 * prevCA00 + dcA11 * prevCA01;
        float gA_cA11 = dcA10 * prevCA10 + dcA11 * prevCA11;

        // Total grad_A[t] = state-path gradient + cumA-path gradient
        if (d == 0) {
            grad_A[gcAi + 0] = fp32_to_bf16(gA00 + gA_cA00);
            grad_A[gcAi + 1] = fp32_to_bf16(gA01 + gA_cA01);
            grad_A[gcAi + 2] = fp32_to_bf16(gA10 + gA_cA10);
            grad_A[gcAi + 3] = fp32_to_bf16(gA11 + gA_cA11);
        }

        // Propagate dcA backward: dcA_accum for step t-1 = A[t]^T @ dcA_accum[t]
        float new_dcA00 = a11 * dcA00 + a21 * dcA10;
        float new_dcA01 = a11 * dcA01 + a21 * dcA11;
        float new_dcA10 = a12 * dcA00 + a22 * dcA10;
        float new_dcA11 = a12 * dcA01 + a22 * dcA11;
        dcA00 = new_dcA00; dcA01 = new_dcA01;
        dcA10 = new_dcA10; dcA11 = new_dcA11;

        // 2. Backward through erasure: h_mod = h_prev - beta*(kTh⊗k)
        // zeta[s] = sum_d(dhm[s,d] * k[d])
        float zeta_0 = block_reduce_to_scalar(dhm0 * k_d, smem, D);
        __syncthreads();
        float zeta_1 = block_reduce_to_scalar(dhm1 * k_d, smem, D);
        __syncthreads();

        // grad_k from erasure (per-thread):
        // -beta * (kTh[s]*dhm[s,d] + h_prev[s,d]*zeta[s]) summed over s
        float gk_erase = -beta * (kTh_0 * dhm0 + hp0 * zeta_0
                                 + kTh_1 * dhm1 + hp1 * zeta_1);

        // grad_beta from erasure: -(kTh[0]*zeta[0] + kTh[1]*zeta[1])
        float gbeta_erase = -(kTh_0 * zeta_0 + kTh_1 * zeta_1);

        // Store grad_K, grad_beta
        grad_K[K_base + t * K_step + d] = fp32_to_bf16(gk_inject + gk_erase);

        if (d == 0) {
            grad_beta[b_base + t * b_step] = gbeta_inject + gbeta_erase;
        }

        // Propagate dh backward to h_prev for next iteration
        // dh_prev[s,d] = dhm[s,d] - beta * zeta[s] * k[d]
        dh0 = dhm0 - beta * zeta_0 * k_d;
        dh1 = dhm1 - beta * zeta_1 * k_d;
    }
}


// Launch Functions

std::tuple<torch::Tensor, torch::Tensor>
intra_chunk_scan_fwd_cuda(
    torch::Tensor A_flat,      // (BNC, C, H, 2, 2) BF16
    torch::Tensor K_flat,      // (BNC, C, H, D)    BF16
    torch::Tensor V_flat,      // (BNC, C, H, 2)    BF16
    torch::Tensor beta_flat    // (BNC, C, H)       BF16
) {
    TORCH_CHECK(A_flat.is_cuda(), "A_flat must be CUDA");
    TORCH_CHECK(K_flat.is_cuda(), "K_flat must be CUDA");
    TORCH_CHECK(V_flat.is_cuda(), "V_flat must be CUDA");
    TORCH_CHECK(beta_flat.is_cuda(), "beta_flat must be CUDA");

    A_flat = A_flat.contiguous();
    K_flat = K_flat.contiguous();
    V_flat = V_flat.contiguous();
    beta_flat = beta_flat.contiguous();

    const int BNC = A_flat.size(0);
    const int C   = A_flat.size(1);
    const int H   = A_flat.size(2);
    const int D   = K_flat.size(3);

    TORCH_CHECK(D <= MAX_HEAD_DIM, "head_dim must be <= ", MAX_HEAD_DIM);

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(A_flat.device());
    auto local_h = torch::empty({BNC, C, H, 2, D}, opts);
    auto cum_A   = torch::empty({BNC, C, H, 2, 2}, opts);

    dim3 grid(BNC, H);
    dim3 block(D);
    cudaStream_t stream = get_cuda_stream();

    intra_chunk_scan_fwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(beta_flat.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(local_h.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(cum_A.data_ptr()),
        BNC, C, H, D
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(local_h, cum_A);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
intra_chunk_scan_bwd_cuda(
    torch::Tensor grad_local_h,  // (BNC, C, H, 2, D) BF16
    torch::Tensor grad_cum_A,    // (BNC, C, H, 2, 2) BF16
    torch::Tensor A_flat,        // (BNC, C, H, 2, 2) BF16
    torch::Tensor K_flat,        // (BNC, C, H, D)    BF16
    torch::Tensor V_flat,        // (BNC, C, H, 2)    BF16
    torch::Tensor beta_flat,     // (BNC, C, H)       BF16
    torch::Tensor local_h,       // (BNC, C, H, 2, D) BF16
    torch::Tensor cum_A          // (BNC, C, H, 2, 2) BF16
) {
    TORCH_CHECK(A_flat.is_cuda(), "A_flat must be CUDA");

    grad_local_h = grad_local_h.contiguous();
    grad_cum_A   = grad_cum_A.contiguous();
    A_flat       = A_flat.contiguous();
    K_flat       = K_flat.contiguous();
    V_flat       = V_flat.contiguous();
    beta_flat    = beta_flat.contiguous();
    local_h      = local_h.contiguous();
    cum_A        = cum_A.contiguous();

    const int BNC = A_flat.size(0);
    const int C   = A_flat.size(1);
    const int H   = A_flat.size(2);
    const int D   = K_flat.size(3);

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(A_flat.device());
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(A_flat.device());

    auto grad_A    = torch::zeros({BNC, C, H, 2, 2}, opts_bf16);
    auto grad_K    = torch::zeros({BNC, C, H, D}, opts_bf16);
    auto grad_V    = torch::zeros({BNC, C, H, 2}, opts_bf16);
    auto grad_beta = torch::zeros({BNC, C, H}, opts_fp32);

    dim3 grid(BNC, H);
    dim3 block(D);
    cudaStream_t stream = get_cuda_stream();

    intra_chunk_scan_bwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_local_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(grad_cum_A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(A_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(beta_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(local_h.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(cum_A.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_A.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_K.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_V.data_ptr()),
        grad_beta.data_ptr<float>(),
        BNC, C, H, D
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(grad_A, grad_K, grad_V, grad_beta);
}

}  // namespace cdssm
