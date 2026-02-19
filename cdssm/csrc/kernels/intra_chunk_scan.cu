// CDSSM v2 Intra-Chunk Delta-Rule Scan — Complex Diagonal State
//
// Fuses the sequential delta-rule scan within each chunk into a single kernel.
// State is now h ∈ R^(N×D) with complex diagonal rotation A_bar ∈ C^(N/2)
// stored as re/im interleaved pairs (conj(mu)).
//
// Algorithm (per chunk, per head):
//   State h[N][D] is register-resident (N up to MAX_STATE_DIM=32).
//   For each timestep t in 0..C-1:
//     kTh[s] = sum_d(h[s,d] * k[d])   (batched retrieval: 2 syncthreads total)
//     h[s,d] -= beta * kTh[s] * k[d]  (selective erasure)
//     Complex diagonal rotation:        (pairs j=0..N/2-1)
//       h_new[2j]   = re*h[2j]   - im*h[2j+1]
//       h_new[2j+1] = re*h[2j+1] + im*h[2j]
//     h[s,d] += beta * v[s] * k[d]    (injection)
//   Also tracks cumulative A_bar product (complex diagonal) for inter-chunk correction.
//
// Grid: (BNC, H)  — one block per (chunk, head)
// Block: D threads — each thread owns column d of the (N,D) state
//
// BF16 I/O, FP32 compute. Register-resident state.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/common.cuh"

namespace cdssm {


// Forward Kernel


__global__ void intra_chunk_scan_fwd_kernel(
    // Inputs (BF16, contiguous)
    const __nv_bfloat16* __restrict__ A_flat,     // (BNC, C, H, N) re/im interleaved
    const __nv_bfloat16* __restrict__ K_flat,     // (BNC, C, H, D)
    const __nv_bfloat16* __restrict__ V_flat,     // (BNC, C, H, N)
    const __nv_bfloat16* __restrict__ beta_flat,  // (BNC, C, H)
    // Outputs (BF16)
    __nv_bfloat16* __restrict__ local_h,          // (BNC, C, H, N, D)
    __nv_bfloat16* __restrict__ cum_A,            // (BNC, C, H, N)
    // Dimensions
    int BNC, int C, int H, int D, int N
) {
    const int chunk_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    if (chunk_idx >= BNC || head_idx >= H) return;

    const int d = threadIdx.x;
    if (d >= D) return;

    // Shared memory for batched kTh reduction
    __shared__ float smem[MAX_WARPS * MAX_STATE_DIM];

    const int warp_id  = d / WARP_SIZE;
    const int lane_id  = d % WARP_SIZE;
    const int num_warps = (D + WARP_SIZE - 1) / WARP_SIZE;

    // Stride computations for contiguous tensors
    // A_flat: (BNC, C, H, N)
    const int A_base = (chunk_idx * C * H + head_idx) * N;
    const int A_step = H * N;

    // K_flat: (BNC, C, H, D)
    const int K_base = (chunk_idx * C * H + head_idx) * D;
    const int K_step = H * D;

    // V_flat: (BNC, C, H, N)
    const int V_base = (chunk_idx * C * H + head_idx) * N;
    const int V_step = H * N;

    // beta_flat: (BNC, C, H)
    const int b_base = chunk_idx * C * H + head_idx;
    const int b_step = H;

    // local_h: (BNC, C, H, N, D)
    const int lh_base = (chunk_idx * C * H + head_idx) * N * D;
    const int lh_step = H * N * D;

    // cum_A: (BNC, C, H, N)
    const int cA_base = (chunk_idx * C * H + head_idx) * N;
    const int cA_step = H * N;

    // Register-resident state: h[N]
    float h[MAX_STATE_DIM];
    #pragma unroll
    for (int s = 0; s < MAX_STATE_DIM; s++) h[s] = 0.0f;

    // Cumulative A_bar product (complex diagonal, identity init)
    // Re/im interleaved: cA[2j]=re, cA[2j+1]=im
    float cA[MAX_STATE_DIM];
    #pragma unroll
    for (int s = 0; s < MAX_STATE_DIM; s += 2) {
        cA[s]     = 1.0f;  // re = 1
        cA[s + 1] = 0.0f;  // im = 0
    }

    for (int t = 0; t < C; t++) {
        // Load A_bar[t] (shared: same for all threads)
        int Ai = A_base + t * A_step;
        float a[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            a[s] = bf16_to_fp32(__ldg(&A_flat[Ai + s]));
        }

        // Load k[t,d] (per-thread), v[t] (shared), beta[t] (shared)
        float k_d  = bf16_to_fp32(__ldg(&K_flat[K_base + t * K_step + d]));
        int Vi = V_base + t * V_step;
        float v[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            v[s] = bf16_to_fp32(__ldg(&V_flat[Vi + s]));
        }
        float beta = bf16_to_fp32(__ldg(&beta_flat[b_base + t * b_step]));

        // 1. Delta-rule retrieval: kTh[s] = sum_d(h[s,d] * k[d])
        //    Batched reduction: amortize to 2 syncthreads total
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
        //    h_new[2j]   = re*h[2j]   - im*h[2j+1]
        //    h_new[2j+1] = re*h[2j+1] + im*h[2j]
        for (int j = 0; j < N / 2; j++) {
            float re = a[2 * j];
            float im = a[2 * j + 1];
            float h_re = h[2 * j];
            float h_im = h[2 * j + 1];
            h[2 * j]     = re * h_re - im * h_im;
            h[2 * j + 1] = re * h_im + im * h_re;
        }

        // 4. Injection
        for (int s = 0; s < N; s++) {
            h[s] += beta * v[s] * k_d;
        }

        // Store local_h[chunk, t, head, s, d]
        int lhi = lh_base + t * lh_step;
        for (int s = 0; s < N; s++) {
            local_h[lhi + s * D + d] = fp32_to_bf16(h[s]);
        }

        // 5. Update cumulative A_bar product: cum_A = A[t] * cum_A (complex diagonal)
        //    For each pair j:
        //      new_re = a_re * cA_re - a_im * cA_im
        //      new_im = a_re * cA_im + a_im * cA_re
        for (int j = 0; j < N / 2; j++) {
            float a_re = a[2 * j];
            float a_im = a[2 * j + 1];
            float c_re = cA[2 * j];
            float c_im = cA[2 * j + 1];
            cA[2 * j]     = a_re * c_re - a_im * c_im;
            cA[2 * j + 1] = a_re * c_im + a_im * c_re;
        }

        // Store cum_A[chunk, t, head, :] — only thread 0 writes
        if (d == 0) {
            int cAi = cA_base + t * cA_step;
            for (int s = 0; s < N; s++) {
                cum_A[cAi + s] = fp32_to_bf16(cA[s]);
            }
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
//   h_mod[s,d] = h_prev[s,d] - beta * kTh[s] * k[d]           (erasure)
//   Complex diagonal rotation:
//     h_evolved[2j,d]   = re*h_mod[2j,d]   - im*h_mod[2j+1,d]
//     h_evolved[2j+1,d] = re*h_mod[2j+1,d] + im*h_mod[2j,d]
//   h_out[s,d] = h_evolved[s,d] + beta * v[s] * k[d]          (injection)
//
// Backward: propagate dh (gradient w.r.t. state) in reverse.

__global__ void intra_chunk_scan_bwd_kernel(
    // Upstream gradients
    const __nv_bfloat16* __restrict__ grad_local_h,  // (BNC, C, H, N, D)
    const __nv_bfloat16* __restrict__ grad_cum_A,    // (BNC, C, H, N)
    // Saved from forward
    const __nv_bfloat16* __restrict__ A_flat,        // (BNC, C, H, N)
    const __nv_bfloat16* __restrict__ K_flat,        // (BNC, C, H, D)
    const __nv_bfloat16* __restrict__ V_flat,        // (BNC, C, H, N)
    const __nv_bfloat16* __restrict__ beta_flat,     // (BNC, C, H)
    const __nv_bfloat16* __restrict__ local_h,       // (BNC, C, H, N, D)
    const __nv_bfloat16* __restrict__ cum_A,         // (BNC, C, H, N)
    // Output gradients
    __nv_bfloat16* __restrict__ grad_A,              // (BNC, C, H, N)
    __nv_bfloat16* __restrict__ grad_K,              // (BNC, C, H, D)
    __nv_bfloat16* __restrict__ grad_V,              // (BNC, C, H, N)
    float* __restrict__ grad_beta,                    // (BNC, C, H) FP32
    // Dimensions
    int BNC, int C, int H, int D, int N
) {
    const int chunk_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    if (chunk_idx >= BNC || head_idx >= H) return;

    const int d = threadIdx.x;
    if (d >= D) return;

    __shared__ float smem[MAX_WARPS * MAX_STATE_DIM];

    const int warp_id  = d / WARP_SIZE;
    const int lane_id  = d % WARP_SIZE;
    const int num_warps = (D + WARP_SIZE - 1) / WARP_SIZE;

    // Strides (same layout as forward)
    const int A_base  = (chunk_idx * C * H + head_idx) * N;
    const int A_step  = H * N;
    const int K_base  = (chunk_idx * C * H + head_idx) * D;
    const int K_step  = H * D;
    const int V_base  = (chunk_idx * C * H + head_idx) * N;
    const int V_step  = H * N;
    const int b_base  = chunk_idx * C * H + head_idx;
    const int b_step  = H;
    const int lh_base = (chunk_idx * C * H + head_idx) * N * D;
    const int lh_step = H * N * D;
    const int cA_base = (chunk_idx * C * H + head_idx) * N;
    const int cA_step = H * N;

    // ---- State gradient accumulator ----
    float dh[MAX_STATE_DIM];
    #pragma unroll
    for (int s = 0; s < MAX_STATE_DIM; s++) dh[s] = 0.0f;

    // ---- CumA gradient accumulator (complex diagonal) ----
    float dcA[MAX_STATE_DIM];
    #pragma unroll
    for (int s = 0; s < MAX_STATE_DIM; s++) dcA[s] = 0.0f;

    for (int t = C - 1; t >= 0; t--) {
        // ---- Add upstream gradient from local_h[t] ----
        int lhi = lh_base + t * lh_step;
        for (int s = 0; s < N; s++) {
            dh[s] += bf16_to_fp32(__ldg(&grad_local_h[lhi + s * D + d]));
        }

        // ---- Load forward values ----
        int Ai = A_base + t * A_step;
        float a[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            a[s] = bf16_to_fp32(__ldg(&A_flat[Ai + s]));
        }

        float k_d  = bf16_to_fp32(__ldg(&K_flat[K_base + t * K_step + d]));
        int Vi = V_base + t * V_step;
        float v[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            v[s] = bf16_to_fp32(__ldg(&V_flat[Vi + s]));
        }
        float beta = bf16_to_fp32(__ldg(&beta_flat[b_base + t * b_step]));

        // h_prev[s,d] = local_h[t-1, s, d], or 0 if t=0
        float hp[MAX_STATE_DIM];
        if (t > 0) {
            int prev_i = lh_base + (t - 1) * lh_step;
            for (int s = 0; s < N; s++) {
                hp[s] = bf16_to_fp32(__ldg(&local_h[prev_i + s * D + d]));
            }
        } else {
            for (int s = 0; s < N; s++) hp[s] = 0.0f;
        }

        // ---- Recompute kTh = h_prev @ k (batched reduction) ----
        float kTh_local[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) kTh_local[s] = hp[s] * k_d;
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

        // ---- Recompute h_mod = h_prev - beta * (kTh outer k) ----
        float hm[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            hm[s] = hp[s] - beta * kTh[s] * k_d;
        }

        
        // Backward through injection: h_out = h_evolved + beta*(v outer k)
        // dh is gradient w.r.t. h_out
        

        // grad_v[s]: sum_d(beta * k_d * dh[s,d]) — batched reduction
        float gv_local[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) gv_local[s] = beta * k_d * dh[s];
        for (int s = 0; s < N; s++) gv_local[s] = warp_reduce_sum(gv_local[s]);
        if (lane_id == 0) {
            for (int s = 0; s < N; s++) smem[warp_id * MAX_STATE_DIM + s] = gv_local[s];
        }
        __syncthreads();
        float gv[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            gv[s] = 0.0f;
            #pragma unroll
            for (int w = 0; w < MAX_WARPS; w++) {
                if (w < num_warps) gv[s] += smem[w * MAX_STATE_DIM + s];
            }
        }
        __syncthreads();

        // grad_k from injection (per-thread): beta * sum_s(v[s]*dh[s,d])
        float gk_inject = 0.0f;
        for (int s = 0; s < N; s++) {
            gk_inject += v[s] * dh[s];
        }
        gk_inject *= beta;

        // grad_beta from injection: sum_{s,d}(v[s]*k[d]*dh[s,d]) — batched reduction
        float gbeta_inject_local = 0.0f;
        for (int s = 0; s < N; s++) {
            gbeta_inject_local += v[s] * k_d * dh[s];
        }
        // Reduce across threads for gbeta
        float gbeta_inject_warp = warp_reduce_sum(gbeta_inject_local);
        if (lane_id == 0) smem[warp_id * MAX_STATE_DIM] = gbeta_inject_warp;
        __syncthreads();
        float gbeta_inject = 0.0f;
        #pragma unroll
        for (int w = 0; w < MAX_WARPS; w++) {
            if (w < num_warps) gbeta_inject += smem[w * MAX_STATE_DIM];
        }
        __syncthreads();

        // Store grad_V (thread 0 writes the N values)
        if (d == 0) {
            int gVi = V_base + t * V_step;
            for (int s = 0; s < N; s++) {
                grad_V[gVi + s] = fp32_to_bf16(gv[s]);
            }
        }

        
        // Backward through rotation: complex diagonal
        //   Forward: h_evolved[2j]   = re*h_mod[2j]   - im*h_mod[2j+1]
        //            h_evolved[2j+1] = re*h_mod[2j+1] + im*h_mod[2j]
        //   A_bar stores conj(mu) = (re, im)
        //
        //   Backward through rotation to get dhm:
        //     Multiply grad by conj of stored conj(mu) = mu = (re, -im):
        //     dhm[2j]   = re*dh[2j]   + im*dh[2j+1]   (note: +im because conj of conj)
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
        //   Forward: h_evolved = stored * h_mod (complex multiply per pair)
        //     h_evolved_re = stored_re * hm_re - stored_im * hm_im
        //     h_evolved_im = stored_re * hm_im + stored_im * hm_re
        //   d/d(stored_re) = dh_re * hm_re + dh_im * hm_im  (summed over D)
        //   d/d(stored_im) = -dh_re * hm_im + dh_im * hm_re (summed over D)
        
        float gA_local[MAX_STATE_DIM];
        for (int j = 0; j < N / 2; j++) {
            float hm_re = hm[2 * j];
            float hm_im = hm[2 * j + 1];
            float dh_re = dh[2 * j];
            float dh_im = dh[2 * j + 1];
            gA_local[2 * j]     = dh_re * hm_re + dh_im * hm_im;   // d/d(stored_re)
            gA_local[2 * j + 1] = -dh_re * hm_im + dh_im * hm_re;  // d/d(stored_im)
        }
        // Batched reduction for grad_A
        for (int s = 0; s < N; s++) gA_local[s] = warp_reduce_sum(gA_local[s]);
        if (lane_id == 0) {
            for (int s = 0; s < N; s++) smem[warp_id * MAX_STATE_DIM + s] = gA_local[s];
        }
        __syncthreads();
        float gA_state[MAX_STATE_DIM];
        for (int s = 0; s < N; s++) {
            gA_state[s] = 0.0f;
            #pragma unroll
            for (int w = 0; w < MAX_WARPS; w++) {
                if (w < num_warps) gA_state[s] += smem[w * MAX_STATE_DIM + s];
            }
        }
        __syncthreads();

        
        // CumA gradient path (register-only, no cross-thread reduction)
        //   cum_A[t] = A[t] * cum_A[t-1] (complex element-wise multiply)
        //   Backward: dcA_accum[t] = grad_cum_A[t] + conj_stored * dcA_accum[t+1]
        //   (At this point dcA holds dcA_accum from step t+1, already propagated)
        

        // Add grad_cum_A[t]
        int gcAi = cA_base + t * cA_step;
        for (int s = 0; s < N; s++) {
            dcA[s] += bf16_to_fp32(__ldg(&grad_cum_A[gcAi + s]));
        }

        // grad_A[t] from cumA path = dcA_accum[t] * conj(cum_A[t-1])
        // cum_A[t-1] is complex diagonal at step t-1
        float prevCA[MAX_STATE_DIM];
        if (t > 0) {
            int pcAi = cA_base + (t - 1) * cA_step;
            for (int s = 0; s < N; s++) {
                prevCA[s] = bf16_to_fp32(__ldg(&cum_A[pcAi + s]));
            }
        } else {
            // cum_A[-1] = identity: re=1, im=0
            for (int j = 0; j < N / 2; j++) {
                prevCA[2 * j]     = 1.0f;
                prevCA[2 * j + 1] = 0.0f;
            }
        }

        // dcA * conj(prevCA) for each complex pair
        // conj(prevCA) for pair j: (prevCA_re, -prevCA_im)
        // (dcA_re + i*dcA_im) * (prevCA_re - i*prevCA_im)
        //   = dcA_re*prevCA_re + dcA_im*prevCA_im
        //   + i*(dcA_im*prevCA_re - dcA_re*prevCA_im)
        float gA_cA[MAX_STATE_DIM];
        for (int j = 0; j < N / 2; j++) {
            float dcA_re  = dcA[2 * j];
            float dcA_im  = dcA[2 * j + 1];
            float pCA_re  = prevCA[2 * j];
            float pCA_im  = prevCA[2 * j + 1];
            gA_cA[2 * j]     = dcA_re * pCA_re + dcA_im * pCA_im;
            gA_cA[2 * j + 1] = dcA_im * pCA_re - dcA_re * pCA_im;
        }

        // Total grad_A[t] = state-path gradient + cumA-path gradient
        if (d == 0) {
            for (int s = 0; s < N; s++) {
                grad_A[gcAi + s] = fp32_to_bf16(gA_state[s] + gA_cA[s]);
            }
        }

        // Propagate dcA backward: dcA_accum for step t-1
        //   dcA_new = conj(A[t]) * dcA (complex multiply)
        //   conj of stored conj(mu) = mu = (re, -im)
        //   (re - i*im) * (dcA_re + i*dcA_im)
        //     = re*dcA_re + im*dcA_im + i*(re*dcA_im - im*dcA_re)
        //   But stored is (re, im) = conj(mu), so conj(stored) = (re, -im):
        //     new_dcA_re = re*dcA_re + im*dcA_im
        //     new_dcA_im = re*dcA_im - im*dcA_re
        for (int j = 0; j < N / 2; j++) {
            float re = a[2 * j];
            float im = a[2 * j + 1];
            float dc_re = dcA[2 * j];
            float dc_im = dcA[2 * j + 1];
            dcA[2 * j]     = re * dc_re + im * dc_im;
            dcA[2 * j + 1] = re * dc_im - im * dc_re;
        }

        
        // Backward through erasure: h_mod = h_prev - beta*(kTh outer k)
        

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
        //   -beta * sum_s(kTh[s]*dhm[s,d] + h_prev[s,d]*zeta[s])
        float gk_erase = 0.0f;
        for (int s = 0; s < N; s++) {
            gk_erase += kTh[s] * dhm[s] + hp[s] * zeta[s];
        }
        gk_erase *= -beta;

        // grad_beta from erasure: -sum_s(kTh[s]*zeta[s])
        float gbeta_erase = 0.0f;
        for (int s = 0; s < N; s++) {
            gbeta_erase -= kTh[s] * zeta[s];
        }

        // Store grad_K, grad_beta
        grad_K[K_base + t * K_step + d] = fp32_to_bf16(gk_inject + gk_erase);

        if (d == 0) {
            grad_beta[b_base + t * b_step] = gbeta_inject + gbeta_erase;
        }

        // Propagate dh backward to h_prev for next iteration
        //   dh_prev[s,d] = dhm[s,d] - beta * zeta[s] * k[d]
        for (int s = 0; s < N; s++) {
            dh[s] = dhm[s] - beta * zeta[s] * k_d;
        }
    }
}



// Launch Functions


std::tuple<torch::Tensor, torch::Tensor>
intra_chunk_scan_fwd_cuda(
    torch::Tensor A_flat,      // (BNC, C, H, N)  BF16
    torch::Tensor K_flat,      // (BNC, C, H, D)  BF16
    torch::Tensor V_flat,      // (BNC, C, H, N)  BF16
    torch::Tensor beta_flat,   // (BNC, C, H)     BF16
    int state_dim              // N (runtime)
) {
    TORCH_CHECK(A_flat.is_cuda(), "A_flat must be CUDA");
    TORCH_CHECK(K_flat.is_cuda(), "K_flat must be CUDA");
    TORCH_CHECK(V_flat.is_cuda(), "V_flat must be CUDA");
    TORCH_CHECK(beta_flat.is_cuda(), "beta_flat must be CUDA");

    A_flat    = A_flat.contiguous();
    K_flat    = K_flat.contiguous();
    V_flat    = V_flat.contiguous();
    beta_flat = beta_flat.contiguous();

    const int BNC = A_flat.size(0);
    const int C   = A_flat.size(1);
    const int H   = A_flat.size(2);
    const int D   = K_flat.size(3);
    const int N   = state_dim;

    TORCH_CHECK(D <= MAX_HEAD_DIM, "head_dim must be <= ", MAX_HEAD_DIM);
    TORCH_CHECK(N <= MAX_STATE_DIM, "state_dim must be <= ", MAX_STATE_DIM);
    TORCH_CHECK(N % 2 == 0, "state_dim must be even (complex pairs)");

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(A_flat.device());
    auto local_h = torch::empty({BNC, C, H, N, D}, opts);
    auto cum_A   = torch::empty({BNC, C, H, N}, opts);

    dim3 grid(BNC, H);
    dim3 block(D);
    size_t smem_bytes = MAX_WARPS * MAX_STATE_DIM * sizeof(float);
    cudaStream_t stream = get_cuda_stream();

    intra_chunk_scan_fwd_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(beta_flat.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(local_h.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(cum_A.data_ptr()),
        BNC, C, H, D, N
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(local_h, cum_A);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
intra_chunk_scan_bwd_cuda(
    torch::Tensor grad_local_h,  // (BNC, C, H, N, D) BF16
    torch::Tensor grad_cum_A,    // (BNC, C, H, N)    BF16
    torch::Tensor A_flat,        // (BNC, C, H, N)    BF16
    torch::Tensor K_flat,        // (BNC, C, H, D)    BF16
    torch::Tensor V_flat,        // (BNC, C, H, N)    BF16
    torch::Tensor beta_flat,     // (BNC, C, H)       BF16
    torch::Tensor local_h,       // (BNC, C, H, N, D) BF16
    torch::Tensor cum_A,         // (BNC, C, H, N)    BF16
    int state_dim                // N (runtime)
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
    const int N   = state_dim;

    TORCH_CHECK(N <= MAX_STATE_DIM, "state_dim must be <= ", MAX_STATE_DIM);
    TORCH_CHECK(N % 2 == 0, "state_dim must be even (complex pairs)");

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(A_flat.device());
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(A_flat.device());

    auto grad_A    = torch::zeros({BNC, C, H, N}, opts_bf16);
    auto grad_K    = torch::zeros({BNC, C, H, D}, opts_bf16);
    auto grad_V    = torch::zeros({BNC, C, H, N}, opts_bf16);
    auto grad_beta = torch::zeros({BNC, C, H}, opts_fp32);

    dim3 grid(BNC, H);
    dim3 block(D);
    size_t smem_bytes = MAX_WARPS * MAX_STATE_DIM * sizeof(float);
    cudaStream_t stream = get_cuda_stream();

    intra_chunk_scan_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
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
        BNC, C, H, D, N
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(grad_A, grad_K, grad_V, grad_beta);
}

}  // namespace cdssm
