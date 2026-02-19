// CDSSM Fused Dynamics Kernel — Complex Diagonal Cayley (v2)
//
// Fuses the ENTIRE dynamics pipeline into a single kernel:
//   1. Parse gate_raw (B, L, G*H) where G = N + 5
//      Layout: [alpha_0..alpha_{N/2-1}, omega_0..omega_{N/2-1},
//               sel_B, sel_C, sel_dt, beta, r_gate] × H
//   2. Per eigenvalue pair j (loop, NOT unrolled for register pressure):
//      a. softplus(alpha_j_raw) → alpha_j
//      b. omega_j = omega_j_raw + position * rope_freqs[h]
//      c. Adaptive timestep (shared across all eigenvalues in a head)
//      d. Cayley discretization → conj(A_bar_j) stored as (re, im)
//      e. Recurrence gate modulation: scale eigenvalue magnitude
//      f. VP scale per eigenvalue: sqrt(1 - |eig_j_eff|^2)
//   3. Fused gates: beta = sigmoid(beta_raw) * sigmoid(sel_B)
//                   sel_C_gate = sigmoid(sel_C_raw)
//
// Grid: (B, cdiv(L, BLOCK_L), H)
// Block: BLOCK_L threads, each processes one (b, l, h) position
//
// BF16 I/O, FP32 compute.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/common.cuh"
#include "../include/cayley_math.cuh"

namespace cdssm {

constexpr int DYN_BLOCK_L = 256;

// Forward Kernel

__global__ void dynamics_fused_fwd_kernel(
    const __nv_bfloat16* __restrict__ gate_raw,  // (B, L, G*H)
    const float* __restrict__ log_dt_scale,       // (H,) FP32
    const float* __restrict__ rope_freqs,         // (H,) FP32
    __nv_bfloat16* __restrict__ A_bar_out,        // (B, L, H, N) BF16
    __nv_bfloat16* __restrict__ vp_scale_out,     // (B, L, H, N/2) BF16
    __nv_bfloat16* __restrict__ beta_out,         // (B, L, H) BF16
    __nv_bfloat16* __restrict__ sel_C_gate_out,   // (B, L, H) BF16
    float gating_c,
    float omega_thresh,
    float adt_delta,
    float adt_smoothness,
    float adt_eps,
    int B, int L, int H, int N  // N = state_dim
) {
    const int b = blockIdx.x;
    const int l_block = blockIdx.y;
    const int h = blockIdx.z;

    const int l = l_block * DYN_BLOCK_L + threadIdx.x;
    if (b >= B || l >= L || h >= H) return;

    const int N_half = N / 2;
    const int G = N + 5;  // gates per head

    // gate_raw layout: (B, L, G*H) with G segments of H each
    const int gate_base = (b * L + l) * (G * H);

    // Per-head parameters
    float log_scale = __ldg(&log_dt_scale[h]);
    float freq = __ldg(&rope_freqs[h]);

    // === Shared scalars (same for all eigenvalue pairs in this head) ===
    // sel_B, sel_C, sel_dt, beta, r_gate are at offsets N_half*2 + {0,1,2,3,4} in gate segments
    int scalar_base = gate_base + N * H;  // after N_half alphas + N_half omegas
    float sel_B_raw  = bf16_to_fp32(__ldg(&gate_raw[scalar_base + 0 * H + h]));
    float sel_C_raw  = bf16_to_fp32(__ldg(&gate_raw[scalar_base + 1 * H + h]));
    float sel_dt_raw = bf16_to_fp32(__ldg(&gate_raw[scalar_base + 2 * H + h]));
    float beta_raw   = bf16_to_fp32(__ldg(&gate_raw[scalar_base + 3 * H + h]));
    float r_gate_raw = bf16_to_fp32(__ldg(&gate_raw[scalar_base + 4 * H + h]));

    float r_gate = sigmoid(r_gate_raw);

    // === Adaptive timestep (use first eigenvalue pair's alpha/omega for base) ===
    // Load alpha_0 and omega_0 for timestep computation
    float alpha0_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + 0 * H + h]));
    float omega0_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + N_half * H + h]));
    float alpha0 = softplus(alpha0_raw);
    float omega0 = omega0_raw + (float)l * freq;

    float dt_scale = softplus(log_scale);
    float omega0_abs = fabsf(omega0);
    float char_freq = alpha0 + omega0_abs + adt_eps;
    float dt_raw_val = dt_scale / char_freq;

    float dt_max = (2.0f - adt_delta) / (alpha0 + adt_eps);
    float blend = sigmoid((omega_thresh - omega0_abs) / adt_smoothness);
    float dt_capped = fminf(dt_raw_val, dt_max);
    float dt_base = blend * dt_capped + (1.0f - blend) * dt_raw_val;
    float dt = dt_base + softplus(sel_dt_raw);

    // === Per-eigenvalue loop (NOT unrolled, reuses registers) ===
    const int blh_idx = (b * L + l) * H + h;
    const int A_out_base = blh_idx * N;
    const int vp_out_base = blh_idx * N_half;

    for (int j = 0; j < N_half; j++) {
        // Load per-eigenvalue gates
        float alpha_raw_j = bf16_to_fp32(__ldg(&gate_raw[gate_base + j * H + h]));
        float omega_raw_j = bf16_to_fp32(__ldg(&gate_raw[gate_base + (N_half + j) * H + h]));

        float alpha_j = softplus(alpha_raw_j);
        float omega_j = omega_raw_j + (float)l * freq;

        // Cayley discretization
        CayleyComplex1D cm = cayley_discretize_complex(alpha_j, omega_j, dt);

        // Eigenvalue magnitude squared
        float eig_sq = cayley_eig_sq(alpha_j, omega_j, dt);
        float eig_sq_clamped = fmaxf(eig_sq, 1e-8f);

        // Recurrence gate modulation
        float exponent = (gating_c * r_gate - 1.0f) / 2.0f;
        float scale = powf(eig_sq_clamped, exponent);

        float a_re = cm.re * scale;
        float a_im = cm.im * scale;

        // Store A_bar (re/im interleaved)
        A_bar_out[A_out_base + 2 * j]     = fp32_to_bf16(a_re);
        A_bar_out[A_out_base + 2 * j + 1] = fp32_to_bf16(a_im);

        // VP scale per eigenvalue pair
        float effective_eig_sq = powf(eig_sq_clamped, gating_c * r_gate);
        float vp = sqrtf(fmaxf(1.0f - effective_eig_sq, EPS_DEFAULT));
        vp = fminf(vp, 1.0f);
        vp_scale_out[vp_out_base + j] = fp32_to_bf16(vp);
    }

    // === Fused gates (scalar per head) ===
    float beta_val = sigmoid(beta_raw) * sigmoid(sel_B_raw);
    float sel_C_val = sigmoid(sel_C_raw);

    beta_out[blh_idx] = fp32_to_bf16(beta_val);
    sel_C_gate_out[blh_idx] = fp32_to_bf16(sel_C_val);
}


// Backward Kernel

__global__ void dynamics_fused_bwd_kernel(
    // Upstream gradients
    const __nv_bfloat16* __restrict__ grad_A_bar,      // (B, L, H, N)
    const __nv_bfloat16* __restrict__ grad_vp_scale,   // (B, L, H, N/2)
    const __nv_bfloat16* __restrict__ grad_beta,       // (B, L, H)
    const __nv_bfloat16* __restrict__ grad_sel_C_gate, // (B, L, H)
    // Saved input
    const __nv_bfloat16* __restrict__ gate_raw,  // (B, L, G*H)
    const float* __restrict__ log_dt_scale,       // (H,)
    const float* __restrict__ rope_freqs,         // (H,)
    // Output gradients
    __nv_bfloat16* __restrict__ grad_gate_raw,    // (B, L, G*H) BF16
    float* __restrict__ grad_log_dt_scale,        // (H,) FP32, atomically accumulated
    // Scalars
    float gating_c,
    float omega_thresh,
    float adt_delta,
    float adt_smoothness,
    float adt_eps,
    int B, int L, int H, int N
) {
    const int b = blockIdx.x;
    const int l_block = blockIdx.y;
    const int h = blockIdx.z;

    const int l = l_block * DYN_BLOCK_L + threadIdx.x;
    if (b >= B || l >= L || h >= H) return;

    const int N_half = N / 2;
    const int G = N + 5;
    const int gate_base = (b * L + l) * (G * H);
    const int blh_idx = (b * L + l) * H + h;

    float log_scale = __ldg(&log_dt_scale[h]);
    float freq = __ldg(&rope_freqs[h]);

    // === Reload shared scalars ===
    int scalar_base = gate_base + N * H;
    float sel_B_raw  = bf16_to_fp32(__ldg(&gate_raw[scalar_base + 0 * H + h]));
    float sel_C_raw  = bf16_to_fp32(__ldg(&gate_raw[scalar_base + 1 * H + h]));
    float sel_dt_raw = bf16_to_fp32(__ldg(&gate_raw[scalar_base + 2 * H + h]));
    float beta_raw_val = bf16_to_fp32(__ldg(&gate_raw[scalar_base + 3 * H + h]));
    float r_gate_raw = bf16_to_fp32(__ldg(&gate_raw[scalar_base + 4 * H + h]));

    float r_gate = sigmoid(r_gate_raw);

    // === Recompute adaptive timestep ===
    float alpha0_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + 0 * H + h]));
    float omega0_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + N_half * H + h]));
    float alpha0 = softplus(alpha0_raw);
    float omega0 = omega0_raw + (float)l * freq;

    float dt_scale_val = softplus(log_scale);
    float omega0_abs = fabsf(omega0);
    float char_freq = alpha0 + omega0_abs + adt_eps;
    float dt_raw_val = dt_scale_val / char_freq;
    float dt_max_val = (2.0f - adt_delta) / (alpha0 + adt_eps);
    float blend = sigmoid((omega_thresh - omega0_abs) / adt_smoothness);
    float dt_capped = fminf(dt_raw_val, dt_max_val);
    bool is_capped = dt_raw_val > dt_max_val;
    float dt_base = blend * dt_capped + (1.0f - blend) * dt_raw_val;
    float sel_dt_sp = softplus(sel_dt_raw);
    float dt = dt_base + sel_dt_sp;

    // === Load upstream scalar gradients ===
    float g_beta_out = bf16_to_fp32(__ldg(&grad_beta[blh_idx]));
    float g_sel_C_out = bf16_to_fp32(__ldg(&grad_sel_C_gate[blh_idx]));

    // === Backward through scalar gates ===
    float sig_beta = sigmoid(beta_raw_val);
    float sig_selB = sigmoid(sel_B_raw);
    float sig_selC = sigmoid(sel_C_raw);

    float g_sel_C_raw = g_sel_C_out * sig_selC * (1.0f - sig_selC);
    float g_beta_raw = g_beta_out * sig_selB * sig_beta * (1.0f - sig_beta);
    float g_sel_B_raw = g_beta_out * sig_beta * sig_selB * (1.0f - sig_selB);

    // === Accumulate dt/r_gate gradients across all eigenvalue pairs ===
    float g_dt_total = 0.0f;
    float g_r_gate_total = 0.0f;

    const int A_grad_base = blh_idx * N;
    const int vp_grad_base = blh_idx * N_half;

    for (int j = 0; j < N_half; j++) {
        // Reload per-eigenvalue gates
        float alpha_raw_j = bf16_to_fp32(__ldg(&gate_raw[gate_base + j * H + h]));
        float omega_raw_j = bf16_to_fp32(__ldg(&gate_raw[gate_base + (N_half + j) * H + h]));

        float alpha_j = softplus(alpha_raw_j);
        float omega_j = omega_raw_j + (float)l * freq;

        // Recompute Cayley
        CayleyComplex1D cm = cayley_discretize_complex(alpha_j, omega_j, dt);
        float eig_sq = cayley_eig_sq(alpha_j, omega_j, dt);
        float eig_sq_clamped = fmaxf(eig_sq, 1e-8f);

        float exponent = (gating_c * r_gate - 1.0f) / 2.0f;
        float scale = powf(eig_sq_clamped, exponent);
        float cr = gating_c * r_gate;
        float effective_eig_sq = powf(eig_sq_clamped, cr);

        float vp_sq = fmaxf(1.0f - effective_eig_sq, EPS_DEFAULT);
        float vp = sqrtf(vp_sq);

        // Load upstream A_bar gradient
        float gA_re = bf16_to_fp32(__ldg(&grad_A_bar[A_grad_base + 2 * j]));
        float gA_im = bf16_to_fp32(__ldg(&grad_A_bar[A_grad_base + 2 * j + 1]));
        float g_vp = bf16_to_fp32(__ldg(&grad_vp_scale[vp_grad_base + j]));

        // --- VP scale backward ---
        float d_eig_eff_sq = 0.0f;
        if (vp > EPS_DEFAULT && vp < 1.0f) {
            d_eig_eff_sq = g_vp * (-0.5f / vp);
        }

        // --- Recurrence gate backward ---
        float d_eig_sq_j = 0.0f;
        float d_r_gate_j = 0.0f;

        // effective_eig_sq = eig_sq^(c*r)
        if (eig_sq_clamped > 1e-8f) {
            d_eig_sq_j += d_eig_eff_sq * cr * powf(eig_sq_clamped, cr - 1.0f);
        }
        d_r_gate_j += d_eig_eff_sq * gating_c * logf(eig_sq_clamped) * effective_eig_sq;

        // A_bar_out = scale * cm → d/d(scale)
        float d_scale = gA_re * cm.re + gA_im * cm.im;
        if (eig_sq_clamped > 1e-8f) {
            d_eig_sq_j += d_scale * exponent * powf(eig_sq_clamped, exponent - 1.0f);
        }
        d_r_gate_j += d_scale * (gating_c / 2.0f) * logf(eig_sq_clamped) * scale;

        // Adjust gA for base Cayley gradient
        gA_re *= scale;
        gA_im *= scale;

        // --- Cayley backward for eigenvalue j ---
        // eig_sq = ((1-ta)^2 + tw^2) / ((1+ta)^2 + tw^2)
        float tau = dt * 0.5f;
        float alpha_c = fmaxf(alpha_j, 0.0f);
        float ta = tau * alpha_c;
        float tw = tau * omega_j;
        float opa = 1.0f + ta;
        float oma = 1.0f - ta;
        float det_numer = opa * opa + tw * tw;
        float inv_det = 1.0f / (det_numer + EPS_DEFAULT);

        // Gradient through eig_sq
        float d_numer_j = d_eig_sq_j * inv_det;
        float d_denom_j = -d_eig_sq_j * eig_sq * inv_det;

        float d_ta_from_eig = d_numer_j * (-2.0f * oma) + d_denom_j * 2.0f * opa;
        float d_tw_from_eig = d_numer_j * 2.0f * tw + d_denom_j * 2.0f * tw;

        // Gradient through Cayley A_bar (conj stored, so adjust signs)
        // cm.re = mu_re, cm.im = -mu_im (conj stored)
        // The upstream grad is for the stored values, so chain rule uses stored values
        // A_bar_re_stored = mu_re, A_bar_im_stored = -mu_im
        // d/d(mu_re) = gA_re, d/d(mu_im) = -gA_im (chain through negation)
        float g_mu_re = gA_re;
        float g_mu_im = -gA_im;  // conj negation

        // mu = N/D, N = (oma, tw), D = (opa, -tw)
        // mu_re = (oma*opa + tw*tw) / det = (1-ta^2-tw^2+2tw^2)...
        // Actually: mu_re = (nr*dr + ni*di)/det, mu_im = (ni*dr - nr*di)/det
        // where nr=oma, ni=tw, dr=opa, di=-tw
        // mu_re = (oma*opa + tw*(-tw))/det = (oma*opa - tw^2)/det = (1-ta^2-tw^2)/det
        // mu_im = (tw*opa - oma*(-tw))/det = (tw*opa + oma*tw)/det = 2*tw/det

        // d(mu_re)/d(ta) = (-2*ta*det - (1-ta^2-tw^2)*2*opa) / det^2
        //                = -(2*ta + (1-ta^2-tw^2)*2*opa/det) / det
        float num_re = 1.0f - ta*ta - tw*tw;
        float da_re_dta = (-2.0f*ta*det_numer - num_re*2.0f*opa) * inv_det * inv_det;
        float da_re_dtw = (-2.0f*tw*det_numer - num_re*2.0f*tw) * inv_det * inv_det;
        float da_im_dta = -2.0f*tw*2.0f*opa * inv_det * inv_det;
        float da_im_dtw = (2.0f*det_numer - 2.0f*tw*2.0f*tw) * inv_det * inv_det;

        float d_ta_from_A = g_mu_re * da_re_dta + g_mu_im * da_im_dta;
        float d_tw_from_A = g_mu_re * da_re_dtw + g_mu_im * da_im_dtw;

        float d_ta_j = d_ta_from_eig + d_ta_from_A;
        float d_tw_j = d_tw_from_eig + d_tw_from_A;

        // ta = tau * alpha, tw = tau * omega, tau = dt/2
        float g_alpha_j = (alpha_j >= 0.0f) ? d_ta_j * tau : 0.0f;
        float g_omega_j = d_tw_j * tau;
        float g_dt_j = d_ta_j * alpha_c * 0.5f + d_tw_j * omega_j * 0.5f;

        g_dt_total += g_dt_j;
        g_r_gate_total += d_r_gate_j;

        // Store per-eigenvalue gate gradients
        float g_alpha_raw_j = g_alpha_j * sigmoid(alpha_raw_j);
        float g_omega_raw_j = g_omega_j;

        grad_gate_raw[gate_base + j * H + h] = fp32_to_bf16(g_alpha_raw_j);
        grad_gate_raw[gate_base + (N_half + j) * H + h] = fp32_to_bf16(g_omega_raw_j);
    }

    // === Backward through adaptive timestep ===
    float g_dt = g_dt_total;
    float g_dt_base = g_dt;
    float g_sel_dt_raw = g_dt * sigmoid(sel_dt_raw);

    float g_dt_raw_adt = g_dt_base * (1.0f - (is_capped ? blend : 0.0f));
    float g_dt_max_adt = g_dt_base * (is_capped ? blend : 0.0f);
    float g_blend = g_dt_base * (dt_capped - dt_raw_val);

    float dblend_domega_abs = -blend * (1.0f - blend) / adt_smoothness;
    float g_dt_scale_val = g_dt_raw_adt / char_freq;
    float g_char_freq = g_dt_raw_adt * (-dt_scale_val / (char_freq * char_freq));
    float g_alpha_from_max = g_dt_max_adt * (-(2.0f - adt_delta) / ((alpha0 + adt_eps) * (alpha0 + adt_eps)));
    float g_alpha_from_freq = g_char_freq;
    float g_omega_abs = g_char_freq + g_blend * dblend_domega_abs;

    float sign_omega = (omega0 >= 0.0f) ? 1.0f : -1.0f;
    float g_omega0_from_adt = g_omega_abs * sign_omega;
    float g_alpha0_from_adt = g_alpha_from_freq + g_alpha_from_max;

    float g_log_scale = g_dt_scale_val * sigmoid(log_scale);

    // Accumulate alpha0/omega0 adt gradients into eigenvalue 0's gradient
    // (alpha0 and omega0 are eigenvalue pair 0)
    float existing_g_alpha0 = bf16_to_fp32(grad_gate_raw[gate_base + 0 * H + h]);
    float existing_g_omega0 = bf16_to_fp32(grad_gate_raw[gate_base + N_half * H + h]);
    existing_g_alpha0 += g_alpha0_from_adt * sigmoid(alpha0_raw);
    existing_g_omega0 += g_omega0_from_adt;
    grad_gate_raw[gate_base + 0 * H + h] = fp32_to_bf16(existing_g_alpha0);
    grad_gate_raw[gate_base + N_half * H + h] = fp32_to_bf16(existing_g_omega0);

    // === Backward through r_gate ===
    float g_r_gate_raw = g_r_gate_total * r_gate * (1.0f - r_gate);

    // === Store scalar gate gradients ===
    grad_gate_raw[scalar_base + 0 * H + h] = fp32_to_bf16(g_sel_B_raw);
    grad_gate_raw[scalar_base + 1 * H + h] = fp32_to_bf16(g_sel_C_raw);
    grad_gate_raw[scalar_base + 2 * H + h] = fp32_to_bf16(g_sel_dt_raw);
    grad_gate_raw[scalar_base + 3 * H + h] = fp32_to_bf16(g_beta_raw);
    grad_gate_raw[scalar_base + 4 * H + h] = fp32_to_bf16(g_r_gate_raw);

    atomicAdd(&grad_log_dt_scale[h], g_log_scale);
}


// Launch Functions

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dynamics_fused_fwd_cuda(
    torch::Tensor gate_raw,       // (B, L, G*H) BF16
    torch::Tensor log_dt_scale,   // (H,) FP32
    torch::Tensor rope_freqs,     // (H,) FP32
    float gating_c,
    float omega_thresh,
    float adt_delta,
    float adt_smoothness,
    float adt_eps,
    int H,
    int state_dim
) {
    TORCH_CHECK(gate_raw.is_cuda(), "gate_raw must be CUDA");

    gate_raw = gate_raw.contiguous();
    log_dt_scale = log_dt_scale.contiguous();
    rope_freqs = rope_freqs.contiguous();

    const int B = gate_raw.size(0);
    const int L = gate_raw.size(1);
    const int N = state_dim;
    const int N_half = N / 2;

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(gate_raw.device());
    auto A_bar_out = torch::empty({B, L, H, N}, opts);
    auto vp_scale_out = torch::empty({B, L, H, N_half}, opts);
    auto beta_out = torch::empty({B, L, H}, opts);
    auto sel_C_gate_out = torch::empty({B, L, H}, opts);

    dim3 grid(B, cdiv(L, DYN_BLOCK_L), H);
    dim3 block(DYN_BLOCK_L);
    cudaStream_t stream = get_cuda_stream();

    dynamics_fused_fwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate_raw.data_ptr()),
        log_dt_scale.data_ptr<float>(),
        rope_freqs.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(A_bar_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(vp_scale_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(beta_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(sel_C_gate_out.data_ptr()),
        gating_c, omega_thresh, adt_delta, adt_smoothness, adt_eps,
        B, L, H, N
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(A_bar_out, vp_scale_out, beta_out, sel_C_gate_out);
}

std::tuple<torch::Tensor, torch::Tensor>
dynamics_fused_bwd_cuda(
    torch::Tensor grad_A_bar,      // (B, L, H, N) BF16
    torch::Tensor grad_vp_scale,   // (B, L, H, N/2) BF16
    torch::Tensor grad_beta,       // (B, L, H) BF16
    torch::Tensor grad_sel_C_gate, // (B, L, H) BF16
    torch::Tensor gate_raw,        // (B, L, G*H) BF16
    torch::Tensor log_dt_scale,    // (H,) FP32
    torch::Tensor rope_freqs,      // (H,) FP32
    float gating_c,
    float omega_thresh,
    float adt_delta,
    float adt_smoothness,
    float adt_eps,
    int H,
    int state_dim
) {
    TORCH_CHECK(gate_raw.is_cuda(), "gate_raw must be CUDA");

    grad_A_bar = grad_A_bar.contiguous();
    grad_vp_scale = grad_vp_scale.contiguous();
    grad_beta = grad_beta.contiguous();
    grad_sel_C_gate = grad_sel_C_gate.contiguous();
    gate_raw = gate_raw.contiguous();
    log_dt_scale = log_dt_scale.contiguous();
    rope_freqs = rope_freqs.contiguous();

    const int B = gate_raw.size(0);
    const int L = gate_raw.size(1);
    const int N = state_dim;
    const int G = N + 5;

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(gate_raw.device());
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(gate_raw.device());
    auto grad_gate_raw = torch::zeros({B, L, G * H}, opts_bf16);
    auto grad_log_dt_scale = torch::zeros({H}, opts_fp32);

    dim3 grid(B, cdiv(L, DYN_BLOCK_L), H);
    dim3 block(DYN_BLOCK_L);
    cudaStream_t stream = get_cuda_stream();

    dynamics_fused_bwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_A_bar.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(grad_vp_scale.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(grad_beta.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(grad_sel_C_gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gate_raw.data_ptr()),
        log_dt_scale.data_ptr<float>(),
        rope_freqs.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(grad_gate_raw.data_ptr()),
        grad_log_dt_scale.data_ptr<float>(),
        gating_c, omega_thresh, adt_delta, adt_smoothness, adt_eps,
        B, L, H, N
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(grad_gate_raw, grad_log_dt_scale);
}

}  // namespace cdssm
