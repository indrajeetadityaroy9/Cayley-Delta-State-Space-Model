// CDSSM Fused Dynamics Kernel — CUDA
//
// Fuses the ENTIRE dynamics pipeline into a single kernel:
//   1. Parse gate_raw (B, L, 7*H) into 7 per-head scalars
//   2. softplus(alpha_raw) → alpha
//   3. omega = omega_raw + position * rope_freqs[h] (RoPE modulation)
//   4. Adaptive timestep: dt_base = softplus(log_dt_scale[h]) / (alpha + |omega| + eps)
//      with smooth safety cap
//   5. dt = dt_base + softplus(sel_dt_raw) (input-dependent adjustment)
//   6. sigmoid(r_gate_raw) → r_gate
//   7. Cayley discretization: (alpha, omega, dt) → A_bar (2×2)
//   8. Recurrence gate modulation: A_bar *= |eig|^(c*r - 1)
//   9. Variance-preserving scale: sqrt(1 - |eig_eff|²)
//  10. beta = sigmoid(beta_raw) * sigmoid(sel_B) (fused write gate)
//  11. sel_C_gate = sigmoid(sel_C) (read gate, passed through)
//
// Replaces: adaptive_dt.cu + cayley_vp.cu + ~15 elementwise PyTorch ops
//
// Grid: (B, cdiv(L, BLOCK_L), H)
// Block: BLOCK_L threads, each processes one (b, l, h) position
//
// BF16 I/O, FP32 compute. Save-for-backward: only gate_raw.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/common.cuh"
#include "../include/cayley_math.cuh"

namespace cdssm {

constexpr int DYN_BLOCK_L = 256;

// Forward Kernel

__global__ void dynamics_fused_fwd_kernel(
    // Input
    const __nv_bfloat16* __restrict__ gate_raw,  // (B, L, 7*H) BF16 — layout: [alpha,omega,sel_B,sel_C,sel_dt,beta,r_gate] each H
    const float* __restrict__ log_dt_scale,       // (H,) FP32 parameter
    const float* __restrict__ rope_freqs,         // (H,) FP32 buffer
    // Outputs
    __nv_bfloat16* __restrict__ A_bar_out,        // (B, L, H, 2, 2) BF16
    __nv_bfloat16* __restrict__ vp_scale_out,     // (B, L, H) BF16
    __nv_bfloat16* __restrict__ beta_out,         // (B, L, H) BF16
    __nv_bfloat16* __restrict__ sel_C_gate_out,   // (B, L, H) BF16
    // Scalars
    float gating_c,
    float omega_thresh,
    float adt_delta,
    float adt_smoothness,
    float adt_eps,
    // Dimensions
    int B, int L, int H
) {
    const int b = blockIdx.x;
    const int l_block = blockIdx.y;
    const int h = blockIdx.z;

    const int l = l_block * DYN_BLOCK_L + threadIdx.x;
    if (b >= B || l >= L || h >= H) return;

    // gate_raw layout: (B, L, 7*H) with segments of size H each
    // Segment offsets: alpha=0, omega=H, sel_B=2H, sel_C=3H, sel_dt=4H, beta=5H, r_gate=6H
    const int gate_base = (b * L + l) * (7 * H);
    float alpha_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + 0*H + h]));
    float omega_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + 1*H + h]));
    float sel_B_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + 2*H + h]));
    float sel_C_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + 3*H + h]));
    float sel_dt_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + 4*H + h]));
    float beta_raw  = bf16_to_fp32(__ldg(&gate_raw[gate_base + 5*H + h]));
    float r_gate_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + 6*H + h]));

    // Per-head parameters
    float log_scale = __ldg(&log_dt_scale[h]);
    float freq = __ldg(&rope_freqs[h]);

    // === Step 1: Activations ===
    float alpha = softplus(alpha_raw);                    // softplus for positivity
    float omega = omega_raw + (float)l * freq;            // RoPE modulation
    float r_gate = sigmoid(r_gate_raw);

    // === Step 2: Adaptive Timestep ===
    float dt_scale = softplus(log_scale);
    float omega_abs = fabsf(omega);
    float char_freq = alpha + omega_abs + adt_eps;
    float dt_raw = dt_scale / char_freq;

    // Safety cap: dt_max = (2 - delta) / (alpha + eps)
    float dt_max = (2.0f - adt_delta) / (alpha + adt_eps);

    // Smooth blend near low-omega regime
    float blend = sigmoid((omega_thresh - omega_abs) / adt_smoothness);
    float dt_capped = fminf(dt_raw, dt_max);
    float dt_base = blend * dt_capped + (1.0f - blend) * dt_raw;

    // Input-dependent adjustment
    float dt = dt_base + softplus(sel_dt_raw);

    // === Step 3: Cayley Discretization ===
    CayleyMatrices cm = cayley_discretize(alpha, omega, dt);
    float a11 = cm.a11, a12 = cm.a12;
    float a21 = cm.a21, a22 = cm.a22;

    // |eigenvalue(A_bar)|²
    float tau = dt * 0.5f;
    float tau_a = tau * fmaxf(alpha, 0.0f);
    float tau_w = tau * omega;
    float numer = (1.0f - tau_a) * (1.0f - tau_a) + tau_w * tau_w;
    float denom = (1.0f + tau_a) * (1.0f + tau_a) + tau_w * tau_w;
    float eig_sq = numer / (denom + EPS_DEFAULT);

    // === Step 4: Recurrence Gate Modulation ===
    // Floor prevents powf(0, negative) = Inf. Value ~compute_eps (conservative).
    // Python derivation: config.eps_eigenvalue_floor (see defaults.py).
    float eig_sq_clamped = fmaxf(eig_sq, 1e-8f);
    float exponent = (gating_c * r_gate - 1.0f) / 2.0f;
    float scale = powf(eig_sq_clamped, exponent);

    a11 *= scale;
    a12 *= scale;
    a21 *= scale;
    a22 *= scale;

    float effective_eig_sq = powf(eig_sq_clamped, gating_c * r_gate);

    // === Step 5: VP Scale ===
    // Physical constraint: VP = sqrt(1 - |eig_eff|²) ∈ [0, 1] (energy conservation).
    // EPS_DEFAULT prevents sqrt(0); clamp to 1.0 prevents VP > 1 (non-physical).
    float vp = sqrtf(fmaxf(1.0f - effective_eig_sq, EPS_DEFAULT));
    vp = fminf(vp, 1.0f);

    // === Step 6: Fused Gates ===
    float beta_val = sigmoid(beta_raw) * sigmoid(sel_B_raw);   // write gate
    float sel_C_val = sigmoid(sel_C_raw);                       // read gate

    // === Store Outputs ===
    const int blh_idx = (b * L + l) * H + h;

    // A_bar: (B, L, H, 2, 2)
    const int A_idx = blh_idx * 4;
    A_bar_out[A_idx + 0] = fp32_to_bf16(a11);
    A_bar_out[A_idx + 1] = fp32_to_bf16(a12);
    A_bar_out[A_idx + 2] = fp32_to_bf16(a21);
    A_bar_out[A_idx + 3] = fp32_to_bf16(a22);

    vp_scale_out[blh_idx] = fp32_to_bf16(vp);
    beta_out[blh_idx] = fp32_to_bf16(beta_val);
    sel_C_gate_out[blh_idx] = fp32_to_bf16(sel_C_val);
}


// Backward Kernel

__global__ void dynamics_fused_bwd_kernel(
    // Upstream gradients
    const __nv_bfloat16* __restrict__ grad_A_bar,      // (B, L, H, 2, 2)
    const __nv_bfloat16* __restrict__ grad_vp_scale,   // (B, L, H)
    const __nv_bfloat16* __restrict__ grad_beta,       // (B, L, H)
    const __nv_bfloat16* __restrict__ grad_sel_C_gate, // (B, L, H)
    // Saved input (for recomputation)
    const __nv_bfloat16* __restrict__ gate_raw,  // (B, L, 7*H)
    const float* __restrict__ log_dt_scale,       // (H,)
    const float* __restrict__ rope_freqs,         // (H,)
    // Output gradients
    __nv_bfloat16* __restrict__ grad_gate_raw,    // (B, L, 7*H) BF16
    float* __restrict__ grad_log_dt_scale,        // (H,) FP32, atomically accumulated
    // Scalars
    float gating_c,
    float omega_thresh,
    float adt_delta,
    float adt_smoothness,
    float adt_eps,
    // Dimensions
    int B, int L, int H
) {
    const int b = blockIdx.x;
    const int l_block = blockIdx.y;
    const int h = blockIdx.z;

    const int l = l_block * DYN_BLOCK_L + threadIdx.x;
    if (b >= B || l >= L || h >= H) return;

    const int gate_base = (b * L + l) * (7 * H);
    const int blh_idx = (b * L + l) * H + h;

    // === Reload raw gate values ===
    float alpha_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + 0*H + h]));
    float omega_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + 1*H + h]));
    float sel_B_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + 2*H + h]));
    float sel_C_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + 3*H + h]));
    float sel_dt_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + 4*H + h]));
    float beta_raw_val = bf16_to_fp32(__ldg(&gate_raw[gate_base + 5*H + h]));
    float r_gate_raw = bf16_to_fp32(__ldg(&gate_raw[gate_base + 6*H + h]));

    float log_scale = __ldg(&log_dt_scale[h]);
    float freq = __ldg(&rope_freqs[h]);

    // === Recompute forward ===
    float alpha = softplus(alpha_raw);
    float omega = omega_raw + (float)l * freq;
    float r_gate = sigmoid(r_gate_raw);

    float dt_scale_val = softplus(log_scale);
    float omega_abs = fabsf(omega);
    float char_freq = alpha + omega_abs + adt_eps;
    float dt_raw_val = dt_scale_val / char_freq;
    float dt_max_val = (2.0f - adt_delta) / (alpha + adt_eps);
    float blend = sigmoid((omega_thresh - omega_abs) / adt_smoothness);
    float dt_capped = fminf(dt_raw_val, dt_max_val);
    bool is_capped = dt_raw_val > dt_max_val;
    float dt_base = blend * dt_capped + (1.0f - blend) * dt_raw_val;
    float sel_dt_sp = softplus(sel_dt_raw);
    float dt = dt_base + sel_dt_sp;

    float tau = dt * 0.5f;
    float alpha_clamped = fmaxf(alpha, 0.0f);
    float tau_a = tau * alpha_clamped;
    float tau_w = tau * omega;
    float opa = 1.0f + tau_a;
    float oma = 1.0f - tau_a;
    float det_numer = opa * opa + tau_w * tau_w;
    float inv_det = 1.0f / (det_numer + EPS_DEFAULT);
    float numer_eig = oma * oma + tau_w * tau_w;
    float eig_sq = numer_eig / (det_numer + EPS_DEFAULT);

    CayleyMatrices cm = cayley_discretize(alpha, omega, dt);

    float eig_sq_clamped = fmaxf(eig_sq, 1e-8f);
    float exponent_val = (gating_c * r_gate - 1.0f) / 2.0f;
    float scale = powf(eig_sq_clamped, exponent_val);
    float effective_eig_sq = powf(eig_sq_clamped, gating_c * r_gate);

    float vp_val_sq = fmaxf(1.0f - effective_eig_sq, EPS_DEFAULT);
    float vp_val = sqrtf(vp_val_sq);

    float sig_beta = sigmoid(beta_raw_val);
    float sig_selB = sigmoid(sel_B_raw);
    float sig_selC = sigmoid(sel_C_raw);

    // === Load upstream gradients ===
    const int A_idx = blh_idx * 4;
    float gAb11 = bf16_to_fp32(__ldg(&grad_A_bar[A_idx + 0]));
    float gAb12 = bf16_to_fp32(__ldg(&grad_A_bar[A_idx + 1]));
    float gAb21 = bf16_to_fp32(__ldg(&grad_A_bar[A_idx + 2]));
    float gAb22 = bf16_to_fp32(__ldg(&grad_A_bar[A_idx + 3]));
    float g_vp = bf16_to_fp32(__ldg(&grad_vp_scale[blh_idx]));
    float g_beta_out = bf16_to_fp32(__ldg(&grad_beta[blh_idx]));
    float g_sel_C_out = bf16_to_fp32(__ldg(&grad_sel_C_gate[blh_idx]));

    // =====================================================
    // BACKWARD: Chain rule through all fused operations
    // =====================================================

    // --- Gradient through sel_C_gate = sigmoid(sel_C_raw) ---
    float g_sel_C_raw = g_sel_C_out * sig_selC * (1.0f - sig_selC);

    // --- Gradient through beta = sigmoid(beta_raw) * sigmoid(sel_B_raw) ---
    float g_beta_raw = g_beta_out * sig_selB * sig_beta * (1.0f - sig_beta);
    float g_sel_B_raw = g_beta_out * sig_beta * sig_selB * (1.0f - sig_selB);

    // --- Gradient through VP scale ---
    float d_eig_eff_sq = 0.0f;
    if (vp_val > EPS_DEFAULT && vp_val < 1.0f) {
        d_eig_eff_sq = g_vp * (-0.5f / vp_val);
    }

    // --- Gradient through recurrence gate ---
    float d_eig_sq_total = 0.0f;
    float d_r_gate = 0.0f;

    // effective_eig_sq = eig_sq^(c*r)
    float cr = gating_c * r_gate;
    if (eig_sq_clamped > 1e-8f) {
        d_eig_sq_total += d_eig_eff_sq * cr * powf(eig_sq_clamped, cr - 1.0f);
    }
    d_r_gate += d_eig_eff_sq * gating_c * logf(eig_sq_clamped) * effective_eig_sq;

    // A_bar_out = scale * A_bar_base → d/d(scale)
    float d_scale = gAb11 * cm.a11 + gAb12 * cm.a12 + gAb21 * cm.a21 + gAb22 * cm.a22;

    // scale = eig_sq^((c*r-1)/2)
    if (eig_sq_clamped > 1e-8f) {
        d_eig_sq_total += d_scale * exponent_val * powf(eig_sq_clamped, exponent_val - 1.0f);
    }
    d_r_gate += d_scale * (gating_c / 2.0f) * logf(eig_sq_clamped) * scale;

    // Adjust gAb for base A_bar gradient (output = scale * base)
    gAb11 *= scale;
    gAb12 *= scale;
    gAb21 *= scale;
    gAb22 *= scale;

    // --- Gradient through eig_sq = numer/denom ---
    float d_numer = d_eig_sq_total * inv_det;
    float d_denom_from_eig = -d_eig_sq_total * eig_sq * inv_det;

    float d_tau_a_from_numer = d_numer * (-2.0f * oma);
    float d_tau_w_from_numer = d_numer * 2.0f * tau_w;
    float d_tau_a_from_denom = d_denom_from_eig * 2.0f * opa;
    float d_tau_w_from_denom = d_denom_from_eig * 2.0f * tau_w;

    // --- Gradient through Cayley A_bar ---
    float num_a11 = oma * opa - tau_w * tau_w;
    float num_a12 = 2.0f * tau_w;
    float da11_dtau_a = (-2.0f * tau_a * det_numer - num_a11 * 2.0f * opa) * inv_det * inv_det;
    float da11_dtau_w = (-2.0f * tau_w * det_numer - num_a11 * 2.0f * tau_w) * inv_det * inv_det;
    float da12_dtau_a = -num_a12 * 2.0f * opa * inv_det * inv_det;
    float da12_dtau_w = (2.0f * det_numer - num_a12 * 2.0f * tau_w) * inv_det * inv_det;

    float d_tau_a_from_A = (gAb11 + gAb22) * da11_dtau_a + (gAb12 - gAb21) * da12_dtau_a;
    float d_tau_w_from_A = (gAb11 + gAb22) * da11_dtau_w + (gAb12 - gAb21) * da12_dtau_w;

    float d_tau_a = d_tau_a_from_numer + d_tau_a_from_denom + d_tau_a_from_A;
    float d_tau_w = d_tau_w_from_numer + d_tau_w_from_denom + d_tau_w_from_A;

    // tau = dt/2, tau_a = tau * alpha, tau_w = tau * omega
    float g_alpha_from_cayley = (alpha >= 0.0f) ? d_tau_a * tau : 0.0f;
    float g_omega_from_cayley = d_tau_w * tau;
    float g_dt_from_cayley = d_tau_a * alpha_clamped * 0.5f + d_tau_w * omega * 0.5f;

    // --- Gradient through dt = dt_base + softplus(sel_dt_raw) ---
    float g_dt = g_dt_from_cayley;
    float g_dt_base = g_dt;
    float g_sel_dt_raw = g_dt * sigmoid(sel_dt_raw);  // softplus_backward

    // --- Gradient through adaptive timestep ---
    // dt_base = blend * dt_capped + (1-blend) * dt_raw
    float g_dt_raw_adt = g_dt_base * (1.0f - (is_capped ? blend : 0.0f));
    float g_dt_max_adt = g_dt_base * (is_capped ? blend : 0.0f);
    float g_blend = g_dt_base * (dt_capped - dt_raw_val);

    // blend = sigmoid((omega_thresh - omega_abs) / smoothness)
    float dblend_domega_abs = -blend * (1.0f - blend) / adt_smoothness;

    // dt_raw = dt_scale / char_freq
    float g_dt_scale_val = g_dt_raw_adt / char_freq;
    float g_char_freq = g_dt_raw_adt * (-dt_scale_val / (char_freq * char_freq));

    // dt_max = (2-delta) / (alpha+eps)
    float g_alpha_from_max = g_dt_max_adt * (-(2.0f - adt_delta) / ((alpha + adt_eps) * (alpha + adt_eps)));

    // char_freq = alpha + |omega| + eps
    float g_alpha_from_freq = g_char_freq;
    float g_omega_abs = g_char_freq + g_blend * dblend_domega_abs;

    // |omega| → omega
    float sign_omega = (omega >= 0.0f) ? 1.0f : -1.0f;
    float g_omega_from_adt = g_omega_abs * sign_omega;
    float g_alpha_from_adt = g_alpha_from_freq + g_alpha_from_max;

    // dt_scale = softplus(log_scale)
    float g_log_scale = g_dt_scale_val * sigmoid(log_scale);

    // --- Gradient through r_gate = sigmoid(r_gate_raw) ---
    float g_r_gate_raw = d_r_gate * r_gate * (1.0f - r_gate);

    // --- Gradient through omega = omega_raw + l * freq ---
    float g_omega_raw = g_omega_from_cayley + g_omega_from_adt;

    // --- Gradient through alpha = softplus(alpha_raw) ---
    float g_alpha_total = g_alpha_from_cayley + g_alpha_from_adt;
    float g_alpha_raw = g_alpha_total * sigmoid(alpha_raw);  // softplus_backward

    // === Store gradient outputs ===
    grad_gate_raw[gate_base + 0*H + h] = fp32_to_bf16(g_alpha_raw);
    grad_gate_raw[gate_base + 1*H + h] = fp32_to_bf16(g_omega_raw);
    grad_gate_raw[gate_base + 2*H + h] = fp32_to_bf16(g_sel_B_raw);
    grad_gate_raw[gate_base + 3*H + h] = fp32_to_bf16(g_sel_C_raw);
    grad_gate_raw[gate_base + 4*H + h] = fp32_to_bf16(g_sel_dt_raw);
    grad_gate_raw[gate_base + 5*H + h] = fp32_to_bf16(g_beta_raw);
    grad_gate_raw[gate_base + 6*H + h] = fp32_to_bf16(g_r_gate_raw);

    // Atomic accumulation for per-head log_dt_scale gradient
    atomicAdd(&grad_log_dt_scale[h], g_log_scale);
}


// Launch Functions

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dynamics_fused_fwd_cuda(
    torch::Tensor gate_raw,       // (B, L, 7*H) BF16
    torch::Tensor log_dt_scale,   // (H,) FP32
    torch::Tensor rope_freqs,     // (H,) FP32
    float gating_c,
    float omega_thresh,
    float adt_delta,
    float adt_smoothness,
    float adt_eps,
    int H
) {
    TORCH_CHECK(gate_raw.is_cuda(), "gate_raw must be CUDA");

    gate_raw = gate_raw.contiguous();
    log_dt_scale = log_dt_scale.contiguous();
    rope_freqs = rope_freqs.contiguous();

    const int B = gate_raw.size(0);
    const int L = gate_raw.size(1);

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(gate_raw.device());
    auto A_bar_out = torch::empty({B, L, H, 2, 2}, opts);
    auto vp_scale_out = torch::empty({B, L, H}, opts);
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
        B, L, H
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(A_bar_out, vp_scale_out, beta_out, sel_C_gate_out);
}

std::tuple<torch::Tensor, torch::Tensor>
dynamics_fused_bwd_cuda(
    torch::Tensor grad_A_bar,      // (B, L, H, 2, 2) BF16
    torch::Tensor grad_vp_scale,   // (B, L, H) BF16
    torch::Tensor grad_beta,       // (B, L, H) BF16
    torch::Tensor grad_sel_C_gate, // (B, L, H) BF16
    torch::Tensor gate_raw,        // (B, L, 7*H) BF16 saved
    torch::Tensor log_dt_scale,    // (H,) FP32
    torch::Tensor rope_freqs,      // (H,) FP32
    float gating_c,
    float omega_thresh,
    float adt_delta,
    float adt_smoothness,
    float adt_eps,
    int H
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

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(gate_raw.device());
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(gate_raw.device());
    auto grad_gate_raw = torch::zeros({B, L, 7 * H}, opts_bf16);
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
        B, L, H
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(grad_gate_raw, grad_log_dt_scale);
}

}  // namespace cdssm
