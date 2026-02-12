// CDSSM Fused Cayley Discretization + Recurrence Gate + VP Scale — CUDA
//
// Fuses three elementwise operations into a single kernel:
//   1. Cayley discretization: (alpha, omega, dt) → A_bar (2×2)
//   2. Recurrence gate modulation: A_bar *= |eig|^(c*r - 1)
//   3. Variance-preserving scale: sqrt(1 - |eig_eff|²)
//
// Eliminates intermediate tensors A_bar (B,L,H,2,2), eig_sq (B,L,H),
// scale (B,L,H) from Python implementation.
//
// Grid: (B, cdiv(L, BLOCK_L), H)
// Block: BLOCK_L threads, each processes one (b, l, h) position
//
// BF16 I/O, FP32 compute. Uses cayley_math.cuh for canonical discretization.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../include/common.cuh"
#include "../include/cayley_math.cuh"

namespace cdssm {

constexpr int BLOCK_L = 256;

// Forward Kernel

__global__ void cayley_vp_fwd_kernel(
    // Inputs (BF16)
    const __nv_bfloat16* __restrict__ alpha,   // (B, L, H)
    const __nv_bfloat16* __restrict__ omega,   // (B, L, H)
    const __nv_bfloat16* __restrict__ dt,      // (B, L, H)
    const __nv_bfloat16* __restrict__ r_gate,  // (B, L, H) or nullptr
    // Outputs (BF16)
    __nv_bfloat16* __restrict__ A_bar_out,     // (B, L, H, 2, 2)
    __nv_bfloat16* __restrict__ vp_scale_out,  // (B, L, H)
    // Scalars
    float gating_c,
    // Dimensions
    int B, int L, int H
) {
    const int b = blockIdx.x;
    const int l_block = blockIdx.y;
    const int h = blockIdx.z;

    const int l = l_block * BLOCK_L + threadIdx.x;
    if (b >= B || l >= L || h >= H) return;

    // Linear index for (b, l, h) in (B, L, H)
    const int idx = (b * L + l) * H + h;

    float a = bf16_to_fp32(__ldg(&alpha[idx]));
    float w = bf16_to_fp32(__ldg(&omega[idx]));
    float d = bf16_to_fp32(__ldg(&dt[idx]));

    // 1. Cayley discretization
    CayleyMatrices cm = cayley_discretize(a, w, d);

    float a11 = cm.a11, a12 = cm.a12;
    float a21 = cm.a21, a22 = cm.a22;

    // Compute |eigenvalue(A_bar)|²
    // For Cayley A_bar with our A = [[-alpha,omega],[-omega,-alpha]]:
    //   |eig|² = ((1-τα)² + (τω)²) / ((1+τα)² + (τω)²)
    float tau = d * 0.5f;
    float tau_a = tau * fmaxf(a, 0.0f);
    float tau_w = tau * w;

    float numer = (1.0f - tau_a) * (1.0f - tau_a) + tau_w * tau_w;
    float denom = (1.0f + tau_a) * (1.0f + tau_a) + tau_w * tau_w;
    float eig_sq = numer / (denom + EPS_DEFAULT);

    // 2. Recurrence gate modulation (if r_gate provided)
    float effective_eig_sq = eig_sq;
    if (r_gate != nullptr) {
        float r = bf16_to_fp32(__ldg(&r_gate[idx]));
        float exponent = (gating_c * r - 1.0f) / 2.0f;

        // |eig|^(c*r - 1) = (eig_sq)^((c*r-1)/2)
        float eig_sq_clamped = fmaxf(eig_sq, 1e-8f);
        float scale = powf(eig_sq_clamped, exponent);

        a11 *= scale;
        a12 *= scale;
        a21 *= scale;
        a22 *= scale;

        // Effective eigenvalue squared for VP scale
        effective_eig_sq = powf(eig_sq_clamped, gating_c * r);
    }

    // 3. Variance-preserving scale: sqrt(1 - |eig_eff|²)
    float vp = sqrtf(fmaxf(1.0f - effective_eig_sq, EPS_DEFAULT));
    vp = fminf(vp, 1.0f);

    // Store A_bar
    const int A_idx = ((b * L + l) * H + h) * 4;
    A_bar_out[A_idx + 0] = fp32_to_bf16(a11);
    A_bar_out[A_idx + 1] = fp32_to_bf16(a12);
    A_bar_out[A_idx + 2] = fp32_to_bf16(a21);
    A_bar_out[A_idx + 3] = fp32_to_bf16(a22);

    // Store VP scale
    vp_scale_out[idx] = fp32_to_bf16(vp);
}

// Backward Kernel

__global__ void cayley_vp_bwd_kernel(
    // Upstream gradients
    const __nv_bfloat16* __restrict__ grad_A_bar,     // (B, L, H, 2, 2)
    const __nv_bfloat16* __restrict__ grad_vp_scale,  // (B, L, H)
    // Inputs (for recomputation)
    const __nv_bfloat16* __restrict__ alpha,   // (B, L, H)
    const __nv_bfloat16* __restrict__ omega,   // (B, L, H)
    const __nv_bfloat16* __restrict__ dt,      // (B, L, H)
    const __nv_bfloat16* __restrict__ r_gate,  // (B, L, H) or nullptr
    // Output gradients (FP32 for accumulation)
    float* __restrict__ grad_alpha,   // (B, L, H)
    float* __restrict__ grad_omega,   // (B, L, H)
    float* __restrict__ grad_dt,      // (B, L, H)
    float* __restrict__ grad_r_gate,  // (B, L, H) or nullptr
    // Scalars
    float gating_c,
    int B, int L, int H
) {
    const int b = blockIdx.x;
    const int l_block = blockIdx.y;
    const int h = blockIdx.z;

    const int l = l_block * BLOCK_L + threadIdx.x;
    if (b >= B || l >= L || h >= H) return;

    const int idx = (b * L + l) * H + h;
    const int A_idx = idx * 4;

    float a = bf16_to_fp32(__ldg(&alpha[idx]));
    float w = bf16_to_fp32(__ldg(&omega[idx]));
    float d = bf16_to_fp32(__ldg(&dt[idx]));

    // Recompute forward values
    float tau = d * 0.5f;
    float a_clamped = fmaxf(a, 0.0f);
    float tau_a = tau * a_clamped;
    float tau_w = tau * w;

    float opa = 1.0f + tau_a;  // 1 + tau*alpha
    float oma = 1.0f - tau_a;  // 1 - tau*alpha

    float det_numer = opa * opa + tau_w * tau_w;  // det(M)
    float inv_det = 1.0f / (det_numer + EPS_DEFAULT);

    float numer_eig = oma * oma + tau_w * tau_w;
    float eig_sq = numer_eig / (det_numer + EPS_DEFAULT);

    // Load upstream gradients
    float gAb11 = bf16_to_fp32(__ldg(&grad_A_bar[A_idx + 0]));
    float gAb12 = bf16_to_fp32(__ldg(&grad_A_bar[A_idx + 1]));
    float gAb21 = bf16_to_fp32(__ldg(&grad_A_bar[A_idx + 2]));
    float gAb22 = bf16_to_fp32(__ldg(&grad_A_bar[A_idx + 3]));
    float g_vp  = bf16_to_fp32(__ldg(&grad_vp_scale[idx]));

    // Compute Cayley A_bar (recompute for backward)
    CayleyMatrices cm = cayley_discretize(a, w, d);

    float scale = 1.0f;
    float eig_sq_clamped = fmaxf(eig_sq, 1e-8f);
    float r = 0.0f;
    float effective_eig_sq = eig_sq;

    if (r_gate != nullptr) {
        r = bf16_to_fp32(__ldg(&r_gate[idx]));
        float exponent = (gating_c * r - 1.0f) / 2.0f;
        scale = powf(eig_sq_clamped, exponent);
        effective_eig_sq = powf(eig_sq_clamped, gating_c * r);
    }

    // VP scale value
    float vp_val_sq = fmaxf(1.0f - effective_eig_sq, EPS_DEFAULT);
    float vp_val = sqrtf(vp_val_sq);

    // ---- Gradient through VP scale ----
    // vp = sqrt(1 - eig_eff_sq), clamped to [eps, 1]
    // d(vp)/d(eig_eff_sq) = -0.5 / vp (when vp > 0)
    float d_eig_eff_sq = 0.0f;
    if (vp_val > EPS_DEFAULT && vp_val < 1.0f) {
        d_eig_eff_sq = g_vp * (-0.5f / vp_val);
    }

    // ---- Gradient through recurrence gate and eig_sq ----
    float d_eig_sq_total = 0.0f;
    float d_r = 0.0f;
    float d_scale = 0.0f;

    if (r_gate != nullptr) {
        // effective_eig_sq = eig_sq^(c*r)
        // d/d(eig_sq) = c*r * eig_sq^(c*r - 1) = c*r * scale^2 / eig_sq_clamped... complex
        // Simpler: d/d(eig_sq) = c*r * eig_sq^(c*r-1) if eig_sq > 0
        float cr = gating_c * r;
        if (eig_sq_clamped > 1e-8f) {
            d_eig_sq_total += d_eig_eff_sq * cr * powf(eig_sq_clamped, cr - 1.0f);
        }

        // d/d(r) from VP: d_eig_eff_sq * d(eig_sq^(c*r))/dr
        //   = d_eig_eff_sq * gating_c * log(eig_sq) * eig_sq^(c*r)
        d_r += d_eig_eff_sq * gating_c * logf(eig_sq_clamped) * effective_eig_sq;

        // A_bar_out = scale * A_bar_base
        // d/d(scale) = sum_{ij}(gAb[ij] * A_bar_base[ij])
        d_scale = gAb11 * cm.a11 + gAb12 * cm.a12 + gAb21 * cm.a21 + gAb22 * cm.a22;

        // scale = eig_sq^((c*r-1)/2)
        // d(scale)/d(eig_sq) = ((c*r-1)/2) * eig_sq^((c*r-3)/2)
        float exp_minus_1 = (gating_c * r - 1.0f) / 2.0f;
        if (eig_sq_clamped > 1e-8f) {
            d_eig_sq_total += d_scale * exp_minus_1 * powf(eig_sq_clamped, exp_minus_1 - 1.0f);
        }

        // d(scale)/d(r) = (c/2) * log(eig_sq) * scale
        d_r += d_scale * (gating_c / 2.0f) * logf(eig_sq_clamped) * scale;

        // Adjust gAb for the base A_bar gradient (since output = scale * base)
        gAb11 *= scale;
        gAb12 *= scale;
        gAb21 *= scale;
        gAb22 *= scale;
    } else {
        // No gate: effective_eig_sq = eig_sq directly
        d_eig_sq_total += d_eig_eff_sq;
    }

    // ---- Gradient through eig_sq = numer/denom ----
    // numer = (1-tau_a)^2 + tau_w^2
    // denom = (1+tau_a)^2 + tau_w^2
    // d(eig_sq)/d(numer) = 1/denom
    // d(eig_sq)/d(denom) = -numer/denom^2 = -eig_sq/denom
    float d_numer = d_eig_sq_total * inv_det;
    float d_denom_from_eig = -d_eig_sq_total * eig_sq * inv_det;

    // d(numer)/d(tau_a) = -2*(1-tau_a)
    // d(numer)/d(tau_w) = 2*tau_w
    float d_tau_a_from_numer = d_numer * (-2.0f * oma);
    float d_tau_w_from_numer = d_numer * 2.0f * tau_w;

    // d(denom)/d(tau_a) = 2*(1+tau_a)
    // d(denom)/d(tau_w) = 2*tau_w
    float d_tau_a_from_denom = d_denom_from_eig * 2.0f * opa;
    float d_tau_w_from_denom = d_denom_from_eig * 2.0f * tau_w;

    // ---- Gradient through Cayley A_bar ----
    // A_bar = M^{-1} @ N where M = I-τA, N = I+τA
    // This is complex; use numerical identity:
    //   A_bar = [[a11, a12], [-a12, a11]] for our specific A structure
    //   a11 = ((1-τα)(1+τα) - (τω)²) / det
    //   a12 = 2τω / det
    //
    // d(a11)/d(tau_a):
    //   a11 = (1 - (τα)² - (τω)²) / det
    //   Numerator for a11: oma*opa - tau_w^2 = 1 - tau_a^2 - tau_w^2
    //   d(num_a11)/d(tau_a) = -2*tau_a
    //   d(a11) = (d(num_a11)*det - num_a11*d(det)) / det^2
    //   But det = (1+τα)² + (τω)², d(det)/d(tau_a) = 2*opa

    float num_a11 = oma * opa - tau_w * tau_w;
    float num_a12 = 2.0f * tau_w;

    // d(a11)/d(tau_a) = (-2*tau_a * det - num_a11 * 2*opa) / det^2
    float da11_dtau_a = (-2.0f * tau_a * det_numer - num_a11 * 2.0f * opa) * inv_det * inv_det;
    // d(a11)/d(tau_w) = (-2*tau_w * det - num_a11 * 2*tau_w) / det^2
    float da11_dtau_w = (-2.0f * tau_w * det_numer - num_a11 * 2.0f * tau_w) * inv_det * inv_det;
    // d(a12)/d(tau_a) = (0 * det - num_a12 * 2*opa) / det^2 = -num_a12*2*opa / det^2
    float da12_dtau_a = -num_a12 * 2.0f * opa * inv_det * inv_det;
    // d(a12)/d(tau_w) = (2 * det - num_a12 * 2*tau_w) / det^2
    float da12_dtau_w = (2.0f * det_numer - num_a12 * 2.0f * tau_w) * inv_det * inv_det;

    // A_bar structure: [[a11, a12], [-a12, a11]] (antisymmetric off-diagonal)
    // So: gAb for base = gAb11*da11 + gAb12*da12 + gAb21*(-da12) + gAb22*da11
    float d_tau_a_from_A = (gAb11 + gAb22) * da11_dtau_a + (gAb12 - gAb21) * da12_dtau_a;
    float d_tau_w_from_A = (gAb11 + gAb22) * da11_dtau_w + (gAb12 - gAb21) * da12_dtau_w;

    // Total tau_a, tau_w gradients
    float d_tau_a = d_tau_a_from_numer + d_tau_a_from_denom + d_tau_a_from_A;
    float d_tau_w = d_tau_w_from_numer + d_tau_w_from_denom + d_tau_w_from_A;

    // tau_a = tau * alpha (clamped), tau_w = tau * omega, tau = dt/2
    // d/d(alpha) = d_tau_a * tau (if alpha >= 0)
    // d/d(omega) = d_tau_w * tau
    // d/d(dt) = d_tau_a * alpha/2 + d_tau_w * omega/2
    float g_alpha = (a >= 0.0f) ? d_tau_a * tau : 0.0f;
    float g_omega = d_tau_w * tau;
    float g_dt = d_tau_a * a_clamped * 0.5f + d_tau_w * w * 0.5f;

    // Store gradients
    grad_alpha[idx] = g_alpha;
    grad_omega[idx] = g_omega;
    grad_dt[idx] = g_dt;
    if (grad_r_gate != nullptr) {
        grad_r_gate[idx] = d_r;
    }
}


// Launch Functions

std::tuple<torch::Tensor, torch::Tensor>
cayley_vp_fwd_cuda(
    torch::Tensor alpha,   // (B, L, H) BF16
    torch::Tensor omega,   // (B, L, H) BF16
    torch::Tensor dt,      // (B, L, H) BF16
    torch::Tensor r_gate,  // (B, L, H) BF16 or empty
    float gating_c
) {
    TORCH_CHECK(alpha.is_cuda(), "alpha must be CUDA");

    alpha = alpha.contiguous();
    omega = omega.contiguous();
    dt = dt.contiguous();

    const int B = alpha.size(0);
    const int L = alpha.size(1);
    const int H = alpha.size(2);

    bool has_gate = r_gate.numel() > 0;
    if (has_gate) r_gate = r_gate.contiguous();

    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(alpha.device());
    auto A_bar_out = torch::empty({B, L, H, 2, 2}, opts);
    auto vp_scale_out = torch::empty({B, L, H}, opts);

    dim3 grid(B, cdiv(L, BLOCK_L), H);
    dim3 block(BLOCK_L);
    cudaStream_t stream = get_cuda_stream();

    cayley_vp_fwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(alpha.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(omega.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dt.data_ptr()),
        has_gate ? reinterpret_cast<const __nv_bfloat16*>(r_gate.data_ptr()) : nullptr,
        reinterpret_cast<__nv_bfloat16*>(A_bar_out.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(vp_scale_out.data_ptr()),
        gating_c,
        B, L, H
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(A_bar_out, vp_scale_out);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
cayley_vp_bwd_cuda(
    torch::Tensor grad_A_bar,     // (B, L, H, 2, 2) BF16
    torch::Tensor grad_vp_scale,  // (B, L, H) BF16
    torch::Tensor alpha,          // (B, L, H) BF16
    torch::Tensor omega,          // (B, L, H) BF16
    torch::Tensor dt,             // (B, L, H) BF16
    torch::Tensor r_gate,         // (B, L, H) BF16 or empty
    float gating_c
) {
    TORCH_CHECK(alpha.is_cuda(), "alpha must be CUDA");

    grad_A_bar = grad_A_bar.contiguous();
    grad_vp_scale = grad_vp_scale.contiguous();
    alpha = alpha.contiguous();
    omega = omega.contiguous();
    dt = dt.contiguous();

    const int B = alpha.size(0);
    const int L = alpha.size(1);
    const int H = alpha.size(2);

    bool has_gate = r_gate.numel() > 0;
    if (has_gate) r_gate = r_gate.contiguous();

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(alpha.device());
    auto grad_alpha = torch::zeros({B, L, H}, opts);
    auto grad_omega = torch::zeros({B, L, H}, opts);
    auto grad_dt    = torch::zeros({B, L, H}, opts);
    auto grad_r     = has_gate ? torch::zeros({B, L, H}, opts) : torch::empty({0}, opts);

    dim3 grid(B, cdiv(L, BLOCK_L), H);
    dim3 block(BLOCK_L);
    cudaStream_t stream = get_cuda_stream();

    cayley_vp_bwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_A_bar.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(grad_vp_scale.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(alpha.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(omega.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dt.data_ptr()),
        has_gate ? reinterpret_cast<const __nv_bfloat16*>(r_gate.data_ptr()) : nullptr,
        grad_alpha.data_ptr<float>(),
        grad_omega.data_ptr<float>(),
        grad_dt.data_ptr<float>(),
        has_gate ? grad_r.data_ptr<float>() : nullptr,
        gating_c,
        B, L, H
    );
    CUDA_CHECK_LAST();

    return std::make_tuple(grad_alpha, grad_omega, grad_dt, grad_r);
}

}  // namespace cdssm
