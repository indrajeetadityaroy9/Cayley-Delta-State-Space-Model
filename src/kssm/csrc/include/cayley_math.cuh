// KSSM Cayley Transform Implementation (SINGLE SOURCE OF TRUTH)
//
// This header provides the AUTHORITATIVE CUDA implementation of the Cayley
// discretization for the KSSM 2x2 state-space model. All CUDA kernels MUST
// use these functions.
//
//
// Mathematical Background
// =======================
//
// The KSSM uses a 2D state-space model with continuous dynamics:
//     dz/dt = A(t)z + Bu
//
// where A = [[-α, ω], [-ω, -α]] is a 2x2 rotation-damping matrix.
//
// The Cayley transform discretizes this as:
//     z_{t+1} = A_bar @ z_t + u_bar
//
// where A_bar = (I - τA)^{-1}(I + τA) with τ = dt/2.
//
// A-Stability Guarantee
// =====================
//
// The Cayley transform guarantees |eigenvalue(A_bar)| ≤ 1 for any α ≥ 0.
// This ensures unconditional numerical stability for arbitrary sequence lengths.
//
// Proof sketch:
// - For A = [[-α, ω], [-ω, -α]] with α ≥ 0, the eigenvalues are λ = -α ± iω
// - These have Re(λ) ≤ 0 (in the left half-plane)
// - The Cayley transform maps the left half-plane to the unit disk
// - Therefore |eigenvalue(A_bar)| ≤ 1
//
// The key numerical stability factor is the determinant:
//     det(M) = (1 + τα)² + (τω)²
//
// This is always ≥ 1 for α ≥ 0, so the inverse is well-conditioned.
// We add eps to the determinant as a safeguard against numerical edge cases.

#pragma once

#include "common.cuh"

namespace kssm {

// ============================================================================
// Cayley Matrix Structures
// ============================================================================

// Discretized matrices from Cayley transform
struct CayleyMatrices {
    float a11, a12, a21, a22;  // A_bar elements (transition matrix)
    float m11, m12, m21, m22;  // M^{-1} elements (for u_bar computation)
};

// Extended structure with intermediates for backward pass
struct CayleyIntermediates {
    // A_bar elements
    float a11, a12, a21, a22;
    // M^{-1} elements
    float m11, m12, m21, m22;
    // Intermediate values needed for gradient computation
    float tau;                   // dt / 2
    float tau_omega;             // tau * omega
    float one_plus_tau_alpha;    // 1 + tau * alpha
    float one_minus_tau_alpha;   // 1 - tau * alpha
    float inv_det;               // 1 / (det_M + eps)
};

// ============================================================================
// Cayley Discretization Functions
// ============================================================================

// Compute the Cayley discretization matrices A_bar and M^{-1}.
//
// This is the CANONICAL implementation of the Cayley transform.
// All KSSM CUDA kernels must use this function.
//
// The Cayley transform discretizes the continuous 2x2 system:
//     A = [[-α, ω], [-ω, -α]]
//
// into the discrete system:
//     A_bar = (I - τA)^{-1}(I + τA), τ = dt/2
//
// Args:
//     alpha: Damping coefficient. Must be ≥ 0 for stability.
//     omega: Frequency coefficient.
//     dt: Timestep. Must be > 0.
//     eps: Small constant for numerical stability (typically 1e-6).
//
// Returns:
//     CayleyMatrices containing A_bar and M^{-1} elements.
//
// Mathematical Details:
//     M = I - τA = [[1 + τα, -τω], [τω, 1 + τα]]
//     det(M) = (1 + τα)² + (τω)²  [always ≥ 1 for α ≥ 0]
//     M^{-1} = (1/det) * [[1 + τα, τω], [-τω, 1 + τα]]
//
//     N = I + τA = [[1 - τα, τω], [-τω, 1 - τα]]
//
//     A_bar = M^{-1} @ N
__device__ __forceinline__ CayleyMatrices cayley_discretize(
    float alpha,
    float omega,
    float dt,
    float eps = EPS_DEFAULT
) {
    // Clamp alpha >= 0 to guarantee A-stability (|eigenvalue(A_bar)| <= 1)
    alpha = fmaxf(alpha, 0.0f);

    // Compute τ = dt/2 (half-timestep for Cayley)
    float tau = dt * 0.5f;

    // Compute frequently-used terms using FMA for precision
    float tau_omega = tau * omega;
    float one_plus_tau_alpha = __fmaf_rn(tau, alpha, 1.0f);
    float one_minus_tau_alpha = __fmaf_rn(-tau, alpha, 1.0f);

    // Determinant of M = I - τA
    // det(M) = (1 + τα)² + (τω)²
    // This is always ≥ 1 for α ≥ 0, ensuring numerical stability
    float det_M = __fmaf_rn(one_plus_tau_alpha, one_plus_tau_alpha,
                            tau_omega * tau_omega);

    // Add eps for numerical safety and compute inverse determinant
    // Use IEEE-compliant division for precision (avoids __fdividef approximation)
    float inv_det = __fdiv_rn(1.0f, det_M + eps);

    CayleyMatrices result;

    // M^{-1} = (1/det) * [[1 + τα, τω], [-τω, 1 + τα]]
    result.m11 = one_plus_tau_alpha * inv_det;
    result.m12 = tau_omega * inv_det;
    result.m21 = -tau_omega * inv_det;
    result.m22 = one_plus_tau_alpha * inv_det;  // Note: m22 = m11

    // N = I + τA = [[1 - τα, τω], [-τω, 1 - τα]]
    // (N elements used inline below)

    // A_bar = M^{-1} @ N (2x2 matrix multiply)
    // Using FMA for precision: a*b + c*d = fma(a,b, c*d)
    float neg_tau_omega = -tau_omega;
    result.a11 = __fmaf_rn(result.m11, one_minus_tau_alpha, result.m12 * neg_tau_omega);
    result.a12 = __fmaf_rn(result.m11, tau_omega, result.m12 * one_minus_tau_alpha);
    result.a21 = __fmaf_rn(result.m21, one_minus_tau_alpha, result.m22 * neg_tau_omega);
    result.a22 = __fmaf_rn(result.m21, tau_omega, result.m22 * one_minus_tau_alpha);

    return result;
}

// Compute Cayley discretization with all intermediate values for backward pass.
//
// This variant returns additional intermediate values needed for gradient
// computation in the backward pass. Use this in backward kernels that need
// to chain through the Cayley transform.
__device__ __forceinline__ CayleyIntermediates cayley_discretize_with_intermediates(
    float alpha,
    float omega,
    float dt,
    float eps = EPS_DEFAULT
) {
    // Clamp alpha >= 0 to guarantee A-stability (|eigenvalue(A_bar)| <= 1)
    alpha = fmaxf(alpha, 0.0f);

    // Compute τ = dt/2 (half-timestep for Cayley)
    float tau = dt * 0.5f;

    // Compute frequently-used terms using FMA for precision
    float tau_omega = tau * omega;
    float one_plus_tau_alpha = __fmaf_rn(tau, alpha, 1.0f);
    float one_minus_tau_alpha = __fmaf_rn(-tau, alpha, 1.0f);

    // Determinant of M = I - τA using FMA
    float det_M = __fmaf_rn(one_plus_tau_alpha, one_plus_tau_alpha,
                            tau_omega * tau_omega);

    // Add eps for numerical safety and compute inverse determinant
    // Use IEEE-compliant division for precision (avoids __fdividef approximation)
    float inv_det = __fdiv_rn(1.0f, det_M + eps);

    CayleyIntermediates result;

    // Store intermediates
    result.tau = tau;
    result.tau_omega = tau_omega;
    result.one_plus_tau_alpha = one_plus_tau_alpha;
    result.one_minus_tau_alpha = one_minus_tau_alpha;
    result.inv_det = inv_det;

    // M^{-1} = (1/det) * [[1 + τα, τω], [-τω, 1 + τα]]
    result.m11 = one_plus_tau_alpha * inv_det;
    result.m12 = tau_omega * inv_det;
    result.m21 = -tau_omega * inv_det;
    result.m22 = one_plus_tau_alpha * inv_det;

    // A_bar = M^{-1} @ N using FMA
    float neg_tau_omega = -tau_omega;
    result.a11 = __fmaf_rn(result.m11, one_minus_tau_alpha, result.m12 * neg_tau_omega);
    result.a12 = __fmaf_rn(result.m11, tau_omega, result.m12 * one_minus_tau_alpha);
    result.a21 = __fmaf_rn(result.m21, one_minus_tau_alpha, result.m22 * neg_tau_omega);
    result.a22 = __fmaf_rn(result.m21, tau_omega, result.m22 * one_minus_tau_alpha);

    return result;
}

// ============================================================================
// Input Discretization
// ============================================================================

// Compute the discretized input u_bar = dt * M^{-1} @ (B * x).
//
// This function computes the input term for the discrete state-space equation:
//     z_{t+1} = A_bar @ z_t + u_bar
//
// Args:
//     m: CayleyMatrices containing M^{-1} elements.
//     B0, B1: Input projection vector [B0, B1].
//     x: Input value.
//     dt: Timestep.
//     u0, u1: Output - elements of u_bar.
//
// Mathematical Details:
//     Bx = B * x = [B0 * x, B1 * x]
//     u_bar = dt * M^{-1} @ Bx
//           = dt * [[m11, m12], [m21, m22]] @ [B0*x, B1*x]
__device__ __forceinline__ void cayley_u_bar(
    const CayleyMatrices& m,
    float B0,
    float B1,
    float x,
    float dt,
    float& u0,
    float& u1
) {
    // Compute B @ x (element-wise since B is per-channel)
    float Bx0 = B0 * x;
    float Bx1 = B1 * x;

    // Compute u_bar = dt * M^{-1} @ Bx
    u0 = dt * (m.m11 * Bx0 + m.m12 * Bx1);
    u1 = dt * (m.m21 * Bx0 + m.m22 * Bx1);
}

// Overload taking individual M^{-1} elements
__device__ __forceinline__ void cayley_u_bar(
    float m11, float m12, float m21, float m22,
    float B0,
    float B1,
    float x,
    float dt,
    float& u0,
    float& u1
) {
    float Bx0 = B0 * x;
    float Bx1 = B1 * x;

    u0 = dt * (m11 * Bx0 + m12 * Bx1);
    u1 = dt * (m21 * Bx0 + m22 * Bx1);
}

// ============================================================================
// State Update
// ============================================================================

// Apply the discrete state-space update: h_new = A_bar @ h + u_bar.
//
// This is the core recurrence that propagates state through time.
//
// Args:
//     a: CayleyMatrices containing A_bar elements.
//     h1, h2: Current state vector [h1, h2].
//     u0, u1: Discretized input from cayley_u_bar().
//     new_h1, new_h2: Output - updated state vector.
//
// Mathematical Details:
//     [new_h1]   [a11 a12] [h1]   [u0]
//     [new_h2] = [a21 a22] [h2] + [u1]
__device__ __forceinline__ void cayley_state_update(
    const CayleyMatrices& a,
    float h1,
    float h2,
    float u0,
    float u1,
    float& new_h1,
    float& new_h2
) {
    new_h1 = a.a11 * h1 + a.a12 * h2 + u0;
    new_h2 = a.a21 * h1 + a.a22 * h2 + u1;
}

// Overload taking individual A_bar elements
__device__ __forceinline__ void cayley_state_update(
    float a11, float a12, float a21, float a22,
    float h1,
    float h2,
    float u0,
    float u1,
    float& new_h1,
    float& new_h2
) {
    new_h1 = a11 * h1 + a12 * h2 + u0;
    new_h2 = a21 * h1 + a22 * h2 + u1;
}

// Combined state update: discretize + compute u_bar + update state
// This is the most common pattern in the evolution kernel
__device__ __forceinline__ void cayley_full_step(
    float alpha,
    float omega,
    float dt,
    float B0,
    float B1,
    float x,
    float h1,
    float h2,
    float& new_h1,
    float& new_h2,
    float eps = EPS_DEFAULT
) {
    // Compute Cayley matrices
    CayleyMatrices cm = cayley_discretize(alpha, omega, dt, eps);

    // Compute discretized input
    float u0, u1;
    cayley_u_bar(cm, B0, B1, x, dt, u0, u1);

    // Update state
    cayley_state_update(cm, h1, h2, u0, u1, new_h1, new_h2);
}

}  // namespace kssm
