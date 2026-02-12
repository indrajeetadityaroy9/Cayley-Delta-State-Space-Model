// CDSSM Cayley Transform Implementation (SINGLE SOURCE OF TRUTH)
//
// This header provides the AUTHORITATIVE CUDA implementation of the Cayley
// discretization for the CDSSM 2x2 state-space model. All CUDA kernels MUST
// use these functions.
//
//
// Mathematical Background
//
// The CDSSM uses a 2D state-space model with continuous dynamics:
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

namespace cdssm {

// Cayley Matrix Structures

// Discretized matrices from Cayley transform
struct CayleyMatrices {
    float a11, a12, a21, a22;  // A_bar elements (transition matrix)
    float m11, m12, m21, m22;  // M^{-1} elements (for u_bar computation)
};

// Cayley Discretization Functions

// Compute the Cayley discretization matrices A_bar and M^{-1}.
//
// This is the CANONICAL implementation of the Cayley transform.
// All CDSSM CUDA kernels must use this function.
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

}  // namespace cdssm
