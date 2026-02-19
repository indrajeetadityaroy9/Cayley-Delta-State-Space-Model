// CDSSM Cayley Transform — Complex Diagonal Implementation
//
// Provides the AUTHORITATIVE CUDA implementation of the Cayley discretization
// for the CDSSM complex diagonal state-space model.
//
// Mathematical Background
//
// The CDSSM uses N/2 independent complex eigenvalue pairs per head.
// For each eigenvalue lambda_j = -alpha_j + i*omega_j (alpha >= 0),
// the Cayley transform produces:
//
//     A_bar_j = (1 + tau*lambda_j) / (1 - tau*lambda_j),  tau = dt/2
//
// A-Stability Guarantee
//
// |A_bar_j|^2 = ((1-ta)^2 + tw^2) / ((1+ta)^2 + tw^2) <= 1
// for any alpha >= 0, unconditionally. This maps the left half-plane
// to the unit disk, guaranteeing numerical stability.
//
// Rotation Direction
//
// We store conj(A_bar_j) = (re, -im) to match the rotation direction
// of the original 2x2 Cayley matrix [[re, im], [-im, re]], which acts
// as multiplication by conj(mu) on complex state h0 + i*h1.

#pragma once

#include "common.cuh"

namespace cdssm {

// Complex Cayley discretization result (stores conj of eigenvalue)
struct CayleyComplex1D {
    float re, im;  // conj(A_bar): re + i*im where im is negated
};

// Compute the complex diagonal Cayley discretization.
//
// For eigenvalue lambda = -alpha + i*omega:
//   A_bar = (1 + tau*lambda) / (1 - tau*lambda)
//   Returns conj(A_bar) for correct rotation direction.
//
// Args:
//   alpha: Damping coefficient (>= 0 for stability, clamped internally)
//   omega: Frequency coefficient
//   dt: Timestep (> 0)
//   eps: Numerical stability constant
//
// Returns:
//   CayleyComplex1D with re and im of conj(A_bar)
__device__ __forceinline__ CayleyComplex1D cayley_discretize_complex(
    float alpha,
    float omega,
    float dt,
    float eps = EPS_DEFAULT
) {
    // Clamp alpha >= 0 for A-stability guarantee
    alpha = fmaxf(alpha, 0.0f);

    float tau = dt * 0.5f;
    float ta = tau * alpha;
    float tw = tau * omega;

    // Numerator N = 1 + tau*lambda = (1-ta) + i*tw
    float nr = 1.0f - ta;
    float ni = tw;

    // Denominator D = 1 - tau*lambda = (1+ta) - i*tw
    float dr = 1.0f + ta;
    float di = -tw;

    // |D|^2 = (1+ta)^2 + tw^2, always >= 1 for alpha >= 0
    float denom = __fmaf_rn(dr, dr, di * di) + eps;
    float inv_d = __fdiv_rn(1.0f, denom);

    // A_bar = N/D via complex division: N * conj(D) / |D|^2
    float mu_re = (nr * dr + ni * di) * inv_d;  // (1-ta^2-tw^2) / |D|^2
    float mu_im = (ni * dr - nr * di) * inv_d;  // 2*tw / |D|^2

    // Store conj(mu) for correct rotation direction
    CayleyComplex1D result;
    result.re = mu_re;
    result.im = -mu_im;  // negate imaginary -> conj(A_bar)
    return result;
}

// Compute |A_bar|^2 from alpha, omega, dt (for VP scale computation)
// Note: |conj(A_bar)|^2 = |A_bar|^2, so conjugation doesn't matter here
__device__ __forceinline__ float cayley_eig_sq(
    float alpha, float omega, float dt, float eps = EPS_DEFAULT
) {
    alpha = fmaxf(alpha, 0.0f);
    float tau = dt * 0.5f;
    float ta = tau * alpha;
    float tw = tau * omega;
    float numer = __fmaf_rn(1.0f - ta, 1.0f - ta, tw * tw);  // (1-ta)^2 + tw^2
    float denom = __fmaf_rn(1.0f + ta, 1.0f + ta, tw * tw);  // (1+ta)^2 + tw^2
    return numer / (denom + eps);
}

}  // namespace cdssm
