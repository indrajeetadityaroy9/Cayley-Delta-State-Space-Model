"""Pure PyTorch reference implementations for KSSM operations.

These implementations are used for:
1. Testing correctness of Triton kernels
2. Phase 2 hybrid backward (PyTorch backward, Triton forward)

All functions use float64 for numerical accuracy in tests.
"""

import torch
from torch import Tensor


def cayley_transform_reference(
    alpha: Tensor,
    omega: Tensor,
    dt: Tensor,
    B: Tensor,
    x: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Compute Cayley (bilinear) transform discretization.

    Discretizes the continuous-time dynamics:
        dz/dt = A(t) z + B x
    where A = [[-alpha, omega], [-omega, -alpha]]

    Using the Cayley transform:
        A_bar = (I - tau*A)^{-1} (I + tau*A)
        u_bar = dt * (I - tau*A)^{-1} B x

    Args:
        alpha: Damping coefficients, shape (batch, seq, d_inner). Must be >= 0.
        omega: Frequency coefficients, shape (batch, seq, d_inner).
        dt: Timestep, shape (batch, seq, d_inner) or broadcastable.
        B: Input projection, shape (d_inner, 2) or (batch, seq, d_inner, 2).
        x: Input, shape (batch, seq, d_inner).
        eps: Epsilon for numerical stability in determinant.

    Returns:
        A_bar: Discretized transition matrix, shape (batch, seq, d_inner, 4).
               Stored as [a11, a12, a21, a22] for each 2x2 block.
        u_bar: Discretized input, shape (batch, seq, d_inner, 2).
    """
    # Ensure float64 for reference
    alpha = alpha.double()
    omega = omega.double()
    dt = dt.double()
    x = x.double()
    if B.dim() == 2:
        B = B.double()
    else:
        B = B.double()

    tau = dt / 2.0

    # Determinant of M = I - tau*A
    # M = [[1 + tau*alpha, -tau*omega], [tau*omega, 1 + tau*alpha]]
    # det(M) = (1 + tau*alpha)^2 + (tau*omega)^2
    det_M = (1.0 + tau * alpha) ** 2 + (tau * omega) ** 2
    inv_det = 1.0 / (det_M + eps)

    # M^{-1} components
    # M^{-1} = (1/det) * [[1 + tau*alpha, tau*omega], [-tau*omega, 1 + tau*alpha]]
    m11 = (1.0 + tau * alpha) * inv_det
    m12 = (tau * omega) * inv_det
    m21 = -(tau * omega) * inv_det
    m22 = (1.0 + tau * alpha) * inv_det

    # N = I + tau*A
    # N = [[1 - tau*alpha, tau*omega], [-tau*omega, 1 - tau*alpha]]
    n11 = 1.0 - tau * alpha
    n12 = tau * omega
    n21 = -tau * omega
    n22 = 1.0 - tau * alpha

    # A_bar = M^{-1} @ N (2x2 @ 2x2)
    a11 = m11 * n11 + m12 * n21
    a12 = m11 * n12 + m12 * n22
    a21 = m21 * n11 + m22 * n21
    a22 = m21 * n12 + m22 * n22

    # Stack into (batch, seq, d_inner, 4)
    A_bar = torch.stack([a11, a12, a21, a22], dim=-1)

    # u_bar = dt * M^{-1} @ B @ x
    # B is (d_inner, 2) or (batch, seq, d_inner, 2)
    if B.dim() == 2:
        # B is (d_inner, 2), x is (batch, seq, d_inner)
        # B @ x for each position: B[d, :] * x[b, t, d] -> (batch, seq, d_inner, 2)
        Bx = x.unsqueeze(-1) * B.unsqueeze(0).unsqueeze(0)  # (batch, seq, d_inner, 2)
    else:
        # B is (batch, seq, d_inner, 2)
        Bx = x.unsqueeze(-1) * B

    # M^{-1} @ Bx: 2x2 @ 2-vector
    u1 = m11 * Bx[..., 0] + m12 * Bx[..., 1]
    u2 = m21 * Bx[..., 0] + m22 * Bx[..., 1]

    # Scale by dt
    u_bar = dt.unsqueeze(-1) * torch.stack([u1, u2], dim=-1)

    return A_bar, u_bar


def evolution_reference(
    A_bar: Tensor,
    u_bar: Tensor,
    initial_state: Tensor | None = None,
) -> Tensor:
    """Sequential state evolution (scan) reference implementation.

    Computes h_t = A_bar[t] @ h_{t-1} + u_bar[t] for t = 0, ..., T-1.

    Args:
        A_bar: Discretized transition matrices, shape (batch, seq, d_inner, 4).
        u_bar: Discretized inputs, shape (batch, seq, d_inner, 2).
        initial_state: Initial state, shape (batch, d_inner, 2). Defaults to zeros.

    Returns:
        states: All states h_0, ..., h_{T-1}, shape (batch, seq, d_inner, 2).
    """
    batch, seq_len, d_inner, _ = A_bar.shape

    # Ensure float64
    A_bar = A_bar.double()
    u_bar = u_bar.double()

    if initial_state is None:
        h = torch.zeros(batch, d_inner, 2, dtype=torch.float64, device=A_bar.device)
    else:
        h = initial_state.double()

    states = []

    for t in range(seq_len):
        # Extract A_bar[t] components
        a11 = A_bar[:, t, :, 0]  # (batch, d_inner)
        a12 = A_bar[:, t, :, 1]
        a21 = A_bar[:, t, :, 2]
        a22 = A_bar[:, t, :, 3]

        # Extract u_bar[t]
        u1 = u_bar[:, t, :, 0]  # (batch, d_inner)
        u2 = u_bar[:, t, :, 1]

        # h_t = A_bar[t] @ h_{t-1} + u_bar[t]
        h1 = h[:, :, 0]
        h2 = h[:, :, 1]

        new_h1 = a11 * h1 + a12 * h2 + u1
        new_h2 = a21 * h1 + a22 * h2 + u2

        h = torch.stack([new_h1, new_h2], dim=-1)
        states.append(h)

    return torch.stack(states, dim=1)  # (batch, seq, d_inner, 2)


@torch.compile(mode="reduce-overhead")
def evolution_backward_reference(
    A_bar: Tensor,
    states: Tensor,
    grad_output: Tensor,
) -> tuple[Tensor, Tensor]:
    """Backward pass for evolution using adjoint state method.

    Computes gradients d_A_bar and d_u_bar given grad_output (dL/d_states).

    The adjoint equation is:
        lambda_{t-1} = A_t^T @ lambda_t + grad_output_{t-1}

    where lambda_T = 0 (or grad_output_T if output at T is used).

    Args:
        A_bar: Discretized transition matrices, shape (batch, seq, d_inner, 4).
        states: Forward pass states, shape (batch, seq, d_inner, 2).
        grad_output: Gradient w.r.t. states, shape (batch, seq, d_inner, 2).

    Returns:
        d_A_bar: Gradient w.r.t. A_bar, shape (batch, seq, d_inner, 4).
        d_u_bar: Gradient w.r.t. u_bar, shape (batch, seq, d_inner, 2).
    """
    batch, seq_len, d_inner, _ = A_bar.shape
    device = A_bar.device

    # Ensure float64
    A_bar = A_bar.double()
    states = states.double()
    grad_output = grad_output.double()

    # d_u_bar[t] = lambda_t = dL/dh_t (after adjoint accumulation)
    # But in simple case where loss depends on all states equally,
    # d_u_bar[t] = grad_output[t]
    d_u_bar = grad_output.clone()

    # d_A_bar[t] = lambda_t @ h_{t-1}^T
    # We need h_{t-1} for each t. h_{-1} = 0 (or initial_state)
    d_A_bar = torch.zeros_like(A_bar)

    # Prepend zero state for h_{-1}
    h_prev = torch.zeros(batch, 1, d_inner, 2, dtype=torch.float64, device=device)
    h_all = torch.cat([h_prev, states[:, :-1, :, :]], dim=1)  # (batch, seq, d_inner, 2)

    # For each timestep, d_A_bar[t] = outer(lambda_t, h_{t-1})
    # lambda_t here is just grad_output[t] in this simple formulation
    # d_A_bar[t, :, k] = lambda_t[:, k] * h_{t-1}[:, l] for the (k,l) element

    # A_bar is stored as [a11, a12, a21, a22]
    # dL/da11 = lambda_1 * h1_{t-1}
    # dL/da12 = lambda_1 * h2_{t-1}
    # dL/da21 = lambda_2 * h1_{t-1}
    # dL/da22 = lambda_2 * h2_{t-1}

    lambda_1 = grad_output[..., 0]  # (batch, seq, d_inner)
    lambda_2 = grad_output[..., 1]
    h1_prev = h_all[..., 0]  # (batch, seq, d_inner)
    h2_prev = h_all[..., 1]

    d_A_bar[..., 0] = lambda_1 * h1_prev  # da11
    d_A_bar[..., 1] = lambda_1 * h2_prev  # da12
    d_A_bar[..., 2] = lambda_2 * h1_prev  # da21
    d_A_bar[..., 3] = lambda_2 * h2_prev  # da22

    return d_A_bar, d_u_bar


def evolution_backward_full_reference(
    A_bar: Tensor,
    states: Tensor,
    grad_output: Tensor,
) -> tuple[Tensor, Tensor]:
    """Full backward pass with reverse adjoint scan.

    This is the mathematically correct backward that propagates gradients
    through the recurrence properly.

    Args:
        A_bar: Discretized transition matrices, shape (batch, seq, d_inner, 4).
        states: Forward pass states, shape (batch, seq, d_inner, 2).
        grad_output: Gradient w.r.t. states, shape (batch, seq, d_inner, 2).

    Returns:
        d_A_bar: Gradient w.r.t. A_bar, shape (batch, seq, d_inner, 4).
        d_u_bar: Gradient w.r.t. u_bar, shape (batch, seq, d_inner, 2).
    """
    batch, seq_len, d_inner, _ = A_bar.shape
    device = A_bar.device

    A_bar = A_bar.double()
    states = states.double()
    grad_output = grad_output.double()

    # Initialize adjoint state
    lambda_state = torch.zeros(batch, d_inner, 2, dtype=torch.float64, device=device)

    d_A_bar = torch.zeros_like(A_bar)
    d_u_bar = torch.zeros(batch, seq_len, d_inner, 2, dtype=torch.float64, device=device)

    # Prepend zero for h_{-1}
    h_prev_all = torch.cat([
        torch.zeros(batch, 1, d_inner, 2, dtype=torch.float64, device=device),
        states[:, :-1, :, :]
    ], dim=1)

    # Reverse scan
    for t in range(seq_len - 1, -1, -1):
        # Add direct gradient contribution
        lambda_state = lambda_state + grad_output[:, t, :, :]

        # d_u_bar[t] = lambda_t
        d_u_bar[:, t, :, :] = lambda_state

        # d_A_bar[t] = outer(lambda_t, h_{t-1})
        h_prev = h_prev_all[:, t, :, :]  # (batch, d_inner, 2)
        lambda_1 = lambda_state[:, :, 0]
        lambda_2 = lambda_state[:, :, 1]
        h1_prev = h_prev[:, :, 0]
        h2_prev = h_prev[:, :, 1]

        d_A_bar[:, t, :, 0] = lambda_1 * h1_prev
        d_A_bar[:, t, :, 1] = lambda_1 * h2_prev
        d_A_bar[:, t, :, 2] = lambda_2 * h1_prev
        d_A_bar[:, t, :, 3] = lambda_2 * h2_prev

        # Propagate: lambda_{t-1} = A_t^T @ lambda_t
        # A^T: swap a12 and a21
        a11 = A_bar[:, t, :, 0]
        a12 = A_bar[:, t, :, 1]
        a21 = A_bar[:, t, :, 2]
        a22 = A_bar[:, t, :, 3]

        new_lambda_1 = a11 * lambda_1 + a21 * lambda_2  # Note: a21 not a12 (transpose)
        new_lambda_2 = a12 * lambda_1 + a22 * lambda_2  # Note: a12 not a21 (transpose)

        lambda_state = torch.stack([new_lambda_1, new_lambda_2], dim=-1)

    return d_A_bar, d_u_bar
