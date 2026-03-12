# Copyright (c) 2025 Anonymous Authors
# Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Linear algebra utilities for sparse optimization."""

import torch


def lsqr_gkl(
    A: torch.Tensor,
    b: torch.Tensor,
    max_iter: int = 1000,
    tol: float = 1e-6,
    x_0: torch.Tensor | None = None,
    device: torch.device | None = None,
):
    """
    Solves the linear least-squares problem min ||Ax - b||_2 using the LSQR algorithm,
    leveraging PyTorch for GPU and Golub-Kahan-Lanczos bidiagonalization.

    Args:
        A: The matrix A (torch.Tensor).
        b: The right-hand side vector b (torch.Tensor).
        max_iter: Maximum number of iterations (int).
        tol: Tolerance for convergence (float).
        x_0: Initial guess for the solution (torch.Tensor, optional).
        device: Device to perform computations on (torch.device, optional).
    Returns:
        x: The solution vector (torch.Tensor).
        info: A dictionary containing information about the convergence.
    """
    assert max_iter > 2, (
        "max_iter must be greater than 2 for LSQR to work properly."
    )
    if device is None:
        device = A.device

    n = A.shape[1]
    A = A.to(device)
    b = b.to(device)

    # initial guess
    if x_0 is None:
        x = torch.zeros(n, dtype=b.dtype, device=device)
    else:
        x = x_0.to(device)
    # initial residual
    norm_b = torch.linalg.norm(b)
    r = b - A @ x
    beta = torch.linalg.norm(r)

    if beta == 0:
        return x, {
            "converged": True,
            "iterations": 0,
            "residual": 0.0,
            "status": "rhs is zero",
        }

    u = r / beta

    # Initial Lanczos vector v1
    v = A.t() @ u
    alpha = torch.linalg.norm(v)

    if alpha == 0:
        return x, {
            "converged": True,
            "iterations": 0,
            "rel_res": 0.0,
            "status": "A.T @ u is zero",
        }

    v = v / alpha

    # working variables
    phi_bar = beta
    rho_bar = alpha
    w = v

    # history
    rel_res = (alpha * phi_bar).abs().item()
    new_rel_res = 0.0
    new_tol = 0.0

    # LSQR iteration
    for k in range(max_iter):
        # bidiagonalization step (Golub-Kahan)
        u_prev = u
        u = A @ v - alpha * u_prev
        beta = torch.linalg.norm(u)
        if beta == 0:
            break
        u = u / beta

        v_prev = v
        v = A.t() @ u - beta * v_prev
        alpha = torch.linalg.norm(v)
        if alpha == 0:
            break
        v = v / alpha

        # apply Givens rotations
        rho = torch.sqrt(rho_bar**2 + beta**2)
        c = rho_bar / rho
        s = beta / rho

        theta = s * alpha
        rho_bar = c * alpha
        phi = c * phi_bar
        phi_bar = -s * phi_bar

        # update solution and auxiliary w
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w

        # Check for convergence

        new_rel_res = (torch.abs(phi_bar) / norm_b).item()
        new_tol = abs(new_rel_res - rel_res)

        if new_tol < tol:
            return x, {
                "converged": True,
                "iterations": k + 1,
                "tol": new_tol,
                "rel_res": new_rel_res,
                "status": "tolerance met",
            }

        rel_res = new_rel_res

    return x, {
        "converged": False,
        "iterations": max_iter,
        "tol": new_tol,
        "rel_res": rel_res,
        "status": "max iters reached",
    }


def hard_threshold(vec: torch.Tensor, alpha: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest elements of vec, selected by magnitude of alpha.

    Args:
        vec: Values tensor.
        alpha: Scores tensor (same shape); top-k selected by these magnitudes.
        k: Number of elements to keep.

    Returns:
        Tensor with only the k largest-alpha entries of vec; rest zeroed.
    """
    if k >= vec.numel():
        return vec
    _, indices = torch.topk(alpha, k)
    result = torch.zeros_like(vec)
    result[indices] = vec[indices]
    return result


def soft_threshold(vec: torch.Tensor, threshold: torch.Tensor | float) -> torch.Tensor:
    """Element-wise soft-thresholding: ``sign(x) * max(abs(x) - threshold, 0)``.

    Args:
        vec: Input tensor.
        threshold: Threshold value (scalar or broadcastable tensor).

    Returns:
        Soft-thresholded tensor.
    """
    return torch.sign(vec) * torch.nn.functional.relu(
        torch.abs(vec) - threshold
    )


@torch.no_grad()
def solve_proximal_adam(
    v_elements: torch.Tensor,
    H_elements: torch.Tensor,
    thresholds: torch.Tensor,
    eps: float = 1e-6,
    max_iter: int = 10,
) -> torch.Tensor:
    """
    Solves for the scalar mu in the Proximal Adam equation using Bisection search.

    Args:
        v_elements: Shape (s1,s2,...,sm). The dense weights (or updates).
        H_elements: Shape (s1,s2,...,sm). The Adam preconditioner (sqrt(v) + eps).
        thresholds: Shape (num_blocks). The target value (eta * lambda).

    Returns:
        mu: Shape (Num_Groups, 1). The scaling factor.
    """
    # 1. Compute Norms ||H * v||_2
    Hv = H_elements * v_elements
    # Hv_norms = torch.linalg.norm(Hv, dim=1, keepdim=True)  # (G, 1)
    Hv_norms = torch.linalg.norm(Hv)

    # 2. Identify Survivors
    # If ||Hv|| <= threshold, the optimal weight is 0. We only solve for the rest.
    # We add a small epsilon to threshold to avoid division by zero in bounds calculation
    is_survivor = Hv_norms > thresholds

    # Prepare output
    mu_solutions = torch.zeros_like(Hv_norms)

    # Filter to active groups to save compute
    indices = torch.nonzero(is_survivor.squeeze()).squeeze()
    if indices.numel() == 0:
        return mu_solutions  # All zero

    v_active = v_elements[indices]
    M_active = H_elements[indices]
    thresh_active = thresholds[indices]
    norm_active = Hv_norms[indices]

    # 3. Compute Bounds (from the derivation)
    # S = ||Hv||^2, so sqrt(S) = norm_active
    # mu_low = (lambda * h_min) / (sqrt(S) - lambda)
    denom = norm_active - thresh_active + eps

    h_min = M_active.min(dim=1, keepdim=True).values
    h_max = M_active.max(dim=1, keepdim=True).values

    mu_low = (thresh_active * h_min) / denom
    mu_high = (thresh_active * h_max) / denom

    # We search for mu such that Zeta(mu) = threshold
    # Zeta(mu) = mu * || (H / (H + mu)) * v ||

    low = mu_low
    high = mu_high

    for _ in range(max_iter):
        mu = (low + high) / 2

        # Compute Zeta(mu)
        # scaling vector = M / (M + mu)
        scaling = M_active / (M_active + mu)

        # weighted_v = scaling * v
        # zeta = mu * ||weighted_v||
        weighted_norm = torch.linalg.norm(
            scaling * v_active, dim=1, keepdim=True
        )
        zeta = mu * weighted_norm

        # Update bounds
        # Zeta is strictly increasing with mu.
        # If zeta < threshold, mu is too small -> low = mu
        # If zeta > threshold, mu is too big -> high = mu
        mask_low = zeta < thresh_active
        low = torch.where(mask_low, mu, low)
        high = torch.where(~mask_low, mu, high)

    # Final estimate
    mu_solutions[indices] = (low + high) / 2
    return mu_solutions
