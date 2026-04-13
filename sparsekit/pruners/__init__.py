# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0
# (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Pruning algorithms.

- :mod:`obs` — Structured OBS with per-row Schur updates.
- :mod:`obd` — OBD and magnitude pruning.
- :mod:`sparsegpt` — SparseGPT column-sequential pruning.
"""

import torch
from torch import Tensor


def compute_hessian(
    X: Tensor,
    device: torch.device | None = None,
    batch_size: int = 4096,
) -> Tensor:
    """Compute Hessian H = X^T X / N.

    Args:
        X: (N, K) input activations (can be on CPU).
        device: target device for H. Defaults to X's device.
        batch_size: rows per batch for memory-friendly accumulation.

    Returns:
        (K, K) Hessian on ``device``.
    """
    N, K = X.shape
    if device is None:
        device = X.device
    H = torch.zeros(K, K, device=device, dtype=torch.float32)
    for i in range(0, N, batch_size):
        X_b = X[i : i + batch_size].to(
            device=device, dtype=torch.float32
        )
        H.addmm_(X_b.T, X_b)
    H /= N
    return H


def output_error(
    W_pruned: Tensor,
    W_orig: Tensor,
    H: Tensor,
    N: int | None = None,
) -> float:
    """Output Frobenius error from pruning.

    When ``N`` is given, returns ``||X dW^T||_F = sqrt(N * tr(dW H dW^T))``.
    Otherwise returns the relative error ``sqrt(tr(dW H dW^T) / tr(W H W^T))``.

    Args:
        W_pruned: (M, K) pruned weights.
        W_orig: (M, K) original weights.
        H: (K, K) Hessian.
        N: number of samples (for absolute error).

    Returns:
        Scalar error value.
    """
    M = W_orig.shape[0]
    chunk = 128
    total = 0.0
    for c0 in range(0, M, chunk):
        dW = (
            W_pruned[c0 : c0 + chunk] - W_orig[c0 : c0 + chunk]
        ).float()
        total += ((dW @ H) * dW).sum().item()
    if N is not None:
        return (total * N) ** 0.5
    ref = 0.0
    for c0 in range(0, M, chunk):
        W = W_orig[c0 : c0 + chunk].float()
        ref += ((W @ H) * W).sum().item()
    if ref <= 0:
        return float("inf")
    return (total / ref) ** 0.5
