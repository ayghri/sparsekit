# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0
# (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Magnitude and OBD (Optimal Brain Damage) pruners."""

import torch
from torch import Tensor


@torch.no_grad()
def magnitude(
    W0: Tensor, scope_size: int, num_keep: int
) -> Tensor:
    """Magnitude pruning: keep ``num_keep`` per scope by |w|.

    Args:
        W0: (M, K) weight matrix.
        scope_size: columns per scope.
        num_keep: columns to keep per scope.

    Returns:
        (M, K) pruned weight tensor.
    """
    M, K = W0.shape
    W = W0.clone()
    num_prune = scope_size - num_keep
    Wv = W.view(M, K // scope_size, scope_size)
    _, idx = Wv.abs().topk(num_prune, dim=-1, largest=False)
    mask = torch.ones_like(Wv)
    mask.scatter_(2, idx, 0.0)
    W *= mask.view(M, K)
    return W


@torch.no_grad()
def obd(
    W0: Tensor,
    H: Tensor,
    scope_size: int,
    num_keep: int,
) -> Tensor:
    """OBD pruning: keep ``num_keep`` per scope by w² * diag(H).

    Args:
        W0: (M, K) weight matrix.
        H: (K, K) Hessian X^T X / N.
        scope_size: columns per scope.
        num_keep: columns to keep per scope.

    Returns:
        (M, K) pruned weight tensor.
    """
    M, K = W0.shape
    W = W0.clone()
    d = torch.diag(H)
    scores = W**2 * d[None, :]
    scores = scores.reshape(M, K // scope_size, scope_size)
    _, keep_idx = scores.topk(num_keep, dim=-1)
    mask = torch.zeros_like(scores)
    mask.scatter_(-1, keep_idx, 1.0)
    W *= mask.reshape(M, K)
    return W
