# Copyright (c) 2025 Anonymous Authors
# Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Utility functions: tensor reshaping, permutations, top-k dispatch, and error types."""

from typing import (
    Optional,
    Tuple,
    Mapping,
    Union,
)

import torch
from torch import Tensor
from torch.nn import Parameter

# ── Type aliases ─────────────────────────────────────────────────────────

Values = Union[Tensor, Mapping[Parameter, Tensor], None]

# ── Top-k dispatch (triton / torch fallback) ─────────────────────────────

try:
    from .kernels import (
        kth_largest as _kth_largest_impl,
        mid_kth_largest as _mid_kth_largest_impl,
    )
except ImportError:

    def _kth_largest_impl(
        x: Tensor, k: int, dim: int = -1, chunk_k: int = 1024,
    ) -> Tensor:
        dim = dim % x.ndim
        K = x.shape[dim]
        return torch.kthvalue(x, K - k + 1, dim=dim).values

    def _mid_kth_largest_impl(
        x: Tensor, k: int, dim: int = -1, chunk_k: int = 1024,
    ) -> Tensor:
        dim = dim % x.ndim
        K = x.shape[dim]
        v1 = torch.kthvalue(x, K - k + 1, dim=dim).values
        v2 = torch.kthvalue(x, K - k, dim=dim).values
        return (v1 + v2) / 2.0


def kth_largest(
    x: Tensor, k: int, dim: int | None = None, **kwargs,
) -> Tensor:
    """Return the k-th largest value along a dimension (or globally).

    Auto-dispatches to the optimal backend (triton / torch) based on (K, k).
    When ``dim is None`` the tensor is flattened first.
    """
    if dim is None:
        return _kth_largest_impl(x.view(-1), k, dim=0, **kwargs).squeeze()
    return _kth_largest_impl(x, k, dim=dim, **kwargs)


def mid_kth_largest(
    x: Tensor, k: int, dim: int | None = None, **kwargs,
) -> Tensor:
    """Midpoint of the k-th and (k+1)-th largest values along a dimension.

    Auto-dispatches to the optimal backend (triton / torch) based on (K, k).
    When ``dim is None`` the tensor is flattened first.
    """
    if dim is None:
        return _mid_kth_largest_impl(x.reshape(-1), k, dim=0, **kwargs).squeeze()
    return _mid_kth_largest_impl(x, k, dim=dim, **kwargs)


# ── Tensor reshaping utilities ───────────────────────────────────────────

def interleave_unsqueeze(t: Tensor) -> Tensor:
    """Insert a singleton dimension after each existing dimension.

    Args:
        t: Input tensor of shape (s1, s2, s3, ...).

    Returns:
        Tensor of shape (s1, 1, s2, 1, s3, 1, ...).
    """
    for i in range(1, 2 * t.ndim, 2):
        t = t.unsqueeze(i)
    return t


def append_odd_dims(t: Tensor) -> Tensor:
    """Put odd-indexed dimensions into trailing dims.

    Args:
        t: Input tensor of shape (s0, s1, s2, s3, ...).

    Returns:
        Tensor of shape (s0, s2, s4, ..., s1, s3, s5, ...).
    """
    ndim = t.ndim
    permutation = list(range(0, ndim, 2)) + list(range(1, ndim, 2))
    return t.permute(permutation)


def merge_odd_dims(t: Tensor) -> Tensor:
    """Move odd-indexed dims to trailing positions and flatten them.

    Args:
        t: Input tensor of shape (s0, s1, s2, s3, ...).

    Returns:
        Tensor of shape (s0, s2, ..., s1*s3*...).
    """
    t = append_odd_dims(t)
    even_shape = [t.shape[i] for i in range(0, t.ndim // 2)]
    return t.reshape(*even_shape, -1)


def unmerge_odd_dims(t: Tensor, odd_dims: Tuple[int, ...]) -> Tensor:
    """Inverse of ``merge_odd_dims``.

    Args:
        t: Tensor of shape (s0, s2, s4, ..., s1*s3*s5*...).
        odd_dims: The original odd-indexed dimensions (s1, s3, s5, ...).

    Returns:
        Tensor restored to original interleaved shape.
    """
    even_dims = t.shape[:-1]
    assert len(even_dims) == len(odd_dims)
    ndim = 2 * len(even_dims)

    # Interleave even and odd dims to get original shape
    original_shape = []
    for i in range(len(even_dims)):
        original_shape.append(even_dims[i])
        original_shape.append(odd_dims[i])

    # Reconstruct intermediate shape and inverse permutation
    permutation = list(range(0, ndim, 2)) + list(range(1, ndim, 2))
    permuted_shape = [original_shape[i] for i in permutation]

    t = t.reshape(permuted_shape)

    inverse_perm = [0] * ndim
    for i, p in enumerate(permutation):
        inverse_perm[p] = i

    return t.permute(inverse_perm)


# ── Permutation utilities ────────────────────────────────────────────────

def normalize_order(
    order: Optional[Tuple[int, ...]], dim: int
) -> Tuple[int, ...]:
    """Validate that ``order`` is a permutation of ``range(dim)``.

    If ``order`` is None or empty, returns the identity permutation.

    Args:
        order: Desired axis ordering.
        dim: Number of dimensions in the target tensor.

    Returns:
        Normalised permutation tuple.
    """
    if not order:
        return tuple(range(dim))
    o = list(order)
    if set(o) != set(range(dim)):
        raise ValueError(
            f"order must be a permutation of 0..{dim - 1}, got {order}"
        )
    return tuple(o)


def inverse_permutation(perm: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute the inverse of a permutation.

    Args:
        perm: A permutation tuple.

    Returns:
        The inverse permutation such that ``perm[inv[i]] == i``.
    """
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


# ── Error types ──────────────────────────────────────────────────────────

class SparseKitError(Exception):
    """Base exception for sparsekit errors."""
    pass


class ShapeMismatchError(SparseKitError):
    """Raised when tensor shapes do not match expected dimensions."""
    def __init__(
        self, expected: Tuple[int, ...], got: Tuple[int, ...], context: str = ""
    ):
        msg = f"Shape mismatch: expected {expected}, got {got}"
        super().__init__(f"{context}: {msg}" if context else msg)


class CouplingError(SparseKitError):
    """Raised when coupling constraints are violated."""
    pass
