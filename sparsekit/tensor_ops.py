# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0
# (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Tensor reshaping, permutation utilities, and top-k dispatch."""

from typing import Optional, Tuple

import torch

from .kernels import kth_largest as _kth_largest_impl
from .kernels import mid_kth_largest as _mid_kth_largest_impl

Tensor = torch.Tensor


def interleave_unsqueeze(t: Tensor) -> Tensor:
    """Insert a singleton dimension after each existing one.

    Args:
        t: Input tensor of shape (s1, s2, s3, ...).

    Returns:
        Tensor of shape (s1, 1, s2, 1, s3, 1, ...).
    """
    for i in range(1, 2 * t.ndim, 2):
        t = t.unsqueeze(i)
    return t


def append_odd_dims(t: Tensor) -> Tensor:
    """Move odd-indexed dimensions to trailing positions.

    Args:
        t: Input tensor of shape (s0, s1, s2, s3, ...).

    Returns:
        Tensor of shape (s0, s2, s4, ..., s1, s3, s5, ...).
    """
    ndim = t.ndim
    perm = list(range(0, ndim, 2)) + list(range(1, ndim, 2))
    return t.permute(perm)


def merge_odd_dims(t: Tensor) -> Tensor:
    """Move odd dims to trailing positions and flatten them.

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
        t: Tensor of shape (s0, s2, ..., s1*s3*...).
        odd_dims: Original odd-indexed dimensions.

    Returns:
        Tensor restored to original interleaved shape.
    """
    even_dims = t.shape[:-1]
    assert len(even_dims) == len(odd_dims)
    ndim = 2 * len(even_dims)

    original_shape = []
    for i in range(len(even_dims)):
        original_shape.append(even_dims[i])
        original_shape.append(odd_dims[i])

    perm = list(range(0, ndim, 2)) + list(range(1, ndim, 2))
    permuted_shape = [original_shape[i] for i in perm]
    t = t.reshape(permuted_shape)

    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return t.permute(inv_perm)


def get_dtype_epsilon(dtype, epsilon):
    if epsilon is None:
        epsilon = torch.finfo(dtype).eps
    assert epsilon >= 0
    return epsilon



def normalize_order(order: Optional[Tuple[int, ...]], dim: int) -> Tuple[int, ...]:
    """Validate that ``order`` is a permutation of range(dim).

    Returns the identity permutation if order is None or empty.

    Args:
        order: Desired axis ordering.
        dim: Number of dimensions.

    Returns:
        Normalised permutation tuple.
    """
    if not order:
        return tuple(range(dim))
    o = list(order)
    if set(o) != set(range(dim)):
        raise ValueError(
            f"order must be a permutation of " f"0..{dim - 1}, got {order}"
        )
    return tuple(o)


def inverse_permutation(perm: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute the inverse of a permutation.

    Args:
        perm: A permutation tuple.

    Returns:
        Inverse such that ``perm[inv[i]] == i``.
    """
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)



def kth_largest(
    x: Tensor,
    k: int,
    dim: int | None = None,
    **kwargs,
) -> Tensor:
    """Return the k-th largest value along a dimension.

    Auto-dispatches to the optimal backend (triton / torch)
    based on (K, k). Flattens first when ``dim is None``.
    """
    if dim is None:
        return _kth_largest_impl(x.view(-1), k, dim=0, **kwargs).squeeze()
    return _kth_largest_impl(x, k, dim=dim, **kwargs)


def mid_kth_largest(
    x: Tensor,
    k: int,
    dim: int | None = None,
    k_weight: float = 1.0,
    **kwargs,
) -> Tensor:
    """Midpoint of k-th and (k+1)-th largest values.

    Auto-dispatches to the optimal backend (triton / torch)
    based on (K, k). Flattens first when ``dim is None``.
    """
    if dim is None:
        return _mid_kth_largest_impl(
            x.reshape(-1), k, dim=0, k_weight=k_weight, **kwargs
        ).squeeze()
    return _mid_kth_largest_impl(x, k, dim=dim, k_weight=k_weight, **kwargs)
