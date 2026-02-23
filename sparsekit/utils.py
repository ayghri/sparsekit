from typing import (
    Optional,
    Tuple,
    Mapping,
    Union,
)

from torch import Tensor
from torch.nn import Parameter

Values = Union[Tensor, Mapping[Parameter, Tensor], None]


class SpastraError(Exception):
    pass


class ShapeMismatchError(SpastraError):
    """Raised when tensor shapes do not match expected dimensions."""
    def __init__(
        self, expected: Tuple[int, ...], got: Tuple[int, ...], context: str = ""
    ):
        msg = f"Shape mismatch: expected {expected}, got {got}"
        super().__init__(f"{context}: {msg}" if context else msg)


class DivisibilityError(SpastraError):
    """Raised when tensor dimension is not divisible by block/group size."""

    def __init__(self, dim: int, size: int, divisor: int, context: str = ""):
        msg = f"Dimension {dim}: size {size} not divisible by {divisor}"
        super().__init__(f"{context}: {msg}" if context else msg)


class CouplingError(SpastraError):
    """Raised when coupling constraints are violated."""

    pass


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
        Tensor of shape (s0, s2, s4, ..., s1,s3,s5, ...).
    """
    ndim = t.ndim
    permutation = list(range(0, ndim, 2)) + list(range(1, ndim, 2))
    return t.permute(permutation)


def merge_odd_dims(t: Tensor) -> Tensor:
    t = append_odd_dims(t)
    even_shape = [t.shape[i] for i in range(0, t.ndim // 2)]
    return t.reshape(*even_shape, -1)


def unmerge_odd_dims(t: Tensor, odd_dims: Tuple[int, ...]) -> Tensor:
    """Inverse of merge_even_dims.

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


def normalize_order(
    order: Optional[Tuple[int, ...]], dim: int
) -> Tuple[int, ...]:
    """
    Validate that *order* is a permutation of ``range(dim)``.
    If *order* is None or empty return the identity permutation.

    Parameters
    ----------
    order : tuple[int] | None
        Desired axis ordering.
    dim   : int
        Number of dimensions in the target tensor.

    Returns
    -------
    Tuple[int, ...]
        Normalised permutation.
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
        The inverse permutation such that perm[inv[i]] == i.
    """
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)
