
"""
Copyright (c) 2025 Ayoub Ghriss and contributors
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.

SparseNode : abstract base class that knows how to view / threshold
BlockSpec   : concrete implementation that treats the whole tensor as a single
              grid of blocks (the most common case).
BlockCoupling  : concrete implementation that merges multiple BlockSpec into one
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Set, Mapping, Iterable

import math

from torch import Tensor
from torch.nn import Parameter
import torch
from abc import abstractmethod, ABC


from .linalg import kth_largest
# from .blocks import SparseNode
# from .blocks import BlockSpec

from .utils import merge_odd_dims
from .utils import normalize_order
from .utils import unmerge_odd_dims
from .utils import inverse_permutation
from .utils import CouplingError
from .utils import Values


from dataclasses import dataclass, field
from typing import (
    Optional,
    Tuple,
    List,
    Mapping,
    Union,
    Iterable,
    Callable,
)
from abc import ABC, abstractmethod
from functools import cached_property

import math
from torch import Tensor
from torch.nn import Parameter
import torch

from .utils import interleave_unsqueeze
from .utils import merge_odd_dims, inverse_permutation, normalize_order
from .utils import ShapeMismatchError
from .utils import Values


@dataclass(frozen=True)
class GridAxesPlan:
    """Tracks the relationship between a full grid shape and its compact form.

    Many parts of the library (threshold tensors) prefer a compact grid shape
    where singleton axes are removed. For coupling (axis permutations) and
    second-order methods (OBS/OBD) we need a stable axis identity, which
    requires keeping singleton axes in the *full* grid.

    This helper provides zero-allocation reshape/squeeze/unsqueeze utilities
    that preserve axis identity.
    """

    full_shape: Tuple[int, ...]

    @cached_property
    def keep_axes(self) -> Tuple[int, ...]:
        return tuple(i for i, s in enumerate(self.full_shape) if s > 1)

    @cached_property
    def drop_axes(self) -> Tuple[int, ...]:
        return tuple(i for i, s in enumerate(self.full_shape) if s == 1)

    @cached_property
    def compact_shape(self) -> Tuple[int, ...]:
        shape = tuple(self.full_shape[i] for i in self.keep_axes)
        return shape if len(shape) > 0 else (1,)

    def compact_to_full(self, t: Tensor) -> Tensor:
        """Unsqueeze a compact-grid tensor to full-grid shape (no alloc)."""
        if tuple(t.shape) == self.full_shape:
            return t
        if tuple(t.shape) != self.compact_shape:
            raise ShapeMismatchError(self.compact_shape, tuple(t.shape), "grid")
        out = t
        # Insert singleton axes in the correct positions.
        for ax in self.drop_axes:
            out = out.unsqueeze(ax)
        # Now out.shape matches full_shape.
        return out

    def compact_to_full_prefix(self, t: Tensor) -> Tensor:
        """Unsqueeze compact grid axes at the front of a tensor.

        Supports tensors of shape ``compact_shape + suffix``, returning
        ``full_shape + suffix`` without allocation.
        """
        if tuple(t.shape[: len(self.full_shape)]) == self.full_shape:
            return t
        if tuple(t.shape[: len(self.compact_shape)]) != self.compact_shape:
            raise ShapeMismatchError(
                self.compact_shape, tuple(t.shape[: len(self.compact_shape)]), "grid_prefix"
            )
        out = t
        for ax in self.drop_axes:
            out = out.unsqueeze(ax)
        return out

    def full_to_compact(self, t: Tensor) -> Tensor:
        """Squeeze full-grid singleton axes to compact-grid shape (no alloc)."""
        if tuple(t.shape) == self.compact_shape:
            return t
        if tuple(t.shape) != self.full_shape:
            raise ShapeMismatchError(self.full_shape, tuple(t.shape), "grid")
        out = t
        # Squeeze only the axes that were dropped.
        for ax in reversed(self.drop_axes):
            out = out.squeeze(ax)
        if len(out.shape) == 0:
            # Keep a 1D singleton tensor rather than scalar.
            out = out.view(1)
        return out

    def full_to_compact_prefix(self, t: Tensor) -> Tensor:
        """Squeeze full grid singleton axes at the front of a tensor.

        Supports tensors of shape ``full_shape + suffix``, returning
        ``compact_shape + suffix`` without allocation.
        """
        if tuple(t.shape[: len(self.compact_shape)]) == self.compact_shape and (
            len(self.drop_axes) == 0 or tuple(t.shape[: len(self.full_shape)]) != self.full_shape
        ):
            # Already compact or ambiguous; return as-is.
            return t
        if tuple(t.shape[: len(self.full_shape)]) != self.full_shape:
            raise ShapeMismatchError(
                self.full_shape, tuple(t.shape[: len(self.full_shape)]), "grid_prefix"
            )
        out = t
        for ax in reversed(self.drop_axes):
            out = out.squeeze(ax)
        return out


@dataclass
class SparseNode(ABC):
    """Abstract base class for block-structured sparse tensors.

    Provides interface for viewing tensors as block grids, computing block
    statistics, and applying soft/hard thresholding operations.
    """

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Full shape of the underlying tensor."""
        pass

    @abstractmethod
    def numel(self) -> int:
        """Total number of elements in the underlying tensor."""
        pass

    @abstractmethod
    def parameters(self) -> Iterable[Parameter]:
        """Iterable of Parameter objects managed by this node."""
        pass

    @abstractmethod
    def block_specs(self) -> Iterable["BlockSpec"]:
        """Iterable of BlockSpec objects contained in this node."""
        pass

    @property
    @abstractmethod
    def data(self) -> Mapping["BlockSpec", Tensor] | Tensor:
        """Raw tensor data of the underlying parameter(s)."""
        pass

    @abstractmethod
    def nnz(self, eps=1e-8) -> int:
        """Count non-zero elements with absolute value > eps."""
        pass

    # --- Block grid shapes ---
    # We distinguish between a **full** grid shape (keeps singleton axes) and a
    # **compact** grid shape (drops singleton axes). The full shape is the
    # semantic shape used for axis identity, coupling permutations, and any
    # algorithm that needs stable axis references (e.g., OBS/OBD). The compact
    # shape is a convenience for user-facing threshold tensors.
    @cached_property
    @abstractmethod
    def block_grid_shape_full(self) -> Tuple[int, ...]:
        """Full block-grid shape (keeps singleton axes)."""
        pass

    @cached_property
    def block_grid_shape(self) -> Tuple[int, ...]:
        """Compact block-grid shape (drops singleton axes)."""
        shape = tuple(s for s in self.block_grid_shape_full if s > 1)
        return shape if len(shape) > 0 else (1,)

    @property
    def block_grid_ndim(self) -> int:
        """Number of dimensions in the **compact** block grid."""
        return len(self.block_grid_shape)

    @property
    def block_grid_ndim_full(self) -> int:
        """Number of dimensions in the **full** block grid."""
        return len(self.block_grid_shape_full)

    @cached_property
    @abstractmethod
    def num_blocks(self) -> int:
        """Total number of blocks in the grid."""
        pass

    @cached_property
    @abstractmethod
    def block_numel(self) -> int:
        """Number of elements per block."""
        pass

    @cached_property
    @abstractmethod
    def _reduction_dim(self) -> int | Tuple[int, ...]:
        """Dimension(s) to reduce over when computing block statistics."""
        pass

    @abstractmethod
    def apply_mask(self, mask):
        """Zero out blocks where mask is True."""
        pass

    @abstractmethod
    def apply_multiplier(self, multiplier: Tensor):
        """Multiply each block by corresponding scalar in multiplier."""
        pass

    @abstractmethod
    def block_view(self, values: Values, squeeze=True) -> Tensor:
        """Return a block-structured view of values."""
        pass

    @abstractmethod
    def block_reduce(
        self, values: Values, reduce_fn: Callable[[Tensor], Tensor]
    ) -> Tensor:
        """Apply reduce_fn over each block and return grid-shaped result."""
        pass

    def _block_lp_fn(self, t: Tensor, p, keepdim=False) -> Tensor:
        """Compute Lp norm over reduction dimensions."""
        return torch.linalg.vector_norm(t, ord=p, dim=self._reduction_dim, keepdim=keepdim)

    def _block_min_fn(self, t: Tensor, keepdim=False) -> Tensor:
        """Compute minimum over reduction dimensions."""
        return torch.amin(t, dim=self._reduction_dim, keepdim=keepdim)

    def _block_max_fn(self, t: Tensor, keepdim=False) -> Tensor:
        """Compute maximum over reduction dimensions."""
        return torch.amax(t, dim=self._reduction_dim, keepdim=keepdim)

    def block_norms(self, values: Values = None, p: int = 2) -> Tensor:
        """Compute Lp norm for each block."""
        return self.block_reduce(values, lambda t: self._block_lp_fn(t, p=p))

    def block_min(self, values: Values = None) -> Tensor:
        """Compute minimum value for each block."""
        return self.block_reduce(values, self._block_min_fn)

    def block_max(self, values: Values = None) -> Tensor:
        """Compute maximum value for each block."""
        return self.block_reduce(values, self._block_max_fn)

    def block_sumsq(self, values: Values = None) -> Tensor:
        """Compute sum of squares for each block (compact grid)."""
        return self.block_reduce(values, lambda t: torch.sum(t * t, dim=self._reduction_dim))

    @abstractmethod
    def _soft_threshold_euclid(self, block_thresholds, eps=1e-20):
        """In-place Euclidean (L2) proximal step."""
        pass

    @abstractmethod
    def _soft_threshold_adam(
        self,
        block_thresholds: Tensor,
        conditioners,
        max_iter=20,
        eps=1e-20,
        atol=1e-8,
    ):
        """In-place Adam-conditioned proximal step."""
        pass

    @torch.no_grad()
    def soft_threshold(
        self,
        block_thresholds,
        conditioners=None,
        scale=False,
        max_iter=20,
        eps=1e-20,
        atol=1e-8,
    ):
        """Apply soft thresholding to shrink block norms.

        Args:
            block_thresholds: Per-block threshold values.
            conditioners: Optional diagonal conditioner for Adam variant.
            scale: If True, scale thresholds by sqrt(block_numel).
            max_iter: Maximum iterations for Adam variant.
            eps: Small constant for numerical stability.
            atol: Absolute tolerance for convergence.
        """
        assert tuple(block_thresholds.shape) == self.block_grid_shape

        if scale:
            block_thresholds = block_thresholds * (self.block_numel**0.5)
        if conditioners is None:
            self._soft_threshold_euclid(block_thresholds, eps=eps)
        else:
            self._soft_threshold_adam(
                block_thresholds=block_thresholds,
                conditioners=conditioners,
                max_iter=max_iter,
                atol=atol,
            )

    @torch.no_grad()
    def hard_threshold(self, thresholds: Tensor, values: Values = None):
        """Zero out blocks with values-based norm below threshold.

        Args:
            thresholds: Per-block threshold values.
            values: Optional values to compute norms from; defaults to data.
        """
        if tuple(thresholds.shape) != self.block_grid_shape:
            raise ValueError(
                f"thresholds shape {thresholds.shape} must match "
                f"block_grid_size {self.block_grid_shape}"
            )

        blocks_to_mask = self.block_norms(values) < thresholds

        self.apply_mask(blocks_to_mask)

    @abstractmethod
    def get_masks(self, block_masks: Tensor) -> Mapping["BlockSpec", Tensor]:
        """Convert block-level mask to element-level masks per BlockSpec."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        return repr(self)

    @abstractmethod
    def __hash__(self) -> int:
        pass


@dataclass
class BlockSpec(SparseNode):
    """Treats the entire tensor as a grid of blocks.

    Attributes:
        param: The underlying Parameter tensor.
        block_shape: Shape of each block in the grid.
        name: Optional name for identification.
    """

    param: Parameter
    block_shape: Tuple[int, ...]
    name: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize block shape after initialization."""
        if len(self.block_shape) == 0:  # if block size empty, default to 1
            self.block_shape = tuple([1 for _ in range(self.param.ndim)])

        if len(self.block_shape) != self.param.ndim:
            raise ValueError(
                f"{self.name} block has len {len(self.block_shape)}:{self.block_shape} "
                f"but tensor is {self.param.ndim}D:{self.param.shape}"
            )
        self.block_shape = tuple(
            [
                bi if bi > 0 else self.shape[i]  # -1 means use the entire dim
                for i, bi in enumerate(self.block_shape)
            ]
        )
        for i, (si, bi) in enumerate(zip(self.shape, self.block_shape)):
            if si % bi != 0:
                raise ValueError(
                    f"dim {i}: size {si} not divisible by block_size[{i}]={bi}"
                )

        assert self.block_numel > 0

    def block_specs(self) -> Iterable["BlockSpec"]:
        """Return self as the only BlockSpec."""
        return [self]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Full shape of the underlying tensor."""
        return tuple(self.param.shape)

    def numel(self) -> int:
        """Number of elements in the underlying tensor."""
        return self.param.numel()

    @property
    def ndim(self) -> int:
        """Number of dimensions in the underlying tensor."""
        return self.param.ndim

    def parameters(self) -> List[Parameter]:
        """List containing the single underlying Parameter."""
        return [self.param]

    @property
    def data(self) -> Tensor:
        """Raw tensor data of the underlying Parameter."""
        return self.param.data

    def set_data(self, data):
        """Copy data into the underlying Parameter tensor."""
        self.param.data.copy_(data)

    def nnz(self, eps=1e-8) -> int:
        """Number of *non-zero* elements (within tolerance)."""
        return int((self.param.data.abs() > eps).sum().item())

    @cached_property
    def _grid_shape(self) -> Tuple[int, ...]:
        """Full grid shape including singleton dimensions."""
        return tuple(si // bi for si, bi in zip(self.shape, self.block_shape))

    @cached_property
    def _grid_plan(self) -> GridAxesPlan:
        """Axis plan for mapping between full and compact block grids."""
        return GridAxesPlan(self._grid_shape)

    @cached_property
    def block_grid_shape_full(self) -> Tuple[int, ...]:
        """Full block-grid shape (keeps singleton axes)."""
        return self._grid_shape

    @cached_property
    def block_numel(self) -> int:
        """Number of elements per block."""
        return math.prod(self.block_shape)

    @cached_property
    def num_blocks(self) -> int:
        """Total number of blocks in the grid."""
        return math.prod(self.block_grid_shape)

    @cached_property
    def _reduction_dim(self) -> Tuple[int, ...]:
        """Odd-indexed dimensions to reduce over for block statistics."""
        return tuple(range(1, 2 * len(self.block_shape), 2))

    def _resolve_values(self, values: Values) -> Tensor:
        """Resolve values to a tensor matching self.shape."""
        if values is None:
            return self.param.data
        if isinstance(values, dict):
            return values[self]
        if isinstance(values, Tensor):
            if values.shape != self.shape:
                raise ShapeMismatchError(
                    self.shape, tuple(values.shape), "values"
                )
            return values
        raise ValueError(
            "values has to be None, Tensor or Dict[BlockSpec, Tensor]"
        )

    def _raw_block_view(self, t: Tensor, merge: bool = False) -> Tensor:
        """Reshape tensor to interleaved block view.

        Args:
            t: Input tensor matching self.shape.
            merge: If True, collapse block dims to trailing dim.

        Returns:
            If merge=False: (B0, b0, B1, b1, ...).
            If merge=True: (B0, B1, ..., b0*b1*...).
        """
        assert t.shape == self.shape
        interleaved_shape = []
        for B, bi in zip(self._grid_shape, self.block_shape):
            interleaved_shape.extend([B, bi])

        view = t.view(*interleaved_shape)

        if merge:
            view = merge_odd_dims(view)

        return view

    def expand_block_tensor(self, block_values: Tensor) -> Tensor:
        """Convert a compact/full block-grid tensor to full grid shape.

        Historically the library used a compact grid shape (singleton axes
        dropped). For coupling and second-order methods we keep the **full**
        grid shape for axis identity. This helper maps either representation
        to the full grid shape without allocation.
        """
        if tuple(block_values.shape) == self.block_grid_shape_full:
            return block_values
        # If it's compact, unsqueeze the dropped axes.
        block_values = self._grid_plan.compact_to_full(block_values)
        return block_values

    def block_view(self, values: Values, squeeze=True) -> Tensor:
        """Return a blocked view of values.

        Args:
            values: Input values (None uses param.data).
            squeeze: If True, remove singleton grid dimensions.

        Returns:
            Tensor with shape (B1, B2, ..., block_numel) if squeeze=True.
        """
        t = self._resolve_values(values)
        view = self._raw_block_view(t, merge=True)
        if squeeze:
            view = view.view(*self.block_grid_shape, -1)
        return view

    def broadcast_block_to_element(
        self,
        block_values: Tensor,
        fake: bool = False,
        materialize: bool = False,
    ) -> Tensor:
        """Broadcast block-grid values to element space.

        This method supports both compact and full block-grid tensors.

        - If ``fake=True``: returns an interleaved tensor of shape
          (B0,1,B1,1,...), suitable for broadcasting against a block view.
        - If ``fake=False`` and ``materialize=False``: returns an interleaved
          broadcast view (stride-0 expansion) of shape (B0,b0,B1,b1,...), which
          can be used for read-only operations without allocating.
        - If ``fake=False`` and ``materialize=True``: returns a materialized
          tensor of shape ``self.shape``.
        """
        # Accept compact or full.
        expanded = self.expand_block_tensor(block_values)
        # expanded: (B0,B1,...)
        for i, bi in enumerate(self.block_shape):
            expanded = expanded.unsqueeze(2 * i + 1)  # (..., Bi, 1)
            if not fake:
                # Expand singleton axis to block extent (stride-0, no alloc)
                expanded = expanded.expand(*expanded.shape[: 2 * i + 1], bi, *expanded.shape[2 * i + 2 :])
        if fake:
            return expanded
        if materialize:
            return expanded.reshape(*self.shape)
        # Non-materialized interleaved broadcast view.
        return expanded

    def apply_mask(self, mask: Tensor):
        """Zero out blocks where mask is True."""
        self.apply_multiplier(~mask)

    def apply_multiplier(self, multiplier: Tensor):
        """Multiply each block by corresponding scalar in multiplier."""
        assert multiplier.shape == self.block_grid_shape

        # Shape (B1, B2, B3,...)
        multiplier = self.expand_block_tensor(multiplier)

        # Shape (B1,1, B2, 1, B3, 1, ...)
        multiplier = interleave_unsqueeze(multiplier)

        # Shape (B1, b1, B2, b2,....)
        b_view = self._raw_block_view(self.param.data, merge=False)
        b_view.mul_(multiplier)

    def block_reduce(
        self, values: Values, reduce_fn: Callable[[Tensor], Tensor]
    ) -> Tensor:
        """Apply reduce_fn over each block and return grid-shaped result."""
        t = self._resolve_values(values)
        t = self._raw_block_view(t, merge=False)
        # reduce_fn(t) returns (B0,B1,...), i.e. full grid.
        reduced_full = reduce_fn(t).view(self.block_grid_shape_full)
        return self._grid_plan.full_to_compact(reduced_full)

    def block_reduce_full(
        self, values: Values, reduce_fn: Callable[[Tensor], Tensor]
    ) -> Tensor:
        """Same as :meth:`block_reduce` but returns the **full** grid shape."""
        t = self._resolve_values(values)
        t = self._raw_block_view(t, merge=False)
        return reduce_fn(t).view(self.block_grid_shape_full)

    def _soft_threshold_euclid(self, block_thresholds, eps=1e-8):
        """In-place Euclidean (L2) proximal step."""

        assert tuple(block_thresholds.shape) == self.block_grid_shape

        block_norms = self.block_norms(self.param.data)

        block_factor = 1 - block_thresholds / (block_norms + eps)
        block_factor.clamp_(min=0.0)

        self.apply_multiplier(block_factor)

    def _soft_threshold_adam(
        self,
        block_thresholds: Tensor,
        conditioners: Values,
        max_iter=20,
        eps=1e-20,
        atol=1e-8,
    ):
        """In-place Adam-conditioned proximal step via bisection."""
        assert block_thresholds.shape == self.block_grid_shape
        assert conditioners is not None
        conditioner = self._resolve_values(conditioners)

        assert isinstance(conditioner, Tensor)
        assert conditioner.shape == self.shape

        if self.block_numel == 1:
            return self._soft_threshold_euclid(
                block_thresholds / conditioner.view(self.block_grid_shape)
            )

        Hv = conditioner * self.param.data
        Hv_norms = self.block_norms(Hv)

        denom = Hv_norms - block_thresholds
        non_survivors = denom <= 0.0
        denom.clamp_(min=0.0).add_(eps)

        h_min = self.block_min(conditioner)
        h_max = self.block_max(conditioner)

        # block_thresholds > 0
        # if Hv_norms < block_thresholds, then denom<0, so we clamp for safety

        mu_low = (block_thresholds * h_min) / denom
        mu_high = (block_thresholds * h_max) / denom

        # (B1, 1, B2,1,...)
        mu_low = self.broadcast_block_to_element(mu_low, fake=True).clamp_(
            min=0.0
        )
        mu_high = self.broadcast_block_to_element(mu_high, fake=True).clamp_(
            min=0.0
        )

        blocked_thresholds = self.broadcast_block_to_element(
            block_thresholds, fake=True
        )

        # (B1, b1, B2, b1,...)
        blocked_conditioner = self._raw_block_view(conditioner, merge=False)
        blocked_Hv = self._raw_block_view(Hv, merge=False)

        mu = (mu_low + mu_high) / 2
        for _ in range(max_iter):
            # Compute Zeta(mu)
            # scaling = H_block / (H_block + mu)

            # ||H / (H+mu) v||
            # (B1, 1, B2,1,...)
            weighted_norm = self._block_lp_fn(
                blocked_Hv / (blocked_conditioner + mu), p=2, keepdim=True
            )
            # zeta = mu * ||weighted_v||
            zeta = mu * weighted_norm

            # Zeta is strictly increasing with mu.
            # If zeta < threshold, mu is too small -> low = mu
            # If zeta > threshold, mu is too big -> high = mu
            mask_low = zeta < blocked_thresholds
            mu_low = torch.where(mask_low, mu, mu_low)
            mu_high = torch.where(~mask_low, mu, mu_high)
            mu = (mu_low + mu_high) / 2
            if (mu_low - mu_high).abs().max() < atol:
                break

        scaling = conditioner / (
            conditioner
            + self.broadcast_block_to_element(mu.view(self.block_grid_shape))
        )

        self.set_data(scaling * self.param.data)

        # only keep survivors
        self.apply_mask(non_survivors)

    def get_masks(self, block_masks) -> Mapping["BlockSpec", Tensor]:
        """Convert block-level mask to element-level mask.

        Args:
            block_masks: Boolean tensor with shape block_grid_shape.

        Returns:
            Dict mapping self to the broadcasted element mask.
        """
        block_masks = self.broadcast_block_to_element(block_masks, materialize=True)
        return {self: block_masks}

    def __repr__(self) -> str:
        """Return string representation with shape information."""
        return (
            f"{self.__class__.__name__}"
            f"(shape={self.shape}, block_shape={self.block_shape}, "
            f"block_grid_shape={self.block_grid_shape}, name={self.name!r})"
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        """Hash based on the underlying Parameter instance."""
        return hash(self.param)


@dataclass
class BlockCoupling(SparseNode):
    """Merges multiple BlockSpec objects into one coupled sparse node.

    Attributes:
        specs: List of BlockSpec objects to couple.
        orders: Axis permutations to align block grids.
        name: Optional name for identification.
    """

    specs: List[BlockSpec]
    orders: List[Tuple[int, ...]]
    name: Optional[str] = None

    _ref_order: Tuple[int] = field(init=False)
    _ref_block_grid_shape: Tuple[int] = field(init=False)
    _reverse_orders: List[Tuple[int, ...]] = field(init=False)

    @cached_property
    def _grid_plan(self) -> GridAxesPlan:
        """Axis plan for mapping between full and compact coupled grids."""
        return GridAxesPlan(self.block_grid_shape_full)

    def block_specs(self) -> Iterable["BlockSpec"]:
        """Return the list of coupled BlockSpec objects."""
        return self.specs

    def __post_init__(self):
        """Validate and compute axis orderings for all specs."""
        if not self.orders:
            self.orders = [
                tuple(range(len(s.block_grid_shape_full))) for s in self.specs
            ]
        if len(self.orders) != len(self.specs):
            raise ValueError("orders must match number of specs.")

        self.orders = [
            normalize_order(o, len(s.block_grid_shape_full))
            for o, s in zip(self.orders, self.specs)
        ]

        self._ref_order = self.orders[0]  # type: ignore
        self._ref_block_grid_shape = ref_permute = tuple(  # type: ignore
            self.specs[0].block_grid_shape_full[i] for i in self._ref_order
        )

        self._reverse_orders = []
        for s, o in zip(self.specs, self.orders):
            Bis = s.block_grid_shape_full
            bperm = tuple(Bis[i] for i in o)
            if bperm != ref_permute:
                raise ValueError(
                    "Incompatible block grid shapes"
                    f"after order: {bperm} vs {ref_permute} "
                    f"(spec {s.name or '<unnamed>'})"
                )
            self._reverse_orders.append(inverse_permutation(o))

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return placeholder shape (-1, -1) for coupled specs."""
        return (-1, -1)

    def numel(self) -> int:
        """Total number of elements across all specs."""
        return sum([s.numel() for s in self.specs])

    def parameters(self) -> List[Parameter]:
        """List of all Parameter objects from coupled specs."""
        return [s.param for s in self.specs]

    @property
    def data(self) -> Mapping[BlockSpec, Tensor]:
        """Dict mapping each spec to its tensor data."""
        return {s: s.param.data for s in self.specs}

    def nnz(self, eps=1e-8) -> int:
        """Count non-zero elements across all specs."""
        return sum([s.nnz(eps=eps) for s in self.specs])

    @cached_property
    def block_grid_shape_full(self) -> Tuple[int, ...]:
        """Reference **full** block grid shape for the coupling."""
        return self._ref_block_grid_shape

    @cached_property
    def num_blocks(self) -> int:
        """Number of blocks (same for all specs after alignment)."""
        return self.specs[0].num_blocks

    @cached_property
    def block_numel(self) -> int:
        """Total elements per block across all specs."""
        return sum([s.block_numel for s in self.specs])

    @cached_property
    def _reduction_dim(self):
        """Reduction dimension for block statistics (last dim)."""
        return -1

    def _resolve_values(self, values: Values) -> Mapping[BlockSpec, Tensor]:
        """Resolve values to a mapping of BlockSpec to Tensor."""
        if values is None:
            return {s: s.param.data for s in self.specs}
        if isinstance(values, dict):
            return {s: values[s] for s in self.specs}
        raise ValueError("values must be Mapping[BlockSpec,Tensor]")

    def _raw_block_view(
        self, spec_values: Mapping[BlockSpec, Tensor]
    ) -> Tensor:
        """Reshape and concatenate all spec values into unified block view.

        Args:
            spec_values: Mapping from BlockSpec to tensor values.

        Returns:
            Concatenated tensor of shape (B0, B1, ..., total_block_numel).
        """
        values = []
        for o, s in zip(self.orders, self.specs):
            # Use the full grid view (keeps singleton axes) for stable
            # coupling semantics. The base class provides a compact view for
            # user-facing thresholds, but coupling/permutation requires the
            # full axis identity.
            values.append(
                s.block_view(spec_values[s], squeeze=False).permute(*o, len(o))
            )
        return torch.concat(values, dim=-1)

    def block_view(self, values: Values, squeeze=True) -> Tensor:
        """Return unified block view of all coupled specs.

        If squeeze=True, singleton grid axes are removed (compact view). For
        coupling semantics and OBS/OBD implementations, prefer squeeze=False.
        """
        view_full = self._raw_block_view(self._resolve_values(values))
        return self._grid_plan.full_to_compact_prefix(view_full) if squeeze else view_full

    def block_reduce(
        self, values: Values, reduce_fn: Callable[[Tensor], Tensor]
    ) -> Tensor:
        """Apply reduce_fn over concatenated block view.

        Returns the **compact** grid by default (drops singleton axes).
        """
        spec_values = self._resolve_values(values)
        concat_values_full = self._raw_block_view(spec_values)
        reduced_full = reduce_fn(concat_values_full)
        return self._grid_plan.full_to_compact(reduced_full)

    def block_reduce_full(
        self, values: Values, reduce_fn: Callable[[Tensor], Tensor]
    ) -> Tensor:
        """Same as :meth:`block_reduce` but returns the **full** coupled grid."""
        spec_values = self._resolve_values(values)
        concat_values_full = self._raw_block_view(spec_values)
        return reduce_fn(concat_values_full)

    # --- Fused per-block statistics ---
    # For common ops (norm/min/max/sumsq) we avoid concatenating block vectors.
    # Concatenation forces extra reads and can allocate a large temporary.

    def block_sumsq(self, values: Values = None) -> Tensor:
        spec_values = self._resolve_values(values)
        out_full: Optional[Tensor] = None
        for o, s in zip(self.orders, self.specs):
            # full grid for this spec, then permute into reference grid.
            v_full = s.block_view(spec_values[s], squeeze=False)
            ss_full = torch.sum(v_full * v_full, dim=-1).permute(*o)
            out_full = ss_full if out_full is None else (out_full + ss_full)
        assert out_full is not None
        return self._grid_plan.full_to_compact(out_full)

    def block_norms(self, values: Values = None, p: int = 2) -> Tensor:
        if p != 2:
            # Fallback to generic reduce for non-Euclidean norms.
            return super().block_norms(values, p=p)
        return torch.sqrt(self.block_sumsq(values))

    def block_min(self, values: Values = None) -> Tensor:
        spec_values = self._resolve_values(values)
        out_full: Optional[Tensor] = None
        for o, s in zip(self.orders, self.specs):
            v_full = s.block_view(spec_values[s], squeeze=False)
            mn_full = torch.amin(v_full, dim=-1).permute(*o)
            out_full = mn_full if out_full is None else torch.minimum(out_full, mn_full)
        assert out_full is not None
        return self._grid_plan.full_to_compact(out_full)

    def block_max(self, values: Values = None) -> Tensor:
        spec_values = self._resolve_values(values)
        out_full: Optional[Tensor] = None
        for o, s in zip(self.orders, self.specs):
            v_full = s.block_view(spec_values[s], squeeze=False)
            mx_full = torch.amax(v_full, dim=-1).permute(*o)
            out_full = mx_full if out_full is None else torch.maximum(out_full, mx_full)
        assert out_full is not None
        return self._grid_plan.full_to_compact(out_full)

    def _soft_threshold_euclid(self, block_thresholds, eps=1e-8):
        """In-place Euclidean (L2) proximal step."""

        assert tuple(block_thresholds.shape) == self.block_grid_shape

        block_norms = self.block_norms({s: s.param.data for s in self.specs})

        block_factor = 1 - block_thresholds / (block_norms + eps)
        block_factor.clamp_(min=0.0)

        self.apply_multiplier(block_factor)

    def _soft_threshold_adam(
        self,
        block_thresholds: Tensor,
        conditioners: Mapping[BlockSpec, Tensor],
        max_iter=20,
        atol=1e-8,
    ):
        """In-place Adam-conditioned proximal step via bisection."""
        assert tuple(block_thresholds.shape) == self.block_grid_shape
        for s in self.specs:
            assert conditioners[s].shape == s.shape

        Hv = {s: conditioners[s] * s.param.data for s in self.specs}
        Hv_norms = self.block_norms(Hv)

        denom = Hv_norms - block_thresholds

        denom = Hv_norms - block_thresholds
        non_survivors = denom <= 0.0
        denom.clamp_(min=0.0)

        h_min = self.block_min(conditioners)
        h_max = self.block_max(conditioners)

        mu_low = ((block_thresholds * h_min) / denom).clamp_(min=0.0)
        mu_high = ((block_thresholds * h_max) / denom).clamp_(min=0.0)

        mu = (mu_low + mu_high) / 2
        for _ in range(max_iter):
            # Compute Zeta(mu)
            # scaling = H_block / (H_block + mu)
            # mu.contiguous()

            # Promote mu to full grid for stable axis mapping, then permute.
            # Compute ||Hv / (H+mu)||_block without materializing element-wise
            # broadcast tensors. Operate in interleaved (blocked) views and
            # reduce over block dimensions.
            mu_full = self._grid_plan.compact_to_full(mu)
            w_sumsq_full: Optional[Tensor] = None
            for o, ro, s in zip(self.orders, self._reverse_orders, self.specs):
                mu_spec_full = mu_full.permute(*ro)
                mu_fake = s.broadcast_block_to_element(mu_spec_full, fake=True)
                Hv_blk = s._raw_block_view(Hv[s], merge=False)
                H_blk = s._raw_block_view(conditioners[s], merge=False)
                w_blk = Hv_blk / (H_blk + mu_fake)
                ss_spec = torch.sum(w_blk * w_blk, dim=s._reduction_dim)
                ss_ref = ss_spec.permute(*o)
                w_sumsq_full = ss_ref if w_sumsq_full is None else (w_sumsq_full + ss_ref)
            assert w_sumsq_full is not None
            weighted_norm = self._grid_plan.full_to_compact(torch.sqrt(w_sumsq_full))
            # zeta = mu * ||weighted_v||
            zeta = mu * weighted_norm

            # Zeta is strictly increasing with mu.
            # If zeta < threshold, mu is too small -> low = mu
            # If zeta > threshold, mu is too big -> high = mu
            mask_low = zeta < block_thresholds

            mu_low = torch.where(mask_low, mu, mu_low)
            mu_high = torch.where(~mask_low, mu, mu_high)

            mu = (mu_low + mu_high) / 2

            if (mu_low - mu_high).abs().max() < atol:
                break

        # Apply scaling in blocked views to avoid element-wise materialization.
        mu_full = self._grid_plan.compact_to_full(mu)
        for ro, s in zip(self._reverse_orders, self.specs):
            mu_spec_full = mu_full.permute(*ro)
            mu_fake = s.broadcast_block_to_element(mu_spec_full, fake=True)
            W_blk = s._raw_block_view(s.param.data, merge=False)
            H_blk = s._raw_block_view(conditioners[s], merge=False)
            W_blk.mul_(H_blk / (H_blk + mu_fake))

        # only keep survivors
        self.apply_mask(non_survivors)

    def get_masks(self, block_masks: Tensor) -> Mapping["BlockSpec", Tensor]:
        """Convert block-level mask to element-level masks for each spec."""
        spec_masks = {}
        # Promote to full grid for permutation, then map to each spec's compact grid.
        block_masks_full = self._grid_plan.compact_to_full(block_masks)
        for ro, s in zip(self._reverse_orders, self.specs):
            m_full = block_masks_full.permute(*ro)
            m_compact = s._grid_plan.full_to_compact(m_full)
            spec_masks.update(s.get_masks(m_compact))
        return spec_masks

    def apply_mask(self, mask: Tensor):
        """Zero out blocks where mask is True across all specs."""
        self.apply_multiplier(~mask)

    def apply_multiplier(self, multiplier: Tensor):
        """Multiply each block by corresponding scalar across all specs."""
        assert (
            multiplier.shape == self.block_grid_shape
        ), "Incompatible Multiplier"
        multiplier_full = self._grid_plan.compact_to_full(multiplier)
        for ro, s in zip(self._reverse_orders, self.specs):
            m_full = multiplier_full.permute(*ro)
            m_compact = s._grid_plan.full_to_compact(m_full)
            s.apply_multiplier(m_compact)

    def __repr__(self) -> str:
        """Return string representation with specs info."""
        specs_str = ",\n\t".join(str(s) for s in self.specs)
        return (
            f"{self.__class__.__name__}"
            f"(block_grid_shape={self.block_grid_shape}, ref_order={self._ref_order}, "
            f"name={self.name!r}, "
            f"BlockSpecs=[\n\t{specs_str}])"
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        """Hash based on hashes of all coupled specs."""
        return hash(tuple(hash(s) for s in self.specs))


if __name__ == "__main__":
    import torch

    v = BlockSpec(
        torch.nn.Parameter(torch.randn(4, 2, 4)),
        block_shape=(2, 1, 2),
    )
    print(v)
    print(v.block_min().shape)
    print(v.block_norms().shape)



class SparseGroup(ABC):
    """
    Abstract base class for a sparse groups
    """

    # Similar to SparseNode, we expose a full/compact group-grid shape.
    @property
    @abstractmethod
    def group_grid_shape_full(self) -> Tuple[int, ...]:
        """Full group-grid shape (keeps singleton axes)."""
        pass

    @property
    def group_grid_shape(self) -> Tuple[int, ...]:
        """Compact group-grid shape (drops singleton axes)."""
        shape = tuple(s for s in self.group_grid_shape_full if s > 1)
        return shape if len(shape) > 0 else (1,)

    @property
    def group_grid_ndim_full(self) -> int:
        return len(self.group_grid_shape_full)

    @property
    def num_groups(self) -> int:
        return math.prod(self.group_grid_shape)

    @property
    @abstractmethod
    def group_numel(self) -> int:
        pass

    @abstractmethod
    def nnz(self, eps=1e-8) -> int:
        pass

    @abstractmethod
    def specs(self) -> Iterable[BlockSpec]:
        pass

    @property
    @abstractmethod
    def data(self) -> Mapping["BlockSpec", Tensor] | Tensor:
        pass

    @abstractmethod
    @torch.no_grad()
    def hard_threshold(
        self,
        thresholds: Optional[Tensor] = None,
        num_nz: Optional[int] = None,
        values: Values = None,
        sparsity: Optional[float] = None,
    ):
        """
        Zeros out blocks in-place based on group-level thresholds.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def soft_threshold(
        self,
        thresholds,
        conditioners,
        scale=False,
        max_iter=20,
        eps=1e-8,
    ):
        pass

    @abstractmethod
    @torch.no_grad()
    def get_masks(
        self,
        num_nz: int,
        grouped_block_scores: Tensor | None = None,
        values: Values = None,
        **kwargs,
    ) -> Mapping[BlockSpec, Tensor]:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        return repr(self)

    @abstractmethod
    def __hash__(self) -> int:
        pass


@dataclass
class GroupSpec(SparseGroup):
    """
    Specification for for N-D sparsification
      param: torch.nn.Parameter with shape s = (s1,...,sm)
      block_size: (b1,...,bm) with si % bi == 0, block grid B=(si//bi)_i
      if bi=-1 -> bi=si
      group_size: (g1,...,gm) with Bi % gi == 0, group grid G = (Gi = Bi//gi)_i
      if gi = -1 -> gi = Bi
    """

    block: SparseNode
    group_shape: Tuple[int, ...]
    name: Optional[str] = None

    def __post_init__(self):
        if not self.group_shape:
            self.group_shape = tuple(-1 for _ in self.block.block_grid_shape_full)

        if len(self.group_shape) != len(self.block.block_grid_shape_full):
            raise ValueError(
                f"group shape {self.group_shape} has len {len(self.group_shape)} "
                f"but block_grid_shape_full = {self.block.block_grid_shape_full}D"
            )
        self.group_shape = tuple(
            [
                self.block.block_grid_shape_full[i] if gi == -1 else gi
                for i, gi in enumerate(self.group_shape)
            ]
        )

        for i, (Bi, gi) in enumerate(zip(self.block.block_grid_shape_full, self.group_shape)):
            if Bi % gi != 0:
                raise ValueError(
                    f"dim {i}: block_grid[{i}]={Bi} "
                    f"not divisible by group_size[{i}]={gi}"
                )

    @property
    def data(self) -> Mapping[BlockSpec, Tensor]:
        data = self.block.data
        if isinstance(data, Tensor):
            assert isinstance(self.block, BlockSpec)
            return {self.block: data}
        else:
            return data

    def specs(self) -> Iterable[BlockSpec]:
        return [s for s in self.block.block_specs()]

    @property
    def _grid_shape(self) -> Tuple[int, ...]:
        return tuple(
            Bi // gi
            for Bi, gi in zip(self.block.block_grid_shape_full, self.group_shape)
        )

    @cached_property
    def _grid_plan(self) -> GridAxesPlan:
        return GridAxesPlan(self._grid_shape)

    @property
    def group_grid_shape_full(self) -> Tuple[int, ...]:
        return self._grid_shape

    @property
    def num_groups(self) -> int:
        # Keep compatibility: number of groups is product of the compact grid.
        return math.prod(self.group_grid_shape)

    def numel(self) -> int:
        return self.block.numel()

    @property
    def group_numel(self) -> int:
        return math.prod(self.group_shape)

    def nnz(self, eps=1e-8) -> int:
        return self.block.nnz(eps)

    def block_to_group(self, b: Tensor, squeeze=True, merge=False) -> Tensor:
        """
        Return a grouped view of b.

        b: shape (B1, B2,...,Bm)
        Returns tensor with shape (G1,g1,G2,g2...,Gm,gm), where Gi=Bi/gi if ``merge=False``

        If ``merge=True`` the block-dimensions are collapsed into the last dim:
        to get (G1,G2,..., g1*g2*...)
        """
        # Accept either compact or full block-grid tensors.
        if tuple(b.shape) == self.block.block_grid_shape:
            # Expand compact -> full using the underlying block's axis plan.
            b = self.block._grid_plan.compact_to_full(b)  # type: ignore[attr-defined]
        assert tuple(b.shape) == self.block.block_grid_shape_full
        inter_shape = [
            Gg
            for pair in zip(self._grid_shape, self.group_shape)
            for Gg in pair
        ]
        view = b.view(*inter_shape)

        if merge:
            view = merge_odd_dims(view)
            if squeeze:
                # Squeeze only singleton group-grid axes (not group elements).
                view = self._grid_plan.full_to_compact_prefix(view)
        elif squeeze:
            # When not merged, keep the interleaved (G,g,...) view; squeezing
            # would drop within-group axes and break axis identity.
            view = view
        return view

    def group_to_block(self, group_values) -> Tensor:
        """
        Broadcast a tensor of group values back to a block view.
        group_values: shape(G1,...,Gm)
        Returns tensor of shape (B1,...,Bm) with Bi=Gi*gi
        """
        # Accept either compact or full group-grid tensors.
        if tuple(group_values.shape) == self.group_grid_shape:
            group_values = self._grid_plan.compact_to_full(group_values)
        assert tuple(group_values.shape) == self.group_grid_shape_full

        inter_values = group_values
        for i, gi in enumerate(self.group_shape):
            inter_values = inter_values.unsqueeze(2 * i + 1)
            # Expand singleton axis to within-group extent (stride-0, no alloc)
            inter_values = inter_values.expand(*inter_values.shape[: 2 * i + 1], gi, *inter_values.shape[2 * i + 2 :])

        # Interleaved (G0,g0,G1,g1,...) -> block grid (B0,B1,...)
        return inter_values.reshape(self.block.block_grid_shape_full)

    def grouped_block_norms(self, values: Values):
        # Compute block norms in **full** grid coordinates, then group.
        bv_full = self.block.block_view(values, squeeze=False)
        block_norms_full = torch.linalg.vector_norm(bv_full, dim=-1)
        group_norms = self.block_to_group(block_norms_full, squeeze=False)
        merged_full = merge_odd_dims(group_norms)
        # Return compact group-grid prefix for thresholds/topk.
        return self._grid_plan.full_to_compact_prefix(merged_full)

    def kth_largest(
        self,
        element_values: Mapping[BlockSpec, Tensor] | None,
        num_nz,
    ):
        """
        Calculates the k-th largest score across all groups from all specs.
        This is used to determine the threshold for pruning.
        """
        grouped_block_scores = self.grouped_block_norms(element_values)
        top_scores = kth_largest(grouped_block_scores, k=num_nz, dim=-1)
        return self._grid_plan.full_to_compact(top_scores)

    @torch.no_grad()
    def hard_threshold(
        self,
        thresholds: Optional[Tensor] = None,
        num_nz: Optional[int] = None,
        values: Optional[Values] = None,
        sparsity: Optional[float] = None,
    ):
        """
        Zeros out blocks in-place based on group-level thresholds.
        """
        if thresholds is None:
            if num_nz is None:
                if sparsity is None:
                    raise ValueError(
                        "Either group_thresholds or kappa or sparsity should be provided"
                    )
                else:
                    num_nz = self.group_numel - int(sparsity * self.group_numel)

            if num_nz == self.group_numel:
                return

            thresholds = self.kth_largest(None, num_nz=num_nz)

        assert thresholds.shape == self.group_grid_shape

        # group->block yields a full block-grid tensor; map to block's compact grid.
        block_thr_full = self.group_to_block(thresholds)
        block_thr = self.block._grid_plan.full_to_compact(block_thr_full)  # type: ignore[attr-defined]
        self.block.hard_threshold(block_thr)

    @torch.no_grad()
    def get_masks(
        self,
        num_nz: int,
        grouped_block_scores: Tensor | None = None,
        values: Values = None,
        grouped_mask: Tensor | None = None,
        **kwargs,
    ) -> Mapping[BlockSpec, Tensor]:
        if grouped_mask is None:
            if grouped_block_scores is None:
                grouped_block_scores = self.grouped_block_norms(values)
            else:
                assert grouped_block_scores.shape == self.group_grid_shape + (
                    self.group_numel,
                )

            indices = torch.topk(grouped_block_scores, k=num_nz, dim=-1)[1]

            grouped_mask = torch.zeros_like(grouped_block_scores).bool()
            grouped_mask.scatter_(-1, indices, True)

        # Convert compact group-grid mask -> full group-grid -> full block-grid
        grouped_mask_full = self._grid_plan.compact_to_full_prefix(grouped_mask)
        block_mask_inter = unmerge_odd_dims(
            grouped_mask_full.view(self._grid_shape + (self.group_numel,)),
            self.group_shape,
        )
        block_mask_full = block_mask_inter.reshape(self.block.block_grid_shape_full)
        block_mask = self.block._grid_plan.full_to_compact(block_mask_full)  # type: ignore[attr-defined]
        return self.block.get_masks(block_mask)

    @torch.no_grad()
    def soft_threshold(
        self,
        thresholds,
        conditioners: Mapping[BlockSpec, Tensor],
        scale=False,
        max_iter=20,
        eps=1e-20,
        atol=1e-8,
    ):
        """
        Applies soft thresholding (proximal operator for L1) to blocks in-place.
        group_lamdas: shape (G1,G2,...,Gm) = self.group_grid_size
        eta_t:
        """
        assert tuple(thresholds.shape) == self.group_grid_shape

        block_lambdas_full = self.group_to_block(thresholds)
        block_lambdas = self.block._grid_plan.full_to_compact(block_lambdas_full)  # type: ignore[attr-defined]
        if scale:
            block_lambdas = block_lambdas * (self.block.block_numel**0.5)

        self.block.soft_threshold(
            block_lambdas,
            conditioners=conditioners,
            max_iter=max_iter,
            eps=eps,
            atol=atol,
        )

    def apply_mask(self, mask):
        assert mask.shape == tuple(self.group_grid_shape + (self.group_numel,))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}[group_shape={self.group_shape}, "
            f"group_grid_shape={self.group_grid_shape}, "
            f"block={self.block}, "
            f"name={self.name}]"
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash((hash(self.block), self.group_shape))


@dataclass
class GroupCoupling(SparseGroup):
    """
    Couples multiple GroupSpec instances.

    - Orders (dimension permutations over the *bin grid*) live here.

    Within each aligned bin, we union groups from all specs
    and hard-threshold parameters in-place (or return masks).
    """

    groups: List[GroupSpec]
    orders: List[Tuple[int, ...]]
    name: Optional[str] = None
    _ref_order: Tuple[int] = field(init=False)
    _ref_group_grid_shape: Tuple[int, ...] = field(init=False)
    _reverse_orders: List[Tuple[int, ...]] = field(init=False)

    @property
    def num_blocks(self) -> int:
        return sum([s.group_numel for s in self.groups])

    @property
    def params(self) -> Set[Parameter]:
        """Expose underlying parameters for optimizer integration."""
        return {p for g in self.groups for p in g.block.parameters()}

    def specs(self) -> Iterable[BlockSpec]:
        return [s for g in self.groups for s in g.specs()]

    @property
    def data(self) -> Mapping[BlockSpec, Tensor]:
        merged: Dict[BlockSpec, Tensor] = {}
        for g in self.groups:
            merged.update(g.data)
        return merged

    def numel(self) -> int:
        return sum([g.block.numel() for g in self.groups])

    @cached_property
    def _grid_plan(self) -> GridAxesPlan:
        return GridAxesPlan(self.group_grid_shape_full)

    @property
    def group_grid_shape_full(self) -> Tuple[int, ...]:
        return self._ref_group_grid_shape

    @property
    def group_numel(self) -> int:
        return sum(g.group_numel for g in self.groups)

    def nnz(self, eps=1e-8) -> int:
        return sum(g.nnz(eps=eps) for g in self.groups)

    def __post_init__(self):
        if not self.orders:
            self.orders = [tuple(range(g.group_grid_ndim_full)) for g in self.groups]
        if len(self.orders) != len(self.groups):
            raise ValueError("orders must match number of specs.")

        self.orders = [
            normalize_order(o, g.group_grid_ndim_full)
            for o, g in zip(self.orders, self.groups)
        ]

        self._ref_order = self.orders[0]  # type: ignore
        self._ref_group_grid_shape = ref_permute = tuple(  # type: ignore
            self.groups[0].group_grid_shape_full[i] for i in self._ref_order
        )

        self._reverse_orders = []

        for g, o in zip(self.groups, self.orders):
            Gi = g.group_grid_shape_full
            gperm = tuple(Gi[i] for i in o)
            if gperm != ref_permute:
                raise CouplingError(
                    "Incompatible grouped shapes "
                    f"after order: {gperm} vs {ref_permute} "
                    f"(spec {g.name or '<unnamed>'})"
                )
            self._reverse_orders.append(inverse_permutation(o))

    def grouped_block_norms(self, values: Values):
        # Each GroupSpec returns a compact group-grid prefix. For coupling we
        # must permute in full coordinates to preserve axis identity, then
        # return a compact prefix again.
        chunks_full = []
        for o, g in zip(self.orders, self.groups):
            scores_c = g.grouped_block_norms(values)
            scores_f = g._grid_plan.compact_to_full_prefix(scores_c)
            chunks_full.append(scores_f.permute(*o, len(o)))
        concatenated_full = torch.cat(chunks_full, dim=-1)
        # Return compact prefix for downstream selection/topk.
        return self._grid_plan.full_to_compact_prefix(concatenated_full)

    def kth_largest(
        self,
        k: int,
        values: Values,
    ) -> Tensor:
        """
        Calculates the k-th largest score across all groups from all specs.
        This is used to determine the threshold for pruning.
        """
        grouped_scores = self.grouped_block_norms(values)

        return kth_largest(grouped_scores, k=k, dim=-1)

    @torch.no_grad()
    def hard_threshold(
        self,
        thresholds: Optional[Tensor] = None,
        num_nz: Optional[int] = None,
        values: Values = None,
        sparsity: Optional[float] = None,
    ):
        """Compute kappa-largest block_norm among coupled groups from
        all specs then sends the threshold to specs to hard-threshold in-place.
        Note that the threshold is across coupled-groups, so some parameters
        might be pruned more than others (it's expected).
        """

        if thresholds is None:
            if num_nz is None:
                if sparsity is None:
                    raise ValueError(
                        "Either group_thresholds or kappa or sparsity should be provided"
                    )
                else:
                    num_nz = self.group_numel - int(sparsity * self.group_numel)

            if num_nz == self.group_numel:
                return

            thresholds = self.kth_largest(k=num_nz, values=values)

        assert thresholds.shape == self.group_grid_shape
        thr_full = self._grid_plan.compact_to_full(thresholds)
        for ro, g in zip(self._reverse_orders, self.groups):
            t_g_full = thr_full.permute(*ro)
            t_g = g._grid_plan.full_to_compact(t_g_full)
            g.hard_threshold(thresholds=t_g)

    @torch.no_grad()
    def soft_threshold(
        self,
        thresholds: Tensor,
        conditioners: Mapping[BlockSpec, Tensor],
        scale=False,
        max_iter=20,
        eps=1e-8,
    ):
        """
        Performs soft thresholding on all coupled parameters.
        """
        assert thresholds.shape == self.group_grid_shape
        thr_full = self._grid_plan.compact_to_full(thresholds)
        for ro, g in zip(self._reverse_orders, self.groups):
            t_g = g._grid_plan.full_to_compact(thr_full.permute(*ro))
            g.soft_threshold(
                t_g,
                conditioners=conditioners,
                scale=scale,
                max_iter=max_iter,
                eps=eps,
            )

    @torch.no_grad()
    def get_masks(
        self,
        num_nz: int,
        grouped_block_scores: Tensor | None = None,
        values: Values = None,
        **kwargs,
    ) -> Mapping[BlockSpec, Tensor]:
        if grouped_block_scores is None:
            grouped_block_scores = self.grouped_block_norms(values)
        else:
            assert grouped_block_scores.shape == self.group_grid_shape + (
                self.group_numel,
            )

        indices = torch.topk(grouped_block_scores, k=num_nz, dim=-1)[1]

        grouped_mask = torch.zeros_like(grouped_block_scores).bool()
        grouped_mask.scatter_(-1, indices, True)

        spec_masks = {}
        slice_start = 0
        for ro, g in zip(self._reverse_orders, self.groups):
            group_slice = grouped_mask[
                ..., slice_start : slice_start + g.group_numel
            ]
            # Promote to full prefix, permute, then compact for that group.
            group_slice_full = self._grid_plan.compact_to_full_prefix(group_slice)
            group_slice_g_full = group_slice_full.permute(*ro, len(ro))
            group_slice_g = g._grid_plan.full_to_compact_prefix(group_slice_g_full)
            spec_masks.update(g.get_masks(num_nz=0, grouped_mask=group_slice_g))
            slice_start += g.group_numel

        return spec_masks

    def __hash__(self):
        return hash(tuple(hash(g) for g in self.groups))

    def __repr__(self):
        return f"GroupCoupling(orders={self.orders}, {', '.join([str(s) for s in self.groups])})"


if __name__ == "__main__":
    from bonsainet.blocks import BlockSpec

    torch.manual_seed(0)
    U = torch.nn.Parameter(torch.randn(4, 8, 2, 2, device="cuda"))
    V = torch.nn.Parameter(torch.randn(8, 16, 2, 2, device="cuda"))

    block_u = BlockSpec(U, block_shape=(2, 2, 2, 2), name="U")
    group_u = GroupSpec(block_u, group_shape=(1, 1))

    block_v = BlockSpec(V, block_shape=(2, 2, 2, 2), name="V")
    group_v = GroupSpec(block_v, group_shape=(1, 4))

    print(group_u)
    print(group_v)

    coupled = GroupCoupling([group_u, group_v], orders=[(0, 1), (1, 0)])
    masks = coupled.hard_threshold(num_nz=2)
    # print(U)
    # print(V)
    # print(masks.squeeze())
    print(block_u.block_norms(None))
    print(block_v.block_norms(None))

    # U = torch.nn.Parameter(torch.randn(4, 8, device="cuda"))
    # V = torch.nn.Parameter(torch.randn(8, 16, device="cuda"))
    # print(U)
    # print(V)

    # group_u = GroupSpec(U, block_s=(2, 2), group_size=(1, 1), name="U")
    # group_v = GroupSpec(V, block_size=(2, 2), group_size=(1, 4), name="V")
    # print(group_u)
    # print(group_v)

    # coupled = GroupCoupling(
    #     [group_u, group_v], orders=[(0, 1), (1, 0)], sparsity=0.5
    # )

    # masks = coupled.hard_threshold(kappa=2)
    # print(masks.squeeze())
    # print(U)
    # print(V)
