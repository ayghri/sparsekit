# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0
# (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Group-structured tensor sparsification."""

from dataclasses import dataclass, field
from typing import (
    Optional,
    Tuple,
    List,
    Mapping,
    Iterable,
    Callable,
)
from abc import ABC, abstractmethod
from functools import cached_property

import math
from torch import Tensor
import torch

from .tensor_ops import interleave_unsqueeze
from .tensor_ops import inverse_permutation, normalize_order
from .types import ShapeMismatchError
from .types import Values
from .view import View


def _vector_norm(x, **kwargs):
    """Wrapper around torch.linalg.vector_norm."""
    # pylint: disable=not-callable
    return torch.linalg.vector_norm(x, **kwargs)


@dataclass
class SparseNode(ABC):
    """Abstract base class for group-structured sparse tensors.

    Provides interface for viewing tensors as group grids, computing group
    statistics, and applying soft/hard thresholding operations.
    """

    def _lp_norm_fn(self, t: Tensor, p, keepdim=False) -> Tensor:
        """Compute Lp norm over reduction dimensions."""
        return _vector_norm(t, ord=p, dim=self._reduction_dim, keepdim=keepdim)

    def _min_fn(self, t: Tensor, keepdim=False) -> Tensor:
        """Compute minimum over reduction dimensions."""
        return torch.amin(t, dim=self._reduction_dim, keepdim=keepdim)

    def _max_fn(self, t: Tensor, keepdim=False) -> Tensor:
        """Compute maximum over reduction dimensions."""
        return torch.amax(t, dim=self._reduction_dim, keepdim=keepdim)

    def norms(self, values: Values = None, p: int = 2) -> Tensor:
        """Compute Lp norm for each group."""
        return self.reduce(values, lambda t: self._lp_norm_fn(t, p=p))

    def min(self, values: Values = None) -> Tensor:
        """Compute minimum value for each group."""
        return self.reduce(values, self._min_fn)

    def max(self, values: Values = None) -> Tensor:
        """Compute maximum value for each group."""
        return self.reduce(values, self._max_fn)

    @property
    def grid_ndim(self) -> int:
        """Number of dimensions in the group grid."""
        return len(self.grid_shape)

    @cached_property
    def num_blocks(self) -> int:
        """Total number of groups in the grid."""
        return math.prod(self.grid_shape)

    @abstractmethod
    def numel(self) -> int:
        """Total number of elements in the underlying tensor."""
        pass

    @cached_property
    @abstractmethod
    def grid_shape(self) -> Tuple[int, ...]:
        """Shape of the group grid (number of groups per dimension)."""
        pass

    @abstractmethod
    def parameters(self) -> Iterable[View]:
        """Iterable of Parameter objects managed by this node."""
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

    @cached_property
    @abstractmethod
    def block_numel(self) -> int:
        """Number of elements per group."""
        pass

    @cached_property
    @abstractmethod
    def _reduction_dim(self) -> int | Tuple[int, ...]:
        """Dimension(s) to reduce over when computing group statistics."""
        pass

    @abstractmethod
    def apply_mask(self, mask: Tensor) -> None:
        """Zero out groups where mask is True."""
        pass

    @abstractmethod
    def apply_multiplier(self, multiplier: Tensor):
        """Multiply each group by corresponding scalar in multiplier."""
        pass

    @abstractmethod
    def block_view(self, values: Values, reorder=True, merge=False) -> Tensor:
        """Return a group-structured view of values."""
        pass

    @abstractmethod
    def reduce(
        self, values: Values, reduce_fn: Callable[[Tensor], Tensor]
    ) -> Tensor:
        """Apply reduce_fn over each group and return grid-shaped result."""
        pass

    def _soft_threshold_euclid(self, block_lambdas: Tensor, eps: float = 1e-8):
        """In-place Euclidean (L2) proximal step."""

        assert tuple(block_lambdas.shape) == self.grid_shape

        block_norms = self.norms(self.data)

        block_factor = 1 - block_lambdas / (block_norms + eps)
        block_factor.clamp_(min=0.0)

        self.apply_multiplier(block_factor)

    @abstractmethod
    def _soft_threshold_diag_cond(
        self,
        block_thresholds: Tensor,
        conditioners: Values,
        max_iter: int = 20,
        eps: float = 1e-20,
        atol: float = 1e-8,
    ) -> None:
        """In-place Adam-conditioned proximal step."""
        pass

    @torch.no_grad()
    def soft_threshold(
        self,
        block_thresholds: Tensor,
        conditioners: Values = None,
        scale: bool = False,
        max_iter: int = 20,
        eps: float = 1e-20,
        atol: float = 1e-8,
    ) -> None:
        """Apply soft thresholding to shrink group norms.

        Args:
            block_thresholds: Per-group threshold values.
            conditioners: Optional diagonal conditioner for Adam variant.
            scale: If True, scale thresholds by sqrt(block_numel).
            max_iter: Maximum iterations for Adam variant.
            eps: Small constant for numerical stability.
            atol: Absolute tolerance for convergence.
        """
        assert tuple(block_thresholds.shape) == self.grid_shape

        if scale:
            block_thresholds = block_thresholds * (self.numel() ** 0.5)
        if conditioners is None:
            self._soft_threshold_euclid(block_thresholds, eps=eps)
        else:
            self._soft_threshold_diag_cond(
                block_thresholds=block_thresholds,
                conditioners=conditioners,
                max_iter=max_iter,
                atol=atol,
            )

    @torch.no_grad()
    def hard_threshold(self, thresholds: Tensor, values: Values = None):
        """Zero out groups with values-based norm below threshold.

        Args:
            thresholds: Per-group threshold values.
            values: Optional values to compute norms from; defaults to data.
        """
        if tuple(thresholds.shape) != self.grid_shape:
            raise ValueError(
                f"thresholds shape {thresholds.shape} must match "
                f"block_grid_size {self.grid_shape}"
            )

        blocks_to_mask = self.norms(values) < thresholds

        self.apply_mask(blocks_to_mask)

    @abstractmethod
    def get_masks(self, block_masks: Tensor) -> Mapping["BlockSpec", Tensor]:
        """Convert group-level mask to element-level masks per BlockSpec."""
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
    """Treats the entire tensor as a grid of groups.

    Attributes:
        param: GroupView providing shape, stride, and write-through data access.
        block_shape: Shape of each group in the grid.
        name: Optional name for identification.
    """

    view: View
    shape: Tuple[int, ...]
    name: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize group shape after initialization."""
        self.view = View.from_existing(self.view)

        if len(self.shape) == 0:  # if group size empty, default to 1
            self.shape = tuple([1 for _ in range(self.view.ndim)])

        if len(self.shape) != self.view.ndim:
            raise ValueError(
                f"{self.name} group has len {len(self.shape)}:{self.shape} "
                f"but tensor is {self.view.ndim}D:{self.view.shape}"
            )
        self.shape = tuple(
            [
                (
                    bi if bi > 0 else self.view.shape[i]
                )  # -1 means use the entire dim
                for i, bi in enumerate(self.shape)
            ]
        )
        for i, (si, bi) in enumerate(zip(self.view.shape, self.shape)):
            if si % bi != 0:
                raise ValueError(
                    f"dim {i}: size {si} not divisible by block_shape[{i}]={bi}"
                )

    @property
    def ndim(self) -> int:
        """Number of dimensions in the underlying tensor."""
        return self.view.ndim

    @property
    def data(self) -> Tensor:
        """Raw tensor data of the underlying Parameter."""
        return self.view.data

    @cached_property
    def grid_shape(self) -> Tuple[int, ...]:
        """Full grid shape including singleton dimensions."""
        return tuple(si // bi for si, bi in zip(self.view.shape, self.shape))

    @cached_property
    def num_blocks(self) -> int:
        """Total number of groups in the grid."""
        return math.prod(self.grid_shape)

    @cached_property
    def block_numel(self) -> int:
        """Number of elements per group."""
        return math.prod(self.shape)

    @cached_property
    def _reduction_dim(self) -> Tuple[int, ...]:
        """Odd-indexed dimensions to reduce over for group statistics."""
        return tuple(range(1, 2 * len(self.shape), 2))

    # ── Methods ──────────────────────────────────────────────────────

    def specs(self) -> Iterable["BlockSpec"]:
        """Return self as the only BlockSpec."""
        return [self]

    def numel(self) -> int:
        """Number of elements in the underlying tensor."""
        return math.prod(self.shape)

    def parameters(self) -> List[View]:
        """List containing the single underlying Parameter."""
        return [self.view]

    def set_data(self, data):
        """Copy data into the underlying Parameter tensor."""
        self.view.data.copy_(data)

    def nnz(self, eps=1e-8) -> int:
        """Number of *non-zero* elements (within tolerance)."""
        return int((self.view.data.abs() > eps).sum().item())

    def _resolve_values(self, values: Values) -> Tensor:
        """Resolve values to a view-shaped tensor.

        If values is a raw Tensor, applies the same as_strided view
        as self.view so that group operations see the correct layout.
        """
        if values is None:
            return self.view.data
        if isinstance(values, dict):
            return values[self]
        if isinstance(values, Tensor):
            if tuple(values.shape) == tuple(self.view.shape):
                return values
            if tuple(values.shape) == tuple(self.view.param.shape):
                return torch.as_strided(
                    values.contiguous(), self.view.shape, self.view.stride
                )
            raise ShapeMismatchError(
                tuple(self.view.shape), tuple(values.shape), "values"
            )
        raise ValueError(
            "values has to be None, Tensor or Dict[BlockSpec, Tensor]"
        )

    def block_view(
        self, values: Values, reorder: bool = True, merge=False
    ) -> Tensor:
        """Reshape tensor to interleaved group view.

        Args:
            values: Input values (None uses param.data).
            reorder: If True, permute grid dims before group dims.
            merge: If True, collapse group dims to trailing dim.
        """
        t = self._resolve_values(values)
        return View.block_view_of(t, self.shape, reorder=reorder, merge=merge)

    def expand_block_tensor(self, block_values: Tensor) -> Tensor:
        """Convert grid-shaped tensor to full grid shape with singletons.

        Args:
            block_values: Tensor with shape block_grid_shape.

        Returns:
            Tensor reshaped to grid_shape.
        """
        return block_values.view(self.grid_shape)

    def broadcast_block_to_element(
        self, block_values: Tensor, fake=False
    ) -> Tensor:
        """Broadcast group grid-shaped tensor to full tensor shape.

        Args:
            block_values: Tensor with shape grid_shape.
            fake: If True, only unsqueeze without repeating.

        Returns:
            Tensor with shape self.shape (or interleaved if fake=True).
        """
        assert tuple(block_values.shape) == self.grid_shape
        return View.broadcast_block_to_element(
            block_values, self.shape, fake=fake
        )

    def apply_mask(self, mask: Tensor):
        """Zero out groups where mask is True."""
        self.apply_multiplier(~mask)

    def apply_multiplier(self, multiplier: Tensor):
        """Multiply each group by corresponding scalar in multiplier."""
        assert multiplier.shape == self.grid_shape
        multiplier = self.expand_block_tensor(multiplier)
        if isinstance(self.view, View):
            self.view.apply_multiplier(multiplier, self.shape)
        else:
            multiplier = interleave_unsqueeze(multiplier)
            b_view = View.block_view_of(
                self.view.data, self.shape, reorder=False
            )
            b_view.mul_(multiplier)

    def reduce(
        self, values: Values, reduce_fn: Callable[[Tensor], Tensor]
    ) -> Tensor:
        """Apply reduce_fn over each group and return grid-shaped result."""
        t = self.block_view(values, reorder=False)
        return reduce_fn(t).view(self.grid_shape)

    def _soft_threshold_diag_cond(
        self,
        block_thresholds: Tensor,
        conditioners: Values,
        max_iter=20,
        eps=1e-20,
        atol=1e-8,
    ):
        """In-place Adam-conditioned proximal step via bisection."""
        assert block_thresholds.shape == self.grid_shape
        assert conditioners is not None
        conditioner = self._resolve_values(conditioners)

        assert isinstance(conditioner, Tensor)
        assert conditioner.shape == self.view.shape

        if self.numel() == 1:
            return self._soft_threshold_euclid(
                block_thresholds / conditioner.view(self.grid_shape)
            )

        hess_weighted = conditioner * self.view.data
        hess_weighted_norms = self.norms(hess_weighted)

        denom = hess_weighted_norms - block_thresholds
        non_survivors = denom <= 0.0
        denom.clamp_(min=0.0).add_(eps)

        h_min = self.min(conditioner)
        h_max = self.max(conditioner)

        # block_thresholds > 0
        # if hess_weighted_norms < block_thresholds, then denom<0,
        # so we clamp for safety

        mu_low = (block_thresholds * h_min) / denom
        mu_high = (block_thresholds * h_max) / denom

        # (B1, 1, B2,1,...)
        mu_low = self.broadcast_block_to_element(mu_low, fake=True).clamp_(
            min=0.0
        )
        mu_high = self.broadcast_block_to_element(mu_high, fake=True).clamp_(
            min=0.0
        )

        grouped_thresholds = self.broadcast_block_to_element(
            block_thresholds, fake=True
        )

        # (B1, b1, B2, b1,...)
        grouped_conditioner = self.block_view(conditioner, reorder=False)
        grouped_hess_weighted = self.block_view(hess_weighted, reorder=False)

        mu = (mu_low + mu_high) / 2
        for _ in range(max_iter):
            # Compute Zeta(mu)
            # scaling = H_group / (H_group + mu)

            # ||H / (H+mu) v||
            # (B1, 1, B2,1,...)
            weighted_norm = self._lp_norm_fn(
                grouped_hess_weighted / (grouped_conditioner + mu),
                p=2,
                keepdim=True,
            )
            # zeta = mu * ||weighted_v||
            zeta = mu * weighted_norm

            # Zeta is strictly increasing with mu.
            # If zeta < threshold, mu is too small -> low = mu
            # If zeta > threshold, mu is too big -> high = mu
            mask_low = zeta < grouped_thresholds
            mu_low = torch.where(mask_low, mu, mu_low)
            mu_high = torch.where(~mask_low, mu, mu_high)
            mu = (mu_low + mu_high) / 2
            if (mu_low - mu_high).abs().max() < atol:
                break

        scaling = conditioner / (
            conditioner
            + self.broadcast_block_to_element(mu.view(self.grid_shape))
        )

        self.set_data(scaling * self.view.data)

        # only keep survivors
        self.apply_mask(non_survivors)

    def get_masks(self, block_masks: Tensor) -> Mapping["BlockSpec", Tensor]:
        """Convert group-level mask to element-level mask.

        Args:
            block_masks: Boolean tensor with shape block_grid_shape.

        Returns:
            Dict mapping self to the broadcasted element mask.
        """
        block_masks = self.broadcast_block_to_element(block_masks)
        return {self: block_masks}

    def __repr__(self) -> str:
        """Return string representation with shape information."""
        return (
            f"{self.__class__.__name__}(group shape={self.shape}, "
            f"grid_shape={self.grid_shape}, name={self.name!r})"
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        """Hash based on the underlying Parameter instance."""
        return hash(self.view)


@dataclass
class BlockCoupling(SparseNode):
    """Merges multiple BlockSpec objects into one coupled sparse node.

    Attributes:
        specs: List of BlockSpec objects to couple.
        orders: Axis permutations to align group grids.
        name: Optional name for identification.
    """

    specs: List[BlockSpec]
    orders: List[Tuple[int, ...]]
    name: Optional[str] = None

    _ref_order: Tuple[int] = field(init=False)
    # _ref_block_grid_shape: Tuple[int] = field(init=False)
    _reverse_orders: List[Tuple[int, ...]] = field(init=False)

    def __post_init__(self):
        """Validate and compute axis orderings for all specs."""
        if not self.orders:
            self.orders = [tuple(range(len(s.grid_shape))) for s in self.specs]
        if len(self.orders) != len(self.specs):
            raise ValueError("orders must match number of specs.")

        self.orders = [
            normalize_order(o, len(s.grid_shape))
            for o, s in zip(self.orders, self.specs)
        ]

        ref_permute = tuple(self.specs[0].grid_shape[i] for i in self.orders[0])

        self._reverse_orders = []
        for s, o in zip(self.specs, self.orders):
            gperm = tuple(s.grid_shape[i] for i in o)
            if gperm != ref_permute:
                raise ValueError(
                    "Incompatible group grid shapes "
                    f"after order: {gperm} vs {ref_permute} "
                    f"(spec {s.name or '<unnamed>'})"
                )
            self._reverse_orders.append(inverse_permutation(o))

    # ── Properties ────────────────────────────────────────────────────

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return placeholder shape (-1, -1) for coupled specs."""
        return (-1, -1)

    @property
    def data(self) -> Mapping[BlockSpec, Tensor]:
        """Dict mapping each spec to its tensor data."""
        return {s: s.view.data for s in self.specs}

    @cached_property
    def grid_shape(self) -> Tuple[int, ...]:
        """Grid shape for the coupling (after order permutation)."""
        return tuple(self.specs[0].grid_shape[i] for i in self.orders[0])

    @cached_property
    def num_blocks(self) -> int:
        """Number of groups (same for all specs after alignment)."""
        return self.specs[0].num_blocks

    @cached_property
    def block_numel(self) -> int:
        """Total elements per group across all specs."""
        return sum([s.block_numel for s in self.specs])

    @cached_property
    def _reduction_dim(self):
        """Reduction dimension for group statistics (last dim)."""
        return -1

    # ── Methods ──────────────────────────────────────────────────────

    def numel(self) -> int:
        """Total number of elements across all specs."""
        return sum([s.numel() for s in self.specs])

    def parameters(self) -> List[View]:
        """List of all Parameter objects from coupled specs."""
        return [s.view for s in self.specs]

    def nnz(self, eps=1e-8) -> int:
        """Count non-zero elements across all specs."""
        return sum([s.nnz(eps=eps) for s in self.specs])

    def _resolve_values(self, values: Values) -> Mapping[BlockSpec, Tensor]:
        """Resolve values to a mapping of BlockSpec to Tensor."""
        if values is None:
            return {s: s.view.data for s in self.specs}
        if isinstance(values, dict):
            return {s: values[s] for s in self.specs}
        raise ValueError("values must be Mapping[BlockSpec,Tensor]")

    def _raw_block_view(
        self, spec_values: Mapping[BlockSpec, Tensor]
    ) -> Tensor:
        """Reshape and concatenate all spec values into unified group view.

        Args:
            spec_values: Mapping from BlockSpec to tensor values.

        Returns:
            Concatenated tensor of shape (B0, B1, ..., total_block_numel).
        """
        values = []
        for o, s in zip(self.orders, self.specs):
            merged = s.block_view(spec_values[s], merge=True)
            # merged shape: (*grid_shape, block_numel)
            ndim = len(s.grid_shape)
            values.append(merged.permute(*o, ndim))
        return torch.concat(values, dim=-1)

    def block_view(
        self, values: Values, reorder: bool = True, merge=False
    ) -> Tensor:
        """Return concatenated group view across all specs."""
        spec_values = self._resolve_values(values)
        return self._raw_block_view(spec_values)

    def reduce(
        self, values: Values, reduce_fn: Callable[[Tensor], Tensor]
    ) -> Tensor:
        """Apply reduce_fn over concatenated group view."""
        spec_values = self._resolve_values(values)
        concat_values = self._raw_block_view(spec_values)
        return reduce_fn(concat_values)

    def _soft_threshold_euclid(self, block_lambdas: Tensor, eps: float = 1e-8):
        """In-place Euclidean (L2) proximal step."""

        assert tuple(block_lambdas.shape) == self.grid_shape

        block_norms = self.norms({s: s.view.data for s in self.specs})

        block_factor = 1 - block_lambdas / (block_norms + eps)
        block_factor.clamp_(min=0.0)

        self.apply_multiplier(block_factor)

    def _soft_threshold_diag_cond(
        self,
        block_thresholds: Tensor,
        conditioners: Values,
        max_iter: int = 20,
        eps: float = 1e-20,
        atol: float = 1e-8,
    ) -> None:
        """In-place Adam-conditioned proximal step via bisection."""
        assert tuple(block_thresholds.shape) == self.grid_shape
        assert isinstance(conditioners, Mapping)
        for s in self.specs:
            assert conditioners[s].shape == s.view.shape

        hess_weighted = {s: conditioners[s] * s.view.data for s in self.specs}
        hess_weighted_norms = self.norms(hess_weighted)

        denom = hess_weighted_norms - block_thresholds

        denom = hess_weighted_norms - block_thresholds
        non_survivors = denom <= 0.0
        denom.clamp_(min=0.0)

        h_min = self.min(conditioners)
        h_max = self.max(conditioners)

        mu_low = ((block_thresholds * h_min) / denom).clamp_(min=0.0)
        mu_high = ((block_thresholds * h_max) / denom).clamp_(min=0.0)

        mu = (mu_low + mu_high) / 2
        for _ in range(max_iter):
            # Compute Zeta(mu)
            # scaling = H_group / (H_group + mu)
            # mu.contiguous()

            weighted_vs = {
                s: hess_weighted[s]
                / (
                    conditioners[s]
                    + s.broadcast_block_to_element(
                        mu.permute(ro).reshape(s.grid_shape).contiguous()
                    )
                )
                for ro, s in zip(self._reverse_orders, self.specs)
            }
            weighted_norm = self.norms(weighted_vs)
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

        for o, s in zip(self._reverse_orders, self.specs):
            s.set_data(
                s.view.data
                * conditioners[s]
                / (
                    conditioners[s]
                    + s.broadcast_block_to_element(
                        mu.permute(o).reshape(s.grid_shape)
                    )
                )
            )

        # only keep survivors
        self.apply_mask(non_survivors)

    def get_masks(self, block_masks: Tensor) -> Mapping["BlockSpec", Tensor]:
        """Convert group-level mask to element-level masks for each spec."""
        spec_masks = {}
        for ro, s in zip(self._reverse_orders, self.specs):
            m = block_masks.permute(ro).reshape(s.grid_shape).contiguous()
            spec_masks.update(s.get_masks(m))
        return spec_masks

    def apply_mask(self, mask: Tensor):
        """Zero out groups where mask is True across all specs."""
        self.apply_multiplier(~mask)

    def apply_multiplier(self, multiplier: Tensor):
        """Multiply each group by corresponding scalar across all specs."""
        assert (
            tuple(multiplier.shape) == self.grid_shape
        ), "Incompatible Multiplier"
        for ro, s in zip(self._reverse_orders, self.specs):
            m = multiplier.permute(ro).reshape(s.grid_shape)
            s.apply_multiplier(m)

    def __repr__(self) -> str:
        """Return string representation with specs info."""
        specs_str = ",\n\t".join(str(s) for s in self.specs)
        return (
            f"{self.__class__.__name__}"
            f"(grid_shape={self.grid_shape}, "
            f"name={self.name!r}, "
            f"BlockSpecs=[\n\t{specs_str}])"
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        """Hash based on hashes of all coupled specs."""
        return hash(tuple(hash(s) for s in self.specs))
