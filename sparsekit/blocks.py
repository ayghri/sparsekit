"""
Copyright (c) 2025 Ayoub Ghriss and contributors
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.

Classes
-------
SparseNode : abstract base class that knows how to view / threshold
BlockSpec   : concrete implementation that treats the whole tensor as a single
              grid of blocks (the most common case).
BlockCoupling  : concrete implementation that merges multiple BlockSpec into one
"""

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
from torch.nn import Parameter
import torch

from .utils import interleave_unsqueeze
from .utils import merge_odd_dims, append_odd_dims
from .utils import inverse_permutation, normalize_order
from .utils import ShapeMismatchError
from .utils import Values


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

    @cached_property
    def rank(self)->int:
        return len(self.shape)

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

    @cached_property
    @abstractmethod
    def grid_shape(self) -> Tuple[int, ...]:
        """Shape of the block grid (number of blocks per dimension)."""
        pass

    @property
    def block_grid_ndim(self) -> int:
        """Number of dimensions in the block grid."""
        return len(self.grid_shape)

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
    def block_view(self, values: Values, reorder=True, merge=False) -> Tensor:
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
        return torch.linalg.vector_norm(
            t, ord=p, dim=self._reduction_dim, keepdim=keepdim
        )

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
        assert tuple(block_thresholds.shape) == self.grid_shape

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
        if tuple(thresholds.shape) != self.grid_shape:
            raise ValueError(
                f"thresholds shape {thresholds.shape} must match "
                f"block_grid_size {self.grid_shape}"
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
    def grid_shape(self) -> Tuple[int, ...]:
        """Full grid shape including singleton dimensions."""
        return tuple(si // bi for si, bi in zip(self.shape, self.block_shape))

    # @cached_property
    # def block_grid_shape(self) -> Tuple[int, ...]:
    #     """
    #     Same as ``_grid_shape`` but removes dimensions that are 1.
    #     This is the shape used by the thresholding logic.
    #     """
    #     shape = tuple(s for s in self._grid_shape if s > 1)
    #     if len(shape) == 0:
    #         return (1,)
    #     return shape

    @cached_property
    def block_numel(self) -> int:
        """Number of elements per block."""
        return math.prod(self.block_shape)

    @cached_property
    def num_blocks(self) -> int:
        """Total number of blocks in the grid."""
        return math.prod(self.grid_shape)

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

    def block_view(
        self, values: Values, reorder: bool = True, merge=False
    ) -> Tensor:
        """Reshape tensor to interleaved block view.

        Args:
            t: Input tensor matching self.shape.
            merge: If True, collapse block dims to trailing dim.

        Returns:
        """

        t = self._resolve_values(values)
        assert t.shape == self.shape
        interleaved_shape = []
        for B, bi in zip(self.grid_shape, self.block_shape):
            interleaved_shape.extend([B, bi])

        view = t.view(*interleaved_shape)
        if reorder or merge:
            view = append_odd_dims(view)
            if merge:
                view = merge_odd_dims(view)

        return view

    def expand_block_tensor(self, block_values: Tensor) -> Tensor:
        #     """Convert grid-shaped tensor to full grid shape with singletons.

        #     Args:
        #         block_values: Tensor with shape block_grid_shape.

        #     Returns:
        #         Tensor reshaped to _grid_shape.
        #     """
        return block_values.view(self.grid_shape)

    # def block_view(self, values: Values, merge=False) -> Tensor:
    #     """Return a blocked view of values.

    #     Args:
    #         values: Input values (None uses param.data).
    #         # squeeze: If True, remove singleton grid dimensions.

    #     Returns:
    #         Tensor with shape (B1, B2, ..., block_numel) if squeeze=True.
    #     """
    #     t = self._resolve_values(values)
    #     view = self._raw_block_view(t, reorder=merge)
    #     # if squeeze:
    #     #     view = view.view(*self.block_grid_shape, -1)
    #     return view

    def broadcast_block_to_element(
        self, block_values: Tensor, fake=False
    ) -> Tensor:
        """Broadcast block grid-shaped tensor to full tensor shape.

        Args:
            block_values: Tensor with shape block_grid_shape.
            fake: If True, only unsqueeze without repeating.

        Returns:
            Tensor with shape self.shape (or interleaved if fake=True).
        """
        assert tuple(block_values.shape) == self.grid_shape

        casted_tensor = block_values

        for i, bi in enumerate(self.block_shape):
            casted_tensor = casted_tensor.unsqueeze(2 * i + 1)
            if not fake:
                casted_tensor = casted_tensor.repeat_interleave(
                    bi, dim=2 * i + 1
                )
        if not fake:
            casted_tensor = casted_tensor.reshape(*self.shape)
        return casted_tensor

    def apply_mask(self, mask: Tensor):
        """Zero out blocks where mask is True."""
        self.apply_multiplier(~mask)

    def apply_multiplier(self, multiplier: Tensor):
        """Multiply each block by corresponding scalar in multiplier."""
        assert multiplier.shape == self.grid_shape

        # Shape (B1, B2, B3,...)
        multiplier = self.expand_block_tensor(multiplier)

        # Shape (B1,1, B2, 1, B3, 1, ...)
        multiplier = interleave_unsqueeze(multiplier)

        # Shape (B1, b1, B2, b2,....)
        b_view = self.block_view(self.param.data, reorder=False)
        b_view.mul_(multiplier)

    def block_reduce(
        self, values: Values, reduce_fn: Callable[[Tensor], Tensor]
    ) -> Tensor:
        """Apply reduce_fn over each block and return grid-shaped result."""
        t = self._resolve_values(values)
        t = self._raw_block_view(t, reorder=False)
        return reduce_fn(t).view(self.grid_shape)

    def _soft_threshold_euclid(self, block_thresholds, eps=1e-8):
        """In-place Euclidean (L2) proximal step."""

        assert tuple(block_thresholds.shape) == self.grid_shape

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
        assert block_thresholds.shape == self.grid_shape
        assert conditioners is not None
        conditioner = self._resolve_values(conditioners)

        assert isinstance(conditioner, Tensor)
        assert conditioner.shape == self.shape

        if self.block_numel == 1:
            return self._soft_threshold_euclid(
                block_thresholds / conditioner.view(self.grid_shape)
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
        blocked_conditioner = self._raw_block_view(conditioner, reorder=False)
        blocked_Hv = self._raw_block_view(Hv, reorder=False)

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
            + self.broadcast_block_to_element(mu.view(self.grid_shape))
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
        block_masks = self.broadcast_block_to_element(block_masks)
        return {self: block_masks}

    def __repr__(self) -> str:
        """Return string representation with shape information."""
        return (
            f"{self.__class__.__name__}"
            f"(shape={self.shape}, block_shape={self.block_shape}, "
            f"block_grid_shape={self.grid_shape}, name={self.name!r})"
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

    def block_specs(self) -> Iterable["BlockSpec"]:
        """Return the list of coupled BlockSpec objects."""
        return self.specs

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

        self._ref_order = self.orders[0]  # type: ignore
        self._ref_block_grid_shape = ref_permute = tuple(  # type: ignore
            self.specs[0].grid_shape[i] for i in self._ref_order
        )

        self._reverse_orders = []
        for s, o in zip(self.specs, self.orders):
            Bis = s.grid_shape
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
    def grid_shape(self) -> Tuple[int, ...]:
        """Reference block grid shape for the coupling."""
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
            return {s: s.param for s in self.specs}
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
            values.append(s.block_view(spec_values[s]).permute(*o, len(o)))
        return torch.concat(values, dim=-1)

    def block_reduce(
        self, values: Values, reduce_fn: Callable[[Tensor], Tensor]
    ) -> Tensor:
        """Apply reduce_fn over concatenated block view."""
        spec_values = self._resolve_values(values)
        concat_values = self._raw_block_view(spec_values)
        return reduce_fn(concat_values)

    def _soft_threshold_euclid(self, block_thresholds, eps=1e-8):
        """In-place Euclidean (L2) proximal step."""

        assert tuple(block_thresholds.shape) == self.grid_shape

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
        assert tuple(block_thresholds.shape) == self.grid_shape
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

            weighted_vs = {
                s: Hv[s]
                / (
                    conditioners[s]
                    + s.broadcast_block_to_element(mu.permute(ro).contiguous())
                )
                for ro, s in zip(self._reverse_orders, self.specs)
            }
            weighted_norm = self.block_norms(weighted_vs)
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
                s.param.data
                * conditioners[s]
                / (
                    conditioners[s]
                    + s.broadcast_block_to_element(mu.permute(o))
                )
            )

        # only keep survivors
        self.apply_mask(non_survivors)

    def get_masks(self, block_masks: Tensor) -> Mapping["BlockSpec", Tensor]:
        """Convert block-level mask to element-level masks for each spec."""
        spec_masks = {}
        for ro, s in zip(self._reverse_orders, self.specs):
            spec_masks.update(s.get_masks(block_masks.permute(ro).contiguous()))
        return spec_masks

    def apply_mask(self, mask: Tensor):
        """Zero out blocks where mask is True across all specs."""
        self.apply_multiplier(~mask)

    def apply_multiplier(self, multiplier: Tensor):
        """Multiply each block by corresponding scalar across all specs."""
        assert multiplier.shape == self.grid_shape, "Incompatible Multiplier"
        for ro, s in zip(self._reverse_orders, self.specs):
            s.apply_multiplier(multiplier.permute(ro))

    def __repr__(self) -> str:
        """Return string representation with specs info."""
        specs_str = ",\n\t".join(str(s) for s in self.specs)
        return (
            f"{self.__class__.__name__}"
            f"(block_grid_shape={self.grid_shape}, ref_order={self._ref_order}, "
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
