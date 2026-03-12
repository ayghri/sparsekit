# Copyright (c) 2025 Anonymous Authors
# Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Group-level sparsity specification over block grids."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Set, Mapping, Iterable
from functools import cached_property

import math

from torch import Tensor
from torch.nn import Parameter
import torch
from abc import abstractmethod, ABC


from .block import SparseNode
from .block import BlockSpec

from .utils import merge_odd_dims, append_odd_dims
from .utils import normalize_order
from .utils import unmerge_odd_dims
from .utils import inverse_permutation
from .utils import CouplingError
from .utils import Values
from .utils import kth_largest


class SparseGroup(ABC):
    """
    Abstract base class for a sparse groups
    """

    @cached_property
    @abstractmethod
    def grid_shape(self) -> Tuple[int, ...]:
        """Shape of the group grid (number of groups per dimension)."""
        pass

    @cached_property
    def num_groups(self) -> int:
        """Total number of groups."""
        return math.prod(self.grid_shape)

    @property
    @abstractmethod
    def group_numel(self) -> int:
        """Number of blocks per group."""
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
        """Raw tensor data of the underlying parameter(s)."""
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
        thresholds: Tensor,
        conditioners: Values,
        scale: bool = False,
        max_iter: int = 20,
        eps: float = 1e-8,
    ) -> None:
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
    """Groups blocks from a SparseNode into sparsification units.

    Divides the block grid into groups of ``group_shape`` blocks.
    Pruning decisions (hard/soft threshold, mask selection) operate at the
    group level: within each group, blocks compete to survive.

    Args:
        block: SparseNode (typically BlockSpec or BlockCoupling) that defines
            the block grid. Must have ``grid_shape`` divisible by ``group_shape``.
        group_shape: Number of blocks per group in each dimension.
            Use -1 to span the entire dimension.
        name: Optional name for identification.
    """

    block: SparseNode
    shape: Tuple[int, ...]
    name: Optional[str] = None

    def __post_init__(self):
        if not self.shape:
            self.shape = tuple(-1 for _ in self.block.grid_shape)

        # Pad with -1 for missing trailing dimensions
        if len(self.shape) < len(self.block.grid_shape):
            self.shape = self.shape + tuple(
                -1 for _ in range(len(self.block.grid_shape) - len(self.shape))
            )

        if len(self.shape) != len(self.block.grid_shape):
            raise ValueError(
                f"group shape {self.shape} has len {len(self.shape)} "
                f"but block_grid_shape = {self.block.grid_shape}D"
            )
        self.shape = tuple(
            [
                self.block.grid_shape[i] if gi == -1 else gi
                for i, gi in enumerate(self.shape)
            ]
        )

        for i, (Bi, gi) in enumerate(zip(self.block.grid_shape, self.shape)):
            if Bi % gi != 0:
                raise ValueError(
                    f"dim {i}: block_grid[{i}]={Bi} "
                    f"not divisible by group_size[{i}]={gi}"
                )

    # ── Properties ────────────────────────────────────────────────────

    @property
    def data(self) -> Mapping[BlockSpec, Tensor]:
        """Dict mapping each BlockSpec to its tensor data."""
        data = self.block.data
        if isinstance(data, Tensor):
            assert isinstance(self.block, BlockSpec)
            return {self.block: data}
        else:
            return data

    @cached_property
    def grid_shape(self) -> Tuple[int, ...]:
        """Full grid shape including singleton dimensions."""
        return tuple(
            Bi // gi for Bi, gi in zip(self.block.grid_shape, self.shape)
        )

    @cached_property
    def numel(self) -> int:
        """Total number of elements across all groups."""
        return math.prod(self.grid_shape)

    @property
    def group_numel(self) -> int:
        """Number of blocks per group."""
        return math.prod(self.shape)

    # ── Methods ──────────────────────────────────────────────────────

    def specs(self) -> Iterable[BlockSpec]:
        return [s for s in self.block.specs()]

    def nnz(self, eps=1e-8) -> int:
        return self.block.nnz(eps=eps)

    def block_to_group(
        self, b: Tensor, reorder: bool = True, merge: bool = False
    ) -> Tensor:
        """Reshape a block-grid tensor into group layout.

        Args:
            b: Tensor with shape ``(B1, B2, ..., Bm, ...)``.
            reorder: If True, permute so group dims precede block dims.
            merge: If True, collapse block dims into a single trailing dim.

        Returns:
            Tensor with shape ``(G1, G2, ..., g1, g2, ..., ...)`` (or merged).
        """
        assert b.shape[: len(self.block.grid_shape)] == self.block.grid_shape
        inter_shape = [
            Gg for pair in zip(self.grid_shape, self.shape) for Gg in pair
        ]
        view = b.view(*inter_shape)
        if reorder or merge:
            view = append_odd_dims(view)
            if merge:
                view = merge_odd_dims(view)
        return view

    def group_to_block(self, group_values: Tensor) -> Tensor:
        """Broadcast group-level values back to the block grid.

        Args:
            group_values: Tensor with shape ``(G1, ..., Gm)``.

        Returns:
            Tensor with shape ``(B1, ..., Bm)`` where ``Bi = Gi * gi``.
        """
        assert tuple(group_values.shape) == self.grid_shape
        inter_values = group_values.view(self.grid_shape)
        for i, gi in enumerate(self.shape):  # type: ignore
            inter_values = inter_values.unsqueeze(2 * i + 1)
            inter_values = inter_values.repeat_interleave(gi, dim=2 * i + 1)
        inter_values = inter_values.view(self.block.grid_shape)
        return inter_values

    def block_norms(self, values: Values) -> Tensor:
        """Compute block L2 norms arranged in group layout.

        Args:
            values: Element values to compute norms from (None uses param data).

        Returns:
            Tensor with shape ``(*grid_shape, group_numel)``.
        """
        block_norms = self.block.norms(values)
        group_norms = self.block_to_group(block_norms, reorder=False)
        merged = merge_odd_dims(group_norms)
        return merged

    def kth_largest(
        self,
        element_values: Mapping[BlockSpec, Tensor] | None,
        num_nz: int,
    ) -> Tensor:
        """
        Calculates the k-th largest score across all groups from all specs.
        This is used to determine the threshold for pruning.
        """
        grouped_block_scores = self.block_norms(element_values)
        top_scores = kth_largest(grouped_block_scores, k=num_nz, dim=-1)
        top_scores = top_scores.view(self.grid_shape)
        return top_scores

    @torch.no_grad()
    def hard_threshold(
        self,
        thresholds: Optional[Tensor] = None,
        num_nz: Optional[int] = None,
        values: Optional[Values] = None,
        sparsity: Optional[float] = None,
    ):
        """Zero out blocks in-place based on group-level thresholds.

        Exactly one of ``thresholds``, ``num_nz``, or ``sparsity`` must be given.

        Args:
            thresholds: Pre-computed per-group thresholds with shape ``grid_shape``.
            num_nz: Number of blocks to keep per group.
            values: Element values for computing block norms.
            sparsity: Fraction of blocks to prune (0.5 = 50% sparse).
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

        assert thresholds.shape == self.grid_shape

        block_thresholds = self.group_to_block(thresholds)
        self.block.hard_threshold(block_thresholds)

    @torch.no_grad()
    def get_masks(
        self,
        num_nz: int,
        grouped_block_scores: Tensor | None = None,
        values: Values = None,
        grouped_mask: Tensor | None = None,
        **kwargs,
    ) -> Mapping[BlockSpec, Tensor]:
        """Compute element-level boolean masks from group-level scores.

        Args:
            num_nz: Number of blocks to keep per group.
            grouped_block_scores: Pre-computed scores with shape
                ``(*grid_shape, group_numel)``.
            values: Element values for computing block norms (if scores not given).
            grouped_mask: Pre-computed boolean mask to use directly.

        Returns:
            Dict mapping each BlockSpec to its element-level boolean mask.
        """
        if grouped_mask is None:
            if grouped_block_scores is None:
                grouped_block_scores = self.block_norms(values)
            else:
                assert grouped_block_scores.shape == self.grid_shape + (
                    self.group_numel,
                )

            indices = torch.topk(grouped_block_scores, k=num_nz, dim=-1)[1]

            grouped_mask = torch.zeros_like(grouped_block_scores).bool()
            grouped_mask.scatter_(-1, indices, True)

        block_mask = unmerge_odd_dims(
            grouped_mask.view(self.grid_shape + (self.group_numel,)),
            self.shape,
        )

        block_mask = block_mask.view(self.block.grid_shape)

        return self.block.get_masks(block_mask)

    @torch.no_grad()
    def soft_threshold(
        self,
        thresholds: Tensor,
        conditioners: Mapping[BlockSpec, Tensor],
        scale: bool = False,
        max_iter: int = 20,
        eps: float = 1e-20,
        atol: float = 1e-8,
    ) -> None:
        """Apply soft thresholding (L1 proximal operator) to blocks in-place.

        Args:
            thresholds: Per-group threshold values with shape ``grid_shape``.
            conditioners: Diagonal preconditioner per BlockSpec.
            scale: If True, scale thresholds by sqrt(block_numel).
            max_iter: Maximum bisection iterations for Adam variant.
            eps: Small constant for numerical stability.
            atol: Absolute tolerance for convergence.
        """
        assert tuple(thresholds.shape) == self.grid_shape

        block_lambdas = self.group_to_block(thresholds)
        if scale:
            block_lambdas = block_lambdas * (self.block.numel()**0.5)

        self.block.soft_threshold(
            block_lambdas,
            conditioners=conditioners,
            max_iter=max_iter,
            eps=eps,
            atol=atol,
        )

    def apply_mask(self, mask: Tensor) -> None:
        assert mask.shape == tuple(self.grid_shape + (self.group_numel,))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}[group_shape={self.shape}, "
            f"grid_shape={self.grid_shape}, "
            f"name={self.name}], "
            f"block={self.block}"
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash((hash(self.block), self.shape))


@dataclass
class GroupCoupling(SparseGroup):
    """Couples multiple GroupSpec instances for joint pruning.

    Aligns group grids from different parameters via dimension permutations
    (``orders``) so they share a common grid shape. Within each aligned group,
    blocks from all specs compete to survive during pruning.

    Args:
        groups: List of GroupSpec instances to couple.
        orders: Dimension permutations to align each group's grid.
            Identity permutation if omitted.
        name: Optional name for identification.
    """

    groups: List[GroupSpec]
    orders: List[Tuple[int, ...]]
    name: Optional[str] = None
    _ref_order: Tuple[int] = field(init=False)
    _ref_group_grid_shape: Tuple[int, ...] = field(init=False)
    _reverse_orders: List[Tuple[int, ...]] = field(init=False)

    def __post_init__(self):
        if not self.orders:
            self.orders = [tuple(range(len(g.grid_shape))) for g in self.groups]
        if len(self.orders) != len(self.groups):
            raise ValueError("orders must match number of specs.")

        self.orders = [
            normalize_order(o, len(g.grid_shape))
            for o, g in zip(self.orders, self.groups)
        ]

        self._ref_order = self.orders[0]  # type: ignore
        self._ref_group_grid_shape = ref_permute = tuple(  # type: ignore
            self.groups[0].grid_shape[i] for i in self._ref_order
        )

        self._reverse_orders = []

        for g, o in zip(self.groups, self.orders):
            gperm = tuple(g.grid_shape[i] for i in o)
            if gperm != ref_permute:
                raise CouplingError(
                    "Incompatible grouped shapes "
                    f"after order: {gperm} vs {ref_permute} "
                    f"(spec {g.name or '<unnamed>'})"
                )
            self._reverse_orders.append(inverse_permutation(o))

    # ── Properties ────────────────────────────────────────────────────

    @property
    def num_blocks(self) -> int:
        """Total number of blocks across all coupled groups."""
        return sum([s.group_numel for s in self.groups])

    @property
    def params(self) -> Set[Parameter]:
        """Expose underlying parameters for optimizer integration."""
        return {p for g in self.groups for p in g.block.parameters()}

    @property
    def data(self) -> Mapping[BlockSpec, Tensor]:
        """Merged dict mapping each BlockSpec to its tensor data."""
        merged: Dict[BlockSpec, Tensor] = {}
        for g in self.groups:
            merged.update(g.data)
        return merged

    @cached_property
    def grid_shape(self) -> Tuple[int, ...]:
        """Reference group grid shape (after order permutation)."""
        return self._ref_group_grid_shape

    @property
    def group_numel(self) -> int:
        """Total blocks per group across all coupled groups."""
        return sum(g.group_numel for g in self.groups)

    # ── Methods ──────────────────────────────────────────────────────

    def specs(self) -> Iterable[BlockSpec]:
        return [s for g in self.groups for s in g.specs()]

    def numel(self) -> int:
        return sum([g.block.numel() for g in self.groups])

    def nnz(self, eps=1e-8) -> int:
        return sum(g.nnz(eps=eps) for g in self.groups)

    def block_norms(self, values: Values) -> Tensor:
        grouped_block_norms = torch.cat(
            [
                g.block_norms(values).permute(o + (len(o),))
                for o, g in zip(self.orders, self.groups)
            ],
            dim=-1,
        )
        assert grouped_block_norms.shape[:-1] == self.grid_shape
        return grouped_block_norms

    def kth_largest(
        self,
        k: int,
        values: Values,
    ) -> Tensor:
        """
        Calculates the k-th largest score across all groups from all specs.
        This is used to determine the threshold for pruning.
        """
        grouped_scores = self.block_norms(values)

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

        assert thresholds.shape == self.grid_shape
        for ro, g in zip(self._reverse_orders, self.groups):
            g.hard_threshold(
                thresholds=thresholds.permute(ro).reshape(g.grid_shape)
            )

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
        assert thresholds.shape == self.grid_shape

        for ro, g in zip(self._reverse_orders, self.groups):
            g.soft_threshold(
                thresholds.permute(ro).reshape(g.grid_shape),
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
            grouped_block_scores = self.block_norms(values)
        else:
            assert grouped_block_scores.shape == self.grid_shape + (
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
            spec_masks.update(
                g.get_masks(
                    num_nz=0, grouped_mask=group_slice.permute(ro + (len(ro),))
                )
            )
            slice_start += g.group_numel

        return spec_masks

    def __hash__(self):
        return hash(tuple(hash(g) for g in self.groups))

    def __repr__(self):
        return f"GroupCoupling(orders={self.orders}, {', '.join([str(s) for s in self.groups])})"
