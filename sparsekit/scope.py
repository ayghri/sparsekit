# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0
# (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Scope-level sparsity specification over block grid."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Set, Mapping, Iterable
from functools import cached_property, lru_cache

import math

from torch import Tensor
import torch
from abc import abstractmethod, ABC


from .view import View
from .block import SparseBlock

from .tensor_ops import merge_odd_dims, append_odd_dims
from .tensor_ops import normalize_order
from .tensor_ops import unmerge_odd_dims
from .tensor_ops import inverse_permutation
from .tensor_ops import kth_largest
from .tensor_ops import mid_kth_largest
from .tensor_ops import get_dtype_epsilon
from .block import Values
from .block import CouplingError


class SparseScope(ABC):
    """Abstract base class for scope-level sparsity specifications."""

    @lru_cache()
    def numscp(self) -> int:
        return math.prod(self.grid_shape)

    @lru_cache()
    def numblk(self) -> int:
        return self.numscp() * self.numblk_per_scope()

    @lru_cache()
    @abstractmethod
    def numblk_per_scope(self) -> int:
        """Number of blocks per scope."""
        pass

    @lru_cache()
    @abstractmethod
    def numel_per_scope(self) -> int:
        """Number of blocks per scope."""
        pass

    @abstractmethod
    def nnz(self, eps=1e-8) -> int:
        pass

    @cached_property
    @abstractmethod
    def grid_shape(self) -> Tuple[int, ...]:
        """Shape of the scope grid (number of scopes per dimension)."""
        pass

    @abstractmethod
    def blocks(self) -> Iterable[SparseBlock]:
        pass

    # @property
    # @abstractmethod
    # def data(self) -> Mapping[SparseBlock, Tensor] | Tensor:
    #     """Raw tensor data of the underlying parameter(s)."""
    #     pass

    @abstractmethod
    @torch.no_grad()
    def hard_threshold(
        self,
        thresholds: Optional[Tensor] = None,
        nnz: Optional[int] = None,
        values: Values = None,
    ):
        """
        Zeros out blocks in-place based on scope-level thresholds.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def soft_threshold(
        self,
        thresholds: Tensor,
        conditioners: Values = None,
        scale: bool = False,
        max_iter: int = 20,
        atol: float = 1e-8,
        eps: float | None = None,
    ) -> None:
        pass

    @abstractmethod
    @torch.no_grad()
    def get_masks(
        self,
        nnz: int,
        values: Values = None,
        block_scores: Tensor | None = None,
        block_mask: Tensor | None = None,
        **kwargs,
    ) -> Mapping[SparseBlock, Tensor]:
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
class ScopeSpec(SparseScope):
    """Organizes blocks from a SparseNode into scopes (decision units).

    Divides the block grid into scopes of ``shape`` blocks.
    Pruning decisions (hard/soft threshold, mask selection) operate at the
    scope level: within each scope, blocks compete to survive.

    Args:
        block: SparseBlock or BlockCoupling that defines the block grid.
            Must have ``grid_shape`` divisible by ``shape``.
        shape: Number of blocks of scope in each dimension.
            Use -1 to span the entire dimension.
        name: Optional name for identification.
    """

    block: SparseBlock
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
                f"scope shape {self.shape} has len {len(self.shape)} "
                f"but grid_shape = {self.block.grid_shape}D"
            )
        self.shape = tuple(
            [
                self.block.grid_shape[i] if gi == -1 else gi
                for i, gi in enumerate(self.shape)
            ]
        )

        for i, (block_idx, gi) in enumerate(
            zip(self.block.grid_shape, self.shape)
        ):
            if block_idx % gi != 0:
                raise ValueError(
                    f"dim {i}: block_grid[{i}]={block_idx} "
                    f"not divisible by "
                    f"block_size[{i}]={gi}"
                )

    @cached_property
    def grid_shape(self) -> Tuple[int, ...]:
        """Full grid shape including singleton dimensions."""
        return tuple(
            block_idx // gi
            for block_idx, gi in zip(self.block.grid_shape, self.shape)
        )

    # @cached_property
    # def scope_numel(self) -> int:
    #     """Total number of elements across all scopes."""
    #     return math.prod(self.grid_shape)

    @lru_cache()
    def numel_per_scope(self) -> int:
        """Number of blocks per scope."""
        return math.prod(self.shape)

    @lru_cache()
    def numblk_per_scope(self) -> int:
        """Number of blocks per scope."""
        return math.prod(self.shape)

    def blocks(self) -> Iterable[SparseBlock]:
        return [self.block]

    def nnz(self, eps=1e-8) -> int:
        return self.block.nnz(eps=eps)

    def block_to_scope(
        self, b: Tensor, reorder: bool = True, merge: bool = False
    ) -> Tensor:
        """Reshape a block-grid tensor into scope layout.

        Args:
            b: Tensor with shape ``(B1, B2, ..., Bm, ...)``.
            reorder: If True, permute so scope dims precede block dims.
            merge: If True, collapse block dims into a single trailing dim.

        Returns:
            Tensor with shape ``(G1, G2, ..., g1, g2, ..., ...)`` (or merged).
        """
        assert b.shape[: len(self.block.grid_shape)] == self.block.grid_shape
        inter_shape = [
            grid_idx
            for pair in zip(self.grid_shape, self.shape)
            for grid_idx in pair
        ]
        view = b.view(*inter_shape)
        if reorder or merge:
            view = append_odd_dims(view)
            if merge:
                view = merge_odd_dims(view)
        return view

    def scope_to_block(self, block_values: Tensor) -> Tensor:
        """Broadcast scope-level values back to the block grid.

        Args:
            block_values: Tensor with shape ``(G1, ..., Gm)``.

        Returns:
            Tensor with shape ``(B1, ..., Bm)`` where ``Bi = Gi * gi``.
        """
        assert tuple(block_values.shape) == self.grid_shape
        inter_values = block_values.view(self.grid_shape)
        for i, gi in enumerate(self.shape):  # type: ignore
            inter_values = inter_values.unsqueeze(2 * i + 1)
            inter_values = inter_values.repeat_interleave(gi, dim=2 * i + 1)
        inter_values = inter_values.view(self.block.grid_shape)
        return inter_values

    def block_norms(self, values: Values) -> Tensor:
        """Compute block L2 norms arranged in scope layout.

        Args:
            values: Element values to compute norms from (None uses param data).

        Returns:
            Tensor with shape ``(*grid_shape, block_numel)``.
        """
        block_norms = self.block.norms(values)
        block_norms = self.block_to_scope(block_norms, reorder=False)
        merged = merge_odd_dims(block_norms)
        return merged

    def kth_largest(self, element_values: Values, nnz: int) -> Tensor:
        """
        Calculates the k-th largest score across all blocks from all specs.
        This is used to determine the threshold for pruning.
        """
        block_scores = self.block_norms(element_values)
        top_scores = kth_largest(block_scores, k=nnz, dim=-1)
        top_scores = top_scores.view(self.grid_shape)
        return top_scores

    def kth_mid(self, element_values: Values, nnz: int, k_weight=1.0) -> Tensor:
        """
        Calculates the k-th largest score across all blocks from all specs.
        This is used to determine the threshold for pruning.
        """
        block_scores = self.block_norms(element_values)
        # top_scores = kth_largest(block_scores, k=nnz, dim=-1)
        top_scores = mid_kth_largest(
            block_scores, k=nnz, dim=-1, k_weight=k_weight
        )
        top_scores = top_scores.view(self.grid_shape)
        return top_scores

    @torch.no_grad()
    def hard_threshold(
        self,
        thresholds: Optional[Tensor] = None,
        nnz: Optional[int] = None,
        values: Values = None,
        # sparsity: Optional[float] = None,
    ):
        """Zero out blocks in-place based on thresholds.

        Exactly one of ``thresholds``, ``nnz``, or
        ``sparsity`` must be given.

        Args:
            thresholds: Per-block thresholds, shape
                ``grid_shape``.
            nnz: Number of blocks to keep per scope.
            values: Element values for computing norms.
            sparsity: Fraction to prune (0.5 = 50%).
        """
        if thresholds is None:
            if nnz is None:
                raise ValueError("One of{thresholds, nnz} should be provided")

            if nnz == self.numblk_per_scope():
                return

            thresholds = self.kth_largest(values, nnz=nnz)

        assert thresholds.shape == self.grid_shape

        block_thresholds = self.scope_to_block(thresholds)
        self.block.hard_threshold(block_thresholds, values=values)

    def sparsity_to_nnz(self, sparsity: float):
        return max(
            self.numblk_per_scope() - int(sparsity * self.numblk_per_scope()),
            0,
        )

    @torch.no_grad()
    def get_masks(
        self,
        nnz: int,
        values: Values = None,
        block_scores: Tensor | None = None,
        block_mask: Tensor | None = None,
        **kwargs,
    ) -> Mapping[SparseBlock, Tensor]:
        """Compute element-level boolean masks from scope-level scores.

        Args:
            nnz: Number of blocks to keep per scope.
            block_scores: Pre-computed scores with shape
                ``(*grid_shape, block_numel)``.
            values: Element values for computing
                block norms (if scores not given).
            block_mask: Pre-computed boolean mask to use directly.

        Returns:
            Dict mapping each SparseBlock to its element-level boolean mask.
        """
        if block_mask is None:
            if block_scores is None:
                block_scores = self.block_norms(values)
            else:
                assert block_scores.shape == self.grid_shape + (
                    self.numblk_per_scope(),
                )

            thresholds = kth_largest(block_scores, k=nnz, dim=-1).unsqueeze(-1)
            # indices = torch.topk(block_scores, k=nnz, dim=-1)[1]
            # block_mask = torch.zeros_like(block_scores).bool()
            # block_mask.scatter_(-1, indices, True)
            block_mask = block_scores >= thresholds

        block_mask = unmerge_odd_dims(
            block_mask.view(self.grid_shape + (self.numblk_per_scope(),)),
            self.shape,
        )

        block_mask = block_mask.view(self.block.grid_shape)

        return self.block.get_masks(block_mask)

    @torch.no_grad()
    def soft_threshold(
        self,
        thresholds: Tensor,
        conditioners: Values = None,
        scale: bool = False,
        max_iter: int = 20,
        atol: float = 1e-8,
        eps: float | None = None,
    ) -> None:
        """Apply soft thresholding (L1 proximal operator) to scopes in-place.

        Args:
            thresholds: Per-scope threshold values
                with shape ``grid_shape``.
            conditioners: Diagonal preconditioner per SparseBlock.
            scale: If True, scale thresholds by sqrt(block_numel).
            max_iter: Maximum bisection iterations for Adam variant.
            eps: Small constant for numerical stability.
            atol: Absolute tolerance for convergence.
        """
        assert tuple(thresholds.shape) == self.grid_shape

        eps = get_dtype_epsilon(thresholds.dtype, eps)
        block_thresholds = self.scope_to_block(thresholds)
        if scale:
            block_thresholds = block_thresholds * (self.block.numel() ** 0.5)

        self.block.soft_threshold(
            block_thresholds,
            conditioners=conditioners,
            max_iter=max_iter,
            atol=atol,
            eps=eps,
        )

    # def apply_mask(self, mask: Tensor) -> None:
    #     assert mask.shape == tuple(self.grid_shape + (self.numblk_per_scope(),))
    #
    def __repr__(self):
        return (
            f"{self.__class__.__name__}[block_shape={self.shape}, "
            f"grid_shape={self.grid_shape}, "
            f"name={self.name}], "
            f"block={self.block}"
        )

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self) -> int:
        return hash((hash(self.block), self.shape))


@dataclass
class ScopeCoupling(SparseScope):
    """Couples multiple ScopeSpec instances for joint pruning.

    Aligns scope grids from different parameters via dimension permutations
    (``orders``) so they share a common grid shape.
    Within each aligned scope, blocks from all specs compete to survive pruning.

    Args:
        scopes: List of ScopeSpec instances to couple.
        orders: Dimension permutations to align each scope's grid.
            Identity permutation if omitted.
        name: Optional name for identification.
    """

    scopes: List[ScopeSpec]
    orders: List[Tuple[int, ...]]
    name: Optional[str] = None
    _ref_order: Tuple[int] = field(init=False)
    _ref_scope_grid_shape: Tuple[int, ...] = field(init=False)
    _reverse_orders: List[Tuple[int, ...]] = field(init=False)

    def __post_init__(self):
        if not self.orders:
            self.orders = [tuple(range(len(g.grid_shape))) for g in self.scopes]
        if len(self.orders) != len(self.scopes):
            raise ValueError("orders must match number of specs.")

        self.orders = [
            normalize_order(o, len(g.grid_shape))
            for o, g in zip(self.orders, self.scopes)
        ]

        self._ref_order = self.orders[0]  # type: ignore
        self._ref_scope_grid_shape = ref_permute = tuple(  # type: ignore
            self.scopes[0].grid_shape[i] for i in self._ref_order
        )

        self._reverse_orders = []

        for g, o in zip(self.scopes, self.orders):
            gperm = tuple(g.grid_shape[i] for i in o)
            if gperm != ref_permute:
                s_name = g.name or "<unnamed"
                raise CouplingError(
                    "Incompatible scope shapes "
                    f"after order: {gperm} vs {ref_permute} (spec {s_name})"
                )
            self._reverse_orders.append(inverse_permutation(o))

    # ── Properties ────────────────────────────────────────────────────

    @lru_cache()
    def numblk(self) -> int:
        """Total number of blocks across all coupled scopes."""
        return sum([s.numblk() for s in self.scopes])

    @property
    def params(self) -> Set[View]:
        """Expose underlying views for optimizer integration."""
        return {p for g in self.scopes for p in g.block.parameters()}

    def blocks(self) -> Iterable[SparseBlock]:
        """Merged dict mapping each SparseBlock to its tensor data."""
        merged = []
        for sc in self.scopes:
            merged.extend(sc.blocks())
        return merged

        # merged: Dict[SparseBlock, Tensor] = {}
        # for g in self.scopes:
        #     merged.update(g.data)
        # return merged

    @cached_property
    def grid_shape(self) -> Tuple[int, ...]:
        """Reference scope grid shape (after order permutation)."""
        return self._ref_scope_grid_shape

    @lru_cache()
    def numblk_per_scope(self) -> int:
        """Total blocks per scope across all coupled scopes."""
        return sum(s.numblk_per_scope() for s in self.scopes)

    @lru_cache()
    def numel_per_scope(self) -> int:
        return sum([s.numel_per_scope() for s in self.scopes])

    def nnz(self, eps=1e-8) -> int:
        return sum(g.nnz(eps=eps) for g in self.scopes)

    def specs(self) -> Iterable[SparseBlock]:
        return [s for g in self.scopes for s in g.blocks()]

    def block_norms(self, values: Values) -> Tensor:
        blocked_block_norms = torch.cat(
            [
                g.block_norms(values).permute(o + (len(o),))
                for o, g in zip(self.orders, self.scopes)
            ],
            dim=-1,
        )
        assert blocked_block_norms.shape[:-1] == self.grid_shape
        return blocked_block_norms

    def kth_largest(
        self,
        k: int,
        values: Values,
    ) -> Tensor:
        """
        Calculates the k-th largest score across all blocks from all specs.
        This is used to determine the threshold for pruning.
        """
        blocked_scores = self.block_norms(values)

        return kth_largest(blocked_scores, k=k, dim=-1)

    @torch.no_grad()
    def hard_threshold(
        self,
        thresholds: Optional[Tensor] = None,
        nnz: Optional[int] = None,
        values: Values = None,
    ):
        """Compute kappa-largest block norm among coupled
        scopes from all specs then sends the threshold
        to specs to hard-threshold in-place.
        Note that the threshold is across coupled scopes,
        so some parameters might be pruned more than
        others (it's expected).
        """

        if thresholds is None:
            if nnz is None:
                raise ValueError("Either thresholds or nnz")

            thresholds = self.kth_largest(k=nnz, values=values)

        assert thresholds.shape == self.grid_shape
        for ro, s in zip(self._reverse_orders, self.scopes):
            s.hard_threshold(
                thresholds=thresholds.permute(ro).reshape(s.grid_shape)
            )

    def sparsity_to_nnz(self, sparsity_to_nnz):
        pass
        # nnz = self.nu - int(sparsity * self.block_numel)

    @torch.no_grad()
    def soft_threshold(
        self,
        thresholds: Tensor,
        conditioners: Values = None,
        scale: bool = False,
        max_iter: int = 20,
        atol: float = 1e-8,
        eps: float | None = None,
    ) -> None:
        """Performs soft thresholding on all coupled parameters."""
        assert thresholds.shape == self.grid_shape

        for ro, g in zip(self._reverse_orders, self.scopes):
            g.soft_threshold(
                thresholds.permute(ro).reshape(g.grid_shape),
                conditioners=conditioners,
                scale=scale,
                max_iter=max_iter,
                atol=atol,
                eps=eps,
            )

    @torch.no_grad()
    def get_masks(
        self,
        nnz: int,
        block_scores: Tensor | None = None,
        values: Values = None,
        **kwargs,
    ) -> Mapping[SparseBlock, Tensor]:
        if block_scores is None:
            block_scores = self.block_norms(values)
        else:
            assert block_scores.shape == self.grid_shape + (
                self.numblk_per_scope(),
            )

        indices = torch.topk(block_scores, k=nnz, dim=-1)[1]

        block_mask = torch.zeros_like(block_scores).bool()
        block_mask.scatter_(-1, indices, True)

        spec_masks = {}
        slice_start = 0
        for ro, g in zip(self._reverse_orders, self.scopes):
            block_slice = block_mask[
                ..., slice_start : slice_start + g.numblk_per_scope()
            ]
            spec_masks.update(
                g.get_masks(
                    nnz=0, block_mask=block_slice.permute(ro + (len(ro),))
                )
            )
            slice_start += g.numblk_per_scope()

        return spec_masks

    def __hash__(self):
        return hash(tuple(hash(g) for g in self.scopes))

    def __repr__(self):
        parts = ", ".join(str(s) for s in self.scopes)
        return f"ScopeCoupling(" f"orders={self.orders}, {parts})"
