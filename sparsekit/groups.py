"""
Copyright (c) 2025 Ayoub Ghriss and contributors
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Set, Mapping, Iterable
from functools import cached_property

import math

from torch import Tensor
from torch.nn import Parameter
import torch
from abc import abstractmethod, ABC


from .linalg import kth_largest
from .blocks import SparseNode
from .blocks import BlockSpec

from .utils import merge_odd_dims, append_odd_dims
from .utils import normalize_order
from .utils import unmerge_odd_dims
from .utils import inverse_permutation
from .utils import CouplingError
from .utils import Values


class SparseGroup(ABC):
    """
    Abstract base class for a sparse groups
    """

    @cached_property
    @abstractmethod
    def grid_shape(self) -> Tuple[int, ...]:
        pass

    @cached_property
    def num_groups(self) -> int:
        return math.prod(self.grid_shape)

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
            self.group_shape = tuple(-1 for _ in self.block.grid_shape)

        if len(self.group_shape) != len(self.block.grid_shape):
            raise ValueError(
                f"group shape {self.group_shape} has len {len(self.group_shape)} "
                f"but block_grid_shape = {self.block.grid_shape}D"
            )
        self.group_shape = tuple(
            [
                self.block.grid_shape[i] if gi == -1 else gi
                for i, gi in enumerate(self.group_shape)
            ]
        )

        for i, (Bi, gi) in enumerate(
            zip(self.block.grid_shape, self.group_shape)
        ):
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

    # @property
    @cached_property
    def grid_shape(self) -> Tuple[int, ...]:
        return tuple(
            Bi // gi for Bi, gi in zip(self.block.grid_shape, self.group_shape)
        )

    # @property
    # def group_grid_shape(self) -> Tuple[int, ...]:
    #     shape = tuple(s for s in self._grid_shape if s > 1)
    #     if len(shape) == 0:
    #         shape = (1,)
    #     return shape

    @cached_property
    def numel(self) -> int:
        return math.prod(self.grid_shape)

    def block_to_group(self, b: Tensor, reorder=True, merge=False) -> Tensor:
        """
        Return a grouped view of b.

        b: shape (B1, B2,...,Bm, ?)
        Returns tensor with shape (G1,G2,...,Gm,g1, g1,,...,gm, ?), where Gi=Bi/gi if ``merge=False``

        If ``merge=True`` the block-dimensions are collapsed into the last dim:
        to get (G1,G2,..., g1*g2*...)
        """
        assert b.shape[:self.block.rank] == self.block.grid_shape
        inter_shape = [
            Gg for pair in zip(self.grid_shape, self.group_shape) for Gg in pair
        ]
        view = b.view(*inter_shape)
        if reorder or merge:
            view = append_odd_dims(view)
            if merge:
                view = merge_odd_dims
        return view


    def group_to_block(self, group_values: Tensor) -> Tensor:
        """
        Broadcast a tensor of group values back to a block view.
        group_values: shape(G1,...,Gm)
        Returns tensor of shape (B1,...,Bm) with Bi=Gi*gi
        """
        assert tuple(group_values.shape) == self.group_grid_shape
        inter_values = group_values.view(self._grid_shape)
        for i, gi in enumerate(self.group_shape):  # type: ignore
            # inter_values = inter_values.unsqueeze(2 * i + 1)
            inter_values = inter_values.repeat_interleave(gi, dim=2 * i + 1)
        inter_values = inter_values.view(self.block.grid_shape)
        return inter_values

    def grouped_block_norms(self, values: Values):
        block_norms = self.block.block_norms(values)
        group_norms = self.block_to_group(block_norms)
        return merge_odd_dims(group_norms)

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
        top_scores = top_scores.view(self.group_grid_shape)
        return top_scores

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

        block_mask = unmerge_odd_dims(
            grouped_mask.view(self._grid_shape + (self.group_numel,)),
            self.group_shape,
        )

        block_mask = block_mask.view(self.block.grid_shape)

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

        block_lambdas = self.group_to_block(thresholds)
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
            f"grid_shape={self.group_grid_shape}, "
            f"name={self.name}], "
            f"block={self.block}"
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

    @property
    def group_grid_shape(self) -> Tuple[int, ...]:
        return self.groups[0].group_grid_shape

    @property
    def group_numel(self) -> int:
        return sum(g.group_numel for g in self.groups)

    def nnz(self, eps=1e-8) -> int:
        return sum(g.nnz(eps=eps) for g in self.groups)

    def __post_init__(self):
        if not self.orders:
            self.orders = [
                tuple(range(g.block.block_grid_ndim)) for g in self.groups
            ]
        if len(self.orders) != len(self.groups):
            raise ValueError("orders must match number of specs.")

        self.orders = [
            normalize_order(o, g.block.block_grid_ndim)
            for o, g in zip(self.orders, self.groups)
        ]

        self._ref_order = self.orders[0]  # type: ignore
        self._ref_group_grid_shape = ref_permute = tuple(  # type: ignore
            self.groups[0].group_grid_shape[i] for i in self._ref_order
        )

        self._reverse_orders = []

        for g, o in zip(self.groups, self.orders):
            Gi = g.group_grid_shape
            gperm = tuple(Gi[i] for i in o)
            if gperm != ref_permute:
                raise CouplingError(
                    "Incompatible grouped shapes "
                    f"after order: {gperm} vs {ref_permute} "
                    f"(spec {g.name or '<unnamed>'})"
                )
            self._reverse_orders.append(inverse_permutation(o))

    def grouped_block_norms(self, values: Values):
        grouped_block_norms = torch.cat(
            [
                g.grouped_block_norms(values).permute(o + (len(o),))
                for o, g in zip(self.orders, self.groups)
            ],
            dim=-1,
        )
        assert grouped_block_norms.shape[:-1] == self.group_grid_shape
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
        for ro, g in zip(self._reverse_orders, self.groups):
            g.hard_threshold(thresholds=thresholds.permute(ro))

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

        for ro, g in zip(self._reverse_orders, self.groups):
            g.soft_threshold(
                thresholds.permute(ro),
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
