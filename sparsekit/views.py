"""
Copyright (c) 2025 Ayoub Ghriss and contributors
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.

Classes
-------
BaseView:

BlockView:
GroupView:

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


@dataclass
class BaseView(ABC):
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
    def nnz(self, eps=1e-8) -> int:
        """Count non-zero elements with absolute value > eps."""
        pass

    @abstractmethod
    def apply_mask(self, mask):
        """Zero out blocks where mask is True."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        return repr(self)

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def apply_multiplier(self, multiplier: Tensor):
        pass

    @abstractmethod
    def get_masks(self) -> Tensor:
        pass

    # @abstractmethod
    # def apply_multiplier(self, multiplier: Tensor):
    #     """Multiply each block by corresponding scalar in multiplier."""
    #     pass

    # @abstractmethod
    # def get_masks(self, block_masks: Tensor) -> Mapping["BlockSpec", Tensor]:
    #     """Convert block-level mask to element-level masks per BlockSpec."""
    #     pass

    # @property
    # @abstractmethod
    # def data(self) -> Mapping["BlockSpec", Tensor] | Tensor:
    #     """Raw tensor data of the underlying parameter(s)."""
    #     pass


@dataclass
class TensorView(BaseView):
    """Treats the entire tensor as a grid of blocks.

    Attributes:
        param: The underlying Parameter tensor.
        shape:
        stride:
        name
    """

    param: Parameter
    # shape: Tuple[int, ...]
    size: Tuple[int, ...]
    stride: Tuple[int, ...]
    offset: int = 0
    name: Optional[str] = ""

    def __post_init__(self):
        self.data = torch.as_strided(self.param.data, self.size, self.stride)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def numel(self) -> int:
        """Number of elements in the underlying tensor."""
        return self.data.numel()
    
    def linear_offset(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (..., rank) long tensor of indices into the view
        returns (...,) element offsets into base storage.
        """
        s = torch.tensor(self.stride, device=idx.device, dtype=idx.dtype)
        return self.offset + (idx * s).sum(dim=-1)

    # @property
    # def ndim(self) -> int:
    #     """Number of dimensions in the underlying tensor."""
    #     return self.param.ndim

    # @property
    # def data(self) -> Tensor:
    #     """Raw tensor data of the underlying Parameter."""
    #     return self.param.data

    # def set_data(self, data):
    #     """Copy data into the underlying Parameter tensor."""
    #     self.param.data.copy_(data)

    # def nnz(self, eps=1e-8) -> int:
    #     """Number of *non-zero* elements (within tolerance)."""
    #     return int((self.param.data.abs() > eps).sum().item())

    # @cached_property
    # def grid_shape(self) -> Tuple[int, ...]:
    #     """Full grid shape including singleton dimensions."""
    #     return tuple(si // bi for si, bi in zip(self.shape, self.block_shape))

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

    # @cached_property
    # def block_numel(self) -> int:
    #     """Number of elements per block."""
    #     return math.prod(self.block_shape)

    # @cached_property
    # def num_blocks(self) -> int:
    #     """Total number of blocks in the grid."""
    #     return math.prod(self.grid_shape)

    # @cached_property
    # def _reduction_dim(self) -> Tuple[int, ...]:
    #     """Odd-indexed dimensions to reduce over for block statistics."""
    #     return tuple(range(1, 2 * len(self.block_shape), 2))

    # def _resolve_values(self, values: Values) -> Tensor:
    #     """Resolve values to a tensor matching self.shape."""
    #     if values is None:
    #         return self.param.data
    #     if isinstance(values, dict):
    #         return values[self]
    #     if isinstance(values, Tensor):
    #         if values.shape != self.shape:
    #             raise ShapeMismatchError(
    #                 self.shape, tuple(values.shape), "values"
    #             )
    #         return values
    #     raise ValueError(
    #         "values has to be None, Tensor or Dict[BlockSpec, Tensor]"
    #     )

    # def tensor_to_block_view(
    #     self, values: Values, reorder: bool = True, merge=False
    # ) -> Tensor:
    #     """Reshape tensor to interleaved block view.

    #     Args:
    #         t: Input tensor matching self.shape.
    #         merge: If True, collapse block dims to trailing dim.

    #     Returns:
    #     """

    #     t = self._resolve_values(values)
    #     assert t.shape == self.shape
    #     interleaved_shape = []
    #     for B, bi in zip(self.grid_shape, self.block_shape):
    #         interleaved_shape.extend([B, bi])
