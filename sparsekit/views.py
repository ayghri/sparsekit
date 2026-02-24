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

from dataclasses import dataclass
from typing import Tuple
import math
import torch
from torch import Tensor
from torch.nn import Parameter


@dataclass
class BlockView:
    """A strided view of a Parameter that duck-types as a Parameter.

    Wraps an ``nn.Parameter`` with an arbitrary ``(size, stride)`` view
    via ``torch.as_strided``.  The ``.data`` property returns a live view
    into the original parameter's storage -- no copy is made, so in-place
    operations on the data write through to the underlying parameter.

    Workflow::

        view  = BlockView(param, size, stride)
        block = BlockSpec(view, block_shape=...)
        group = GroupSpec(block, group_shape=...)
        # ... pruning / thresholding -- writes go to param directly ...

    Args:
        param: The underlying ``nn.Parameter``.
        size:  Shape of the strided view.
        stride: Strides (in elements) for each dimension.
    """

    param: Parameter
    size: Tuple[int, ...]
    stride: Tuple[int, ...]

    def __post_init__(self):
        self.size = tuple(self.size)
        self.stride = tuple(self.stride)
        if len(self.size) != len(self.stride):
            raise ValueError(
                f"size has {len(self.size)} dims but "
                f"stride has {len(self.stride)}"
            )

    @classmethod
    def from_reshape(
        cls, param: Parameter, size: Tuple[int, ...]
    ) -> "BlockView":
        """View that is a contiguous reshape (row-major strides)."""
        size = tuple(size)
        if math.prod(size) != param.numel():
            raise ValueError(
                f"cannot reshape {tuple(param.shape)} "
                f"({param.numel()} elems) to {size} "
                f"({math.prod(size)} elems)"
            )
        strides: list = []
        s = 1
        for sz in reversed(size):
            strides.append(s)
            s *= sz
        return cls(param, size, tuple(reversed(strides)))

    @classmethod
    def from_existing(cls, param: "Parameter | BlockView") -> "BlockView":
        """Wrap a Parameter or pass through an existing BlockView."""
        if isinstance(param, BlockView):
            return param
        return cls(
            param,
            size=tuple(param.shape),
            stride=tuple(param.data.stride()),
        )

    @property
    def data(self) -> Tensor:
        """Live ``as_strided`` view into the parameter's storage."""
        return torch.as_strided(self.param.data, self.size, self.stride)

    @data.setter
    def data(self, value: Tensor):
        torch.as_strided(self.param.data, self.size, self.stride).copy_(value)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.size

    @property
    def ndim(self) -> int:
        return len(self.size)

    def numel(self) -> int:
        return math.prod(self.size)

    @staticmethod
    def block_view_of(
        t: Tensor,
        block_shape: Tuple[int, ...],
        reorder: bool = True,
        merge: bool = False,
    ) -> Tensor:
        """Block-structured view of *t* via ``as_strided``.

        Args:
            t: Tensor of shape ``(s0, s1, …, sm)``.
            block_shape: ``(b0, b1, …, bm)`` with ``si % bi == 0``.
            reorder: Permute grid dims before block dims.
            merge: Flatten block dims into a single trailing dim
                   (implies *reorder*).

        Returns:
            ``reorder=False``: ``(B0, b0, B1, b1, …)`` interleaved view.
            ``reorder=True``:  ``(B0, B1, …, b0, b1, …)`` reordered view.
            ``merge=True``:    ``(B0, B1, …, b0*b1*…)`` merged view.
        """
        t_stride = t.stride()
        interleaved_size: list[int] = []
        interleaved_stride: list[int] = []
        for si, bi, di in zip(t.shape, block_shape, t_stride):
            interleaved_size.extend([si // bi, bi])
            interleaved_stride.extend([bi * di, di])

        view = torch.as_strided(
            t,
            tuple(interleaved_size),
            tuple(interleaved_stride),
            t.storage_offset(),
        )

        if reorder or merge:
            ndim = len(block_shape)
            perm = list(range(0, 2 * ndim, 2)) + list(range(1, 2 * ndim, 2))
            view = view.permute(*perm)
            if merge:
                grid_shape = tuple(
                    si // bi for si, bi in zip(t.shape, block_shape)
                )
                view = view.reshape(*grid_shape, -1)

        return view

    @staticmethod
    def broadcast_block_to_element(
        block_values: Tensor,
        block_shape: Tuple[int, ...],
        fake: bool = False,
    ) -> Tensor:
        """Expand a grid-shaped tensor to the full element shape.

        Args:
            block_values: ``(B0, B1, …)`` grid tensor.
            block_shape: ``(b0, b1, …)`` block sizes per dim.
            fake: If True only unsqueeze (for broadcasting against
                  an interleaved view) without expanding.

        Returns:
            ``fake=True``:  ``(B0, 1, B1, 1, …)``
            ``fake=False``: ``(s0, s1, …)`` with ``si = Bi * bi``.
        """
        t = block_values
        for i, bi in enumerate(block_shape):
            t = t.unsqueeze(2 * i + 1)
            if not fake:
                t = t.repeat_interleave(bi, dim=2 * i + 1)
        if not fake:
            full_shape = tuple(
                Bi * bi for Bi, bi in zip(block_values.shape, block_shape)
            )
            t = t.reshape(full_shape)
        return t

    def apply_multiplier(
        self, multiplier: Tensor, block_shape: Tuple[int, ...]
    ):
        """In-place multiply each block of ``self.data`` by a grid scalar.

        Args:
            multiplier: ``(B0, B1, …)`` grid-shaped tensor.
            block_shape: Block sizes per dimension.
        """
        m = multiplier
        for i in range(multiplier.ndim):
            m = m.unsqueeze(2 * i + 1)
        b_view = BlockView.block_view_of(self.data, block_shape, reorder=False)
        b_view.mul_(m)

    def apply_mask(self, mask: Tensor, block_shape: Tuple[int, ...]):
        """Zero out blocks of ``self.data`` where *mask* is True.

        Args:
            mask: ``(B0, B1, …)`` boolean grid tensor.
            block_shape: Block sizes per dimension.
        """
        self.apply_multiplier(~mask, block_shape)

    def linear_offset(self, idx: Tensor) -> Tensor:
        """Map multi-dim view indices to linear offsets in *param*'s storage.

        Args:
            idx: ``(..., ndim)`` long tensor of view-space indices.

        Returns:
            ``(...,)`` tensor of flat storage offsets.
        """
        s = torch.tensor(self.stride, device=idx.device, dtype=idx.dtype)
        return (idx * s).sum(dim=-1)

    def __hash__(self) -> int:
        return hash((id(self.param), self.size, self.stride))

    def __eq__(self, other) -> bool:
        if not isinstance(other, BlockView):
            return NotImplemented
        return (
            self.param is other.param
            and self.size == other.size
            and self.stride == other.stride
        )

    def __repr__(self) -> str:
        return (
            f"BlockView(param.shape={tuple(self.param.shape)}, "
            f"size={self.size}, stride={self.stride})"
        )


# @dataclass
# class BaseView(ABC):
#     """Abstract base class for block-structured sparse tensors.

#     Provides interface for viewing tensors as block grids, computing block
#     statistics, and applying soft/hard thresholding operations.
#     """

#     @property
#     @abstractmethod
#     def shape(self) -> Tuple[int, ...]:
#         """Full shape of the underlying tensor."""
#         pass

#     @abstractmethod
#     def numel(self) -> int:
#         """Total number of elements in the underlying tensor."""
#         pass

#     @abstractmethod
#     def parameters(self) -> Iterable[Parameter]:
#         """Iterable of Parameter objects managed by this node."""
#         pass

#     @abstractmethod
#     def nnz(self, eps=1e-8) -> int:
#         """Count non-zero elements with absolute value > eps."""
#         pass

#     @abstractmethod
#     def apply_mask(self, mask):
#         """Zero out blocks where mask is True."""
#         pass

#     @abstractmethod
#     def __repr__(self) -> str:
#         pass

#     def __str__(self) -> str:
#         return repr(self)

#     @abstractmethod
#     def __hash__(self) -> int:
#         pass

#     @abstractmethod
#     def apply_multiplier(self, multiplier: Tensor):
#         pass

#     @abstractmethod
#     def get_masks(self) -> Tensor:
#         pass

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


# @dataclass
# class TensorView(BaseView):
#     """Treats the entire tensor as a grid of blocks.

#     Attributes:
#         param: The underlying Parameter tensor.
#         shape:
#         stride:
#         name
#     """

#     param: Parameter
#     # shape: Tuple[int, ...]
#     size: Tuple[int, ...]
#     stride: Tuple[int, ...]
#     offset: int = 0
#     name: Optional[str] = ""

#     def __post_init__(self):
#         self.data = torch.as_strided(self.param.data, self.size, self.stride)

#     @property
#     def shape(self) -> Tuple[int, ...]:
#         return self.data.shape

#     def numel(self) -> int:
#         """Number of elements in the underlying tensor."""
#         return self.data.numel()

#     def linear_offset(self, idx: torch.Tensor) -> torch.Tensor:
#         """
#         idx: (..., rank) long tensor of indices into the view
#         returns (...,) element offsets into base storage.
#         """
#         s = torch.tensor(self.stride, device=idx.device, dtype=idx.dtype)
#         return self.offset + (idx * s).sum(dim=-1)

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
