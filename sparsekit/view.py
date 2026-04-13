# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0
# (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Strided parameter views with write-through to original storage."""

from dataclasses import dataclass
from typing import Tuple
import math
import torch
from torch import Tensor
from torch.nn import Parameter


@dataclass
class View:
    """A strided view of a Parameter that duck-types as a Parameter.

    Wraps an ``nn.Parameter`` with an arbitrary ``(size, stride)`` view
    via ``torch.as_strided``.  The ``.data`` property returns a live view
    into the original parameter's storage -- no copy is made, so in-place
    operations on the data write through to the underlying parameter.

    Workflow::

        view  = View(param, shape, stride)
        block = BlockSpec(view, shape=...)
        scope = ScopeSpec(block, shape=...)
        # ... pruning / thresholding -- writes go to param directly ...

    Args:
        param: The underlying Tensor/Parameter.
        shape:  Shape of the strided view.
        stride: Strides (in elements) for each dimension.
    """

    param: Parameter | Tensor
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]

    def __post_init__(self):
        self.shape = tuple(self.shape)
        self.stride = tuple(self.stride)
        if len(self.shape) != len(self.stride):
            raise ValueError(
                f"size has {len(self.shape)} dims but "
                f"stride has {len(self.stride)}"
            )

    @classmethod
    def from_existing(cls, param: "Tensor | Parameter | View") -> "View":
        """Wrap a Parameter or pass through an existing View."""
        if isinstance(param, View):
            return param
        return cls(
            param,
            shape=tuple(param.shape),
            stride=tuple(param.data.stride()),
        )

    @property
    def data(self) -> Tensor:
        """Live ``as_strided`` view into the parameter's storage."""
        return torch.as_strided(self.param.data, self.shape, self.stride)

    @data.setter
    def data(self, value: Tensor):
        torch.as_strided(self.param.data, self.shape, self.stride).copy_(value)

    @property
    def ndim(self) -> int:
        """Number of dimensions in the view."""
        return len(self.shape)

    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def cosize(self) -> int:
        """Maximum linear offset reachable by this view + 1."""
        return sum((s - 1) * d for s, d in zip(self.shape, self.stride)) + 1

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
        interleaved_shape: list[int] = []
        interleaved_stride: list[int] = []
        for si, bi, di in zip(t.shape, block_shape, t_stride):
            interleaved_shape.extend([si // bi, bi])
            interleaved_stride.extend([bi * di, di])

        view = torch.as_strided(
            t,
            tuple(interleaved_shape),
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
            block_shape: ``(b0, b1, …)`` Block shape.
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
                block_idx * bi
                for block_idx, bi in zip(block_values.shape, block_shape)
            )
            t = t.reshape(full_shape)
        return t

    def apply_multiplier(
        self, multiplier: Tensor, block_shape: Tuple[int, ...]
    ):
        """In-place multiply each block of ``self.data`` by a grid scalar.

        Args:
            multiplier: ``(B0, B1, …)`` grid-shaped tensor.
            block_shape: Block shape.
        """
        m = multiplier
        for i in range(multiplier.ndim):
            m = m.unsqueeze(2 * i + 1)
        b_view = View.block_view_of(self.data, block_shape, reorder=False)
        b_view.mul_(m)

    def apply_mask(self, mask: Tensor, block_shape: Tuple[int, ...]):
        """Zero out blocks of ``self.data`` where *mask* is True.

        Args:
            mask: ``(B0, B1, …)`` boolean grid tensor.
            block_shape: Block shape.
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

    def to_param_rc(self, idx: Tensor) -> Tuple[Tensor, Tensor]:
        """Map view-space indices to ``(row, col)`` in the 2-D *param*.

        Args:
            idx: ``(..., ndim)`` long tensor of view-space indices.

        Returns:
            ``(rows, cols)`` — each ``(...,)`` tensor of param-space indices.
        """
        assert self.param.ndim == 2, "param must be 2D"
        flat = self.linear_offset(idx)
        num_cols = self.param.shape[1]
        return flat // num_cols, flat % num_cols

    def __hash__(self) -> int:
        return hash((id(self.param), self.shape, self.stride))

    def __eq__(self, other) -> bool:
        if not isinstance(other, View):
            return NotImplemented
        return (
            self.param is other.param
            and self.shape == other.shape
            and self.stride == other.stride
        )

    def __repr__(self) -> str:
        return (
            f"View(param.shape={tuple(self.param.shape)}, "
            f"shape={self.shape}, stride={self.stride})"
        )
