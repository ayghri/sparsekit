# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0
# (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Visualization utilities for S³ sparsity layouts."""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List, Tuple, Union

from sparsekit.block import BlockSpec, BlockCoupling
from sparsekit.scope import ScopeSpec, ScopeCoupling

# Fill colors for blocks (pastel) — cycle if more blocks than entries
_COLORS = [
    "#AED6F1",
    "#A9DFBF",
    "#F9E79F",
    "#F1948A",
    "#C39BD3",
    "#F0B27A",
    "#76D7C4",
    "#F7DC6F",
    "#85C1E9",
    "#FDEBD0",
    "#D5F5E3",
    "#FDEDEC",
    "#EAF2FF",
    "#FEF9E7",
    "#F4ECF7",
    "#E8DAEF",
    "#D1F2EB",
    "#FAD7A0",
    "#D2B4DE",
    "#A2D9CE",
]

# Outline colors for scopes (high-contrast, saturated) — cycle if needed
_SCOPE_COLORS = [
    "#C0392B",
    "#1A5276",
    "#1E8449",
    "#784212",
    "#6C3483",
    "#117A65",
    "#B7950B",
    "#922B21",
    "#1F618D",
    "#196F3D",
    "#4A235A",
    "#0E6655",
    "#7B241C",
    "#154360",
    "#145A32",
]

# Inset (in data units) for block outlines — keeps outlines inside the block
_INSET = 0.07
# Line width for block outlines
_OUTLINE_LW = 2.5


def _merge_intervals(
    intervals: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Merge a sorted list of (start, end) intervals
    into non-overlapping ones."""
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _draw_scope_outlines(
    ax: plt.Axes,
    sm: np.ndarray,
    num_rows: int,
    _num_cols: int,  # pylint: disable=invalid-name
    block_offset: int,
) -> None:
    """Draw one inset perimeter per contiguous scope region.

    For each scope, boundary edges are merged into
    maximal collinear segments, then each segment is
    shifted inward by _INSET so outlines from adjacent
    scopes never overlap. Segment endpoints are also
    trimmed by _INSET so perpendicular segments meet
    cleanly at corners.
    """
    e = _INSET
    unique_sids = sorted(int(s) for s in np.unique(sm) if s >= 0)
    ngc = len(_SCOPE_COLORS)

    for sid in unique_sids:
        color = _SCOPE_COLORS[(sid + block_offset) % ngc]
        cells = set(zip(*np.where(sm == sid)))

        # Accumulate raw perimeter edges, keyed by their axis-position:
        #   top/bottom edges  → dict[y_abs] = [(x_start, x_end), ...]
        #   left/right edges  → dict[x_abs] = [(y_start, y_end), ...]
        top: dict = {}
        bot: dict = {}
        left: dict = {}
        right: dict = {}

        for row, col in cells:
            yt = num_rows - row  # y of top edge
            yb = num_rows - row - 1  # y of bottom

            if (row - 1, col) not in cells:
                top.setdefault(yt, []).append((col, col + 1))
            if (row + 1, col) not in cells:
                bot.setdefault(yb, []).append((col, col + 1))
            if (row, col - 1) not in cells:
                left.setdefault(col, []).append((yb, yt))
            if (row, col + 1) not in cells:
                right.setdefault(col + 1, []).append((yb, yt))

        # Draw merged horizontal segments (top: shift inward = down; bot: up)
        for y_abs, segs in top.items():
            for xs, xe in _merge_intervals(segs):
                ax.plot(
                    [xs + e, xe - e],
                    [y_abs - e, y_abs - e],
                    color=color,
                    lw=_OUTLINE_LW,
                    solid_capstyle="butt",
                    zorder=4,
                )
        for y_abs, segs in bot.items():
            for xs, xe in _merge_intervals(segs):
                ax.plot(
                    [xs + e, xe - e],
                    [y_abs + e, y_abs + e],
                    color=color,
                    lw=_OUTLINE_LW,
                    solid_capstyle="butt",
                    zorder=4,
                )

        # Draw merged vertical segments
        # (left: shift inward = right; right: left)
        for x_abs, segs in left.items():
            for ys, ye in _merge_intervals(segs):
                ax.plot(
                    [x_abs + e, x_abs + e],
                    [ys + e, ye - e],
                    color=color,
                    lw=_OUTLINE_LW,
                    solid_capstyle="butt",
                    zorder=4,
                )
        for x_abs, segs in right.items():
            for ys, ye in _merge_intervals(segs):
                ax.plot(
                    [x_abs - e, x_abs - e],
                    [ys + e, ye - e],
                    color=color,
                    lw=_OUTLINE_LW,
                    solid_capstyle="butt",
                    zorder=4,
                )


def _c_strides(shape: Tuple[int, ...]) -> List[int]:
    """C-order (row-major) flat strides for a shape tuple."""
    strides, s = [], 1
    for dim in reversed(shape):
        strides.insert(0, s)
        s *= dim
    return strides


def _build_maps(
    block_spec: BlockSpec,
    scope_spec: Optional[ScopeSpec] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Build (M, num_cols) integer maps of block_id and
    scope_id for one parameter.

    Returns:
        block_map: (M, num_cols) int32 array; -1 for
            elements not covered by the view.
        scope_map: (M, num_cols) int32 array; -1 if
            no scope_spec or not covered.
        (M, num_cols): physical parameter shape.
    """
    view = block_spec.view
    block_shape = block_spec.shape
    grid_shape = block_spec.grid_shape

    param = view.param
    if param.ndim == 2:
        M, num_cols = param.shape
    elif param.ndim == 1:
        M, num_cols = 1, param.shape[0]
    else:
        M = math.prod(param.shape[:-1])
        num_cols = param.shape[-1]

    # Coordinate grids over all view dimensions
    grids = torch.meshgrid(
        *[torch.arange(s) for s in view.shape],
        indexing="ij",
    )

    # Linear offset into param storage
    linear_offset = torch.zeros(view.shape, dtype=torch.long)
    for g, d in zip(grids, view.stride):
        linear_offset.add_(g.long() * d)

    param_rows = (linear_offset // num_cols).numpy()
    param_cols = (linear_offset % num_cols).numpy()

    # Block flat index
    bg_strides = _c_strides(grid_shape)
    block_ids = torch.zeros(view.shape, dtype=torch.long)
    for g, b, s in zip(grids, block_shape, bg_strides):
        block_ids.add_((g.long() // b) * s)

    block_map = np.full((M, num_cols), -1, dtype=np.int32)
    block_map[param_rows, param_cols] = block_ids.numpy()

    scope_map = np.full((M, num_cols), -1, dtype=np.int32)
    if scope_spec is not None:
        ss_strides = _c_strides(scope_spec.grid_shape)
        scope_ids = torch.zeros(view.shape, dtype=torch.long)
        for g, b, gs, s in zip(
            grids,
            block_shape,
            scope_spec.shape,
            ss_strides,
        ):
            scope_ids.add_((g.long() // (b * gs)) * s)
        scope_map[param_rows, param_cols] = scope_ids.numpy()

    return block_map, scope_map, (M, num_cols)


def _draw_param(
    block_spec: BlockSpec,
    scope_spec: Optional[ScopeSpec],
    ax: plt.Axes,
    max_rows: int,
    max_cols: int,
    color_offset: int = 0,
    block_offset: int = 0,
    label: Optional[str] = None,
):
    """Render one parameter's block/scope layout on *ax*.

    Args:
        block_spec: BlockSpec for this parameter.
        scope_spec: Optional ScopeSpec for scope outlines.
        ax: Axes to draw on.
        max_rows: Maximum number of param rows to display.
        max_cols: Maximum number of param cols to display.
        color_offset: Shift into _COLORS so coupled
            params use different colors.
        block_offset: Shift into _SCOPE_COLORS so
            scope colors stay consistent.
        label: Optional subtitle drawn below the axes.
    """
    bm_map, sm_map, (M, num_cols) = _build_maps(block_spec, scope_spec)

    num_rows = min(M, max_rows)
    C = min(num_cols, max_cols)
    bm = bm_map[:num_rows, :C]
    sm = sm_map[:num_rows, :C]

    nc = len(_COLORS)

    # ── Cell fill (color = block identity) ────────
    for row in range(num_rows):
        for col in range(C):
            bid = int(bm[row, col])
            y = num_rows - row - 1
            facecolor = (
                _COLORS[(bid + color_offset) % nc] if bid >= 0 else "white"
            )
            ax.add_patch(
                mpatches.Rectangle(
                    (col, y),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor="none",
                    zorder=1,
                )
            )

    # ── Thin cell-level grid ─────────────────────
    for r in range(num_rows + 1):
        ax.axhline(r, color="#cccccc", linewidth=0.4, zorder=2)
    for c in range(C + 1):
        ax.axvline(c, color="#cccccc", linewidth=0.4, zorder=2)

    # ── Scope outlines ───────────────────────────
    if scope_spec is not None:
        _draw_scope_outlines(ax, sm, num_rows, C, block_offset)

    # Outer border
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

    ax.set_xlim(0, C)
    ax.set_ylim(0, num_rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if label is not None:
        ax.set_title(label, fontsize=9, pad=3)


def draw_layout(
    spec: Union[
        ScopeSpec,
        BlockSpec,
        ScopeCoupling,
        BlockCoupling,
    ],
    max_rows: int = 32,
    max_cols: int = 64,
    title: Optional[str] = None,
    cell_size: float = 0.28,
) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
    """Draw the sparsity layout of *spec*.

    Visual encoding:

    - **Fill color**: elements sharing the same block
      get the same pastel fill.
    - **Outline color**: blocks in the same scope
      are encapsulated by a single colored perimeter.
      Contiguous cells share one outline; disconnected
      regions within the same scope each get their
      own outline in the same color.
    - **Thin lines**: cell grid.

    For coupled specs, each coupled parameter is drawn
    as a separate subplot with a shared color/hatch
    scheme so that coupled blocks are visually
    consistent across parameters.

    Args:
        spec: A ScopeSpec, BlockSpec,
            ScopeCoupling, or BlockCoupling.
        max_rows: Max param rows rendered per subplot.
        max_cols: Max param cols rendered per subplot.
        title: Overall figure title.
        cell_size: Approx size (inches) per cell.

    Returns:
        (fig, ax) for single-param specs or
        (fig, [ax, ...]) for coupled specs.
    """
    # Collect (block_spec, scope_spec, label) triples
    triples: List[
        Tuple[
            BlockSpec,
            Optional[ScopeSpec],
            Optional[str],
        ]
    ] = []

    if isinstance(spec, ScopeSpec):
        block = spec.block
        if isinstance(block, BlockSpec):
            triples.append((block, spec, block.name))
        elif isinstance(block, BlockCoupling):
            for s in block.specs:
                triples.append((s, None, s.name or "param"))
        else:
            raise TypeError(f"Unsupported block type: {type(block)}")

    elif isinstance(spec, BlockSpec):
        triples.append((spec, None, spec.name or "param"))

    elif isinstance(spec, ScopeCoupling):
        for sc in spec.scopes:
            block = sc.block
            if isinstance(block, BlockSpec):
                triples.append((block, sc, block.name))
            elif isinstance(block, BlockCoupling):
                for s in block.specs:
                    triples.append((s, None, s.name or "param"))

    elif isinstance(spec, BlockCoupling):
        for s in spec.specs:
            triples.append((s, None, s.name or "param"))

    else:
        raise TypeError(f"Unsupported spec type: {type(spec)}")

    n = len(triples)
    if n == 0:
        raise ValueError("No parameters found in spec.")

    # Figure size based on display cells
    num_rows = min(
        (
            triples[0][0].view.param.shape[0]
            if triples[0][0].view.param.ndim >= 1
            else max_rows
        ),
        max_rows,
    )
    C = min(
        (
            triples[0][0].view.param.shape[-1]
            if triples[0][0].view.param.ndim >= 1
            else max_cols
        ),
        max_cols,
    )
    w = max(C * cell_size + 0.1, 1.0)
    h = max(num_rows * cell_size + 0.1, 1.0)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(w * n + 0.3 * (n - 1), h),
        squeeze=False,
    )
    axes = axes[0]

    # Color/hatch offsets: coupled params stagger so
    # their block colors don't clash
    for i, (bs, gs, lbl) in enumerate(triples):
        # Stagger color offset so two coupled params
        # with the same block layout still have
        # visually distinct blocks across subplots.
        color_offset = i * (len(_COLORS) // max(n, 1))
        _draw_param(
            bs,
            gs,
            axes[i],
            max_rows=max_rows,
            max_cols=max_cols,
            color_offset=color_offset,
            block_offset=0,  # same hatch = same block position across params
            label=lbl,
        )

    if title:
        fig.suptitle(title, fontsize=10, y=1.01)

    fig.tight_layout()

    return fig, (axes[0] if n == 1 else list(axes))
