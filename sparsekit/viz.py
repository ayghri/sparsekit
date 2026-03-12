# Copyright (c) 2025 Anonymous Authors
# Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Visualization utilities for S³ sparsity layouts."""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List, Tuple, Union

from .block import BlockSpec, BlockCoupling
from .group import GroupSpec, GroupCoupling


# Fill colors for blocks (pastel) — cycle if more blocks than entries
_COLORS = [
    "#AED6F1", "#A9DFBF", "#F9E79F", "#F1948A", "#C39BD3",
    "#F0B27A", "#76D7C4", "#F7DC6F", "#85C1E9", "#FDEBD0",
    "#D5F5E3", "#FDEDEC", "#EAF2FF", "#FEF9E7", "#F4ECF7",
    "#E8DAEF", "#D1F2EB", "#FAD7A0", "#D2B4DE", "#A2D9CE",
]

# Outline colors for groups (high-contrast, saturated) — cycle if needed
_GROUP_COLORS = [
    "#C0392B", "#1A5276", "#1E8449", "#784212", "#6C3483",
    "#117A65", "#B7950B", "#922B21", "#1F618D", "#196F3D",
    "#4A235A", "#0E6655", "#7B241C", "#154360", "#145A32",
]

# Inset (in data units) for group outlines — keeps outlines inside the group
_INSET = 0.07
# Line width for group outlines
_OUTLINE_LW = 2.5


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge a sorted list of (start, end) intervals into non-overlapping ones."""
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _draw_group_outlines(
    ax: plt.Axes,
    gm: np.ndarray,
    R: int,
    C: int,
    group_offset: int,
) -> None:
    """Draw one inset perimeter per contiguous group region.

    For each group, boundary edges are merged into maximal collinear segments,
    then each segment is shifted inward by _INSET so outlines from adjacent
    groups never overlap. Segment endpoints are also trimmed by _INSET so
    perpendicular segments meet cleanly at corners.
    """
    e = _INSET
    unique_gids = sorted(int(g) for g in np.unique(gm) if g >= 0)
    ngc = len(_GROUP_COLORS)

    for gid in unique_gids:
        color = _GROUP_COLORS[(gid + group_offset) % ngc]
        cells = set(zip(*np.where(gm == gid)))

        # Accumulate raw perimeter edges, keyed by their axis-position:
        #   top/bottom edges  → dict[y_abs] = [(x_start, x_end), ...]
        #   left/right edges  → dict[x_abs] = [(y_start, y_end), ...]
        top:   dict = {}
        bot:   dict = {}
        left:  dict = {}
        right: dict = {}

        for (row, col) in cells:
            yt = R - row        # y of top edge of this cell
            yb = R - row - 1    # y of bottom edge

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
                ax.plot([xs + e, xe - e], [y_abs - e, y_abs - e],
                        color=color, lw=_OUTLINE_LW, solid_capstyle='butt', zorder=4)
        for y_abs, segs in bot.items():
            for xs, xe in _merge_intervals(segs):
                ax.plot([xs + e, xe - e], [y_abs + e, y_abs + e],
                        color=color, lw=_OUTLINE_LW, solid_capstyle='butt', zorder=4)

        # Draw merged vertical segments (left: shift inward = right; right: left)
        for x_abs, segs in left.items():
            for ys, ye in _merge_intervals(segs):
                ax.plot([x_abs + e, x_abs + e], [ys + e, ye - e],
                        color=color, lw=_OUTLINE_LW, solid_capstyle='butt', zorder=4)
        for x_abs, segs in right.items():
            for ys, ye in _merge_intervals(segs):
                ax.plot([x_abs - e, x_abs - e], [ys + e, ye - e],
                        color=color, lw=_OUTLINE_LW, solid_capstyle='butt', zorder=4)


def _c_strides(shape: Tuple[int, ...]) -> List[int]:
    """C-order (row-major) flat strides for a shape tuple."""
    strides, s = [], 1
    for dim in reversed(shape):
        strides.insert(0, s)
        s *= dim
    return strides


def _build_maps(
    block_spec: BlockSpec,
    group_spec: Optional[GroupSpec] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Build (M, K) integer maps of block_id and group_id for one parameter.

    Returns:
        block_map: (M, K) int32 array; -1 for elements not covered by the view.
        group_map: (M, K) int32 array; -1 if no group_spec or not covered.
        (M, K): physical parameter shape.
    """
    view = block_spec.view
    block_shape = block_spec.shape
    grid_shape = block_spec.grid_shape

    param = view.param
    if param.ndim == 2:
        M, K = param.shape
    elif param.ndim == 1:
        M, K = 1, param.shape[0]
    else:
        M = math.prod(param.shape[:-1])
        K = param.shape[-1]

    # Coordinate grids over all view dimensions
    grids = torch.meshgrid(
        *[torch.arange(s) for s in view.shape], indexing='ij'
    )

    # Linear offset into param storage: Σ coord_i * stride_i
    linear_offset = torch.zeros(view.shape, dtype=torch.long)
    for g, d in zip(grids, view.stride):
        linear_offset.add_(g.long() * d)

    param_rows = (linear_offset // K).numpy()
    param_cols = (linear_offset % K).numpy()

    # Block flat index = Σ_i floor(coord_i / block_shape_i) * block_grid_stride_i
    bg_strides = _c_strides(grid_shape)
    block_ids = torch.zeros(view.shape, dtype=torch.long)
    for g, b, s in zip(grids, block_shape, bg_strides):
        block_ids.add_((g.long() // b) * s)

    block_map = np.full((M, K), -1, dtype=np.int32)
    block_map[param_rows, param_cols] = block_ids.numpy()

    group_map = np.full((M, K), -1, dtype=np.int32)
    if group_spec is not None:
        gg_strides = _c_strides(group_spec.grid_shape)
        group_ids = torch.zeros(view.shape, dtype=torch.long)
        for g, b, gs, s in zip(
            grids, block_shape, group_spec.shape, gg_strides
        ):
            group_ids.add_((g.long() // (b * gs)) * s)
        group_map[param_rows, param_cols] = group_ids.numpy()

    return block_map, group_map, (M, K)


def _draw_param(
    block_spec: BlockSpec,
    group_spec: Optional[GroupSpec],
    ax: plt.Axes,
    max_rows: int,
    max_cols: int,
    color_offset: int = 0,
    group_offset: int = 0,
    label: Optional[str] = None,
):
    """Render one parameter's block/group layout on *ax*.

    Args:
        block_spec: BlockSpec for this parameter.
        group_spec: Optional GroupSpec for group outlines.
        ax: Axes to draw on.
        max_rows: Maximum number of param rows to display.
        max_cols: Maximum number of param cols to display.
        color_offset: Shift into _COLORS so coupled params use different colors.
        group_offset: Shift into _GROUP_COLORS so group colors stay consistent.
        label: Optional subtitle drawn below the axes.
    """
    block_map, group_map, (M, K) = _build_maps(block_spec, group_spec)

    R = min(M, max_rows)
    C = min(K, max_cols)
    bm = block_map[:R, :C]
    gm = group_map[:R, :C]

    nc = len(_COLORS)
    ngc = len(_GROUP_COLORS)

    # ── Cell fill (color = block identity) ──────────────────────────
    for row in range(R):
        for col in range(C):
            bid = int(bm[row, col])
            y = R - row - 1
            facecolor = _COLORS[(bid + color_offset) % nc] if bid >= 0 else "white"
            ax.add_patch(mpatches.Rectangle(
                (col, y), 1, 1,
                facecolor=facecolor,
                edgecolor="none",
                zorder=1,
            ))

    # ── Thin cell-level grid ─────────────────────────────────────────
    for r in range(R + 1):
        ax.axhline(r, color="#cccccc", linewidth=0.4, zorder=2)
    for c in range(C + 1):
        ax.axvline(c, color="#cccccc", linewidth=0.4, zorder=2)

    # ── Group outlines — inset perimeter per contiguous region ──────
    if group_spec is not None:
        _draw_group_outlines(ax, gm, R, C, group_offset)

    # Outer border
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")

    ax.set_xlim(0, C)
    ax.set_ylim(0, R)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if label is not None:
        ax.set_title(label, fontsize=9, pad=3)


def draw_layout(
    spec: Union[GroupSpec, BlockSpec, GroupCoupling, BlockCoupling],
    max_rows: int = 32,
    max_cols: int = 64,
    title: Optional[str] = None,
    cell_size: float = 0.28,
) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
    """Draw the S³ sparsity layout of *spec*.

    Visual encoding:
    - **Fill color**: elements sharing the same block get the same pastel fill.
    - **Outline color**: blocks belonging to the same group are encapsulated by
      a single colored perimeter. Contiguous cells share one outline; disconnected
      regions within the same group each get their own outline in the same color.
    - **Thin lines**: cell grid.

    For coupled specs (GroupCoupling / BlockCoupling), each coupled parameter
    is drawn as a separate subplot with a shared color/hatch scheme so that
    coupled groups are visually consistent across parameters.

    Args:
        spec: A GroupSpec, BlockSpec, GroupCoupling, or BlockCoupling.
        max_rows: Maximum number of parameter rows rendered per subplot.
        max_cols: Maximum number of parameter columns rendered per subplot.
        title: Overall figure title.
        cell_size: Approximate size (inches) of each displayed cell.

    Returns:
        (fig, ax) for single-param specs or (fig, [ax, ...]) for coupled specs.
    """
    # ── Collect (block_spec, group_spec, label) triples ──────────────
    triples: List[Tuple[BlockSpec, Optional[GroupSpec], Optional[str]]] = []

    if isinstance(spec, GroupSpec):
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

    elif isinstance(spec, GroupCoupling):
        for g in spec.groups:
            block = g.block
            if isinstance(block, BlockSpec):
                triples.append((block, g, block.name))
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

    # ── Figure size based on display cells ───────────────────────────
    R = min(triples[0][0].view.param.shape[0] if triples[0][0].view.param.ndim >= 1 else max_rows, max_rows)
    C = min(triples[0][0].view.param.shape[-1] if triples[0][0].view.param.ndim >= 1 else max_cols, max_cols)
    w = max(C * cell_size + 0.1, 1.0)
    h = max(R * cell_size + 0.1, 1.0)
    fig, axes = plt.subplots(
        1, n,
        figsize=(w * n + 0.3 * (n - 1), h),
        squeeze=False,
    )
    axes = axes[0]

    # Color/hatch offsets: coupled params stagger so their block colors don't clash
    for i, (bs, gs, lbl) in enumerate(triples):
        # Stagger color offset so two coupled params with the same block layout
        # still have visually distinct blocks across subplots.
        color_offset = i * (len(_COLORS) // max(n, 1))
        _draw_param(
            bs, gs, axes[i],
            max_rows=max_rows,
            max_cols=max_cols,
            color_offset=color_offset,
            group_offset=0,   # same hatch = same group position across params
            label=lbl,
        )

    if title:
        fig.suptitle(title, fontsize=10, y=1.01)

    fig.tight_layout()

    return fig, (axes[0] if n == 1 else list(axes))
