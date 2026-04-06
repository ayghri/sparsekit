# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0
# (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
# type: ignore [reachability]
"""Triton kernels for k-th largest selection.

Uses tl.topk for efficient partial sort — only materializes k values
instead of sorting the full dimension.

Five operations with auto-dispatch based on (K, k):
  - kth_largest: auto-dispatches to torch,
    single-load triton, or streaming triton
  - mid_kth_largest: auto-dispatches to torch,
    single-load triton, or streaming triton
  - streaming_kth_largest: chunked topk via
    join+reshape merge (direct access)
  - streaming_mid_kth_largest: chunked midpoint
    via join+reshape merge (direct access)
  - radix_kth_largest: chunked topk via
    radix-key tl.maximum merge (direct access)
"""

import torch
import triton
import triton.language as tl
import math
from torch import Tensor

# ── Dispatch thresholds (from benchmarks on A100) ──────────────────────
# M*K below this → torch (small tensor, launch overhead dominates)
_SMALL_TENSOR = 1_000
# next_power_of_2(K) above this → can't single-load, need streaming or torch
_SINGLE_LOAD_LIMIT = 1024


def _block_m_heuristic(block_k: int, max_registers: int = 1024) -> int:
    return int(max(min(max_registers // 2 ** math.ceil(math.log2(block_k)), 256), 1))


def _reduce_dim_strides(x: Tensor, dim: int):
    """Compute (x_flat, n_rows, n_cols, stride_row, stride_col, out_shape)
    for reducing along ``dim``, zero-copy when possible."""
    ndim = x.ndim
    dim = dim % ndim
    n_cols = x.shape[dim]
    stride_col = x.stride(dim)

    if ndim == 1:
        return x, 1, n_cols, 1, stride_col, ()

    if ndim == 2:
        row_dim = 1 - dim
        n_rows = x.shape[row_dim]
        stride_row = x.stride(row_dim)
        out_shape = (n_rows,)
        return x, n_rows, n_cols, stride_row, stride_col, out_shape

    # N-D: permute dim to last, flatten leading dims
    perm = [i for i in range(ndim) if i != dim] + [dim]
    x_perm = x.permute(perm)
    out_shape = x_perm.shape[:-1]
    n_rows = x_perm.shape[:-1].numel()

    # Check if leading dims are contiguous so we can flatten without copy
    leading_contiguous = True
    for i in range(len(out_shape) - 1):
        if x_perm.stride(i) != x_perm.stride(i + 1) * x_perm.shape[i + 1]:
            leading_contiguous = False
            break

    if leading_contiguous:
        stride_row = x_perm.stride(len(out_shape) - 1)
        stride_col = x_perm.stride(-1)
        x_flat = x_perm.as_strided(
            (n_rows, n_cols),
            (stride_row, stride_col),
            storage_offset=x_perm.storage_offset(),
        )
    else:
        x_flat = x_perm.contiguous()
        stride_row = n_cols
        stride_col = 1

    return x_flat, n_rows, n_cols, stride_row, stride_col, out_shape


# ── kth_largest ──────────────────────────────────────────────────────────


@triton.jit
def _kth_largest_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_row,
    stride_col,
    kth: tl.constexpr,
    kth_next_p2: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    """Single-load topk kernel for k-th largest selection."""
    group_idx = tl.program_id(0)
    col_offsets = tl.arange(0, block_k)
    row_indices = group_idx * block_m + tl.arange(0, block_m)

    ptrs = row_indices[:, None] * stride_row + col_offsets[None, :] * stride_col
    row_mask = row_indices < n_rows
    col_mask = col_offsets < n_cols
    mask = row_mask[:, None] & col_mask[None, :]

    data = tl.load(x_ptr + ptrs, mask=mask, other=float("-inf"))
    topk_vals = tl.topk(data, kth_next_p2)

    select = tl.arange(0, kth_next_p2)[None, :]
    kth_val = tl.sum(tl.where(select == (kth - 1), topk_vals, 0.0), axis=1)

    tl.store(out_ptr + row_indices, kth_val.to(out_ptr.dtype.element_ty), mask=row_mask)


def kth_largest(
    x: Tensor,
    k: int,
    dim: int = -1,
    chunk_k: int = 1024,
) -> Tensor:
    """Return the k-th largest value along a dimension.

    Auto-dispatches between torch, single-load triton, and streaming triton
    based on dimension size K and rank k.

    Args:
        x: Input tensor of any shape.
        k: 1-based rank (k=1 is the maximum).
        dim: Dimension to reduce over (default: last).
        chunk_k: Columns per streaming chunk when streaming path is used.

    Returns:
        Tensor with ``dim`` removed, containing the k-th largest value.
    """
    ndim = x.ndim
    assert ndim >= 1, "Input must be at least 1-D"
    dim = dim % ndim
    num_cols = x.shape[dim]
    assert 1 <= k <= num_cols, f"k={k} out of range for dim size {num_cols}"

    block_k = triton.next_power_of_2(num_cols)
    topk = triton.next_power_of_2(k)
    n_elems = x.numel() // num_cols

    # Dispatch
    if n_elems * num_cols < _SMALL_TENSOR or block_k > _SINGLE_LOAD_LIMIT:
        return torch.kthvalue(x, num_cols - k + 1, dim=dim).values

    # Single-load triton
    x_flat, n_rows, n_cols, stride_row, stride_col, out_shape = _reduce_dim_strides(
        x, dim
    )

    block_m = _block_m_heuristic(block_k)
    num_warps = max(1, min(16, block_m * block_k // 256))
    num_blocks = triton.cdiv(n_rows, block_m)

    out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    _kth_largest_kernel[(num_blocks,)](
        x_ptr=x_flat,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        stride_row=stride_row,
        stride_col=stride_col,
        kth=k,
        kth_next_p2=topk,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )

    if len(out_shape) == 0:
        return out.squeeze()
    return out.view(out_shape)


# ── mid_kth_largest ──────────────────────────────────────────────────────


@triton.jit
def _mid_kth_largest_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_row,
    stride_col,
    k_val: tl.constexpr,
    k_val_next_p2: tl.constexpr,
    k_weight: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    """topk(topk), then average positions k-1 and k (0-indexed)."""
    group_idx = tl.program_id(0)
    col_offsets = tl.arange(0, block_k)
    row_indices = group_idx * block_m + tl.arange(0, block_m)

    ptrs = row_indices[:, None] * stride_row + col_offsets[None, :] * stride_col
    row_mask = row_indices < n_rows
    col_mask = col_offsets < n_cols
    mask = row_mask[:, None] & col_mask[None, :]

    data = tl.load(x_ptr + ptrs, mask=mask, other=float("-inf"))
    topk_vals = tl.topk(data, k_val_next_p2)

    select = tl.arange(0, k_val_next_p2)[None, :]
    val_k = tl.sum(tl.where(select == (k_val - 1), topk_vals, 0.0), axis=1)
    val_k1 = tl.sum(tl.where(select == k_val, topk_vals, 0.0), axis=1)

    mid = (val_k + k_weight * val_k1) / (1.0 + k_weight)
    tl.store(out_ptr + row_indices, mid.to(out_ptr.dtype.element_ty), mask=row_mask)


def mid_kth_largest(
    x: Tensor,
    k: int,
    dim: int = -1,
    k_weight: float = 1.0,
) -> Tensor:
    """Midpoint of the k-th and (k+1)-th largest values along a dimension.

    Auto-dispatches between torch, single-load triton, and streaming triton
    based on dimension size K and rank k.

    Args:
        x: Input tensor of any shape.
        k: 1-based rank (k=1 gives midpoint of max and 2nd-max).
        dim: Dimension to reduce over (default: last).
        chunk_k: Columns per streaming chunk when streaming path is used.

    Returns:
        Tensor with ``dim`` removed.
    """
    ndim = x.ndim
    assert ndim >= 1, "Input must be at least 1-D"
    dim = dim % ndim
    num_cols = x.shape[dim]
    assert (
        1 <= k and k + 1 <= num_cols
    ), f"k={k} out of range: need k+1 <= dim size {num_cols}"

    kp1 = k + 1
    block_k = triton.next_power_of_2(num_cols)
    topk = triton.next_power_of_2(kp1)
    n_elems = x.numel() // num_cols

    # Dispatch
    if n_elems * num_cols < _SMALL_TENSOR or block_k > _SINGLE_LOAD_LIMIT:
        v1 = torch.kthvalue(x, num_cols - k, dim=dim).values
        v2 = torch.kthvalue(x, num_cols - k + 1, dim=dim).values
        return (v1 + k_weight * v2) / (1.0 + k_weight)

    # Single-load triton
    x_flat, n_rows, n_cols, stride_row, stride_col, out_shape = _reduce_dim_strides(
        x, dim
    )

    block_m = _block_m_heuristic(block_k)
    num_warps = max(1, min(16, block_m * block_k // 256))
    num_blocks = triton.cdiv(n_rows, block_m)

    out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    _mid_kth_largest_kernel[(num_blocks,)](
        x_ptr=x_flat,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        stride_row=stride_row,
        stride_col=stride_col,
        k_val_next_p2=topk,
        k_val=k,
        block_m=block_m,
        block_k=block_k,
        k_weight=k_weight,
        num_warps=num_warps,
    )

    if len(out_shape) == 0:
        return out.squeeze()
    return out.view(out_shape)
