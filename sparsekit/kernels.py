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
_SMALL_TENSOR = 100_000
# next_power_of_2(K) above this → can't single-load, need streaming or torch
_SINGLE_LOAD_LIMIT = 1024
# next_power_of_2(k) above this with large K → torch beats streaming
_STREAMING_TOPK_LIMIT = 128


def _block_m_heuristic(block_k: int, max_registers: int = 1024) -> int:
    return int(
        max(min(max_registers // 2 ** math.ceil(math.log2(block_k)), 256), 1)
    )


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
    topk: tl.constexpr,
    kth: tl.constexpr,
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
    topk_vals = tl.topk(data, topk)

    select = tl.arange(0, topk)[None, :]
    kth_val = tl.sum(tl.where(select == (kth - 1), topk_vals, 0.0), axis=1)

    tl.store(out_ptr + row_indices, kth_val, mask=row_mask)


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
    if n_elems * num_cols < _SMALL_TENSOR or (
        block_k > _SINGLE_LOAD_LIMIT and topk > _STREAMING_TOPK_LIMIT
    ):
        return torch.kthvalue(x, num_cols - k + 1, dim=dim).values
    if block_k > _SINGLE_LOAD_LIMIT:
        return streaming_kth_largest(x, k, dim, chunk_k)

    # Single-load triton
    x_flat, n_rows, n_cols, stride_row, stride_col, out_shape = (
        _reduce_dim_strides(x, dim)
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
        topk=topk,
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
    topk: tl.constexpr,
    k_val: tl.constexpr,
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
    topk_vals = tl.topk(data, topk)

    select = tl.arange(0, topk)[None, :]
    val_k = tl.sum(tl.where(select == (k_val - 1), topk_vals, 0.0), axis=1)
    val_k1 = tl.sum(tl.where(select == k_val, topk_vals, 0.0), axis=1)

    mid = (val_k + val_k1) / 2.0
    tl.store(out_ptr + row_indices, mid, mask=row_mask)


def mid_kth_largest(
    x: Tensor,
    k: int,
    dim: int = -1,
    chunk_k: int = 1024,
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
    if n_elems * num_cols < _SMALL_TENSOR or (
        block_k > _SINGLE_LOAD_LIMIT and topk > _STREAMING_TOPK_LIMIT
    ):
        v1 = torch.kthvalue(x, num_cols - k + 1, dim=dim).values
        v2 = torch.kthvalue(x, num_cols - k, dim=dim).values
        return (v1 + v2) / 2.0
    if block_k > _SINGLE_LOAD_LIMIT:
        return streaming_mid_kth_largest(x, k, dim, chunk_k)

    # Single-load triton
    x_flat, n_rows, n_cols, stride_row, stride_col, out_shape = (
        _reduce_dim_strides(x, dim)
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
        topk=topk,
        k_val=k,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )

    if len(out_shape) == 0:
        return out.squeeze()
    return out.view(out_shape)


# ── streaming_kth_largest ────────────────────────────────────────────────


@triton.jit
def _streaming_kth_largest_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_row,
    stride_col,
    topk: tl.constexpr,
    double_topk: tl.constexpr,
    kth: tl.constexpr,
    block_m: tl.constexpr,
    chunk_k: tl.constexpr,
    n_chunks: tl.constexpr,
):
    """Streaming topk: load chunk_k cols at a time, merge with running buffer.

    Each iteration:
      1. Load chunk_k columns, topk -> local topk winners
      2. join(buf, local) -> (block_m, topk, 2)
      3. reshape -> (block_m, 2*topk)
      4. topk -> new buf (block_m, topk)
    """
    group_idx = tl.program_id(0)
    row_indices = group_idx * block_m + tl.arange(0, block_m)
    row_mask = row_indices < n_rows

    buf = tl.full([block_m, topk], float("-inf"), dtype=tl.float32)

    for c in range(n_chunks):
        col_start = c * chunk_k
        col_offsets = col_start + tl.arange(0, chunk_k)
        col_mask = col_offsets < n_cols
        mask = row_mask[:, None] & col_mask[None, :]
        ptrs = (
            row_indices[:, None] * stride_row
            + col_offsets[None, :] * stride_col
        )
        chunk = tl.load(x_ptr + ptrs, mask=mask, other=float("-inf"))

        # Local top-k from this chunk
        local_topk = tl.topk(chunk, topk)

        # Merge with running buffer: concat then topk
        joined = tl.join(buf, local_topk)
        combined = tl.reshape(joined, [block_m, double_topk])
        buf = tl.topk(combined, topk)

    # Extract k-th largest
    select = tl.arange(0, topk)[None, :]
    kth_val = tl.sum(tl.where(select == (kth - 1), buf, 0.0), axis=1)

    tl.store(out_ptr + row_indices, kth_val, mask=row_mask)


def streaming_kth_largest(
    x: Tensor,
    k: int,
    dim: int = -1,
    chunk_k: int = 1024,
) -> Tensor:
    """Streaming k-th largest for large K dimensions.

    Processes the reduction dimension in chunks of ``chunk_k`` columns,
    maintaining a running top-k buffer via repeated topk + merge.
    Register pressure is O(chunk_k) per row instead of O(K).

    Args:
        x: Input tensor of any shape.
        k: 1-based rank (k=1 is the maximum).
        dim: Dimension to reduce over (default: last).
        chunk_k: Columns per streaming chunk (default 1024).

    Returns:
        Tensor with ``dim`` removed, containing the k-th largest value.
    """
    ndim = x.ndim
    assert ndim >= 1, "Input must be at least 1-D"
    dim = dim % ndim
    n = x.shape[dim]
    assert 1 <= k <= n, f"k={k} out of range for dim size {n}"

    x_flat, n_rows, n_cols, stride_row, stride_col, out_shape = (
        _reduce_dim_strides(x, dim)
    )

    topk = triton.next_power_of_2(k)
    chunk_k = triton.next_power_of_2(max(chunk_k, topk))
    double_topk = 2 * topk
    n_chunks = triton.cdiv(n_cols, chunk_k)
    block_m = _block_m_heuristic(chunk_k)
    num_warps = max(1, min(16, block_m * chunk_k // 256))
    num_blocks = triton.cdiv(n_rows, block_m)

    out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    _streaming_kth_largest_kernel[(num_blocks,)](
        x_ptr=x_flat,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        stride_row=stride_row,
        stride_col=stride_col,
        topk=topk,
        double_topk=double_topk,
        kth=k,
        block_m=block_m,
        chunk_k=chunk_k,
        n_chunks=n_chunks,
        num_warps=num_warps,
    )

    if len(out_shape) == 0:
        return out.squeeze()
    return out.view(out_shape)


# ── streaming_mid_kth_largest ──────────────────────────────────────────


@triton.jit
def _streaming_mid_kth_largest_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_row,
    stride_col,
    topk: tl.constexpr,
    double_topk: tl.constexpr,
    k_val: tl.constexpr,
    block_m: tl.constexpr,
    chunk_k: tl.constexpr,
    n_chunks: tl.constexpr,
):
    """Streaming midpoint topk: extracts mid(k, k+1)."""
    group_idx = tl.program_id(0)
    row_indices = group_idx * block_m + tl.arange(0, block_m)
    row_mask = row_indices < n_rows

    buf = tl.full([block_m, topk], float("-inf"), dtype=tl.float32)

    for c in range(n_chunks):
        col_start = c * chunk_k
        col_offsets = col_start + tl.arange(0, chunk_k)
        col_mask = col_offsets < n_cols
        mask = row_mask[:, None] & col_mask[None, :]
        ptrs = (
            row_indices[:, None] * stride_row
            + col_offsets[None, :] * stride_col
        )
        chunk = tl.load(x_ptr + ptrs, mask=mask, other=float("-inf"))

        local_topk = tl.topk(chunk, topk)

        joined = tl.join(buf, local_topk)
        combined = tl.reshape(joined, [block_m, double_topk])
        buf = tl.topk(combined, topk)

    # Extract midpoint of k-th and (k+1)-th largest
    select = tl.arange(0, topk)[None, :]
    val_k = tl.sum(tl.where(select == (k_val - 1), buf, 0.0), axis=1)
    val_k1 = tl.sum(tl.where(select == k_val, buf, 0.0), axis=1)

    mid = (val_k + val_k1) / 2.0
    tl.store(out_ptr + row_indices, mid, mask=row_mask)


def streaming_mid_kth_largest(
    x: Tensor,
    k: int,
    dim: int = -1,
    chunk_k: int = 1024,
) -> Tensor:
    """Streaming midpoint of k-th and (k+1)-th largest for large K dimensions.

    Same streaming approach as ``streaming_kth_largest`` but returns
    the midpoint of the k-th and (k+1)-th largest values.

    Args:
        x: Input tensor of any shape.
        k: 1-based rank (k=1 gives midpoint of max and 2nd-max).
        dim: Dimension to reduce over (default: last).
        chunk_k: Columns per streaming chunk (default 1024).

    Returns:
        Tensor with ``dim`` removed.
    """
    ndim = x.ndim
    assert ndim >= 1, "Input must be at least 1-D"
    dim = dim % ndim
    n = x.shape[dim]
    assert (
        1 <= k and k + 1 <= n
    ), f"k={k} out of range: need k+1 <= dim size {n}"

    x_flat, n_rows, n_cols, stride_row, stride_col, out_shape = (
        _reduce_dim_strides(x, dim)
    )

    kp1 = k + 1
    topk = triton.next_power_of_2(kp1)
    chunk_k = triton.next_power_of_2(max(chunk_k, topk))
    double_topk = 2 * topk
    n_chunks = triton.cdiv(n_cols, chunk_k)
    block_m = _block_m_heuristic(chunk_k)
    num_warps = max(1, min(16, block_m * chunk_k // 256))
    num_blocks = triton.cdiv(n_rows, block_m)

    out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    _streaming_mid_kth_largest_kernel[(num_blocks,)](
        x_ptr=x_flat,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        stride_row=stride_row,
        stride_col=stride_col,
        topk=topk,
        double_topk=double_topk,
        k_val=k,
        block_m=block_m,
        chunk_k=chunk_k,
        n_chunks=n_chunks,
        num_warps=num_warps,
    )

    if len(out_shape) == 0:
        return out.squeeze()
    return out.view(out_shape)


# ── radix-key helpers ──────────────────────────────────────────────────


@triton.jit
def _get_topmask_and_fullmask(x):
    """Masks for IEEE-754 sign-magnitude → sortable-uint mapping."""
    tl.static_assert(
        x.dtype.is_int_unsigned(), "must be passed as unsigned bits"
    )
    tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
    fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
    tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
    fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
    return tm_arr, fm_arr


@triton.jit
def _fpval_to_key(x):
    """Map float bits (as uint) to a uint that sorts in the same order."""
    tm, fm = _get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) != 0, fm, tm)


@triton.jit
def _key_to_fpval(x):
    """Inverse of _fpval_to_key."""
    tm, fm = _get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) == 0, fm, tm)


# ── radix_kth_largest ──────────────────────────────────────────────────


@triton.jit
def _radix_kth_largest_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_row,
    stride_col,
    n_pad: tl.constexpr,
    n_act: tl.constexpr,
    kth: tl.constexpr,
    block_n: tl.constexpr,
    block_m: tl.constexpr,
):
    """Radix-key streaming k-th largest.

    Packs float bits into uint keys via _fpval_to_key so that unsigned
    comparison preserves float ordering. Merges chunks via element-wise
    tl.maximum on sorted accumulator (after tl.bitonic_merge) instead
    of join+reshape+topk.
    """
    x_nbits: tl.constexpr = x_ptr.dtype.element_ty.primitive_bitwidth
    x_utype: tl.constexpr = tl.dtype(f"uint{x_nbits}")
    if x_nbits < 16:
        y_nbits: tl.constexpr = 32
    else:
        y_nbits: tl.constexpr = x_nbits * 2
    x_ultype: tl.constexpr = tl.dtype(f"uint{y_nbits}")
    x_dtype: tl.constexpr = x_ptr.dtype.element_ty

    pid = tl.program_id(0)
    offs_m = pid * block_m + tl.arange(0, block_m)
    mask_m = offs_m[:, None] < n_rows

    # Peel first (masked) iteration from the end
    loop_iterations: tl.constexpr = n_pad // block_n - 1
    offs_n = loop_iterations * block_n + tl.arange(0, block_n)
    mask_n = offs_n[None, :] < n_cols

    x_ptrs = x_ptr + offs_m[:, None] * stride_row + offs_n[None, :] * stride_col
    x = tl.load(x_ptrs, mask=(mask_m & mask_n), other=float("-inf"))
    x = _fpval_to_key(x.to(x_utype, bitcast=True))
    x = (x.to(x_ultype) << 16) | (n_pad - offs_n)[None, :]
    acc = tl.topk(x, n_act, dim=1)

    # Stream remaining chunks right-to-left
    for _ in (tl.static_range if loop_iterations <= 4 else range)(
        loop_iterations
    ):
        acc = tl.bitonic_merge(acc)
        x_ptrs -= block_n * stride_col
        offs_n -= block_n
        x = tl.load(x_ptrs, mask=mask_m, other=float("-inf"))
        x = _fpval_to_key(x.to(x_utype, bitcast=True))
        x = (x.to(x_ultype) << 16) | (n_pad - offs_n)[None, :]
        acc = tl.maximum(acc, tl.topk(x, n_act, dim=1))

    # Final sort descending, unpack values
    acc = tl.sort(acc, dim=1, descending=True)
    y_values_raw = (acc >> 16).to(x_utype)
    y_values = _key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)

    # k-th largest (0-indexed position kth-1 in descending-sorted top-n_act)
    select = tl.arange(0, n_act)[None, :]
    kth_val = tl.sum(tl.where(select == (kth - 1), y_values, 0.0), axis=1)

    tl.store(out_ptr + offs_m, kth_val, mask=offs_m < n_rows)


def radix_kth_largest(
    x: Tensor,
    k: int,
    dim: int = -1,
    block_n: int = 1024,
) -> Tensor:
    """Radix-key streaming k-th largest for large K dimensions.

    Uses the same approach as MoE routing top-k (topk_triton.py):
    packs float bits into sortable unsigned keys, streams chunks
    right-to-left, merges via tl.maximum on bitonic-merged accumulator.

    Args:
        x: Input tensor of any shape.
        k: 1-based rank (k=1 is the maximum).
        dim: Dimension to reduce over (default: last).
        block_n: Columns per streaming chunk (default 1024).

    Returns:
        Tensor with ``dim`` removed, containing the k-th largest value.
    """
    ndim = x.ndim
    assert ndim >= 1, "Input must be at least 1-D"
    dim = dim % ndim
    n = x.shape[dim]
    assert 1 <= k <= n, f"k={k} out of range for dim size {n}"

    x_flat, n_rows, n_cols, stride_row, stride_col, out_shape = (
        _reduce_dim_strides(x, dim)
    )

    n_act = triton.next_power_of_2(k)
    block_n = triton.next_power_of_2(max(block_n, n_act))
    n_pad = triton.cdiv(n_cols, block_n) * block_n
    block_m = _block_m_heuristic(block_n)
    num_warps = max(1, min(16, block_m * block_n // 256))
    num_blocks = triton.cdiv(n_rows, block_m)

    out = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    _radix_kth_largest_kernel[(num_blocks,)](
        x_ptr=x_flat,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
        stride_row=stride_row,
        stride_col=stride_col,
        n_pad=n_pad,
        n_act=n_act,
        kth=k,
        block_n=block_n,
        block_m=block_m,
        num_warps=num_warps,
    )

    if len(out_shape) == 0:
        return out.squeeze()
    return out.view(out_shape)
