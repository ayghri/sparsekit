import torch
import triton
import triton.language as tl


@triton.jit
def top_kth_mid_kernel(
    x_ptr,
    mid_ptr,
    kth,
    stride_k,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_K)
    row_indices = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)

    ptr_offsets = row_indices[:, None] * stride_k + col_offsets[None, :]

    row_mask = row_indices < n_rows  # [BLOCK_M]
    col_mask = col_offsets < n_cols  # [BLOCK_K]
    mask = row_mask[:, None] & col_mask[None, :]  # [BLOCK_M, BLOCK_K]
    data = tl.load(x_ptr + ptr_offsets, mask=mask, other=float("inf"))

    # sorted_data = tl.topk(data, kth, dim=-1, descending=False)
    topk = tl.topk(data, k=kth, dim=1)

    val_k = tl.sum(
        tl.where(col_offsets[None, :] == (kth - 1), topk, 0.0), axis=1
    )
    val_k1 = tl.sum(tl.where(col_offsets[None, :] == kth, topk, 0.0), axis=1)

    mid_k = (val_k + val_k1) / 2.0

    out_offsets = row_indices
    tl.store(mid_ptr + out_offsets, mid_k, mask=row_mask)


def get_k_and_k1_triton(x: torch.Tensor, k: int):
    """
    Compute the midpoint between k-th and (k+1)-th smallest values along the last dimension.

    Args:
        x: Input tensor of any shape (..., n_cols)
        k: The rank (1-based) for kth smallest value

    Returns:
        Tensor of shape (...) containing (kth + (k+1)th) / 2 for each row
    """
    original_shape = x.shape
    cols = original_shape[-1]

    assert k >= 1, f"k must be >= 1, got {k}"
    assert k + 1 <= cols, f"k+1={k + 1} exceeds n_cols={cols}"

    x_flat = x.view(-1, cols)
    rows = x_flat.shape[0]

    # Determine block sizes
    BLOCK_K = triton.next_power_of_2(cols)

    # Choose BLOCK_M based on BLOCK_K to balance occupancy
    # Constraint: BLOCK_M * BLOCK_K should not exceed ~4096 elements for 2D sort
    # to avoid register pressure issues
    if BLOCK_K <= 4:
        BLOCK_M = 256
    elif BLOCK_K <= 8:
        BLOCK_M = 128
    elif BLOCK_K <= 16:
        BLOCK_M = 64
    elif BLOCK_K <= 32:
        BLOCK_M = 32
    elif BLOCK_K <= 64:
        BLOCK_M = 16
    elif BLOCK_K <= 128:
        BLOCK_M = 8
    elif BLOCK_K <= 256:
        BLOCK_M = 4
    elif BLOCK_K <= 512:
        BLOCK_M = 2
    else:
        # For BLOCK_K >= 1024, use BLOCK_M=1 (one row per program)
        BLOCK_M = 1

    # Allocate output buffer
    mid_k = torch.empty(rows, device=x.device, dtype=x.dtype)

    # Number of row blocks
    num_blocks = triton.cdiv(rows, BLOCK_M)

    # Heuristics for num_warps
    num_warps = max(1, min(16, BLOCK_M * BLOCK_K // 256))

    grid = (num_blocks,)

    top_kth_mid_kernel[grid](
        x_ptr=x_flat,
        mid_ptr=mid_k,
        kth=k,
        stride_k=x_flat.stride(0),
        n_rows=rows,
        n_cols=cols,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,  # type: ignore
    )

    # Reshape output to match input shape (minus last dimension)
    output_shape = original_shape[:-1]
    if len(output_shape) == 0:
        # Input was 1D, return scalar
        return mid_k.squeeze()
    return mid_k.view(output_shape)


@triton.jit
def kth_largest_kernel(
    x_ptr,
    kth_ptr,
    n_rows,
    n_cols,
    stride_k,
    kth: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_K)
    row_indices = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)

    ptr_offsets = row_indices[:, None] * stride_k + col_offsets[None, :]

    row_mask = row_indices < n_rows
    col_mask = col_offsets < n_cols
    mask = row_mask[:, None] & col_mask[None, :]

    data = tl.load(x_ptr + ptr_offsets, mask=mask, other=float("-inf"))
    data = tl.topk(data, kth)

    # After topk, data has shape (BLOCK_M, kth), so use kth for indexing
    select_offsets = tl.arange(0, kth)[None, :]

    kth_value = tl.sum(tl.where(select_offsets == (kth - 1), data, 0.0), axis=1)

    tl.store(kth_ptr + row_indices, kth_value, mask=row_mask)


def mid_k_triton(x: torch.Tensor, k: int):
    """
    Compute the midpoint between k-th and (k+1)-th smallest values along the last dimension.

    Args:
        x: Input tensor of any shape (..., n_cols)
        k: The rank (1-based) for kth smallest value

    Returns:
        Tensor of shape (...) containing (kth + (k+1)th) / 2 for each row
    """
    original_shape = x.shape
    cols = original_shape[-1]

    assert k >= 1, f"k must be >= 1, got {k}"
    assert k + 1 <= cols, f"k+1={k + 1} exceeds n_cols={cols}"

    x_flat = x.view(-1, cols)
    rows = x_flat.shape[0]
    k = k + 1

    BLOCK_K = triton.next_power_of_2(cols)
    kth_nearst_pow2 = triton.next_power_of_2(k)

    if BLOCK_K <= 4:
        BLOCK_M = 256
    elif BLOCK_K <= 8:
        BLOCK_M = 128
    elif BLOCK_K <= 16:
        BLOCK_M = 64
    elif BLOCK_K <= 32:
        BLOCK_M = 32
    elif BLOCK_K <= 64:
        BLOCK_M = 16
    elif BLOCK_K <= 128:
        BLOCK_M = 8
    elif BLOCK_K <= 256:
        BLOCK_M = 4
    elif BLOCK_K <= 512:
        BLOCK_M = 2
    else:
        # For BLOCK_K >= 1024, use BLOCK_M=1 (one row per program)
        BLOCK_M = 1

    # Allocate output buffer
    mid_k = torch.empty(rows, device=x.device, dtype=x.dtype)

    # Number of row blocks
    num_blocks = triton.cdiv(rows, BLOCK_M)

    # # Heuristics for num_warps
    num_warps = max(1, min(16, BLOCK_M * BLOCK_K // 256))

    grid = (num_blocks,)
    # grid = lambda meta: (triton.cdiv(rows, meta['BLOCK_M']),)

    mid_k_kernel[grid](
        x_ptr=x_flat,
        mid_ptr=mid_k,
        kth=k,
        stride_k=x_flat.stride(0),
        n_rows=rows,
        n_cols=cols,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        kth_nearst_pow2=kth_nearst_pow2,
        num_warps=num_warps,
    )

    # Reshape output to match input shape (minus last dimension)
    output_shape = original_shape[:-1]
    if len(output_shape) == 0:
        # Input was 1D, return scalar
        return mid_k.squeeze()
    return mid_k.view(output_shape)


def kth_largest(x: torch.Tensor, k: int, dim: int = -1):
    """
    Compute the midpoint between k-th and (k+1)-th smallest values along the last dimension.

    Args:
        x: Input tensor of any shape (..., n_cols)
        k: The rank (1-based) for kth smallest value

    Returns:
        Tensor of shape (...) containing (kth + (k+1)th) / 2 for each row
    """
    original_shape = x.shape
    cols = original_shape[-1]
    assert dim == -1, "Only last dimension is supported"
    assert k >= 1, f"k must be >= 1, got {k}"
    assert k + 1 <= cols, f"k+1={k + 1} exceeds n_cols={cols}"

    x_flat = x.view(-1, cols)
    rows = x_flat.shape[0]

    BLOCK_K = triton.next_power_of_2(cols)

    if BLOCK_K <= 4:
        BLOCK_M = 256
    elif BLOCK_K <= 8:
        BLOCK_M = 128
    elif BLOCK_K <= 16:
        BLOCK_M = 64
    elif BLOCK_K <= 32:
        BLOCK_M = 32
    elif BLOCK_K <= 64:
        BLOCK_M = 16
    elif BLOCK_K <= 128:
        BLOCK_M = 8
    elif BLOCK_K <= 256:
        BLOCK_M = 4
    elif BLOCK_K <= 512:
        BLOCK_M = 2
    else:
        # For BLOCK_K >= 1024, use BLOCK_M=1 (one row per program)
        BLOCK_M = 1

    # # Heuristics for num_warps
    num_warps = max(1, min(16, BLOCK_M * BLOCK_K // 256))

    # Allocate output buffer
    kthlargest_vals = torch.empty(rows, device=x.device, dtype=x.dtype)

    # Number of row blocks
    num_blocks = triton.cdiv(rows, BLOCK_M)

    # # Heuristics for num_warps
    num_warps = max(1, min(16, BLOCK_M * BLOCK_K // 256))

    grid = (num_blocks,)
    # grid = lambda meta: (triton.cdiv(rows, meta['BLOCK_M']),)

    kth_largest_kernel[grid](
        x_ptr=x_flat,
        kth_ptr=kthlargest_vals,
        n_rows=rows,
        n_cols=cols,
        kth=k,
        stride_k=x_flat.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )

    # Reshape output to match input shape (minus last dimension)
    output_shape = original_shape[:-1]
    if len(output_shape) == 0:
        # Input was 1D, return scalar
        return kthlargest_vals.squeeze()
    return kthlargest_vals.view(output_shape)
