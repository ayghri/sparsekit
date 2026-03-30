"""
Sparse MMA in Triton: A[16,256] × B[256,16] → C[16,16]

B has 1:2 structured sparsity on columns (N dimension):
  For each pair (col_j, col_{j+8}), j=0..7, exactly one is non-zero.

Storage:
  B_compressed: [256, 8]   — only the non-zero columns
  masks:        [16]        — one uint8 per 16-row K-tile (16 tiles × 16 rows = 256)

Each mask byte encodes sparsity for a [16, 8] compressed tile:
  bit i = 0  →  column i   is non-zero
  bit i = 1  →  column i+8 is non-zero

Kernel strategy (single program, no grid needed for one tile):
  - Two accumulators: acc_lo[16,8] (cols 0-7), acc_hi[16,8] (cols 8-15)
  - Loop over 16 K-tiles:
      partial = A_tile[16,16] @ B_comp_tile[16,8]  →  [16,8]
      acc_lo += where(bit == 0, partial, 0)
      acc_hi += where(bit == 1, partial, 0)
  - Store [acc_lo | acc_hi] → C[16,16]

No scatter instructions. No atomics. The paired structure (offset by 8)
means column j in the partial always maps to index j in either the lo or
hi half — so tl.where is all we need.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Compression utility
# ---------------------------------------------------------------------------


def compress_b(B: torch.Tensor):
    """
    B: [K, N], N-dim has 1:2 column sparsity (pairs at offset 8).
    Returns B_comp [K, N//2], masks [N_tiles * K_tiles] stored as [K//16]
    per 16-column N-tile.

    For simplicity here N=16, so one column-tile, and K//16 mask bytes.
    """
    K, N = B.shape
    assert N % 16 == 0 and K % 16 == 0

    n_tiles = N // 16
    k_tiles = K // 16
    B_comp = torch.empty(K, N // 2, dtype=B.dtype, device=B.device)
    # masks laid out as [n_tile, k_tile] but for N=16 it's just [k_tiles]
    masks = torch.empty(n_tiles * k_tiles, dtype=torch.uint8, device=B.device)

    for nt in range(n_tiles):
        col_base = nt * 16
        for kt in range(k_tiles):
            row_base = kt * 16
            mask_val = 0
            for i in range(8):
                col_a = B[row_base : row_base + 16, col_base + i]
                col_b = B[row_base : row_base + 16, col_base + i + 8]

                a_zero = (col_a == 0).all()
                b_zero = (col_b == 0).all()
                assert a_zero or b_zero, (
                    f"K-tile {kt}, pair ({col_base+i},{col_base+i+8}): "
                    "one column must be all-zero"
                )

                if b_zero:
                    B_comp[row_base : row_base + 16, nt * 8 + i] = col_a
                else:
                    mask_val |= 1 << i
                    B_comp[row_base : row_base + 16, nt * 8 + i] = col_b

            masks[nt * k_tiles + kt] = mask_val

    return B_comp, masks


# ---------------------------------------------------------------------------
# Core kernel: A[16,256] × B_sparse[256,16] → C[16,16]
# ---------------------------------------------------------------------------


# @triton.jit
# def sparse_mma_kernel(
#     a_ptr,
#     b_comp_ptr,
#     c_ptr,
#     mask_ptr,
#     stride_am,
#     stride_ak,
#     stride_bk,
#     stride_bn,
#     stride_cm,
#     stride_cn,
#     NUM_K_TILES: tl.constexpr,
# ):
#     """
#     C[16,16] = A[16, K] @ B_sparse[K, 16]
#
#     B stored as B_comp[K, 8] + masks[K//16].
#     Uses split accumulators to avoid scatter.
#     """
#     offs_m = tl.arange(0, 16)
#     offs_8 = tl.arange(0, 8)
#
#     # acc_low = tl.zeros((16, 8), dtype = c_ptr.dtype)
#     acc_high = tl.zeros((16, 8), dtype=tl.float32)
#
#     for tile in range(NUM_K_TILES):
#         # ---- Load sparsity mask for this K-tile (1 byte) ----
#         mask_val = tl.load(mask_ptr + tile).to(tl.int32)
#         bits = (mask_val >> offs_8) & 1  # [8], each 0 or 1
#
#         # ---- Load A tile [16, 16] ----
#         k_base = tile * 16
#         offs_k = k_base + tl.arange(0, 16)
#         a = tl.load(
#             a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
#         )
#
#         # ---- Load B_comp tile [16, 8] ----
#         b = tl.load(
#             b_comp_ptr
#             + offs_k[:, None] * stride_bk
#             + offs_8[None, :] * stride_bn
#         )
#
#         # ---- MMA: [16,16] @ [16,8] = [16,8] ----
#         partial = tl.dot(a, b)  # [16, 8], fp32
#
#         # ---- Route to correct half using the mask ----
#         # bit=0 → column j lives in lo half → add to acc_lo
#         # bit=1 → column j+8 lives in hi half → add to acc_hi
#         is_lo = bits[None, :] == 0  # [1, 8] broadcast to [16, 8]
#         acc_low += tl.where(is_lo, partial, 0.0)
#         acc_high += tl.where(is_lo, 0.0, partial)
#
#     # ---- Store C[16,16] = [acc_lo | acc_hi] ----
#     offs_lo = tl.arange(0, 8)  # columns 0..7
#     offs_hi = tl.arange(0, 8) + 8  # columns 8..15
#
#     tl.store(
#         c_ptr + offs_m[:, None] * stride_cm + offs_lo[None, :] * stride_cn,
#         acc_low,
#     )
#     tl.store(
#         c_ptr + offs_m[:, None] * stride_cm + offs_hi[None, :] * stride_cn,
#         acc_high,
#     )
#


@triton.jit
def sparse_matmul_kernel(
    a_ptr,
    b_comp_ptr,
    c_ptr,
    mask_ptr,
    M,
    # K,
    # N: tl.constexpr,  # must be multiple of 16
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    NUM_K_TILES: tl.constexpr,  # K // 16
    block_m: tl.constexpr,
):
    """
    C[M, N] = A[M, K] @ B_sparse[K, N]

    Grid: (M // BLOCK_M,  N // 16)
    Each program handles BLOCK_M rows × 16 output columns.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)  # which 16-column tile of N

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    mask_m = offs_m < M
    offs_8 = tl.arange(0, 8)

    # Mask array for this N-tile: masks are laid out as [n_tile, k_tile]
    mask_base = pid_n * NUM_K_TILES

    # Compressed B column offset for this N-tile
    # acc_hi = tl.zeros((block_m, 8), dtype=tl.float32)
    # acc_low=tl.zeros((block_m, 10) , dttype=tl.float16)
    comp_col_base = pid_n * 8

    for tile in range(NUM_K_TILES):
        mask_val = tl.load(mask_ptr + mask_base + tile).to(tl.int32)
        bits = (mask_val >> offs_8) & 1

        k_base = tile * 16
        offs_k = k_base + tl.arange(0, 16)

        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None],
            other=0.0,
        )

        b = tl.load(
            b_comp_ptr
            + offs_k[:, None] * stride_bk
            + (comp_col_base + offs_8)[None, :] * stride_bn,
        )

        partial = tl.dot(a, b)

        is_lo = bits[None, :] == 0
        acc_lo += tl.where(is_lo, partial, 0.0)
        acc_hi += tl.where(is_lo, 0.0, partial)

    # Store
    n_base = pid_n * 16
    offs_lo = n_base + tl.arange(0, 8)
    offs_hi = n_base + tl.arange(0, 8) + 8

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_lo[None, :] * stride_cn,
        acc_lo,
        mask=mask_m[:, None],
    )
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_hi[None, :] * stride_cn,
        acc_hi,
        mask=mask_m[:, None],
    )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def sparse_mma_16x256x16(A, B_comp, masks):
    """A[16,256] × B_sparse[256,16] → C[16,16]"""
    M, K = A.shape
    assert M == 16 and B_comp.shape == (K, 8) and masks.shape[0] == K // 16

    C = torch.empty(16, 16, dtype=torch.float32, device=A.device)

    sparse_mma_kernel[(1,)](
        A,
        B_comp,
        C,
        masks,
        K,
        A.stride(0),
        A.stride(1),
        B_comp.stride(0),
        B_comp.stride(1),
        C.stride(0),
        C.stride(1),
        NUM_K_TILES=K // 16,
    )
    return C


def sparse_matmul(A, B_comp, masks, N_orig, K_orig):
    """General: A[M, K] × B_sparse[K, N] → C[M, N]"""
    M, K = A.shape
    assert K == K_orig
    assert N_orig % 16 == 0

    C = torch.empty(M, N_orig, dtype=torch.float32, device=A.device)
    num_k_tiles = K // 16
    BLOCK_M = 16

    grid = (triton.cdiv(M, BLOCK_M), N_orig // 16)

    sparse_matmul_kernel[grid](
        A,
        B_comp,
        C,
        masks,
        M,
        K,
        N_orig,
        A.stride(0),
        A.stride(1),
        B_comp.stride(0),
        B_comp.stride(1),
        C.stride(0),
        C.stride(1),
        NUM_K_TILES=num_k_tiles,
        block_m=BLOCK_M,
    )
    return C


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def make_sparse_b(K, N, device="cuda", dtype=torch.float16):
    """B [K, N] with 1:2 column sparsity, pattern can vary per K-tile."""
    B = torch.randn(K, N, device=device, dtype=dtype)
    for nt in range(N // 16):
        col_base = nt * 16
        for kt in range(K // 16):
            row_base = kt * 16
            for i in range(8):
                if torch.randint(0, 2, (1,)).item() == 0:
                    B[row_base : row_base + 16, col_base + i + 8] = 0
                else:
                    B[row_base : row_base + 16, col_base + i] = 0
    return B


def test_single():
    torch.manual_seed(42)
    dev, dt = "cuda", torch.float16
    M, K, N = 16, 256, 16

    A = torch.randn(M, K, device=dev, dtype=dt)
    B = make_sparse_b(K, N, device=dev, dtype=dt)

    C_ref = A.float() @ B.float()

    B_comp, masks = compress_b(B)
    print(f"B_comp shape: {B_comp.shape}  masks shape: {masks.shape}")
    assert B_comp.shape == (K, N // 2)

    C_sparse = sparse_mma_16x256x16(A, B_comp, masks)

    err = (C_sparse - C_ref).abs().max().item()
    rel = err / C_ref.abs().max().item()
    print(f"[16×256 @ 256×16]  max_err={err:.4e}  rel_err={rel:.4e}")
    assert rel < 1e-3, f"Relative error too large: {rel}"
    print("PASSED\n")


def test_larger():
    torch.manual_seed(99)
    dev, dt = "cuda", torch.float16
    M, K, N = 64, 256, 64

    A = torch.randn(M, K, device=dev, dtype=dt)
    B = make_sparse_b(K, N, device=dev, dtype=dt)

    C_ref = A.float() @ B.float()

    B_comp, masks = compress_b(B)
    C_sparse = sparse_matmul(A, B_comp, masks, N, K)

    err = (C_sparse - C_ref).abs().max().item()
    rel = err / C_ref.abs().max().item()
    print(f"[{M}×{K} @ {K}×{N}]  max_err={err:.4e}  rel_err={rel:.4e}")
    assert rel < 1e-3, f"Relative error too large: {rel}"
    print("PASSED\n")


if __name__ == "__main__":
    test_single()
    test_larger()
