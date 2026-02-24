"""
Copyright (c) 2025 Ayoub Ghriss and contributors
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.

FP4 quantization with OBS full-column compensation.

Progressive block quantization: each block of contiguous columns is quantized
to FP4 (shared scale per block), with quantization error compensated across all
K columns using C = H^{-1} (frozen inverse Hessian).

Supports MXFP4 (UE8M0 scale) and NVFP4 (E4M3 scale).
"""

from typing import Callable

import torch
import torch.linalg as LA
from torch import Tensor

from .obs import StructuredOBS


# MXFP4 codebook (doubled values to avoid half-integer arithmetic)
_DOUBLED_MXFP4 = torch.tensor(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=torch.float32,
)


def mxfp4_quantize(W: Tensor, block_size: int = 16) -> Tensor:
    """Simulate MXFP4 quantization with UE8M0 shared exponent.

    Each block of `block_size` contiguous columns shares a single scale
    derived from the block's absolute maximum:
        scale = 2^(floor(log2(amax)) - 3)

    Values are rounded to the nearest point in the MXFP4 codebook
    {0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-12} * scale.

    Args:
        W: (M, K) weight tensor (K must be divisible by block_size).
        block_size: Number of columns per quantization block.

    Returns:
        Dequantized tensor (same shape as W).
    """
    M, K = W.shape
    assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"
    B = K // block_size
    device = W.device

    codebook = _DOUBLED_MXFP4.to(device=device, dtype=torch.float32)

    x = W.float().view(M, B, block_size)

    # UE8M0 scale: 2^(floor(log2(amax)) - 3)
    amax = x.abs().amax(dim=-1)  # (M, B)
    safe_amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    scale_exp = safe_amax.log2().floor() - 3.0
    scale = torch.pow(2.0, scale_exp)  # (M, B)
    scale = torch.where(amax > 0, scale, torch.zeros_like(scale))

    # Round to nearest codebook value
    # possible_values: (M, B, 1, 16), x: (M, B, block_size, 1)
    possible = (scale.unsqueeze(-1) * codebook).unsqueeze(-2)  # (M, B, 1, 16)
    deltas = x.unsqueeze(-1) - possible  # (M, B, block_size, 16)
    best_idx = deltas.abs().argmin(dim=-1)  # (M, B, block_size)
    quant_doubled = codebook[best_idx]  # (M, B, block_size)
    dequant = scale.unsqueeze(-1) * quant_doubled  # (M, B, block_size)

    return dequant.view(M, K).to(W.dtype)


def _gptq_core(
    W: Tensor,
    H: Tensor,
    fp4_block_size: int,
    codebook: Tensor,
    scale_fn: Callable[[Tensor], Tensor],
    damp: float,
) -> Tensor:
    """Column-by-column quantization with Cholesky-based forward propagation.

    Processes ALL K columns in one pass — no 128-column window limitation.
    W and H should already be column-permuted if non-standard order is desired.

    Args:
        W: (M, K) weight tensor.
        H: (K, K) Hessian.
        fp4_block_size: FP4 block size for scale computation.
        codebook: (N,) codebook tensor.
        scale_fn: (M, block_size) -> (M,) computes per-row scale.
        damp: Damping factor for H.

    Returns:
        (M, K) quantized tensor.
    """
    M, K = W.shape
    device = W.device
    W = W.clone().float()
    H = H.clone().float()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    damp_val = damp * torch.mean(torch.diag(H))
    diag_idx = torch.arange(K, device=device)
    H[diag_idx, diag_idx] += damp_val

    L = LA.cholesky(H)
    Hinv_full = torch.cholesky_inverse(L)
    Hinv = LA.cholesky(Hinv_full, upper=True)

    Q = torch.zeros_like(W)
    scale = torch.zeros(M, device=device)

    for i in range(K):
        if i % fp4_block_size == 0:
            end = min(i + fp4_block_size, K)
            scale = scale_fn(W[:, i:end])

        w = W[:, i]
        q = _round_to_codebook(w, scale, codebook)
        Q[:, i] = q

        err = (w - q) / Hinv[i, i]
        W[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)

    return Q


def _mxfp4_block_scale(block_vals: Tensor) -> Tensor:
    """Compute UE8M0 (power-of-2) scale from block values.

    Args:
        block_vals: (M, block_size) current column values.

    Returns:
        (M,) per-row scale.
    """
    amax = block_vals.abs().amax(dim=-1)  # (M,)
    safe_amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    scale = torch.pow(2.0, safe_amax.log2().floor() - 3.0)
    return torch.where(amax > 0, scale, torch.zeros_like(scale))


@torch.no_grad()
def quantize_obs(
    W: Tensor,
    H: Tensor,
    block_size: int = 16,
    damp: float = 1e-4,
    C: Tensor | None = None,
    order: str = "largest_first",
    quantize_fn: Callable[[Tensor], Tensor] | None = None,
    mask: Tensor | None = None,
    codebook: Tensor | None = None,
    scale_fn: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Block quantization with OBS full-row compensation via frozen C = H^{-1}.

    For each FP4 block P (16 columns):
      1. Quantize: Q_P = fp4(W[:, P])
      2. Error: E_P = W[:, P] - Q_P
      3. Compensation: delta = E_P @ inv(C_PP) @ C[P, :]
      4. Update ALL remaining unquantized columns: W -= delta

    Unlike GPTQ (which compensates forward within 128-column windows via
    Cholesky), this compensates bidirectionally across ALL K columns using
    the full inverse Hessian.

    Args:
        W: (M, K) weight tensor (modified in-place).
        H: (K, K) Hessian matrix (X^T X / N).
        block_size: Quantization block size (default 16).
        damp: Damping factor for H regularization.
        C: Precomputed (K, K) damped inverse. Computed if None.
        order: Block processing order:
            "largest_first" — blocks with highest quant loss first
            "smallest_first" — blocks with lowest quant loss first
            "left_to_right" — natural column order (like GPTQ)
        quantize_fn: Block quantize function for ordering scores.
            (M, block_size) -> (M, block_size). Defaults to MXFP4.
        mask: (M, K) bool tensor. True = active, False = frozen.
        codebook: (N,) codebook tensor. Defaults to _DOUBLED_MXFP4.
        scale_fn: (M, block_size) -> (M,) computes per-row scale from
            block values. Defaults to UE8M0 (MXFP4).

    Returns:
        The quantized weight tensor.
    """
    is_param = isinstance(W, torch.nn.Parameter)
    W_data = W.data if is_param else W
    M, K = W_data.shape
    assert K % block_size == 0
    B = K // block_size
    device = W_data.device

    if C is None:
        C = StructuredOBS.compute_inverse(H, damp)

    W_work = W_data.clone().float()
    if codebook is None:
        codebook = _DOUBLED_MXFP4
    codebook = codebook.to(device=device, dtype=torch.float32)
    if scale_fn is None:
        scale_fn = _mxfp4_block_scale

    # Block column indices: (B, block_size)
    block_cols = torch.arange(K, device=device).view(B, block_size)

    eye_bs = 1e-8 * torch.eye(block_size, device=device)

    # Determine processing order using initial C
    if order == "left_to_right":
        block_order = torch.arange(B, device=device)
    else:
        if quantize_fn is None:
            qfn = lambda w: _quantize_block(w, w.device)
        else:
            qfn = quantize_fn
        Q_init = _full_quantize(W_work, block_size, qfn)
        E_init = W_work - Q_init

        C_PP_all = C[block_cols[:, :, None], block_cols[:, None, :]]
        C_PP_inv = LA.inv(C_PP_all + eye_bs)
        E_blocks = E_init[:, block_cols.reshape(-1)].view(M, B, block_size)
        temp = torch.einsum("mbp,bpq->mbq", E_blocks, C_PP_inv)
        scores = (temp * E_blocks).sum(dim=2).sum(dim=0)

        if order == "largest_first":
            block_order = torch.argsort(scores, descending=True)
        elif order == "smallest_first":
            block_order = torch.argsort(scores, descending=False)
        else:
            raise ValueError(f"Unknown order={order!r}")

    # Sequential block processing with updated C
    active = torch.ones(K, dtype=torch.bool, device=device)

    for idx in range(B):
        b = block_order[idx].item()
        cols = block_cols[b]

        # Current active columns and C restricted to them
        active_cols = torch.where(active)[0]
        n_active = active_cols.shape[0]

        if idx == 0:
            C_act = C
        else:
            H_aa = H[active_cols[:, None], active_cols[None, :]]
            C_act = StructuredOBS.compute_inverse(H_aa, damp)

        # Map block cols to local indices in active space
        abs_to_local = torch.full((K,), -1, dtype=torch.long, device=device)
        abs_to_local[active_cols] = torch.arange(n_active, device=device)
        local_cols = abs_to_local[cols]  # (bs,)

        # 1. Quantize block from current values
        W_P = W_work[:, cols]  # (M, bs)
        scale = scale_fn(W_P)
        possible = (scale.unsqueeze(-1) * codebook).unsqueeze(-2)
        best_idx = (W_P.unsqueeze(-1) - possible).abs().argmin(dim=-1)
        Q_P = scale.unsqueeze(-1) * codebook[best_idx]

        # 2. OBS compensation: delta = E_P @ inv(C_PP) @ C[P, active]
        E_P = W_P - Q_P  # (M, bs)
        C_PP = C_act[local_cols[:, None], local_cols[None, :]]  # (bs, bs)
        C_PP_inv = LA.inv(C_PP + eye_bs)
        C_P_act = C_act[local_cols, :]  # (bs, n_active)
        comp = C_PP_inv @ C_P_act  # (bs, n_active)

        delta = E_P @ comp  # (M, n_active)

        # Apply to all active columns except current block
        if n_active == K:
            W_work -= delta
        else:
            W_work.scatter_add_(
                1, active_cols.unsqueeze(0).expand(M, -1), -delta
            )

        # 3. Fix quantized block, mark inactive
        W_work[:, cols] = Q_P
        active[cols] = False

    if is_param:
        W.data.copy_(W_work)
    else:
        W.copy_(W_work)
    return W_work


def _round_to_codebook(w: Tensor, scale: Tensor, codebook: Tensor) -> Tensor:
    """Round each element to nearest FP4 codebook value.

    w: (M,) values, scale: (M,) per-row scale.
    Returns: (M,) dequantized values.
    """
    possible = scale.unsqueeze(-1) * codebook  # (M, 16)
    idx = (w.unsqueeze(-1) - possible).abs().argmin(dim=-1)
    return scale * codebook[idx]


def _full_quantize(
    W: Tensor, block_size: int, quantize_fn: Callable[[Tensor], Tensor]
) -> Tensor:
    """Apply quantize_fn block-by-block across all columns."""
    M, K = W.shape
    B = K // block_size
    out = torch.empty_like(W)
    for b in range(B):
        c0 = b * block_size
        out[:, c0:c0 + block_size] = quantize_fn(W[:, c0:c0 + block_size])
    return out


def _quantize_block(W_P: Tensor, device: torch.device) -> Tensor:
    """Quantize a single block of columns to MXFP4.

    W_P: (M, block_size) float tensor.
    Returns: (M, block_size) dequantized values.
    """
    codebook = _DOUBLED_MXFP4.to(device=device, dtype=torch.float32)
    M, bs = W_P.shape

    amax = W_P.abs().amax(dim=-1)  # (M,)
    safe_amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    scale_exp = safe_amax.log2().floor() - 3.0
    scale = torch.pow(2.0, scale_exp)
    scale = torch.where(amax > 0, scale, torch.zeros_like(scale))  # (M,)

    # possible: (M, 1, 16), W_P: (M, bs, 1)
    possible = (scale.unsqueeze(-1) * codebook).unsqueeze(-2)  # (M, 1, 16)
    deltas = W_P.unsqueeze(-1) - possible  # (M, bs, 16)
    best_idx = deltas.abs().argmin(dim=-1)  # (M, bs)
    quant_doubled = codebook[best_idx]
    return scale.unsqueeze(-1) * quant_doubled  # (M, bs)
