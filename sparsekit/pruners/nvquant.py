# Copyright (c) 2025 Anonymous Authors
# Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""NVFP4 quantization: FP4 (E2M1) values with E4M3 FP8 per-block scales.

Compared to MXFP4 (UE8M0 scale = pure power of 2), the E4M3 scale has
3 mantissa bits, allowing finer-grained scale selection that better matches
the block's dynamic range.
"""

import torch
from torch import Tensor

from .quant import quantize_obs as _quantize_obs


# FP4 E2M1 codebook (doubled to avoid half-integer values)
_DOUBLED_FP4 = torch.tensor(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=torch.float32,
)

# Max absolute doubled codebook value
_DOUBLED_FP4_MAX = 12.0


def _build_e4m3_table(device: torch.device) -> Tensor:
    """Build a sorted tensor of all positive E4M3fn FP8 values.

    E4M3fn (NVIDIA convention, no NaN):
      - 4 exponent bits, 3 mantissa bits, bias = 7
      - Subnormal (e=0): 2^(-6) * m/8,  m in [1..7]
      - Normal (e=1..15): 2^(e-7) * (1 + m/8),  m in [0..7]
      - Max = 2^8 * 1.875 = 448 (no NaN, e=15 m=7 is valid)

    Returns:
        Sorted 1D tensor of unique positive E4M3fn values (~120 values).
    """
    vals = []
    # Subnormals: e=0, m=1..7
    for m in range(1, 8):
        vals.append(2.0 ** (-6) * m / 8.0)
    # Normals: e=1..15, m=0..7
    for e in range(1, 16):
        for m in range(8):
            vals.append(2.0 ** (e - 7) * (1.0 + m / 8.0))
    vals = sorted(set(vals))
    return torch.tensor(vals, dtype=torch.float32, device=device)


def _round_to_e4m3(x: Tensor, e4m3_table: Tensor) -> Tensor:
    """Round each element of x to the nearest E4M3fn value.

    Args:
        x: (...,) positive float tensor.
        e4m3_table: (N,) sorted positive E4M3fn values.

    Returns:
        Same shape as x, with each value snapped to nearest E4M3fn.
    """
    # searchsorted finds insertion point; compare left and right neighbors
    flat = x.reshape(-1)
    idx = torch.searchsorted(e4m3_table, flat)
    idx = idx.clamp(1, e4m3_table.shape[0] - 1)

    left = e4m3_table[idx - 1]
    right = e4m3_table[idx]
    use_left = (flat - left).abs() <= (right - flat).abs()
    result = torch.where(use_left, left, right)
    return result.view(x.shape)


def nvfp4_quantize(W: Tensor, block_size: int = 16) -> Tensor:
    """Simulate NVFP4 quantization with E4M3 FP8 per-block scale.

    Each block of `block_size` contiguous columns gets a scale factor
    that is the nearest E4M3 FP8 value to (amax / 6), where 6 is the
    max representable FP4 E2M1 value. This allows finer scale granularity
    than MXFP4's power-of-2 (UE8M0) scales.

    Args:
        W: (M, K) weight tensor (K must be divisible by block_size).
        block_size: Number of columns per quantization block.

    Returns:
        Dequantized tensor (same shape as W).
    """
    M, K = W.shape
    assert K % block_size == 0
    B = K // block_size
    device = W.device

    codebook = _DOUBLED_FP4.to(device=device, dtype=torch.float32)
    e4m3_table = _build_e4m3_table(device)

    x = W.float().view(M, B, block_size)

    # Ideal scale: amax / max_doubled_codebook (= amax / 12)
    amax = x.abs().amax(dim=-1)  # (M, B)
    ideal_scale = amax / _DOUBLED_FP4_MAX  # (M, B)

    # Round to nearest E4M3 representable value
    # For amax=0, we'll get scale≈0; handle by setting scale=0 explicitly
    safe_ideal = torch.where(ideal_scale > 0, ideal_scale, torch.ones_like(ideal_scale))
    scale = _round_to_e4m3(safe_ideal, e4m3_table)  # (M, B)
    scale = torch.where(amax > 0, scale, torch.zeros_like(scale))

    # Round to nearest codebook value
    possible = (scale.unsqueeze(-1) * codebook).unsqueeze(-2)  # (M, B, 1, 16)
    deltas = x.unsqueeze(-1) - possible  # (M, B, bs, 16)
    best_idx = deltas.abs().argmin(dim=-1)  # (M, B, bs)
    quant_doubled = codebook[best_idx]
    dequant = scale.unsqueeze(-1) * quant_doubled  # (M, B, bs)

    return dequant.view(M, K).to(W.dtype)


def _nvfp4_quantize_block(W_P: Tensor) -> Tensor:
    """Quantize a single block of columns to NVFP4 with E4M3 FP8 scale.

    Args:
        W_P: (M, block_size) float tensor.

    Returns:
        (M, block_size) dequantized values using E4M3 per-row scale.
    """
    device = W_P.device
    codebook = _DOUBLED_FP4.to(device=device, dtype=torch.float32)
    e4m3_table = _build_e4m3_table(device)
    M, bs = W_P.shape

    amax = W_P.abs().amax(dim=-1)  # (M,)
    ideal_scale = amax / _DOUBLED_FP4_MAX

    safe_ideal = torch.where(ideal_scale > 0, ideal_scale, torch.ones_like(ideal_scale))
    scale = _round_to_e4m3(safe_ideal, e4m3_table)
    scale = torch.where(amax > 0, scale, torch.zeros_like(scale))  # (M,)

    possible = (scale.unsqueeze(-1) * codebook).unsqueeze(-2)  # (M, 1, 16)
    deltas = W_P.unsqueeze(-1) - possible  # (M, bs, 16)
    best_idx = deltas.abs().argmin(dim=-1)
    quant_doubled = codebook[best_idx]
    return scale.unsqueeze(-1) * quant_doubled


def _nvfp4_block_scale(block_vals: Tensor) -> Tensor:
    """Compute E4M3 FP8 scale from block values.

    Args:
        block_vals: (M, block_size) current column values.

    Returns:
        (M,) per-row scale.
    """
    device = block_vals.device
    e4m3_table = _build_e4m3_table(device)
    amax = block_vals.abs().amax(dim=-1)  # (M,)
    ideal = amax / _DOUBLED_FP4_MAX
    safe_ideal = torch.where(ideal > 0, ideal, torch.ones_like(ideal))
    scale = _round_to_e4m3(safe_ideal, e4m3_table)
    return torch.where(amax > 0, scale, torch.zeros_like(scale))


@torch.no_grad()
def quantize_nvfp4_obs(
    W: Tensor,
    H: Tensor,
    block_size: int = 16,
    damp: float = 1e-4,
    C: Tensor | None = None,
    order: str = "largest_first",
) -> Tensor:
    """Progressive NVFP4 quantization with OBS full-column compensation.

    Same as quantize_obs but uses E4M3 FP8 scales instead of UE8M0.

    Args:
        W: (M, K) weight tensor.
        H: (K, K) Hessian matrix.
        block_size: Block size (default 16).
        damp: Damping factor.
        C: Precomputed H^{-1}.
        order: "largest_first", "smallest_first", or "left_to_right".

    Returns:
        Quantized weight tensor.
    """
    return _quantize_obs(
        W, H,
        block_size=block_size,
        damp=damp,
        C=C,
        order=order,
        quantize_fn=_nvfp4_quantize_block,
        codebook=_DOUBLED_FP4,
        scale_fn=_nvfp4_block_scale,
    )
