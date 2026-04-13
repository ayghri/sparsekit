# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0
# (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""SparseGPT column-sequential pruning (tensor-level API).

Reference: Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be
Accurately Pruned in One-Shot", 2023.
"""

import torch
import torch.linalg as LA
from torch import Tensor


def _prepare_hinv(H: Tensor, W: Tensor, damp: float = 0.01):
    """Damp H, handle dead columns, return upper-Cholesky of H^{-1}.

    Zeros dead columns in W **in-place**.

    Returns:
        Hinv: (K, K) upper-triangular Cholesky factor of (H+damp*I)^{-1}.
    """
    K = H.shape[0]
    device = H.device
    H = H.clone().float()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    damp_val = damp * torch.mean(torch.diag(H))
    diag_idx = torch.arange(K, device=device)
    H[diag_idx, diag_idx] += damp_val

    L = LA.cholesky(H)
    return LA.cholesky(torch.cholesky_inverse(L), upper=True)


@torch.no_grad()
def sparsegpt(
    W0: Tensor,
    H: Tensor,
    block_size: int = 1,
    scope_size: int = 4,
    blocksize: int = 128,
    damp: float = 0.01,
) -> Tensor:
    """SparseGPT N:M structured pruning.

    Within each scope of ``scope_size`` columns (grouped into
    ``block_size``-column blocks), keeps the half with highest
    w²/d² scores. Compensates column-by-column left-to-right.

    Args:
        W0: (M, K) weight matrix.
        H: (K, K) Hessian X^T X / N.
        block_size: columns per block (1 for scalar, >1 for block).
        scope_size: columns per scope (4 for 2:4, 8 for 4:8).
        blocksize: sequential processing chunk size.
        damp: damping as fraction of mean(diag(H)).

    Returns:
        (M, K) pruned weight tensor.
    """
    M, K = W0.shape
    device = W0.device
    W = W0.clone().float()

    Hinv = _prepare_hinv(H, W, damp)

    bpg = scope_size // block_size
    num_prune = bpg // 2

    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        mask1 = torch.zeros_like(W1, dtype=torch.bool)

        for i in range(count):
            col = i1 + i
            if col % scope_size == 0:
                end = min(i + scope_size, count)
                if end - i == scope_size:
                    col_scores = W1[:, i:end] ** 2 / (
                        torch.diag(Hinv1)[i:end].reshape(1, -1)
                    ) ** 2
                    if block_size > 1:
                        bscores = col_scores.view(
                            M, bpg, block_size
                        ).sum(-1)
                        _, bot = bscores.topk(
                            num_prune, dim=1, largest=False
                        )
                        bot_cols = (
                            bot.unsqueeze(-1) * block_size
                            + torch.arange(
                                block_size, device=device
                            )
                        ).view(M, num_prune * block_size)
                        mask1[:, i:end].scatter_(1, bot_cols, True)
                    else:
                        _, bot = col_scores.topk(
                            num_prune, dim=1, largest=False
                        )
                        mask1[:, i:end].scatter_(1, bot, True)

            w = W1[:, i]
            d = Hinv1[i, i]
            q = w.clone()
            q[mask1[:, i]] = 0.0
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= (
                err1.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            )
            Err1[:, i] = err1

        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return W


@torch.no_grad()
def sparsegpt_coupled_24(
    W0: Tensor,
    H: Tensor,
    blocksize: int = 128,
    damp: float = 0.01,
) -> Tensor:
    """SparseGPT for coupled 2:4 sparsity.

    Paired mask selection at 16-column boundaries: columns
    (j, j+8) share the same mask bit for j=0..7 within each
    16-column tile.

    Args:
        W0: (M, K) weight matrix, K divisible by 16.
        H: (K, K) Hessian.
        blocksize: sequential processing chunk size.
        damp: damping factor.

    Returns:
        (M, K) pruned weight tensor.
    """
    M, K = W0.shape
    device = W0.device
    W = W0.clone().float()

    Hinv = _prepare_hinv(H, W, damp)

    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        mask1 = torch.zeros_like(W1, dtype=torch.bool)

        for i in range(count):
            col = i1 + i
            if col % 16 == 0 and i + 16 <= count:
                d_sq = (
                    torch.diag(Hinv1)[i : i + 16].reshape(1, -1)
                    ** 2
                )
                col_scores = W1[:, i : i + 16] ** 2 / d_sq
                # First half-pair: cols 0-3 coupled with 8-11
                ps0 = col_scores[:, :4] + col_scores[:, 8:12]
                _, bot0 = ps0.topk(2, dim=1, largest=False)
                pmask0 = torch.zeros(
                    M, 4, dtype=torch.bool, device=device
                )
                pmask0.scatter_(1, bot0, True)
                mask1[:, i : i + 4] |= pmask0
                mask1[:, i + 8 : i + 12] |= pmask0
                # Second half-pair: cols 4-7 coupled with 12-15
                ps1 = col_scores[:, 4:8] + col_scores[:, 12:16]
                _, bot1 = ps1.topk(2, dim=1, largest=False)
                pmask1 = torch.zeros(
                    M, 4, dtype=torch.bool, device=device
                )
                pmask1.scatter_(1, bot1, True)
                mask1[:, i + 4 : i + 8] |= pmask1
                mask1[:, i + 12 : i + 16] |= pmask1

            w = W1[:, i]
            d = Hinv1[i, i]
            q = w.clone()
            q[mask1[:, i]] = 0.0
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= (
                err1.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            )
            Err1[:, i] = err1

        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return W


@torch.no_grad()
def sparsegpt_block16(
    W0: Tensor,
    H: Tensor,
    blocksize: int = 128,
    damp: float = 0.01,
) -> Tensor:
    """SparseGPT for 16-column blocks with 8-row coupling.

    Processes 16-row chunks. Within each chunk, 8 row-pairs
    compete: the row with lower block-norm is pruned.

    Args:
        W0: (M, K) weight matrix, M divisible by 16.
        H: (K, K) Hessian.
        blocksize: sequential processing chunk size.
        damp: damping factor.

    Returns:
        (M, K) pruned weight tensor.
    """
    M, K = W0.shape
    W = W0.clone().float()

    Hinv = _prepare_hinv(H, W, damp)

    BLK = 16
    CHUNK = 16
    PAIRS = 8

    for c0 in range(0, M, CHUNK):
        Wc = W[c0 : c0 + CHUNK]
        for i1 in range(0, K, blocksize):
            i2 = min(i1 + blocksize, K)
            count = i2 - i1
            W1 = Wc[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            mask1 = torch.zeros_like(W1, dtype=torch.bool)
            for i in range(count):
                col = i1 + i
                if col % BLK == 0 and i + BLK <= count:
                    diag_sq = (
                        torch.diag(Hinv1)[i : i + BLK].reshape(
                            1, -1
                        )
                        ** 2
                    )
                    row_scores = (
                        W1[:, i : i + BLK] ** 2 / diag_sq
                    ).sum(dim=1)
                    for p in range(PAIRS):
                        r0, r1 = p, p + PAIRS
                        if row_scores[r0] <= row_scores[r1]:
                            mask1[r0, i : i + BLK] = True
                        else:
                            mask1[r1, i : i + BLK] = True
                w = W1[:, i]
                d = Hinv1[i, i]
                q = w.clone()
                q[mask1[:, i]] = 0.0
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= (
                    err1.unsqueeze(1)
                    * Hinv1[i, i:].unsqueeze(0)
                )
                Err1[:, i] = err1
            Wc[:, i1:i2] = Q1
            Wc[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return W
