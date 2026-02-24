"""
4:8 structured sparsity benchmark with block_shape=(1,2), group_shape=(1,4).

Each group of 8 contiguous columns contains 4 blocks of 2. We prune 2 blocks
(4 columns) and keep 2 blocks (4 columns). Elements within a block are always
pruned/kept together.

Methods:
  - Magnitude:  prune 2 lowest-norm blocks per group
  - Wanda:      prune by |w| * ||x||_2 (weight * input feature norm)
  - SparseGPT:  Cholesky-based, column-sequential, paired selection
  - OBS full:   optimal subset (C(4,2)=6 combos) + full-column compensation
  - OBS local:  optimal subset + within-group compensation only

W (2560, 9728), X (244449, 9728)
"""

import time
import torch
import torch.linalg as LA

from sparsekit import BlockSpec, GroupSpec, StructuredOBS


def progress(msg):
    print(msg, flush=True)


DEVICE = torch.device("cuda:1")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


def compute_H(X, batch_size=4096):
    N, K = X.shape
    H = torch.zeros(K, K, device=DEVICE, dtype=torch.float32)
    for i in range(0, N, batch_size):
        X_b = X[i:i+batch_size].to(device=DEVICE, dtype=torch.float32)
        H.addmm_(X_b.T, X_b)
    H /= N
    return H


def compute_loss(W_quant, W0, H, N, chunk=128):
    M = W0.shape[0]
    total = 0.0
    for c0 in range(0, M, chunk):
        dW = W_quant[c0:c0+chunk] - W0[c0:c0+chunk]
        total += ((dW @ H) * dW).sum().item()
    return total * N


# ── Magnitude 4:8 (block pairs) ──────────────────────────────────────────

def magnitude_48(W0):
    """Prune 2 of 4 blocks (each block = 2 contiguous columns) per group of 8."""
    W = W0.clone()
    M, K = W.shape
    assert K % 8 == 0
    # Reshape into groups of 4 blocks of 2
    Wg = W.view(M, K // 8, 4, 2)
    # Block norms: (M, G, 4)
    block_norms = Wg.abs().sum(dim=-1)
    # Find 2 smallest-norm blocks per group
    _, bot = block_norms.topk(2, dim=-1, largest=False)
    mask = torch.zeros(M, K // 8, 4, dtype=torch.bool, device=W.device)
    mask.scatter_(2, bot, True)
    # Zero out pruned blocks
    Wg[mask.unsqueeze(-1).expand_as(Wg)] = 0.0
    return W


# ── Wanda 4:8 (block pairs) ───────────────────────────────────────────────

def wanda_48(W0, H):
    """Prune 2 of 4 blocks per group using Wanda metric: |w| * ||x||_2.

    ||x_j||_2 = sqrt(N * H[j,j]), but since N is constant across columns
    we use sqrt(H[j,j]) for ranking.
    """
    W = W0.clone()
    M, K = W.shape
    assert K % 8 == 0
    # Input feature norms: sqrt(diag(H))
    x_norms = torch.sqrt(torch.diag(H))  # (K,)
    # Wanda scores: |w_ij| * ||x_j||
    scores = W.abs() * x_norms.unsqueeze(0)  # (M, K)
    # Reshape into groups of 4 blocks of 2
    scores_g = scores.view(M, K // 8, 4, 2)
    # Block scores: sum within each block pair
    block_scores = scores_g.sum(dim=-1)  # (M, G, 4)
    # Find 2 smallest-score blocks per group
    _, bot = block_scores.topk(2, dim=-1, largest=False)
    mask = torch.zeros(M, K // 8, 4, dtype=torch.bool, device=W.device)
    mask.scatter_(2, bot, True)
    # Zero out pruned blocks
    Wg = W.view(M, K // 8, 4, 2)
    Wg[mask.unsqueeze(-1).expand_as(Wg)] = 0.0
    return W


# ── SparseGPT 4:8 (block pairs) ──────────────────────────────────────────

def sparsegpt_48(W0, H, blocksize=128):
    """SparseGPT for 4:8 structured sparsity with block_shape=(1,2).

    At each group boundary (every 8 columns), scores 4 blocks of 2 by
    sum of w^2/d^2 within each block, selects 2 lowest-score blocks to prune.
    Error is propagated column-by-column via Cholesky(H^{-1}).
    """
    M, K = W0.shape
    device = W0.device
    W = W0.clone().float()
    H = H.clone().float()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0

    damp = 0.01 * torch.mean(torch.diag(H))
    diag_idx = torch.arange(K, device=device)
    H[diag_idx, diag_idx] += damp

    L = LA.cholesky(H)
    Hinv_full = torch.cholesky_inverse(L)
    Hinv = LA.cholesky(Hinv_full, upper=True)

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

            # At group boundary (every 8 columns): select which 2 of 4
            # blocks to prune using block-level scores
            if col % 8 == 0:
                end = min(i + 8, count)
                if end - i == 8:
                    # Score each column: w^2 / d^2
                    col_scores = W1[:, i:end] ** 2 / (
                        torch.diag(Hinv1)[i:end].reshape(1, -1)
                    ) ** 2
                    # Sum pairs to get block scores: (M, 4)
                    block_scores = col_scores.view(M, 4, 2).sum(dim=-1)
                    # Find 2 lowest-score blocks
                    _, bot = block_scores.topk(2, dim=1, largest=False)
                    # Expand block indices to column indices within group
                    bot_cols = (bot.unsqueeze(-1) * 2 +
                                torch.arange(2, device=device)).view(M, 4)
                    mask1[:, i:end].scatter_(1, bot_cols, True)

            w = W1[:, i]
            d = Hinv1[i, i]

            q = w.clone()
            q[mask1[:, i]] = 0.0
            Q1[:, i] = q

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err1

        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return W


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    progress("Loading data...")
    W0 = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
    M, K = W0.shape
    N = X_cpu.shape[0]
    progress(f"  W: {W0.shape}, X: {X_cpu.shape}")

    progress("Computing H...")
    t0 = time.time()
    H = compute_H(X_cpu)
    torch.cuda.synchronize(DEVICE)
    progress(f"  H computed in {time.time() - t0:.1f}s")
    del X_cpu

    ref = compute_loss(W0, torch.zeros_like(W0), H, N)
    progress(f"Reference ||X W0^T||_F^2 = {ref:.4e}")

    progress("\nPrecomputing C = H^{-1}...")
    t0 = time.time()
    C = StructuredOBS.compute_inverse(H, damp=1e-4)
    torch.cuda.synchronize(DEVICE)
    progress(f"  C computed in {time.time() - t0:.1f}s")

    results = []

    progress(f"\n{'='*70}")
    progress("4:8 structured sparsity: block_shape=(1,2), group_shape=(1,4)")
    progress("  Keep 2 of 4 blocks (4 of 8 columns), C(4,2)=6 subsets")
    progress(f"  K={K} -> {K//8} groups")
    progress(f"{'='*70}")

    # ── Magnitude ──
    progress("\n  [Magnitude] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_mag = magnitude_48(W0)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_mag, W0, H, N)
    results.append(("Magnitude", loss, t))
    progress(f"  [Magnitude] Loss={loss:.4e}, Time={t:.3f}s")
    del W_mag

    # ── Wanda ──
    progress("\n  [Wanda] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_wanda = wanda_48(W0, H)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_wanda, W0, H, N)
    results.append(("Wanda", loss, t))
    progress(f"  [Wanda] Loss={loss:.4e}, Time={t:.3f}s")
    del W_wanda

    # ── SparseGPT ──
    progress("\n  [SparseGPT] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_sgpt = sparsegpt_48(W0, H, blocksize=128)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_sgpt, W0, H, N)
    results.append(("SparseGPT", loss, t))
    progress(f"  [SparseGPT] Loss={loss:.4e}, Time={t:.1f}s")
    del W_sgpt

    # ── OBS local ──
    progress("\n  [OBS local] Running...")
    W_local = torch.nn.Parameter(W0.clone())
    block_local = BlockSpec(W_local, block_shape=(1, 2))
    group_local = GroupSpec(block_local, group_shape=(1, 4))
    solver_local = StructuredOBS(group_local, H, damp=1e-4, C=C)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    solver_local.prune(num_nz=2, compensate="local")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_local.data, W0, H, N)
    results.append(("OBS local", loss, t))
    progress(f"  [OBS local] Loss={loss:.4e}, Time={t:.1f}s")
    del W_local, block_local, group_local, solver_local

    # ── OBS full ──
    progress("\n  [OBS full] Running...")
    W_full = torch.nn.Parameter(W0.clone())
    block_full = BlockSpec(W_full, block_shape=(1, 2))
    group_full = GroupSpec(block_full, group_shape=(1, 4))
    solver_full = StructuredOBS(group_full, H, damp=1e-4, C=C)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    solver_full.prune(num_nz=2, compensate="full")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_full.data, W0, H, N)
    results.append(("OBS full", loss, t))
    progress(f"  [OBS full] Loss={loss:.4e}, Time={t:.1f}s")
    del W_full, block_full, group_full, solver_full

    # ══════════════════════════════════════════════════════════════════════
    # Report
    # ══════════════════════════════════════════════════════════════════════
    progress(f"\n{'='*70}")
    progress("Results")
    progress(f"{'='*70}")
    progress(f"\n  {'Method':<25} {'Loss':>14} {'Norm.':>10} {'Time':>10}")
    progress(f"  {'-'*61}")
    for name, loss, t in results:
        progress(f"  {name:<25} {loss:>14.4e} {loss/ref*100:>8.4f}% {t:>8.3f}s")

    r = {name: (loss, t) for name, loss, t in results}

    def compare(a, b):
        la, lb = r[a][0], r[b][0]
        if la < lb:
            progress(f"  {a} beats {b} by {(1 - la/lb)*100:.2f}%")
        else:
            progress(f"  {b} beats {a} by {(1 - lb/la)*100:.2f}%")

    progress(f"\n  --- vs Magnitude ---")
    compare("Wanda", "Magnitude")
    compare("SparseGPT", "Magnitude")
    compare("OBS local", "Magnitude")
    compare("OBS full", "Magnitude")

    progress(f"\n  --- vs SparseGPT ---")
    compare("OBS full", "SparseGPT")
    compare("Wanda", "SparseGPT")

    progress(f"\n  --- OBS variants ---")
    compare("OBS full", "OBS local")

    best_name, best_loss, _ = min(results, key=lambda x: x[1])
    progress(f"\n  Best: {best_name} ({best_loss:.4e}, {best_loss/ref*100:.4f}%)")


if __name__ == "__main__":
    main()
