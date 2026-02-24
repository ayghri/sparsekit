"""
True OBS for 16-col block coupling with 8-row-apart pairing.

Structure (same as bench_block_obs.py):
  Chunks of 16 rows. Within each chunk, 8 pairs: row p <-> row p+8.
  At each group of 16 contiguous columns, prune one row's block, keep the other.
  C(2,1)=2 choices per group. 50% sparsity.

True OBS: per-row C = H^{-1} with rank-16 Schur updates after each group.
Batched ng groups at a time: select for ng groups using current C,
then jointly compensate and Schur-update per row.

W (2560, 9728), X (244449, 9728), chunk=16, blk=16, ng=16.
"""

import time
from itertools import combinations

import torch
import torch.linalg as LA

from sparsekit import StructuredOBS

DEVICE = torch.device("cuda:1")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"
CHUNK = 16
BLK = 16
PAIRS = CHUNK // 2


def progress(msg):
    print(msg, flush=True)


def compute_H(X, batch_size=4096):
    N, K = X.shape
    H = torch.zeros(K, K, device=DEVICE, dtype=torch.float32)
    for i in range(0, N, batch_size):
        X_b = X[i:i+batch_size].to(device=DEVICE, dtype=torch.float32)
        H.addmm_(X_b.T, X_b)
    H /= N
    return H


def compute_loss(W_pruned, W0, H, N, chunk=128):
    M = W0.shape[0]
    total = 0.0
    for c0 in range(0, M, chunk):
        dW = W_pruned[c0:c0+chunk] - W0[c0:c0+chunk]
        total += ((dW @ H) * dW).sum().item()
    return total * N


def measure_sparsity(W_pruned):
    return (W_pruned.abs() < 1e-10).sum().item() / W_pruned.numel() * 100


def check_block_coupling(W_pruned, label):
    """Verify each pair prunes exactly one block of 16 per group."""
    M, K = W_pruned.shape
    G = K // BLK
    violations = 0
    for c0 in range(0, M, CHUNK):
        Wc = W_pruned[c0:c0+CHUNK]
        zeros = (Wc.abs() < 1e-10).view(CHUNK, G, BLK).all(dim=2)  # (16, G) — True if block is all-zero
        for p in range(PAIRS):
            r0, r1 = p, p + PAIRS
            # Exactly one of the two should be all-zero per group
            both_zero = (zeros[r0] & zeros[r1]).sum().item()
            neither_zero = (~zeros[r0] & ~zeros[r1]).sum().item()
            violations += both_zero + neither_zero
    if violations > 0:
        progress(f"  WARNING: {label} has {violations} coupling violations!")
    else:
        progress(f"  {label}: coupling OK")
    return violations == 0


# ── Magnitude ─────────────────────────────────────────────────────────────

def magnitude_block(W0):
    M, K = W0.shape
    W = W0.clone()
    G = K // BLK
    for c0 in range(0, M, CHUNK):
        Wc = W[c0:c0+CHUNK]
        W_blocked = Wc.view(PAIRS, 2, G, BLK)
        norms = W_blocked.norm(dim=3)
        prune_j = norms.argmin(dim=1)
        mask = torch.zeros(PAIRS, 2, G, dtype=torch.bool, device=W0.device)
        mask.scatter_(1, prune_j.unsqueeze(1), True)
        W_blocked[mask.unsqueeze(3).expand_as(W_blocked)] = 0.0
    return W


# ── SparseGPT ─────────────────────────────────────────────────────────────

def sparsegpt_block(W0, H, blocksize=128):
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

    for c0 in range(0, M, CHUNK):
        Wc = W[c0:c0+CHUNK]

        for i1 in range(0, K, blocksize):
            i2 = min(i1 + blocksize, K)
            count = i2 - i1

            W1 = Wc[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            mask1 = torch.zeros_like(W1, dtype=torch.bool)

            for i in range(count):
                global_col = i1 + i
                if global_col % BLK == 0 and i + BLK <= count:
                    diag_sq = torch.diag(Hinv1)[i:i+BLK].reshape(1, -1) ** 2
                    row_scores = (W1[:, i:i+BLK] ** 2 / diag_sq).sum(dim=1)

                    for p in range(PAIRS):
                        r0, r1 = p, p + PAIRS
                        if row_scores[r0] <= row_scores[r1]:
                            mask1[r0, i:i+BLK] = True
                        else:
                            mask1[r1, i:i+BLK] = True

                w = W1[:, i]
                d = Hinv1[i, i]
                q = w.clone()
                q[mask1[:, i]] = 0.0
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err1

            Wc[:, i1:i2] = Q1
            Wc[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return W


# ── True OBS block coupling ───────────────────────────────────────────────

def true_obs_block(W0, C_base, ng=16):
    """True OBS with per-row Schur updates for block coupling.

    For each chunk of 16 rows (8 pairs):
      Batch ng groups at a time:
        1. Score each group: OBS cost per row, pick cheaper row to prune per pair
        2. Collect pruned columns per row across ng groups
        3. Joint compensation + rank-(n_pruned*16) Schur update per row
    """
    M, K = W0.shape
    device = W0.device
    G = K // BLK
    W = W0.clone().float()

    eye_blk = 1e-8 * torch.eye(BLK, device=device)
    n_chunks = M // CHUNK

    for ci in range(n_chunks):
        c0 = ci * CHUNK
        if ci % 20 == 0:
            progress(f"    chunk {ci}/{n_chunks}")
        Wc = W[c0:c0+CHUNK]  # view

        # Per-row C for this chunk
        C = C_base.unsqueeze(0).expand(CHUNK, -1, -1).clone()  # (16, K, K)

        for batch_start in range(0, G, ng):
            batch_end = min(batch_start + ng, G)
            n_g = batch_end - batch_start

            # Per-row list of pruned column index tensors
            row_pruned = [[] for _ in range(CHUNK)]

            # Score each group, select which row to prune per pair
            for g in range(batch_start, batch_end):
                col_idx = torch.arange(g * BLK, (g + 1) * BLK, device=device)

                C_bb = C[:, col_idx][:, :, col_idx]  # (16, BLK, BLK)
                C_bb_inv = LA.inv(C_bb + eye_blk)
                W_b = Wc[:, col_idx]  # (16, BLK)
                costs = (torch.bmm(W_b.unsqueeze(1), C_bb_inv).squeeze(1) * W_b).sum(1)

                # Per-pair: prune row with lower cost
                for p in range(PAIRS):
                    r = p if costs[p] <= costs[p + PAIRS] else p + PAIRS
                    row_pruned[r].append(col_idx)

            # Joint compensation + Schur update per row
            for r in range(CHUNK):
                if not row_pruned[r]:
                    continue
                all_cols = torch.cat(row_pruned[r])  # (n_pruned * BLK,)
                np_total = all_cols.shape[0]

                eye_n = 1e-8 * torch.eye(np_total, device=device)
                C_col = C[r, :, all_cols].clone()        # (K, np_total)
                C_PP = C_col[all_cols, :]                 # (np_total, np_total)
                C_PP_inv = torch.inverse(C_PP + eye_n)
                wb = Wc[r, all_cols].clone()              # (np_total,)

                # Compensation
                Wc[r] -= C_col @ (C_PP_inv @ wb)
                Wc[r, all_cols] = 0.0

                # Schur update
                tmp = C_col @ C_PP_inv                    # (K, np_total)
                C_r_rows = C[r, all_cols, :].clone()      # (np_total, K)
                C[r].addmm_(tmp, C_r_rows, alpha=-1.0)
                C[r, all_cols, :] = 0.0
                C[r, :, all_cols] = 0.0

    return W


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    progress("Loading data...")
    W0 = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
    M, K = W0.shape
    N = X_cpu.shape[0]
    G = K // BLK
    n_chunks = M // CHUNK
    progress(f"  W: {W0.shape}, X: {X_cpu.shape}")
    progress(f"  {n_chunks} chunks of {CHUNK} rows, {PAIRS} pairs/chunk, {G} groups of {BLK} cols")

    progress("Computing H...")
    t0 = time.time()
    H = compute_H(X_cpu)
    torch.cuda.synchronize(DEVICE)
    progress(f"  H computed in {time.time() - t0:.1f}s")
    del X_cpu

    ref = compute_loss(W0, torch.zeros_like(W0), H, N)
    progress(f"Reference ||X W0^T||_F^2 = {ref:.4e}")

    progress("\nComputing C = H^{-1}...")
    t0 = time.time()
    C = StructuredOBS.compute_inverse(H, damp=1e-4)
    torch.cuda.synchronize(DEVICE)
    progress(f"  C computed in {time.time() - t0:.1f}s")

    results = []

    # ── Magnitude ──
    progress("\n  [Magnitude] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_mag = magnitude_block(W0)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_mag, W0, H, N)
    sp = measure_sparsity(W_mag)
    check_block_coupling(W_mag, "Magnitude")
    results.append(("Magnitude", loss, t, sp))
    progress(f"  [Magnitude] Loss={loss:.4e}, Time={t:.3f}s, Sp={sp:.1f}%")
    del W_mag

    # ── SparseGPT ──
    progress("\n  [SparseGPT] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_sgpt = sparsegpt_block(W0, H)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_sgpt, W0, H, N)
    sp = measure_sparsity(W_sgpt)
    check_block_coupling(W_sgpt, "SparseGPT")
    results.append(("SparseGPT", loss, t, sp))
    progress(f"  [SparseGPT] Loss={loss:.4e}, Time={t:.1f}s, Sp={sp:.1f}%")
    del W_sgpt

    # ── True OBS ng=16 ──
    progress("\n  [True OBS ng=16] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_obs = true_obs_block(W0, C, ng=16)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_obs, W0, H, N)
    sp = measure_sparsity(W_obs)
    check_block_coupling(W_obs, "True OBS ng=16")
    results.append(("True OBS ng=16", loss, t, sp))
    progress(f"  [True OBS ng=16] Loss={loss:.4e}, Time={t:.1f}s, Sp={sp:.1f}%")
    del W_obs

    # ── Report ──
    loss_sgpt = results[1][1]
    progress(f"\n  {'Method':<25} {'Loss':>14} {'Norm.':>10} {'Sp':>6} {'Time':>8} {'vs SGPT':>10}")
    progress(f"  {'-'*77}")
    for name, loss, t, sp in results:
        vs = (1 - loss / loss_sgpt) * 100 if loss < loss_sgpt else -(1 - loss_sgpt / loss) * 100
        progress(f"  {name:<25} {loss:>14.4e} {loss/ref*100:>8.4f}% {sp:>4.0f}% {t:>7.1f}s {vs:>+9.2f}%")

    best_name, best_loss, best_t, _ = min(results, key=lambda x: x[1])
    progress(f"\n  Best: {best_name} ({best_loss:.4e}, {best_loss/ref*100:.4f}%, {best_t:.1f}s)")


if __name__ == "__main__":
    main()
