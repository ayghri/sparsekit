"""
Coupled 2:4 sparsity: pairs of elements 8 columns apart.

BlockView (M, K/16, 8, 2):(K, 16, 1, 8)
  element [m, g, i, j] -> W[m, g*16 + i + j*8]
  So block (i, j=0..1) pairs column (g*16+i) with column (g*16+i+8).

BlockSpec block_shape=(1,1,1,2), GroupSpec group_shape=(1,1,4,1):
  Each group has 4 paired blocks. Keep 2, prune 2 -> 2:4 on paired elements.
  Two sub-groups per 16-col block:
    Sub-group 0: pairs (0,8), (1,9), (2,10), (3,11)
    Sub-group 1: pairs (4,12), (5,13), (6,14), (7,15)

Methods:
  - Magnitude:  prune 2 lowest-norm pairs per sub-group
  - Wanda:      prune by |w|*||x||_2 summed per pair
  - SparseGPT:  Cholesky-based, column-sequential, paired selection
  - OBS full:   optimal subset (C(4,2)=6) + full-column compensation

W (2560, 9728), X (244449, 9728)
"""

import time
import torch
import torch.linalg as LA

from sparsekit import BlockSpec, GroupSpec, StructuredOBS
from sparsekit.views import BlockView


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


def _apply_coupled_mask(W, mask_fn):
    """Apply a mask function that scores 4 pairs per sub-group.

    mask_fn(W_16, sub_cols_a, sub_cols_b) -> (M, G, 4) pair scores.
    Prunes 2 lowest-scoring pairs per sub-group.
    """
    M, K = W.shape
    G = K // 16
    W_16 = W.view(M, G, 16)
    mask = torch.zeros(M, G, 16, dtype=torch.bool, device=W.device)

    for a, b in [(slice(0, 4), slice(8, 12)),
                 (slice(4, 8), slice(12, 16))]:
        pair_scores = mask_fn(W_16, a, b)  # (M, G, 4)
        _, bot = pair_scores.topk(2, dim=-1, largest=False)
        pmask = torch.zeros(M, G, 4, dtype=torch.bool, device=W.device)
        pmask.scatter_(2, bot, True)
        mask[:, :, a] |= pmask
        mask[:, :, b] |= pmask

    W_16[mask] = 0.0


# ── Magnitude ────────────────────────────────────────────────────────────

def magnitude_coupled_24(W0):
    """Prune 2 of 4 paired blocks per sub-group by L1 norm."""
    W = W0.clone()
    _apply_coupled_mask(W, lambda w, a, b: w[:, :, a].abs() + w[:, :, b].abs())
    return W


# ── Wanda ─────────────────────────────────────────────────────────────────

def wanda_coupled_24(W0, H):
    """Prune by Wanda metric: |w| * sqrt(diag(H)), summed per pair."""
    W = W0.clone()
    M, K = W.shape
    x_norms = torch.sqrt(torch.diag(H))  # (K,)
    scores = W.abs() * x_norms.unsqueeze(0)  # (M, K)
    S = scores.view(M, K // 16, 16)

    _apply_coupled_mask(W, lambda w, a, b: S[:, :, a] + S[:, :, b])
    return W


# ── SparseGPT ─────────────────────────────────────────────────────────────

def sparsegpt_coupled_24(W0, H, blocksize=128):
    """SparseGPT for coupled 2:4: column-sequential with paired selection.

    At each 16-column boundary, scores 4 pairs per sub-group using
    w^2/d^2 summed over the 2 columns in each pair, selects 2 lowest-score
    pairs to prune. Error propagated column-by-column via Cholesky(H^{-1}).
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

            # At 16-column boundary: score pairs for both sub-groups
            if col % 16 == 0 and i + 16 <= count:
                d_sq = torch.diag(Hinv1)[i:i+16].reshape(1, -1) ** 2
                col_scores = W1[:, i:i+16] ** 2 / d_sq  # (M, 16)

                # Sub-group 0: pairs (0,8), (1,9), (2,10), (3,11)
                ps0 = col_scores[:, :4] + col_scores[:, 8:12]  # (M, 4)
                _, bot0 = ps0.topk(2, dim=1, largest=False)
                pmask0 = torch.zeros(M, 4, dtype=torch.bool, device=device)
                pmask0.scatter_(1, bot0, True)
                mask1[:, i:i+4] |= pmask0
                mask1[:, i+8:i+12] |= pmask0

                # Sub-group 1: pairs (4,12), (5,13), (6,14), (7,15)
                ps1 = col_scores[:, 4:8] + col_scores[:, 12:16]  # (M, 4)
                _, bot1 = ps1.topk(2, dim=1, largest=False)
                pmask1 = torch.zeros(M, 4, dtype=torch.bool, device=device)
                pmask1.scatter_(1, bot1, True)
                mask1[:, i+4:i+8] |= pmask1
                mask1[:, i+12:i+16] |= pmask1

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

    # results: list of (name, loss, time, sparsity)
    results = []

    def measure_sparsity(W_pruned):
        return (W_pruned.abs() < 1e-10).sum().item() / W_pruned.numel() * 100

    progress(f"\n{'='*70}")
    progress("Coupled 2:4 sparsity: pairs of elements 8 columns apart")
    progress("  BlockView (M, K/16, 8, 2):(K, 16, 1, 8)")
    progress("  block_shape=(1,1,1,2), group_shape=(1,1,4,1)")
    progress("  Keep 2 of 4 pairs per sub-group, C(4,2)=6 subsets")
    progress(f"  K={K} -> {K//16} groups of 16 cols, {K//16*2} sub-groups")
    progress(f"{'='*70}")

    # ── Magnitude ──
    progress("\n  [Magnitude] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_mag = magnitude_coupled_24(W0)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_mag, W0, H, N)
    sp = measure_sparsity(W_mag)
    results.append(("Magnitude", loss, t, sp))
    progress(f"  [Magnitude] Loss={loss:.4e}, Time={t:.3f}s, Sparsity={sp:.1f}%")
    del W_mag

    # ── Wanda ──
    progress("\n  [Wanda] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_wanda = wanda_coupled_24(W0, H)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_wanda, W0, H, N)
    sp = measure_sparsity(W_wanda)
    results.append(("Wanda", loss, t, sp))
    progress(f"  [Wanda] Loss={loss:.4e}, Time={t:.3f}s, Sparsity={sp:.1f}%")
    del W_wanda

    # ── SparseGPT variants ──
    for bs in [128, 256, 512, 1024]:
        label = f"SparseGPT bs={bs}"
        progress(f"\n  [{label}] Running...")
        torch.cuda.synchronize(DEVICE)
        t0 = time.time()
        W_sgpt = sparsegpt_coupled_24(W0, H, blocksize=bs)
        torch.cuda.synchronize(DEVICE)
        t = time.time() - t0
        loss = compute_loss(W_sgpt, W0, H, N)
        sp = measure_sparsity(W_sgpt)
        results.append((label, loss, t, sp))
        progress(f"  [{label}] Loss={loss:.4e}, Time={t:.1f}s, Sparsity={sp:.1f}%")
        del W_sgpt

    # ── OBS full ──
    progress("\n  [OBS full] Running...")
    W_full = torch.nn.Parameter(W0.clone())
    view = BlockView(W_full, size=(M, K // 16, 8, 2), stride=(K, 16, 1, 8))
    block = BlockSpec(view, block_shape=(1, 1, 1, 2))
    group = GroupSpec(block, group_shape=(1, 1, 4, 1))
    solver = StructuredOBS(group, H, damp=1e-4, C=C)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    solver.prune(num_nz=2, compensate="full")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_full.data, W0, H, N)
    sp = measure_sparsity(W_full.data)
    results.append(("OBS full", loss, t, sp))
    progress(f"  [OBS full] Loss={loss:.4e}, Time={t:.1f}s, Sparsity={sp:.1f}%")
    del W_full, view, block, group, solver

    # ── OBS split variants ──
    def run_obs_split(n_splits):
        label = f"OBS split={n_splits}"
        progress(f"\n  [{label}] Running...")
        W_sp = torch.nn.Parameter(W0.clone())
        v = BlockView(W_sp, size=(M, K // 16, 8, 2), stride=(K, 16, 1, 8))
        b = BlockSpec(v, block_shape=(1, 1, 1, 2))
        g = GroupSpec(b, group_shape=(1, 1, 4, 1))
        s = StructuredOBS(g, H, damp=1e-4, C=C)
        torch.cuda.synchronize(DEVICE)
        t0 = time.time()
        s.prune(num_nz=2, compensate="split", n_splits=n_splits)
        torch.cuda.synchronize(DEVICE)
        t = time.time() - t0
        loss = compute_loss(W_sp.data, W0, H, N)
        sp = measure_sparsity(W_sp.data)
        results.append((label, loss, t, sp))
        progress(f"  [{label}] Loss={loss:.4e}, Time={t:.1f}s, Sparsity={sp:.1f}%")
        del W_sp, v, b, g, s

    run_obs_split(2)
    run_obs_split(4)

    # ══════════════════════════════════════════════════════════════════════
    # Report
    # ══════════════════════════════════════════════════════════════════════
    progress(f"\n{'='*70}")
    progress("Results")
    progress(f"{'='*70}")
    progress(f"\n  {'Method':<25} {'Loss':>14} {'Norm.':>10} {'Sparsity':>10} {'Time':>10}")
    progress(f"  {'-'*73}")
    for name, loss, t, sp in results:
        progress(f"  {name:<25} {loss:>14.4e} {loss/ref*100:>8.4f}% {sp:>8.1f}% {t:>8.3f}s")

    r = {name: (loss, t, sp) for name, loss, t, sp in results}

    def compare(a, b):
        la, lb = r[a][0], r[b][0]
        if la < lb:
            progress(f"  {a} beats {b} by {(1 - la/lb)*100:.2f}%")
        else:
            progress(f"  {b} beats {a} by {(1 - lb/la)*100:.2f}%")

    # Find best SparseGPT variant
    sgpt_results = [(n, l) for n, l, _, _ in results if n.startswith("SparseGPT")]
    best_sgpt = min(sgpt_results, key=lambda x: x[1])[0] if sgpt_results else None

    if best_sgpt:
        progress(f"\n  --- vs {best_sgpt} ---")
        for name, loss, t, sp in results:
            if name.startswith("SparseGPT"):
                continue
            compare(name, best_sgpt)

    best_name, best_loss, best_t, best_sp = min(results, key=lambda x: x[1])
    progress(f"\n  Best: {best_name} ({best_loss:.4e}, {best_loss/ref*100:.4f}%, "
             f"sparsity={best_sp:.1f}%, {best_t:.1f}s)")

    # ── Diagnostic: verify mask pattern ──
    progress(f"\n{'='*70}")
    progress("Mask pattern diagnostic")
    progress(f"{'='*70}")
    progress("  (. = zero, X = nonzero)")

    def check_sparsity(label, W_pruned):
        """Print 16-col mask pattern and verify sparsity + coupling."""
        mask_16 = (W_pruned[0, :16] == 0).int().cpu().tolist()
        progress(f"\n  {label}:")
        progress(f"   row 0, cols 0-15: {''.join(['.' if m else 'X' for m in mask_16])}")
        progress(f"   sub-grp 0 (0-3 | 8-11): "
                 f"{''.join(['.' if mask_16[i] else 'X' for i in range(4)])} | "
                 f"{''.join(['.' if mask_16[i] else 'X' for i in range(8,12)])}")
        progress(f"   sub-grp 1 (4-7 | 12-15): "
                 f"{''.join(['.' if mask_16[i] else 'X' for i in range(4,8)])} | "
                 f"{''.join(['.' if mask_16[i] else 'X' for i in range(12,16)])}")

        # 4-row x 16-col tile
        progress(f"   4x16 tile (rows 0-3):")
        for row in range(4):
            m = (W_pruned[row, :16] == 0).int().cpu().tolist()
            progress(f"     r{row}: {''.join(['.' if v else 'X' for v in m])}")

        total_zero = (W_pruned.abs() < 1e-10).sum().item()
        total_elem = W_pruned.numel()
        pct = total_zero / total_elem * 100
        progress(f"   Sparsity: {total_zero}/{total_elem} ({pct:.1f}%), expected 50%")

        # Coupling check
        violations = 0
        for row in range(min(10, M)):
            for g in range(K // 16):
                for i in range(8):
                    c1, c2 = g * 16 + i, g * 16 + i + 8
                    z1 = (W_pruned[row, c1].abs() < 1e-10).item()
                    z2 = (W_pruned[row, c2].abs() < 1e-10).item()
                    if z1 != z2:
                        violations += 1
        if violations == 0:
            progress(f"   Coupling: OK (0 violations in 10 rows)")
        else:
            progress(f"   Coupling: {violations} VIOLATIONS in 10 rows!")

    # Magnitude
    check_sparsity("Magnitude", magnitude_coupled_24(W0))

    # SparseGPT
    check_sparsity("SparseGPT", sparsegpt_coupled_24(W0, H))

    # OBS split=2
    W_sp2 = torch.nn.Parameter(W0.clone())
    v2 = BlockView(W_sp2, size=(M, K // 16, 8, 2), stride=(K, 16, 1, 8))
    b2 = BlockSpec(v2, block_shape=(1, 1, 1, 2))
    g2 = GroupSpec(b2, group_shape=(1, 1, 4, 1))
    s2 = StructuredOBS(g2, H, damp=1e-4, C=C)
    s2.prune(num_nz=2, compensate="split", n_splits=2)
    check_sparsity("OBS split=2", W_sp2.data)
    del W_sp2, v2, b2, g2, s2


if __name__ == "__main__":
    main()
