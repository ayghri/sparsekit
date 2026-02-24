"""
StructuredOBS vs SparseGPT benchmark on real LLM data.

W (2560, 9728), X (244449, 9728)

Contiguous 2:4 sparsity:
  SparseGPT: Cholesky-based, magnitude mask, column-sequential error prop
  OBS local: optimal subset mask, within-group compensation only
  OBS full:  optimal subset mask + sequential full-column compensation
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


def load_data():
    progress("Loading data...")
    W = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    progress(f"  W: {W.shape} {W.dtype}")
    X = torch.load(X_PATH, map_location="cpu", weights_only=True)
    progress(f"  X: {X.shape} {X.dtype}")
    return W, X


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


# ── SparseGPT 2:4 ────────────────────────────────────────────────────────

def sparsegpt_24(W0, H, blocksize=128):
    """SparseGPT fasterprune for 2:4 structured sparsity."""
    M, K = W0.shape
    device = W0.device
    W = W0.clone().float()
    H = H.clone().float()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

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
            w = W1[:, i]
            d = Hinv1[i, i]

            if i % 4 == 0:
                end = min(i + 4, count)
                if end - i == 4:
                    tmp = W1[:, i:end] ** 2 / (torch.diag(Hinv1)[i:end].reshape(1, -1)) ** 2
                    _, bot = tmp.topk(2, dim=1, largest=False)
                    mask1[:, i:end].scatter_(1, bot, True)

            q = w.clone()
            q[mask1[:, i]] = 0.0
            Q1[:, i] = q

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err1

        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return W


# ── Hybrid: OBS subset selection + SparseGPT compensation ────────────────

def hybrid_obs_gptq_24(W0, H, C, blocksize=128):
    """OBS-optimal subset selection + SparseGPT Cholesky compensation.

    At each group boundary, uses C(4,2)=6 enumeration with inv(C[P,P]) cost
    to pick which 2 columns to prune (better than SparseGPT's w^2/d^2).
    Error propagation uses SparseGPT's Cholesky scheme.
    """
    from itertools import combinations

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

    # Precompute all 6 prune subsets
    all_subs = torch.tensor(
        list(combinations(range(4), 2)), device=device, dtype=torch.long
    )  # (6, 2)
    n_subs = all_subs.shape[0]
    eye2 = 1e-8 * torch.eye(2, device=device)

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

            if col % 4 == 0:
                end = min(i + 4, count)
                if end - i == 4:
                    # OBS subset selection using C = H^{-1}
                    abs_cols = torch.arange(col, col + 4, device=device)
                    C_gg = C[abs_cols][:, abs_cols]  # (4, 4)
                    W_g = W1[:, i:i+4]               # (M, 4)

                    all_costs = torch.empty(n_subs, M, device=device)
                    for si in range(n_subs):
                        pidx = all_subs[si]
                        C_PP = C_gg[pidx][:, pidx]   # (2, 2)
                        C_PP_inv = torch.inverse(C_PP + eye2)
                        W_P = W_g[:, pidx]            # (M, 2)
                        all_costs[si] = (W_P @ C_PP_inv * W_P).sum(dim=1)

                    best_si = all_costs.argmin(dim=0)  # (M,)
                    best_prune = all_subs[best_si]     # (M, 2)
                    mask1[:, i:i+4].scatter_(1, best_prune, True)

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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    W0, X_cpu = load_data()
    M, K = W0.shape
    N = X_cpu.shape[0]

    progress(f"\nMatrix: M={M}, K={K}, N={N}")

    progress("Computing H...")
    t0 = time.time()
    H = compute_H(X_cpu)
    torch.cuda.synchronize(DEVICE)
    progress(f"  H computed in {time.time() - t0:.1f}s")
    del X_cpu

    ref_norm = compute_loss(W0, torch.zeros_like(W0), H, N)
    progress(f"Reference ||X W0^T||_F^2 = {ref_norm:.4e}")

    progress(f"\n{'='*70}")
    progress("Contiguous 2:4 sparsity via BlockSpec/GroupSpec")
    progress(f"{'='*70}")

    # Precompute C = H^{-1} (shared across all methods)
    progress("\n  Precomputing C = H^{-1}...")
    t0 = time.time()
    C = StructuredOBS.compute_inverse(H, damp=1e-4)
    torch.cuda.synchronize(DEVICE)
    progress(f"  C computed in {time.time() - t0:.1f}s")

    def measure_sparsity(W_pruned):
        return (W_pruned.abs() < 1e-10).sum().item() / W_pruned.numel() * 100

    # ── SparseGPT ──
    progress("\n  [SparseGPT] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_sgpt = sparsegpt_24(W0, H, blocksize=128)
    torch.cuda.synchronize(DEVICE)
    t_sgpt = time.time() - t0
    loss_sgpt = compute_loss(W_sgpt, W0, H, N)
    sp_sgpt = measure_sparsity(W_sgpt)
    progress(f"  [SparseGPT] Loss={loss_sgpt:.4e}, Time={t_sgpt:.1f}s, Sparsity={sp_sgpt:.1f}%")
    del W_sgpt

    # ── Hybrid: OBS selection + GPTQ compensation ──
    progress("\n  [Hybrid OBS+GPTQ] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_hyb = hybrid_obs_gptq_24(W0, H, C, blocksize=128)
    torch.cuda.synchronize(DEVICE)
    t_hyb = time.time() - t0
    loss_hyb = compute_loss(W_hyb, W0, H, N)
    sp_hyb = measure_sparsity(W_hyb)
    progress(f"  [Hybrid OBS+GPTQ] Loss={loss_hyb:.4e}, Time={t_hyb:.1f}s, Sparsity={sp_hyb:.1f}%")
    del W_hyb

    # ── OBS local (independent groups) ──
    progress("\n  [OBS local] Running via StructuredOBS(compensate='local')...")
    W_local = torch.nn.Parameter(W0.clone())
    block_local = BlockSpec(W_local, block_shape=(1, 1))
    group_local = GroupSpec(block_local, group_shape=(1, 4))
    solver_local = StructuredOBS(group_local, H, damp=1e-4, C=C)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    solver_local.prune(num_nz=2, compensate="local")
    torch.cuda.synchronize(DEVICE)
    t_local = time.time() - t0
    loss_local = compute_loss(W_local.data, W0, H, N)
    sp_local = measure_sparsity(W_local.data)
    progress(f"  [OBS local] Loss={loss_local:.4e}, Time={t_local:.1f}s, Sparsity={sp_local:.1f}%")
    del W_local, block_local, group_local, solver_local

    # ── OBS full (sequential full-column compensation) ──
    progress("\n  [OBS full] Running via StructuredOBS(compensate='full')...")
    W_full = torch.nn.Parameter(W0.clone())
    block_full = BlockSpec(W_full, block_shape=(1, 1))
    group_full = GroupSpec(block_full, group_shape=(1, 4))
    solver_full = StructuredOBS(group_full, H, damp=1e-4, C=C)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    solver_full.prune(num_nz=2, compensate="full")
    torch.cuda.synchronize(DEVICE)
    t_full = time.time() - t0
    loss_full = compute_loss(W_full.data, W0, H, N)
    sp_full = measure_sparsity(W_full.data)
    progress(f"  [OBS full] Loss={loss_full:.4e}, Time={t_full:.1f}s, Sparsity={sp_full:.1f}%")
    del W_full, block_full, group_full, solver_full

    # ── OBS split variants ──
    results = [
        ("SparseGPT", loss_sgpt, t_sgpt, sp_sgpt),
        ("Hybrid OBS+GPTQ", loss_hyb, t_hyb, sp_hyb),
        ("OBS local", loss_local, t_local, sp_local),
        ("OBS full", loss_full, t_full, sp_full),
    ]

    for n_splits in [2, 4, 8, 16, 32, 64]:
        label = f"OBS split={n_splits}"
        progress(f"\n  [{label}] Running...")
        W_sp = torch.nn.Parameter(W0.clone())
        block_sp = BlockSpec(W_sp, block_shape=(1, 1))
        group_sp = GroupSpec(block_sp, group_shape=(1, 4))
        solver_sp = StructuredOBS(group_sp, H, damp=1e-4, C=C)
        torch.cuda.synchronize(DEVICE)
        t0 = time.time()
        solver_sp.prune(num_nz=2, compensate="split", n_splits=n_splits)
        torch.cuda.synchronize(DEVICE)
        t_sp = time.time() - t0
        loss_sp = compute_loss(W_sp.data, W0, H, N)
        sp_sp = measure_sparsity(W_sp.data)
        progress(f"  [{label}] Loss={loss_sp:.4e}, Time={t_sp:.1f}s, Sparsity={sp_sp:.1f}%")
        results.append((label, loss_sp, t_sp, sp_sp))
        del W_sp, block_sp, group_sp, solver_sp

    # ── Report ──
    progress(f"\n  {'Method':<30} {'Loss':>14} {'Norm.':>10} {'Sparsity':>10} {'Time':>10}")
    progress(f"  {'-'*78}")
    for name, loss, t, sp in results:
        progress(f"  {name:<30} {loss:>14.4e} {loss/ref_norm*100:>8.4f}% {sp:>8.1f}% {t:>8.1f}s")

    progress(f"\n  --- vs SparseGPT ---")
    for name, loss, _, _ in results[1:]:
        if loss < loss_sgpt:
            progress(f"  {name} beats SparseGPT by {(1 - loss/loss_sgpt)*100:.2f}%")
        else:
            progress(f"  SparseGPT beats {name} by {(1 - loss_sgpt/loss)*100:.2f}%")

    progress(f"\n  --- vs OBS full ---")
    for name, loss, _, _ in results[4:]:
        if loss < loss_full:
            progress(f"  {name} beats OBS full by {(1 - loss/loss_full)*100:.2f}%")
        else:
            progress(f"  OBS full beats {name} by {(1 - loss_full/loss)*100:.2f}%")


if __name__ == "__main__":
    main()
