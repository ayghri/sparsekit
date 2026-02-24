"""
True OBS group ordering: largest-loss-first vs left-to-right.

Pre-score all groups using initial C, sort by OBS cost (largest first),
then process in that order with ng=16 batched Schur updates.

Idea: groups with highest pruning cost benefit most from compensating
across many still-active columns, so process them first.

B=32 rows, K=9728, 2432 groups of 4, ng=16.
"""

import time
from itertools import combinations

import torch
import torch.linalg as LA

from sparsekit import StructuredOBS

DEVICE = torch.device("cuda:1")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"
B = 32
NG = 16


def compute_H(X, batch_size=4096):
    N, K = X.shape
    H = torch.zeros(K, K, device=DEVICE, dtype=torch.float32)
    for i in range(0, N, batch_size):
        X_b = X[i:i+batch_size].to(device=DEVICE, dtype=torch.float32)
        H.addmm_(X_b.T, X_b)
    H /= N
    return H


def compute_loss(W_pruned, W0, H, N):
    dW = W_pruned - W0
    return ((dW @ H) * dW).sum().item() * N


def measure_sparsity(W_pruned):
    return (W_pruned.abs() < 1e-10).sum().item() / W_pruned.numel() * 100


def check_24(W_pruned, label):
    M_, K_ = W_pruned.shape
    zeros = (W_pruned.abs() < 1e-10).view(M_, K_ // 4, 4).sum(dim=2)
    ok = (zeros == 2).all().item()
    if not ok:
        bad = (zeros != 2).sum().item()
        print(f"  WARNING: {label} has {bad} groups violating 2:4!", flush=True)
    return ok


# ── SparseGPT 2:4 ────────────────────────────────────────────────────────

def sparsegpt_24(W0, H, blocksize=128):
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
            if i % 4 == 0 and i + 4 <= count:
                tmp = W1[:, i:i+4] ** 2 / (torch.diag(Hinv1)[i:i+4].reshape(1, -1)) ** 2
                _, bot = tmp.topk(2, dim=1, largest=False)
                mask1[:, i:i+4].scatter_(1, bot, True)

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


# ── True OBS with configurable group ordering ────────────────────────────

def true_obs_ordered(W0, C_base, n_groups=NG, order="left_to_right"):
    """True OBS with batched Schur updates and configurable group order.

    Args:
        order: "left_to_right", "largest_first", "smallest_first"
    """
    R, K = W0.shape
    device = W0.device
    num_groups = K // 4

    C = C_base.unsqueeze(0).expand(R, -1, -1).clone()
    W = W0.clone().float()
    arange_R = torch.arange(R, device=device)

    all_subs = torch.tensor(
        list(combinations(range(4), 2)), device=device, dtype=torch.long
    )
    eye2 = 1e-8 * torch.eye(2, device=device)

    # Pre-score all groups to determine ordering and best subsets
    group_costs = torch.zeros(R, num_groups, device=device)
    group_best_si = torch.zeros(R, num_groups, dtype=torch.long, device=device)

    for g in range(num_groups):
        base = g * 4
        cols = torch.arange(base, base + 4, device=device)
        C_g = C_base[cols][:, cols].unsqueeze(0).expand(R, -1, -1)
        W_g = W[:, cols]

        best_cost = torch.full((R,), float("inf"), device=device)
        best_si = torch.zeros(R, dtype=torch.long, device=device)

        for si in range(6):
            pidx = all_subs[si]
            C_PP = C_g[:, pidx][:, :, pidx]
            W_P = W_g[:, pidx]
            C_PP_inv = LA.inv(C_PP + eye2)
            cost = (torch.bmm(W_P.unsqueeze(1), C_PP_inv).squeeze(1) * W_P).sum(1)
            better = cost < best_cost
            best_cost[better] = cost[better]
            best_si[better] = si

        group_costs[:, g] = best_cost
        group_best_si[:, g] = best_si

    # Determine group ordering (by mean cost across rows)
    mean_costs = group_costs.mean(dim=0)  # (num_groups,)
    if order == "largest_first":
        group_order = torch.argsort(mean_costs, descending=True)
    elif order == "smallest_first":
        group_order = torch.argsort(mean_costs, descending=False)
    else:  # left_to_right
        group_order = torch.arange(num_groups, device=device)

    # Process groups in determined order, batched by n_groups
    num_blocks = (num_groups + n_groups - 1) // n_groups

    for blk in range(num_blocks):
        idx_start = blk * n_groups
        idx_end = min(idx_start + n_groups, num_groups)
        batch_groups = group_order[idx_start:idx_end]
        n_g = batch_groups.shape[0]

        # Re-score each group using current C (not initial C)
        block_pruned = []
        for g in batch_groups.tolist():
            base = g * 4
            cols = torch.arange(base, base + 4, device=device)

            C_g = C[:, cols][:, :, cols]
            W_g = W[:, cols]

            best_cost = torch.full((R,), float("inf"), device=device)
            best_si = torch.zeros(R, dtype=torch.long, device=device)

            for si in range(6):
                pidx = all_subs[si]
                C_PP = C_g[:, pidx][:, :, pidx]
                W_P = W_g[:, pidx]
                C_PP_inv = LA.inv(C_PP + eye2)
                cost = (torch.bmm(W_P.unsqueeze(1), C_PP_inv).squeeze(1) * W_P).sum(1)
                better = cost < best_cost
                best_cost[better] = cost[better]
                best_si[better] = si

            pruned_cols = all_subs[best_si] + base
            block_pruned.append(pruned_cols)

        # Combine: (R, 2*n_g)
        all_p = torch.cat(block_pruned, dim=1)
        np_total = all_p.shape[1]

        eye_n = 1e-8 * torch.eye(np_total, device=device)
        pc_exp = all_p.unsqueeze(1).expand(R, K, np_total)
        C_col_P = C.gather(2, pc_exp)

        C_PP = C_col_P.gather(1, all_p.unsqueeze(2).expand(R, np_total, np_total))
        C_PP_inv = LA.inv(C_PP + eye_n)

        W_P = W.gather(1, all_p)
        comp = torch.bmm(C_col_P, C_PP_inv)
        delta = torch.bmm(comp, W_P.unsqueeze(2)).squeeze(2)
        W -= delta
        W.scatter_(1, all_p, torch.zeros(R, np_total, device=device))

        L = LA.cholesky(C_PP_inv + eye_n)
        U = torch.bmm(C_col_P, L)
        C.baddbmm_(U, U.transpose(1, 2), alpha=-1.0)

        for j in range(np_total):
            pc_j = all_p[:, j]
            C[arange_R, pc_j, :] = 0.0
            C[arange_R, :, pc_j] = 0.0

    return W


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading data...", flush=True)
    W0_full = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X = torch.load(X_PATH, map_location="cpu", weights_only=True)
    N = X.shape[0]
    print(f"  W: {W0_full.shape}, X: {X.shape}", flush=True)

    print("Computing H...", flush=True)
    H = compute_H(X)
    del X
    torch.cuda.synchronize(DEVICE)

    print("Computing C = H^{-1}...", flush=True)
    C = StructuredOBS.compute_inverse(H, damp=1e-4)
    torch.cuda.synchronize(DEVICE)

    W0 = W0_full[:B]
    ref = compute_loss(W0, torch.zeros_like(W0), H, N)

    print(f"\nFirst {B} rows, K={W0.shape[1]}, {W0.shape[1]//4} groups, ng={NG}", flush=True)
    print(f"Reference ||X W0^T||_F^2 = {ref:.4e}", flush=True)

    results = []

    # ── SparseGPT ──
    print(f"\n  [SparseGPT] Running...", flush=True)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_sgpt = sparsegpt_24(W0, H)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_sgpt, W0, H, N)
    sp = measure_sparsity(W_sgpt)
    check_24(W_sgpt, "SparseGPT")
    results.append(("SparseGPT", loss, t, sp))
    print(f"  [SparseGPT] Loss={loss:.4e}, Time={t:.3f}s, Sp={sp:.1f}%", flush=True)
    del W_sgpt

    # ── True OBS: left-to-right ──
    print(f"\n  [Left-to-right] Running...", flush=True)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_ltr = true_obs_ordered(W0, C, order="left_to_right")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_ltr, W0, H, N)
    sp = measure_sparsity(W_ltr)
    check_24(W_ltr, "Left-to-right")
    results.append(("Left-to-right", loss, t, sp))
    print(f"  [Left-to-right] Loss={loss:.4e}, Time={t:.1f}s, Sp={sp:.1f}%", flush=True)
    del W_ltr

    # ── True OBS: largest first ──
    print(f"\n  [Largest first] Running...", flush=True)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_lf = true_obs_ordered(W0, C, order="largest_first")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_lf, W0, H, N)
    sp = measure_sparsity(W_lf)
    check_24(W_lf, "Largest first")
    results.append(("Largest first", loss, t, sp))
    print(f"  [Largest first] Loss={loss:.4e}, Time={t:.1f}s, Sp={sp:.1f}%", flush=True)
    del W_lf

    # ── True OBS: smallest first ──
    print(f"\n  [Smallest first] Running...", flush=True)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_sf = true_obs_ordered(W0, C, order="smallest_first")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_sf, W0, H, N)
    sp = measure_sparsity(W_sf)
    check_24(W_sf, "Smallest first")
    results.append(("Smallest first", loss, t, sp))
    print(f"  [Smallest first] Loss={loss:.4e}, Time={t:.1f}s, Sp={sp:.1f}%", flush=True)
    del W_sf

    # ── Report ──
    loss_sgpt = results[0][1]
    loss_ltr = results[1][1]
    print(f"\n  {'Method':<25} {'Loss':>14} {'Norm.':>10} {'Sp':>6} {'Time':>8} {'vs SGPT':>10} {'vs L2R':>10}", flush=True)
    print(f"  {'-'*87}", flush=True)
    for name, loss, t, sp in results:
        vs_s = (1 - loss / loss_sgpt) * 100 if loss < loss_sgpt else -(1 - loss_sgpt / loss) * 100
        vs_l = (1 - loss / loss_ltr) * 100 if loss < loss_ltr else -(1 - loss_ltr / loss) * 100
        print(f"  {name:<25} {loss:>14.4e} {loss/ref*100:>8.4f}% {sp:>4.0f}% {t:>7.1f}s {vs_s:>+9.2f}% {vs_l:>+9.2f}%", flush=True)


if __name__ == "__main__":
    main()
