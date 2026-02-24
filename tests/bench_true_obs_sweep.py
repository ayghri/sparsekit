"""
True OBS group-size sweep: how many groups to batch before Schur update?

Each batch of N groups:
  1. Select best 2:4 subset per group independently (6 subsets each, current C)
  2. Compensate W for all 2N pruned columns at once
  3. One rank-2N Schur update to C

B=32 rows, K=9728, 2432 groups of 4.
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


# ── True OBS: batched N groups at a time ─────────────────────────────────

def true_obs_batched(W0, C_base, n_groups=1):
    """True OBS processing n_groups contiguous groups per step.

    Each step:
      1. Select best 2:4 subset per group (6 subsets, current C)
      2. Joint compensation for all 2*n_groups pruned columns
      3. One rank-(2*n_groups) Schur update via Cholesky decomposition
    """
    R, K = W0.shape
    device = W0.device
    num_groups = K // 4

    C = C_base.unsqueeze(0).expand(R, -1, -1).clone()  # (R, K, K)
    W = W0.clone().float()
    arange_R = torch.arange(R, device=device)

    all_subs = torch.tensor(
        list(combinations(range(4), 2)), device=device, dtype=torch.long
    )  # (6, 2)
    eye2 = 1e-8 * torch.eye(2, device=device)

    num_blocks = (num_groups + n_groups - 1) // n_groups

    for blk in range(num_blocks):
        g_start = blk * n_groups
        g_end = min(g_start + n_groups, num_groups)
        n_g = g_end - g_start

        # Select best subset per group using current C
        block_pruned = []
        for g in range(g_start, g_end):
            base = g * 4
            cols = torch.arange(base, base + 4, device=device)

            C_g = C[:, cols][:, :, cols]  # (R, 4, 4)
            W_g = W[:, cols]              # (R, 4)

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

            pruned_cols = all_subs[best_si] + base  # (R, 2)
            block_pruned.append(pruned_cols)

        # Combine all pruned cols: (R, 2*n_g)
        all_p = torch.cat(block_pruned, dim=1)
        np_total = all_p.shape[1]

        # Joint compensation + Schur update
        eye_n = 1e-8 * torch.eye(np_total, device=device)
        pc_exp = all_p.unsqueeze(1).expand(R, K, np_total)
        C_col_P = C.gather(2, pc_exp)                               # (R, K, np)

        C_PP = C_col_P.gather(1, all_p.unsqueeze(2).expand(R, np_total, np_total))
        C_PP_inv = LA.inv(C_PP + eye_n)

        # Weight compensation
        W_P = W.gather(1, all_p)
        comp = torch.bmm(C_col_P, C_PP_inv)
        delta = torch.bmm(comp, W_P.unsqueeze(2)).squeeze(2)        # (R, K)
        W -= delta
        W.scatter_(1, all_p, torch.zeros(R, np_total, device=device))

        # Schur update: C -= C[:,P] @ inv(C[P,P]) @ C[P,:]
        L = LA.cholesky(C_PP_inv + eye_n)
        U = torch.bmm(C_col_P, L)                                   # (R, K, np)
        C.baddbmm_(U, U.transpose(1, 2), alpha=-1.0)

        # Zero pruned rows/cols in C
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

    print(f"\nFirst {B} rows, K={W0.shape[1]}, {W0.shape[1]//4} groups", flush=True)
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

    # ── True OBS sweep ──
    for ng in [1, 2, 4, 8, 16, 32, 64, 128]:
        label = f"True OBS ng={ng}"
        print(f"\n  [{label}] Running...", flush=True)
        torch.cuda.synchronize(DEVICE)
        t0 = time.time()
        W_obs = true_obs_batched(W0, C, n_groups=ng)
        torch.cuda.synchronize(DEVICE)
        t = time.time() - t0
        loss = compute_loss(W_obs, W0, H, N)
        sp = measure_sparsity(W_obs)
        check_24(W_obs, label)
        results.append((label, loss, t, sp))
        print(f"  [{label}] Loss={loss:.4e}, Time={t:.1f}s, Sp={sp:.1f}%", flush=True)
        del W_obs

    # ── Report ──
    loss_sgpt = results[0][1]
    print(f"\n  {'Method':<25} {'Loss':>14} {'Norm.':>10} {'Sp':>6} {'Time':>8} {'vs SGPT':>10}", flush=True)
    print(f"  {'-'*77}", flush=True)
    for name, loss, t, sp in results:
        vs = (1 - loss / loss_sgpt) * 100 if loss < loss_sgpt else -(1 - loss_sgpt / loss) * 100
        sign = "+" if vs >= 0 else ""
        print(f"  {name:<25} {loss:>14.4e} {loss/ref*100:>8.4f}% {sp:>4.0f}% {t:>7.1f}s {sign}{vs:>8.2f}%", flush=True)

    best_name, best_loss, best_t, _ = min(results, key=lambda x: x[1])
    print(f"\n  Best: {best_name} ({best_loss:.4e}, {best_loss/ref*100:.4f}%, {best_t:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
