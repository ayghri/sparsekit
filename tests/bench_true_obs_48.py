"""
True OBS for 4:8 structured sparsity with contiguous block pairing.

block_shape=(1,2): pairs of 2 contiguous columns form a block.
group_shape=(1,4): 4 blocks per group → 8 columns per group.
Keep 2 of 4 blocks (4 of 8 columns). C(4,2)=6 subsets.

True OBS: per-row C = H^{-1} with rank-4 Schur updates per group.
Batched ng groups at a time.

B=32 rows, K=9728, 1216 groups of 8.
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


def check_48(W_pruned, label):
    """Verify each group of 8 has exactly 4 zeros (2 pruned blocks of 2) per row."""
    M_, K_ = W_pruned.shape
    zeros = (W_pruned.abs() < 1e-10).view(M_, K_ // 8, 4, 2).all(dim=3)  # (M, G, 4) block-level
    pruned_per_group = zeros.sum(dim=2)  # (M, G)
    ok = (pruned_per_group == 2).all().item()
    if not ok:
        bad = (pruned_per_group != 2).sum().item()
        print(f"  WARNING: {label} has {bad} groups violating 4:8!", flush=True)
    return ok


# ── SparseGPT 4:8 ─────────────────────────────────────────────────────────

def sparsegpt_48(W0, H, blocksize=128):
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
            if col % 8 == 0:
                end = min(i + 8, count)
                if end - i == 8:
                    col_scores = W1[:, i:end] ** 2 / (
                        torch.diag(Hinv1)[i:end].reshape(1, -1)
                    ) ** 2
                    block_scores = col_scores.view(M, 4, 2).sum(dim=-1)  # (M, 4)
                    _, bot = block_scores.topk(2, dim=1, largest=False)
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


# ── Magnitude 4:8 ─────────────────────────────────────────────────────────

def magnitude_48(W0):
    W = W0.clone()
    M, K = W.shape
    Wg = W.view(M, K // 8, 4, 2)
    block_norms = Wg.abs().sum(dim=-1)  # (M, G, 4)
    _, bot = block_norms.topk(2, dim=-1, largest=False)
    mask = torch.zeros(M, K // 8, 4, dtype=torch.bool, device=W.device)
    mask.scatter_(2, bot, True)
    Wg[mask.unsqueeze(-1).expand_as(Wg)] = 0.0
    return W


# ── True OBS 4:8 batched ──────────────────────────────────────────────────

def true_obs_48(W0, C_base, ng=1):
    """True OBS for 4:8 with per-row Schur, batched ng groups at a time.

    Each group: 8 columns = 4 blocks of 2. Prune 2 blocks (4 cols).
    C(4,2)=6 subsets. Rank-4 Schur update per group.
    """
    R, K = W0.shape
    device = W0.device
    num_groups = K // 8

    C = C_base.unsqueeze(0).expand(R, -1, -1).clone()  # (R, K, K)
    W = W0.clone().float()
    arange_R = torch.arange(R, device=device)

    # Block prune subsets: which 2 of 4 blocks to prune
    all_subs = torch.tensor(
        list(combinations(range(4), 2)), device=device, dtype=torch.long
    )  # (6, 2)

    # Expand block indices to column offsets within group of 8
    # block i -> columns [2i, 2i+1]
    blk_to_cols = (all_subs.unsqueeze(-1) * 2 +
                   torch.arange(2, device=device)).view(6, 4)  # (6, 4)

    eye4 = 1e-8 * torch.eye(4, device=device)

    num_batches = (num_groups + ng - 1) // ng

    for blk in range(num_batches):
        g_start = blk * ng
        g_end = min(g_start + ng, num_groups)

        block_pruned = []
        for g in range(g_start, g_end):
            base = g * 8
            group_cols = torch.arange(base, base + 8, device=device)

            C_g = C[:, group_cols][:, :, group_cols]  # (R, 8, 8)
            W_g = W[:, group_cols]                     # (R, 8)

            best_cost = torch.full((R,), float("inf"), device=device)
            best_si = torch.zeros(R, dtype=torch.long, device=device)

            for si in range(6):
                col_offsets = blk_to_cols[si]              # (4,)
                C_PP = C_g[:, col_offsets][:, :, col_offsets]
                W_P = W_g[:, col_offsets]
                C_PP_inv = LA.inv(C_PP + eye4)
                cost = (torch.bmm(W_P.unsqueeze(1), C_PP_inv).squeeze(1) * W_P).sum(1)
                better = cost < best_cost
                best_cost[better] = cost[better]
                best_si[better] = si

            # Pruned columns (absolute)
            pruned_offsets = blk_to_cols[best_si]          # (R, 4)
            pruned_cols = pruned_offsets + base             # (R, 4)
            block_pruned.append(pruned_cols)

        # Combine: (R, 4 * n_g)
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
    K = W0.shape[1]
    ref = compute_loss(W0, torch.zeros_like(W0), H, N)

    print(f"\nFirst {B} rows, K={K}, {K//8} groups of 8 (4 blocks of 2)", flush=True)
    print(f"Reference ||X W0^T||_F^2 = {ref:.4e}", flush=True)

    results = []

    # ── Magnitude ──
    print(f"\n  [Magnitude] Running...", flush=True)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_mag = magnitude_48(W0)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_mag, W0, H, N)
    sp = measure_sparsity(W_mag)
    check_48(W_mag, "Magnitude")
    results.append(("Magnitude", loss, t, sp))
    print(f"  [Magnitude] Loss={loss:.4e}, Time={t:.3f}s, Sp={sp:.1f}%", flush=True)
    del W_mag

    # ── SparseGPT ──
    print(f"\n  [SparseGPT] Running...", flush=True)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_sgpt = sparsegpt_48(W0, H)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_sgpt, W0, H, N)
    sp = measure_sparsity(W_sgpt)
    check_48(W_sgpt, "SparseGPT")
    results.append(("SparseGPT", loss, t, sp))
    print(f"  [SparseGPT] Loss={loss:.4e}, Time={t:.3f}s, Sp={sp:.1f}%", flush=True)
    del W_sgpt

    # ── True OBS sweep ──
    for ng_val in [1, 2, 4, 8, 16, 32, 64]:
        label = f"True OBS ng={ng_val}"
        print(f"\n  [{label}] Running...", flush=True)
        torch.cuda.synchronize(DEVICE)
        t0 = time.time()
        W_obs = true_obs_48(W0, C, ng=ng_val)
        torch.cuda.synchronize(DEVICE)
        t = time.time() - t0
        loss = compute_loss(W_obs, W0, H, N)
        sp = measure_sparsity(W_obs)
        check_48(W_obs, label)
        results.append((label, loss, t, sp))
        print(f"  [{label}] Loss={loss:.4e}, Time={t:.1f}s, Sp={sp:.1f}%", flush=True)
        del W_obs

    # ── Report ──
    loss_sgpt = results[1][1]
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
