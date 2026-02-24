"""
OBS full (frozen C) vs True OBS variants on first B rows.

True OBS maintains (B, K, K) C tensor — one inverse per row.
Memory: B * K^2 * 4 bytes. For K=9728:
  B=32:  ~11 GB
  B=64:  ~23 GB  (OOM)
"""

import time
from itertools import combinations

import torch
import torch.linalg as LA

from sparsekit import BlockSpec, GroupSpec, StructuredOBS

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


# ── True OBS: greedy element-by-element ──────────────────────────────────

def true_obs_element(W0, C_base):
    """Original true OBS: pick globally cheapest element, rank-1 Schur."""
    B, K = W0.shape
    device = W0.device
    num_groups = K // 4
    total_prunes = 2 * num_groups

    C = C_base.unsqueeze(0).expand(B, -1, -1).clone()
    w = W0.clone().float()

    remaining = torch.full((B, num_groups), 2, dtype=torch.long, device=device)
    pruned = torch.zeros(B, K, dtype=torch.bool, device=device)
    arange_B = torch.arange(B, device=device)

    for step in range(total_prunes):
        if step % 500 == 0:
            print(f"    step {step}/{total_prunes}", flush=True)

        diag_C = torch.diagonal(C, dim1=1, dim2=2)
        scores = w ** 2 / (diag_C.abs() + 1e-10)
        scores[pruned] = float("inf")
        group_done = (remaining <= 0).repeat_interleave(4, dim=1)
        scores[group_done] = float("inf")

        p = scores.argmin(dim=1)

        p_col = p.view(B, 1, 1).expand(B, K, 1)
        c_col = C.gather(2, p_col).squeeze(2)
        p_row = p.view(B, 1, 1).expand(B, 1, K)
        c_row = C.gather(1, p_row).squeeze(1)

        c_pp = C[arange_B, p, p]
        safe_cpp = c_pp.abs() + 1e-12

        wp = w[arange_B, p]
        w -= (wp / safe_cpp).unsqueeze(1) * c_col
        w[arange_B, p] = 0.0

        inv_sqrt = (1.0 / safe_cpp.sqrt()).unsqueeze(1)
        C.baddbmm_(
            (c_col * inv_sqrt).unsqueeze(2),
            (c_row * inv_sqrt).unsqueeze(1),
            alpha=-1.0,
        )
        C[arange_B, p, :] = 0.0
        C[arange_B, :, p] = 0.0

        pruned[arange_B, p] = True
        remaining[arange_B, p // 4] -= 1

    return w


# ── True OBS: C(4,2) group-level with Schur ─────────────────────────────

def true_obs_groups(W0, C_base):
    """True OBS with C(4,2) subset selection per group + per-row Schur updates.

    For each group left-to-right:
      1. Enumerate all 6 subsets, pick best per row using current per-row C
      2. Compensate all K columns using per-row C
      3. Rank-2 Schur update to C (via baddbmm_)
    """
    B, K = W0.shape
    device = W0.device
    num_groups = K // 4

    C = C_base.unsqueeze(0).expand(B, -1, -1).clone()  # (B, K, K)
    W = W0.clone().float()
    arange_B = torch.arange(B, device=device)

    all_subs = torch.tensor(
        list(combinations(range(4), 2)), device=device, dtype=torch.long
    )  # (6, 2)
    n_subs = all_subs.shape[0]
    eye2 = 1e-8 * torch.eye(2, device=device)

    for g in range(num_groups):
        if g % 200 == 0:
            print(f"    group {g}/{num_groups}", flush=True)

        base = g * 4
        cols = torch.arange(base, base + 4, device=device)

        # Score all 6 subsets per row using current per-row C
        C_g = C[:, cols][:, :, cols]  # (B, 4, 4)
        W_g = W[:, cols]              # (B, 4)

        best_cost = torch.full((B,), float("inf"), device=device)
        best_si = torch.zeros(B, dtype=torch.long, device=device)

        for si in range(n_subs):
            pidx = all_subs[si]
            C_PP = C_g[:, pidx][:, :, pidx]  # (B, 2, 2)
            W_P = W_g[:, pidx]                # (B, 2)
            C_PP_inv = LA.inv(C_PP + eye2)
            cost = (torch.bmm(W_P.unsqueeze(1), C_PP_inv).squeeze(1) * W_P).sum(1)
            better = cost < best_cost
            best_cost[better] = cost[better]
            best_si[better] = si

        # Per-row pruned columns (absolute)
        pruned_local = all_subs[best_si]   # (B, 2)
        pruned_cols = pruned_local + base   # (B, 2)

        # Gather C[:, :, P] per row — (B, K, 2)
        pc_exp = pruned_cols.unsqueeze(1).expand(B, K, 2)
        C_col_P = C.gather(2, pc_exp)

        # C[P, P] — (B, 2, 2) via double gather
        C_PP = C_col_P.gather(1, pruned_cols.unsqueeze(2).expand(B, 2, 2))
        C_PP_inv = LA.inv(C_PP + eye2)

        # Weight compensation: W -= C[:,:,P] @ inv(C[P,P]) @ W[P]
        W_P = W.gather(1, pruned_cols)               # (B, 2)
        comp = torch.bmm(C_col_P, C_PP_inv)          # (B, K, 2)
        delta = torch.bmm(comp, W_P.unsqueeze(2)).squeeze(2)  # (B, K)
        W -= delta
        W.scatter_(1, pruned_cols, torch.zeros(B, 2, device=device))

        # Schur update: C -= C[:,:,P] @ inv(C[P,P]) @ C[P,:,:]
        # = C -= comp @ C_col_P^T
        # Decompose via Cholesky for memory-efficient baddbmm_:
        #   comp @ C_col_P^T = (C_col_P @ L) @ (C_col_P @ L)^T  where L L^T = C_PP_inv
        L = LA.cholesky(C_PP_inv + eye2)              # (B, 2, 2)
        U = torch.bmm(C_col_P, L)                     # (B, K, 2)
        C.baddbmm_(U, U.transpose(1, 2), alpha=-1.0)  # in-place rank-2 update

        # Zero pruned rows/cols
        for j in range(2):
            pc_j = pruned_cols[:, j]
            C[arange_B, pc_j, :] = 0.0
            C[arange_B, :, pc_j] = 0.0

    return W


# ── True OBS: 2 groups at a time (rank-4 Schur) ─────────────────────────

def true_obs_groups2(W0, C_base):
    """True OBS processing 2 contiguous groups (8 cols) at a time.

    For each pair of groups:
      1. Enumerate C(4,2)*C(4,2)=36 joint subsets (2:4 constraint per group)
      2. Pick best per row using current per-row C
      3. Compensate + rank-4 Schur update
    """
    B, K = W0.shape
    device = W0.device
    num_groups = K // 4
    num_pairs = num_groups // 2

    C = C_base.unsqueeze(0).expand(B, -1, -1).clone()  # (B, K, K)
    W = W0.clone().float()
    arange_B = torch.arange(B, device=device)

    # All prune subsets for a single group of 4 (prune 2)
    single_subs = list(combinations(range(4), 2))  # 6 subsets

    # Joint subsets: pairs of single-group subsets -> 36 combos
    # Each gives 4 pruned columns within an 8-col block
    joint_subs = []
    for s0 in single_subs:
        for s1 in single_subs:
            # s0 are local indices in first group (0-3)
            # s1 are local indices in second group (4-7)
            joint_subs.append(list(s0) + [x + 4 for x in s1])
    joint_subs = torch.tensor(joint_subs, device=device, dtype=torch.long)  # (36, 4)
    n_joint = joint_subs.shape[0]
    eye4 = 1e-8 * torch.eye(4, device=device)

    for gp in range(num_pairs):
        if gp % 100 == 0:
            print(f"    group-pair {gp}/{num_pairs}", flush=True)

        base = gp * 8
        cols = torch.arange(base, base + 8, device=device)

        C_g = C[:, cols][:, :, cols]  # (B, 8, 8)
        W_g = W[:, cols]              # (B, 8)

        best_cost = torch.full((B,), float("inf"), device=device)
        best_si = torch.zeros(B, dtype=torch.long, device=device)

        for si in range(n_joint):
            pidx = joint_subs[si]
            C_PP = C_g[:, pidx][:, :, pidx]  # (B, 4, 4)
            W_P = W_g[:, pidx]                # (B, 4)
            C_PP_inv = LA.inv(C_PP + eye4)
            cost = (torch.bmm(W_P.unsqueeze(1), C_PP_inv).squeeze(1) * W_P).sum(1)
            better = cost < best_cost
            best_cost[better] = cost[better]
            best_si[better] = si

        pruned_local = joint_subs[best_si]     # (B, 4)
        pruned_cols = pruned_local + base       # (B, 4)

        # Gather C[:, :, P] per row — (B, K, 4)
        pc_exp = pruned_cols.unsqueeze(1).expand(B, K, 4)
        C_col_P = C.gather(2, pc_exp)

        # C[P, P] — (B, 4, 4)
        C_PP = C_col_P.gather(1, pruned_cols.unsqueeze(2).expand(B, 4, 4))
        C_PP_inv = LA.inv(C_PP + eye4)

        # Weight compensation
        W_P = W.gather(1, pruned_cols)
        comp = torch.bmm(C_col_P, C_PP_inv)
        delta = torch.bmm(comp, W_P.unsqueeze(2)).squeeze(2)
        W -= delta
        W.scatter_(1, pruned_cols, torch.zeros(B, 4, device=device))

        # Rank-4 Schur update via Cholesky decomposition
        L = LA.cholesky(C_PP_inv + eye4)
        U = torch.bmm(C_col_P, L)              # (B, K, 4)
        C.baddbmm_(U, U.transpose(1, 2), alpha=-1.0)

        for j in range(4):
            pc_j = pruned_cols[:, j]
            C[arange_B, pc_j, :] = 0.0
            C[arange_B, :, pc_j] = 0.0

    # Handle leftover group if odd number
    if num_groups % 2 == 1:
        g = num_groups - 1
        base = g * 4
        cols = torch.arange(base, base + 4, device=device)
        single = torch.tensor(list(combinations(range(4), 2)),
                              device=device, dtype=torch.long)
        eye2 = 1e-8 * torch.eye(2, device=device)
        C_g = C[:, cols][:, :, cols]
        W_g = W[:, cols]
        best_cost = torch.full((B,), float("inf"), device=device)
        best_si = torch.zeros(B, dtype=torch.long, device=device)
        for si in range(6):
            pidx = single[si]
            C_PP = C_g[:, pidx][:, :, pidx]
            W_P = W_g[:, pidx]
            C_PP_inv = LA.inv(C_PP + eye2)
            cost = (torch.bmm(W_P.unsqueeze(1), C_PP_inv).squeeze(1) * W_P).sum(1)
            better = cost < best_cost
            best_cost[better] = cost[better]
            best_si[better] = si
        pruned_local = single[best_si]
        pruned_cols = pruned_local + base
        pc_exp = pruned_cols.unsqueeze(1).expand(B, K, 2)
        C_col_P = C.gather(2, pc_exp)
        C_PP = C_col_P.gather(1, pruned_cols.unsqueeze(2).expand(B, 2, 2))
        C_PP_inv = LA.inv(C_PP + eye2)
        W_P = W.gather(1, pruned_cols)
        comp = torch.bmm(C_col_P, C_PP_inv)
        delta = torch.bmm(comp, W_P.unsqueeze(2)).squeeze(2)
        W -= delta
        W.scatter_(1, pruned_cols, torch.zeros(B, 2, device=device))

    return W


# ── SparseGPT 2:4 ────────────────────────────────────────────────────────

def sparsegpt_24(W0, H, blocksize=128):
    """SparseGPT fasterprune for 2:4 structured sparsity."""
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


# ── Magnitude 2:4 ────────────────────────────────────────────────────────

def magnitude_24(W0):
    W = W0.clone()
    M, K = W.shape
    Wg = W.view(M, K // 4, 4)
    _, idx = Wg.abs().topk(2, dim=2, largest=False)
    mask = torch.zeros_like(Wg, dtype=torch.bool)
    mask.scatter_(2, idx, True)
    Wg[mask] = 0.0
    return W


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    W0_full = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X = torch.load(X_PATH, map_location="cpu", weights_only=True)
    N = X.shape[0]
    print(f"  W: {W0_full.shape}, X: {X.shape}")

    print("Computing H...")
    H = compute_H(X)
    del X
    torch.cuda.synchronize(DEVICE)

    print("Computing C = H^{-1}...")
    C = StructuredOBS.compute_inverse(H, damp=1e-4)
    torch.cuda.synchronize(DEVICE)

    W0 = W0_full[:B]
    ref = compute_loss(W0, torch.zeros_like(W0), H, N)

    print(f"\nFirst {B} rows, K={W0.shape[1]}")
    print(f"Reference ||X W0^T||_F^2 = {ref:.4e}")

    def measure_sparsity(W_pruned):
        return (W_pruned.abs() < 1e-10).sum().item() / W_pruned.numel() * 100

    def check_24(W_pruned, label):
        """Verify every group of 4 has exactly 2 zeros per row."""
        M_, K_ = W_pruned.shape
        zeros = (W_pruned.abs() < 1e-10).view(M_, K_ // 4, 4).sum(dim=2)  # (M, G)
        ok = (zeros == 2).all().item()
        if not ok:
            bad = (zeros != 2).sum().item()
            print(f"  WARNING: {label} has {bad} groups violating 2:4!")
        return ok

    results = []

    # ── Magnitude ──
    print(f"\n  [Magnitude] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_mag = magnitude_24(W0)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_mag, W0, H, N)
    sp = measure_sparsity(W_mag)
    check_24(W_mag, "Magnitude")
    results.append(("Magnitude", loss, t, sp))
    print(f"  [Magnitude] Loss={loss:.4e}, Time={t:.3f}s, Sp={sp:.1f}%")
    del W_mag

    # ── SparseGPT ──
    print(f"\n  [SparseGPT] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_sgpt = sparsegpt_24(W0, H)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_sgpt, W0, H, N)
    sp = measure_sparsity(W_sgpt)
    check_24(W_sgpt, "SparseGPT")
    results.append(("SparseGPT", loss, t, sp))
    print(f"  [SparseGPT] Loss={loss:.4e}, Time={t:.3f}s, Sp={sp:.1f}%")
    del W_sgpt

    # ── OBS full (frozen C) ──
    print(f"\n  [OBS full] Running...")
    W_p = torch.nn.Parameter(W0.clone())
    block = BlockSpec(W_p, block_shape=(1, 1))
    group = GroupSpec(block, group_shape=(1, 4))
    solver = StructuredOBS(group, H, damp=1e-4, C=C)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    solver.prune(num_nz=2, compensate="full")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_p.data, W0, H, N)
    sp = measure_sparsity(W_p.data)
    check_24(W_p.data, "OBS full")
    results.append(("OBS full", loss, t, sp))
    print(f"  [OBS full] Loss={loss:.4e}, Time={t:.1f}s, Sp={sp:.1f}%")
    del W_p, block, group, solver

    # ── True OBS element-by-element ──
    print(f"\n  [True OBS element] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_true_e = true_obs_element(W0, C)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_true_e, W0, H, N)
    sp = measure_sparsity(W_true_e)
    check_24(W_true_e, "True OBS element")
    results.append(("True OBS element", loss, t, sp))
    print(f"  [True OBS element] Loss={loss:.4e}, Time={t:.1f}s, Sp={sp:.1f}%")
    del W_true_e

    # ── True OBS with C(4,2) groups ──
    print(f"\n  [True OBS groups] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_true_g = true_obs_groups(W0, C)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_true_g, W0, H, N)
    sp = measure_sparsity(W_true_g)
    check_24(W_true_g, "True OBS groups")
    results.append(("True OBS groups", loss, t, sp))
    print(f"  [True OBS groups] Loss={loss:.4e}, Time={t:.1f}s, Sp={sp:.1f}%")
    del W_true_g

    # ── True OBS with 2 groups at a time ──
    print(f"\n  [True OBS groups×2] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_true_g2 = true_obs_groups2(W0, C)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_true_g2, W0, H, N)
    sp = measure_sparsity(W_true_g2)
    check_24(W_true_g2, "True OBS groups×2")
    results.append(("True OBS groups×2", loss, t, sp))
    print(f"  [True OBS groups×2] Loss={loss:.4e}, Time={t:.1f}s, Sp={sp:.1f}%")
    del W_true_g2

    # ── Report ──
    print(f"\n  {'Method':<25} {'Loss':>14} {'Norm.':>10} {'Sparsity':>10} {'Time':>10}")
    print(f"  {'-'*73}")
    for name, loss, t, sp in results:
        print(f"  {name:<25} {loss:>14.4e} {loss/ref*100:>8.4f}% {sp:>8.1f}% {t:>8.1f}s")

    def compare(a_name, a_loss, b_name, b_loss):
        if a_loss < b_loss:
            print(f"  {a_name} beats {b_name} by {(1 - a_loss/b_loss)*100:.2f}%")
        else:
            print(f"  {b_name} beats {a_name} by {(1 - b_loss/a_loss)*100:.2f}%")

    r = {name: loss for name, loss, _, _ in results}
    print(f"\n  --- Key comparisons ---")
    compare("True OBS groups", r["True OBS groups"], "SparseGPT", r["SparseGPT"])
    compare("True OBS groups×2", r["True OBS groups×2"], "SparseGPT", r["SparseGPT"])
    compare("True OBS element", r["True OBS element"], "SparseGPT", r["SparseGPT"])
    compare("True OBS groups×2", r["True OBS groups×2"], "True OBS groups", r["True OBS groups"])
    compare("OBS full", r["OBS full"], "SparseGPT", r["SparseGPT"])


if __name__ == "__main__":
    main()
