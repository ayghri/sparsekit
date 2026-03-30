"""
Test: Cholesky-sequential OBS vs Greedy-Schur OBS.

The Cholesky insight: if we eliminate columns LEFT-TO-RIGHT, the Schur
complement C_new = U_{22}^T U_{22} — just drop the first row of U.
No explicit Schur update needed.

For group elimination (16 cols at a time), dropping 16 rows of U gives
the exact updated C for the remaining columns.

Question: how much quality do we lose by fixing the column order to L->R
(enabling the Cholesky trick) vs fully greedy (best quality, needs Schur)?
"""

import time
import torch
import torch.linalg as LA

DEVICE = torch.device("cuda:1")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


def compute_H(X, device, batch_size=4096):
    N, K = X.shape
    H = torch.zeros(K, K, device=device, dtype=torch.float32)
    for i in range(0, N, batch_size):
        X_b = X[i:i+batch_size].to(device=device, dtype=torch.float32)
        H.addmm_(X_b.T, X_b)
    H /= N
    return H


# ─── Greedy Schur OBS (gold standard) ──────────────────────────────────

def obs_greedy_schur(w_row, C, group_size=4, n_prune=2):
    """Per-row greedy OBS with explicit Schur complement."""
    K = w_row.shape[0]
    C = C.clone()
    w = w_row.clone()
    num_scopes = K // group_size
    remaining = torch.full((num_scopes,), n_prune, device=w.device)
    pruned = torch.zeros(K, dtype=torch.bool, device=w.device)

    for step in range(n_prune * num_scopes):
        diag = C.diagonal()
        scores = w ** 2 / (diag.abs() + 1e-10)
        scores[pruned] = float("inf")
        group_done = (remaining <= 0).repeat_interleave(group_size)
        scores[group_done] = float("inf")

        p = scores.argmin().item()
        if scores[p] == float("inf"):
            break

        c_pp = C[p, p]
        c_p = C[:, p].clone()
        w -= (w[p] / c_pp) * c_p
        w[p] = 0.0
        C.addr_(c_p, c_p, alpha=(-1.0 / c_pp).item())
        C[p, :] = 0.0
        C[:, p] = 0.0
        pruned[p] = True
        remaining[p // group_size] -= 1

    return w


# ─── Sequential Cholesky OBS (fast) ────────────────────────────────────

def obs_sequential_chol(w_row, C, group_size=4, n_prune=2):
    """
    OBS with LEFT-TO-RIGHT block processing using Cholesky trick.

    Process blocks in order g=0, 1, 2, ..., G-1.
    For each block, decide which n_prune elements to prune (greedy within block).
    Use U = chol(C, upper=True). After processing a block of `group_size` cols,
    C_remaining = U[group_size:, group_size:]^T @ U[group_size:, group_size:]
    — just drop rows from U.

    The within-block decisions use the full H^{-1} info (current U).
    The across-block ordering is fixed (left-to-right).
    """
    K = w_row.shape[0]
    U = LA.cholesky(C, upper=True).clone()
    w = w_row.clone()
    num_scopes = K // group_size

    for g in range(num_scopes):
        gs = g * group_size
        ge = gs + group_size

        # Current C for remaining columns [gs:] is U[gs:, gs:]^T @ U[gs:, gs:]
        # We only need the group_size × group_size group for within-block scoring
        # and the group_size × (K-gs) group for compensation.
        U_active = U[gs:, gs:]  # (K-gs, K-gs) upper triangular

        # Extract the block group: first group_size columns of U_active
        # C_group = U_active^T @ U_active restricted to first group_size cols
        U_grp = U_active[:, :group_size]  # (K-gs, group_size)

        # C_gg[i,j] = (U_grp^T @ U_grp)[i,j] for i,j in [0, group_size)
        # But we need per-element OBS within this block.

        # For greedy within-block: iterate n_prune times
        # Maintain a local "active" mask within the block
        local_pruned = torch.zeros(group_size, dtype=torch.bool, device=w.device)

        for prune_step in range(n_prune):
            # Diagonal of current C for columns in this block
            # After previous eliminations within block, C has been updated
            # We track via a local V (small, group_size cols)

            # Actually, let's use the exact Cholesky structure.
            # The block columns are [gs:ge] in the original indexing,
            # which are [0:group_size] in U_active.
            # For within-block greedy, we need C[i,i] for i in block.
            # C = U_active^T @ U_active
            # C[i,i] = ||U_active[:, i]||^2

            # After pruning column i within block:
            # C_new = C - C[:,i] C[i,:]^T / C[i,i]
            # Which corresponds to removing column i from U_active.

            # For small group_size (4), just compute C_gg directly
            C_gg = U_grp.T @ U_grp  # (group_size, group_size) — tiny

            diag_gg = C_gg.diagonal()
            w_grp = w[gs:ge]
            scores = w_grp ** 2 / (diag_gg.abs() + 1e-10)
            scores[local_pruned] = float("inf")

            p_local = scores.argmin().item()
            p_global = gs + p_local

            c_pp = C_gg[p_local, p_local]

            # Full compensation: need C[:, p_global] for ALL remaining cols
            # C[:, p_global] for cols [gs:] = U_active^T @ U_active[:, p_local]
            u_p = U_active[:, p_local].clone()
            c_col = U_active.T @ u_p  # (K-gs,)

            # Compensate all remaining weights
            w[gs:] -= (w[p_global] / c_pp) * c_col
            w[p_global] = 0.0

            # Update U_active: rank-1 downdate
            # U_active_new such that U_active_new^T @ U_active_new = C - v v^T / c_pp
            # where v = U_active[:, p_local]
            # Use the V-projection: V = (I - u_p u_p^T / c_pp) @ U_active
            vtU = u_p @ U_active  # (K-gs,)
            U_active -= torch.outer(u_p, vtU) / c_pp
            U_active[:, p_local] = 0.0
            U_grp = U_active[:, :group_size]

            local_pruned[p_local] = True

        # After processing this block, advance U to drop the block columns.
        # The remaining C for columns [ge:] is:
        # U_active[group_size:, group_size:]^T @ U_active[group_size:, group_size:]
        # But U_active has been modified by the downdates. We need to
        # write back the modified U_active into U.
        U[gs:, gs:] = U_active
        # The "advance" happens naturally in the next iteration: gs += group_size

    return w


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    print("=== Correctness test (K=32) ===")
    torch.manual_seed(42)
    K, N = 32, 128
    X = torch.randn(N, K, device=DEVICE)
    w_row = torch.randn(K, device=DEVICE)
    H = X.T @ X / N
    damp = 1e-4 * torch.mean(torch.diag(H))
    H[torch.arange(K, device=DEVICE), torch.arange(K, device=DEVICE)] += damp
    C = LA.inv(H)

    w_greedy = obs_greedy_schur(w_row, C)
    w_seq = obs_sequential_chol(w_row, C)

    loss_greedy = ((w_row - w_greedy) @ H @ (w_row - w_greedy)).item()
    loss_seq = ((w_row - w_seq) @ H @ (w_row - w_seq)).item()
    print(f"Greedy loss: {loss_greedy:.6f}")
    print(f"Sequential loss: {loss_seq:.6f}")
    print(f"Sequential / Greedy: {loss_seq / loss_greedy:.4f}x")
    print(f"Greedy advantage: {(1 - loss_greedy / loss_seq) * 100:.2f}%")

    # Profile at multiple scales
    for K in [256, 1024, 4096]:
        print(f"\n=== Profile K={K} ===")
        torch.manual_seed(42)
        N = 2 * K
        X = torch.randn(N, K, device=DEVICE)
        w_row = torch.randn(K, device=DEVICE)
        H = X.T @ X / N
        damp = 1e-4 * torch.mean(torch.diag(H))
        H[torch.arange(K, device=DEVICE), torch.arange(K, device=DEVICE)] += damp
        C = LA.inv(H)

        torch.cuda.synchronize(DEVICE)
        t0 = time.time()
        w_greedy = obs_greedy_schur(w_row, C)
        torch.cuda.synchronize(DEVICE)
        t_greedy = time.time() - t0

        torch.cuda.synchronize(DEVICE)
        t0 = time.time()
        w_seq = obs_sequential_chol(w_row, C)
        torch.cuda.synchronize(DEVICE)
        t_seq = time.time() - t0

        loss_greedy = ((w_row - w_greedy) @ H @ (w_row - w_greedy)).item()
        loss_seq = ((w_row - w_seq) @ H @ (w_row - w_seq)).item()

        print(f"Greedy:     {t_greedy:.3f}s, loss={loss_greedy:.4f}")
        print(f"Sequential: {t_seq:.3f}s, loss={loss_seq:.4f}")
        print(f"Speedup: {t_greedy / t_seq:.2f}x")
        print(f"Quality gap: {(loss_seq / loss_greedy - 1) * 100:.2f}% (seq is worse)")

    # Real LLM data
    print("\n=== Real LLM data (single row, K=9728) ===")
    W0 = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
    H = compute_H(X_cpu, DEVICE)
    del X_cpu
    K = W0.shape[1]
    damp = 1e-4 * torch.mean(torch.diag(H))
    H[torch.arange(K, device=DEVICE), torch.arange(K, device=DEVICE)] += damp
    C = LA.inv(H)
    w_row = W0[0]

    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    w_greedy = obs_greedy_schur(w_row, C)
    torch.cuda.synchronize(DEVICE)
    t_greedy = time.time() - t0

    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    w_seq = obs_sequential_chol(w_row, C)
    torch.cuda.synchronize(DEVICE)
    t_seq = time.time() - t0

    loss_greedy = ((w_row - w_greedy) @ H @ (w_row - w_greedy)).item()
    loss_seq = ((w_row - w_seq) @ H @ (w_row - w_seq)).item()

    print(f"Greedy:     {t_greedy:.3f}s, loss={loss_greedy:.6e}")
    print(f"Sequential: {t_seq:.3f}s, loss={loss_seq:.6e}")
    print(f"Speedup: {t_greedy / t_seq:.2f}x")
    print(f"Quality gap: {(loss_seq / loss_greedy - 1) * 100:.2f}%")


if __name__ == "__main__":
    main()
