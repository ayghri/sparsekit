"""
Large-scale 2:4 structured sparsity benchmark on real LLM data.

Loads layer_0 W (2560, 9728) and X (244449, 9728), prunes with:
  1. Structured OBS  — greedy per-element, full H^{-1} + Schur complement
  2. SparseGPT       — Cholesky(H^{-1}), one-shot N:M selection, blockwise error prop
  3. Magnitude       — prune smallest-magnitude pairs per group of 4

Reports reconstruction loss = ||X @ (W_pruned - W0)^T||_F^2 and wall-clock time.
"""

import time
import torch
import torch.linalg as LA

DEVICE = torch.device("cuda:1")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"
NUM_ROWS = 32  # rows to benchmark


def load_data(device):
    print(f"  Loading W from {W_PATH}...")
    W = torch.load(W_PATH, map_location=device, weights_only=True).float()
    print(f"  W loaded: {W.shape} {W.dtype} on {W.device}")
    print(f"  Loading X from {X_PATH}...")
    X = torch.load(X_PATH, map_location="cpu", weights_only=True)
    print(f"  X loaded: {X.shape} {X.dtype} on cpu")
    return W, X


def compute_H(X, device, batch_size=4096):
    """Compute H = X^T X / N by streaming batches of X to GPU."""
    N, K = X.shape
    H = torch.zeros(K, K, device=device, dtype=torch.float32)
    for i in range(0, N, batch_size):
        X_b = X[i:i+batch_size].to(device=device, dtype=torch.float32)
        H.addmm_(X_b.T, X_b)
    H /= N
    return H


# ─── Magnitude 2:4 ─────────────────────────────────────────────────────

def magnitude_24(W0):
    """Prune 2 of every 4 consecutive columns per row by magnitude."""
    W = W0.clone()
    M, K = W.shape
    assert K % 4 == 0
    Wg = W.view(M, K // 4, 4)
    _, idx = Wg.abs().topk(2, dim=2, largest=False)
    mask = torch.zeros_like(Wg, dtype=torch.bool)
    mask.scatter_(2, idx, True)
    Wg[mask] = 0.0
    return W


# ─── SparseGPT 2:4 ─────────────────────────────────────────────────────

def sparsegpt_24(W0, H, blocksize=128):
    """
    SparseGPT fasterprune for 2:4 structured sparsity.
    Direct adaptation of the reference implementation.
    """
    M, K = W0.shape
    device = W0.device
    W = W0.clone().float()

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


# ─── Structured OBS 2:4 (batched baddbmm_) ──────────────────────────────

def obs_24(W0, C_base):
    """
    True greedy OBS for 2:4 sparsity, batched over rows via baddbmm_.

    Each row maintains its own C = H^{-1}. Per step:
      1. Score all candidates: w[k]^2 / C[k,k]
      2. Per-row argmin picks cheapest column
      3. Compensate + fused Schur via baddbmm_ (one kernel for all B rows)
    """
    B, K = W0.shape
    device = W0.device
    num_groups = K // 4
    total_prunes = 2 * num_groups

    C = C_base.unsqueeze(0).expand(B, -1, -1).clone()  # (B, K, K) f32
    w = W0.clone().float()

    remaining = torch.full((B, num_groups), 2, dtype=torch.long, device=device)
    pruned = torch.zeros(B, K, dtype=torch.bool, device=device)
    arange_B = torch.arange(B, device=device)

    for step in range(total_prunes):
        if step % 500 == 0:
            print(f"    step {step}/{total_prunes}")

        # Score all (B, K)
        diag_C = torch.diagonal(C, dim1=1, dim2=2)
        scores = w ** 2 / (diag_C.abs() + 1e-10)
        scores[pruned] = float("inf")
        group_done = (remaining <= 0).repeat_interleave(4, dim=1)
        scores[group_done] = float("inf")

        # Per-row argmin
        p = scores.argmin(dim=1)  # (B,)

        # Gather C[:, p] and C[p, :] for each row
        p_col = p.view(B, 1, 1).expand(B, K, 1)
        c_col = C.gather(2, p_col).squeeze(2)  # (B, K)
        p_row = p.view(B, 1, 1).expand(B, 1, K)
        c_row = C.gather(1, p_row).squeeze(1)  # (B, K)

        c_pp = C[arange_B, p, p]  # (B,)
        safe_cpp = c_pp.abs() + 1e-12

        # Weight compensation: w -= (w[p] / c_pp) * C[:, p]
        wp = w[arange_B, p]
        w -= (wp / safe_cpp).unsqueeze(1) * c_col
        w[arange_B, p] = 0.0

        # Fused Schur: C -= outer(c_col, c_row) / c_pp via baddbmm_
        inv_sqrt = (1.0 / safe_cpp.sqrt()).unsqueeze(1)
        C.baddbmm_(
            (c_col * inv_sqrt).unsqueeze(2),  # (B, K, 1)
            (c_row * inv_sqrt).unsqueeze(1),  # (B, 1, K)
            alpha=-1.0,
        )

        # Zero out pruned row/col
        C[arange_B, p, :] = 0.0
        C[arange_B, :, p] = 0.0

        pruned[arange_B, p] = True
        remaining[arange_B, p // 4] -= 1

    return w


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    W0_full, X_cpu = load_data(DEVICE)
    M_full, K = W0_full.shape
    N = X_cpu.shape[0]

    # Slice to NUM_ROWS for benchmarking
    W0 = W0_full[:NUM_ROWS]
    M = NUM_ROWS
    print(f"Benchmarking {M} of {M_full} rows, K={K}")
    print(f"Groups of 4 -> {K // 4} groups, prune 2/4 -> {K // 2} zeros per row")

    print("Computing H (batched)...")
    t0 = time.time()
    H = compute_H(X_cpu, DEVICE)
    torch.cuda.synchronize(DEVICE)
    t_H = time.time() - t0
    print(f"H computed in {t_H:.1f}s, shape {H.shape}")
    del X_cpu

    # Loss via trace: ||X(W_p - W0)^T||_F^2 = N * tr(dW H dW^T)
    def compute_loss(W_pruned):
        dW = W_pruned - W0
        return ((dW @ H) * dW).sum().item() * N

    ref_norm = ((W0 @ H) * W0).sum().item() * N
    print(f"Reference ||X W0^T||_F^2 = {ref_norm:.2e}")

    results = []

    # ── 1. Magnitude 2:4 ──
    print("\n[1/3] Magnitude 2:4...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_mag = magnitude_24(W0)
    torch.cuda.synchronize(DEVICE)
    t_mag = time.time() - t0
    loss_mag = compute_loss(W_mag)
    results.append(("Magnitude", loss_mag, t_mag))
    print(f"  Loss: {loss_mag:.4e}, Time: {t_mag:.3f}s")
    del W_mag

    # ── 2. SparseGPT 2:4 ──
    print("\n[2/3] SparseGPT 2:4...")
    H_sgpt = H.clone()
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_sgpt = sparsegpt_24(W0, H_sgpt, blocksize=128)
    torch.cuda.synchronize(DEVICE)
    t_sgpt = time.time() - t0
    loss_sgpt = compute_loss(W_sgpt)
    results.append(("SparseGPT", loss_sgpt, t_sgpt))
    print(f"  Loss: {loss_sgpt:.4e}, Time: {t_sgpt:.3f}s")
    del W_sgpt, H_sgpt

    # ── 3. Structured OBS 2:4 ──
    print("\n[3/3] Structured OBS 2:4...")
    print("  Computing C = H^{-1}...")
    t0 = time.time()
    damp = 1e-4 * torch.mean(torch.diag(H))
    diag_idx = torch.arange(K, device=DEVICE)
    H_reg = H.clone()
    H_reg[diag_idx, diag_idx] += damp
    C_base = LA.inv(H_reg)
    torch.cuda.synchronize(DEVICE)
    t_inv = time.time() - t0
    print(f"  C computed in {t_inv:.1f}s")

    print(f"  Greedy OBS: {M} rows batched, {K//2} prunes/row...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_obs = obs_24(W0, C_base)
    torch.cuda.synchronize(DEVICE)
    t_obs = time.time() - t0
    loss_obs = compute_loss(W_obs)
    results.append(("Structured OBS", loss_obs, t_obs + t_inv))
    print(f"  Loss: {loss_obs:.4e}, Time: {t_obs + t_inv:.2f}s (inv: {t_inv:.1f}s + prune: {t_obs:.1f}s)")
    del W_obs, C_base

    # ── Report ──
    print(f"\n{'='*65}")
    print(f"{'Method':<20} {'Loss':>14} {'Time':>10} {'vs Mag':>10}")
    print(f"{'-'*65}")
    for name, loss, t in results:
        ratio = loss / results[0][1] if results[0][1] > 0 else float("inf")
        print(f"{name:<20} {loss:>14.4e} {t:>9.3f}s {ratio:>9.4f}x")

    print(f"\n--- Analysis ---")
    loss_obs = results[2][1]
    loss_sgpt = results[1][1]
    loss_mag = results[0][1]

    if loss_obs <= loss_sgpt:
        pct = (1 - loss_obs / loss_sgpt) * 100
        print(f"OBS beats SparseGPT by {pct:.2f}%")
    else:
        pct = (loss_obs / loss_sgpt - 1) * 100
        print(f"SparseGPT beats OBS by {pct:.2f}%")

    if loss_obs <= loss_mag:
        pct = (1 - loss_obs / loss_mag) * 100
        print(f"OBS beats Magnitude by {pct:.2f}%")

    if loss_sgpt <= loss_mag:
        pct = (1 - loss_sgpt / loss_mag) * 100
        print(f"SparseGPT beats Magnitude by {pct:.2f}%")

    print(f"\n--- Normalized loss (degradation / reference) ---")
    for name, loss, t in results:
        print(f"  {name:<20} {loss / ref_norm * 100:.4f}%")


if __name__ == "__main__":
    main()
