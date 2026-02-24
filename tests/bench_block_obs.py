"""
Block-structured OBS benchmark on real LLM data — FULL MATRIX.

W (2560, 9728) processed in chunks of 16 rows:
  BlockView:    size=(8, 2, K), stride=(K, 8K, 1)
  block_shape:  (1, 1, 16)  — each block = 16 contiguous columns in one row
  group_shape:  (1, 2, 1)   — couples block at row i with row i+8
  nnz:          1            — keep 1 of 2 blocks per group (50% sparsity)

Methods:
  1. OBS       — greedy block selection, full H^{-1} + rank-16 Schur complement
  2. SparseGPT — Cholesky(H^{-1}), one-shot selection, blockwise error prop
  3. Magnitude — prune smaller-norm block per group
"""

import time
import torch
import torch.linalg as LA

DEVICE = torch.device("cuda:1")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"
CHUNK = 16      # rows per chunk (must be even for pairing)
BLK = 16        # structural block size (columns)
PAIRS = CHUNK // 2


def load_data(device):
    print(f"  Loading W from {W_PATH}...")
    W = torch.load(W_PATH, map_location=device, weights_only=True).float()
    print(f"  W loaded: {W.shape} {W.dtype}")
    print(f"  Loading X from {X_PATH}...")
    X = torch.load(X_PATH, map_location="cpu", weights_only=True)
    print(f"  X loaded: {X.shape} {X.dtype}")
    return W, X


def compute_H(X, device, batch_size=4096):
    N, K = X.shape
    H = torch.zeros(K, K, device=device, dtype=torch.float32)
    for i in range(0, N, batch_size):
        X_b = X[i:i+batch_size].to(device=device, dtype=torch.float32)
        H.addmm_(X_b.T, X_b)
    H /= N
    return H


# ─── Magnitude ──────────────────────────────────────────────────────────

def magnitude_block(W0):
    """Prune the smaller-norm block per group, chunk by chunk."""
    M, K = W0.shape
    W = W0.clone()
    G = K // BLK
    pairs = CHUNK // 2

    for c0 in range(0, M, CHUNK):
        c1 = c0 + CHUNK
        Wc = W[c0:c1]
        W_blocked = Wc.view(pairs, 2, G, BLK)
        norms = W_blocked.norm(dim=3)
        prune_j = norms.argmin(dim=1)
        mask = torch.zeros(pairs, 2, G, dtype=torch.bool, device=W0.device)
        mask.scatter_(1, prune_j.unsqueeze(1), True)
        W_blocked[mask.unsqueeze(3).expand_as(W_blocked)] = 0.0

    return W


# ─── SparseGPT ─────────────────────────────────────────────────────────

def sparsegpt_block(W0, H, blocksize=128):
    """SparseGPT for block-structured group pruning, processes all rows."""
    M, K = W0.shape
    device = W0.device
    pairs = CHUNK // 2
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

    # Process each chunk of CHUNK rows
    for c0 in range(0, M, CHUNK):
        c1 = c0 + CHUNK
        Wc = W[c0:c1]  # (CHUNK, K) view

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

                    for p in range(pairs):
                        r0, r1 = p, p + pairs
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


# ─── Structured OBS (true greedy, batched) ─────────────────────────────

def obs_block_chunk(Wc, C_base):
    """Greedy block-OBS for one chunk of CHUNK rows, fused addmm_ Schur."""
    R, K = Wc.shape
    device = Wc.device
    G = K // BLK
    pairs = R // 2

    C = C_base.unsqueeze(0).expand(R, -1, -1).clone()  # (R, K, K) f32
    w = Wc.clone().float()

    done = torch.zeros(pairs, G, dtype=torch.bool, device=device)
    arange_G = torch.arange(G, device=device)
    eye16 = 1e-4 * torch.eye(BLK, device=device)

    for step in range(G):
        C_view = C.view(R, G, BLK, G, BLK)
        C_diag = C_view[:, arange_G, :, arange_G, :]
        C_diag_inv = LA.inv(
            C_diag.reshape(-1, BLK, BLK) + eye16
        ).reshape(R, G, BLK, BLK)

        W_blocked = w.view(R, G, BLK)
        temp = torch.einsum("rgb,rgbc->rgc", W_blocked, C_diag_inv)
        scores = (temp * W_blocked).sum(dim=2)

        scores_pair = scores.view(pairs, 2, G)
        scores_pair[done.unsqueeze(1).expand(pairs, 2, G)] = float("inf")

        flat = scores_pair.reshape(pairs, 2 * G)
        best_flat = flat.argmin(dim=1)
        best_j = best_flat // G
        best_g = best_flat % G
        best_r = torch.arange(pairs, device=device) * 2 + best_j

        for p in range(pairs):
            r, g = best_r[p].item(), best_g[p].item()
            cs = BLK * g

            C_bb_inv = LA.inv(C[r, cs:cs+BLK, cs:cs+BLK] + eye16)
            wb = w[r, cs:cs+BLK].clone()
            C_col = C[r, :, cs:cs+BLK].clone()

            w[r] -= C_col @ (C_bb_inv @ wb)
            w[r, cs:cs+BLK] = 0.0

            tmp = C_col @ C_bb_inv
            C[r].addmm_(tmp, C[r, cs:cs+BLK, :], alpha=-1.0)
            C[r, cs:cs+BLK, :] = 0.0
            C[r, :, cs:cs+BLK] = 0.0

            done[p, g] = True

    return w


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    W0, X_cpu = load_data(DEVICE)
    M, K = W0.shape
    N = X_cpu.shape[0]
    G = K // BLK
    n_chunks = M // CHUNK

    print(f"Full matrix: M={M}, K={K}, block={BLK}, groups={G}")
    print(f"Chunks: {n_chunks} x {CHUNK} rows, pairs={PAIRS}/chunk")
    print(f"Each group: 2 blocks of {BLK} elements, keep 1 -> 50% sparsity")

    print("Computing H (batched)...")
    t0 = time.time()
    H = compute_H(X_cpu, DEVICE)
    torch.cuda.synchronize(DEVICE)
    print(f"H computed in {time.time() - t0:.1f}s")
    del X_cpu

    def compute_loss(W_pruned):
        # Batched to avoid OOM: sum over chunks of rows
        total = 0.0
        for c0 in range(0, M, CHUNK):
            dW = W_pruned[c0:c0+CHUNK] - W0[c0:c0+CHUNK]
            total += ((dW @ H) * dW).sum().item()
        return total * N

    ref_norm = 0.0
    for c0 in range(0, M, CHUNK):
        ref_norm += ((W0[c0:c0+CHUNK] @ H) * W0[c0:c0+CHUNK]).sum().item()
    ref_norm *= N
    print(f"Reference ||X W0^T||_F^2 = {ref_norm:.2e}")

    results = []

    # ── 1. Magnitude ──
    print("\n[1/3] Magnitude...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_mag = magnitude_block(W0)
    torch.cuda.synchronize(DEVICE)
    t_mag = time.time() - t0
    loss_mag = compute_loss(W_mag)
    results.append(("Magnitude", loss_mag, t_mag))
    print(f"  Loss: {loss_mag:.4e}, Time: {t_mag:.3f}s")
    del W_mag

    # ── 2. SparseGPT ──
    print("\n[2/3] SparseGPT...")
    H_sgpt = H.clone()
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_sgpt = sparsegpt_block(W0, H_sgpt, blocksize=128)
    torch.cuda.synchronize(DEVICE)
    t_sgpt = time.time() - t0
    loss_sgpt = compute_loss(W_sgpt)
    results.append(("SparseGPT", loss_sgpt, t_sgpt))
    print(f"  Loss: {loss_sgpt:.4e}, Time: {t_sgpt:.3f}s")
    del W_sgpt, H_sgpt

    # ── 3. Structured OBS ──
    print("\n[3/3] Structured OBS...")
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

    W_obs = W0.clone()
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    for ci in range(n_chunks):
        c0 = ci * CHUNK
        c1 = c0 + CHUNK
        print(f"  chunk {ci+1}/{n_chunks} (rows {c0}-{c1-1})")
        W_obs[c0:c1] = obs_block_chunk(W0[c0:c1], C_base)
    torch.cuda.synchronize(DEVICE)
    t_obs = time.time() - t0
    loss_obs = compute_loss(W_obs)
    results.append(("Structured OBS", loss_obs, t_obs + t_inv))
    print(f"  Loss: {loss_obs:.4e}, Time: {t_obs + t_inv:.1f}s (inv: {t_inv:.1f}s + prune: {t_obs:.1f}s)")
    del W_obs, C_base

    # ── Report ──
    print(f"\n{'='*75}")
    print(f"{'Method':<20} {'Loss':>14} {'Norm. Loss':>12} {'Time':>10} {'vs Mag':>10}")
    print(f"{'-'*75}")
    for name, loss, t in results:
        ratio = loss / results[0][1] if results[0][1] > 0 else float("inf")
        nloss = loss / ref_norm * 100
        print(f"{name:<20} {loss:>14.4e} {nloss:>10.4f}% {t:>9.1f}s {ratio:>9.4f}x")

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


if __name__ == "__main__":
    main()
