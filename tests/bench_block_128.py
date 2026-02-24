"""
Block config (16-col, 8-row coupled) on first 128 rows:
  OBS full vs True OBS vs SparseGPT.

128 rows = 8 chunks of 16 rows.
True OBS: (16, K, K) C per chunk = 5.7 GB — fits easily.
"""

import time
from itertools import combinations

import torch
import torch.linalg as LA

from sparsekit import BlockSpec, GroupSpec, StructuredOBS
from sparsekit.views import BlockView

DEVICE = torch.device("cuda:1")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"

CHUNK = 16
BLK = 16
PAIRS = CHUNK // 2
NROWS = 128


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


# ── True OBS block (greedy, per-row Schur) ────────────────────────────────

def true_obs_block_chunk(Wc, C_base):
    """Greedy block-OBS for one chunk: per-row C with rank-16 Schur."""
    R, K = Wc.shape
    device = Wc.device
    G = K // BLK
    pairs = R // 2

    C = C_base.unsqueeze(0).expand(R, -1, -1).clone()  # (R, K, K)
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


# ── SparseGPT block ──────────────────────────────────────────────────────

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
                w = W1[:, i]; d = Hinv1[i, i]
                q = w.clone(); q[mask1[:, i]] = 0.0; Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err1
            Wc[:, i1:i2] = Q1
            Wc[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return W


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("Loading data...", flush=True)
    W0_full = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()
    X = torch.load(X_PATH, map_location="cpu", weights_only=True)
    M, K = W0_full.shape
    N = X.shape[0]

    print("Computing H...", flush=True)
    H = compute_H(X)
    del X
    torch.cuda.synchronize(DEVICE)

    print("Computing C = H^{-1}...", flush=True)
    C = StructuredOBS.compute_inverse(H, damp=1e-4)
    torch.cuda.synchronize(DEVICE)

    W0 = W0_full[:NROWS]
    n_chunks = NROWS // CHUNK
    ref = compute_loss(W0, torch.zeros_like(W0), H, N)

    print(f"\nBlock config on first {NROWS} rows ({n_chunks} chunks of {CHUNK})")
    print(f"  block=16 cols, group=2 rows (8 apart), nnz=1, 50% sparsity")
    print(f"  Reference = {ref:.4e}", flush=True)

    results = []

    # SparseGPT block
    print(f"\n  [SparseGPT] Running...", flush=True)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_s = sparsegpt_block(W0, H)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_s, W0, H, N)
    results.append(("SparseGPT", loss, t))
    print(f"  [SparseGPT] Loss={loss:.4e}, Time={t:.1f}s", flush=True)
    del W_s

    # True OBS block (greedy + Schur)
    print(f"\n  [True OBS] Running...", flush=True)
    W_t = W0.clone()
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    for ci in range(n_chunks):
        c0 = ci * CHUNK
        print(f"    chunk {ci}/{n_chunks}", flush=True)
        W_t[c0:c0+CHUNK] = true_obs_block_chunk(W0[c0:c0+CHUNK], C)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_t, W0, H, N)
    results.append(("True OBS", loss, t))
    print(f"  [True OBS] Loss={loss:.4e}, Time={t:.1f}s", flush=True)
    del W_t

    # OBS full block (frozen C, via StructuredOBS)
    print(f"\n  [OBS full] Running...", flush=True)
    W_o = torch.nn.Parameter(W0.clone())
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    for ci in range(n_chunks):
        c0 = ci * CHUNK
        W_chunk = torch.nn.Parameter(W_o.data[c0:c0+CHUNK].clone())
        view = BlockView(W_chunk, size=(PAIRS, 2, K), stride=(K, PAIRS * K, 1))
        block = BlockSpec(view, block_shape=(1, 1, BLK))
        group = GroupSpec(block, group_shape=(1, 2, 1))
        solver = StructuredOBS(group, H, damp=1e-4, C=C)
        solver.prune(num_nz=1, compensate="full")
        W_o.data[c0:c0+CHUNK] = W_chunk.data
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_o.data, W0, H, N)
    results.append(("OBS full", loss, t))
    print(f"  [OBS full] Loss={loss:.4e}, Time={t:.1f}s", flush=True)
    del W_o

    # Report
    print(f"\n  {'Method':<20} {'Loss':>14} {'Norm.':>10} {'Time':>10}")
    print(f"  {'-'*56}")
    for name, loss, t in results:
        print(f"  {name:<20} {loss:>14.4e} {loss/ref*100:>8.4f}% {t:>8.1f}s")

    best_name, best_loss, _ = min(results, key=lambda x: x[1])
    print(f"\n  --- vs {best_name} ---")
    for name, loss, _ in results:
        if name == best_name:
            continue
        print(f"  {best_name} beats {name} by {(1 - best_loss/loss)*100:.2f}%")


if __name__ == "__main__":
    main()
