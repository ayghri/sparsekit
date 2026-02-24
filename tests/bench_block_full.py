"""
OBS full vs SparseGPT: 2:4 and 16-contiguous-8-row-coupled.

Config A — 2:4 contiguous:
  block_shape=(1,1), group_shape=(1,4), nnz=2
  All 2560 rows processed together.

Config B — 16-col blocks, 8-row coupled:
  BlockView(size=(8,2,K), stride=(K,8K,1)) on 16-row chunks
  block_shape=(1,1,16), group_shape=(1,2,1), nnz=1
  160 chunks of 16 rows each.
"""

import time
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


def progress(msg):
    print(msg, flush=True)


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
            w = W1[:, i]; d = Hinv1[i, i]
            if i % 4 == 0:
                end = min(i + 4, count)
                if end - i == 4:
                    tmp = W1[:, i:end] ** 2 / (torch.diag(Hinv1)[i:end].reshape(1, -1)) ** 2
                    _, bot = tmp.topk(2, dim=1, largest=False)
                    mask1[:, i:end].scatter_(1, bot, True)
            q = w.clone(); q[mask1[:, i]] = 0.0; Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err1
        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return W


# ── SparseGPT block (16-col, 8-row coupled) ──────────────────────────────

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
        c1 = c0 + CHUNK
        Wc = W[c0:c1]
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

    # Precompute C
    progress("\nPrecomputing C = H^{-1}...")
    t0 = time.time()
    C = StructuredOBS.compute_inverse(H, damp=1e-4)
    torch.cuda.synchronize(DEVICE)
    progress(f"  C computed in {time.time() - t0:.1f}s")

    results = {}

    # ══════════════════════════════════════════════════════════════════════
    # Config A: 2:4 contiguous
    # ══════════════════════════════════════════════════════════════════════
    progress(f"\n{'='*70}")
    progress("Config A: 2:4 contiguous (block=1x1, group=1x4, nnz=2)")
    progress(f"{'='*70}")

    # SparseGPT 2:4
    progress("\n  [SparseGPT 2:4] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_s24 = sparsegpt_24(W0, H, blocksize=128)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_s24, W0, H, N)
    results["SparseGPT 2:4"] = (loss, t)
    progress(f"  [SparseGPT 2:4] Loss={loss:.4e}, Time={t:.1f}s")
    del W_s24

    # OBS full 2:4
    progress("\n  [OBS full 2:4] Running...")
    W_o24 = torch.nn.Parameter(W0.clone())
    block = BlockSpec(W_o24, block_shape=(1, 1))
    group = GroupSpec(block, group_shape=(1, 4))
    solver = StructuredOBS(group, H, damp=1e-4, C=C)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    solver.prune(num_nz=2, compensate="full")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_o24.data, W0, H, N)
    results["OBS full 2:4"] = (loss, t)
    progress(f"  [OBS full 2:4] Loss={loss:.4e}, Time={t:.1f}s")
    del W_o24, block, group, solver

    # ══════════════════════════════════════════════════════════════════════
    # Config B: 16-contiguous, 8-row coupled
    # ══════════════════════════════════════════════════════════════════════
    progress(f"\n{'='*70}")
    progress("Config B: 16-col blocks, 8-row coupled (block=1x1x16, group=1x2x1, nnz=1)")
    progress(f"  160 chunks of 16 rows, BlockView(8,2,K) stride=(K,8K,1)")
    progress(f"{'='*70}")

    n_chunks = M // CHUNK

    # SparseGPT block
    progress("\n  [SparseGPT block] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_sb = sparsegpt_block(W0, H, blocksize=128)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_sb, W0, H, N)
    results["SparseGPT block"] = (loss, t)
    progress(f"  [SparseGPT block] Loss={loss:.4e}, Time={t:.1f}s")
    del W_sb

    # True OBS block — greedy per-row C with Schur updates, chunk by chunk
    progress("\n  [True OBS block] Running...")
    W_tb = W0.clone()
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    for ci in range(n_chunks):
        c0 = ci * CHUNK
        c1 = c0 + CHUNK
        if ci % 20 == 0:
            progress(f"    chunk {ci}/{n_chunks} (rows {c0}-{c1-1})")
        W_tb[c0:c1] = true_obs_block_chunk(W0[c0:c1], C)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_tb, W0, H, N)
    results["True OBS block"] = (loss, t)
    progress(f"  [True OBS block] Loss={loss:.4e}, Time={t:.1f}s")
    del W_tb

    # OBS full block — process chunks via BlockView + StructuredOBS
    progress("\n  [OBS full block] Running...")
    W_ob = torch.nn.Parameter(W0.clone())
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    for ci in range(n_chunks):
        c0 = ci * CHUNK
        # Create a sub-parameter for this chunk
        W_chunk = torch.nn.Parameter(W_ob.data[c0:c0+CHUNK].clone())

        # BlockView: (8, 2, K) with stride (K, 8K, 1)
        view = BlockView(W_chunk, size=(PAIRS, 2, K), stride=(K, PAIRS * K, 1))
        block = BlockSpec(view, block_shape=(1, 1, BLK))
        group = GroupSpec(block, group_shape=(1, 2, 1))
        solver = StructuredOBS(group, H, damp=1e-4, C=C)
        solver.prune(num_nz=1, compensate="full")

        # Write back
        W_ob.data[c0:c0+CHUNK] = W_chunk.data
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_ob.data, W0, H, N)
    results["OBS full block"] = (loss, t)
    progress(f"  [OBS full block] Loss={loss:.4e}, Time={t:.1f}s")
    del W_ob

    # ══════════════════════════════════════════════════════════════════════
    # Report
    # ══════════════════════════════════════════════════════════════════════
    progress(f"\n{'='*70}")
    progress("Results")
    progress(f"{'='*70}")
    progress(f"\n  {'Method':<25} {'Loss':>14} {'Norm.':>10} {'Time':>10}")
    progress(f"  {'-'*61}")
    for name in ["SparseGPT 2:4", "OBS full 2:4", "SparseGPT block", "OBS full block"]:
        loss, t = results[name]
        progress(f"  {name:<25} {loss:>14.4e} {loss/ref*100:>8.4f}% {t:>8.1f}s")

    progress(f"\n  --- Config A: 2:4 ---")
    l_s, l_o = results["SparseGPT 2:4"][0], results["OBS full 2:4"][0]
    progress(f"  OBS full beats SparseGPT by {(1 - l_o/l_s)*100:.2f}%")

    progress(f"\n  --- Config B: block ---")
    l_s, l_o = results["SparseGPT block"][0], results["OBS full block"][0]
    if l_o < l_s:
        progress(f"  OBS full beats SparseGPT by {(1 - l_o/l_s)*100:.2f}%")
    else:
        progress(f"  SparseGPT beats OBS full by {(1 - l_s/l_o)*100:.2f}%")

    progress(f"\n  --- Cross-config ---")
    l_24 = results["OBS full 2:4"][0]
    l_blk = results["OBS full block"][0]
    progress(f"  OBS 2:4 vs OBS block: {l_24:.4e} vs {l_blk:.4e}")
    if l_blk < l_24:
        progress(f"  Block is {(1 - l_blk/l_24)*100:.2f}% better (less loss)")
    else:
        progress(f"  2:4 is {(1 - l_24/l_blk)*100:.2f}% better (less loss)")


if __name__ == "__main__":
    main()
