"""Profile optimized OBS variants."""

import time
import torch
import torch.linalg as LA

DEVICE = torch.device("cuda:1")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"
NUM_ROWS = 16
BLK = 16
PAIRS = 8


def compute_H(X, device, batch_size=4096):
    N, K = X.shape
    H = torch.zeros(K, K, device=device, dtype=torch.float32)
    for i in range(0, N, batch_size):
        X_b = X[i:i+batch_size].to(device=device, dtype=torch.float32)
        H.addmm_(X_b.T, X_b)
    H /= N
    return H


def setup():
    W0 = torch.load(W_PATH, map_location=DEVICE, weights_only=True).float()[:NUM_ROWS]
    X_cpu = torch.load(X_PATH, map_location="cpu", weights_only=True)
    H = compute_H(X_cpu, DEVICE)
    del X_cpu
    R, K = W0.shape
    damp = 1e-4 * torch.mean(torch.diag(H))
    diag_idx = torch.arange(K, device=DEVICE)
    H[diag_idx, diag_idx] += damp
    C_base = LA.inv(H)
    return W0, H, C_base


def bench_original_f64(W0, C_base, n_steps=50):
    """Original: f64, sequential pairs, separate mm + sub."""
    R, K = W0.shape
    G = K // BLK
    C = C_base.double().unsqueeze(0).expand(R, -1, -1).clone()
    w = W0.clone().double()
    done = torch.zeros(PAIRS, G, dtype=torch.bool, device=DEVICE)
    arange_G = torch.arange(G, device=DEVICE)
    eye16 = 1e-6 * torch.eye(BLK, device=DEVICE, dtype=torch.float64)

    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    for step in range(n_steps):
        C_view = C.view(R, G, BLK, G, BLK)
        C_diag = C_view[:, arange_G, :, arange_G, :]
        C_diag_inv = LA.inv(C_diag.reshape(-1, BLK, BLK) + eye16).reshape(R, G, BLK, BLK)
        W_blocked = w.view(R, G, BLK)
        temp = torch.einsum("rgb,rgbc->rgc", W_blocked, C_diag_inv)
        scores = (temp * W_blocked).sum(dim=2)
        scores_pair = scores.view(PAIRS, 2, G)
        scores_pair[done.unsqueeze(1).expand(PAIRS, 2, G)] = float("inf")
        flat = scores_pair.reshape(PAIRS, 2 * G)
        best_flat = flat.argmin(dim=1)
        best_r = torch.arange(PAIRS, device=DEVICE) * 2 + best_flat // G
        best_g = best_flat % G

        for p in range(PAIRS):
            r, g = best_r[p].item(), best_g[p].item()
            cs, ce = BLK * g, BLK * g + BLK
            C_bb_inv = LA.inv(C[r, cs:ce, cs:ce].clone() + eye16)
            wb = w[r, cs:ce].clone()
            C_col = C[r, :, cs:ce].clone()
            w[r] -= C_col @ (C_bb_inv @ wb)
            w[r, cs:ce] = 0.0
            tmp = C_col @ C_bb_inv
            C[r] -= tmp @ C[r, cs:ce, :].clone()
            C[r, cs:ce, :] = 0.0
            C[r, :, cs:ce] = 0.0
            done[p, g] = True

    torch.cuda.synchronize(DEVICE)
    return time.time() - t0


def bench_f32_batched(W0, C_base, n_steps=50):
    """Optimized: f32, batched pairs via gather/scatter + baddbmm."""
    R, K = W0.shape
    G = K // BLK
    C = C_base.unsqueeze(0).expand(R, -1, -1).clone()  # f32
    w = W0.clone().float()
    done = torch.zeros(PAIRS, G, dtype=torch.bool, device=DEVICE)
    arange_G = torch.arange(G, device=DEVICE)
    eye16 = 1e-6 * torch.eye(BLK, device=DEVICE)

    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    for step in range(n_steps):
        # Score
        C_view = C.view(R, G, BLK, G, BLK)
        C_diag = C_view[:, arange_G, :, arange_G, :]
        C_diag_inv = LA.inv(C_diag.reshape(-1, BLK, BLK) + eye16).reshape(R, G, BLK, BLK)
        W_blocked = w.view(R, G, BLK)
        temp = torch.einsum("rgb,rgbc->rgc", W_blocked, C_diag_inv)
        scores = (temp * W_blocked).sum(dim=2)
        scores_pair = scores.view(PAIRS, 2, G)
        scores_pair[done.unsqueeze(1).expand(PAIRS, 2, G)] = float("inf")
        flat = scores_pair.reshape(PAIRS, 2 * G)
        best_flat = flat.argmin(dim=1)
        best_j = best_flat // G
        best_g = best_flat % G
        best_r = torch.arange(PAIRS, device=DEVICE) * 2 + best_j  # (PAIRS,)

        # Batched update: gather columns for all 8 pairs at once
        # C[best_r[p], :, cs:ce] for each p where cs=16*best_g[p]
        # Use a loop to gather (8 pairs is tiny)
        C_cols = torch.zeros(PAIRS, K, BLK, device=DEVICE)
        C_rows = torch.zeros(PAIRS, BLK, K, device=DEVICE)
        C_bb_invs = torch.zeros(PAIRS, BLK, BLK, device=DEVICE)
        wbs = torch.zeros(PAIRS, BLK, device=DEVICE)

        for p in range(PAIRS):
            r, g = best_r[p].item(), best_g[p].item()
            cs = BLK * g
            C_cols[p] = C[r, :, cs:cs+BLK]
            C_rows[p] = C[r, cs:cs+BLK, :]
            C_bb_invs[p] = LA.inv(C[r, cs:cs+BLK, cs:cs+BLK] + eye16)
            wbs[p] = w[r, cs:cs+BLK]

        # Batched weight compensation
        # delta[p] = C_cols[p] @ C_bb_invs[p] @ wbs[p]
        deltas = torch.bmm(C_cols, torch.bmm(C_bb_invs, wbs.unsqueeze(2))).squeeze(2)  # (P, K)
        for p in range(PAIRS):
            r, g = best_r[p].item(), best_g[p].item()
            w[r] -= deltas[p]
            w[r, BLK*g:BLK*g+BLK] = 0.0

        # Batched Schur: C[r] -= (C_cols @ C_bb_inv) @ C_rows
        tmp = torch.bmm(C_cols, C_bb_invs)  # (P, K, BLK)
        schur = torch.bmm(tmp, C_rows)       # (P, K, K)
        for p in range(PAIRS):
            r, g = best_r[p].item(), best_g[p].item()
            cs = BLK * g
            C[r] -= schur[p]
            C[r, cs:cs+BLK, :] = 0.0
            C[r, :, cs:cs+BLK] = 0.0
            done[p, g] = True

    torch.cuda.synchronize(DEVICE)
    return time.time() - t0


def bench_f32_fused(W0, C_base, n_steps=50):
    """Optimized: f32, loop over pairs but use addmm_ to avoid alloc."""
    R, K = W0.shape
    G = K // BLK
    C = C_base.unsqueeze(0).expand(R, -1, -1).clone()
    w = W0.clone().float()
    done = torch.zeros(PAIRS, G, dtype=torch.bool, device=DEVICE)
    arange_G = torch.arange(G, device=DEVICE)
    eye16 = 1e-6 * torch.eye(BLK, device=DEVICE)

    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    for step in range(n_steps):
        C_view = C.view(R, G, BLK, G, BLK)
        C_diag = C_view[:, arange_G, :, arange_G, :]
        C_diag_inv = LA.inv(C_diag.reshape(-1, BLK, BLK) + eye16).reshape(R, G, BLK, BLK)
        W_blocked = w.view(R, G, BLK)
        temp = torch.einsum("rgb,rgbc->rgc", W_blocked, C_diag_inv)
        scores = (temp * W_blocked).sum(dim=2)
        scores_pair = scores.view(PAIRS, 2, G)
        scores_pair[done.unsqueeze(1).expand(PAIRS, 2, G)] = float("inf")
        flat = scores_pair.reshape(PAIRS, 2 * G)
        best_flat = flat.argmin(dim=1)
        best_r = torch.arange(PAIRS, device=DEVICE) * 2 + best_flat // G
        best_g = best_flat % G

        for p in range(PAIRS):
            r, g = best_r[p].item(), best_g[p].item()
            cs = BLK * g

            C_bb_inv = LA.inv(C[r, cs:cs+BLK, cs:cs+BLK] + eye16)
            wb = w[r, cs:cs+BLK].clone()
            C_col = C[r, :, cs:cs+BLK].clone()  # (K, BLK)

            # Weight comp
            w[r] -= C_col @ (C_bb_inv @ wb)
            w[r, cs:cs+BLK] = 0.0

            # Fused Schur: C[r] -= (C_col @ C_bb_inv) @ C[r, cs:ce, :]
            tmp = C_col @ C_bb_inv  # (K, BLK)
            C_row = C[r, cs:cs+BLK, :]  # (BLK, K) — contiguous slice
            # In-place: C[r].addmm_(tmp, C_row, alpha=-1.0)
            C[r].addmm_(tmp, C_row, alpha=-1.0)
            C[r, cs:cs+BLK, :] = 0.0
            C[r, :, cs:cs+BLK] = 0.0
            done[p, g] = True

    torch.cuda.synchronize(DEVICE)
    return time.time() - t0


def main():
    print("Loading data...")
    W0, H, C_base = setup()
    R, K = W0.shape
    G = K // BLK
    n_steps = 50
    print(f"R={R}, K={K}, G={G}, profiling {n_steps} steps\n")

    t1 = bench_original_f64(W0, C_base, n_steps)
    print(f"Original (f64, sequential):  {t1:.3f}s  ({t1/n_steps*1000:.1f}ms/step, est full: {t1/n_steps*G:.1f}s)")

    t2 = bench_f32_batched(W0, C_base, n_steps)
    print(f"f32 + batched bmm:           {t2:.3f}s  ({t2/n_steps*1000:.1f}ms/step, est full: {t2/n_steps*G:.1f}s)")

    t3 = bench_f32_fused(W0, C_base, n_steps)
    print(f"f32 + fused addmm_:          {t3:.3f}s  ({t3/n_steps*1000:.1f}ms/step, est full: {t3/n_steps*G:.1f}s)")

    print(f"\nSpeedup vs original: batched={t1/t2:.2f}x, fused={t1/t3:.2f}x")


if __name__ == "__main__":
    main()
