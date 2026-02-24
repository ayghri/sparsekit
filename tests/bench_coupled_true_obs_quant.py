"""
Coupled 2:4 pruning (True OBS ng=16) + MXFP4 quantization (OBS compensated).

Step 1: Coupled 2:4 pruning.
  Pairs: column i and column i+8 within each 16-col group.
  Two sub-groups of 4 pairs: {(0,8),(1,9),(2,10),(3,11)} and {(4,12),(5,13),(6,14),(7,15)}.
  Prune 2 of 4 pairs per sub-group -> 50% sparsity, C(4,2)=6 subsets.
  True OBS with per-row Schur updates, batched ng=16 groups.

Step 2: MXFP4 quantization of non-zero values.
  block_size=32 columns -> 16 non-zeros per block (after 2:4).
  OBS frozen-C compensation with mask to preserve sparsity, largest-first ordering.

W (2560, 9728), X (244449, 9728).
"""

import time
from itertools import combinations

import torch
import torch.linalg as LA

from sparsekit import StructuredOBS, mxfp4_quantize, quantize_obs
from sparsekit.pruners.quant import _quantize_block

DEVICE = torch.device("cuda:1")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"
NG = 16
QUANT_BLK = 32


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


def measure_sparsity(W):
    return (W.abs() < 1e-10).sum().item() / W.numel() * 100


def check_coupled_24(W, label):
    """Verify coupled 2:4: paired columns (i, i+8) within each 16-col group
    are both zero or both non-zero."""
    M, K = W.shape
    G = K // 16
    W16 = W.view(M, G, 16)
    for sub_a, sub_b in [(slice(0, 4), slice(8, 12)),
                         (slice(4, 8), slice(12, 16))]:
        za = (W16[:, :, sub_a].abs() < 1e-10)
        zb = (W16[:, :, sub_b].abs() < 1e-10)
        if (za != zb).any().item():
            progress(f"  WARNING: {label} coupling violated!")
            return False
        if (za.sum(dim=-1) != 2).any().item():
            progress(f"  WARNING: {label} not 2:4!")
            return False
    progress(f"  {label}: coupled 2:4 OK")
    return True


# ── SparseGPT coupled 2:4 ─────────────────────────────────────────────────

def sparsegpt_coupled_24(W0, H, blocksize=128):
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
            if col % 16 == 0 and i + 16 <= count:
                d_sq = torch.diag(Hinv1)[i:i+16].reshape(1, -1) ** 2
                col_scores = W1[:, i:i+16] ** 2 / d_sq

                ps0 = col_scores[:, :4] + col_scores[:, 8:12]
                _, bot0 = ps0.topk(2, dim=1, largest=False)
                pmask0 = torch.zeros(M, 4, dtype=torch.bool, device=device)
                pmask0.scatter_(1, bot0, True)
                mask1[:, i:i+4] |= pmask0
                mask1[:, i+8:i+12] |= pmask0

                ps1 = col_scores[:, 4:8] + col_scores[:, 12:16]
                _, bot1 = ps1.topk(2, dim=1, largest=False)
                pmask1 = torch.zeros(M, 4, dtype=torch.bool, device=device)
                pmask1.scatter_(1, bot1, True)
                mask1[:, i+4:i+8] |= pmask1
                mask1[:, i+12:i+16] |= pmask1

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


# ── True OBS coupled 2:4 ──────────────────────────────────────────────────

def true_obs_coupled_24(W0, C_base, ng=NG, row_chunk=16):
    """True OBS for coupled 2:4 with per-row Schur updates.

    Pairs: (i, i+8) within each 16-col group.
    Two sub-groups of 4 pairs each. Prune 2 of 4 pairs per sub-group.
    C(4,2)=6 subsets per sub-group.
    """
    M, K = W0.shape
    device = W0.device
    G = K // 16
    W = W0.clone().float()

    all_subs = torch.tensor(
        list(combinations(range(4), 2)), device=device, dtype=torch.long
    )  # (6, 2)

    n_chunks = (M + row_chunk - 1) // row_chunk

    for ci in range(n_chunks):
        r0 = ci * row_chunk
        r1 = min(r0 + row_chunk, M)
        R = r1 - r0
        if ci % 10 == 0:
            progress(f"    chunk {ci}/{n_chunks} (rows {r0}-{r1})")

        Wc = W[r0:r1]
        C = C_base.unsqueeze(0).expand(R, -1, -1).clone()  # (R, K, K)
        arange_R = torch.arange(R, device=device)

        for batch_start in range(0, G, ng):
            batch_end = min(batch_start + ng, G)

            block_pruned = []
            for g in range(batch_start, batch_end):
                base = g * 16

                for sub_off_a, sub_off_b in [(0, 8), (4, 12)]:
                    pair_cols = torch.stack([
                        torch.arange(base + sub_off_a, base + sub_off_a + 4, device=device),
                        torch.arange(base + sub_off_b, base + sub_off_b + 4, device=device),
                    ], dim=1)  # (4, 2)

                    best_cost = torch.full((R,), float("inf"), device=device)
                    best_si = torch.zeros(R, dtype=torch.long, device=device)

                    eye4 = 1e-8 * torch.eye(4, device=device)
                    for si in range(6):
                        prune_pairs = all_subs[si]
                        prune_col_idx = pair_cols[prune_pairs].view(-1)  # (4,)

                        C_PP = C[:, prune_col_idx][:, :, prune_col_idx]
                        C_PP_inv = LA.inv(C_PP + eye4)
                        W_P = Wc[:, prune_col_idx]
                        cost = (torch.bmm(W_P.unsqueeze(1), C_PP_inv).squeeze(1) * W_P).sum(1)

                        better = cost < best_cost
                        best_cost[better] = cost[better]
                        best_si[better] = si

                    pruned_pairs = all_subs[best_si]  # (R, 2)
                    pruned_cols = pair_cols[pruned_pairs.view(-1)].view(R, 2, 2).reshape(R, 4)
                    block_pruned.append(pruned_cols)

            all_p = torch.cat(block_pruned, dim=1)
            np_total = all_p.shape[1]

            eye_n = 1e-8 * torch.eye(np_total, device=device)
            pc_exp = all_p.unsqueeze(1).expand(R, K, np_total)
            C_col_P = C.gather(2, pc_exp)

            C_PP = C_col_P.gather(1, all_p.unsqueeze(2).expand(R, np_total, np_total))
            C_PP_inv = LA.inv(C_PP + eye_n)

            W_P = Wc.gather(1, all_p)
            comp = torch.bmm(C_col_P, C_PP_inv)
            delta = torch.bmm(comp, W_P.unsqueeze(2)).squeeze(2)
            Wc -= delta
            Wc.scatter_(1, all_p, torch.zeros(R, np_total, device=device))

            L_chol = LA.cholesky(C_PP_inv + eye_n)
            U = torch.bmm(C_col_P, L_chol)
            C.baddbmm_(U, U.transpose(1, 2), alpha=-1.0)

            for j in range(np_total):
                pc_j = all_p[:, j]
                C[arange_R, pc_j, :] = 0.0
                C[arange_R, :, pc_j] = 0.0

    return W


# ── Sparse MXFP4 quantize_fn ──────────────────────────────────────────────

def _sparse_quantize_block(W_P, nz_mask_block):
    """quantize_fn for quantize_obs that preserves zeros.

    Returns a closure over nz_mask_block that maps block index to mask.
    """
    Q = _quantize_block(W_P, W_P.device)
    Q[~nz_mask_block] = 0.0
    return Q


# ── Main ──────────────────────────────────────────────────────────────────

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

    progress("\nComputing C = H^{-1}...")
    t0 = time.time()
    C = StructuredOBS.compute_inverse(H, damp=1e-4)
    torch.cuda.synchronize(DEVICE)
    progress(f"  C computed in {time.time() - t0:.1f}s")

    results = []

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: Coupled 2:4 pruning
    # ═══════════════════════════════════════════════════════════════════════
    progress(f"\n{'='*70}")
    progress("Step 1: Coupled 2:4 pruning")
    progress(f"{'='*70}")

    # ── SparseGPT ──
    progress("\n  [SparseGPT coupled 2:4] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_sgpt = sparsegpt_coupled_24(W0, H)
    torch.cuda.synchronize(DEVICE)
    t_sgpt = time.time() - t0
    loss_sgpt = compute_loss(W_sgpt, W0, H, N)
    check_coupled_24(W_sgpt, "SparseGPT")
    progress(f"  Prune loss={loss_sgpt:.4e}, Time={t_sgpt:.1f}s, "
             f"Sp={measure_sparsity(W_sgpt):.1f}%")
    W_sgpt_cpu = W_sgpt.cpu(); del W_sgpt
    torch.cuda.empty_cache()

    # ── True OBS ──
    progress("\n  [True OBS coupled 2:4 ng=16] Running...")
    H_cpu = H.cpu(); del H; torch.cuda.empty_cache()
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_obs = true_obs_coupled_24(W0, C, ng=NG)
    torch.cuda.synchronize(DEVICE)
    t_obs = time.time() - t0
    H = H_cpu.to(DEVICE); del H_cpu
    loss_obs = compute_loss(W_obs, W0, H, N)
    check_coupled_24(W_obs, "True OBS")
    progress(f"  Prune loss={loss_obs:.4e}, Time={t_obs:.1f}s, "
             f"Sp={measure_sparsity(W_obs):.1f}%")
    vs = (1 - loss_obs / loss_sgpt) * 100
    progress(f"\n  True OBS vs SparseGPT (prune only): {vs:+.2f}%")
    W_obs_cpu = W_obs.cpu(); del W_obs
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════════
    # Step 2: MXFP4 quantization (block_size=32, 16 nnz/block)
    # ═══════════════════════════════════════════════════════════════════════
    progress(f"\n{'='*70}")
    progress(f"Step 2: MXFP4 quantization (block_size={QUANT_BLK})")
    progress(f"{'='*70}")

    def run_quant(W_pruned_cpu, prune_label, prune_time, quant_label, quant_fn):
        W_p = W_pruned_cpu.to(DEVICE)
        nz_mask = W_p.abs() > 1e-10
        torch.cuda.synchronize(DEVICE)
        t0 = time.time()
        W_q = quant_fn(W_p, nz_mask)
        torch.cuda.synchronize(DEVICE)
        t = prune_time + time.time() - t0
        loss = compute_loss(W_q, W0, H, N)
        sp = measure_sparsity(W_q)
        name = f"{prune_label} + {quant_label}"
        # Verify MXFP4: re-quantize and check
        Q_check = mxfp4_quantize(W_q, block_size=QUANT_BLK)
        diff = (W_q - Q_check).abs()
        viol = (diff[nz_mask] > 1e-6).sum().item()
        ok = "OK" if viol == 0 else f"{viol} violations!"
        progress(f"  [{name}] Loss={loss:.4e}, Norm={loss/ref*100:.4f}%, "
                 f"Sp={sp:.1f}%, Time={t:.1f}s, MXFP4={ok}")
        results.append((name, loss, t, sp))
        del W_p, W_q, Q_check
        torch.cuda.empty_cache()

    def naive_quant(W_p, nz_mask):
        Q = mxfp4_quantize(W_p, block_size=QUANT_BLK)
        Q[~nz_mask] = 0.0
        return Q

    def obs_quant(W_p, nz_mask):
        W_work = W_p.clone()
        # quantize_fn that preserves zeros
        block_cols = torch.arange(K, device=DEVICE).view(-1, QUANT_BLK)
        nz_blocks = {b: nz_mask[:, block_cols[b]] for b in range(K // QUANT_BLK)}

        def qfn(W_block):
            # Find which block this is by matching column values
            # quantize_obs passes W_work[:, cols] which is a view
            Q = _quantize_block(W_block, W_block.device)
            return Q

        quantize_obs(W_work, H, block_size=QUANT_BLK, damp=1e-4, C=C,
                     order="largest_first", mask=nz_mask)
        # Re-zero pruned positions (quant may have drifted them slightly)
        W_work[~nz_mask] = 0.0
        return W_work

    # All combos
    for W_cpu, plabel, ptime in [(W_sgpt_cpu, "SGPT", t_sgpt),
                                  (W_obs_cpu, "TrueOBS", t_obs)]:
        run_quant(W_cpu, plabel, ptime, "Naive", naive_quant)
        run_quant(W_cpu, plabel, ptime, "OBS", obs_quant)

    del W_sgpt_cpu, W_obs_cpu

    # ═══════════════════════════════════════════════════════════════════════
    # Report
    # ═══════════════════════════════════════════════════════════════════════
    progress(f"\n{'='*70}")
    progress("Summary")
    progress(f"{'='*70}")

    baseline_loss = results[0][1]  # SGPT + Naive
    progress(f"\n  {'Method':<30} {'Loss':>14} {'Norm.':>10} {'Sp':>6} "
             f"{'Time':>8} {'vs baseline':>12}")
    progress(f"  {'-'*84}")
    for name, loss, t, sp in results:
        vs = (1 - loss / baseline_loss) * 100
        progress(f"  {name:<30} {loss:>14.4e} {loss/ref*100:>8.4f}% {sp:>4.0f}% "
                 f"{t:>7.1f}s {vs:>+11.2f}%")

    best_name, best_loss, best_t, _ = min(results, key=lambda x: x[1])
    progress(f"\n  Best: {best_name} ({best_loss:.4e}, "
             f"{best_loss/ref*100:.4f}%, {best_t:.1f}s)")


if __name__ == "__main__":
    main()
