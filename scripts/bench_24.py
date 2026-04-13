"""
Exact greedy OBS for 2:4 structured sparsity — row-by-row.

Each row independently maintains its own C = (H + damp·I)⁻¹.
At each step:
  1. Score ALL remaining groups of 4 (closed-form 2×2 OBS cost)
  2. Pick the group with HIGHEST loss increase
  3. Prune best 2-of-4, compensate all active columns
  4. Rank-2 Schur update to C, zero pruned rows/cols
  Repeat until all groups are pruned.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_24.py [--rows N] [--chunk 16]
"""

import argparse
import time
from itertools import combinations

import torch
import torch.linalg as LA
from tqdm import tqdm

from sparsekit import StructuredOBS
from sparsekit.pruners import compute_hessian, output_error
from sparsekit.pruners.obd import magnitude
from sparsekit.pruners.sparsegpt import sparsegpt

DEVICE = torch.device("cuda")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


def progress(msg):
    print(msg, flush=True)


def measure_sparsity(W):
    return (W.abs() < 1e-10).sum().item() / W.numel() * 100


def check_24(W):
    M, K = W.shape
    zeros = (W.abs() < 1e-10).view(M, K // 4, 4)
    bad = (zeros.sum(-1) != 2).sum().item()
    if bad:
        progress(f"  WARNING: {bad} scopes violating 2:4!")
    return bad == 0


# ── Exact greedy OBS ───────────────────────────────────────────────────


@torch.no_grad()
def exact_greedy_24(
    W0,
    C_base,
    chunk_size=16,
    c_dtype=torch.float16,
    order="highest",
):
    """
    Row-by-row exact greedy OBS for 2:4 structured sparsity.

    Each row gets its own C = (H + damp·I)⁻¹, updated via rank-2 Schur
    complement after every pruning step.

    Args:
        W0:         (M, K) original weights.
        C_base:     (K, K) shared damped inverse Hessian.
        chunk_size: rows processed in parallel (each gets own C copy).
        c_dtype:    dtype for per-row C (fp16 halves memory, uses TC).
        order:      "highest" prunes largest-cost group first,
                    "lowest" prunes smallest-cost group first.
    """
    M, K = W0.shape
    device = W0.device
    n_groups = K // 4
    assert K % 4 == 0

    W = W0.clone().float()

    # Column indices per group: (n_groups, 4)
    group_cols = torch.arange(K, device=device).reshape(n_groups, 4)

    # All C(4,2)=6 pruning subsets (which 2 of 4 to zero)
    subs = torch.tensor(
        list(combinations(range(4), 2)), device=device, dtype=torch.long
    )  # (6, 2)

    # Precompute flat index arrays for vectorized scoring
    p0_all = group_cols[:, subs[:, 0]]  # (n_groups, 6)
    p1_all = group_cols[:, subs[:, 1]]
    p0_flat = p0_all.reshape(-1)  # (n_groups*6,)
    p1_flat = p1_all.reshape(-1)

    n_chunks = (M + chunk_size - 1) // chunk_size
    eye2 = 1e-8 * torch.eye(2, device=device)

    for ci in tqdm(range(n_chunks), desc="Row chunks"):
        r0 = ci * chunk_size
        r1 = min(r0 + chunk_size, M)
        B = r1 - r0

        w = W[r0:r1]  # (B, K) fp32
        C = (
            C_base.to(c_dtype).unsqueeze(0).expand(B, -1, -1).clone()
        )  # (B, K, K)

        pruned = torch.zeros(
            B, n_groups, dtype=torch.bool, device=device
        )
        arange_B = torch.arange(B, device=device)

        for _it in range(n_groups):
            # ── 1. Score all unpruned groups (closed-form 2×2) ───
            a = C[:, p0_flat, p0_flat].reshape(B, n_groups, 6).float()
            bv = C[:, p0_flat, p1_flat].reshape(B, n_groups, 6).float()
            d = C[:, p1_flat, p1_flat].reshape(B, n_groups, 6).float()
            det = (a * d - bv * bv).clamp(min=1e-12)

            w0 = w[:, p0_flat].reshape(B, n_groups, 6)
            w1 = w[:, p1_flat].reshape(B, n_groups, 6)

            # cost = w_P^T C[P,P]^{-1} w_P
            cost = (
                d * w0**2 - 2 * bv * w0 * w1 + a * w1**2
            ) / det  # (B, n_groups, 6)

            # Best (min-cost) subset per group
            best_cost, best_sub = cost.min(dim=2)  # (B, n_groups)

            # ── 2. Pick group to prune ───────────────────────────
            if order == "highest":
                best_cost[pruned] = -float("inf")
                best_g = best_cost.argmax(dim=1)  # (B,)
            else:
                best_cost[pruned] = float("inf")
                best_g = best_cost.argmin(dim=1)

            best_s = best_sub[arange_B, best_g]  # (B,)
            prune_local = subs[best_s]  # (B, 2) local 0-3
            P = group_cols[best_g].gather(
                1, prune_local
            )  # (B, 2) absolute cols

            # ── 3. Compensate all active columns ─────────────────
            P_exp = P.unsqueeze(1).expand(B, K, 2)
            CcolP = C.gather(2, P_exp).float()  # (B, K, 2)

            aa = C[arange_B, P[:, 0], P[:, 0]].float()
            bb = C[arange_B, P[:, 0], P[:, 1]].float()
            dd = C[arange_B, P[:, 1], P[:, 1]].float()
            det2 = (aa * dd - bb * bb).clamp(min=1e-12)

            # C[P,P]^{-1} via closed-form 2×2 inverse
            Cpp_inv = torch.empty(B, 2, 2, device=device)
            Cpp_inv[:, 0, 0] = dd / det2
            Cpp_inv[:, 0, 1] = -bb / det2
            Cpp_inv[:, 1, 0] = -bb / det2
            Cpp_inv[:, 1, 1] = aa / det2

            # w -= C[:, P] @ C[P,P]^{-1} @ w[P]
            wp = w.gather(1, P)  # (B, 2)
            delta = torch.bmm(
                CcolP,
                torch.bmm(Cpp_inv, wp.unsqueeze(2)),
            ).squeeze(2)
            w -= delta
            w.scatter_(1, P, 0.0)

            # ── 4. Rank-2 Schur update ───────────────────────────
            # C -= factor @ factor^T  where factor = C[:,P] @ chol(C[P,P]^{-1})
            lpp = LA.cholesky(Cpp_inv + eye2)  # (B, 2, 2)
            factor = torch.bmm(
                CcolP.to(c_dtype), lpp.to(c_dtype)
            )  # (B, K, 2)
            C.baddbmm_(
                factor,
                factor.transpose(1, 2),
                alpha=-1.0,
            )

            # Zero pruned rows/cols for numerical hygiene
            for j in range(2):
                C[arange_B, P[:, j], :] = 0
                C[arange_B, :, P[:, j]] = 0

            pruned[arange_B, best_g] = True

        W[r0:r1] = w
        del C

    return W


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Exact greedy OBS benchmark for 2:4"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=0,
        help="Use first N rows (0=all)",
    )
    parser.add_argument("--chunk", type=int, default=16)
    args = parser.parse_args()

    W0 = torch.load(
        W_PATH, map_location=DEVICE, weights_only=True
    ).float()
    if args.rows > 0:
        W0 = W0[: args.rows]
    X_cpu = torch.load(
        X_PATH, map_location="cpu", weights_only=True
    )
    M, K = W0.shape
    N = X_cpu.shape[0]
    progress(f"W: {W0.shape}, X: {X_cpu.shape}")

    progress("Computing H...")
    H = compute_hessian(X_cpu, device=DEVICE)
    del X_cpu
    torch.cuda.synchronize(DEVICE)

    ref_out = output_error(W0, torch.zeros_like(W0), H, N)
    progress(f"Reference ||X W^T||_F = {ref_out:.4e}")

    progress("Computing C = (H + damp·I)^{-1}...")
    C = StructuredOBS.compute_inverse(H, damp=1e-4)
    torch.cuda.synchronize(DEVICE)

    results = []

    # ── Magnitude ──
    progress("\n[Magnitude]")
    t0 = time.time()
    W_mag = magnitude(W0, scope_size=4, num_keep=2)
    t = time.time() - t0
    loss = output_error(W_mag, W0, H, N)
    check_24(W_mag)
    results.append(("Magnitude", loss, t))
    progress(
        f"  Loss={loss:.4e} ({loss/ref_out*100:.4f}%)  "
        f"Time={t:.1f}s"
    )
    del W_mag

    # ── SparseGPT ──
    progress("\n[SparseGPT]")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_sgpt = sparsegpt(W0, H, scope_size=4)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss_sgpt = output_error(W_sgpt, W0, H, N)
    check_24(W_sgpt)
    results.append(("SparseGPT", loss_sgpt, t))
    progress(
        f"  Loss={loss_sgpt:.4e} ({loss_sgpt/ref_out*100:.4f}%)  "
        f"Time={t:.1f}s"
    )
    mask_sgpt = W_sgpt.abs() < 1e-10
    del W_sgpt

    # ── Exact Greedy OBS (highest-first) ──
    progress("\n[Exact Greedy OBS — highest-first]")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_greedy = exact_greedy_24(
        W0, C, chunk_size=args.chunk, order="highest"
    )
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = output_error(W_greedy, W0, H, N)
    check_24(W_greedy)
    mask_greedy = W_greedy.abs() < 1e-10
    overlap = (
        (mask_greedy & mask_sgpt).sum().item()
        / mask_sgpt.sum().item()
    )
    results.append(("Greedy OBS (highest)", loss, t))
    progress(
        f"  Loss={loss:.4e}|{loss**2} ({loss/ref_out*100:.4f}%)  "
        f"Time={t:.1f}s"
    )
    progress(f"  Mask overlap w/ SparseGPT: {overlap*100:.2f}%")
    del W_greedy, mask_greedy

    # ── Report ──
    progress(
        f"\n{'Method':<30} {'Loss':>14} {'Loss%':>8} "
        f"{'Time':>8} {'vs SGPT':>10}"
    )
    progress("-" * 76)
    for name, loss_val, t in results:
        vs = (
            (1 - loss_val / loss_sgpt) * 100
            if loss_val < loss_sgpt
            else -(1 - loss_sgpt / loss_val) * 100
        )
        sign = "+" if vs >= 0 else ""
        progress(
            f"{name:<30} {loss_val:>14.4e} "
            f"{loss_val/ref_out*100:>6.4f}% "
            f"{t:>7.1f}s {sign}{vs:>8.2f}%"
        )


if __name__ == "__main__":
    main()
