"""
True OBS benchmark using sparsekit BlockSpec/ScopeSpec/StructuredOBS.

Compares SparseGPT baseline against True OBS (per-row Schur complement)
implemented via the sparsekit structured pruning abstractions.

Patterns: 2:4, 4:8, coupled 2:4 use StructuredOBS.prune_true_obs().
Block-16 (8-row coupled) uses direct implementation (row coupling).

Usage:
    python scripts/structured_obs.py [--pattern 24|48|coupled24|block16] [--rows 128] [--ng 64]
"""

import argparse
import time

import torch

from sparsekit import BlockSpec, ScopeSpec, StructuredOBS
from sparsekit.view import View
from sparsekit.pruners import compute_hessian, output_error
from sparsekit.pruners.sparsegpt import (
    sparsegpt,
    sparsegpt_coupled_24,
    sparsegpt_block16,
)

DEVICE = torch.device("cuda")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"


def progress(msg):
    print(msg, flush=True)


def measure_sparsity(W):
    return (W.abs() < 1e-10).sum().item() / W.numel() * 100


def check_pattern(W, block_size, scope_size, label):
    M, K = W.shape
    bpg = scope_size // block_size
    num_prune = bpg // 2
    zeros = (W.abs() < 1e-10).view(
        M, K // scope_size, bpg, block_size
    )
    block_dead = zeros.all(dim=-1)
    pruned_per_scope = block_dead.sum(dim=-1)
    ok = (pruned_per_scope == num_prune).all().item()
    if not ok:
        bad = (pruned_per_scope != num_prune).sum().item()
        progress(
            f"  WARNING: {label} has {bad} scopes violating pattern!"
        )
    return ok


def check_coupled_24(W, label=""):
    M, K = W.shape
    Wv = W.as_strided(
        size=(M, K // 16, 8, 2), stride=(K, 16, 1, 8)
    )
    pair_dead = (Wv.abs() < 1e-10).all(dim=-1)
    pair_dead_g = pair_dead.view(M, K // 16, 2, 4)
    pruned_per_scope = pair_dead_g.sum(dim=-1)
    ok = (pruned_per_scope == 2).all().item()
    if not ok:
        bad = (pruned_per_scope != 2).sum().item()
        progress(
            f"  WARNING: {label} has {bad} scopes violating coupled 2:4!"
        )
    return ok


BLK16 = 16
CHUNK16 = 16
PAIRS16 = 8


def check_block16(W, label=""):
    M, K = W.shape
    G = K // BLK16
    bad = 0
    for c0 in range(0, M, CHUNK16):
        Wc = W[c0 : c0 + CHUNK16]
        block_dead = (
            Wc.view(CHUNK16, G, BLK16).abs() < 1e-10
        ).all(dim=-1)
        for p in range(PAIRS16):
            pair_dead = (
                block_dead[p].long()
                + block_dead[p + PAIRS16].long()
            )
            bad += (pair_dead != 1).sum().item()
    ok = bad == 0
    if not ok:
        progress(
            f"  WARNING: {label} has {bad} scopes violating block-16 pattern!"
        )
    return ok


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        choices=["24", "48", "coupled24", "block16"],
        default="24",
    )
    parser.add_argument("--ng", type=int, default=64)
    parser.add_argument(
        "--rows",
        type=int,
        default=0,
        help="Use only first N rows (0=all)",
    )
    args = parser.parse_args()

    if args.pattern == "24":
        block_size, scope_size = 1, 4
        pattern_label = "2:4"
    elif args.pattern == "48":
        block_size, scope_size = 2, 8
        pattern_label = "4:8"
    elif args.pattern == "coupled24":
        block_size, scope_size = 1, 4
        pattern_label = "Coupled 2:4"
    else:
        block_size, scope_size = 16, 32
        pattern_label = "Block-16 (8-row coupled)"

    bpg = scope_size // block_size
    nnz = bpg // 2

    progress(
        f"Pattern: {pattern_label}, ng={args.ng}, "
        f"rows={args.rows or 'all'}"
    )

    progress("Loading data...")
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
    progress(f"  W: {W0.shape}, X: {X_cpu.shape}")

    progress("Computing H...")
    t0 = time.time()
    H = compute_hessian(X_cpu, device=DEVICE)
    torch.cuda.synchronize(DEVICE)
    progress(f"  H computed in {time.time() - t0:.1f}s")
    del X_cpu

    ref_out = output_error(W0, torch.zeros_like(W0), H, N)
    ref_w = W0.norm().item()
    progress(
        f"Reference ||X W^T||_F = {ref_out:.4e},  ||W||_F = {ref_w:.4e}"
    )

    progress("Computing C = H^{-1}...")
    t0 = time.time()
    C = StructuredOBS.compute_inverse(H, damp=1e-4)
    torch.cuda.synchronize(DEVICE)
    progress(f"  C computed in {time.time() - t0:.1f}s")

    results = []

    # ── SparseGPT ──
    progress("\n[SparseGPT] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    if args.pattern == "coupled24":
        W_sgpt = sparsegpt_coupled_24(W0, H)
    elif args.pattern == "block16":
        W_sgpt = sparsegpt_block16(W0, H)
    else:
        W_sgpt = sparsegpt(
            W0, H, block_size=block_size, scope_size=scope_size
        )
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    out_loss = output_error(W_sgpt, W0, H, N)
    sp = measure_sparsity(W_sgpt)
    if args.pattern == "coupled24":
        check_coupled_24(W_sgpt, "SparseGPT")
    elif args.pattern == "block16":
        check_block16(W_sgpt, "SparseGPT")
    else:
        check_pattern(W_sgpt, block_size, scope_size, "SparseGPT")
    mask_sgpt = W_sgpt.abs() < 1e-10
    results.append(("SparseGPT", out_loss, t, sp, 1.0))
    progress(
        f"  Out={out_loss:.4e} ({out_loss/ref_out*100:.4f}%),  "
        f"Sp={sp:.0f}%,  Time={t:.1f}s"
    )
    del W_sgpt

    # ── True OBS ──
    lbl = f"True OBS ng={args.ng} ord indep"
    progress(f"\n[{lbl}] Running...")

    W_param = torch.nn.Parameter(W0.clone())
    if args.pattern == "block16":
        n_blk_chunks = M // CHUNK16
        torch.cuda.synchronize(DEVICE)
        t0 = time.time()
        for ci in range(n_blk_chunks):
            c0 = ci * CHUNK16
            chunk_data = W_param.data[c0 : c0 + CHUNK16]
            view = View(
                chunk_data,
                shape=(PAIRS16, 2, K),
                stride=(K, PAIRS16 * K, 1),
            )
            block = BlockSpec(view, shape=(1, 1, BLK16))
            scope = ScopeSpec(block, shape=(1, 2, 1))
            obs = StructuredOBS(scope, H, inv_h=C)
            obs.prune_true_obs(
                nnz=1,
                ng=args.ng,
                chunk_size=PAIRS16,
                order="largest_first",
                progress_fn=(
                    lambda m, _ci=ci: print(
                        f"\r  blk {_ci + 1}/{n_blk_chunks}: {m}",
                        end="",
                        flush=True,
                    )
                ),
            )
        torch.cuda.synchronize(DEVICE)
        t = time.time() - t0
    elif args.pattern == "coupled24":
        view = View(
            W_param,
            shape=(M, K // 16, 8, 2),
            stride=(K, 16, 1, 8),
        )
        block = BlockSpec(view, shape=(1, 1, 1, 2))
        scope = ScopeSpec(block, shape=(1, 1, 4, 1))
        obs = StructuredOBS(scope, H, inv_h=C)

        torch.cuda.synchronize(DEVICE)
        t0 = time.time()
        obs.prune_true_obs(
            nnz=nnz,
            ng=args.ng,
            order="largest_first",
            scoring="independent",
            progress_fn=lambda m: print(
                f"\r  {m}", end="", flush=True
            ),
        )
        torch.cuda.synchronize(DEVICE)
        t = time.time() - t0
    else:
        block = BlockSpec(W_param, shape=(1, block_size))
        scope = ScopeSpec(block, shape=(1, bpg))
        obs = StructuredOBS(scope, H, inv_h=C)

        torch.cuda.synchronize(DEVICE)
        t0 = time.time()
        obs.prune_true_obs(
            nnz=nnz,
            ng=args.ng,
            order="largest_first",
            scoring="independent",
            progress_fn=lambda m: print(
                f"\r  {m}", end="", flush=True
            ),
        )
        torch.cuda.synchronize(DEVICE)
        t = time.time() - t0

    W_obs = W_param.data

    out_loss = output_error(W_obs, W0, H, N)
    sp = measure_sparsity(W_obs)
    if args.pattern == "coupled24":
        check_coupled_24(W_obs, lbl)
    elif args.pattern == "block16":
        check_block16(W_obs, lbl)
    else:
        check_pattern(W_obs, block_size, scope_size, lbl)
    mask_obs = W_obs.abs() < 1e-10
    overlap = (
        (mask_obs & mask_sgpt).sum().item()
        / mask_sgpt.sum().item()
    )
    results.append((lbl, out_loss, t, sp, overlap))
    progress(
        f"\n  Out={out_loss:.4e} ({out_loss/ref_out*100:.4f}%),  "
        f"Sp={sp:.0f}%,  "
        f"Mask overlap={overlap*100:.2f}%,  Time={t:.1f}s"
    )
    del W_obs

    # ── Report ──
    out_sgpt = results[0][1]
    progress(
        f"\n{'Method':<30} {'||XdW^T||':>14} {'out%':>8} "
        f"{'Sp':>6} {'Mask OL':>8} {'Time':>8} {'vs SGPT':>10}"
    )
    progress("-" * 90)
    for name, out_loss, t, sp, ol in results:
        vs = (
            (1 - out_loss / out_sgpt) * 100
            if out_loss < out_sgpt
            else -(1 - out_sgpt / out_loss) * 100
        )
        sign = "+" if vs >= 0 else ""
        progress(
            f"{name:<30} {out_loss:>14.4e} {out_loss/ref_out*100:>6.4f}% "
            f"{sp:>4.0f}% {ol*100:>6.2f}% "
            f"{t:>7.1f}s {sign}{vs:>8.2f}%"
        )


if __name__ == "__main__":
    main()
