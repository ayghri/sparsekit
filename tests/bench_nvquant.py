"""
MXFP4 vs NVFP4 quantization benchmark.

MXFP4: FP4 (E2M1) values with UE8M0 scale (power-of-2 only)
NVFP4: FP4 (E2M1) values with E4M3 FP8 scale (finer granularity)

Methods:
  - Naive:     quantize directly, no error compensation
  - GPTQ:      column-sequential, forward-only via Cholesky(H^{-1}), left-to-right
  - GPTQ-ord:  GPTQ with OBS-scored column permutation (largest-first)
  - OBS:       block-sequential, updated C = inv(H[active,active]), left-to-right
  - OBS-ord:   same as OBS but largest-first block ordering

W (2560, 9728), X (244449, 9728)
"""

import time
import torch
import torch.linalg as LA

from sparsekit import (
    StructuredOBS,
    mxfp4_quantize, quantize_obs,
    nvfp4_quantize, quantize_nvfp4_obs,
)
from sparsekit.pruners.nvquant import (
    _DOUBLED_FP4, _DOUBLED_FP4_MAX,
    _build_e4m3_table, _round_to_e4m3,
)


def progress(msg):
    print(msg, flush=True)


DEVICE = torch.device("cuda:1")
W_PATH = "/buckets/checkpoints/layer_0_W.cpt"
X_PATH = "/buckets/checkpoints/layer_0_X.cpt"
BLOCK_SIZE = 16


def compute_H(X, batch_size=4096):
    N, K = X.shape
    H = torch.zeros(K, K, device=DEVICE, dtype=torch.float32)
    for i in range(0, N, batch_size):
        X_b = X[i:i+batch_size].to(device=DEVICE, dtype=torch.float32)
        H.addmm_(X_b.T, X_b)
    H /= N
    return H


def compute_loss(W_quant, W0, H, N, chunk=128):
    M = W0.shape[0]
    total = 0.0
    for c0 in range(0, M, chunk):
        dW = W_quant[c0:c0+chunk] - W0[c0:c0+chunk]
        total += ((dW @ H) * dW).sum().item()
    return total * N


# ── GPTQ reference implementations ───────────────────────────────────────

_DOUBLED_MXFP4 = torch.tensor(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=torch.float32,
)


def _round_to_fp4(w, scale, codebook):
    """Round each element of w to nearest FP4 value given per-row scale.

    w: (M,) values, scale: (M,) per-row scale.
    Returns: (M,) dequantized values.
    """
    possible = scale.unsqueeze(-1) * codebook  # (M, 16)
    idx = (w.unsqueeze(-1) - possible).abs().argmin(dim=-1)  # (M,)
    return scale * codebook[idx]


def gptq_mxfp4(W0, H, fp4_block_size=16, blocksize=128):
    """GPTQ for MXFP4: Cholesky-based quantization with forward error propagation.

    At each FP4 block boundary, computes UE8M0 scale from current column values.
    Each column is quantized using that fixed scale, with error propagated forward.
    """
    M, K = W0.shape
    device = W0.device
    W = W0.clone().float()
    H = H.clone().float()
    codebook = _DOUBLED_MXFP4.to(device=device, dtype=torch.float32)

    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    damp = 0.01 * torch.mean(torch.diag(H))
    diag_idx = torch.arange(K, device=device)
    H[diag_idx, diag_idx] += damp

    L = LA.cholesky(H)
    Hinv_full = torch.cholesky_inverse(L)
    Hinv = LA.cholesky(Hinv_full, upper=True)

    scale = torch.zeros(M, device=device)

    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            col = i1 + i
            # At FP4 block boundary: compute scale from current column values
            if col % fp4_block_size == 0:
                end = min(i + fp4_block_size, count)
                amax = W1[:, i:end].abs().amax(dim=-1)  # (M,)
                safe_amax = torch.where(amax > 0, amax, torch.ones_like(amax))
                scale = torch.pow(2.0, safe_amax.log2().floor() - 3.0)
                scale = torch.where(amax > 0, scale, torch.zeros_like(scale))

            w = W1[:, i]
            q = _round_to_fp4(w, scale, codebook)
            Q1[:, i] = q

            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err

        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return W


def gptq_nvfp4(W0, H, fp4_block_size=16, blocksize=128):
    """GPTQ for NVFP4: same as gptq_mxfp4 but with E4M3 FP8 scales."""
    M, K = W0.shape
    device = W0.device
    W = W0.clone().float()
    H = H.clone().float()
    codebook = _DOUBLED_FP4.to(device=device, dtype=torch.float32)
    e4m3_table = _build_e4m3_table(device)

    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    damp = 0.01 * torch.mean(torch.diag(H))
    diag_idx = torch.arange(K, device=device)
    H[diag_idx, diag_idx] += damp

    L = LA.cholesky(H)
    Hinv_full = torch.cholesky_inverse(L)
    Hinv = LA.cholesky(Hinv_full, upper=True)

    scale = torch.zeros(M, device=device)

    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            col = i1 + i
            # At FP4 block boundary: compute E4M3 scale
            if col % fp4_block_size == 0:
                end = min(i + fp4_block_size, count)
                amax = W1[:, i:end].abs().amax(dim=-1)  # (M,)
                ideal = amax / _DOUBLED_FP4_MAX
                safe_ideal = torch.where(ideal > 0, ideal, torch.ones_like(ideal))
                scale = _round_to_e4m3(safe_ideal, e4m3_table)
                scale = torch.where(amax > 0, scale, torch.zeros_like(scale))

            w = W1[:, i]
            q = _round_to_fp4(w, scale, codebook)
            Q1[:, i] = q

            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err

        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    return W


# ── GPTQ with OBS block ordering ──────────────────────────────────────────

def gptq_ordered_mxfp4(W0, H, fp4_block_size=16, blocksize=128, C=None, damp=1e-4):
    """GPTQ with OBS-ordered columns: score blocks by OBS loss, permute so
    highest-loss blocks come first, run standard GPTQ, then unpermute.

    Combines GPTQ's Cholesky-based forward propagation with OBS's
    largest-first ordering.
    """
    M, K = W0.shape
    device = W0.device
    B = K // fp4_block_size

    if C is None:
        C = StructuredOBS.compute_inverse(H, damp)

    # Score each block by OBS quantization loss
    block_cols = torch.arange(K, device=device).view(B, fp4_block_size)
    Q_naive = mxfp4_quantize(W0, block_size=fp4_block_size)
    E = W0.float() - Q_naive.float()
    E_blocks = E[:, block_cols.reshape(-1)].view(M, B, fp4_block_size)

    eye_bs = 1e-8 * torch.eye(fp4_block_size, device=device)
    C_PP = C[block_cols[:, :, None], block_cols[:, None, :]]
    C_PP_inv = LA.inv(C_PP + eye_bs)

    temp = torch.einsum("mbp,bpq->mbq", E_blocks, C_PP_inv)
    scores = (temp * E_blocks).sum(dim=2).sum(dim=0)  # (B,)

    block_order = torch.argsort(scores, descending=True)
    perm = block_cols[block_order].reshape(-1)
    inv_perm = torch.argsort(perm)

    # Run standard GPTQ on permuted data
    W_result_perm = gptq_mxfp4(
        W0[:, perm], H[perm][:, perm],
        fp4_block_size=fp4_block_size, blocksize=blocksize,
    )
    return W_result_perm[:, inv_perm]


def gptq_ordered_nvfp4(W0, H, fp4_block_size=16, blocksize=128, C=None, damp=1e-4):
    """GPTQ with OBS-ordered columns for NVFP4."""
    M, K = W0.shape
    device = W0.device
    B = K // fp4_block_size

    if C is None:
        C = StructuredOBS.compute_inverse(H, damp)

    block_cols = torch.arange(K, device=device).view(B, fp4_block_size)
    Q_naive = nvfp4_quantize(W0, block_size=fp4_block_size)
    E = W0.float() - Q_naive.float()
    E_blocks = E[:, block_cols.reshape(-1)].view(M, B, fp4_block_size)

    eye_bs = 1e-8 * torch.eye(fp4_block_size, device=device)
    C_PP = C[block_cols[:, :, None], block_cols[:, None, :]]
    C_PP_inv = LA.inv(C_PP + eye_bs)

    temp = torch.einsum("mbp,bpq->mbq", E_blocks, C_PP_inv)
    scores = (temp * E_blocks).sum(dim=2).sum(dim=0)

    block_order = torch.argsort(scores, descending=True)
    perm = block_cols[block_order].reshape(-1)
    inv_perm = torch.argsort(perm)

    W_result_perm = gptq_nvfp4(
        W0[:, perm], H[perm][:, perm],
        fp4_block_size=fp4_block_size, blocksize=blocksize,
    )
    return W_result_perm[:, inv_perm]


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

    progress("\nPrecomputing C = H^{-1}...")
    t0 = time.time()
    C = StructuredOBS.compute_inverse(H, damp=1e-4)
    torch.cuda.synchronize(DEVICE)
    progress(f"  C computed in {time.time() - t0:.1f}s")

    results = []

    def check_quantized(W_q, W0, name, quant_fn):
        """Report ||W_q - quant(W_q)||_max and loss of re-quantized version."""
        W_re = quant_fn(W_q, block_size=BLOCK_SIZE)
        dist = (W_q - W_re).abs().max().item()
        loss_re = compute_loss(W_re, W0, H, N)
        progress(f"    {name}: dist to quant = {dist:.4e}, "
                 f"re-quant loss = {loss_re:.4e} ({loss_re/ref*100:.4f}%)")

    progress(f"\n{'='*70}")
    progress(f"FP4 quantization comparison (block_size={BLOCK_SIZE})")
    progress(f"  MXFP4: UE8M0 scale (power-of-2)")
    progress(f"  NVFP4: E4M3 FP8 scale (finer granularity)")
    progress(f"  Methods: Naive / GPTQ / GPTQ-ord / OBS / OBS-ord")
    progress(f"{'='*70}")

    # ══════════════════════════════════════════════════════════════════════
    # MXFP4
    # ══════════════════════════════════════════════════════════════════════

    # ── Naive MXFP4 ──
    progress("\n  [Naive MXFP4] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_naive_mx = mxfp4_quantize(W0, block_size=BLOCK_SIZE)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_naive_mx, W0, H, N)
    results.append(("Naive MXFP4", loss, t))
    progress(f"  [Naive MXFP4] Loss={loss:.4e}, Time={t:.2f}s")
    check_quantized(W_naive_mx, W0, "Naive MXFP4", mxfp4_quantize)
    del W_naive_mx

    # ── GPTQ MXFP4 ──
    progress("\n  [GPTQ MXFP4] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_gptq_mx = gptq_mxfp4(W0, H, fp4_block_size=BLOCK_SIZE)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_gptq_mx, W0, H, N)
    results.append(("GPTQ MXFP4", loss, t))
    progress(f"  [GPTQ MXFP4] Loss={loss:.4e}, Time={t:.1f}s")
    check_quantized(W_gptq_mx, W0, "GPTQ MXFP4", mxfp4_quantize)
    del W_gptq_mx

    # ── GPTQ-ordered MXFP4 ──
    progress("\n  [GPTQ-ord MXFP4] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_gptqord_mx = gptq_ordered_mxfp4(W0, H, fp4_block_size=BLOCK_SIZE, C=C)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_gptqord_mx, W0, H, N)
    results.append(("GPTQ-ord MXFP4", loss, t))
    progress(f"  [GPTQ-ord MXFP4] Loss={loss:.4e}, Time={t:.1f}s")
    check_quantized(W_gptqord_mx, W0, "GPTQ-ord MXFP4", mxfp4_quantize)
    del W_gptqord_mx

    # ── OBS MXFP4 (left-to-right) ──
    progress("\n  [OBS MXFP4] Running (left-to-right)...")
    W_obs_mx = W0.clone()
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    quantize_obs(W_obs_mx, H, block_size=BLOCK_SIZE, damp=1e-4, C=C,
                 order="left_to_right")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_obs_mx, W0, H, N)
    results.append(("OBS MXFP4", loss, t))
    progress(f"  [OBS MXFP4] Loss={loss:.4e}, Time={t:.1f}s")
    check_quantized(W_obs_mx, W0, "OBS MXFP4", mxfp4_quantize)
    del W_obs_mx

    # ── OBS-ord MXFP4 (largest-first) ──
    progress("\n  [OBS-ord MXFP4] Running (largest-first)...")
    W_obsord_mx = W0.clone()
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    quantize_obs(W_obsord_mx, H, block_size=BLOCK_SIZE, damp=1e-4, C=C,
                 order="largest_first")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_obsord_mx, W0, H, N)
    results.append(("OBS-ord MXFP4", loss, t))
    progress(f"  [OBS-ord MXFP4] Loss={loss:.4e}, Time={t:.1f}s")
    check_quantized(W_obsord_mx, W0, "OBS-ord MXFP4", mxfp4_quantize)
    del W_obsord_mx

    # ══════════════════════════════════════════════════════════════════════
    # NVFP4
    # ══════════════════════════════════════════════════════════════════════

    # ── Naive NVFP4 ──
    progress("\n  [Naive NVFP4] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_naive_nv = nvfp4_quantize(W0, block_size=BLOCK_SIZE)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_naive_nv, W0, H, N)
    results.append(("Naive NVFP4", loss, t))
    progress(f"  [Naive NVFP4] Loss={loss:.4e}, Time={t:.2f}s")
    check_quantized(W_naive_nv, W0, "Naive NVFP4", nvfp4_quantize)
    del W_naive_nv

    # ── GPTQ NVFP4 ──
    progress("\n  [GPTQ NVFP4] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_gptq_nv = gptq_nvfp4(W0, H, fp4_block_size=BLOCK_SIZE)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_gptq_nv, W0, H, N)
    results.append(("GPTQ NVFP4", loss, t))
    progress(f"  [GPTQ NVFP4] Loss={loss:.4e}, Time={t:.1f}s")
    check_quantized(W_gptq_nv, W0, "GPTQ NVFP4", nvfp4_quantize)
    del W_gptq_nv

    # ── GPTQ-ordered NVFP4 ──
    progress("\n  [GPTQ-ord NVFP4] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_gptqord_nv = gptq_ordered_nvfp4(W0, H, fp4_block_size=BLOCK_SIZE, C=C)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_gptqord_nv, W0, H, N)
    results.append(("GPTQ-ord NVFP4", loss, t))
    progress(f"  [GPTQ-ord NVFP4] Loss={loss:.4e}, Time={t:.1f}s")
    check_quantized(W_gptqord_nv, W0, "GPTQ-ord NVFP4", nvfp4_quantize)
    del W_gptqord_nv

    # ── OBS NVFP4 (left-to-right) ──
    progress("\n  [OBS NVFP4] Running (left-to-right)...")
    W_obs_nv = W0.clone()
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    quantize_nvfp4_obs(W_obs_nv, H, block_size=BLOCK_SIZE, damp=1e-4, C=C,
                       order="left_to_right")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_obs_nv, W0, H, N)
    results.append(("OBS NVFP4", loss, t))
    progress(f"  [OBS NVFP4] Loss={loss:.4e}, Time={t:.1f}s")
    check_quantized(W_obs_nv, W0, "OBS NVFP4", nvfp4_quantize)
    del W_obs_nv

    # ── OBS-ord NVFP4 (largest-first) ──
    progress("\n  [OBS-ord NVFP4] Running (largest-first)...")
    W_obsord_nv = W0.clone()
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    quantize_nvfp4_obs(W_obsord_nv, H, block_size=BLOCK_SIZE, damp=1e-4, C=C,
                       order="largest_first")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_obsord_nv, W0, H, N)
    results.append(("OBS-ord NVFP4", loss, t))
    progress(f"  [OBS-ord NVFP4] Loss={loss:.4e}, Time={t:.1f}s")
    check_quantized(W_obsord_nv, W0, "OBS-ord NVFP4", nvfp4_quantize)
    del W_obsord_nv

    # ══════════════════════════════════════════════════════════════════════
    # Report
    # ══════════════════════════════════════════════════════════════════════
    progress(f"\n{'='*70}")
    progress("Results")
    progress(f"{'='*70}")
    progress(f"\n  {'Method':<25} {'Loss':>14} {'Norm.':>10} {'Time':>10}")
    progress(f"  {'-'*61}")
    for name, loss, t in results:
        progress(f"  {name:<25} {loss:>14.4e} {loss/ref*100:>8.4f}% {t:>8.2f}s")

    # Named lookups
    r = {name: (loss, t) for name, loss, t in results}

    def compare(a, b):
        la, lb = r[a][0], r[b][0]
        if la < lb:
            progress(f"  {a} beats {b} by {(1 - la/lb)*100:.2f}%")
        else:
            progress(f"  {b} beats {a} by {(1 - lb/la)*100:.2f}%")

    progress(f"\n  --- MXFP4: vs Naive ---")
    for name in ["GPTQ MXFP4", "GPTQ-ord MXFP4", "OBS MXFP4", "OBS-ord MXFP4"]:
        loss = r[name][0]
        base = r["Naive MXFP4"][0]
        progress(f"  {name} beats Naive by {(1 - loss/base)*100:.2f}%")

    progress(f"\n  --- MXFP4: ordering effect ---")
    compare("GPTQ MXFP4", "GPTQ-ord MXFP4")
    compare("OBS MXFP4", "OBS-ord MXFP4")

    progress(f"\n  --- MXFP4: OBS vs GPTQ ---")
    compare("OBS MXFP4", "GPTQ MXFP4")
    compare("OBS-ord MXFP4", "GPTQ-ord MXFP4")

    progress(f"\n  --- NVFP4: vs Naive ---")
    for name in ["GPTQ NVFP4", "GPTQ-ord NVFP4", "OBS NVFP4", "OBS-ord NVFP4"]:
        loss = r[name][0]
        base = r["Naive NVFP4"][0]
        progress(f"  {name} beats Naive by {(1 - loss/base)*100:.2f}%")

    progress(f"\n  --- NVFP4: ordering effect ---")
    compare("GPTQ NVFP4", "GPTQ-ord NVFP4")
    compare("OBS NVFP4", "OBS-ord NVFP4")

    progress(f"\n  --- NVFP4: OBS vs GPTQ ---")
    compare("OBS NVFP4", "GPTQ NVFP4")
    compare("OBS-ord NVFP4", "GPTQ-ord NVFP4")

    progress(f"\n  --- Scale format: NVFP4 vs MXFP4 ---")
    for method in ["Naive", "GPTQ", "GPTQ-ord", "OBS", "OBS-ord"]:
        mx = r[f"{method} MXFP4"][0]
        nv = r[f"{method} NVFP4"][0]
        if nv < mx:
            progress(f"  {method}: NVFP4 beats MXFP4 by {(1 - nv/mx)*100:.2f}%")
        else:
            progress(f"  {method}: MXFP4 beats NVFP4 by {(1 - mx/nv)*100:.2f}%")

    best_name, best_loss, _ = min(results, key=lambda x: x[1])
    progress(f"\n  Best overall: {best_name} ({best_loss:.4e}, "
             f"{best_loss/ref*100:.4f}%)")


if __name__ == "__main__":
    main()
