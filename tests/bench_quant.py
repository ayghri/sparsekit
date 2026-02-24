"""
MXFP4 quantization benchmark: naive vs OBS-compensated.

W (2560, 9728), X (244449, 9728)

Naive MXFP4:       quantize W directly, no error compensation
OBS largest-first: progressive quantization, largest-loss blocks first
OBS left-to-right: progressive quantization, natural column order
OBS smallest-first: progressive quantization, smallest-loss blocks first
"""

import time
import torch

from sparsekit import StructuredOBS, mxfp4_quantize, quantize_obs


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

    # ── Naive MXFP4 ──
    progress(f"\n{'='*70}")
    progress(f"MXFP4 quantization (block_size={BLOCK_SIZE})")
    progress(f"{'='*70}")

    progress("\n  [Naive MXFP4] Running...")
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    W_naive = mxfp4_quantize(W0, block_size=BLOCK_SIZE)
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_naive, W0, H, N)
    results.append(("Naive MXFP4", loss, t))
    progress(f"  [Naive MXFP4] Loss={loss:.4e}, Time={t:.1f}s")
    del W_naive

    # ── OBS left-to-right ──
    progress("\n  [OBS left-to-right] Running...")
    W_ltr = W0.clone()
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    quantize_obs(W_ltr, H, block_size=BLOCK_SIZE, damp=1e-4, C=C,
                 order="left_to_right")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_ltr, W0, H, N)
    results.append(("OBS left-to-right", loss, t))
    progress(f"  [OBS left-to-right] Loss={loss:.4e}, Time={t:.1f}s")
    del W_ltr

    # ── OBS largest-first ──
    progress("\n  [OBS largest-first] Running...")
    W_lf = W0.clone()
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    quantize_obs(W_lf, H, block_size=BLOCK_SIZE, damp=1e-4, C=C,
                 order="largest_first")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_lf, W0, H, N)
    results.append(("OBS largest-first", loss, t))
    progress(f"  [OBS largest-first] Loss={loss:.4e}, Time={t:.1f}s")
    del W_lf

    # ── OBS smallest-first ──
    progress("\n  [OBS smallest-first] Running...")
    W_sf = W0.clone()
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
    quantize_obs(W_sf, H, block_size=BLOCK_SIZE, damp=1e-4, C=C,
                 order="smallest_first")
    torch.cuda.synchronize(DEVICE)
    t = time.time() - t0
    loss = compute_loss(W_sf, W0, H, N)
    results.append(("OBS smallest-first", loss, t))
    progress(f"  [OBS smallest-first] Loss={loss:.4e}, Time={t:.1f}s")
    del W_sf

    # ── Report ──
    progress(f"\n{'='*70}")
    progress("Results")
    progress(f"{'='*70}")
    progress(f"\n  {'Method':<25} {'Loss':>14} {'Norm.':>10} {'Time':>10}")
    progress(f"  {'-'*61}")
    for name, loss, t in results:
        progress(f"  {name:<25} {loss:>14.4e} {loss/ref*100:>8.4f}% {t:>8.1f}s")

    progress(f"\n  --- vs Naive MXFP4 ---")
    naive_loss = results[0][1]
    for name, loss, _ in results[1:]:
        if loss < naive_loss:
            progress(f"  {name} beats Naive by {(1 - loss/naive_loss)*100:.2f}%")
        else:
            progress(f"  Naive beats {name} by {(1 - naive_loss/loss)*100:.2f}%")

    best_name, best_loss, _ = min(results, key=lambda x: x[1])
    progress(f"\n  Best: {best_name} ({best_loss:.4e})")


if __name__ == "__main__":
    main()
