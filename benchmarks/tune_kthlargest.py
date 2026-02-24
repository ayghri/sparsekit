"""
Tune Triton kth_largest kernel parameters for m=2048.
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np


@triton.jit
def kth_largest_kernel_tuned(
    x_ptr,
    kth_ptr,
    n_rows,
    n_cols,
    stride_k,
    kth: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_K)
    row_indices = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)

    ptr_offsets = row_indices[:, None] * stride_k + col_offsets[None, :]

    row_mask = row_indices < n_rows
    col_mask = col_offsets < n_cols
    mask = row_mask[:, None] & col_mask[None, :]

    data = tl.load(x_ptr + ptr_offsets, mask=mask, other=float("-inf"))
    data = tl.topk(data, kth)

    select_offsets = tl.arange(0, kth)[None, :]
    kth_value = tl.sum(tl.where(select_offsets == (kth - 1), data, 0.0), axis=1)

    tl.store(kth_ptr + row_indices, kth_value, mask=row_mask)


def kth_largest_tuned(x: torch.Tensor, k: int, BLOCK_M: int, BLOCK_K: int, num_warps: int):
    original_shape = x.shape
    cols = original_shape[-1]
    x_flat = x.view(-1, cols)
    rows = x_flat.shape[0]

    kthlargest_vals = torch.empty(rows, device=x.device, dtype=x.dtype)
    num_blocks = triton.cdiv(rows, BLOCK_M)

    grid = (num_blocks,)
    kth_largest_kernel_tuned[grid](
        x_ptr=x_flat,
        kth_ptr=kthlargest_vals,
        n_rows=rows,
        n_cols=cols,
        kth=k,
        stride_k=x_flat.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )

    output_shape = original_shape[:-1]
    if len(output_shape) == 0:
        return kthlargest_vals.squeeze()
    return kthlargest_vals.view(output_shape)


def benchmark_fn(fn, warmup=5, runs=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times), np.std(times)


def torch_topk_kth(x, k, dim=-1):
    topk_vals, _ = torch.topk(x, k, dim=dim, largest=True, sorted=False)
    return topk_vals.min(dim=dim).values


def main():
    print("Tuning Triton kernel for m=2048, k=1024")
    print("=" * 80)

    m = 2048
    k = 1024

    # BLOCK_K must be >= m and power of 2
    block_k_options = [2048]

    # BLOCK_M options
    block_m_options = [1, 2, 4]

    # num_warps options
    num_warps_options = [4, 8, 16, 32]

    test_sizes = [
        (16, 16384),
        (64, 65536),
        (256, 262144),
    ]

    for n, rows in test_sizes:
        print(f"\nRows={rows} (n={n})")
        print(f"{'BLOCK_K':>8} {'BLOCK_M':>8} {'warps':>6} | {'time_ms':>10} {'vs_topk':>10}")
        print("-" * 55)

        x = torch.randn(rows, m, device="cuda")

        # Baseline: topk
        t_topk, _ = benchmark_fn(lambda: torch_topk_kth(x, k))
        print(f"{'topk':>8} {'-':>8} {'-':>6} | {t_topk:>10.3f} {'1.00x':>10}")

        results = []

        for block_k in block_k_options:
            for block_m in block_m_options:
                for num_warps in num_warps_options:
                    try:
                        # Need to capture values properly for lambda
                        def make_fn(bk, bm, nw):
                            return lambda: kth_largest_tuned(x, k, bm, bk, nw)

                        t, _ = benchmark_fn(make_fn(block_k, block_m, num_warps))
                        speedup = t_topk / t
                        results.append((block_k, block_m, num_warps, t, speedup))
                        print(f"{block_k:>8} {block_m:>8} {num_warps:>6} | {t:>10.3f} {speedup:>9.2f}x")
                    except Exception as e:
                        print(f"{block_k:>8} {block_m:>8} {num_warps:>6} | ERROR: {str(e)[:30]}")

        # Best config
        if results:
            best = min(results, key=lambda r: r[3])
            print(f"\nBest: BLOCK_K={best[0]}, BLOCK_M={best[1]}, warps={best[2]} -> {best[3]:.3f}ms ({best[4]:.2f}x vs topk)")

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
