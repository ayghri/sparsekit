"""
Tune Triton kth_largest - exploring k values and num_stages.
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np


@triton.jit
def kth_largest_kernel_v2(
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


def kth_largest_v2(x: torch.Tensor, k: int, BLOCK_M: int, BLOCK_K: int, num_warps: int, num_stages: int):
    original_shape = x.shape
    cols = original_shape[-1]
    x_flat = x.view(-1, cols)
    rows = x_flat.shape[0]

    kthlargest_vals = torch.empty(rows, device=x.device, dtype=x.dtype)
    num_blocks = triton.cdiv(rows, BLOCK_M)

    grid = (num_blocks,)
    kth_largest_kernel_v2[grid](
        x_ptr=x_flat,
        kth_ptr=kthlargest_vals,
        n_rows=rows,
        n_cols=cols,
        kth=k,
        stride_k=x_flat.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
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
    print("=" * 80)
    print("Part 1: Testing different k values for m=2048")
    print("=" * 80)

    rows = 65536
    m = 2048

    k_values = [64, 128, 256, 512, 1024]

    print(f"\nRows={rows}, m={m}")
    print(f"{'k':>6} {'k%':>6} | {'triton':>10} {'topk':>10} {'ratio':>10}")
    print("-" * 50)

    for k in k_values:
        x = torch.randn(rows, m, device="cuda")

        def make_triton_fn(kval):
            return lambda: kth_largest_v2(x, kval, 1, 2048, 16, 1)

        def make_topk_fn(kval):
            return lambda: torch_topk_kth(x, kval)

        t_triton, _ = benchmark_fn(make_triton_fn(k))
        t_topk, _ = benchmark_fn(make_topk_fn(k))
        ratio = t_triton / t_topk

        print(f"{k:>6} {100*k/m:>5.0f}% | {t_triton:>10.3f} {t_topk:>10.3f} {ratio:>10.2f}x")

        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("Part 2: Testing num_stages for m=2048, k=1024")
    print("=" * 80)

    rows = 65536
    m = 2048
    k = 1024

    x = torch.randn(rows, m, device="cuda")
    t_topk, _ = benchmark_fn(lambda: torch_topk_kth(x, k))
    print(f"\ntopk baseline: {t_topk:.3f} ms")

    print(f"\n{'stages':>8} {'warps':>8} | {'time_ms':>10} {'ratio':>10}")
    print("-" * 45)

    for num_stages in [1, 2, 3, 4]:
        for num_warps in [8, 16, 32]:
            def make_fn(ns, nw):
                return lambda: kth_largest_v2(x, k, 1, 2048, nw, ns)

            t, _ = benchmark_fn(make_fn(num_stages, num_warps))
            ratio = t / t_topk
            print(f"{num_stages:>8} {num_warps:>8} | {t:>10.3f} {ratio:>10.2f}x")

    print("\n" + "=" * 80)
    print("Part 3: Comparison across different m values")
    print("=" * 80)

    rows = 65536
    m_values = [128, 256, 512, 1024, 2048, 4096]

    print(f"\nRows={rows}, k=50% of m")
    print(f"{'m':>6} {'k':>6} | {'triton':>10} {'topk':>10} {'ratio':>10} {'winner':>10}")
    print("-" * 65)

    for m in m_values:
        k = m // 2
        block_k = triton.next_power_of_2(m)

        x = torch.randn(rows, m, device="cuda")

        try:
            def make_triton_fn():
                return lambda: kth_largest_v2(x, k, 1, block_k, 16, 1)

            t_triton, _ = benchmark_fn(make_triton_fn())
        except Exception as e:
            t_triton = float('inf')

        t_topk, _ = benchmark_fn(lambda: torch_topk_kth(x, k))
        ratio = t_triton / t_topk
        winner = "triton" if t_triton < t_topk else "topk"

        print(f"{m:>6} {k:>6} | {t_triton:>10.3f} {t_topk:>10.3f} {ratio:>10.2f}x {winner:>10}")

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
