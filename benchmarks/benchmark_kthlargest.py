"""
Benchmark: kth largest implementations

Compares performance of:
1. Triton kernel (sparsekit.kernels.kth_largest)
2. PyTorch kthvalue (torch.kthvalue)
3. PyTorch topk (torch.topk + take last element)
4. PyTorch sort (full sort + index)

Tensor shapes: (1024*n, m) for various n and m values
k = 50% of m (approximately)
"""

import time
import torch
import numpy as np
from typing import Callable, Tuple, List, Dict
import triton

# Import implementations
from sparsekit.kernels import kth_largest as triton_kth_largest
from sparsekit.linalg import kth_largest as torch_kthvalue_impl


def benchmark_fn(fn: Callable, warmup: int = 5, runs: int = 20) -> Tuple[float, float]:
    """Benchmark a function, return (mean_ms, std_ms)."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms

    return np.mean(times), np.std(times)


def torch_topk_kth(x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """Get kth largest using topk and taking the last element."""
    topk_vals, _ = torch.topk(x, k, dim=dim, largest=True, sorted=False)
    return topk_vals.min(dim=dim).values


def torch_sort_kth(x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """Get kth largest using full sort."""
    sorted_vals, _ = torch.sort(x, dim=dim, descending=True)
    return sorted_vals.select(dim, k - 1)


def check_correctness(results: Dict[str, torch.Tensor], reference_name: str = "torch_kthvalue") -> Dict[str, float]:
    """Check correctness of all results against a reference."""
    ref = results[reference_name]
    max_diffs = {}
    for name, result in results.items():
        if name == reference_name:
            max_diffs[name] = 0.0
        else:
            max_diffs[name] = (result - ref).abs().max().item()
    return max_diffs


def run_benchmark_grid(
    n_values: List[int],
    m_values: List[int],
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> List[Dict]:
    """Run benchmarks for all combinations of n and m."""
    results = []

    for m in m_values:
        k = max(1, m // 2)  # k = 50% of m

        print(f"\n{'='*80}")
        print(f"Column size m={m}, k={k} (50%)")
        print(f"{'='*80}")
        print(f"{'n':>6} | {'rows':>10} | {'triton':>12} | {'kthvalue':>12} | {'topk':>12} | {'sort':>12} | {'best':>8}")
        print("-" * 80)

        for n in n_values:
            rows = 1024 * n

            # Skip if triton BLOCK_K would be too large (triton.topk limitation)
            block_k = triton.next_power_of_2(m)
            if block_k > 8192:  # Triton has limits on block size
                print(f"{n:>6} | {rows:>10} | {'SKIP':>12} | {'SKIP':>12} | {'SKIP':>12} | {'SKIP':>12} |")
                continue

            # Create test tensor
            x = torch.randn(rows, m, device=device, dtype=dtype)

            # Define benchmark functions
            def triton_fn():
                return triton_kth_largest(x, k, dim=-1)

            def kthvalue_fn():
                return torch_kthvalue_impl(x, k, dim=-1)

            def topk_fn():
                return torch_topk_kth(x, k, dim=-1)

            def sort_fn():
                return torch_sort_kth(x, k, dim=-1)

            # Check correctness first
            try:
                triton_result = triton_fn()
                kthvalue_result = kthvalue_fn()
                topk_result = topk_fn()
                sort_result = sort_fn()

                # Verify correctness
                ref = kthvalue_result
                triton_diff = (triton_result - ref).abs().max().item()
                topk_diff = (topk_result - ref).abs().max().item()
                sort_diff = (sort_result - ref).abs().max().item()

                if triton_diff > 1e-4 or topk_diff > 1e-4 or sort_diff > 1e-4:
                    print(f"WARNING: Large diff at n={n}, m={m}: triton={triton_diff:.2e}, topk={topk_diff:.2e}, sort={sort_diff:.2e}")
            except Exception as e:
                print(f"{n:>6} | {rows:>10} | ERROR: {str(e)[:50]}")
                continue

            # Benchmark each method
            try:
                triton_time, triton_std = benchmark_fn(triton_fn)
            except Exception:
                triton_time, triton_std = float('inf'), 0.0

            try:
                kthvalue_time, kthvalue_std = benchmark_fn(kthvalue_fn)
            except Exception:
                kthvalue_time, kthvalue_std = float('inf'), 0.0

            try:
                topk_time, topk_std = benchmark_fn(topk_fn)
            except Exception:
                topk_time, topk_std = float('inf'), 0.0

            # Skip sort for very large tensors (OOM risk)
            if rows * m > 500_000_000:  # ~500M elements
                sort_time, sort_std = float('inf'), 0.0
            else:
                try:
                    sort_time, sort_std = benchmark_fn(sort_fn)
                except Exception:
                    sort_time, sort_std = float('inf'), 0.0

            # Clear cache to avoid OOM
            torch.cuda.empty_cache()

            # Find best method
            times = {
                'triton': triton_time,
                'kthvalue': kthvalue_time,
                'topk': topk_time,
                'sort': sort_time,
            }
            best = min(times, key=times.get)

            print(f"{n:>6} | {rows:>10} | {triton_time:>9.3f} ms | {kthvalue_time:>9.3f} ms | {topk_time:>9.3f} ms | {sort_time:>9.3f} ms | {best:>8}")

            results.append({
                'n': n,
                'm': m,
                'k': k,
                'rows': rows,
                'triton_ms': triton_time,
                'kthvalue_ms': kthvalue_time,
                'topk_ms': topk_time,
                'sort_ms': sort_time,
                'best': best,
            })

    return results


def print_summary(results: List[Dict]):
    """Print a summary of which method is best for each regime."""
    print("\n" + "=" * 80)
    print("SUMMARY: Best method by regime")
    print("=" * 80)

    # Group by m value
    m_values = sorted(set(r['m'] for r in results))

    for m in m_values:
        m_results = [r for r in results if r['m'] == m]

        print(f"\nm={m}:")
        triton_wins = sum(1 for r in m_results if r['best'] == 'triton')
        kthvalue_wins = sum(1 for r in m_results if r['best'] == 'kthvalue')
        topk_wins = sum(1 for r in m_results if r['best'] == 'topk')
        sort_wins = sum(1 for r in m_results if r['best'] == 'sort')

        total = len(m_results)
        print(f"  Triton:   {triton_wins:>3}/{total} ({100*triton_wins/total:>5.1f}%)")
        print(f"  kthvalue: {kthvalue_wins:>3}/{total} ({100*kthvalue_wins/total:>5.1f}%)")
        print(f"  topk:     {topk_wins:>3}/{total} ({100*topk_wins/total:>5.1f}%)")
        print(f"  sort:     {sort_wins:>3}/{total} ({100*sort_wins/total:>5.1f}%)")

        # Average speedup of triton over kthvalue
        if m_results:
            avg_speedup = np.mean([r['kthvalue_ms'] / r['triton_ms'] for r in m_results if r['triton_ms'] < float('inf')])
            print(f"  Avg triton speedup vs kthvalue: {avg_speedup:.2f}x")


def main():
    print("=" * 80)
    print("KTH LARGEST BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    if device == "cpu":
        print("WARNING: Running on CPU. Triton kernels require CUDA.")
        return

    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")

    # Benchmark parameters
    # n values: exponential scale from 1 to 1024
    n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # m values: from small to large
    m_values = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    print(f"\nBenchmarking tensor shapes (1024*n, m) with k=m//2")
    print(f"n values: {n_values}")
    print(f"m values: {m_values}")

    results = run_benchmark_grid(n_values, m_values, device=device)

    print_summary(results)

    # Also test a few specific large configurations
    print("\n" + "=" * 80)
    print("EXTRA: Large tensor tests")
    print("=" * 80)

    large_configs = [
        (1024, 64),    # 1M rows, small m
        (1024, 256),   # 1M rows, medium m
        (1024, 1024),  # 1M rows, large m
        (512, 2048),   # 512K rows, very large m
    ]

    print(f"{'config':>15} | {'triton':>12} | {'kthvalue':>12} | {'topk':>12} | {'sort':>12} | {'best':>8}")
    print("-" * 80)

    for n, m in large_configs:
        rows = 1024 * n
        k = m // 2

        x = torch.randn(rows, m, device=device)

        try:
            triton_time, _ = benchmark_fn(lambda: triton_kth_largest(x, k, dim=-1))
        except Exception:
            triton_time = float('inf')

        try:
            kthvalue_time, _ = benchmark_fn(lambda: torch_kthvalue_impl(x, k, dim=-1))
        except Exception:
            kthvalue_time = float('inf')

        try:
            topk_time, _ = benchmark_fn(lambda: torch_topk_kth(x, k, dim=-1))
        except Exception:
            topk_time = float('inf')

        if rows * m > 500_000_000:
            sort_time = float('inf')
        else:
            try:
                sort_time, _ = benchmark_fn(lambda: torch_sort_kth(x, k, dim=-1))
            except Exception:
                sort_time = float('inf')

        torch.cuda.empty_cache()

        times = {'triton': triton_time, 'kthvalue': kthvalue_time, 'topk': topk_time, 'sort': sort_time}
        best = min(times, key=times.get)

        config_str = f"({rows}, {m})"
        print(f"{config_str:>15} | {triton_time:>9.3f} ms | {kthvalue_time:>9.3f} ms | {topk_time:>9.3f} ms | {sort_time:>9.3f} ms | {best:>8}")


if __name__ == "__main__":
    main()
