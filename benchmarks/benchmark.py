"""
Benchmark: Python vs C++ Backend

Compares performance and correctness of the pure Python implementation
against the C++/LibTorch backend.
"""

import time
import torch
import numpy as np
from contextlib import contextmanager
from typing import Callable, Tuple, Dict, Any

# Import both implementations
from sparsekit import blocks as py_blocks
from sparsekit import groups as py_groups
from sparsekit import linalg as py_linalg
from sparsekit import utils as py_utils

import sparsekit._C as cpp

# Timing utilities
@contextmanager
def timer():
    """Context manager for timing code blocks."""
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    yield result
    result["elapsed"] = time.perf_counter() - start


def benchmark_fn(fn: Callable, warmup: int = 3, runs: int = 10) -> Tuple[float, float]:
    """Benchmark a function, return (mean_ms, std_ms)."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append((time.perf_counter() - start) * 1000)  # ms

    return np.mean(times), np.std(times)


def check_close(a: torch.Tensor, b: torch.Tensor, name: str, atol: float = 1e-5) -> bool:
    """Check if two tensors are close, print result."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False
    max_diff = (a - b).abs().max().item()
    match = max_diff < atol
    status = "MATCH" if match else "MISMATCH"
    print(f"  {name}: {status} (max_diff={max_diff:.2e})")
    return match


def print_speedup(py_time: float, cpp_time: float, name: str):
    """Print timing comparison."""
    speedup = py_time / cpp_time if cpp_time > 0 else float('inf')
    print(f"  {name}: Python={py_time:.3f}ms, C++={cpp_time:.3f}ms, Speedup={speedup:.2f}x")


# ============================================================================
# Benchmarks
# ============================================================================

def benchmark_interleave_unsqueeze(sizes: list):
    """Benchmark interleave_unsqueeze."""
    print("\n" + "=" * 60)
    print("Benchmark: interleave_unsqueeze")
    print("=" * 60)

    for size in sizes:
        t = torch.randn(*size)
        print(f"\nShape: {size}")

        # Python
        def py_fn():
            return py_utils.interleave_unsqueeze(t.clone())

        # C++
        def cpp_fn():
            return cpp.utils.interleave_unsqueeze(t.clone())

        # Correctness
        py_result = py_fn()
        cpp_result = cpp_fn()
        check_close(py_result, cpp_result, "Result")

        # Speed
        py_time, _ = benchmark_fn(py_fn)
        cpp_time, _ = benchmark_fn(cpp_fn)
        print_speedup(py_time, cpp_time, "Time")


def benchmark_merge_odd_dims(sizes: list):
    """Benchmark merge_odd_dims."""
    print("\n" + "=" * 60)
    print("Benchmark: merge_odd_dims")
    print("=" * 60)

    for size in sizes:
        t = torch.randn(*size)
        print(f"\nShape: {size}")

        # Python
        def py_fn():
            return py_utils.merge_odd_dims(t.clone())

        # C++
        def cpp_fn():
            return cpp.utils.merge_odd_dims(t.clone())

        # Correctness
        py_result = py_fn()
        cpp_result = cpp_fn()
        check_close(py_result, cpp_result, "Result")

        # Speed
        py_time, _ = benchmark_fn(py_fn)
        cpp_time, _ = benchmark_fn(cpp_fn)
        print_speedup(py_time, cpp_time, "Time")


def benchmark_kth_largest(sizes: list):
    """Benchmark kth_largest."""
    print("\n" + "=" * 60)
    print("Benchmark: kth_largest")
    print("=" * 60)

    for size in sizes:
        t = torch.randn(*size)
        k = min(10, size[-1])
        print(f"\nShape: {size}, k={k}")

        # Python
        def py_fn():
            return py_linalg.kth_largest(t, k, dim=-1)

        # C++
        def cpp_fn():
            return cpp.ops.kth_largest(t, k, -1, False)

        # Correctness
        py_result = py_fn()
        cpp_result = cpp_fn()
        check_close(py_result, cpp_result, "Result")

        # Speed
        py_time, _ = benchmark_fn(py_fn)
        cpp_time, _ = benchmark_fn(cpp_fn)
        print_speedup(py_time, cpp_time, "Time")


def benchmark_block_spec(sizes: list):
    """Benchmark BlockSpec operations."""
    print("\n" + "=" * 60)
    print("Benchmark: BlockSpec")
    print("=" * 60)

    for tensor_shape, block_shape in sizes:
        print(f"\nTensor: {tensor_shape}, Block: {block_shape}")

        # Create parameters
        py_param = torch.nn.Parameter(torch.randn(*tensor_shape))
        cpp_param = torch.nn.Parameter(py_param.data.clone())

        # Create BlockSpecs
        py_block = py_blocks.BlockSpec(py_param, block_shape, "py_block")
        cpp_block = cpp.BlockSpec(cpp_param, list(block_shape), "cpp_block")

        # --- block_norms ---
        print("\n  block_norms:")

        def py_norms():
            return py_block.block_norms(py_param.data)

        def cpp_norms():
            return cpp_block.block_norms(cpp_param.data, 2)

        py_result = py_norms()
        cpp_result = cpp_norms()
        check_close(py_result, cpp_result, "    Result")

        py_time, _ = benchmark_fn(py_norms)
        cpp_time, _ = benchmark_fn(cpp_norms)
        print_speedup(py_time, cpp_time, "    Time")

        # --- block_view ---
        print("\n  block_view:")

        def py_view():
            return py_block.block_view(py_param.data)

        def cpp_view():
            return cpp_block.block_view(cpp_param.data, True)

        py_result = py_view()
        cpp_result = cpp_view()
        check_close(py_result, cpp_result, "    Result")

        py_time, _ = benchmark_fn(py_view)
        cpp_time, _ = benchmark_fn(cpp_view)
        print_speedup(py_time, cpp_time, "    Time")

        # --- hard_threshold ---
        print("\n  hard_threshold:")
        thresholds = torch.ones(py_block.grid_shape) * 1.0

        # Reset params
        py_param.data.copy_(torch.randn(*tensor_shape))
        cpp_param.data.copy_(py_param.data)

        def py_hard():
            py_param.data.copy_(cpp_param.data)  # Reset
            py_block.hard_threshold(thresholds)

        def cpp_hard():
            cpp_param.data.copy_(py_param.data)  # Reset to same
            cpp_block.hard_threshold(thresholds, None)

        # For correctness, run once with same input
        test_data = torch.randn(*tensor_shape)
        py_param.data.copy_(test_data)
        cpp_param.data.copy_(test_data)
        py_block.hard_threshold(thresholds)
        cpp_block.hard_threshold(thresholds, None)
        check_close(py_param.data, cpp_param.data, "    Result")

        py_time, _ = benchmark_fn(py_hard)
        cpp_time, _ = benchmark_fn(cpp_hard)
        print_speedup(py_time, cpp_time, "    Time")

        # --- soft_threshold (Euclidean) ---
        print("\n  soft_threshold (Euclidean):")
        thresholds = torch.ones(py_block.grid_shape) * 0.5

        # For correctness
        test_data = torch.randn(*tensor_shape)
        py_param.data.copy_(test_data)
        cpp_param.data.copy_(test_data)
        py_block.soft_threshold(thresholds, conditioners=None)
        cpp_block.soft_threshold(thresholds, None, False, 20, 1e-20, 1e-8)
        check_close(py_param.data, cpp_param.data, "    Result")

        def py_soft():
            py_param.data.copy_(test_data)
            py_block.soft_threshold(thresholds, conditioners=None)

        def cpp_soft():
            cpp_param.data.copy_(test_data)
            cpp_block.soft_threshold(thresholds, None, False, 20, 1e-20, 1e-8)

        py_time, _ = benchmark_fn(py_soft)
        cpp_time, _ = benchmark_fn(cpp_soft)
        print_speedup(py_time, cpp_time, "    Time")


def benchmark_group_spec(sizes: list):
    """Benchmark GroupSpec operations."""
    print("\n" + "=" * 60)
    print("Benchmark: GroupSpec")
    print("=" * 60)

    for tensor_shape, block_shape, group_shape in sizes:
        print(f"\nTensor: {tensor_shape}, Block: {block_shape}, Group: {group_shape}")

        # Create parameters
        py_param = torch.nn.Parameter(torch.randn(*tensor_shape))
        cpp_param = torch.nn.Parameter(py_param.data.clone())

        # Create BlockSpecs
        py_block = py_blocks.BlockSpec(py_param, block_shape, "py_block")
        cpp_block = cpp.BlockSpec(cpp_param, list(block_shape), "cpp_block")

        # Create GroupSpecs
        py_group = py_groups.GroupSpec(py_block, group_shape, "py_group")
        cpp_group = cpp.GroupSpec(cpp_block, list(group_shape), "cpp_group")

        # --- grouped_block_norms ---
        print("\n  grouped_block_norms:")

        def py_norms():
            return py_group.grouped_block_norms(None)

        def cpp_norms():
            return cpp_group.grouped_block_norms(None)

        py_result = py_norms()
        cpp_result = cpp_norms()
        check_close(py_result, cpp_result, "    Result")

        py_time, _ = benchmark_fn(py_norms)
        cpp_time, _ = benchmark_fn(cpp_norms)
        print_speedup(py_time, cpp_time, "    Time")

        # --- hard_threshold with sparsity ---
        print("\n  hard_threshold (sparsity=0.5):")

        test_data = torch.randn(*tensor_shape)
        py_param.data.copy_(test_data)
        cpp_param.data.copy_(test_data)
        py_group.hard_threshold(sparsity=0.5)
        cpp_group.hard_threshold(None, None, None, 0.5)
        check_close(py_param.data, cpp_param.data, "    Result")

        def py_hard():
            py_param.data.copy_(test_data)
            py_group.hard_threshold(sparsity=0.5)

        def cpp_hard():
            cpp_param.data.copy_(test_data)
            cpp_group.hard_threshold(None, None, None, 0.5)

        py_time, _ = benchmark_fn(py_hard)
        cpp_time, _ = benchmark_fn(cpp_hard)
        print_speedup(py_time, cpp_time, "    Time")


def main():
    print("=" * 60)
    print("SparseKit Benchmark: Python vs C++ Backend")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # User-requested benchmarks: (1024*n, 64) with k=32
    # For n = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
    print("\n" + "=" * 60)
    print("Benchmark: Requested Shapes (1024*n, 64) with block (32, 32)")
    print("=" * 60)

    n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    for n in n_values:
        rows = 1024 * n
        cols = 64
        block_size = 32

        # Skip if not divisible
        if rows % block_size != 0 or cols % block_size != 0:
            print(f"\nSkipping n={n}: ({rows}, {cols}) not divisible by {block_size}")
            continue

        print(f"\n{'='*60}")
        print(f"n={n}: Tensor ({rows}, {cols}), Block ({block_size}, {block_size})")
        print(f"  Blocks: {rows//block_size} x {cols//block_size} = {(rows//block_size) * (cols//block_size)}")
        print(f"{'='*60}")

        # Create parameters
        py_param = torch.nn.Parameter(torch.randn(rows, cols))
        cpp_param = torch.nn.Parameter(py_param.data.clone())

        # Create BlockSpecs
        py_block = py_blocks.BlockSpec(py_param, (block_size, block_size), "py_block")
        cpp_block = cpp.BlockSpec(cpp_param, [block_size, block_size], "cpp_block")

        # --- block_norms ---
        print("\n  block_norms:")

        def py_norms():
            return py_block.block_norms(py_param.data)

        def cpp_norms():
            return cpp_block.block_norms(cpp_param.data, 2)

        py_result = py_norms()
        cpp_result = cpp_norms()
        check_close(py_result, cpp_result, "    Result")

        py_time, _ = benchmark_fn(py_norms)
        cpp_time, _ = benchmark_fn(cpp_norms)
        print_speedup(py_time, cpp_time, "    Time")

        # --- hard_threshold ---
        print("\n  hard_threshold:")
        thresholds = torch.ones(py_block.grid_shape) * 1.0

        test_data = torch.randn(rows, cols)
        py_param.data.copy_(test_data)
        cpp_param.data.copy_(test_data)
        py_block.hard_threshold(thresholds)
        cpp_block.hard_threshold(thresholds, None)
        check_close(py_param.data, cpp_param.data, "    Result")

        def py_hard():
            py_param.data.copy_(test_data)
            py_block.hard_threshold(thresholds)

        def cpp_hard():
            cpp_param.data.copy_(test_data)
            cpp_block.hard_threshold(thresholds, None)

        py_time, _ = benchmark_fn(py_hard)
        cpp_time, _ = benchmark_fn(cpp_hard)
        print_speedup(py_time, cpp_time, "    Time")

        # --- soft_threshold (Euclidean) ---
        print("\n  soft_threshold (Euclidean):")
        thresholds = torch.ones(py_block.grid_shape) * 0.5

        test_data = torch.randn(rows, cols)
        py_param.data.copy_(test_data)
        cpp_param.data.copy_(test_data)
        py_block.soft_threshold(thresholds, conditioners=None)
        cpp_block.soft_threshold(thresholds, None, False, 20, 1e-20, 1e-8)
        check_close(py_param.data, cpp_param.data, "    Result")

        def py_soft():
            py_param.data.copy_(test_data)
            py_block.soft_threshold(thresholds, conditioners=None)

        def cpp_soft():
            cpp_param.data.copy_(test_data)
            cpp_block.soft_threshold(thresholds, None, False, 20, 1e-20, 1e-8)

        py_time, _ = benchmark_fn(py_soft)
        cpp_time, _ = benchmark_fn(cpp_soft)
        print_speedup(py_time, cpp_time, "    Time")

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
