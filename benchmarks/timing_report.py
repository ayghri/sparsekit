"""
Benchmark timing report for SparseKit Python vs C++ backend.
Reports actual timing values in milliseconds.
"""

import time
import torch
import numpy as np
from torch.nn import Parameter

from sparsekit import blocks as py_blocks
import sparsekit._C as cpp


def benchmark_fn(fn, warmup=3, runs=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times), np.std(times)


BLOCK_SIZE = 4

def main():
    n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    print("=" * 90)
    print("BLOCK_NORMS TIMING (ms)")
    print("=" * 90)
    print(f"{'n':>6} | {'Shape':>16} | {'Blocks':>10} | {'Python (ms)':>16} | {'C++ (ms)':>16}")
    print("-" * 90)

    for n in n_values:
        rows = 1024 * n
        cols = 64
        block_size = BLOCK_SIZE

        py_param = Parameter(torch.randn(rows, cols))
        cpp_param = Parameter(py_param.data.clone())

        py_block = py_blocks.BlockSpec(py_param, (block_size, block_size), "py_block")
        cpp_block = cpp.BlockSpec(cpp_param, [block_size, block_size], "cpp_block")

        def py_norms():
            return py_block.block_norms(py_param.data)

        def cpp_norms():
            return cpp_block.block_norms(cpp_param.data, 2)

        py_time, py_std = benchmark_fn(py_norms)
        cpp_time, cpp_std = benchmark_fn(cpp_norms)

        num_blocks = (rows // block_size) * (cols // block_size)
        shape_str = f"{rows}x{cols}"
        print(f"{n:>6} | {shape_str:>16} | {num_blocks:>10} | {py_time:>10.3f} +/-{py_std:>4.2f} | {cpp_time:>10.3f} +/-{cpp_std:>4.2f}")

    print()
    print("=" * 90)
    print("HARD_THRESHOLD TIMING (ms)")
    print("=" * 90)
    print(f"{'n':>6} | {'Shape':>16} | {'Blocks':>10} | {'Python (ms)':>16} | {'C++ (ms)':>16}")
    print("-" * 90)

    for n in n_values:
        rows = 1024 * n
        cols = 64
        block_size = BLOCK_SIZE

        py_param = Parameter(torch.randn(rows, cols))
        cpp_param = Parameter(py_param.data.clone())

        py_block = py_blocks.BlockSpec(py_param, (block_size, block_size), "py_block")
        cpp_block = cpp.BlockSpec(cpp_param, [block_size, block_size], "cpp_block")

        thresholds = torch.ones(py_block.grid_shape) * 1.0
        test_data = torch.randn(rows, cols)

        def py_hard():
            py_param.data.copy_(test_data)
            py_block.hard_threshold(thresholds)

        def cpp_hard():
            cpp_param.data.copy_(test_data)
            cpp_block.hard_threshold(thresholds, None)

        py_time, py_std = benchmark_fn(py_hard)
        cpp_time, cpp_std = benchmark_fn(cpp_hard)

        num_blocks = (rows // block_size) * (cols // block_size)
        shape_str = f"{rows}x{cols}"
        print(f"{n:>6} | {shape_str:>16} | {num_blocks:>10} | {py_time:>10.3f} +/-{py_std:>4.2f} | {cpp_time:>10.3f} +/-{cpp_std:>4.2f}")

    print()
    print("=" * 90)
    print("SOFT_THRESHOLD (Euclidean) TIMING (ms)")
    print("=" * 90)
    print(f"{'n':>6} | {'Shape':>16} | {'Blocks':>10} | {'Python (ms)':>16} | {'C++ (ms)':>16}")
    print("-" * 90)

    for n in n_values:
        rows = 1024 * n
        cols = 64
        block_size = BLOCK_SIZE

        py_param = Parameter(torch.randn(rows, cols))
        cpp_param = Parameter(py_param.data.clone())

        py_block = py_blocks.BlockSpec(py_param, (block_size, block_size), "py_block")
        cpp_block = cpp.BlockSpec(cpp_param, [block_size, block_size], "cpp_block")

        thresholds = torch.ones(py_block.grid_shape) * 0.5
        test_data = torch.randn(rows, cols)

        def py_soft():
            py_param.data.copy_(test_data)
            py_block.soft_threshold(thresholds, conditioners=None)

        def cpp_soft():
            cpp_param.data.copy_(test_data)
            cpp_block.soft_threshold(thresholds, None, False, 20, 1e-20, 1e-8)

        py_time, py_std = benchmark_fn(py_soft)
        cpp_time, cpp_std = benchmark_fn(cpp_soft)

        num_blocks = (rows // block_size) * (cols // block_size)
        shape_str = f"{rows}x{cols}"
        print(f"{n:>6} | {shape_str:>16} | {num_blocks:>10} | {py_time:>10.3f} +/-{py_std:>4.2f} | {cpp_time:>10.3f} +/-{cpp_std:>4.2f}")

    print()
    print("=" * 90)
    print("SOFT_THRESHOLD (Adam with diagonal conditioner) TIMING (ms)")
    print("=" * 90)
    print(f"{'n':>6} | {'Shape':>16} | {'Blocks':>10} | {'Python (ms)':>16} | {'C++ (ms)':>16}")
    print("-" * 90)

    for n in n_values:
        rows = 1024 * n
        cols = 64
        block_size = BLOCK_SIZE

        py_param = Parameter(torch.randn(rows, cols))
        cpp_param = Parameter(py_param.data.clone())

        py_block = py_blocks.BlockSpec(py_param, (block_size, block_size), "py_block")
        cpp_block = cpp.BlockSpec(cpp_param, [block_size, block_size], "cpp_block")

        thresholds = torch.ones(py_block.grid_shape) * 0.5
        test_data = torch.randn(rows, cols)
        # Diagonal conditioner (simulating Adam's sqrt(v) + eps)
        conditioner = torch.rand(rows, cols).abs() + 0.1

        def py_soft_adam():
            py_param.data.copy_(test_data)
            py_block.soft_threshold(thresholds, conditioners=conditioner)

        def cpp_soft_adam():
            cpp_param.data.copy_(test_data)
            cpp_block.soft_threshold(thresholds, conditioner, False, 20, 1e-20, 1e-8)

        py_time, py_std = benchmark_fn(py_soft_adam)
        cpp_time, cpp_std = benchmark_fn(cpp_soft_adam)

        num_blocks = (rows // block_size) * (cols // block_size)
        shape_str = f"{rows}x{cols}"
        print(f"{n:>6} | {shape_str:>16} | {num_blocks:>10} | {py_time:>10.3f} +/-{py_std:>4.2f} | {cpp_time:>10.3f} +/-{cpp_std:>4.2f}")


if __name__ == "__main__":
    main()
