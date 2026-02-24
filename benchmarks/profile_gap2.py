"""
Profile the gap between Python and C++ for large tensors.
"""

import time
import torch
from torch.nn import Parameter

from sparsekit import blocks as py_blocks
import sparsekit._C as cpp

torch.set_num_threads(1)

n = 4096
rows = 1024 * n
cols = 64
block_size = 32

print(f"Tensor shape: ({rows}, {cols})")
print(f"Block grid: ({rows // block_size}, {cols // block_size})")
print()

data = torch.randn(rows, cols)
interleaved_shape = (rows // block_size, block_size, cols // block_size, block_size)
reduction_dims = (1, 3)

# ============================================================================
# Raw tensor operations - no wrapper overhead
# ============================================================================
print("=" * 70)
print("RAW TENSOR OPERATIONS (no wrapper)")
print("=" * 70)

t = data.clone()

# Python's approach: sum(t**p).pow(1/p)
print("\n--- Python _block_lp_fn style ---")
start = time.perf_counter()
for _ in range(10):
    v = t.view(*interleaved_shape)
    norms = torch.sum(v**2, dim=reduction_dims).pow(0.5)
t_py = (time.perf_counter() - start) / 10 * 1000
print(f"  torch.sum(v**2, dim).pow(0.5): {t_py:.3f} ms")

# C++ approach: torch.norm
print("\n--- C++ block_lp_norm style ---")
start = time.perf_counter()
for _ in range(10):
    v = t.view(*interleaved_shape)
    norms = torch.norm(v, p=2, dim=reduction_dims)
t_cpp = (time.perf_counter() - start) / 10 * 1000
print(f"  torch.norm(v, p=2, dim):       {t_cpp:.3f} ms")

# Alternative: linalg.vector_norm
print("\n--- torch.linalg.vector_norm ---")
start = time.perf_counter()
for _ in range(10):
    v = t.view(*interleaved_shape)
    norms = torch.linalg.vector_norm(v, dim=reduction_dims)
t_linalg = (time.perf_counter() - start) / 10 * 1000
print(f"  torch.linalg.vector_norm:      {t_linalg:.3f} ms")

# ============================================================================
# Full wrapper comparison
# ============================================================================
print("\n" + "=" * 70)
print("FULL WRAPPER COMPARISON")
print("=" * 70)

py_param = Parameter(data.clone())
cpp_param = Parameter(data.clone())

py_block = py_blocks.BlockSpec(py_param, (block_size, block_size), "py")
cpp_block = cpp.BlockSpec(cpp_param, [block_size, block_size], "cpp")

# Python
start = time.perf_counter()
for _ in range(10):
    result = py_block.block_norms(py_param.data)
t_py_wrapper = (time.perf_counter() - start) / 10 * 1000
print(f"\n  Python block_norms: {t_py_wrapper:.3f} ms")

# C++
start = time.perf_counter()
for _ in range(10):
    result = cpp_block.block_norms(cpp_param.data, 2)
t_cpp_wrapper = (time.perf_counter() - start) / 10 * 1000
print(f"  C++ block_norms:    {t_cpp_wrapper:.3f} ms")

# ============================================================================
# Breakdown of Python wrapper
# ============================================================================
print("\n" + "=" * 70)
print("PYTHON WRAPPER BREAKDOWN")
print("=" * 70)

# _raw_block_view
start = time.perf_counter()
for _ in range(10):
    v = py_block._raw_block_view(py_param.data, reorder=False)
t1 = (time.perf_counter() - start) / 10 * 1000
print(f"\n  _raw_block_view(merge=False): {t1:.3f} ms  shape={v.shape}")

# _block_lp_fn
start = time.perf_counter()
for _ in range(10):
    v = py_block._raw_block_view(py_param.data, reorder=False)
    norms = py_block._block_lp_fn(v, p=2)
t2 = (time.perf_counter() - start) / 10 * 1000
print(f"  _raw_block_view + _block_lp_fn: {t2:.3f} ms")

# block_reduce
start = time.perf_counter()
for _ in range(10):
    result = py_block.block_reduce(py_param.data, lambda t: py_block._block_lp_fn(t, p=2))
t3 = (time.perf_counter() - start) / 10 * 1000
print(f"  block_reduce (full):           {t3:.3f} ms")

# ============================================================================
# Breakdown of C++ wrapper
# ============================================================================
print("\n" + "=" * 70)
print("C++ WRAPPER BREAKDOWN")
print("=" * 70)

# raw_block_view
start = time.perf_counter()
for _ in range(10):
    v = cpp_block.raw_block_view(cpp_param.data, False)
t1 = (time.perf_counter() - start) / 10 * 1000
print(f"\n  raw_block_view(merge=False): {t1:.3f} ms  shape={tuple(v.shape)}")

# ops.block_lp_norm directly
start = time.perf_counter()
for _ in range(10):
    v = cpp_block.raw_block_view(cpp_param.data, False)
    norms = cpp.ops.block_lp_norm(v, 2.0, [1, 3], False)
t2 = (time.perf_counter() - start) / 10 * 1000
print(f"  raw_block_view + block_lp_norm: {t2:.3f} ms")

# block_norms
start = time.perf_counter()
for _ in range(10):
    result = cpp_block.block_norms(cpp_param.data, 2)
t3 = (time.perf_counter() - start) / 10 * 1000
print(f"  block_norms (full):            {t3:.3f} ms")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n  Raw torch.sum(v**2).pow(0.5): {t_py:.3f} ms")
print(f"  Raw torch.norm(v, p=2, dim):  {t_cpp:.3f} ms")
print(f"  Python wrapper block_norms:   {t_py_wrapper:.3f} ms")
print(f"  C++ wrapper block_norms:      {t_cpp_wrapper:.3f} ms")
print(f"\n  Gap Python - C++:             {t_py_wrapper - t_cpp_wrapper:.3f} ms")
