"""
Detailed profiling of Python vs C++ backend for n=1024.
"""

import time
import torch
from torch.nn import Parameter

from sparsekit import blocks as py_blocks
import sparsekit._C as cpp

torch.set_num_threads(1)  # Single thread for consistent profiling

n = 1024
rows = 1024 * n  # 1048576
cols = 64
block_size = 32

print(f"Tensor shape: ({rows}, {cols})")
print(f"Block shape: ({block_size}, {block_size})")
print(f"Block grid: ({rows // block_size}, {cols // block_size})")
print()

# Create test data
data = torch.randn(rows, cols)

# ============================================================================
# Profile block_norms
# ============================================================================
print("=" * 70)
print("PROFILING: block_norms")
print("=" * 70)

# Python implementation - step by step
print("\n--- Python Implementation Breakdown ---")
py_param = Parameter(data.clone())
py_block = py_blocks.BlockSpec(py_param, (block_size, block_size), "py_block")

# Step 1: block_view
start = time.perf_counter()
for _ in range(10):
    bv = py_block.block_view(py_param.data)
t1 = (time.perf_counter() - start) / 10 * 1000
print(f"  block_view:           {t1:.3f} ms   shape={bv.shape}")

# Step 2: norm computation
start = time.perf_counter()
for _ in range(10):
    norms = torch.linalg.vector_norm(bv, dim=-1)
t2 = (time.perf_counter() - start) / 10 * 1000
print(f"  vector_norm:          {t2:.3f} ms   shape={norms.shape}")

# Full call
start = time.perf_counter()
for _ in range(10):
    result = py_block.block_norms(py_param.data)
t_total = (time.perf_counter() - start) / 10 * 1000
print(f"  TOTAL block_norms:    {t_total:.3f} ms")

# C++ implementation - step by step
print("\n--- C++ Implementation Breakdown ---")
cpp_param = Parameter(data.clone())
cpp_block = cpp.BlockSpec(cpp_param, [block_size, block_size], "cpp_block")

# Step 1: raw_block_view (without merge)
start = time.perf_counter()
for _ in range(10):
    rbv = cpp_block.raw_block_view(cpp_param.data, False)
t1 = (time.perf_counter() - start) / 10 * 1000
print(f"  raw_block_view:       {t1:.3f} ms   shape={tuple(rbv.shape)}")

# Step 2: raw_block_view (with merge)
start = time.perf_counter()
for _ in range(10):
    rbv_merged = cpp_block.raw_block_view(cpp_param.data, True)
t2 = (time.perf_counter() - start) / 10 * 1000
print(f"  raw_block_view+merge: {t2:.3f} ms   shape={tuple(rbv_merged.shape)}")

# Step 3: block_view (with squeeze)
start = time.perf_counter()
for _ in range(10):
    bv = cpp_block.block_view(cpp_param.data, True)
t3 = (time.perf_counter() - start) / 10 * 1000
print(f"  block_view:           {t3:.3f} ms   shape={tuple(bv.shape)}")

# Step 4: Full block_norms
start = time.perf_counter()
for _ in range(10):
    result = cpp_block.block_norms(cpp_param.data, 2)
t_total = (time.perf_counter() - start) / 10 * 1000
print(f"  TOTAL block_norms:    {t_total:.3f} ms   shape={tuple(result.shape)}")

# ============================================================================
# Compare view operations directly
# ============================================================================
print("\n" + "=" * 70)
print("COMPARING VIEW OPERATIONS")
print("=" * 70)

# Python's view approach
print("\n--- Python view approach ---")
t = data.clone()
start = time.perf_counter()
for _ in range(10):
    # interleave_unsqueeze equivalent
    for i in range(t.dim() - 1, -1, -1):
        t_view = t.unsqueeze(i + 1)
    # reshape to block grid
    grid_shape = (rows // block_size, block_size, cols // block_size, block_size)
    t_view = t.view(*grid_shape)
t1 = (time.perf_counter() - start) / 10 * 1000
print(f"  Simple view+reshape:  {t1:.3f} ms")

# Python's actual block_view
start = time.perf_counter()
for _ in range(10):
    bv = py_block.block_view(py_param.data)
t2 = (time.perf_counter() - start) / 10 * 1000
print(f"  py_block.block_view:  {t2:.3f} ms")

# C++ block_view
start = time.perf_counter()
for _ in range(10):
    bv = cpp_block.block_view(cpp_param.data, True)
t3 = (time.perf_counter() - start) / 10 * 1000
print(f"  cpp_block.block_view: {t3:.3f} ms")

# Direct LibTorch operations (through Python)
print("\n--- Direct tensor ops comparison ---")
t = data.clone()

# Simple view
start = time.perf_counter()
for _ in range(10):
    v = t.view(rows // block_size, block_size, cols // block_size, block_size)
t1 = (time.perf_counter() - start) / 10 * 1000
print(f"  t.view(...):          {t1:.3f} ms")

# View + permute
start = time.perf_counter()
for _ in range(10):
    v = t.view(rows // block_size, block_size, cols // block_size, block_size)
    v = v.permute(0, 2, 1, 3)
t2 = (time.perf_counter() - start) / 10 * 1000
print(f"  view + permute:       {t2:.3f} ms")

# View + permute + reshape
start = time.perf_counter()
for _ in range(10):
    v = t.view(rows // block_size, block_size, cols // block_size, block_size)
    v = v.permute(0, 2, 1, 3)
    v = v.reshape(rows // block_size, cols // block_size, -1)
t3 = (time.perf_counter() - start) / 10 * 1000
print(f"  view+permute+reshape: {t3:.3f} ms")

# Full norm computation
start = time.perf_counter()
for _ in range(10):
    v = t.view(rows // block_size, block_size, cols // block_size, block_size)
    v = v.permute(0, 2, 1, 3)
    v = v.reshape(rows // block_size, cols // block_size, -1)
    norms = torch.linalg.vector_norm(v, dim=-1)
t4 = (time.perf_counter() - start) / 10 * 1000
print(f"  full block_norms:     {t4:.3f} ms")

# ============================================================================
# Profile C++ utils functions
# ============================================================================
print("\n" + "=" * 70)
print("PROFILING C++ UTILS FUNCTIONS")
print("=" * 70)

t = data.clone()

# interleave_unsqueeze
start = time.perf_counter()
for _ in range(10):
    iu = cpp.utils.interleave_unsqueeze(t)
t1 = (time.perf_counter() - start) / 10 * 1000
print(f"  interleave_unsqueeze: {t1:.3f} ms   shape={tuple(iu.shape)}")

# merge_odd_dims on interleaved
start = time.perf_counter()
for _ in range(10):
    iu = cpp.utils.interleave_unsqueeze(t)
    merged = cpp.utils.merge_odd_dims(iu)
t2 = (time.perf_counter() - start) / 10 * 1000
print(f"  interleave+merge:     {t2:.3f} ms   shape={tuple(merged.shape)}")

# Compare with direct approach
print("\n--- Direct vs C++ utils ---")
start = time.perf_counter()
for _ in range(10):
    # Direct: just view
    v = t.view(rows // block_size, block_size, cols // block_size, block_size)
t1 = (time.perf_counter() - start) / 10 * 1000
print(f"  Direct view:          {t1:.3f} ms")

start = time.perf_counter()
for _ in range(10):
    # C++ utils path
    iu = cpp.utils.interleave_unsqueeze(t)
t2 = (time.perf_counter() - start) / 10 * 1000
print(f"  C++ interleave:       {t2:.3f} ms")
