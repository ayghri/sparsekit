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

# ============================================================================
# Profile Python block_norms step by step
# ============================================================================
print("=" * 70)
print("PYTHON block_norms breakdown")
print("=" * 70)

py_param = Parameter(data.clone())
py_block = py_blocks.BlockSpec(py_param, (block_size, block_size), "py")

# Read Python source to understand what it does
import inspect
print("\nPython block_view source:")
print(inspect.getsource(py_block.block_view))

print("\nPython block_norms source:")
print(inspect.getsource(py_block.block_norms))

# Time individual steps
print("\n--- Timing breakdown ---")

# Step 1: raw_block_view equivalent
start = time.perf_counter()
for _ in range(10):
    interleaved = py_block._interleave_unsqueeze(py_param.data)
t1 = (time.perf_counter() - start) / 10 * 1000
print(f"  _interleave_unsqueeze: {t1:.3f} ms")

start = time.perf_counter()
for _ in range(10):
    merged = py_block._merge_odd_dims(interleaved)
t2 = (time.perf_counter() - start) / 10 * 1000
print(f"  _merge_odd_dims:       {t2:.3f} ms")

start = time.perf_counter()
for _ in range(10):
    bv = py_block.block_view(py_param.data)
t3 = (time.perf_counter() - start) / 10 * 1000
print(f"  block_view total:      {t3:.3f} ms  shape={bv.shape}")

start = time.perf_counter()
for _ in range(10):
    norms = torch.linalg.vector_norm(bv, dim=-1)
t4 = (time.perf_counter() - start) / 10 * 1000
print(f"  vector_norm:           {t4:.3f} ms")

start = time.perf_counter()
for _ in range(10):
    result = py_block.block_norms(py_param.data)
t_total = (time.perf_counter() - start) / 10 * 1000
print(f"  TOTAL block_norms:     {t_total:.3f} ms")

# ============================================================================
# Profile C++ block_norms step by step
# ============================================================================
print("\n" + "=" * 70)
print("C++ block_norms breakdown")
print("=" * 70)

cpp_param = Parameter(data.clone())
cpp_block = cpp.BlockSpec(cpp_param, [block_size, block_size], "cpp")

# Step 1: raw_block_view (no merge)
start = time.perf_counter()
for _ in range(10):
    rbv = cpp_block.raw_block_view(cpp_param.data, False)
t1 = (time.perf_counter() - start) / 10 * 1000
print(f"  raw_block_view:        {t1:.3f} ms  shape={tuple(rbv.shape)}")

# Step 2: block_norms
start = time.perf_counter()
for _ in range(10):
    result = cpp_block.block_norms(cpp_param.data, 2)
t_total = (time.perf_counter() - start) / 10 * 1000
print(f"  TOTAL block_norms:     {t_total:.3f} ms")

# ============================================================================
# Compare raw tensor operations
# ============================================================================
print("\n" + "=" * 70)
print("RAW TENSOR OPERATIONS (no wrapper overhead)")
print("=" * 70)

t = data.clone()
interleaved_shape = (rows // block_size, block_size, cols // block_size, block_size)

# Method 1: Python's approach (permute + contiguous + reshape + vector_norm)
print("\n--- Python-style approach ---")
start = time.perf_counter()
for _ in range(10):
    v = t.view(*interleaved_shape)
    v = v.permute(0, 2, 1, 3)
    v = v.contiguous()  # This is the expensive part
    v = v.view(rows // block_size, cols // block_size, -1)
    norms = torch.linalg.vector_norm(v, dim=-1)
t_py_style = (time.perf_counter() - start) / 10 * 1000
print(f"  Total: {t_py_style:.3f} ms")

# Method 2: C++-style approach (view + norm over dims)
print("\n--- C++-style approach ---")
start = time.perf_counter()
for _ in range(10):
    v = t.view(*interleaved_shape)
    norms = torch.norm(v, p=2, dim=(1, 3))
t_cpp_style = (time.perf_counter() - start) / 10 * 1000
print(f"  Total: {t_cpp_style:.3f} ms")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Python wrapper block_norms: {t_total:.3f} ms")
print(f"  C++ wrapper block_norms:    {t_total:.3f} ms")
print(f"  Raw Python-style ops:       {t_py_style:.3f} ms")
print(f"  Raw C++-style ops:          {t_cpp_style:.3f} ms")
