"""
Profile the block_lp_norm operation specifically.
"""

import time
import torch
from torch.nn import Parameter

import sparsekit._C as cpp

torch.set_num_threads(1)

n = 1024
rows = 1024 * n  # 1048576
cols = 64
block_size = 32

print(f"Tensor shape: ({rows}, {cols})")
print(f"Block shape: ({block_size}, {block_size})")
print(f"Block grid: ({rows // block_size}, {cols // block_size})")
print()

data = torch.randn(rows, cols)

# Shape after raw_block_view (no merge): (32768, 32, 2, 32)
# reduction_dims = [1, 3]
interleaved_shape = (rows // block_size, block_size, cols // block_size, block_size)
reduction_dims = [1, 3]

print("=" * 70)
print("PROFILING: Different norm computation strategies")
print("=" * 70)

# Get interleaved view (no contiguous copy needed)
t = data.view(*interleaved_shape)
print(f"\nInterleaved shape: {t.shape}")
print(f"Is contiguous: {t.is_contiguous()}")

# Strategy 1: C++ block_lp_norm approach (iterative sum)
print("\n--- Strategy 1: Iterative sum (C++ approach) ---")
start = time.perf_counter()
for _ in range(10):
    powered = torch.pow(torch.abs(t), 2)
    summed = powered
    for dim in reversed(reduction_dims):
        summed = summed.sum(dim, keepdim=False)
    result1 = torch.pow(summed, 0.5)
t1 = (time.perf_counter() - start) / 10 * 1000
print(f"  Time: {t1:.3f} ms   shape={result1.shape}")

# Strategy 2: Single flatten + vector_norm (Python approach)
print("\n--- Strategy 2: Permute + contiguous + flatten + vector_norm ---")
start = time.perf_counter()
for _ in range(10):
    # Permute to group block dims together
    tp = t.permute(0, 2, 1, 3)  # (32768, 2, 32, 32)
    tc = tp.contiguous()        # EXPENSIVE: copy
    tf = tc.view(rows // block_size, cols // block_size, -1)  # (32768, 2, 1024)
    result2 = torch.linalg.vector_norm(tf, dim=-1)
t2 = (time.perf_counter() - start) / 10 * 1000
print(f"  Time: {t2:.3f} ms   shape={result2.shape}")

# Strategy 3: Use torch.norm with multiple dims directly
print("\n--- Strategy 3: torch.norm with tuple of dims ---")
start = time.perf_counter()
for _ in range(10):
    result3 = torch.norm(t, p=2, dim=(1, 3))
t3 = (time.perf_counter() - start) / 10 * 1000
print(f"  Time: {t3:.3f} ms   shape={result3.shape}")

# Strategy 4: Flatten reduction dims first, then single norm
print("\n--- Strategy 4: Reshape to merge reduction dims, single norm ---")
start = time.perf_counter()
for _ in range(10):
    # (32768, 32, 2, 32) -> flatten the block dims
    t_flat = t.reshape(rows // block_size, cols // block_size, -1)  # needs contiguous
t4_reshape = (time.perf_counter() - start) / 10 * 1000
print(f"  Reshape time: {t4_reshape:.3f} ms")

# Check if reshape alone needs copy
print(f"\n  After t.reshape(...): is_contiguous={t_flat.is_contiguous()}")

start = time.perf_counter()
for _ in range(10):
    # Permute first to make reshape cheap
    tp = t.permute(0, 2, 1, 3)
    # Check if this permuted view can be reshaped without copy
t_permute = (time.perf_counter() - start) / 10 * 1000
print(f"  Permute time: {t_permute:.3f} ms, is_contiguous={tp.is_contiguous()}")

# Strategy 5: torch.linalg.vector_norm with dim tuple
print("\n--- Strategy 5: torch.linalg.vector_norm with dim tuple ---")
start = time.perf_counter()
for _ in range(10):
    result5 = torch.linalg.vector_norm(t, dim=(1, 3))
t5 = (time.perf_counter() - start) / 10 * 1000
print(f"  Time: {t5:.3f} ms   shape={result5.shape}")

# Verify all results are the same
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)
print(f"Strategy 1 == Strategy 3: {torch.allclose(result1, result3)}")
print(f"Strategy 2 == Strategy 3: {torch.allclose(result2, result3)}")
print(f"Strategy 3 == Strategy 5: {torch.allclose(result3, result5)}")

# Now profile what C++ is actually doing
print("\n" + "=" * 70)
print("C++ block_lp_norm via Python binding")
print("=" * 70)

start = time.perf_counter()
for _ in range(10):
    result_cpp = cpp.ops.block_lp_norm(t, 2.0, reduction_dims, False)
t_cpp = (time.perf_counter() - start) / 10 * 1000
print(f"  cpp.ops.block_lp_norm: {t_cpp:.3f} ms   shape={tuple(result_cpp.shape)}")
print(f"  Matches torch.norm:    {torch.allclose(result_cpp, result3)}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Iterative sum (C++ impl):     {t1:.3f} ms")
print(f"  Permute+contiguous+flatten:   {t2:.3f} ms")
print(f"  torch.norm(dim=(1,3)):        {t3:.3f} ms  <-- FASTEST")
print(f"  torch.linalg.vector_norm:     {t5:.3f} ms")
print(f"  C++ binding block_lp_norm:    {t_cpp:.3f} ms")
