# SparseKit Benchmarks

Benchmarks comparing the pure Python implementation against the C++/LibTorch backend.

## Running Benchmarks

```bash
# Activate environment
mamba activate skit

# Run benchmark
python benchmarks/benchmark.py
```

## Results Summary

All operations produce **identical results** (within floating point tolerance ~1e-6).

### Speedup by Operation

| Operation | Small Tensors | Large Tensors |
|-----------|---------------|---------------|
| `interleave_unsqueeze` | 1.0-1.2x | ~1.0x |
| `merge_odd_dims` | 1.0-1.1x | ~1.0x |
| `kth_largest` | ~1.0x | ~1.0x |
| `block_norms` | 1.2-1.5x | ~1.0x |
| `block_view` | 1.4-1.5x | 1.4x |
| `hard_threshold` | **1.8-2.0x** | 1.1-1.2x |
| `soft_threshold` | **1.4-1.7x** | 1.1-1.4x |
| `grouped_block_norms` | **2.0-3.0x** | ~1.0x |
| `group hard_threshold` | **1.7-2.0x** | 1.1x |

### Key Observations

1. **Correctness**: All results match between Python and C++ (max diff < 1e-5)

2. **Best Speedups** (1.5-3x):
   - `grouped_block_norms` on small tensors (2-3x)
   - `hard_threshold` operations (1.8-2x)
   - `soft_threshold` operations (1.4-1.7x)

3. **Minimal Difference** (~1.0x):
   - Pure tensor operations (`kth_largest`, `merge_odd_dims`)
   - Large tensor operations (dominated by PyTorch compute)

4. **Why C++ Helps**:
   - Reduced Python object creation/destruction overhead
   - Fewer Python function calls in multi-step operations
   - Operations that compose multiple PyTorch calls benefit most

## Sample Output

```
============================================================
Benchmark: BlockSpec
============================================================

Tensor: (128, 128), Block: (8, 8)

  block_norms:
      Result: MATCH (max_diff=9.54e-07)
      Time: Python=0.044ms, C++=0.036ms, Speedup=1.21x

  hard_threshold:
      Result: MATCH (max_diff=0.00e+00)
      Time: Python=0.087ms, C++=0.043ms, Speedup=2.04x

  soft_threshold (Euclidean):
      Result: MATCH (max_diff=2.38e-07)
      Time: Python=0.069ms, C++=0.045ms, Speedup=1.53x
```

## What's Tested

### Utility Functions
- `interleave_unsqueeze` - Insert singleton dims after each existing dim
- `merge_odd_dims` - Collapse odd-indexed dims into trailing dim
- `kth_largest` - Find k-th largest element along dimension

### BlockSpec Operations
- `block_norms` - Compute Lp norm for each block
- `block_view` - Reshape tensor to block-structured view
- `hard_threshold` - Zero blocks below threshold
- `soft_threshold` - Shrink block norms (Euclidean proximal)

### GroupSpec Operations
- `grouped_block_norms` - Compute norms grouped across blocks
- `hard_threshold` - Group-level thresholding with sparsity target
