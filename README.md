# SparseKit

**SparseKit** is the reference implementation of **S³ (Structured Sparsity Specification)**,
a unified framework for expressing and pruning structured sparse neural networks.

## Browsing the Documentation

Full HTML documentation is bundled in `docs/_build/html/`.
Serve it locally with the included helper script:

```bash
python serve_docs.py
```

Then open **http://localhost:8000** in your browser.

The documentation covers:

- **Quickstart** — install, construct a `GroupSpec`, prune with `StructuredOBS`
- **Concepts** — View, Block, Group, Coupling explained with examples
- **API Reference** — full docstrings for every public class and function
- **Results** — single-layer and end-to-end benchmark tables

## Library Overview

```
sparsekit/
├── view.py        # View  — zero-copy strided parameter wrapper (torch.as_strided)
├── block.py       # BlockSpec / BlockCoupling — atomic sparsity unit
├── group.py       # GroupSpec / GroupCoupling — sparsity decision scope
├── linalg.py      # Utility solvers (LSQR, proximal, thresholds)
├── utils.py       # kth_largest, layout helpers
├── kernels.py     # Triton kernels (auto-dispatched for large K/k)
├── builder.py     # SparsityBuilder fluent API
├── viz.py         # draw_layout() — visualize sparsity patterns
└── pruners/
    ├── obs.py     # StructuredOBS — S-OBS with per-row Schur updates
    ├── quant.py   # quantize_obs, mxfp4_quantize
    └── nvquant.py # nvfp4_quantize, quantize_nvfp4_obs
```

## Quick Example

```python
import torch
from torch.nn import Parameter
from sparsekit import View, BlockSpec, GroupSpec, StructuredOBS

M, K = 2560, 9728
W = Parameter(torch.randn(M, K, device="cuda"))
X = torch.randn(1024, K, device="cuda")          # calibration inputs

# Express 2:4 sparsity
v     = View.from_existing(W)
block = BlockSpec(v, shape=(1, 1))
group = GroupSpec(block, shape=(1, 4))

# Prune with Structured OBS
H   = (X.T @ X) / X.shape[0]
obs = StructuredOBS(group, H)
obs.prune_true_obs(num_nz=2)                     # keep 2 of 4, in-place
```

Any of the four experimental patterns replaces the three `View/BlockSpec/GroupSpec`
lines above; the `StructuredOBS` call is identical.

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.8
- Triton 3.4.0
- CUDA (for Triton kernels; CPU fallback available)
