# SparseKit

**SparseKit** is the reference implementation of **S3 (Structured Sparsity Specification)**,
a unified framework for expressing and pruning structured sparse neural networks.

## Library Overview

```
sparsekit/
├── view.py          # View         — zero-copy strided parameter wrapper (torch.as_strided)
├── block.py         # BlockSpec / BlockCoupling   — atomic pruning unit (block)
├── scope.py         # ScopeSpec / ScopeCoupling   — decision scope
├── builder.py       # SparsityBuilder fluent API
├── linalg.py        # Utility solvers (proximal, thresholds)
├── tensor_ops.py    # kth_largest, layout helpers
├── kernels.py       # Triton kernels (auto-dispatched for large K/k)
├── viz.py           # draw_layout() — visualize sparsity patterns
├── pruners/
│   ├── obs.py       # StructuredOBS — S-OBS with per-row Schur updates
│   ├── sparsegpt.py # SparseGPT column-sequential pruning
│   └── obd.py       # OBD and magnitude pruning
└── training/
    ├── data.py      # Calibration data loaders (C4)
    └── hooks.py     # ModuleInputCatcher, transfer_to_device
```

**Terminology:**
- **Block** — atomic pruning unit: the smallest set of weights pruned or kept together.
- **Scope** — decision scope: a set of blocks that compete; the pruning budget is enforced per scope.

## Quick Example

```python
import torch
from torch.nn import Parameter
from sparsekit import BlockSpec, ScopeSpec, StructuredOBS

M, K = 2560, 9728
W = Parameter(torch.randn(M, K, device="cuda"))
X = torch.randn(1024, K, device="cuda")          # calibration inputs

# Express 2:4 sparsity: scalar blocks, scopes of 4
block = BlockSpec(W, shape=(1, 1))
scope = ScopeSpec(block, shape=(1, 4))

# Prune with Structured OBS
hessian = (X.T @ X) / X.shape[0]
obs = StructuredOBS(scope, hessian)
obs.prune_true_obs(nnz=2)                     # keep 2 of 4, in-place
```

Any of the four experimental patterns replaces the two `BlockSpec/ScopeSpec`
lines above; the `StructuredOBS` call is identical.

## Sparsity Patterns

| Pattern | Block shape | Scope shape | Description |
|---|---|---|---|
| 2:4 | `(1, 1)` | `(1, 4)` | Keep 2 of 4 contiguous columns |
| 4:8 | `(1, 2)` | `(1, 4)` | Keep 2 of 4 column-pairs |
| Coupled 2:4 | `(1, 1, 1, 2)` | `(1, 1, 4, 1)` | Pair columns 8 apart via `View` |
| 16-col block | `(1, 1, 16)` | `(1, 2, 1)` | 16-col blocks, 8-row coupling |

## Reproducing Paper Results

**Table 1** (single-layer, 4 patterns):
```bash
python scripts/structured_obs.py --pattern 24         --ng 64   # 2:4
python scripts/structured_obs.py --pattern 48         --ng 64   # 4:8
python scripts/structured_obs.py --pattern coupled24  --ng 64   # Coupled 2:4
python scripts/structured_obs.py --pattern block16    --ng 64   # 16-col block, 8-row coupled
```

**Table 2 + Figures** (end-to-end LLM pruning):
```bash
# SparseGPT baseline
python scripts/prune_gpt.py --method sparsegpt_24 --model Qwen/Qwen3-1.7B

# S-OBS (True OBS)
python scripts/prune_gpt.py --method true_obs_24 --model Qwen/Qwen3-1.7B --ng 64
```

**Plots** (from saved CSVs):
```bash
python scripts/plot_results.py experiments/results --model Qwen3-1.7B
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.4
- Triton >= 3.0
- CUDA GPU

Additional for LLM experiments (`prune_gpt.py`):
- `transformers`, `datasets`, `lm_eval`, `pandas`
