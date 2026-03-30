"""
2:4 OBS test using sparsekit's BlockSpec + ScopeSpec + StructuredOBS.

Test 1 — Contiguous 2:4:
  W:            (M, K) = (32, 16)
  BlockSpec:    param=W, block_shape=(1, 1)  -> grid_shape=(32, 16)
  ScopeSpec:    block_shape=(1, 4)            -> block_grid=(32, 4)
  num_nz:       2  (keep 2 of 4 elements per block -> 50% sparsity)

Test 2 — Non-contiguous via GroupView:
  W:            (M, K) = (32, 16)
  GroupView:    size=(32, 8, 2), stride=(16, 1, 8)
                view[i,j,k] = W[i, j + 8k]
  BlockSpec:    param=view, block_shape=(1, 1, 2) -> grid_shape=(32, 8, 1)
                each group couples columns {j, j+8}
  ScopeSpec:    block_shape=(1, 4, 1)  -> block_grid=(32, 2, 1)
  num_nz:       2  (keep 2 of 4 groups per block)
"""

import torch
import torch.linalg as LA

from sparsekit.view import View
from sparsekit.block import BlockSpec
from sparsekit.scope import ScopeSpec
from sparsekit.pruners.obs import StructuredOBS


def magnitude_prune_via_group(W0, group_shape, scope_shape, num_nz, param_factory):
    """Magnitude pruning using ScopeSpec.get_masks."""
    param = param_factory(W0.clone())
    group = BlockSpec(param, shape=group_shape)
    g = ScopeSpec(group, shape=scope_shape)
    masks = g.get_masks(num_nz=num_nz)
    for spec, mask in masks.items():
        spec.view.data[~mask] = 0.0
    if isinstance(param, View):
        return param.param.data
    return param.data


# ── Test 1: Contiguous 2:4 ──────────────────────────────────────────────

def test_contiguous_24():
    print("=" * 60)
    print("Test 1: Contiguous 2:4")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cpu")

    M, K = 32, 16
    N = 128
    num_nz = 2
    group_shape = (1, 1)
    scope_shape = (1, 4)

    X = torch.randn(N, K, device=device)
    W0 = torch.randn(M, K, device=device)
    H = X.T @ X / N
    Y0 = X @ W0.T

    # OBS
    W_obs = torch.nn.Parameter(W0.clone())
    block_obs = BlockSpec(W_obs, shape=group_shape)
    group_obs = ScopeSpec(block_obs, shape=scope_shape)
    print(f"  BlockSpec: {block_obs}")
    print(f"  ScopeSpec: {group_obs}")

    solver = StructuredOBS(group_obs, H, damp=1e-4)
    solver.prune(num_nz=num_nz)
    loss_obs = ((X @ W_obs.data.T - Y0) ** 2).sum().item()

    # Magnitude
    W_mag = magnitude_prune_via_group(
        W0, group_shape, scope_shape, num_nz,
        param_factory=torch.nn.Parameter,
    )
    loss_mag = ((X @ W_mag.T - Y0) ** 2).sum().item()

    print(f"\n  {'Method':<20} {'Loss':>12}")
    print(f"  {'-'*34}")
    print(f"  {'Structured OBS':<20} {loss_obs:>12.4f}")
    print(f"  {'Magnitude':<20} {loss_mag:>12.4f}")

    if loss_obs < loss_mag:
        pct = (1 - loss_obs / loss_mag) * 100
        print(f"  OBS beats Magnitude by {pct:.1f}%")

    # Sparsity check
    ok = True
    for r in range(M):
        for g in range(K // 4):
            cols = W_obs.data[r, g*4:(g+1)*4]
            nnz = (cols.abs() > 1e-8).sum().item()
            if nnz != num_nz:
                print(f"  FAIL: row {r}, block {g}: nnz={nnz}")
                ok = False
    print(f"  Sparsity check: {'PASS' if ok else 'FAIL'}")

    assert ok, "Sparsity constraint violated!"
    assert loss_obs < loss_mag, f"OBS should beat magnitude: {loss_obs} >= {loss_mag}"
    print("  All checks passed!\n")


# ── Test 2: Non-contiguous via GroupView ─────────────────────────────────

def test_blockview_24():
    print("=" * 60)
    print("Test 2: Non-contiguous via GroupView")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cpu")

    M, K = 32, 16
    N = 128
    num_nz = 2
    view_size = (32, 8, 2)
    view_stride = (16, 1, 8)
    group_shape = (1, 1, 2)
    scope_shape = (1, 4, 1)

    X = torch.randn(N, K, device=device)
    W0 = torch.randn(M, K, device=device)
    H = X.T @ X / N
    Y0 = X @ W0.T

    # OBS
    W_obs = torch.nn.Parameter(W0.clone())
    view_obs = View(W_obs, shape=view_size, stride=view_stride)
    block_obs = BlockSpec(view_obs, shape=group_shape)
    group_obs = ScopeSpec(block_obs, shape=scope_shape)

    print(f"  GroupView: size={view_size}, stride={view_stride}")
    print(f"  BlockSpec: grid_shape={block_obs.grid_shape}, group_shape={group_shape}")
    print(f"  ScopeSpec: grid_shape={group_obs.grid_shape}, scope_shape={scope_shape}")

    solver = StructuredOBS(group_obs, H, damp=1e-4)
    solver.prune(num_nz=num_nz)
    loss_obs = ((X @ W_obs.data.T - Y0) ** 2).sum().item()

    # Magnitude
    def make_view(W):
        p = torch.nn.Parameter(W)
        return View(p, shape=view_size, stride=view_stride)

    W_mag_param = torch.nn.Parameter(W0.clone())
    W_mag_view = View(W_mag_param, shape=view_size, stride=view_stride)
    block_mag = BlockSpec(W_mag_view, shape=group_shape)
    group_mag = ScopeSpec(block_mag, shape=scope_shape)
    masks = group_mag.get_masks(num_nz=num_nz)
    for spec, mask in masks.items():
        spec.view.data[~mask] = 0.0
    loss_mag = ((X @ W_mag_param.data.T - Y0) ** 2).sum().item()

    print(f"\n  {'Method':<20} {'Loss':>12}")
    print(f"  {'-'*34}")
    print(f"  {'Structured OBS':<20} {loss_obs:>12.4f}")
    print(f"  {'Magnitude':<20} {loss_mag:>12.4f}")

    if loss_obs < loss_mag:
        pct = (1 - loss_obs / loss_mag) * 100
        print(f"  OBS beats Magnitude by {pct:.1f}%")

    # Sparsity check: each block of 4 groups (j=0..3 and j=4..7) should have
    # exactly 2 groups kept. Each group couples columns {j, j+8}.
    ok = True
    v = View(torch.nn.Parameter(W_obs.data), shape=view_size, stride=view_stride)
    bv = View.block_view_of(v.data, group_shape, reorder=True, merge=True)
    # bv shape: (32, 8, 1, 2) -> after merge: (32, 8, 1, 2) -> grid (32,8,1), block_numel=2
    # Actually with merge=True: (32, 8, 1, 2) but grid_shape=(32,8,1) so merged = (32,8,1,2)
    # Check via norms
    all_norms = bv.norm(dim=-1)  # (32, 8, 1)
    # Blocks: along dim 1, blocks of 4
    for r in range(M):
        for g in range(2):
            scope_norms = all_norms[r, g*4:(g+1)*4, 0]
            nnz = (scope_norms > 1e-8).sum().item()
            if nnz != num_nz:
                print(f"  FAIL: row {r}, block {g}: nnz={nnz}")
                ok = False
    print(f"  Sparsity check: {'PASS' if ok else 'FAIL'}")

    assert ok, "Sparsity constraint violated!"
    assert loss_obs < loss_mag, f"OBS should beat magnitude: {loss_obs} >= {loss_mag}"
    print("  All checks passed!\n")


if __name__ == "__main__":
    test_contiguous_24()
    test_blockview_24()
    print("All tests passed!")
