"""
2:4 OBS test using sparsekit's BlockSpec + GroupSpec + StructuredOBS.

Test 1 — Contiguous 2:4:
  W:            (M, K) = (32, 16)
  BlockSpec:    param=W, block_shape=(1, 1)  -> grid_shape=(32, 16)
  GroupSpec:    group_shape=(1, 4)            -> group_grid=(32, 4)
  num_nz:       2  (keep 2 of 4 elements per group -> 50% sparsity)

Test 2 — Non-contiguous via BlockView:
  W:            (M, K) = (32, 16)
  BlockView:    size=(32, 8, 2), stride=(16, 1, 8)
                view[i,j,k] = W[i, j + 8k]
  BlockSpec:    param=view, block_shape=(1, 1, 2) -> grid_shape=(32, 8, 1)
                each block couples columns {j, j+8}
  GroupSpec:    group_shape=(1, 4, 1)  -> group_grid=(32, 2, 1)
  num_nz:       2  (keep 2 of 4 blocks per group)
"""

import torch
import torch.linalg as LA

from sparsekit.view import View
from sparsekit.block import BlockSpec
from sparsekit.group import GroupSpec
from sparsekit.pruners.obs import StructuredOBS


def magnitude_prune_via_group(W0, block_shape, group_shape, num_nz, param_factory):
    """Magnitude pruning using GroupSpec.get_masks."""
    param = param_factory(W0.clone())
    block = BlockSpec(param, shape=block_shape)
    g = GroupSpec(block, shape=group_shape)
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
    block_shape = (1, 1)
    group_shape = (1, 4)

    X = torch.randn(N, K, device=device)
    W0 = torch.randn(M, K, device=device)
    H = X.T @ X / N
    Y0 = X @ W0.T

    # OBS
    W_obs = torch.nn.Parameter(W0.clone())
    block_obs = BlockSpec(W_obs, shape=block_shape)
    group_obs = GroupSpec(block_obs, shape=group_shape)
    print(f"  BlockSpec: {block_obs}")
    print(f"  GroupSpec: {group_obs}")

    solver = StructuredOBS(group_obs, H, damp=1e-4)
    solver.prune(num_nz=num_nz)
    loss_obs = ((X @ W_obs.data.T - Y0) ** 2).sum().item()

    # Magnitude
    W_mag = magnitude_prune_via_group(
        W0, block_shape, group_shape, num_nz,
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
                print(f"  FAIL: row {r}, group {g}: nnz={nnz}")
                ok = False
    print(f"  Sparsity check: {'PASS' if ok else 'FAIL'}")

    assert ok, "Sparsity constraint violated!"
    assert loss_obs < loss_mag, f"OBS should beat magnitude: {loss_obs} >= {loss_mag}"
    print("  All checks passed!\n")


# ── Test 2: Non-contiguous via BlockView ─────────────────────────────────

def test_blockview_24():
    print("=" * 60)
    print("Test 2: Non-contiguous via BlockView")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device("cpu")

    M, K = 32, 16
    N = 128
    num_nz = 2
    view_size = (32, 8, 2)
    view_stride = (16, 1, 8)
    block_shape = (1, 1, 2)
    group_shape = (1, 4, 1)

    X = torch.randn(N, K, device=device)
    W0 = torch.randn(M, K, device=device)
    H = X.T @ X / N
    Y0 = X @ W0.T

    # OBS
    W_obs = torch.nn.Parameter(W0.clone())
    view_obs = View(W_obs, shape=view_size, stride=view_stride)
    block_obs = BlockSpec(view_obs, shape=block_shape)
    group_obs = GroupSpec(block_obs, shape=group_shape)

    print(f"  BlockView: size={view_size}, stride={view_stride}")
    print(f"  BlockSpec: grid_shape={block_obs.grid_shape}, block_shape={block_shape}")
    print(f"  GroupSpec: grid_shape={group_obs.grid_shape}, group_shape={group_shape}")

    solver = StructuredOBS(group_obs, H, damp=1e-4)
    solver.prune(num_nz=num_nz)
    loss_obs = ((X @ W_obs.data.T - Y0) ** 2).sum().item()

    # Magnitude
    def make_view(W):
        p = torch.nn.Parameter(W)
        return View(p, shape=view_size, stride=view_stride)

    W_mag_param = torch.nn.Parameter(W0.clone())
    W_mag_view = View(W_mag_param, shape=view_size, stride=view_stride)
    block_mag = BlockSpec(W_mag_view, shape=block_shape)
    group_mag = GroupSpec(block_mag, shape=group_shape)
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

    # Sparsity check: each group of 4 blocks (j=0..3 and j=4..7) should have
    # exactly 2 blocks kept. Each block couples columns {j, j+8}.
    ok = True
    v = View(torch.nn.Parameter(W_obs.data), shape=view_size, stride=view_stride)
    bv = View.block_view_of(v.data, block_shape, reorder=True, merge=True)
    # bv shape: (32, 8, 1, 2) -> after merge: (32, 8, 1, 2) -> grid (32,8,1), block_numel=2
    # Actually with merge=True: (32, 8, 1, 2) but grid_shape=(32,8,1) so merged = (32,8,1,2)
    # Check via norms
    block_norms = bv.norm(dim=-1)  # (32, 8, 1)
    # Groups: along dim 1, groups of 4
    for r in range(M):
        for g in range(2):
            group_norms = block_norms[r, g*4:(g+1)*4, 0]
            nnz = (group_norms > 1e-8).sum().item()
            if nnz != num_nz:
                print(f"  FAIL: row {r}, group {g}: nnz={nnz}")
                ok = False
    print(f"  Sparsity check: {'PASS' if ok else 'FAIL'}")

    assert ok, "Sparsity constraint violated!"
    assert loss_obs < loss_mag, f"OBS should beat magnitude: {loss_obs} >= {loss_mag}"
    print("  All checks passed!\n")


if __name__ == "__main__":
    test_contiguous_24()
    test_blockview_24()
    print("All tests passed!")
