import torch
import torch.linalg as LA
from itertools import product as iterproduct
from dataclasses import dataclass, field
from typing import List, Callable, Tuple, Any


@dataclass
class Block:
    id: int
    col_indices: List[int]
    row_indices: List[int]
    constraint_group: Any  # hashable
    c_group: Any           # hashable – blocks sharing the same c_group evolve one H⁻¹ copy


# ---------------------------------------------------------------------------
# General: derive blocks from a strided + reshaped layout
# ---------------------------------------------------------------------------

def blocks_from_layout(
    k: int,
    view_shape: Tuple[int, ...],
    view_strides: Tuple[int, ...],
    reshape_shape: Tuple[int, ...],
    block_shape: Tuple[int, ...],
    constraint_group_fn: Callable,   # block_grid_pos -> hashable
    c_group_fn: Callable,            # block_grid_pos -> hashable
) -> List[Block]:
    """
    Enumerate all pruning blocks from a layout defined by:
        W_l = torch.as_strided(W, view_shape, view_strides)
        W_l = W_l.reshape(reshape_shape)

    Each block occupies `block_shape` elements in the reshaped view.
    The function maps every block back to original (row, col) indices in W.

    Args:
        k:               number of columns in the original W (m, k)
        view_shape:      shape given to as_strided
        view_strides:    strides given to as_strided (in *elements*, matching W.stride())
        reshape_shape:   shape after .reshape()
        block_shape:     pruning block shape (same ndim as reshape_shape)
        constraint_group_fn:  maps block grid position -> constraint group id
        c_group_fn:           maps block grid position -> C-group id
    """
    ndim = len(reshape_shape)
    assert len(block_shape) == ndim
    grid_shape = tuple(reshape_shape[d] // block_shape[d] for d in range(ndim))

    # Pre-compute contiguous strides for reshape and view shapes (row-major)
    def _contig_strides(shape):
        s = [1] * len(shape)
        for d in range(len(shape) - 2, -1, -1):
            s[d] = s[d + 1] * shape[d + 1]
        return s

    rs_strides = _contig_strides(reshape_shape)
    vs_strides = _contig_strides(view_shape)

    blocks: List[Block] = []
    bid = 0

    for block_pos in iterproduct(*(range(g) for g in grid_shape)):
        elem_ranges = [
            range(block_pos[d] * block_shape[d], (block_pos[d] + 1) * block_shape[d])
            for d in range(ndim)
        ]

        rows, cols = set(), set()
        for elem_idx in iterproduct(*elem_ranges):
            # reshape index -> flat (contiguous row-major in reshape_shape)
            flat = sum(elem_idx[d] * rs_strides[d] for d in range(ndim))

            # flat -> view index (unravel in view_shape, row-major)
            view_idx = []
            rem = flat
            for d in range(len(view_shape)):
                view_idx.append(rem // vs_strides[d])
                rem %= vs_strides[d]

            # view index -> offset in original W using the *as_strided* strides
            offset = sum(view_idx[d] * view_strides[d] for d in range(len(view_shape)))

            rows.add(offset // k)
            cols.add(offset % k)

        blocks.append(Block(
            id=bid,
            col_indices=sorted(cols),
            row_indices=sorted(rows),
            constraint_group=constraint_group_fn(block_pos),
            c_group=c_group_fn(block_pos),
        ))
        bid += 1

    return blocks


# ---------------------------------------------------------------------------
# Core: structured OBS pruning (greedy, iterative)
# ---------------------------------------------------------------------------

def structured_obs(
    W: torch.Tensor,
    H_inv: torch.Tensor,
    blocks: List[Block],
    n_keep: int,
    damping: float = 0.0,
) -> torch.Tensor:
    """
    Greedy structured OBS pruning.

    For every constraint group, iteratively removes blocks until only `n_keep`
    remain, always choosing the block with minimum OBS cost across all
    constraint groups that share the same C-group.

    Args:
        W:       (m, k) weight matrix
        H_inv:   (k, k) Hessian inverse
        blocks:  list of Block objects (from blocks_from_layout or manual)
        n_keep:  number of blocks to *keep* per constraint group
        damping: optional extra diagonal damping for C_SS inversion

    Returns:
        Pruned W (new tensor).
    """
    W = W.clone()
    dev = W.device

    # Partition blocks by c_group
    c_groups: dict[Any, List[Block]] = {}
    for b in blocks:
        c_groups.setdefault(b.c_group, []).append(b)

    for cg_id, cg_blocks in c_groups.items():
        C = H_inv.clone()

        # Partition into constraint sub-groups
        constraints: dict[Any, List[Block]] = {}
        for b in cg_blocks:
            constraints.setdefault(b.constraint_group, []).append(b)

        # How many to prune per constraint group
        n_prune = {cid: len(bs) - n_keep for cid, bs in constraints.items()}
        prune_count = {cid: 0 for cid in constraints}
        alive = {b.id: b for b in cg_blocks}     # id -> Block
        alive_by_cg = {cid: {b.id for b in bs} for cid, bs in constraints.items()}

        total = sum(n_prune.values())

        for _step in range(total):
            best_cost = float('inf')
            best = None

            for cid, bids in alive_by_cg.items():
                if prune_count[cid] >= n_prune[cid]:
                    continue
                for bid in bids:
                    b = alive[bid]
                    S = torch.tensor(b.col_indices, device=dev, dtype=torch.long)
                    C_SS = C[S][:, S]
                    if damping > 0:
                        C_SS = C_SS + damping * torch.eye(len(S), device=dev, dtype=C.dtype)
                    C_SS_inv = LA.inv(C_SS)
                    R = torch.tensor(b.row_indices, device=dev, dtype=torch.long)
                    w_S = W[R][:, S]
                    # cost = 0.5 * Σ_r  w_S[r] @ C_SS⁻¹ @ w_S[r]
                    cost = 0.5 * (w_S @ C_SS_inv * w_S).sum().item()

                    if cost < best_cost:
                        best_cost = cost
                        best = (b, S, R, C_SS_inv, cid)

            b, S, R, C_SS_inv, cid = best

            # --- OBS weight update for affected rows ---
            w_S = W[R][:, S].clone()              # (|R|, |S|)
            W[R] -= (w_S @ C_SS_inv) @ C[S]       # (|R|, k)

            # --- Zero the pruned positions ---
            W[R.unsqueeze(1), S.unsqueeze(0)] = 0.0

            # --- Rank-|S| update to the Hessian inverse ---
            C -= C[:, S] @ C_SS_inv @ C[S]

            # --- Bookkeeping ---
            del alive[b.id]
            alive_by_cg[cid].discard(b.id)
            prune_count[cid] += 1

    return W


# ---------------------------------------------------------------------------
# Verify sparsity pattern
# ---------------------------------------------------------------------------

def verify_pattern(W_pruned, view_shape, view_strides, reshape_shape,
                   block_shape, constraint_group_fn, n_keep, k, atol=1e-6):
    """Check that every constraint group has exactly n_keep surviving blocks."""
    blocks = blocks_from_layout(k, view_shape, view_strides, reshape_shape,
                                block_shape, constraint_group_fn, lambda p: 0)
    groups: dict[Any, int] = {}
    for b in blocks:
        alive = not all(
            abs(W_pruned[r, c]) < atol
            for r in b.row_indices for c in b.col_indices
        )
        groups.setdefault(b.constraint_group, 0)
        if alive:
            groups[b.constraint_group] += 1

    ok = all(v == n_keep for v in groups.values())
    if not ok:
        bad = {k: v for k, v in groups.items() if v != n_keep}
        print(f"FAIL – groups with wrong count: {bad}")
    else:
        print(f"OK – all {len(groups)} constraint groups have exactly {n_keep} surviving blocks")
    return ok


# ===================================================================
# Example: the specific layout from the problem
# ===================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lamb = 1e-4
    num_samples = 64
    m, k = 56, 64

    X = torch.randn(num_samples, k, device=device)
    W = torch.randn(m, k, device=device)
    Y = X @ W.T   # targets (if needed)

    H = (X.T @ X) / num_samples + lamb * torch.eye(k, device=device)
    C = LA.inv(H)

    # --- Layout specification ---
    view_shape    = (56, 4, 8, 2)
    view_strides  = (k, 16, 1, 8)        # as_strided strides (in elements)
    reshape_shape = (14, 4, 4, 8, 2)
    block_shape   = (1,  4, 1, 1, 2)     # block to prune/keep

    # Constraint: within each (group_i, block_k, half), keep 2 out of 4 blocks
    # half = 0 for positions 0..3, half = 1 for positions 4..7
    # block_grid_pos has shape (14, 1, 4, 8, 1) → pos = (i, 0, m, p, 0)
    constraint_group_fn = lambda pos: (pos[0], pos[2], 0 if pos[3] < 4 else 1)
    c_group_fn          = lambda pos: pos[0]   # independent H⁻¹ per group-of-4-rows

    blocks = blocks_from_layout(
        k, view_shape, view_strides, reshape_shape, block_shape,
        constraint_group_fn, c_group_fn,
    )
    print(f"Total blocks: {len(blocks)}")
    print(f"Block col indices example (block 0): {blocks[0].col_indices}")
    print(f"Block row indices example (block 0): {blocks[0].row_indices}")

    # --- Compute OBS objective before pruning (should be 0) ---
    def obs_objective(W_new, W_orig, H):
        dW = W_new - W_orig
        return 0.5 * (dW @ H * dW).sum().item()

    W_orig = W.clone()
    print(f"\nOBS objective before pruning: {obs_objective(W, W_orig, H):.6e}")

    # --- Prune ---
    W_pruned = structured_obs(W, C, blocks, n_keep=2)

    print(f"OBS objective after pruning:  {obs_objective(W_pruned, W_orig, H):.6e}")

    # --- MSE on training data ---
    mse_before = ((X @ W_orig.T - Y) ** 2).mean().item()
    mse_after  = ((X @ W_pruned.T - Y) ** 2).mean().item()
    print(f"\nMSE before: {mse_before:.6e}")
    print(f"MSE after:  {mse_after:.6e}")

    # --- Sparsity ---
    nnz = (W_pruned.abs() > 1e-8).sum().item()
    total = W_pruned.numel()
    print(f"\nNon-zeros: {nnz}/{total} ({100*nnz/total:.1f}%)")

    # --- Verify constraint satisfaction ---
    verify_pattern(W_pruned, view_shape, view_strides, reshape_shape,
                   block_shape, constraint_group_fn, n_keep=2, k=k)

