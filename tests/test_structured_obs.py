"""
Structured OBS test using BlockView + BlockSpec.

Configuration:
  W:            (M, K) = (32, 16)
  BlockView:    size=(32, 8, 2), stride=(16, 1, 8)
                view[i, j, k] = W[i, j + 8*k]
  block_shape:  (1, 1, 2)   -> grid_shape = (32, 8, 1)
                each block couples columns {j, j+8} for a given row i
  group_shape:  (1, 4, 1)   -> group_grid = (32, 2, 1)
                group (r,0,0): blocks j=0..3 -> columns {0,8},{1,9},{2,10},{3,11}
                group (r,1,0): blocks j=4..7 -> columns {4,12},{5,13},{6,14},{7,15}
  nnz:          2  (keep 2 of 4 blocks per group, prune 2)

Methods benchmarked:
  1. Structured OBS — full H^{-1}, greedy block selection, Schur complement update
  2. SparseGPT      — Cholesky(H^{-1}), sequential column processing, one-shot selection
  3. Magnitude       — prune smallest-norm blocks per group
  4. Naive zero      — same pattern as OBS, but no weight compensation
"""

import torch
import torch.linalg as LA

from sparsekit.view import View
from sparsekit.block import BlockSpec


# ─── helpers ────────────────────────────────────────────────────────────

def block_col_indices(view: View, block_shape, device="cpu"):
    """
    For each block in the grid, compute which K-columns of the original
    (M, K) matrix it touches.

    Returns:
        col_idx: (*grid_shape, block_numel) long tensor of K-column indices.
    """
    grid_shape = tuple(s // b for s, b in zip(view.shape, block_shape))
    rank = len(block_shape)

    ranges = [torch.arange(b, device=device) for b in block_shape]
    offsets = torch.stack(torch.meshgrid(*ranges, indexing="ij"), dim=-1).reshape(-1, rank)

    grid_ranges = [torch.arange(g, device=device) for g in grid_shape]
    grid_pts = torch.stack(torch.meshgrid(*grid_ranges, indexing="ij"), dim=-1)

    bs = torch.tensor(block_shape, device=device)
    grid_elem = grid_pts.unsqueeze(-2) * bs + offsets

    flat_offsets = view.linear_offset(grid_elem)
    K = view.param.shape[1]
    return flat_offsets % K


def build_group_mapping(grid_shape, group_shape, group_grid, device):
    """
    Build vectorized mappings from block-columns to groups.

    Returns:
        block_to_group:       (Bk_total,) group index per block-column
        block_within_group:   (Bk_total,) index within group per block-column
    """
    grid_rest = grid_shape[1:]
    group_rest = group_shape[1:]
    gg_rest = group_grid[1:]

    Bk_total = 1
    for g in grid_rest:
        Bk_total *= g

    bc_ranges = [torch.arange(g, device=device) for g in grid_rest]
    bc_grid = torch.stack(torch.meshgrid(*bc_ranges, indexing="ij"), dim=-1).reshape(Bk_total, -1)

    group_pos = bc_grid // torch.tensor(group_rest, device=device)
    gg_strides = []
    s = 1
    for g in reversed(list(gg_rest)):
        gg_strides.append(s)
        s *= g
    block_to_group = (group_pos * torch.tensor(list(reversed(gg_strides)), device=device)).sum(dim=-1)

    within_pos = bc_grid % torch.tensor(group_rest, device=device)
    gs_strides = []
    s = 1
    for g in reversed(group_rest):
        gs_strides.append(s)
        s *= g
    block_within_group = (within_pos * torch.tensor(list(reversed(gs_strides)), device=device)).sum(dim=-1)

    return block_to_group, block_within_group


# ─── Structured OBS solver (vectorized) ─────────────────────────────────

class StructuredOBS:
    """
    Vectorized structured OBS.

    One C = H^{-1} per group row.  Within each group row, greedily prune
    the cheapest block, update W and C, repeat.
    All group rows are independent and processed in parallel (batched).
    """

    def __init__(self, W, view, block_shape, group_shape, nnz, X, damp=1e-4):
        self.W = W
        self.view = view
        self.block_shape = tuple(block_shape)
        self.group_shape = tuple(group_shape)
        self.nnz = nnz

        M, K = W.shape
        self.M, self.K = M, K
        device = W.device

        self.grid_shape = tuple(s // b for s, b in zip(view.shape, block_shape))
        self.group_grid = tuple(g // gg for g, gg in zip(self.grid_shape, group_shape))

        self.blocks_per_group = 1
        for gs in group_shape:
            self.blocks_per_group *= gs
        self.num_prune = self.blocks_per_group - nnz

        col_idx_full = block_col_indices(view, block_shape, device=device)
        Bk_total = 1
        for g in self.grid_shape[1:]:
            Bk_total *= g
        self.bk = col_idx_full.shape[-1]
        self.col_idx = col_idx_full[0].reshape(Bk_total, self.bk)
        self.Bk_total = Bk_total

        self.rows_per_group = self.group_shape[0] * self.block_shape[0]
        self.num_group_rows = self.group_grid[0]

        self.num_groups_per_row = 1
        for gg in self.group_grid[1:]:
            self.num_groups_per_row *= gg

        self.block_to_group, self.block_within_group = build_group_mapping(
            self.grid_shape, self.group_shape, self.group_grid, device,
        )

        H = X.T @ X / X.shape[0]
        C = LA.inv(H + damp * torch.eye(K, device=device))
        self.C = C.unsqueeze(0).expand(self.num_group_rows, -1, -1).clone()

        self.pruned = torch.zeros(self.num_group_rows, Bk_total, dtype=torch.bool, device=device)

        self.remaining = torch.full(
            (self.num_group_rows, self.num_groups_per_row),
            self.num_prune, dtype=torch.long, device=device,
        )

    def _gather_C_block(self, cols):
        """Extract C[r, cols, :][:, cols] for all group rows."""
        R, K, bk = self.num_group_rows, self.K, self.bk
        row_idx = cols.unsqueeze(-1).expand(-1, -1, K)
        C_rows = torch.gather(self.C, 1, row_idx)
        col_idx = cols.unsqueeze(1).expand(-1, bk, -1)
        C_pp = torch.gather(C_rows, 2, col_idx)
        return C_pp, C_rows

    def compute_scores(self):
        """Vectorized OBS scores for all (group_row, block_col) pairs."""
        R, bk, device = self.num_group_rows, self.bk, self.W.device

        W_rows = self.W.data.view(R, self.rows_per_group, self.K)
        ci = self.col_idx.unsqueeze(0).unsqueeze(0).expand(R, self.rows_per_group, -1, -1)
        W_p = torch.gather(
            W_rows.unsqueeze(2).expand(-1, -1, self.Bk_total, -1), 3, ci,
        )

        ci_all = self.col_idx.unsqueeze(0).expand(R, -1, -1)
        ci_row = ci_all.unsqueeze(-1).expand(-1, -1, -1, self.K)
        C_rows = torch.gather(
            self.C.unsqueeze(1).expand(-1, self.Bk_total, -1, -1), 2, ci_row,
        )
        ci_col = ci_all.unsqueeze(2).expand(-1, -1, bk, -1)
        C_pp = torch.gather(C_rows, 3, ci_col)

        C_pp_inv = LA.inv(C_pp + 1e-6 * torch.eye(bk, device=device))

        temp = torch.einsum("rjpb,rpba->rjpa", W_p, C_pp_inv)
        scores = (temp * W_p).sum(dim=(1, 3))

        scores[self.pruned] = float("inf")
        return scores

    def batch_update(self, selections):
        """OBS weight update + Schur complement for all group rows."""
        R, bk, K, device = self.num_group_rows, self.bk, self.K, self.W.device

        active = selections >= 0
        if not active.any():
            return

        sel = selections.clamp(min=0)
        pruned_cols = self.col_idx[sel]

        W_rows = self.W.data.view(R, self.rows_per_group, K)
        pc = pruned_cols.unsqueeze(1).expand(-1, self.rows_per_group, -1)
        W_p = torch.gather(W_rows, 2, pc)

        C_pp, C_row_p = self._gather_C_block(pruned_cols)
        pc_C = pruned_cols.unsqueeze(1).expand(-1, K, -1)
        C_col_p = torch.gather(self.C, 2, pc_C)

        C_pp_inv = LA.inv(C_pp + 1e-6 * torch.eye(bk, device=device))
        temp = torch.bmm(C_col_p, C_pp_inv)
        delta = torch.einsum("rjb,rkb->rjk", W_p, temp)

        active_f = active.float().view(R, 1, 1)
        W_rows -= delta * active_f

        zeros = torch.zeros(R, self.rows_per_group, bk, device=device)
        current = torch.gather(W_rows, 2, pc)
        W_rows.scatter_(2, pc, torch.where(active.view(R, 1, 1).expand_as(current), zeros, current))

        schur = torch.bmm(temp, C_row_p)
        self.C -= schur * active_f

        r_active = torch.where(active)[0]
        if r_active.numel() > 0:
            active_cols = pruned_cols[r_active]
            for b in range(bk):
                self.C[r_active, active_cols[:, b], :] = 0.0
                self.C[r_active, :, active_cols[:, b]] = 0.0

        self.pruned[r_active, sel[r_active]] = True
        active_groups = self.block_to_group[sel[r_active]]
        self.remaining[r_active, active_groups] -= 1

    def prune(self):
        """Greedy structured OBS."""
        total_prunes = self.num_prune * self.num_groups_per_row
        for step in range(total_prunes):
            scores = self.compute_scores()

            group_done = self.remaining <= 0
            block_group_done = group_done[:, self.block_to_group]
            scores[block_group_done] = float("inf")

            selections = scores.argmin(dim=1)
            all_done = (scores == float("inf")).all(dim=1)
            selections[all_done] = -1

            if (selections == -1).all():
                break

            self.batch_update(selections)


# ─── SparseGPT adapted to our block/group structure ─────────────────────

class SparseGPTBlockPruner:
    """
    SparseGPT-style pruner adapted to arbitrary BlockView structure.

    Key differences from true OBS (StructuredOBS):
      1. Uses Cholesky(H^{-1}) instead of full H^{-1} + Schur complement.
         The upper-triangular Cholesky factor L^T encodes a specific
         column ordering: column i is "processed" using L^T[i,i] as
         the effective inverse-Hessian diagonal and L^T[i, i+1:] for
         error propagation to later columns.
      2. One-shot block selection: within each group, scores are computed
         once from W^2 / diag(Hinv)^2 and pruning decisions are made
         without re-scoring.
      3. Sequential column processing: columns are processed left-to-right,
         errors propagate forward only.  No global Schur complement update.

    This means SparseGPT's quality depends on the column ordering aligning
    well with the Cholesky factorization. For non-contiguous block patterns
    (like our strided view), this ordering may be suboptimal.
    """

    def __init__(self, W, view, block_shape, group_shape, nnz, X, damp=1e-4):
        M, K = W.shape
        self.M, self.K = M, K
        self.W = W
        self.view = view
        self.block_shape = tuple(block_shape)
        self.group_shape = tuple(group_shape)
        self.nnz = nnz
        device = W.device

        self.grid_shape = tuple(s // b for s, b in zip(view.shape, block_shape))
        self.group_grid = tuple(g // gg for g, gg in zip(self.grid_shape, group_shape))

        self.blocks_per_group = 1
        for gs in group_shape:
            self.blocks_per_group *= gs
        self.num_prune = self.blocks_per_group - nnz

        col_idx_full = block_col_indices(view, block_shape, device=device)
        Bk_total = 1
        for g in self.grid_shape[1:]:
            Bk_total *= g
        self.bk = col_idx_full.shape[-1]
        self.col_idx = col_idx_full[0].reshape(Bk_total, self.bk)
        self.Bk_total = Bk_total

        self.rows_per_group = self.group_shape[0] * self.block_shape[0]
        self.num_group_rows = self.group_grid[0]

        self.num_groups_per_row = 1
        for gg in self.group_grid[1:]:
            self.num_groups_per_row *= gg

        self.block_to_group, _ = build_group_mapping(
            self.grid_shape, self.group_shape, self.group_grid, device,
        )

        # Compute Hinv = upper Cholesky of H^{-1}, exactly as SparseGPT does
        H = X.T @ X / X.shape[0]  # (K, K)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W.data[:, dead] = 0

        damp_val = damp * torch.mean(torch.diag(H))
        diag = torch.arange(K, device=device)
        H[diag, diag] += damp_val

        # H -> cholesky -> cholesky_inverse -> upper cholesky
        L = LA.cholesky(H)
        Hinv_full = torch.cholesky_inverse(L)
        self.Hinv = LA.cholesky(Hinv_full, upper=True)  # (K, K) upper triangular

        # pruned mask
        self.pruned = torch.zeros(self.num_group_rows, Bk_total, dtype=torch.bool, device=device)

    def prune(self):
        """
        SparseGPT fasterprune adapted to our block/group structure.

        For each group, one-shot select which blocks to prune based on
        W^2 / diag(Hinv)^2 scores, then process columns sequentially
        propagating errors via the Cholesky factor.
        """
        device = self.W.device
        R = self.num_group_rows
        K = self.K
        bk = self.bk

        W = self.W.data.clone().float()  # (M, K)
        Hinv = self.Hinv  # (K, K) upper triangular

        # ── Step 1: compute block scores and decide pruning mask ──
        # SparseGPT scores: for each element, w^2 / (Hinv[i,i])^2
        # For blocks: sum over the block's elements
        diag_Hinv = torch.diag(Hinv)  # (K,)

        # Score per block = sum_{c in block_cols} sum_{r in rows} W[r,c]^2 / Hinv[c,c]^2
        # W_rows: (R, rows_per_group, K)
        W_rows = W.view(R, self.rows_per_group, K)

        # Gather per-block weights: (R, rows_per_group, Bk_total, bk)
        ci = self.col_idx.unsqueeze(0).unsqueeze(0).expand(R, self.rows_per_group, -1, -1)
        W_p = torch.gather(
            W_rows.unsqueeze(2).expand(-1, -1, self.Bk_total, -1), 3, ci,
        )

        # Gather per-block diag(Hinv): (Bk_total, bk)
        diag_p = diag_Hinv[self.col_idx]  # (Bk_total, bk)

        # Block score = sum over (rows, bk) of W_p^2 / diag_p^2
        scores = (W_p ** 2 / (diag_p.unsqueeze(0).unsqueeze(0) ** 2)).sum(dim=(1, 3))  # (R, Bk_total)

        # For each group, prune the num_prune lowest-score blocks
        for g in range(self.num_groups_per_row):
            in_group = (self.block_to_group == g)
            group_scores = scores[:, in_group]  # (R, blocks_per_group)
            _, bot_idx = group_scores.topk(self.num_prune, dim=1, largest=False)
            group_block_indices = torch.where(in_group)[0]
            prune_idx = group_block_indices[bot_idx]
            row_idx = torch.arange(R, device=device).unsqueeze(1).expand_as(prune_idx)
            self.pruned[row_idx, prune_idx] = True

        # ── Step 2: build per-row element-level mask ──
        # mask[r, k] = True if column k should be zeroed for group row r
        # Each group row has its own pattern since different blocks may be pruned
        mask = torch.zeros(R, K, dtype=torch.bool, device=device)
        pruned_cols_all = self.col_idx.unsqueeze(0).expand(R, -1, -1)  # (R, Bk_total, bk)
        pruned_exp = self.pruned.unsqueeze(-1).expand_as(pruned_cols_all)
        # Scatter True into mask
        mask.scatter_(1, pruned_cols_all[pruned_exp].view(R, -1)
                      if pruned_exp.any()
                      else torch.zeros(R, 0, dtype=torch.long, device=device),
                      True)
        # Build mask properly: for each (r, p) where pruned, mark col_idx[p] in mask[r]
        for r in range(R):
            pruned_blocks = torch.where(self.pruned[r])[0]
            if pruned_blocks.numel() > 0:
                cols = self.col_idx[pruned_blocks].reshape(-1)
                mask[r, cols] = True

        # ── Step 3: SparseGPT sequential column processing ──
        # Process columns left to right, propagating errors via Hinv
        # W_work: (R, rows_per_group, K)
        W_work = W.view(R, self.rows_per_group, K).clone()

        for i in range(K):
            w_i = W_work[:, :, i]  # (R, rows_per_group)
            d_i = Hinv[i, i]       # scalar

            # Quantize: zero out if masked
            q_i = w_i.clone()
            q_i[mask[:, i].unsqueeze(1).expand_as(q_i)] = 0.0

            # Error
            err_i = (w_i - q_i) / d_i  # (R, rows_per_group)

            # Write back
            W_work[:, :, i] = q_i

            # Propagate error to remaining columns
            if i + 1 < K:
                # Hinv[i, i+1:] is the propagation row
                prop = Hinv[i, i + 1:]  # (K - i - 1,)
                # W_work[:, :, i+1:] -= err_i.unsqueeze(-1) @ prop.unsqueeze(0)
                W_work[:, :, i + 1:] -= err_i.unsqueeze(-1) * prop.unsqueeze(0).unsqueeze(0)

        # Write back to W
        self.W.data.copy_(W_work.view(self.M, K))


# ─── Magnitude pruning baseline ─────────────────────────────────────────

def magnitude_prune(W0, view, block_shape, group_shape, nnz, device="cpu"):
    """Keep the `nnz` highest-norm blocks per group, zero the rest."""
    W = W0.clone()
    M, K = W.shape

    grid_shape = tuple(s // b for s, b in zip(view.shape, block_shape))
    group_grid = tuple(g // gg for g, gg in zip(grid_shape, group_shape))

    Bk_total = 1
    for g in grid_shape[1:]:
        Bk_total *= g

    col_idx = block_col_indices(view, block_shape, device=device)[0].reshape(Bk_total, -1)

    rows_per_group = group_shape[0] * block_shape[0]
    num_group_rows = group_grid[0]

    block_to_group, _ = build_group_mapping(grid_shape, group_shape, group_grid, device)
    num_groups_per_row = 1
    for gg in group_grid[1:]:
        num_groups_per_row *= gg
    blocks_per_group = 1
    for gs in group_shape:
        blocks_per_group *= gs

    # Block norms: (R, Bk_total)
    W_rows = W.view(num_group_rows, rows_per_group, K)
    ci = col_idx.unsqueeze(0).unsqueeze(0).expand(num_group_rows, rows_per_group, -1, -1)
    W_p = torch.gather(W_rows.unsqueeze(2).expand(-1, -1, Bk_total, -1), 3, ci)
    norms = W_p.norm(dim=(1, 3))

    pruned_mask = torch.zeros(num_group_rows, Bk_total, dtype=torch.bool, device=device)
    num_prune = blocks_per_group - nnz

    for g in range(num_groups_per_row):
        in_group = (block_to_group == g)
        group_norms = norms[:, in_group]
        _, bot_idx = group_norms.topk(num_prune, dim=1, largest=False)
        group_block_indices = torch.where(in_group)[0]
        prune_idx = group_block_indices[bot_idx]
        row_idx = torch.arange(num_group_rows, device=device).unsqueeze(1).expand_as(prune_idx)
        pruned_mask[row_idx, prune_idx] = True

    # Zero pruned blocks
    for r in range(num_group_rows):
        cols_to_zero = col_idx[pruned_mask[r]].reshape(-1).unique()
        if cols_to_zero.numel() > 0:
            row_s = r * rows_per_group
            row_e = row_s + rows_per_group
            W[row_s:row_e, cols_to_zero] = 0.0

    return W


# ─── Apply a pruning mask without compensation ──────────────────────────

def apply_pruned_pattern(W0, pruned, col_idx, num_group_rows, rows_per_group):
    """Zero out W0 using the same block-pruning pattern, no compensation."""
    W = W0.clone()
    pruned_cols_all = col_idx.unsqueeze(0).expand(num_group_rows, -1, -1)
    pruned_exp = pruned.unsqueeze(-1).expand_as(pruned_cols_all)
    for r in range(num_group_rows):
        cols = pruned_cols_all[r][pruned_exp[r]].unique()
        if cols.numel() > 0:
            row_s = r * rows_per_group
            row_e = row_s + rows_per_group
            W[row_s:row_e, cols] = 0.0
    return W


# ─── main ───────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    device = torch.device("cpu")

    M, K = 32, 16
    N = 128

    X = torch.randn(N, K, device=device)
    W = torch.nn.Parameter(torch.randn(M, K, device=device))
    W0 = W.data.clone()

    view_size = (32, 8, 2)
    view_stride = (16, 1, 8)
    block_shape = (1, 1, 2)
    group_shape = (1, 4, 1)
    nnz = 2

    # ── Verify view mapping ──
    view = View(W, shape=view_size, stride=view_stride)
    v = view.data
    for i in range(3):
        for j in range(3):
            for k in range(2):
                assert torch.isclose(v[i, j, k], W.data[i, j + 8 * k])
    print("View mapping verified: view[i,j,k] = W[i, j+8k]")

    col_idx = block_col_indices(view, block_shape, device=device)
    grid_shape = tuple(s // b for s, b in zip(view_size, block_shape))
    for j in range(8):
        cols = col_idx[0, j, 0].sort().values.tolist()
        assert cols == sorted([j, j + 8]), f"Block (0,{j},0): got {cols}"
    print(f"Block column indices verified, grid={grid_shape}")

    group_grid = tuple(g // gg for g, gg in zip(grid_shape, group_shape))
    print(f"Group grid: {group_grid}, blocks_per_group=4, nnz={nnz}")

    Y0 = X @ W0.T

    # ── 1. Structured OBS ──
    W_obs = torch.nn.Parameter(W0.clone())
    view_obs = View(W_obs, shape=view_size, stride=view_stride)
    solver = StructuredOBS(
        W=W_obs, view=view_obs, block_shape=block_shape,
        group_shape=group_shape, nnz=nnz, X=X, damp=1e-4,
    )
    solver.prune()
    loss_obs = ((X @ W_obs.data.T - Y0) ** 2).sum().item()

    # ── 2. SparseGPT ──
    W_sgpt = torch.nn.Parameter(W0.clone())
    view_sgpt = View(W_sgpt, shape=view_size, stride=view_stride)
    sgpt = SparseGPTBlockPruner(
        W=W_sgpt, view=view_sgpt, block_shape=block_shape,
        group_shape=group_shape, nnz=nnz, X=X, damp=1e-4,
    )
    sgpt.prune()
    loss_sgpt = ((X @ W_sgpt.data.T - Y0) ** 2).sum().item()

    # ── 3. Magnitude ──
    view_mag = View(torch.nn.Parameter(W0.clone()), shape=view_size, stride=view_stride)
    W_mag = magnitude_prune(W0, view_mag, block_shape, group_shape, nnz, device=device)
    loss_mag = ((X @ W_mag.T - Y0) ** 2).sum().item()

    # ── 4. Naive zero (OBS pattern, no compensation) ──
    W_naive = apply_pruned_pattern(
        W0, solver.pruned, solver.col_idx,
        solver.num_group_rows, solver.rows_per_group,
    )
    loss_naive = ((X @ W_naive.T - Y0) ** 2).sum().item()

    # ── 5. Naive zero (SparseGPT pattern, no compensation) ──
    W_sgpt_naive = apply_pruned_pattern(
        W0, sgpt.pruned, sgpt.col_idx,
        sgpt.num_group_rows, sgpt.rows_per_group,
    )
    loss_sgpt_naive = ((X @ W_sgpt_naive.T - Y0) ** 2).sum().item()

    # ── Report ──
    print(f"\n{'Method':<30} {'Loss':>12} {'vs Mag':>10}")
    print("-" * 54)
    methods = [
        ("Structured OBS", loss_obs),
        ("SparseGPT", loss_sgpt),
        ("Magnitude", loss_mag),
        ("Naive zero (OBS pattern)", loss_naive),
        ("Naive zero (SGPT pattern)", loss_sgpt_naive),
    ]
    for name, loss in methods:
        ratio = loss / loss_mag if loss_mag > 0 else float("inf")
        print(f"{name:<30} {loss:>12.2f} {ratio:>9.3f}x")

    # ── Sparsity verification ──
    print("\n--- Sparsity verification ---")
    for label, pruner in [("OBS", solver), ("SparseGPT", sgpt)]:
        ok = True
        for g in range(pruner.num_groups_per_row):
            in_group = (pruner.block_to_group == g)
            cnt = pruner.pruned[:, in_group].sum(dim=1)
            if not (cnt == pruner.num_prune).all():
                ok = False
        print(f"  {label}: {pruner.num_prune} pruned per group in every row: {ok}")

    # ── Analysis ──
    print("\n--- Analysis ---")
    if loss_obs <= loss_sgpt:
        pct = (1 - loss_obs / loss_sgpt) * 100
        print(f"OBS beats SparseGPT by {pct:.1f}% (lower loss)")
    else:
        pct = (loss_obs / loss_sgpt - 1) * 100
        print(f"SparseGPT beats OBS by {pct:.1f}%")

    if loss_obs <= loss_mag:
        pct = (1 - loss_obs / loss_mag) * 100
        print(f"OBS beats Magnitude by {pct:.1f}%")
    else:
        print(f"Magnitude beats OBS")

    # OBS compensation value
    comp_pct = (1 - loss_obs / loss_naive) * 100 if loss_naive > 0 else 0
    print(f"OBS compensation reduces loss by {comp_pct:.1f}% vs naive zero (same pattern)")

    sgpt_comp_pct = (1 - loss_sgpt / loss_sgpt_naive) * 100 if loss_sgpt_naive > 0 else 0
    print(f"SparseGPT compensation reduces loss by {sgpt_comp_pct:.1f}% vs naive zero (same pattern)")


if __name__ == "__main__":
    main()
