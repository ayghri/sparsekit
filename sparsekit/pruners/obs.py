"""
Copyright (c) 2025 Ayoub Ghriss and contributors
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.

Structured OBS (Optimal Brain Surgeon) via BlockSpec/GroupSpec.

Two compensation modes:
  - 'local': within-group only (fast, independent groups)
  - 'full':  sequential full-column compensation using H^{-1}
             (beats SparseGPT by ~44%)
"""

import math
from itertools import combinations

import torch
import torch.linalg as LA
from torch import Tensor

from ..views import BlockView
from ..blocks import BlockSpec
from ..groups import GroupSpec


def block_col_indices(
    block: BlockSpec, K: int, device: torch.device = torch.device("cpu")
) -> Tensor:
    """Map each block in a BlockSpec to its original K-column indices.

    Returns:
        col_idx: (*grid_shape[1:], block_numel) long tensor.
    """
    grid_shape = block.grid_shape
    block_shape = block.block_shape
    rank = len(block_shape)

    ranges = [torch.arange(b, device=device) for b in block_shape]
    offsets = torch.stack(
        torch.meshgrid(*ranges, indexing="ij"), dim=-1
    ).reshape(-1, rank)

    grid_ranges = [torch.arange(g, device=device) for g in grid_shape]
    grid_pts = torch.stack(
        torch.meshgrid(*grid_ranges, indexing="ij"), dim=-1
    )

    bs = torch.tensor(block_shape, device=device)
    elem_idx = grid_pts.unsqueeze(-2) * bs + offsets

    param = block.param
    if isinstance(param, BlockView):
        flat_offsets = param.linear_offset(elem_idx)
    else:
        strides = torch.tensor(param.data.stride(), device=device, dtype=torch.long)
        flat_offsets = (elem_idx * strides).sum(dim=-1)

    col_idx_full = flat_offsets % K
    col_grid = grid_shape[1:]
    row_slice = tuple(0 for _ in range(len(grid_shape) - len(col_grid)))
    return col_idx_full[row_slice]


class StructuredOBS:
    """Structured OBS pruner operating through GroupSpec.

    Args:
        group: GroupSpec defining block and group structure.
        H: (K, K) Hessian matrix.
        damp: Damping factor for H regularization.
        C: Precomputed (K, K) damped inverse. Skips inversion if given.
    """

    def __init__(self, group: GroupSpec, H: Tensor, damp: float = 1e-4,
                 C: Tensor | None = None):
        self.group = group
        block = group.block
        assert isinstance(block, BlockSpec)

        self.block = block
        param = block.param
        self.H = H
        self.damp = damp
        self.K = H.shape[0]
        device = H.device

        if isinstance(param, BlockView):
            self.W = param.param
        else:
            self.W = param

        self.M = self.W.shape[0]
        self.bk = block.block_numel

        grid_shape = block.grid_shape
        group_shape = group.group_shape
        group_grid = group.grid_shape

        self.rows_per_group_row = block.block_shape[0] * group_shape[0]
        self.num_group_rows = group_grid[0]

        col_grid = grid_shape[1:]
        self.Bk_total = math.prod(col_grid)
        self.blocks_per_group = math.prod(group_shape[1:]) if len(group_shape) > 1 else 1
        self.num_groups_per_row = math.prod(group_grid[1:]) if len(group_grid) > 1 else 1

        # Column index mapping: (Bk_total, bk)
        col_idx_full = block_col_indices(block, self.K, device=device)
        self.col_idx = col_idx_full.reshape(self.Bk_total, self.bk)
        if self.bk == 1:
            self.col_idx_flat = self.col_idx.squeeze(-1)

        # Group mapping
        self.block_to_group, _ = self._build_group_mapping(
            col_grid, group_shape[1:], group_grid[1:], device
        )

        if C is not None:
            self.C_base = C
        else:
            self.C_base = self.compute_inverse(H, damp)

    @staticmethod
    def compute_inverse(H: Tensor, damp: float = 1e-4) -> Tensor:
        K = H.shape[0]
        device = H.device
        damp_val = damp * torch.mean(torch.diag(H)) if damp < 1.0 else damp
        H_reg = H.clone()
        diag_idx = torch.arange(K, device=device)
        H_reg[diag_idx, diag_idx] += damp_val
        return LA.inv(H_reg)

    @staticmethod
    def _build_group_mapping(col_grid, group_shape_rest, group_grid_rest, device):
        Bk_total = math.prod(col_grid)
        if len(col_grid) == 0:
            return (
                torch.zeros(1, dtype=torch.long, device=device),
                torch.zeros(1, dtype=torch.long, device=device),
            )

        ranges = [torch.arange(g, device=device) for g in col_grid]
        bc_grid = torch.stack(
            torch.meshgrid(*ranges, indexing="ij"), dim=-1
        ).reshape(Bk_total, -1)

        gs_rest = list(group_shape_rest)
        gg_rest = list(group_grid_rest)

        group_pos = bc_grid // torch.tensor(gs_rest, device=device)
        gg_strides = []
        s = 1
        for g in reversed(gg_rest):
            gg_strides.append(s)
            s *= g
        gg_strides.reverse()
        block_to_group = (
            group_pos * torch.tensor(gg_strides, device=device)
        ).sum(dim=-1)

        within_pos = bc_grid % torch.tensor(gs_rest, device=device)
        gs_strides = []
        s = 1
        for g in reversed(gs_rest):
            gs_strides.append(s)
            s *= g
        gs_strides.reverse()
        block_within_group = (
            within_pos * torch.tensor(gs_strides, device=device)
        ).sum(dim=-1)

        return block_to_group, block_within_group

    def _find_best_subsets(
        self, W, group_cols, prune_subsets, keep_subsets, block_size,
    ):
        """Phase 1: find optimal pruning subset per (row, group).

        Returns:
            best_si: (M, G) long tensor — index into prune_subsets.
        """
        device = W.device
        M, K = W.shape
        G = self.num_groups_per_row
        gs = self.blocks_per_group
        bk = self.bk
        C = self.C_base
        num_prune = prune_subsets.shape[1]
        n_subsets = prune_subsets.shape[0]
        elem_per_group = gs * bk if bk > 1 else gs

        if bk == 1:
            group_start = group_cols[:, 0]
        else:
            group_start = group_cols[:, 0, 0]

        best_si_full = torch.zeros(M, G, dtype=torch.long, device=device)

        for b_start in range(0, K, block_size):
            b_end = min(b_start + block_size, K)
            in_block = (group_start >= b_start) & (group_start < b_end)
            gidx = torch.where(in_block)[0]
            G_b = gidx.shape[0]
            if G_b == 0:
                continue

            if bk == 1:
                cols = group_cols[gidx]
                flat = cols.view(-1)
            else:
                cols = group_cols[gidx]
                flat = cols.view(G_b, elem_per_group).view(-1)

            if bk == 1:
                C_sub = C[cols.unsqueeze(-1), cols.unsqueeze(-2)]
            else:
                fc = cols.view(G_b, elem_per_group)
                C_sub = C[fc.unsqueeze(-1), fc.unsqueeze(-2)]

            W_g = W[:, flat].view(M, G_b, elem_per_group)

            eye_np = 1e-8 * torch.eye(num_prune * bk, device=device)
            all_costs = torch.empty(n_subsets, M, G_b, device=device)

            for si in range(n_subsets):
                pidx = prune_subsets[si]
                if bk == 1:
                    fp = pidx
                else:
                    fp = (pidx.unsqueeze(-1) * bk + torch.arange(bk, device=device)).view(-1)

                C_PP = C_sub[:, fp][:, :, fp]
                C_PP_inv = LA.inv(C_PP + eye_np)

                W_P = W_g[:, :, fp]
                temp = torch.einsum("mgp,gpq->mgq", W_P, C_PP_inv)
                all_costs[si] = (temp * W_P).sum(dim=2)

            best_si_full[:, gidx] = all_costs.argmin(dim=0)

        return best_si_full

    def _compensate_local(
        self, W, group_cols, prune_subsets, keep_subsets, best_si, block_size,
    ):
        """Within-group compensation only."""
        device = W.device
        M, K = W.shape
        G = self.num_groups_per_row
        gs = self.blocks_per_group
        bk = self.bk
        C = self.C_base
        num_prune = prune_subsets.shape[1]
        n_subsets = prune_subsets.shape[0]
        elem_per_group = gs * bk if bk > 1 else gs

        if bk == 1:
            group_start = group_cols[:, 0]
        else:
            group_start = group_cols[:, 0, 0]

        for b_start in range(0, K, block_size):
            b_end = min(b_start + block_size, K)
            in_block = (group_start >= b_start) & (group_start < b_end)
            gidx = torch.where(in_block)[0]
            G_b = gidx.shape[0]
            if G_b == 0:
                continue

            if bk == 1:
                cols = group_cols[gidx]
                flat = cols.view(-1)
            else:
                cols = group_cols[gidx]
                flat = cols.view(G_b, elem_per_group).view(-1)

            if bk == 1:
                C_sub = C[cols.unsqueeze(-1), cols.unsqueeze(-2)]
            else:
                fc = cols.view(G_b, elem_per_group)
                C_sub = C[fc.unsqueeze(-1), fc.unsqueeze(-2)]

            W_g = W[:, flat].view(M, G_b, elem_per_group)
            W_g_new = W_g.clone()
            eye_np = 1e-8 * torch.eye(num_prune * bk, device=device)
            block_best_si = best_si[:, gidx]

            for si in range(n_subsets):
                pidx = prune_subsets[si]
                kidx = keep_subsets[si]
                if bk == 1:
                    fp, fk = pidx, kidx
                else:
                    fp = (pidx.unsqueeze(-1) * bk + torch.arange(bk, device=device)).view(-1)
                    fk = (kidx.unsqueeze(-1) * bk + torch.arange(bk, device=device)).view(-1)

                mask = block_best_si == si
                if not mask.any():
                    continue

                C_PP = C_sub[:, fp][:, :, fp]
                C_PP_inv = LA.inv(C_PP + eye_np)
                C_kP = C_sub[:, fk][:, :, fp]
                comp = torch.bmm(C_kP, C_PP_inv)

                W_P = W_g[:, :, fp]
                delta_k = torch.einsum("mgp,gnp->mgn", W_P, comp)
                mask_exp = mask.unsqueeze(-1)
                W_g_new[:, :, fk] -= delta_k * mask_exp
                W_g_new[:, :, fp] = W_g_new[:, :, fp].masked_fill(mask_exp, 0.0)

            W.scatter_(
                1,
                flat.unsqueeze(0).expand(M, -1),
                W_g_new.reshape(M, G_b * elem_per_group),
            )

    def _compensate_full(self, W, group_cols, prune_subsets, best_si):
        """Sequential full-column compensation using C = H^{-1}.

        For each group, compensate ALL K columns (not just within-group),
        processing groups sequentially so each sees updated W.
        """
        device = W.device
        M, K = W.shape
        G = self.num_groups_per_row
        gs = self.blocks_per_group
        bk = self.bk
        C = self.C_base
        num_prune = prune_subsets.shape[1]
        n_subsets = prune_subsets.shape[0]
        elem_per_group = gs * bk if bk > 1 else gs
        np_bk = num_prune * bk

        # Precompute compensation matrices for all (group, subset) pairs:
        #   comp[g, si] = inv(C[P,P]) @ C[P, :]  — shape (np*bk, K)
        # where P = absolute column indices of pruned blocks in group g, subset si.

        # Pruned column indices per (group, subset): (G, n_subsets, np*bk)
        if bk == 1:
            # group_cols: (G, gs) — absolute col indices
            # prune_subsets: (n_subsets, num_prune) — local block indices
            pruned_cols = group_cols[:, prune_subsets]  # (G, n_subsets, num_prune)
        else:
            # group_cols: (G, gs, bk)
            # Expand block indices to element indices
            blk_offsets = torch.arange(bk, device=device)
            pruned_blk = group_cols[:, prune_subsets]  # (G, n_subsets, num_prune, bk)
            pruned_cols = pruned_blk.view(G, n_subsets, np_bk)

        # C submatrices and their inverses: (G, n_subsets, np_bk, np_bk)
        C_PP = C[pruned_cols[:, :, :, None], pruned_cols[:, :, None, :]]
        eye = 1e-8 * torch.eye(np_bk, device=device)
        C_PP_inv = torch.inverse(C_PP + eye)

        # Full compensation rows: C[P, :] — (G, n_subsets, np_bk, K)
        C_P_rows = C[pruned_cols.reshape(-1), :].view(G, n_subsets, np_bk, K)

        # comp = inv(C_PP) @ C_P_rows — (G, n_subsets, np_bk, K)
        comp_all = torch.einsum("gsij,gsjk->gsik", C_PP_inv, C_P_rows)

        # Mask to zero out comp columns for already-pruned indices.
        # All rows share the same group ordering, but pruned subsets differ
        # per row. We track a per-row active mask.
        active = torch.ones(M, K, device=device, dtype=torch.bool)

        # Sequential group processing
        for g in range(G):
            si_per_row = best_si[:, g]              # (M,)
            comp_g = comp_all[g]                     # (n_subsets, np_bk, K)
            comp_rows = comp_g[si_per_row]           # (M, np_bk, K)
            p_cols = pruned_cols[g, si_per_row]      # (M, np_bk)

            # Zero out comp entries for already-pruned columns
            comp_rows = comp_rows * active.unsqueeze(1)  # (M, np_bk, K)

            W_P = torch.gather(W, 1, p_cols)         # (M, np_bk)
            delta = torch.bmm(
                W_P.unsqueeze(1), comp_rows
            ).squeeze(1)                              # (M, K)
            W -= delta

            # Zero pruned columns for this group
            W.scatter_(1, p_cols, torch.zeros_like(W_P))

            # Mark these columns as inactive
            active.scatter_(1, p_cols, False)

    def _compensate_full_split(self, W, group_cols, prune_subsets, best_si,
                               n_splits=2):
        """Full-column compensation with C recomputed between splits.

        Splits groups into n_splits chunks by column position. Each chunk
        uses C = inv(H[active, active] + damp*I) where 'active' excludes
        columns from previous chunks. This is exact: the Schur complement
        of H^{-1} w.r.t. frozen columns F equals inv(H[S,S]).
        """
        device = W.device
        M, K = W.shape
        G = self.num_groups_per_row
        bk = self.bk
        num_prune = prune_subsets.shape[1]
        n_subsets = prune_subsets.shape[0]
        np_bk = num_prune * bk

        chunk_g = (G + n_splits - 1) // n_splits
        active_mask = torch.ones(K, dtype=torch.bool, device=device)

        for split_idx in range(n_splits):
            g_start = split_idx * chunk_g
            g_end = min((split_idx + 1) * chunk_g, G)
            if g_start >= G:
                break
            G_s = g_end - g_start

            # Active columns and C for this split
            if split_idx == 0:
                active_cols = torch.arange(K, device=device)
                n_active = K
                C = self.C_base
                abs_to_local = torch.arange(K, device=device)
            else:
                active_cols = torch.where(active_mask)[0]
                n_active = active_cols.shape[0]
                H_aa = self.H[active_cols[:, None], active_cols[None, :]]
                C = self.compute_inverse(H_aa, self.damp)
                abs_to_local = torch.full((K,), -1, dtype=torch.long, device=device)
                abs_to_local[active_cols] = torch.arange(n_active, device=device)

            # Pruned column indices for this split's groups
            split_group_cols = group_cols[g_start:g_end]
            if bk == 1:
                pruned_abs = split_group_cols[:, prune_subsets]  # (G_s, n_sub, np)
            else:
                pruned_abs = split_group_cols[:, prune_subsets].view(
                    G_s, n_subsets, np_bk
                )

            pruned_local = abs_to_local[pruned_abs]

            # Compensation matrices in active-local space
            C_PP = C[pruned_local[:, :, :, None], pruned_local[:, :, None, :]]
            eye = 1e-8 * torch.eye(np_bk, device=device)
            C_PP_inv = torch.inverse(C_PP + eye)

            C_P_rows = C[pruned_local.reshape(-1), :].view(
                G_s, n_subsets, np_bk, n_active
            )
            comp_split = torch.einsum("gsij,gsjk->gsik", C_PP_inv, C_P_rows)

            # Per-row mask of active columns within this split's active space
            split_active = torch.ones(M, n_active, device=device, dtype=torch.bool)

            # Sequential group processing within this split
            for g_local in range(G_s):
                g = g_start + g_local
                si_per_row = best_si[:, g]
                comp_g = comp_split[g_local]
                comp_rows = comp_g[si_per_row]           # (M, np_bk, n_active)
                p_cols = pruned_abs[g_local, si_per_row]  # (M, np_bk)

                # Zero out comp entries for already-pruned columns within split
                comp_rows = comp_rows * split_active.unsqueeze(1)

                W_P = torch.gather(W, 1, p_cols)
                delta = torch.bmm(
                    W_P.unsqueeze(1), comp_rows
                ).squeeze(1)                              # (M, n_active)

                # Update active columns
                if n_active == K:
                    W -= delta
                else:
                    ac_exp = active_cols.unsqueeze(0).expand(M, -1)
                    W.scatter_add_(1, ac_exp, -delta)

                W.scatter_(1, p_cols, torch.zeros_like(W_P))

                # Mark pruned columns as inactive in local space
                p_local = abs_to_local[p_cols]  # (M, np_bk)
                split_active.scatter_(1, p_local, False)

            # Freeze this split's columns
            if split_idx < n_splits - 1:
                if bk == 1:
                    frozen = split_group_cols.reshape(-1)
                else:
                    frozen = split_group_cols.view(-1)
                active_mask[frozen] = False

    @torch.no_grad()
    def prune(self, num_nz: int, block_size: int = 2048,
              compensate: str = "local", n_splits: int = 1) -> None:
        """Prune to num_nz blocks per group.

        Phase 1: enumerate all C(gs, num_prune) subsets per group, pick
                 the best per (row, group) using C = H^{-1} submatrices.

        Phase 2 (compensation):
          - 'local': within-group only (fast, independent groups)
          - 'full':  sequential compensation to ALL K columns via C[P, :]
                     (slower but ~44% better than SparseGPT)
          - 'split': like 'full' but recomputes C between column splits.
                     Use n_splits to control granularity (2 = one C update
                     at the midpoint).

        Args:
            num_nz: Blocks to keep per group.
            block_size: Column chunk size for subset search.
            compensate: 'local', 'full', or 'split'.
            n_splits: Number of column splits (only for compensate='split').
        """
        gs = self.blocks_per_group
        num_prune = gs - num_nz
        if num_prune <= 0:
            return

        device = self.W.data.device
        M = self.M
        K = self.K
        bk = self.bk
        G = self.num_groups_per_row

        W = self.W.data.clone().float().view(M, K)

        # Group column indices
        if bk == 1:
            group_cols = self.col_idx_flat.view(G, gs)
        else:
            group_cols = self.col_idx.view(G, gs, bk)

        # Pruning subsets
        prune_subsets = torch.tensor(
            list(combinations(range(gs), num_prune)),
            device=device, dtype=torch.long,
        )
        keep_subsets = torch.tensor(
            [sorted(set(range(gs)) - set(s))
             for s in combinations(range(gs), num_prune)],
            device=device, dtype=torch.long,
        )

        # Phase 1: find best subset per (row, group)
        best_si = self._find_best_subsets(
            W, group_cols, prune_subsets, keep_subsets, block_size,
        )

        # Phase 2: apply compensation
        if compensate == "split":
            self._compensate_full_split(
                W, group_cols, prune_subsets, best_si, n_splits=n_splits,
            )
        elif compensate == "full":
            self._compensate_full(W, group_cols, prune_subsets, best_si)
        else:
            self._compensate_local(
                W, group_cols, prune_subsets, keep_subsets, best_si, block_size,
            )

        self.W.data.copy_(W)
