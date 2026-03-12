# Copyright (c) 2025 Anonymous Authors
# Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Structured OBS (Optimal Brain Surgeon) via BlockSpec/GroupSpec.

Compensation modes: ``local``, ``full``, ``split``, ``interleaved``.
"""

import math
from itertools import combinations

import torch
import torch.linalg as LA
from torch import Tensor

from ..view import View
from ..block import BlockSpec
from ..group import GroupSpec


def _block_flat_offsets(
    block: BlockSpec, device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Compute flat storage offsets for every element of every block.

    Returns:
        flat_offsets: ``(*grid_shape, block_numel)`` long tensor.
    """
    grid_shape = block.grid_shape
    block_shape = block.shape
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

    param = block.view
    if isinstance(param, View):
        return param.linear_offset(elem_idx)
    else:
        strides = torch.tensor(param.data.stride(), device=device, dtype=torch.long)
        return (elem_idx * strides).sum(dim=-1)


def block_col_indices(
    block: BlockSpec, K: int, device: torch.device = torch.device("cpu")
) -> Tensor:
    """Map each block in a BlockSpec to its original K-column indices.

    Returns:
        col_idx: ``(*grid_shape[1:], block_numel)`` long tensor.
    """
    flat_offsets = _block_flat_offsets(block, device)
    col_idx_full = flat_offsets % K
    col_grid = block.grid_shape[1:]
    row_slice = tuple(0 for _ in range(len(block.grid_shape) - len(col_grid)))
    return col_idx_full[row_slice]


def block_param_rc(
    block: BlockSpec, K: int, device: torch.device = torch.device("cpu")
) -> tuple[Tensor, Tensor]:
    """Map each block to ``(param_row, param_col)`` per element.

    Unlike :func:`block_col_indices` this returns the **full** grid
    (including the row dimension) and both row and column indices.

    Returns:
        row_idx: ``(*grid_shape, block_numel)`` long tensor.
        col_idx: ``(*grid_shape, block_numel)`` long tensor.
    """
    flat_offsets = _block_flat_offsets(block, device)
    return flat_offsets // K, flat_offsets % K


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
        param = block.view
        self.H = H
        self.damp = damp
        self.K = H.shape[0]
        device = H.device

        if isinstance(param, View):
            self.W = param.param
        else:
            self.W = param

        self.M = self.W.shape[0]
        self.bk = block.numel()

        grid_shape = block.grid_shape
        group_shape = group.shape
        group_grid = group.grid_shape

        self.rows_per_group_row = block.shape[0] * group_shape[0]
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

        # Row coupling detection: do blocks within the same group span
        # different param rows?  (e.g. block-16 with 8-row coupling)
        row_full, _ = block_param_rc(block, self.K, device=device)
        row_slice = tuple(0 for _ in range(len(grid_shape) - len(col_grid)))
        row_cg = row_full[row_slice].reshape(self.Bk_total, self.bk)

        sort_perm = torch.argsort(self.block_to_group)
        gs = self.blocks_per_group
        G = self.num_groups_per_row
        sorted_row_base = row_cg[sort_perm].view(G, gs, self.bk)
        block_rows = sorted_row_base[:, :, 0]
        self.row_coupled = bool(
            (block_rows.max(1).values - block_rows.min(1).values > 0).any()
        )

        if self.row_coupled:
            self.group_sort_perm = sort_perm
            self.full_row_idx = row_full  # (*grid_shape, bk)
            self.group_col_map = self.col_idx[sort_perm].view(G, gs, self.bk)

        if C is not None:
            self.C_base = C
        else:
            self.C_base = self.compute_inverse(H, damp)

    @staticmethod
    def compute_inverse(H: Tensor, damp: float = 1e-4) -> Tensor:
        """Compute damped inverse of the Hessian matrix.

        Args:
            H: (K, K) symmetric positive semi-definite Hessian.
            damp: Damping factor. If < 1.0, scaled by mean diagonal;
                otherwise used as absolute value.

        Returns:
            (K, K) inverse of the damped Hessian: (H + damp * I)^{-1}.
        """
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
        C=None,
    ):
        """Phase 1: find optimal pruning subset per (row, group).

        Args:
            C: Optional inverse matrix. Uses self.C_base if not provided.
               When provided, group_cols must be in C's coordinate space.

        Returns:
            best_si: (M, G_input) long tensor — index into prune_subsets.
        """
        device = W.device
        M, K = W.shape
        gs = self.blocks_per_group
        bk = self.bk
        if C is None:
            C = self.C_base
        num_prune = prune_subsets.shape[1]
        n_subsets = prune_subsets.shape[0]
        elem_per_group = gs * bk if bk > 1 else gs
        G = group_cols.shape[0]

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

        # Sequential group processing.
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

            split_active = torch.ones(M, n_active, device=device, dtype=torch.bool)

            # Sequential group processing within this split
            for g_local in range(G_s):
                g = g_start + g_local
                si_per_row = best_si[:, g]
                comp_rows = comp_split[g_local][si_per_row]  # (M, np_bk, n_active)
                p_cols = pruned_abs[g_local, si_per_row]     # (M, np_bk)

                comp_rows = comp_rows * split_active.unsqueeze(1)

                W_P = torch.gather(W, 1, p_cols)
                delta = torch.bmm(
                    W_P.unsqueeze(1), comp_rows
                ).squeeze(1)                                  # (M, n_active)

                if n_active == K:
                    W -= delta
                else:
                    ac_exp = active_cols.unsqueeze(0).expand(M, -1)
                    W.scatter_add_(1, ac_exp, -delta)

                W.scatter_(1, p_cols, torch.zeros_like(W_P))

                p_local = abs_to_local[p_cols]
                split_active.scatter_(1, p_local, False)

            # Freeze this split's columns
            if split_idx < n_splits - 1:
                if bk == 1:
                    frozen = split_group_cols.reshape(-1)
                else:
                    frozen = split_group_cols.view(-1)
                active_mask[frozen] = False

    def _interleaved(self, W, group_cols, prune_subsets, keep_subsets,
                      n_splits=16, block_size=2048):
        """Interleaved selection + compensation with C recomputed between splits.

        Unlike split compensation (which selects masks once then compensates),
        this re-selects which columns to prune at each split using the updated C.
        Uses a single shared C (not per-row), so O(K²) memory.

        Key: _find_best_subsets is called with local-space group_cols and the
        recomputed C, so mask selection uses the Schur-updated inverse.
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

            # Recompute C on active columns
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

            # Map group cols to local active-column space for C indexing
            split_group_cols = group_cols[g_start:g_end]
            if bk == 1:
                split_group_cols_local = abs_to_local[split_group_cols]
            else:
                split_group_cols_local = abs_to_local[split_group_cols]

            # Project W to active columns for subset scoring
            W_active = W[:, active_cols]

            # Re-select masks using updated C and local-space group cols
            best_si_split = self._find_best_subsets(
                W_active, split_group_cols_local, prune_subsets, keep_subsets,
                block_size, C=C,
            )

            # Build compensation matrices in local space
            if bk == 1:
                pruned_local = split_group_cols_local[:, prune_subsets]
                pruned_abs = split_group_cols[:, prune_subsets]
            else:
                pruned_local = abs_to_local[
                    split_group_cols[:, prune_subsets].view(G_s, n_subsets, np_bk)
                ]
                pruned_abs = split_group_cols[:, prune_subsets].view(
                    G_s, n_subsets, np_bk
                )

            C_PP = C[pruned_local[:, :, :, None], pruned_local[:, :, None, :]]
            eye = 1e-8 * torch.eye(np_bk, device=device)
            C_PP_inv = torch.inverse(C_PP + eye)

            C_P_rows = C[pruned_local.reshape(-1), :].view(
                G_s, n_subsets, np_bk, n_active
            )
            comp_split = torch.einsum("gsij,gsjk->gsik", C_PP_inv, C_P_rows)

            split_active = torch.ones(M, n_active, device=device, dtype=torch.bool)

            for g_local in range(G_s):
                si_per_row = best_si_split[:, g_local]
                comp_rows = comp_split[g_local][si_per_row]
                p_cols = pruned_abs[g_local, si_per_row]

                comp_rows = comp_rows * split_active.unsqueeze(1)

                W_P = torch.gather(W, 1, p_cols)
                delta = torch.bmm(
                    W_P.unsqueeze(1), comp_rows
                ).squeeze(1)

                if n_active == K:
                    W -= delta
                else:
                    ac_exp = active_cols.unsqueeze(0).expand(M, -1)
                    W.scatter_add_(1, ac_exp, -delta)

                W.scatter_(1, p_cols, torch.zeros_like(W_P))

                p_local = abs_to_local[p_cols]
                split_active.scatter_(1, p_local, False)

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
          - 'interleaved': re-selects masks AND compensates at each split
                     using recomputed C. Single shared C (O(K²) memory).

        Args:
            num_nz: Blocks to keep per group.
            block_size: Column chunk size for subset search.
            compensate: 'local', 'full', 'split', or 'interleaved'.
            n_splits: Number of column splits (for 'split'/'interleaved').
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

        if compensate == "interleaved":
            self._interleaved(
                W, group_cols, prune_subsets, keep_subsets,
                n_splits=n_splits, block_size=block_size,
            )
            self.W.data.copy_(W.view_as(self.W.data))
            return

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

        self.W.data.copy_(W.view_as(self.W.data))

    # ── True OBS (per-row C with Schur updates) ──────────────────────

    def _prescore_groups_order(self, W, col_map, prune_subsets,
                               sub_to_cols, np_cols):
        """Pre-score groups by OBS cost using shared C_base.

        Returns group indices sorted by descending total cost (highest-cost
        groups first, so they are processed while C is most accurate).
        """
        device = W.device
        M = W.shape[0]
        G = col_map.shape[0]
        n_subs = prune_subsets.shape[0]
        C = self.C_base
        eye_np = 1e-8 * torch.eye(np_cols, device=device)
        scores = torch.zeros(G, device=device)

        for g in range(G):
            cols = col_map[g]
            C_g = C[cols][:, cols]
            W_g = W[:, cols]
            best_cost = torch.full((M,), float("inf"), device=device)
            for si in range(n_subs):
                co = sub_to_cols[si]
                C_PP_inv = torch.inverse(C_g[co][:, co] + eye_np)
                w_P = W_g[:, co]
                cost = (w_P @ C_PP_inv * w_P).sum(1)
                better = cost < best_cost
                best_cost[better] = cost[better]
            scores[g] = best_cost.sum()

        return torch.argsort(scores, descending=True)

    # ── Row-coupled True OBS ─────────────────────────────────────────

    def _prescore_coupled_groups(self, Wc, local_row_map, n_vr, eye_bk):
        """Pre-score row-coupled groups using shared C_base.

        Returns group indices sorted descending by OBS cost.
        Fully vectorized over all G groups.
        """
        gcm = self.group_col_map  # (G, gs, bk)
        G, gs, bk = gcm.shape
        device = Wc.device
        C = self.C_base  # (K, K)

        # C_PP for all groups via flat indexing (single gather)
        cols = gcm[:, 0, :]  # (G, bk) — column indices per group
        K = C.shape[0]
        ci = cols.unsqueeze(2).expand(-1, -1, bk)  # (G, bk, bk)
        cj = cols.unsqueeze(1).expand(-1, bk, -1)  # (G, bk, bk)
        flat_idx = ci * K + cj  # (G, bk, bk)
        C_PP = C.reshape(-1)[flat_idx.reshape(-1)].view(G, bk, bk)

        C_PP_inv = torch.linalg.inv(C_PP.float() + eye_bk)  # (G, bk, bk)

        # Score all (n_vr, G, gs) items vectorized
        block_scores = torch.empty(n_vr, G, gs, device=device)
        for b in range(gs):
            lr = local_row_map[:, :, b]  # (n_vr, G)
            lr_exp = lr.unsqueeze(2).expand(-1, -1, bk)  # (n_vr, G, bk)
            cols_exp = cols.unsqueeze(0).expand(n_vr, -1, -1)  # (n_vr, G, bk)
            w_blk = Wc[lr_exp.reshape(-1), cols_exp.reshape(-1)].view(
                n_vr, G, bk
            )
            # w^T C_PP^{-1} w: einsum over (n_vr, G, bk) x (G, bk, bk)
            temp = torch.einsum("ngb,gbc->ngc", w_blk, C_PP_inv)
            block_scores[:, :, b] = (temp * w_blk).sum(2)

        scores = block_scores.min(dim=2).values.sum(dim=0)  # (G,)
        return torch.argsort(scores, descending=True)

    @torch.no_grad()
    def _prune_true_obs_coupled(self, num_nz, ng, chunk_size, order,
                                progress_fn):
        """True OBS for row-coupled groups (blocks span different param rows).

        Groups processed sequentially; vectorized across view-rows using
        flat indexing (no per-element Python loops).
        """
        gs = self.blocks_per_group
        bk = self.bk
        G = self.num_groups_per_row
        K = self.K
        M = self.M
        KK = K * K
        device = self.C_base.device
        num_prune = gs - num_nz

        num_gr = self.num_group_rows
        gcm = self.group_col_map     # (G, gs, bk)
        gsp = self.group_sort_perm   # (Bk_total,)

        eye_bk = 1e-4 * torch.eye(bk, device=device)

        W = self.W.data.clone().float().view(M, K)

        n_chunks = (num_gr + chunk_size - 1) // chunk_size

        for ci in range(n_chunks):
            vr0 = ci * chunk_size
            vr1 = min(vr0 + chunk_size, num_gr)
            n_vr = vr1 - vr0

            if progress_fn:
                progress_fn(
                    f"chunk {ci + 1}/{n_chunks} ({vr0}/{num_gr} view-rows)"
                )

            # Build chunk row map: (n_vr, G, gs, bk)
            chunk_row_map = torch.empty(
                n_vr, G, gs, bk, device=device, dtype=torch.long,
            )
            for vr_local in range(n_vr):
                row_cg = self.full_row_idx[vr0 + vr_local].reshape(
                    self.Bk_total, bk
                )
                chunk_row_map[vr_local] = row_cg[gsp].view(G, gs, bk)

            unique_rows = chunk_row_map[:, :, :, 0].reshape(-1).unique()
            B_param = unique_rows.shape[0]
            p2l = torch.full((M,), -1, device=device, dtype=torch.long)
            p2l[unique_rows] = torch.arange(B_param, device=device)

            local_row_map = p2l[chunk_row_map[:, :, :, 0]]  # (n_vr, G, gs)

            Wc = W[unique_rows].clone()
            C = torch.empty(
                B_param, K, K, device=device, dtype=torch.float16,
            )
            C[:] = self.C_base.half()
            pruned_mask = torch.ones(B_param, K, device=device)

            # Flat views for gather indexing
            C_flat = C.reshape(-1)
            Wc_flat = Wc.reshape(-1)

            # Pre-compute k_range for C_col extraction
            k_range = torch.arange(K, device=device)

            # Group ordering
            if order == "largest_first":
                if progress_fn:
                    progress_fn("Pre-scoring groups...")
                group_order = self._prescore_coupled_groups(
                    Wc, local_row_map, n_vr, eye_bk,
                )
            else:
                group_order = torch.arange(G, device=device)

            # ── Sequential scoring, compensation, and Schur ──
            arange_vr = torch.arange(n_vr, device=device)

            for gi in range(G):
                g = group_order[gi]
                cols = gcm[g, 0, :]  # (bk,)
                rows_g = local_row_map[:, g, :]  # (n_vr, gs)

                # Score via fancy indexing (fewer kernels)
                N_score = n_vr * gs
                sc_rows = rows_g.reshape(N_score)
                C_PP = C[:, cols, :][:, :, cols][sc_rows]
                C_PP_inv = torch.linalg.inv(
                    C_PP.float() + eye_bk
                )
                W_blk = Wc[sc_rows][:, cols]
                temp = torch.bmm(
                    W_blk.unsqueeze(1), C_PP_inv
                ).squeeze(1)
                scores = (temp * W_blk).sum(1).view(n_vr, gs)

                _, prune_bi = scores.topk(
                    num_prune, dim=1, largest=False,
                )

                for pi in range(num_prune):
                    pb = prune_bi[:, pi]
                    flat_rows = rows_g[arange_vr, pb]

                    # Reuse C_PP_inv
                    si = arange_vr * gs + pb
                    Cinv = C_PP_inv[si]

                    # C_col from current C
                    C_col = C[:, :, cols][flat_rows]  # (n_vr, K, bk)

                    # comp, delta
                    comp = torch.bmm(C_col.float(), Cinv)
                    w_P = Wc[flat_rows][:, cols]
                    delta = torch.bmm(
                        comp, w_P.unsqueeze(2)
                    ).squeeze(2)

                    # W update & zero
                    Wc.index_add_(0, flat_rows, -delta)
                    Wc[flat_rows.unsqueeze(1).expand(
                        -1, bk
                    ), cols.unsqueeze(0).expand(n_vr, -1)] = 0.0

                    # Schur: fp16, fused addmm over n_vr
                    comp_h = comp.half()
                    for vr in range(n_vr):
                        r = flat_rows[vr]
                        torch.addmm(
                            C[r], comp_h[vr], C_col[vr].T,
                            beta=1, alpha=-1, out=C[r],
                        )

                    # Mask update
                    pruned_mask[flat_rows.unsqueeze(1).expand(
                        -1, bk
                    ), cols.unsqueeze(0).expand(n_vr, -1)] = 0.0

            Wc *= pruned_mask
            W[unique_rows] = Wc
            del C, C_flat

        if progress_fn:
            progress_fn("")

        self.W.data.copy_(W.view_as(self.W.data))

    @torch.no_grad()
    def prune_true_obs(self, num_nz: int, ng: int = 64,
                       chunk_size: int = 16,
                       order: str = "left_to_right",
                       scoring: str = "independent",
                       c_dtype=None,
                       progress_fn=None) -> None:
        """Per-row True OBS with Schur complement updates.

        Each row maintains its own C = inv(H), updated via Schur complement
        after pruning.  Processes ``ng`` groups simultaneously per batch.

        Args:
            num_nz: Blocks to keep per group.
            ng: Number of groups to process per batch.
            chunk_size: Rows to process simultaneously.
            order: ``"left_to_right"`` or ``"largest_first"``.
            scoring: ``"joint"`` (enumerate subsets) or ``"independent"``
                (per-element w²/diag(C) + topk).
            c_dtype: Dtype for per-row C matrices. Default ``None`` uses
                fp16 for tensor-core Schur updates. Use ``torch.float32``
                for better numerical stability with ill-conditioned H.
            progress_fn: Optional ``callable(str)`` for progress messages.
        """
        gs = self.blocks_per_group
        bk = self.bk
        G = self.num_groups_per_row
        K = self.K
        M = self.M
        device = self.C_base.device
        num_prune = gs - num_nz

        if num_prune <= 0:
            return

        if self.row_coupled:
            self._prune_true_obs_coupled(
                num_nz, ng, chunk_size, order, progress_fn,
            )
            return

        epg = gs * bk if bk > 1 else gs  # elements per group

        # Group column indices from BlockSpec mapping
        if bk == 1:
            group_cols = self.col_idx_flat.view(G, gs)
        else:
            group_cols = self.col_idx.view(G, gs, bk)
        col_map = group_cols.reshape(G, epg)

        # Pruning subsets
        prune_subsets = torch.tensor(
            list(combinations(range(gs), num_prune)),
            device=device, dtype=torch.long,
        )
        n_subs = prune_subsets.shape[0]

        if bk == 1:
            sub_to_cols = prune_subsets
        else:
            sub_to_cols = (
                prune_subsets.unsqueeze(-1) * bk
                + torch.arange(bk, device=device)
            ).view(n_subs, num_prune * bk)
        np_cols = sub_to_cols.shape[1]
        use_closed_form = np_cols == 2

        # Group ordering
        if order == "largest_first":
            if progress_fn:
                progress_fn("Pre-scoring groups for ordering...")
            W_flat = self.W.data.clone().float().view(M, K)
            group_order = self._prescore_groups_order(
                W_flat, col_map, prune_subsets, sub_to_cols, np_cols,
            )
            del W_flat
            if progress_fn:
                progress_fn("Pre-scoring done.")
        else:
            group_order = torch.arange(G, device=device)

        W = self.W.data.clone().float().view(M, K)

        n_chunks = (M + chunk_size - 1) // chunk_size
        num_batches = (G + ng - 1) // ng

        for ci in range(n_chunks):
            c0 = ci * chunk_size
            c1 = min(c0 + chunk_size, M)
            B = c1 - c0

            if progress_fn and n_chunks > 4:
                progress_fn(
                    f"chunk {ci + 1}/{n_chunks} ({c0}/{M} rows)"
                )

            Wc = W[c0:c1]
            _c_dt = c_dtype or torch.float16
            C = self.C_base.to(_c_dt).unsqueeze(0).expand(B, -1, -1).clone()
            pruned_mask = torch.ones(B, K, device=device)

            for blk in range(num_batches):
                batch_gids = group_order[
                    blk * ng : min((blk + 1) * ng, G)
                ]
                n_g = batch_gids.shape[0]

                # Column indices for this batch (from BlockSpec mapping)
                batch_col_map = col_map[batch_gids]  # (n_g, epg)
                base_cols = batch_col_map.reshape(-1)

                # ── C diagonal blocks for scoring: (B, n_g, epg, epg) ──
                ri = batch_col_map.unsqueeze(2).expand(n_g, epg, epg)
                ci_idx = batch_col_map.unsqueeze(1).expand(n_g, epg, epg)
                C_diag = C[
                    :, ri.reshape(-1), ci_idx.reshape(-1)
                ].view(B, n_g, epg, epg).float()

                W_all = Wc[:, base_cols].view(B, n_g, epg)

                # ── Scoring ──
                if scoring == "independent":
                    C_diag_vec = torch.diagonal(
                        C_diag, dim1=-2, dim2=-1
                    )
                    if bk == 1:
                        elem_cost = W_all ** 2 / (C_diag_vec + 1e-8)
                        _, prune_idx = elem_cost.topk(
                            num_prune, dim=-1, largest=False
                        )
                    else:
                        block_cost = (
                            W_all ** 2 / (C_diag_vec + 1e-8)
                        ).view(B, n_g, gs, bk).sum(-1)
                        _, blk_idx = block_cost.topk(
                            num_prune, dim=-1, largest=False
                        )
                        prune_idx = (
                            blk_idx.unsqueeze(-1) * bk
                            + torch.arange(bk, device=device)
                        ).view(B, n_g, np_cols)
                    pruned_local = prune_idx
                else:
                    # Joint: enumerate all C(gs, num_prune) subsets
                    best_cost = torch.full(
                        (B, n_g), float("inf"), device=device
                    )
                    best_si = torch.zeros(
                        B, n_g, dtype=torch.long, device=device
                    )
                    for si in range(n_subs):
                        co = sub_to_cols[si]
                        W_P = W_all[:, :, co]

                        if use_closed_form:
                            a = C_diag[:, :, co[0], co[0]]
                            b_ = C_diag[:, :, co[0], co[1]]
                            d = C_diag[:, :, co[1], co[1]]
                            det = a * d - b_ * b_ + 1e-8
                            w0 = W_P[:, :, 0]
                            w1 = W_P[:, :, 1]
                            cost = (
                                w0 * w0 * d
                                - 2 * w0 * w1 * b_
                                + w1 * w1 * a
                            ) / det
                        else:
                            C_PP = C_diag[:, :, co][:, :, :, co]
                            eye_pp = 1e-8 * torch.eye(
                                np_cols, device=device
                            )
                            C_PP_inv = LA.inv(
                                (C_PP + eye_pp).reshape(
                                    B * n_g, np_cols, np_cols
                                )
                            )
                            W_flat = W_P.reshape(
                                B * n_g, 1, np_cols
                            )
                            cost = (
                                torch.bmm(W_flat, C_PP_inv).squeeze(1)
                                * W_P.reshape(B * n_g, np_cols)
                            ).sum(1).view(B, n_g)

                        better = cost < best_cost
                        best_cost[better] = cost[better]
                        best_si[better] = si

                    pruned_local = sub_to_cols[
                        best_si.view(-1)
                    ].view(B, n_g, np_cols)

                # Map local group indices → absolute column indices
                g_exp = torch.arange(
                    n_g, device=device
                ).view(1, n_g, 1).expand(B, n_g, np_cols)
                all_p = batch_col_map[
                    g_exp, pruned_local
                ].reshape(B, n_g * np_cols)
                np_total = all_p.shape[1]

                # ── Gather C columns ──
                pc_exp = all_p.unsqueeze(1).expand(B, K, np_total)
                C_col_P = C.gather(2, pc_exp)

                # ── Compensation ──
                eye_n = 1e-8 * torch.eye(np_total, device=device)
                C_PP = C_col_P.gather(
                    1,
                    all_p.unsqueeze(2).expand(B, np_total, np_total),
                ).float()
                C_PP_inv = LA.inv(C_PP + eye_n)

                W_P = Wc.gather(1, all_p)
                comp = torch.bmm(C_col_P.float(), C_PP_inv)
                delta = torch.bmm(
                    comp, W_P.unsqueeze(2)
                ).squeeze(2)
                Wc -= delta
                Wc.scatter_(
                    1, all_p,
                    torch.zeros(B, np_total, device=device),
                )

                # ── Schur update ──
                Lpp = LA.cholesky(C_PP_inv + eye_n)
                U = torch.bmm(C_col_P, Lpp.to(_c_dt))
                C.baddbmm_(U, U.transpose(1, 2), alpha=-1.0)

                pruned_mask.scatter_(1, all_p, 0.0)

            # Re-apply sparsity mask (fp16 residuals in C cause
            # tiny compensations to previously pruned weights)
            Wc *= pruned_mask
            del C

        if progress_fn and n_chunks > 4:
            progress_fn("")  # clear line

        self.W.data.copy_(W.view_as(self.W.data))
