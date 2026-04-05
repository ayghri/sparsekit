# Copyright (c) 2026 - Ayoub Ghriss & Contributors
# Licensed under CC BY-NC 4.0
# (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
# Non-commercial use only; contact us for commercial licensing.
"""Structured OBS (Optimal Brain Surgeon) via BlockSpec/ScopeSpec.

Compensation modes: ``local``, ``full``, ``split``, ``interleaved``.
"""

import math
from itertools import combinations

import torch
import torch.linalg as LA
from torch import Tensor

from ..view import View
from ..block import BlockSpec
from ..scope import ScopeSpec


def _block_flat_offsets(
    block: BlockSpec,
    device: torch.device = torch.device("cpu"),
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
    grid_pts = torch.stack(torch.meshgrid(*grid_ranges, indexing="ij"), dim=-1)

    bs = torch.tensor(block_shape, device=device)
    elem_idx = grid_pts.unsqueeze(-2) * bs + offsets

    param = block.view
    if isinstance(param, View):
        return param.linear_offset(elem_idx)
    else:
        strides = torch.tensor(
            param.data.stride(), device=device, dtype=torch.long
        )
        return (elem_idx * strides).sum(dim=-1)


def block_col_indices(
    block: BlockSpec,
    num_cols: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Map each block to its original column indices.

    Returns:
        col_idx: ``(*grid_shape[1:], block_numel)``
            long tensor.
    """
    flat_offsets = _block_flat_offsets(block, device)
    col_idx_full = flat_offsets % num_cols
    col_grid = block.grid_shape[1:]
    row_slice = tuple(
        0
        for _ in range(
            len(block.grid_shape) - len(col_grid)
        )
    )
    return col_idx_full[row_slice]


def block_param_rc(
    block: BlockSpec,
    num_cols: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[Tensor, Tensor]:
    """Map each block to ``(param_row, param_col)``.

    Unlike :func:`block_col_indices` this returns the
    **full** grid (including the row dimension) and both
    row and column indices.

    Returns:
        row_idx: ``(*grid_shape, block_numel)`` long
            tensor.
        col_idx: ``(*grid_shape, block_numel)`` long
            tensor.
    """
    flat_offsets = _block_flat_offsets(block, device)
    return (
        flat_offsets // num_cols,
        flat_offsets % num_cols,
    )


class StructuredOBS:
    """Structured OBS pruner operating through ScopeSpec.

    Args:
        scope: ScopeSpec defining the block and scope
            structure.
        hessian: (K, K) Hessian matrix.
        damp: Damping factor for hessian regularization.
        inv_h: Precomputed (K, K) damped inverse. Skips
            inversion if given.
    """

    def __init__(
        self,
        scope: ScopeSpec,
        hessian: Tensor,
        damp: float = 1e-4,
        inv_h: Tensor | None = None,
    ):
        self.scope = scope
        block = scope.block
        assert isinstance(block, BlockSpec)

        self.block = block
        param = block.view
        self.hessian = hessian
        self.damp = damp
        self.num_cols = hessian.shape[0]
        device = hessian.device

        if isinstance(param, View):
            self.W = param.param
        else:
            self.W = param

        self.M = self.W.shape[0]
        self.bk = block.block_numel

        grid_shape = block.grid_shape
        scope_shape = scope.shape
        scope_grid = scope.grid_shape

        self.rows_per_scope_row = block.shape[0] * scope_shape[0]
        self.num_scope_rows = scope_grid[0]

        col_grid = grid_shape[1:]
        self.total_blocks = math.prod(col_grid)
        self.blocks_per_scope = (
            math.prod(scope_shape[1:]) if len(scope_shape) > 1 else 1
        )
        self.num_scopes_per_row = (
            math.prod(scope_grid[1:]) if len(scope_grid) > 1 else 1
        )

        # Block column indices: (total_blocks, bk)
        col_idx_full = block_col_indices(block, self.num_cols, device=device)
        self.col_idx = col_idx_full.reshape(self.total_blocks, self.bk)
        if self.bk == 1:
            self.col_idx_flat = self.col_idx.squeeze(-1)

        # Block mapping
        self.block_to_scope, _ = self._build_block_mapping(
            col_grid, scope_shape[1:], scope_grid[1:], device
        )

        # Row coupling detection: do blocks within the same scope span
        # different param rows?  (e.g. block-16 with 8-row coupling)
        row_full, _ = block_param_rc(block, self.num_cols, device=device)
        row_slice = tuple(0 for _ in range(len(grid_shape) - len(col_grid)))
        row_cg = row_full[row_slice].reshape(self.total_blocks, self.bk)

        sort_perm = torch.argsort(self.block_to_scope)
        gs = self.blocks_per_scope
        num_parts = self.num_scopes_per_row
        sorted_row_base = row_cg[sort_perm].view(
            num_parts, gs, self.bk
        )
        block_rows = sorted_row_base[:, :, 0]
        self.row_coupled = bool(
            (
                block_rows.max(1).values
                - block_rows.min(1).values
                > 0
            ).any()
        )

        if self.row_coupled:
            self.block_sort_perm = sort_perm
            self.full_row_idx = row_full
            self.block_col_map = self.col_idx[
                sort_perm
            ].view(num_parts, gs, self.bk)

        if inv_h is not None:
            self.inv_hessian = inv_h
        else:
            self.inv_hessian = self.compute_inverse(
                hessian, damp
            )

    @staticmethod
    def compute_inverse(
        hessian: Tensor, damp: float = 1e-4
    ) -> Tensor:
        """Compute damped inverse of the Hessian matrix.

        Args:
            hessian: (K, K) symmetric positive semi-definite
                Hessian.
            damp: Damping factor. If < 1.0, scaled by mean
                diagonal; otherwise used as absolute value.

        Returns:
            (K, K) inverse of the damped Hessian:
            ``(H + damp * I)^{-1}``.
        """
        num_cols = hessian.shape[0]
        device = hessian.device
        damp_val = (
            damp * torch.mean(torch.diag(hessian))
            if damp < 1.0
            else damp
        )
        hessian_reg = hessian.clone()
        diag_idx = torch.arange(num_cols, device=device)
        hessian_reg[diag_idx, diag_idx] += damp_val
        return LA.inv(hessian_reg)  # pylint: disable=not-callable

    @staticmethod
    def _build_block_mapping(
        col_grid, block_shape_rest, block_grid_rest, device
    ):
        """Build mapping from block index to scope index."""
        total_blocks = math.prod(col_grid)
        if len(col_grid) == 0:
            return (
                torch.zeros(1, dtype=torch.long, device=device),
                torch.zeros(1, dtype=torch.long, device=device),
            )

        ranges = [torch.arange(g, device=device) for g in col_grid]
        bc_grid = torch.stack(
            torch.meshgrid(*ranges, indexing="ij"), dim=-1
        ).reshape(total_blocks, -1)

        gs_rest = list(block_shape_rest)
        gg_rest = list(block_grid_rest)

        block_pos = bc_grid // torch.tensor(gs_rest, device=device)
        gg_strides = []
        s = 1
        for g in reversed(gg_rest):
            gg_strides.append(s)
            s *= g
        gg_strides.reverse()
        block_to_scope = (
            block_pos * torch.tensor(gg_strides, device=device)
        ).sum(dim=-1)

        within_pos = bc_grid % torch.tensor(gs_rest, device=device)
        gs_strides = []
        s = 1
        for g in reversed(gs_rest):
            gs_strides.append(s)
            s *= g
        gs_strides.reverse()
        block_within_scope = (
            within_pos * torch.tensor(gs_strides, device=device)
        ).sum(dim=-1)

        return block_to_scope, block_within_scope

    def _find_best_subsets(  # pylint: disable=too-many-locals
        self,
        W,
        block_cols,
        prune_subsets,
        keep_subsets,  # pylint: disable=unused-argument
        block_size,
        inv_h=None,
    ):
        """Phase 1: find optimal pruning subset per (row, block).

        Args:
            inv_h: Optional inverse matrix. Uses
                self.inv_hessian if not provided. When provided,
                block_cols must be in inv_h's coordinate space.

        Returns:
            best_si: ``(M, G_input)`` long tensor -- index into
                prune_subsets.
        """
        device = W.device
        M, num_cols = W.shape
        gs = self.blocks_per_scope
        bk = self.bk
        if inv_h is None:
            inv_h = self.inv_hessian
        num_prune = prune_subsets.shape[1]
        n_subsets = prune_subsets.shape[0]
        elem_per_block = gs * bk if bk > 1 else gs
        num_parts = block_cols.shape[0]

        if bk == 1:
            block_start = block_cols[:, 0]
        else:
            block_start = block_cols[:, 0, 0]

        best_si_full = torch.zeros(
            M, num_parts, dtype=torch.long, device=device
        )

        for b_start in range(0, num_cols, block_size):
            b_end = min(b_start + block_size, num_cols)
            in_block = (
                (block_start >= b_start)
                & (block_start < b_end)
            )
            gidx = torch.where(in_block)[0]
            num_batch_parts = gidx.shape[0]
            if num_batch_parts == 0:
                continue

            if bk == 1:
                cols = block_cols[gidx]
                flat = cols.view(-1)
            else:
                cols = block_cols[gidx]
                flat = cols.view(
                    num_batch_parts, elem_per_block
                ).view(-1)

            if bk == 1:
                inv_sub = inv_h[
                    cols.unsqueeze(-1),
                    cols.unsqueeze(-2),
                ]
            else:
                fc = cols.view(
                    num_batch_parts, elem_per_block
                )
                inv_sub = inv_h[
                    fc.unsqueeze(-1), fc.unsqueeze(-2)
                ]

            weight_block = W[:, flat].view(
                M, num_batch_parts, elem_per_block
            )

            eye_np = 1e-8 * torch.eye(
                num_prune * bk, device=device
            )
            all_costs = torch.empty(
                n_subsets, M, num_batch_parts,
                device=device,
            )

            for si in range(n_subsets):
                pidx = prune_subsets[si]
                if bk == 1:
                    fp = pidx
                else:
                    fp = (
                        pidx.unsqueeze(-1) * bk
                        + torch.arange(bk, device=device)
                    ).view(-1)

                inv_pruned = inv_sub[:, fp][:, :, fp]
                inv_pruned_inv = LA.inv(  # pylint: disable=not-callable
                    inv_pruned + eye_np
                )

                weight_pruned = weight_block[:, :, fp]
                temp = torch.einsum(
                    "mgp,gpq->mgq",
                    weight_pruned,
                    inv_pruned_inv,
                )
                all_costs[si] = (
                    (temp * weight_pruned).sum(dim=2)
                )

            best_si_full[:, gidx] = all_costs.argmin(
                dim=0
            )

        return best_si_full

    def _compensate_local(
        self,
        W,
        block_cols,
        prune_subsets,
        keep_subsets,
        best_si,
        block_size,
    ):
        """Within-block compensation only."""
        device = W.device
        M, num_cols = W.shape
        gs = self.blocks_per_scope
        bk = self.bk
        inv_h = self.inv_hessian
        num_prune = prune_subsets.shape[1]
        n_subsets = prune_subsets.shape[0]
        elem_per_block = gs * bk if bk > 1 else gs

        if bk == 1:
            block_start = block_cols[:, 0]
        else:
            block_start = block_cols[:, 0, 0]

        for b_start in range(0, num_cols, block_size):
            b_end = min(b_start + block_size, num_cols)
            in_block = (
                (block_start >= b_start)
                & (block_start < b_end)
            )
            gidx = torch.where(in_block)[0]
            num_batch_parts = gidx.shape[0]
            if num_batch_parts == 0:
                continue

            if bk == 1:
                cols = block_cols[gidx]
                flat = cols.view(-1)
            else:
                cols = block_cols[gidx]
                flat = cols.view(
                    num_batch_parts, elem_per_block
                ).view(-1)

            if bk == 1:
                inv_sub = inv_h[
                    cols.unsqueeze(-1),
                    cols.unsqueeze(-2),
                ]
            else:
                fc = cols.view(
                    num_batch_parts, elem_per_block
                )
                inv_sub = inv_h[
                    fc.unsqueeze(-1), fc.unsqueeze(-2)
                ]

            weight_block = W[:, flat].view(
                M, num_batch_parts, elem_per_block
            )
            weight_block_new = weight_block.clone()
            eye_np = 1e-8 * torch.eye(
                num_prune * bk, device=device
            )
            block_best_si = best_si[:, gidx]

            for si in range(n_subsets):
                pidx = prune_subsets[si]
                kidx = keep_subsets[si]
                if bk == 1:
                    fp, fk = pidx, kidx
                else:
                    fp = (
                        pidx.unsqueeze(-1) * bk
                        + torch.arange(
                            bk, device=device
                        )
                    ).view(-1)
                    fk = (
                        kidx.unsqueeze(-1) * bk
                        + torch.arange(
                            bk, device=device
                        )
                    ).view(-1)

                mask = block_best_si == si
                if not mask.any():
                    continue

                inv_pruned = inv_sub[:, fp][:, :, fp]
                inv_pruned_inv = LA.inv(  # pylint: disable=not-callable
                    inv_pruned + eye_np
                )
                inv_keep_prune = (
                    inv_sub[:, fk][:, :, fp]
                )
                comp = torch.bmm(
                    inv_keep_prune, inv_pruned_inv
                )

                weight_pruned = (
                    weight_block[:, :, fp]
                )
                delta_k = torch.einsum(
                    "mgp,gnp->mgn",
                    weight_pruned,
                    comp,
                )
                mask_exp = mask.unsqueeze(-1)
                weight_block_new[:, :, fk] -= (
                    delta_k * mask_exp
                )
                weight_block_new[
                    :, :, fp
                ] = weight_block_new[
                    :, :, fp
                ].masked_fill(
                    mask_exp, 0.0
                )

            W.scatter_(
                1,
                flat.unsqueeze(0).expand(M, -1),
                weight_block_new.reshape(
                    M, num_batch_parts * elem_per_block
                ),
            )

    def _compensate_full(
        self, W, block_cols, prune_subsets, best_si
    ):
        """Sequential full-column compensation.

        Uses inv_hessian. For each block, compensate ALL K
        columns (not just within-block), processing blocks
        sequentially so each sees updated W.
        """
        device = W.device
        M, num_cols = W.shape
        num_parts = self.num_scopes_per_row
        bk = self.bk
        inv_h = self.inv_hessian
        num_prune = prune_subsets.shape[1]
        n_subsets = prune_subsets.shape[0]
        np_bk = num_prune * bk

        # Precompute compensation matrices for all
        # (block, subset) pairs:
        #   comp[g, si] = inv(C[P,P]) @ C[P, :]
        #   shape (np*bk, K)
        # where P = absolute column indices of pruned
        # groups in block g, subset si.

        # Pruned column indices per (block, subset):
        # (num_parts, n_subsets, np*bk)
        if bk == 1:
            pruned_cols = block_cols[
                :, prune_subsets
            ]
        else:
            pruned_blk = block_cols[
                :, prune_subsets
            ]
            pruned_cols = pruned_blk.view(
                num_parts, n_subsets, np_bk
            )

        inv_pruned = inv_h[
            pruned_cols[:, :, :, None],
            pruned_cols[:, :, None, :],
        ]
        eye = 1e-8 * torch.eye(np_bk, device=device)
        inv_pruned_inv = torch.inverse(
            inv_pruned + eye
        )

        inv_prune_rows = inv_h[
            pruned_cols.reshape(-1), :
        ].view(num_parts, n_subsets, np_bk, num_cols)

        comp_all = torch.einsum(
            "gsij,gsjk->gsik",
            inv_pruned_inv,
            inv_prune_rows,
        )

        active = torch.ones(
            M, num_cols, device=device, dtype=torch.bool
        )

        for g in range(num_parts):
            si_per_row = best_si[:, g]
            comp_g = comp_all[g]
            comp_rows = comp_g[si_per_row]
            p_cols = pruned_cols[g, si_per_row]

            comp_rows = (
                comp_rows * active.unsqueeze(1)
            )

            weight_pruned = torch.gather(
                W, 1, p_cols
            )
            delta = torch.bmm(
                weight_pruned.unsqueeze(1), comp_rows
            ).squeeze(1)
            W -= delta

            W.scatter_(
                1,
                p_cols,
                torch.zeros_like(weight_pruned),
            )

            active.scatter_(1, p_cols, False)

    def _compensate_full_split(
        self,
        W,
        block_cols,
        prune_subsets,
        best_si,
        n_splits=2,
    ):
        """Full-column compensation with C recomputed.

        Splits blocks into n_splits chunks by column position.
        Each chunk uses C = inv(H[active, active] + damp*I)
        where 'active' excludes columns from previous chunks.
        """
        device = W.device
        M, num_cols = W.shape
        num_parts = self.num_scopes_per_row
        bk = self.bk
        num_prune = prune_subsets.shape[1]
        n_subsets = prune_subsets.shape[0]
        np_bk = num_prune * bk

        chunk_g = (
            (num_parts + n_splits - 1) // n_splits
        )
        active_mask = torch.ones(
            num_cols, dtype=torch.bool, device=device
        )

        for split_idx in range(n_splits):
            g_start = split_idx * chunk_g
            g_end = min(
                (split_idx + 1) * chunk_g, num_parts
            )
            if g_start >= num_parts:
                break
            num_split_parts = g_end - g_start

            if split_idx == 0:
                active_cols = torch.arange(
                    num_cols, device=device
                )
                n_active = num_cols
                inv_h = self.inv_hessian
                abs_to_local = torch.arange(
                    num_cols, device=device
                )
            else:
                active_cols = torch.where(
                    active_mask
                )[0]
                n_active = active_cols.shape[0]
                hessian_active = self.hessian[
                    active_cols[:, None],
                    active_cols[None, :],
                ]
                inv_h = self.compute_inverse(
                    hessian_active, self.damp
                )
                abs_to_local = torch.full(
                    (num_cols,),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
                abs_to_local[active_cols] = torch.arange(
                    n_active, device=device
                )

            split_block_cols = block_cols[
                g_start:g_end
            ]
            if bk == 1:
                pruned_abs = split_block_cols[
                    :, prune_subsets
                ]
            else:
                pruned_abs = split_block_cols[
                    :, prune_subsets
                ].view(
                    num_split_parts, n_subsets, np_bk
                )

            pruned_local = abs_to_local[pruned_abs]

            inv_pruned = inv_h[
                pruned_local[:, :, :, None],
                pruned_local[:, :, None, :],
            ]
            eye = 1e-8 * torch.eye(
                np_bk, device=device
            )
            inv_pruned_inv = torch.inverse(
                inv_pruned + eye
            )

            inv_prune_rows = inv_h[
                pruned_local.reshape(-1), :
            ].view(
                num_split_parts,
                n_subsets,
                np_bk,
                n_active,
            )
            comp_split = torch.einsum(
                "gsij,gsjk->gsik",
                inv_pruned_inv,
                inv_prune_rows,
            )

            split_active = torch.ones(
                M,
                n_active,
                device=device,
                dtype=torch.bool,
            )

            for g_local in range(num_split_parts):
                g = g_start + g_local
                si_per_row = best_si[:, g]
                comp_rows = comp_split[g_local][
                    si_per_row
                ]
                p_cols = pruned_abs[
                    g_local, si_per_row
                ]

                comp_rows = (
                    comp_rows * split_active.unsqueeze(1)
                )

                weight_pruned = torch.gather(
                    W, 1, p_cols
                )
                delta = torch.bmm(
                    weight_pruned.unsqueeze(1),
                    comp_rows,
                ).squeeze(1)

                if n_active == num_cols:
                    W -= delta
                else:
                    ac_exp = active_cols.unsqueeze(
                        0
                    ).expand(M, -1)
                    W.scatter_add_(
                        1, ac_exp, -delta
                    )

                W.scatter_(
                    1,
                    p_cols,
                    torch.zeros_like(weight_pruned),
                )

                p_local = abs_to_local[p_cols]
                split_active.scatter_(
                    1, p_local, False
                )

            if split_idx < n_splits - 1:
                if bk == 1:
                    frozen = split_block_cols.reshape(
                        -1
                    )
                else:
                    frozen = split_block_cols.view(-1)
                active_mask[frozen] = False

    def _interleaved(
        self,
        W,
        block_cols,
        prune_subsets,
        keep_subsets,
        n_splits=16,
        block_size=2048,
    ):
        """Interleaved selection + compensation.

        Unlike split compensation (which selects masks once
        then compensates), this re-selects which columns to
        prune at each split using the updated C. Uses a
        single shared C (not per-row), so O(K^2) memory.

        Key: _find_best_subsets is called with local-space
        block_cols and the recomputed C, so mask selection
        uses the Schur-updated inverse.
        """
        device = W.device
        M, num_cols = W.shape
        num_parts = self.num_scopes_per_row
        bk = self.bk
        num_prune = prune_subsets.shape[1]
        n_subsets = prune_subsets.shape[0]
        np_bk = num_prune * bk

        chunk_g = (
            (num_parts + n_splits - 1) // n_splits
        )
        active_mask = torch.ones(
            num_cols, dtype=torch.bool, device=device
        )

        for split_idx in range(n_splits):
            g_start = split_idx * chunk_g
            g_end = min(
                (split_idx + 1) * chunk_g, num_parts
            )
            if g_start >= num_parts:
                break
            num_split_parts = g_end - g_start

            if split_idx == 0:
                active_cols = torch.arange(
                    num_cols, device=device
                )
                n_active = num_cols
                inv_h = self.inv_hessian
                abs_to_local = torch.arange(
                    num_cols, device=device
                )
            else:
                active_cols = torch.where(
                    active_mask
                )[0]
                n_active = active_cols.shape[0]
                hessian_active = self.hessian[
                    active_cols[:, None],
                    active_cols[None, :],
                ]
                inv_h = self.compute_inverse(
                    hessian_active, self.damp
                )
                abs_to_local = torch.full(
                    (num_cols,),
                    -1,
                    dtype=torch.long,
                    device=device,
                )
                abs_to_local[active_cols] = torch.arange(
                    n_active, device=device
                )

            split_block_cols = block_cols[
                g_start:g_end
            ]
            if bk == 1:
                split_block_cols_local = (
                    abs_to_local[split_block_cols]
                )
            else:
                split_block_cols_local = (
                    abs_to_local[split_block_cols]
                )

            weight_active = W[:, active_cols]

            best_si_split = self._find_best_subsets(
                weight_active,
                split_block_cols_local,
                prune_subsets,
                keep_subsets,
                block_size,
                inv_h=inv_h,
            )

            if bk == 1:
                pruned_local = (
                    split_block_cols_local[
                        :, prune_subsets
                    ]
                )
                pruned_abs = split_block_cols[
                    :, prune_subsets
                ]
            else:
                pruned_local = abs_to_local[
                    split_block_cols[
                        :, prune_subsets
                    ].view(
                        num_split_parts,
                        n_subsets,
                        np_bk,
                    )
                ]
                pruned_abs = split_block_cols[
                    :, prune_subsets
                ].view(
                    num_split_parts,
                    n_subsets,
                    np_bk,
                )

            inv_pruned = inv_h[
                pruned_local[:, :, :, None],
                pruned_local[:, :, None, :],
            ]
            eye = 1e-8 * torch.eye(
                np_bk, device=device
            )
            inv_pruned_inv = torch.inverse(
                inv_pruned + eye
            )

            inv_prune_rows = inv_h[
                pruned_local.reshape(-1), :
            ].view(
                num_split_parts,
                n_subsets,
                np_bk,
                n_active,
            )
            comp_split = torch.einsum(
                "gsij,gsjk->gsik",
                inv_pruned_inv,
                inv_prune_rows,
            )

            split_active = torch.ones(
                M,
                n_active,
                device=device,
                dtype=torch.bool,
            )

            for g_local in range(num_split_parts):
                si_per_row = best_si_split[
                    :, g_local
                ]
                comp_rows = comp_split[g_local][
                    si_per_row
                ]
                p_cols = pruned_abs[
                    g_local, si_per_row
                ]

                comp_rows = (
                    comp_rows
                    * split_active.unsqueeze(1)
                )

                weight_pruned = torch.gather(
                    W, 1, p_cols
                )
                delta = torch.bmm(
                    weight_pruned.unsqueeze(1),
                    comp_rows,
                ).squeeze(1)

                if n_active == num_cols:
                    W -= delta
                else:
                    ac_exp = active_cols.unsqueeze(
                        0
                    ).expand(M, -1)
                    W.scatter_add_(
                        1, ac_exp, -delta
                    )

                W.scatter_(
                    1,
                    p_cols,
                    torch.zeros_like(weight_pruned),
                )

                p_local = abs_to_local[p_cols]
                split_active.scatter_(
                    1, p_local, False
                )

            if split_idx < n_splits - 1:
                if bk == 1:
                    frozen = split_block_cols.reshape(
                        -1
                    )
                else:
                    frozen = split_block_cols.view(-1)
                active_mask[frozen] = False

    @torch.no_grad()
    def prune(
        self,
        num_nz: int,
        block_size: int = 2048,
        compensate: str = "local",
        n_splits: int = 1,
    ) -> None:
        """Prune to num_nz groups per block.

        Phase 1: enumerate all C(gs, num_prune) subsets per block, pick
                 the best per (row, block) using C = H^{-1} submatrices.

        Phase 2 (compensation):
          - 'local': within-block only (fast, independent blocks)
          - 'full':  sequential compensation to ALL K columns via C[P, :]
                     (slower but ~44% better than SparseGPT)
          - 'split': like 'full' but recomputes C between column splits.
                     Use n_splits to control granularity (2 = one C update
                     at the midpoint).
          - 'interleaved': re-selects masks AND compensates at each split
                     using recomputed C. Single shared C (O(K²) memory).

        Args:
            num_nz: Groups to keep per block.
            block_size: Column chunk size for subset search.
            compensate: 'local', 'full', 'split', or 'interleaved'.
            n_splits: Number of column splits (for 'split'/'interleaved').
        """
        gs = self.blocks_per_scope
        num_prune = gs - num_nz
        if num_prune <= 0:
            return

        device = self.W.data.device
        M = self.M
        num_cols = self.num_cols
        bk = self.bk
        num_parts = self.num_scopes_per_row

        W = self.W.data.clone().float().view(
            M, num_cols
        )

        # Block column indices
        if bk == 1:
            block_cols = self.col_idx_flat.view(
                num_parts, gs
            )
        else:
            block_cols = self.col_idx.view(
                num_parts, gs, bk
            )

        # Pruning subsets
        prune_subsets = torch.tensor(
            list(combinations(range(gs), num_prune)),
            device=device,
            dtype=torch.long,
        )
        keep_subsets = torch.tensor(
            [
                sorted(set(range(gs)) - set(s))
                for s in combinations(range(gs), num_prune)
            ],
            device=device,
            dtype=torch.long,
        )

        if compensate == "interleaved":
            self._interleaved(
                W,
                block_cols,
                prune_subsets,
                keep_subsets,
                n_splits=n_splits,
                block_size=block_size,
            )
            self.W.data.copy_(W.view_as(self.W.data))
            return

        # Phase 1: find best subset per (row, block)
        best_si = self._find_best_subsets(
            W,
            block_cols,
            prune_subsets,
            keep_subsets,
            block_size,
        )

        # Phase 2: apply compensation
        if compensate == "split":
            self._compensate_full_split(
                W,
                block_cols,
                prune_subsets,
                best_si,
                n_splits=n_splits,
            )
        elif compensate == "full":
            self._compensate_full(W, block_cols, prune_subsets, best_si)
        else:
            self._compensate_local(
                W,
                block_cols,
                prune_subsets,
                keep_subsets,
                best_si,
                block_size,
            )

        self.W.data.copy_(W.view_as(self.W.data))

    # ── True OBS (per-row C with Schur updates) ──────────────────────

    def _prescore_blocks_order(
        self,
        W,
        col_map,
        prune_subsets,
        sub_to_cols,
        np_cols,
    ):
        """Pre-score blocks by OBS cost.

        Uses shared inv_hessian. Returns block indices sorted
        by descending total cost (highest-cost blocks first,
        so they are processed while C is most accurate).
        """
        device = W.device
        M = W.shape[0]
        num_parts = col_map.shape[0]
        n_subs = prune_subsets.shape[0]
        inv_h = self.inv_hessian
        eye_np = 1e-8 * torch.eye(
            np_cols, device=device
        )
        scores = torch.zeros(
            num_parts, device=device
        )

        for g in range(num_parts):
            cols = col_map[g]
            inv_block = inv_h[cols][:, cols]
            weight_block = W[:, cols]
            best_cost = torch.full(
                (M,), float("inf"), device=device
            )
            for si in range(n_subs):
                co = sub_to_cols[si]
                inv_pruned_inv = torch.inverse(
                    inv_block[co][:, co] + eye_np
                )
                weight_pruned = weight_block[:, co]
                cost = (
                    weight_pruned
                    @ inv_pruned_inv
                    * weight_pruned
                ).sum(1)
                better = cost < best_cost
                best_cost[better] = cost[better]
            scores[g] = best_cost.sum()

        return torch.argsort(scores, descending=True)

    # ── Row-coupled True OBS ─────────────────────────────────────────

    def _prescore_coupled_blocks(
        self,
        weight_chunk,
        local_row_map,
        n_vr,
        eye_bk,
    ):
        """Pre-score row-coupled blocks.

        Uses shared inv_hessian. Returns block indices
        sorted descending by OBS cost. Fully vectorized
        over all num_parts blocks.
        """
        gcm = self.block_col_map
        num_parts, gs, bk = gcm.shape
        device = weight_chunk.device
        inv_h = self.inv_hessian

        cols = gcm[:, 0, :]
        num_cols = inv_h.shape[0]
        ci = cols.unsqueeze(2).expand(
            -1, -1, bk
        )
        cj = cols.unsqueeze(1).expand(
            -1, bk, -1
        )
        flat_idx = ci * num_cols + cj
        inv_pruned = inv_h.reshape(-1)[
            flat_idx.reshape(-1)
        ].view(num_parts, bk, bk)

        inv_pruned_inv = torch.linalg.inv(  # pylint: disable=not-callable
            inv_pruned.float() + eye_bk
        )

        block_scores = torch.empty(
            n_vr, num_parts, gs, device=device
        )
        for b in range(gs):
            lr = local_row_map[:, :, b]
            lr_exp = lr.unsqueeze(2).expand(
                -1, -1, bk
            )
            cols_exp = cols.unsqueeze(0).expand(
                n_vr, -1, -1
            )
            weight_blk = weight_chunk[
                lr_exp.reshape(-1),
                cols_exp.reshape(-1),
            ].view(n_vr, num_parts, bk)
            temp = torch.einsum(
                "ngb,gbc->ngc",
                weight_blk,
                inv_pruned_inv,
            )
            block_scores[:, :, b] = (
                (temp * weight_blk).sum(2)
            )

        scores = (
            block_scores.min(dim=2).values.sum(dim=0)
        )
        return torch.argsort(scores, descending=True)

    @torch.no_grad()
    def _prune_true_obs_coupled(
        self,
        num_nz,
        ng,  # pylint: disable=unused-argument
        chunk_size,
        order,
        progress_fn,
    ):
        """True OBS for row-coupled blocks.

        Groups span different param rows. Blocks processed
        sequentially; vectorized across view-rows using
        flat indexing (no per-element Python loops).
        """
        gs = self.blocks_per_scope
        bk = self.bk
        num_parts = self.num_scopes_per_row
        num_cols = self.num_cols
        M = self.M
        device = self.inv_hessian.device
        num_prune = gs - num_nz

        num_gr = self.num_scope_rows
        gcm = self.block_col_map
        gsp = self.block_sort_perm

        eye_bk = 1e-4 * torch.eye(bk, device=device)

        W = self.W.data.clone().float().view(
            M, num_cols
        )

        n_chunks = (
            (num_gr + chunk_size - 1) // chunk_size
        )

        for ci in range(n_chunks):
            vr0 = ci * chunk_size
            vr1 = min(vr0 + chunk_size, num_gr)
            n_vr = vr1 - vr0

            if progress_fn:
                progress_fn(
                    f"chunk {ci + 1}/{n_chunks}"
                    f" ({vr0}/{num_gr} view-rows)"
                )

            chunk_row_map = torch.empty(
                n_vr,
                num_parts,
                gs,
                bk,
                device=device,
                dtype=torch.long,
            )
            for vr_local in range(n_vr):
                row_cg = self.full_row_idx[
                    vr0 + vr_local
                ].reshape(self.total_blocks, bk)
                chunk_row_map[vr_local] = (
                    row_cg[gsp].view(
                        num_parts, gs, bk
                    )
                )

            unique_rows = (
                chunk_row_map[:, :, :, 0]
                .reshape(-1)
                .unique()
            )
            b_param = unique_rows.shape[0]
            p2l = torch.full(
                (M,),
                -1,
                device=device,
                dtype=torch.long,
            )
            p2l[unique_rows] = torch.arange(
                b_param, device=device
            )

            local_row_map = p2l[
                chunk_row_map[:, :, :, 0]
            ]

            weight_chunk = W[unique_rows].clone()
            inv_h = torch.empty(
                b_param,
                num_cols,
                num_cols,
                device=device,
                dtype=torch.float16,
            )
            inv_h[:] = self.inv_hessian.half()
            pruned_mask = torch.ones(
                b_param, num_cols, device=device
            )

            inv_flat = inv_h.reshape(-1)

            if order == "largest_first":
                if progress_fn:
                    progress_fn(
                        "Pre-scoring blocks..."
                    )
                block_order = (
                    self._prescore_coupled_blocks(
                        weight_chunk,
                        local_row_map,
                        n_vr,
                        eye_bk,
                    )
                )
            else:
                block_order = torch.arange(
                    num_parts, device=device
                )

            arange_vr = torch.arange(
                n_vr, device=device
            )

            for gi in range(num_parts):
                g = block_order[gi]
                cols = gcm[g, 0, :]
                rows_g = local_row_map[:, g, :]

                n_score = n_vr * gs
                sc_rows = rows_g.reshape(n_score)
                inv_pruned = inv_h[
                    :, cols, :
                ][:, :, cols][sc_rows]
                inv_pruned_inv = torch.linalg.inv(  # pylint: disable=not-callable
                    inv_pruned.float() + eye_bk
                )
                weight_blk = weight_chunk[
                    sc_rows
                ][:, cols]
                temp = torch.bmm(
                    weight_blk.unsqueeze(1),
                    inv_pruned_inv,
                ).squeeze(1)
                scores = (
                    (temp * weight_blk)
                    .sum(1)
                    .view(n_vr, gs)
                )

                _, prune_bi = scores.topk(
                    num_prune,
                    dim=1,
                    largest=False,
                )

                for pi in range(num_prune):
                    pb = prune_bi[:, pi]
                    flat_rows = rows_g[
                        arange_vr, pb
                    ]

                    si = arange_vr * gs + pb
                    c_inv = inv_pruned_inv[si]

                    inv_col = inv_h[
                        :, :, cols
                    ][flat_rows]

                    comp = torch.bmm(
                        inv_col.float(), c_inv
                    )
                    weight_pruned = (
                        weight_chunk[flat_rows][
                            :, cols
                        ]
                    )
                    delta = torch.bmm(
                        comp,
                        weight_pruned.unsqueeze(2),
                    ).squeeze(2)

                    weight_chunk.index_add_(
                        0, flat_rows, -delta
                    )
                    weight_chunk[
                        flat_rows.unsqueeze(1).expand(
                            -1, bk
                        ),
                        cols.unsqueeze(0).expand(
                            n_vr, -1
                        ),
                    ] = 0.0

                    comp_h = comp.half()
                    for vr in range(n_vr):
                        r = flat_rows[vr]
                        torch.addmm(
                            inv_h[r],
                            comp_h[vr],
                            inv_col[vr].T,
                            beta=1,
                            alpha=-1,
                            out=inv_h[r],
                        )

                    pruned_mask[
                        flat_rows.unsqueeze(1).expand(
                            -1, bk
                        ),
                        cols.unsqueeze(0).expand(
                            n_vr, -1
                        ),
                    ] = 0.0

            weight_chunk *= pruned_mask
            W[unique_rows] = weight_chunk
            del inv_h, inv_flat

        if progress_fn:
            progress_fn("")

        self.W.data.copy_(W.view_as(self.W.data))

    @torch.no_grad()
    def prune_true_obs(
        self,
        num_nz: int,
        ng: int = 64,
        chunk_size: int = 16,
        order: str = "left_to_right",
        scoring: str = "independent",
        c_dtype=None,
        progress_fn=None,
    ) -> None:
        """Per-row True OBS with Schur complement updates.

        Each row maintains its own C = inv(H), updated via
        Schur complement after pruning. Processes ``ng``
        blocks simultaneously per batch.

        Args:
            num_nz: Groups to keep per block.
            ng: Number of blocks to process per batch.
            chunk_size: Rows to process simultaneously.
            order: ``"left_to_right"`` or
                ``"largest_first"``.
            scoring: ``"joint"`` (enumerate subsets) or
                ``"independent"`` (per-element w^2/diag(C)
                + topk).
            c_dtype: Dtype for per-row C matrices. Default
                ``None`` uses fp16 for tensor-core Schur
                updates.
            progress_fn: Optional ``callable(str)`` for
                progress messages.
        """
        gs = self.blocks_per_scope
        bk = self.bk
        num_parts = self.num_scopes_per_row
        num_cols = self.num_cols
        M = self.M
        device = self.inv_hessian.device
        num_prune = gs - num_nz

        if num_prune <= 0:
            return

        if self.row_coupled:
            self._prune_true_obs_coupled(
                num_nz,
                ng,
                chunk_size,
                order,
                progress_fn,
            )
            return

        epg = gs * bk if bk > 1 else gs

        if bk == 1:
            block_cols = self.col_idx_flat.view(
                num_parts, gs
            )
        else:
            block_cols = self.col_idx.view(
                num_parts, gs, bk
            )
        col_map = block_cols.reshape(num_parts, epg)

        prune_subsets = torch.tensor(
            list(combinations(range(gs), num_prune)),
            device=device,
            dtype=torch.long,
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

        if order == "largest_first":
            if progress_fn:
                progress_fn(
                    "Pre-scoring blocks for ordering..."
                )
            weight_flat = (
                self.W.data.clone()
                .float()
                .view(M, num_cols)
            )
            block_order = (
                self._prescore_blocks_order(
                    weight_flat,
                    col_map,
                    prune_subsets,
                    sub_to_cols,
                    np_cols,
                )
            )
            del weight_flat
            if progress_fn:
                progress_fn("Pre-scoring done.")
        else:
            block_order = torch.arange(
                num_parts, device=device
            )

        W = self.W.data.clone().float().view(
            M, num_cols
        )

        n_chunks = (M + chunk_size - 1) // chunk_size
        num_batches = (num_parts + ng - 1) // ng

        for ci in range(n_chunks):
            c0 = ci * chunk_size
            c1 = min(c0 + chunk_size, M)
            B = c1 - c0

            if progress_fn and n_chunks > 4:
                progress_fn(
                    f"chunk {ci + 1}/{n_chunks}"
                    f" ({c0}/{M} rows)"
                )

            weight_chunk = W[c0:c1]
            c_dt = c_dtype or torch.float16
            inv_h = (
                self.inv_hessian.to(c_dt)
                .unsqueeze(0)
                .expand(B, -1, -1)
                .clone()
            )
            pruned_mask = torch.ones(
                B, num_cols, device=device
            )

            for blk in range(num_batches):
                batch_gids = block_order[
                    blk * ng : min(
                        (blk + 1) * ng, num_parts
                    )
                ]
                n_g = batch_gids.shape[0]

                batch_col_map = col_map[batch_gids]
                base_cols = batch_col_map.reshape(-1)

                ri = batch_col_map.unsqueeze(2).expand(
                    n_g, epg, epg
                )
                ci_idx = (
                    batch_col_map.unsqueeze(1).expand(
                        n_g, epg, epg
                    )
                )
                inv_diag = (
                    inv_h[
                        :,
                        ri.reshape(-1),
                        ci_idx.reshape(-1),
                    ]
                    .view(B, n_g, epg, epg)
                    .float()
                )

                weight_all = weight_chunk[
                    :, base_cols
                ].view(B, n_g, epg)

                if scoring == "independent":
                    inv_diag_vec = torch.diagonal(
                        inv_diag, dim1=-2, dim2=-1
                    )
                    if bk == 1:
                        elem_cost = weight_all**2 / (
                            inv_diag_vec + 1e-8
                        )
                        _, prune_idx = elem_cost.topk(
                            num_prune,
                            dim=-1,
                            largest=False,
                        )
                    else:
                        block_cost = (
                            (
                                weight_all**2
                                / (inv_diag_vec + 1e-8)
                            )
                            .view(B, n_g, gs, bk)
                            .sum(-1)
                        )
                        _, blk_idx = block_cost.topk(
                            num_prune,
                            dim=-1,
                            largest=False,
                        )
                        prune_idx = (
                            blk_idx.unsqueeze(-1) * bk
                            + torch.arange(
                                bk, device=device
                            )
                        ).view(B, n_g, np_cols)
                    pruned_local = prune_idx
                else:
                    best_cost = torch.full(
                        (B, n_g),
                        float("inf"),
                        device=device,
                    )
                    best_si = torch.zeros(
                        B,
                        n_g,
                        dtype=torch.long,
                        device=device,
                    )
                    for si in range(n_subs):
                        co = sub_to_cols[si]
                        weight_pruned = (
                            weight_all[:, :, co]
                        )

                        if use_closed_form:
                            a = inv_diag[
                                :, :, co[0], co[0]
                            ]
                            b_ = inv_diag[
                                :, :, co[0], co[1]
                            ]
                            d = inv_diag[
                                :, :, co[1], co[1]
                            ]
                            det = (
                                a * d - b_ * b_ + 1e-8
                            )
                            w0 = weight_pruned[
                                :, :, 0
                            ]
                            w1 = weight_pruned[
                                :, :, 1
                            ]
                            cost = (
                                w0 * w0 * d
                                - 2 * w0 * w1 * b_
                                + w1 * w1 * a
                            ) / det
                        else:
                            inv_pruned = inv_diag[
                                :, :, co
                            ][:, :, :, co]
                            eye_pp = (
                                1e-8
                                * torch.eye(
                                    np_cols,
                                    device=device,
                                )
                            )
                            inv_pruned_inv = LA.inv(  # pylint: disable=not-callable
                                (
                                    inv_pruned + eye_pp
                                ).reshape(
                                    B * n_g,
                                    np_cols,
                                    np_cols,
                                )
                            )
                            weight_flat = (
                                weight_pruned.reshape(
                                    B * n_g,
                                    1,
                                    np_cols,
                                )
                            )
                            cost = (
                                (
                                    torch.bmm(
                                        weight_flat,
                                        inv_pruned_inv,
                                    ).squeeze(1)
                                    * weight_pruned.reshape(
                                        B * n_g,
                                        np_cols,
                                    )
                                )
                                .sum(1)
                                .view(B, n_g)
                            )

                        better = cost < best_cost
                        best_cost[better] = cost[
                            better
                        ]
                        best_si[better] = si

                    pruned_local = sub_to_cols[
                        best_si.view(-1)
                    ].view(B, n_g, np_cols)

                g_exp = (
                    torch.arange(n_g, device=device)
                    .view(1, n_g, 1)
                    .expand(B, n_g, np_cols)
                )
                all_p = batch_col_map[
                    g_exp, pruned_local
                ].reshape(B, n_g * np_cols)
                np_total = all_p.shape[1]

                pc_exp = all_p.unsqueeze(1).expand(
                    B, num_cols, np_total
                )
                inv_col_prune = inv_h.gather(
                    2, pc_exp
                )

                eye_n = 1e-8 * torch.eye(
                    np_total, device=device
                )
                inv_pruned = inv_col_prune.gather(
                    1,
                    all_p.unsqueeze(2).expand(
                        B, np_total, np_total
                    ),
                ).float()
                inv_pruned_inv = LA.inv(  # pylint: disable=not-callable
                    inv_pruned + eye_n
                )

                weight_pruned = weight_chunk.gather(
                    1, all_p
                )
                comp = torch.bmm(
                    inv_col_prune.float(),
                    inv_pruned_inv,
                )
                delta = torch.bmm(
                    comp,
                    weight_pruned.unsqueeze(2),
                ).squeeze(2)
                weight_chunk -= delta
                weight_chunk.scatter_(
                    1,
                    all_p,
                    torch.zeros(
                        B, np_total, device=device
                    ),
                )

                lpp = LA.cholesky(  # pylint: disable=not-callable
                    inv_pruned_inv + eye_n
                )
                schur_factor = torch.bmm(
                    inv_col_prune, lpp.to(c_dt)
                )
                inv_h.baddbmm_(
                    schur_factor,
                    schur_factor.transpose(1, 2),
                    alpha=-1.0,
                )

                pruned_mask.scatter_(
                    1, all_p, 0.0
                )

            weight_chunk *= pruned_mask
            del inv_h

        if progress_fn and n_chunks > 4:
            progress_fn("")

        self.W.data.copy_(W.view_as(self.W.data))
