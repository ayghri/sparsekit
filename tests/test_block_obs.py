import torch
from typing import List
from dataclasses import dataclass


@dataclass
class GroupConfig:
    """Configuration for group structure."""
    n: int = 256          # Output dim
    k: int = 128          # Input dim
    group_row: int = 4   # Group size along output dim
    group_col: int = 4   # Group size along input dim
    block_row: int = 1    # Block size along output dim (in groups)
    block_col: int = 4    # Block size along input dim (in groups)
    groups_to_prune: int = 2  # Groups to prune per block
    
    @property
    def num_blocks_row(self) -> int:
        return self.n // self.group_row  # 16
    
    @property
    def num_blocks_col(self) -> int:
        return self.k // self.group_col  # 8
    
    @property
    def num_scopes_row(self) -> int:
        return self.num_blocks_row // self.block_row  # 16
    
    @property
    def num_scopes_col(self) -> int:
        return self.num_blocks_col // self.block_col  # 2
    
    @property
    def blocks_per_scope(self) -> int:
        return self.block_row * self.block_col  # 4


class GroupedWeightView:
    """Handles blocked view B with shape (num_blocks_row, num_blocks_col, group_row, group_col)."""
    
    def __init__(self, weight: torch.Tensor, config: GroupConfig):
        self.config = config
        self.n, self.k = weight.shape
        assert self.n == config.n and self.k == config.k
        
        # Reshape to blocked view: (16, 8, 16, 16)
        self.B = weight.view(
            config.num_blocks_row, config.group_row,
            config.num_blocks_col, config.group_col
        ).permute(0, 2, 1, 3).contiguous()  # (n_groups_r, n_groups_c, blk_r, blk_c)
        
    def to_grouped(self) -> 'BlockedWeightView':
        """Convert to grouped view G with shape (16, 2, 4, 16, 16)."""
        cfg = self.config
        # Reshape groups into blocks: (16, 2, 4, 16, 16)
        # block_row=1, block_col=4 means we block 1x4 groups together
        G = self.B.view(
            cfg.num_scopes_row, cfg.block_row,  # 16, 1
            cfg.num_scopes_col, cfg.block_col,  # 2, 4
            cfg.group_row, cfg.group_col        # 16, 16
        ).permute(0, 2, 1, 3, 4, 5).contiguous()  # (n_grp_r, n_grp_c, grp_r, grp_c, blk_r, blk_c)
        
        # Flatten block dimensions: (16, 2, 4, 16, 16)
        G = G.view(
            cfg.num_scopes_row, cfg.num_scopes_col,
            cfg.blocks_per_scope,
            cfg.group_row, cfg.group_col
        )
        return BlockedWeightView(G, self.config)
    
    def from_grouped(self, G_view: 'BlockedWeightView') -> torch.Tensor:
        """Convert grouped view back to blocked view."""
        cfg = self.config
        G = G_view.G  # (16, 2, 4, 16, 16)
        
        # Reshape to separate block dimensions: (16, 2, 1, 4, 16, 16)
        G = G.view(
            cfg.num_scopes_row, cfg.num_scopes_col,
            cfg.block_row, cfg.block_col,
            cfg.group_row, cfg.group_col
        )
        
        # Permute back to blocked structure: (16, 8, 16, 16)
        B = G.permute(0, 2, 1, 3, 4, 5).contiguous()
        B = B.view(
            cfg.num_blocks_row, cfg.num_blocks_col,
            cfg.group_row, cfg.group_col
        )
        return B


class BlockedWeightView:
    """Handles grouped view G with shape (n_grp_r, n_grp_c, blocks_per_scope, blk_r, blk_c)."""
    
    def __init__(self, G: torch.Tensor, config: GroupConfig):
        self.G = G  # (16, 2, 4, 16, 16)
        self.config = config
        
    def flatten_groups(self) -> torch.Tensor:
        """Flatten each group to a vector: (16, 2, 4, 256)."""
        cfg = self.config
        return self.G.view(
            cfg.num_scopes_row, cfg.num_scopes_col,
            cfg.blocks_per_scope,
            cfg.group_row * cfg.group_col
        )


class StructuredOBSSolver:
    """
    Vectorized Structured OBS implementation.
    
    For each block row (16 blocks), we maintain one H = X^T X and C = H^{-1}.
    Since groups within a block share the same input features (columns of X),
    they share the same Hessian structure.
    """
    
    def __init__(self, X: torch.Tensor, config: GroupConfig, device: str = 'cuda'):
        """
        Args:
            X: Input matrix of shape (m, k) = (batch, 128)
            config: Group configuration
            device: Device to run on
        """
        self.config = config
        self.device = device
        self.X = X.to(device)  # (m, 128)
        
        # Compute H = X^T X for each block column
        # Since blocks in the same column share the same input features,
        # we need H for each group column position
        self._compute_hessians()
        
        # Track pruned groups: (n_grp_r, n_grp_c, blocks_per_scope)
        self.pruned_mask = torch.zeros(
            config.num_scopes_row, config.num_scopes_col,
            config.blocks_per_scope,
            dtype=torch.bool, device=device
        )
        
        # Track remaining groups to prune per block
        self.remaining_to_prune = torch.full(
            (config.num_scopes_row, config.num_scopes_col),
            config.groups_to_prune,
            dtype=torch.int32, device=device
        )
        
    def _compute_hessians(self):
        """Compute group-diagonal Hessian and its inverse for each block."""
        cfg = self.config
        
        # H = X^T X has shape (k, k) = (128, 128)
        # For group structure, we consider the group-diagonal approximation
        H_full = self.X.T @ self.X  # (128, 128)
        
        # For each group column position, extract the corresponding H group
        # Group size along input dim is cfg.group_col = 16
        # We have cfg.num_blocks_col = 8 group columns
        
        # Reshape H to group view: (8, 16, 8, 16)
        H_scopes = H_full.view(
            cfg.num_blocks_col, cfg.group_col,
            cfg.num_blocks_col, cfg.group_col
        ).permute(0, 2, 1, 3)  # (n_blk_c, n_blk_c, blk_c, blk_c)
        
        # For structured OBS, we use group-diagonal approximation
        # H_b = H_scopes[i, i] for group column i
        
        # For grouped view, each block spans cfg.block_col group columns
        # So H for a block is the submatrix covering those columns
        
        # Precompute C = H^{-1} for each possible block of group columns
        # Block spans columns [j*block_col*group_col : (j+1)*block_col*group_col]
        
        self.H_inv_list = []  # List of C for each block column index
        
        for grp_c in range(cfg.num_scopes_col):
            start_col = grp_c * cfg.block_col * cfg.group_col
            end_col = (grp_c + 1) * cfg.block_col * cfg.group_col
            
            # Extract H for this block column
            H_grp = H_full[start_col:end_col, start_col:end_col]  # (64, 64)
            
            # Add regularization for numerical stability
            H_grp = H_grp + 1e-5 * torch.eye(H_grp.shape[0], device=self.device)
            
            # Compute inverse
            C_grp = torch.linalg.inv(H_grp)  # (64, 64)
            self.H_inv_list.append(C_grp)
        
        # Store C in blocked form for easier indexing: (2, 4, 16, 4, 16)
        # C_grp_col[b1, :, b2, :] gives C_{b1, b2}
        self.C_blocked = torch.stack([
            C.view(cfg.block_col, cfg.group_col, cfg.block_col, cfg.group_col)
            .permute(0, 2, 1, 3)  # (grp_c, grp_c, blk_c, blk_c)
            for C in self.H_inv_list
        ])  # (n_grp_c, grp_c, grp_c, blk_c, blk_c)
        
    def compute_pruning_scores(self, G_view: BlockedWeightView) -> torch.Tensor:
        """
        Compute OBS pruning scores for all groups.
        
        Score for group b: Tr[b @ (C_{b,b})^{-1} @ b^T]
        
        Args:
            G_view: Grouped weight view with shape (16, 2, 4, 16, 16)
            
        Returns:
            scores: Tensor of shape (16, 2, 4) with pruning scores
        """
        cfg = self.config
        G = G_view.G  # (n_grp_r, n_grp_c, blk_per_grp, blk_r, blk_c)
        
        # Flatten spatial dims of each group: (16, 2, 4, 256)
        G_flat = G_view.flatten_groups()
        
        scores = torch.zeros(
            cfg.num_scopes_row, cfg.num_scopes_col, cfg.blocks_per_scope,
            device=self.device
        )
        
        # For each block column, compute scores using corresponding C
        for grp_c in range(cfg.num_scopes_col):
            C = self.H_inv_list[grp_c]  # (64, 64)
            
            # Extract C_{b,b} groups for each group position
            # C is (64, 64), each group is (16, 16)
            # C_{b,b} for group b is the diagonal group
            
            # Reshape C to group view: (4, 16, 4, 16)
            C_scopes = C.view(
                cfg.blocks_per_scope, cfg.group_col,
                cfg.blocks_per_scope, cfg.group_col
            ).permute(0, 2, 1, 3)  # (4, 4, 16, 16)
            
            # Diagonal groups C_{b,b}: (4, 16, 16)
            C_diag = torch.diagonal(C_scopes, dim1=0, dim2=1).permute(2, 0, 1)  # (16, 4, 16)
            # Actually, let's do it properly:
            C_diag = torch.stack([C_scopes[b, b] for b in range(cfg.blocks_per_scope)])  # (4, 16, 16)
            
            # Compute (C_{b,b})^{-1} for each b
            C_bb_inv = torch.linalg.inv(C_diag + 1e-6 * torch.eye(cfg.group_col, device=self.device))  # (4, 16, 16)
            
            # For each block row and each group in the block
            for grp_r in range(cfg.num_scopes_row):
                for b in range(cfg.blocks_per_scope):
                    if self.pruned_mask[grp_r, grp_c, b]:
                        scores[grp_r, grp_c, b] = float('inf')
                        continue
                    
                    # Get group weights: (16, 16) -> flatten to (256,)
                    b_vec = G_flat[grp_r, grp_c, b]  # (256,)
                    
                    # Reshape to (blk_r, blk_c) = (16, 16)
                    b_mat = b_vec.view(cfg.group_row, cfg.group_col)  # (16, 16)
                    
                    # Compute Tr[b @ (C_{b,b})^{-1} @ b^T]
                    # = sum_{i,j} b_{i,j} * (C_{b,b}^{-1} @ b^T)_{j,i}
                    # = sum_i (b @ C_{b,b}^{-1} @ b^T)_{i,i}
                    
                    # b: (16, 16), C_bb_inv[b]: (16, 16)
                    temp = b_mat @ C_bb_inv[b]  # (16, 16)
                    score = torch.trace(temp @ b_mat.T)  # scalar
                    
                    scores[grp_r, grp_c, b] = score
        
        return scores
    
    def compute_pruning_scores_vectorized(self, G_view: BlockedWeightView) -> torch.Tensor:
        """
        Fully vectorized version of pruning scores.
        """
        cfg = self.config
        G = G_view.G  # (16, 2, 4, 16, 16)
        
        # Reshape to (n_grp_r, n_grp_c, blk_per_grp, blk_r, blk_c)
        scores = torch.zeros(
            cfg.num_scopes_row, cfg.num_scopes_col, cfg.blocks_per_scope,
            device=self.device
        )
        
        for grp_c in range(cfg.num_scopes_col):
            C = self.H_inv_list[grp_c]  # (64, 64)
            
            # Get all C_{b,b} groups: (4, 16, 16)
            C_scopes = C.view(cfg.blocks_per_scope, cfg.group_col, cfg.blocks_per_scope, cfg.group_col)
            C_scopes = C_scopes.permute(0, 2, 1, 3)  # (4, 4, 16, 16)
            C_diag = torch.stack([C_scopes[b, b] for b in range(cfg.blocks_per_scope)])  # (4, 16, 16)
            C_bb_inv = torch.linalg.inv(C_diag + 1e-6 * torch.eye(cfg.group_col, device=self.device))
            
            # G[:, grp_c] has shape (16, 4, 16, 16)
            G_grp = G[:, grp_c]  # (16, 4, 16, 16)
            
            # For each group position b
            for b in range(cfg.blocks_per_scope):
                b_weights = G_grp[:, b]  # (16, 16, 16)
                
                # Compute score for all block rows at once
                # b_weights: (16, 16, 16), C_bb_inv[b]: (16, 16)
                # We want: Tr[b @ C_bb_inv @ b^T] for each of 16 matrices
                
                # (16, 16, 16) @ (16, 16) -> (16, 16, 16)
                temp = b_weights @ C_bb_inv[b]  # (16, 16, 16)
                
                # b_weights @ C_bb_inv @ b_weights^T -> (16, 16, 16) @ (16, 16, 16).T
                # = (16, 16, 16) @ (16, 16, 16) permuted
                result = temp @ b_weights.transpose(-2, -1)  # (16, 16, 16)
                
                # Trace along last two dims
                score = torch.diagonal(result, dim1=-2, dim2=-1).sum(-1)  # (16,)
                
                scores[:, grp_c, b] = score
        
        # Mask pruned groups
        scores = torch.where(self.pruned_mask, torch.tensor(float('inf'), device=self.device), scores)
        
        return scores
    
    def update_weights(self, G_view: BlockedWeightView, grp_r: int, grp_c: int, b: int):
        """
        Apply OBS update after pruning group b from block (grp_r, grp_c).
        
        Update rule:
        G_{i,:}^{unpruned} <- G_{i,:}^{unpruned} - C_{:,b} @ (C_{b,b})^{-1} @ G_{i,b}
        
        Args:
            G_view: Grouped weight view to update in-place
            grp_r: Block row index
            grp_c: Block column index
            b: Group index within block to prune
        """
        cfg = self.config
        G = G_view.G  # (16, 2, 4, 16, 16)
        
        # Get C for this block column
        C = self.H_inv_list[grp_c]  # (64, 64)
        
        # Extract C_{:,b} and C_{b,b}
        # C is (64, 64), group size is 16
        # Group b corresponds to rows/cols [b*16 : (b+1)*16]
        
        start_idx = b * cfg.group_col
        end_idx = (b + 1) * cfg.group_col
        
        C_col_b = C[:, start_idx:end_idx]  # (64, 16) - all columns for group b rows
        C_bb = C[start_idx:end_idx, start_idx:end_idx]  # (16, 16)
        C_bb_inv = torch.linalg.inv(C_bb + 1e-6 * torch.eye(cfg.group_col, device=self.device))
        
        # Get the weight group to prune
        G_ib = G[grp_r, grp_c, b]  # (16, 16)
        
        # Compute update factor: C_{:,b} @ (C_{b,b})^{-1} @ G_{i,b}
        # C_col_b: (64, 16), C_bb_inv: (16, 16), G_ib: (16, 16)
        
        # First: C_bb_inv @ G_ib -> we need to flatten G_ib
        G_ib_flat = G_ib.view(-1)  # (256,)
        
        # Actually, the formula is for vectorized weights
        # Let's flatten everything properly
        
        # G[grp_r, grp_c] has shape (4, 16, 16) -> flatten to (4, 256)
        G_grp = G[grp_r, grp_c].view(cfg.blocks_per_scope, -1)  # (4, 256)
        
        # C is (64, 64), reshape to (4, 16, 4, 16) -> (4, 4, 16, 16)
        C_scopes = C.view(cfg.blocks_per_scope, cfg.group_col, cfg.blocks_per_scope, cfg.group_col)
        C_scopes = C_scopes.permute(0, 2, 1, 3)  # (4, 4, 16, 16)
        
        # C_{:,b} means all group rows, group column b
        # Shape: (4, 16, 16) for each source group, stacked: (4, 4, 16, 16) where last dim is b
        C_col_b_scopes = C_scopes[:, b, :, :]  # (4, 16, 16)
        
        # C_{b,b}
        C_bb_block = C_scopes[b, b]  # (16, 16)
        C_bb_inv_block = torch.linalg.inv(C_bb_block + 1e-6 * torch.eye(cfg.group_col, device=self.device))
        
        # G_{i,b}
        G_ib_vec = G_grp[b]  # (256,)
        
        # Compute (C_{b,b})^{-1} @ G_{i,b} (in group form)
        # G_ib_vec: (256,) = (16, 16) flattened
        G_ib_mat = G_ib_vec.view(cfg.group_row, cfg.group_col)  # (16, 16)
        
        # C_bb_inv_block: (16, 16), G_ib_mat: (16, 16)
        temp = C_bb_inv_block @ G_ib_mat.T  # (16, 16) - wait, dimensions don't match
        
        # Actually, C relates to input features (columns of W)
        # So C_{b,b} is (16, 16) and G_{i,b} is (16, 16)
        # The product C_{:,b} @ (C_{b,b})^{-1} @ G_{i,b} needs care
        
        # Let's think: C is (k_per_group, k_per_group) = (64, 64)
        # G_{i,b} is (out_block, in_block) = (16, 16)
        
        # The formula G_{i,:}^{unpruned} suggests we're updating all input groups
        # So we flatten input dim: G_{i,:} has shape (4*16,) = (64,)
        
        G_i_flat = G_grp.view(-1)  # (1024,) = (4*256)
        
        # But that's not right either. Let's reconsider.
        
        # Actually, for OBS, if W is (n, k), and we prune based on input features (columns),
        # then the update is for each output row independently.
        
        # G[grp_r, grp_c, :, :, :] is (4, 16, 16) = (groups, out_dim, in_dim_per_block)
        # For each output position j in [0, 15], we have a vector of 64 input weights
        
        # Reshape: (4, 16, 16) -> (16, 4, 16) = (out_pos, groups, in_dim)
        G_per_output = G[grp_r, grp_c].permute(1, 0, 2)  # (16, 4, 16)
        G_per_output = G_per_output.reshape(cfg.group_row, -1)  # (16, 64)
        
        # Now for each output position, we have a 64-dim vector
        # C is (64, 64)
        
        # Group b corresponds to indices [b*16 : (b+1)*16] in the 64-dim space
        start = b * cfg.group_col
        end = (b + 1) * cfg.group_col
        
        C_col_b = C[:, start:end]  # (64, 16)
        C_bb = C[start:end, start:end]  # (16, 16)
        C_bb_inv = torch.linalg.inv(C_bb + 1e-6 * torch.eye(cfg.group_col, device=self.device))
        
        # For each output position j
        for j in range(cfg.group_row):
            w_jb = G_per_output[j, start:end]  # (16,) - weights for group b, output j
            
            # Update: w_j,unpruned <- w_j,unpruned - C_{:,b} @ C_{b,b}^{-1} @ w_jb
            update = C_col_b @ C_bb_inv @ w_jb  # (64,)
            
            # Create mask for unpruned indices
            unpruned_mask = torch.ones(64, dtype=torch.bool, device=self.device)
            unpruned_mask[start:end] = False
            
            # Apply update only to unpruned
            G_per_output[j, unpruned_mask] -= update[unpruned_mask]
        
        # Reshape back
        G[grp_r, grp_c] = G_per_output.view(cfg.group_row, cfg.blocks_per_scope, cfg.group_col).permute(1, 0, 2)
        
    def update_weights_vectorized(self, G_view: BlockedWeightView, grp_r: int, grp_c: int, b: int):
        """
        Vectorized version of OBS weight update for a single block.
        """
        cfg = self.config
        G = G_view.G
        
        C = self.H_inv_list[grp_c]  # (64, 64)
        
        start = b * cfg.group_col
        end = (b + 1) * cfg.group_col
        
        C_col_b = C[:, start:end]  # (64, 16)
        C_bb = C[start:end, start:end]  # (16, 16)
        C_bb_inv = torch.linalg.inv(C_bb + 1e-6 * torch.eye(cfg.group_col, device=self.device))
        
        # G[grp_r, grp_c]: (4, 16, 16) -> (16, 64) for all outputs
        G_grp = G[grp_r, grp_c].permute(1, 0, 2).reshape(cfg.group_row, -1)  # (16, 64)
        
        # Extract group b for all outputs: (16, 16)
        W_b = G_grp[:, start:end]  # (16, 16)
        
        # Compute update for all outputs at once: (16, 64)
        # C_col_b @ C_bb_inv @ W_b^T -> (64, 16) @ (16, 16) @ (16, 16) = wait
        
        # Correct: for each row j, we want C_col_b @ C_bb_inv @ W_b[j]
        # C_col_b: (64, 16), C_bb_inv: (16, 16), W_b: (16, 16)
        
        # (64, 16) @ (16, 16) = (64, 16)
        temp = C_col_b @ C_bb_inv  # (64, 16)
        
        # (64, 16) @ (16, 16).T = (64, 16) @ (16, 16) = (64, 16)
        # But we need (16, 64) to match G_grp shape
        
        # Actually: for each j, update_j = temp @ W_b[j] = (64, 16) @ (16,) = (64,)
        # So for all j: (16, 64) = W_b @ temp.T = (16, 16) @ (16, 64) = (16, 64)
        update = W_b @ temp.T  # (16, 64)
        
        # Mask pruned group
        unpruned_mask = torch.ones(64, dtype=torch.bool, device=self.device)
        unpruned_mask[start:end] = False
        
        # Apply update
        G_grp[:, unpruned_mask] -= update[:, unpruned_mask]
        
        # Zero out pruned group
        G_grp[:, start:end] = 0
        
        # Reshape back
        G[grp_r, grp_c] = G_grp.view(cfg.group_row, cfg.blocks_per_scope, cfg.group_col).permute(1, 0, 2)
        
    def update_C(self, grp_c: int, b: int):
        """
        Update C to remove pruned group b using Schur complement.
        
        C <- C - C_{:,b} @ (C_{b,b})^{-1} @ C_{b,:}
        """
        cfg = self.config
        C = self.H_inv_list[grp_c]  # (64, 64)
        
        start = b * cfg.group_col
        end = (b + 1) * cfg.group_col
        
        C_col_b = C[:, start:end]  # (64, 16)
        C_bb = C[start:end, start:end]  # (16, 16)
        C_bb_inv = torch.linalg.inv(C_bb + 1e-6 * torch.eye(cfg.group_col, device=self.device))
        C_b_row = C[start:end, :]  # (16, 64)
        
        # Schur complement update
        update = C_col_b @ C_bb_inv @ C_b_row  # (64, 64)
        C_new = C - update
        
        # Zero out the pruned group rows/cols for numerical stability
        C_new[start:end, :] = 0
        C_new[:, start:end] = 0
        
        self.H_inv_list[grp_c] = C_new
        
    def prune_step(self, G_view: BlockedWeightView) -> bool:
        """
        Perform one pruning step across all blocks.
        
        Returns:
            True if pruning was performed, False if no more groups to prune
        """
        cfg = self.config
        
        # Check if we have groups left to prune
        if self.remaining_to_prune.sum() == 0:
            return False
        
        # Compute scores for all groups
        scores = self.compute_pruning_scores_vectorized(G_view)
        
        # For each block that still needs pruning, find the group with minimum score
        for grp_r in range(cfg.num_scopes_row):
            for grp_c in range(cfg.num_scopes_col):
                if self.remaining_to_prune[grp_r, grp_c] > 0:
                    # Get scores for this block
                    grp_scores = scores[grp_r, grp_c]  # (4,)
                    
                    # Find minimum
                    b_to_prune = torch.argmin(grp_scores).item()
                    
                    # Prune this group
                    self._prune_block(G_view, grp_r, grp_c, b_to_prune)
                    
        return True
    
    def _prune_block(self, G_view: BlockedWeightView, grp_r: int, grp_c: int, b: int):
        """Prune a specific group and update state."""
        # Update weights
        self.update_weights_vectorized(G_view, grp_r, grp_c, b)
        
        # Update C
        self.update_C(grp_c, b)
        
        # Mark as pruned
        self.pruned_mask[grp_r, grp_c, b] = True
        self.remaining_to_prune[grp_r, grp_c] -= 1
        
    def prune_all(self, W0: torch.Tensor) -> torch.Tensor:
        """
        Prune all groups according to config.
        
        Args:
            W0: Original weight matrix (256, 128)
            
        Returns:
            Pruned weight matrix
        """
        cfg = self.config
        
        # Create blocked view
        B_view = GroupedWeightView(W0, cfg)
        
        # Convert to grouped view
        G_view = B_view.to_grouped()
        
        # Prune until done
        step = 0
        while self.prune_step(G_view):
            step += 1
            if step >= cfg.groups_to_prune * cfg.num_scopes_row * cfg.num_scopes_col:
                break
        
        # Convert back to weight matrix
        B_pruned = B_view.from_grouped(G_view)
        
        # Reshape to original
        W_pruned = B_pruned.permute(0, 2, 1, 3).reshape(cfg.n, cfg.k)
        
        return W_pruned


class FullyVectorizedStructuredOBS(StructuredOBSSolver):
    """
    Fully vectorized version that processes all blocks simultaneously.
    """
    
    def __init__(self, X: torch.Tensor, config: GroupConfig, device: str = 'cuda'):
        super().__init__(X, config, device)
        
        # Precompute C in a more accessible format
        # Stack all C matrices: (n_grp_c, 64, 64)
        self.C_stacked = torch.stack(self.H_inv_list)  # (2, 64, 64)
        
    def compute_all_scores(self, G_view: BlockedWeightView) -> torch.Tensor:
        """
        Compute scores for all groups in all blocks simultaneously.
        """
        cfg = self.config
        G = G_view.G  # (16, 2, 4, 16, 16)
        
        # Reshape G to (n_grp_r, n_grp_c, blocks_per_scope, blk_r, blk_c)
        # We need to compute Tr[G_{i,b} @ (C_{b,b})^{-1} @ G_{i,b}^T]
        
        # For each block column, C is different
        # For each block column grp_c, C_stacked[grp_c] is (64, 64)
        
        scores = torch.zeros(
            cfg.num_scopes_row, cfg.num_scopes_col, cfg.blocks_per_scope,
            device=self.device
        )
        
        for grp_c in range(cfg.num_scopes_col):
            C = self.C_stacked[grp_c]  # (64, 64)
            
            # Extract diagonal groups C_{b,b}: (4, 16, 16)
            C_scopes = C.view(cfg.blocks_per_scope, cfg.group_col, cfg.blocks_per_scope, cfg.group_col)
            C_scopes = C_scopes.permute(0, 2, 1, 3)  # (4, 4, 16, 16)
            
            # Get diagonal
            C_bb = torch.stack([C_scopes[b, b] for b in range(cfg.blocks_per_scope)])  # (4, 16, 16)
            C_bb_inv = torch.linalg.inv(C_bb + 1e-5 * torch.eye(cfg.group_col, device=self.device))
            
            # G[:, grp_c]: (16, 4, 16, 16)
            G_grp = G[:, grp_c]  # (16, 4, 16, 16)
            
            # Reshape to (16, 4, 16, 16) where last two are (out, in)
            # We want Tr[G_b @ C_bb_inv[b] @ G_b^T] for each block row and group
            
            # Method: batch matrix multiply
            # G_grp: (16, 4, 16, 16)
            # For each b in [0,3], we have 16 matrices of shape (16, 16)
            
            for b in range(cfg.blocks_per_scope):
                G_b = G_grp[:, b]  # (16, 16, 16) - 16 matrices of (16, 16)
                
                # Compute G_b @ C_bb_inv[b]: (16, 16, 16) @ (16, 16)
                temp = G_b @ C_bb_inv[b]  # (16, 16, 16)
                
                # Compute trace of temp @ G_b^T
                # = sum_i (temp @ G_b^T)_{i,i}
                # = sum_i sum_k temp_{i,k} * G_b_{i,k}
                
                result = temp @ G_b.transpose(-2, -1)  # (16, 16, 16)
                trace = torch.diagonal(result, dim1=-2, dim2=-1).sum(-1)  # (16,)
                
                scores[:, grp_c, b] = trace
        
        # Apply pruned mask
        scores = torch.where(self.pruned_mask, torch.tensor(float('inf'), device=self.device), scores)
        
        return scores
    
    def batch_update_weights(self, G_view: BlockedWeightView, prune_indices: torch.Tensor):
        """
        Update weights for multiple pruned groups simultaneously.
        
        Args:
            G_view: Weight view
            prune_indices: (n_grp_r, n_grp_c) tensor of group indices to prune
        """
        cfg = self.config
        G = G_view.G
        
        # Process each block column separately (since they have different C)
        for grp_c in range(cfg.num_scopes_col):
            C = self.C_stacked[grp_c]  # (64, 64)
            
            # Get all block rows for this column
            groups_to_prune = prune_indices[:, grp_c]  # (16,)
            
            # For each unique group to prune in this column
            for b in range(cfg.blocks_per_scope):
                mask = (groups_to_prune == b) & (~self.pruned_mask[:, grp_c, b])
                if mask.sum() == 0:
                    continue
                
                grp_r_list = torch.where(mask)[0]
                
                start = b * cfg.group_col
                end = (b + 1) * cfg.group_col
                
                C_col_b = C[:, start:end]  # (64, 16)
                C_bb = C[start:end, start:end]  # (16, 16)
                C_bb_inv = torch.linalg.inv(C_bb + 1e-6 * torch.eye(cfg.group_col, device=self.device))
                
                # Precompute temp = C_col_b @ C_bb_inv: (64, 16)
                temp = C_col_b @ C_bb_inv
                
                for grp_r in grp_r_list:
                    # G[grp_r, grp_c]: (4, 16, 16) -> (16, 64)
                    G_grp = G[grp_r, grp_c].permute(1, 0, 2).reshape(cfg.group_row, -1)  # (16, 64)
                    
                    W_b = G_grp[:, start:end]  # (16, 16)
                    
                    # Update: (16, 16) @ (16, 64) = (16, 64)
                    update = W_b @ temp.T  # (16, 64)
                    
                    unpruned_mask = torch.ones(64, dtype=torch.bool, device=self.device)
                    unpruned_mask[start:end] = False
                    
                    G_grp[:, unpruned_mask] -= update[:, unpruned_mask]
                    G_grp[:, start:end] = 0
                    
                    # Reshape back
                    G[grp_r, grp_c] = G_grp.view(cfg.group_row, cfg.blocks_per_scope, cfg.group_col).permute(1, 0, 2)
                    
                    # Mark pruned
                    self.pruned_mask[grp_r, grp_c, b] = True
                    self.remaining_to_prune[grp_r, grp_c] -= 1
    
    def batch_update_C(self, grp_c: int, pruned_scopes: List[int]):
        """Update C after pruning multiple groups."""
        cfg = self.config
        C = self.C_stacked[grp_c].clone()
        
        for b in pruned_scopes:
            if self.pruned_mask[0, grp_c, b]:  # Check if already pruned
                continue
                
            start = b * cfg.group_col
            end = (b + 1) * cfg.group_col
            
            C_col_b = C[:, start:end]
            C_bb = C[start:end, start:end]
            C_bb_inv = torch.linalg.inv(C_bb + 1e-6 * torch.eye(cfg.group_col, device=self.device))
            C_b_row = C[start:end, :]
            
            C = C - C_col_b @ C_bb_inv @ C_b_row
            
            # Zero out
            C[start:end, :] = 0
            C[:, start:end] = 0
        
        self.C_stacked[grp_c] = C


# Example usage
def demo():
    """Demonstrate the structured OBS pruning."""
    torch.manual_seed(42)
    
    config = GroupConfig(
        n=256, k=128,
        group_row=16, group_col=16,
        block_row=1, block_col=4,
        groups_to_prune=2
    )
    
    # Create random input and weights
    m = 64  # batch size
    X = torch.randn(m, config.k)
    W0 = torch.randn(config.n, config.k)
    
    # Initialize solver
    solver = FullyVectorizedStructuredOBS(X, config, device='cpu')
    
    # Prune
    W_pruned = solver.prune_all(W0)
    
    # Check sparsity pattern
    B_view = GroupedWeightView(W_pruned, config)
    G_view = B_view.to_grouped()
    
    print("Grouped view shape:", G_view.G.shape)
    print("Expected: (16, 2, 4, 16, 16)")
    
    # Check which groups are zero
    G_flat = G_view.flatten_groups()  # (16, 2, 4, 256)
    block_norms = G_flat.norm(dim=-1)  # (16, 2, 4)
    
    print("\nBlock norms (should see 2 zeros per block):")
    for grp_r in range(config.num_scopes_row):
        for grp_c in range(config.num_scopes_col):
            print(f"Block ({grp_r}, {grp_c}): {block_norms[grp_r, grp_c].tolist()}")
    
    # Compute loss
    loss_orig = torch.norm(X @ W0.T - X @ W0.T)**2
    loss_pruned = torch.norm(X @ W_pruned.T - X @ W0.T)**2
    
    print(f"\nOriginal loss: {loss_orig.item():.6f}")
    print(f"Pruned loss: {loss_pruned.item():.6f}")
    print(f"Relative increase: {(loss_pruned/loss_orig - 1)*100:.2f}%")


if __name__ == "__main__":
    demo()