import torch
from typing import List
from dataclasses import dataclass


@dataclass
class BlockConfig:
    """Configuration for block structure."""
    n: int = 256          # Output dim
    k: int = 128          # Input dim
    block_row: int = 4   # Block size along output dim
    block_col: int = 4   # Block size along input dim
    group_row: int = 1    # Group size along output dim (in blocks)
    group_col: int = 4    # Group size along input dim (in blocks)
    blocks_to_prune: int = 2  # Blocks to prune per group
    
    @property
    def num_blocks_row(self) -> int:
        return self.n // self.block_row  # 16
    
    @property
    def num_blocks_col(self) -> int:
        return self.k // self.block_col  # 8
    
    @property
    def num_groups_row(self) -> int:
        return self.num_blocks_row // self.group_row  # 16
    
    @property
    def num_groups_col(self) -> int:
        return self.num_blocks_col // self.group_col  # 2
    
    @property
    def blocks_per_group(self) -> int:
        return self.group_row * self.group_col  # 4


class BlockedWeightView:
    """Handles blocked view B with shape (num_blocks_row, num_blocks_col, block_row, block_col)."""
    
    def __init__(self, weight: torch.Tensor, config: BlockConfig):
        self.config = config
        self.n, self.k = weight.shape
        assert self.n == config.n and self.k == config.k
        
        # Reshape to blocked view: (16, 8, 16, 16)
        self.B = weight.view(
            config.num_blocks_row, config.block_row,
            config.num_blocks_col, config.block_col
        ).permute(0, 2, 1, 3).contiguous()  # (n_blocks_r, n_blocks_c, blk_r, blk_c)
        
    def to_grouped(self) -> 'GroupedWeightView':
        """Convert to grouped view G with shape (16, 2, 4, 16, 16)."""
        cfg = self.config
        # Reshape blocks into groups: (16, 2, 4, 16, 16)
        # group_row=1, group_col=4 means we group 1x4 blocks together
        G = self.B.view(
            cfg.num_groups_row, cfg.group_row,  # 16, 1
            cfg.num_groups_col, cfg.group_col,  # 2, 4
            cfg.block_row, cfg.block_col        # 16, 16
        ).permute(0, 2, 1, 3, 4, 5).contiguous()  # (n_grp_r, n_grp_c, grp_r, grp_c, blk_r, blk_c)
        
        # Flatten group dimensions: (16, 2, 4, 16, 16)
        G = G.view(
            cfg.num_groups_row, cfg.num_groups_col,
            cfg.blocks_per_group,
            cfg.block_row, cfg.block_col
        )
        return GroupedWeightView(G, self.config)
    
    def from_grouped(self, G_view: 'GroupedWeightView') -> torch.Tensor:
        """Convert grouped view back to blocked view."""
        cfg = self.config
        G = G_view.G  # (16, 2, 4, 16, 16)
        
        # Reshape to separate group dimensions: (16, 2, 1, 4, 16, 16)
        G = G.view(
            cfg.num_groups_row, cfg.num_groups_col,
            cfg.group_row, cfg.group_col,
            cfg.block_row, cfg.block_col
        )
        
        # Permute back to blocked structure: (16, 8, 16, 16)
        B = G.permute(0, 2, 1, 3, 4, 5).contiguous()
        B = B.view(
            cfg.num_blocks_row, cfg.num_blocks_col,
            cfg.block_row, cfg.block_col
        )
        return B


class GroupedWeightView:
    """Handles grouped view G with shape (n_grp_r, n_grp_c, blocks_per_group, blk_r, blk_c)."""
    
    def __init__(self, G: torch.Tensor, config: BlockConfig):
        self.G = G  # (16, 2, 4, 16, 16)
        self.config = config
        
    def flatten_blocks(self) -> torch.Tensor:
        """Flatten each block to a vector: (16, 2, 4, 256)."""
        cfg = self.config
        return self.G.view(
            cfg.num_groups_row, cfg.num_groups_col,
            cfg.blocks_per_group,
            cfg.block_row * cfg.block_col
        )


class StructuredOBSSolver:
    """
    Vectorized Structured OBS implementation.
    
    For each group row (16 groups), we maintain one H = X^T X and C = H^{-1}.
    Since blocks within a group share the same input features (columns of X),
    they share the same Hessian structure.
    """
    
    def __init__(self, X: torch.Tensor, config: BlockConfig, device: str = 'cuda'):
        """
        Args:
            X: Input matrix of shape (m, k) = (batch, 128)
            config: Block configuration
            device: Device to run on
        """
        self.config = config
        self.device = device
        self.X = X.to(device)  # (m, 128)
        
        # Compute H = X^T X for each group column
        # Since groups in the same column share the same input features,
        # we need H for each block column position
        self._compute_hessians()
        
        # Track pruned blocks: (n_grp_r, n_grp_c, blocks_per_group)
        self.pruned_mask = torch.zeros(
            config.num_groups_row, config.num_groups_col,
            config.blocks_per_group,
            dtype=torch.bool, device=device
        )
        
        # Track remaining blocks to prune per group
        self.remaining_to_prune = torch.full(
            (config.num_groups_row, config.num_groups_col),
            config.blocks_to_prune,
            dtype=torch.int32, device=device
        )
        
    def _compute_hessians(self):
        """Compute block-diagonal Hessian and its inverse for each group."""
        cfg = self.config
        
        # H = X^T X has shape (k, k) = (128, 128)
        # For block structure, we consider the block-diagonal approximation
        H_full = self.X.T @ self.X  # (128, 128)
        
        # For each block column position, extract the corresponding H block
        # Block size along input dim is cfg.block_col = 16
        # We have cfg.num_blocks_col = 8 block columns
        
        # Reshape H to block view: (8, 16, 8, 16)
        H_blocks = H_full.view(
            cfg.num_blocks_col, cfg.block_col,
            cfg.num_blocks_col, cfg.block_col
        ).permute(0, 2, 1, 3)  # (n_blk_c, n_blk_c, blk_c, blk_c)
        
        # For structured OBS, we use block-diagonal approximation
        # H_b = H_blocks[i, i] for block column i
        
        # For grouped view, each group spans cfg.group_col block columns
        # So H for a group is the submatrix covering those columns
        
        # Precompute C = H^{-1} for each possible group of block columns
        # Group spans columns [j*group_col*block_col : (j+1)*group_col*block_col]
        
        self.H_inv_list = []  # List of C for each group column index
        
        for grp_c in range(cfg.num_groups_col):
            start_col = grp_c * cfg.group_col * cfg.block_col
            end_col = (grp_c + 1) * cfg.group_col * cfg.block_col
            
            # Extract H for this group column
            H_grp = H_full[start_col:end_col, start_col:end_col]  # (64, 64)
            
            # Add regularization for numerical stability
            H_grp = H_grp + 1e-5 * torch.eye(H_grp.shape[0], device=self.device)
            
            # Compute inverse
            C_grp = torch.linalg.inv(H_grp)  # (64, 64)
            self.H_inv_list.append(C_grp)
        
        # Store C in blocked form for easier indexing: (2, 4, 16, 4, 16)
        # C_grp_col[b1, :, b2, :] gives C_{b1, b2}
        self.C_blocked = torch.stack([
            C.view(cfg.group_col, cfg.block_col, cfg.group_col, cfg.block_col)
            .permute(0, 2, 1, 3)  # (grp_c, grp_c, blk_c, blk_c)
            for C in self.H_inv_list
        ])  # (n_grp_c, grp_c, grp_c, blk_c, blk_c)
        
    def compute_pruning_scores(self, G_view: GroupedWeightView) -> torch.Tensor:
        """
        Compute OBS pruning scores for all blocks.
        
        Score for block b: Tr[b @ (C_{b,b})^{-1} @ b^T]
        
        Args:
            G_view: Grouped weight view with shape (16, 2, 4, 16, 16)
            
        Returns:
            scores: Tensor of shape (16, 2, 4) with pruning scores
        """
        cfg = self.config
        G = G_view.G  # (n_grp_r, n_grp_c, blk_per_grp, blk_r, blk_c)
        
        # Flatten spatial dims of each block: (16, 2, 4, 256)
        G_flat = G_view.flatten_blocks()
        
        scores = torch.zeros(
            cfg.num_groups_row, cfg.num_groups_col, cfg.blocks_per_group,
            device=self.device
        )
        
        # For each group column, compute scores using corresponding C
        for grp_c in range(cfg.num_groups_col):
            C = self.H_inv_list[grp_c]  # (64, 64)
            
            # Extract C_{b,b} blocks for each block position
            # C is (64, 64), each block is (16, 16)
            # C_{b,b} for block b is the diagonal block
            
            # Reshape C to block view: (4, 16, 4, 16)
            C_blocks = C.view(
                cfg.blocks_per_group, cfg.block_col,
                cfg.blocks_per_group, cfg.block_col
            ).permute(0, 2, 1, 3)  # (4, 4, 16, 16)
            
            # Diagonal blocks C_{b,b}: (4, 16, 16)
            C_diag = torch.diagonal(C_blocks, dim1=0, dim2=1).permute(2, 0, 1)  # (16, 4, 16)
            # Actually, let's do it properly:
            C_diag = torch.stack([C_blocks[b, b] for b in range(cfg.blocks_per_group)])  # (4, 16, 16)
            
            # Compute (C_{b,b})^{-1} for each b
            C_bb_inv = torch.linalg.inv(C_diag + 1e-6 * torch.eye(cfg.block_col, device=self.device))  # (4, 16, 16)
            
            # For each group row and each block in the group
            for grp_r in range(cfg.num_groups_row):
                for b in range(cfg.blocks_per_group):
                    if self.pruned_mask[grp_r, grp_c, b]:
                        scores[grp_r, grp_c, b] = float('inf')
                        continue
                    
                    # Get block weights: (16, 16) -> flatten to (256,)
                    b_vec = G_flat[grp_r, grp_c, b]  # (256,)
                    
                    # Reshape to (blk_r, blk_c) = (16, 16)
                    b_mat = b_vec.view(cfg.block_row, cfg.block_col)  # (16, 16)
                    
                    # Compute Tr[b @ (C_{b,b})^{-1} @ b^T]
                    # = sum_{i,j} b_{i,j} * (C_{b,b}^{-1} @ b^T)_{j,i}
                    # = sum_i (b @ C_{b,b}^{-1} @ b^T)_{i,i}
                    
                    # b: (16, 16), C_bb_inv[b]: (16, 16)
                    temp = b_mat @ C_bb_inv[b]  # (16, 16)
                    score = torch.trace(temp @ b_mat.T)  # scalar
                    
                    scores[grp_r, grp_c, b] = score
        
        return scores
    
    def compute_pruning_scores_vectorized(self, G_view: GroupedWeightView) -> torch.Tensor:
        """
        Fully vectorized version of pruning scores.
        """
        cfg = self.config
        G = G_view.G  # (16, 2, 4, 16, 16)
        
        # Reshape to (n_grp_r, n_grp_c, blk_per_grp, blk_r, blk_c)
        scores = torch.zeros(
            cfg.num_groups_row, cfg.num_groups_col, cfg.blocks_per_group,
            device=self.device
        )
        
        for grp_c in range(cfg.num_groups_col):
            C = self.H_inv_list[grp_c]  # (64, 64)
            
            # Get all C_{b,b} blocks: (4, 16, 16)
            C_blocks = C.view(cfg.blocks_per_group, cfg.block_col, cfg.blocks_per_group, cfg.block_col)
            C_blocks = C_blocks.permute(0, 2, 1, 3)  # (4, 4, 16, 16)
            C_diag = torch.stack([C_blocks[b, b] for b in range(cfg.blocks_per_group)])  # (4, 16, 16)
            C_bb_inv = torch.linalg.inv(C_diag + 1e-6 * torch.eye(cfg.block_col, device=self.device))
            
            # G[:, grp_c] has shape (16, 4, 16, 16)
            G_grp = G[:, grp_c]  # (16, 4, 16, 16)
            
            # For each block position b
            for b in range(cfg.blocks_per_group):
                b_weights = G_grp[:, b]  # (16, 16, 16)
                
                # Compute score for all group rows at once
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
        
        # Mask pruned blocks
        scores = torch.where(self.pruned_mask, torch.tensor(float('inf'), device=self.device), scores)
        
        return scores
    
    def update_weights(self, G_view: GroupedWeightView, grp_r: int, grp_c: int, b: int):
        """
        Apply OBS update after pruning block b from group (grp_r, grp_c).
        
        Update rule:
        G_{i,:}^{unpruned} <- G_{i,:}^{unpruned} - C_{:,b} @ (C_{b,b})^{-1} @ G_{i,b}
        
        Args:
            G_view: Grouped weight view to update in-place
            grp_r: Group row index
            grp_c: Group column index
            b: Block index within group to prune
        """
        cfg = self.config
        G = G_view.G  # (16, 2, 4, 16, 16)
        
        # Get C for this group column
        C = self.H_inv_list[grp_c]  # (64, 64)
        
        # Extract C_{:,b} and C_{b,b}
        # C is (64, 64), block size is 16
        # Block b corresponds to rows/cols [b*16 : (b+1)*16]
        
        start_idx = b * cfg.block_col
        end_idx = (b + 1) * cfg.block_col
        
        C_col_b = C[:, start_idx:end_idx]  # (64, 16) - all columns for block b rows
        C_bb = C[start_idx:end_idx, start_idx:end_idx]  # (16, 16)
        C_bb_inv = torch.linalg.inv(C_bb + 1e-6 * torch.eye(cfg.block_col, device=self.device))
        
        # Get the weight block to prune
        G_ib = G[grp_r, grp_c, b]  # (16, 16)
        
        # Compute update factor: C_{:,b} @ (C_{b,b})^{-1} @ G_{i,b}
        # C_col_b: (64, 16), C_bb_inv: (16, 16), G_ib: (16, 16)
        
        # First: C_bb_inv @ G_ib -> we need to flatten G_ib
        G_ib_flat = G_ib.view(-1)  # (256,)
        
        # Actually, the formula is for vectorized weights
        # Let's flatten everything properly
        
        # G[grp_r, grp_c] has shape (4, 16, 16) -> flatten to (4, 256)
        G_grp = G[grp_r, grp_c].view(cfg.blocks_per_group, -1)  # (4, 256)
        
        # C is (64, 64), reshape to (4, 16, 4, 16) -> (4, 4, 16, 16)
        C_blocks = C.view(cfg.blocks_per_group, cfg.block_col, cfg.blocks_per_group, cfg.block_col)
        C_blocks = C_blocks.permute(0, 2, 1, 3)  # (4, 4, 16, 16)
        
        # C_{:,b} means all block rows, block column b
        # Shape: (4, 16, 16) for each source block, stacked: (4, 4, 16, 16) where last dim is b
        C_col_b_blocks = C_blocks[:, b, :, :]  # (4, 16, 16)
        
        # C_{b,b}
        C_bb_block = C_blocks[b, b]  # (16, 16)
        C_bb_inv_block = torch.linalg.inv(C_bb_block + 1e-6 * torch.eye(cfg.block_col, device=self.device))
        
        # G_{i,b}
        G_ib_vec = G_grp[b]  # (256,)
        
        # Compute (C_{b,b})^{-1} @ G_{i,b} (in block form)
        # G_ib_vec: (256,) = (16, 16) flattened
        G_ib_mat = G_ib_vec.view(cfg.block_row, cfg.block_col)  # (16, 16)
        
        # C_bb_inv_block: (16, 16), G_ib_mat: (16, 16)
        temp = C_bb_inv_block @ G_ib_mat.T  # (16, 16) - wait, dimensions don't match
        
        # Actually, C relates to input features (columns of W)
        # So C_{b,b} is (16, 16) and G_{i,b} is (16, 16)
        # The product C_{:,b} @ (C_{b,b})^{-1} @ G_{i,b} needs care
        
        # Let's think: C is (k_per_group, k_per_group) = (64, 64)
        # G_{i,b} is (out_block, in_block) = (16, 16)
        
        # The formula G_{i,:}^{unpruned} suggests we're updating all input blocks
        # So we flatten input dim: G_{i,:} has shape (4*16,) = (64,)
        
        G_i_flat = G_grp.view(-1)  # (1024,) = (4*256)
        
        # But that's not right either. Let's reconsider.
        
        # Actually, for OBS, if W is (n, k), and we prune based on input features (columns),
        # then the update is for each output row independently.
        
        # G[grp_r, grp_c, :, :, :] is (4, 16, 16) = (blocks, out_dim, in_dim_per_block)
        # For each output position j in [0, 15], we have a vector of 64 input weights
        
        # Reshape: (4, 16, 16) -> (16, 4, 16) = (out_pos, blocks, in_dim)
        G_per_output = G[grp_r, grp_c].permute(1, 0, 2)  # (16, 4, 16)
        G_per_output = G_per_output.reshape(cfg.block_row, -1)  # (16, 64)
        
        # Now for each output position, we have a 64-dim vector
        # C is (64, 64)
        
        # Block b corresponds to indices [b*16 : (b+1)*16] in the 64-dim space
        start = b * cfg.block_col
        end = (b + 1) * cfg.block_col
        
        C_col_b = C[:, start:end]  # (64, 16)
        C_bb = C[start:end, start:end]  # (16, 16)
        C_bb_inv = torch.linalg.inv(C_bb + 1e-6 * torch.eye(cfg.block_col, device=self.device))
        
        # For each output position j
        for j in range(cfg.block_row):
            w_jb = G_per_output[j, start:end]  # (16,) - weights for block b, output j
            
            # Update: w_j,unpruned <- w_j,unpruned - C_{:,b} @ C_{b,b}^{-1} @ w_jb
            update = C_col_b @ C_bb_inv @ w_jb  # (64,)
            
            # Create mask for unpruned indices
            unpruned_mask = torch.ones(64, dtype=torch.bool, device=self.device)
            unpruned_mask[start:end] = False
            
            # Apply update only to unpruned
            G_per_output[j, unpruned_mask] -= update[unpruned_mask]
        
        # Reshape back
        G[grp_r, grp_c] = G_per_output.view(cfg.block_row, cfg.blocks_per_group, cfg.block_col).permute(1, 0, 2)
        
    def update_weights_vectorized(self, G_view: GroupedWeightView, grp_r: int, grp_c: int, b: int):
        """
        Vectorized version of OBS weight update for a single group.
        """
        cfg = self.config
        G = G_view.G
        
        C = self.H_inv_list[grp_c]  # (64, 64)
        
        start = b * cfg.block_col
        end = (b + 1) * cfg.block_col
        
        C_col_b = C[:, start:end]  # (64, 16)
        C_bb = C[start:end, start:end]  # (16, 16)
        C_bb_inv = torch.linalg.inv(C_bb + 1e-6 * torch.eye(cfg.block_col, device=self.device))
        
        # G[grp_r, grp_c]: (4, 16, 16) -> (16, 64) for all outputs
        G_grp = G[grp_r, grp_c].permute(1, 0, 2).reshape(cfg.block_row, -1)  # (16, 64)
        
        # Extract block b for all outputs: (16, 16)
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
        
        # Mask pruned block
        unpruned_mask = torch.ones(64, dtype=torch.bool, device=self.device)
        unpruned_mask[start:end] = False
        
        # Apply update
        G_grp[:, unpruned_mask] -= update[:, unpruned_mask]
        
        # Zero out pruned block
        G_grp[:, start:end] = 0
        
        # Reshape back
        G[grp_r, grp_c] = G_grp.view(cfg.block_row, cfg.blocks_per_group, cfg.block_col).permute(1, 0, 2)
        
    def update_C(self, grp_c: int, b: int):
        """
        Update C to remove pruned block b using Schur complement.
        
        C <- C - C_{:,b} @ (C_{b,b})^{-1} @ C_{b,:}
        """
        cfg = self.config
        C = self.H_inv_list[grp_c]  # (64, 64)
        
        start = b * cfg.block_col
        end = (b + 1) * cfg.block_col
        
        C_col_b = C[:, start:end]  # (64, 16)
        C_bb = C[start:end, start:end]  # (16, 16)
        C_bb_inv = torch.linalg.inv(C_bb + 1e-6 * torch.eye(cfg.block_col, device=self.device))
        C_b_row = C[start:end, :]  # (16, 64)
        
        # Schur complement update
        update = C_col_b @ C_bb_inv @ C_b_row  # (64, 64)
        C_new = C - update
        
        # Zero out the pruned block rows/cols for numerical stability
        C_new[start:end, :] = 0
        C_new[:, start:end] = 0
        
        self.H_inv_list[grp_c] = C_new
        
    def prune_step(self, G_view: GroupedWeightView) -> bool:
        """
        Perform one pruning step across all groups.
        
        Returns:
            True if pruning was performed, False if no more blocks to prune
        """
        cfg = self.config
        
        # Check if we have blocks left to prune
        if self.remaining_to_prune.sum() == 0:
            return False
        
        # Compute scores for all blocks
        scores = self.compute_pruning_scores_vectorized(G_view)
        
        # For each group that still needs pruning, find the block with minimum score
        for grp_r in range(cfg.num_groups_row):
            for grp_c in range(cfg.num_groups_col):
                if self.remaining_to_prune[grp_r, grp_c] > 0:
                    # Get scores for this group
                    grp_scores = scores[grp_r, grp_c]  # (4,)
                    
                    # Find minimum
                    b_to_prune = torch.argmin(grp_scores).item()
                    
                    # Prune this block
                    self._prune_block(G_view, grp_r, grp_c, b_to_prune)
                    
        return True
    
    def _prune_block(self, G_view: GroupedWeightView, grp_r: int, grp_c: int, b: int):
        """Prune a specific block and update state."""
        # Update weights
        self.update_weights_vectorized(G_view, grp_r, grp_c, b)
        
        # Update C
        self.update_C(grp_c, b)
        
        # Mark as pruned
        self.pruned_mask[grp_r, grp_c, b] = True
        self.remaining_to_prune[grp_r, grp_c] -= 1
        
    def prune_all(self, W0: torch.Tensor) -> torch.Tensor:
        """
        Prune all blocks according to config.
        
        Args:
            W0: Original weight matrix (256, 128)
            
        Returns:
            Pruned weight matrix
        """
        cfg = self.config
        
        # Create blocked view
        B_view = BlockedWeightView(W0, cfg)
        
        # Convert to grouped view
        G_view = B_view.to_grouped()
        
        # Prune until done
        step = 0
        while self.prune_step(G_view):
            step += 1
            if step >= cfg.blocks_to_prune * cfg.num_groups_row * cfg.num_groups_col:
                break
        
        # Convert back to weight matrix
        B_pruned = B_view.from_grouped(G_view)
        
        # Reshape to original
        W_pruned = B_pruned.permute(0, 2, 1, 3).reshape(cfg.n, cfg.k)
        
        return W_pruned


class FullyVectorizedStructuredOBS(StructuredOBSSolver):
    """
    Fully vectorized version that processes all groups simultaneously.
    """
    
    def __init__(self, X: torch.Tensor, config: BlockConfig, device: str = 'cuda'):
        super().__init__(X, config, device)
        
        # Precompute C in a more accessible format
        # Stack all C matrices: (n_grp_c, 64, 64)
        self.C_stacked = torch.stack(self.H_inv_list)  # (2, 64, 64)
        
    def compute_all_scores(self, G_view: GroupedWeightView) -> torch.Tensor:
        """
        Compute scores for all blocks in all groups simultaneously.
        """
        cfg = self.config
        G = G_view.G  # (16, 2, 4, 16, 16)
        
        # Reshape G to (n_grp_r, n_grp_c, blocks_per_group, blk_r, blk_c)
        # We need to compute Tr[G_{i,b} @ (C_{b,b})^{-1} @ G_{i,b}^T]
        
        # For each group column, C is different
        # For each group column grp_c, C_stacked[grp_c] is (64, 64)
        
        scores = torch.zeros(
            cfg.num_groups_row, cfg.num_groups_col, cfg.blocks_per_group,
            device=self.device
        )
        
        for grp_c in range(cfg.num_groups_col):
            C = self.C_stacked[grp_c]  # (64, 64)
            
            # Extract diagonal blocks C_{b,b}: (4, 16, 16)
            C_blocks = C.view(cfg.blocks_per_group, cfg.block_col, cfg.blocks_per_group, cfg.block_col)
            C_blocks = C_blocks.permute(0, 2, 1, 3)  # (4, 4, 16, 16)
            
            # Get diagonal
            C_bb = torch.stack([C_blocks[b, b] for b in range(cfg.blocks_per_group)])  # (4, 16, 16)
            C_bb_inv = torch.linalg.inv(C_bb + 1e-5 * torch.eye(cfg.block_col, device=self.device))
            
            # G[:, grp_c]: (16, 4, 16, 16)
            G_grp = G[:, grp_c]  # (16, 4, 16, 16)
            
            # Reshape to (16, 4, 16, 16) where last two are (out, in)
            # We want Tr[G_b @ C_bb_inv[b] @ G_b^T] for each group row and block
            
            # Method: batch matrix multiply
            # G_grp: (16, 4, 16, 16)
            # For each b in [0,3], we have 16 matrices of shape (16, 16)
            
            for b in range(cfg.blocks_per_group):
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
    
    def batch_update_weights(self, G_view: GroupedWeightView, prune_indices: torch.Tensor):
        """
        Update weights for multiple pruned blocks simultaneously.
        
        Args:
            G_view: Weight view
            prune_indices: (n_grp_r, n_grp_c) tensor of block indices to prune
        """
        cfg = self.config
        G = G_view.G
        
        # Process each group column separately (since they have different C)
        for grp_c in range(cfg.num_groups_col):
            C = self.C_stacked[grp_c]  # (64, 64)
            
            # Get all group rows for this column
            blocks_to_prune = prune_indices[:, grp_c]  # (16,)
            
            # For each unique block to prune in this column
            for b in range(cfg.blocks_per_group):
                mask = (blocks_to_prune == b) & (~self.pruned_mask[:, grp_c, b])
                if mask.sum() == 0:
                    continue
                
                grp_r_list = torch.where(mask)[0]
                
                start = b * cfg.block_col
                end = (b + 1) * cfg.block_col
                
                C_col_b = C[:, start:end]  # (64, 16)
                C_bb = C[start:end, start:end]  # (16, 16)
                C_bb_inv = torch.linalg.inv(C_bb + 1e-6 * torch.eye(cfg.block_col, device=self.device))
                
                # Precompute temp = C_col_b @ C_bb_inv: (64, 16)
                temp = C_col_b @ C_bb_inv
                
                for grp_r in grp_r_list:
                    # G[grp_r, grp_c]: (4, 16, 16) -> (16, 64)
                    G_grp = G[grp_r, grp_c].permute(1, 0, 2).reshape(cfg.block_row, -1)  # (16, 64)
                    
                    W_b = G_grp[:, start:end]  # (16, 16)
                    
                    # Update: (16, 16) @ (16, 64) = (16, 64)
                    update = W_b @ temp.T  # (16, 64)
                    
                    unpruned_mask = torch.ones(64, dtype=torch.bool, device=self.device)
                    unpruned_mask[start:end] = False
                    
                    G_grp[:, unpruned_mask] -= update[:, unpruned_mask]
                    G_grp[:, start:end] = 0
                    
                    # Reshape back
                    G[grp_r, grp_c] = G_grp.view(cfg.block_row, cfg.blocks_per_group, cfg.block_col).permute(1, 0, 2)
                    
                    # Mark pruned
                    self.pruned_mask[grp_r, grp_c, b] = True
                    self.remaining_to_prune[grp_r, grp_c] -= 1
    
    def batch_update_C(self, grp_c: int, pruned_blocks: List[int]):
        """Update C after pruning multiple blocks."""
        cfg = self.config
        C = self.C_stacked[grp_c].clone()
        
        for b in pruned_blocks:
            if self.pruned_mask[0, grp_c, b]:  # Check if already pruned
                continue
                
            start = b * cfg.block_col
            end = (b + 1) * cfg.block_col
            
            C_col_b = C[:, start:end]
            C_bb = C[start:end, start:end]
            C_bb_inv = torch.linalg.inv(C_bb + 1e-6 * torch.eye(cfg.block_col, device=self.device))
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
    
    config = BlockConfig(
        n=256, k=128,
        block_row=16, block_col=16,
        group_row=1, group_col=4,
        blocks_to_prune=2
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
    B_view = BlockedWeightView(W_pruned, config)
    G_view = B_view.to_grouped()
    
    print("Grouped view shape:", G_view.G.shape)
    print("Expected: (16, 2, 4, 16, 16)")
    
    # Check which blocks are zero
    G_flat = G_view.flatten_blocks()  # (16, 2, 4, 256)
    block_norms = G_flat.norm(dim=-1)  # (16, 2, 4)
    
    print("\nBlock norms (should see 2 zeros per group):")
    for grp_r in range(config.num_groups_row):
        for grp_c in range(config.num_groups_col):
            print(f"Group ({grp_r}, {grp_c}): {block_norms[grp_r, grp_c].tolist()}")
    
    # Compute loss
    loss_orig = torch.norm(X @ W0.T - X @ W0.T)**2
    loss_pruned = torch.norm(X @ W_pruned.T - X @ W0.T)**2
    
    print(f"\nOriginal loss: {loss_orig.item():.6f}")
    print(f"Pruned loss: {loss_pruned.item():.6f}")
    print(f"Relative increase: {(loss_pruned/loss_orig - 1)*100:.2f}%")


if __name__ == "__main__":
    demo()